from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
import os
import math
import yaml
import shutil

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

# Configure PyTorch SDPA for optimal performance (updated for PyTorch 2.5+)
try:
    # Try new API first (PyTorch 2.5+)
    from torch.nn.attention import SDPBackend
    torch.nn.attention.sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
except (AttributeError, ImportError):
    try:
        # Fallback to older API
        torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_mem_efficient=True,
            enable_math=False
        )
    except:
        # If both fail, continue with defaults
        pass

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
from adam_atan2 import AdamATan2

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')

    name: str
    loss: LossConfig


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_path: str

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    eval_save_outputs: List[str] = []


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int
    
    # NEW: Enhanced tracking for batch rejection
    total_batches_processed: int = 0
    total_batches_rejected: int = 0
    consecutive_rejections: int = 0


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,

        dataset_path=config.data_path,

        rank=rank,
        num_replicas=world_size,
        
        **kwargs
    ), split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=None,

        num_workers=1,
        prefetch_factor=8,

        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, dataset.metadata


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, world_size: int):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore

        batch_size=config.global_batch_size // world_size,

        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False  # Non-autoregressive
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model, dynamic=False)  # type: ignore

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Optimizers and lr
    optimizers = [
        CastedSparseEmbeddingSignSGD_Distributed(
            model.model.puzzle_emb.buffers(),  # type: ignore
            
            lr=0,  # Needs to be set by scheduler
            weight_decay=config.puzzle_emb_weight_decay,

            world_size=world_size
        ),
        AdamATan2(
            model.parameters(),

            lr=0,  # Needs to be set by scheduler
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2)
        )
    ]
    optimizer_lrs = [
        config.puzzle_emb_lr,
        config.lr
    ]

    return model, optimizers, optimizer_lrs


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, world_size: int):
    # Estimated total training steps
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)

    # Model
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, world_size=world_size)

    return TrainState(
        step=0,
        total_steps=total_steps,

        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None,
        
        # Initialize enhanced tracking
        total_batches_processed=0,
        total_batches_rejected=0,
        consecutive_rejections=0
    )


def save_train_state(config: PretrainConfig, train_state: TrainState):
    # FIXME: Only saved model.
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f"step_{train_state.step}"))


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )


# ============================================================================
# Enhanced Training Functions with Batch Rejection & Resampling
# ============================================================================

def is_batch_rejected(metrics, config):
    """
    Determine if a batch should be rejected based on anti-cheat metrics
    """
    blocked_learning = metrics.get("blocked_learning", torch.tensor(0.0)).item()
    path_ratio = metrics.get("path_ratio", torch.tensor(0.0)).item()
    
    # Get rejection thresholds from config with sensible defaults
    max_path_ratio = getattr(config.arch.loss, 'max_path_ratio', 0.1)
    rejection_threshold = getattr(config.arch, 'rejection_path_ratio_threshold', max_path_ratio * 1.5)  # 50% buffer
    
    # Reject if:
    # 1. Explicitly blocked by loss function (blocked_learning > 0.5)
    # 2. Path ratio exceeds threshold (spam detection)
    should_reject = (blocked_learning > 0.5) or (path_ratio > rejection_threshold)
    
    return should_reject, {
        "blocked_learning": blocked_learning,
        "path_ratio": path_ratio,
        "rejection_threshold": rejection_threshold
    }


def train_batch_with_rejection(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    """
    Enhanced train_batch function with rejection capability
    Returns: (accepted: bool, metrics: dict or None)
    """
    train_state.step += 1
    train_state.total_batches_processed += 1
    
    if train_state.step > train_state.total_steps:
        return True, None  # Accept to end training

    # To device
    batch = {k: v.cuda() for k, v in batch.items()}

    # Init carry if it is None
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)

    # Forward pass (always compute to get metrics)
    train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])

    # Check if batch should be rejected
    should_reject, rejection_info = is_batch_rejected(metrics, config)
    
    if should_reject:
        # REJECT BATCH - No gradient computation, no optimizer step
        train_state.total_batches_rejected += 1
        train_state.consecutive_rejections += 1
        
        if rank == 0:
            print(f"[Step {train_state.step}] REJECTED batch - path_ratio: {rejection_info['path_ratio']:.3f}, "
                  f"blocked: {rejection_info['blocked_learning']:.1f}, "
                  f"consecutive: {train_state.consecutive_rejections}")
        
        # Return rejection metrics for logging
        rejection_metrics = {
            "train/batch_rejected": 1.0,
            "train/rejection_path_ratio": rejection_info['path_ratio'],
            "train/rejection_blocked": rejection_info['blocked_learning'],
            "train/consecutive_rejections": float(train_state.consecutive_rejections),
            "train/total_rejection_rate": train_state.total_batches_rejected / train_state.total_batches_processed,
        }
        return False, rejection_metrics  # Rejected
    
    # ACCEPT BATCH - Normal training
    train_state.consecutive_rejections = 0  # Reset consecutive counter
    
    # Backward pass
    ((1 / global_batch_size) * loss).backward()

    # Allreduce gradients
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
            
    # Apply optimizer
    lr_this_step = None    
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)
        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
        optim.step()
        optim.zero_grad()

    # Process metrics for accepted batches
    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())
        metric_keys = list(sorted(metrics.keys()))
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            
            # Postprocess
            count = max(reduced_metrics["count"], 1)
            reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}
            
            # Add training metadata
            reduced_metrics["train/lr"] = lr_this_step
            reduced_metrics["train/batch_rejected"] = 0.0  # Mark as accepted
            reduced_metrics["train/consecutive_rejections"] = 0.0
            reduced_metrics["train/total_rejection_rate"] = train_state.total_batches_rejected / train_state.total_batches_processed
            
            return True, reduced_metrics  # Accepted

    return True, None  # Accepted but no metrics


class BatchResampler:
    """
    Helper class to handle batch resampling when rejections occur
    """
    def __init__(self, dataloader, max_consecutive_rejections=10):
        self.dataloader = dataloader
        self.dataloader_iter = iter(dataloader)
        self.max_consecutive_rejections = max_consecutive_rejections
        
    def get_next_batch(self):
        """Get next batch from dataloader, handling StopIteration"""
        try:
            return next(self.dataloader_iter)
        except StopIteration:
            # Restart dataloader
            try:
                self.dataloader_iter = iter(self.dataloader)
                return next(self.dataloader_iter)
            except StopIteration:
                # Empty dataloader - this shouldn't happen
                print("Warning: Empty dataloader encountered")
                return None
        except Exception as e:
            print(f"Error getting next batch: {e}")
            return None
    
    def get_batch_with_rejection_handling(self, config, train_state, rank, world_size, progress_bar=None):
        """
        Get a batch and train on it, handling rejections with resampling
        Returns: (success: bool, final_metrics: dict or None)
        """
        attempts = 0
        final_metrics = None
        
        while attempts < self.max_consecutive_rejections:
            # Get next batch
            batch_data = self.get_next_batch()
            if batch_data is None:
                # No more data available
                return False, None
            
            set_name, batch, global_batch_size = batch_data
            
            # Try training on this batch
            accepted, metrics = train_batch_with_rejection(
                config, train_state, batch, global_batch_size, rank, world_size
            )
            
            if accepted:
                # Batch was accepted - we're done
                final_metrics = metrics
                if rank == 0 and progress_bar is not None and metrics is not None:
                    progress_bar.update(train_state.step - progress_bar.n)
                return True, final_metrics
            else:
                # Batch was rejected - log and try next batch
                attempts += 1
                if rank == 0 and metrics is not None:
                    # Log rejection metrics immediately
                    if wandb.run is not None:
                        wandb.log(metrics, step=train_state.step)
        
        # Too many consecutive rejections - force accept the last batch to prevent infinite loop
        if rank == 0:
            print(f"WARNING: {self.max_consecutive_rejections} consecutive rejections at step {train_state.step}. "
                  "This indicates the model may be stuck in a degenerate state.")
        
        # Force accept by temporarily disabling rejection
        batch_data = self.get_next_batch()
        if batch_data is not None:
            set_name, batch, global_batch_size = batch_data
            
            # Ensure batch is on correct device
            batch = {k: v.cuda() for k, v in batch.items()}
            
            # Force accept by skipping rejection check (just do normal training)
            train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])
            ((1 / global_batch_size) * loss).backward()
            
            # Apply optimizer
            for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
                lr_this_step = compute_lr(base_lr, config, train_state)
                for param_group in optim.param_groups:
                    param_group['lr'] = lr_this_step
                optim.step()
                optim.zero_grad()
            
            if rank == 0:
                print(f"Force-accepted batch at step {train_state.step}")
        
        return True, None


def evaluate(config: PretrainConfig, train_state: TrainState, eval_loader: torch.utils.data.DataLoader, eval_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    with torch.inference_mode():
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}
        
        all_preds = {}

        metric_keys = []
        metric_values = None
        metric_global_batch_size = [0 for _ in range(len(set_ids))]
        
        carry = None
        for set_name, batch, global_batch_size in eval_loader:
            # To device
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)  # type: ignore

            # Forward
            while True:
                carry, _, metrics, preds, all_finish = train_state.model(carry=carry, batch=batch, return_keys=config.eval_save_outputs)
                
                if all_finish:
                    break

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        all_preds.setdefault(k, [])
                        all_preds[k].append(v.cpu())  # Move to CPU for saving GPU memory
                        
            del carry, preds, batch, all_finish

            # Aggregate
            set_id = set_ids[set_name]
            
            if metric_values is None:
                metric_keys = list(sorted(metrics.keys()))  # Sort keys to guarantee all processes use the same order.
                metric_values = torch.zeros((len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda")
                
            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])
            metric_global_batch_size[set_id] += global_batch_size

        if len(all_preds) and config.checkpoint_path is not None:
            all_preds = {k: torch.cat(v, dim=0) for k, v in all_preds.items()}

            os.makedirs(config.checkpoint_path, exist_ok=True)
            torch.save(all_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}"))

        # Logging
        # Reduce to rank 0
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)
            
            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {set_name: {metric_name: reduced_metrics[set_id, metric_id] for metric_id, metric_name in enumerate(metric_keys)}
                                   for set_id, set_name in enumerate(set_ids)}
                
                # Postprocess
                for set_name, metrics in reduced_metrics.items():
                    count = metrics.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in metrics.items()}

                return reduced_metrics


def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code
    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_path).capitalize()} ACT-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        
    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)

    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    eval_loader,  eval_metadata  = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)

    # Train state
    train_state = init_train_state(config, train_metadata, world_size=WORLD_SIZE)

    # Progress bar and logger
    progress_bar = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)

        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), settings=wandb.Settings(_disable_stats=True))  # type: ignore
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
        save_code_and_config(config)

    # Enhanced Training Loop with Batch Rejection & Resampling
    max_consecutive_rejections = getattr(config.arch, 'max_consecutive_rejections', 10)
    enable_batch_rejection = getattr(config.arch, 'enable_batch_rejection', True)
    
    if RANK == 0:
        print(f"Enhanced training with batch rejection: {enable_batch_rejection}")
        print(f"Max consecutive rejections: {max_consecutive_rejections}")
        if enable_batch_rejection:
            # Pre-calculate estimated batches for logging
            estimated_examples = train_metadata.total_groups * train_metadata.mean_puzzle_examples * train_epochs_per_iter
            estimated_batches = max(int(estimated_examples / config.global_batch_size), 10)
            print(f"Estimated {estimated_batches} batches per epoch (from {estimated_examples:.0f} examples)")

    for _iter_id in range(total_iters):
        if RANK == 0:
            print(f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

        ############ Enhanced Training Iteration with Rejection Handling
        train_state.model.train()
        
        if enable_batch_rejection:
            # Use enhanced batch resampler
            batch_resampler = BatchResampler(train_loader, max_consecutive_rejections)
            
            # Process batches with rejection handling
            epoch_batches_processed = 0
            # Estimate batches per epoch: dataset_size / batch_size (approximate)
            estimated_examples_per_epoch = train_metadata.total_groups * train_metadata.mean_puzzle_examples * train_epochs_per_iter
            target_batches_per_epoch = max(int(estimated_examples_per_epoch / config.global_batch_size), 10)  # At least 10 batches
            
            while epoch_batches_processed < target_batches_per_epoch and train_state.step < train_state.total_steps:
                success, metrics = batch_resampler.get_batch_with_rejection_handling(
                    config, train_state, RANK, WORLD_SIZE, progress_bar
                )
                
                if not success:
                    break  # No more data or training complete
                
                if RANK == 0 and metrics is not None:
                    wandb.log(metrics, step=train_state.step)
                
                epoch_batches_processed += 1
                
                # Log progress periodically
                if RANK == 0 and epoch_batches_processed % 50 == 0:
                    rejection_rate = train_state.total_batches_rejected / max(train_state.total_batches_processed, 1)
                    print(f"  Processed {epoch_batches_processed}/{target_batches_per_epoch} batches, "
                          f"rejection rate: {rejection_rate:.2%}, "
                          f"consecutive: {train_state.consecutive_rejections}")
        
        else:
            # Original training loop (no rejection)
            for set_name, batch, global_batch_size in train_loader:
                accepted, metrics = train_batch_with_rejection(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)
                
                if RANK == 0 and metrics is not None:
                    wandb.log(metrics, step=train_state.step)
                    progress_bar.update(train_state.step - progress_bar.n)  # type: ignore

        ############ Evaluation
        train_state.model.eval()
        metrics = evaluate(config, train_state, eval_loader, eval_metadata, rank=RANK, world_size=WORLD_SIZE)

        if RANK == 0 and metrics is not None:
            wandb.log(metrics, step=train_state.step)
            
            # Enhanced logging for rejection statistics
            if enable_batch_rejection:
                rejection_stats = {
                    "epoch/total_batches_processed": train_state.total_batches_processed,
                    "epoch/total_batches_rejected": train_state.total_batches_rejected, 
                    "epoch/overall_rejection_rate": train_state.total_batches_rejected / max(train_state.total_batches_processed, 1),
                    "epoch/consecutive_rejections": train_state.consecutive_rejections,
                }
                wandb.log(rejection_stats, step=train_state.step)
                
                print(f"Epoch {_iter_id} complete. Overall rejection rate: {rejection_stats['epoch/overall_rejection_rate']:.2%}")
            
        ############ Checkpointing
        if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
            save_train_state(config, train_state)

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    
    # Final summary
    if RANK == 0 and enable_batch_rejection:
        final_rejection_rate = train_state.total_batches_rejected / max(train_state.total_batches_processed, 1)
        print(f"\n🏁 Training Complete!")
        print(f"📊 Final Statistics:")
        print(f"   Total batches processed: {train_state.total_batches_processed}")
        print(f"   Total batches rejected: {train_state.total_batches_rejected}")
        print(f"   Overall rejection rate: {final_rejection_rate:.2%}")
        print(f"   Final consecutive rejections: {train_state.consecutive_rejections}")
        
        if final_rejection_rate > 0.5:
            print("⚠️  Warning: High rejection rate suggests model may be struggling with anti-cheat constraints")
        elif final_rejection_rate < 0.05:
            print("✅ Low rejection rate indicates model learned to avoid cheating")
    
    wandb.finish()


if __name__ == "__main__":
    launch()