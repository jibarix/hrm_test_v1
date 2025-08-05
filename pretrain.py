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

    # DEBUGGING OPTIONS
    debug_mode: bool = False
    verbose_training: bool = False


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int


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

    # DEBUGGING: Print model info
    if config.debug_mode:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f">>> DEBUG: Model has {total_params:,} parameters ({trainable_params:,} trainable)")

    # Parameter separation with verification
    print("\n>>> Model Parameter Structure:")
    param_names = {}
    for name, param in model.named_parameters():
        category = "other"
        if 'puzzle_emb' in name:
            category = "puzzle_emb"
        elif 'q_head' in name.lower():
            category = "q_head"
        
        if category not in param_names:
            param_names[category] = []
        param_names[category].append(name)
    
    for category, names in param_names.items():
        print(f"\n{category} parameters ({len(names)}):")
        for name in names[:3]:
            print(f"  - {name}")
        if len(names) > 3:
            print(f"  ... and {len(names) - 3} more")
    
    # Separate parameters into groups
    puzzle_emb_params = []
    q_head_params = []
    main_params = []
    
    for name, param in model.named_parameters():
        if 'puzzle_emb' in name:
            puzzle_emb_params.append(param)
        elif 'q_head' in name.lower():
            q_head_params.append(param)
        else:
            main_params.append(param)
    
    print(f"\n>>> Final Parameter Grouping:")
    print(f"  Puzzle Embedding Params: {len(puzzle_emb_params)}")
    print(f"  Q-Head Params: {len(q_head_params)}")
    print(f"  Main Model Params: {len(main_params)}")
    
    if len(q_head_params) == 0:
        print("WARNING: No Q-head parameters found! Check model structure.")
        main_params = [p for n, p in model.named_parameters() if 'puzzle_emb' not in n]

    # Create optimizers
    optimizers = [
        CastedSparseEmbeddingSignSGD_Distributed(
            model.model.puzzle_emb.buffers(),  # type: ignore
            lr=0,
            weight_decay=config.puzzle_emb_weight_decay,
            world_size=world_size
        ),
        AdamATan2(
            main_params,
            lr=0,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2)
        )
    ]
    
    optimizer_lrs = [config.puzzle_emb_lr, config.lr]
    
    if len(q_head_params) > 0:
        q_head_lr_ratio = getattr(config, 'q_head_lr_ratio', 0.1)
        optimizers.append(
            AdamATan2(
                q_head_params,
                lr=0,
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        )
        optimizer_lrs.append(config.lr * q_head_lr_ratio)
        print(f"  Q-Head LR will be: {config.lr * q_head_lr_ratio} (main LR * {q_head_lr_ratio})")

    return model, optimizers, optimizer_lrs


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if base_lr == 0:
        return 0
        
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    if current_step >= num_training_steps:
        return base_lr * min_ratio
    
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    lr = base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))
    
    if lr == 0 and min_ratio > 0:
        lr = base_lr * min_ratio
        
    return lr


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, world_size: int):
    # Calculate total steps more carefully
    examples_per_epoch = int(train_metadata.total_groups * train_metadata.mean_puzzle_examples)
    batches_per_epoch = (examples_per_epoch + config.global_batch_size - 1) // config.global_batch_size
    total_batches = config.epochs * batches_per_epoch
    
    # For step counting, we'll count batches, not samples
    total_steps = total_batches
    
    if config.debug_mode:
        print(f">>> DEBUG: Training calculation:")
        print(f"    Examples per epoch: {examples_per_epoch}")
        print(f"    Batches per epoch: {batches_per_epoch}")
        print(f"    Total epochs: {config.epochs}")
        print(f"    Total batches (steps): {total_steps}")

    # Model
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, world_size=world_size)

    return TrainState(
        step=0,
        total_steps=total_steps,
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None
    )


def save_train_state(config: PretrainConfig, train_state: TrainState):
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    checkpoint_name = f"step_{train_state.step}"
    torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, checkpoint_name))
    
    if config.debug_mode:
        print(f">>> DEBUG: Saved checkpoint {checkpoint_name}")


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    lr = cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )
    
    return lr


def debug_model_output(outputs, batch, config, step_count):
    """Debug function to analyze model outputs"""
    if not config.debug_mode:
        return
    
    print(f"\n>>> DEBUG: Model Output Analysis (Step {step_count})")
    
    if "logits" in outputs:
        logits = outputs["logits"]
        print(f"    Logits shape: {logits.shape}")
        
        # Get predictions
        predictions = torch.argmax(logits, dim=-1)
        
        # Analyze first example
        first_pred = predictions[0].cpu().numpy()
        first_labels = batch["labels"][0].cpu().numpy()
        
        # Count tokens
        from collections import Counter
        pred_counter = Counter(first_pred)
        label_counter = Counter(first_labels)
        
        print(f"    First example predictions: {dict(list(pred_counter.items())[:5])}")
        print(f"    First example true labels: {dict(list(label_counter.items())[:5])}")
        
        # Check for the "all PATH" problem
        path_tokens = pred_counter.get(9, 0)  # PATH token = 9
        if path_tokens == len(first_pred):
            print(f"    ⚠️ WARNING: Model outputs PATH for ALL tokens!")
        elif path_tokens > 0:
            print(f"    PATH tokens: {path_tokens}/{len(first_pred)} ({path_tokens/len(first_pred)*100:.1f}%)")
        
        # Check unique predictions
        unique_preds = len(set(first_pred))
        print(f"    Unique prediction tokens: {unique_preds}/10 possible")
        
        if unique_preds == 1:
            print(f"    ⚠️ WARNING: Model only predicts one token type: {first_pred[0]}")


def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    # Increment step counter (counting batches, not samples)
    train_state.step += 1

    if train_state.step > train_state.total_steps:
        return

    # To device
    batch = {k: v.cuda() for k, v in batch.items()}

    # DEBUGGING: Print batch info
    if config.debug_mode and train_state.step <= 3:
        print(f"\n>>> DEBUG: Batch info (step {train_state.step})")
        for k, v in batch.items():
            if hasattr(v, 'shape'):
                print(f"    {k}: {v.shape}, dtype: {v.dtype}")
        
        # Check batch content
        first_input = batch["inputs"][0].cpu().numpy()
        first_label = batch["labels"][0].cpu().numpy()
        
        from collections import Counter
        input_counter = Counter(first_input)
        label_counter = Counter(first_label)
        
        print(f"    First example input tokens: {dict(list(input_counter.items())[:5])}")
        print(f"    First example label tokens: {dict(list(label_counter.items())[:5])}")
        
        path_in_labels = label_counter.get(9, 0)
        print(f"    PATH tokens in labels: {path_in_labels}/{len(first_label)} ({path_in_labels/len(first_label)*100:.1f}%)")

    # Init carry if it is None
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)

    # Detach carry state
    if train_state.carry is not None:
        carry = train_state.carry
        if hasattr(carry, 'inner_carry') and hasattr(carry.inner_carry, 'z_H'):
             carry.inner_carry.z_H = carry.inner_carry.z_H.detach()
        if hasattr(carry, 'inner_carry') and hasattr(carry.inner_carry, 'z_L'):
             carry.inner_carry.z_L = carry.inner_carry.z_L.detach()
        if hasattr(carry, 'steps'):
            carry.steps = carry.steps.detach()
        if hasattr(carry, 'halted'):
            carry.halted = carry.halted.detach()
        if hasattr(carry, 'current_data') and isinstance(carry.current_data, dict):
             carry.current_data = {k: v.detach() for k, v in carry.current_data.items()}
        train_state.carry = carry

    # Forward pass
    train_state.carry, loss, metrics, outputs, all_halted = train_state.model(
        carry=train_state.carry, batch=batch, return_keys=["logits"]
    )

    # DEBUGGING: Analyze outputs
    debug_model_output(outputs, batch, config, train_state.step)

    # DEBUGGING: Check loss and gradients
    if config.debug_mode and train_state.step <= 10:
        print(f"\n>>> DEBUG: Training step {train_state.step}")
        print(f"    Raw loss: {loss.item():.6f}")
        print(f"    Scaled loss: {(loss / global_batch_size).item():.6f}")
        print(f"    All halted: {all_halted}")
        
        if len(metrics) > 0:
            print(f"    Metrics: {list(metrics.keys())}")
            for k, v in metrics.items():
                if hasattr(v, 'item'):
                    print(f"      {k}: {v.item():.6f}")

    # Backward pass
    scaled_loss = loss / global_batch_size
    scaled_loss.backward()

    # DEBUGGING: Check gradients
    if config.debug_mode and train_state.step <= 3:
        grad_norms = []
        for name, param in train_state.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append((name, grad_norm))
        
        # Show top 5 gradient norms
        grad_norms.sort(key=lambda x: x[1], reverse=True)
        print(f"    Top gradient norms:")
        for name, norm in grad_norms[:5]:
            print(f"      {name}: {norm:.6f}")
        
        total_grad_norm = sum(norm for _, norm in grad_norms)
        print(f"    Total gradient norm: {total_grad_norm:.6f}")
        
        if total_grad_norm < 1e-8:
            print(f"    ⚠️ WARNING: Very small gradients - learning might be slow!")

    # Gradient Clipping
    total_norm = torch.nn.utils.clip_grad_norm_(train_state.model.parameters(), max_norm=1.0)
    
    if config.debug_mode and train_state.step <= 3:
        print(f"    Gradient norm before clipping: {total_norm:.6f}")

    # Allreduce
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

    # DEBUGGING: Print learning rate
    if config.debug_mode and (train_state.step <= 5 or train_state.step % 100 == 0):
        warmup_done = train_state.step >= config.lr_warmup_steps
        print(f"    Learning rate: {lr_this_step:.2e} {'(warmup)' if not warmup_done else '(normal)'}")

    # Reduce metrics
    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())

        metric_keys = list(sorted(metrics.keys()))
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            
            count = max(reduced_metrics["count"], 1)
            reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}

            reduced_metrics["train/lr"] = lr_this_step
            
            # DEBUGGING: Print key metrics
            if config.verbose_training or (config.debug_mode and train_state.step <= 20):
                lm_loss = reduced_metrics.get("train/lm_loss", 0)
                accuracy = reduced_metrics.get("train/accuracy", 0)
                exact_acc = reduced_metrics.get("train/exact_accuracy", 0)
                print(f"    >>> Step {train_state.step}: Loss={lm_loss:.4f}, Acc={accuracy:.1%}, ExactAcc={exact_acc:.1%}")
            
            return reduced_metrics


def evaluate(config: PretrainConfig, train_state: TrainState, eval_loader: torch.utils.data.DataLoader, eval_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    if config.debug_mode:
        print(f"\n>>> DEBUG: Starting evaluation at step {train_state.step}")
    
    with torch.inference_mode():
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}
        
        all_preds = {}

        metric_keys = []
        metric_values = None
        metric_global_batch_size = [0 for _ in range(len(set_ids))]
        
        carry = None
        batch_count = 0
        
        for set_name, batch, global_batch_size in eval_loader:
            batch_count += 1
            
            # To device
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)

            # Forward
            step_count = 0
            max_steps = 20
            
            while step_count < max_steps:
                carry, _, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=config.eval_save_outputs
                )
                step_count += 1
                
                if all_finish:
                    break
            
            # DEBUGGING: Check evaluation outputs
            if config.debug_mode and batch_count <= 2:
                print(f"    Eval batch {batch_count}: {step_count} steps, halted: {all_finish}")
                if "logits" in preds:
                    debug_model_output(preds, batch, config, f"eval-{batch_count}")

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        all_preds.setdefault(k, [])
                        all_preds[k].append(v.cpu())
                        
            del carry, preds, batch, all_finish

            # Aggregate
            set_id = set_ids[set_name]
            
            if metric_values is None:
                metric_keys = list(sorted(metrics.keys()))
                metric_values = torch.zeros((len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda")
                
            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])
            metric_global_batch_size[set_id] += global_batch_size

        if len(all_preds) and config.checkpoint_path is not None:
            all_preds = {k: torch.cat(v, dim=0) for k, v in all_preds.items()}

            os.makedirs(config.checkpoint_path, exist_ok=True)
            torch.save(all_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}"))

        # Logging
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
                
                # DEBUGGING: Print evaluation results
                if config.debug_mode:
                    print(f">>> DEBUG: Evaluation results:")
                    for set_name, metrics in reduced_metrics.items():
                        acc = metrics.get("accuracy", 0)
                        exact_acc = metrics.get("exact_accuracy", 0)
                        lm_loss = metrics.get("lm_loss", 0)
                        print(f"    {set_name}: Loss={lm_loss:.4f}, Acc={acc:.1%}, ExactAcc={exact_acc:.1%}")

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
        config = PretrainConfig(**hydra_config)

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

    return objects[0]


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1

    # Initialize distributed training if in distributed environment
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        
    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # DEBUGGING: Print config
    if config.debug_mode and RANK == 0:
        print(f"\n>>> DEBUG: Training Configuration")
        print(f"    Data path: {config.data_path}")
        print(f"    Batch size: {config.global_batch_size}")
        print(f"    Learning rate: {config.lr}")
        print(f"    Warmup steps: {config.lr_warmup_steps}")
        print(f"    Epochs: {config.epochs}")
        print(f"    Debug mode: {config.debug_mode}")
        print(f"    Verbose training: {config.verbose_training}")

    # Seed RNGs
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
        progress_bar = tqdm.tqdm(total=train_state.total_steps, desc="Training")

        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), settings=wandb.Settings(_disable_stats=True))
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
        save_code_and_config(config)

    # Training Loop
    for iter_id in range(total_iters):
        if RANK == 0:
            print(f"\n{'='*60}")
            print(f"[Rank {RANK}, World Size {WORLD_SIZE}]: Starting Epoch {iter_id * train_epochs_per_iter}")
            print(f"{'='*60}")

        # Train Iter
        train_state.model.train()
        batch_count = 0
        
        for set_name, batch, global_batch_size in train_loader:
            batch_count += 1
            
            if config.debug_mode and RANK == 0 and batch_count <= 3:
                print(f"\n--- Processing batch {batch_count} ---")
            
            metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                if progress_bar:
                    progress_bar.update(1)
                    
                    # Update progress bar description with current metrics
                    if "train/lm_loss" in metrics and "train/accuracy" in metrics:
                        loss_val = metrics["train/lm_loss"]
                        acc_val = metrics["train/accuracy"]
                        progress_bar.set_description(f"Training (Loss: {loss_val:.3f}, Acc: {acc_val:.1%})")

        # Evaluation
        if RANK == 0:
            print(f"\n--- Starting Evaluation ---")
        
        train_state.model.eval()
        metrics = evaluate(config, train_state, eval_loader, eval_metadata, rank=RANK, world_size=WORLD_SIZE)

        if RANK == 0 and metrics is not None:
            wandb.log(metrics, step=train_state.step)
            
            # Print evaluation summary
            print(f"--- Evaluation Complete ---")
            for set_name, set_metrics in metrics.items():
                acc = set_metrics.get("accuracy", 0)
                exact_acc = set_metrics.get("exact_accuracy", 0)
                lm_loss = set_metrics.get("lm_loss", 0)
                print(f"  {set_name}: Loss={lm_loss:.4f}, Accuracy={acc:.1%}, Exact={exact_acc:.1%}")
            
        # Checkpointing
        if RANK == 0 and (config.checkpoint_every_eval or (iter_id == total_iters - 1)):
            save_train_state(config, train_state)

    # Finalize
    if RANK == 0:
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE!")
        print(f"Final step: {train_state.step}")
        print(f"{'='*60}")
        
        if progress_bar:
            progress_bar.close()
    
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()