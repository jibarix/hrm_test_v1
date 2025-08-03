from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn


IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    valid_mask = labels != ignore_index
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


def spatial_connectivity_loss(logits, labels, grid_width=40, path_token_id=6, spatial_penalty_weight=10.0):
    """
    FUNDAMENTAL FIX: Penalize spatially impossible path transitions
    
    This addresses the core problem where the model treats the 1600-token sequence
    as a flat sequence without understanding 2D spatial relationships.
    
    Args:
        logits: Model predictions [batch_size, seq_len, vocab_size]
        labels: Ground truth [batch_size, seq_len] 
        grid_width: Width of the 2D grid (40 for 40x40 grid)
        path_token_id: Token ID for path predictions (6 in simplified system)
        spatial_penalty_weight: Weight for spatial penalty
    
    Returns:
        Spatial connectivity penalty (scalar tensor)
    """
    predicted_tokens = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]
    batch_size = predicted_tokens.shape[0]
    
    if batch_size == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    
    total_spatial_penalty = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    
    for b in range(batch_size):
        # Find all predicted path positions for this batch element
        path_mask = (predicted_tokens[b] == path_token_id)
        path_positions = path_mask.nonzero().squeeze(-1)  # [num_path_tokens]
        
        if len(path_positions) <= 1:
            continue  # No penalty for 0 or 1 path tokens
        
        # Check spatial connectivity of predicted path
        batch_penalty = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        
        for i in range(len(path_positions) - 1):
            pos1 = path_positions[i].item()
            pos2 = path_positions[i + 1].item()
            
            # Convert flat sequence indices to 2D grid coordinates
            r1, c1 = pos1 // grid_width, pos1 % grid_width
            r2, c2 = pos2 // grid_width, pos2 % grid_width
            
            # Calculate Manhattan distance between consecutive path positions
            manhattan_dist = abs(r1 - r2) + abs(c1 - c2)
            
            # Spatially adjacent cells should have Manhattan distance = 1
            if manhattan_dist > 1:
                # Penalize impossible spatial jumps
                jump_penalty = (manhattan_dist - 1) * spatial_penalty_weight
                batch_penalty += torch.tensor(jump_penalty, device=logits.device, dtype=logits.dtype)
        
        total_spatial_penalty += batch_penalty
    
    # Return average penalty across batch
    return total_spatial_penalty / batch_size


def path_connectivity_loss(logits, labels, grid_width=40, path_token_id=6, connectivity_weight=5.0):
    """
    OPTIONAL ENHANCEMENT: Penalize disconnected path components
    
    This ensures predicted paths form connected components rather than
    scattered individual path tokens across the grid.
    
    Args:
        logits: Model predictions [batch_size, seq_len, vocab_size]
        labels: Ground truth [batch_size, seq_len]
        grid_width: Width of the 2D grid (40)  
        path_token_id: Token ID for path predictions (6)
        connectivity_weight: Weight for connectivity penalty
        
    Returns:
        Path connectivity penalty (scalar tensor)
    """
    predicted_tokens = torch.argmax(logits, dim=-1)
    batch_size = predicted_tokens.shape[0]
    
    if batch_size == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    
    total_connectivity_penalty = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    
    for b in range(batch_size):
        # Convert flat predictions to 2D grid
        pred_grid = predicted_tokens[b].reshape(grid_width, grid_width)
        path_positions = (pred_grid == path_token_id).nonzero()  # [num_paths, 2]
        
        if len(path_positions) <= 1:
            continue
        
        # Count connected components using flood fill
        visited = torch.zeros_like(pred_grid, dtype=torch.bool)
        connected_components = 0
        
        for pos in path_positions:
            r, c = pos[0].item(), pos[1].item()
            if not visited[r, c] and pred_grid[r, c] == path_token_id:
                # Start flood fill from this unvisited path position
                connected_components += 1
                stack = [(r, c)]
                
                while stack:
                    curr_r, curr_c = stack.pop()
                    if (curr_r < 0 or curr_r >= grid_width or 
                        curr_c < 0 or curr_c >= grid_width or
                        visited[curr_r, curr_c] or
                        pred_grid[curr_r, curr_c] != path_token_id):
                        continue
                    
                    visited[curr_r, curr_c] = True
                    
                    # Add 4-connected neighbors to stack
                    for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                        stack.append((curr_r + dr, curr_c + dc))
        
        # Penalty increases with number of disconnected components
        if connected_components > 1:
            disconnection_penalty = (connected_components - 1) * connectivity_weight
            total_connectivity_penalty += torch.tensor(disconnection_penalty, device=logits.device, dtype=logits.dtype)
    
    return total_connectivity_penalty / batch_size


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str, 
                 # Spatial connectivity parameters
                 enable_spatial_loss: bool = True,
                 spatial_penalty_weight: float = 10.0,
                 enable_connectivity_loss: bool = False,  
                 connectivity_weight: float = 5.0,
                 grid_width: int = 40,
                 path_token_id: int = 6):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        
        # Spatial loss configuration
        self.enable_spatial_loss = enable_spatial_loss
        self.spatial_penalty_weight = spatial_penalty_weight
        self.enable_connectivity_loss = enable_connectivity_loss
        self.connectivity_weight = connectivity_weight
        self.grid_width = grid_width
        self.path_token_id = path_token_id
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        # Correctness
        with torch.no_grad():
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Standard losses
        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")

        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })

        # FUNDAMENTAL SPATIAL FIX: Add spatial connectivity penalty
        spatial_loss = torch.tensor(0.0, device=outputs["logits"].device, dtype=outputs["logits"].dtype)
        connectivity_loss = torch.tensor(0.0, device=outputs["logits"].device, dtype=outputs["logits"].dtype)
        
        if self.enable_spatial_loss:
            spatial_loss = spatial_connectivity_loss(
                outputs["logits"], labels,
                grid_width=self.grid_width,
                path_token_id=self.path_token_id,
                spatial_penalty_weight=self.spatial_penalty_weight
            )
            metrics["spatial_loss"] = spatial_loss.detach()
        
        if self.enable_connectivity_loss:
            connectivity_loss = path_connectivity_loss(
                outputs["logits"], labels,  
                grid_width=self.grid_width,
                path_token_id=self.path_token_id,
                connectivity_weight=self.connectivity_weight
            )
            metrics["connectivity_loss"] = connectivity_loss.detach()

        # Q continue (bootstrapping target loss)
        q_continue_loss = torch.tensor(0.0, device=outputs["logits"].device, dtype=outputs["logits"].dtype)
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")
            metrics["q_continue_loss"] = q_continue_loss.detach()

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        # TOTAL LOSS: Standard ACT loss + spatial fixes
        total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss) + spatial_loss + connectivity_loss

        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()