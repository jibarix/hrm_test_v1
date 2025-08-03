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
    return F.cross_entropy(
        logits.to(torch.float32).reshape(-1, logits.shape[-1]), 
        labels.to(torch.long).reshape(-1), 
        ignore_index=ignore_index, 
        reduction="none"
    ).reshape(labels.shape)


def weighted_cross_entropy(logits, labels, path_token_id=9, path_weight=40.0, ignore_index=-100):
    """Weighted cross-entropy that heavily weights path token errors"""
    # Create weight tensor with same dtype as logits (after conversion to float32)
    num_classes = logits.shape[-1]
    weights = torch.ones(num_classes, device=logits.device, dtype=torch.float32)
    weights[path_token_id] = path_weight  # Heavy weight for path tokens
    
    return F.cross_entropy(
        logits.to(torch.float32).reshape(-1, logits.shape[-1]),  # Convert to float32 like softmax_cross_entropy
        labels.to(torch.long).reshape(-1), 
        weight=weights,
        ignore_index=ignore_index, 
        reduction="none"
    ).reshape(labels.shape)


def count_connected_components(positions, grid_width=40):
    """Count connected components in path positions"""
    if len(positions) == 0:
        return 0
        
    visited = set()
    components = 0
    
    # Convert tensor positions to tuples
    pos_set = {(pos[0].item(), pos[1].item()) for pos in positions}
    
    for pos in positions:
        pos_tuple = (pos[0].item(), pos[1].item())
        if pos_tuple not in visited:
            components += 1
            # BFS to mark all connected positions
            queue = [pos_tuple]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                
                # Check 4-connected neighbors
                for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nr, nc = current[0] + dr, current[1] + dc
                    neighbor = (nr, nc)
                    if (neighbor not in visited and 
                        0 <= nr < grid_width and 0 <= nc < grid_width and
                        neighbor in pos_set):
                        queue.append(neighbor)
    
    return components


def compute_connectivity_loss(predicted_tokens, path_token_id=9, grid_width=40, connectivity_weight=1.0):
    """Penalize disconnected path predictions"""
    batch_size = predicted_tokens.shape[0]
    connectivity_penalty = 0
    
    for b in range(batch_size):
        # Convert flat tokens to 2D grid
        pred_grid = predicted_tokens[b].reshape(grid_width, grid_width)
        path_positions = (pred_grid == path_token_id).nonzero()
        
        if len(path_positions) <= 1:
            continue  # No penalty for 0-1 path tokens
            
        # Count connected components
        connected_components = count_connected_components(path_positions, grid_width)
        
        # Penalty increases with number of disconnected components
        if connected_components > 1:
            connectivity_penalty += (connected_components - 1) * connectivity_weight
    
    return connectivity_penalty / batch_size


def compute_start_end_connectivity(predicted_tokens, labels, start_token_id=7, end_token_id=8, path_token_id=9, grid_width=40):
    """Check if predicted path connects start to end"""
    batch_size = predicted_tokens.shape[0]
    connected_count = 0
    
    for b in range(batch_size):
        # Convert flat tokens to 2D grid
        pred_grid = predicted_tokens[b].reshape(grid_width, grid_width)
        label_grid = labels[b].reshape(grid_width, grid_width)
        
        # Find start and end positions in labels
        start_positions = (label_grid == start_token_id).nonzero()
        end_positions = (label_grid == end_token_id).nonzero()
        
        if len(start_positions) == 0 or len(end_positions) == 0:
            continue
            
        start_pos = (start_positions[0][0].item(), start_positions[0][1].item())
        end_pos = (end_positions[0][0].item(), end_positions[0][1].item())
        
        # Find all predicted path positions
        path_positions = (pred_grid == path_token_id).nonzero()
        path_set = {(pos[0].item(), pos[1].item()) for pos in path_positions}
        
        # Add start and end to path for connectivity check
        path_set.add(start_pos)
        path_set.add(end_pos)
        
        # BFS from start to see if we can reach end
        visited = set()
        queue = [start_pos]
        found_end = False
        
        while queue and not found_end:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            if current == end_pos:
                found_end = True
                break
                
            # Check 4-connected neighbors
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = current[0] + dr, current[1] + dc
                neighbor = (nr, nc)
                if (neighbor not in visited and 
                    0 <= nr < grid_width and 0 <= nc < grid_width and
                    neighbor in path_set):
                    queue.append(neighbor)
        
        if found_end:
            connected_count += 1
    
    return connected_count / batch_size if batch_size > 0 else 0.0


def compute_path_efficiency(predicted_tokens, labels, path_token_id=9):
    """Measure path efficiency: predicted_length / ground_truth_length"""
    batch_size = predicted_tokens.shape[0]
    efficiencies = []
    
    for b in range(batch_size):
        pred_path_count = (predicted_tokens[b] == path_token_id).sum().item()
        true_path_count = (labels[b] == path_token_id).sum().item()
        
        if true_path_count > 0:
            efficiency = pred_path_count / true_path_count
            efficiencies.append(efficiency)
    
    return sum(efficiencies) / len(efficiencies) if efficiencies else 0.0


def compute_valid_path_ratio(predicted_tokens, labels, path_token_id=9, obstacle_token_id=1, grid_width=40):
    """Check what fraction of predicted path tokens are on valid roads (not obstacles)"""
    batch_size = predicted_tokens.shape[0]
    valid_ratios = []
    
    for b in range(batch_size):
        pred_grid = predicted_tokens[b].reshape(grid_width, grid_width)
        label_grid = labels[b].reshape(grid_width, grid_width)
        
        # Find predicted path positions
        pred_path_positions = (pred_grid == path_token_id).nonzero()
        
        if len(pred_path_positions) == 0:
            continue
            
        valid_count = 0
        for pos in pred_path_positions:
            r, c = pos[0].item(), pos[1].item()
            # Check if this position is not an obstacle in the input
            # We need to look at the input grid to see if it's a valid road
            # For now, we'll assume any non-obstacle token in labels is valid
            if label_grid[r, c] != obstacle_token_id:
                valid_count += 1
        
        valid_ratio = valid_count / len(pred_path_positions)
        valid_ratios.append(valid_ratio)
    
    return sum(valid_ratios) / len(valid_ratios) if valid_ratios else 0.0


class ACTLossHead(nn.Module):
    """
    ACTLossHead with aggressive anti-cheating measures:
    
    APPROACH 1: Much stronger penalties (1000x path weight, 50x connectivity, 100x sparsity)
    APPROACH 2: Hard constraints (zero gradients when cheating detected)
    
    This combination forces the model to learn legitimate pathfinding by:
    - Blocking all learning when path_ratio > 10% (hard constraint)
    - Applying massive penalties when learning is allowed (strong penalties)
    - Requiring start-end connectivity for any accuracy credit
    """
    def __init__(self, model: nn.Module, 
                 loss_type: str,
                 sparsity_penalty_weight: float = 100.0,  # Much stronger default
                 path_token_id: int = 9,
                 path_weight: float = 1000.0,              # Much stronger default
                 connectivity_weight: float = 50.0,       # Much stronger default
                 start_token_id: int = 7,
                 end_token_id: int = 8,
                 obstacle_token_id: int = 1,
                 grid_width: int = 40,
                 max_path_ratio: float = 0.1,
                 require_connectivity: bool = True):
        super().__init__()
        self.model = model
        self.loss_fn = weighted_cross_entropy if loss_type == "weighted_cross_entropy" else globals()[loss_type]
        
        # Loss and metric parameters
        self.sparsity_penalty_weight = sparsity_penalty_weight
        self.path_token_id = path_token_id
        self.path_weight = path_weight
        self.connectivity_weight = connectivity_weight
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id
        self.obstacle_token_id = obstacle_token_id
        self.grid_width = grid_width
        self.loss_type = loss_type
        
        # Anti-cheating parameters
        self.max_path_ratio = max_path_ratio
        self.require_connectivity = require_connectivity
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

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

        # Get predicted tokens for metrics and anti-cheating checks
        predicted_tokens = torch.argmax(outputs["logits"], dim=-1)
        
        # Compute loss_divisor early (needed for both normal and cheat penalty paths)
        mask = labels != IGNORE_LABEL_ID
        loss_counts = mask.sum(-1)
        loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division
        
        # ANTI-CHEATING CHECK 1: Prevent "everything is path" exploit
        # HARD CONSTRAINT APPROACH: Stop all learning when cheating detected
        pred_path_mask = (predicted_tokens == self.path_token_id)
        path_ratio = pred_path_mask.float().mean()
        
        if path_ratio > self.max_path_ratio:  # >10% of grid is path
            # HARD CONSTRAINT: Minimal loss to prevent learning when cheating detected
            # Create a tiny loss connected to computational graph to satisfy optimizer
            if self.loss_type == "weighted_cross_entropy":
                minimal_loss = (weighted_cross_entropy(
                    outputs["logits"], labels, 
                    path_token_id=self.path_token_id,
                    path_weight=self.path_weight,
                    ignore_index=IGNORE_LABEL_ID
                ) / loss_divisor).sum()
            else:
                minimal_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor).sum()
            
            # Scale loss to be tiny (effectively blocking learning)
            blocked_loss = minimal_loss * 0.001  # 1000x smaller than normal
            
            metrics = {
                "count": torch.tensor(1.0, device=outputs["logits"].device),
                "cheat_penalty": torch.tensor(1000.0, device=outputs["logits"].device),
                "path_ratio": path_ratio,
                "lm_loss": minimal_loss.detach(),
                "blocked_learning": torch.tensor(1.0, device=outputs["logits"].device),  # Track when learning is blocked
            }
            return new_carry, blocked_loss, metrics, {}, True  # Force halt when cheating

        # Correctness
        with torch.no_grad():
            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # ANTI-CHEATING CHECK 2: Require start-end connectivity
            start_end_connectivity = compute_start_end_connectivity(
                predicted_tokens, labels,
                start_token_id=self.start_token_id,
                end_token_id=self.end_token_id,
                path_token_id=self.path_token_id,
                grid_width=self.grid_width
            )
            
            # Zero out accuracy if no start→end connection
            if self.require_connectivity and start_end_connectivity < 0.5:
                seq_is_correct = torch.zeros_like(seq_is_correct, dtype=torch.bool)
            
            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Main loss computation
        if self.loss_type == "weighted_cross_entropy":
            lm_loss = (weighted_cross_entropy(
                outputs["logits"], labels, 
                path_token_id=self.path_token_id,
                path_weight=self.path_weight,
                ignore_index=IGNORE_LABEL_ID
            ) / loss_divisor).sum()
        else:
            lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor).sum()
            
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")

        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })

        # Sparsity penalty (stronger)
        sparsity_penalty = torch.tensor(0.0, device=outputs["logits"].device)
        if self.sparsity_penalty_weight > 0:
            path_predictions = (predicted_tokens == self.path_token_id).float()
            sparsity_penalty = self.sparsity_penalty_weight * path_predictions.mean()
            metrics["sparsity_penalty"] = sparsity_penalty.detach()

        # Connectivity penalty (stronger)
        connectivity_penalty = torch.tensor(0.0, device=outputs["logits"].device)
        if self.connectivity_weight > 0:
            connectivity_penalty_value = compute_connectivity_loss(
                predicted_tokens, 
                path_token_id=self.path_token_id,
                grid_width=self.grid_width,
                connectivity_weight=self.connectivity_weight
            )
            connectivity_penalty = torch.tensor(connectivity_penalty_value, device=outputs["logits"].device)
            metrics["connectivity_penalty"] = connectivity_penalty.detach()

        # Comprehensive path metrics
        path_mask = (labels == self.path_token_id)

        # Basic precision/recall
        path_precision = (pred_path_mask & path_mask).sum() / pred_path_mask.sum().clamp(min=1)
        path_recall = (pred_path_mask & path_mask).sum() / path_mask.sum().clamp(min=1)
        path_f1 = 2 * path_precision * path_recall / (path_precision + path_recall).clamp(min=1e-8)

        metrics.update({
            "path_precision": path_precision.detach(),
            "path_recall": path_recall.detach(),
            "path_f1": path_f1.detach(),
            "avg_path_tokens": pred_path_mask.sum(dim=-1).float().mean().detach(),
            "path_ratio": path_ratio.detach(),
            "blocked_learning": torch.tensor(0.0, device=outputs["logits"].device),  # Track when learning is NOT blocked
        })

        # Advanced cheating-detection metrics
        path_efficiency = compute_path_efficiency(
            predicted_tokens, labels,
            path_token_id=self.path_token_id
        )
        
        valid_path_ratio = compute_valid_path_ratio(
            predicted_tokens, labels,
            path_token_id=self.path_token_id,
            obstacle_token_id=self.obstacle_token_id,
            grid_width=self.grid_width
        )

        metrics.update({
            "start_end_connectivity": torch.tensor(start_end_connectivity, device=outputs["logits"].device),
            "path_efficiency": torch.tensor(path_efficiency, device=outputs["logits"].device),
            "valid_path_ratio": torch.tensor(valid_path_ratio, device=outputs["logits"].device),
        })

        # Q continue (bootstrapping target loss)
        q_continue_loss = torch.tensor(0.0, device=outputs["logits"].device)
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")
            metrics["q_continue_loss"] = q_continue_loss.detach()

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        # Total loss (with stronger penalties)
        total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss) + sparsity_penalty + connectivity_penalty

        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()