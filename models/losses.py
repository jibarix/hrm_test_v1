from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import heapq
from collections import deque

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


# ============================================================================
# NEW: Enhanced Anti-Cheat Loss Functions
# ============================================================================

def compute_path_budget_loss(predicted_tokens, labels, path_token_id=9, grid_size=1600):
    """
    Direct supervision on path token count - match predicted vs true path length
    This prevents "spam everything as path" by enforcing exact token count matching
    """
    batch_size = predicted_tokens.shape[0]
    
    # Count path tokens in predictions and labels
    pred_path_mask = (predicted_tokens == path_token_id)
    true_path_mask = (labels == path_token_id)
    
    pred_count = pred_path_mask.float().sum(dim=-1)  # [B]
    true_count = true_path_mask.float().sum(dim=-1)  # [B]
    
    # L1 loss on token counts, normalized by grid size
    budget_loss = ((pred_count - true_count).abs() / grid_size).mean()
    
    return budget_loss


def compute_coverage_penalty(predicted_tokens, labels, path_token_id=9):
    """
    Penalize over-coverage: predicted paths that don't overlap with ground truth
    """
    batch_size = predicted_tokens.shape[0]
    
    pred_path_mask = (predicted_tokens == path_token_id)
    true_path_mask = (labels == path_token_id)
    
    # Count path tokens that are predicted but not in ground truth (false positives)
    false_positive_mask = pred_path_mask & (~true_path_mask)
    false_positive_count = false_positive_mask.float().sum(dim=-1)  # [B]
    
    # Normalize by total predicted path tokens to get over-coverage ratio
    pred_count = pred_path_mask.float().sum(dim=-1).clamp(min=1)  # Avoid division by zero
    over_coverage_ratio = false_positive_count / pred_count
    
    return over_coverage_ratio.mean()


def extract_positions_from_tokens(tokens, token_id, grid_width=40):
    """Extract 2D positions of specific tokens from flattened sequence"""
    positions = []
    batch_size = tokens.shape[0]
    
    for b in range(batch_size):
        token_positions = (tokens[b] == token_id).nonzero().squeeze(-1)
        if len(token_positions) > 0:
            # Convert flat indices to 2D coordinates
            first_pos = token_positions[0].item()
            row = first_pos // grid_width
            col = first_pos % grid_width
            positions.append([row, col])
        else:
            positions.append([0, 0])  # Fallback
    
    return torch.tensor(positions, device=tokens.device)


def _dijkstra_path(cost_grid, road_mask, start, end, grid_width):
    """Simple Dijkstra implementation for path projection"""
    # Priority queue: (cost, row, col)
    pq = [(0.0, start[0], start[1])]
    visited = set()
    came_from = {}
    distances = {start: 0.0}
    
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    while pq:
        current_cost, row, col = heapq.heappop(pq)
        
        if (row, col) in visited:
            continue
        visited.add((row, col))
        
        # Check if we reached the end
        if (row, col) == end:
            # Reconstruct path
            path = []
            current = end
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        # Explore neighbors
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            
            # Check bounds and road validity
            if (0 <= nr < grid_width and 0 <= nc < grid_width and 
                road_mask[nr, nc] > 0 and (nr, nc) not in visited):
                
                # Cost = current + movement cost based on cell cost
                move_cost = current_cost + cost_grid[nr, nc] + 1.0  # Base movement cost
                
                if (nr, nc) not in distances or move_cost < distances[(nr, nc)]:
                    distances[(nr, nc)] = move_cost
                    came_from[(nr, nc)] = (row, col)
                    heapq.heappush(pq, (move_cost, nr, nc))
    
    return []  # No path found


def shortest_path_projection(cost_grid, start_pos, end_pos, road_mask, grid_width=40):
    """
    Project cost grid to nearest feasible path using BFS/Dijkstra
    Args:
        cost_grid: [B, H, W] costs for each cell (-log p(PATH))
        start_pos: [B, 2] start positions
        end_pos: [B, 2] end positions  
        road_mask: [B, H, W] valid road mask (1=road, 0=obstacle)
    Returns:
        projected_mask: [B, H, W] binary mask of projected path
    """
    batch_size = cost_grid.shape[0]
    projected_paths = torch.zeros_like(cost_grid)
    
    for b in range(batch_size):
        # Extract batch elements
        costs = cost_grid[b].cpu().numpy()
        roads = road_mask[b].cpu().numpy()
        start = (start_pos[b][0].item(), start_pos[b][1].item())
        end = (end_pos[b][0].item(), end_pos[b][1].item())
        
        # Run Dijkstra's algorithm with costs
        path = _dijkstra_path(costs, roads, start, end, grid_width)
        
        # Convert path to binary mask
        if path:
            for (r, c) in path:
                if 0 <= r < grid_width and 0 <= c < grid_width:
                    projected_paths[b, r, c] = 1.0
    
    return projected_paths


def apply_structured_projection(logits, labels, path_token_id=9, start_token_id=7, end_token_id=8, 
                               obstacle_token_id=1, grid_width=40, projection_weight=1.0):
    """
    Apply structured projection during training to enforce feasible paths
    Uses Straight-Through Estimator for gradient flow
    """
    batch_size = logits.shape[0]
    device = logits.device
    
    with torch.no_grad():
        # Get probabilities for path token
        probs = torch.softmax(logits, dim=-1)
        path_probs = probs[..., path_token_id].view(batch_size, grid_width, grid_width)
        
        # Create cost grid: -log p(PATH) clamped for numerical stability
        cost_grid = (-path_probs.clamp(min=1e-6).log())
        
        # Extract start/end positions from labels
        start_positions = extract_positions_from_tokens(labels, start_token_id, grid_width)
        end_positions = extract_positions_from_tokens(labels, end_token_id, grid_width)
        
        # Create road mask (anything that's not an obstacle)
        road_mask = (labels.view(batch_size, grid_width, grid_width) != obstacle_token_id).float()
        
        # Project to feasible paths
        projected_mask = shortest_path_projection(cost_grid, start_positions, end_positions, road_mask, grid_width)
        projected_mask_flat = projected_mask.view(batch_size, -1)
    
    # Straight-Through Estimator: use projected mask in forward, but gradients from original logits
    original_path_logits = logits[..., path_token_id]
    projected_path_logits = projected_mask_flat.detach() + (original_path_logits - original_path_logits.detach())
    
    # Binary cross-entropy loss on projected paths
    projection_loss = F.binary_cross_entropy_with_logits(
        projected_path_logits, projected_mask_flat.float(), reduction='mean'
    )
    
    return projection_loss * projection_weight, projected_mask_flat


def is_valid_solution(metrics, max_path_ratio=0.10, min_connectivity=0.8, min_valid_ratio=0.95):
    """Check if a solution meets validity criteria for ACT rewards"""
    return (
        metrics.get("start_end_connectivity", 0) >= min_connectivity and
        metrics.get("valid_path_ratio", 0) >= min_valid_ratio and
        metrics.get("path_ratio", 1.0) <= max_path_ratio
    )


# ============================================================================
# Enhanced ACTLossHead with Comprehensive Anti-Cheat System
# ============================================================================

class ACTLossHead(nn.Module):
    """
    Enhanced ACTLossHead with comprehensive anti-cheating system:
    
    LEVEL 1: Weighted cross-entropy (original)
    LEVEL 2: Hard constraints (block learning when cheating)
    LEVEL 3: Enhanced penalties (connectivity, sparsity) 
    LEVEL 4: Direct supervision (budget loss, coverage penalty)
    LEVEL 5: Structured projection (force feasible paths)
    LEVEL 6: ACT validity gating (only reward valid solutions)
    """
    def __init__(self, model: nn.Module, 
                 loss_type: str,
                 # Basic parameters
                 sparsity_penalty_weight: float = 100.0,
                 path_token_id: int = 9,
                 path_weight: float = 1000.0,
                 connectivity_weight: float = 50.0,
                 start_token_id: int = 7,
                 end_token_id: int = 8,
                 obstacle_token_id: int = 1,
                 grid_width: int = 40,
                 max_path_ratio: float = 0.1,
                 require_connectivity: bool = True,
                 # NEW: Enhanced anti-cheat parameters
                 budget_loss_weight: float = 10.0,
                 coverage_penalty_weight: float = 5.0,
                 structured_projection_weight: float = 2.0,
                 enable_projection: bool = True,
                 act_validity_gating: bool = True,
                 min_connectivity_for_reward: float = 0.8,
                 min_valid_path_ratio: float = 0.95):
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
        
        # NEW: Enhanced anti-cheat parameters
        self.budget_loss_weight = budget_loss_weight
        self.coverage_penalty_weight = coverage_penalty_weight
        self.structured_projection_weight = structured_projection_weight
        self.enable_projection = enable_projection
        self.act_validity_gating = act_validity_gating
        self.min_connectivity_for_reward = min_connectivity_for_reward
        self.min_valid_path_ratio = min_valid_path_ratio
        
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
        
        # LEVEL 2: HARD CONSTRAINT - Prevent "everything is path" exploit
        pred_path_mask = (predicted_tokens == self.path_token_id)
        path_ratio = pred_path_mask.float().mean()
        
        if path_ratio > self.max_path_ratio:  # >10% of grid is path
            # HARD CONSTRAINT: Minimal loss to prevent learning when cheating detected
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
                "blocked_learning": torch.tensor(1.0, device=outputs["logits"].device),
            }
            return new_carry, blocked_loss, metrics, {}, True  # Force halt when cheating

        # NORMAL TRAINING PATH - Comprehensive loss computation
        
        # Correctness computation with enhanced validity checks
        with torch.no_grad():
            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Compute comprehensive quality metrics
            start_end_connectivity = compute_start_end_connectivity(
                predicted_tokens, labels,
                start_token_id=self.start_token_id,
                end_token_id=self.end_token_id,
                path_token_id=self.path_token_id,
                grid_width=self.grid_width
            )
            
            valid_path_ratio = compute_valid_path_ratio(
                predicted_tokens, labels,
                path_token_id=self.path_token_id,
                obstacle_token_id=self.obstacle_token_id,
                grid_width=self.grid_width
            )
            
            path_efficiency = compute_path_efficiency(
                predicted_tokens, labels,
                path_token_id=self.path_token_id
            )
            
            # LEVEL 6: ACT Validity Gating - Only reward truly valid solutions
            if self.act_validity_gating:
                batch_metrics = {
                    "start_end_connectivity": start_end_connectivity,
                    "path_ratio": path_ratio.item(),
                    "valid_path_ratio": valid_path_ratio
                }
                
                # Override accuracy based on validity
                if not is_valid_solution(batch_metrics, 
                                       max_path_ratio=self.max_path_ratio,
                                       min_connectivity=self.min_connectivity_for_reward,
                                       min_valid_ratio=self.min_valid_path_ratio):
                    seq_is_correct = torch.zeros_like(seq_is_correct, dtype=torch.bool)
            
            # Standard connectivity requirement
            elif self.require_connectivity and start_end_connectivity < 0.5:
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

        # LEVEL 1: Main loss computation (weighted cross-entropy)
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

        # LEVEL 3: Enhanced sparsity penalty
        sparsity_penalty = torch.tensor(0.0, device=outputs["logits"].device)
        if self.sparsity_penalty_weight > 0:
            path_predictions = (predicted_tokens == self.path_token_id).float()
            sparsity_penalty = self.sparsity_penalty_weight * path_predictions.mean()
            metrics["sparsity_penalty"] = sparsity_penalty.detach()

        # LEVEL 3: Enhanced connectivity penalty
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

        # LEVEL 4: NEW - Path budget loss (direct supervision on token count)
        budget_loss = torch.tensor(0.0, device=outputs["logits"].device)
        if self.budget_loss_weight > 0:
            budget_loss_value = compute_path_budget_loss(
                predicted_tokens, labels,
                path_token_id=self.path_token_id,
                grid_size=self.grid_width * self.grid_width
            )
            budget_loss = self.budget_loss_weight * budget_loss_value
            metrics["budget_loss"] = budget_loss.detach()

        # LEVEL 4: NEW - Coverage penalty (penalize false positive paths)
        coverage_penalty = torch.tensor(0.0, device=outputs["logits"].device)
        if self.coverage_penalty_weight > 0:
            coverage_penalty_value = compute_coverage_penalty(
                predicted_tokens, labels,
                path_token_id=self.path_token_id
            )
            coverage_penalty = self.coverage_penalty_weight * coverage_penalty_value
            metrics["coverage_penalty"] = coverage_penalty.detach()
            metrics["over_coverage_ratio"] = torch.tensor(coverage_penalty_value, device=outputs["logits"].device)

        # LEVEL 5: NEW - Structured projection (force feasible paths)
        projection_loss = torch.tensor(0.0, device=outputs["logits"].device)
        projected_paths = None
        
        if self.enable_projection and self.structured_projection_weight > 0:
            try:
                projection_loss_value, projected_paths = apply_structured_projection(
                    outputs["logits"], labels,
                    path_token_id=self.path_token_id,
                    start_token_id=self.start_token_id,
                    end_token_id=self.end_token_id,
                    obstacle_token_id=self.obstacle_token_id,
                    grid_width=self.grid_width,
                    projection_weight=self.structured_projection_weight
                )
                projection_loss = projection_loss_value
                metrics["projection_loss"] = projection_loss.detach()
                
                # Track how much projection changed the prediction
                if projected_paths is not None:
                    original_path_mask = (predicted_tokens == self.path_token_id).float()
                    projection_diff = (projected_paths - original_path_mask).abs().sum(dim=-1).mean()
                    metrics["projection_diff"] = projection_diff.detach()
                    
            except Exception as e:
                # Fallback: if projection fails, continue without it
                print(f"Projection failed: {e}")
                projection_loss = torch.tensor(0.0, device=outputs["logits"].device)

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
            "blocked_learning": torch.tensor(0.0, device=outputs["logits"].device),  # Not blocked
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

        # TOTAL LOSS: Combination of all loss components
        total_loss = (lm_loss + 0.5 * (q_halt_loss + q_continue_loss) + 
                     sparsity_penalty + connectivity_penalty + 
                     budget_loss + coverage_penalty + projection_loss)

        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()