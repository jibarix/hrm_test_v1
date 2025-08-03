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


def is_valid_solution(metrics, max_path_ratio=0.10, min_connectivity=0.8, min_valid_ratio=0.95):
    """Check if a solution meets validity criteria for ACT rewards"""
    return (
        metrics.get("start_end_connectivity", 0) >= min_connectivity and
        metrics.get("valid_path_ratio", 0) >= min_valid_ratio and
        metrics.get("path_ratio", 1.0) <= max_path_ratio
    )


# ============================================================================
# NEW: Curriculum Learning System
# ============================================================================

class CurriculumScheduler:
    """Manages progressive constraint tightening during training"""
    
    def __init__(self, config):
        self.config = config
        self.warmup_steps = getattr(config, 'warmup_steps', 0)
        self.curriculum_stages = getattr(config, 'curriculum_stages', [])
        self.current_stage = None
        self._last_logged_step = -1
        
    def get_current_constraints(self, step):
        """Get constraints for current training step"""
        # Warmup phase - no constraints
        if step < self.warmup_steps:
            return {
                'max_path_ratio': 1.0,  # Allow anything
                'path_weight': 1.0,     # Minimal penalty
                'connectivity_weight': 0.0,
                'require_connectivity': False,
                'enable_batch_rejection': False,
                'sparsity_penalty_weight': 0.0,
                'budget_loss_weight': 0.0,
                'coverage_penalty_weight': 0.0,
                'stage_name': 'warmup'
            }
        
        # Find current curriculum stage
        current_stage = self.curriculum_stages[0] if self.curriculum_stages else {}
        stage_index = 0
        
        for i, stage in enumerate(self.curriculum_stages):
            if step >= stage['start_step']:
                current_stage = stage
                stage_index = i
            else:
                break
        
        # FIXED: Create a mutable copy to avoid OmegaConf struct error
        if hasattr(current_stage, '_content'):  # OmegaConf DictConfig
            current_constraints = dict(current_stage)
        else:  # Regular dict
            current_constraints = current_stage.copy()
        
        # Add stage identification metadata
        current_constraints['stage_name'] = f'stage_{stage_index + 1}'
        current_constraints['stage_index'] = stage_index
                
        return current_constraints
        
    def should_apply_constraint(self, step, constraint_name):
        """Check if specific constraint should be applied at current step"""
        constraints = self.get_current_constraints(step)
        return constraints.get(constraint_name, False)
    
    def log_stage_transition(self, step, constraints, rank=0):
        """Log curriculum stage transitions"""
        if rank == 0 and step != self._last_logged_step:
            stage_name = constraints.get('stage_name', 'unknown')
            max_ratio = constraints.get('max_path_ratio', 1.0)
            path_weight = constraints.get('path_weight', 1.0)
            connectivity_weight = constraints.get('connectivity_weight', 0.0)
            
            print(f"🎓 [Step {step}] Curriculum {stage_name}: "
                  f"max_path_ratio={max_ratio:.2f}, "
                  f"path_weight={path_weight:.0f}, "
                  f"connectivity_weight={connectivity_weight:.1f}")
            
            self._last_logged_step = step


# ============================================================================
# Enhanced ACTLossHead with Curriculum Learning Support
# ============================================================================

class ACTLossHead(nn.Module):
    """
    Enhanced ACTLossHead with curriculum learning and comprehensive anti-cheat system
    """
    def __init__(self, model: nn.Module, 
                 loss_type: str,
                 # Curriculum learning parameters
                 curriculum_learning: bool = False,
                 warmup_steps: int = 0,
                 curriculum_stages: list = None,
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
                 # Enhanced anti-cheat parameters
                 budget_loss_weight: float = 10.0,
                 coverage_penalty_weight: float = 5.0,
                 structured_projection_weight: float = 2.0,
                 enable_projection: bool = False,  # Disabled due to bfloat16 issues
                 act_validity_gating: bool = True,
                 min_connectivity_for_reward: float = 0.8,
                 min_valid_path_ratio: float = 0.95,
                 **kwargs):
        super().__init__()
        self.model = model
        self.loss_fn = weighted_cross_entropy if loss_type == "weighted_cross_entropy" else globals()[loss_type]
        self.loss_type = loss_type
        
        # Store all original parameters for curriculum
        self.original_params = {
            'sparsity_penalty_weight': sparsity_penalty_weight,
            'path_token_id': path_token_id,
            'path_weight': path_weight,
            'connectivity_weight': connectivity_weight,
            'start_token_id': start_token_id,
            'end_token_id': end_token_id,
            'obstacle_token_id': obstacle_token_id,
            'grid_width': grid_width,
            'max_path_ratio': max_path_ratio,
            'require_connectivity': require_connectivity,
            'budget_loss_weight': budget_loss_weight,
            'coverage_penalty_weight': coverage_penalty_weight,
            'structured_projection_weight': structured_projection_weight,
            'enable_projection': enable_projection,
            'act_validity_gating': act_validity_gating,
            'min_connectivity_for_reward': min_connectivity_for_reward,
            'min_valid_path_ratio': min_valid_path_ratio,
        }
        self.original_params.update(kwargs)
        
        # Curriculum learning setup
        self.curriculum_learning = curriculum_learning
        if curriculum_learning:
            # Create a config object for curriculum scheduler
            curriculum_config = type('Config', (), {
                'warmup_steps': warmup_steps,
                'curriculum_stages': curriculum_stages or []
            })()
            self.curriculum_scheduler = CurriculumScheduler(curriculum_config)
        else:
            self.curriculum_scheduler = None
            
        # Initialize with starting parameters
        self._update_parameters(self.original_params)
        
        # Training step counter
        self.current_step = 0
        self._last_stage_index = -1
        
    def _update_parameters(self, params):
        """Update internal parameters (used for curriculum)"""
        # Token IDs (static)
        self.path_token_id = params.get('path_token_id', 9)
        self.start_token_id = params.get('start_token_id', 7)
        self.end_token_id = params.get('end_token_id', 8)
        self.obstacle_token_id = params.get('obstacle_token_id', 1)
        self.grid_width = params.get('grid_width', 40)
        
        # Dynamic parameters (updated by curriculum)
        self.path_weight = params.get('path_weight', 40.0)
        self.max_path_ratio = params.get('max_path_ratio', 0.1)
        self.connectivity_weight = params.get('connectivity_weight', 1.0)
        self.require_connectivity = params.get('require_connectivity', True)
        self.sparsity_penalty_weight = params.get('sparsity_penalty_weight', 0.0)
        self.budget_loss_weight = params.get('budget_loss_weight', 0.0)
        self.coverage_penalty_weight = params.get('coverage_penalty_weight', 0.0)
        
        # ACT parameters (can be curriculum-adjusted)
        self.act_validity_gating = params.get('act_validity_gating', True)
        self.min_connectivity_for_reward = params.get('min_connectivity_for_reward', 0.8)
        self.min_valid_path_ratio = params.get('min_valid_path_ratio', 0.95)
        
        # Static parameters
        self.structured_projection_weight = params.get('structured_projection_weight', 0.0)
        self.enable_projection = params.get('enable_projection', False)
        
    def set_training_step(self, step, rank=0):
        """Update current training step and adjust constraints if using curriculum"""
        self.current_step = step
        
        if self.curriculum_learning and self.curriculum_scheduler:
            # Get constraints for current step
            current_constraints = self.curriculum_scheduler.get_current_constraints(step)
            
            # Update parameters based on curriculum stage
            updated_params = self.original_params.copy()
            updated_params.update(current_constraints)
            self._update_parameters(updated_params)
            
            # Log stage transitions
            current_stage_index = current_constraints.get('stage_index', -1)
            if current_stage_index != self._last_stage_index:
                self.curriculum_scheduler.log_stage_transition(step, current_constraints, rank)
                self._last_stage_index = current_stage_index
    
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        
        # Model logits
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        # Get predicted tokens for metrics and anti-cheating checks
        predicted_tokens = torch.argmax(outputs["logits"], dim=-1)
        
        # Compute loss_divisor early (needed for both normal and cheat penalty paths)
        mask = labels != IGNORE_LABEL_ID
        loss_counts = mask.sum(-1)
        loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division
        
        # CURRICULUM-AWARE CONSTRAINT CHECKING
        pred_path_mask = (predicted_tokens == self.path_token_id)
        path_ratio = pred_path_mask.float().mean()
        
        # Check if we should apply hard constraints (curriculum-aware)
        should_block = False
        current_max_ratio = self.max_path_ratio
        enable_blocking = True
        
        if self.curriculum_learning and self.curriculum_scheduler:
            constraints = self.curriculum_scheduler.get_current_constraints(self.current_step)
            enable_blocking = constraints.get('enable_batch_rejection', False)
            current_max_ratio = constraints.get('max_path_ratio', 1.0)
            should_block = enable_blocking and (path_ratio > current_max_ratio)
        else:
            # Original behavior
            should_block = path_ratio > self.max_path_ratio
        
        if should_block:
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
                "curriculum_step": torch.tensor(float(self.current_step), device=outputs["logits"].device),
                "curriculum_max_path_ratio": torch.tensor(current_max_ratio, device=outputs["logits"].device),
                "curriculum_enable_blocking": torch.tensor(float(enable_blocking), device=outputs["logits"].device),
            }
            return new_carry, blocked_loss, metrics, {}, True  # Force halt when cheating

        # NORMAL TRAINING PATH - Comprehensive loss computation with curriculum awareness
        
        # Correctness computation with curriculum-aware validity checks
        with torch.no_grad():
            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Compute comprehensive quality metrics
            start_end_connectivity = 1.0  # Default if not checking
            if self.require_connectivity:
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
            
            # CURRICULUM-AWARE ACT Validity Gating
            if self.act_validity_gating:
                batch_metrics = {
                    "start_end_connectivity": start_end_connectivity,
                    "path_ratio": path_ratio.item(),
                    "valid_path_ratio": valid_path_ratio
                }
                
                # Use curriculum-adjusted thresholds
                if not is_valid_solution(batch_metrics, 
                                       max_path_ratio=current_max_ratio,
                                       min_connectivity=self.min_connectivity_for_reward,
                                       min_valid_ratio=self.min_valid_path_ratio):
                    seq_is_correct = torch.zeros_like(seq_is_correct, dtype=torch.bool)
            
            # Standard connectivity requirement (curriculum-aware)
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

        # CURRICULUM-ADJUSTED MAIN LOSS COMPUTATION
        if self.loss_type == "weighted_cross_entropy":
            lm_loss = (weighted_cross_entropy(
                outputs["logits"], labels, 
                path_token_id=self.path_token_id,
                path_weight=self.path_weight,  # Now curriculum-adjusted
                ignore_index=IGNORE_LABEL_ID
            ) / loss_divisor).sum()
        else:
            lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor).sum()
            
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")

        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })

        # CURRICULUM-ADJUSTED PENALTY SYSTEM
        
        # Level 3: Enhanced sparsity penalty
        sparsity_penalty = torch.tensor(0.0, device=outputs["logits"].device)
        if self.sparsity_penalty_weight > 0:
            path_predictions = (predicted_tokens == self.path_token_id).float()
            sparsity_penalty = self.sparsity_penalty_weight * path_predictions.mean()
            metrics["sparsity_penalty"] = sparsity_penalty.detach()

        # Level 3: Enhanced connectivity penalty
        connectivity_penalty = torch.tensor(0.0, device=outputs["logits"].device)
        if self.connectivity_weight > 0:
            connectivity_penalty_value = compute_connectivity_loss(
                predicted_tokens, 
                path_token_id=self.path_token_id,
                grid_width=self.grid_width,
                connectivity_weight=self.connectivity_weight  # Now curriculum-adjusted
            )
            connectivity_penalty = torch.tensor(connectivity_penalty_value, device=outputs["logits"].device)
            metrics["connectivity_penalty"] = connectivity_penalty.detach()

        # Level 4: Path budget loss (direct supervision on token count)
        budget_loss = torch.tensor(0.0, device=outputs["logits"].device)
        if self.budget_loss_weight > 0:
            budget_loss_value = compute_path_budget_loss(
                predicted_tokens, labels,
                path_token_id=self.path_token_id,
                grid_size=self.grid_width * self.grid_width
            )
            budget_loss = self.budget_loss_weight * budget_loss_value
            metrics["budget_loss"] = budget_loss.detach()

        # Level 4: Coverage penalty (penalize false positive paths)
        coverage_penalty = torch.tensor(0.0, device=outputs["logits"].device)
        if self.coverage_penalty_weight > 0:
            coverage_penalty_value = compute_coverage_penalty(
                predicted_tokens, labels,
                path_token_id=self.path_token_id
            )
            coverage_penalty = self.coverage_penalty_weight * coverage_penalty_value
            metrics["coverage_penalty"] = coverage_penalty.detach()
            metrics["over_coverage_ratio"] = coverage_penalty_value.clone().detach()

        # Level 5: Structured projection (disabled for now due to bfloat16 issues)
        projection_loss = torch.tensor(0.0, device=outputs["logits"].device)
        if self.enable_projection and self.structured_projection_weight > 0:
            # Implementation would go here, but disabled for stability
            metrics["projection_loss"] = projection_loss.detach()

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
            
            # Curriculum tracking metrics
            "curriculum_step": torch.tensor(float(self.current_step), device=outputs["logits"].device),
            "curriculum_max_path_ratio": torch.tensor(current_max_ratio, device=outputs["logits"].device),
            "curriculum_path_weight": torch.tensor(self.path_weight, device=outputs["logits"].device),
            "curriculum_connectivity_weight": torch.tensor(self.connectivity_weight, device=outputs["logits"].device),
            "curriculum_enable_blocking": torch.tensor(float(enable_blocking), device=outputs["logits"].device),
        })

        # Q continue (bootstrapping target loss)
        q_continue_loss = torch.tensor(0.0, device=outputs["logits"].device)
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")
            metrics["q_continue_loss"] = q_continue_loss.detach()

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        # TOTAL LOSS: Combination of all curriculum-adjusted loss components
        total_loss = (lm_loss + 0.5 * (q_halt_loss + q_continue_loss) + 
                     sparsity_penalty + connectivity_penalty + 
                     budget_loss + coverage_penalty + projection_loss)

        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()


# ============================================================================
# Utility Functions for Training Integration
# ============================================================================

def update_loss_curriculum_step(loss_head, step, rank=0):
    """Call this in your training loop to update curriculum step"""
    if hasattr(loss_head, 'set_training_step'):
        loss_head.set_training_step(step, rank)
    elif hasattr(loss_head, '_orig_mod') and hasattr(loss_head._orig_mod, 'set_training_step'):
        # Handle torch.compile wrapped models
        loss_head._orig_mod.set_training_step(step, rank)


def get_curriculum_status(loss_head):
    """Get current curriculum status for logging"""
    if hasattr(loss_head, 'curriculum_scheduler') and loss_head.curriculum_scheduler:
        constraints = loss_head.curriculum_scheduler.get_current_constraints(loss_head.current_step)
        return {
            'step': loss_head.current_step,
            'stage': constraints.get('stage_name', 'unknown'),
            'max_path_ratio': constraints.get('max_path_ratio', 1.0),
            'path_weight': constraints.get('path_weight', 1.0),
            'connectivity_weight': constraints.get('connectivity_weight', 0.0),
            'enable_blocking': constraints.get('enable_batch_rejection', False)
        }
    return None