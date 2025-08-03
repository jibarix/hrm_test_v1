from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding


@dataclass
class HierarchicalReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class HierarchicalReasoningModel_ACTV1Carry:
    inner_carry: HierarchicalReasoningModel_ACTV1InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class HierarchicalReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"


class HierarchicalReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post Norm
        # Self Attention
        hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class HierarchicalReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[HierarchicalReasoningModel_ACTV1Block]):
        super().__init__()

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)

        return hidden_states


# ============================================================================
# Enhanced Validity Functions for ACT (imported from losses.py logic)
# ============================================================================

def compute_start_end_connectivity_fast(predicted_tokens, labels, start_token_id=7, end_token_id=8, path_token_id=9, grid_width=40):
    """
    Fast version of start-end connectivity check for ACT (single batch element)
    Returns 1.0 if connected, 0.0 if not connected
    """
    batch_size = predicted_tokens.shape[0]
    if batch_size == 0:
        return 0.0
    
    # Take first batch element for speed (ACT processes one at a time anyway)
    pred_grid = predicted_tokens[0].reshape(grid_width, grid_width)
    label_grid = labels[0].reshape(grid_width, grid_width)
    
    # Find start and end positions in labels
    start_positions = (label_grid == start_token_id).nonzero()
    end_positions = (label_grid == end_token_id).nonzero()
    
    if len(start_positions) == 0 or len(end_positions) == 0:
        return 0.0
        
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
    
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        
        if current == end_pos:
            return 1.0  # Found connection
            
        # Check 4-connected neighbors
        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
            nr, nc = current[0] + dr, current[1] + dc
            neighbor = (nr, nc)
            if (neighbor not in visited and 
                0 <= nr < grid_width and 0 <= nc < grid_width and
                neighbor in path_set):
                queue.append(neighbor)
    
    return 0.0  # No connection found


def compute_valid_path_ratio_fast(predicted_tokens, labels, path_token_id=9, obstacle_token_id=1, grid_width=40):
    """Fast version of valid path ratio check for ACT"""
    if predicted_tokens.shape[0] == 0:
        return 0.0
    
    pred_grid = predicted_tokens[0].reshape(grid_width, grid_width)
    label_grid = labels[0].reshape(grid_width, grid_width)
    
    # Find predicted path positions
    pred_path_positions = (pred_grid == path_token_id).nonzero()
    
    if len(pred_path_positions) == 0:
        return 1.0  # No paths predicted, so all are "valid"
        
    valid_count = 0
    for pos in pred_path_positions:
        r, c = pos[0].item(), pos[1].item()
        # Check if this position is not an obstacle in the input
        if label_grid[r, c] != obstacle_token_id:
            valid_count += 1
    
    return valid_count / len(pred_path_positions)


def is_valid_solution_fast(predicted_tokens, labels, max_path_ratio=0.10, min_connectivity=0.8, 
                          min_valid_ratio=0.95, path_token_id=9, start_token_id=7, 
                          end_token_id=8, obstacle_token_id=1, grid_width=40):
    """
    Fast validity check for ACT rewards
    Returns True if solution meets all validity criteria
    """
    if predicted_tokens.shape[0] == 0:
        return False
    
    # Check path ratio (sparsity)
    pred_path_mask = (predicted_tokens == path_token_id)
    path_ratio = pred_path_mask.float().mean().item()
    
    if path_ratio > max_path_ratio:
        return False
    
    # Check connectivity
    connectivity = compute_start_end_connectivity_fast(
        predicted_tokens, labels, start_token_id, end_token_id, path_token_id, grid_width
    )
    
    if connectivity < min_connectivity:
        return False
    
    # Check valid path ratio (paths on roads, not obstacles)
    valid_ratio = compute_valid_path_ratio_fast(
        predicted_tokens, labels, path_token_id, obstacle_token_id, grid_width
    )
    
    if valid_ratio < min_valid_ratio:
        return False
    
    return True


class HierarchicalReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        self.embed_scale  = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            raise NotImplementedError()

        # Reasoning Layers
        self.H_level = HierarchicalReasoningModel_ACTV1ReasoningModule(layers=[HierarchicalReasoningModel_ACTV1Block(self.config) for _i in range(self.config.H_layers)])
        self.L_level = HierarchicalReasoningModel_ACTV1ReasoningModule(layers=[HierarchicalReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])
        
        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: HierarchicalReasoningModel_ACTV1InnerCarry):
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: HierarchicalReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            for _H_step in range(self.config.H_cycles):
                for _L_step in range(self.config.L_cycles):
                    if not ((_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)):
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)

                if not (_H_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, z_L, **seq_info)

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step grad
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.H_level(z_H, z_L, **seq_info)

        # LM Outputs
        new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]

        # Q head
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class HierarchicalReasoningModel_ACTV1(nn.Module):
    """Enhanced ACT wrapper with validity-based rewards."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_ACTV1Config(**config_dict)
        self.inner = HierarchicalReasoningModel_ACTV1_Inner(self.config)
        
        # NEW: ACT validity gating parameters (can be set from config)
        self.act_validity_gating = getattr(self.config, 'act_validity_gating', True)
        self.min_connectivity_for_reward = getattr(self.config, 'min_connectivity_for_reward', 0.8)
        self.min_valid_path_ratio = getattr(self.config, 'min_valid_path_ratio', 0.95)
        self.max_path_ratio_for_reward = getattr(self.config, 'max_path_ratio', 0.1)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return HierarchicalReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: HierarchicalReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        
        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):
                # ENHANCED: Validity-based halt signal instead of raw accuracy
                
                if self.act_validity_gating:
                    # NEW: Use validity-based rewards instead of raw accuracy
                    predicted_tokens = torch.argmax(logits, dim=-1)
                    
                    # Check if current solution is valid for each batch element
                    validity_rewards = []
                    for b in range(predicted_tokens.shape[0]):
                        is_valid = is_valid_solution_fast(
                            predicted_tokens[b:b+1], 
                            new_current_data["labels"][b:b+1],
                            max_path_ratio=self.max_path_ratio_for_reward,
                            min_connectivity=self.min_connectivity_for_reward,
                            min_valid_ratio=self.min_valid_path_ratio,
                            path_token_id=9, start_token_id=7, end_token_id=8, obstacle_token_id=1
                        )
                        validity_rewards.append(1.0 if is_valid else 0.0)
                    
                    validity_rewards = torch.tensor(validity_rewards, device=logits.device, dtype=torch.float32)
                    
                    # Halt signal based on validity, not raw Q-values
                    # Only halt if solution is valid AND Q-halt > Q-continue
                    valid_solutions = validity_rewards > 0.5
                    q_wants_halt = q_halt_logits > q_continue_logits
                    halted = halted | (valid_solutions & q_wants_halt)
                    
                else:
                    # Original behavior: halt based on Q-values only
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration: minimum steps before considering halting
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                # ENHANCED: Compute target Q using validity instead of raw accuracy
                if self.act_validity_gating:
                    # Get next step predictions for target Q computation
                    next_carry, next_logits, (next_q_halt_logits, next_q_continue_logits) = self.inner(new_inner_carry, new_current_data)
                    next_predicted_tokens = torch.argmax(next_logits, dim=-1)
                    
                    # Compute next step validity rewards
                    next_validity_rewards = []
                    for b in range(next_predicted_tokens.shape[0]):
                        is_valid = is_valid_solution_fast(
                            next_predicted_tokens[b:b+1], 
                            new_current_data["labels"][b:b+1],
                            max_path_ratio=self.max_path_ratio_for_reward,
                            min_connectivity=self.min_connectivity_for_reward,
                            min_valid_ratio=self.min_valid_path_ratio,
                            path_token_id=9, start_token_id=7, end_token_id=8, obstacle_token_id=1
                        )
                        next_validity_rewards.append(1.0 if is_valid else 0.0)
                    
                    next_validity_rewards = torch.tensor(next_validity_rewards, device=logits.device, dtype=torch.float32)
                    
                    # Target Q-continue: reward if next step would be valid
                    outputs["target_q_continue"] = torch.sigmoid(
                        torch.where(
                            is_last_step, 
                            next_validity_rewards,  # Use validity reward instead of Q-halt
                            torch.maximum(next_q_halt_logits, next_q_continue_logits)
                        )
                    )
                    
                    # Store validity rewards for logging in loss function
                    outputs["validity_rewards"] = validity_rewards
                    outputs["next_validity_rewards"] = next_validity_rewards
                    
                else:
                    # Original Q-learning target computation
                    next_q_halt_logits, next_q_continue_logits = self.inner(new_inner_carry, new_current_data)[-1]
                    
                    # Use raw accuracy instead of validity (original behavior)
                    predicted_tokens = torch.argmax(logits, dim=-1)
                    seq_is_correct = (predicted_tokens == new_current_data["labels"]).all(dim=-1)
                    
                    outputs["target_q_continue"] = torch.sigmoid(
                        torch.where(
                            is_last_step, 
                            seq_is_correct.float(),
                            torch.maximum(next_q_halt_logits, next_q_continue_logits)
                        )
                    )

        return HierarchicalReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs