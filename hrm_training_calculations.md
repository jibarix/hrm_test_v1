# HRM Training Step Calculations and Configuration Guide

## Overview

This document explains how training steps, epochs, and evaluation cycles work in the Hierarchical Reasoning Model (HRM), based on analysis of the original paper codebase and empirical testing.

## Key Configuration Principle: 10:1 Ratio

**Critical Rule**: Always maintain the paper's evaluation frequency ratio:
```yaml
eval_interval = epochs / 10  # 10% evaluation frequency
```

### Original Paper Configuration
```yaml
epochs: 100000
eval_interval: 10000
# Ratio: 10000/100000 = 0.1 (10% evaluation frequency)
# Result: 10 total evaluation cycles
```

### Scaled Configuration Examples
```yaml
# Small scale (recommended for testing)
epochs: 100
eval_interval: 10

# Medium scale 
epochs: 1000
eval_interval: 100

# Large scale (closer to paper)
epochs: 10000
eval_interval: 1000
```

## HRM's Group-Based Epoch Definition

### Traditional vs HRM Epochs

**Traditional Deep Learning:**
- 1 epoch = 1 complete pass through entire dataset
- 960 examples ÷ 32 batch_size = 30 steps per epoch

**HRM's Group-Based Sampling:**
- 1 epoch = 1 pass through all **groups**, sampling from each
- Groups: 240 (base scenarios)
- Examples per group: 4 (vehicle variants)
- Samples per epoch: ~240 (one sample per group)
- Steps per epoch: 240 ÷ 32 = 7.5 ≈ 8 steps

## Step Calculation Formula

### Basic Calculation
```
Examples per HRM epoch = Number of groups ≈ Number of base scenarios
Steps per HRM epoch = Examples per HRM epoch ÷ Global batch size
Steps per evaluation cycle = Steps per HRM epoch × eval_interval
Total steps = (epochs ÷ eval_interval) × Steps per evaluation cycle
```

### Example: City Logistics Dataset
```yaml
# Dataset properties
Total examples: 960
Base scenarios (groups): 240
Vehicle variants per scenario: 4
Global batch size: 32

# Configuration
epochs: 100
eval_interval: 10

# Calculations
Examples per HRM epoch: ~240 (one per group)
Steps per HRM epoch: 240 ÷ 32 = 7.5 ≈ 8 steps
Steps per evaluation cycle: 8 × 10 = 80 steps  
Number of evaluation cycles: 100 ÷ 10 = 10 cycles
Total training steps: 10 × 80 = 800 steps
```

## Training Progress Patterns

### Correct Pattern (eval_interval: 10)
```
[Rank 0, World Size 1]: Epoch 0   | Step 75   # First evaluation cycle
[Rank 0, World Size 1]: Epoch 10  | Step 150  # Second evaluation cycle  
[Rank 0, World Size 1]: Epoch 20  | Step 225  # Third evaluation cycle
```
- **Continuous training** for 10 epochs between evaluations
- **~75-80 steps** per evaluation cycle

### Broken Pattern (eval_interval: 1)
```
[Rank 0, World Size 1]: Epoch 0   | Step 7    # Frequent interruptions
[Rank 0, World Size 1]: Epoch 1   | Step 14   # Poor data utilization
[Rank 0, World Size 1]: Epoch 2   | Step 21   # Slow training
```
- **Evaluation every epoch** interrupts group sampling
- **Only ~7 steps** before restart
- **Inefficient** data loading and GPU utilization

## Code Implementation Details

### Dataset Configuration (`puzzle_dataset.py`)
```python
def _iter_train(self):
    for set_name, dataset in self._data.items():
        # Group-based sampling for epochs_per_iter cycles
        group_order = np.concatenate([
            rng.permutation(dataset["group_indices"].size - 1) 
            for _i in range(self.config.epochs_per_iter)
        ])
        
        # Sample one example per group per epoch
        start_index, batch_indices, batch_puzzle_indices = _sample_batch(
            rng, group_order, puzzle_indices, group_indices, 
            start_index, global_batch_size
        )
```

### Training Loop (`pretrain.py`)
```python
# Critical calculation
train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
total_iters = config.epochs // train_epochs_per_iter

# Example with eval_interval: 10
# train_epochs_per_iter = 10
# total_iters = 100 // 10 = 10 evaluation cycles
```

## Performance Implications

### Correct Configuration Benefits
- **Continuous training**: Full utilization of group sampling
- **Better convergence**: Longer training cycles allow proper gradient accumulation
- **Efficient GPU usage**: Reduced evaluation overhead
- **Stable learning**: Consistent data distribution per cycle

### Incorrect Configuration Problems
- **Premature sampling termination**: Group sampling interrupted too frequently
- **Poor data utilization**: Only subset of dataset seen per cycle
- **Slow training**: Excessive evaluation overhead
- **Unstable gradients**: Short training cycles prevent proper optimization

## Dataset Structure Requirements

HRM expects datasets with group structure:
```
all__inputs.npy: (N, sequence_length)           # All examples
all__labels.npy: (N, sequence_length)           # All labels  
all__puzzle_identifiers.npy: (N,)               # Puzzle type IDs
all__puzzle_indices.npy: (N+1,)                 # Example boundaries
all__group_indices.npy: (G+1,)                  # Group boundaries
```

Where:
- **N**: Total number of examples
- **G**: Number of groups (base scenarios)
- **Group size**: N ÷ G examples per group (e.g., vehicle variants)

## Configuration Validation

### Check Your Setup
```python
# Verify group structure
groups = len(group_indices) - 1
examples_per_group = total_examples / groups
steps_per_epoch = groups / global_batch_size
steps_per_cycle = steps_per_epoch * eval_interval

print(f"Groups: {groups}")
print(f"Examples per group: {examples_per_group}")  
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Steps per evaluation cycle: {steps_per_cycle}")
```

### Recommended Configurations

| Scale | Examples | Groups | Batch Size | Epochs | Eval Interval | Steps per Cycle |
|-------|----------|--------|------------|---------|---------------|-----------------|
| Small | 960      | 240    | 32         | 100     | 10            | 75              |
| Medium| 960      | 240    | 32         | 1000    | 100           | 750             |
| Large | 960      | 240    | 32         | 10000   | 1000          | 7500            |

## Troubleshooting

### Symptom: Too few steps per epoch
- **Cause**: eval_interval too small (frequent evaluation)
- **Fix**: Increase eval_interval to maintain 10:1 ratio

### Symptom: Training stops early  
- **Cause**: total_iters calculation incorrect
- **Fix**: Verify epochs divisible by eval_interval

### Symptom: Slow training progress
- **Cause**: Excessive evaluation overhead
- **Fix**: Reduce evaluation frequency (larger eval_interval)

### Symptom: Poor convergence
- **Cause**: Training cycles too short for group sampling
- **Fix**: Use longer continuous training periods (larger eval_interval)

## Batch Size Impact Analysis

### Warmup Phase: Examples vs Gradient Updates

**Key Principle**: Warmup steps count **examples seen**, not gradient updates.

```yaml
# Example: City Logistics (960 examples)
lr_warmup_steps: 1920  # Always 2x examples, regardless of batch size

# Different batch sizes, same warmup:
global_batch_size: 32   # 1920 ÷ 32 = 60 gradient updates during warmup
global_batch_size: 64   # 1920 ÷ 64 = 30 gradient updates during warmup  
global_batch_size: 128  # 1920 ÷ 128 = 15 gradient updates during warmup
```

### Batch Size Effects on Training Phases

| Phase | Metric | Small Batch (32) | Large Batch (128) | Impact |
|-------|--------|------------------|-------------------|---------|
| **Warmup** | Examples seen | 1920 | 1920 | ✅ Same |
| **Warmup** | Gradient updates | 60 | 15 | ⚠️ Fewer updates |
| **Warmup** | LR ramp duration | Longer | Shorter | ⚠️ Steep LR increase |
| **Main Training** | Steps per epoch | 7.5 | 1.9 | 📉 Much fewer |
| **Main Training** | GPU utilization | Lower | Higher | 📈 Better efficiency |
| **Main Training** | Gradient noise | Higher | Lower | ⚖️ Trade-off |

### Detailed Batch Size Scenarios

#### Small Batch Size (32) - Your Current Setup
```yaml
global_batch_size: 32
lr_warmup_steps: 1920

# Warmup phase
Gradient updates during warmup: 1920 ÷ 32 = 60 updates
LR increase per update: target_lr ÷ 60 = gradual ramp

# Main training phase  
Steps per epoch: 240 groups ÷ 32 = 7.5 steps
Steps per eval cycle (10 epochs): 75 steps
GPU memory usage: Lower
Training stability: Higher (more gradient updates)
```

#### Large Batch Size (128) - Memory Permitting
```yaml
global_batch_size: 128
lr_warmup_steps: 1920  # Same! Still 2x data exposure

# Warmup phase
Gradient updates during warmup: 1920 ÷ 128 = 15 updates  
LR increase per update: target_lr ÷ 15 = steep ramp ⚠️

# Main training phase
Steps per epoch: 240 groups ÷ 128 = 1.875 ≈ 2 steps
Steps per eval cycle (10 epochs): ~20 steps
GPU memory usage: Higher
Training stability: Lower (fewer gradient updates)
```

#### Paper Scale Batch Size (768) - Original Setup
```yaml
global_batch_size: 768
lr_warmup_steps: 1920

# Warmup phase  
Gradient updates during warmup: 1920 ÷ 768 = 2.5 updates ⚠️⚠️
LR increase per update: target_lr ÷ 2.5 = very steep ramp

# Main training phase
Steps per epoch: 240 groups ÷ 768 = 0.31 steps
Fractional steps per epoch: Problematic for group sampling
```

### Critical Batch Size Constraints

#### Upper Limit: Group Size Constraint
```
Maximum effective batch_size ≤ Number of groups
For City Logistics: batch_size ≤ 240 groups
```

**Why**: HRM's group sampling cannot provide more examples per batch than available groups.

#### Warmup Stability Constraint
```
Minimum gradient updates during warmup ≥ 10-20 updates
This requires: batch_size ≤ lr_warmup_steps ÷ 10
For City Logistics: batch_size ≤ 1920 ÷ 10 = 192
```

**Why**: Too few gradient updates during warmup cause unstable LR ramping.

### Recommended Batch Size Ranges

| Dataset Size | Groups | Recommended Batch Size | Reasoning |
|--------------|--------|------------------------|-----------|
| 960 (City Logistics) | 240 | 32-64 | Balanced updates & stability |
| ~1000 (ARC) | ~250 | 64-128 | Paper used 768 (special case) |
| 4000+ | 1000+ | 128-256 | Can handle larger batches |

### Batch Size Optimization Strategy

1. **Start Conservative**: Use smaller batch size for stability
2. **Monitor GPU**: Increase batch size until memory limit
3. **Check Warmup**: Ensure ≥15 gradient updates during warmup
4. **Validate Groups**: Batch size should not exceed group count
5. **Test Convergence**: Larger batches may need LR adjustment

### Configuration Examples

#### Conservative (Recommended for Testing)
```yaml
global_batch_size: 32
lr_warmup_steps: 1920
# Results: 60 warmup updates, 7.5 steps/epoch
```

#### Balanced (Good Performance/Memory Trade-off)
```yaml
global_batch_size: 64  
lr_warmup_steps: 1920
# Results: 30 warmup updates, 3.75 steps/epoch
```

#### Aggressive (Maximum Memory Utilization)
```yaml
global_batch_size: 128
lr_warmup_steps: 1920
# Results: 15 warmup updates, 1.875 steps/epoch
```

### Warning: Batch Size Too Large

**Symptoms of excessive batch size**:
- Very few gradient updates during warmup (< 10)
- Unstable early training (steep LR ramp)
- Fractional steps per epoch (< 1.0)
- Poor group sampling diversity

**Fix**: Reduce batch size to maintain stable warmup and adequate steps per epoch.

## Critical Warmup Configuration

### Paper's Warmup Strategy: "2x Data Exposure Rule"

The original paper configuration reveals a crucial warmup principle:

```yaml
# Original paper config
data_path: data/arc-aug-1000    # ~1000 examples
global_batch_size: 768
lr_warmup_steps: 2000           # CRITICAL: 2x the dataset size
```

**Key insight**: `lr_warmup_steps = 2 × dataset_size`

### Warmup Calculation Formula

```
lr_warmup_steps = 2 × total_examples
```

This ensures the model sees **every example twice** during the warmup period before entering the main training phase.

**CRITICAL**: Warmup steps are **independent of batch size** - they represent total examples seen, not gradient updates.

### Dataset-Specific Warmup Examples

| Dataset | Total Examples | Warmup Steps | Reasoning |
|---------|----------------|--------------|-----------|
| ARC (paper) | ~1000 | 2000 | 2x data exposure |
| City Logistics | 960 | 1920 | 2x data exposure |
| Custom dataset | N | 2×N | Universal rule |

### Why 2x Data Exposure is Essential

1. **Gradient Stability**: HRM's hierarchical modules need stable gradient flow
2. **Q-Learning Bootstrap**: ACT's halting mechanism requires initial experience
3. **Sparse Embedding Init**: Puzzle embeddings need multiple exposures to initialize properly
4. **Hierarchical Convergence**: H and L modules need time to establish cooperation

### Implementation in Your Config

```yaml
# For City Logistics dataset (960 examples)
lr_warmup_steps: 1920  # 2 × 960 examples

# Alternative calculation for safety margin
lr_warmup_steps: 2000  # Slightly above 2x for robustness
```

### Warmup vs Main Training Timeline

```
Steps 1-1920:    Warmup period (LR: 0 → target_lr)
                 Model sees each example ~2 times
                 Establishes stable hierarchical dynamics

Steps 1920+:    Main training (LR: target_lr → min_lr)
                Group-based sampling with established dynamics
```

### Code Implementation Details

From `pretrain.py`:
```python
def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, 
    num_training_steps: int, min_ratio: float = 0.0
):
    if current_step < num_warmup_steps:
        # Linear warmup: 0 → base_lr over warmup_steps
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))
    
    # Main training: cosine decay
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))))
```

### Paper Methodology Validation

The paper comment in your logistics config reveals this principle:
```yaml
# Paper methodology validation:
# - Paper: 2000 warmup steps with 960 samples = 2.08x data passes
# - Our config: 1920 warmup steps with 960 samples = 2.0x data passes  
# - This matches the "seeing data 2x before pretrain" observation
```

### Warning: Insufficient Warmup Consequences

**Too few warmup steps** (`< 2 × dataset_size`):
- Unstable hierarchical convergence
- Poor Q-learning initialization
- Gradient explosion in early training
- Suboptimal puzzle embedding learning

**Example of broken config**:
```yaml
lr_warmup_steps: 100  # Only 0.1x dataset size - BROKEN
```

## Summary

The HRM uses a **group-based epoch definition** rather than traditional full-dataset epochs. This requires:

1. **Maintain 10:1 ratio**: `eval_interval = epochs / 10`
2. **Use 2x warmup rule**: `lr_warmup_steps = 2 × total_examples`
3. **Understand group sampling**: Each epoch samples ~1 example per group
4. **Calculate steps correctly**: Based on groups, not total examples
5. **Allow continuous training**: Longer cycles between evaluations

Following these principles ensures optimal HRM training performance and proper utilization of the hierarchical reasoning architecture.