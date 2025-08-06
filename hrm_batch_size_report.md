# HRM Training Configuration Report: Batch Size Scaling and Sampling Regimes

**Technical Report on Hierarchical Reasoning Model Training Dynamics**

*Analysis of the ARC Configuration Paradox and Guidelines for Optimal Batch Size Selection*

---

## Executive Summary

This report addresses a critical configuration paradox in Hierarchical Reasoning Model (HRM) training: the original ARC-AGI experiments used a global batch size (768) that significantly exceeds the theoretical constraint imposed by the group-based sampling architecture. Through detailed analysis, we identify two distinct sampling regimes in HRM training and provide practical guidelines for batch size optimization across different dataset characteristics.

**Key Findings:**
- ARC represents a special case where `batch_size > number_of_groups` creates a "survey sampling" regime
- The 768 batch size provides comprehensive gradient signals covering nearly all puzzle types per update
- Hardware constraints (8-GPU scaling) aligned favorably with ARC's group structure
- Different reasoning tasks benefit from different batch size strategies

---

## Problem Statement

### The Apparent Contradiction

The original HRM paper configuration for ARC-AGI-1 presents several constraints that appear mutually incompatible:

```yaml
# Original ARC Configuration
data_path: data/arc-aug-1000      # ~1000 examples
global_batch_size: 768            # 76.8% of dataset per batch
lr_warmup_steps: 2000            # 2x dataset exposure rule

# Derived metrics:
gradient_updates_during_warmup: 2.6    # 2000 ÷ 768
batch_coverage: 76.8%                  # 768 ÷ 1000
```

This configuration appears to violate established HRM training principles:

1. **Group Constraint**: `batch_size ≤ number_of_groups` (theoretical limit)
2. **Warmup Stability**: Minimum 10-20 gradient updates for stable hierarchical initialization
3. **Data Efficiency**: Balanced exposure across puzzle types

### Research Questions

1. How does ARC's group structure differ from typical HRM datasets?
2. What sampling strategy enables batch sizes exceeding group counts?
3. How does warmup behave with comprehensive gradient signals?
4. What are the generalization principles for other datasets?

---

## Analysis: ARC as a Special Case

### Dataset Structure Analysis

**ARC-AGI-1 Composition:**
- **Source data**: 400 original ARC puzzles + ConceptARC corpus
- **Total groups**: ~400-500 (each original puzzle file = 1 group)
- **Augmentation strategy**: Rotations, flips, color permutations within groups
- **Final dataset**: ~960 examples after augmentation
- **Examples-to-groups ratio**: 2:1 to 2.5:1

**Comparison with Other HRM Datasets:**

| Dataset | Examples | Groups | Ratio | Typical Batch Size |
|---------|----------|--------|-------|-------------------|
| ARC-AGI-1 | 960 | ~450 | 2.1:1 | 768 (survey regime) |
| City Logistics | 960 | 240 | 4:1 | 32 (traditional regime) |
| Sudoku-Extreme | 1000 | ~250 | 4:1 | 32 (traditional regime) |

### Sampling Regime Classification

We identify two distinct sampling regimes in HRM training:

#### **Traditional Sampling Regime** (`batch_size ≤ groups`)
```python
# Each batch = random subset of puzzle types
batch_groups = random_sample(all_groups, batch_size)
gradient_signal = partial_coverage_with_variance
training_dynamics = higher_variance_frequent_updates
```

**Characteristics:**
- Each gradient update sees a random subset of puzzle types
- Higher gradient variance between batches
- More training steps required for convergence
- Better exploration of local optima

#### **Survey Sampling Regime** (`batch_size > groups`)
```python
# Each batch = comprehensive coverage of all puzzle types
batch_groups = nearly_all_groups + some_duplicates
gradient_signal = comprehensive_coverage_low_variance
training_dynamics = stable_comprehensive_updates
```

**Characteristics:**
- Each gradient update sees nearly every puzzle type
- Lower gradient variance between batches
- Fewer training steps required for convergence
- Stronger convergence to global optima

---

## Technical Deep Dive: Why ARC Works

### 1. Hardware-Driven Configuration

The 768 batch size was primarily optimized for **8-GPU distributed training**:

```yaml
# Distributed configuration
num_gpus: 8
global_batch_size: 768
per_gpu_batch_size: 96    # 768 ÷ 8
memory_utilization: optimal
```

This hardware constraint happened to align favorably with ARC's group structure, enabling the survey sampling regime.

### 2. Gradient Quality vs Quantity Trade-off

**Traditional Warmup (Small Batches):**
- Many gradient updates (20-50)
- Each update covers ~10-20% of puzzle types
- Gradual learning rate ramp
- Higher noise, slower convergence

**ARC Warmup (Large Batches):**
- Few gradient updates (2.6)
- Each update covers ~90%+ of puzzle types
- Rapid but stable learning rate ramp
- Lower noise, faster convergence

### 3. Hierarchical Architecture Benefits

The HRM's hierarchical structure particularly benefits from comprehensive gradient signals:

**H-Module (High-Level):**
- Requires global context across puzzle types
- Benefits from seeing diverse examples simultaneously
- Establishes abstract reasoning patterns

**L-Module (Low-Level):**
- Handles detailed computations within contexts set by H-module
- Benefits from stable, comprehensive guidance

### 4. Task-Specific Advantages

**ARC Reasoning Characteristics:**
- Short reasoning chains (few transformations)
- Pattern recognition focused
- Benefits from comprehensive pattern exposure
- Less dependent on extensive search/backtracking

This contrasts with tasks like Sudoku that require extensive iterative refinement and benefit from more frequent gradient updates.

---

## Practical Guidelines for Batch Size Selection

### Decision Framework

```yaml
# Step 1: Analyze Dataset Structure
dataset_analysis:
  total_examples: count_examples()
  total_groups: count_base_scenarios()
  ratio: total_examples / total_groups
  complexity: assess_reasoning_requirements()

# Step 2: Determine Hardware Constraints
hardware_constraints:
  num_gpus: available_gpus
  memory_per_gpu: gpu_memory_gb
  target_per_gpu_batch: 32-128  # Sweet spot for most hardware

# Step 3: Select Sampling Regime
if (target_global_batch <= total_groups):
  regime: "traditional"
  benefits: ["exploration", "frequent_updates", "robust_convergence"]
  drawbacks: ["higher_variance", "more_steps_needed"]
else:
  regime: "survey" 
  benefits: ["stable_gradients", "fast_convergence", "comprehensive_coverage"]
  drawbacks: ["less_exploration", "hardware_intensive"]
```

### Configuration Recommendations by Dataset Size

#### **Small Datasets (≤1000 examples)**
```yaml
# Can afford survey regime if groups < 200
recommended_batch_size: min(total_groups * 1.5, hardware_limit)
lr_warmup_steps: 2 * total_examples
min_gradient_updates_warmup: 5
strategy: "survey_regime_with_comprehensive_coverage"
```

#### **Medium Datasets (1K-10K examples)**
```yaml
# Balance between regimes
recommended_batch_size: min(64-128, total_groups * 0.8)
lr_warmup_steps: 2 * total_examples  
min_gradient_updates_warmup: 15
strategy: "traditional_regime_with_good_coverage"
```

#### **Large Datasets (>10K examples)**
```yaml
# Traditional regime recommended
recommended_batch_size: 32-128  # Hardware constrained
lr_warmup_steps: 2 * total_examples
min_gradient_updates_warmup: 20
strategy: "traditional_regime_with_exploration"
```

### Task-Specific Considerations

#### **Short-Horizon Reasoning Tasks**
- Pattern recognition (ARC)
- Simple logical inference
- Spatial transformations

**Recommended Configuration:**
```yaml
batch_size_strategy: "larger_batches_preferred"
reasoning: "Benefits from comprehensive pattern exposure"
max_batch_size: min(total_groups * 2, hardware_limit)
```

#### **Long-Horizon Reasoning Tasks**
- Complex search problems (Sudoku)
- Multi-step planning
- Extensive backtracking required

**Recommended Configuration:**
```yaml
batch_size_strategy: "smaller_batches_preferred" 
reasoning: "Benefits from frequent gradient updates during search"
max_batch_size: min(64, total_groups * 0.5)
```

---

## Implementation Guidelines

### Configuration Validation Checklist

```python
def validate_hrm_config(config):
    """Validate HRM training configuration for potential issues."""
    
    # 1. Basic constraints
    assert config.global_batch_size % config.num_replicas == 0
    assert config.lr_warmup_steps >= 2 * config.dataset_size
    
    # 2. Sampling regime analysis
    groups = count_groups(config.dataset_path)
    ratio = config.global_batch_size / groups
    
    if ratio > 1.0:
        print(f"⚠️  Survey sampling regime detected (ratio: {ratio:.1f})")
        print("   - Ensure hardware can handle large batches")
        print("   - Expect faster but less exploratory training")
    else:
        print(f"✅ Traditional sampling regime (ratio: {ratio:.1f})")
        
    # 3. Warmup stability check
    warmup_updates = config.lr_warmup_steps / config.global_batch_size
    if warmup_updates < 5:
        print(f"⚠️  Very few warmup updates ({warmup_updates:.1f})")
        print("   - Monitor for training instability")
        print("   - Consider reducing batch size or extending warmup")
    
    # 4. Hardware efficiency
    per_gpu_batch = config.global_batch_size / config.num_replicas
    if per_gpu_batch < 16 or per_gpu_batch > 256:
        print(f"⚠️  Suboptimal per-GPU batch size ({per_gpu_batch})")
        
    return True
```

### Monitoring During Training

**Key Metrics to Track:**

1. **Gradient Norms During Warmup**
   - Should increase gradually, not spike
   - Monitor for instability in first 100 steps

2. **Loss Convergence Patterns**
   - Survey regime: Faster initial convergence
   - Traditional regime: More gradual convergence

3. **GPU Utilization**
   - Target: >85% GPU memory usage
   - Target: >90% GPU compute utilization

4. **Training Stability**
   - Watch for loss spikes during warmup
   - Monitor learning rate scaling behavior

### Common Pitfalls and Solutions

#### **Problem: Batch Size Too Large**
```yaml
symptoms:
  - Loss spikes during warmup
  - Very steep learning rate ramp
  - Poor GPU memory utilization
  
solutions:
  - Reduce global_batch_size to hardware sweet spot
  - Extend lr_warmup_steps if needed
  - Consider gradient accumulation for effective large batches
```

#### **Problem: Batch Size Too Small**
```yaml
symptoms:
  - Slow convergence
  - High gradient variance
  - Poor GPU utilization
  
solutions:
  - Increase batch_size within memory constraints
  - Use gradient accumulation
  - Consider distributed training
```

#### **Problem: Mismatched Regime for Task**
```yaml
# Long-horizon task with large batches
symptoms:
  - Convergence to suboptimal solutions
  - Poor exploration of solution space
  
solutions:
  - Reduce batch_size for more frequent updates
  - Increase training epochs to compensate
  
# Short-horizon task with small batches  
symptoms:
  - Slow convergence
  - High training variance
  
solutions:
  - Increase batch_size for stability
  - Reduce training epochs due to faster convergence
```

---

## Future Research Directions

### 1. Adaptive Batch Size Scheduling
Investigate dynamic batch size adjustment during training:
- Start with survey regime for fast initial convergence
- Transition to traditional regime for fine-grained optimization

### 2. Group Structure Optimization
Explore optimal group definitions for different task types:
- Semantic grouping vs random grouping
- Hierarchical group structures
- Dynamic group resampling

### 3. Hardware-Aware Configuration
Develop automated configuration tools:
- Optimize batch size for specific hardware configurations
- Balance memory usage vs computational efficiency
- Multi-node scaling considerations

### 4. Task-Specific Sampling Strategies
Design specialized sampling approaches:
- Curriculum learning within sampling regimes
- Difficulty-aware batch composition
- Multi-task learning considerations

---

## Conclusion

The ARC configuration paradox reveals important insights about HRM's training dynamics. The apparent violation of typical constraints actually demonstrates the model's ability to leverage comprehensive gradient signals effectively. Key takeaways:

1. **Two Sampling Regimes**: Traditional and survey sampling each have distinct advantages
2. **Hardware Alignment**: Optimal batch sizes often emerge from hardware constraints
3. **Task Specificity**: Different reasoning tasks benefit from different batch size strategies
4. **Configuration Validation**: Systematic analysis prevents common pitfalls

**Recommendations for Practitioners:**
- Analyze dataset group structure before selecting batch size
- Validate warmup stability with your chosen configuration
- Monitor training dynamics to ensure regime alignment
- Consider task-specific reasoning requirements

The HRM architecture's flexibility allows it to excel in both sampling regimes, but understanding when and how to apply each regime is crucial for optimal performance.

---

**Authors:** HRM Training Analysis Team  
**Date:** January 2025  
**Version:** 1.0