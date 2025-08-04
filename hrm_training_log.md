2025-08-03 07:00 PM

feat: Implement spatial connectivity fix for HRM pathfinding training

PROBLEM SOLVED:
- HRM was treating 1600-token sequences as flat arrays, causing spatial hallucinations
- Model learned to "teleport" between grid positions and wrap around edges
- Training became stuck with path_ratio=1.0 cheating (predicting entire grid as paths)
- Complex curriculum/batch-rejection systems were masking the fundamental issue

SOLUTION IMPLEMENTED:
- Added spatial_connectivity_loss() function to enforce 2D grid constraints
- Penalizes Manhattan distance > 1 between consecutive PATH tokens
- Converts flat sequence indices back to 2D coordinates for validation
- Clean surgical fix without curriculum complexity

CHANGES MADE:
1. Simplified dataset generator (logistics_game.html):
   - Reduced from 10 to 6 tokens (PAD, OBSTACLE, ROAD, SLOW_ROAD, START, END, PATH)
   - Single vehicle type, removed time-based complexity
   - Maintained A* oracle and 40×40 grid infrastructure

2. Updated dataset converter (build_logistics_dataset.py):
   - Adapted for simplified 6-token system
   - Updated file paths and token mappings
   - Maintained HRM compatibility

3. Enhanced loss function (losses.py):
   - Added spatial_connectivity_loss() with configurable penalty weight
   - Added optional path_connectivity_loss() for fragment detection
   - Integrated spatial fixes into ACTLossHead
   - Fixed tensor/float type issues

4. Updated training config (logistics_routing.yaml):
   - Enabled spatial loss with enable_spatial_loss: true
   - Set path_token_id: 6 for simplified system
   - Updated data_path for simple-pathfinding-1k dataset
   - Maintained SDPA compatibility for Windows

VALIDATION FRAMEWORK:
- Key metrics: train/spatial_loss, train/exact_accuracy, train/path_ratio
- Expected progression: spatial_loss decreases → accuracy improves → no more cheating
- Clean test environment without confounding curriculum variables

TECHNICAL DETAILS:
- Maintains 27M parameter count and 1600-token sequences
- Uses SDPA_ready baseline code (no batch rejection complexity)
- Windows compatible with DISABLE_COMPILE=1
- Preserves all HRM architectural innovations

This represents a fundamental fix to the 2D→1D representation problem that was
causing HRM spatial reasoning failures, implemented through minimal surgical
changes to the loss function rather than complex training procedures.


2025-08-03 08:55 PM

Implement hybrid loss approach for HRM pathfinding class imbalance

PROBLEM ADDRESSED:
- Severe class imbalance (97.5% non-path vs 2.5% path tokens) causing model to cheat
- Model predicting entire grid as paths (path_ratio=1.0) to guarantee coverage
- Spatial connectivity loss alone insufficient - model found loopholes

SOLUTION IMPLEMENTED:
- Added Focal Loss: Focuses learning on hard examples, reduces weight of easy examples
- Added Dice Loss: Directly measures path overlap between predictions and ground truth  
- Combined with existing spatial connectivity loss for comprehensive approach
- Research-backed solution for sparse sequence generation under extreme class imbalance

CHANGES MADE:
1. Enhanced models/losses.py:
   - Added focal_loss() function with alpha=0.25, gamma=2.0 (research recommended)
   - Added dice_loss() function for path overlap measurement
   - Extended ACTLossHead with hybrid loss parameters
   - Modified forward() to use focal loss when enabled, add dice loss component
   - Fixed tensor contiguity and dtype issues for CUDA compatibility

2. Updated config/logistics_routing.yaml:
   - Added enable_focal_loss: true, enable_dice_loss: true
   - Set dice_weight: 5.0 for balanced loss contribution
   - Maintained all existing HRM architectural parameters

TECHNICAL APPROACH:
- Backward compatible: hybrid approach disabled by default
- Maintains HRM's multi-timescale recurrence architecture unchanged
- Uses research-proven techniques (Focal Loss, Dice Loss) for class imbalance
- Avoids complex curriculum learning in favor of loss function engineering

EXPECTED OUTCOMES:
- Model should learn to predict sparse, connected paths instead of cheating
- train/focal_loss and train/dice_loss should decrease over training
- train/exact_accuracy should increase without path_ratio=1.0 spam
- Eliminates need for complex anti-cheating systems through principled loss design

This represents a fundamental solution to the sparse pathfinding class imbalance
problem using established deep learning techniques for imbalanced data.


2025-08-03 10:00 PM

Uniform distances: Flat 6-60 distribution
True uniform positions: Zero edge bias
Uniform obstacles: Flat 8-52% distribution
Uniform slow roads: Flat 5-45% distribution
Multiple map themes: 6 different generation strategies
Enhanced uniqueness: Multi-factor checking
Path complexity range: 2-200 step paths