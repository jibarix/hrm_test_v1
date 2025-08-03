# 🗽 NYC Logistics & Routing Simulator

A research-oriented traffic routing simulation designed to generate HRM-compatible training data for Hierarchical Reasoning Models and test AI pathfinding capabilities in complex urban environments.

## 🎯 Project Overview

This interactive simulation recreates Manhattan's grid system with realistic Monday traffic patterns, serving as both a data generation tool and evaluation platform for advanced AI reasoning models. The simulator generates 1000 training examples in HRM-compatible format, following the methodology described in the HRM research paper for creating complex reasoning benchmarks.

## 🧠 Research Context

The simulator is designed to support the 3-phase HRM research pipeline:

1. **Oracle Data Generation** - Perfect A* pathfinding creates ground truth training data
2. **HRM Model Training** - Neural networks learn routing patterns from examples  
3. **Oracle Replacement** - Trained HRM models replace algorithmic pathfinding

## ✨ Features

### 🏙️ Realistic NYC Environment

- Manhattan-style grid with major avenues and cross streets
- Central Park as large obstacle area
- Broadway diagonal route with unique properties
- Time-based traffic patterns reflecting real Monday conditions

### 🚗 Vehicle Diversity & Constraints

- 🚲 **Bike**: Can use all roads, fastest in traffic
- 🚗 **Car**: Can use all roads, moderate speed
- 🚐 **Van**: Restricted to major avenues + Broadway
- 🚛 **Truck**: Major avenues only, slowest movement

### ⏰ Dynamic Traffic Simulation

- **Rush Hours (8 AM, 5 PM)**: Heavy congestion on major routes
- **Lunch Hour (12 PM)**: Midtown delivery restrictions
- **Late Night (11 PM-5 AM)**: Construction and road closures
- **Off-Peak**: Light traffic with occasional obstacles

### 🤖 HRM Training Pipeline

- Automated data collection with 1000 diverse scenarios
- Integer token-based format compatible with HRM models
- JSON export for external model training
- Integration points for HRM model deployment

## 🚀 Quick Start

### Dataset Generation

1. Save the HTML file as `logistics_game.html`
2. Open in any modern web browser (Chrome, Firefox, Safari, Edge)
3. No additional dependencies or setup required

### HRM Dataset Creation

1. **Select Vehicle Type**: Choose from bike, car, van, or truck
2. **Choose Time**: Pick different Monday time slots (6 AM - 11 PM)
3. **Generate Dataset**: Click "🤖 Generate HRM Dataset" to create 1000 training examples
4. **Wait for Collection**: Progress shows "850/1000 (1247 attempts)" format
5. **Download Dataset**: Click "📥 Download HRM Data" for 7 JSON files

## Generated Files

The simulator creates 7 files in HRM-compatible format:

- `nyc_routing_1000ex_train_dataset.json` - Metadata and configuration
- `nyc_routing_1000ex_train_all__inputs.json` - Input grid sequences (1600 tokens each)
- `nyc_routing_1000ex_train_all__labels.json` - Output path sequences (1600 tokens each)
- `nyc_routing_1000ex_train_all__puzzle_identifiers.json` - Puzzle grouping identifiers
- `nyc_routing_1000ex_train_all__puzzle_indices.json` - Example boundary indices
- `nyc_routing_1000ex_train_all__group_indices.json` - Group boundary indices
- `nyc_routing_1000ex_identifiers.json` - Task identifier mapping

## Dataset Conversion

Convert JSON to HRM .npy format:

```bash
# Create the conversion script
# Save as dataset/build_logistics_dataset.py (see conversion script below)

# Run conversion
python dataset/build_logistics_dataset.py --source-dir dataset/raw-data/Logistics --output-dir data/logistics-routing-1k

# Expected output structure:
data/logistics-routing-1k/
├── train/
│   ├── all__inputs.npy
│   ├── all__labels.npy
│   ├── all__puzzle_identifiers.npy
│   ├── all__puzzle_indices.npy
│   ├── all__group_indices.npy
│   └── dataset.json
├── test/
│   └── (same files)
└── identifiers.json
```

## HRM Model Training

Train the Hierarchical Reasoning Model:

```bash
# Single GPU training
python pretrain.py data_path=data/logistics-routing-1k epochs=10000 eval_interval=1000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0

# Multi-GPU training (8 GPUs)
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/logistics-routing-1k epochs=10000 eval_interval=1000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0
```

## Expected Results

- **Runtime**: ~1-2 hours (similar to Maze 30x30 Hard)
- **Dataset size**: 1000 examples (800 train, 200 test)
- **Expected accuracy**: 80-95% (routing complexity similar to maze navigation)
- **Memory**: ~8GB GPU memory for single GPU training

## 🔬 Technical Details

### HRM Token Format

Current Implementation: HRM-Compatible Integer Sequences

- **Flattened Grid**: 40x40 grid converted to 1600-token sequence
- **Integer Mapping**: Each cell type mapped to specific integer token
- **Perfect Information**: Complete knowledge of traffic conditions
- **Multi-criteria**: Balances distance vs. time costs
- **Constraint-aware**: Respects vehicle road restrictions

```javascript
// HRM Token Mapping
const HRM_TOKEN_MAP = {
    PAD: 0,           // Padding token
    OBSTACLE: 1,      // Buildings/Central Park
    SMALL_ROAD: 2,    // Side Streets  
    LARGE_ROAD: 3,    // Major Avenues
    BROADWAY: 4,      // Broadway diagonal
    TRAFFIC_JAM: 5,   // Heavy Traffic
    ROAD_CLOSURE: 6,  // Road Closure
    START: 7,         // Start Point
    END: 8,           // End Point
    PATH: 9           // Optimal Route
};
```

**Token Sequence Example:**
```
Input:  [1, 1, 3, 5, 2, 7, 1, 3, 8, 2, ...] // 1600 tokens (map with start/end)
Output: [0, 0, 9, 9, 9, 9, 0, 0, 0, 0, ...] // 1600 tokens (path grid)
```

## Dataset Conversion Script

Create `dataset/build_logistics_dataset.py`:

```python
from typing import Optional
import os
import json
import numpy as np
from pathlib import Path

from argdantic import ArgParser
from pydantic import BaseModel
from common import PuzzleDatasetMetadata

cli = ArgParser()

class DataProcessConfig(BaseModel):
    source_dir: str = "dataset/raw-data/Logistics"
    output_dir: str = "data/logistics-routing-1k"
    
def convert_dataset(config: DataProcessConfig):
    # Read the JSON files
    inputs = json.load(open(f"{config.source_dir}/nyc_routing_1000ex_train_all__inputs.json"))
    labels = json.load(open(f"{config.source_dir}/nyc_routing_1000ex_train_all__labels.json"))
    puzzle_identifiers = json.load(open(f"{config.source_dir}/nyc_routing_1000ex_train_all__puzzle_identifiers.json"))
    puzzle_indices = json.load(open(f"{config.source_dir}/nyc_routing_1000ex_train_all__puzzle_indices.json"))
    group_indices = json.load(open(f"{config.source_dir}/nyc_routing_1000ex_train_all__group_indices.json"))
    metadata = json.load(open(f"{config.source_dir}/nyc_routing_1000ex_train_dataset.json"))
    
    # Convert to numpy arrays
    data = {
        "inputs": np.array(inputs, dtype=np.int32),
        "labels": np.array(labels, dtype=np.int32),  
        "puzzle_identifiers": np.array(puzzle_identifiers, dtype=np.int32),
        "puzzle_indices": np.array(puzzle_indices, dtype=np.int32),
        "group_indices": np.array(group_indices, dtype=np.int32)
    }
    
    # Create train/test split (80/20)
    split_idx = int(0.8 * len(data["inputs"]))
    
    for split_name, start_idx, end_idx in [("train", 0, split_idx), ("test", split_idx, len(data["inputs"]))]:
        save_dir = os.path.join(config.output_dir, split_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Slice data for this split
        split_data = {k: v[start_idx:end_idx] for k, v in data.items() if k not in ["puzzle_indices", "group_indices"]}
        
        # Adjust indices
        split_data["puzzle_indices"] = data["puzzle_indices"] - start_idx
        split_data["group_indices"] = data["group_indices"] - start_idx
        
        # Save .npy files
        for k, v in split_data.items():
            np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
            
        # Create metadata
        split_metadata = PuzzleDatasetMetadata(
            seq_len=metadata["seq_len"],
            vocab_size=metadata["vocab_size"],
            pad_id=metadata["pad_id"],
            ignore_label_id=metadata["ignore_label_id"],
            blank_identifier_id=metadata["blank_identifier_id"],
            num_puzzle_identifiers=metadata["num_puzzle_identifiers"],
            total_groups=len(split_data["group_indices"]) - 1,
            mean_puzzle_examples=1.0,
            sets=["all"]
        )
        
        # Save metadata
        with open(os.path.join(save_dir, "dataset.json"), "w") as f:
            json.dump(split_metadata.model_dump(), f)
    
    # Save identifiers
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)

@cli.command(singleton=True)
def main(config: DataProcessConfig):
    convert_dataset(config)

if __name__ == "__main__":
    cli()
```

### HRM Training Data Format

```json
{
  "metadata": {
    "task": "nyc_logistics_routing",
    "seq_len": 1600,
    "vocab_size": 10,
    "map_dimensions": {"width": 40, "height": 40},
    "token_mapping": {"PAD": 0, "OBSTACLE": 1, ...}
  },
  "inputs": [[1,1,3,5,2,7,1,3,8,2,...], ...],  // 1000 examples x 1600 tokens
  "labels": [[0,0,9,9,9,9,0,0,0,0,...], ...]   // 1000 examples x 1600 tokens
}
```

## 🔄 HRM Integration Workflow

The complete pipeline follows the 3-phase HRM research methodology:

### Phase 1: Oracle Data Generation (Current)
- A* pathfinding creates ground truth training data
- 1000 examples with perfect optimal routes
- HRM-compatible integer token format
- 40x40 grid flattened to 1600-token sequences

### Phase 2: HRM Model Training
```bash
# Convert dataset to .npy format
python dataset/build_logistics_dataset.py --source-dir dataset/raw-data/Logistics --output-dir data/logistics-routing-1k

# Train HRM model
python pretrain.py data_path=data/logistics-routing-1k epochs=10000 eval_interval=1000

# Expected: 80-95% accuracy on routing tasks
```

### Phase 3: Oracle Replacement (Future)
```javascript
// FUTURE: Replace A* with trained HRM model
const hrmInput = gridToHRMSequence(grid, trafficGrid, startPos, endPos, []);
const pathGrid = await callTrainedHRM(hrmInput);
const optimalPath = parseHRMPath(pathGrid);
```

## 📊 Research Applications

### NYC Logistics Routing Benchmark

- **Complex Reasoning Task**: Multi-step planning with vehicle constraints
- **Dynamic Conditions**: Traffic patterns affect optimal strategies  
- **Perfect Training Data**: Mathematical ground truth from A* oracle
- **Progressive Difficulty**: Vehicle restrictions create challenge levels

### HRM Model Evaluation

- **Path Optimality**: Compare HRM routes vs. A* optimal routes
- **Generalization**: Test on unseen traffic/time combinations
- **Constraint Handling**: Verify vehicle restriction compliance
- **Reasoning Depth**: Analyze multi-step planning capabilities

### Expected Performance Comparison

| Vehicle Type | A* Oracle Accuracy | HRM Expected Accuracy | Complexity |
|--------------|--------------------|-----------------------|------------|
| 🚲 Bike | 100% | 90-95% | Low (all roads) |
| 🚗 Car | 100% | 85-92% | Medium (all roads) |
| 🚐 Van | 100% | 80-88% | High (limited roads) |
| 🚛 Truck | 100% | 75-85% | Expert (major avenues only) |

## 🎮 Game Mechanics

### Token Representation

- **Grid Size**: 40x40 Manhattan-style layout
- **Sequence Length**: 1600 tokens per example
- **Vocabulary Size**: 10 tokens (PAD + 9 map elements)
- **Input Format**: Map state with start/end points
- **Output Format**: Binary path grid (0=empty, 9=path)

### Traffic Patterns

| Time | Period | Congestion Level | Construction |
|------|--------|------------------|--------------|
| 6:00 AM | Early Morning | 10% | Yes |
| 8:00 AM | Rush Hour | 40% | No |
| 12:00 PM | Lunch Peak | 25% | No |
| 17:00 PM | Evening Rush | 45% | No |
| 23:00 PM | Late Night | 10% | Yes |

## 📁 File Structure

```
logistics_game.html (Complete Simulator)
├── HRM Dataset Generation
├── A* Pathfinding Engine
├── NYC Traffic Simulation
├── Vehicle Constraint System
└── Token Conversion Pipeline

Generated Dataset Files:
dataset/raw-data/Logistics/
├── nyc_routing_1000ex_train_dataset.json
├── nyc_routing_1000ex_train_all__inputs.json
├── nyc_routing_1000ex_train_all__labels.json
├── nyc_routing_1000ex_train_all__puzzle_identifiers.json
├── nyc_routing_1000ex_train_all__puzzle_indices.json
├── nyc_routing_1000ex_train_all__group_indices.json
└── nyc_routing_1000ex_identifiers.json

Converted HRM Format:
data/logistics-routing-1k/
├── train/ (800 examples)
├── test/ (200 examples)
└── identifiers.json
```

## 🎯 Ready to train your HRM model on NYC traffic routing! 

The dataset provides 1000 examples of optimal pathfinding with realistic traffic constraints and vehicle limitations, formatted specifically for Hierarchical Reasoning Model training following the research methodology.