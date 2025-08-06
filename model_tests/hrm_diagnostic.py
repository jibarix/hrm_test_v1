#!/usr/bin/env python3
"""
HRM Model Diagnostic Script
Analyzes what the trained model is actually outputting
"""

import sys
import os
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import numpy as np
from collections import Counter

# --- Model & Training Imports ---
from pretrain import PretrainConfig, create_model
from puzzle_dataset import PuzzleDatasetMetadata

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f">>> Using device: {DEVICE}")

# Update this path to your trained model
CHECKPOINT_PATH = "checkpoints/City-logistics-1k ACT-torch/HierarchicalReasoningModel_ACTV1 cordial-vole/step_22400"

def load_model_and_config():
    """Load HRM model and configuration"""
    try:
        # Load configuration
        config_path = f"{'/'.join(CHECKPOINT_PATH.split('/')[:-1])}/all_config.yaml"
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
            config_dict['data_path'] = 'data/city-logistics-1k'  # Updated path
            config = PretrainConfig(**config_dict)

        # Create dummy metadata
        dummy_metadata = PuzzleDatasetMetadata(
            pad_id=0, ignore_label_id=-100, blank_identifier_id=0,
            vocab_size=10, seq_len=1600, num_puzzle_identifiers=1,
            total_groups=1, mean_puzzle_examples=1.0, sets=['all']
        )

        # Load model
        print(">>> Loading model...")
        model, _, _ = create_model(config, dummy_metadata, world_size=1)
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
        model.to(DEVICE)
        model.eval()
        print(">>> Model loaded successfully!")
        
        return model, config
        
    except FileNotFoundError as e:
        print(f"❌ Model checkpoint not found: {e}")
        print(f"   Expected path: {CHECKPOINT_PATH}")
        print("   Make sure you've trained a model first!")
        return None, None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

# --- Token Analysis ---
HRM_TOKEN_MAP = {
    "PAD": 0, "OBSTACLE": 1, "SMALL_ROAD": 2, "LARGE_ROAD": 3, "DIAGONAL": 4,
    "TRAFFIC_JAM": 5, "ROAD_CLOSURE": 6, "START": 7, "END": 8, "PATH": 9
}

TOKEN_NAMES = {v: k for k, v in HRM_TOKEN_MAP.items()}

def create_simple_test_input():
    """Create a simple test routing scenario"""
    MAP_SIZE = 40
    sequence = [HRM_TOKEN_MAP["OBSTACLE"]] * (MAP_SIZE * MAP_SIZE)  # Default to obstacles
    
    # Create a simple grid with roads
    for i in range(MAP_SIZE * MAP_SIZE):
        y = i // MAP_SIZE
        x = i % MAP_SIZE
        
        # Create road network
        if y == 10 or y == 20 or y == 30:  # Horizontal roads
            sequence[i] = HRM_TOKEN_MAP["LARGE_ROAD"]
        elif x == 10 or x == 20 or x == 30:  # Vertical roads
            sequence[i] = HRM_TOKEN_MAP["LARGE_ROAD"]
        elif (y % 5 == 0 and 5 <= x <= 35) or (x % 5 == 0 and 5 <= y <= 35):  # Additional roads
            sequence[i] = HRM_TOKEN_MAP["SMALL_ROAD"]
    
    # Add some traffic
    for i in range(0, len(sequence), 50):
        if sequence[i] in [HRM_TOKEN_MAP["LARGE_ROAD"], HRM_TOKEN_MAP["SMALL_ROAD"]]:
            if np.random.random() < 0.3:  # 30% chance of traffic
                sequence[i] = HRM_TOKEN_MAP["TRAFFIC_JAM"]
    
    # Add start and end points
    start_idx = 5 * MAP_SIZE + 5  # Position (5, 5)
    end_idx = 35 * MAP_SIZE + 35  # Position (35, 35)
    sequence[start_idx] = HRM_TOKEN_MAP["START"]
    sequence[end_idx] = HRM_TOKEN_MAP["END"]
    
    return sequence

def analyze_model_output(model):
    """Analyze what the model outputs"""
    print("\n" + "="*60)
    print("🔍 ANALYZING HRM MODEL OUTPUT")
    print("="*60)
    
    # Create test input
    test_input = create_simple_test_input()
    input_tensor = torch.tensor(test_input, dtype=torch.long).unsqueeze(0).to(DEVICE)
    
    print(f"📊 Input sequence length: {len(test_input)}")
    input_counter = Counter(test_input)
    print("📊 Input token distribution:")
    for token_id, count in sorted(input_counter.items()):
        token_name = TOKEN_NAMES.get(token_id, f"UNK_{token_id}")
        percentage = (count / len(test_input)) * 100
        print(f"   {token_name}: {count} ({percentage:.1f}%)")
    
    # Run model
    print("\n🧠 Running HRM model...")
    with torch.no_grad():
        dummy_batch = {
            "inputs": input_tensor,
            "labels": torch.zeros_like(input_tensor),
            "puzzle_identifiers": torch.tensor([0], dtype=torch.long).to(DEVICE)
        }
        
        # Initialize carry
        carry = model.initial_carry(dummy_batch)
        
        # Robust carry-to-device function
        def move_to_device(obj, device):
            if hasattr(obj, 'to'):
                return obj.to(device)
            if isinstance(obj, (list, tuple)):
                return type(obj)(move_to_device(x, device) for x in obj)
            if isinstance(obj, dict):
                return {k: move_to_device(v, device) for k, v in obj.items()}
            if hasattr(obj, '__dict__'):
                for attr, value in obj.__dict__.items():
                    setattr(obj, attr, move_to_device(value, device))
            return obj
        
        carry = move_to_device(carry, DEVICE)
        
        # Run inference with step logging
        step_count = 0
        final_outputs = None
        reasoning_steps = []
        
        print("   Starting HRM reasoning process...")
        
        while True:
            carry, _, _, outputs, all_halted = model(return_keys=["logits"], carry=carry, batch=dummy_batch)
            step_count += 1
            
            # Log intermediate reasoning
            if outputs and "logits" in outputs:
                predicted_tokens = torch.argmax(outputs["logits"], dim=-1).squeeze(0).cpu().tolist()
                path_count = predicted_tokens.count(HRM_TOKEN_MAP["PATH"])
                reasoning_steps.append(path_count)
                print(f"   Step {step_count}: {path_count} PATH tokens predicted")
            
            if all_halted:
                final_outputs = outputs
                print(f"   ✅ Model halted after {step_count} steps")
                break
            
            if step_count > 20:  # Safety limit
                print("   ⚠️  Stopping after 20 steps (safety limit)")
                final_outputs = outputs
                break
    
    # Analyze reasoning progression
    if len(reasoning_steps) > 1:
        print(f"\n🧠 REASONING PROGRESSION:")
        print(f"   PATH tokens by step: {reasoning_steps}")
        if reasoning_steps[-1] > reasoning_steps[0]:
            print("   ✅ Model is building up a path during reasoning")
        elif all(x == reasoning_steps[0] for x in reasoning_steps):
            print("   ⚠️  Model output is static (not reasoning)")
        else:
            print("   🔄 Model is refining its path during reasoning")
    
    # Analyze final output
    if final_outputs and "logits" in final_outputs:
        logits = final_outputs["logits"]
        predicted_tokens = torch.argmax(logits, dim=-1).squeeze(0).cpu().tolist()
        
        print(f"\n📊 FINAL OUTPUT ANALYSIS:")
        print(f"   Output sequence length: {len(predicted_tokens)}")
        output_counter = Counter(predicted_tokens)
        print("   Final token distribution:")
        for token_id, count in sorted(output_counter.items()):
            token_name = TOKEN_NAMES.get(token_id, f"UNK_{token_id}")
            percentage = (count / len(predicted_tokens)) * 100
            print(f"     {token_name}: {count} ({percentage:.1f}%)")
        
        # Critical analysis
        path_count = output_counter.get(HRM_TOKEN_MAP["PATH"], 0)
        total_tokens = len(predicted_tokens)
        path_percentage = (path_count / total_tokens) * 100
        
        print(f"\n🎯 CRITICAL DIAGNOSIS:")
        print(f"   PATH tokens: {path_count} out of {total_tokens} ({path_percentage:.1f}%)")
        
        if path_count == total_tokens:
            print("   🚨 CRITICAL PROBLEM: Model outputs PATH for EVERY token!")
            print("   🚨 This means the model learned 'output PATH everywhere'")
            print("   🚨 Likely cause: Corrupted training labels")
        elif path_count > total_tokens * 0.8:
            print("   🚨 MAJOR PROBLEM: Model outputs too many PATH tokens")
            print("   🚨 Model may have overfitted to 'output mostly PATH'")
        elif path_count > total_tokens * 0.3:
            print("   ⚠️  WARNING: Model outputs many PATH tokens")
            print("   ⚠️  May indicate incomplete learning")
        elif path_count == 0:
            print("   ⚠️  WARNING: Model outputs NO PATH tokens")
            print("   ⚠️  Model hasn't learned to predict paths")
        elif 10 <= path_count <= 100:  # Reasonable range for 40x40 grid
            print("   ✅ PATH token count looks reasonable for pathfinding")
        else:
            print(f"   🤔 PATH count ({path_count}) needs manual inspection")
        
        # Structure preservation check
        preserved_start = HRM_TOKEN_MAP["START"] in predicted_tokens
        preserved_end = HRM_TOKEN_MAP["END"] in predicted_tokens
        
        print(f"\n🎯 STRUCTURE PRESERVATION:")
        print(f"   START token preserved: {'✅' if preserved_start else '❌'}")
        print(f"   END token preserved: {'✅' if preserved_end else '❌'}")
        
        if not preserved_start or not preserved_end:
            print("   ⚠️  Model is not preserving critical structure")
        
        # Check for reasonable token variety
        unique_tokens = len(output_counter)
        print(f"   Token variety: {unique_tokens}/10 possible tokens")
        if unique_tokens < 3:
            print("   ⚠️  Very low token variety - model may be collapsed")
        
        return predicted_tokens, path_percentage
    
    return None, 0

def visualize_grid_output(predicted_tokens):
    """Visualize the output as a grid"""
    if not predicted_tokens:
        return
        
    print(f"\n🗺️  OUTPUT GRID VISUALIZATION (first 10x10):")
    print("="*50)
    
    # Reshape to 40x40 grid and show first 10x10
    try:
        grid = np.array(predicted_tokens).reshape(40, 40)
        
        # Print header
        print("    " + "".join(f"{i:4}" for i in range(10)))
        print("   " + "-" * 40)
        
        for y in range(10):
            row_tokens = []
            for x in range(10):
                token = grid[y, x]
                token_name = TOKEN_NAMES.get(token, f"?{token}")
                if token_name == "PATH":
                    row_tokens.append("🔵")  # PATH
                elif token_name == "START":
                    row_tokens.append("🟢")  # START
                elif token_name == "END":
                    row_tokens.append("🔴")  # END
                elif token_name == "OBSTACLE":
                    row_tokens.append("⬛")  # OBSTACLE
                elif token_name == "LARGE_ROAD":
                    row_tokens.append("⬜")  # LARGE_ROAD
                elif token_name == "SMALL_ROAD":
                    row_tokens.append("◻️")   # SMALL_ROAD
                elif token_name == "TRAFFIC_JAM":
                    row_tokens.append("🟡")  # TRAFFIC
                else:
                    row_tokens.append("❓")  # UNKNOWN
            
            print(f"{y:2}| " + "".join(f"{token:>4}" for token in row_tokens))
            
    except Exception as e:
        print(f"❌ Error visualizing grid: {e}")

def main():
    """Run model diagnostics"""
    print("🔧 HRM Model Output Diagnostic")
    print("="*50)
    
    # Load model
    model, config = load_model_and_config()
    if model is None:
        return
    
    try:
        # Analyze output
        predicted_tokens, path_percentage = analyze_model_output(model)
        
        # Visualize if we got output
        if predicted_tokens:
            visualize_grid_output(predicted_tokens)
        
        # Final recommendations
        print(f"\n🎯 RECOMMENDATIONS:")
        if path_percentage > 80:
            print("   🚨 Model needs retraining - check training labels")
            print("   🚨 Verify dataset with dataset_diagnostic.py")
        elif path_percentage == 0:
            print("   📚 Model needs more training - continue training")
            print("   📚 Check loss curves and learning rate")
        elif 1 <= path_percentage <= 10:
            print("   ✅ Model appears to be learning correctly")
            print("   ✅ Continue training or test on full dataset")
        else:
            print("   🔍 Model behavior needs investigation")
            print("   🔍 Check training logs and loss curves")
        
        print("\n" + "="*60)
        print("🎯 DIAGNOSIS COMPLETE")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()