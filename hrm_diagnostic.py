#!/usr/bin/env python3
"""
HRM Model Diagnostic Script
Analyzes what the trained model is actually outputting
"""

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

# Load configuration
config_path = f"{'/'.join(CHECKPOINT_PATH.split('/')[:-1])}/all_config.yaml"
with open(config_path, "r") as f:
    config_dict = yaml.safe_load(f)
    config_dict['data_path'] = 'data/logistics-routing-1k' 
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

# --- Token Analysis ---
HRM_TOKEN_MAP = {
    "PAD": 0, "OBSTACLE": 1, "SMALL_ROAD": 2, "LARGE_ROAD": 3, "DIAGONAL": 4,
    "TRAFFIC_JAM": 5, "ROAD_CLOSURE": 6, "START": 7, "END": 8, "PATH": 9
}

TOKEN_NAMES = {v: k for k, v in HRM_TOKEN_MAP.items()}

def create_simple_test_input():
    """Create a simple test routing scenario"""
    MAP_SIZE = 40
    sequence = [HRM_TOKEN_MAP["PAD"]] * (MAP_SIZE * MAP_SIZE)
    
    # Create a simple grid with roads
    for i in range(MAP_SIZE * MAP_SIZE):
        y = i // MAP_SIZE
        x = i % MAP_SIZE
        
        # Default to obstacles
        if y < 2 or y >= MAP_SIZE - 2 or x < 2 or x >= MAP_SIZE - 2:
            sequence[i] = HRM_TOKEN_MAP["OBSTACLE"]
        elif y % 4 == 0 or x % 4 == 0:  # Roads every 4 cells
            sequence[i] = HRM_TOKEN_MAP["LARGE_ROAD"]
        else:
            sequence[i] = HRM_TOKEN_MAP["OBSTACLE"]
    
    # Add start and end
    start_idx = 4 * MAP_SIZE + 4  # Position (4, 4)
    end_idx = 35 * MAP_SIZE + 35  # Position (35, 35)
    sequence[start_idx] = HRM_TOKEN_MAP["START"]
    sequence[end_idx] = HRM_TOKEN_MAP["END"]
    
    return sequence

def analyze_model_output():
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
        
        # Move carry to device (simplified version)
        def move_to_device(obj, device):
            if hasattr(obj, 'to'):
                return obj.to(device)
            if hasattr(obj, '__dict__'):
                for attr, value in obj.__dict__.items():
                    setattr(obj, attr, move_to_device(value, device))
            return obj
        
        carry = move_to_device(carry, DEVICE)
        
        # Run inference
        step_count = 0
        final_outputs = None
        
        while True:
            carry, _, _, outputs, all_halted = model(return_keys=["logits"], carry=carry, batch=dummy_batch)
            step_count += 1
            
            if all_halted:
                final_outputs = outputs
                break
            
            if step_count > 20:  # Safety limit
                print("⚠️  Stopping after 20 steps (safety limit)")
                final_outputs = outputs
                break
    
    print(f"✅ Model completed in {step_count} steps")
    
    # Analyze final output
    if final_outputs and "logits" in final_outputs:
        logits = final_outputs["logits"]
        predicted_tokens = torch.argmax(logits, dim=-1).squeeze(0).cpu().tolist()
        
        print(f"\n📊 Output sequence length: {len(predicted_tokens)}")
        output_counter = Counter(predicted_tokens)
        print("📊 Output token distribution:")
        for token_id, count in sorted(output_counter.items()):
            token_name = TOKEN_NAMES.get(token_id, f"UNK_{token_id}")
            percentage = (count / len(predicted_tokens)) * 100
            print(f"   {token_name}: {count} ({percentage:.1f}%)")
        
        # Specific analysis of PATH tokens
        path_count = output_counter.get(HRM_TOKEN_MAP["PATH"], 0)
        total_tokens = len(predicted_tokens)
        path_percentage = (path_count / total_tokens) * 100
        
        print(f"\n🎯 PATH TOKEN ANALYSIS:")
        print(f"   PATH tokens: {path_count} out of {total_tokens} ({path_percentage:.1f}%)")
        
        if path_count == total_tokens:
            print("   🚨 PROBLEM: Model is outputting PATH for EVERY token!")
            print("   🚨 This indicates the model hasn't learned proper pathfinding")
        elif path_count > total_tokens * 0.5:
            print("   ⚠️  WARNING: Model is outputting too many PATH tokens")
            print("   ⚠️  This suggests overfitting or incorrect learning")
        elif path_count == 0:
            print("   ⚠️  WARNING: Model is not outputting any PATH tokens")
            print("   ⚠️  This suggests the model hasn't learned to predict paths")
        else:
            print("   ✅ Path token count looks reasonable")
        
        # Check if model preserves input structure
        preserved_start = HRM_TOKEN_MAP["START"] in predicted_tokens
        preserved_end = HRM_TOKEN_MAP["END"] in predicted_tokens
        
        print(f"\n🎯 STRUCTURE PRESERVATION:")
        print(f"   START token preserved: {preserved_start}")
        print(f"   END token preserved: {preserved_end}")
        
        if not preserved_start or not preserved_end:
            print("   ⚠️  Model is not preserving start/end positions")
        
        # Sample output for manual inspection
        print(f"\n📝 SAMPLE OUTPUT (first 100 tokens):")
        sample_tokens = predicted_tokens[:100]
        sample_names = [TOKEN_NAMES.get(t, f"UNK_{t}") for t in sample_tokens]
        
        # Print in groups of 10 for readability
        for i in range(0, len(sample_names), 10):
            group = sample_names[i:i+10]
            print(f"   {i:3d}-{i+9:3d}: {' '.join(group)}")
    
    print("\n" + "="*60)
    print("🎯 DIAGNOSIS COMPLETE")
    print("="*60)

def main():
    """Run model diagnostics"""
    try:
        analyze_model_output()
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()