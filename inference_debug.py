#!/usr/bin/env python3
"""
HRM Inference Debug Script
Test different inference modes to isolate the issue
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

CHECKPOINT_PATH = "checkpoints/City-logistics-1k ACT-torch/HierarchicalReasoningModel_ACTV1 cordial-vole/step_22400"

# Load configuration
config_path = f"{'/'.join(CHECKPOINT_PATH.split('/')[:-1])}/all_config.yaml"
with open(config_path, "r") as f:
    config_dict = yaml.safe_load(f)
    config_dict['data_path'] = 'data/city-logistics-1k' 
    config = PretrainConfig(**config_dict)

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

HRM_TOKEN_MAP = {
    "PAD": 0, "OBSTACLE": 1, "SMALL_ROAD": 2, "LARGE_ROAD": 3, "DIAGONAL": 4,
    "TRAFFIC_JAM": 5, "ROAD_CLOSURE": 6, "START": 7, "END": 8, "PATH": 9
}

TOKEN_NAMES = {v: k for k, v in HRM_TOKEN_MAP.items()}

def load_real_training_example():
    """Load a real training example to test on"""
    inputs = np.load('data/city-logistics-1k/train/all__inputs.npy')
    labels = np.load('data/city-logistics-1k/train/all__labels.npy')
    
    # Take first example
    test_input = inputs[0]
    true_labels = labels[0]
    
    print(f"📊 Real training example:")
    input_counter = Counter(test_input)
    label_counter = Counter(true_labels)
    
    print("Input distribution:")
    for token_id, count in sorted(input_counter.items()):
        if count > 0:
            token_name = TOKEN_NAMES.get(token_id, f"UNK_{token_id}")
            print(f"   {token_name}: {count}")
    
    print("True label distribution:")
    for token_id, count in sorted(label_counter.items()):
        if count > 0:
            token_name = TOKEN_NAMES.get(token_id, f"UNK_{token_id}")
            print(f"   {token_name}: {count}")
    
    return test_input, true_labels

def test_training_mode_inference():
    """Test model in training mode (like during training)"""
    print("\n" + "="*60)
    print("🎯 TESTING TRAINING MODE INFERENCE")
    print("="*60)
    
    test_input, true_labels = load_real_training_example()
    
    input_tensor = torch.tensor(test_input, dtype=torch.long).unsqueeze(0).to(DEVICE)
    label_tensor = torch.tensor(true_labels, dtype=torch.long).unsqueeze(0).to(DEVICE)
    
    # Create training-style batch
    batch = {
        "inputs": input_tensor,
        "labels": label_tensor,
        "puzzle_identifiers": torch.tensor([0], dtype=torch.long).to(DEVICE)
    }
    
    # Set model to training mode (like during training)
    model.train()
    
    with torch.no_grad():
        # Initialize carry
        carry = model.initial_carry(batch)
        
        # Move to device
        def move_to_device(obj, device):
            if hasattr(obj, 'to'):
                return obj.to(device)
            if hasattr(obj, '__dict__'):
                for attr, value in obj.__dict__.items():
                    setattr(obj, attr, move_to_device(value, device))
            return obj
        
        carry = move_to_device(carry, DEVICE)
        
        # Single forward pass (like during training)
        carry, loss, metrics, outputs, all_halted = model(
            return_keys=["logits"], carry=carry, batch=batch
        )
        
        print(f"✅ Training mode inference completed")
        print(f"📊 Loss: {loss.item():.4f}")
        print(f"📊 All halted: {all_halted}")
        
        if "logits" in outputs:
            logits = outputs["logits"]
            predicted_tokens = torch.argmax(logits, dim=-1).squeeze(0).cpu().tolist()
            
            output_counter = Counter(predicted_tokens)
            print("Predicted token distribution:")
            for token_id, count in sorted(output_counter.items()):
                if count > 0:
                    token_name = TOKEN_NAMES.get(token_id, f"UNK_{token_id}")
                    percentage = (count / len(predicted_tokens)) * 100
                    print(f"   {token_name}: {count} ({percentage:.1f}%)")
            
            # Check accuracy
            true_labels_tensor = batch["labels"].squeeze(0).cpu()
            predicted_tensor = torch.tensor(predicted_tokens)
            
            # Only check non-pad positions
            non_pad_mask = (true_labels_tensor != 0)
            if non_pad_mask.sum() > 0:
                accuracy = (predicted_tensor[non_pad_mask] == true_labels_tensor[non_pad_mask]).float().mean()
                print(f"📊 Accuracy on non-pad tokens: {accuracy.item():.1%}")

def test_evaluation_mode_inference():
    """Test model in evaluation mode (like during inference)"""
    print("\n" + "="*60)
    print("🔍 TESTING EVALUATION MODE INFERENCE")
    print("="*60)
    
    test_input, true_labels = load_real_training_example()
    
    input_tensor = torch.tensor(test_input, dtype=torch.long).unsqueeze(0).to(DEVICE)
    
    # Create inference-style batch (no labels)
    batch = {
        "inputs": input_tensor,
        "labels": torch.zeros_like(input_tensor),  # Dummy labels
        "puzzle_identifiers": torch.tensor([0], dtype=torch.long).to(DEVICE)
    }
    
    # Set model to evaluation mode (like during inference)
    model.eval()
    
    with torch.no_grad():
        # Initialize carry
        carry = model.initial_carry(batch)
        
        def move_to_device(obj, device):
            if hasattr(obj, 'to'):
                return obj.to(device)
            if hasattr(obj, '__dict__'):
                for attr, value in obj.__dict__.items():
                    setattr(obj, attr, move_to_device(value, device))
            return obj
        
        carry = move_to_device(carry, DEVICE)
        
        # Multi-step inference (like in your Flask server)
        step_count = 0
        while True:
            carry, _, _, outputs, all_halted = model(
                return_keys=["logits"], carry=carry, batch=batch
            )
            step_count += 1
            
            if all_halted or step_count > 20:
                break
        
        print(f"✅ Evaluation mode inference completed in {step_count} steps")
        print(f"📊 All halted: {all_halted}")
        
        if "logits" in outputs:
            logits = outputs["logits"]
            predicted_tokens = torch.argmax(logits, dim=-1).squeeze(0).cpu().tolist()
            
            output_counter = Counter(predicted_tokens)
            print("Predicted token distribution:")
            for token_id, count in sorted(output_counter.items()):
                if count > 0:
                    token_name = TOKEN_NAMES.get(token_id, f"UNK_{token_id}")
                    percentage = (count / len(predicted_tokens)) * 100
                    print(f"   {token_name}: {count} ({percentage:.1f}%)")

def test_single_step_inference():
    """Test model with single step (no ACT)"""
    print("\n" + "="*60)
    print("⚡ TESTING SINGLE STEP INFERENCE (NO ACT)")
    print("="*60)
    
    test_input, true_labels = load_real_training_example()
    
    input_tensor = torch.tensor(test_input, dtype=torch.long).unsqueeze(0).to(DEVICE)
    label_tensor = torch.tensor(true_labels, dtype=torch.long).unsqueeze(0).to(DEVICE)
    
    batch = {
        "inputs": input_tensor,
        "labels": label_tensor,
        "puzzle_identifiers": torch.tensor([0], dtype=torch.long).to(DEVICE)
    }
    
    model.eval()
    
    with torch.no_grad():
        # Initialize carry
        carry = model.initial_carry(batch)
        
        def move_to_device(obj, device):
            if hasattr(obj, 'to'):
                return obj.to(device)
            if hasattr(obj, '__dict__'):
                for attr, value in obj.__dict__.items():
                    setattr(obj, attr, move_to_device(value, device))
            return obj
        
        carry = move_to_device(carry, DEVICE)
        
        # Force single step by setting all_halted manually
        carry, _, _, outputs, _ = model(
            return_keys=["logits"], carry=carry, batch=batch
        )
        
        print(f"✅ Single step inference completed")
        
        if "logits" in outputs:
            logits = outputs["logits"]
            predicted_tokens = torch.argmax(logits, dim=-1).squeeze(0).cpu().tolist()
            
            output_counter = Counter(predicted_tokens)
            print("Predicted token distribution:")
            for token_id, count in sorted(output_counter.items()):
                if count > 0:
                    token_name = TOKEN_NAMES.get(token_id, f"UNK_{token_id}")
                    percentage = (count / len(predicted_tokens)) * 100
                    print(f"   {token_name}: {count} ({percentage:.1f}%)")

def main():
    """Run all inference tests"""
    try:
        test_training_mode_inference()
        test_evaluation_mode_inference()
        test_single_step_inference()
        
        print("\n" + "="*60)
        print("🎯 INFERENCE DEBUG COMPLETE")
        print("="*60)
        print("\n🔍 Analysis:")
        print("- If TRAINING MODE works but EVALUATION MODE fails:")
        print("  → ACT (halting) mechanism issue during inference")
        print("- If SINGLE STEP works but MULTI-STEP fails:")
        print("  → Problem with iterative reasoning/carry state")
        print("- If ALL modes output PATH everywhere:")
        print("  → Model weights/architecture issue")
        
    except Exception as e:
        print(f"❌ Error during inference debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()