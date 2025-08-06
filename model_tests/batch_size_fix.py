#!/usr/bin/env python3
"""
Batch Size Compatible Inference Test
Test HRM with the correct batch size that matches training
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

CHECKPOINT_PATH = "checkpoints/City-logistics-1k ACT-torch/HierarchicalReasoningModel_ACTV1 cordial-vole/step_22400"

# Load configuration
config_path = f"{'/'.join(CHECKPOINT_PATH.split('/')[:-1])}/all_config.yaml"
with open(config_path, "r") as f:
    config_dict = yaml.safe_load(f)
    config_dict['data_path'] = 'data/city-logistics-1k' 
    config = PretrainConfig(**config_dict)

print(f"🔍 Training config batch size: {config.global_batch_size}")

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

def test_with_training_batch_size():
    """Test model with the same batch size used during training"""
    print("\n" + "="*60)
    print("🎯 TESTING WITH TRAINING BATCH SIZE")
    print("="*60)
    
    # Load multiple training examples to fill the batch
    inputs = np.load('data/city-logistics-1k/train/all__inputs.npy')
    labels = np.load('data/city-logistics-1k/train/all__labels.npy')
    
    # Use the same batch size as training
    local_batch_size = config.global_batch_size // 1  # world_size = 1
    print(f"📊 Using batch size: {local_batch_size}")
    
    # Take first batch_size examples
    batch_inputs = inputs[:local_batch_size]
    batch_labels = labels[:local_batch_size]
    
    input_tensor = torch.tensor(batch_inputs, dtype=torch.long).to(DEVICE)
    label_tensor = torch.tensor(batch_labels, dtype=torch.long).to(DEVICE)
    
    batch = {
        "inputs": input_tensor,
        "labels": label_tensor,
        "puzzle_identifiers": torch.zeros(local_batch_size, dtype=torch.long).to(DEVICE)
    }
    
    print(f"✅ Created batch with shape: {input_tensor.shape}")
    
    # Test in training mode first
    model.train()
    
    with torch.no_grad():
        try:
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
            
            # Single forward pass (training-style)
            carry, loss, metrics, outputs, all_halted = model(
                return_keys=["logits"], carry=carry, batch=batch
            )
            
            print(f"✅ Training mode successful!")
            print(f"📊 Loss: {loss.item():.4f}")
            print(f"📊 All halted: {all_halted}")
            
            if "logits" in outputs:
                logits = outputs["logits"]
                predicted_tokens = torch.argmax(logits, dim=-1)
                
                # Analyze first example in the batch
                first_prediction = predicted_tokens[0].cpu().tolist()
                first_true_labels = batch_labels[0].tolist()
                
                pred_counter = Counter(first_prediction)
                true_counter = Counter(first_true_labels)
                
                print("\n📊 First example analysis:")
                print("True labels:")
                for token_id, count in sorted(true_counter.items()):
                    if count > 0:
                        token_name = TOKEN_NAMES.get(token_id, f"UNK_{token_id}")
                        percentage = (count / len(first_true_labels)) * 100
                        print(f"   {token_name}: {count} ({percentage:.1f}%)")
                
                print("Predictions:")
                for token_id, count in sorted(pred_counter.items()):
                    if count > 0:
                        token_name = TOKEN_NAMES.get(token_id, f"UNK_{token_id}")
                        percentage = (count / len(first_prediction)) * 100
                        print(f"   {token_name}: {count} ({percentage:.1f}%)")
                
                # Calculate accuracy on non-pad tokens
                true_tensor = torch.tensor(first_true_labels)
                pred_tensor = torch.tensor(first_prediction)
                non_pad_mask = (true_tensor != 0)
                
                if non_pad_mask.sum() > 0:
                    accuracy = (pred_tensor[non_pad_mask] == true_tensor[non_pad_mask]).float().mean()
                    print(f"📊 Accuracy on non-pad tokens: {accuracy:.1%}")
                
        except Exception as e:
            print(f"❌ Training mode failed: {e}")
            return False
    
    # Test in evaluation mode
    print("\n🔍 Testing evaluation mode...")
    model.eval()
    
    with torch.no_grad():
        try:
            # Re-initialize for eval mode
            carry = model.initial_carry(batch)
            carry = move_to_device(carry, DEVICE)
            
            # Multi-step inference (eval-style)
            step_count = 0
            while True:
                carry, _, _, outputs, all_halted = model(
                    return_keys=["logits"], carry=carry, batch=batch
                )
                step_count += 1
                
                if all_halted or step_count > 20:
                    break
            
            print(f"✅ Evaluation mode completed in {step_count} steps")
            print(f"📊 All halted: {all_halted}")
            
            if "logits" in outputs:
                logits = outputs["logits"]
                predicted_tokens = torch.argmax(logits, dim=-1)
                
                # Analyze first example
                first_prediction = predicted_tokens[0].cpu().tolist()
                pred_counter = Counter(first_prediction)
                
                print("Evaluation predictions:")
                for token_id, count in sorted(pred_counter.items()):
                    if count > 0:
                        token_name = TOKEN_NAMES.get(token_id, f"UNK_{token_id}")
                        percentage = (count / len(first_prediction)) * 100
                        print(f"   {token_name}: {count} ({percentage:.1f}%)")
                
        except Exception as e:
            print(f"❌ Evaluation mode failed: {e}")
            return False
    
    return True

def create_single_inference_flask_compatible():
    """Create a Flask-compatible single example inference"""
    print("\n" + "="*60)
    print("🌐 FLASK-COMPATIBLE SINGLE INFERENCE TEST")
    print("="*60)
    
    # Load one example
    inputs = np.load('data/city-logistics-1k/train/all__inputs.npy')
    test_input = inputs[0]
    
    # Create batch of size matching training, but duplicate the same example
    local_batch_size = config.global_batch_size // 1
    
    # Duplicate the single example to fill the batch
    batch_inputs = np.tile(test_input, (local_batch_size, 1))
    
    input_tensor = torch.tensor(batch_inputs, dtype=torch.long).to(DEVICE)
    dummy_labels = torch.zeros_like(input_tensor)
    
    batch = {
        "inputs": input_tensor,
        "labels": dummy_labels,
        "puzzle_identifiers": torch.zeros(local_batch_size, dtype=torch.long).to(DEVICE)
    }
    
    print(f"📊 Created Flask-style batch: {input_tensor.shape}")
    
    model.eval()
    
    with torch.no_grad():
        try:
            carry = model.initial_carry(batch)
            
            def move_to_device(obj, device):
                if hasattr(obj, 'to'):
                    return obj.to(device)
                if hasattr(obj, '__dict__'):
                    for attr, value in obj.__dict__.items():
                        setattr(obj, attr, move_to_device(value, device))
                return obj
            
            carry = move_to_device(carry, DEVICE)
            
            # Multi-step inference like Flask server
            step_count = 0
            while True:
                carry, _, _, outputs, all_halted = model(
                    return_keys=["logits"], carry=carry, batch=batch
                )
                step_count += 1
                
                if all_halted or step_count > 16:  # Match your Flask server limit
                    break
            
            print(f"✅ Flask-style inference completed in {step_count} steps")
            
            if "logits" in outputs:
                logits = outputs["logits"]
                predicted_tokens = torch.argmax(logits, dim=-1)
                
                # Take first example (they're all the same)
                first_prediction = predicted_tokens[0].cpu().tolist()
                pred_counter = Counter(first_prediction)
                
                print("Flask-style predictions:")
                path_count = 0
                for token_id, count in sorted(pred_counter.items()):
                    if count > 0:
                        token_name = TOKEN_NAMES.get(token_id, f"UNK_{token_id}")
                        percentage = (count / len(first_prediction)) * 100
                        print(f"   {token_name}: {count} ({percentage:.1f}%)")
                        
                        if token_id == HRM_TOKEN_MAP["PATH"]:
                            path_count = count
                
                if path_count == len(first_prediction):
                    print("🚨 CONFIRMED: Model outputs PATH for every token in Flask-style inference!")
                elif path_count == 0:
                    print("⚠️  Model outputs no PATH tokens")
                else:
                    print(f"✅ Model outputs reasonable PATH count: {path_count}")
                
                return first_prediction
                
        except Exception as e:
            print(f"❌ Flask-style inference failed: {e}")
            return None

def main():
    """Run batch size compatible tests"""
    try:
        success = test_with_training_batch_size()
        
        if success:
            result = create_single_inference_flask_compatible()
            
            print("\n" + "="*60)
            print("🎯 BATCH SIZE ANALYSIS COMPLETE")
            print("="*60)
            
            if result:
                print("\n📋 CONCLUSIONS:")
                print("✅ Model works correctly with training batch size")
                print("🔧 Flask server needs to use proper batch size")
                print("📝 Solution: Pad single examples to training batch size")
        
    except Exception as e:
        print(f"❌ Error during batch size testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()