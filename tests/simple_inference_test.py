#!/usr/bin/env python3
"""
Simple HRM Inference Test
Tests if the trained model actually learned by running inference on training data
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
CHECKPOINT_PATH = "checkpoints/City-logistics-1k ACT-torch/HierarchicalReasoningModel_ACTV1 violet-potoo/step_2240"

def load_model_and_config():
    """Load the trained model and its configuration"""
    print(">>> Loading model configuration...")
    
    # Load configuration
    config_path = f"{'/'.join(CHECKPOINT_PATH.split('/')[:-1])}/all_config.yaml"
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
        config_dict['data_path'] = 'data/city-logistics-1k'
        config = PretrainConfig(**config_dict)
    
    # Create dummy metadata
    dummy_metadata = PuzzleDatasetMetadata(
        pad_id=0, ignore_label_id=-100, blank_identifier_id=0,
        vocab_size=10, seq_len=1600, num_puzzle_identifiers=1,
        total_groups=1, mean_puzzle_examples=1.0, sets=['all']
    )
    
    # Create and load model
    print(">>> Creating model...")
    model, _, _ = create_model(config, dummy_metadata, world_size=1)
    
    print(">>> Loading checkpoint...")
    try:
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
    except:
        # Handle torch.compile wrapped models
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
        cleaned_state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned_state_dict)
    
    model.to(DEVICE)
    model.eval()
    print(">>> Model loaded successfully!")
    
    return model, config

def load_training_data():
    """Load real training data for testing"""
    print(">>> Loading training data...")
    
    inputs = np.load('data/city-logistics-1k/train/all__inputs.npy')
    labels = np.load('data/city-logistics-1k/train/all__labels.npy')
    
    print(f">>> Loaded {len(inputs)} training examples")
    return inputs, labels

def test_single_example(model, config, inputs, labels, example_idx=0):
    """Test model on a single training example"""
    print(f"\n>>> Testing on training example {example_idx}")
    
    # Get the example
    test_input = inputs[example_idx]
    true_labels = labels[example_idx]
    
    # Analyze true data
    print("True data analysis:")
    input_counter = Counter(test_input)
    label_counter = Counter(true_labels)
    
    path_tokens_true = label_counter.get(9, 0)  # PATH token = 9
    print(f"  True PATH tokens: {path_tokens_true} out of {len(true_labels)}")
    print(f"  True PATH percentage: {path_tokens_true/len(true_labels)*100:.1f}%")
    
    # Create proper batch (matching training format)
    batch_size = config.global_batch_size
    
    # Duplicate example to fill batch
    batch_inputs = np.tile(test_input, (batch_size, 1))
    batch_labels = np.tile(true_labels, (batch_size, 1))
    
    input_tensor = torch.tensor(batch_inputs, dtype=torch.long).to(DEVICE)
    label_tensor = torch.tensor(batch_labels, dtype=torch.long).to(DEVICE)
    
    batch = {
        "inputs": input_tensor,
        "labels": label_tensor,
        "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.long).to(DEVICE)
    }
    
    print(f">>> Created batch with shape: {input_tensor.shape}")
    
    return batch, test_input, true_labels

def run_inference(model, batch):
    """Run inference on the batch"""
    print(">>> Running inference...")
    
    with torch.no_grad():
        # Initialize carry state
        carry = model.initial_carry(batch)
        
        # Move carry to device properly
        def move_to_device(obj, device):
            if hasattr(obj, 'to'):
                return obj.to(device)
            if hasattr(obj, '__dict__'):
                for attr, value in obj.__dict__.items():
                    setattr(obj, attr, move_to_device(value, device))
            return obj
        
        carry = move_to_device(carry, DEVICE)
        
        # Run ACT inference (like during training)
        step_count = 0
        max_steps = 20
        
        while step_count < max_steps:
            carry, _, metrics, outputs, all_halted = model(
                return_keys=["logits"], carry=carry, batch=batch
            )
            step_count += 1
            
            if all_halted:
                print(f">>> Model halted after {step_count} steps")
                break
        
        if step_count >= max_steps:
            print(f">>> Model reached max steps ({max_steps})")
        
        return outputs, metrics, step_count

def analyze_results(outputs, true_labels, test_input):
    """Analyze the model's predictions"""
    print("\n>>> Analyzing results...")
    
    if "logits" not in outputs:
        print("❌ No logits in output!")
        return False
    
    logits = outputs["logits"]
    print(f"Logits shape: {logits.shape}")
    
    # Get predictions for first example (they're all identical)
    predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()
    
    # Analyze predictions
    pred_counter = Counter(predictions)
    true_counter = Counter(true_labels)
    
    print("\nPrediction analysis:")
    print(f"  Predicted tokens: {dict(pred_counter)}")
    
    path_tokens_pred = pred_counter.get(9, 0)  # PATH token = 9
    path_tokens_true = true_counter.get(9, 0)
    
    print(f"  Predicted PATH tokens: {path_tokens_pred}")
    print(f"  True PATH tokens: {path_tokens_true}")
    print(f"  PATH token ratio: {path_tokens_pred/path_tokens_true:.2f}x" if path_tokens_true > 0 else "N/A")
    
    # Check if model is just outputting PATH everywhere
    if path_tokens_pred == len(predictions):
        print("❌ CRITICAL: Model outputs PATH for every token!")
        print("❌ This means the model never learned proper pathfinding")
        return False
    
    # Check accuracy on non-PAD tokens
    non_pad_mask = (true_labels != 0)
    if non_pad_mask.sum() > 0:
        accuracy = (predictions[non_pad_mask] == true_labels[non_pad_mask]).mean()
        print(f"  Accuracy on non-PAD tokens: {accuracy:.1%}")
        
        if accuracy > 0.8:
            print("✅ Model shows good accuracy!")
            return True
        else:
            print("⚠️ Model accuracy is low")
            return False
    
    return False

def main():
    """Main test function"""
    print("="*60)
    print("🧠 SIMPLE HRM INFERENCE TEST")
    print("="*60)
    
    try:
        # Load model
        model, config = load_model_and_config()
        
        # Load training data
        inputs, labels = load_training_data()
        
        # Test on multiple examples
        success_count = 0
        test_examples = [0, 1, 2, 10, 50]  # Test on various examples
        
        for example_idx in test_examples:
            if example_idx >= len(inputs):
                continue
                
            print(f"\n{'='*40}")
            print(f"Testing Example {example_idx}")
            print(f"{'='*40}")
            
            # Prepare test
            batch, test_input, true_labels = test_single_example(
                model, config, inputs, labels, example_idx
            )
            
            # Run inference
            outputs, metrics, steps = run_inference(model, batch)
            
            # Analyze results
            success = analyze_results(outputs, true_labels, test_input)
            
            if success:
                success_count += 1
            
            print(f"Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
        
        # Final verdict
        print(f"\n{'='*60}")
        print("🎯 FINAL VERDICT")
        print(f"{'='*60}")
        
        success_rate = success_count / len(test_examples)
        print(f"Success rate: {success_count}/{len(test_examples)} ({success_rate:.1%})")
        
        if success_rate >= 0.8:
            print("✅ MODEL IS PROPERLY TRAINED!")
            print("✅ Issue is likely in Flask server or inference pipeline")
        elif success_rate >= 0.4:
            print("⚠️ MODEL PARTIALLY TRAINED")
            print("⚠️ May need more training or has convergence issues")
        else:
            print("❌ MODEL NOT PROPERLY TRAINED")
            print("❌ Training likely failed or checkpoint is corrupted")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()