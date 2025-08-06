#!/usr/bin/env python3
"""
HRM Training Debug Script
Diagnoses common issues with HRM training setup and provides fixes.
"""

import sys
import os
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from pathlib import Path
import torch
from collections import Counter

def check_dataset_structure(data_path="data/city-logistics-1k"):
    """Check if dataset is properly formatted for HRM training"""
    print(f"🔍 Checking dataset structure at: {data_path}")
    
    data_dir = Path(data_path)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    
    # Check if directories exist
    if not data_dir.exists():
        print(f"❌ Dataset directory not found: {data_dir}")
        print("📝 You need to generate the dataset first!")
        return False
    
    if not train_dir.exists():
        print(f"❌ Train directory not found: {train_dir}")
        return False
    
    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        return False
    
    # Check required files
    required_files = [
        "all__inputs.npy",
        "all__labels.npy", 
        "all__puzzle_identifiers.npy",
        "all__puzzle_indices.npy",
        "all__group_indices.npy",
        "dataset.json"
    ]
    
    print("\n📂 Checking required files...")
    for split in ["train", "test"]:
        split_dir = data_dir / split
        print(f"\n  {split.upper()} split:")
        
        for file_name in required_files:
            file_path = split_dir / file_name
            if file_path.exists():
                if file_name.endswith('.npy'):
                    try:
                        arr = np.load(file_path)
                        print(f"    ✅ {file_name}: shape {arr.shape}, dtype {arr.dtype}")
                    except Exception as e:
                        print(f"    ❌ {file_name}: Error loading - {e}")
                else:
                    print(f"    ✅ {file_name}: Found")
            else:
                print(f"    ❌ {file_name}: Missing")
                return False
    
    return True

def validate_dataset_content(data_path="data/city-logistics-1k"):
    """Validate dataset content and format"""
    print(f"\n🧪 Validating dataset content...")
    
    data_dir = Path(data_path)
    
    try:
        # Load train data
        train_inputs = np.load(data_dir / "train" / "all__inputs.npy")
        train_labels = np.load(data_dir / "train" / "all__labels.npy")
        train_puzzle_ids = np.load(data_dir / "train" / "all__puzzle_identifiers.npy")
        train_puzzle_indices = np.load(data_dir / "train" / "all__puzzle_indices.npy")
        train_group_indices = np.load(data_dir / "train" / "all__group_indices.npy")
        
        # Load test data
        test_inputs = np.load(data_dir / "test" / "all__inputs.npy")
        test_labels = np.load(data_dir / "test" / "all__labels.npy")
        
        print(f"📊 Dataset Statistics:")
        print(f"  Train examples: {len(train_inputs)}")
        print(f"  Test examples: {len(test_inputs)}")
        print(f"  Sequence length: {train_inputs.shape[1] if len(train_inputs.shape) > 1 else 'N/A'}")
        print(f"  Train groups: {len(train_group_indices) - 1}")
        
        # Check for expected HRM logistics dataset size
        expected_train = 960  # 240 base scenarios × 4 vehicles
        expected_test = 400   # 100 base scenarios × 4 vehicles
        
        if len(train_inputs) != expected_train:
            print(f"⚠️  Train size mismatch: {len(train_inputs)} vs expected {expected_train}")
        
        if len(test_inputs) != expected_test:
            print(f"⚠️  Test size mismatch: {len(test_inputs)} vs expected {expected_test}")
        
        # Check token distribution
        print(f"\n📈 Token Distribution Analysis:")
        all_inputs = np.concatenate([train_inputs.flatten(), test_inputs.flatten()])
        all_labels = np.concatenate([train_labels.flatten(), test_labels.flatten()])
        
        input_counts = Counter(all_inputs)
        label_counts = Counter(all_labels)
        
        print("  Input tokens:")
        for token_id, count in sorted(input_counts.items())[:10]:  # Top 10
            percentage = count / len(all_inputs) * 100
            print(f"    Token {token_id}: {count:,} ({percentage:.2f}%)")
        
        print("  Label tokens:")
        for token_id, count in sorted(label_counts.items())[:10]:  # Top 10
            percentage = count / len(all_labels) * 100
            print(f"    Token {token_id}: {count:,} ({percentage:.2f}%)")
        
        # Check for all zeros (common error)
        if np.all(train_inputs == 0):
            print("❌ All input tokens are zero - dataset generation failed!")
            return False
        
        if np.all(train_labels == 0):
            print("❌ All label tokens are zero - path generation failed!")
            return False
        
        print("✅ Dataset content validation passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error validating dataset: {e}")
        return False

def check_training_config(config_path="config/logistics_routing.yaml"):
    """Check training configuration for common issues"""
    print(f"\n⚙️ Checking training configuration...")
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"📋 Configuration Summary:")
        print(f"  Data path: {config.get('data_path', 'Not set')}")
        print(f"  Global batch size: {config.get('global_batch_size', 'Not set')}")
        print(f"  Epochs: {config.get('epochs', 'Not set')}")
        print(f"  Learning rate: {config.get('lr', 'Not set')}")
        print(f"  Eval interval: {config.get('eval_interval', 'Not set')}")
        
        # Check for potential issues
        batch_size = config.get('global_batch_size', 256)
        if batch_size > 960:  # Our training set size
            print(f"⚠️  Batch size ({batch_size}) larger than training set (960)")
            print("   Consider reducing global_batch_size to 64 or 128")
        
        data_path = config.get('data_path', '')
        if not os.path.exists(data_path):
            print(f"❌ Data path in config doesn't exist: {data_path}")
            return False
        
        print("✅ Configuration validation passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error checking config: {e}")
        return False

def test_data_loading():
    """Test actual data loading with PuzzleDataset"""
    print(f"\n🔄 Testing data loading...")
    
    try:
        from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
        
        # Create dataset config (mimicking pretrain.py)
        config = PuzzleDatasetConfig(
            seed=0,
            dataset_path="data/city-logistics-1k",
            global_batch_size=32,  # Smaller for testing
            test_set_mode=False,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1
        )
        
        print("📦 Creating training dataset...")
        train_dataset = PuzzleDataset(config, split="train")
        
        print("📊 Dataset metadata:")
        print(f"  Vocab size: {train_dataset.metadata.vocab_size}")
        print(f"  Sequence length: {train_dataset.metadata.seq_len}")
        print(f"  Total groups: {train_dataset.metadata.total_groups}")
        print(f"  Mean examples per group: {train_dataset.metadata.mean_puzzle_examples}")
        
        print("🔄 Testing batch generation...")
        batch_count = 0
        for set_name, batch, global_batch_size in train_dataset:
            print(f"  Batch {batch_count}:")
            print(f"    Set: {set_name}")
            print(f"    Batch size: {global_batch_size}")
            print(f"    Input shape: {batch['inputs'].shape}")
            print(f"    Label shape: {batch['labels'].shape}")
            print(f"    Puzzle IDs shape: {batch['puzzle_identifiers'].shape}")
            
            batch_count += 1
            if batch_count >= 3:  # Test first 3 batches
                break
        
        if batch_count == 0:
            print("❌ No batches generated - data loading failed!")
            return False
        
        print("✅ Data loading test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing data loading: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_fix_commands():
    """Generate commands to fix common issues"""
    print(f"\n🔧 Fix Commands:")
    
    print("1. Generate dataset (if missing):")
    print("   - Open logistics_game.html in browser")
    print("   - Click '📊 Generate Paper Dataset'")
    print("   - Download ZIP and extract to dataset/raw-data/CityLogistics/")
    
    print("\n2. Convert dataset to HRM format:")
    print("   python dataset/build_logistics_dataset.py \\")
    print("     --source-dir dataset/raw-data/CityLogistics \\")
    print("     --output-dir data/city-logistics-1k")
    
    print("\n3. Validate dataset:")
    print("   python data_analysis_script.py")
    
    print("\n4. Fix training config (if needed):")
    print("   # Edit config/logistics_routing.yaml")
    print("   # Reduce global_batch_size to 64 or 128")
    print("   # Verify data_path points to correct location")
    
    print("\n5. Run training with debug:")
    print("   $env:DISABLE_COMPILE=1")
    print("   python pretrain.py --config-name=logistics_routing")

def main():
    """Main debugging function"""
    print("🧠 HRM Training Debug Tool")
    print("="*50)
    
    issues_found = []
    
    # Check dataset structure
    if not check_dataset_structure():
        issues_found.append("Dataset structure")
    
    # Validate content if structure is OK
    elif not validate_dataset_content():
        issues_found.append("Dataset content")
    
    # Check training config
    if not check_training_config():
        issues_found.append("Training configuration")
    
    # Test data loading
    if not test_data_loading():
        issues_found.append("Data loading")
    
    print("\n" + "="*50)
    if issues_found:
        print(f"❌ Issues found: {', '.join(issues_found)}")
        generate_fix_commands()
    else:
        print("✅ All checks passed! Training should work.")
        print("\nIf training still hangs, try:")
        print("1. Reduce batch size in config")
        print("2. Check GPU memory with nvidia-smi")
        print("3. Run with smaller epochs (epochs: 1) for testing")
    
    print("\n🎯 Next: Run this script to identify specific issues!")

if __name__ == "__main__":
    main()
