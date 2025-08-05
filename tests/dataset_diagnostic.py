#!/usr/bin/env python3
"""
Dataset Quality Diagnostic
Check if the training data is corrupted
"""

import numpy as np
import json
from collections import Counter

# Token mapping
HRM_TOKEN_MAP = {
    "PAD": 0, "OBSTACLE": 1, "SMALL_ROAD": 2, "LARGE_ROAD": 3, "DIAGONAL": 4,
    "TRAFFIC_JAM": 5, "ROAD_CLOSURE": 6, "START": 7, "END": 8, "PATH": 9
}

TOKEN_NAMES = {v: k for k, v in HRM_TOKEN_MAP.items()}

def analyze_dataset():
    """Analyze the training dataset quality"""
    print("="*60)
    print("🔍 ANALYZING TRAINING DATASET QUALITY")
    print("="*60)
    
    try:
        # Load training data
        print("📂 Loading training data...")
        inputs = np.load('data/city-logistics-1k/train/all__inputs.npy')
        labels = np.load('data/city-logistics-1k/train/all__labels.npy')
        
        print(f"✅ Loaded {len(inputs)} training examples")
        print(f"   Input shape: {inputs.shape}")
        print(f"   Label shape: {labels.shape}")
        
        # Analyze first few examples
        print(f"\n📊 ANALYZING FIRST 5 EXAMPLES:")
        
        for i in range(min(5, len(inputs))):
            print(f"\n--- Example {i} ---")
            
            # Input analysis
            input_tokens = inputs[i]
            input_counter = Counter(input_tokens)
            print("Input tokens:")
            for token_id, count in sorted(input_counter.items()):
                if count > 0:
                    token_name = TOKEN_NAMES.get(token_id, f"UNK_{token_id}")
                    percentage = (count / len(input_tokens)) * 100
                    print(f"   {token_name}: {count} ({percentage:.1f}%)")
            
            # Label analysis
            label_tokens = labels[i]
            label_counter = Counter(label_tokens)
            print("Label tokens:")
            for token_id, count in sorted(label_counter.items()):
                if count > 0:
                    token_name = TOKEN_NAMES.get(token_id, f"UNK_{token_id}")
                    percentage = (count / len(label_tokens)) * 100
                    print(f"   {token_name}: {count} ({percentage:.1f}%)")
            
            # Critical checks
            path_tokens_in_labels = label_counter.get(HRM_TOKEN_MAP["PATH"], 0)
            total_label_tokens = len(label_tokens)
            path_percentage = (path_tokens_in_labels / total_label_tokens) * 100
            
            if path_percentage > 50:
                print(f"   🚨 CRITICAL: {path_percentage:.1f}% PATH tokens in labels!")
            elif path_percentage == 0:
                print(f"   ⚠️  WARNING: No PATH tokens in labels")
            else:
                print(f"   ✅ PATH percentage looks reasonable: {path_percentage:.1f}%")
        
        # Overall dataset statistics
        print(f"\n📈 OVERALL DATASET STATISTICS:")
        
        # Flatten all inputs and labels for overall analysis
        all_inputs = inputs.flatten()
        all_labels = labels.flatten()
        
        input_overall = Counter(all_inputs)
        label_overall = Counter(all_labels)
        
        print("\nOverall input distribution:")
        for token_id, count in sorted(input_overall.items()):
            if count > 0:
                token_name = TOKEN_NAMES.get(token_id, f"UNK_{token_id}")
                percentage = (count / len(all_inputs)) * 100
                print(f"   {token_name}: {count:,} ({percentage:.1f}%)")
        
        print("\nOverall label distribution:")
        for token_id, count in sorted(label_overall.items()):
            if count > 0:
                token_name = TOKEN_NAMES.get(token_id, f"UNK_{token_id}")
                percentage = (count / len(all_labels)) * 100
                print(f"   {token_name}: {count:,} ({percentage:.1f}%)")
        
        # Critical analysis
        total_path_labels = label_overall.get(HRM_TOKEN_MAP["PATH"], 0)
        total_labels = len(all_labels)
        overall_path_percentage = (total_path_labels / total_labels) * 100
        
        print(f"\n🎯 CRITICAL ANALYSIS:")
        print(f"   Overall PATH percentage in labels: {overall_path_percentage:.1f}%")
        
        if overall_path_percentage > 30:
            print("   🚨 DATASET IS CORRUPTED: Too many PATH tokens!")
            print("   🚨 This explains why the model outputs PATH everywhere")
            print("   🚨 Expected: 2-5% PATH tokens, Got: {:.1f}%".format(overall_path_percentage))
        elif overall_path_percentage < 0.5:
            print("   ⚠️  Very few PATH tokens - might be sparse labeling issue")
        else:
            print("   ✅ PATH token percentage looks reasonable")
        
        # Check for ignore tokens (padding)
        ignore_tokens = label_overall.get(-100, 0) + label_overall.get(0, 0)  # Common ignore values
        if ignore_tokens > 0:
            ignore_percentage = (ignore_tokens / total_labels) * 100
            print(f"   📝 Ignore/Pad tokens: {ignore_tokens:,} ({ignore_percentage:.1f}%)")
        
    except FileNotFoundError as e:
        print(f"❌ Dataset files not found: {e}")
        print("   Make sure you've converted the dataset to .npy format")
    except Exception as e:
        print(f"❌ Error analyzing dataset: {e}")
        import traceback
        traceback.print_exc()

def visualize_example():
    """Visualize a single example to see the problem"""
    print("\n" + "="*60)
    print("🖼️  VISUALIZING TRAINING EXAMPLE")
    print("="*60)
    
    try:
        inputs = np.load('data/city-logistics-1k/train/all__inputs.npy')
        labels = np.load('data/city-logistics-1k/train/all__labels.npy')
        
        # Take first example
        input_seq = inputs[0].reshape(40, 40)
        label_seq = labels[0].reshape(40, 40)
        
        print("Input grid (first 10x10 section):")
        for y in range(10):
            row = []
            for x in range(10):
                token = input_seq[y, x]
                token_name = TOKEN_NAMES.get(token, f"?{token}")
                row.append(token_name[:4].ljust(4))
            print("   " + " ".join(row))
        
        print("\nLabel grid (first 10x10 section):")
        for y in range(10):
            row = []
            for x in range(10):
                token = label_seq[y, x]
                token_name = TOKEN_NAMES.get(token, f"?{token}")
                if token_name == "PATH":
                    row.append("PATH")
                else:
                    row.append(token_name[:4].ljust(4))
            print("   " + " ".join(row))
        
    except Exception as e:
        print(f"❌ Error visualizing: {e}")

def main():
    analyze_dataset()
    visualize_example()
    
    print("\n" + "="*60)
    print("🎯 DIAGNOSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()