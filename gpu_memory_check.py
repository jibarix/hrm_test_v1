#!/usr/bin/env python3
"""
GPU Memory Check Script
Estimates memory usage for different batch sizes with HRM model.
"""

import torch
import gc

def check_gpu_memory():
    """Check available GPU memory"""
    print("🔍 GPU Memory Check")
    print("="*40)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return False
    
    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
    
    print(f"🎮 GPU: {gpu_name}")
    print(f"💾 Total Memory: {total_memory:.1f} GB")
    
    # Clear any existing memory
    torch.cuda.empty_cache()
    gc.collect()
    
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    cached = torch.cuda.memory_reserved(device) / 1024**3
    free = total_memory - cached
    
    print(f"📊 Current Usage:")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Cached: {cached:.2f} GB")
    print(f"  Free: {free:.2f} GB")
    
    return True

def estimate_model_memory():
    """Estimate HRM model memory usage"""
    print(f"\n🧠 HRM Model Memory Estimation")
    print("="*40)
    
    # HRM model parameters (from config/arch/hrm_v1.yaml)
    hidden_size = 256
    num_heads = 8
    H_layers = 4
    L_layers = 4
    seq_len = 1600
    vocab_size = 10
    
    # Estimate parameter count
    # Input embedding
    emb_params = vocab_size * hidden_size
    
    # Each transformer layer (attention + MLP)
    attention_params = hidden_size * hidden_size * 4  # qkv + output
    mlp_params = hidden_size * (hidden_size * 4) * 2  # up + down projection
    layer_params = attention_params + mlp_params
    
    total_layers = H_layers + L_layers
    total_params = emb_params + (total_layers * layer_params)
    
    # Add output head and other components
    total_params += hidden_size * vocab_size  # output head
    total_params += hidden_size * 2  # initial states
    
    param_memory_gb = total_params * 4 / 1024**3  # 4 bytes per float32
    
    print(f"📈 Parameter Estimation:")
    print(f"  Total parameters: ~{total_params/1e6:.1f}M")
    print(f"  Parameter memory: ~{param_memory_gb:.2f} GB")
    
    return total_params, param_memory_gb

def estimate_batch_memory(batch_size):
    """Estimate memory usage for a given batch size"""
    seq_len = 1600
    hidden_size = 256
    
    # Input tensors
    input_memory = batch_size * seq_len * 4 / 1024**3  # int32
    
    # Hidden states (rough estimate)
    hidden_memory = batch_size * seq_len * hidden_size * 4 / 1024**3  # float32
    
    # Gradients (roughly same as parameters)
    total_memory = input_memory + hidden_memory * 3  # activations + gradients
    
    return total_memory

def test_batch_sizes():
    """Test different batch sizes to find optimal"""
    print(f"\n📊 Batch Size Memory Analysis")
    print("="*40)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - skipping memory test")
        return
    
    _, param_memory = estimate_model_memory()
    
    batch_sizes = [16, 32, 64, 128, 256]
    
    print(f"{'Batch Size':<12} {'Est. Memory':<12} {'Total':<12} {'Status'}")
    print("-" * 50)
    
    for batch_size in batch_sizes:
        batch_memory = estimate_batch_memory(batch_size)
        total_memory = param_memory + batch_memory
        
        # Check if it fits in 8GB (RTX 3070 Ti)
        if total_memory < 6.0:  # Leave 2GB headroom
            status = "✅ Safe"
        elif total_memory < 7.0:
            status = "⚠️  Risky"
        else:
            status = "❌ Too large"
        
        print(f"{batch_size:<12} {batch_memory:.2f} GB{'':<4} {total_memory:.2f} GB{'':<4} {status}")

def recommend_settings():
    """Recommend optimal settings"""
    print(f"\n🎯 Recommendations for RTX 3070 Ti")
    print("="*40)
    
    print("✅ Recommended settings:")
    print("  global_batch_size: 64")
    print("  hidden_size: 256 (current)")
    print("  Enable gradient checkpointing if available")
    print("  Use mixed precision (fp16) if supported")
    
    print("\n⚠️  If still OOM (Out of Memory):")
    print("  1. Reduce global_batch_size to 32")
    print("  2. Reduce hidden_size to 128")
    print("  3. Reduce max_steps in ACT")
    
    print("\n🚀 Performance tips:")
    print("  1. Use $env:DISABLE_COMPILE=1 (already doing)")
    print("  2. Close other GPU applications")
    print("  3. Monitor with: nvidia-smi")

def main():
    print("🔧 HRM GPU Memory Analysis Tool")
    print("="*50)
    
    check_gpu_memory()
    estimate_model_memory()
    test_batch_sizes()
    recommend_settings()
    
    print(f"\n🎯 Next Steps:")
    print("1. Use the fixed config with global_batch_size: 64")
    print("2. Monitor GPU memory during training")
    print("3. Reduce batch size further if needed")

if __name__ == "__main__":
    main()
