#!/usr/bin/env python3
"""
HRM GPU Memory Check Script
Accurate memory estimation for Hierarchical Reasoning Model training.
Based on HRM architecture and city logistics dataset requirements.
"""

import torch
import gc

def check_gpu_memory():
    """Check available GPU memory"""
    print("🔍 GPU Memory Check")
    print("="*40)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return False, 0
    
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
    
    return True, total_memory

def estimate_hrm_parameters():
    """Estimate HRM model parameters based on actual architecture"""
    print(f"\n🧠 HRM Model Parameter Estimation")
    print("="*40)
    
    # HRM config (from config/arch/hrm_v1.yaml)
    hidden_size = 256
    num_heads = 4  # Corrected from 8
    expansion = 4  # SwiGLU expansion factor
    H_layers = 4
    L_layers = 4
    seq_len = 1600  # 40x40 grid
    vocab_size = 10
    puzzle_emb_ndim = hidden_size  # 256
    num_puzzle_identifiers = 1  # City logistics has 1 puzzle type
    
    params = {}
    
    # 1. Input/Output layers
    params['input_embedding'] = vocab_size * hidden_size
    params['output_head'] = hidden_size * vocab_size
    
    # 2. Puzzle embeddings (sparse, batch-dependent)
    params['puzzle_embedding'] = num_puzzle_identifiers * puzzle_emb_ndim
    
    # 3. Single transformer block parameters
    def transformer_block_params(hidden_size, num_heads, expansion):
        # Attention: qkv projection + output projection
        attn_params = hidden_size * (hidden_size * 3) + hidden_size * hidden_size
        
        # SwiGLU MLP: gate_up (2x expanded) + down
        mlp_inner = int(expansion * hidden_size * 2 / 3)
        # Round to multiple of 256 (as in code)
        mlp_inner = ((mlp_inner + 255) // 256) * 256
        mlp_params = hidden_size * (mlp_inner * 2) + mlp_inner * hidden_size
        
        # RMSNorm parameters (scale only, no bias)
        norm_params = hidden_size * 2  # Two norm layers per block
        
        return attn_params + mlp_params + norm_params
    
    block_params = transformer_block_params(hidden_size, num_heads, expansion)
    
    # 4. H-module and L-module
    params['H_module'] = H_layers * block_params
    params['L_module'] = L_layers * block_params
    
    # 5. Initial states (learnable)
    params['initial_states'] = hidden_size * 2  # H_init + L_init
    
    # 6. Q-head for ACT
    params['q_head'] = hidden_size * 2 + 2  # Linear layer + bias for halt/continue
    
    # 7. Position embeddings (if using learned pos encodings)
    params['position_embedding'] = (seq_len + (puzzle_emb_ndim // hidden_size)) * hidden_size
    
    total_params = sum(params.values())
    param_memory_gb = total_params * 4 / 1024**3  # float32
    
    print(f"📈 HRM Parameter Breakdown:")
    for component, count in params.items():
        print(f"  {component:<20}: {count/1e6:>6.2f}M")
    print(f"  {'='*20}  {'='*8}")
    print(f"  {'Total':<20}: {total_params/1e6:>6.2f}M")
    print(f"  Parameter memory: {param_memory_gb:.3f} GB")
    
    return total_params, param_memory_gb

def estimate_hrm_training_memory(batch_size, total_params):
    """Estimate complete HRM training memory usage"""
    
    # Dataset parameters
    seq_len = 1600
    hidden_size = 256
    
    # 1. Input tensors (batch_size, seq_len)
    input_memory = batch_size * seq_len * 4 / 1024**3  # int32
    label_memory = batch_size * seq_len * 4 / 1024**3  # int32
    
    # 2. HRM-specific activations
    # Carry states (zH, zL) for hierarchical reasoning
    carry_memory = batch_size * seq_len * hidden_size * 2 * 4 / 1024**3  # float32
    
    # 3. Transformer activations (rough estimate)
    # Each layer needs intermediate activations
    total_layers = 8  # 4 H + 4 L
    activation_memory = batch_size * seq_len * hidden_size * total_layers * 4 / 1024**3
    
    # 4. Deep supervision: Multiple forward passes
    # HRM uses deep supervision with multiple segments
    supervision_multiplier = 2  # Conservative estimate for supervision overhead
    
    # 5. Optimizer states (AdamATan2)
    # Adam stores momentum (m) and variance (v) for each parameter
    optimizer_memory = total_params * 4 * 2 / 1024**3  # 2x params for m,v
    
    # 6. Gradients
    gradient_memory = total_params * 4 / 1024**3
    
    # Total training memory
    model_memory = input_memory + label_memory + carry_memory + activation_memory
    training_memory = model_memory * supervision_multiplier + optimizer_memory + gradient_memory
    
    breakdown = {
        'Input/Labels': input_memory + label_memory,
        'Carry States': carry_memory,
        'Activations': activation_memory,
        'Supervision Overhead': model_memory * (supervision_multiplier - 1),
        'Optimizer States': optimizer_memory,
        'Gradients': gradient_memory
    }
    
    return training_memory, breakdown

def test_batch_sizes(total_params, param_memory, gpu_memory):
    """Test different batch sizes for HRM training"""
    print(f"\n📊 HRM Batch Size Analysis")
    print("="*60)
    
    # HRM-specific constraints
    max_theoretical_batch = 240  # Number of groups in city logistics
    
    # Test practical batch sizes
    batch_sizes = [16, 32, 48, 64, 96, 128]
    
    print(f"{'Batch':<8} {'Training':<10} {'Total':<10} {'GPU %':<8} {'Status':<12} {'Notes'}")
    print("-" * 65)
    
    for batch_size in batch_sizes:
        if batch_size > max_theoretical_batch:
            status = "❌ Too large"
            notes = f"Exceeds groups ({max_theoretical_batch})"
            print(f"{batch_size:<8} {'N/A':<10} {'N/A':<10} {'N/A':<8} {status:<12} {notes}")
            continue
            
        training_memory, breakdown = estimate_hrm_training_memory(batch_size, total_params)
        total_memory = param_memory + training_memory
        gpu_usage = (total_memory / gpu_memory) * 100 if gpu_memory > 0 else 0
        
        # Determine status
        if total_memory < gpu_memory * 0.75:  # 75% usage threshold
            status = "✅ Safe"
            notes = "Recommended"
        elif total_memory < gpu_memory * 0.90:  # 90% usage threshold
            status = "⚠️  Risky"
            notes = "Monitor closely"
        else:
            status = "❌ OOM"
            notes = "Will crash"
        
        print(f"{batch_size:<8} {training_memory:.2f} GB{'':<2} {total_memory:.2f} GB{'':<2} {gpu_usage:>5.1f}%{'':<1} {status:<12} {notes}")

def analyze_hrm_constraints():
    """Analyze HRM-specific training constraints"""
    print(f"\n🎯 HRM Training Constraints")
    print("="*40)
    
    print("📋 Dataset constraints:")
    print("  • City logistics: 960 examples, 240 groups")
    print("  • Max theoretical batch_size: 240 (group count)")
    print("  • Recommended batch_size: 32-64 (paper guidance)")
    print("  • Group sampling: Each epoch samples ~240 examples")
    
    print(f"\n🧠 HRM architecture constraints:")
    print("  • Hierarchical modules: 2 (H + L)")
    print("  • Deep supervision: Multiple forward passes")
    print("  • ACT mechanism: Variable computation steps")
    print("  • Carry states: Additional memory overhead")
    
    print(f"\n⚡ Memory optimization tips:")
    print("  • Use gradient checkpointing")
    print("  • Enable mixed precision (fp16)")
    print("  • Reduce ACT max_steps if needed")
    print("  • Monitor with: nvidia-smi")

def recommend_optimal_config(gpu_memory):
    """Recommend optimal configuration based on GPU memory"""
    print(f"\n🎯 Optimal Configuration Recommendations")
    print("="*50)
    
    if gpu_memory >= 12.0:  # RTX 3080 Ti, RTX 4070 Ti, etc.
        print("🚀 High-end GPU detected:")
        print("  global_batch_size: 64")
        print("  hidden_size: 256 (current)")
        print("  Enable mixed precision")
        
    elif gpu_memory >= 8.0:  # RTX 3070 Ti, RTX 4060 Ti, etc.
        print("⚖️  Mid-range GPU detected:")
        print("  global_batch_size: 32")
        print("  hidden_size: 256 (current)")
        print("  Enable gradient checkpointing")
        print("  Enable mixed precision")
        
    elif gpu_memory >= 6.0:  # RTX 3060, etc.
        print("💾 Budget GPU detected:")
        print("  global_batch_size: 16")
        print("  hidden_size: 128 (reduce from 256)")
        print("  Enable all memory optimizations")
        
    else:
        print("⚠️  Insufficient GPU memory:")
        print("  Consider CPU training or cloud GPU")
    
    print(f"\n📝 Recommended config/logistics_routing.yaml:")
    if gpu_memory >= 8.0:
        batch_size = 32
    elif gpu_memory >= 6.0:
        batch_size = 16
    else:
        batch_size = 8
        
    print(f"  global_batch_size: {batch_size}")
    print(f"  lr: 1e-4")
    print(f"  lr_warmup_steps: 1920  # 2x dataset size")
    print(f"  epochs: 100")
    print(f"  eval_interval: 10")

def main():
    print("🔧 HRM GPU Memory Analysis Tool")
    print("="*50)
    print("Analyzing memory requirements for City Logistics training...")
    
    # Check GPU
    gpu_available, gpu_memory = check_gpu_memory()
    
    # Estimate model parameters
    total_params, param_memory = estimate_hrm_parameters()
    
    if gpu_available:
        # Test batch sizes
        test_batch_sizes(total_params, param_memory, gpu_memory)
        
        # HRM-specific analysis
        analyze_hrm_constraints()
        
        # Recommendations
        recommend_optimal_config(gpu_memory)
    else:
        print("\n⚠️  GPU not available - showing parameter estimates only")
        analyze_hrm_constraints()
    
    print(f"\n🎯 Next Steps:")
    print("1. Use recommended batch size from analysis above")
    print("2. Start with conservative settings")
    print("3. Monitor GPU memory during first few batches")
    print("4. Adjust if you see OOM errors")

if __name__ == "__main__":
    main()