#!/usr/bin/env python3
"""
Quick validation script to test training speed with optimized config
Run this before full training to estimate completion time
"""

import time
import torch
import os
from omegaconf import OmegaConf

# Add project root to path if needed
import sys
sys.path.append('.')

from pretrain import PretrainConfig, init_train_state, create_dataloader, train_batch

def quick_speed_test():
    """Test training speed with optimized config"""
    
    # Load optimized config
    config_dict = {
        'data_path': 'data/logistics-routing-1k',
        'global_batch_size': 256,
        'epochs': 50,
        'eval_interval': 10,
        'lr': 5e-4,
        'lr_min_ratio': 0.5,
        'lr_warmup_steps': 10,
        'beta1': 0.9,
        'beta2': 0.95,
        'weight_decay': 0.1,
        'puzzle_emb_weight_decay': 0.1,
        'puzzle_emb_lr': 5e-4,
        'seed': 0,
        'checkpoint_every_eval': False,
        'eval_save_outputs': ['inputs', 'labels', 'logits'],
        'project_name': 'Logistics-Minimal-Test',
        'run_name': None,
        'checkpoint_path': None,
        'arch': {
            'name': 'hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1',
            'loss': {
                'name': 'losses@ACTLossHead',
                'loss_type': 'stablemax_cross_entropy'
            },
            'halt_exploration_prob': 0.1,
            'halt_max_steps': 2,
            'H_cycles': 1,
            'L_cycles': 1,
            'H_layers': 1,
            'L_layers': 1,
            'hidden_size': 128,
            'num_heads': 2,
            'expansion': 2,
            'puzzle_emb_ndim': 128,
            'pos_encodings': 'rope',
            'rms_norm_eps': 1e-5,
            'rope_theta': 10000.0,
            'forward_dtype': 'bfloat16'
        }
    }
    
    config = PretrainConfig(**config_dict)
    
    print("🚀 Testing optimized HRM training speed...")
    print(f"Model: {config.arch.hidden_size}D, {config.arch.H_layers}+{config.arch.L_layers} layers")
    print(f"Cycles: H={config.arch.H_cycles}, L={config.arch.L_cycles}")
    print(f"Batch size: {config.global_batch_size}")
    
    # Check if data exists
    if not os.path.exists(config.data_path):
        print(f"❌ Dataset not found at {config.data_path}")
        print("Please generate the dataset first with the logistics game HTML")
        return
    
    try:
        # Create dataloader
        train_loader, train_metadata = create_dataloader(
            config, "train", 
            test_set_mode=False, 
            epochs_per_iter=1, 
            global_batch_size=config.global_batch_size, 
            rank=0, 
            world_size=1
        )
        
        # Initialize model
        train_state = init_train_state(config, train_metadata, world_size=1)
        
        print(f"✅ Model initialized: {sum(p.numel() for p in train_state.model.parameters())/1e6:.1f}M parameters")
        print(f"📊 Dataset: {train_metadata.total_groups} groups")
        
        # Test a few training steps
        test_steps = 5
        print(f"\n⏱️  Testing {test_steps} training steps...")
        
        start_time = time.time()
        step_times = []
        
        for i, (set_name, batch, global_batch_size) in enumerate(train_loader):
            if i >= test_steps:
                break
                
            step_start = time.time()
            
            # Move to GPU
            batch = {k: v.cuda() for k, v in batch.items()}
            
            # Initialize carry if needed
            if train_state.carry is None:
                with torch.device("cuda"):
                    train_state.carry = train_state.model.initial_carry(batch)
            
            # Train one step
            metrics = train_batch(config, train_state, batch, global_batch_size, rank=0, world_size=1)
            
            step_time = time.time() - step_start
            step_times.append(step_time)
            
            print(f"  Step {i+1}: {step_time:.2f}s")
            if metrics:
                for k, v in metrics.items():
                    if 'loss' in k or 'accuracy' in k:
                        print(f"    {k}: {v:.4f}")
        
        avg_step_time = sum(step_times) / len(step_times)
        total_steps = (config.epochs * train_metadata.total_groups) // config.global_batch_size
        estimated_hours = (total_steps * avg_step_time) / 3600
        
        print(f"\n📈 Performance Results:")
        print(f"  Average step time: {avg_step_time:.2f}s")
        print(f"  Total steps needed: {total_steps:,}")
        print(f"  Estimated total time: {estimated_hours:.1f} hours")
        
        if estimated_hours < 3:
            print("✅ Great! Training should complete in reasonable time")
        elif estimated_hours < 8:
            print("⚠️  Training will take several hours but is manageable")
        else:
            print("❌ Still too slow. Consider further optimizations:")
            print("   - Reduce epochs further (200-300)")
            print("   - Increase batch size to 256")
            print("   - Use even smaller model (hidden_size=128)")
        
        # Memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  GPU memory used: {memory_used:.1f} GB / 8 GB")
            
            if memory_used > 7:
                print("⚠️  High memory usage. Consider reducing batch size")
        
    except Exception as e:
        print(f"❌ Error during speed test: {e}")
        print("This might indicate a configuration issue")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_speed_test()
