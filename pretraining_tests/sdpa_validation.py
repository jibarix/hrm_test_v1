"""
Validation script to verify SDPA setup is working correctly
Run this before starting training to ensure everything is configured properly
"""

import torch
import torch.nn.functional as F

def validate_sdpa_setup():
    """Validate that SDPA is properly configured and working"""
    
    print("🔍 Validating SDPA Configuration...")
    
    # 1. Check PyTorch version
    print(f"✓ PyTorch version: {torch.__version__}")
    
    # 2. Check CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.version.cuda}")
        print(f"✓ Device: {torch.cuda.get_device_name()}")
    else:
        print("⚠️  CUDA not available - SDPA will run on CPU")
    
    # 3. Check SDPA support (PyTorch 2.5+ has good defaults, no manual config needed)
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        print("✓ SDPA support detected (using PyTorch defaults)")
    else:
        print("❌ SDPA not supported in this PyTorch version")
        return False
    
    # 4. Test SDPA functionality
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create test tensors
    batch_size, num_heads, seq_len, head_dim = 2, 8, 512, 64
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    
    # Test SDPA
    try:
        with torch.cuda.amp.autocast():
            output = F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=0.0)
        print(f"✓ SDPA test successful - output shape: {output.shape}")
        
        # Test causal attention
        output_causal = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        print(f"✓ Causal SDPA test successful - output shape: {output_causal.shape}")
        
    except Exception as e:
        print(f"❌ SDPA test failed: {e}")
        return False
    
    # 5. Performance comparison (optional)
    if torch.cuda.is_available():
        print("\n⚡ Running performance test...")
        
        # Warmup
        for _ in range(10):
            _ = F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=0.0)
        
        # Benchmark
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        for _ in range(100):
            _ = F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=0.0)
        end_time.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time)
        
        print(f"✓ Average SDPA time: {elapsed_time/100:.2f}ms")
    
    print("\n🎉 SDPA validation completed successfully!")
    print("🚀 You're ready to start training with native PyTorch attention!")
    return True

def check_dependencies():
    """Check that required dependencies are available"""
    print("\n📦 Checking Dependencies...")
    
    # Map package names to their import names
    required_packages = {
        'torch': 'torch',
        'adam_atan2': 'adam_atan2', 
        'einops': 'einops',
        'tqdm': 'tqdm',
        'pydantic': 'pydantic',
        'argdantic': 'argdantic',
        'wandb': 'wandb',
        'omegaconf': 'omegaconf',
        'hydra-core': 'hydra',  # Package name vs import name
        'huggingface_hub': 'huggingface_hub'
    }
    
    missing_packages = []
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name}")
        except ImportError:
            missing_packages.append(package_name)
            print(f"❌ {package_name}")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✓ All dependencies available")
    return True

if __name__ == "__main__":
    print("City Logistics & Routing HRM - SDPA Validation")
    print("=" * 50)
    
    deps_ok = check_dependencies()
    if not deps_ok:
        print("\n❌ Dependency check failed. Please install missing packages.")
        exit(1)
    
    sdpa_ok = validate_sdpa_setup()
    if not sdpa_ok:
        print("\n❌ SDPA validation failed. Check your PyTorch installation.")
        exit(1)
    
    print("\n" + "=" * 50)
    print("🎯 Ready to train! Run your training command:")
    print("python pretrain.py data_path=data/logistics-routing-1k ...")