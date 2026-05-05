"""
Quick CUDA Test for V7P3R V3.0
==============================

Simple verification that your RTX 4070 Ti is working with PyTorch
"""

import torch
import time

def main():
    print("🚀 V7P3R V3.0 CUDA QUICK TEST")
    print("=" * 50)
    
    # Basic CUDA info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print()
        
        # Test GPU performance
        print("🔥 GPU PERFORMANCE TEST")
        print("-" * 30)
        
        device = torch.device("cuda")
        
        # Create test tensors
        size = (1000, 1000)
        a = torch.randn(size, device=device)
        b = torch.randn(size, device=device)
        
        # Warm up GPU
        for _ in range(10):
            c = torch.matmul(a, b)
        
        # Time GPU operations
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(100):
            c = torch.matmul(a, b)
        
        torch.cuda.synchronize()
        end = time.time()
        
        gpu_time = (end - start) / 100 * 1000  # ms per operation
        print(f"✅ GPU Matrix Multiplication: {gpu_time:.2f} ms")
        
        # Test CPU for comparison
        print("\n🖥️  CPU COMPARISON")
        print("-" * 20)
        
        a_cpu = a.cpu()
        b_cpu = b.cpu()
        
        start = time.time()
        for _ in range(10):  # Fewer iterations for CPU
            c_cpu = torch.matmul(a_cpu, b_cpu)
        end = time.time()
        
        cpu_time = (end - start) / 10 * 1000  # ms per operation
        print(f"⚡ CPU Matrix Multiplication: {cpu_time:.2f} ms")
        
        speedup = cpu_time / gpu_time
        print(f"🚀 GPU Speedup: {speedup:.1f}x faster!")
        
        # Memory test
        print(f"\n💾 GPU Memory Test")
        print("-" * 20)
        
        # Create larger tensor to test memory
        large_tensor = torch.randn(5000, 5000, device=device)
        memory_used = torch.cuda.memory_allocated(0) / 1e6  # MB
        
        print(f"✅ Large tensor created: {large_tensor.shape}")
        print(f"📊 GPU memory used: {memory_used:.1f} MB")
        
        # Clean up
        del large_tensor
        torch.cuda.empty_cache()
        
        print("\n🎉 YOUR RTX 4070 Ti IS READY FOR V3 TRAINING!")
        print("=" * 50)
        print("Recommendations for V3 training:")
        print("• Use batch sizes of 32-64 for optimal performance")
        print("• Your 12.9 GB VRAM can handle large models")
        print("• GPU acceleration will speed up training significantly")
        print("• Monitor GPU usage with: nvidia-smi")
        
    else:
        print("❌ CUDA not available")
        print("Please check NVIDIA drivers and CUDA installation")

if __name__ == "__main__":
    main()