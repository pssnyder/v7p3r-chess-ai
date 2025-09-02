#!/usr/bin/env python3
"""
Test GPU acceleration for V7P3R Chess AI v2.0
Quick test to verify GPU setup and performance
"""

import time
import sys
import os

def test_gpu_availability():
    """Test if GPU is available for training"""
    print("=== GPU Availability Test ===")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"CUDA available: Yes")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
                
            return True
        else:
            print("CUDA available: No")
            return False
            
    except ImportError:
        print("PyTorch not installed")
        return False

def test_gpu_model():
    """Test GPU model creation and basic operations"""
    print("\n=== GPU Model Test ===")
    
    try:
        from v7p3r_gpu_model import V7P3RGPU_LSTM, GPUChessFeatureExtractor
        import torch
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Create model
        model = V7P3RGPU_LSTM(
            input_size=816,
            hidden_size=128,
            num_layers=2,
            output_size=64,
            device=device
        )
        
        print(f"Model created successfully")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        batch_size = 4
        seq_length = 1
        test_input = torch.randn(batch_size, seq_length, 816, device=device)
        
        start_time = time.time()
        with torch.no_grad():
            output, hidden = model(test_input)
        inference_time = time.time() - start_time
        
        print(f"Forward pass successful")
        print(f"Output shape: {output.shape}")
        print(f"Inference time: {inference_time*1000:.2f}ms")
        
        # Test feature extractor
        feature_extractor = GPUChessFeatureExtractor()
        feature_extractor.to(device)
        
        import chess
        board = chess.Board()
        features = feature_extractor.extract_features(board)
        print(f"Feature extraction successful: {len(features)} features")
        
        return True
        
    except Exception as e:
        print(f"GPU model test failed: {e}")
        return False

def test_genetic_trainer():
    """Test GPU genetic trainer initialization"""
    print("\n=== GPU Genetic Trainer Test ===")
    
    try:
        from v7p3r_gpu_genetic_trainer import V7P3RGPUGeneticTrainer
        
        config = {
            'population_size': 4,  # Small for testing
            'games_per_evaluation': 1,
            'max_parallel_evaluations': 2,
            'tournament_size': 2,
            'mutation_rate': 0.1,
            'max_moves_per_game': 50,
            'model': {
                'input_size': 816,
                'hidden_size': 64,  # Smaller for testing
                'num_layers': 1,
                'output_size': 32
            },
            'bounty_config': {
                'center_control_weight': 1.0,
                'piece_development_weight': 1.0,
                'king_safety_weight': 1.0,
                'tactical_weight': 1.0
            }
        }
        
        trainer = V7P3RGPUGeneticTrainer(config)
        print(f"Trainer initialized successfully")
        print(f"Device: {trainer.device}")
        
        # Create small population
        population = trainer.create_random_population(2)
        print(f"Population created: {len(population)} individuals")
        
        # Test evaluation (very quick)
        print("Testing individual evaluation...")
        start_time = time.time()
        fitness = trainer.evaluate_individual_gpu(population[0], num_games=1)
        eval_time = time.time() - start_time
        
        print(f"Evaluation completed: fitness={fitness:.2f}, time={eval_time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"Genetic trainer test failed: {e}")
        return False

def run_performance_comparison():
    """Compare CPU vs GPU performance"""
    print("\n=== Performance Comparison ===")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("GPU not available for comparison")
            return
        
        from v7p3r_gpu_model import V7P3RGPU_LSTM
        
        # CPU test
        print("Testing CPU performance...")
        cpu_model = V7P3RGPU_LSTM(816, 128, 2, 64, device='cpu')
        test_input = torch.randn(8, 1, 816)
        
        start_time = time.time()
        for _ in range(10):
            with torch.no_grad():
                output, _ = cpu_model(test_input)
        cpu_time = time.time() - start_time
        
        # GPU test
        print("Testing GPU performance...")
        gpu_model = V7P3RGPU_LSTM(816, 128, 2, 64, device='cuda')
        test_input_gpu = test_input.cuda()
        
        # Warmup
        with torch.no_grad():
            gpu_model(test_input_gpu)
        
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(10):
            with torch.no_grad():
                output, _ = gpu_model(test_input_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        print(f"CPU time: {cpu_time:.3f}s")
        print(f"GPU time: {gpu_time:.3f}s")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
        
    except Exception as e:
        print(f"Performance comparison failed: {e}")

def main():
    """Run all GPU tests"""
    print("V7P3R Chess AI v2.0 - GPU Acceleration Test")
    print("=" * 50)
    
    # Test 1: GPU availability
    gpu_available = test_gpu_availability()
    
    # Test 2: GPU model
    if gpu_available:
        model_works = test_gpu_model()
        
        # Test 3: Genetic trainer
        if model_works:
            trainer_works = test_genetic_trainer()
            
            # Test 4: Performance comparison
            if trainer_works:
                run_performance_comparison()
    
    print("\n" + "=" * 50)
    
    if gpu_available:
        print("✅ GPU acceleration is ready!")
        print("Use '--gpu' flag in training script to enable GPU acceleration")
        print("\nExample: python train_v7p3r_v2.py --preset quick --gpu")
    else:
        print("❌ GPU acceleration not available")
        print("Install PyTorch with CUDA support for GPU acceleration")
        print("Visit: https://pytorch.org/get-started/locally/")

if __name__ == "__main__":
    main()
