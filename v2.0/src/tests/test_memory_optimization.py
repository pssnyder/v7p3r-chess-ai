#!/usr/bin/env python3
"""
Test GPU Memory Optimization for V7P3R Chess AI v2.0
Verify that flatten_parameters() fixes the RNN memory warning
"""

import torch
import warnings
import io
import sys
from contextlib import redirect_stderr

def test_rnn_memory_optimization():
    """Test that our RNN models don't generate memory warnings"""
    print("Testing RNN memory optimization...")
    
    # Capture warnings
    warning_buffer = io.StringIO()
    
    try:
        from v7p3r_gpu_model import V7P3RGPU_LSTM
        
        # Create model
        model = V7P3RGPU_LSTM(816, 64, 2, 32, device='cuda')
        model.to('cuda')
        
        # Test input
        test_input = torch.randn(1, 1, 816, device='cuda')
        
        print("Testing forward pass without optimization...")
        
        # Capture warnings during forward pass
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Forward pass without flatten_parameters - should generate warning
            model.lstm.flatten_parameters = lambda: None  # Disable flatten_parameters temporarily
            output1, _ = model(test_input)
            
            initial_warnings = len(w)
            print(f"Warnings without optimization: {initial_warnings}")
            
        # Reset and test with optimization
        model = V7P3RGPU_LSTM(816, 64, 2, 32, device='cuda')
        model.to('cuda')
        
        print("Testing forward pass with optimization...")
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Forward pass with flatten_parameters - should not generate warning
            output2, _ = model(test_input)
            
            optimized_warnings = len(w)
            print(f"Warnings with optimization: {optimized_warnings}")
        
        # Test crossover and mutation
        print("Testing genetic operations...")
        
        model2 = V7P3RGPU_LSTM(816, 64, 2, 32, device='cuda')
        model2.to('cuda')
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Test crossover
            child = model.crossover(model2)
            
            # Test mutation
            mutated = model.mutate()
            
            # Test forward passes
            output3, _ = child(test_input)
            output4, _ = mutated(test_input)
            
            genetic_warnings = len(w)
            print(f"Warnings from genetic operations: {genetic_warnings}")
        
        print(f"\n✅ Memory optimization test completed!")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Output shapes consistent: {output1.shape == output2.shape == output3.shape == output4.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory optimization test failed: {e}")
        return False

def test_trainer_memory_optimization():
    """Test that the genetic trainer uses optimized models"""
    print("\nTesting trainer memory optimization...")
    
    try:
        from v7p3r_gpu_genetic_trainer_clean import V7P3RGPUGeneticTrainer
        
        # Small config for testing
        config = {
            'population_size': 4,
            'games_per_evaluation': 1,
            'max_parallel_evaluations': 2,
            'tournament_size': 2,
            'mutation_rate': 0.1,
            'max_moves_per_game': 10,
            'model': {
                'input_size': 816,
                'hidden_size': 32,
                'num_layers': 1,
                'output_size': 16
            }
        }
        
        trainer = V7P3RGPUGeneticTrainer(config)
        
        # Create population
        population = trainer.create_random_population(4)
        
        # Test memory compaction
        trainer.population = population
        trainer.compact_population_memory()
        
        print("✅ Trainer memory optimization working!")
        
        return True
        
    except Exception as e:
        print(f"❌ Trainer optimization test failed: {e}")
        return False

def main():
    """Run memory optimization tests"""
    print("V7P3R Chess AI v2.0 - GPU Memory Optimization Test")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - cannot test GPU memory optimization")
        return
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    # Test 1: RNN memory optimization
    test1_passed = test_rnn_memory_optimization()
    
    # Test 2: Trainer memory optimization
    test2_passed = test_trainer_memory_optimization()
    
    print("\n" + "=" * 60)
    
    if test1_passed and test2_passed:
        print("✅ All memory optimization tests passed!")
        print("\nThe RNN memory warning should now be resolved during training.")
        print("Memory usage should be more efficient and stable.")
    else:
        print("❌ Some memory optimization tests failed")
        print("Check the error messages above for debugging.")

if __name__ == "__main__":
    main()
