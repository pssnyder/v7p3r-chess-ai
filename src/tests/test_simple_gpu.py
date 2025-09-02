#!/usr/bin/env python3
"""
Simple GPU Training Test for V7P3R Chess AI v2.0
Quick test of GPU-accelerated training
"""

import time
import json

def test_simple_gpu_training():
    """Test GPU training with minimal configuration"""
    print("Testing GPU-accelerated training...")
    
    try:
        from v7p3r_gpu_genetic_trainer import V7P3RGPUGeneticTrainer
        
        # Minimal test configuration
        config = {
            'population_size': 4,
            'games_per_evaluation': 1,
            'max_parallel_evaluations': 2,
            'tournament_size': 2,
            'mutation_rate': 0.1,
            'max_moves_per_game': 20,  # Very short games for testing
            'model': {
                'input_size': 816,
                'hidden_size': 32,  # Small for testing
                'num_layers': 1,
                'output_size': 16
            },
            'bounty_config': {}
        }
        
        print("Creating GPU trainer...")
        trainer = V7P3RGPUGeneticTrainer(config)
        
        print(f"Using device: {trainer.device}")
        
        # Test one generation
        print("Running 1 generation test...")
        start_time = time.time()
        
        report = trainer.train(1)
        
        training_time = time.time() - start_time
        
        print(f"\nTest completed!")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Best fitness: {report['training_summary']['best_fitness_achieved']:.2f}")
        print(f"Device used: {report['training_summary']['device_used']}")
        
        return True
        
    except Exception as e:
        print(f"GPU training test failed: {e}")
        return False

def main():
    """Run simple GPU training test"""
    print("V7P3R Chess AI v2.0 - Simple GPU Training Test")
    print("=" * 50)
    
    success = test_simple_gpu_training()
    
    print("\n" + "=" * 50)
    
    if success:
        print("✅ GPU training is working!")
        print("\nReady for full training. Use:")
        print("python train_v7p3r_v2.py --preset quick --gpu")
    else:
        print("❌ GPU training test failed")
        print("Check error messages above for debugging")

if __name__ == "__main__":
    main()
