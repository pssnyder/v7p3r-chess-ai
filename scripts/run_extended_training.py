#!/usr/bin/env python3
"""
Extended Training Script for V7P3R Chess AI v2.0
Continues training from the best GPU model found in the models directory.
"""

import os
import sys
import argparse

# Add src directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '..', 'src')
sys.path.insert(0, src_dir)

# Add individual module directories to path
sys.path.insert(0, os.path.join(src_dir, 'core'))
sys.path.insert(0, os.path.join(src_dir, 'models'))
sys.path.insert(0, os.path.join(src_dir, 'evaluation'))
sys.path.insert(0, os.path.join(src_dir, 'training'))

from incremental_trainer import V7P3RIncrementalTrainer

def main():
    parser = argparse.ArgumentParser(description="Extended Training for V7P3R Chess AI v2.0")
    parser.add_argument('--generations', type=int, default=50, help='Number of generations to train')
    parser.add_argument('--population', type=int, default=16, help='Population size')
    parser.add_argument('--games', type=int, default=3, help='Games per individual')
    parser.add_argument('--workers', type=int, default=4, help='Parallel workers')
    
    args = parser.parse_args()
    
    # Configuration for extended training
    config = {
        'population_size': args.population,
        'games_per_evaluation': args.games,
        'max_parallel_evaluations': args.workers,
        'tournament_size': 3,
        'mutation_rate': 0.1,
        'max_moves_per_game': 120,
        'model': {
            'input_size': 816,
            'hidden_size': 128,
            'num_layers': 2,
            'output_size': 64
        }
    }
    
    print("=" * 60)
    print("V7P3R Chess AI v2.0 - Extended Training Session")
    print("=" * 60)
    print(f"Generations: {args.generations}")
    print(f"Population size: {args.population}")
    print(f"Games per individual: {args.games}")
    print(f"Parallel workers: {args.workers}")
    print(f"Continuing from best GPU model found in models/")
    print("=" * 60)
    
    # Create incremental trainer (will automatically find and load best model)
    trainer = V7P3RIncrementalTrainer(config, load_previous_best=True)
    
    try:
        # Start training
        print("Starting extended training session...")
        report = trainer.train(args.generations)
        
        print("\nTraining completed successfully!")
        print(f"Best fitness achieved: {report.get('best_fitness_achieved', 'N/A')}")
        
        # Save detailed report
        trainer.save_incremental_report(report)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Progress has been saved")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
