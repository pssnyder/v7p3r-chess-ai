#!/usr/bin/env python3
"""
Debug script to test fitness evaluation diversity
"""

import os
import sys
import torch
import chess

# Add src directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '..', 'src')
sys.path.insert(0, src_dir)
sys.path.insert(0, os.path.join(src_dir, 'core'))
sys.path.insert(0, os.path.join(src_dir, 'models'))
sys.path.insert(0, os.path.join(src_dir, 'evaluation'))
sys.path.insert(0, os.path.join(src_dir, 'training'))

from v7p3r_gpu_genetic_trainer_clean import V7P3RGPUGeneticTrainer

def test_fitness_evaluation():
    """Test if fitness evaluation produces diverse results"""
    
    # Configuration
    config = {
        'population_size': 4,
        'games_per_evaluation': 2,
        'max_parallel_evaluations': 2,
        'tournament_size': 3,
        'mutation_rate': 0.1,
        'max_moves_per_game': 50,  # Shorter games for testing
        'model': {
            'input_size': 816,
            'hidden_size': 128,
            'num_layers': 2,
            'output_size': 64
        }
    }
    
    print("Creating trainer...")
    trainer = V7P3RGPUGeneticTrainer(config)
    
    print("Creating random population...")
    population = trainer.create_random_population(4)
    
    print("Evaluating population...")
    fitness_scores = trainer.evaluate_population_parallel(population, num_games_per_individual=2)
    
    print(f"\nFitness scores:")
    for i, score in enumerate(fitness_scores):
        print(f"Individual {i}: {score:.2f}")
    
    # Check for diversity
    unique_scores = set(fitness_scores)
    print(f"\nUnique scores: {len(unique_scores)}")
    print(f"All scores identical: {len(unique_scores) == 1}")
    
    if len(unique_scores) == 1:
        print("❌ All fitness scores are identical - this is the bug!")
        print(f"Score value: {fitness_scores[0]}")
    else:
        print("✅ Fitness scores show diversity")
        
    # Test individual game evaluation
    print(f"\nTesting individual game evaluation...")
    test_model = population[0]
    
    game_scores = []
    for i in range(5):
        score = trainer.play_evaluation_game_gpu(test_model)
        game_scores.append(score)
        print(f"Game {i+1}: {score:.2f}")
    
    unique_game_scores = set(game_scores)
    print(f"\nUnique game scores: {len(unique_game_scores)}")
    if len(unique_game_scores) == 1:
        print("❌ All game scores are identical - deterministic evaluation!")
    else:
        print("✅ Game scores show diversity")

if __name__ == "__main__":
    test_fitness_evaluation()
