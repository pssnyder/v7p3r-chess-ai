#!/usr/bin/env python3
"""
Train V7P3R Chess AI 2.0 using Genetic Algorithm
Start the genetic training process with customizable parameters.
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

from v7p3r_genetic_trainer import GeneticTrainer, GeneticConfig


def load_config(config_path: str) -> GeneticConfig:
    """Load genetic algorithm configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        genetic_data = data.get('genetic_config', {})
        
        return GeneticConfig(
            population_size=genetic_data.get('population_size', 128),
            generations=genetic_data.get('generations', 1000),
            mutation_rate=genetic_data.get('mutation_rate', 0.15),
            mutation_strength=genetic_data.get('mutation_strength', 0.02),
            crossover_rate=genetic_data.get('crossover_rate', 0.7),
            elite_percentage=genetic_data.get('elite_percentage', 0.1),
            tournament_size=genetic_data.get('tournament_size', 5),
            games_per_individual=genetic_data.get('games_per_individual', 3),
            max_moves_per_game=genetic_data.get('max_moves_per_game', 200),
            parallel_workers=genetic_data.get('parallel_workers', min(8, mp.cpu_count())),
            save_frequency=genetic_data.get('save_frequency', 10),
            extended_bounty=genetic_data.get('extended_bounty', True)
        )
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Using default configuration")
        return GeneticConfig()


def main():
    parser = argparse.ArgumentParser(description="Train V7P3R Chess AI 2.0")
    parser.add_argument('--config', type=str, default='config_v2.json',
                        help='Path to configuration file')
    parser.add_argument('--population', type=int, help='Population size')
    parser.add_argument('--generations', type=int, help='Number of generations')
    parser.add_argument('--workers', type=int, help='Number of parallel workers')
    parser.add_argument('--games', type=int, help='Games per individual')
    parser.add_argument('--quick', action='store_true', 
                        help='Quick training mode (reduced parameters)')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.population:
        config.population_size = args.population
    if args.generations:
        config.generations = args.generations
    if args.workers:
        config.parallel_workers = args.workers
    if args.games:
        config.games_per_individual = args.games
    
    # Quick mode for testing
    if args.quick:
        config.population_size = 32
        config.generations = 20
        config.games_per_individual = 1
        config.max_moves_per_game = 100
        config.save_frequency = 5
        print("Quick training mode enabled")
    
    print("=" * 60)
    print("V7P3R Chess AI 2.0 - Genetic Algorithm Training")
    print("=" * 60)
    print(f"Population size: {config.population_size}")
    print(f"Generations: {config.generations}")
    print(f"Games per individual: {config.games_per_individual}")
    print(f"Parallel workers: {config.parallel_workers}")
    print(f"Max moves per game: {config.max_moves_per_game}")
    print(f"Extended bounty system: {config.extended_bounty}")
    print(f"Mutation rate: {config.mutation_rate}")
    print(f"Crossover rate: {config.crossover_rate}")
    print("=" * 60)
    
    # Create trainer and start training
    trainer = GeneticTrainer(config)
    
    if args.resume:
        print(f"Resuming training from: {args.resume}")
        # TODO: Implement resume functionality
    
    try:
        trainer.train()
        print("\nTraining completed successfully!")
        print(f"Best model saved to: models/genetic/v7p3r_2.0_final.json")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Progress has been saved")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
