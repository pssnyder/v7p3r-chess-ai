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
from v7p3r_gpu_genetic_trainer_clean import V7P3RGPUGeneticTrainer
from training_configs import (
    get_initial_exploration_config, get_development_config, 
    get_production_config, get_quick_test_config, 
    get_bounty_tuning_config, get_config_for_hardware,
    TRAINING_PHASES
)


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
    parser.add_argument('--preset', type=str, 
                        choices=['quick', 'initial', 'development', 'production', 'bounty-tuning'],
                        help='Use preset training configuration')
    parser.add_argument('--population', type=int, help='Population size')
    parser.add_argument('--generations', type=int, help='Number of generations')
    parser.add_argument('--workers', type=int, help='Number of parallel workers')
    parser.add_argument('--games', type=int, help='Games per individual')
    parser.add_argument('--auto-hardware', action='store_true',
                        help='Automatically configure based on hardware')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--monitor', action='store_true', 
                        help='Start training monitor in background')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU acceleration (requires PyTorch/CUDA)')
    
    args = parser.parse_args()
    
    # Select configuration
    if args.preset:
        if args.preset == 'quick':
            config = get_quick_test_config()
        elif args.preset == 'initial':
            config = get_initial_exploration_config()
        elif args.preset == 'development':
            config = get_development_config()
        elif args.preset == 'production':
            config = get_production_config()
        elif args.preset == 'bounty-tuning':
            config = get_bounty_tuning_config()
    elif args.auto_hardware:
        cpu_cores = mp.cpu_count()
        # Estimate RAM (simplified)
        ram_gb = 16  # Default assumption
        config = get_config_for_hardware(cpu_cores, ram_gb)
        print(f"Auto-detected {cpu_cores} CPU cores, using appropriate configuration")
    else:
        # Load from JSON config
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
    
    # Display configuration info
    print("=" * 60)
    print("V7P3R Chess AI 2.0 - Genetic Algorithm Training")
    print("=" * 60)
    
    if args.preset:
        phase_info = None
        for phase_name, phase_data in TRAINING_PHASES.items():
            if args.preset in phase_name:
                phase_info = phase_data
                break
        
        if phase_info:
            print(f"Training Phase: {phase_info['description']}")
            print(f"Expected Duration: {phase_info['duration']}")
            print(f"Goals: {', '.join(phase_info['goals'])}")
            print(f"Success Metrics: {', '.join(phase_info['success_metrics'])}")
            print("-" * 60)
    
    print(f"Population size: {config.population_size}")
    print(f"Generations: {config.generations}")
    print(f"Games per individual: {config.games_per_individual}")
    print(f"Parallel workers: {config.parallel_workers}")
    print(f"Max moves per game: {config.max_moves_per_game}")
    print(f"Extended bounty system: {config.extended_bounty}")
    print(f"Mutation rate: {config.mutation_rate}")
    print(f"Crossover rate: {config.crossover_rate}")
    print(f"Save frequency: every {config.save_frequency} generations")
    
    # Estimate runtime
    estimated_minutes = estimate_runtime(config)
    print(f"Estimated runtime: {estimated_minutes} minutes")
    print("=" * 60)
    
    # Start monitoring if requested
    monitor_process = None
    if args.monitor:
        import subprocess
        monitor_process = subprocess.Popen([
            'python', 'training_monitor.py', '--watch'
        ])
        print("Training monitor started in background")
    
    # Create trainer and start training
    use_gpu = args.gpu if hasattr(args, 'gpu') else False
    
    # Check for GPU availability
    try:
        import torch
        if torch.cuda.is_available() and use_gpu:
            print(f"GPU detected: {torch.cuda.get_device_name()}")
            print("Using GPU-accelerated genetic trainer")
            
            # Convert config to GPU trainer format
            gpu_config = {
                'population_size': config.population_size,
                'games_per_evaluation': config.games_per_individual,
                'max_parallel_evaluations': config.parallel_workers,
                'tournament_size': config.tournament_size,
                'mutation_rate': config.mutation_rate,
                'max_moves_per_game': config.max_moves_per_game,
                'model': {
                    'input_size': 816,
                    'hidden_size': 128,
                    'num_layers': 2,
                    'output_size': 64
                }
            }
            
            trainer = V7P3RGPUGeneticTrainer(gpu_config)
        else:
            if use_gpu:
                print("GPU requested but not available, falling back to CPU trainer")
            trainer = GeneticTrainer(config)
    except ImportError:
        if use_gpu:
            print("PyTorch not installed, falling back to CPU trainer")
        trainer = GeneticTrainer(config)
    
    if args.resume:
        print(f"Resuming training from: {args.resume}")
        # TODO: Implement resume functionality
    
    try:
        # Use the appropriate train method based on trainer type
        if isinstance(trainer, V7P3RGPUGeneticTrainer):
            trainer.train(config.generations)
        else:
            trainer.train()
        
        print("\nTraining completed successfully!")
        print(f"Best model saved to: models/genetic/v7p3r_2.0_final.json")
        
        # Generate final report
        print("\nGenerating training report...")
        from training_monitor import TrainingMonitor
        monitor = TrainingMonitor()
        monitor.generate_report()
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Progress has been saved")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Stop monitor if running
        if monitor_process:
            monitor_process.terminate()


def estimate_runtime(config: GeneticConfig) -> int:
    """Estimate training runtime in minutes"""
    # Rough estimates based on empirical testing
    time_per_game = 2  # seconds per game
    games_per_generation = config.population_size * config.games_per_individual
    time_per_generation = (games_per_generation * time_per_game) / config.parallel_workers
    total_time = time_per_generation * config.generations
    return int(total_time / 60)


if __name__ == "__main__":
    main()
