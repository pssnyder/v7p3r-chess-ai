#!/usr/bin/env python3
"""
Extended Training Session for V7P3R Chess AI v2.0
Optimized for 2-3 hour intensive training with tournament preparation
"""

import time
import os
from datetime import datetime, timedelta
from incremental_trainer import V7P3RIncrementalTrainer

def create_tournament_training_config():
    """Create optimized config for tournament-ready AI"""
    return {
        'population_size': 32,           # Larger population for better diversity
        'games_per_evaluation': 4,       # More games per evaluation for stability
        'max_parallel_evaluations': 8,   # Use more CPU cores
        'tournament_size': 5,            # Competitive selection
        'mutation_rate': 0.12,           # Moderate mutation for exploration
        'max_moves_per_game': 150,       # Allow complex games
        'model': {
            'input_size': 816,
            'hidden_size': 256,          # Larger network for tournament play
            'num_layers': 3,             # Deeper network
            'output_size': 128           # More output neurons
        }
    }

def estimate_training_time(config, generations):
    """Estimate training time based on configuration"""
    # Rough estimates based on previous runs
    time_per_individual = 2.5  # seconds per individual per game
    time_per_generation = (
        config['population_size'] * 
        config['games_per_evaluation'] * 
        time_per_individual / 
        config['max_parallel_evaluations']
    )
    
    total_time = generations * time_per_generation
    return total_time

def run_extended_training_session(target_hours=2.5):
    """Run extended training session for tournament preparation"""
    
    print("=" * 80)
    print("V7P3R Chess AI v2.0 - Extended Tournament Training Session")
    print("=" * 80)
    
    config = create_tournament_training_config()
    
    # Calculate optimal number of generations for target time
    time_per_gen = estimate_training_time(config, 1)
    target_seconds = target_hours * 3600
    estimated_generations = int(target_seconds / time_per_gen)
    
    print(f"Training Configuration:")
    print(f"  Population Size: {config['population_size']}")
    print(f"  Games per Individual: {config['games_per_evaluation']}")
    print(f"  Parallel Evaluations: {config['max_parallel_evaluations']}")
    print(f"  Model Size: {config['model']['hidden_size']} hidden, {config['model']['num_layers']} layers")
    print(f"  Estimated Time per Generation: {time_per_gen:.1f} seconds")
    print(f"  Target Training Time: {target_hours:.1f} hours")
    print(f"  Estimated Generations: {estimated_generations}")
    print("=" * 80)
    
    # Create incremental trainer
    trainer = V7P3RIncrementalTrainer(config, load_previous_best=True)
    
    # Tournament-focused bounty configuration
    tournament_bounties = {
        'tactical_weight': 5.0,           # Strong tactical play
        'piece_development_weight': 3.0,  # Good opening principles
        'center_control_weight': 2.5,     # Solid positional understanding
        'king_safety_weight': 4.0,        # Tournament safety
        'attack_defense_weight': 3.5,     # Balanced aggressive play
        'mate_threats_weight': 6.0,       # Finish games decisively
        'piece_coordination_weight': 3.0   # Advanced coordination
    }
    
    training_modifications = {
        'max_moves_per_game': 150,        # Allow complex tournament games
        'mutation_rate': 0.12,            # Exploration vs exploitation balance
        'tournament_size': 5              # Competitive selection pressure
    }
    
    print("Starting extended training session...")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        report = trainer.train_with_modified_fitness(
            num_generations=estimated_generations,
            bounty_modifications=tournament_bounties,
            fitness_modifications=training_modifications
        )
        
        training_time = time.time() - start_time
        
        print("=" * 80)
        print("TRAINING SESSION COMPLETED!")
        print(f"Actual Training Time: {training_time/3600:.2f} hours")
        print(f"Best Fitness Achieved: {report['training_summary']['best_fitness_achieved']:.2f}")
        print(f"Total Generations: {report['training_summary']['total_generations']}")
        print("=" * 80)
        
        # Save detailed tournament report
        tournament_report = {
            'session_info': {
                'session_type': 'tournament_preparation',
                'target_hours': target_hours,
                'actual_hours': training_time / 3600,
                'config_used': config,
                'bounty_config': tournament_bounties,
                'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'session_purpose': 'Tournament engine preparation'
            },
            'training_results': report
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tournament_report_path = f"reports/tournament_training_session_{timestamp}.json"
        
        import json
        with open(tournament_report_path, 'w') as f:
            json.dump(tournament_report, f, indent=2)
        
        print(f"Tournament training report saved: {tournament_report_path}")
        
        return report, tournament_report_path
        
    except KeyboardInterrupt:
        training_time = time.time() - start_time
        print(f"\nTraining interrupted after {training_time/3600:.2f} hours")
        print("Progress has been saved and can be resumed")
        return None, None

def main():
    """Main training session"""
    print("Preparing for extended tournament training session...")
    
    # Ask user for confirmation
    response = input("This will run a 2-3 hour intensive training session. Continue? (y/N): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return
    
    # Run the training session
    report, report_path = run_extended_training_session(target_hours=2.5)
    
    if report:
        print("\n" + "=" * 80)
        print("TRAINING SESSION SUMMARY")
        print("=" * 80)
        print(f"✅ Best fitness: {report['training_summary']['best_fitness_achieved']:.2f}")
        print(f"✅ Training time: {report['training_summary']['total_training_time']/3600:.2f} hours")
        print(f"✅ Generations completed: {report['training_summary']['total_generations']}")
        print(f"✅ Device used: {report['training_summary']['device_used']}")
        print("=" * 80)
        
        print("\nNext steps:")
        print("1. Run: python package_tournament_engine.py")
        print("2. Test the packaged engine")
        print("3. Enter into Arena tournaments")
        
        # Check if we should proceed to packaging
        package_now = input("\nProceed to package the tournament engine now? (y/N): ")
        if package_now.lower() == 'y':
            print("Proceeding to engine packaging...")
            os.system("python package_tournament_engine.py")

if __name__ == "__main__":
    main()
