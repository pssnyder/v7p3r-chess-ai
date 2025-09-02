#!/usr/bin/env python3
"""
Demo: Incremental Training with Bounty Modifications
Shows how to continue training while modifying the fitness function
"""

from incremental_trainer import V7P3RIncrementalTrainer

def demo_tactical_focus_training():
    """Demo: Emphasize tactical play over positional play"""
    
    config = {
        'population_size': 16,
        'games_per_evaluation': 2,
        'max_parallel_evaluations': 4,
        'tournament_size': 3,
        'mutation_rate': 0.1,
        'max_moves_per_game': 80,
        'model': {
            'input_size': 816,
            'hidden_size': 128,
            'num_layers': 2,
            'output_size': 64
        }
    }
    
    print("=== Demo: Tactical Focus Training ===")
    print("This will load your previous best model and train it to be more tactical")
    
    trainer = V7P3RIncrementalTrainer(config, load_previous_best=True)
    
    # Modify bounties to emphasize tactics
    bounty_modifications = {
        'tactical_weight': 6.0,           # Much higher tactical emphasis
        'piece_development_weight': 2.0,  # Moderate piece development
        'center_control_weight': 1.0,     # Lower center control
        'king_safety_weight': 3.0         # Maintain king safety
    }
    
    fitness_modifications = {
        'mutation_rate': 0.08,  # Lower mutation for fine-tuning
    }
    
    report = trainer.train_with_modified_fitness(
        num_generations=10,
        bounty_modifications=bounty_modifications,
        fitness_modifications=fitness_modifications
    )
    
    trainer.save_incremental_report(report, bounty_modifications, fitness_modifications)
    
    print(f"✅ Tactical training completed! Best fitness: {report['training_summary']['best_fitness_achieved']:.2f}")
    return report

def demo_positional_focus_training():
    """Demo: Emphasize positional play over tactics"""
    
    config = {
        'population_size': 16,
        'games_per_evaluation': 2,
        'max_parallel_evaluations': 4,
        'tournament_size': 3,
        'mutation_rate': 0.1,
        'max_moves_per_game': 120,  # Longer games for positional play
        'model': {
            'input_size': 816,
            'hidden_size': 128,
            'num_layers': 2,
            'output_size': 64
        }
    }
    
    print("\n=== Demo: Positional Focus Training ===")
    print("This will train for positional understanding and long-term planning")
    
    trainer = V7P3RIncrementalTrainer(config, load_previous_best=True)
    
    # Modify bounties to emphasize positional play
    bounty_modifications = {
        'tactical_weight': 2.0,           # Lower tactical emphasis
        'piece_development_weight': 4.0,  # High piece development
        'center_control_weight': 3.5,     # High center control
        'king_safety_weight': 4.0,        # High king safety
        'positional_weight': 3.0          # If available
    }
    
    fitness_modifications = {
        'max_moves_per_game': 120,  # Allow longer positional games
        'mutation_rate': 0.06,      # Very fine tuning
    }
    
    report = trainer.train_with_modified_fitness(
        num_generations=15,
        bounty_modifications=bounty_modifications,
        fitness_modifications=fitness_modifications
    )
    
    trainer.save_incremental_report(report, bounty_modifications, fitness_modifications)
    
    print(f"✅ Positional training completed! Best fitness: {report['training_summary']['best_fitness_achieved']:.2f}")
    return report

def demo_aggressive_play_training():
    """Demo: Train for aggressive, attacking play"""
    
    config = {
        'population_size': 20,
        'games_per_evaluation': 3,
        'max_parallel_evaluations': 4,
        'tournament_size': 4,
        'mutation_rate': 0.12,
        'max_moves_per_game': 100,
        'model': {
            'input_size': 816,
            'hidden_size': 128,
            'num_layers': 2,
            'output_size': 64
        }
    }
    
    print("\n=== Demo: Aggressive Play Training ===")
    print("This will train for aggressive, attacking chess")
    
    trainer = V7P3RIncrementalTrainer(config, load_previous_best=True)
    
    # Modify bounties for aggressive play
    bounty_modifications = {
        'tactical_weight': 7.0,           # Very high tactics
        'attack_defense_weight': 5.0,     # High attack emphasis
        'piece_development_weight': 3.0,  # Quick development
        'center_control_weight': 2.5,     # Moderate center control
        'king_safety_weight': 2.0,        # Lower king safety (more risk)
        'mate_threats_weight': 8.0        # Prioritize mate threats
    }
    
    fitness_modifications = {
        'mutation_rate': 0.12,    # Higher mutation for exploration
        'tournament_size': 4      # More competitive selection
    }
    
    report = trainer.train_with_modified_fitness(
        num_generations=12,
        bounty_modifications=bounty_modifications,
        fitness_modifications=fitness_modifications
    )
    
    trainer.save_incremental_report(report, bounty_modifications, fitness_modifications)
    
    print(f"✅ Aggressive training completed! Best fitness: {report['training_summary']['best_fitness_achieved']:.2f}")
    return report

def main():
    """Run all training demos"""
    
    print("V7P3R Chess AI v2.0 - Incremental Training Demos")
    print("=" * 60)
    print("These demos will build upon your existing trained models")
    print("and specialize them for different playing styles.")
    print("=" * 60)
    
    demos = [
        ("Tactical Focus", demo_tactical_focus_training),
        ("Positional Focus", demo_positional_focus_training), 
        ("Aggressive Play", demo_aggressive_play_training)
    ]
    
    print("Available demos:")
    for i, (name, _) in enumerate(demos):
        print(f"{i+1}. {name}")
    
    choice = input("\nSelect demo (1-3) or 'all' to run all: ").strip().lower()
    
    if choice == 'all':
        for name, demo_func in demos:
            print(f"\n{'='*20} {name} {'='*20}")
            demo_func()
    elif choice in ['1', '2', '3']:
        idx = int(choice) - 1
        name, demo_func = demos[idx]
        print(f"\n{'='*20} {name} {'='*20}")
        demo_func()
    else:
        print("Invalid choice. Running tactical focus demo...")
        demo_tactical_focus_training()
    
    print("\n" + "="*60)
    print("All training demos completed!")
    print("Check the 'models/' directory for your new specialized models.")
    print("Check 'reports/' for detailed training reports.")

if __name__ == "__main__":
    main()
