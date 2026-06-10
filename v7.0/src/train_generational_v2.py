"""
V7.1 Generational Training - WITH PRETRAINED BASELINE

CHANGE: Generation 0 is now pre-trained (not random initialization)

This solves the "all draws" problem by ensuring the baseline
can actually play chess before we try to improve on it.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from generational_trainer import GenerationalTrainer


def main():
    print("=" * 80)
    print("V7P3R v7.1 - GENERATIONAL TRAINING WITH PRETRAINED BASELINE")
    print("=" * 80)
    print()
    print("CHANGE: Generation 0 will be pre-trained with 500 self-play games")
    print("This creates a competent baseline before generational training")
    print()
    print("=" * 80)
    print()
    
    # Configuration
    PROFILE_PATH = "../profiles/dark_forest_assassin.json"
    STOCKFISH_PATH = "../../stockfish.exe"
    OUTPUT_DIR = "../training/v7_generational_v2"
    
    # Tablebase path
    TABLEBASE_PATH = r"E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\pgn_training_data\pgn_data_endgames\3-4-5_pieces_Syzygy\3-4-5"
    
    # Training parameters
    BASELINE_GAMES = 500        # Pre-train Generation 0
    SELFPLAY_GAMES = 100        # Games per generation for training
    EVALUATION_GAMES = 6        # Games for evaluation (3 White, 3 Black)
    MAX_GENERATIONS = 10        # Number of generations to train
    
    # Initialize trainer
    print("Initializing generational trainer...")
    print("-" * 80)
    
    trainer = GenerationalTrainer(
        profile_path=PROFILE_PATH,
        stockfish_path=STOCKFISH_PATH,
        output_dir=OUTPUT_DIR,
        opening_book_pgn=None,
        tablebase_path=TABLEBASE_PATH
    )
    
    print()
    print("=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"BASELINE (Gen 0) games: {BASELINE_GAMES} ⭐ NEW")
    print(f"Self-play games per generation: {SELFPLAY_GAMES}")
    print(f"Evaluation match format: {EVALUATION_GAMES} games")
    print(f"Maximum generations: {MAX_GENERATIONS}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print()
    print("=" * 80)
    print()
    
    input("Press ENTER to start training (or Ctrl+C to cancel)...")
    print()
    
    # PRE-TRAIN GENERATION 0
    print("=" * 80)
    print("PHASE 1: PRE-TRAINING GENERATION 0 BASELINE")
    print("=" * 80)
    print(f"Training with {BASELINE_GAMES} self-play games to create competent baseline")
    print()
    
    trainer.initialize_first_generation()
    
    # Train baseline
    trainer.train_new_generation(
        selfplay_games=BASELINE_GAMES,
        batch_size=256,
        train_every_n_games=50
    )
    
    # Accept baseline as Gen 0
    print()
    print("[OK] Baseline training complete - accepting as Generation 0")
    
    # Save as best
    import torch
    best_path = Path(OUTPUT_DIR) / "best_model.pt"
    torch.save(trainer.new_model.state_dict(), best_path)
    trainer.best_model = trainer.new_model
    trainer.best_trainer = trainer.new_trainer
    print(f"[OK] Baseline saved as best model")
    
    print()
    print("=" * 80)
    print("PHASE 2: GENERATIONAL TRAINING")
    print("=" * 80)
    print()
    
    # Run generational training from this baseline
    for gen in range(MAX_GENERATIONS):
        trainer.train_new_generation(
            selfplay_games=SELFPLAY_GAMES,
            batch_size=256,
            train_every_n_games=10
        )
        
        result = trainer.evaluate_generation(num_games=EVALUATION_GAMES)
        
        if result.accepted:
            trainer.accept_generation(result)
        else:
            trainer.reject_generation(result)
        
        print()
    
    print("=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Final generation: {trainer.current_generation}")
    print(f"Total trained: {len(trainer.generation_history)}")
    print(f"Accepted: {sum(1 for g in trainer.generation_history if g['accepted'])}")
    print(f"Rejected: {sum(1 for g in trainer.generation_history if not g['accepted'])}")
    print("=" * 80)


if __name__ == "__main__":
    main()
