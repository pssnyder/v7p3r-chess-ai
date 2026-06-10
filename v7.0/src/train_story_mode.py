"""
V7P3R v7.0 - "Chess as Story" Training System

Revolutionary training philosophy:
- Opening (moves 1-10): Learn fundamentals (90% Stockfish weight)
- Middlegame Chaos (moves 21-40): Maximum personality (10% Stockfish weight)
- Endgame Precision (moves 61+): Stockfish + Tablebases (50-100% weight)

This script runs self-play training with:
1. Opening book forcing (fast-forward 8-10 moves to interesting positions)
2. Phase-aware dynamic weighting (sinusoidal transitions)
3. Tablebase oracle for perfect endgame knowledge

User's Vision:
"get the engine finding 'decisive' play, not necessarily 'good' play"
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from selfplay_trainer import SelfPlayTrainer


def main():
    print("=" * 80)
    print("V7P3R v7.0 - 'CHESS AS STORY' TRAINING SYSTEM")
    print("=" * 80)
    print()
    print("Training Philosophy:")
    print("  Opening (1-10):        90% Stockfish → Learn fundamentals")
    print("  Early Mid (11-20):     80% → 40% Stockfish → Building complexity")
    print("  Deep Mid (21-40):      30% → 10% Stockfish → MAXIMUM CHAOS")
    print("  Late Mid (41-60):      20% → 50% Stockfish → Refining tactics")
    print("  Endgame (61+):         50% Stockfish → Precision conversion")
    print("  Tablebase (≤7 pieces): 100% Perfect → Math solves endgames")
    print()
    print("=" * 80)
    print()
    
    # Configuration
    PROFILE_PATH = "../profiles/dark_forest_assassin.json"
    STOCKFISH_PATH = "../../stockfish.exe"
    OUTPUT_DIR = "../training/v7_story_training"
    
    # Optional: Opening book and tablebases
    OPENING_BOOK_PGN = None  # Use default aggressive openings
    TABLEBASE_PATH = r"E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\pgn_training_data\pgn_data_endgames\3-4-5_pieces_Syzygy\3-4-5"
    
    # Training parameters
    NUM_GAMES = 100
    BATCH_SIZE = 256
    TRAIN_EVERY_N_GAMES = 10
    
    # Initialize trainer
    print("Initializing trainer...")
    print("-" * 80)
    
    trainer = SelfPlayTrainer(
        profile_path=PROFILE_PATH,
        stockfish_path=STOCKFISH_PATH,
        output_dir=OUTPUT_DIR,
        opening_book_pgn=OPENING_BOOK_PGN,
        tablebase_path=TABLEBASE_PATH,
        use_opening_book=True,   # Enable opening forcing
        use_tablebases=True      # Enable tablebase oracle
    )
    
    print()
    print("=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Games: {NUM_GAMES}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Train Every: {TRAIN_EVERY_N_GAMES} games")
    print(f"Output Directory: {OUTPUT_DIR}")
    print()
    print(f"Opening Book: {'Enabled (12 aggressive lines)' if trainer.use_opening_book else 'Disabled'}")
    print(f"Tablebases: {'Enabled' if trainer.use_tablebases else 'Disabled'}")
    print()
    print("=" * 80)
    print()
    
    # Run training
    input("Press ENTER to start training (or Ctrl+C to cancel)...")
    print()
    
    trainer.train_from_selfplay(
        num_games=NUM_GAMES,
        batch_size=BATCH_SIZE,
        train_every_n_games=TRAIN_EVERY_N_GAMES
    )
    
    print()
    print("=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print()
    print(f"Results saved to: {OUTPUT_DIR}")
    print()
    print("Expected Improvements vs Baseline (v7_selfplay):")
    print("  ✓ More decisive middlegame play (personality maximized)")
    print("  ✓ Better endgame conversion (tablebase integration)")
    print("  ✓ Faster training (opening forcing skips repetitive moves)")
    print("  ✓ Reduced max-move games (78% → ~20% expected)")
    print()
    print("Next Steps:")
    print("  1. Review training_report.json for metrics")
    print("  2. Compare stats to baseline (v7_selfplay/)")
    print("  3. Play games with new model to validate personality")
    print("  4. Tournament test against previous versions")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
