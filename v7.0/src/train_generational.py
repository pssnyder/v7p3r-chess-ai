"""
V7P3R v7.1 - Generational Training with Revised Weight Curve

MAJOR CHANGES from v7.0:
1. GENERATIONAL ARCHITECTURE:
   - New model vs Old model evaluation (6-game matches)
   - Win rate > 50% required for acceptance
   - MEANINGFUL win/loss metrics (unlike pure self-play)

2. REVISED WEIGHT CURVE:
   - Opening (1-10): 90% SF (same)
   - Early MG (11-20): 90% → 10% SF (steeper drop)
   - Deep MG (21-40): 20% SF (controlled chaos, up from 10%)
   - Late MG (41-60): 20% → 80% SF (steeper recovery)
   - Endgame (61+): 100% SF (perfect technique, up from 50%)

3. EVALUATION MATCHES:
   - 6 games per generation (3 White, 3 Black)
   - Alternating colors for fairness
   - Tiebreaker: wins > draws

This should address:
- Color bias (equal perspective)
- Meaningful progress metrics (vs previous generation)
- Better endgame conversion (100% SF in endgame)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from generational_trainer import GenerationalTrainer


def main():
    print("=" * 80)
    print("V7P3R v7.1 - GENERATIONAL TRAINING SYSTEM")
    print("=" * 80)
    print()
    print("Weight Curve (REVISED):")
    print("  Opening (1-10):        90% SF → Learn fundamentals")
    print("  Early MG (11-20):      90% → 10% SF → Enter chaos")
    print("  Deep MG (21-40):       20% SF → CONTROLLED CHAOS (up from 10%)")
    print("  Late MG (41-60):       20% → 80% SF → Return to precision")
    print("  Endgame (61+):         100% SF → Perfect technique (up from 50%)")
    print("  Tablebase (≤5 pieces): 100% Perfect → Math solves endgames")
    print()
    print("Generational Architecture:")
    print("  1. Train new model via self-play (100 games)")
    print("  2. Evaluate new vs best (6-game match: 3 White, 3 Black)")
    print("  3. Accept only if win rate > 50%")
    print("  4. Repeat for N generations")
    print()
    print("=" * 80)
    print()
    
    # Configuration
    PROFILE_PATH = "../profiles/dark_forest_assassin.json"
    STOCKFISH_PATH = "../../stockfish.exe"
    OUTPUT_DIR = "../training/v7_generational"
    
    # Tablebase path
    TABLEBASE_PATH = r"E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\pgn_training_data\pgn_data_endgames\3-4-5_pieces_Syzygy\3-4-5"
    
    # Training parameters
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
        opening_book_pgn=None,  # Use default aggressive openings
        tablebase_path=TABLEBASE_PATH
    )
    
    print()
    print("=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Self-play games per generation: {SELFPLAY_GAMES}")
    print(f"Evaluation match format: {EVALUATION_GAMES} games")
    print(f"  - {EVALUATION_GAMES//2} as White (new model)")
    print(f"  - {EVALUATION_GAMES//2} as Black (new model)")
    print(f"Maximum generations: {MAX_GENERATIONS}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print()
    print("Acceptance Criteria:")
    print("  Win rate > 50%")
    print("  OR win rate = 50% AND wins > draws (tiebreaker)")
    print()
    print("=" * 80)
    print()
    
    # Run training
    input("Press ENTER to start generational training (or Ctrl+C to cancel)...")
    print()
    
    trainer.run_full_cycle(
        selfplay_games=SELFPLAY_GAMES,
        evaluation_games=EVALUATION_GAMES,
        max_generations=MAX_GENERATIONS
    )
    
    print()
    print("=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print()
    print(f"Results saved to: {OUTPUT_DIR}")
    print()
    print("What Improved:")
    print("  ✓ MEANINGFUL metrics (new vs old, not self vs self)")
    print("  ✓ Color-balanced evaluation (3 White, 3 Black)")
    print("  ✓ Better endgame conversion (100% SF, up from 50%)")
    print("  ✓ Controlled middlegame chaos (20% SF, up from 10%)")
    print("  ✓ Generational improvement tracking")
    print()
    print("Next Steps:")
    print("  1. Review generation_history.json for progress")
    print("  2. Analyze accepted vs rejected generations")
    print("  3. Load best_model.pt for tournament play")
    print("  4. Compare win rates across generations")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
