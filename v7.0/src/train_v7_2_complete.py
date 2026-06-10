"""
V7.2 Training Workflow - Supervised Bootstrap + Generational Refinement

WORKFLOW:
  1. Clean PGN files (pgn_preprocessor.py)
  2. Supervised pre-training from GM games (supervised_gm_trainer.py)
  3. Generational self-play refinement (generational_trainer.py)

This is the "Matrix plug-in" approach:
  - Fast baseline from grandmaster knowledge (minutes)
  - Then self-play refinement for creativity (hours)
"""

import sys
from pathlib import Path
import torch
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent))

from supervised_gm_trainer import SupervisedGMTrainer
from generational_trainer import GenerationalTrainer
from pgn_preprocessor import PGNPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_complete_workflow(
    clean_pgns: bool = False,
    supervised_epochs: int = 10,
    selfplay_games: int = 100,
    num_generations: int = 10
):
    """
    Run complete V7.2 training workflow.
    
    Args:
        clean_pgns: Whether to clean PGN files first
        supervised_epochs: Epochs for supervised pre-training
        selfplay_games: Games per generation for self-play
        num_generations: Number of generations to train
    """
    print("=" * 80)
    print("V7P3R v7.2 - COMPLETE TRAINING WORKFLOW")
    print("=" * 80)
    print()
    print("This combines supervised learning from grandmaster games")
    print("with generational self-play refinement.")
    print()
    print("WORKFLOW:")
    print("  1. Clean PGN files (if needed)")
    print("  2. Supervised pre-training (~10 minutes)")
    print("  3. Generational self-play (~10-15 hours)")
    print()
    print("=" * 80)
    print()
    
    # Configuration paths
    PROFILE_PATH = "../profiles/dark_forest_assassin.json"
    STOCKFISH_PATH = "../../stockfish.exe"
    TABLEBASE_PATH = r"E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\pgn_training_data\pgn_data_endgames\3-4-5_pieces_Syzygy\3-4-5"
    
    PGN_DIRS = [
        Path(r"E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\pgn_training_data\pgn_data_important_games"),
        Path(r"E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\pgn_training_data\pgn_data_tactics"),
        Path(r"E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\pgn_training_data\pgn_data_general"),
    ]
    
    SUPERVISED_OUTPUT = "../training/v7_supervised"
    GENERATIONAL_OUTPUT = "../training/v7_generational_v2"
    
    # =========================================================================
    # STEP 1: CLEAN PGN FILES (Optional)
    # =========================================================================
    if clean_pgns:
        print("=" * 80)
        print("STEP 1: CLEANING PGN FILES")
        print("=" * 80)
        print()
        
        preprocessor = PGNPreprocessor()
        
        for pgn_dir in PGN_DIRS:
            if pgn_dir.exists():
                logger.info(f"Cleaning: {pgn_dir.name}")
                preprocessor.process_directory(
                    pgn_dir,
                    pattern="*.pgn",
                    overwrite=True
                )
        
        print()
        print("✓ PGN cleaning complete")
        print()
    else:
        print("Skipping PGN cleaning (using existing cleaned files)")
        print()
    
    # =========================================================================
    # STEP 2: SUPERVISED PRE-TRAINING
    # =========================================================================
    print("=" * 80)
    print("STEP 2: SUPERVISED PRE-TRAINING FROM GM GAMES")
    print("=" * 80)
    print()
    
    supervised_trainer = SupervisedGMTrainer(
        profile_path=PROFILE_PATH,
        output_dir=SUPERVISED_OUTPUT
    )
    
    # Load positions from cleaned PGNs
    for pgn_dir in PGN_DIRS:
        cleaned_dir = pgn_dir / "cleaned"
        if cleaned_dir.exists():
            logger.info(f"Loading from: {cleaned_dir}")
            supervised_trainer.load_games_from_directory(
                cleaned_dir,
                pattern="*_clean.pgn",
                winner_only=True
            )
    
    if supervised_trainer.positions_extracted == 0:
        logger.error("No positions extracted! Check that cleaned PGN files exist.")
        logger.error("Run with clean_pgns=True first.")
        return
    
    logger.info(f"Total positions: {supervised_trainer.positions_extracted}")
    logger.info(f"Total games: {supervised_trainer.games_processed}")
    
    # Train
    print()
    logger.info("Starting supervised training...")
    
    losses = supervised_trainer.train_on_positions(
        epochs=supervised_epochs,
        batch_size=256,
        learning_rate=0.001
    )
    
    supervised_trainer.save_training_stats()
    
    print()
    logger.info(f"✓ Supervised training complete - Final loss: {losses[-1]:.4f}")
    
    # =========================================================================
    # STEP 3: GENERATIONAL SELF-PLAY TRAINING
    # =========================================================================
    print()
    print("=" * 80)
    print("STEP 3: GENERATIONAL SELF-PLAY REFINEMENT")
    print("=" * 80)
    print()
    
    # Initialize generational trainer
    gen_trainer = GenerationalTrainer(
        profile_path=PROFILE_PATH,
        stockfish_path=STOCKFISH_PATH,
        output_dir=GENERATIONAL_OUTPUT,
        tablebase_path=TABLEBASE_PATH
    )
    
    # Initialize network structure first
    gen_trainer.initialize_first_generation()
    
    # Load supervised model weights into Generation 0
    supervised_model_path = Path(SUPERVISED_OUTPUT) / "supervised_final.pt"
    
    if not supervised_model_path.exists():
        logger.error(f"Supervised model not found: {supervised_model_path}")
        return
    
    logger.info("Loading supervised model as Generation 0...")
    gen_trainer.best_model.load_state_dict(torch.load(supervised_model_path))
    
    # Save as gen_0000_supervised.pt (overwrite the random init)
    gen_0_path = Path(GENERATIONAL_OUTPUT) / "gen_0000_initial.pt"
    torch.save(gen_trainer.best_model.state_dict(), gen_0_path)
    logger.info(f"✓ Saved supervised baseline as: {gen_0_path.name}")
    
    print()
    logger.info(f"Starting generational training ({num_generations} generations)...")
    print()
    
    # Run generational training
    for gen in range(num_generations):
        logger.info(f"=" * 80)
        logger.info(f"GENERATION {gen + 1}/{num_generations}")
        logger.info(f"=" * 80)
        
        # Train new generation
        gen_trainer.train_new_generation(
            selfplay_games=selfplay_games,
            batch_size=256,
            train_every_n_games=10
        )
        
        # Evaluate
        result = gen_trainer.evaluate_generation(num_games=6)
        
        # Accept or reject
        if result.accepted:
            gen_trainer.accept_generation(result)
            logger.info(f"✓ Generation {gen + 1} ACCEPTED ({result.win_rate:.1%} win rate)")
        else:
            gen_trainer.reject_generation(result)
            logger.info(f"✗ Generation {gen + 1} REJECTED ({result.win_rate:.1%} win rate)")
        
        print()
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print()
    print("SUPERVISED PRE-TRAINING:")
    print(f"  Games processed: {supervised_trainer.games_processed}")
    print(f"  Positions trained: {supervised_trainer.positions_extracted}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print()
    print("GENERATIONAL REFINEMENT:")
    print(f"  Generations trained: {len(gen_trainer.generation_history)}")
    print(f"  Accepted: {sum(1 for g in gen_trainer.generation_history if g['accepted'])}")
    print(f"  Rejected: {sum(1 for g in gen_trainer.generation_history if not g['accepted'])}")
    print(f"  Final generation: {gen_trainer.current_generation}")
    print()
    print("OUTPUTS:")
    print(f"  Supervised model: {SUPERVISED_OUTPUT}")
    print(f"  Generational models: {GENERATIONAL_OUTPUT}")
    print("=" * 80)


def main():
    """Interactive workflow launcher."""
    print("=" * 80)
    print("V7P3R v7.2 - TRAINING WORKFLOW LAUNCHER")
    print("=" * 80)
    print()
    print("Options:")
    print("  1. Complete workflow (clean PGNs + supervised + generational)")
    print("  2. Supervised pre-training only")
    print("  3. Generational training only (requires existing supervised model)")
    print("  4. Quick test (1 epoch supervised, 2 generations)")
    print()
    
    choice = input("Select option (1-4): ").strip()
    
    if choice == '1':
        print()
        print("Running COMPLETE workflow...")
        print("This will take ~10-15 hours total.")
        print()
        input("Press ENTER to confirm...")
        run_complete_workflow(
            clean_pgns=True,
            supervised_epochs=10,
            selfplay_games=100,
            num_generations=10
        )
    
    elif choice == '2':
        print()
        print("Running SUPERVISED pre-training only...")
        print()
        input("Press ENTER to confirm...")
        run_complete_workflow(
            clean_pgns=False,
            supervised_epochs=10,
            selfplay_games=0,  # Will be skipped
            num_generations=0  # Will be skipped
        )
    
    elif choice == '3':
        print()
        print("Running GENERATIONAL training only...")
        print("Make sure supervised model exists!")
        print()
        input("Press ENTER to confirm...")
        
        from generational_trainer import GenerationalTrainer
        
        PROFILE_PATH = "../profiles/dark_forest_assassin.json"
        STOCKFISH_PATH = "../../stockfish.exe"
        SUPERVISED_OUTPUT = "../training/supervised_gm"
        GENERATIONAL_OUTPUT = "../training/v7_generational_v2"
        TABLEBASE_PATH = r"E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\pgn_training_data\pgn_data_endgames\3-4-5_pieces_Syzygy\3-4-5"
        
        trainer = GenerationalTrainer(
            profile_path=PROFILE_PATH,
            stockfish_path=STOCKFISH_PATH,
            output_dir=GENERATIONAL_OUTPUT,
            tablebase_path=TABLEBASE_PATH
        )
        
        # Initialize network structure
        trainer.initialize_first_generation()
        
        # Load supervised model weights
        supervised_path = Path(SUPERVISED_OUTPUT) / "supervised_final.pt"
        if supervised_path.exists():
            print(f"\n[OK] Loading supervised model: {supervised_path}")
            trainer.best_model.load_state_dict(torch.load(supervised_path))
            print(f"[OK] Generation 0 initialized from supervised training")
        else:
            print(f"\n[WARNING] Supervised model not found: {supervised_path}")
            print(f"[INFO] Starting with random initialization instead")
        
        for gen in range(10):
            trainer.train_new_generation(selfplay_games=100)
            result = trainer.evaluate_generation(num_games=6)
            
            if result.accepted:
                trainer.accept_generation(result)
            else:
                trainer.reject_generation(result)
    
    elif choice == '4':
        print()
        print("Running QUICK TEST...")
        print("This will take ~30-60 minutes.")
        print()
        input("Press ENTER to confirm...")
        run_complete_workflow(
            clean_pgns=False,
            supervised_epochs=1,
            selfplay_games=20,
            num_generations=2
        )
    
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
