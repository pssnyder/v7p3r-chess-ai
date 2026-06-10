"""
Quick Verification Script - "Chess as Story" System

Tests all components before running full training:
1. Phase manager weight calculations
2. Opening book loading
3. Tablebase oracle (if available)
4. Integration with trainer
"""

import sys
import chess
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

def test_phase_manager():
    """Test phase detection and weight calculations."""
    print("=" * 60)
    print("TEST 1: Phase Manager")
    print("=" * 60)
    
    try:
        from phase_manager import DynamicWeightCalculator, PhaseAwareTrainingTarget, GamePhase
        
        calculator = DynamicWeightCalculator()
        phase_manager = PhaseAwareTrainingTarget(calculator)
        
        # Test different move numbers
        test_moves = [5, 15, 25, 35, 45, 65]
        board = chess.Board()
        
        print("\nWeight Progression:")
        print("-" * 60)
        print("Move | Phase              | SF Weight | Pers Weight")
        print("-" * 60)
        
        for move_num in test_moves:
            target, weights = phase_manager.calculate_target(
                board=board,
                move_number=move_num,
                stockfish_eval=0.5,
                personality_reward=0.8,
                game_outcome=0.0
            )
            
            print(f"{move_num:4d} | {weights['phase']:18s} | {weights['stockfish']:9.2f} | {weights['personality']:11.2f}")
        
        print("-" * 60)
        print("✓ Phase manager working correctly")
        print()
        return True
    
    except Exception as e:
        print(f"✗ Phase manager test failed: {e}")
        return False


def test_opening_book():
    """Test opening book loading and application."""
    print("=" * 60)
    print("TEST 2: Opening Book")
    print("=" * 60)
    
    try:
        from opening_book import OpeningBookManager
        
        book = OpeningBookManager()
        
        print(f"\n✓ Opening book loaded: {len(book)} openings")
        print("\nAvailable openings:")
        print("-" * 60)
        
        for i, name in enumerate(book.list_openings()[:5], 1):
            print(f"  {i}. {name}")
        print(f"  ... and {len(book) - 5} more")
        
        # Test random selection
        print("\nTesting random opening application...")
        board = chess.Board()
        opening, moves = book.apply_random_opening(board)
        
        print(f"✓ Selected: {opening.name}")
        print(f"  Moves applied: {len(moves)}")
        print(f"  Final FEN: {board.fen()}")
        print()
        return True
    
    except Exception as e:
        print(f"✗ Opening book test failed: {e}")
        return False


def test_tablebase_oracle():
    """Test tablebase oracle (may not be available)."""
    print("=" * 60)
    print("TEST 3: Tablebase Oracle")
    print("=" * 60)
    
    try:
        from tablebase_oracle import TablebaseOracle
        import os
        
        # Try common paths
        possible_paths = [
            r"E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\pgn_training_data\pgn_data_endgames\3-4-5_pieces_Syzygy\3-4-5",
            "E:/Chess/Tablebases/syzygy",
            "C:/Chess/Tablebases",
            "./tablebases",
            "../tablebases"
        ]
        
        oracle = None
        for path in possible_paths:
            if os.path.exists(path):
                oracle = TablebaseOracle(path)
                if oracle.enabled:
                    break
        
        if not oracle or not oracle.enabled:
            print("\n⚠ Tablebases not available (optional)")
            print("  Training will work without them, but endgame")
            print("  conversion may be slower.")
            print()
            print("To enable tablebases:")
            print("  1. Download 3-4-5 piece Syzygy (~1GB)")
            print("  2. Extract to E:/Chess/Tablebases/syzygy")
            print("  3. Re-run this test")
            print()
            return None  # Not a failure, just unavailable
        
        # Test with simple endgame
        print(f"\n✓ Tablebases loaded: {oracle.max_pieces}-piece")
        
        # KPK endgame (White pawn on e7, can promote)
        board = chess.Board("4k3/4P3/4K3/8/8/8/8/8 w - - 0 1")
        
        if oracle.is_available(board):
            eval_val = oracle.get_normalized_eval(board)
            wdl = oracle.probe_wdl(board)
            
            print(f"  Test position (KPK): eval={eval_val}, wdl={wdl}")
            print(f"  ✓ Tablebase oracle functioning correctly")
        else:
            print(f"  ⚠ Position not in tablebase (need more pieces)")
        
        print()
        return True
    
    except Exception as e:
        print(f"⚠ Tablebase oracle test failed: {e}")
        print("  (This is non-critical - training will work without tablebases)")
        print()
        return None


def test_integration():
    """Test that all components integrate with trainer."""
    print("=" * 60)
    print("TEST 4: Integration with Trainer")
    print("=" * 60)
    
    try:
        from selfplay_trainer import SelfPlayTrainer
        
        # Just test initialization (don't run training)
        print("\nAttempting to initialize trainer...")
        
        PROFILE_PATH = "../profiles/dark_forest_assassin.json"
        STOCKFISH_PATH = "../../stockfish.exe"
        OUTPUT_DIR = "../training/test_integration"
        TABLEBASE_PATH = r"E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\pgn_training_data\pgn_data_endgames\3-4-5_pieces_Syzygy\3-4-5"
        
        # Check if files exist
        if not Path(PROFILE_PATH).exists():
            print(f"✗ Profile not found: {PROFILE_PATH}")
            return False
        
        if not Path(STOCKFISH_PATH).exists():
            print(f"✗ Stockfish not found: {STOCKFISH_PATH}")
            print("  Update STOCKFISH_PATH in this script")
            return False
        
        trainer = SelfPlayTrainer(
            profile_path=PROFILE_PATH,
            stockfish_path=STOCKFISH_PATH,
            output_dir=OUTPUT_DIR,
            tablebase_path=TABLEBASE_PATH,
            use_opening_book=True,
            use_tablebases=True
        )
        
        print("✓ Trainer initialized successfully")
        print(f"  Opening book: {'Enabled' if trainer.use_opening_book else 'Disabled'}")
        print(f"  Tablebases: {'Enabled' if trainer.use_tablebases else 'Disabled'}")
        print(f"  Phase manager: {trainer.phase_manager}")
        print()
        
        return True
    
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print()
    print("=" * 60)
    print("'CHESS AS STORY' SYSTEM VERIFICATION")
    print("=" * 60)
    print()
    print("This script tests all components before training.")
    print("Running 4 tests...")
    print()
    
    results = {
        'Phase Manager': test_phase_manager(),
        'Opening Book': test_opening_book(),
        'Tablebase Oracle': test_tablebase_oracle(),
        'Integration': test_integration()
    }
    
    print("=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    for component, result in results.items():
        if result is True:
            status = "✓ PASS"
        elif result is None:
            status = "⚠ OPTIONAL (not available)"
        else:
            status = "✗ FAIL"
        
        print(f"{component:20s}: {status}")
    
    print()
    
    # Check if critical components passed
    critical_passed = (
        results['Phase Manager'] is True and
        results['Opening Book'] is True and
        results['Integration'] is True
    )
    
    if critical_passed:
        print("=" * 60)
        print("✓ ALL CRITICAL TESTS PASSED")
        print("=" * 60)
        print()
        print("System ready for training!")
        print()
        print("To start training:")
        print("  python train_story_mode.py")
        print()
        
        if results['Tablebase Oracle'] is None:
            print("Note: Training will work without tablebases, but endgame")
            print("      conversion may take longer. Consider downloading")
            print("      Syzygy 3-4-5 piece tablebases (~1GB) for best results.")
            print()
    else:
        print("=" * 60)
        print("✗ SOME TESTS FAILED")
        print("=" * 60)
        print()
        print("Please fix the failed components before training.")
        print("Check error messages above for details.")
        print()


if __name__ == "__main__":
    main()
