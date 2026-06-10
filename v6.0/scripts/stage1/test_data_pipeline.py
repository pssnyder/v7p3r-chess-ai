"""
Test Multi-Source Data Pipeline - validates all data loaders and Stockfish integration.

This script tests the v6.1 data infrastructure:
1. Loads small batches from each data source
2. Validates positions with Stockfish
3. Reports statistics and performance
4. Verifies class balance

Run this before integrating into train_policy.py to ensure all components work.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.stage1.data_sources import MultiSourceDataLoader
from scripts.stage1.stockfish_validator import StockfishValidator
import json
import time


def find_data_paths():
    """Find data paths based on workspace structure."""
    # Base paths in workspace
    chess_engines_base = Path("E:/Programming Stuff/Chess Engines")
    
    paths = {
        'lichess_db': chess_engines_base / "Chess Engine Playground/engine-metrics/raw_data/pgn_training_data/json_data_lichess_evaluations_db/lichess_db_eval.jsonl",
        'v7p3r_bad': Path(__file__).parent.parent.parent / "data/stage1/v7p3r_bad_positions.jsonl",
        'opening_pgn': chess_engines_base / "Chess PGNs/training_data/pgn_data_openings",
        'tactics_csv': chess_engines_base / "Chess PGNs/training_data/csv_data_puzzles",
        'endgame_pgn': chess_engines_base / "Chess PGNs/training_data/pgn_data_endgames",
        'v7p3r_pgn': chess_engines_base / "Chess Engine Playground/engine-metrics/raw_data/game_records/Engine Battle 202512"
    }
    
    return paths


def test_individual_loaders(paths):
    """Test each loader individually."""
    print("\n" + "="*60)
    print("Testing Individual Data Loaders")
    print("="*60 + "\n")
    
    # Test Lichess loader
    if paths['lichess_db'].exists():
        print("Testing LichessDBLoader...")
        from scripts.stage1.data_sources.lichess_loader import LichessDBLoader
        
        try:
            loader = LichessDBLoader(str(paths['lichess_db']))
            positions = loader.load_batch(10)
            print(f"  ✓ Loaded {len(positions)} positions from Lichess DB")
            
            if positions:
                print(f"    Sample: {positions[0]['fen'][:30]}...")
                print(f"    Eval: {positions[0]['eval_cp']}cp, Grade: {positions[0]['grade']}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    else:
        print("  ⚠ Lichess DB not found, skipping")
    
    # Test V7P3R loader
    if paths['v7p3r_bad'].exists():
        print("\nTesting V7P3RGameLoader...")
        from scripts.stage1.data_sources.v7p3r_loader import V7P3RGameLoader
        
        try:
            loader = V7P3RGameLoader(
                str(paths['v7p3r_bad']),
                str(paths['v7p3r_pgn']) if paths['v7p3r_pgn'].exists() else None
            )
            positions = loader.load_batch(10)
            print(f"  ✓ Loaded {len(positions)} positions from V7P3R games")
            
            if positions:
                print(f"    Sample: {positions[0]['fen'][:30]}...")
                if 'eval_drop' in positions[0]:
                    print(f"    Eval drop: {positions[0].get('eval_drop', 0)}cp")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    else:
        print("  ⚠ V7P3R bad positions not found, skipping")
    
    # Test Opening loader
    if paths['opening_pgn'].exists():
        print("\nTesting OpeningPGNLoader...")
        from scripts.stage1.data_sources.opening_loader import OpeningPGNLoader
        
        try:
            loader = OpeningPGNLoader(str(paths['opening_pgn']))
            positions = loader.load_batch(10)
            print(f"  ✓ Loaded {len(positions)} positions from opening PGNs")
            
            if positions:
                print(f"    Sample: {positions[0]['fen'][:30]}...")
                print(f"    Opening: {positions[0].get('opening', 'Unknown')}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    else:
        print("  ⚠ Opening PGNs not found, skipping")
    
    # Test Tactics loader
    if paths['tactics_csv'].exists():
        print("\nTesting TacticsLoader...")
        from scripts.stage1.data_sources.tactics_loader import TacticsLoader
        
        try:
            loader = TacticsLoader(str(paths['tactics_csv']))
            positions = loader.load_batch(10)
            print(f"  ✓ Loaded {len(positions)} positions from tactics")
            
            if positions:
                print(f"    Sample: {positions[0]['fen'][:30]}...")
                print(f"    Themes: {positions[0].get('themes', [])}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    else:
        print("  ⚠ Tactics CSV not found, skipping")
    
    # Test Endgame loader
    if paths['endgame_pgn'].exists():
        print("\nTesting EndgameLoader...")
        from scripts.stage1.data_sources.endgame_loader import EndgameLoader
        
        try:
            loader = EndgameLoader(str(paths['endgame_pgn']))
            positions = loader.load_batch(10)
            print(f"  ✓ Loaded {len(positions)} positions from endgames")
            
            if positions:
                print(f"    Sample: {positions[0]['fen'][:30]}...")
                print(f"    Pieces: {positions[0].get('piece_count', 0)}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    else:
        print("  ⚠ Endgame PGNs not found, skipping")


def test_multi_source_loader(paths):
    """Test the multi-source orchestrator."""
    print("\n" + "="*60)
    print("Testing Multi-Source Data Loader")
    print("="*60 + "\n")
    
    try:
        loader = MultiSourceDataLoader(
            lichess_db_path=str(paths['lichess_db']),
            v7p3r_bad_positions=str(paths['v7p3r_bad']),
            opening_pgn_dir=str(paths['opening_pgn']),
            tactics_csv_path=str(paths['tactics_csv']),
            endgame_pgn_dir=str(paths['endgame_pgn']),
            v7p3r_pgn_dir=str(paths['v7p3r_pgn']) if paths['v7p3r_pgn'].exists() else None,
            seed=42,
            shuffle=True
        )
        
        print("Loading mixed batch of 100 positions...")
        start_time = time.time()
        batch = loader.load_batch(100)
        load_time = time.time() - start_time
        
        print(f"\n✓ Loaded {len(batch)} positions in {load_time:.2f}s")
        
        # Analyze batch composition
        sources = {}
        labels = {0: 0, 1: 0}
        
        for pos in batch:
            source = pos['source']
            sources[source] = sources.get(source, 0) + 1
            label = pos.get('label', 1)
            labels[label] = labels.get(label, 0) + 1
        
        print("\nBatch Composition:")
        for source, count in sorted(sources.items()):
            print(f"  {source:12s}: {count:3d} ({count/len(batch)*100:.1f}%)")
        
        print("\nLabel Distribution:")
        total_labels = sum(labels.values())
        for label, count in sorted(labels.items()):
            label_name = "Bad" if label == 0 else "Good"
            print(f"  {label_name:4s}: {count:3d} ({count/total_labels*100:.1f}%)")
        
        # Print loader stats
        loader.print_summary()
        
        return batch
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_stockfish_validator(sample_positions):
    """Test Stockfish validation with caching."""
    print("\n" + "="*60)
    print("Testing Stockfish Validator")
    print("="*60 + "\n")
    
    if not sample_positions:
        print("⚠ No positions to validate, skipping")
        return
    
    try:
        # Use first 10 positions for testing
        test_batch = sample_positions[:10]
        
        print(f"Validating {len(test_batch)} positions...")
        print("(This may take ~1 second per position on first run)")
        
        validator = StockfishValidator(
            stockfish_path="stockfish",  # Assumes stockfish in PATH
            analysis_time=0.1,
            min_depth=15
        )
        
        start_time = time.time()
        validated = validator.validate_batch(test_batch, update_in_place=True)
        validation_time = time.time() - start_time
        
        print(f"\n✓ Validated {len(validated)} positions in {validation_time:.2f}s")
        print(f"  Avg: {validation_time/len(validated)*1000:.0f}ms per position")
        
        # Show sample validations
        print("\nSample Validations:")
        for i, pos in enumerate(validated[:3]):
            print(f"\n  Position {i+1}:")
            print(f"    FEN: {pos['fen'][:50]}...")
            print(f"    Eval: {pos.get('eval_cp', 0):+5d}cp")
            print(f"    Grade: {pos.get('grade', 1)}/5")
            if pos.get('mate_in'):
                print(f"    Mate in: {pos['mate_in']}")
        
        # Print cache stats
        validator.print_stats()
        
        # Test cache hit on second run
        print("\nTesting cache (re-validating same positions)...")
        start_time = time.time()
        validator.validate_batch(test_batch[:3], update_in_place=False)
        cache_time = time.time() - start_time
        
        print(f"✓ Re-validated 3 positions in {cache_time*1000:.0f}ms (should be cached)")
        
        validator.close()
        
    except FileNotFoundError:
        print("✗ Stockfish not found in PATH")
        print("  Please install Stockfish or provide path")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


def test_feature_calculation(sample_positions):
    """Test feature calculation on sample positions."""
    print("\n" + "="*60)
    print("Testing Feature Calculation")
    print("="*60 + "\n")
    
    if not sample_positions:
        print("⚠ No positions to test, skipping")
        return
    
    # Check if positions have features
    features_present = all('features' in pos for pos in sample_positions)
    
    if features_present:
        print(f"✓ All {len(sample_positions)} positions have features")
        
        # Count feature dimensions
        sample_features = sample_positions[0]['features']
        feature_count = len([k for k in sample_features.keys() if k.startswith('F')])
        
        print(f"  Feature count: {feature_count}")
        
        # Show sample features
        print("\nSample Feature Values (Position 1):")
        feature_items = sorted(
            [(k, v) for k, v in sample_features.items() if k.startswith('F')],
            key=lambda x: x[0]
        )[:10]
        
        for feat, val in feature_items:
            if isinstance(val, (int, float)):
                print(f"  {feat}: {val:.4f}")
            else:
                print(f"  {feat}: {val}")
        
    else:
        print("✗ Some positions missing features")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("V6.1 Multi-Source Data Pipeline Test Suite")
    print("="*80)
    
    # Find data paths
    paths = find_data_paths()
    
    print("\nData Paths:")
    for name, path in paths.items():
        exists = "✓" if path.exists() else "✗"
        print(f"  {exists} {name:12s}: {path}")
    
    # Run tests
    test_individual_loaders(paths)
    
    sample_batch = test_multi_source_loader(paths)
    
    if sample_batch:
        test_feature_calculation(sample_batch)
        test_stockfish_validator(sample_batch)
    
    print("\n" + "="*80)
    print("Test Suite Complete")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
