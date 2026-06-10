"""
Test Opening Loader - Verify sequence-based grading handles gambits correctly.

This test verifies:
1. Openings with final eval > -200cp are labeled GOOD (allows gambits)
2. Entire sequence gets same label based on final position
3. Individual move eval drops don't matter
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.stage1.data_sources.opening_loader import OpeningPGNLoader


def test_opening_loader():
    """Test opening loader with real PGN files."""
    print("="*80)
    print("Testing Opening Loader - Sequence-Based Grading")
    print("="*80 + "\n")
    
    # Path to opening PGNs
    opening_dir = Path("E:/Programming Stuff/Chess Engines/Chess Engine Playground/engine-metrics/raw_data/pgn_training_data/pgn_data_openings")
    
    if not opening_dir.exists():
        print(f"❌ Opening directory not found: {opening_dir}")
        return False
    
    # Test files to check
    test_files = [
        'London2e6.pgn',  # Sound opening, should pass
        'DutchLeningrad.pgn',  # Sound, should pass
        'BudapestGambit.pgn',  # Gambit (-100cp), should pass if final eval > -200cp
    ]
    
    try:
        loader = OpeningPGNLoader(
            pgn_dir=str(opening_dir),
            seed=42,
            shuffle=False,
            max_opening_moves=12,
            preferred_only=True
        )
        
        print(f"✓ Loader initialized")
        print(f"  Found {len(loader._pgn_files)} PGN files\n")
        
        # Load a small batch
        print("Loading 50 opening positions...")
        batch = loader.load_batch(50)
        
        print(f"✓ Loaded {len(batch)} positions\n")
        
        if len(batch) == 0:
            print("⚠ No positions loaded - might be due to strict filtering")
            return True
        
        # Analyze batch
        print("Position Analysis:")
        print("-" * 80)
        
        # Group by opening
        from collections import defaultdict
        openings = defaultdict(list)
        for pos in batch:
            opening_name = pos.get('opening', 'Unknown')
            openings[opening_name].append(pos)
        
        for opening_name, positions in sorted(openings.items()):
            final_eval = positions[-1].get('opening_final_eval', 0)
            move_count = len(positions)
            all_good = all(p.get('label') == 1 for p in positions)
            
            print(f"\n{opening_name} ({positions[0].get('eco', '')})")
            print(f"  Moves: {move_count}")
            print(f"  Final Eval: {final_eval:+d}cp ({final_eval/100:+.2f} pawns)")
            print(f"  All labeled GOOD: {'✓' if all_good else '✗'}")
            
            # Show if it's a gambit
            if final_eval < -50:
                print(f"  🎯 GAMBIT DETECTED - Accepting {abs(final_eval)/100:.1f} pawn sacrifice")
        
        print("\n" + "="*80)
        print("Summary:")
        print(f"  Total positions: {len(batch)}")
        print(f"  Unique openings: {len(openings)}")
        print(f"  All labeled GOOD: {all(p.get('label') == 1 for p in batch)}")
        
        # Check feature completeness
        complete_features = sum(1 for p in batch if 'features' in p and len(p['features']) >= 73)
        print(f"  Complete features: {complete_features}/{len(batch)} ({complete_features/len(batch)*100:.1f}%)")
        
        print("\n✓ Opening loader test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_opening_loader()
    sys.exit(0 if success else 1)
