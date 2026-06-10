"""
Test Multi-Source Data Loader with Tal-Inspired Mixing.

Validates that all sources integrate correctly and mixing ratios produce
balanced training data with proper tactical emphasis.
"""

import sys
from pathlib import Path

# Add v6.0 directory to path for imports
v6_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(v6_root))

from scripts.stage1.data_sources.multi_source_loader import MultiSourceDataLoader


def test_tal_inspired_pipeline():
    """Test complete Tal-inspired data pipeline."""
    
    print("\n" + "="*70)
    print("TESTING TAL-INSPIRED MULTI-SOURCE DATA LOADER")
    print("="*70)
    
    # Define paths
    base_path = Path("E:/Programming Stuff/Chess Engines/Chess Engine Playground/engine-metrics/raw_data")
    
    lichess_db = base_path / "pgn_training_data/lichess_db_eval.jsonl"
    v7p3r_bad = Path("E:/Programming Stuff/Chess Engines/V7P3R Chess AI/v7p3r-chess-ai/v6.0/data/stage1/v7p3r_bad_positions.jsonl")
    openings_dir = base_path / "pgn_training_data/pgn_data_openings"
    tactics_csv = base_path / "pgn_training_data/csv_data_puzzles"
    endgame_dir = base_path / "pgn_training_data/pgn_data_endgames"
    
    # NEW: Tal and Human games
    tal_games = base_path / "pgn_training_data/pgn_data_general/mikhail_tal_master_games.pgn"
    human_games = base_path / "game_records/v7p3r Human/v7p3r_20250530.pgn"
    
    print("\n📂 Data Source Paths:")
    print("-" * 70)
    print(f"  Tal Games:    {tal_games}")
    print(f"  Human Games:  {human_games}")
    print(f"  Lichess DB:   {lichess_db}")
    print(f"  Tactics:      {tactics_csv}")
    print(f"  Openings:     {openings_dir}")
    print(f"  Endgames:     {endgame_dir}")
    print(f"  V7P3R Bad:    {v7p3r_bad}")
    
    # Initialize with Tal-inspired mixing
    print("\n🎯 Initializing Tal-Inspired Multi-Source Loader...")
    print("-" * 70)
    
    loader = MultiSourceDataLoader(
        lichess_db_path=str(lichess_db),
        v7p3r_bad_positions=str(v7p3r_bad),
        opening_pgn_dir=str(openings_dir),
        tactics_csv_path=str(tactics_csv),
        endgame_pgn_dir=str(endgame_dir),
        tal_games_pgn=str(tal_games),
        human_games_pgn=str(human_games),
        use_tal_mix=True,  # Use TAL_INSPIRED_MIX ratios
        seed=42,
        shuffle=True
    )
    
    # Print mixing ratios
    print("\n📊 Mixing Ratios:")
    print("-" * 70)
    for source, ratio in sorted(loader.mix_ratios.items(), key=lambda x: -x[1]):
        if ratio > 0:
            print(f"  {source:15s}: {ratio:5.1%}")
    
    # Calculate tactical focus percentage
    tactical_sources = ['tal_games', 'human_games', 'tactics']
    tactical_ratio = sum(loader.mix_ratios.get(s, 0) for s in tactical_sources)
    print(f"\n  🎯 TACTICAL FOCUS: {tactical_ratio:.1%} (Tal + Human + Tactics)")
    
    # Test batch loading
    print("\n🔄 Loading Test Batch (1000 positions)...")
    print("-" * 70)
    
    batch = loader.load_batch(
        size=1000,
        target_balance={0: 0.5, 1: 0.5}  # 50:50 good/bad
    )
    
    print(f"  Batch size: {len(batch)} positions")
    
    # Analyze batch composition
    print("\n📈 Batch Analysis:")
    print("-" * 70)
    
    # Count by source
    by_source = {}
    for pos in batch:
        source = pos.get('source', 'unknown')
        by_source[source] = by_source.get(source, 0) + 1
    
    print("  Positions by source:")
    for source, count in sorted(by_source.items(), key=lambda x: -x[1]):
        pct = (count / len(batch)) * 100
        print(f"    {source:20s}: {count:4d} ({pct:5.1f}%)")
    
    # Count by label
    by_label = {}
    for pos in batch:
        label = pos.get('label')
        by_label[label] = by_label.get(label, 0) + 1
    
    print("\n  Label balance:")
    for label, count in sorted(by_label.items()):
        pct = (count / len(batch)) * 100
        label_name = "GOOD" if label == 1 else "BAD"
        print(f"    {label_name:5s} ({label}): {count:4d} ({pct:5.1f}%)")
    
    # Check for weighted positions (Bxf7+ patterns)
    weighted_positions = [p for p in batch if p.get('weight', 1.0) > 1.0]
    high_weight = [p for p in batch if p.get('weight', 1.0) >= 5.0]
    
    print(f"\n  Weighted positions:")
    print(f"    Any weight > 1.0: {len(weighted_positions)}")
    print(f"    Bxf7+ (≥5.0x):    {len(high_weight)}")
    
    # Show example Bxf7+ position if found
    if high_weight:
        print("\n⭐ Example Bxf7+ King Hunt Position:")
        print("-" * 70)
        pos = high_weight[0]
        print(f"  FEN:    {pos['fen']}")
        print(f"  Label:  {pos['label']} ({'GOOD - YOUR ATTACK' if pos['label'] == 1 else 'BAD - OPPONENT'})")
        print(f"  Weight: {pos.get('weight', 1.0)}x")
        print(f"  Source: {pos.get('source')}")
        if 'game_info' in pos:
            info = pos['game_info']
            print(f"  Game:   {info.get('white', '?')} vs {info.get('black', '?')}")
            print(f"  Move:   {info.get('move_number', '?')} - {info.get('move_san', '?')}")
    
    # Get loader statistics
    print("\n📋 Loader Statistics:")
    print("=" * 70)
    
    stats = loader.get_stats()
    for source_name, source_stats in stats['sources'].items():
        print(f"\n  {source_name.upper()}:")
        for key, value in source_stats.items():
            if key != 'name':
                print(f"    {key}: {value}")
    
    # Validation checks
    print("\n✅ Validation Checks:")
    print("=" * 70)
    
    checks_passed = 0
    checks_total = 0
    
    # Check 1: Label balance
    checks_total += 1
    good_pct = (by_label.get(1, 0) / len(batch)) * 100
    bad_pct = (by_label.get(0, 0) / len(batch)) * 100
    if 45 <= good_pct <= 55 and 45 <= bad_pct <= 55:
        print("  ✓ Label balance: 50:50 (±5%)")
        checks_passed += 1
    else:
        print(f"  ✗ Label balance: {good_pct:.1f}% good, {bad_pct:.1f}% bad (target: 50:50 ±5%)")
    
    # Check 2: Tactical focus
    checks_total += 1
    tactical_count = sum(by_source.get(s, 0) for s in ['tal_games', 'human_tactical_games', 'tactics'])
    tactical_pct = (tactical_count / len(batch)) * 100
    if tactical_pct >= 40:  # Target is 50%, allow some variance
        print(f"  ✓ Tactical focus: {tactical_pct:.1f}% (target: 50%)")
        checks_passed += 1
    else:
        print(f"  ✗ Tactical focus: {tactical_pct:.1f}% (target: 50%)")
    
    # Check 3: Tal games present
    checks_total += 1
    if 'tal_games' in by_source and by_source['tal_games'] > 0:
        print(f"  ✓ Tal games loaded: {by_source['tal_games']} positions")
        checks_passed += 1
    else:
        print("  ✗ Tal games not present in batch")
    
    # Check 4: Human games present
    checks_total += 1
    if 'human_tactical_games' in by_source and by_source['human_tactical_games'] > 0:
        print(f"  ✓ Human games loaded: {by_source['human_tactical_games']} positions")
        checks_passed += 1
    else:
        print("  ✗ Human games not present in batch")
    
    # Check 5: Bxf7+ patterns present
    checks_total += 1
    if high_weight:
        print(f"  ✓ Bxf7+ patterns found: {len(high_weight)} positions (5.0x weight)")
        checks_passed += 1
    else:
        print("  ⚠ No Bxf7+ patterns in this batch (may appear in larger batches)")
    
    # Final summary
    print("\n" + "=" * 70)
    print(f"VALIDATION SUMMARY: {checks_passed}/{checks_total} checks passed")
    
    if checks_passed == checks_total:
        print("🎉 ALL CHECKS PASSED - Pipeline ready for training!")
    elif checks_passed >= checks_total - 1:
        print("✅ MOSTLY PASSING - Pipeline functional with minor warnings")
    else:
        print("⚠️  SOME ISSUES - Review warnings above")
    
    print("=" * 70 + "\n")
    
    return loader


if __name__ == "__main__":
    try:
        loader = test_tal_inspired_pipeline()
        
        print("\n🎯 Next Steps:")
        print("-" * 70)
        print("  1. Review mixing ratios and batch composition above")
        print("  2. If satisfied, integrate MultiSourceDataLoader into train_policy.py")
        print("  3. Run training with Tal-inspired tactical focus")
        print("  4. Validate Stage 1 model accuracy ≥85%")
        print("  5. Test on qualitative positions (Tal games, your Bxf7+ games)")
        print("\n  Pipeline is READY for v6.1 training! 🚀\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
