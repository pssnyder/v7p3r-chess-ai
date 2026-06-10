"""
Test the massive bad position miner on a small sample.

This tests:
1. PGN parsing works
2. Eval extraction works
3. Fast feature extraction works
4. JSONL output format correct
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.stage1.mine_bad_positions_massive import mine_directory

# Test on one month of engine battles
test_directory = Path("E:/Programming Stuff/Chess Engines/Chess Engine Playground/engine-metrics/raw_data/game_records/Engine Battle 202512")
output_file = Path(__file__).parent.parent.parent / "data" / "stage1" / "bad_positions_test.jsonl"

# Mine just 100 positions as a test
print("Testing bad position miner on Engine Battle 202512...")
mine_directory(test_directory, output_file, target_positions=100)

# Verify output
if output_file.exists():
    import json
    print("\n✅ Test successful! Sample records:")
    with open(output_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            record = json.loads(line)
            print(f"\n   Record {i+1}:")
            print(f"      FEN: {record['fen'][:50]}...")
            print(f"      Label: {record['label']} (BAD)")
            print(f"      Features: {len(record['features'])} dims")
            print(f"      Weight: {record['weight']}")
            print(f"      Eval drop: {record['eval_drop_cp']:.0f}cp")
            print(f"      Game: {record['game_info']['white']} vs {record['game_info']['black']}")
else:
    print("\n❌ Test failed - no output file created")
