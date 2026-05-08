#!/usr/bin/env python3
"""
Quick Start: Move Ordering Training Pipeline

Runs a quick test to verify the entire pipeline works:
1. Preprocesses a small subset of puzzles (1000) with Stockfish
2. Trains move ordering model for a few epochs
3. Validates top-k accuracy

Run this first to verify everything works before launching full 4M puzzle training.
"""

import subprocess
import sys
from pathlib import Path
import time

def run_command(cmd: str, description: str):
    """Run a command and print status"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n✅ {description} completed in {elapsed:.1f}s")
    else:
        print(f"\n❌ {description} failed!")
        sys.exit(1)
    
    return result

def main():
    print("""
╔════════════════════════════════════════════════════════════╗
║        V7P3RAI v4.0 - Move Ordering Quick Start          ║
║                                                            ║
║  This script will:                                         ║
║  1. Preprocess 1,000 puzzles with Stockfish (test)       ║
║  2. Train move ordering model (5 epochs)                  ║
║  3. Validate top-k accuracy                               ║
║                                                            ║
║  Expected runtime: ~15-30 minutes                         ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    input("Press Enter to start...")
    
    # Step 1: Preprocess puzzles (test with 1000 puzzles)
    run_command(
        "python scripts/preprocess_puzzles_with_stockfish.py "
        "--max-puzzles 1000 "
        "--rating-min 800 "
        "--rating-max 1800 "
        "--stockfish-time 0.5 "
        "--output-dir data/preprocessed_puzzles",
        "Step 1: Preprocessing 1,000 puzzles with Stockfish"
    )
    
    # Find the generated dataset
    data_dir = Path("data/preprocessed_puzzles")
    json_files = list(data_dir.glob("enriched_puzzles_compact_*.json"))
    
    if not json_files:
        print("❌ No preprocessed data found!")
        sys.exit(1)
    
    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    print(f"\n📁 Using dataset: {latest_file}")
    
    # Step 2: Test dataset loading
    run_command(
        "python src/training/puzzle_dataset.py",
        "Step 2: Testing dataset loading"
    )
    
    # Step 3: Test model
    run_command(
        "python src/models/move_ordering_network.py",
        "Step 3: Testing model architecture"
    )
    
    # Step 4: Train model (short test)
    run_command(
        f"python scripts/train_move_ordering.py "
        f"--data-path {latest_file} "
        f"--batch-size 32 "
        f"--num-epochs 5 "
        f"--learning-rate 0.001 "
        f"--checkpoint-dir models/stage1_themes/test "
        f"--num-workers 4",
        "Step 4: Training move ordering model (5 epochs)"
    )
    
    print("""
╔════════════════════════════════════════════════════════════╗
║                   ✅ QUICK START COMPLETE!                 ║
╚════════════════════════════════════════════════════════════╝

Next steps for FULL training on 4M puzzles:

1. Preprocess full dataset (will take 8-12 hours):
   
   python scripts/preprocess_puzzles_with_stockfish.py \\
     --max-puzzles 4000000 \\
     --rating-min 600 \\
     --rating-max 2500 \\
     --stockfish-time 1.0 \\
     --batch-size 1000

2. Train on full dataset (will take 2-4 days):
   
   python scripts/train_move_ordering.py \\
     --data-path data/preprocessed_puzzles/enriched_puzzles_compact_<TIMESTAMP>.json \\
     --batch-size 64 \\
     --num-epochs 100 \\
     --learning-rate 0.001 \\
     --early-stopping-patience 10 \\
     --checkpoint-dir models/stage1_themes/full

3. Validate final model:
   
   python scripts/validate_agents.py --model models/stage1_themes/full/best_checkpoint.pt

Good luck! 🚀
    """)

if __name__ == '__main__':
    main()
