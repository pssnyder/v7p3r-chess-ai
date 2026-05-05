#!/usr/bin/env python3
"""Quick analysis of V7P3R vs Stockfish eval differences"""

import json

with open('data/corrections_eval_training_v18.3_depth10.json') as f:
    corrections = json.load(f)

# Sort by eval difference
corrections.sort(key=lambda x: x['eval_difference'], reverse=True)

print("="*80)
print("V7P3R v18.3 (depth 10) vs Stockfish (0.5s) - Top 10 Biggest Disagreements")
print("="*80)
print()

for i, d in enumerate(corrections[:10], 1):
    print(f"#{i} - Eval Difference: {d['eval_difference']}cp")
    print(f"   V7P3R eval: {d['v7p3r_eval']:+5d}cp | Stockfish eval: {d['stockfish_eval']:+5d}cp")
    print(f"   V7P3R played: {d['v7p3r_played']:6s} | Stockfish best: {d['stockfish_best']:6s}")
    print(f"   Stockfish top-5: {[m for m, s in d['stockfish_top5'][:5]]}")
    print(f"   FEN: {d['fen'][:70]}")
    print()

# Statistics
print("="*80)
print("Disagreement Categories:")
print("="*80)
huge_diff = [d for d in corrections if d['eval_difference'] > 300]
large_diff = [d for d in corrections if 200 <= d['eval_difference'] <= 300]
med_diff = [d for d in corrections if 100 <= d['eval_difference'] < 200]

print(f"Huge (>300cp):     {len(huge_diff):3d} ({len(huge_diff)/len(corrections)*100:.1f}%)")
print(f"Large (200-300cp): {len(large_diff):3d} ({len(large_diff)/len(corrections)*100:.1f}%)")
print(f"Medium (100-200cp):{len(med_diff):3d} ({len(med_diff)/len(corrections)*100:.1f}%)")
print()

# Check if V7P3R's moves are in Stockfish top-5
v7p3r_in_top5 = sum(1 for d in corrections if d['v7p3r_played'] in [m for m, s in d['stockfish_top5']])
print(f"V7P3R's move in Stockfish top-5: {v7p3r_in_top5}/{len(corrections)} ({v7p3r_in_top5/len(corrections)*100:.1f}%)")
print()

# Average eval difference
avg_diff = sum(d['eval_difference'] for d in corrections) / len(corrections)
print(f"Average eval difference: {avg_diff:.1f}cp")
