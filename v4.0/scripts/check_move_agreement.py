#!/usr/bin/env python3
"""Check if V7P3R's moves match Stockfish's recommendation"""

import json

# Read the raw position data (not the processed training examples)
with open('data/positions_multidepth_test.json') as f:
    positions = json.load(f)

print("="*80)
print("V7P3R vs Stockfish Move Agreement")
print("="*80)
print()

matches_depth2 = 0
matches_depth5 = 0
matches_depth10 = 0
in_top5_depth2 = 0
in_top5_depth5 = 0
in_top5_depth10 = 0
total = len(positions)

for i, pos in enumerate(positions, 1):
    sf_best = pos['stockfish_best']
    sf_top5 = [move for move, score in pos['stockfish_top5']]
    
    v7p3r_d2 = pos['v7p3r_depth2']['best_move'] if pos['v7p3r_depth2'] else None
    v7p3r_d5 = pos['v7p3r_depth5']['best_move'] if pos['v7p3r_depth5'] else None
    v7p3r_d10 = pos['v7p3r_depth10']['best_move'] if pos['v7p3r_depth10'] else None
    
    print(f"Position #{i}:")
    print(f"  Stockfish best: {sf_best}")
    print(f"  Stockfish top-5: {sf_top5}")
    print(f"  V7P3R depth-2:  {v7p3r_d2} {'✓ MATCH' if v7p3r_d2 == sf_best else '✗ differs'} {'(in top-5)' if v7p3r_d2 in sf_top5 else '(NOT in top-5)'}")
    print(f"  V7P3R depth-5:  {v7p3r_d5} {'✓ MATCH' if v7p3r_d5 == sf_best else '✗ differs'} {'(in top-5)' if v7p3r_d5 in sf_top5 else '(NOT in top-5)'}")
    print(f"  V7P3R depth-10: {v7p3r_d10} {'✓ MATCH' if v7p3r_d10 == sf_best else '✗ differs'} {'(in top-5)' if v7p3r_d10 in sf_top5 else '(NOT in top-5)'}")
    print()
    
    if v7p3r_d2 == sf_best:
        matches_depth2 += 1
    if v7p3r_d5 == sf_best:
        matches_depth5 += 1
    if v7p3r_d10 == sf_best:
        matches_depth10 += 1
    
    if v7p3r_d2 in sf_top5:
        in_top5_depth2 += 1
    if v7p3r_d5 in sf_top5:
        in_top5_depth5 += 1
    if v7p3r_d10 in sf_top5:
        in_top5_depth10 += 1

print("="*80)
print("Summary:")
print("="*80)
print(f"Exact match with Stockfish #1 best:")
print(f"  Depth-2:  {matches_depth2}/{total} ({matches_depth2/total*100:.1f}%)")
print(f"  Depth-5:  {matches_depth5}/{total} ({matches_depth5/total*100:.1f}%)")
print(f"  Depth-10: {matches_depth10}/{total} ({matches_depth10/total*100:.1f}%)")
print()
print(f"In Stockfish top-5:")
print(f"  Depth-2:  {in_top5_depth2}/{total} ({in_top5_depth2/total*100:.1f}%)")
print(f"  Depth-5:  {in_top5_depth5}/{total} ({in_top5_depth5/total*100:.1f}%)")
print(f"  Depth-10: {in_top5_depth10}/{total} ({in_top5_depth10/total*100:.1f}%)")
