#!/usr/bin/env python3
"""Analyze multi-depth evaluation patterns"""

import json

with open('data/eval_multidepth_test.json') as f:
    examples = json.load(f)

print("="*80)
print("Multi-Depth Training Examples Analysis")
print("="*80)
print()

for i, ex in enumerate(examples, 1):
    print(f"Position #{i}")
    print(f"  FEN: {ex['fen'][:60]}...")
    print(f"  Candidate moves: {ex['candidate_moves'][:3]}")
    print(f"  Correct move: {ex['correct_move']}")
    print(f"  V7P3R eval: {ex['v7p3r_eval']:+5d}cp | Stockfish eval: {ex['stockfish_eval']:+5d}cp")
    print(f"  Use V7P3R: {ex['use_v7p3r']} | Confidence: {ex['confidence']:.1f}")
    print()

# Count learning signals
from collections import Counter
signals = Counter()
for ex in examples:
    # Infer signal type from confidence and use_v7p3r
    if ex['use_v7p3r'] and ex['confidence'] == 1.0:
        signals['shallow_correct'] += 1
    elif ex['use_v7p3r'] and ex['confidence'] == 0.8:
        signals['avoid_overthinking'] += 1
    elif not ex['use_v7p3r'] and ex['confidence'] == 1.0:
        signals['correction_needed'] += 1

print("="*80)
print("Learning Signals:")
print("="*80)
for signal, count in signals.most_common():
    print(f"  {signal}: {count} ({count/len(examples)*100:.1f}%)")
