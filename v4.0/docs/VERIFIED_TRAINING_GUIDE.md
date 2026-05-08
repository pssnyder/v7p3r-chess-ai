# V7P3R AI Evaluator - Verified Training System

## Overview

This system trains a neural network to mimic V7P3R's evaluation brain using **active learning** with evaluation verification:

1. **Primary Training Signal**: V7P3R's evaluations (58 feature-based system)
2. **Verification Layer**: Lichess Stockfish database (95GB, millions of positions)
3. **Active Learning**: Flag discrepancies → Fix V7P3R → Retrain
4. **Result**: AI that plays like V7P3R but faster and more robust

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Position                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
┌───────────────┐          ┌────────────────┐
│  V7P3R Engine │          │ Lichess DB     │
│  Evaluation   │          │ (Stockfish)    │
└───────┬───────┘          └────────┬───────┘
        │                           │
        │  58 Features              │  Eval Score
        │  + Eval Score             │  + Best Move
        │                           │
        └─────────────┬─────────────┘
                      │
                      ▼
            ┌─────────────────┐
            │  Verification   │
            │  System         │
            └────────┬────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
    ┌────────┐            ┌──────────┐
    │ Match  │            │ Mismatch │
    │ (Train)│            │ (Flag)   │
    └────┬───┘            └─────┬────┘
         │                      │
         ▼                      ▼
    ┌────────────┐         ┌─────────────┐
    │ AI Model   │         │ Review      │
    │ Training   │         │ → Fix V7P3R │
    └────────────┘         └──────┬──────┘
                                  │
                                  ▼
                           ┌──────────────┐
                           │ Retrain on   │
                           │ Corrections  │
                           └──────────────┘
```

## Quick Start

### 1. Build Lichess Database Index (One-Time, 10-15 minutes)

```bash
cd "e:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v4.0"

python src/data/lichess_eval_indexer.py
```

This creates a fast lookup index (~500MB) for the 95GB Lichess evaluation database.

**Note**: Only needs to be done once. Subsequent runs load the index in <1 second.

### 2. Run Verified Training

```bash
python scripts/train_verified_pipeline.py \
  --v7p3r-engine "e:\Programming Stuff\Chess Engines\V7P3R Chess Engine\v7p3r-chess-engine\lichess\engines\V7P3R_v18.3_20251229\v7p3r_uci.py" \
  --lichess-db "E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\pgn_training_data\json_data_lichess_evaluations_db\lichess_db_eval.jsonl\lichess_db_eval.jsonl" \
  --puzzle-path "data/preprocessed_puzzles/checkpoint_100000_100000.json" \
  --max-positions 10000 \
  --max-eval-diff 100 \
  --num-epochs 10 \
  --batch-size 64
```

### 3. Review Flagged Positions

After training, check flagged positions:

```bash
ls flags/eval_discrepancies/
cat flags/eval_discrepancies/flagged_training_batch.jsonl
```

Each flagged position shows:
- V7P3R's evaluation vs Stockfish's evaluation
- Eval difference in centipawns
- Reason for flagging
- V7P3R's feature breakdown (58 features)

### 4. Fix V7P3R Evaluation Bugs

Review flagged positions and identify patterns:
- Missing tactical evaluations?
- Pawn structure bugs?
- Endgame evaluation errors?

Update V7P3R engine code to fix issues, then retrain.

## Training Parameters

### Verification Thresholds

```python
# Agreement levels (centipawns difference)
PERFECT_MATCH = 10      # High confidence (1.0)
GOOD_MATCH = 50         # Medium confidence (0.7)
ACCEPTABLE = 100        # Low confidence (0.3)
SIGNIFICANT = 200       # Flag for review
MAJOR = > 200           # Definite flag
```

**Recommendation**: Start with `--max-eval-diff 100` (flag significant differences only)

### Training Workflow

1. **Initial Training**: Use V7P3R evals where Stockfish agrees (≤100cp diff)
2. **Review Flags**: Analyze positions flagged for large disagreements
3. **Fix V7P3R**: Update evaluation functions based on findings
4. **Retrain**: Run training again with improved V7P3R
5. **Iterate**: Repeat until flag rate drops below acceptable level

## File Structure

```
v4.0/
├── src/
│   ├── evaluation/
│   │   └── v7p3r_ai_evaluator.py      # 58 feature extraction
│   ├── training/
│   │   ├── v7p3r_reward_system.py     # Reward calculation
│   │   └── eval_verification_system.py # Verification & flagging
│   └── data/
│       └── lichess_eval_indexer.py    # Fast 95GB DB lookup
├── scripts/
│   ├── train_verified_pipeline.py     # Main training script
│   └── train_v7p3r_imitation.py       # Alternative (no verification)
├── docs/
│   ├── V7P3R_Evaluation_Functions_Catalog.md  # 58 extracted functions
│   └── VERIFIED_TRAINING_GUIDE.md     # This file
├── flags/
│   └── eval_discrepancies/            # Flagged positions for review
└── checkpoints/
    └── verified_training/             # Trained models
```

## Key Concepts

### V7P3R's 58 Evaluation Features

The AI learns these evaluation components from V7P3R:

**Material & Positional (3)**
- Material balance
- Piece-square tables
- PST optimization bonus

**King Safety (5)**
- Basic castling safety
- Enhanced castling evaluation
- Pawn shield
- Complex king safety
- Endgame king centralization

**Pawn Structure (5)**
- Passed pawns
- Isolated pawns
- Doubled pawns
- Backward pawns
- Pawn chains

**Piece-Specific (6)**
- Bishop pair
- Knight outposts
- Rook open files
- Rook on 7th rank
- Queen mobility
- Knight mobility

**Mobility & Control (2)**
- General piece mobility
- Center control

**Positional (6)**
- Space advantage
- Development
- Piece coordination
- Pawn majority
- Weak squares
- Strong squares

**Tactical (7)**
- Pin opportunities
- Fork opportunities
- Check threats
- Hanging pieces
- Trapped pieces
- X-ray attacks
- Discovered attacks

**Endgame (5)**
- Pawn promotion proximity
- King-pawn opposition
- Zugzwang detection
- Wrong bishop
- King activity

**Safety & Stability (4)**
- Move safety (hanging)
- Move safety (pinned)
- Move safety (tactical)
- Position stability

**Position Context (4)**
- Game phase detection
- Material balance context
- Tactical density
- Time pressure factor

**Modular System (4 - v18.3 only)**
- Evaluation profile selection
- Module activation count
- Cost efficiency
- Criticality weighting

**Bitboard Infrastructure (3)**
- Fast attack generation
- Fast mobility calculation
- Fast safety checks

**Utilities (3)**
- Tempo bonus
- Draw detection
- Mate distance

### Verification Confidence Weighting

Training samples are weighted by verification confidence:

```python
confidence = {
    'perfect_match':  1.0,  # Learn strongly from these
    'good_match':     0.7,  # Moderate learning
    'acceptable':     0.3,  # Weak learning
    'flagged':        0.0   # Excluded from training
}
```

This ensures the AI learns most from positions where V7P3R is most reliable.

### Active Learning Loop

```
1. Train on V7P3R evals (verified positions only)
2. Flag positions where V7P3R disagrees with Stockfish
3. Analyze flags to find V7P3R evaluation bugs
4. Fix bugs in V7P3R engine code
5. Retrain on corrected evaluations
6. Measure improvement (flag rate should decrease)
7. Repeat until flag rate < 5%
```

## Performance Targets

### Indexer
- Index build time: 10-15 minutes (one-time)
- Index size: ~500MB for 95GB database
- Lookup time: <1ms per position
- RAM usage: <1GB during lookup

### Training
- Verification speed: ~1000 positions/minute
- Training speed: ~500 positions/second (GPU)
- Expected flag rate: 10-20% initially → <5% after corrections

### Model
- Feature extraction: <1ms per position
- Evaluation prediction: <0.5ms per position
- Total speedup vs V7P3R: 10-100x faster

## Example Training Session

```bash
# Step 1: Build index (one-time)
python src/data/lichess_eval_indexer.py

# Output:
# Scanning lichess_db_eval.jsonl...
# This will take 10-15 minutes (one-time operation)
# Indexing: 100%|██████████| 95.0GB/95.0GB
# Indexed 12,345,678 positions
# ✓ Index built successfully
# - Index size: 524.3 MB

# Step 2: Run verified training
python scripts/train_verified_pipeline.py \
  --v7p3r-engine "path/to/v18.3/v7p3r_uci.py" \
  --lichess-db "path/to/lichess_db_eval.jsonl" \
  --puzzle-path "data/preprocessed_puzzles/checkpoint_100000_100000.json" \
  --max-positions 10000 \
  --max-eval-diff 100 \
  --num-epochs 10

# Output:
# ================================================================================
# V7P3R Verified Training Pipeline
# ================================================================================
# Device: cuda
# V7P3R Engine: v18.3
# Max Eval Difference: 100 cp
# Max Positions: 10000
#
# STEP 1: Creating Verified Training Dataset
# ✓ Index loaded successfully
#   - Positions: 12,345,678
# Verifying positions (V7P3R vs Stockfish)...
# 100%|██████████| 10000/10000 [10:23<00:00, 16.04it/s]
#
# Evaluation Verification Statistics
# ================================================================================
# Total positions verified: 10,000
#
# Agreement Levels:
#   Perfect matches (≤10cp):   6,234 (62.3%)
#   Good matches (≤50cp):      2,145 (21.4%)
#   Acceptable (≤100cp):         987 ( 9.9%)
#   Significant diff (>100cp):   512 ( 5.1%)
#   Major disagreements:         122 ( 1.2%)
#
# Flagged for Review:            634 ( 6.3%)
# Not in Lichess DB:             543 ( 5.4%)
# Database coverage:            94.6%
#
# ✓ Saved 634 flagged positions to flags/eval_discrepancies/flagged_training_batch.jsonl
#
# STEP 2: Creating Training Dataset
# Verified training dataset:
#   Total verified: 10,000
#   Eligible for training: 9,366
#   Flagged (excluded): 634
#
# Train samples: 8,429
# Val samples: 937
#
# STEP 3: Training Model
# Model parameters: 2,456,321
#
# Epoch 1/10: 100%|██████████| 132/132 [01:23<00:00]
#   Train Loss: 1245.6732
#   Val Loss: 1198.4521
#   ✓ Saved best model
#
# [... training continues ...]
#
# Epoch 10/10: 100%|██████████| 132/132 [01:21<00:00]
#   Train Loss: 423.1234
#   Val Loss: 456.7890
#
# ================================================================================
# Training Complete!
# ================================================================================
# Best validation loss: 456.7890
#
# Next Steps:
# 1. Review flagged positions in: flags/eval_discrepancies
# 2. Fix V7P3R evaluation bugs identified
# 3. Retrain on corrected positions
# 4. Integrate puzzle analysis data

# Step 3: Review flagged positions
head flags/eval_discrepancies/flagged_training_batch.jsonl

# Example flagged position:
# {
#   "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
#   "v7p3r_score": 45,
#   "v7p3r_best_move": "d2d3",
#   "lichess_score": -78,
#   "lichess_best_move": "d2d4",
#   "eval_difference": 123,
#   "agreement_level": "significant",
#   "flag_reason": "Eval difference: 123cp (V7P3R: 45, Stockfish: -78)",
#   "features": [0.12, -0.45, 0.67, ...]  # 58 features
# }

# Analysis: V7P3R sees position as equal, Stockfish sees slight Black advantage
# Possible issue: V7P3R not penalizing weak d3 pawn move enough
# Fix: Update pawn structure evaluation in v7p3r.py
```

## Integration with Puzzle Analysis

You can also integrate puzzle analysis data as mentioned:

```python
# After main training, run puzzle analysis
python scripts/puzzle_analyzer.py \
  --engine v7p3r \
  --puzzle-file puzzles.json \
  --output puzzle_analysis.json

# Feed puzzle performance data into training
# (Future enhancement - not yet implemented)
```

## Questions?

Common questions about this system:

**Q: Why use V7P3R evals instead of just training on Stockfish?**
A: We want the AI to preserve V7P3R's unique personality/playing style. Training directly on Stockfish would lose V7P3R's character.

**Q: What's the point of verification if we use V7P3R's evals anyway?**
A: Verification helps us find bugs in V7P3R's evaluation logic. When we find discrepancies, we can fix V7P3R itself, making it stronger.

**Q: How often should I review flagged positions?**
A: After each training run. Look for patterns in the flags - are certain position types consistently wrong?

**Q: Can I adjust the flagging threshold?**
A: Yes! Use `--max-eval-diff` parameter. Lower = more strict (more flags), higher = more lenient (fewer flags).

**Q: How long does the first run take?**
A: Index build: 10-15min (one-time). Training: ~30min for 10K positions + 10 epochs (with GPU).

**Q: Do I need the full 95GB database?**
A: For best coverage, yes. But you can test with a smaller subset initially.
