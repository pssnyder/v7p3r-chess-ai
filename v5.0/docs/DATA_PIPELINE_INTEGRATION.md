# V5.0 Data Pipeline Integration
**How BigQuery + Puzzles + Self-Play → Unified Training Dataset**

---

## 🎯 Three Sources, One Format

All three data sources feed into the **identical** unified training format defined in `UNIFIED_TRAINING_DATASET.md`:

```
┌─────────────────────────────────────────────────────────┐
│              STAGE 1: Data Collection                    │
│         (Each produces unified format)                   │
└──────────────┬──────────────────────────────────────────┘
               │
     ┌─────────┼─────────┐
     │         │         │
┌────▼─────┐ ┌▼─────┐ ┌─▼──────┐
│ BigQuery │ │Puzzle│ │Self-   │
│ Extract  │ │Solver│ │Play    │
│ (PGNs)   │ │      │ │        │
└────┬─────┘ └┬─────┘ └─┬──────┘
     │        │         │
     │ 492k   │ 15k     │ 5k positions
     │ moves  │ attempts│ games
     └────────┴─────────┘
              │
    ┌─────────▼──────────┐
    │ raw/*.jsonl        │
    │ (Unified format)   │
    └─────────┬──────────┘
              │
┌─────────────▼───────────────────────────────────────────┐
│         STAGE 2: Stockfish Post-Analysis                 │
│  (Same analysis for all sources)                         │
└─────────────┬───────────────────────────────────────────┘
              │
    ┌─────────▼──────────┐
    │ analyzed/*.jsonl   │
    │ + top_5_moves      │
    │ + quality grading  │
    └─────────┬──────────┘
              │
┌─────────────▼───────────────────────────────────────────┐
│         STAGE 3: Feature Extraction                      │
│  (Same features for all sources)                         │
└─────────────┬───────────────────────────────────────────┘
              │
    ┌─────────▼──────────────┐
    │ training/              │
    │ unified_training       │
    │ _dataset.jsonl         │
    └────────────────────────┘
```

---

## 📊 BigQuery → Unified Format Mapping

### Source Tables

**conformed_layer.game_data** (5,069 games)
- Game metadata (opponent, ELO, time control, result)
- Engine version, game type
- Used for filtering and context

**conformed_layer.moves** (1,350,163 moves)
- Move details (UCI, SAN, FEN before/after)
- Game phase, material balance
- Capture/check flags
- V7P3R evaluation (if available)

### Field Mapping

#### BigQuery → Metadata Block
```python
BigQuery Field                    →  Unified Field
─────────────────────────────────────────────────────────
game_id                          →  metadata.source_details.game_id
white, black                     →  metadata.source_details.white/black
result                           →  metadata.source_details.result
time_control                     →  metadata.source_details.time_control
date                             →  metadata.source_details.date
engine_version                   →  metadata.v7p3r_version
```

#### BigQuery → Position Block
```python
BigQuery Field                    →  Unified Field
─────────────────────────────────────────────────────────
fen_before                       →  position.fen
move_number                      →  position.move_number
game_phase                       →  position.game_phase
material_balance                 →  position.material.balance
piece_count                      →  (used to calculate phase_score)

# Calculated from FEN:
board.is_check()                 →  position.tactical_state.in_check
board.has_castling_rights()      →  position.tactical_state.*_can_castle_*
board.legal_moves.count()        →  position.tactical_state.num_legal_moves
```

#### BigQuery → Engine Decision Block
```python
BigQuery Field                    →  Unified Field
─────────────────────────────────────────────────────────
move_uci                         →  engine_decision.move_uci
move_san                         →  engine_decision.move_san
v7p3r_eval_cp                    →  engine_decision.evaluation.total_cp
is_capture                       →  engine_decision.move_type.is_capture
is_check                         →  engine_decision.move_type.is_check
is_castle                        →  engine_decision.move_type.is_castling
piece                            →  engine_decision.move_type.piece_moved

# NOT AVAILABLE in BigQuery (search stats):
depth_reached                    →  engine_decision.search.* = None
nodes_searched                   →  (historical data doesn't have this)
tt_hits, cache_hits, etc.        →  (only live profiling has this)
```

### Data Quality Notes

**✅ Available in BigQuery:**
- Position context (FEN, game phase, material)
- Move details (UCI, SAN, captures, checks)
- Game metadata (opponent, ELO, time control)
- Some V7P3R evaluations (v7p3r_eval_cp)

**❌ NOT Available in BigQuery:**
- Search statistics (depth, nodes, time)
- Evaluation breakdown (material vs PST vs strategic)
- TT/cache/killer hits
- PV line

**📝 Implication:**
- Historical BigQuery data provides **position + move** context
- Live profiling (v18.3.1) will provide **search + eval breakdown**
- Both feed into same unified format with different fields populated

---

## 🔄 Data Source Comparison

| Feature | BigQuery (Historical) | Puzzle Solver | Self-Play | Live Profiling |
|---------|----------------------|---------------|-----------|----------------|
| **Position (FEN)** | ✅ | ✅ | ✅ | ✅ |
| **Move Played** | ✅ | ✅ | ✅ | ✅ |
| **Game Phase** | ✅ | ✅ | ✅ | ✅ |
| **Material Balance** | ✅ | ✅ | ✅ | ✅ |
| **Eval Total** | ⚠️ Partial | ✅ | ✅ | ✅ |
| **Eval Breakdown** | ❌ | ❌ | ✅ | ✅ |
| **Search Stats** | ❌ | ❌ | ✅ | ✅ |
| **PV Line** | ❌ | ❌ | ✅ | ✅ |
| **Stockfish Top-5** | 🔄 Stage 2 | 🔄 Stage 2 | 🔄 Stage 2 | 🔄 Stage 2 |
| **Volume** | 492k moves | 15k puzzles | 5k games | Future data |

**Key Insight**: All sources provide **position + move**. Only live data provides **search details**. All get **Stockfish analysis** in Stage 2.

---

## 📂 File Organization

```
v5.0/
├── data/
│   ├── raw/                                    # Stage 1: Raw collection
│   │   ├── pgn_extractions/
│   │   │   ├── bigquery_records_20260506.jsonl      # ← BigQuery extract
│   │   │   └── local_pgn_records_20260507.jsonl     # ← Local PGN files
│   │   ├── puzzle_training/
│   │   │   └── puzzle_attempts_20260506.jsonl       # ← Puzzle solver
│   │   └── selfplay/
│   │       └── selfplay_episode_042.jsonl           # ← Self-play games
│   │
│   ├── analyzed/                               # Stage 2: + Stockfish
│   │   ├── bigquery_analyzed_20260506.jsonl
│   │   ├── puzzle_analyzed_20260506.jsonl
│   │   └── selfplay_analyzed_20260506.jsonl
│   │
│   ├── training/                               # Stage 3: + Features
│   │   ├── unified_training_dataset.jsonl          # ← FINAL MERGED
│   │   ├── train_split.jsonl                       # 80% for training
│   │   ├── val_split.jsonl                         # 10% for validation
│   │   └── test_split.jsonl                        # 10% for testing
│   │
│   └── metadata/
│       ├── dataset_stats.json                      # Quality metrics
│       ├── source_breakdown.json                   # Records per source
│       └── feature_distributions.json              # Feature statistics
│
└── scripts/
    ├── extract_from_bigquery.py                    # ✅ Created
    ├── extract_from_local_pgns.py                  # 🔄 TODO
    ├── collect_puzzle_data.py                      # 🔄 TODO
    ├── run_selfplay.py                             # 🔄 TODO
    ├── analyze_with_stockfish.py                   # 🔄 TODO (universal)
    ├── extract_features.py                         # 🔄 TODO (universal)
    └── merge_datasets.py                           # 🔄 TODO (final step)
```

---

## 🚀 Execution Workflow

### Step 1: Extract from All Sources

**BigQuery (Historical Games)**
```bash
# With service account credentials
python scripts/extract_from_bigquery.py \
    --credentials /path/to/service-account-key.json \
    --min-elo 1200 \
    --game-types lichess_rated lichess_casual tournament

# Or with gcloud auth (Application Default Credentials)
gcloud auth application-default login
python scripts/extract_from_bigquery.py --min-elo 1200

# Expected output:
# → v5.0/data/raw/pgn_extractions/bigquery_records_20260506.jsonl
# → ~492k records
```

**Puzzle Training** (Future)
```bash
python scripts/collect_puzzle_data.py \
    --puzzle-db lichess_puzzles.db \
    --max-puzzles 15000 \
    --min-rating 1400

# → v5.0/data/raw/puzzle_training/puzzle_attempts_20260506.jsonl
```

**Self-Play** (Future)
```bash
python scripts/run_selfplay.py \
    --episodes 100 \
    --model-version v5.0_epoch_0 \
    --games-per-episode 50

# → v5.0/data/raw/selfplay/selfplay_episode_042.jsonl
```

---

### Step 2: Stockfish Analysis (Universal)

```bash
# Analyze ALL raw sources with same script
python scripts/analyze_with_stockfish.py \
    --input data/raw/pgn_extractions/bigquery_records_20260506.jsonl \
    --output data/analyzed/bigquery_analyzed_20260506.jsonl \
    --depth 20 \
    --multipv 5 \
    --parallel 8

python scripts/analyze_with_stockfish.py \
    --input data/raw/puzzle_training/puzzle_attempts_20260506.jsonl \
    --output data/analyzed/puzzle_analyzed_20260506.jsonl \
    --depth 20 \
    --multipv 5 \
    --parallel 8

# Same analysis for all sources!
```

**Stockfish Adds:**
- `stockfish_analysis.top_5_moves` - Best 5 moves with evals
- `stockfish_analysis.engine_move_evaluation` - Quality grade
- `stockfish_analysis.position_evaluation` - Win probabilities

---

### Step 3: Feature Extraction (Universal)

```bash
# Extract features from ALL analyzed sources
python scripts/extract_features.py \
    --input data/analyzed/bigquery_analyzed_20260506.jsonl \
    --output data/training/bigquery_features_20260506.jsonl

python scripts/extract_features.py \
    --input data/analyzed/puzzle_analyzed_20260506.jsonl \
    --output data/training/puzzle_features_20260506.jsonl

# Same feature extraction for all sources!
```

**Feature Extraction Adds:**
- `features.board_tensor` - 8x8x12 numpy array (base64 encoded)
- `features.material_balance_normalized` - -1 to +1
- `features.pst_scores` - PST advantage
- `features.strategic_features` - V7P3R heuristics
- `features.game_phase_vector` - One-hot encoding

---

### Step 4: Merge Into Final Dataset

```bash
python scripts/merge_datasets.py \
    --inputs \
        data/training/bigquery_features_20260506.jsonl \
        data/training/puzzle_features_20260506.jsonl \
        data/training/selfplay_features_20260506.jsonl \
    --output data/training/unified_training_dataset.jsonl \
    --train-split 0.8 \
    --val-split 0.1 \
    --test-split 0.1

# Output:
# → unified_training_dataset.jsonl (all 512k records)
# → train_split.jsonl (410k records, 80%)
# → val_split.jsonl (51k records, 10%)
# → test_split.jsonl (51k records, 10%)
```

---

## 📊 Expected Dataset Composition

| Source | Records | % of Total | Quality Grade |
|--------|---------|-----------|---------------|
| BigQuery (rated games) | 390,000 | 76% | ★★★★☆ High (competitive) |
| BigQuery (casual games) | 90,000 | 18% | ★★★☆☆ Medium |
| Puzzle Training | 15,000 | 3% | ★★★★★ Excellent (tactical) |
| Self-Play | 15,000 | 3% | ★★★☆☆ Medium (exploration) |
| **TOTAL** | **510,000** | **100%** | **Mixed Quality** |

**Quality Distribution (After Stockfish Analysis):**
- Excellent moves (top-1): ~36% (184k)
- Good moves (top-3): ~44% (224k)
- Inaccuracies (top-5): ~14% (71k)
- Mistakes: ~5% (26k)
- Blunders: ~1% (5k)

---

## 🎯 Training Strategy

### Supervised Learning (Policy + Value Heads)
```python
# Use Stockfish top-5 for policy head targets
policy_target = softmax([move1_cp, move2_cp, move3_cp, move4_cp, move5_cp])

# Use Stockfish eval for value head targets
value_target = tanh(stockfish_best_eval_cp / 100.0)
```

### Reinforcement Learning (Move Quality Rewards)
```python
# Reward based on Stockfish quality grade
rewards = {
    "excellent": +1.0,   # Top move
    "good": +0.5,        # Top-3
    "inaccuracy": -0.2,  # Top-5
    "mistake": -0.5,     # Eval loss 50-100cp
    "blunder": -1.0      # Eval loss >100cp
}

# Plus V7P3R personality bonus
if move_matches_v7p3r_style(move, position):
    reward += 0.3  # Encourage V7P3R-like play
```

---

## ✅ Current Status

**Completed:**
- ✅ Unified training dataset schema designed
- ✅ BigQuery extraction script created
- ✅ Field mapping documented

**In Progress:**
- 🔄 BigQuery authentication setup
- 🔄 Test extraction with 100 records

**TODO:**
- ⏳ Stockfish analysis script (universal for all sources)
- ⏳ Feature extraction script
- ⏳ Dataset merger script
- ⏳ Puzzle solver data collection
- ⏳ Self-play engine implementation

---

## 🎯 Next Immediate Steps

1. **Test BigQuery Extraction**
   ```bash
   python scripts/extract_from_bigquery.py --limit 100
   ```

2. **Build Stockfish Analyzer**
   - Universal script for all sources
   - Batch processing with parallel workers
   - Progress tracking and resume capability

3. **Validate Data Quality**
   - Check unified format compliance
   - Verify Stockfish analysis quality
   - Inspect feature distributions

4. **Start Training**
   - Once 50k+ records analyzed
   - Begin with supervised learning
   - Add RL training later

Ready to collect 500k+ training examples! 🚀
