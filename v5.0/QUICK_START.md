# V5.0 Quick Start Guide
**Getting Started with Unified Training Dataset Collection**

---

## 🎯 Goal

Build a **unified training dataset** from three sources:
1. **BigQuery historical games** (492k v7p3r moves)
2. **Puzzle training** (15k tactical positions)
3. **Self-play** (5k exploration games)

All sources produce **identical data format** → analyzed by Stockfish → used for AI training.

---

## 📚 Key Documents

| Document | Purpose |
|----------|---------|
| [UNIFIED_TRAINING_DATASET.md](UNIFIED_TRAINING_DATASET.md) | Complete schema definition (5 blocks: metadata, position, engine_decision, stockfish_analysis, features) |
| [DATA_PIPELINE_INTEGRATION.md](docs/DATA_PIPELINE_INTEGRATION.md) | How all 3 sources integrate, field mappings, execution workflow |
| [V7P3R_AI_Data_Preparation.md](docs/V7P3R_AI_Data_Preparation.md) | BigQuery database schema, connection details, query examples |

---

## 🗄️ BigQuery Data Available

**Project**: `chess-engine-metrics-agent`

**Tables:**
- `conformed_layer.game_data` - 5,069 unique games with metadata
- `conformed_layer.moves` - 1,350,163 moves (492k v7p3r moves after filtering)

**Connection:**
```bash
# Authenticate
gcloud auth application-default login

# Test connection
python scripts/extract_from_bigquery.py --limit 100
```

**What BigQuery Provides:**
- ✅ Position (FEN, game phase, material)
- ✅ Move details (UCI, SAN, captures, checks)
- ✅ Game context (opponent, ELO, time control)
- ⚠️ Partial eval (some games have v7p3r_eval_cp)
- ❌ Search stats (not recorded in historical data)

**What's Missing (added in other stages):**
- Stockfish analysis → Stage 2
- Move quality grading → Stage 2
- Feature tensors → Stage 3

---

## 🚀 Data Collection Pipeline

### **Stage 1: Raw Collection**

Extract data from each source into unified format:

```bash
# Source 1: BigQuery (Historical)
python scripts/extract_from_bigquery.py \
    --min-elo 1200 \
    --game-types lichess_rated lichess_casual tournament
# → data/raw/pgn_extractions/bigquery_records_20260506.jsonl
# → ~492k records

# Source 2: Puzzles (TODO)
python scripts/collect_puzzle_data.py \
    --max-puzzles 15000 \
    --min-rating 1400
# → data/raw/puzzle_training/puzzle_attempts_20260506.jsonl
# → ~15k records

# Source 3: Self-Play (TODO)
python scripts/run_selfplay.py \
    --episodes 100 \
    --games-per-episode 50
# → data/raw/selfplay/selfplay_episode_042.jsonl
# → ~5k records
```

### **Stage 2: Stockfish Analysis**

Analyze ALL sources with same script:

```bash
python scripts/analyze_with_stockfish.py \
    --input data/raw/pgn_extractions/bigquery_records_20260506.jsonl \
    --output data/analyzed/bigquery_analyzed_20260506.jsonl \
    --depth 20 \
    --multipv 5 \
    --parallel 8

# Repeat for puzzle and selfplay data
```

**Stockfish Adds:**
- Top 5 best moves with evaluations
- Engine move quality grade (excellent/good/inaccuracy/mistake/blunder)
- Position win probabilities

### **Stage 3: Feature Extraction**

Extract features for neural network:

```bash
python scripts/extract_features.py \
    --input data/analyzed/bigquery_analyzed_20260506.jsonl \
    --output data/training/bigquery_features_20260506.jsonl

# Repeat for all sources
```

**Features Added:**
- Board tensor (8x8x12 numpy array)
- Material balance normalized (-1 to +1)
- PST scores
- Strategic features (V7P3R heuristics)
- Game phase one-hot encoding

### **Stage 4: Merge Dataset**

Combine all sources:

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
```

**Final Outputs:**
- `unified_training_dataset.jsonl` - All 512k records
- `train_split.jsonl` - 410k records (80%)
- `val_split.jsonl` - 51k records (10%)
- `test_split.jsonl` - 51k records (10%)

---

## 📁 Directory Structure

```
v5.0/
├── README.md                           # V5.0 project overview
├── QUICK_START.md                      # ← YOU ARE HERE
├── UNIFIED_TRAINING_DATASET.md         # Schema definition
│
├── docs/
│   ├── DATA_PIPELINE_INTEGRATION.md    # Pipeline integration guide
│   └── V7P3R_AI_Data_Preparation.md    # BigQuery details
│
├── data/
│   ├── raw/                            # Stage 1 output
│   │   ├── pgn_extractions/
│   │   ├── puzzle_training/
│   │   └── selfplay/
│   ├── analyzed/                       # Stage 2 output
│   ├── training/                       # Stage 3 + 4 output
│   └── metadata/                       # Quality metrics
│
├── scripts/
│   ├── extract_from_bigquery.py        # ✅ Created (Stage 1)
│   ├── extract_from_local_pgns.py      # 🔄 TODO
│   ├── collect_puzzle_data.py          # 🔄 TODO
│   ├── run_selfplay.py                 # 🔄 TODO
│   ├── analyze_with_stockfish.py       # 🔄 TODO (Stage 2)
│   ├── extract_features.py             # 🔄 TODO (Stage 3)
│   └── merge_datasets.py               # 🔄 TODO (Stage 4)
│
└── static_engines/
    └── V7P3R_v18.3.1/                  # Simple profiling engine
        └── src/
            ├── v7p3r_engine.py
            ├── v7p3r_evaluators.py     # Fast eval only (no modularity)
            └── v7p3r_profiler.py       # Data logger
```

---

## ✅ Current Status

**Completed:**
- ✅ Unified training dataset schema designed
- ✅ BigQuery extraction script created
- ✅ V18.3.1 consolidation (3 files, fast eval only)
- ✅ Documentation complete

**Next Steps:**
1. **Test BigQuery extraction**
   ```bash
   python scripts/extract_from_bigquery.py --limit 100
   ```

2. **Build Stockfish analyzer** (universal for all sources)

3. **Create feature extractor**

4. **Validate with small dataset** (1,000 records end-to-end)

5. **Scale to full dataset** (500k records)

---

## 🎯 Training Targets

Once dataset is ready:

### **Supervised Learning**
```python
# Policy head: Learn move probabilities from Stockfish top-5
policy_target = softmax(stockfish_top_5_evals / temperature)

# Value head: Learn position evaluation from Stockfish
value_target = tanh(stockfish_best_eval_cp / 100.0)
```

### **Reinforcement Learning**
```python
# Reward based on move quality
reward = {
    "excellent": +1.0,   # Played top move
    "good": +0.5,        # Top-3
    "inaccuracy": -0.2,  # Top-5
    "mistake": -0.5,
    "blunder": -1.0
}

# Plus V7P3R personality bonus
if move_matches_v7p3r_style:
    reward += 0.3
```

---

## 💡 Key Insights

1. **One Format, Three Sources**: All data sources produce identical schema
2. **BigQuery = Historical Data**: 492k moves from past games
3. **Stockfish = Ground Truth**: Provides move quality grading
4. **V7P3R Personality = Reward Shaping**: Favor engine's style when quality acceptable
5. **Mixed Quality OK**: Train on mistakes too (with corrective targets)

---

## 🚀 Getting Started Now

```bash
# 1. Authenticate with BigQuery
gcloud auth application-default login

# 2. Test extraction (100 records)
cd v5.0
python scripts/extract_from_bigquery.py --limit 100

# 3. Inspect output
head -1 data/raw/pgn_extractions/bigquery_records_*.jsonl | python -m json.tool

# 4. Verify unified format compliance
# Check: metadata, position, engine_decision blocks present
```

**Expected output**: JSONL file with records matching `UNIFIED_TRAINING_DATASET.md` schema.

---

## 📊 Dataset Quality Goals

| Metric | Target |
|--------|--------|
| Total positions | 500,000+ |
| Excellent moves (top-1) | 36% |
| Good moves (top-3) | 44% |
| Inaccuracies (top-5) | 14% |
| Mistakes | 5% |
| Blunders | 1% |
| Source distribution | 75% BigQuery, 15% puzzles, 10% self-play |
| Phase distribution | 25% opening, 50% middlegame, 25% endgame |

---

Ready to collect training data! 🎯
