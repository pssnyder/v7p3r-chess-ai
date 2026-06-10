# V7P3R AI v6.0 - Current Status

**Last Updated:** Phase 1 Data Preparation (In Progress)

## ✅ Completed

### Setup & Infrastructure
- [x] v6.0 directory structure created
- [x] Zobrist hashing utility (`zobrist_hashing.py`)
- [x] Data filtering script (`filter_dataset.py`)
- [x] Transposition graph builder (`build_graph.py`)
- [x] Feature extraction utilities (copied from v5.0)
- [x] Configuration file (`stage1_config.yaml`)
- [x] Documentation (`README.md`, `IMPLEMENTATION_PLAN.md`)
- [x] Utility scripts (`quick_start_data_prep.ps1`, `check_progress.ps1`)

### Phase 1a: Data Filtering ✅ COMPLETE
**Script:** `filter_dataset.py`  
**Runtime:** ~1 hour  
**Status:** ✅ Successfully completed

**Input:**
- v5.3 merged dataset: 6,313,414 positions (23.7 GB)

**Output:**
- `good_positions.jsonl`: 5,719,272 positions (22.68 GB)
- `bad_positions.jsonl`: 69,240 positions (248.78 MB)

**Key Metrics:**
- Imbalance ratio: 82.6:1 (good:bad)
- C0BR4 excluded: 492,654 positions (failed Stockfish analysis)
- Parsing errors: 0
- Grade 1 handling: 32,248 excluded (no eval data for ≤50cp filtering)

**Source Breakdown:**
- Lichess puzzles: 5,622,293 (all Grade 0 - optimal moves)
- V7P3R games: 181,467 positions
  - Good (G0): 96,979
  - Bad (G2-G5): 52,240
  - Excluded (G1 no eval): 32,248

**Quality:**
- Zero parsing errors on 6.3M records
- Clean binary classification achieved
- V7P3R personality data preserved (181k positions)

## 🚧 In Progress

### Phase 1b: Transposition Graph Building
**Script:** `build_graph.py`  
**Started:** Just now  
**Status:** 🚧 Running (Phase 1: Indexing)

**Current Progress:**
- Indexed: 100,000 records
- Unique positions: 99,909
- Duplicates detected: 91 (via Zobrist hashing)

**Process:**
1. **Phase 1:** Index all 5.7M positions by Zobrist hash → Detect transpositions
2. **Phase 2:** Compute tactical features for each unique position
3. **Phase 3:** Find K=10 nearest neighbors via tactical similarity
4. **Phase 4:** Build graph adjacency list
5. **Phase 5:** Save to `transposition_graph.pkl`

**Expected Output:**
- Unique positions: ~5.6M (expect ~100k transpositions in 5.7M records)
- Graph edges: ~56M (10 neighbors × 5.6M nodes)
- File size: ~500 MB (pickled adjacency list)
- Similarity metric: Shared tactical features (hanging pieces, pins, forks, king attacks, passed pawns)

**ETA:** 2-3 hours (brute-force K-NN is O(n²))

**Optimization Note:**
Production version should use FAISS or Annoy for approximate nearest neighbors (O(n log n) instead of O(n²)). Current implementation prioritizes correctness over speed for initial v6.0.

## 📋 Next Steps

### Phase 2: Stage 1 Training Implementation (2-3 days)
**Goal:** Implement graph-augmented neural network for binary classification

**Tasks:**
1. Create `train_policy.py` in `scripts/stage1/`
2. Implement architecture:
   - Input: 325D position features + transposition embeddings
   - Embedding layer: 325 → 512
   - Hidden layers: 512 → 256 → 128
   - Transposition attention: Attend to K=10 neighbor embeddings
   - Output: Binary classification (sigmoid activation)
3. Implement loss function:
   - Weighted BCE: good=0.006, bad=1.0 (handle 82.6:1 imbalance)
   - Graph regularization: L2(prediction_i - avg(predictions_neighbors))
   - Total: α * BCE + β * graph_reg (α=1.0, β=0.1)
4. Implement data pipeline:
   - Load graph + position features
   - Create TensorFlow Dataset with neighbor lookup
   - Train/val/test split: 80/10/10
5. Implement validation metrics:
   - Standard: accuracy, precision, recall, F1
   - Transposition consistency: Correlation between similar positions
   - V7P3R style matching: Agreement on V7P3R game moves
6. Train model:
   - Batch size: 2048
   - Epochs: 100 (early stopping patience=10)
   - Learning rate: 0.001 (Adam optimizer)
   - Expected runtime: 8-12 hours on GPU

**Performance Targets:**
- Accuracy: 95%+ (binary easier than 6-class)
- Bad recall: 60%+ (catch tactical blunders)
- Transposition consistency: 0.8+ correlation
- V7P3R style match: 60%+ agreement

### Phase 3: Stage 2 Self-Play Implementation (3-5 days)
**Goal:** Expand knowledge beyond training set via reinforcement learning

**Tasks:**
1. Create `self_play.py` in `scripts/stage2/`
2. Implement self-play framework:
   - Generate starting positions from opening book
   - Apply policy network (Stage 1) + epsilon-greedy (ε=0.2)
   - Play out games to random depth (10-30 moves)
3. Implement Stockfish feedback loop:
   - Evaluate new positions (0.5s analysis)
   - Compare AI intuition vs Stockfish evaluation
   - If disagreement >100cp: Add to training set
4. Implement graph expansion:
   - Compute Zobrist hash for new positions
   - Find K=10 nearest neighbors in existing graph
   - Add new nodes + edges to transposition graph
5. Implement incremental learning:
   - Collect batches of 10k new positions
   - Retrain Stage 1 policy with expanded dataset
   - Validate improvement via hold-out test set
6. Run self-play:
   - Target: 1000+ games
   - Expected discoveries: 200-300k new positions
   - Runtime: 2-3 days

**Performance Targets:**
- Dataset growth: 5.7M → 6M+ positions (+5% coverage)
- Graph density: +10% edges (new connections discovered)
- Performance: +5-10% improvement over Stage 1 baseline
- Coverage: Fill gaps in opening/endgame positions

### Phase 4: Evaluation & Testing (1-2 days)
**Goal:** Validate v6.0 against v5.0 and document performance

**Tasks:**
1. Benchmark against v5.0:
   - Test on hold-out puzzle set (10k positions)
   - Compare: accuracy, precision, recall, F1
   - Analyze: where v6.0 outperforms v5.0
2. V7P3R style validation:
   - Test on V7P3R game positions
   - Measure agreement with V7P3R actual moves
   - Verify personality preservation
3. Transposition analysis:
   - Test prediction consistency on transpositions
   - Measure correlation between similar positions
   - Validate graph regularization benefit
4. Generate performance report:
   - Comprehensive metrics comparison
   - Visualization: confusion matrices, ROC curves
   - Documentation: v6.0 improvements over v5.0

## 📊 Performance Comparison (Projected)

| Metric | v5.0 Multi-Class | v6.0 Binary + Graph | Improvement |
|--------|------------------|---------------------|-------------|
| **Task** | 6-grade classification | Binary (good/bad) | Simplified |
| **Accuracy** | ~85% (multi-class) | ~95% (binary) | +10% |
| **Bad recall** | ~40% | ~60% | +20% |
| **Training data** | 6.3M positions | 5.7M curated | Quality over quantity |
| **Architecture** | Standard NN | Graph-augmented NN | Transposition modeling |
| **Learning** | Supervised only | Supervised + RL | Knowledge expansion |
| **Style preservation** | Not validated | V7P3R style check | Personality maintained |

## 💡 Key Innovations

### 1. Binary Classification
**Rationale:** Chess move quality is inherently binary at practical level
- **Good moves:** Near-optimal (eval loss ≤50cp)
- **Bad moves:** Tactical/positional mistakes (eval loss >50cp)

**Benefits:**
- Simpler task → higher accuracy
- Aligns with engine needs (avoid bad moves, not rank good moves)
- 82.6:1 imbalance reflects reality (blunders rare in puzzles)

### 2. Transposition Network
**Rationale:** Similar positions should inform each other during learning

**Implementation:**
- Zobrist hashing for position identity
- Tactical feature similarity for edge creation
- K=10 nearest neighbors per position
- Graph regularization in loss function

**Benefits:**
- Generalization: Learn patterns, not memorize positions
- Consistency: Similar positions receive similar predictions
- Attention mechanism: Neighbor embeddings provide context
- Inspired by AlphaZero's MCTS value sharing

### 3. Two-Stage Learning
**Rationale:** Combine human expertise with machine exploration

**Stage 1:** Supervised learning from 5.7M curated positions
- Learn from human puzzle expertise (Lichess)
- Preserve V7P3R playing style
- Build strong binary classification baseline

**Stage 2:** Self-play reinforcement learning
- Generate new positions via epsilon-greedy
- Stockfish feedback corrects AI mistakes
- Expand transposition graph with discoveries
- Fill gaps in opening/endgame knowledge

**Benefits:**
- Best of both worlds: Human intuition + machine exploration
- Continuous improvement via feedback loop
- Dataset grows with experience
- Discovers patterns beyond static training set

## 🎯 Success Criteria

### Phase 1 (Data Preparation) - ✅ 50% Complete
- [x] Filter dataset: 6.3M → 5.7M good + 69k bad
- [🚧] Build transposition graph: ~5.6M nodes, ~56M edges

### Phase 2 (Stage 1 Training)
- [ ] Implement graph NN architecture
- [ ] Train binary classifier (8-12 hours GPU)
- [ ] Achieve 95%+ accuracy on validation set
- [ ] Validate transposition consistency (0.8+ correlation)
- [ ] Verify V7P3R style match (60%+ agreement)

### Phase 3 (Stage 2 Self-Play)
- [ ] Implement self-play framework
- [ ] Run 1000+ games with Stockfish feedback
- [ ] Discover 200-300k new positions
- [ ] Expand transposition graph (+10% edges)
- [ ] Improve performance +5-10% over Stage 1

### Phase 4 (Evaluation)
- [ ] Benchmark vs v5.0 on hold-out test set
- [ ] Document performance improvements
- [ ] Validate V7P3R personality preservation
- [ ] Generate comprehensive performance report

## 📈 Timeline

- ✅ **Setup & Infrastructure:** 1 day (COMPLETE)
- 🚧 **Phase 1a - Data Filtering:** 1 hour (COMPLETE)
- 🚧 **Phase 1b - Graph Building:** 2-3 hours (IN PROGRESS - 1% done)
- **Phase 2 - Stage 1 Implementation:** 2-3 days
- **Phase 2 - Stage 1 Training:** 1-2 days (8-12 hours GPU)
- **Phase 3 - Stage 2 Implementation:** 3-5 days
- **Phase 3 - Self-Play:** 2-3 days (1000+ games)
- **Phase 4 - Evaluation:** 1-2 days
- **Total:** ~2 weeks to full v6.0 deployment

## 📁 File Structure

```
v6.0/
├── data/
│   ├── stage1/
│   │   ├── good_positions.jsonl          ✅ 5,719,272 positions (22.68 GB)
│   │   ├── bad_positions.jsonl           ✅ 69,240 positions (248.78 MB)
│   │   └── transposition_graph.pkl       🚧 Building...
│   ├── stage2/                           (Self-play outputs)
│   └── raw/                              (References v5.0 data)
│
├── models/                                (Trained models)
│   ├── stage1_policy.h5                  (TODO)
│   └── position_embeddings.npy           (TODO)
│
├── scripts/
│   ├── utils/
│   │   ├── zobrist_hashing.py            ✅
│   │   ├── calculate_features.py         ✅ (from v5.0)
│   │   └── temporal_feature_calculator.py ✅ (from v5.0)
│   ├── stage1/
│   │   ├── filter_dataset.py             ✅
│   │   ├── build_graph.py                🚧 Running
│   │   └── train_policy.py               (TODO)
│   └── stage2/
│       └── self_play.py                   (TODO)
│
├── configs/
│   └── stage1_config.yaml                ✅
│
├── docs/
│   ├── README.md                         ✅
│   ├── IMPLEMENTATION_PLAN.md            ✅
│   └── STATUS.md                         ✅ (This file)
│
├── quick_start_data_prep.ps1             ✅
├── check_progress.ps1                    ✅
└── DATA_SOURCE.txt                       ✅ (Points to v5.0 data)
```

## 🔍 Monitoring Commands

**Check data preparation progress:**
```powershell
cd "E:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v6.0"
.\check_progress.ps1
```

**View filtering results:**
```powershell
# Good positions summary
Get-Content data\stage1\good_positions.jsonl | Select-Object -First 5

# Bad positions summary
Get-Content data\stage1\bad_positions.jsonl | Select-Object -First 5
```

**Check graph statistics (after completion):**
```powershell
# Will be in build_graph.py terminal output
```

## 📞 Contact Points

**Questions/Issues:**
- Data quality concerns → See filtering report in terminal output
- Graph building performance → Expected ~2-3 hours for 5.7M positions
- Training implementation → Refer to IMPLEMENTATION_PLAN.md
- Performance targets → See success criteria above

---

**Next Action:** Wait for graph building to complete (~2-3 hours), then implement Stage 1 training (`train_policy.py`).
