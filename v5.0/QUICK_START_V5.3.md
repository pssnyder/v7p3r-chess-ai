# V7P3R AI v5.3 - Quick Start Guide

## 🎯 Strategic Pivot Summary

**Problem:** v5.1/v5.2 plateaued at 45% accuracy due to data starvation  
**Solution:** Expand to 750k positions + optimize for "good move prediction" (grades 0-2)  
**New Metric:** Good Move Rate (% predictions in grades 0-2) - Target: >70%

---

## 📁 New Files Created

### Data Collection
- `scripts/multi_engine_puzzle_solver.py` - V7P3R engines solve puzzles
- `run_data_expansion_v5.3.ps1` - Master pipeline orchestrator

### Evaluation  
- `src/good_move_metrics.py` - New evaluation metrics focused on good move prediction

### Documentation
- `docs/V5.3_STRATEGY_PIVOT.md` - Complete strategy explanation
- `docs/TRAINING_PLATEAU_DIAGNOSTIC.md` - Why v5.2 failed (from earlier)

---

## 🚀 Getting Started (3 Options)

### Option 1: Full Automated Pipeline (RECOMMENDED)

```powershell
# Run entire data expansion pipeline
.\run_data_expansion_v5.3.ps1
```

**Timeline:** ~12-15 hours total  
**Output:** 750k preprocessed positions ready for training

**Steps:**
1. Multi-engine puzzle solving (3-4 hours)
2. Full Lichess extraction (2-3 hours)
3. C0BR4 integration (4-6 hours)
4. Dataset merging and preprocessing (2-3 hours)

### Option 2: Manual Step-by-Step

**Step 1: Multi-Engine Puzzle Solving**
```powershell
cd "E:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v5.0"
python scripts\multi_engine_puzzle_solver.py
```
- **Time:** 3-4 hours
- **Output:** 40k positions (V7P3R character data)
- **Location:** `data/multi_engine_puzzles/`

**Step 2: Full Lichess Extraction**
```powershell
python scripts\extract_puzzle_sequences.py `
    --input "E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\pgn_training_data\csv_data_puzzles\lichess_db_puzzle.csv" `
    --output data\puzzles\puzzle_sequences_full.jsonl `
    --num-puzzles 100000 `
    --rating-min 1500 `
    --rating-max 2500
```
- **Time:** 2-3 hours  
- **Output:** ~450k positions (tactical training)

**Step 3: C0BR4 Integration** (need to create script first)
```powershell
# TODO: Create integrate_cobra_games.py
python scripts\integrate_cobra_games.py
```

**Step 4: Merge and Preprocess**
```powershell
# TODO: Create merge_datasets_v5.3.py
python scripts\merge_datasets_v5.3.py
python scripts\balance_grade_distribution.py
python scripts\preprocess_dataset_v5.1.py --input data\final\v7p3r_ai_v5.3_merged.jsonl --output data\preprocessed_v5.3 --version v5.3
```

### Option 3: Start with Multi-Engine Only (FASTEST)

**Just run the new multi-engine puzzle solver:**
```powershell
python scripts\multi_engine_puzzle_solver.py
```

**Why?**
- Test new infrastructure (3-4 hours vs 12-15 hours)
- Get immediate V7P3R-character data
- Validate puzzle grading approach
- Can merge with existing 324k later

**Then train on combined data:**
- Existing 324k + new 40k = 364k positions
- Better data-to-param ratio (1.1 → 1.3)
- Should still see 2-3% improvement

---

## 📊 How to Evaluate with New Metrics

### After Training v5.3

```powershell
# Load model and evaluate
python scripts\evaluate_with_good_move_metrics.py --model models/v5.3/best_model.pth --data data/preprocessed_v5.3
```

**Output:**
```
🎯 PRIMARY METRICS
Good Move Rate (Grades 0-2):     68.5%  ← Target: >70%
Excellent Move Rate (Grades 0-1): 42.3%  ← Target: >40%
Bad Avoidance:                    86.2%  ← Target: >85%

✅ 2/3 criteria met - Continue training
```

### Compare with v5.1 Baseline

```powershell
# Retroactive evaluation of v5.1
python scripts\evaluate_with_good_move_metrics.py --model models/v5.1_tpf/best_model.pth --data data/preprocessed_v5.1
```

**Expected v5.1 Baseline:**
- Good Move Rate: ~51%
- Excellent Move Rate: ~28%
- Bad Avoidance: ~62%

---

## 🎯 Success Criteria

### Minimum Viable (Deploy to Testing)
- ✅ Good Move Rate >65%
- ✅ Excellent Move Rate >35%
- ✅ Bad Avoidance >80%

### Target Performance (Production Ready)
- 🎯 Good Move Rate >70%
- 🎯 Excellent Move Rate >40%
- 🎯 Bad Avoidance >85%

### Comparison Table

| Metric | v5.1 | v5.3 (Target) | Improvement |
|--------|------|---------------|-------------|
| **Data Size** | 324k | 750k | +131% |
| **Data-to-Param** | 1.0 | 2.3 | +130% |
| **Good Move Rate** | ~51% | >70% | +19pp |
| **Excellent Move Rate** | ~28% | >40% | +12pp |
| **Bad Avoidance** | ~62% | >85% | +23pp |

---

## 🔄 Training Pipeline

### Create v5.3 Training Config

```yaml
# configs/training_config_v5.3_expanded.yaml
model:
  input_dim: 325  # Same as v5.1 (temporal features)
  shared_dims: [256, 256, 128, 64]  # Revert from v5.2 wide
  policy_dims: [64]
  value_dims: [32]
  dropout: 0.3

training:
  batch_size: 512  # Increased from 256 (more data available)
  learning_rate: 0.001
  weight_decay: 0.0001
  epochs: 100
  
  # Reduced class weights (after oversampling)
  class_weights: [1.0, 2.0, 1.5, 1.5, 1.5, 1.0]
  
  early_stopping_patience: 20
  
paths:
  data_dir: "data/preprocessed_v5.3"
  checkpoint_dir: "models/v5.3"
```

### Train v5.3

```powershell
python src\train.py `
    --config configs\training_config_v5.3_expanded.yaml `
    --data-dir data\preprocessed_v5.3 `
    --checkpoint-dir models\v5.3
```

**Expected Training:**
- Duration: ~25-30 min per epoch (larger dataset)
- Convergence: Epochs 40-60
- Validation loss: Should reach <1.35 (vs 1.44 in v5.1)

---

## 🧪 Test Good Move Metrics Script

```powershell
# Test metrics on simulated data
cd src
python good_move_metrics.py
```

**Output:**
```
============================================================
V7P3R AI - GOOD MOVE FOCUSED EVALUATION
============================================================

🎯 PRIMARY METRICS (Optimization Targets)
Good Move Rate (Grades 0-2):     70.15%
Excellent Move Rate (Grades 0-1): 45.23%
Bad Move Avoidance:               85.12%

✅ SUCCESS CRITERIA
✅ Good Move Rate >70%
✅ Excellent Move Rate >40%
✅ Bad Avoidance >85%
✅ Grade 0 Precision >60%
✅ Good Move Accuracy >80%

Passed: 5/5 criteria

🎉 MODEL READY FOR DEPLOYMENT!
============================================================
```

---

## 📦 Missing Scripts to Create

### 1. C0BR4 Integration Script

```python
# scripts/integrate_cobra_games.py
# TODO: Create script to:
# 1. Load C0BR4 PGN files
# 2. Extract positions
# 3. Run Stockfish grading
# 4. Calculate features
# 5. Output JSONL format
```

### 2. Dataset Merger for v5.3

```python
# scripts/merge_datasets_v5.3.py
# TODO: Create script to merge:
# 1. Multi-engine puzzle results
# 2. Full Lichess puzzles
# 3. C0BR4 games
# 4. V7P3R historical games
```

### 3. Grade Balancing Script

```python
# scripts/balance_grade_distribution.py
# TODO: Oversample rare grades (1, 3, 4) to balance distribution
# Target: Each grade 15-20% of dataset
```

### 4. Evaluation Script with New Metrics

```python
# scripts/evaluate_with_good_move_metrics.py
# TODO: Load model, run inference, calculate good move metrics
```

---

## 🎬 Recommended First Step

**START HERE** (Fastest validation):

```powershell
# 1. Test multi-engine puzzle solver on small sample
cd "E:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v5.0"

# Edit multi_engine_puzzle_solver.py:
# Change: PUZZLES_PER_ENGINE = 10000
# To: PUZZLES_PER_ENGINE = 100  (for quick test)

python scripts\multi_engine_puzzle_solver.py
```

**Expected output (100 puzzles × 4 engines):**
```
V7P3R v18.4 Puzzle Solver
Processing 100 puzzles...
✅ Completed! Solved 98 puzzles

📊 Summary Statistics
Total puzzles solved: 98
Exact puzzle matches: 42 (42.9%)

Move Quality Distribution:
  Grade 0:    28 (28.6%) ████████████
  Grade 1:    15 (15.3%) ███████
  Grade 2:    22 (22.4%) ███████████
  Grade 3:    18 (18.4%) █████████
  Grade 4:    10 (10.2%) █████
  Grade 5:     5 ( 5.1%) ██

🎯 Good Moves (Grades 0-2): 65 (66.3%)
```

**If this works:** Scale up to 10k puzzles per engine.  
**If this fails:** Debug before full pipeline.

---

## 📞 Support / Issues

- Review `docs/V5.3_STRATEGY_PIVOT.md` for strategic background
- Check `docs/TRAINING_PLATEAU_DIAGNOSTIC.md` for why v5.2 failed
- Existing pipeline scripts: `run_full_pipeline_v5.1_temporal.ps1`

---

*Last Updated: May 10, 2026*  
*Version: v5.3 Strategy*  
*Status: Ready to execute*
