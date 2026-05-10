# V7P3R AI v5.1 + TPF - Quick Start Checklist
## Complete Pipeline Execution Guide

**Goal**: Train fresh 100-epoch model with 262 features (106 v5.1 + 156 temporal)  
**Expected**: 56-60% accuracy, all grades predicted (not binary)  
**Time**: ~12-15 hours end-to-end

---

## ✅ Pre-Flight Checklist

### 1. Verify Data Files
```powershell
# Puzzle database
Test-Path "E:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v5.0\data\puzzles\lichess_db_puzzle.jsonl"
# Should return: True

# Game data
Test-Path "E:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v5.0\data\final\v7p3r_ai_v5_training_dataset_complete.jsonl"
# Should return: True

# Stockfish
Test-Path "E:\Programming Stuff\Chess Engines\Tournament Engines\Stockfish\stockfish-windows-x86-64-avx2.exe"
# Should return: True
```

### 2. Verify Python Environment
```powershell
cd "E:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v5.0"

# Check dependencies
python -c "import chess; import torch; import sklearn; print('✓ All dependencies installed')"

# Check GPU (optional but recommended)
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"}')"
```

### 3. Review Implementation Files
```powershell
# Verify all new files were created
Test-Path "scripts\temporal_feature_calculator.py"          # ✓ CREATED
Test-Path "scripts\extract_puzzle_sequences.py"             # ✓ CREATED
Test-Path "run_full_pipeline_v5.1_temporal.ps1"             # ✓ CREATED
Test-Path "docs\V5.1_TPF_IMPLEMENTATION_SUMMARY.md"         # ✓ CREATED

# Verify updates
Select-String -Path "scripts\preprocess_dataset_v5.1.py" -Pattern "move_encoding"
# Should show: move_encoding features added

Select-String -Path "configs\training_config.yaml" -Pattern "input_dim: 262"
# Should show: input_dim updated to 262
```

---

## 🚀 Execution Options

### Option A: Full Automated Pipeline (Recommended)
```powershell
# One command to rule them all
.\run_full_pipeline_v5.1_temporal.ps1

# Interactive prompts will ask:
# - Extract puzzles? (y/n) → Answer 'y'
# - Recalculate features? (y/n) → Answer 'y'
# - Start training? (y/n) → Answer 'y'

# Total time: ~12-15 hours
```

### Option B: Phased Execution (Monitor Each Step)
```powershell
# Phase 1: Puzzle Extraction (~30 minutes)
python scripts\extract_puzzle_sequences.py --input data\puzzles\lichess_db_puzzle.jsonl --output data\puzzles\puzzle_sequences_with_features.jsonl --limit 20000 --feature-set standard

# Verify output
Get-Content data\puzzles\puzzle_sequences_with_features.jsonl | Measure-Object -Line
# Expected: ~60,000 lines (positions)

# Phase 2: Game Feature Calculation (4-6 hours)
python scripts\calculate_features.py --input data\final\v7p3r_ai_v5_training_dataset_complete.jsonl --output data\final\v7p3r_ai_v5.1_game_features.jsonl --feature-set standard

# Verify output
Get-Content data\final\v7p3r_ai_v5.1_game_features.jsonl | Measure-Object -Line
# Expected: ~230,930 lines

# Phase 3: Merge Datasets
Get-Content data\puzzles\puzzle_sequences_with_features.jsonl, data\final\v7p3r_ai_v5.1_game_features.jsonl | Set-Content data\final\v7p3r_ai_v5.1_with_temporal.jsonl

# Verify merge
Get-Content data\final\v7p3r_ai_v5.1_with_temporal.jsonl | Measure-Object -Line
# Expected: ~291,000 lines

# Phase 4: Split Dataset (~1 minute)
python scripts\split_dataset.py --input data\final\v7p3r_ai_v5.1_with_temporal.jsonl --output-dir data\analysis\splits_v5.1 --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 --seed 42

# Verify splits
Get-ChildItem data\analysis\splits_v5.1
# Expected: train.jsonl, validation.jsonl, test.jsonl, split_stats.json

# Phase 5: Preprocessing (~30 minutes)
$env:V7P3R_SPLIT_DIR = "data\analysis\splits_v5.1"
python scripts\preprocess_dataset_v5.1.py

# Verify preprocessing
Get-Content data\preprocessed_v5.1\preprocessing_stats.json
# Check: input_dim should be 262

# Phase 6: Training (6-8 hours)
python src\train.py --epochs 100 --batch-size 256 --data-dir data\preprocessed_v5.1 --checkpoint-dir checkpoints_v5.1_temporal --early-stopping-patience 15 --log-interval 10

# Monitor training (separate terminal)
Get-Content checkpoints_v5.1_temporal\training_log.txt -Wait

# Phase 7: Evaluation (~5 minutes)
python src\evaluate.py --checkpoint checkpoints_v5.1_temporal\best_model.pth --data-dir data\preprocessed_v5.1 --output-dir evaluation_results_v5.1_temporal
```

### Option C: Quick Test (Skip Heavy Operations)
```powershell
# Use existing puzzle sequences (if already extracted)
# Use existing game features (if already calculated)
# Only run preprocessing + training

.\run_full_pipeline_v5.1_temporal.ps1 -SkipPuzzleExtraction -SkipFeatureCalculation

# Time: ~7-9 hours (just preprocessing + training + eval)
```

---

## 📊 Monitoring & Validation

### During Puzzle Extraction
```
Expected output every 100 puzzles:
  Processed 100/20000 puzzles (300 positions)
  Processed 200/20000 puzzles (600 positions)
  ...
  Processed 20000/20000 puzzles (60000 positions)
  
Red flags:
  ❌ "Feature calculation failed" errors >5%
  ❌ Output file <50,000 positions
```

### During Feature Calculation
```
Expected output every 1,000 positions:
  Processed 1,000/230,930 positions (0.4%)
  Processed 2,000/230,930 positions (0.9%)
  ...
  Avg time per position: 75-100ms
  
Red flags:
  ❌ Time per position >200ms
  ❌ "Stockfish timeout" errors
  ❌ CPU usage <50% (Stockfish not running)
```

### During Preprocessing
```
Expected output:
  Loading dataset splits...
    Train: 233,544 positions
    Val:   29,193 positions
    Test:  29,193 positions
  
  Extracting features...
    Numerical: 67 features
    Boolean: 50 features
    Categorical: 145 one-hot features
  
  Fitting transformers on training data...
    StandardScaler fitted on 233,544 positions
    Game phase categories: 3
    Material categories: 5
    Move type categories: 4
    Move encoding categories: 65 (squares + -1) and 7 (pieces)
  
  Transforming validation and test sets...
    X_train: (233544, 262)
    X_val:   (29193, 262)
    X_test:  (29193, 262)
  
Red flags:
  ❌ X_train shape is not (*, 262)
  ❌ "KeyError" for temporal features
  ❌ NaN values in output arrays
```

### During Training (CRITICAL)
```
Epoch 1 (Baseline establishment):
  Train: policy_acc ~0.35-0.40
  Val:   policy_acc ~0.38-0.42
  
Epoch 10 (Early learning):
  Train: policy_acc ~0.45-0.48
  Val:   policy_acc ~0.46-0.49
  
Epoch 25 (Breaking binary barrier):
  Train: policy_acc ~0.50-0.53
  Val:   policy_acc ~0.51-0.54
  ⚠️ CHECK CONFUSION MATRIX: All grades should have >5% predictions
  
Epoch 50 (Solidifying spectrum):
  Train: policy_acc ~0.54-0.57
  Val:   policy_acc ~0.55-0.58
  ✓ All grades should have >15% predictions
  
Epoch 100 (Target achievement):
  Train: policy_acc ~0.57-0.61
  Val:   policy_acc ~0.56-0.60
  ✓ GOAL ACHIEVED: 56-60% accuracy, all grades predicted
  
Red flags:
  ❌ Policy accuracy stuck <0.50 after epoch 30
  ❌ Grades 1-4 still show 0% predictions after epoch 25
  ❌ Validation loss increasing (overfitting)
  ❌ GPU utilization <50% (bottleneck somewhere)
  
Green flags:
  ✅ Validation accuracy tracking training closely (±2%)
  ✅ Learning rate reduces at epochs ~30, ~50, ~70
  ✅ Value MAE steadily decreasing
  ✅ Confusion matrix balanced across grades
```

---

## 🎯 Success Validation

### Confusion Matrix Analysis
```powershell
# After training completes
python src\evaluate.py --checkpoint checkpoints_v5.1_temporal\best_model.pth --data-dir data\preprocessed_v5.1 --output-dir evaluation_results_v5.1_temporal

# Check confusion matrix
Get-Content evaluation_results_v5.1_temporal\confusion_matrix.txt
```

**Expected (SUCCESS)**:
```
Confusion Matrix (Validation Set):
        Predicted
Actual    0    1    2    3    4    5
   0   [5412  340  287  213  105  832]  ← ALL COLUMNS HAVE VALUES
   1   [ 298  263  189  142  214  208]  ← GRADE 1 PREDICTED!
   2   [ 324  221  452  298  276  239]  ← GRADE 2 PREDICTED!
   3   [ 378  268  352  788  512  329]  ← GRADE 3 PREDICTED!
   4   [ 542  389  441  694 1543  800]  ← GRADE 4 PREDICTED!
   5   [1220  634  712  889 1915 9474]

Per-Grade Accuracy:
  Grade 0: 75.3% ✓
  Grade 1: 20.0% ✓ (vs 0% baseline)
  Grade 2: 25.0% ✓ (vs 0% baseline)
  Grade 3: 30.0% ✓ (vs 0% baseline)
  Grade 4: 35.0% ✓ (vs 0% baseline)
  Grade 5: 80.0% ✓

Overall Accuracy: 57.8% ✓ (vs 49.1% baseline)
```

**FAILURE (Binary Classification Persists)**:
```
Confusion Matrix (Validation Set):
        Predicted
Actual    0    1    2    3    4    5
   0   [5412   0    0    0    0  1777]  ← ONLY COLUMNS 0 AND 5
   1   [ 614   0    0    0    0   700]  ← GRADE 1 NEVER PREDICTED
   2   [ 843   0    0    0    0   967]  ← GRADE 2 NEVER PREDICTED
   3   [1225   0    0    0    0  1402]  ← GRADE 3 NEVER PREDICTED
   4   [2057   0    0    0    0  2352]  ← GRADE 4 NEVER PREDICTED
   5   [2023   0    0    0    0  9821]

❌ ROLLBACK TO v5.0 - Binary classification not fixed
```

### Temporal Feature Impact Analysis
```python
# Compare puzzle sequences vs game positions
# Puzzle sequences have has_history=1 (temporal features populated)
# Game positions have has_history=0 (temporal features masked)

# Expected: Puzzle accuracy 5-10% higher than game accuracy
# Confirms temporal features provide value
```

---

## 🐛 Troubleshooting Guide

### Issue 1: Out of Memory During Preprocessing
```powershell
# Reduce memory footprint
$env:V7P3R_BATCH_SIZE = "1000"  # Process in smaller batches
python scripts\preprocess_dataset_v5.1.py
```

### Issue 2: Training Too Slow (No GPU)
```yaml
# Edit configs/training_config.yaml
training:
  batch_size: 128  # Reduce from 256
  num_workers: 2   # Reduce from 4
```

### Issue 3: Binary Classification Still Occurs
```python
# Verify class weights in src/train.py
class_weights = torch.tensor([1.0, 5.0, 3.5, 2.5, 1.8, 1.0])
policy_criterion = nn.CrossEntropyLoss(weight=class_weights)

# If weights not applied, model will collapse to binary
```

### Issue 4: Temporal Features Not Used
```powershell
# Verify preprocessing detected move encoding
Get-Content data\preprocessed_v5.1\preprocessing_stats.json | ConvertFrom-Json | Select-Object -ExpandProperty feature_breakdown

# Should show:
# - move_encoding: 3 raw features → 135 one-hot
```

### Issue 5: Stockfish Not Found
```powershell
# Update path in scripts/calculate_features.py
$stockfishPath = "E:\Programming Stuff\Chess Engines\Tournament Engines\Stockfish\stockfish-windows-x86-64-avx2.exe"
```

---

## 📝 Post-Training Checklist

### Immediate Analysis
- [ ] Confusion matrix shows all 6 grades predicted
- [ ] Overall accuracy ≥56%
- [ ] Grades 1-4 each show ≥15% individual accuracy
- [ ] Value MAE <0.20
- [ ] No overfitting (train_loss ≈ val_loss)

### Documentation Updates
- [ ] Update `docs/MODEL_ARCHITECTURE.md` with TPF details
- [ ] Document training results in project changelog
- [ ] Save confusion matrix and metrics
- [ ] Compare v5.0 vs v5.1+TPF side-by-side

### Future Enhancements (If Successful)
- [ ] Test self-play integration with temporal context
- [ ] Implement Phase 2: Opening book features
- [ ] Explore reinforcement learning with temporal state
- [ ] Publish results (if >60% accuracy achieved)

---

## 🎉 Victory Conditions

**Minimum Success** (Deploy to Production):
- ✅ 56%+ overall accuracy
- ✅ All grades predicted (not binary)
- ✅ Stable training (no divergence)

**Excellent Success** (Announce to Community):
- ✅ 58-60% overall accuracy
- ✅ 65-70% puzzle sequence accuracy
- ✅ Temporal ablation shows 8% impact

**Game-Changing Success** (Publish Paper):
- ✅ 60%+ overall accuracy (10% improvement)
- ✅ Temporal features fundamental to performance
- ✅ Generalizes to self-play and reinforcement learning

---

## 🚦 Final Status

**ALL SYSTEMS READY FOR LAUNCH! 🚀**

| Component | Status | Notes |
|-----------|--------|-------|
| Temporal Feature Calculator | ✅ | `scripts/temporal_feature_calculator.py` |
| Puzzle Sequence Extractor | ✅ | `scripts/extract_puzzle_sequences.py` |
| Preprocessing Script | ✅ | Updated for 262 features |
| Model Architecture | ✅ | Unchanged (designed for expansion) |
| Training Config | ✅ | `input_dim: 262` |
| Master Pipeline | ✅ | `run_full_pipeline_v5.1_temporal.ps1` |
| Documentation | ✅ | Complete implementation summary |

**READY TO START TRAINING!**

---

## 🚀 Launch Command

```powershell
cd "E:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v5.0"

# Full automated pipeline
.\run_full_pipeline_v5.1_temporal.ps1

# Or manual phased execution (see Option B above)
```

**Your AI is about to gain MEMORY! 🧠**  
**Expected completion: 12-15 hours**  
**Expected accuracy: 56-60% (vs 49% baseline)**  
**Expected impact: Binary classification → Full grade spectrum**

---

**Good luck! May your gradients flow smoothly and your accuracy soar! 🚀**
