# V7P3R AI v5.0 - Quick Reference Card

## 🎯 Dataset Overview

| Metric | Value |
|--------|-------|
| **Total Positions** | 228,666 |
| **File Size** | 548.31 MB |
| **PGN Positions** | 210,054 (92%) |
| **Puzzle Positions** | 18,612 (8%) |
| **Grading Depth** | Stockfish 16, depth 15 |
| **Error Rate** | 0% (perfect run) |

## 📂 Key Files

| File | Description | Size | Records |
|------|-------------|------|---------|
| `data/final/v7p3r_ai_v5_training_dataset_complete.jsonl` | **Master dataset** | 548 MB | 228,666 |
| `data/analysis/splits/train.jsonl` | Training set (80%) | 460 MB | 182,930 |
| `data/analysis/splits/validation.jsonl` | Validation set (10%) | 57 MB | 22,864 |
| `data/analysis/splits/test.jsonl` | Test set (10%) | 58 MB | 22,872 |
| `data/analysis/dataset_analysis.md` | Statistics report | 8 KB | - |

## 📊 Key Statistics

### Move Quality (V7P3R Performance)
- **Grade 5 (best)**: 40.14% ⭐
- **Grade 4 (2nd)**: 15.18%
- **Grade 3 (3rd)**: 9.03%
- **Grades 2-0**: 35.65% (learning opportunities)

### Game Phases
- **Opening**: 80.54%
- **Middlegame**: 15.80%
- **Endgame**: 3.66%

### Move Types
- **Quiet**: 64.16%
- **Captures**: 25.89%
- **Checks**: 12.74%
- **Other**: 2.21%

## 🔧 Quick Commands

### Analyze Dataset
```bash
python scripts/analyze_dataset.py \
  --input "data/final/v7p3r_ai_v5_training_dataset_complete.jsonl" \
  --output "data/analysis" \
  --create-splits
```

### Sample Random Records
```powershell
# Sample 10 random positions
Get-Content "data/final/v7p3r_ai_v5_training_dataset_complete.jsonl" | 
  Get-Random -Count 10 | 
  ConvertFrom-Json | 
  ConvertTo-Json -Depth 10
```

### Check Grade Distribution
```powershell
# Count by grade
Get-Content "data/final/v7p3r_ai_v5_training_dataset_complete.jsonl" | 
  ForEach-Object { ($_ | ConvertFrom-Json).stockfish_analysis.move_quality_grade } | 
  Group-Object | 
  Sort-Object Name | 
  Select-Object Name, Count
```

### Verify File Integrity
```powershell
# Count lines (should be 228,666)
(Get-Content "data/final/v7p3r_ai_v5_training_dataset_complete.jsonl" | Measure-Object -Line).Lines

# Check for JSON errors
Get-Content "data/final/v7p3r_ai_v5_training_dataset_complete.jsonl" | 
  ForEach-Object { try { $_ | ConvertFrom-Json } catch { "Error: $_" } }
```

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **COMPLETION_SUMMARY.md** | Comprehensive completion report |
| **DATASET_COMPLETE.md** | Dataset details and schema |
| **PIPELINE_STATUS.md** | Pipeline progress tracking |
| **TRAINING_PIPELINE_QUICKSTART.md** | Quick start guide |
| **UNIFIED_TRAINING_DATASET.md** | Data schema specification |
| **docs/V7P3R_FEATURE_SET_DEFINITION.md** | Feature catalog (130+ heuristics) |

## 🚀 Next Steps

### 1. Build PyTorch Dataset Loader
Create `scripts/dataset_loader.py`:
```python
class V7P3RDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_file):
        # Load and parse JSONL
        # Extract features and labels
        
    def __getitem__(self, idx):
        # Return (features, policy_label, value_label)
        
    def __len__(self):
        return self.num_positions
```

### 2. Design Neural Network
```python
class V7P3RNet(nn.Module):
    def __init__(self, num_features=20):
        # Input normalization
        # Embedding layer
        # Hidden layers
        # Policy head (6-way classification)
        # Value head (regression)
```

### 3. Train Model
```python
for epoch in range(num_epochs):
    for features, policy_labels, value_labels in train_loader:
        # Forward pass
        policy_out, value_out = model(features)
        
        # Calculate losses
        policy_loss = criterion_ce(policy_out, policy_labels)
        value_loss = criterion_mse(value_out, value_labels)
        total_loss = policy_loss + lambda * value_loss
        
        # Backprop and optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
```

## 🎓 Training Labels

### Policy Head (Classification)
- **Source**: `stockfish_analysis.move_quality_grade`
- **Values**: 0, 1, 2, 3, 4, 5
- **Type**: Categorical (6 classes)
- **Loss**: CrossEntropyLoss

### Value Head (Regression)
- **Source**: `stockfish_analysis.best_move_eval_cp`
- **Values**: Centipawns (e.g., -500 to +500)
- **Type**: Continuous
- **Loss**: MSELoss or HuberLoss

## 📈 Success Metrics

| Metric | Baseline | Target | Stretch Goal |
|--------|----------|--------|--------------|
| **Policy Accuracy** | 16.7% (random) | >50% | >70% |
| **Top-3 Accuracy** | 50% | >75% | >90% |
| **Value MAE** | - | <100 cp | <50 cp |
| **Value Correlation** | - | r>0.7 | r>0.9 |

## 🔍 Data Schema Quick Reference

Every record has 5 blocks:
1. **metadata**: Source, file, IDs, version, game info
2. **position**: FEN, move number, phase, material, checks
3. **engine_decision**: Move UCI/SAN, type flags, eval
4. **stockfish_analysis**: Top-5 moves, ranks, **grades** (0-5)
5. **features**: 20+ observations (F001-F053)

## ⚡ Performance Benchmarks

| Stage | Speed | Total Time |
|-------|-------|------------|
| **PGN Extraction** | 5,250 pos/sec | 40 sec |
| **Feature Calc** | 3,750 pos/sec | 56 sec |
| **Stockfish Grading** | 4.1 pos/sec | 14.2 hours |
| **Analysis** | 38,000 pos/sec | 10 sec |

## 🎯 Key Features (Top 10)

| Feature | Description | Type |
|---------|-------------|------|
| F001_position_fen | Full FEN string | String |
| F002_game_phase | opening/middlegame/endgame | Categorical |
| F003_material_balance_cp | Material in centipawns | Integer |
| F030_white_piece_mobility | White piece mobility count | Integer |
| F030_black_piece_mobility | Black piece mobility count | Integer |
| F005_total_piece_count | Total pieces on board | Integer |
| F010_*_king_castled | Has king castled? | Boolean |
| F011_*_king_has_pawn_shield | Has pawn shield? | Boolean |
| F032_*_has_bishop_pair | Has bishop pair? | Boolean |
| F050_is_capture | Is move a capture? | Boolean |

## 💾 Storage Requirements

| Component | Size |
|-----------|------|
| Final dataset | 548 MB |
| Train split | 460 MB |
| Val split | 57 MB |
| Test split | 58 MB |
| Analysis reports | 8 MB |
| **Total** | **1.1 GB** |

## 🔒 Data Quality

| Check | Result |
|-------|--------|
| **JSON Validity** | ✅ 100% valid |
| **Complete Records** | ✅ All 5 blocks present |
| **Stockfish Grades** | ✅ 228,666 / 228,666 |
| **Features** | ✅ 228,666 / 228,666 |
| **Errors** | ✅ 0 errors |

---

**Status**: ✅ READY FOR MODEL TRAINING  
**Date**: May 7, 2026  
**Total Time**: 14.5 hours  
**Next**: Build PyTorch dataset loader

*Quick reference for V7P3R AI v5.0 training pipeline*
