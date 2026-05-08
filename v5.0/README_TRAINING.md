# V7P3R AI v5.0 - Training System

Complete neural network training pipeline for V7P3R chess position evaluation.

## Quick Start

```powershell
# 1. Test the pipeline (2 epochs)
.\quick_start_train.ps1

# 2. If test passes, start full training (100 epochs)
python src/train.py --config configs/training_config.yaml
```

---

## Architecture

**Model**: Dual-head neural network with residual connections
- **Input**: 26 features (heuristics as observations)
- **Shared Embedding**: 256 → 256 → 128 → 64 (with residuals + batch norm)
- **Policy Head**: 6-class move quality classification (grades 0-5)
- **Value Head**: Position evaluation regression ([-1, 1])
- **Parameters**: ~163k (0.62 MB)

**Design Philosophy**: Built for future expansion (26 → 40+ features) without architectural changes.

---

## Dataset

**Total**: 230,930 positions (as of May 7, 2026)
- **Training**: 184,742 (80%)
- **Validation**: 23,089 (10%)
- **Test**: 23,099 (10%)

**Sources**:
- V7P3R game history (210,054 positions)
- Puzzle analyses (20,876 positions)

**Grading**: Stockfish depth-15 analysis (0-5 quality scale)

---

## Training Pipeline

### 1. Preprocessing ✅ COMPLETE

```bash
python scripts/preprocess_dataset.py
```

**Output**: `data/preprocessed/` (46 MB total)
- Normalized numerical features (StandardScaler)
- One-hot encoded categoricals (game phase, material advantage)
- Saved transformers for inference

### 2. Training

```bash
# Full training (100 epochs)
python src/train.py --config configs/training_config.yaml

# Resume from checkpoint
python src/train.py --resume checkpoints/latest_checkpoint.pth
```

**Features**:
- Dual-head loss (policy + value)
- Early stopping (patience: 15 epochs)
- Learning rate scheduling (ReduceLROnPlateau)
- Gradient clipping (prevent explosions)
- Model checkpointing (save best + periodic)
- Training history logging

**Hyperparameters** (configurable in `training_config.yaml`):
- Batch size: 256
- Learning rate: 1e-3 (adaptive)
- Dropout: 0.3
- Loss weights: policy=1.0, value=0.1
- Optimizer: AdamW with weight decay

### 3. Evaluation

```bash
# Evaluate on test set
python src/evaluate.py --checkpoint checkpoints/best_model.pth

# Save predictions for analysis
python src/evaluate.py --checkpoint checkpoints/best_model.pth --save-predictions
```

**Output**: `evaluation_results/`
- Detailed metrics report (Markdown)
- Confusion matrix
- Per-grade performance
- Correlation analysis

---

## Performance Targets

### Policy Head (Move Quality)
- ✅ **Accuracy >50%**: Exact grade match
- ✅ **Top-2 Accuracy >75%**: Within ±1 grade
- ✅ **Top-3 Accuracy >85%**: Within ±2 grades

### Value Head (Position Evaluation)
- ✅ **MAE <0.15**: ~1500cp average error
- ✅ **Correlation >0.80**: Strong alignment with Stockfish

**Baseline (Random)**: 16.7% accuracy, 0.30 MAE

---

## Project Structure

```
v5.0/
├── src/
│   ├── model.py          # PyTorch model definition (163k params)
│   ├── dataset.py        # Dataset classes and loaders
│   ├── train.py          # Training pipeline
│   └── evaluate.py       # Evaluation and metrics
├── scripts/
│   ├── preprocess_dataset.py     # Data preprocessing
│   ├── extract_v7p3r_pgns.py     # PGN position extraction
│   ├── calculate_features.py    # Feature calculation
│   ├── grade_with_stockfish.py  # Stockfish grading
│   └── analyze_dataset.py       # Dataset statistics
├── configs/
│   ├── training_config.yaml     # Full training config (100 epochs)
│   └── test_config.yaml         # Quick test config (2 epochs)
├── data/
│   ├── preprocessed/    # ✅ Preprocessed arrays (46 MB)
│   ├── final/           # ✅ Master dataset (575 MB)
│   └── analysis/        # ✅ Statistics and splits
├── checkpoints/         # Saved model checkpoints
├── evaluation_results/  # Evaluation reports
├── docs/
│   ├── MODEL_ARCHITECTURE.md        # Complete architecture specs
│   ├── PREPROCESSING_STRATEGY.md    # Preprocessing details
│   ├── DATA_PROCESSING_PIPELINE.md  # Full pipeline documentation
│   └── DATASET_CREATION_SUMMARY.md  # Dataset creation process
└── quick_start_train.ps1  # Quick start test script
```

---

## Training Workflow

### Standard Training Run

```powershell
# 1. Quick test (2 epochs - verify pipeline works)
.\quick_start_train.ps1

# 2. Full training (100 epochs - ~2-4 hours on GPU)
python src/train.py --config configs/training_config.yaml

# 3. Monitor progress
#    - Training history: checkpoints/training_history.json
#    - Live updates in terminal

# 4. Evaluate best model
python src/evaluate.py --checkpoint checkpoints/best_model.pth
```

### Resuming Training

```bash
# Resume from interruption
python src/train.py --resume checkpoints/latest_checkpoint.pth

# Continue from specific epoch
python src/train.py --resume checkpoints/checkpoint_epoch_50.pth
```

### Hyperparameter Tuning

```bash
# 1. Copy config
cp configs/training_config.yaml configs/experiment_1.yaml

# 2. Edit hyperparameters
#    - Adjust learning_rate, batch_size, dropout, etc.

# 3. Train with new config
python src/train.py --config configs/experiment_1.yaml
```

---

## Monitoring Training

### Key Metrics to Watch

**During Training**:
- **Train Loss**: Should decrease steadily
- **Val Loss**: Should decrease (with some noise)
- **Policy Accuracy**: Should increase toward 50%+
- **Learning Rate**: Will adapt based on val loss

**Warning Signs**:
- Val loss increasing while train loss decreases → **Overfitting**
  - Solution: Increase dropout, reduce model complexity
- Both losses stuck → **Underfitting**
  - Solution: Increase model capacity, reduce regularization
- NaN losses → **Exploding gradients**
  - Solution: Lower learning rate, check gradient clipping

### Checkpoints

**Auto-saved**:
- `latest_checkpoint.pth`: Every epoch
- `best_model.pth`: When val loss improves
- `checkpoint_epoch_N.pth`: Every 10 epochs

**Contents**:
- Model weights
- Optimizer state
- Training history
- Configuration

---

## Advanced Features

### Custom Loss Weighting

Edit `training_config.yaml`:
```yaml
training:
  policy_weight: 1.0   # Increase for better move quality
  value_weight: 0.2    # Increase for better position eval
```

**Recommendation**: Start with 1.0/0.1, adjust based on validation metrics.

### Gradient Accumulation (Larger Effective Batch Size)

```python
# In train.py, modify training loop:
accumulation_steps = 4  # Effective batch = 256 * 4 = 1024

for batch_idx, batch in enumerate(train_loader):
    loss = compute_loss(...)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Mixed Precision Training (Faster on GPU)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    policy_logits, value = model(features)
    loss = compute_loss(...)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## Troubleshooting

### "CUDA out of memory"
- Reduce `batch_size` in config (256 → 128)
- Use CPU: Set `CUDA_VISIBLE_DEVICES=""`

### "Preprocessed data missing"
- Run: `python scripts/preprocess_dataset.py`

### "No improvement for N epochs"
- Normal! Early stopping will trigger
- Or: Reduce `early_stopping_patience`

### Training too slow
- Enable GPU (see requirements below)
- Reduce `num_workers` if CPU bottleneck
- Use smaller test config for experiments

---

## Requirements

```bash
# Core dependencies
pip install torch numpy scikit-learn pyyaml

# Optional (for visualization)
pip install tensorboard matplotlib
```

**GPU Recommended**: CUDA-capable GPU (NVIDIA) for 10x speedup

---

## Expected Training Time

| Hardware | Time (100 epochs) |
|----------|-------------------|
| **GPU (RTX 3060)** | ~2-3 hours |
| **GPU (RTX 4090)** | ~1-2 hours |
| **CPU (i7)** | ~12-20 hours |

**Quick Test** (2 epochs): 2-5 minutes

---

## Next Steps After Training

1. **Evaluate Model**: `python src/evaluate.py --checkpoint checkpoints/best_model.pth`

2. **Review Metrics**: Check `evaluation_results/evaluation_report.md`

3. **Deploy for Inference**: Use trained model to evaluate new positions

4. **Iterate**:
   - Add more training data (incremental updates)
   - Expand feature set (26 → 40+ features)
   - Fine-tune hyperparameters

---

## Documentation

- **[MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md)**: Complete model design
- **[PREPROCESSING_STRATEGY.md](PREPROCESSING_STRATEGY.md)**: Feature engineering
- **[INCREMENTAL_UPDATE_WORKFLOW.md](INCREMENTAL_UPDATE_WORKFLOW.md)**: Adding new data
- **[DATASET_CHANGELOG.md](data/final/DATASET_CHANGELOG.md)**: Dataset version history

---

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review documentation in `docs/`
3. Check training logs in `checkpoints/training_history.json`

---

*Last Updated: May 7, 2026*  
*V7P3R AI Development Team*
