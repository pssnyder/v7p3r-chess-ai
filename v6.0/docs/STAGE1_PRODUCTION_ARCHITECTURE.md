# V7P3R AI v6.1 - Stage 1 Production Architecture
## Position Evaluator (GOOD vs BAD) - PRODUCTION MODEL

**Created**: 2026-05-30  
**Status**: ✅ **PRODUCTION READY**  
**Model File**: `models/position_evaluator_best.pth`  
**Training Date**: 2026-05-30  

---

## Executive Summary

Successfully trained a **binary position classifier** that distinguishes GOOD chess positions from BAD positions with **87.76% F1 score** (exceeds 82% target by 5.76 percentage points).

**Key Achievement**: Trained on **1.648 million real positions** from V7P3R games with perfect class balance (50/50 good/bad).

---

## Model Architecture

### Network Type
**Simple Feed-Forward Neural Network** (no graph augmentation, no attention mechanisms)

### Layer Structure
```python
PositionEvaluator(
    input_dim=19,
    hidden_dims=[512, 256, 128],
    dropout=0.3
)
```

**Detailed Architecture**:
```
Input: 19 features
    ↓
Linear(19 → 512)
    ↓
BatchNorm1d(512)
    ↓
ReLU()
    ↓
Dropout(0.3)
    ↓
Linear(512 → 256)
    ↓
BatchNorm1d(256)
    ↓
ReLU()
    ↓
Dropout(0.3)
    ↓
Linear(256 → 128)
    ↓
BatchNorm1d(128)
    ↓
ReLU()
    ↓
Dropout(0.3)
    ↓
Linear(128 → 1)
    ↓
Sigmoid()
    ↓
Output: P(good) ∈ [0, 1]
```

**Total Parameters**: ~206,209 parameters
- Layer 1: 19 × 512 + 512 = 10,240
- Layer 2: 512 × 256 + 256 = 131,328
- Layer 3: 256 × 128 + 128 = 32,896
- Output: 128 × 1 + 1 = 129
- BatchNorm params: ~3 × (512 + 256 + 128) = 2,688
- **Total**: ~177,281 trainable parameters

### Activation Functions
- **Hidden Layers**: ReLU (Rectified Linear Unit)
- **Output Layer**: Sigmoid (binary classification)

### Regularization
- **Dropout**: 0.3 (30% dropout after each hidden layer)
- **Batch Normalization**: Applied after each linear layer before activation
- **Early Stopping**: Best model saved based on validation F1 score

---

## Feature Engineering

### Fast 19-Dimensional Feature Vector

**Feature extraction function**: `extract_fast_features(fen: str) -> List[float]`

#### Feature Breakdown:

**1. Piece Counts (12 features)**
- F01: White Pawns (0-8)
- F02: White Knights (0-10)
- F03: White Bishops (0-10)
- F04: White Rooks (0-10)
- F05: White Queens (0-9)
- F06: White Kings (0-1)
- F07: Black Pawns (0-8)
- F08: Black Knights (0-10)
- F09: Black Bishops (0-10)
- F10: Black Rooks (0-10)
- F11: Black Queens (0-9)
- F12: Black Kings (0-1)

**2. Material Balance (1 feature)**
- F13: Material imbalance (White - Black material)
  - Piece values: Pawn=1, Knight=3, Bishop=3, Rook=5, Queen=9
  - Range: typically -39 to +39

**3. Positional Features (4 features)**
- F14: Side to move (1=White, 0=Black)
- F15: White kingside castling rights (1=yes, 0=no)
- F16: White queenside castling rights (1=yes, 0=no)
- F17: In check (1=yes, 0=no)

**4. Mobility Features (2 features)**
- F18: Legal moves for current player (0-100+)
- F19: Legal moves for opponent (0-100+)

### Feature Preprocessing

**Normalization**: StandardScaler from sklearn
- Fit on training set
- Transform both train and validation sets
- Scaler saved in model checkpoint for inference

**No Missing Values**: All features guaranteed to be numeric (extracted directly from FEN via chess library)

**No Feature Selection**: All 19 features used (simple, interpretable feature set)

---

## Training Dataset

### Dataset Composition

**Total Positions**: 1,648,000
- **GOOD positions**: 824,000 (label=1)
- **BAD positions**: 824,000 (label=0)
- **Class balance**: Perfect 50/50 split

### Data Sources

**GOOD Positions** (`good_positions.jsonl`):
- Source: V7P3R engine self-play games
- Selection criteria: Positions where V7P3R played strong moves
- Original size: 5.7M positions (sampled 824k for balance)

**BAD Positions** (`bad_positions_massive.jsonl`):
- Source: V7P3R, C0BR4, human games (510 PGN files total)
- Selection criteria: Positions BEFORE blunders (eval drops ≥50cp)
- Mining process: Detected 824k mistakes across all game sources
- Weighting by severity:
  - Small mistake (50-99cp): weight 0.5
  - Mistake (100-199cp): weight 1.0
  - Blunder (200-299cp): weight 2.0
  - Major blunder (300+cp): weight 3.0

### Data Split

**Training Set**: 1,318,400 positions (80%)
- GOOD: 659,200
- BAD: 659,200

**Validation Set**: 329,600 positions (20%)
- GOOD: 164,800
- BAD: 164,800

**Stratified Split**: Maintained 50/50 class balance in both train and validation

**Random Seed**: 42 (reproducible splits)

---

## Training Configuration

### Hyperparameters

```python
CONFIG = {
    'epochs': 20,
    'batch_size': 512,
    'learning_rate': 0.001,
    'hidden_dims': [512, 256, 128],
    'dropout': 0.3,
    'train_val_split': 0.8,
    'random_seed': 42,
    'max_positions': 1_648_000,
}
```

### Optimizer
- **Type**: Adam (Adaptive Moment Estimation)
- **Learning Rate**: 0.001 (default)
- **Betas**: (0.9, 0.999)
- **Weight Decay**: 0 (no L2 regularization - using dropout instead)

### Loss Function
**Weighted Binary Cross-Entropy Loss**

```python
criterion = nn.BCELoss(reduction='none')

# Per-sample loss
loss_per_sample = criterion(outputs, labels)

# Apply position weights (based on blunder severity)
weighted_loss = (loss_per_sample * weights).mean()
```

**Weighting Strategy**:
- GOOD positions: weight = 1.0 (uniform)
- BAD positions: weight = 0.5 to 3.0 (based on eval drop severity)

### Training Loop
- **Epochs**: 20 total
- **Batch processing**: Mini-batch gradient descent
- **Shuffle**: Yes (every epoch)
- **Model saving**: Best validation F1 score (early stopping)

---

## Performance Results

### Final Metrics (Epoch 18 - Best Model)

| Metric | Value | Target | Delta |
|--------|-------|--------|-------|
| **F1 Score** | **87.76%** | ≥82% | **+5.76%** ✅ |
| **Accuracy** | **88.31%** | ≥85% | **+3.31%** ✅ |
| **Precision** | **92.08%** | ≥80% | **+12.08%** ✅ |
| **Recall** | **83.82%** | ≥80% | **+3.82%** ✅ |

**Training Loss**: 0.4008 (epoch 18)  
**Validation Loss**: 0.3813 (epoch 18)  

### Learning Curve

```
Epoch  | Train Loss | Val Loss | Val Acc | Val Prec | Val Rec | Val F1
-------|------------|----------|---------|----------|---------|--------
  1    | 0.4663     | 0.4251   | 86.51%  | 91.36%   | 80.64%  | 85.67%
  5    | 0.4234     | 0.4036   | 87.46%  | 91.45%   | 82.64%  | 86.82%
 10    | 0.4109     | 0.3895   | 87.84%  | 92.00%   | 82.90%  | 87.21%
 15    | 0.4034     | 0.3828   | 88.09%  | 92.20%   | 83.22%  | 87.48%
 18    | 0.4008     | 0.3813   | 88.31%  | 92.08%   | 83.82%  | 87.76% ⭐
 20    | 0.3988     | 0.3777   | 87.88%  | 92.76%   | 82.16%  | 87.14%
```

**Best Epoch**: 18 (highest validation F1 score)

### Interpretation

**Precision 92.08%**: When model predicts "GOOD position," it's correct 92% of the time.
- Very reliable for finding promising positions
- Low false positive rate (only 8% wrong "good" predictions)

**Recall 83.82%**: Model catches 84% of all good positions
- Misses ~16% of good opportunities
- Conservative but accurate

**F1 87.76%**: Excellent balanced performance
- Significantly exceeds target (82%)
- Well-balanced precision/recall trade-off

**No Overfitting**: Validation loss decreasing consistently
- Training and validation losses track closely
- Good generalization to unseen data

---

## Model Deployment

### Saved Model File

**Location**: `models/position_evaluator_best.pth`

**Checkpoint Contents**:
```python
checkpoint = {
    'epoch': 18,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler': StandardScaler (fitted),
    'config': CONFIG,
    'val_f1': 0.8776,
}
```

### Loading the Model

```python
import torch
import pickle
from pathlib import Path

# Load checkpoint
checkpoint_path = Path("models/position_evaluator_best.pth")
checkpoint = torch.load(checkpoint_path)

# Restore scaler
scaler = checkpoint['scaler']

# Rebuild model
from train_balanced import PositionEvaluator
model = PositionEvaluator(
    input_dim=19,
    hidden_dims=[512, 256, 128],
    dropout=0.3
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Loaded model from epoch {checkpoint['epoch']}")
print(f"Validation F1: {checkpoint['val_f1']:.4f}")
```

### Inference Example

```python
import chess
from train_balanced import extract_fast_features

def evaluate_position(fen: str) -> float:
    """
    Evaluate a chess position.
    
    Args:
        fen: FEN string of position to evaluate
        
    Returns:
        Probability that position is GOOD (0.0 to 1.0)
    """
    # Extract features
    features = extract_fast_features(fen)
    
    # Normalize
    features_normalized = scaler.transform([features])
    
    # Convert to tensor
    features_tensor = torch.FloatTensor(features_normalized)
    
    # Predict
    with torch.no_grad():
        prob_good = model(features_tensor).item()
    
    return prob_good

# Example usage
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
score = evaluate_position(fen)
print(f"P(good position) = {score:.2%}")

if score >= 0.5:
    print("✅ Recommendation: GOOD position")
else:
    print("❌ Recommendation: BAD position (avoid)")
```

---

## Comparison to Previous Architectures

### V5.0 Architecture (DEPRECATED)
- **Features**: 325 dimensions
- **Network**: Graph-augmented with transposition attention
- **Complexity**: Very high (millions of parameters)
- **Training**: Slow, complex data pipeline
- **Status**: Abandoned due to feature calculation bottleneck

### V6.0 (Early Attempt - FAILED)
- **Features**: 76-92 dimensions
- **Network**: Graph-augmented policy network
- **Issue**: Feature calculation too slow, hung on large datasets
- **Status**: Abandoned

### V6.1 Production (CURRENT - SUCCESS)
- **Features**: 19 dimensions (fast extraction)
- **Network**: Simple feed-forward with batch norm
- **Complexity**: Low (~200k parameters)
- **Training**: Fast (1.6M positions in ~45 minutes)
- **Status**: ✅ **PRODUCTION**

**Key Insight**: Simpler is better! Stripped down to essential features and architecture, achieved superior results.

---

## Next Steps

### Stage 2: Move Selection
With Stage 1 evaluator proven, move to Stage 2:
1. **Input**: Current position + list of legal moves
2. **Process**: Evaluate each candidate move's resulting position using Stage 1 evaluator
3. **Output**: Ranked list of moves (highest P(good) first)
4. **Integration**: Replace or augment V7P3R's current move selection

### Potential Improvements (Future)
- **More data**: Expand to full 5.7M good positions (currently using 824k)
- **Ensemble**: Train multiple models with different random seeds, average predictions
- **Feature expansion**: Add 5-10 more tactical features (pins, forks, checks)
- **Multi-class**: Instead of binary GOOD/BAD, predict 5 quality grades
- **GPU training**: Enable CUDA for faster training on larger datasets

---

## File Locations

**Training Script**: `scripts/stage1/train_balanced.py`  
**Model File**: `models/position_evaluator_best.pth`  
**Data Files**:
- Good: `data/stage1/good_positions.jsonl` (5.7M positions)
- Bad: `data/stage1/bad_positions_massive.jsonl` (824k positions)

**Mining Script**: `scripts/stage1/mine_bad_positions_massive.py`  
**Test Script**: `scripts/stage1/test_massive_miner.py`

---

## Conclusion

The V7P3R AI v6.1 Stage 1 Position Evaluator is a **production-ready model** that successfully learned to distinguish good chess positions from bad positions with **87.76% F1 score** on a massive dataset of 1.6 million real game positions.

The model's simplicity (19 features, 200k parameters) makes it fast and deployable, while its performance (exceeding all targets) proves that effective chess position evaluation doesn't require complex architectures—just the right features and balanced training data.

**Ready for Stage 2 integration** into V7P3R's tactical decision-making pipeline. 🎯
