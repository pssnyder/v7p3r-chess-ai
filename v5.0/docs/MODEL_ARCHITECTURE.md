# V7P3R AI v5.0/v5.1 - Model Architecture Design

## Overview

**Model Type**: Dual-Head Neural Network (AlphaZero-style supervised learning)  
**Framework**: PyTorch  
**Current Version**: v5.1 (expanded features)  
**Input**: 92+ dimensional feature vector (heuristics as observations, expandable post one-hot encoding)  
**Outputs**: 
- **Policy Head**: Move quality classification (0-5 grades)
- **Value Head**: Position evaluation regression (-1 to +1)

### Version Comparison

| Version | Raw Features | Post-Encoding | Status | Policy Accuracy | Notes |
|---------|-------------|---------------|--------|-----------------|-------|
| **v5.0** | 26 | 26 | ✅ Baseline | 49.12% | Binary classification problem (only predicts grades 0/5) |
| **v5.1** | 92+ | 98-105 | 🔄 Current | **Target: 54-57%** | Expanded features fix binary classification |

**Key v5.1 Changes:**
- ✅ **Tactical features** (F040-F049): Pins, forks, skewers, discovered attacks, trapped pieces, back rank threats
- ✅ **Rook placement** (F060-F064): Open/semi-open files, 7th rank, connected rooks, activity score
- ✅ **Enhanced pawn structure** (F024-F029): Backward pawns, chains, islands, advanced pawns
- ✅ **Multi-move context** (F100-F114): Top-5 Stockfish evals, eval gaps, move diversity - **CRITICAL for fixing binary classification**
- ✅ **Class weights**: Forces model to learn rare grades (1-4) with weights `[1.0, 5.0, 3.5, 2.5, 1.8, 1.0]`

---

## Architecture Diagram

### v5.1 Architecture (Current - Expanded Features)

```
Input (98-105 features after one-hot encoding)
    ↓
    [92+ raw features: tactical, rook placement, pawn structure, multi-move context]
    [One-hot: game_phase (3), material_cat (5), move_types (3×4=12)]
    ↓
Shared Embedding Layers (DESIGNED FOR FUTURE EXPANSION)
    ├── Dense(256) → BatchNorm → ReLU → Dropout(0.3)
    ├── Dense(256) → BatchNorm → ReLU → Dropout(0.3) [+ Residual]
    ├── Dense(128) → BatchNorm → ReLU → Dropout(0.3) [+ Residual]
    └── Dense(64) → BatchNorm → ReLU → Dropout(0.2)
    ↓
   [Fork into dual heads]
    ↓                        ↓
Policy Head              Value Head
├── Dense(64) → ReLU    ├── Dense(32) → ReLU
├── Dropout(0.2)        ├── Dropout(0.2)
└── Dense(6) → Softmax  └── Dense(1) → Tanh
    ↓                        ↓
6-class probs           Position eval
(WITH CLASS WEIGHTS)    (advantage)
[1.0, 5.0, 3.5, 2.5, 1.8, 1.0]
```

**v5.1 Design Philosophy**: 
- **Same architecture depth** (256→256→128→64) - designed for expansion from day 1
- **Increased input capacity**: 26 → 98-105 features (no architectural changes needed)
- **Class weights**: Force model to learn grades 1-4 (not just 0 and 5)
- **Multi-move context**: F100-F114 provide full top-5 Stockfish analysis (fixes binary classification)

### v5.0 Architecture (Baseline)

```
Input (26 features - limited featureset)
    ↓
    [6 numerical, 12 boolean, 3 game_phase, 5 material_cat]
    ↓
[Same shared embedding layers - 256→256→128→64]
    ↓
[Same dual heads - policy (6 classes) + value (regression)]
    ↓
Binary classification problem:
  - Only predicts grades 0 or 5
  - Insufficient features to distinguish 2nd/3rd/4th best moves
  - NO class weights (model biased toward common grades)
```

**Design Philosophy**: Deeper architecture (256→256→128→64) provides capacity for future featureset expansion without architectural disruption. Allows smooth model transfer if heuristics grow from 26 to 100+ features.

---

## Detailed Layer Specifications

### Input Layer

#### v5.1 (Current - Expanded Features)

**Raw Features**: 92+ before one-hot encoding  
**Post-Encoding**: 98-105 features (depends on categorical cardinality)

**Feature Breakdown:**

**Numerical Features (~52)**:
- Core position: material_balance_cp, total_piece_count
- Pawn structure: passed_pawn_count (×2), backward_pawn_count (×2), pawn_chain_length (×2), advanced_pawn_count (×2), pawn_island_count (×2)
- Piece activity: piece_mobility (×2), pieces_on_strong_squares (×2)
- Tactical: pieces_under_attack (×2), en_prise_value (×2), trapped_piece_count (×2)
- Rook placement: rooks_on_open_files (×2), rooks_on_semi_open_files (×2), rook_activity_score (×2)
- Knights: knight_outposts (×2), knight_mobility_avg (×2)
- Center control: center_pawn_count (×2), center_control_score (×2), space_advantage (×2)
- Development: pieces_developed (×2)
- **Multi-move context (F100-F114 - CRITICAL)**: best_move_eval_cp, second_move_eval_cp, third_move_eval_cp, fourth_move_eval_cp, fifth_move_eval_cp, eval_gap_best_to_second, eval_gap_second_to_third, v7p3r_move_eval_cp, v7p3r_eval_loss, move_diversity_score, position_sharpness, alternative_move_quality

**Boolean Features (~34)**:
- King safety: king_castled (×2), king_has_pawn_shield (×2), king_under_attack (×2)
- Pawn structure: has_passed_pawns (×2), has_doubled_pawns (×2), has_isolated_pawns (×2)
- Piece pairs: has_bishop_pair (×2)
- Tactical: has_hanging_pieces (×2), has_fork_threat (×2), has_pin (×2), has_skewer (×2), has_discovered_attack (×2), back_rank_threat (×2)
- Move context: is_capture, is_check, is_promotion, is_castling
- Rook placement: rook_on_7th_rank (×2), connected_rooks (×2)

**Categorical Features (→ One-Hot Encoded)**:
- `game_phase` (3 categories): opening, middlegame, endgame → **3 features**
- `material_advantage_category` (5 categories): balanced, white_advantage, white_winning, black_advantage, black_winning → **5 features**
- `best_move_type` (4 categories): quiet, capture, check, promotion → **4 features**
- `second_move_type` (4 categories) → **4 features**
- `v7p3r_move_type` (4 categories) → **4 features**

**Total: ~52 numerical + ~34 boolean + 20 one-hot = 106 features**

**Input shape**: `(batch_size, 106)`  
**Data type**: `torch.float32`

---

#### v5.0 (Baseline - Limited Features)

**Total Features**: 26 (all features)

**Feature Breakdown:**

**Numerical Features (6)**:
- material_balance_cp
- total_piece_count
- white_piece_mobility, black_piece_mobility
- white_pieces_on_strong_squares, black_pieces_on_strong_squares

**Boolean Features (12)**:
- white_king_castled, black_king_castled
- white_king_has_pawn_shield, black_king_has_pawn_shield
- white_king_under_attack, black_king_under_attack
- white_has_bishop_pair, black_has_bishop_pair
- is_capture, is_check, is_promotion, is_castling

**Categorical Features (→ One-Hot)**:
- `game_phase` (3 categories) → **3 features**
- `material_advantage_category` (5 categories) → **5 features**

**Total: 6 numerical + 12 boolean + 8 one-hot = 26 features**

**Input shape**: `(batch_size, 26)`

**❌ Problem**: Insufficient features to distinguish tactical nuances between 2nd/3rd/4th best moves → binary classification

### Shared Embedding Network

**Purpose**: Learn abstract position representations from heuristics  
**Design**: Deeper architecture supports featureset expansion (v5.0: 26 → v5.1: 106 features)

**Layer 1** - Wide feature extraction
- **Input**: 26 (v5.0) or 106 (v5.1)
- **Output**: 256
- **Batch Normalization**: Yes (normalize activations)
- **Activation**: ReLU
- **Dropout**: 0.3
- **Initialization**: He initialization (for ReLU)
- **Note**: Same output dimension regardless of input size - architecture designed for expansion

**Layer 2** - Deep feature refinement with residual connection
- **Input**: 256
- **Output**: 256
- **Batch Normalization**: Yes
- **Activation**: ReLU
- **Dropout**: 0.3
- **Residual Connection**: Yes (direct skip - same dimensions)

**Layer 3** - Feature compression with residual connection
- **Input**: 256
- **Output**: 128
- **Batch Normalization**: Yes
- **Activation**: ReLU
- **Dropout**: 0.3
- **Residual Connection**: Yes (with projection layer: 256→128)

**Layer 4** - Final compressed representation
- **Input**: 128
- **Output**: 64
- **Batch Normalization**: Yes
- **Activation**: ReLU
- **Dropout**: 0.2
- **Purpose**: Compact shared representation for dual heads (same for both versions)

---

### 🔗 Residual Connections Explained

**What are they?**  
Residual connections (from ResNet) add the *input* of a layer directly to its *output*, creating a "skip connection":

```
Traditional layer:
    x → [Dense → BatchNorm → ReLU] → y

Residual layer:
    x → [Dense → BatchNorm → ReLU] → y
    └────────────────────────────────┘
           x + y (element-wise sum)
```

**Why use them?**
1. **Gradient flow**: Prevents vanishing gradients in deep networks
2. **Easier training**: Allows layers to learn "refinements" instead of full transformations
3. **Future expansion**: When adding features, model can keep old patterns via skip connections

**When do we need a projection?**  
When input/output dimensions differ (e.g., 256→128), we need a projection layer:
```python
# Layer 3: 256 → 128 needs projection
residual = Dense(128)(x)  # Project 256 → 128
y = Dense(128)(x)
output = residual + y      # Now dimensions match
```

**Implementation in V7P3R AI**:
- Layer 2: 256→256 (same dims, direct skip)
- Layer 3: 256→128 (different dims, needs projection)

### Policy Head (Move Quality Classification)

**Purpose**: Predict move quality grade (0-5)

**Layer 1** - Feature extraction
- **Input**: 64
- **Output**: 64
- **Activation**: ReLU
- **Dropout**: 0.2

**Initial weights (adjustable via config):
alpha = 1.0  # Policy loss weight
beta = 0.1   # Value loss weight (scaled down due to loss magnitude differences)
```

**Tuning Strategy**: Start with `beta=0.1`, monitor validation metrics:
- If policy overfits but value underfits → increase beta to 0.2
- If value overfits but policy underfits → decrease beta to 0.05
- Adjust in config.yaml without code changes

### Policy Loss - CrossEntropyLoss (with Class Weights in v5.1)

#### v5.1 (Current - Class Weighted)
```python
import torch
import torch.nn as nn

# Class weights to handle imbalance and fix binary classification
class_weights = torch.tensor([1.0, 5.0, 3.5, 2.5, 1.8, 1.0]).to(device)

policy_loss = nn.CrossEntropyLoss(weight=class_weights)(
    policy_logits,      # Shape: (batch, 6)
    policy_targets      # Shape: (batch,) - integer grades 0-5
)
```

**Why Class Weights?**  
v5.0 suffered from **binary classification** - model only predicted grades 0 or 5. This happened because:
1. **Class imbalance**: Grades 0 (24.6%) and 5 (40.6%) dominate the dataset
2. **Insufficient features**: 26 features couldn't distinguish 2nd vs 3rd vs 4th best moves
3. **Model bias**: Learned to predict common grades (0/5) and ignore rare grades (1-4)

**Class Weight Strategy**:
- **Grade 0** (24.6%): weight = 1.0 (baseline)
- **Grade 1** (4.5%): weight = 5.0 (heavily boost rare grade)
- **Grade 2** (6.2%): weight = 3.5
- **Grade 3** (9.0%): weight = 2.5
- **Grade 4** (15.1%): weight = 1.8
- **Grade 5** (40.6%): weight = 1.0 (baseline, most common)

This forces the model to **pay attention to rare grades** instead of ignoring them.

#### v5.0 (Baseline - No Class Weights)
```python
policy_loss = nn.CrossEntropyLoss()(
    policy_logits,      # Shape: (batch, 6)
    policy_targets      # Shape: (batch,) - integer grades 0-5
)
```

**Result**: Binary classification - only predicts 0 or 5, ignores grades 1-4 entirely.

### Value Loss - HuberLoss ✅ (User Selected)
```python
value_loss = nn.HuberLoss(delta=0.5)(
    value_predictions,  # Shape: (batch, 1)
    value_targets       # Shape: (batch, 1) - normalized evals
)
```

**Why Huber?** (User's choice confirmed)
- ✅ More robust to outlier evaluations (e.g., mate scores, tactical blunders)
- Behaves like MSE for small errors (smooth gradients)
- Behaves like L1 for large errors (robust to outliers)
- `delta=0.5` balances both regimes for chess evaluation)
- **Activation**: Tanh (bounds output to [-1, 1])
- **Loss**: MSELoss or HuberLoss

**Output Interpretation**:
```python
# Example output: 0.35
# Interpretation: +350cp advantage for White
# Range: -1.0 (Black winning) to +1.0 (White winning)
```

---

## Loss Function

### Combined Loss (Weighted Multi-Task)

```python
total_loss = alpha * policy_loss + beta * value_loss

# Recommended weights:
alpha = 1.0  # Policy loss weight
beta = 0.1   # Value loss weight (scaled down due to MSE magnitude)
```

### Policy Loss - CrossEntropyLoss
```python
import torch.nn as nn

policy_loss = nn.CrossEntropyLoss()(
    policy_logits,      # Shape: (batch, 6)
    policy_targets      # Shape: (batch,) - integer grades 0-5
)
```

### Value Loss - HuberLoss (Robust to Outliers)
```python
value_loss = nn.HuberLoss(delta=0.5)(
    value_predictions,  # Shape: (batch, 1)
    value_targets       # Shape: (batch, 1) - normalized evals
)
```

**Why Huber over MSE?**
- More robust to outlier evaluations (e.g., mate scores)
- Behaves like MSE for small errors, L1 for large errors
- `delta=0.5` balances both regimes

---

## PyTResidualBlock(nn.Module):
    """Residual block with batch normalization"""
    
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super(ResidualBlock, self).__init__()
        
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Projection layer if dimensions don't match
        self.projection = None
        if in_dim != out_dim:
            self.projection = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        identity = x
        
        # Main path
        out = self.linear(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Residual connection
        if self.projection is not None:
            identity = self.projection(identity)
        
        out = out + identity  # Skip connection
        return out


class V7P3R_AI_v5(nn.Module):
    """
    V7P3R AI v5.0 - Dual-head neural network with residual connections
    Learns move quality and position evaluation from V7P3R game history
    
    Architecture designed for future expansion:
    - Deep network (256→256→128→64) accommodates feature growth
    - Residual connections enable smooth gradient flow
    - Batch normalization for stable training
    """
    
    def __init__(self, 
                 input_dim=26,
                 shared_dims=[256, 256, 128, 64],
                 policy_hidden=64,
                 value_hidden=32,
                 dropout=0.3,
                 use_residuals=True):
        super(V7P3R_AI_v5, self).__init__()
        
        self.use_residuals = use_residuals
        
        # Initial projection to shared dimension
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, shared_dims[0]),
            nn.BatchNorm1d(shared_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Shared embedding network with residual blocks
        if use_residuals:
            self.shared_blocks = nn.ModuleList([
                ResidualBlock(shared_dims[i], shared_dims[i+1], dropout)
                for i in range(len(shared_dims) - 1)
            ])
        else:
            # Fallback to sequential (for ablation studies)
            layers = []
            for i in range(len(shared_dims) - 1):
                layers.extend([
                    nn.Linear(shared_dims[i], shared_dims[i+1]),
                    nn.BatchNorm1d(shared_dims[i+1]),
                    nn.ReLU(),
                    nn.Dropout(dropout if i < len(shared_dims) - 2 else dropout * 0.7)
                ])
            self.shared_sequential = nn.Sequential(*layers)
        
        # Policy head (move quality classification)
        self.policy = nn.Sequential(
            nn.Linear(shared_dims[-1], policy_hidden),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(policy_hidden, 6)  # 6 classes (grades 0-5)
        )
        
        # Value head (position evaluation)
        self.value = nn.Sequential(
            nn.Linear(shared_dims[-1], value_hidden),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(value_hidden, 1),
            nn.Tanh()  # Bound to [-1, 1] ✅ User selected
        )
    
    def forward(self, x):
        """
        Forward pass through dual-head network
        
        Args:
            x: Input features
               - v5.0: (batch_size, 26)
               - v5.1: (batch_size, 106)
            
        Returns:
            policy_logits: (batch_size, 6) - unnormalized class scores
            value: (batch_size, 1) - position evaluation [-1, 1]
        """
        # Project input to shared dimension
        x = self.input_proj(x)
        
        # Shared embedding with residuals
        if self.use_residuals:
            for block in self.shared_blocks:
                x = block(x)
        else:
            x = self.shared_sequential(x)
        
        # Dual heads
        policy_logits = self.policy(x)
        value = self.value(x)
        
        return policy_logits, value


# Model instantiation

# v5.1 (current - expanded features)
model_v5_1 = V7P3R_AI_v5(
    input_dim=106,             # Expanded from 26 to 106 (post one-hot encoding)
    shared_dims=[256, 256, 128, 64],  # Same depth - designed for expansion
    policy_hidden=64,
    value_hidden=32,
    dropout=0.3,
    use_residuals=True         # Enable residual connections
)

# v5.0 (baseline)
model_v5_0 = V7P3R_AI_v5(
    input_dim=26,              # Original feature count
    shared_dims=[256, 256, 128, 64],
    policy_hidden=64,
    value_hidden=32,
    dropout=0.3,
    use_residuals=True
)

# Parameter count comparison
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

v5_1_total, v5_1_trainable = count_parameters(model_v5_1)
v5_0_total, v5_0_trainable = count_parameters(model_v5_0)

print("=" * 60)
print("V7P3R AI - Parameter Count Comparison")
print("=" * 60)
print(f"v5.0 (26 features):")
print(f"  Total parameters:      {v5_0_total:,}")
print(f"  Trainable parameters:  {v5_0_trainable:,}")
print(f"  Model size:            {v5_0_total / 1000:.1f}k parameters")
print(f"  Memory footprint:      ~{v5_0_total * 4 / 1024 / 1024:.2f} MB")
print()
print(f"v5.1 (106 features):")
print(f"  Total parameters:      {v5_1_total:,}")
print(f"  Trainable parameters:  {v5_1_trainable:,}")
print(f"  Model size:            {v5_1_total / 1000:.1f}k parameters")
print(f"  Memory footprint:      ~{v5_1_total * 4 / 1024 / 1024:.2f} MB")
print()
print(f"Increase: +{v5_1_total - v5_0_total:,} parameters (+{100*(v5_1_total - v5_0_total)/v5_0_total:.1f}%)")
print("=" * 60)

# Expected output:
# v5.0: ~164,491 parameters (0.62 MB)
# v5.1: ~208,000 parameters (0.79 MB) - increase mostly from input layer (26→106)
```

---

## Training Configuration

### Optimizer - AdamW

```python
import torch.optim as optim

optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3,              # Initial learning rate
    weight_decay=1e-4,    # L2 regularization
    betas=(0.9, 0.999)
)
```

**Why AdamW?**
- Improved weight decay handling vs Adam
- Better generalization
- Standard choice for supervised learning

### Learning Rate Scheduler

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,           # Reduce by 50%
    patience=5,           # After 5 epochs without improvement
    min_lr=1e-6
)
```

### Training Hyperparameters

```python
# v5.1 Configuration (Current - Expanded Features)
config_v5_1 = {
    # Model Architecture
    'input_dim': 106,           # ✅ Expanded from 26 to 106 features
    'shared_dims': [256, 256, 128, 64],  # Same depth - designed for expansion
    'policy_hidden': 64,
    'value_hidden': 32,
    'use_residuals': True,
    
    # Data
    'batch_size': 256,
    'num_workers': 4,
    
    # Training
    'epochs': 100,
    'early_stopping_patience': 15,
    
    # Loss weights
    'policy_weight': 1.0,
    'value_weight': 0.1,
    
    # Class weights (NEW - fixes binary classification)
    'use_class_weights': True,
    'class_weights': [1.0, 5.0, 3.5, 2.5, 1.8, 1.0],  # Force learning of rare grades
    
    # Optimizer
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    
    # Scheduler
    'lr_patience': 5,
    'lr_factor': 0.5,
    
    # Regularization
    'dropout': 0.3,
    'batch_norm': True,
    
    # Loss functions
    'value_loss': 'huber',
    'huber_delta': 0.5,
    
    # Evaluation
    'eval_every': 1,
    'save_best_only': True
}

# v5.0 Configuration (Baseline)
config_v5_0 = {
    # Model Architecture
    'input_dim': 26,            # Limited features
    'shared_dims': [256, 256, 128, 64],
    'policy_hidden': 64,
    'value_hidden': 32,
    'use_residuals': True,
    
    # Data
    'batch_size': 256,
    'num_workers': 4,
    
    # Training
    'epochs': 100,
    'early_stopping_patience': 15,
    
    # Loss weights
    'policy_weight': 1.0,
    'value_weight': 0.1,
    
    # Class weights
    'use_class_weights': False,  # ❌ Not used - caused binary classification
    
    # Optimizer
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    
    # Scheduler
    'lr_patience': 5,
    'lr_factor': 0.5,
    
    # Regularization
    'dropout': 0.3,
    'batch_norm': True,
    
    # Loss functions
    'value_loss': 'huber',
    'huber_delta': 0.5,
    
    # Evaluation
    'eval_every': 1,
    'save_best_only': True
}
```

---

## DataLoader Implementation

```python
import torch
from torch.utils.data import Dataset, DataLoader

class V7P3RDataset(Dataset):
    """PyTorch dataset for V7P3R training data"""
    
    def __init__(self, X, policy_targets, value_targets):
        self.X = torch.FloatTensor(X)
        self.policy = torch.LongTensor(policy_targets)
        self.value = torch.FloatTensor(value_targets).reshape(-1, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'features': self.X[idx],
            'policy_target': self.policy[idx],
            'value_target': self.value[idx]
        }

# Create datasets
train_dataset = V7P3RDataset(X_train, y_train['policy'], y_train['value'])
val_dataset = V7P3RDataset(X_val, y_val['policy'], y_val['value'])

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True,
    num_workers=4,
    pin_memory=True  # Faster GPU transfer
)

val_loader = DataLoader(
    val_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)
```

---

## Training Loop

```python
def train_epoch(model, loader, optimizer, policy_weight=1.0, value_weight=0.1):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    policy_correct = 0
    total_samples = 0
    
    for batch in loader:
        # Get data
        features = batch['features'].to(device)
        policy_targets = batch['policy_target'].to(device)
        value_targets = batch['value_target'].to(device)
        
        # Forward pass
        policy_logits, value_preds = model(features)
        
        # Compute losses
        policy_loss = nn.CrossEntropyLoss()(policy_logits, policy_targets)
        value_loss = nn.HuberLoss(delta=0.5)(value_preds, value_targets)
        
        loss = policy_weight * policy_loss + value_weight * value_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item() * features.size(0)
        policy_correct += (policy_logits.argmax(1) == policy_targets).sum().item()
        total_samples += features.size(0)
    
    return {
        'loss': total_loss / total_samples,
        'policy_accuracy': policy_correct / total_samples
    }

def evaluate(model, loader, policy_weight=1.0, value_weight=0.1):
    """Evaluate on validation set"""
    model.eval()
    total_loss = 0
    policy_correct = 0
    total_samples = 0
    value_mae = 0
    
    with torch.no_grad():
        for batch in loader:
            features = batch['features'].to(device)
            policy_targets = batch['policy_target'].to(device)
            value_targets = batch['value_target'].to(device)
            
            policy_logits, value_preds = model(features)
            
            policy_loss = nn.CrossEntropyLoss()(policy_logits, policy_targets)
            value_loss = nn.HuberLoss(delta=0.5)(value_preds, value_targets)
            
            loss = policy_weight * policy_loss + value_weight * value_loss
            
            total_loss += loss.item() * features.size(0)
            policy_correct += (policy_logits.argmax(1) == policy_targets).sum().item()
            value_mae += torch.abs(value_preds - value_targets).sum().item()
            total_samples += features.size(0)
    
    return {
        'loss': total_loss / total_samples,
        'policy_accuracy': policy_correct / total_samples,
        'value_mae': value_mae / total_samples
    }
```

---

## Model Evaluation Metrics

### Policy Head Metrics
- **Accuracy**: % of exact grade predictions
- **Top-2 Accuracy**: % within best 2 grades
- **Confusion Matrix**: See where model confuses grades
- **Per-Grade F1 Score**: Performance on each quality level

### Value Head Metrics
- **MAE** (Mean Absolute Error): Average |prediction - target|
- **MSE** (Mean Squared Error): Penalizes large errors
- **Correlation**: Pearson correlation with Stockfish evals

### Combined Metrics
- **Total Loss**: Weighted combination
- **Training Time**: Seconds per epoch
- **GPU Memory**: Peak usage

---

## Expected Performance Targets

### Baseline (Random)
- **Policy Accuracy**: 16.7% (1/6 random guess)
- **Value MAE**: ~0.3 (mean absolute eval error)

### Target Performance (After Training)
- **Policy Accuracy**: >50% (exact grade match)
- **Top-2 Accuracy**: >75% (within 1 grade)
- **Value MAE**: <0.15 (within 1500cp on average)

### Stretch Goals
- **Policy Accuracy**: >65%
- **Top-2 Accuracy**: >85%
- **Value MAE**: <0.10 (within 1000cp)

---

## ✅ Architecture Decisions (User Confirmed)

1. **Architecture depth**: ✅ **Deeper architecture (256→256→128→64)** - Designed for future featureset expansion (26 → 40+ features) with smooth model transfer
2. **Loss weighting**: ✅ **policy=1.0, value=0.1** - Start optimal, adjustable via config (increase to 0.2 if value underperforms)
3. **Dropout rate**: ✅ **0.3** - Mid-range with room to expand if needed
4. **Batch size**: ✅ **256** - Mid-range, can increase to 512 if pushing model harder
5. **Value activation**: ✅ **Tanh (bounded)** - Keeps data relative and simpler [-1, 1]
6. **Batch normalization**: ✅ **Yes** - Normalize data to keep things clean
7. **Residual connections**: ✅ **Yes** - Enables gradient flow and supports future expansion (see explanation above)
8. **Value loss**: ✅ **Huber** - Robust to outlier evaluations (mate scores, blunders)

---

## 🎯 Design Philosophy Summary

This architecture is **future-proof** and **expansion-ready**:
- **Deep network** (4 shared layers) accommodates feature growth without restructuring
- **Residual connections** enable smooth gradient flow and feature addition
- **Batch normalization** ensures stable training with varied feature scales
- **Configurable hyperparameters** allow tuning without code changes
- **26 → 40+ feature capacity** built-in from day one

When you add new heuristics (e.g., F060-F080 series):
1. Update `input_dim` in config (e.g., 26 → 38)
2. Retrain with new features - architecture handles it automatically
3. Optional: Use transfer learning to preserve learned patterns

---

## Implementation Checklist

- [ ] Create `scripts/preprocess_dataset.py` (save preprocessed arrays)
- [ ] Create `src/model.py` (PyTorch model definition)
- [ ] Create `src/dataset.py` (PyTorch dataset class)
- [ ] Create `src/train.py` (training loop with logging)
- [ ] Create `src/evaluate.py` (evaluation and metrics)
- [ ] Create `configs/training_config.yaml` (hyperparameters)
- [ ] Test model forward pass with sample batch
- [ ] Verify loss computation and gradient flow
- [ ] Run 1-epoch test training to validate pipeline
- [ ] Full training run with early stopping

---

*Last Updated: May 7, 2026 @ 1:05 PM*  
*Ready for user alignment and implementation*
