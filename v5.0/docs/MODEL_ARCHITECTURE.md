# V7P3R AI v5.0 - Model Architecture Design

## Overview

**Model Type**: Dual-Head Neural Network (AlphaZero-style supervised learning)  
**Framework**: PyTorch  
**Input**: 26-dimensional feature vector (heuristics as observations)  
**Outputs**: 
- **Policy Head**: Move quality classification (0-5 grades)
- **Value Head**: Position evaluation regression (-1 to +1)

---

## Architecture Diagram

```
Input (26 features)
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
(move quality)          (advantage)
```

**Design Philosophy**: Deeper architecture (256→256→128→64) provides capacity for future featureset expansion without architectural disruption. Allows smooth model transfer if heuristics grow from 26 to 40+ features.

---

## Detailed Layer Specifications

### Input Layer
- **Dimensions**: 26 features
  - 6 numerical (scaled)
  - 12 boolean (0/1)
  - 3 game phase (one-hot)
  - 5 material category (one-hot)
- **Input shape**: `(batch_size, 26)`
- **Data type**: `torch.float32`

### Shared Embedding Network

**Purpose**: Learn abstract position representations from heuristics  
**Design**: Deeper architecture supports future featureset expansion (26 → 40+ features)

**Layer 1** - Wide feature extraction
- **Input**: 26 (expandable to 40+)
- **Output**: 256
- **Batch Normalization**: Yes (normalize activations)
- **Activation**: ReLU
- **Dropout**: 0.3
- **Initialization**: He initialization (for ReLU)

**Layer 2** - Deep feature refinement with residual connection
- **Input**: 256
- **Output**: 256
- **Batch Normalization**: Yes
- **Activation**: ReLU
- **Dropout**: 0.3
- **Residual Connection**: Yes (see explanation below)

**Layer 3** - Feature compression with residual connection
- **Input**: 256
- **Output**: 128
- **Batch Normalization**: Yes
- **Activation**: ReLU
- **Dropout**: 0.3
- **Residual Connection**: Yes (with projection layer)

**Layer 4** - Final compressed representation
- **Input**: 128
- **Output**: 64
- **Batch Normalization**: Yes
- **Activation**: ReLU
- **Dropout**: 0.2
- **Purpose**: Compact shared representation for dual heads

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

### Policy Loss - CrossEntropyLoss
```python
import torch.nn as nn

policy_loss = nn.CrossEntropyLoss()(
    policy_logits,      # Shape: (batch, 6)
    policy_targets      # Shape: (batch,) - integer grades 0-5
)
```

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
            x: Input features (batch_size, 26) - expandable to 40+
            
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
model = V7P3R_AI_v5(
    input_dim=26,              # Expandable to 40+ features
    shared_dims=[256, 256, 128, 64],  # Deeper for future growth
    policy_hidden=64,
    value_hidden=32,
    dropout=0.3,
    use_residuals=True         # Enable residual connections
)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"\nModel capacity: {total_params / 1000:.1f}k parameters")

# Expected: ~140k-160k parameters (deeper model with residualsters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Expected: ~25k-30k parameters (lightweight model)
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
config = {
    # Model Architecture (designed for expansion)
    'input_dim': 26,            # Current features (expandable to 40+)
    'shared_dims': [256, 256, 128, 64],  # ✅ User approved: Deeper for future growth
    'policy_hidden': 64,
    'value_hidden': 32,
    'use_residuals': True,      # ✅ Residual connections enabled
    
    # Data
    'batch_size': 256,          # ✅ User approved: Mid-range, expandable
    'num_workers': 4,           # For DataLoader
    
    # Training
    'epochs': 100,
    'early_stopping_patience': 15,
    
    # Loss weights (adjustable)
    'policy_weight': 1.0,       # ✅ User approved: Start optimal, adjust as needed
    'value_weight': 0.1,        # ✅ Adjustable: increase to 0.2 if value underperforms
    
    # Optimizer
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    
    # Scheduler
    'lr_patience': 5,
    'lr_factor': 0.5,
    
    # Regularization
    'dropout': 0.3,             # ✅ User approved: Mid-range, expandable
    'batch_norm': True,         # ✅ User approved: Normalize for clean data
    
    # Loss functions
    'value_loss': 'huber',      # ✅ User selected: Robust to outliers
    'huber_delta': 0.5,
    
    # Evaluation
    'eval_every': 1,            # Evaluate every N epochs
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
