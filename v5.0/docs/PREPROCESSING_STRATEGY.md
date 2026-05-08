# V7P3R AI v5.0 - Preprocessing Strategy

## Summary

Based on dataset analysis of 230,930 positions, here's the preprocessing plan:

---

## Dataset Health

✅ **No null values** - All 21 feature fields are fully populated  
✅ **Consistent schema** - All records follow unified format  
✅ **230,930 positions** ready for preprocessing  

---

## Feature Categories

### 1. Metadata (NOT for training)
- **F001_position_fen**: Position identifier (tracking only)
- Should be **excluded from training features**

### 2. Numerical Features (6 total) → StandardScaler

| Feature | Range | Notes |
|---------|-------|-------|
| F003_material_balance_cp | [-5400, 4400] | Large range, needs scaling |
| F005_total_piece_count | [3, 32] | Count feature |
| F030_white_piece_mobility | [0, 89] | Mobility score |
| F030_black_piece_mobility | [0, 108] | Mobility score (large range) |
| F031_white_pieces_on_strong_squares | [0, 4] | Count feature |
| F031_black_pieces_on_strong_squares | [0, 3] | Count feature |

**Preprocessing**: StandardScaler (zero mean, unit variance)
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numerical_scaled = scaler.fit_transform(numerical_features)
```

### 3. Boolean Features (12 total) → Already 0/1

- F010_white_king_castled
- F010_black_king_castled
- F011_white_king_has_pawn_shield
- F011_black_king_has_pawn_shield
- F012_white_king_under_attack
- F012_black_king_under_attack
- F032_white_has_bishop_pair
- F032_black_has_bishop_pair
- F050_is_capture
- F051_is_check
- F052_is_castling
- F053_is_promotion

**Preprocessing**: ✅ None needed (already binary)

### 4. Categorical Features → One-Hot Encoding

#### F002_game_phase (3 values) → 3 binary features
- `opening` → [1, 0, 0]
- `middlegame` → [0, 1, 0]
- `endgame` → [0, 0, 1]

**Preprocessing**:
```python
from sklearn.preprocessing import OneHotEncoder
phase_encoder = OneHotEncoder(sparse=False)
phase_encoded = phase_encoder.fit_transform(game_phase.reshape(-1, 1))
# Output: 3 columns
```

#### F004_material_advantage_category (5 values) → 5 binary features
- `black_winning` → [1, 0, 0, 0, 0]
- `black_advantage` → [0, 1, 0, 0, 0]
- `equal` → [0, 0, 1, 0, 0]
- `white_advantage` → [0, 0, 0, 1, 0]
- `white_winning` → [0, 0, 0, 0, 1]

**Preprocessing**:
```python
material_encoder = OneHotEncoder(sparse=False)
material_encoded = material_encoder.fit_transform(material_category.reshape(-1, 1))
# Output: 5 columns
```

---

## Final Feature Vector Dimensions

| Feature Type | Count | Notes |
|--------------|-------|-------|
| Numerical (scaled) | 6 | StandardScaler applied |
| Boolean (raw) | 12 | Already 0/1 |
| Game phase (one-hot) | 3 | One-hot encoded |
| Material category (one-hot) | 5 | One-hot encoded |
| **TOTAL INPUT FEATURES** | **26** | Fed to neural network |

**Concatenation Order**:
```
[numerical_6, boolean_12, game_phase_3, material_category_5] → 26-dim vector
```

---

## Target Variables (Labels)

### 1. Policy Head Target
- **move_quality_grade**: 0-5 integer
- **Preprocessing**: One-hot encode to 6 classes
  - Grade 0 → [1, 0, 0, 0, 0, 0]
  - Grade 1 → [0, 1, 0, 0, 0, 0]
  - ...
  - Grade 5 → [0, 0, 0, 0, 0, 1]

### 2. Value Head Target
- **best_move_eval**: Centipawn evaluation from Stockfish
- **Preprocessing**: 
  - Clip to [-10000, 10000] (remove extreme outliers)
  - Normalize to [-1, 1] range: `normalized = eval / 10000`
  - Or: Use tanh transformation: `tanh(eval / 1000)`

---

## Preprocessing Pipeline Implementation

### Step 1: Load and Split
```python
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load pre-split datasets
train_data = load_jsonl('data/analysis/splits/train.jsonl')
val_data = load_jsonl('data/analysis/splits/validation.jsonl')
test_data = load_jsonl('data/analysis/splits/test.jsonl')
```

### Step 2: Extract Features
```python
def extract_features(records):
    """Extract features from dataset records"""
    numerical = []
    boolean = []
    game_phase = []
    material_cat = []
    
    for record in records:
        f = record['features']
        
        # Numerical
        numerical.append([
            f['F003_material_balance_cp'],
            f['F005_total_piece_count'],
            f['F030_white_piece_mobility'],
            f['F030_black_piece_mobility'],
            f['F031_white_pieces_on_strong_squares'],
            f['F031_black_pieces_on_strong_squares']
        ])
        
        # Boolean
        boolean.append([
            f['F010_white_king_castled'],
            f['F010_black_king_castled'],
            f['F011_white_king_has_pawn_shield'],
            f['F011_black_king_has_pawn_shield'],
            f['F012_white_king_under_attack'],
            f['F012_black_king_under_attack'],
            f['F032_white_has_bishop_pair'],
            f['F032_black_has_bishop_pair'],
            f['F050_is_capture'],
            f['F051_is_check'],
            f['F052_is_castling'],
            f['F053_is_promotion']
        ])
        
        # Categorical
        game_phase.append(f['F002_game_phase'])
        material_cat.append(f['F004_material_advantage_category'])
    
    return {
        'numerical': np.array(numerical, dtype=np.float32),
        'boolean': np.array(boolean, dtype=np.float32),
        'game_phase': np.array(game_phase).reshape(-1, 1),
        'material_cat': np.array(material_cat).reshape(-1, 1)
    }
```

### Step 3: Fit Transformers on Training Data
```python
# Extract training features
train_features = extract_features(train_data)

# Fit scalers/encoders on TRAINING DATA ONLY
scaler = StandardScaler()
scaler.fit(train_features['numerical'])

phase_encoder = OneHotEncoder(sparse=False)
phase_encoder.fit(train_features['game_phase'])

material_encoder = OneHotEncoder(sparse=False)
material_encoder.fit(train_features['material_cat'])
```

### Step 4: Transform All Splits
```python
def preprocess_features(features, scaler, phase_enc, material_enc):
    """Apply preprocessing transformations"""
    numerical_scaled = scaler.transform(features['numerical'])
    phase_encoded = phase_enc.transform(features['game_phase'])
    material_encoded = material_enc.transform(features['material_cat'])
    
    # Concatenate all features
    X = np.concatenate([
        numerical_scaled,           # 6 features
        features['boolean'],        # 12 features
        phase_encoded,              # 3 features
        material_encoded            # 5 features
    ], axis=1)  # Total: 26 features
    
    return X

# Transform all splits
X_train = preprocess_features(train_features, scaler, phase_encoder, material_encoder)
X_val = preprocess_features(val_features, scaler, phase_encoder, material_encoder)
X_test = preprocess_features(test_features, scaler, phase_encoder, material_encoder)
```

### Step 5: Extract Targets
```python
def extract_targets(records):
    """Extract target variables for dual-head model"""
    policy_targets = []  # move quality grades
    value_targets = []   # position evaluations
    
    for record in records:
        stockfish = record['stockfish_analysis']
        
        # Policy: move quality grade (0-5)
        policy_targets.append(stockfish['move_quality_grade'])
        
        # Value: position evaluation (clipped and normalized)
        eval_cp = stockfish['best_move_eval']
        eval_cp = np.clip(eval_cp, -10000, 10000)
        eval_normalized = eval_cp / 10000.0  # Range: [-1, 1]
        value_targets.append(eval_normalized)
    
    return {
        'policy': np.array(policy_targets, dtype=np.int64),
        'value': np.array(value_targets, dtype=np.float32)
    }

y_train = extract_targets(train_data)
y_val = extract_targets(val_data)
y_test = extract_targets(test_data)
```

### Step 6: Save Preprocessed Data
```python
import pickle

# Save transformers
with open('data/preprocessed/transformers.pkl', 'wb') as f:
    pickle.dump({
        'scaler': scaler,
        'phase_encoder': phase_encoder,
        'material_encoder': material_encoder
    }, f)

# Save preprocessed arrays (optional - can load from JSONL)
np.savez_compressed('data/preprocessed/train.npz',
                   X=X_train, 
                   policy=y_train['policy'],
                   value=y_train['value'])
```

---

## Data Augmentation (Optional)

### Position Perspective Flip
Since chess is symmetric, we can **flip White/Black perspective**:

```python
def flip_perspective(features):
    """Flip position from White's view to Black's view"""
    flipped = features.copy()
    
    # Swap White/Black features (indices 2-5 for mobility/strong squares)
    flipped[:, 2], flipped[:, 3] = features[:, 3].copy(), features[:, 2].copy()
    flipped[:, 4], flipped[:, 5] = features[:, 5].copy(), features[:, 4].copy()
    
    # Swap White/Black booleans (indices 6-17)
    # Swap king castled
    flipped[:, 6], flipped[:, 7] = features[:, 7].copy(), features[:, 6].copy()
    # Swap pawn shield
    flipped[:, 8], flipped[:, 9] = features[:, 9].copy(), features[:, 8].copy()
    # Swap under attack
    flipped[:, 10], flipped[:, 11] = features[:, 11].copy(), features[:, 10].copy()
    # Swap bishop pair
    flipped[:, 12], flipped[:, 13] = features[:, 13].copy(), features[:, 12].copy()
    
    # Flip material balance sign
    flipped[:, 0] = -features[:, 0]
    
    # Flip material advantage category (reverse order)
    # Implementation depends on one-hot encoding details
    
    return flipped
```

**Note**: This doubles effective dataset size but needs careful handling of:
- Material balance sign flip
- Material advantage category reversal
- Position evaluation sign flip for value target

**Recommendation**: Start WITHOUT augmentation, add later if overfitting occurs.

---

## Implementation Script: `scripts/preprocess_dataset.py`

Create a standalone script that:
1. Loads train/val/test splits
2. Fits transformers on training data
3. Applies transformations to all splits
4. Saves preprocessed arrays and transformers
5. Generates preprocessing statistics report

---

## Next Steps

1. ✅ Preprocessing strategy defined
2. 🔲 Implement `preprocess_dataset.py` script
3. 🔲 Run preprocessing on full dataset
4. 🔲 Define PyTorch model architecture (dual-head)
5. 🔲 Implement training loop
6. 🔲 Hyperparameter tuning

---

## Key Decisions Made

✅ **StandardScaler** for numerical features (zero mean, unit variance)  
✅ **One-hot encoding** for game_phase (3) and material_advantage (5)  
✅ **Boolean features** kept as-is (already 0/1)  
✅ **F001_position_fen** excluded from training (metadata only)  
✅ **26-dimensional input** feature vector  
✅ **Dual targets**: Policy (6 classes) + Value (regression)  
⏸️ **Data augmentation** deferred until after initial training  

---

*Last Updated: May 7, 2026 @ 1:00 PM*
