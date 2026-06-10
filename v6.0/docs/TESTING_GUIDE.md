# V7P3R AI v6.0 - Testing & Validation Guide

## Quick Test: Data Loading (1-2 minutes)

Test that the data loader can successfully load and process our filtered dataset:

```powershell
cd "E:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v6.0"

python -c @"
from scripts.stage1.train_policy import DataLoader
from pathlib import Path

print('Testing DataLoader...')
base = Path('.')
loader = DataLoader(
    str(base / 'data/stage1/good_positions.jsonl'),
    str(base / 'data/stage1/bad_positions.jsonl'),
    str(base / 'data/stage1/transposition_graph.pkl'),
    {}
)

# Load small sample
data = loader.load(max_samples=1000)  # 1k good + 1k bad

print(f\"\\nData shapes:\")
print(f\"  Train: {data['train'][0].shape}\")
print(f\"  Val: {data['val'][0].shape}\")
print(f\"  Test: {data['test'][0].shape}\")
print(f\"  Features: {data['feature_dim']}\")

print('\\n✅ Data loading works!')
"@
```

**Expected Output:**
```
Testing DataLoader...
✅ Loaded graph: 1,288 nodes
✅ Loaded 1,000 good + 1,000 bad positions
📊 Total dataset: 2,000 positions
   Good: 1,000 (50.0%)
   Bad:  1,000 (50.0%)
...
Data shapes:
  Train: (1600, 95)
  Val: (200, 95)
  Test: (200, 95)
  Features: 95

✅ Data loading works!
```

**Potential Issues:**
- **Feature mismatch**: Not all positions have all 325 features (only 95 numeric features)
- **File paths**: Ensure working directory is v6.0/
- **Memory**: 1k sample should work on any machine

---

## Quick Test: Model Forward Pass (1-2 minutes)

Test that the model can be constructed and perform a forward pass:

```powershell
python -c @"
import tensorflow as tf
from scripts.stage1.train_policy import GraphAugmentedPolicyNetwork
import numpy as np

print('Testing GraphAugmentedPolicyNetwork...')

# Create model
config = {'use_graph_attention': False}
model = GraphAugmentedPolicyNetwork(input_dim=95, config=config)

# Create dummy batch
batch_size = 32
features = np.random.randn(batch_size, 95).astype(np.float32)
neighbors = None  # Not using graph attention yet

# Forward pass
output = model((features, neighbors), training=False)

print(f\"\\nOutput shape: {output.shape}\")
print(f\"Output range: [{output.numpy().min():.4f}, {output.numpy().max():.4f}]\")
print(f\"Output mean: {output.numpy().mean():.4f}\")

assert output.shape == (batch_size, 1), 'Wrong output shape!'
assert output.numpy().min() >= 0.0, 'Output not in [0,1]!'
assert output.numpy().max() <= 1.0, 'Output not in [0,1]!'

print('\\n✅ Model forward pass works!')
"@
```

**Expected Output:**
```
Testing GraphAugmentedPolicyNetwork...

Output shape: (32, 1)
Output range: [0.4123, 0.6789]
Output mean: 0.5234

✅ Model forward pass works!
```

**Potential Issues:**
- **NaN outputs**: Check initialization, batch norm
- **Wrong shape**: Verify output layer configuration
- **Not in [0,1]**: Sigmoid activation should guarantee this

---

## Subset Training (30-60 minutes)

Train on a small subset to validate the full pipeline:

```powershell
python scripts/stage1/train_subset.py
```

**Create this file first:**

```python
# scripts/stage1/train_subset.py
from train_policy import DataLoader, GraphAugmentedPolicyNetwork, Trainer
from pathlib import Path

print("SUBSET TRAINING TEST")
print("=" * 60)

# Configuration
config = {
    'epochs': 5,  # Just 5 epochs
    'batch_size': 256,
    'learning_rate': 0.001,
    'use_graph_attention': False,
    'output_dir': 'models/stage1_test',
}

# Paths
base_path = Path(__file__).parent.parent.parent
good_path = base_path / "data" / "stage1" / "good_positions.jsonl"
bad_path = base_path / "data" / "stage1" / "bad_positions.jsonl"
graph_path = base_path / "data" / "stage1" / "transposition_graph.pkl"

# Create output directory
output_dir = base_path / config['output_dir']
output_dir.mkdir(parents=True, exist_ok=True)

# Load SUBSET (10k positions)
loader = DataLoader(str(good_path), str(bad_path), str(graph_path), config)
data = loader.load(max_samples=5000)  # 5k good + all bad

print(f"\n📊 Subset training: {data['train'][0].shape[0]:,} positions")

# Train
trainer = Trainer(config)
history = trainer.train(data, loader)

print("\n✅ Subset training complete!")
print(f"   Final train loss: {history.history['loss'][-1]:.4f}")
print(f"   Final val loss: {history.history['val_loss'][-1]:.4f}")
print(f"   Final val accuracy: {history.history['val_accuracy'][-1]:.4f}")
```

**Expected Runtime:** 30-60 minutes
**Expected Output:**
```
Epoch 1/5
...
Epoch 5/5
...
✅ Subset training complete!
   Final train loss: 0.2345
   Final val loss: 0.2567
   Final val accuracy: 0.9123
```

**Success Criteria:**
- ✅ Training completes without errors
- ✅ Loss decreases over epochs
- ✅ Val accuracy > 85% (easy on subset)
- ✅ No NaN losses
- ✅ Reasonable training speed (~5-10 min/epoch on CPU)

**Potential Issues:**
- **Very slow**: Check if using CPU instead of GPU
- **NaN loss**: Reduce learning rate to 0.0001
- **Low accuracy**: Check class weighting, data imbalance
- **Memory error**: Reduce batch_size to 128

---

## Full Training (8-12 hours)

Once subset training works, run on full dataset:

```powershell
python scripts/stage1/train_policy.py
```

**Monitor with TensorBoard:**
```powershell
tensorboard --logdir models/stage1/logs
```

**Success Criteria:**
- ✅ F1 score ≥ 55% on test set
- ✅ Bad recall ≥ 60%
- ✅ Val loss stops improving (early stopping triggers)
- ✅ No overfitting (train/val loss gap < 0.05)

---

## Evaluation

After training, evaluate on test set:

```python
# scripts/stage1/evaluate.py
from train_policy import DataLoader
import tensorflow as tf
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Load model
model_path = Path("models/stage1/stage1_policy_best.h5")
model = tf.keras.models.load_model(model_path)

# Load test data
base_path = Path(__file__).parent.parent.parent
loader = DataLoader(
    str(base_path / "data/stage1/good_positions.jsonl"),
    str(base_path / "data/stage1/bad_positions.jsonl"),
    str(base_path / "data/stage1/transposition_graph.pkl"),
    {}
)

data = loader.load()
X_test, y_test, hash_test = data['test']

# Predict
y_pred_proba = model.predict((X_test, None), batch_size=2048)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Metrics
print("=" * 60)
print("TEST SET EVALUATION")
print("=" * 60)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Bad', 'Good']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print("\nDetailed Metrics:")
print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"  Precision (Bad): {precision_score(y_test, y_pred, pos_label=0):.4f}")
print(f"  Recall (Bad):    {recall_score(y_test, y_pred, pos_label=0):.4f}")
print(f"  F1 (Bad):        {f1_score(y_test, y_pred, pos_label=0):.4f}")
print(f"  AUC:       {roc_auc_score(y_test, y_pred_proba):.4f}")
```

---

## Troubleshooting

### Data Loading Fails

**Symptom:** FileNotFoundError, JSON parsing errors
**Fix:**
- Check working directory: `cd "E:\...\v7p3r-chess-ai\v6.0"`
- Verify files exist: `ls data/stage1/`
- Check JSON format: `python -c "import json; json.loads(open('data/stage1/good_positions.jsonl').readline())"`

### Model Won't Compile

**Symptom:** TensorFlow errors, shape mismatches
**Fix:**
- Update TensorFlow: `pip install --upgrade tensorflow`
- Check input_dim matches data: Should be 95 (not 325)
- Simplify model: Remove batch norm, dropout temporarily

### NaN Loss During Training

**Symptom:** Loss becomes NaN after a few batches
**Fix:**
- Lower learning rate: 0.001 → 0.0001
- Reduce batch size: 2048 → 512
- Check for invalid features (inf, NaN): Add assertions in data loader
- Add gradient clipping: `optimizer.clipnorm=1.0`

### Low Performance (F1 < 40%)

**Symptom:** Model trains but performs poorly
**Fix:**
- Check class weighting is applied correctly
- Verify data split is stratified
- Increase model capacity: [512,256,128] → [1024,512,256,128]
- Train longer: 100 epochs → 200 epochs
- Try different threshold: 0.5 → optimal from ROC curve

### Overfitting (Val loss >> Train loss)

**Symptom:** Train loss low, val loss high
**Fix:**
- Increase dropout: 0.3 → 0.5
- Add L2 regularization: `kernel_regularizer=l2(0.01)`
- Reduce model capacity
- Use more data: Remove max_samples limit

### GPU Not Being Used

**Symptom:** Training very slow (>30 min/epoch)
**Fix:**
- Check GPU availability: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
- Install CUDA if needed
- Force GPU: `with tf.device('/GPU:0'): ...`

---

## Performance Benchmarks

### Expected Training Speed

| Hardware | Batch Size | Time/Epoch | Total Time |
|----------|-----------|-----------|-----------|
| CPU (8 cores) | 2048 | 25-30 min | 25-30 hours |
| GPU (RTX 3060) | 2048 | 8-12 min | 8-12 hours |
| GPU (RTX 4090) | 4096 | 3-5 min | 3-5 hours |

### Expected Performance

| Metric | Minimum | Target | Stretch |
|--------|---------|--------|---------|
| Accuracy | 90% | 95% | 97% |
| Bad Recall | 50% | 60% | 70% |
| Bad Precision | 40% | 50% | 60% |
| F1 (Bad) | 45% | 55% | 65% |
| AUC | 0.75 | 0.85 | 0.90 |

---

## Next Steps After Success

1. ✅ **Celebrate** - First working v6.0 model!
2. **Error Analysis** - Which positions are hardest?
3. **Hyperparameter Tuning** - Can we improve further?
4. **Enable Graph Attention** - Add transposition network
5. **Expand Graph** - Build larger similarity graph
6. **Stage 2 Planning** - Self-play reinforcement learning

---

Last Updated: May 24, 2026
