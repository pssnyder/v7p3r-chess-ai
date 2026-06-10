# V7P3R AI v6.0 - Stage 1 Training Architecture

## Overview

This document defines the complete architecture for Stage 1 training: a graph-augmented neural network for binary move classification (Good vs Bad).

## Training Objective

**Task:** Binary classification of chess positions based on move quality

**Input:** Chess position features (325D) + transposition graph structure  
**Output:** Probability that position represents a "good" move (0-1 scale)

**Decision Boundary:**
- P(good) ≥ 0.5 → Recommend this move (near-optimal)
- P(good) < 0.5 → Avoid this move (tactical/positional mistake)

## Data Pipeline

### Dataset Structure

**Training Data:**
- Good positions: 5,719,272 (82.6%)
- Bad positions: 69,240 (17.4%)
- Imbalance ratio: 82.6:1

**Transposition Graph:**
- Nodes: 1,288 positions (sample)
- Edges: 8,980 similarity links
- Average degree: 13.94 neighbors/node

**Train/Val/Test Split:**
- Train: 80% (~4.6M positions)
- Validation: 10% (~572k positions)
- Test: 10% (~572k positions)

### Feature Processing

**Input Features:** 325 dimensions from v5.0 feature extractor
- Material features (16): piece counts, material balance, piece values
- King safety (24): attack squares, king exposure, pawn shield
- Pawn structure (18): isolated, doubled, passed pawns, chains
- Mobility (32): piece movement freedom, control metrics
- Tactical patterns (40): pins, forks, skewers, discovered attacks
- Positional (195): piece-square tables, board control, development

**Feature Preprocessing:**
1. **Drop zero-variance features** (identified in data analysis)
2. **Normalize numeric features** to [0, 1] range
3. **Convert boolean features** to {0, 1} integers
4. **Handle missing values** (fill with 0 or feature mean)

**Feature Engineering (Optional for v6.1):**
- PCA dimensionality reduction (325 → 256)
- Feature interaction terms
- Temporal features from move sequences

### Batching Strategy

**Batch Composition:**
- Batch size: 2048 positions
- Mixed sampling: 80% good + 20% bad (mitigate imbalance)
- Shuffle: Yes (every epoch)
- Drop last: No (use all data)

**Graph Neighbor Lookup:**
- For positions in transposition graph: Load K=10 neighbor features
- For positions NOT in graph: Use zero-padding or mean neighbor embedding

## Neural Network Architecture

### Model: GraphAugmentedPolicyNetwork

**Overall Structure:**
```
Input (325D features)
    ↓
Position Embedding Layer (325 → 512)
    ↓
[If position in graph]
    Neighbor Embedding Lookup (K=10 neighbors × 512D each)
    ↓
    Transposition Attention (attend to K neighbors)
    ↓
    Concatenate: [position_emb (512) + attended_neighbors (512)] = 1024D
[Else]
    Use position_emb only = 512D
    ↓
Hidden Layer 1 (1024/512 → 512, ReLU, Dropout 0.3)
    ↓
Hidden Layer 2 (512 → 256, ReLU, Dropout 0.3)
    ↓
Hidden Layer 3 (256 → 128, ReLU, Dropout 0.3)
    ↓
Output Layer (128 → 1, Sigmoid)
    ↓
P(good) ∈ [0, 1]
```

### Layer Details

**1. Position Embedding Layer**
```python
Dense(325 → 512, activation='relu')
BatchNormalization()
```
- Transforms raw features to learned representation
- Captures non-linear feature interactions

**2. Transposition Attention Mechanism**
```python
# For each position with graph neighbors:
Q = Linear(position_emb, 512)  # Query from current position
K = Linear(neighbor_embs, 512)  # Keys from neighbors (K=10)
V = Linear(neighbor_embs, 512)  # Values from neighbors

# Attention weights
attn_weights = Softmax(Q @ K.T / sqrt(512))

# Attended neighbor representation
attended = attn_weights @ V  # Weighted sum of neighbor values

# Combine with position embedding
combined = Concatenate([position_emb, attended])  # 1024D
```

**Attention Intuition:**
- Similar positions (graph neighbors) inform current prediction
- Model learns which neighbors are most relevant
- Enforces consistency across transpositions

**3. Hidden Layers**
```python
Layer 1: Dense(1024/512 → 512) + ReLU + Dropout(0.3) + BatchNorm
Layer 2: Dense(512 → 256) + ReLU + Dropout(0.3) + BatchNorm
Layer 3: Dense(256 → 128) + ReLU + Dropout(0.3)
```
- Progressive dimensionality reduction
- Dropout prevents overfitting
- BatchNorm stabilizes training

**4. Output Layer**
```python
Dense(128 → 1, activation='sigmoid')
```
- Single output: P(good move)
- Sigmoid bounds output to [0, 1]

### Regularization Techniques

**1. Dropout (p=0.3)**
- Applied after each hidden layer
- Prevents co-adaptation of neurons
- Improves generalization

**2. Batch Normalization**
- Stabilizes training dynamics
- Allows higher learning rates
- Reduces internal covariate shift

**3. L2 Weight Regularization (λ=0.0001)**
- Penalizes large weights
- Prevents overfitting to training data

**4. Graph Regularization** (see Loss Function)

## Loss Function

### Composite Loss

**Total Loss = α · BCE_Loss + β · Graph_Regularization_Loss**

Where:
- α = 1.0 (primary task weight)
- β = 0.1 (graph smoothness weight)

### 1. Weighted Binary Cross-Entropy (BCE)

**Formula:**
```
BCE = -[w_good · y · log(ŷ) + w_bad · (1-y) · log(1-ŷ)]

Where:
  y = true label (0 or 1)
  ŷ = predicted probability
  w_good = 1/82.6 ≈ 0.012 (inverse class frequency)
  w_bad = 1.0
```

**Class Weights Calculation:**
```python
# Based on 82.6:1 imbalance
total = 5,719,272 + 69,240 = 5,788,512
weight_good = total / (2 * 5,719,272) = 0.506
weight_bad = total / (2 * 69,240) = 41.82

# Normalize so bad weight = 1.0
weight_good = 0.506 / 41.82 = 0.012
weight_bad = 1.0
```

**Rationale:**
- Handles severe class imbalance (82.6:1)
- Forces model to pay attention to rare "bad" examples
- Prevents trivial "always predict good" solution

### 2. Graph Regularization Loss

**Formula:**
```
Graph_Reg = (1/N) · Σ_i [(ŷ_i - mean(ŷ_neighbors_i))²]

Where:
  N = number of positions with graph neighbors
  ŷ_i = predicted probability for position i
  ŷ_neighbors_i = predictions for K=10 neighbors of position i
```

**Implementation:**
```python
def graph_regularization_loss(predictions, neighbor_indices, graph):
    """
    Encourage similar positions to have similar predictions.
    
    Args:
        predictions: Model predictions for batch (N,)
        neighbor_indices: Graph neighbor lookup (N, K)
        graph: Transposition graph structure
    
    Returns:
        Scalar loss (L2 divergence from neighbor consensus)
    """
    reg_loss = 0.0
    count = 0
    
    for i, pred_i in enumerate(predictions):
        neighbors = neighbor_indices[i]
        
        if neighbors is not None:  # Position has graph neighbors
            neighbor_preds = [predictions[j] for j in neighbors if j < len(predictions)]
            
            if neighbor_preds:
                mean_neighbor_pred = np.mean(neighbor_preds)
                reg_loss += (pred_i - mean_neighbor_pred) ** 2
                count += 1
    
    return reg_loss / count if count > 0 else 0.0
```

**Rationale:**
- Enforces prediction consistency across similar positions
- Leverages graph structure for better generalization
- Inspired by graph neural networks (smoothness assumption)
- Helps model learn chess patterns rather than memorizing positions

## Training Procedure

### Optimizer

**Adam Optimizer**
- Learning rate: 0.001 (default)
- β1: 0.9 (momentum)
- β2: 0.999 (RMSprop component)
- ε: 1e-8 (numerical stability)

**Learning Rate Schedule:**
```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,      # Reduce by 50%
    patience=5,      # Wait 5 epochs
    min_lr=1e-6      # Minimum LR
)
```

**Rationale:**
- Adam adapts learning rate per parameter
- LR schedule prevents plateaus
- Allows fine-tuning in later epochs

### Training Loop

**Hyperparameters:**
- Epochs: 100 (with early stopping)
- Batch size: 2048
- Gradient clipping: norm ≤ 1.0

**Early Stopping:**
```python
EarlyStopping(
    monitor='val_loss',
    patience=10,     # Wait 10 epochs
    restore_best_weights=True
)
```

**Checkpointing:**
```python
ModelCheckpoint(
    filepath='models/stage1_policy_epoch_{epoch:02d}_val_{val_loss:.4f}.h5',
    monitor='val_f1',
    save_best_only=True
)
```

### Pseudocode

```python
# Initialization
model = GraphAugmentedPolicyNetwork(input_dim=325)
optimizer = Adam(lr=0.001)
train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2048)

# Training loop
for epoch in range(100):
    # Training phase
    model.train()
    train_loss = 0.0
    
    for batch in train_loader:
        features, labels, neighbor_indices = batch
        
        # Forward pass
        predictions = model(features, neighbor_indices)
        
        # Calculate losses
        bce_loss = weighted_bce(predictions, labels)
        graph_reg = graph_regularization_loss(predictions, neighbor_indices)
        total_loss = bce_loss + 0.1 * graph_reg
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        clip_grad_norm(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += total_loss.item()
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_predictions = []
    val_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            features, labels, neighbor_indices = batch
            predictions = model(features, neighbor_indices)
            
            val_loss += calculate_loss(predictions, labels, neighbor_indices)
            val_predictions.extend(predictions)
            val_labels.extend(labels)
    
    # Calculate metrics
    metrics = calculate_metrics(val_predictions, val_labels)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Early stopping check
    if early_stopping.should_stop(val_loss):
        break
    
    # Logging
    print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_f1={metrics['f1']:.4f}")
```

## Evaluation Metrics

### Standard Binary Classification Metrics

**1. Accuracy**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
- Target: ≥95% (binary easier than multi-class)
- Measures overall correctness

**2. Precision (for "Bad" class)**
```
Precision = TP / (TP + FP)
```
- Target: ≥50%
- Measures: When model says "bad", how often is it correct?
- Important: Avoid false alarms (rejecting good moves)

**3. Recall (for "Bad" class)**
```
Recall = TP / (TP + FN)
```
- Target: ≥60%
- Measures: Of all truly bad moves, how many did we catch?
- Critical: Must detect blunders to avoid losing games

**4. F1 Score (for "Bad" class)**
```
F1 = 2 · (Precision · Recall) / (Precision + Recall)
```
- Target: ≥55%
- Harmonic mean of precision and recall
- Primary metric for model selection

**5. ROC-AUC**
```
AUC = Area under ROC curve
```
- Target: ≥0.85
- Measures discrimination ability across all thresholds

### Graph-Specific Metrics

**6. Transposition Consistency**
```
Consistency = Correlation(predictions_i, mean_predictions_neighbors_i)

For all positions i with graph neighbors
```
- Target: ≥0.80
- Measures: Do similar positions get similar predictions?
- Validates graph regularization effectiveness

**7. Graph Smoothness**
```
Smoothness = 1 - (1/N) · Σ_i |ŷ_i - mean(ŷ_neighbors_i)|
```
- Target: ≥0.90
- Measures: Average prediction difference from neighbors
- Lower difference = smoother graph predictions

### V7P3R Style Metrics

**8. V7P3R Move Agreement**
```
Agreement = (# of V7P3R moves classified as "good") / (# of V7P3R moves)
```
- Target: ≥60%
- Uses V7P3R game positions from training set
- Validates personality preservation

**9. V7P3R Style Consistency**
```
Consistency = Correlation(model_predictions, v7p3r_actual_choices)

On V7P3R game positions only
```
- Target: ≥0.50
- Measures how well model matches V7P3R playing style

### Confusion Matrix Analysis

```
                    Predicted
                Good        Bad
Actual  Good    TN          FP    ← False alarms (reject good moves)
        Bad     FN          TP    ← Misses (accept bad moves)
```

**Key Insights:**
- **FN (False Negatives):** Missed blunders - CRITICAL to minimize
- **FP (False Positives):** Rejected good moves - Less critical but hurts style
- **Class balance:** Expect high TN (most moves are good)

## Implementation Plan

### Phase 1: Data Pipeline (Day 1)

**Tasks:**
1. Load filtered datasets (good_positions.jsonl, bad_positions.jsonl)
2. Load transposition graph (transposition_graph.pkl)
3. Create feature preprocessor:
   - Identify and drop zero-variance features
   - Normalize numeric features
   - Convert boolean to int
4. Implement train/val/test split (80/10/10)
5. Create TensorFlow Dataset with batching
6. Implement neighbor lookup for graph positions

**Validation:**
- Verify batch shapes: features (2048, 325), labels (2048,)
- Check class balance in batches
- Test neighbor lookup correctness

### Phase 2: Model Implementation (Day 1-2)

**Tasks:**
1. Implement Position Embedding layer
2. Implement Transposition Attention mechanism
3. Implement Hidden layers with dropout + batch norm
4. Implement Output layer (sigmoid)
5. Assemble full GraphAugmentedPolicyNetwork
6. Test forward pass on sample batch

**Validation:**
- Verify output shape: (batch_size, 1)
- Check output range: [0, 1] (sigmoid)
- Test with/without graph neighbors

### Phase 3: Loss & Training (Day 2)

**Tasks:**
1. Implement weighted BCE loss
2. Implement graph regularization loss
3. Configure Adam optimizer
4. Implement learning rate scheduler
5. Implement early stopping
6. Add model checkpointing
7. Add TensorBoard logging

**Validation:**
- Test loss calculation on sample batch
- Verify gradient flow (no NaN/Inf)
- Test checkpoint save/load

### Phase 4: Training Execution (Day 2-3)

**Tasks:**
1. Run initial training (10 epochs, monitor metrics)
2. Tune hyperparameters if needed:
   - Learning rate
   - Batch size
   - Dropout rate
   - Graph weight (β)
3. Full training run (100 epochs with early stopping)
4. Monitor TensorBoard for:
   - Loss curves (train/val)
   - Metric curves (accuracy, F1, AUC)
   - Learning rate changes

**Expected Timeline:**
- 1 epoch ≈ 30-60 minutes (5.7M positions, batch size 2048)
- Full training ≈ 8-12 hours (if early stopping at epoch ~20-30)

### Phase 5: Evaluation (Day 3)

**Tasks:**
1. Load best checkpoint
2. Evaluate on test set:
   - Standard metrics (accuracy, precision, recall, F1, AUC)
   - Graph metrics (transposition consistency, smoothness)
   - V7P3R style metrics
3. Generate confusion matrix
4. Analyze errors:
   - Which positions are hardest to classify?
   - Are there systematic failure patterns?
5. Save evaluation report (JSON + Markdown)

**Deliverables:**
- `stage1_evaluation_report.json`
- `stage1_evaluation_report.md`
- Confusion matrix visualization
- Top-20 hardest positions analysis

## Performance Targets

### Minimum Acceptable Performance (MVP)

| Metric | Target | Rationale |
|--------|--------|-----------|
| Accuracy | ≥90% | Binary task should be easier than 6-class |
| Bad Recall | ≥50% | Catch at least half of tactical blunders |
| F1 Score | ≥45% | Balanced performance on minority class |
| Transposition Consistency | ≥0.70 | Graph regularization shows some effect |
| V7P3R Agreement | ≥55% | Maintain some personality despite imbalance |

### Target Performance (Production Ready)

| Metric | Target | Rationale |
|--------|--------|-----------|
| Accuracy | ≥95% | High confidence in predictions |
| Bad Recall | ≥60% | Catch majority of tactical mistakes |
| Bad Precision | ≥50% | Limit false alarms |
| F1 Score | ≥55% | Strong performance on rare class |
| ROC-AUC | ≥0.85 | Good discrimination across thresholds |
| Transposition Consistency | ≥0.80 | Strong graph effect |
| Graph Smoothness | ≥0.90 | Predictions very similar to neighbors |
| V7P3R Agreement | ≥60% | Preserve V7P3R playing style |

### Stretch Goals (Excellent Performance)

| Metric | Target | Rationale |
|--------|--------|-----------|
| Accuracy | ≥97% | Near-perfect on majority class |
| Bad Recall | ≥70% | Catch most blunders |
| F1 Score | ≥65% | Very strong on rare class |
| ROC-AUC | ≥0.90 | Excellent discrimination |
| Transposition Consistency | ≥0.85 | Very strong graph effect |
| V7P3R Agreement | ≥65% | Strong style preservation |

## Comparison with v5.0

| Aspect | v5.0 Multi-Class | v6.0 Binary + Graph |
|--------|------------------|---------------------|
| **Task** | 6-grade classification | Binary (good/bad) |
| **Classes** | 6 (Grades 0-5) | 2 (Good vs Bad) |
| **Architecture** | Standard feedforward | Graph-augmented with attention |
| **Input** | 325D features | 325D + graph structure |
| **Hidden layers** | [512, 256, 128] | [1024, 512, 256, 128] |
| **Parameters** | ~400k | ~800k (2x due to attention) |
| **Training data** | 6.3M all sources | 5.7M curated (Lichess + V7P3R) |
| **Class weights** | Uniform per class | Severe imbalance (82:1) |
| **Graph structure** | None | 1.3k nodes, 9k edges |
| **Expected accuracy** | ~85% (multi-class) | ~95% (binary) |
| **Expected F1** | ~0.40 (weighted avg) | ~0.55 (on "bad" class) |
| **Training time** | ~6 hours | ~10 hours (larger model) |
| **Inference speed** | ~1000 pos/sec | ~800 pos/sec (attention overhead) |

## Risk Mitigation

### Risk 1: Extreme Class Imbalance (82:1)

**Mitigation:**
- Weighted loss function (bad weight = 41.8x good weight)
- Mixed batch sampling (20% bad positions per batch)
- Monitor precision AND recall (not just accuracy)
- Use F1 score as primary metric (balances P&R)

### Risk 2: Transposition Graph Too Small (1.3k nodes)

**Mitigation:**
- Graph regularization weight starts low (β=0.1)
- Monitor transposition consistency metric
- Can disable graph regularization if no benefit (β=0)
- Future: Expand to full 5.6M graph with FAISS

### Risk 3: Overfitting to Training Data

**Mitigation:**
- Dropout (p=0.3) in all hidden layers
- L2 weight regularization (λ=0.0001)
- Early stopping (patience=10)
- Large validation set (10% = 572k positions)
- Cross-validation on V7P3R holdout positions

### Risk 4: V7P3R Style Lost in Imbalance

**Mitigation:**
- Explicitly track V7P3R agreement metric
- Validate on V7P3R-only test set
- If agreement <50%, increase V7P3R sampling in batches
- Future Stage 2: Self-play will reinforce V7P3R style

### Risk 5: Training Time Too Long

**Mitigation:**
- Start with subset training (1M positions, 10 epochs)
- If performance acceptable, scale to full dataset
- Use GPU acceleration (TensorFlow/PyTorch)
- Optimize data loading (prefetch, multi-threading)

## Next Steps After Stage 1

### If Performance Targets Met (F1 ≥55%, V7P3R ≥60%)

**Proceed to Stage 2: Self-Play Reinforcement Learning**
1. Implement self-play framework
2. Generate new positions via epsilon-greedy
3. Stockfish feedback loop
4. Expand transposition graph
5. Retrain with augmented dataset

### If Performance Below Target

**Iterate on Stage 1:**
1. Hyperparameter tuning (grid search)
2. Feature engineering (PCA, interaction terms)
3. Data augmentation (position mirroring, color swapping)
4. Architecture changes (deeper network, different attention)
5. Expand transposition graph (10k or 100k nodes)

### Regardless of Performance

**Documentation & Analysis:**
1. Save trained model + checkpoints
2. Generate comprehensive evaluation report
3. Error analysis: Which positions are hard?
4. Feature importance analysis
5. Visualization: t-SNE embeddings, attention weights

---

**Timeline Summary:**
- Day 1: Data pipeline + model implementation
- Day 2: Training setup + initial runs
- Day 3: Full training + evaluation
- **Total:** 3 days to trained Stage 1 model

**Success Criteria:**
- F1 score ≥55% on test set
- V7P3R agreement ≥60%
- Transposition consistency ≥0.80
- No critical errors or NaN losses during training
