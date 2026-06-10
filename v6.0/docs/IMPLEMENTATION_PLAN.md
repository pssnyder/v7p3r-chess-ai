# V7P3R AI v6.0 - Implementation Plan

## Overview
Two-stage learning architecture with graph-based transposition network for chess move evaluation.

## Stage 1: Binary Classification with Transposition Graph

### Architecture
- **Task**: Binary classification (Good vs Bad moves)
- **Good Moves**: Grade 0 (always) + Grade 1 if eval_diff ≤ 50cp
- **Bad Moves**: Grades 2-5 (tactical blunders, positional mistakes, losing moves)
- **Network**: Graph-augmented neural network with transposition attention

### Data Pipeline
```
v5.3 Merged Dataset (6.3M)
    ↓
Filter by binary classification + exclude C0BR4
    ↓
Good positions (~5.7M) + Bad positions (~33k)
    ↓
Zobrist hash + index positions
    ↓
Build transposition graph (K=10 neighbors)
    ↓
Graph-augmented neural network training
    ↓
Policy network: P(good | position)
```

### Graph Structure
- **Nodes**: Chess positions (Zobrist hash as ID)
- **Edges**: Similarity links via tactical feature matching
- **Features**: 325 features from v5.0 (material, king safety, pawn structure, mobility, tactics)
- **Similarity metric**: Sum of shared tactical features (hanging pieces, pins, forks, king attacks, passed pawns)
- **K-Neighbors**: 10 most similar positions per node

### Training Approach
1. **Input**: Position features (325D) + transposition embeddings from K neighbors
2. **Architecture**: 
   - Embedding layer: 325 → 512
   - Hidden layers: 512 → 256 → 128
   - Transposition attention: Attend to K=10 neighbor embeddings
   - Output: Binary classification (sigmoid)
3. **Loss**: Weighted binary cross-entropy + graph regularization
   - BCE weight: good=0.006, bad=1.0 (170:1 imbalance)
   - Graph regularization: L2(prediction_i - prediction_neighbors)
   - Total loss: α * BCE + β * graph_reg (α=1.0, β=0.1)
4. **Validation**: 
   - Standard metrics: accuracy, precision, recall, F1
   - Transposition consistency: Correlation between similar positions
   - V7P3R style matching: Agreement on V7P3R game moves

### Expected Performance
- **Good accuracy**: 95%+ (most puzzle moves are optimal)
- **Bad recall**: 60%+ (catch tactical blunders)
- **Transposition consistency**: 0.8+ correlation
- **V7P3R style match**: 60%+ agreement on V7P3R game moves

## Stage 2: Self-Play Reinforcement Learning

### Architecture
- **Task**: Expand knowledge beyond training set via self-play
- **Policy**: Stage 1 binary classifier (guides exploration)
- **Feedback**: Stockfish evaluations on new positions
- **Learning**: Temporal difference when intuition differs from Stockfish

### Self-Play Pipeline
```
Generate position via Stage 1 policy + epsilon-greedy
    ↓
Apply candidate move (policy network suggests good moves)
    ↓
Stockfish evaluates new position (0.5s analysis)
    ↓
Compare: AI intuition vs Stockfish evaluation
    ↓
IF disagreement > 100cp: Add position to training set
    ↓
Update transposition graph with new positions
    ↓
Retrain Stage 1 policy with expanded dataset
```

### Learning Strategy
1. **Exploration**: ε-greedy (ε=0.2) to try non-policy moves
2. **Feedback loop**: Stockfish corrects AI mistakes
3. **Graph expansion**: Add new positions to transposition network
4. **Incremental learning**: Update policy with new data batches
5. **Style preservation**: Maintain V7P3R personality via style matching validation

### Expected Outcomes
- **Dataset growth**: 5.7M → 6M+ positions (300k+ self-play discoveries)
- **Transposition graph density**: 10% increase in edge count
- **Performance**: 5-10% improvement over Stage 1 baseline
- **Coverage**: Fill gaps in opening/endgame knowledge

## Implementation Status

### ✅ Completed
- [x] v6.0 directory structure created
- [x] Zobrist hashing utility (zobrist_hashing.py)
- [x] Data filtering script (filter_dataset.py)
- [x] Transposition graph builder (build_graph.py)
- [x] Feature extraction utilities copied from v5.0
- [x] Configuration file (stage1_config.yaml)
- [x] README documentation
- [x] Quick start script (quick_start_data_prep.ps1)

### 🚧 In Progress
- [ ] **Data filtering** (RUNNING NOW - 1-2 hours)
  - Processing 6.3M positions from v5.3 merged dataset
  - Expected output: 5.7M good + 33k bad positions
  - Progress updates every 100k records

### 📋 Next Steps
1. **Complete data preparation** (~2-3 hours total)
   - Wait for filtering to complete
   - Run graph builder (build_graph.py)
   - Verify statistics (imbalance ratio, exclusions, Zobrist uniqueness)

2. **Implement Stage 1 training** (2-3 days)
   - Create train_policy.py in scripts/stage1/
   - Graph-augmented neural network
   - Transposition attention mechanism
   - Weighted loss + graph regularization
   - Train/val/test split (80/10/10)
   - Validation metrics implementation

3. **Implement Stage 2 self-play** (3-5 days)
   - Create self_play.py in scripts/stage2/
   - Self-play framework with epsilon-greedy
   - Stockfish feedback loop
   - Temporal difference learning
   - Graph expansion logic
   - Run 1000+ self-play games

4. **Evaluation and testing** (1-2 days)
   - Benchmark against v5.0 multi-class model
   - Test V7P3R style preservation
   - Validate transposition consistency
   - Generate performance report

## Comparison with v5.0

| Aspect | v5.0 Multi-Class | v6.0 Binary + Graph |
|--------|------------------|---------------------|
| **Classification** | 6 grades (0-5) | Binary (good/bad) |
| **Architecture** | Standard feedforward NN | Graph-augmented NN |
| **Features** | 325D position features | 325D + transposition embeddings |
| **Training data** | 6.3M positions (all sources) | 5.7M positions (Lichess + V7P3R) |
| **Learning** | Supervised only | Supervised + RL self-play |
| **Transpositions** | Not modeled | Explicit graph structure |
| **Imbalance handling** | Class weights only | Class weights + graph regularization |
| **Style matching** | No validation | V7P3R style consistency check |
| **Knowledge expansion** | Fixed training set | Self-play discovery |

## Key Insights

### Why Binary Classification?
- **Reality of chess**: Most good moves are nearly optimal (eval diff <50cp)
- **Simplicity**: Engine needs to know "avoid this move" vs "this move is fine"
- **Imbalance is correct**: 170:1 ratio reflects that blunders are rare in puzzle positions
- **V7P3R personality**: 194k V7P3R positions preserve playing style

### Why Transposition Network?
- **Chess structure**: Similar positions should have similar evaluations
- **Generalization**: Learn patterns, not memorize positions
- **Attention mechanism**: Neighboring positions provide context
- **Graph regularization**: Encourage smooth predictions across similar positions
- **Inspired by AlphaZero**: Position similarity as learning signal

### Why Two-Stage Learning?
- **Stage 1**: Learn from 5.7M high-quality puzzle positions (human expertise)
- **Stage 2**: Expand knowledge via self-play (discover new patterns)
- **Feedback loop**: Stockfish corrects AI mistakes in real-time
- **Continuous improvement**: Graph grows with experience

## Performance Metrics

### Stage 1 Targets
- Accuracy: 95%+ (binary classification is easier than 6-class)
- Bad recall: 60%+ (catch tactical blunders)
- Transposition consistency: 0.8+ (similar positions agree)
- V7P3R style match: 60%+ (personality preserved)

### Stage 2 Targets
- Dataset growth: 300k+ new positions
- Graph density: 10% edge increase
- Performance improvement: 5-10% over Stage 1
- Coverage expansion: Fill opening/endgame gaps

## Timeline Estimate
- ✅ Setup & data prep: 1 day (COMPLETED)
- 🚧 Stage 1 implementation: 2-3 days (IN PROGRESS)
- Stage 1 training: 1-2 days (8-12 hours GPU time)
- Stage 2 implementation: 3-5 days
- Stage 2 self-play: 2-3 days (1000+ games)
- Evaluation & testing: 1-2 days
- **Total**: ~2 weeks to full v6.0 deployment
