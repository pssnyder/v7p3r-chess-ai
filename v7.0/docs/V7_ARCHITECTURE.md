# V7P3R v7.0/v7.1 - Unified Feature-Based Architecture

## 🎯 Core Philosophy

**"Let the model learn chess through gameplay, not pre-labeled positions"**

V7 abandons the 2-stage architecture (Stage 1 position evaluator → Stage 2 complexity manager) in favor of a unified approach that learns from actual chess gameplay with Stockfish as oracle.

## 📋 Version Summary

| Version | Training Method | Key Features | Status |
|---------|----------------|--------------|--------|
| **v7.0** | Pure self-play | Phase-aware weighting, 50% endgame SF | Legacy |
| **v7.1** | Generational | New vs old eval, 100% endgame SF, 20% MG SF | ⭐ **Current** |

**Recommendation**: Use v7.1 for all new training. See [V7.1_GENERATIONAL_TRAINING.md](V7.1_GENERATIONAL_TRAINING.md).

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     CHESS POSITION                           │
│            (represented as FEN string)                       │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│          COMPREHENSIVE FEATURE EXTRACTOR                     │
│                                                              │
│  ┌───────────────────────────────────────────────────┐     │
│  │ Stage 1 Features (19 dims)                        │     │
│  │ - 12 piece counts                                 │     │
│  │ - Material balance                                │     │
│  │ - Castling rights, check, mobility                │     │
│  └───────────────────────────────────────────────────┘     │
│                                                              │
│  ┌───────────────────────────────────────────────────┐     │
│  │ Heuristic Features (24 dims)                      │     │
│  │ - Bishop pair, passed pawns                       │     │
│  │ - Doubled/isolated pawns                          │     │
│  │ - King safety, active rooks                       │     │
│  │ - Development, mobility (normalized)              │     │
│  └───────────────────────────────────────────────────┘     │
│                                                              │
│  ┌───────────────────────────────────────────────────┐     │
│  │ Complexity Features (8 dims)                      │     │
│  │ - Legal moves, captures, checks                   │     │
│  │ - Piece tension, center control                   │     │
│  │ - Game phase, move diversity                      │     │
│  │ - Forest darkness score (V7P3R custom)            │     │
│  └───────────────────────────────────────────────────┘     │
│                                                              │
│               OUTPUT: 51-dim feature vector                  │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│              NEURAL NETWORK (Value Head)                     │
│                                                              │
│  Input Layer:    51 features                                │
│  Hidden Layer 1: 256 nodes (ReLU + BatchNorm + Dropout)     │
│  Hidden Layer 2: 128 nodes (ReLU + BatchNorm + Dropout)     │
│  Hidden Layer 3: 64 nodes  (ReLU + BatchNorm + Dropout)     │
│  Output Layer:   1 node    (Tanh) → position value [-1, 1]  │
│                                                              │
│  OUTPUT: Position evaluation from mover's perspective        │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│          MONTE CARLO TREE SEARCH (MCTS)                      │
│                                                              │
│  1. For each legal move, create child position              │
│  2. Extract 51 features from child position                 │
│  3. Evaluate child with neural network                      │
│  4. Select move with best value (+ exploration bonus)       │
│                                                              │
│  OUTPUT: Best move to play                                  │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│              SELF-PLAY TRAINING LOOP                         │
│                                                              │
│  1. Play full game using MCTS + current network             │
│  2. After each move, record:                                │
│     - Position features (51-dim)                            │
│     - Move played                                           │
│     - Stockfish evaluation (oracle)                         │
│  3. At end of game, backpropagate:                          │
│     - Game result (win/loss/draw)                           │
│     - Stockfish evaluations                                 │
│     - Custom personality rewards                            │
│  4. Train network on collected experience                   │
│  5. Repeat                                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎓 Training Strategy

### v7.1 Generational Training (Current)

**Architecture**: AlphaZero-inspired generational evolution

```
┌─────────────────────────────────────────┐
│  Generation 0: Random initialization    │
│           ↓                              │
│  Self-play training (100 games)         │
│           ↓                              │
│  Generation 1: Trained model            │
│           ↓                              │
│  Evaluate vs Gen 0 (6-game match)       │
│           ├─ Win rate > 50% → Accept    │
│           └─ Win rate ≤ 50% → Reject    │
│           ↓                              │
│  Generation 2: Train from best so far   │
│           ↓                              │
│  Repeat for N generations...            │
└─────────────────────────────────────────┘
```

**Key Differences from v7.0:**
- New model evaluated against **previous best** (not itself)
- Win/loss metrics are **meaningful** (new must beat old)
- Color balance enforced (3 White, 3 Black)
- Revised weight curve: 20% MG, 100% endgame

### v7.0 Self-Play Training (Legacy)

**Note**: v7.0 has a critical flaw - same model plays both sides, so win rate just measures % of decisive games, not improvement.

### Stockfish as Oracle (Both Versions)

Instead of pre-labeling positions, Stockfish provides real-time feedback during self-play:

- **After each move**: Query Stockfish for position evaluation
- **Objective signal**: "Is this position winning/losing?"
- **No labeling bias**: Model learns from actual gameplay outcomes

### Custom Personality Rewards

V7P3R's Tal-style aggression is encoded as reward weights:

```python
# Base reward: Stockfish evaluation + game outcome
base_reward = stockfish_eval * 0.7 + game_result * 0.3

# Personality bonuses (applied to feature values)
personality_bonus = (
    forest_darkness_score * 0.15 +      # Reward complexity
    piece_tension * 0.10 +               # Reward active positions
    passed_pawns_advantage * 0.05 -      # Standard chess
    king_safety_advantage * -0.05        # Tolerate king risk
)

# Final reward
total_reward = base_reward + personality_bonus
```

**Key insight**: Model learns "good chess" from Stockfish, but personality weights bias it toward V7P3R's aggressive style.

---

## 🔥 Why This Works (Solving v2.0's Failure)

### The v2.0 Problem

- **Features**: 19 fast features (piece counts, material, mobility)
- **Labels**: Heuristic sentiment (bishop pair, pawn structure, king safety)
- **Result**: F1 = 0.6263 ❌ (vs 0.8957 for outcome-based)
- **Root cause**: Feature-label mismatch

**The 19 features couldn't predict the 24 heuristic labels because they weren't designed to!**

### The V7 Solution

- **Features**: ALL 51 features (fast + heuristics + complexity)
- **Labels**: Real-time Stockfish evaluation during gameplay
- **Training**: Self-play with actual move outcomes
- **Result**: Model learns correlations naturally through experience

**No pre-labeling needed. No feature-label mismatch. Pure reinforcement learning.**

---

## 📊 Feature Breakdown (51 dimensions)

| Category | Features | Description | Count |
|----------|----------|-------------|-------|
| **Piece Counts** | white_pawns, white_knights, ..., black_kings | Basic material | 12 |
| **Material** | material_balance | White - Black material | 1 |
| **State** | side_to_move, castling, in_check | Board state | 4 |
| **Mobility** | current_mobility, opponent_mobility | Legal moves | 2 |
| **Bishop Pair** | white/black_bishop_pair, advantage | Both bishops present | 3 |
| **Passed Pawns** | white/black_passed_pawns, advantage | Unstoppable pawns | 3 |
| **Doubled Pawns** | white/black_doubled_pawns, disadvantage | Structural weakness | 3 |
| **Isolated Pawns** | white/black_isolated_pawns, disadvantage | Pawn structure | 3 |
| **King Safety** | white/black_king_pawn_shield, advantage | Pawn shield strength | 3 |
| **Active Rooks** | white/black_active_rooks, advantage | Rooks on open files/7th | 3 |
| **Development** | white/black_development_score, advantage | Pieces off back rank | 3 |
| **Mobility (norm)** | white/black_mobility_normalized, advantage | Attacked squares / 64 | 3 |
| **Complexity** | legal_moves, captures, checks, tension | Tactical density | 4 |
| **Strategic** | center_control, game_phase, move_diversity | Position type | 3 |
| **V7P3R Custom** | forest_darkness_score | Complexity metric | 1 |
| **TOTAL** | | | **51** |

---

## 🎮 Self-Play Training Pipeline

### Phase 1: Data Collection (Self-Play)

```python
# Play 1000 games
for game_id in range(1000):
    board = chess.Board()
    game_data = []
    
    while not board.is_game_over():
        # Extract features
        features = extractor.extract_all_features(board)
        
        # Neural network evaluates position
        position_value = network.evaluate(features)
        
        # MCTS selects move
        move = mcts.select_move(board, network)
        
        # Query Stockfish for oracle evaluation
        stockfish_eval = stockfish.evaluate(board.fen())
        
        # Record experience
        game_data.append({
            'features': features,
            'move': move,
            'stockfish_eval': stockfish_eval,
            'forest_darkness': features[50],  # Custom metric
        })
        
        # Make move
        board.push(move)
    
    # Game over - record outcome
    result = board.result()
    backpropagate_rewards(game_data, result)
```

### Phase 2: Training (Experience Replay)

```python
# Sample mini-batch from collected games
batch = sample_experience_buffer(batch_size=256)

# Extract features and targets
X = np.array([exp['features'] for exp in batch])
y_stockfish = np.array([exp['stockfish_eval'] for exp in batch])
y_result = np.array([exp['game_result'] for exp in batch])

# Apply personality rewards
personality_rewards = calculate_personality_bonus(batch)

# Combined target
y_target = (y_stockfish * 0.7 + 
            y_result * 0.3 + 
            personality_rewards * 0.2)

# Train network
network.train(X, y_target)
```

### Phase 3: Iteration

1. Play 100 games with current network
2. Collect ~3000 positions (avg 30 moves/game)
3. Train for 10 epochs on collected data
4. Evaluate improvement (play vs previous version)
5. If win rate > 55%, accept new network
6. Repeat

---

## 🎯 V7P3R Personality Integration

### Complexity Seeking (Tal-Style)

```python
# Reward high forest darkness scores
if forest_darkness > 0.6:
    personality_bonus += 0.15
```

### Material Sacrifice Tolerance

```python
# Don't penalize material loss if complexity increases
if material_loss <= 5 and forest_darkness_increase > 0.2:
    personality_bonus += 0.10
```

### King Risk Acceptance

```python
# Tolerate reduced king safety if attacking
if king_safety_loss <= 2 and opponent_king_under_pressure:
    personality_bonus += 0.08
```

### Center Control Emphasis

```python
# Reward center control more than Stockfish would
personality_bonus += center_control * 0.05
```

---

## 📈 Expected Results

### Advantages Over v1-v6

1. **No feature-label mismatch** - Features and training signal are aligned
2. **No pre-labeling bias** - Model learns from actual gameplay
3. **Comprehensive features** - 51 dims capture everything observable
4. **Personality integration** - Custom rewards shape playing style
5. **Continuous improvement** - Self-play generates infinite training data

### Predicted Performance

- **Opening**: Strong (development, center control features)
- **Midgame**: Aggressive (forest darkness rewards)
- **Endgame**: Competent (game phase, passed pawns features)
- **Tactics**: Sharp (piece tension, checks, captures features)
- **Strategy**: Tal-inspired (complexity > safety)

---

## 🔬 Comparison to AlphaZero

| Aspect | AlphaZero | V7P3R v7.0 |
|--------|-----------|------------|
| **Input** | Raw board (19x8x8 planes) | 51 explicit features |
| **Training** | Pure self-play | Self-play + Stockfish oracle |
| **Personality** | Emergent | Explicit (reward weights) |
| **Computational** | Massive (5000 TPUs) | Modest (CPU/single GPU) |
| **Interpretability** | Black box | Transparent (feature-based) |
| **Data Efficiency** | Low (millions of games) | High (thousands of games) |

**V7P3R's advantage**: Explicit features make training faster and results interpretable. Custom rewards encode personality directly.

---

## 🚀 Implementation Roadmap

### Phase 1: Core Components (Week 1)
- [x] Comprehensive feature extractor (51 dims)
- [ ] Neural network value head
- [ ] MCTS move selection
- [ ] Stockfish oracle integration

### Phase 2: Training Loop (Week 2)
- [ ] Self-play data collection
- [ ] Experience replay buffer
- [ ] Personality reward calculation
- [ ] Training pipeline

### Phase 3: Validation (Week 3)
- [ ] Play 100 games vs Stockfish (handicapped)
- [ ] Measure personality alignment (complexity metrics)
- [ ] Compare to v18.6.3 baseline
- [ ] Tune hyperparameters

### Phase 4: Production (Week 4)
- [ ] UCI integration
- [ ] Time management
- [ ] Opening book
- [ ] Deploy to Lichess

---

## 💡 Key Insights

1. **Feature engineering matters** - 51 explicit features outperform 19 for this approach
2. **Training signal quality > dataset size** - Real gameplay beats pre-labeled positions
3. **Personality through rewards** - Custom weights shape playing style without compromising strength
4. **Stockfish as teacher** - Learn "good chess" first, add personality second
5. **Iterative improvement** - Self-play generates better data as model improves

---

## 📝 Research Questions

- How many self-play games needed for convergence?
- Optimal personality reward weight (currently 0.2)?
- Should Stockfish weight decrease over time (bootstrap then self-play)?
- Can we add policy head (move probabilities) in addition to value head?
- Would curriculum learning help (simple → complex positions)?

---

## 🎉 Why This is Exciting

**V7 combines the best of all V6 research**:
- Stage 1's fast features (efficiency)
- Heuristic grading's comprehensive evaluation (quality)
- Stage 2's complexity metrics (personality)
- Continuous improvement cycle (adaptability)

**Plus eliminates the problems**:
- No feature-label mismatch
- No pre-labeling bias
- No dataset staleness
- No 2-stage coordination issues

**Result**: A single, unified, self-improving chess engine that plays Tal-style aggressive chess while continuously learning from experience.

---

*"The model doesn't memorize positions - it learns to think about chess through the lens of ~50 observable features, guided by Stockfish's wisdom and V7P3R's personality."*
