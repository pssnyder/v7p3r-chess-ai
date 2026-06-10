# V7 Training Workflow

## Version Overview

- **v7.0**: Pure self-play training (same model vs itself)
- **v7.1**: Generational training (new model vs previous best) ⭐ **RECOMMENDED**

---

## v7.1 Generational Training (Recommended)

### High-Level Overview

```
┌──────────────────────────────────────────────────────────────┐
│              GENERATIONAL TRAINING CYCLE                      │
│                                                                │
│  Generation N:                                                │
│  [Train via Self-Play] → [Evaluate vs Gen N-1] → Decision    │
│                                    ↓                           │
│                          Win Rate > 50%?                      │
│                          ├─ Yes → Accept as Gen N+1           │
│                          └─ No  → Keep Gen N-1                │
│                                                                │
│  Repeat for 10-50 generations                                 │
└──────────────────────────────────────────────────────────────┘
```

**Philosophy**: Like AlphaZero - each generation must prove it's better than the last through competitive play.

**Key Improvements over v7.0:**
- ✅ **Meaningful metrics**: Win rate measures actual improvement
- ✅ **Color balance**: 3 games as White, 3 as Black
- ✅ **Better endgame**: 100% Stockfish weight (up from 50%)
- ✅ **Controlled chaos**: 20% SF in middlegame (up from 10%)

**Usage:**
```bash
cd v7.0/src
python train_generational.py  # Runs full generational cycle
```

See [V7.1_GENERATIONAL_TRAINING.md](V7.1_GENERATIONAL_TRAINING.md) for complete details.

---

## v7.0 Self-Play Training (Legacy)

### High-Level Overview

```
┌──────────────────────────────────────────────────────────┐
│                  SELF-PLAY TRAINING LOOP                  │
│                                                            │
│  [Play 10 Games] → [Collect Data] → [Train Network]      │
│         ↑                                     │            │
│         └─────────────────────────────────────┘            │
│                   Repeat 10 times                          │
│              (100 games total in this run)                 │
└──────────────────────────────────────────────────────────┘
```

**Philosophy**: Learn good chess from Stockfish, add personality through reward shaping, improve network through gameplay experience.

**Note**: v7.0 has a critical flaw - win/loss metrics are meaningless because the same model plays both sides. Use v7.1 instead.

---

## Detailed Workflow Stages

### Stage 1: Initialization (Once Per Training Session)

```
┌─────────────────────────────────────────────────┐
│           INITIALIZATION STAGE                   │
├─────────────────────────────────────────────────┤
│                                                  │
│  1. Load Neural Network                          │
│     ├─ V7ValueNetwork (51 input → 1 output)     │
│     ├─ 55,425 parameters                         │
│     └─ Random weights (untrained)                │
│                                                  │
│  2. Start Stockfish Oracle                       │
│     ├─ Depth: 15                                 │
│     ├─ Time: 1000ms per position                 │
│     └─ Thread: 1, Hash: 128MB                    │
│                                                  │
│  3. Load Personality Profile                     │
│     ├─ DarkForestAssassin.json                   │
│     ├─ 14 personality parameters                 │
│     └─ Creates PersonalityRewardCalculator       │
│                                                  │
│  4. Initialize Experience Buffer                 │
│     ├─ Empty at start                            │
│     └─ Will store game experiences               │
│                                                  │
└─────────────────────────────────────────────────┘
```

**What This Does:**
- **Network**: Creates the "brain" that will learn chess
- **Stockfish**: Objective teacher providing "good chess" signal
- **Personality**: Your custom "Dark Forest Assassin" style overlay
- **Buffer**: Memory storage for learning from games

**Duration**: ~1-2 seconds

---

### Stage 2: Game Playing (Repeated 10 Times)

```
┌────────────────────────────────────────────────────────────┐
│                    PLAY SINGLE GAME                         │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  FOR EACH MOVE (up to 200 moves):                          │
│                                                             │
│    ┌─────────────────────────────────────────────┐         │
│    │  A. Position Evaluation                      │         │
│    │     ├─ Extract 51 features from board        │         │
│    │     ├─ Feed to neural network                │         │
│    │     └─ Get value for each legal move         │         │
│    └─────────────────────────────────────────────┘         │
│                   ↓                                         │
│    ┌─────────────────────────────────────────────┐         │
│    │  B. Move Selection (Temperature Sampling)    │         │
│    │     ├─ Apply softmax with temp=0.3           │         │
│    │     ├─ Higher temp = more exploration        │         │
│    │     └─ Select move probabilistically         │         │
│    └─────────────────────────────────────────────┘         │
│                   ↓                                         │
│    ┌─────────────────────────────────────────────┐         │
│    │  C. Make Move on Board                       │         │
│    │     └─ Update chess position                 │         │
│    └─────────────────────────────────────────────┘         │
│                   ↓                                         │
│    ┌─────────────────────────────────────────────┐         │
│    │  D. Evaluate Position (Training Signal)      │         │
│    │     ├─ Stockfish evaluation (~67ms)          │         │
│    │     ├─ Personality reward calculation        │         │
│    │     ├─ Feature extraction (51 dims)          │         │
│    │     └─ Store experience in buffer            │         │
│    └─────────────────────────────────────────────┘         │
│                                                             │
│  LOOP until:                                                │
│    - Checkmate found                                        │
│    - Stalemate/draw                                         │
│    - 200 moves reached (current limit)                      │
│                                                             │
│  Save PGN: training/v7_selfplay/game_XXXX.pgn              │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

**What's Happening Under the Hood:**

#### A. Position Evaluation (Per Move)
```python
# Extract comprehensive features (51 dimensions)
features = extractor.extract_all_features(board)
# Features include:
# - Stage 1: Basic material, mobility, castling (19 dims)
# - Heuristics: Bishop pair, passed pawns, king safety (24 dims)  
# - Complexity: Forest darkness, tension, diversity (8 dims)

# Feed to network
value = network.predict(features)  # Returns value in [-1, 1]
```

#### B. Temperature Sampling
```python
# Evaluate all legal moves
move_values = []
for move in legal_moves:
    board.push(move)
    move_values.append(network.predict(board))
    board.pop()

# Softmax with temperature
probabilities = softmax(move_values / temperature)

# Select move (weighted random)
selected_move = np.random.choice(legal_moves, p=probabilities)
```

**Why Temperature?**
- `temp = 0.0`: Always pick best move (exploitation, no learning)
- `temp = 1.0`: More random (pure exploration)
- **`temp = 0.3`**: Balanced (mostly best moves, some exploration)

#### C. Training Signal Collection
```python
# Stockfish evaluation
sf_eval = oracle.evaluate(board)  # Returns normalized score [-1, 1]

# Personality reward
personality = calculator.calculate_total_reward(board, features)
# Components:
#  - Complexity rewards (forest_darkness * 0.15)
#  - Sacrifice bonus (material_sacrifice * 0.10)
#  - King safety penalty (king_risk * -0.03)
#  - Strategic bonuses (center, passed pawns, etc.)

# Store experience
experience = GameExperience(
    fen=board.fen(),
    features=features,  # 51 dims
    stockfish_eval=sf_eval,  # -1 to +1
    personality_reward=personality,  # 0 to ~0.4
    game_outcome=None,  # Will be filled after game ends
    forest_darkness=features['forest_darkness'],
    # ... other metadata
)
```

**Duration**: 
- ~6-9 moves/second (depends on position complexity)
- 200-move game: 20-30 seconds

---

### Stage 3: Data Collection (After Each Game)

```
┌────────────────────────────────────────────────────────┐
│              GAME DATA PROCESSING                       │
├────────────────────────────────────────────────────────┤
│                                                         │
│  1. Determine Game Outcome                              │
│     ├─ Checkmate: +1.0 for winner, -1.0 for loser     │
│     ├─ Stalemate/Draw: 0.0 for both                    │
│     └─ Max moves (200): 0.0 (inconclusive)             │
│                                                         │
│  2. Update All Experiences with Outcome                 │
│     └─ Propagate game result to all positions          │
│                                                         │
│  3. Calculate Training Target (per position)            │
│     ┌──────────────────────────────────────────────┐   │
│     │  target = 0.7 * stockfish_eval               │   │
│     │          + 0.2 * personality_reward           │   │
│     │          + 0.1 * game_outcome                 │   │
│     └──────────────────────────────────────────────┘   │
│                                                         │
│  4. Add to Experience Buffer                            │
│     ├─ Stores all positions from game                  │
│     └─ Typical game: ~200 experiences                  │
│                                                         │
│  5. Save Game Record                                    │
│     ├─ PGN file (for replay/analysis)                  │
│     └─ GameResult metadata (moves, duration, stats)    │
│                                                         │
└────────────────────────────────────────────────────────┘
```

**Example Training Target Calculation:**

Position from Game 1, Move 47:
```python
stockfish_eval = +0.139 (White slightly better)
personality_reward = +0.282 (High complexity, sacrifices)
game_outcome = 0.0 (Max moves, no winner)

target = 0.7 * 0.139 + 0.2 * 0.282 + 0.1 * 0.0
       = 0.0973 + 0.0564 + 0.0
       = 0.1537  # Network should output ~0.15 for this position
```

**What This Means:**
- **70% Stockfish**: Network learns objectively good chess
- **20% Personality**: Network learns aggressive/complex style
- **10% Outcome**: Network learns what actually wins games

**Duration**: <1 second per game

---

### Stage 4: Network Training (Every 10 Games)

```
┌──────────────────────────────────────────────────────────────┐
│                  NEURAL NETWORK TRAINING                      │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  TRIGGER: After games 10, 20, 30, ... 100                    │
│                                                               │
│  1. Experience Buffer Status                                  │
│     ├─ 10 games played                                        │
│     ├─ ~2,000 positions collected                            │
│     └─ Each has: features (51d), target (scalar)             │
│                                                               │
│  2. Split Data (80/20 Train/Val)                              │
│     ├─ Training: 1,600 positions                             │
│     └─ Validation: 400 positions                             │
│                                                               │
│  3. Training Loop (1 epoch)                                   │
│     ┌───────────────────────────────────────────────┐        │
│     │  FOR EACH BATCH (256 positions):              │        │
│     │                                                │        │
│     │    A. Forward Pass                             │        │
│     │       ├─ Network predicts values               │        │
│     │       └─ predictions: [256 x 1] tensor         │        │
│     │                                                │        │
│     │    B. Calculate Loss                           │        │
│     │       ├─ MSE = mean((pred - target)²)         │        │
│     │       └─ Example: loss = 0.4867                │        │
│     │                                                │        │
│     │    C. Backward Pass                            │        │
│     │       ├─ Calculate gradients                   │        │
│     │       └─ Update 55,425 parameters              │        │
│     │                                                │        │
│     │    D. Optimizer Step (Adam)                    │        │
│     │       ├─ Learning rate: 0.001                  │        │
│     │       └─ Weight decay: 1e-5                    │        │
│     │                                                │        │
│     └───────────────────────────────────────────────┘        │
│                                                               │
│  4. Validation                                                │
│     ├─ Evaluate on 400 held-out positions                    │
│     └─ Check if network is overfitting                       │
│                                                               │
│  5. Save Checkpoint                                           │
│     ├─ model_game_0010.pt (network weights)                  │
│     └─ stats_game_0010.json (training metrics)               │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

**Under the Hood: Network Update Math**

```python
# Training batch (simplified)
batch_features = torch.tensor([...])  # [256, 51]
batch_targets = torch.tensor([...])   # [256, 1]

# Forward pass
predictions = network(batch_features)  # [256, 1]

# Loss calculation
loss = MSELoss(predictions, batch_targets)
# Example: predictions = [0.15, -0.32, 0.08, ...]
#          targets     = [0.12, -0.28, 0.10, ...]
#          loss        = mean((0.15-0.12)² + (-0.32-(-0.28))² + ...)
#                      = 0.4867

# Backward pass
loss.backward()  # Computes ∂loss/∂weight for all 55,425 parameters

# Optimizer update
optimizer.step()  # weight_new = weight_old - lr * gradient
```

**What Gets Updated:**
- Layer 1 (input → hidden1): 51 × 256 = 13,056 weights
- Layer 2 (hidden1 → hidden2): 256 × 128 = 32,768 weights
- Layer 3 (hidden2 → output): 128 × 64 = 8,192 weights
- Output (value head): 64 × 1 = 64 weights
- Biases: 256 + 128 + 64 + 1 = 449
- **Total: 55,425 parameters updated**

**Why Only 1 Epoch Per Training?**
- We're doing **continual learning** (not offline training)
- Network sees new data every 10 games
- Multiple epochs risk overfitting to recent games
- Next 10 games will provide fresh data

**Duration**: 
- ~0.1 seconds (very fast with only 2,000 samples)
- Would be longer with more games (e.g., 100 games = ~1-2 seconds)

---

### Stage 5: Progress Tracking (Every 10 Games)

```
┌─────────────────────────────────────────────────────┐
│              CHECKPOINT & METRICS                    │
├─────────────────────────────────────────────────────┤
│                                                      │
│  1. Save Network Checkpoint                          │
│     └─ model_game_0010.pt (can resume if crash)     │
│                                                      │
│  2. Calculate Statistics                             │
│     ├─ Win/Draw Rate                                 │
│     ├─ Avg Forest Darkness                           │
│     ├─ Avg Personality Reward                        │
│     ├─ Total Sacrifices                              │
│     └─ Training Loss                                 │
│                                                      │
│  3. Save Statistics JSON                             │
│     └─ stats_game_0010.json                          │
│                                                      │
│  4. Display Progress Summary                         │
│     └─ Terminal output with metrics                  │
│                                                      │
└─────────────────────────────────────────────────────┘
```

**Example stats_game_0010.json:**
```json
{
  "total_experiences": 1953,
  "total_games": 10,
  "avg_forest_darkness": 0.348,
  "avg_personality_reward": 0.267,
  "total_sacrifices": 15,
  "win_rate": 0.2,
  "draw_rate": 0.0
}
```

**What You Learn:**
- **Win rate 0.2 (20%)**: 2 out of 10 games ended in checkmate
- **Draw rate 0.0**: No draws (most games hit 200-move limit)
- **Forest darkness 0.348**: Good complexity (personality emerging)
- **Sacrifices 15**: 1.5 sacrifices/game (aggressive style working)

---

## Complete 100-Game Training Timeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                     FULL TRAINING SESSION                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Games 1-10   → Play → Train → Checkpoint (model_game_0010.pt)      │
│  [Network Version 1]                                                  │
│                                                                       │
│  Games 11-20  → Play → Train → Checkpoint (model_game_0020.pt)      │
│  [Network Version 2 - slightly better at chess]                      │
│                                                                       │
│  Games 21-30  → Play → Train → Checkpoint (model_game_0030.pt)      │
│  [Network Version 3 - learning patterns]                             │
│                                                                       │
│  Games 31-40  → Play → Train → Checkpoint (model_game_0040.pt)      │
│  [Network Version 4 - personality + chess converging]                │
│                                                                       │
│  Games 41-50  → Play → Train → Checkpoint (model_game_0050.pt)      │
│  [Network Version 5 - midpoint]                                      │
│                                                                       │
│  Games 51-60  → Play → Train → Checkpoint (model_game_0060.pt)      │
│  [Network Version 6 - improving]                                     │
│                                                                       │
│  Games 61-70  → Play → Train → Checkpoint (model_game_0070.pt)      │
│  [Network Version 7 - more refined]                                  │
│                                                                       │
│  Games 71-80  → Play → Train → Checkpoint (model_game_0080.pt)      │
│  [Network Version 8 - near complete]                                 │
│                                                                       │
│  Games 81-90  → Play → Train → Checkpoint (model_game_0090.pt)      │
│  [Network Version 9 - polishing]                                     │
│                                                                       │
│  Games 91-100 → Play → Train → Final (model_final.pt)               │
│  [Network Version 10 - COMPLETE]                                     │
│                                                                       │
│  Generate Final Report: training_report.json                         │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

**Expected Evolution:**

| Checkpoint | Games Seen | Experiences | Expected Behavior |
|------------|------------|-------------|-------------------|
| game_0010  | 10         | ~2,000      | Random play, learning basics |
| game_0030  | 30         | ~6,000      | Recognizing material, avoiding blunders |
| game_0050  | 50         | ~10,000     | Simple tactics, personality emerging |
| game_0070  | 70         | ~14,000     | Complex sacrifices, forest darkness increasing |
| game_0100  | 100        | ~20,000     | Full Dark Forest Assassin style |

---

## Key Performance Metrics

### System Resources (Measured)

```
┌─────────────────────────────────────────────────┐
│  CPU Usage: 0.0% (minimal during training)      │
│  Memory: 316 MB (0.1% of system)                │
│  Stockfish Time: 67.1ms avg per position        │
│  Positions/sec: 8.5 during training              │
│  Game Speed: 6-9 moves/sec during play          │
└─────────────────────────────────────────────────┘
```

### Training Speed Estimate

```
Single Game:
  - 200 moves × 67ms Stockfish = 13.4s evaluation time
  - Movement + feature extraction: ~7-16s
  - Total: ~20-30s per game

10 Games:
  - 10 × 25s average = 250s (~4 minutes)
  - Training: 0.1s (negligible)
  - Total per checkpoint: ~4 minutes

100 Games:
  - 10 checkpoints × 4 minutes = 40 minutes
  - ESTIMATED TOTAL: 35-45 minutes
```

---

## Data Flow Diagram

```
                    ┌──────────────────────┐
                    │   Chess Position     │
                    │   (Board State)      │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Feature Extraction  │
                    │   (51 dimensions)    │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼────────┐  ┌───▼────────┐  ┌───▼──────────┐
    │ Neural Network   │  │ Stockfish  │  │ Personality  │
    │ (Move Selection) │  │ (Truth)    │  │ (Style)      │
    └─────────┬────────┘  └───┬────────┘  └───┬──────────┘
              │               │                │
              │        ┌──────▼────────────────▼───────┐
              │        │  Training Target = 0.7S      │
              │        │                 + 0.2P       │
              │        │                 + 0.1O       │
              │        └──────┬───────────────────────┘
              │               │
              │        ┌──────▼────────────┐
              │        │ Experience Buffer │
              │        │ (Memory)          │
              │        └──────┬────────────┘
              │               │
              │         [Every 10 games]
              │               │
              │        ┌──────▼────────────┐
              └────────►  Network Training │
                       │  (Gradient Desc)  │
                       └───────────────────┘
```

---

## What Makes This Different from Traditional Training?

### Traditional Supervised Learning:
```
┌──────────────────────────────────────────┐
│ Fixed Dataset → Train → Done             │
│                                           │
│ - Data collected first                    │
│ - Network trained on all data             │
│ - No feedback loop                        │
└──────────────────────────────────────────┘
```

### V7 Self-Play Training:
```
┌─────────────────────────────────────────────────────┐
│ Play → Learn → Improve → Play Better → Learn More  │
│   ↑                                            │     │
│   └────────────────────────────────────────────┘     │
│                                                      │
│ - Data generated during training                     │
│ - Network plays its own games                        │
│ - Continuous improvement loop                        │
│ - Personality emerges from reward shaping            │
└─────────────────────────────────────────────────────┘
```

---

## Expected Training Outcomes

### After 100 Games (Baseline):

**Quantitative:**
- Win rate: ~20-30% (2-3 out of 10 end in checkmate)
- Avg forest darkness: 0.35-0.40
- Sacrifices/game: 1-2
- Max-move games: ~70% (7-8 out of 10)

**Qualitative:**
- Network learns basic chess rules
- Understands material balance
- Begins to recognize tactical patterns
- Personality (aggression) starts emerging
- **BUT**: Struggles to finish games (endgame weakness)

### After Tablebase Integration (Projected):

**Quantitative:**
- Natural conclusion rate: 70%+ (vs current 10-20%)
- Avg game length: 130 moves (vs 195)
- Checkmate rate: 90%+ in decisive games

**Qualitative:**
- Same aggressive middlegame (forest darkness preserved)
- **Perfect endgame technique** (tablebase lookups)
- Games actually finish with checkmate
- Technical precision in conversions

---

## Troubleshooting Guide

### Issue: All Games Hit 200-Move Limit

**Why:**
- Network has no concept of checkmate conversion
- Optimizes for complexity/position quality, not mate
- No endgame database to guide final moves

**Solution:**
- Implement Syzygy tablebase integration
- See: V7_TABLEBASE_INTEGRATION.md

### Issue: Low Win Rate (<10%)

**Why:**
- Network still random (early training)
- Need more games for pattern recognition
- Temperature too high (too exploratory)

**Solution:**
- Run more training games (100 → 500+)
- Reduce temperature from 0.3 → 0.2
- Check Stockfish oracle is running

### Issue: Loss Not Decreasing

**Why:**
- Learning rate too high/low
- Not enough training data
- Network capacity too small/large

**Solution:**
- Check learning rate schedule (ReduceLROnPlateau)
- Increase batch size (256 → 512)
- Train for more epochs per checkpoint (1 → 3)

### Issue: Personality Not Emerging

**Why:**
- Personality weight too low (20%)
- Stockfish signal dominating (70%)
- Position features not capturing complexity

**Solution:**
- Increase personality weight (20% → 30%)
- Check forest_darkness calculation
- Verify PersonalityRewardCalculator working

---

## Next Steps After 100-Game Baseline

1. **Analyze Results**
   - Run `monitor_training.py`
   - Check forest darkness trends
   - Measure checkmate conversion rate

2. **Implement Tablebases**
   - Download Syzygy 3-4-5 (~1 GB)
   - Integrate `tablebase_oracle.py`
   - Run 100-game comparison

3. **Scale Training**
   - 500-1000 games for stronger network
   - Multi-threading for faster data collection
   - Distributed Stockfish for parallel evaluation

4. **Deploy to UCI Engine**
   - Integrate trained network into V7P3R engine
   - Add time management
   - Deploy to Lichess for real games

---

## Summary: "Under the Hood" in One Sentence

**Each game: Network guesses moves → Stockfish says "actually this is better" → Personality says "but make it spicy" → Game outcome says "and this approach won/lost" → Network updates weights to do better next time → Repeat 100 times → Network learns to play chess with your style.**

---

**Status**: Training in progress (Games 1-10 completed, checkpoint saved)  
**Next Checkpoint**: After Game 20 (~8 minutes from now)
