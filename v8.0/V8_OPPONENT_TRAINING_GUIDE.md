# V8.0 Opponent-Based Training System

## Overview

Enhanced training architecture that replaces pure self-play with **opponent diversity training**. The model learns by playing against a curated pool of UCI chess engines with varying styles and strengths.

---

## Key Changes from Baseline v8.0

### ❌ **Removed: Pure Self-Play**
- **Problem**: Echo chamber effect - model plays only against itself
- **Symptoms**: Mobility-only focus (91.7%), high draw rates (78-87%), plateauing after Gen 10
- **Result**: Limited strategic diversity, catastrophic forgetting (king safety: 87.8% → 0%)

### ✅ **Added: Opponent Diversity Training**
- **Solution**: Train against 6 different UCI opponents with varying strengths/styles
- **Benefits**: 
  - Learn to exploit specific weaknesses
  - Forced exploration of multiple strategies
  - More robust feature learning
  - Faster convergence to strong play

---

## System Architecture

### 1. **UCI Interface** (`v8_uci.py`)
- Full UCI protocol implementation
- Compatible with Arena, Cutechess, etc.
- Loads trained Gen 10 network
- Supports both **deployment** and **training** use
- Temperature control (deterministic vs creative play)
- Opening book integration (100 variations)
- Tablebase support (5-piece Syzygy)

**Usage:**
```bash
python src/v8_uci.py
# Responds to standard UCI commands: uci, isready, position, go, quit
```

### 2. **Opponent Manager** (`opponent_manager.py`)
- Manages pool of 6 UCI opponents
- Weighted selection (prioritize v7p3r versions)
- Rotation strategies: round-robin, weighted-random, weakest-first
- Statistics tracking per opponent
- Subprocess lifecycle management

**Opponent Pool:**
```
Random Opponent        - ELO  600 (10% weight, random baseline)
Material Opponent v2.0 - ELO 1100 (15% weight, tactical style)
Positional Opponent v2 - ELO 1200 (15% weight, positional style)
V7P3R v17.1            - ELO 1700 (20% weight, balanced) ← PRIMARY TARGET
V7P3R v17.8            - ELO 1800 (20% weight, aggressive) ← PRIMARY TARGET
V7P3R v18.3            - ELO 1850 (20% weight, balanced) ← PRIMARY TARGET
```

### 3. **UCI Game Executor** (`uci_game_executor.py`)
- Executes games between v8.0 and UCI opponents
- Alternates moves via UCI protocol
- Collects positions + outcomes for training
- Fast games (3s/move, 200 move limit)
- Handles time controls, tablebase stops

**Game Flow:**
1. Launch opponent engine (subprocess)
2. Apply opening book macro (diversity)
3. Alternate moves (v8 neural network vs opponent UCI)
4. Stop at checkmate/stalemate/tablebase/max moves
5. Collect all v8 positions with final outcome labels
6. Clean up opponent subprocess

### 4. **Enhanced Trainer** (`train_v8_opponents.py`)
- Replaces `train_v8.py` pure self-play with opponent-based games
- Same network architecture (V8ValueNetwork + RewardShaper)
- Same speed (~300-400 games/hour, no Stockfish analysis)
- Enhanced statistics: per-opponent win rates, ELO progression

**Training Loop:**
```
For each generation (20 total):
  1. Play 100 games vs opponents (weighted random selection)
     - 50 games as white, 50 as black (balanced)
     - ~20 games vs v17.1, ~20 vs v17.8, ~20 vs v18.3, ~40 vs themed
  2. Collect experiences (positions + outcomes)
  3. Train value network (3 epochs, batch 512)
  4. Train reward shaper (learn feature importance)
  5. Save generation checkpoint
  6. Print opponent statistics
```

---

## Training Configuration

```python
num_generations = 20             # More than baseline (10)
games_per_generation = 100       # Same as baseline
batch_size = 512                 # Larger than baseline (256)
max_moves_per_game = 200         # Same as baseline
movetime_ms = 3000               # 3 seconds/move (fast)
temperature = 0.3                # Some exploration
```

**Expected Duration:** 3-4 hours (similar to baseline 10-gen training)

**Expected Outcomes:**
- **Total games:** ~2000 (vs baseline 1000)
- **Speed:** ~400-500 games/hour
- **Win rate targets:**
  - Random Opponent: >95% by Gen 20
  - Material/Positional: >80% by Gen 20
  - V7P3R v17.1: >60% by Gen 20
  - V7P3R v17.8: >50% by Gen 20
  - V7P3R v18.3: >40% by Gen 20

---

## File Structure

```
v8.0/
├── src/
│   ├── v8_uci.py                    ← UCI interface (NEW)
│   ├── opponent_manager.py          ← Opponent pool manager (NEW)
│   ├── uci_game_executor.py         ← Game execution (NEW)
│   ├── train_v8_opponents.py        ← Enhanced trainer (NEW)
│   ├── network.py                   ← V8ValueNetwork (unchanged)
│   ├── reward_shaper.py             ← RewardShaper (unchanged)
│   ├── opening_selector.py          ← Opening book (unchanged)
│   ├── comprehensive_features.py    ← Feature extraction (unchanged)
│   └── tablebase_oracle.py          ← Syzygy integration (unchanged)
│
├── training/
│   ├── v8_generational/             ← Baseline self-play (10 gen)
│   └── v8_opponent_training/        ← Enhanced opponent training (20 gen) NEW
│
├── START_OPPONENT_TRAINING.bat      ← Launch script (NEW)
└── V8_OPPONENT_TRAINING_GUIDE.md    ← This file (NEW)
```

---

## Goals & Success Criteria

### Goal #1: Beat All Historical v7p3r Engines
- **Target:** Win majority games vs v18.3, v17.8, v17.1
- **Metric:** >50% win rate by Gen 20
- **Validation:** Post-training tournament (100 games per version)

### Goal #2: Reach Tablebase in 20-30 Moves
- **Target:** Efficient conversion to won endgames
- **Metric:** Average moves to tablebase <30
- **Validation:** Track tablebase-finish rate (should be >40%)

### Goal #3: Break Mobility-Only Focus
- **Problem:** Baseline Gen 10 learned 91.7% mobility, 0% king safety
- **Target:** More balanced feature weights
- **Metric:** All feature groups >10% weight by Gen 20
- **Validation:** Visualize learned weights each generation

### Goal #4: Reduce Draw Rate
- **Problem:** Baseline 78-87% draws (echo chamber)
- **Target:** <50% draw rate vs themed opponents
- **Metric:** Track draws per opponent type
- **Validation:** More decisive wins/losses vs strong opponents

---

## Running the Training

### Quick Start
```bash
cd v8.0
START_OPPONENT_TRAINING.bat
```

### Manual Launch
```bash
cd v8.0/src
python train_v8_opponents.py
```

### Monitoring Progress
Training prints detailed logs:
- Game results (win/draw/loss per game pair)
- Speed metrics (games/hour)
- Per-opponent statistics (every 5 generations)
- Feature weight evolution (visualized every 5 generations)

### Checkpoints
Saved to `v8.0/training/v8_opponent_training/`:
```
gen_0001_value_network.pt       ← Trained value network
gen_0001_reward_shaper.pt       ← Learned feature weights
gen_0001_stats.json             ← Generation statistics
```

---

## Post-Training Validation

### 1. Test Against Opponents (100-game tournaments)
```bash
# Use engine-tester framework
cd engine-tester
python run_tournament.py \
  --engine1 "../v7p3r-chess-ai/v8.0/src/v8_uci.py" \
  --engine2 "../Tournament Engines/V7P3R/V7P3R_v18.3/src/v7p3r_uci.py" \
  --games 100 \
  --time-control "5+4"
```

### 2. Puzzle Solving Validation
```bash
# Validate tactical strength
cd v7p3r-chess-ai/v8.0/src
python test_puzzle_solving.py --model ../training/v8_opponent_training/gen_0020_value_network.pt
```

### 3. Arena GUI Deployment
- Add v8.0 to Arena engines list
- Run gauntlet vs v7p3r versions
- Track ELO progression

---

## Advantages Over Baseline v8.0

| Aspect | Baseline (Self-Play) | Enhanced (Opponents) |
|--------|---------------------|---------------------|
| **Training games** | 1000 (10 gen × 100) | 2000 (20 gen × 100) |
| **Opponent diversity** | 1 (self) | 6 (varied styles) |
| **Draw rate** | 78-87% | Expected <50% |
| **Feature learning** | Single pattern (mobility) | Multi-pattern (balanced) |
| **Catastrophic forgetting** | High risk | Lower risk |
| **Strategic robustness** | Low (echo chamber) | High (diverse challenges) |
| **Validation** | Self-consistency | External benchmarks |
| **Deployment readiness** | Low confidence | High confidence |

---

## Troubleshooting

### Issue: Opponent Engine Fails to Launch
**Symptoms:** "Failed to launch opponent" errors
**Cause:** Missing .bat file or incorrect path
**Fix:** Verify opponent paths in `opponent_manager.py` line 372+

### Issue: Training Hangs on Opponent Games
**Symptoms:** No progress after "vs Opponent" message
**Cause:** Opponent engine not responding to UCI commands
**Fix:** Test opponent manually: `python opponent_path.py`, type `uci`, check for `uciok`

### Issue: Very Low Win Rate vs All Opponents
**Symptoms:** <20% win rate even vs Random Opponent
**Cause:** Network not learning (possibly random weights)
**Fix:** Verify baseline Gen 10 network loads correctly, check loss decreasing

### Issue: High Draw Rate (>80%)
**Symptoms:** Most games ending in draws
**Cause:** Temperature too low (deterministic play), repetition threshold issues
**Fix:** Increase temperature to 0.5, check opponent isn't drawing excessively

---

## Next Steps After Training

1. **Validate Gen 20 Model**
   - Test puzzle solving accuracy
   - Run 100-game tournament vs v18.3
   - Check feature weight balance

2. **Compare to Baseline**
   - Gen 20 (opponents) vs Gen 10 (self-play)
   - 100-game match
   - Analyze strategic differences

3. **Deploy to Production**
   - If Gen 20 beats v18.3, deploy to Lichess bot
   - Update deployment log
   - Monitor live performance

4. **Iterate Training** (if needed)
   - Add stronger opponents (Stockfish depth 5?)
   - Increase games per generation
   - Tune opponent weights
   - Add reward shaping to prioritize quick wins

---

## References

- **Baseline Training Results:** `V8_TRAINING_RESULTS.md`
- **Architecture Summary:** `V8_ARCHITECTURE_SUMMARY.md`
- **Deployment Guide:** `V8_DEPLOYMENT_GUIDE.md`
- **Version Management:** `../.github/instructions/version_management.instructions.md`

---

**Created:** 2025-01-29  
**Status:** Ready for training  
**Expected Completion:** Gen 20 by 2025-01-29 (3-4 hours)
