# V7P3R AI v6.0 - Two-Stage Learning Architecture

**Status**: Data preparation phase
**Architecture**: Graph-based transposition network with two-stage learning

---

## Overview

V7P3R AI v6.0 introduces a fundamentally different learning approach inspired by how humans learn chess:

### **Stage 1: Positional Memory Network**
- Learn from 5.7M+ positions (Lichess puzzles + V7P3R games)
- Binary classification: Good moves (G0-G1) vs Bad moves (G2-G5)
- Build transposition graph linking similar positions
- Output: Policy network with V7P3R personality encoded

### **Stage 2: Self-Play Reinforcement**
- Generate new positions through guided self-play
- Evaluate with Stage 1 intuition
- Get corrected by Stockfish when wrong
- Expand knowledge graph with new positions

---

## Project Structure

```
v6.0/
├── data/
│   ├── raw/                    # Reference to v5.0 data (symlink/pointer)
│   ├── stage1/                 # Filtered binary dataset + graph
│   │   ├── good_positions.jsonl   (G0 + filtered G1)
│   │   ├── bad_positions.jsonl    (G2-G5)
│   │   └── transposition_graph.pkl
│   └── stage2/                 # Self-play outputs
│       ├── expanded_graph.pkl
│       └── self_play_games.jsonl
├── models/
│   ├── stage1_policy.h5        # Trained policy network
│   └── stage2_policy.h5        # Refined policy network
├── scripts/
│   ├── utils/                  # Shared utilities
│   │   ├── calculate_features.py
│   │   ├── temporal_feature_calculator.py
│   │   └── zobrist_hashing.py
│   ├── stage1/                 # Stage 1 scripts
│   │   ├── filter_dataset.py   # Binary classification filtering
│   │   ├── build_graph.py      # Transposition graph builder
│   │   └── train_policy.py     # (TODO) Graph NN training
│   └── stage2/                 # Stage 2 scripts
│       ├── self_play.py        # (TODO) Self-play framework
│       └── td_learning.py      # (TODO) Temporal difference learning
├── configs/
│   └── stage1_config.yaml
└── docs/
    └── TWO_STAGE_ARCHITECTURE.md (comprehensive design doc)
```

---

## Quick Start

### Phase 1: Data Preparation (Current)

**Step 1: Filter dataset for binary classification**
```powershell
cd E:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v6.0
python scripts\stage1\filter_dataset.py
```

**Output:**
- `data/stage1/good_positions.jsonl` (~5.7M positions, G0 + filtered G1)
- `data/stage1/bad_positions.jsonl` (~33k positions, G2-G5)

**Step 2: Build transposition graph**
```powershell
python scripts\stage1\build_graph.py
```

**Output:**
- `data/stage1/transposition_graph.pkl` (position similarity network)

### Phase 2: Stage 1 Training (Next)

**TODO**: Implement graph-augmented neural network
- Input: 325 features + transposition attention
- Output: Binary probability P(Good Position)
- Training: Weighted loss for imbalanced data

### Phase 3: Stage 2 Self-Play (Future)

**TODO**: Implement self-play framework
- Generate positions using Stage 1 policy
- Evaluate with Stockfish
- Update model via temporal difference learning
- Expand transposition graph

---

## Key Differences from v5.0

| Aspect | v5.0 | v6.0 |
|--------|------|------|
| **Task** | Multi-class (6 grades) | Binary (Good vs Bad) |
| **Architecture** | Standard feedforward NN | Graph-augmented NN |
| **Data** | All sources (incl. failed C0BR4) | Lichess + V7P3R only |
| **Grading** | All grades 0-5 | G0 + filtered G1 (eval variance) |
| **Learning** | Supervised only | Two-stage (supervised + RL) |
| **Novelty** | Temporal features (TPF) | Transposition graph |

---

## Current Status

✅ **Completed:**
- v6.0 directory structure
- Zobrist hashing utility
- Data filtering script
- Transposition graph builder

🚧 **In Progress:**
- Data preparation (run filtering + graph building)

📋 **TODO:**
- Stage 1 training script (graph NN)
- Stage 2 self-play framework
- UCI integration

---

## Data Source

The v6.0 pipeline uses the v5.3 merged dataset from v5.0:
- **Location**: `E:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v5.0\data\final\v7p3r_ai_v5.3_merged.jsonl`
- **Size**: 23.7 GB (6,313,414 positions)
- **Sources**: Lichess puzzles (5.6M) + V7P3R games (194k) + C0BR4 games (492k)

**Filtering Strategy:**
- ✅ Keep: Lichess puzzles + V7P3R games
- ❌ Exclude: C0BR4 games (failed Stockfish analysis)
- ✅ Good: Grade 0 + Grade 1 (if eval diff ≤50cp)
- ❌ Bad: Grades 2-5 (for negative reinforcement)

---

## Architecture Philosophy

### Human Learning Analogy

**Stage 1** = Studying chess books, master games, tactics puzzles
- Builds pattern recognition: "This position type is strong"
- Learns principles: "Control the center, develop pieces"
- Memorizes key ideas: "Passed pawns should be pushed"

**Stage 2** = Playing practice games with a coach
- Tries moves based on learned patterns
- Coach corrects mistakes: "That was a blunder, here's why"
- Updates understanding through experience

### AI Implementation

**Stage 1** = Supervised learning on 5.7M positions
- Input: Position features
- Output: Binary probability P(Good)
- Loss: Weighted binary cross-entropy + graph regularization
- Goal: Compress 5.7M positions into neural network weights

**Stage 2** = Reinforcement learning through self-play
- Input: Stage 1 policy network
- Process: Generate positions → Evaluate → Get Stockfish feedback → Update
- Output: Enhanced policy + expanded transposition graph
- Goal: Generalize beyond training set

---

## Expected Performance

### Stage 1 Targets
- Classification accuracy: >95% on good positions, >70% on bad
- Transposition consistency: Similar positions get similar scores (>0.8 correlation)
- V7P3R style match: >60% agreement with V7P3R's actual moves

### Stage 2 Targets
- Position coverage: Add 50k+ new positions to graph
- Self-consistency: AI predictions match Stockfish >75% on new positions
- Game quality: <20% blunders in self-play games

### Final Engine Targets
- Move quality: >60% Grade 0-1 moves in live play
- Speed: <5 seconds per move (rapid games)
- Win rate: >45% vs 1500-1700 ELO opponents
- Personality: Maintains V7P3R's aggressive style

---

## Next Steps

1. **Run data filtering** (~1-2 hours)
2. **Build transposition graph** (~2-3 hours)
3. **Analyze filtered dataset** (verify quality)
4. **Implement Stage 1 training** (graph NN)
5. **Train Stage 1 model** (~6-12 hours)
6. **Implement Stage 2 self-play** (RL framework)
7. **Run Stage 2 training** (1000+ games)
8. **Integrate with UCI** (deployment)

---

## Requirements

```bash
# Install dependencies
pip install chess numpy tensorflow networkx scikit-learn

# Optional (for faster graph NN search)
pip install faiss-cpu  # or faiss-gpu
```

---

## Contact

For questions or issues, refer to the comprehensive design document:
- `docs/TWO_STAGE_ARCHITECTURE.md`

Or review v5.0 for comparison:
- `../v5.0/QUICK_START_V5.3.md`
