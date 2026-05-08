# Stage 1 Implementation Guide
# Pattern Recognition & Move Ordering

**Status**: 🚧 In Progress  
**Goal**: Train V7P3R Themes Agent on 4M puzzle library  
**ELO Target**: +100-150 improvement

## Overview

Stage 1 focuses on training a neural network to recognize tactical patterns across 4,000,000 chess puzzles and integrate it as an intelligent move ordering system for the V7P3R Chess Engine.

## Prerequisites

### Data Requirements
- ✅ 4M puzzle database available locally
- ✅ Puzzles include themes/categories (pins, forks, mates, etc.)
- ✅ Puzzles have solutions (best moves)

### Hardware Requirements
- GPU: NVIDIA RTX 4070 Ti (or similar)
- RAM: 16GB+ recommended
- Storage: 50GB+ free space

### Software Requirements
- Python 3.11+
- PyTorch 2.0+
- CUDA 12.0+ (for GPU acceleration)
- All dependencies from requirements.txt

## Implementation Steps

### Step 1: Data Preparation (Week 1)

#### 1.1 Locate Puzzle Database
```bash
# Verify puzzle database exists
ls -la "E:/Programming Stuff/Chess Engines/Chess PGNs/training_data/fen_data_lichess_puzzles_db/"

# Expected format: CSV with columns [FEN, Moves, Rating, Themes]
# Or: Multiple CSV files for different puzzle sets
```

#### 1.2 Create Puzzle Dataset Class
Create `src/training/puzzle_dataset.py`:
- Parse puzzle database (CSV/JSON)
- Extract position (FEN), solution moves, themes
- Convert to training examples
- Implement data augmentation (optional: mirror positions)

**Key Components**:
```python
class PuzzleDataset(torch.utils.data.Dataset):
    def __init__(self, puzzle_db_path, split='train'):
        # Load puzzles
        # Split into train/val/test (80/10/10)
        pass
    
    def __getitem__(self, idx):
        # Return:
        # - position_features (690-dim)
        # - theme_labels (50-dim multi-hot)
        # - best_move_encoding (64-dim)
        # - move_score (float)
        pass
```

#### 1.3 Port ChessState Feature Extractor from v3.0
```bash
# Copy from v3.0
cp ../v3.0/src/training/chess_state.py src/core/chess_state_extractor.py

# Update imports and class name
# ChessState → ChessStateExtractor
# Verify 690-dimensional output
```

**Testing**:
```bash
python -c "
from src.core.chess_state_extractor import ChessStateExtractor
import chess

extractor = ChessStateExtractor()
board = chess.Board()
features = extractor.extract(board)
assert len(features) == 690, f'Expected 690 features, got {len(features)}'
print('✅ Feature extractor working')
"
```

#### 1.4 Data Pipeline Testing
```python
# Test script: scripts/test_data_pipeline.py
from src.training.puzzle_dataset import PuzzleDataset
from torch.utils.data import DataLoader

# Load small sample
dataset = PuzzleDataset('data/puzzles/4M_puzzle_library/', split='train')
print(f"Total puzzles: {len(dataset)}")

# Test batch loading
loader = DataLoader(dataset, batch_size=64, shuffle=True)
batch = next(iter(loader))
print(f"Batch shapes: {[x.shape for x in batch]}")
```

**Success Criteria**:
- ✅ Can load all 4M puzzles without errors
- ✅ Train/val/test splits correct (3.2M/400K/400K)
- ✅ Feature extraction working (690-dim output)
- ✅ Batch loading fast (<1 second per batch)

---

### Step 2: Model Architecture (Week 1-2)

#### 2.1 Theme Classifier Network
Already implemented in `src/agents/v7p3r_themes_agent.py`

**Architecture**:
```
Input: 690 features
→ Linear(690 → 512) + ReLU + Dropout(0.3)
→ Linear(512 → 384) + ReLU + Dropout(0.3)
→ Linear(384 → 256) + ReLU
→ Linear(256 → 128) + ReLU
→ Linear(128 → 50) + Sigmoid  # Multi-label classification
Output: 50 theme probabilities
```

#### 2.2 Move Ranking Network
Already implemented in `src/agents/v7p3r_themes_agent.py`

**Architecture**:
```
Input: 690 position features + 64 move encoding = 754 total
→ Linear(754 → 512) + ReLU + Dropout(0.3)
→ Linear(512 → 256) + ReLU + Dropout(0.3)
→ Linear(256 → 128) + ReLU
→ Linear(128 → 1) + Sigmoid  # Move quality score [0, 1]
Output: Move score
```

#### 2.3 Test Models (Untrained)
```bash
python src/agents/v7p3r_themes_agent.py
# Should output:
# Dominant theme: <random> (confidence: <low>)
# Top 5 moves: [...]
# Inference time: <X>ms
```

---

### Step 3: Training (Week 2-3)

#### 3.1 Configure Training
Edit `config/training_config.json`:
```json
{
  "stage1": {
    "data": {
      "puzzle_database_path": "data/puzzles/4M_puzzle_library/",
      "puzzle_count": 4000000,
      "train_split": 0.8,
      "val_split": 0.1,
      "test_split": 0.1
    },
    "training": {
      "batch_size": 64,
      "learning_rate": 0.001,
      "epochs": 100,
      "early_stopping_patience": 10
    }
  }
}
```

#### 3.2 Launch Training
```bash
cd v4.0

# Single GPU training
python scripts/stage1_train_themes.py --config config/training_config.json

# Resume from checkpoint (if needed)
python scripts/stage1_train_themes.py --resume models/stage1_themes/checkpoint_epoch_50.pth
```

#### 3.3 Monitor Training
```bash
# TensorBoard (if enabled)
tensorboard --logdir=models/stage1_themes/logs

# Watch logs
tail -f logs/training.log
```

**Expected Training Time**:
- 4M puzzles, 64 batch size = ~62,500 batches/epoch
- RTX 4070 Ti: ~2-3 hours per epoch
- 100 epochs: ~200-300 hours (~8-12 days)
- **With early stopping**: Likely converges in 30-50 epochs (~3-6 days)

#### 3.4 Validation Metrics to Monitor
- **Theme Classification Accuracy**: Target >90%
- **Top-5 Move Accuracy**: Target >85% (best move in top 5 ranked)
- **Top-10 Move Accuracy**: Target >95%
- **Validation Loss**: Should decrease consistently
- **Inference Speed**: <5ms per position

---

### Step 4: Validation & Testing (Week 3)

#### 4.1 Load Trained Model
```python
from src.agents.v7p3r_themes_agent import V7P3RThemesAgent

agent = V7P3RThemesAgent(model_path='models/stage1_themes/final_model.pth')
```

#### 4.2 Run Validation Suite
```bash
# Full validation
python scripts/validate_agents.py --agent themes --test-puzzles 10000

# Expected output:
# {
#   "agent": "themes",
#   "theme_accuracy": 0.92,
#   "top5_accuracy": 0.87,
#   "top10_accuracy": 0.96,
#   "avg_inference_ms": 3.2,
#   "passed": true
# }
```

#### 4.3 Manual Testing
```python
import chess
from src.agents.v7p3r_themes_agent import V7P3RThemesAgent

agent = V7P3RThemesAgent(model_path='models/stage1_themes/final_model.pth')

# Test position: Scholar's Mate threat
board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 4")

# Categorize position
themes = agent.categorize_position(board)
print(f"Dominant theme: {themes.dominant_theme}")
print(f"Theme scores: {themes.themes}")

# Expected: High probability for "mate_in_1" or "king_attack"

# Rank moves
ranking = agent.rank_moves(board, time_budget=2.0)
print(f"Top move: {ranking.ranked_moves[0]}")  # Expected: Qxf7#
```

---

### Step 5: Engine Integration (Week 3-4)

#### 5.1 Create AI Move Sorter Module
Create `../../v7p3r-chess-engine/src/v7p3r_ai_move_ordering.py`:
```python
from v7p3r_themes_agent import V7P3RThemesAgent

class AIMoveSorter:
    def __init__(self, themes_agent: V7P3RThemesAgent):
        self.themes_agent = themes_agent
        
    def sort_moves(self, board, legal_moves, time_remaining):
        time_budget = self.calculate_time_budget(time_remaining)
        ranking = self.themes_agent.rank_moves(board, time_budget)
        return ranking.ranked_moves
```

#### 5.2 Modify V7P3R Engine
Edit `../../v7p3r-chess-engine/src/v7p3r.py`:
```python
class V7P3REngine:
    def __init__(self):
        # Add AI move sorter
        self.ai_move_sorter = AIMoveSorter(themes_agent)
        self.use_ai_ordering = True  # Feature flag
        
    def search(self, board, time_limit=3.0):
        legal_moves = list(board.legal_moves)
        
        # AI-powered move ordering
        if self.use_ai_ordering:
            ordered_moves = self.ai_move_sorter.sort_moves(
                board, legal_moves, self.time_manager.remaining_time()
            )
        else:
            ordered_moves = self.traditional_move_sort(legal_moves)
        
        # ... rest of search
```

#### 5.3 Testing Integration
```bash
cd ../../v7p3r-chess-engine

# Test AI move ordering in isolation
python testing/test_ai_move_ordering.py

# Test full engine with AI
python src/v7p3r_uci.py
# UCI command: position startpos
# UCI command: go movetime 1000
# Verify AI move ordering is used (check logs)
```

---

### Step 6: Performance Validation (Week 4)

#### 6.1 Move Ordering Efficiency Test
Measure alpha-beta cutoffs with/without AI ordering:
```python
# Test script: testing/test_move_ordering_efficiency.py
from v7p3r import V7P3REngine
import chess

positions = load_test_positions(1000)  # Diverse test positions

# Baseline (traditional ordering)
engine_baseline = V7P3REngine(use_ai_ordering=False)
nodes_baseline = 0
for pos in positions:
    result = engine_baseline.search(pos, depth=5)
    nodes_baseline += engine_baseline.nodes_searched

# AI-enhanced ordering
engine_ai = V7P3REngine(use_ai_ordering=True)
nodes_ai = 0
for pos in positions:
    result = engine_ai.search(pos, depth=5)
    nodes_ai += engine_ai.nodes_searched

improvement = (nodes_baseline - nodes_ai) / nodes_baseline * 100
print(f"Node reduction: {improvement:.1f}%")  # Target: >30%
```

**Success Criteria**:
- ✅ Node reduction: >30%
- ✅ Effective depth increase: +0.5 to +1.0 plies
- ✅ Total move time: Still <10ms average

#### 6.2 ELO Measurement
```bash
# Use engine-tester for tournament
cd ../../../engine-tester

# Run 500-game tournament
python run_tournament.py \
  --engine1 "../../v7p3r-chess-engine/src/v7p3r_uci.py" \
  --engine1-name "V7P3R-v18.4-AI" \
  --engine2 "../../Tournament Engines/V7P3R/V7P3R_v18.4.bat" \
  --engine2-name "V7P3R-v18.4-baseline" \
  --games 500 \
  --time-control "5+4"

# Expected result: V7P3R-v18.4-AI scores 60-65% (ELO +70 to +120)
```

**Success Criteria**:
- ✅ Win rate: >55%
- ✅ ELO improvement: +100-150
- ✅ No crashes or timeouts
- ✅ Blunders/game: Same or lower

---

## Success Metrics Summary

### Training Metrics
- [  ] Theme classification accuracy: >90%
- [  ] Top-5 move inclusion: >85%
- [  ] Top-10 move inclusion: >95%
- [  ] Inference speed: <5ms per position
- [  ] 4M puzzle coverage: 100%

### Integration Metrics
- [  ] Alpha-beta cutoffs: +30% improvement
- [  ] Effective search depth: +0.5-1.0 plies
- [  ] Move time maintained: <10ms average
- [  ] Zero crashes in 1000-game test

### Performance Metrics
- [  ] ELO improvement: +100-150
- [  ] Win rate vs baseline: 60-65%
- [  ] Puzzle solving improvement: +10-20%

---

## Troubleshooting

### Issue: GPU Out of Memory
**Solution**: Reduce batch size in `config/training_config.json`
```json
"batch_size": 32  // or 16
```

### Issue: Training too slow
**Solution**: 
1. Enable mixed precision training
2. Reduce puzzle count for initial tests
3. Use data parallelism if multiple GPUs available

### Issue: Poor theme accuracy (<80%)
**Solution**:
1. Check puzzle database quality (themes correctly labeled?)
2. Increase model capacity (more layers/neurons)
3. Train longer (more epochs)
4. Adjust learning rate

### Issue: Integration causes engine slowdown
**Solution**:
1. Check inference speed (<5ms required)
2. Enable model caching for repeated positions
3. Use lower candidate counts for fast time controls

---

## Next Steps

Once Stage 1 is complete and validated:
1. Update `config/agent_config.json`: Set `themes_agent.enabled = true`
2. Deploy to GCP staging environment for live testing
3. Begin Stage 2: Historical Game Analysis & Correction

---

**Document Status**: Ready for Implementation  
**Last Updated**: April 18, 2026  
**Next Action**: Prepare puzzle database and start data pipeline development
