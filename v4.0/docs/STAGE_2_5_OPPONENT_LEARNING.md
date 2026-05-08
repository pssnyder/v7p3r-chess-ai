# Stage 2.5: Opponent Pattern Learning

## 📋 Overview

After Stage 2's corrective training showed regression (75.2% top-5 vs Stage 1's 86.6%), we're pivoting to a better approach:

**Learn from ALL moves in historical games, not just mistakes.**

## 🎯 Strategy

### What Changed
- ❌ **Old**: Learn to avoid V7P3R's blunders (caused catastrophic forgetting)
- ✅ **New**: Learn best moves from ALL positions (V7P3R + opponents)

### Why This Works
1. **Preserve Stage 1 Knowledge**: Continue training from 86.6% top-5 checkpoint
2. **Learn Opponent Patterns**: If they beat V7P3R, learn their winning moves
3. **Game Phase Awareness**: Context-appropriate moves (opening/middlegame/endgame)
4. **Curriculum Learning**: Mix puzzles + games for balanced training

## 📊 Data Sources

### Puzzle Dataset (Stage 1)
- **100,000 positions** from Lichess puzzle database
- Stockfish-enriched with top-10 moves
- **86.6% top-5 accuracy** achieved

### Game Position Dataset (Stage 2.5)
- **All V7P3R historical games** (5,107+ games, ~350,000+ positions)
- Both V7P3R and opponent moves included
- Game phase classification:
  - **Opening**: Moves 1-20 (first ~25%)
  - **Middlegame**: Moves 21-50 (middle ~50%)
  - **Endgame**: Moves 51+ (final ~25%)

## 🔄 Workflow

### Step 1: Extract Game Positions
```bash
cd v4.0
python scripts/extract_game_positions.py \
    --pgn-dir "E:/Programming Stuff/Chess Engines/Chess Engine Playground/engine-metrics/raw_data/game_records/Lichess V7P3R Bot" \
    --output data/stage2_games/historical_positions.json \
    --analysis-time 0.5 \
    --max-games 1000
```

**Output**: JSON file with all positions, Stockfish analysis, game phase classification

### Step 2: Continue Training from Stage 1
```bash
python scripts/train_combined_dataset.py \
    --stage1-checkpoint models/stage1_themes/best_checkpoint.pt \
    --puzzle-data data/preprocessed_puzzles/enriched_puzzles_compact_20260420_003909.json \
    --game-data data/stage2_games/historical_positions.json \
    --batch-size 32 \
    --num-epochs 50 \
    --learning-rate 5e-5 \
    --patience 15
```

**Output**: Combined model trained on puzzles + games

## 🎯 Expected Results

### Success Criteria
- ✅ Maintain **≥85% top-5 accuracy** on puzzles (no regression)
- ✅ Learn **game-specific patterns** from historical data
- ✅ **Game phase awareness** (different strategies for opening/middlegame/endgame)
- ✅ **Outperform V7P3R v18.4** in head-to-head tournament

### If Successful
Deploy as **V7P3R primary engine**, replacing minimax search with AI-based move selection.

## 📈 Advantages Over Stage 2 Corrective

| Aspect | Stage 2 (Corrective) | Stage 2.5 (Opponent Learning) |
|--------|----------------------|-------------------------------|
| **Data Source** | V7P3R's mistakes only | ALL moves (V7P3R + opponents) |
| **Learning Focus** | Avoid bad moves | Find best moves |
| **Stage 1 Preservation** | ❌ Catastrophic forgetting | ✅ Curriculum learning |
| **Result** | 75.2% top-5 (regression) | Target: ≥86% top-5 |
| **Blunder Avoidance** | 57.3% | Natural byproduct of best moves |

## 🔬 Technical Details

### Dataset Format
Each position contains:
- **FEN**: Board position
- **Position Features**: 690-dim vector (from ChessStateExtractor)
- **Top Moves**: Stockfish's top-10 moves with evaluations
- **Move Weights**: Exponential decay (1.0, 0.8, 0.6, 0.4, 0.2, ...)
- **Game Phase**: Opening/Middlegame/Endgame
- **Player/Opponent**: Track who played this move
- **Result**: Game outcome (for future analysis)

### Training Configuration
- **Starting Point**: Stage 1 checkpoint (epoch 95, 86.6% top-5)
- **Learning Rate**: 5e-5 (lower than Stage 1 for fine-tuning)
- **Batch Size**: 32
- **Early Stopping**: Patience 15 epochs
- **Device**: CPU (can use CUDA if available)

### Model Architecture
- **Same as Stage 1**: MoveOrderingNetwork
- **Parameters**: 1.6M
- **Output**: Move scores + theme probabilities

## 📝 Next Steps

1. **Extract positions** from all V7P3R PGN files
2. **Train combined model** (puzzles + games)
3. **Validate performance**:
   - Puzzle accuracy (maintain ≥85% top-5)
   - Game pattern recognition
   - Head-to-head vs V7P3R v18.4
4. **If successful**: Deploy as V7P3R primary engine
5. **If needs improvement**: Add opening book + tablebase (Stage 3)

## 🚀 Future Enhancements

### Stage 3 (If Stage 2.5 Successful)
- **Opening Book Integration**: Bias towards proven opening lines
- **Tablebase Integration**: Perfect endgame play
- **Target**: 75%+ opening win rate, 85%+ endgame conversion

### Stage 4 (Final Stage)
- **Reinforcement Learning**: Self-play for middlegame tactics
- **Target**: Final ELO 1800-2000 actual

## 📊 Performance Tracking

### Baseline (V7P3R v18.4)
- **ELO**: ~1600 Lichess (~1200-1400 actual)
- **Win Rate**: 66.8% overall
- **Blunders/Game**: ~6.0

### Target (V7P3RAI v4.0)
- **Puzzle Accuracy**: ≥85% top-5 (maintain Stage 1)
- **Game Accuracy**: ≥80% top-5 on historical positions
- **Win Rate vs V7P3R**: ≥55%
- **Blunders/Game**: <5.0
- **Deployment Ready**: If win rate ≥55% in 50+ game tournament
