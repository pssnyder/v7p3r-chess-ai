# Stage 2.5 Quick Start Guide

## 🎯 Goal

Train V7P3RAI to find the **best moves** in real V7P3R game positions, using Stockfish as the teacher.

## 📝 The Simple Approach

**What we're doing:**
1. Take every position from V7P3R's historical games (5,107+ games)
2. Ask Stockfish "what's the best move here?" (top-5 moves)
3. Train the AI on those positions with Stockfish's answers
4. Result: AI learns what it **should** have done in real games

**Why this works:**
- ✅ Real positions V7P3R will face on Lichess
- ✅ Stockfish-corrected moves (always best, regardless of what was played)
- ✅ No confusion about good vs bad positions
- ✅ Continues from Stage 1's 86.6% success (curriculum learning)
- ✅ No catastrophic forgetting (same task: find best move)

## 🚀 3-Step Workflow

### Step 1: Quick Test (10 games, ~700 positions)
```bash
cd v4.0
run_stage2_5_test.bat
```

**Expected:**
- Time: ~5 minutes
- Output: `data/stage2_games/test_positions_10games.json`
- Size: ~2-3 MB
- Validates pipeline works

### Step 2: Extract 1000 Games (~70,000 positions)
```bash
run_stage2_5_extract_1000games.bat
```

**Expected:**
- Time: ~10 hours
- Output: `data/stage2_games/historical_positions_1000games.json`
- Size: ~200-300 MB
- Reasonable dataset for training

### Step 3: Train Combined Model (Puzzles + Games)
```bash
run_stage2_5_train.bat
```

**Expected:**
- Time: ~2-3 hours
- Input: 100K puzzles + 70K game positions = 170K total
- Target: Maintain ≥85% top-5 accuracy
- Output: `models/stage2_combined/best_checkpoint.pt`

## 📊 Dataset Details

### Source Data
- **File**: `E:\...\Lichess V7P3R Bot\lichess_v7p3r_bot_2026-04-09.pgn`
- **Games**: 5,107 (all V7P3R games to date)
- **Positions**: ~350,000 (all moves from all games)
- **Game Phases**: Classified as opening/middlegame/endgame

### Position Format
Each position contains:
```json
{
  "fen": "...",
  "game_phase": "middlegame",
  "top_moves": [
    {"uci": "e2e4", "san": "e4", "score": 50, "weight": 1.0},
    {"uci": "d2d4", "san": "d4", "score": 40, "weight": 0.8},
    {"uci": "g1f3", "san": "Nf3", "score": 35, "weight": 0.6},
    {"uci": "c2c4", "san": "c4", "score": 30, "weight": 0.4},
    {"uci": "e2e3", "san": "e3", "score": 25, "weight": 0.2}
  ],
  "position_features": [690 floats],
  "player": "v7p3r_bot",
  "opponent": "opponent_name",
  "result": "1-0"
}
```

### Game Phase Classification
- **Opening**: Moves 1-20 (first ~25%)
- **Middlegame**: Moves 21-50 (middle ~50%)
- **Endgame**: Moves 51+ (final ~25%)

## 🎯 Success Criteria

### Stage 2.5 (Combined Training)
- ✅ Maintain **≥85% top-5 accuracy** on puzzles (no regression from Stage 1)
- ✅ Learn **game-specific patterns** from historical data
- ✅ **Game phase awareness** (different strategies per phase)

### Production Deployment
- ✅ **Win rate ≥55%** vs V7P3R v18.4 (50+ games)
- ✅ **Blunders/game <5.0** (currently 6.0)
- ✅ **Time forfeit rate <10%**

## 🔧 Advanced Options

### Extract Subset of Games
```bash
python scripts/extract_game_positions.py \
    --pgn-file "path/to/file.pgn" \
    --output data/stage2_games/positions.json \
    --analysis-time 0.5 \
    --num-top-moves 5 \
    --max-games 100
```

### Faster Extraction (Less Accurate)
```bash
python scripts/extract_game_positions.py \
    --analysis-time 0.2 \
    --num-top-moves 3
```

### More Accurate Extraction (Slower)
```bash
python scripts/extract_game_positions.py \
    --analysis-time 1.0 \
    --num-top-moves 10
```

## 📈 What's Different from Stage 2 Corrective?

| Aspect | Stage 2 (Corrective) | Stage 2.5 (Opponent Learning) |
|--------|----------------------|-------------------------------|
| **Focus** | Avoid V7P3R's mistakes | Learn best moves from all positions |
| **Data** | Only blunder positions | ALL positions (both players) |
| **Labels** | Mixed (good/bad) | Always Stockfish best |
| **Consistency** | Confusing (avoid vs select) | Clear (always select best) |
| **Result** | 75.2% top-5 (regression) | Target: ≥85% top-5 (maintain) |
| **Forgetting** | Catastrophic | None (curriculum learning) |

## 🚨 Common Issues

### "FileNotFoundError: PGN file not found"
Check the path in `run_stage2_5_*.bat` files or pass `--pgn-file` argument.

### "Stockfish engine not found"
Update `--stockfish-path` in script or batch file.

### Training shows regression
- Check if Stage 1 checkpoint exists: `models/stage1_themes/best_checkpoint.pt`
- Verify learning rate is low (5e-5) for continued training
- Ensure curriculum learning (mix puzzles + games)

### GPU not used
- Check PyTorch installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Add `--device cuda` to training command

## 📝 Next Steps After Training

1. **Validate Performance**
   - Test on puzzle dataset (should maintain ≥85% top-5)
   - Test on game positions (should show improvement)

2. **Create UCI Wrapper**
   - Load model in UCI engine
   - Generate move candidates
   - Return best move to lichess-bot

3. **Tournament Testing**
   - Run 50-100 games vs V7P3R v18.4
   - Measure win rate, blunders/game, time usage

4. **Deployment Decision**
   - If win rate ≥55%: Deploy as primary V7P3R engine
   - If win rate 50-55%: Hybrid (AI + minimax validation)
   - If win rate <50%: Use as agent layer only

## 🎓 Key Insight

**Stage 2.5's approach is fundamentally different:**

- **Stage 2**: "Don't do this bad move, do this good move instead"
  - Problem: Confusing (two opposite tasks)
  - Result: Model forgot how to do original task
  
- **Stage 2.5**: "Here's a position, what's the best move?"
  - Same task as Stage 1 (puzzles)
  - Just more positions from real games
  - Model already knows how to do this!

This is **curriculum learning**: Start with simple (puzzles), add complexity (games), but keep the task the same (find best move).
