# V7.1 Quick Start Guide

## TL;DR - Just Run This

```bash
cd "E:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v7.0\src"
python train_generational.py
```

That's it! The script will:
1. Initialize generation 0 (random weights)
2. Train 10 generations via self-play
3. Evaluate each generation vs previous best (6 games)
4. Accept only generations that beat the old model
5. Save best model and training history

## What Changed from v7.0

### Major Fixes:
1. **Generational training** - New model vs old model (not self vs self)
2. **Better endgame** - 100% Stockfish in endgame (was 50%)
3. **Controlled chaos** - 20% SF in middlegame (was 10%)
4. **Color balance** - 3 games as White, 3 as Black

### Expected Results:
- **Max-move games:** Should drop from 78% to ~30%
- **Win metrics:** Actually meaningful (new beats old)
- **Color bias:** Should be ~50/50, not 0-100

## Configuration

Edit `train_generational.py` to change:

```python
SELFPLAY_GAMES = 100        # Games per generation (try 200-500 later)
EVALUATION_GAMES = 6        # Match format (3 White, 3 Black)
MAX_GENERATIONS = 10        # Total generations to train
```

## Output Files

```
training/v7_generational/
├── gen_0000_initial.pt          # Generation 0 baseline
├── gen_0001_trained.pt          # Generation 1
├── gen_0002_trained.pt          # Generation 2
├── ...
├── best_model.pt                # Current best (load this for play)
└── generation_history.json      # Win/loss records per generation
```

## Monitoring Progress

### During Training:
Watch for:
- **Self-play games:** Forest darkness (~0.3-0.4 is good)
- **Training loss:** Should decrease over time
- **Evaluation wins:** New model should win >50% to be accepted

### After Training:
```bash
cd training/v7_generational
cat generation_history.json
```

Look for:
```json
{
  "current_generation": 10,
  "generations": [
    {
      "generation_number": 1,
      "wins_as_white": 2,
      "wins_as_black": 2,
      "total_wins": 4,
      "win_rate": 0.67,
      "accepted": true  ← Good! New model was better
    },
    {
      "generation_number": 2,
      "wins_as_white": 1,
      "wins_as_black": 1,
      "total_wins": 2,
      "win_rate": 0.33,
      "accepted": false  ← Rejected, kept old model
    }
  ]
}
```

## Success Criteria

### Healthy Training:
- ✅ Acceptance rate: 60-80% (steady improvement)
- ✅ Win rate trends upward across generations
- ✅ Color-balanced wins (both White and Black)

### Unhealthy Training:
- ❌ All generations rejected (too hard, increase self-play games)
- ❌ All generations accepted (too easy, increase evaluation games)
- ❌ Random win rates (unstable, check network convergence)

## Troubleshooting

### "Max-move games still at 78%"
**Fix:** Increase endgame training focus
```python
# In generational_trainer.py
DynamicWeightCalculator(
    endgame_sf_weight=1.0,     # Already at max
    tablebase_sf_weight=1.0    # Already at max
)
```
**Next step:** Add explicit endgame penalties for wandering

### "All games are draws"
**Fix:** Lower temperature in evaluation
```python
# In generational_trainer.py, line ~220
temperature=0.1  # Already low, try 0.05
```

### "Color bias still present"
**Check:** Are games alternating properly?
```python
# In generational_trainer.py, line ~213
new_plays_white = (game_num % 2 == 1)  # Should alternate
```

### "Training loss not decreasing"
**Fix:** Increase batch size or learning rate
```python
# In network.py
learning_rate=0.001  # Try 0.002
```

## Next Steps After v7.1

### If Max-Move Rate Improves (< 40%):
1. Increase generations to 20-50
2. Increase self-play games to 200-500
3. Tournament test best_model.pt
4. Deploy to production

### If Max-Move Rate Still High (> 50%):
1. Add explicit endgame conversion rewards
2. Train on endgame puzzles (Lucena, Philidor)
3. Penalize repetitions when ahead
4. Use 6-piece tablebases

### Production Deployment:
```bash
# Copy best model to v7p3r-chess-engine
cp training/v7_generational/best_model.pt ../../v7p3r-chess-engine/models/v7_gen_10.pt

# Update engine to load it
# (Integration code TBD)
```

## Weight Curve Reference

```
Opening (1-10):        90% SF → Learn fundamentals
Early MG (11-20):      90% → 10% SF → Enter chaos
Deep MG (21-40):       20% SF → CONTROLLED CHAOS
Late MG (41-60):       20% → 80% SF → Return to precision
Endgame (61+):         100% SF → Perfect technique
Tablebase (≤7 pieces): 100% Perfect
```

## Key Differences from v7.0

| Metric | v7.0 (Self-Play) | v7.1 (Generational) |
|--------|------------------|---------------------|
| Opponent | Same model | Previous best |
| Win metric | % decisive | % new beats old |
| Color balance | Random | 50/50 forced |
| Endgame SF | 50% | 100% |
| Middlegame SF | 10% | 20% |
| Max-move games | 78% | TBD (~30% expected) |
| White wins | 0% | TBD (~50% expected) |

## Time Estimates

- **Generation 0:** ~5 minutes (initialization)
- **Each generation:** ~30-60 minutes (100 games + training)
- **Full 10 generations:** ~5-10 hours total

Adjust `SELFPLAY_GAMES` lower for faster iteration.
