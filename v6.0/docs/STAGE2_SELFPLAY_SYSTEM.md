# Stage 2 Self-Play System - Effort Prediction Learning
**V7P3R AI v6.1 - Teaching Computational Resource Allocation**  
**Created**: 2026-05-31  
**Status**: 🚀 **READY FOR EXECUTION**

---

## Core Concept: Teaching Effort Prediction

**What is "effort"?**  
Effort = computational resources required to make a good move decision in a given position.

**Measurable Proxies**:
- **Processing ticks**: Nodes searched, legal moves evaluated
- **Time spent**: Actual seconds consumed
- **Depth reached**: Search depth achieved within time budget
- **Complexity encountered**: Branching factor, tactical density

**Goal**: Train Stage 2 model to predict required effort BEFORE spending it.

---

## Diverse Training Scenario Generation

### Time Pressure Scenarios (8 scenarios)

Each scenario creates different effort allocation challenges:

#### Bullet (1+2s) - 57 games (20%)
```
Early:    60s remaining → Learn "I have time to think"
Midgame:  30s remaining → Learn "Must be efficient"
Endgame:   8s remaining → Learn "EMERGENCY: instant moves only"
```

#### Blitz (5+4s) - 170 games (60% - Primary Training)
```
Early:    300s remaining → Learn "Deep calculation available"
Midgame:  120s remaining → Learn "Balanced resource use"
Endgame:   25s remaining → Learn "Time pressure tactics"
```

#### Rapid (15+10s) - 57 games (20%)
```
Early:    900s remaining → Learn "Maximum depth exploration"
Midgame:  600s remaining → Learn "Thorough position analysis"
```

### Position Complexity Scenarios

**Simple Positions** (low effort):
- Few pieces remaining (endgame)
- Forced moves (only 1-2 legal options)
- Quiet positions (no tactics)
- **Expected learning**: "Spend minimal time here"

**Complex Positions** (high effort):
- Many legal moves (30+ options)
- Tactical density (pins, forks, threats)
- High branching factor (>35 average)
- Material imbalances with compensation
- **Expected learning**: "Need deep thought here"

**Forest Darkness** positions (Tal-style):
- Sacrifice opportunities
- Multiple tactical motifs
- Unclear evaluation
- **Expected learning**: "Maximum effort required, but rewarding"

---

## Data Collection: What Gets Recorded

### Per-Position Data Structure
```json
{
  "game_id": "selfplay_20260531_143022_5847",
  "position_id": "selfplay_20260531_143022_5847_move_15",
  "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 5",
  "move_number": 15,
  "side_to_move": "white",
  
  "stage1_features": [2,2,2,2,1,1,2,2,2,2,1,1,0,1,1,1,1,0,28,32],
  
  "complexity_metrics": {
    "legal_moves_count": 34,
    "capture_moves_count": 3,
    "check_moves_count": 1,
    "pieces_under_attack": 2,
    "pieces_undefended": 0,
    "branching_factor_1ply": 28.5,
    "forest_darkness_score": 0.45,
    "tactical_density": 3
  },
  
  "time_state": {
    "time_white": 118.3,
    "time_black": 125.7,
    "increment": 4.0,
    "time_budget": 8.2,
    "time_remaining": 118.3
  },
  
  "move_played": "f3g5",
  "time_spent": 6.8,
  "nodes_searched": 34,
  "eval_cp": 25,
  
  "labels": {
    "complexity_score": 4.5,        // Ground truth: how complex WAS this?
    "time_allocation": 0.83,         // Ground truth: what fraction DID we use?
    "processing_tick_count": 34,     // Ground truth: how much effort DID we need?
    "effort_metric": 5.0             // Nodes per second (efficiency)
  }
}
```

### Key Insight: Ground Truth from Actual Play

**Before the move**: We DON'T know effort required (that's what we're predicting)  
**After the move**: We KNOW effort required (actual ticks spent, time used)

**Training data labels** = actual effort that was necessary to make the move.

**Stage 2 model learns**: "Positions with features X,Y,Z required effort E" → predict E for new positions.

---

## Effort Prediction Learning Objectives

### 1. Complexity Score Prediction (0-10 scale)

**Input**: Position features (legal moves, tactics, branching factor)  
**Output**: Predicted complexity score  
**Ground Truth**: Calculated from actual branching factor + tactical density

**Learning Examples**:
- "8 legal moves, no tactics, quiet position" → complexity = 1.5
- "34 legal moves, 3 captures, check available, 2 pieces hanging" → complexity = 4.5
- "42 legal moves, sacrifice opportunity, unclear eval, deep tactics" → complexity = 8.2

### 2. Time Allocation Prediction (0-1 fraction)

**Input**: Complexity + time remaining + time pressure  
**Output**: Predicted time fraction to use  
**Ground Truth**: Actual time_spent / time_budget

**Learning Examples**:
- "Simple position, 300s remaining" → time_allocation = 0.1 (use 10% of budget)
- "Complex position, 120s remaining" → time_allocation = 0.6 (use 60% of budget)
- "Any position, 8s remaining (emergency)" → time_allocation = 0.05 (instant move)

### 3. Processing Tick Count Estimation

**Input**: Complexity + position type  
**Output**: Predicted nodes needed  
**Ground Truth**: Actual nodes searched

**Learning Examples**:
- "Forced move (1 legal option)" → ticks = 1 (trivial)
- "Standard middlegame (30 options)" → ticks = 30-50 (evaluate each)
- "Tactical position (deep search needed)" → ticks = 200+ (calculate variations)

---

## Diverse Scenario Distribution

### Time Control Distribution
```
Blitz (5+4):   170 games (60%)
  ├─ Early:     68 games (40%)  ← Learn "abundant time" strategies
  ├─ Midgame:   60 games (35%)  ← Learn "balanced" strategies
  └─ Endgame:   42 games (25%)  ← Learn "time pressure" strategies

Bullet (1+2):   57 games (20%)
  ├─ Early:     17 games (30%)  ← Learn "move fast" from start
  ├─ Midgame:   20 games (35%)  ← Learn "increment farming"
  └─ Endgame:   20 games (35%)  ← Learn "emergency heuristics"

Rapid (15+10):  57 games (20%)
  ├─ Early:     29 games (50%)  ← Learn "deep calculation"
  └─ Midgame:   28 games (50%)  ← Learn "thorough analysis"
```

**Total**: 284 games (median historical manual tuning benchmark)

### Expected Position Type Distribution (estimated)

```
Simple positions (complexity 0-3):     ~40% of positions
  ├─ Opening book (early moves)
  ├─ Quiet middlegame
  └─ Simple endgames

Moderate positions (complexity 3-6):   ~45% of positions
  ├─ Standard middlegame
  ├─ Early tactics
  └─ Material imbalances

Complex positions (complexity 6-10):   ~15% of positions
  ├─ Tactical complications
  ├─ Sacrifice opportunities
  └─ Unclear evaluations ("forest darkness")
```

---

## How Model Learns Effort Allocation

### Training Loss Function

```python
def stage2_loss(predictions, targets):
    # Complexity prediction (how complex is this?)
    complexity_loss = MSE(pred['complexity_score'], target['complexity_score'])
    
    # Time allocation (how much time should I use?)
    time_loss = MSE(pred['time_allocation'], target['time_allocation'])
    
    # Confidence (how sure am I?)
    confidence_loss = MSE(pred['confidence'], target['confidence'])
    
    # Weighted combination
    total_loss = 0.4 * complexity_loss + 0.4 * time_loss + 0.2 * confidence_loss
    return total_loss
```

### Learning Patterns

**Pattern 1: Time Pressure → Lower Allocation**
```
Input:  complexity=5.0, time_remaining=10s  → Prediction: time_allocation=0.1
Input:  complexity=5.0, time_remaining=300s → Prediction: time_allocation=0.5

Learning: "Same complexity, less time → must allocate less"
```

**Pattern 2: Complexity → Higher Allocation**
```
Input:  complexity=2.0, time_remaining=120s → Prediction: time_allocation=0.2
Input:  complexity=8.0, time_remaining=120s → Prediction: time_allocation=0.7

Learning: "More complexity → need more time, even with same budget"
```

**Pattern 3: Emergency Override**
```
Input:  complexity=ANY, time_remaining<10s → Prediction: time_allocation=0.05

Learning: "Severe time pressure → instant moves regardless of complexity"
```

**Pattern 4: Effort Efficiency**
```
Simple positions:  High effort/benefit ratio → allocate minimal time
Complex positions: Low effort/benefit ratio initially → need deep search

Learning: "Don't waste time on simple positions, invest in complex ones"
```

---

## Expected Training Outcomes

### Quantitative Targets

**Complexity Prediction**:
- MSE ≤ 1.0 (on 0-10 scale)
- MAE ≤ 0.7
- **Interpretation**: Predictions within ±0.7 complexity points on average

**Time Allocation Prediction**:
- MSE ≤ 0.05 (on 0-1 scale)
- MAE ≤ 0.15
- **Interpretation**: Predictions within ±15% of actual time used

**Processing Tick Estimation**:
- Log-scale error (ticks vary widely: 1 to 1000+)
- Within 2x of actual (predicted 50 vs actual 25-100 acceptable)

### Qualitative Behaviors

**Learned Heuristics**:
1. "Opening moves are simple → instant moves"
2. "Tactical positions need deep thought → allocate more time"
3. "Time pressure overrides complexity → instant moves"
4. "Endgames with few pieces are simple → quick calculation"
5. "Sacrifice opportunities are complex → deep analysis needed"

**Emergency Behaviors** (should emerge automatically):
1. "Time <10s → skip complex analysis, use heuristics"
2. "Time <30s → reduce search depth, rely on Stage 1 more"
3. "Time abundant → explore deeper, consider multiple lines"

---

## Next Steps: Execution Plan

### Phase 1: Generate Training Data (1-2 days)
```bash
# Run batch self-play (284 games)
python scripts/stage2/run_batch_selfplay.py \
  --model models/position_evaluator_best.pth \
  --output data/stage2/selfplay_batch_284 \
  --games 284

# Expected output:
# - 284 game PGN files
# - 284 position JSONL files
# - ~5,000-10,000 total positions
# - Batch report with statistics
```

### Phase 2: Verify Data Quality (1 hour)
```bash
# Check feature compatibility
python scripts/stage2/verify_compatibility.py \
  --model models/position_evaluator_best.pth \
  --data data/stage2/selfplay_batch_284 \
  --save-map

# Expected output:
# - ✓ All compatibility checks pass
# - Feature mapping JSON saved
# - Ready for Stage 2 training
```

### Phase 3: Train Stage 2 Model (2-3 days)
```bash
# Train complexity/time manager
python scripts/stage2/train_stage2.py \
  --data data/stage2/selfplay_batch_284 \
  --epochs 30 \
  --batch-size 256 \
  --lr 0.0005

# Expected output:
# - Trained ComplexityTimeManager model
# - Trained MovePriorityRanker model
# - Validation metrics meeting targets
```

### Phase 4: Integration Testing (1-2 days)
```bash
# Test integrated engine (static + Stage 1 + Stage 2)
python scripts/engine/test_integrated_engine.py \
  --stage1-model models/position_evaluator_best.pth \
  --stage2-model models/complexity_time_manager_best.pth \
  --test-positions testing/tactical_positions.fen

# Expected output:
# - Inference time <100ms per position
# - Time allocations sensible
# - Emergency behaviors working
```

---

## Success Criteria

**The model has successfully learned effort prediction when**:

✓ **Simple positions** → predicted complexity <3.0, time_allocation <0.2  
✓ **Complex positions** → predicted complexity >6.0, time_allocation >0.5  
✓ **Time pressure** → time_allocation drops to <0.1 regardless of complexity  
✓ **Abundant time** → time_allocation increases for same complexity  
✓ **Processing ticks** → predicted within 2x of actual need  

**Bonus: Emergent Behaviors**:
- Model learns opening positions are simple (without being told)
- Model learns endgames vary (simple K+Q vs complex R+B)
- Model learns time pressure overrides complexity (emergency heuristic)
- Model learns increment farming in bullet (use less time per move)

---

## Can AI Learn Faster Than Human?

**Human Benchmark**: 284 games median (from historical V7P3R manual tuning)

**AI Training**: 284 games self-play

**Comparison Metric**: Performance improvement per game
- Human: Learned through trial/error, manual analysis
- AI: Learns from ground truth labels, supervised learning

**Expected Outcome**: 
- AI achieves similar performance improvements in ≤284 games
- Demonstrates data efficiency matching human learning
- If AI needs >284 games → human learning more efficient
- If AI needs <284 games → AI learning more efficient

**This is the quantitative answer to: "Can this AI learn faster than I did?"**

---

## Status

✅ **Self-play infrastructure created**  
✅ **Feature compatibility verified**  
✅ **Batch runner implemented**  
✅ **Effort prediction framework defined**  
🚀 **Ready to generate training data**  

**Next Command**:
```bash
python scripts/stage2/run_batch_selfplay.py
```
