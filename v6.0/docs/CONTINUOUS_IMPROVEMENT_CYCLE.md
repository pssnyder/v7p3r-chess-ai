# Continuous Improvement Cycle: Stage 1 ↔ Stage 2 Feedback Loop

**Created**: June 1, 2026  
**Author**: V7P3R Chess AI Team  
**Status**: Architecture Defined, Implementation In Progress

---

## Executive Summary

This document defines the **continuous self-improvement cycle** between Stage 1 (Position Evaluator) and Stage 2 (Complexity & Time Manager). The key insight is that Stage 2 self-play generates high-quality training data that can be fed back into Stage 1, creating a virtuous loop of incremental learning without requiring full retraining.

### Core Principle
> "Each cycle of self-play doesn't just train Stage 2 — it strengthens Stage 1, which in turn produces better Stage 2 data, creating compounding improvement over time."

---

## The Feedback Loop

```
┌─────────────────────────────────────────────────────────────┐
│                 CONTINUOUS IMPROVEMENT CYCLE                 │
└─────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────┐
    │  Stage 1 Position Evaluator (v1.0)           │
    │  1.648M positions, F1=87.76%                 │
    └────────────────┬─────────────────────────────┘
                     │ Move selection
                     ↓
    ┌──────────────────────────────────────────────┐
    │  Stage 2 Self-Play (284 games)               │
    │  Generates 7,557 positions with labels       │
    └────────────────┬─────────────────────────────┘
                     │ Extract GOOD/BAD labels
                     ↓
    ┌──────────────────────────────────────────────┐
    │  Data Augmentation                           │
    │  Balance, filter, metadata tracking          │
    └────────────────┬─────────────────────────────┘
                     │ Merge datasets
                     ↓
    ┌──────────────────────────────────────────────┐
    │  Incremental Stage 1 Training                │
    │  1.648M → 1.656M positions                   │
    │  Train 5-10 epochs (not full 20)             │
    └────────────────┬─────────────────────────────┘
                     │ Improved model
                     ↓
    ┌──────────────────────────────────────────────┐
    │  Stage 1 Position Evaluator (v1.1)           │
    │  Enhanced with self-play knowledge           │
    └────────────────┬─────────────────────────────┘
                     │ Better move selection
                     ↓
    ┌──────────────────────────────────────────────┐
    │  Stage 2 Training                            │
    │  ComplexityTimeManager + MovePriorityRanker  │
    └────────────────┬─────────────────────────────┘
                     │ Deploy trained model
                     ↓
    ┌──────────────────────────────────────────────┐
    │  Stage 2 Verification Games                  │
    │  Test in diverse scenarios                   │
    └────────────────┬─────────────────────────────┘
                     │ New positions
                     ↓
                  [LOOP BACK TO TOP]
```

---

## Data Flow Architecture

### Phase 1: Self-Play Generation (Current)
**Input**: Stage 1 model (F1=87.76%)  
**Process**: Monte Carlo self-play with 8 time scenarios  
**Output**: 7,557 positions with rich metadata

**Position Structure**:
```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "eval_cp": 99.99,           # Stage 1 evaluation
  "scenario": "blitz_midgame", # Time control context
  "move_number": 1,
  "stage1_features": [19-dim], # Already extracted
  "complexity_metrics": {...}, # Stage 2 labels
  "time_state": {...},         # Scenario metadata
  "move_played": "e2e4",
  "game_result": "1-0"
}
```

### Phase 2: Label Extraction
**Labeling Strategy**:
- `GOOD`: eval_cp > +50 (clear advantage for side-to-move)
- `BAD`: eval_cp < -50 (clear disadvantage for side-to-move)
- `NEUTRAL`: -50 ≤ eval_cp ≤ +50 (skip or balance separately)

**Expected Distribution** (from 7,557 positions):
- ~30-40% GOOD (winning positions)
- ~30-40% BAD (losing positions)
- ~20-40% NEUTRAL (balanced positions)

**Balancing**: Undersample majority class to maintain 50/50 GOOD/BAD ratio

### Phase 3: Dataset Augmentation
**Original Dataset**: 1.648M positions (824k GOOD, 824k BAD)  
**New Data**: ~3,000-4,000 balanced positions from self-play  
**Augmented Dataset**: ~1.652M positions

**Metadata Tracking** (critical for future analysis):
```json
{
  "dataset_version": "1.1",
  "total_positions": 1652000,
  "good_count": 826000,
  "bad_count": 826000,
  "source_distribution": {
    "original_training_data": 1648000,
    "selfplay_batch_284": 4000
  },
  "color_balance": {
    "white_positions": 826000,
    "black_positions": 826000
  },
  "scenario_distribution": {
    "blitz": 2400,
    "bullet": 800,
    "rapid": 800
  }
}
```

### Phase 4: Incremental Training
**Key Insight**: We DON'T retrain from scratch — we continue training existing model.

**Configuration**:
- Load existing `position_evaluator_best.pth` (epoch 18 weights)
- Train for **5-10 additional epochs** (not full 20)
- **Lower learning rate**: 0.0001 (vs original 0.001) to preserve knowledge
- **Monitor for catastrophic forgetting**: Track performance on original validation set

**Success Criteria**:
- F1 score ≥ 0.8776 (original performance)
- Accuracy ≥ 0.8831 (original performance)
- No significant degradation on original test positions
- Ideally: slight improvement from enhanced diversity

### Phase 5: Stage 2 Training
**With Improved Stage 1**:
- Better move selection in self-play → higher quality Stage 2 training data
- More diverse position coverage → better generalization
- Scenario-aware patterns → richer complexity modeling

---

## Data Distribution Tracking

### Critical Metadata to Monitor

As the dataset grows through multiple feedback cycles, we must track:

#### 1. **Color Balance**
```python
{
  "white_to_move_good": count,
  "white_to_move_bad": count,
  "black_to_move_good": count,
  "black_to_move_bad": count
}
```

**Why**: Current 58.1% White / 40.5% Black bias in self-play suggests we may need to oversample Black positions in future cycles.

#### 2. **Scenario Distribution**
```python
{
  "bullet_early": count,
  "bullet_midgame": count,
  "bullet_endgame": count,
  "blitz_early": count,
  "blitz_midgame": count,
  "blitz_endgame": count,
  "rapid_early": count,
  "rapid_midgame": count
}
```

**Why**: Stage 2 complexity modeling needs balanced representation across all time controls.

#### 3. **Move Diversity per Position**
**Future Enhancement**: Track multiple legal moves per position with scenario context.

**Example**:
```json
{
  "fen": "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
  "legal_moves": [
    {
      "move": "e4e5",
      "eval_cp": 25,
      "scenario_preference": {"bullet": 0.8, "blitz": 0.7, "rapid": 0.6},
      "complexity_added": 3.2
    },
    {
      "move": "Nc3",
      "eval_cp": 22,
      "scenario_preference": {"bullet": 0.5, "blitz": 0.9, "rapid": 0.95},
      "complexity_added": 4.1
    }
  ]
}
```

**Use Case**: Stage 2 can learn that certain tactics (e.g., aggressive pawn pushes) are more effective in bullet/blitz than rapid, and weight complexity scores accordingly.

---

## Incremental Learning Validation

### Preventing Catastrophic Forgetting

**Test Protocol**:
1. **Before Incremental Training**: Save original model's performance on held-out test set
2. **After Incremental Training**: Re-evaluate on same held-out test set
3. **Compare Metrics**:
   - F1 score change: Should be ≥ -0.01 (max 1% degradation acceptable)
   - Accuracy change: Should be ≥ -0.01
   - Precision/Recall change: Should be stable

**Example Validation**:
```
Original Model (epoch 18):
  F1: 0.8776, Acc: 0.8831, Prec: 0.9208, Rec: 0.8382

Incremental Model (epoch 18 + 10):
  F1: 0.8801, Acc: 0.8855, Prec: 0.9221, Rec: 0.8410

Delta: F1 +0.0025, Acc +0.0024 ✅ NO CATASTROPHIC FORGETTING
```

### Dataset Growth Tracking

**Version Log** (`models/dataset_versions.json`):
```json
{
  "versions": [
    {
      "version": "1.0",
      "date": "2026-05-30",
      "total_positions": 1648000,
      "model_checkpoint": "position_evaluator_best.pth",
      "f1_score": 0.8776
    },
    {
      "version": "1.1",
      "date": "2026-06-01",
      "total_positions": 1652000,
      "added_positions": 4000,
      "source": "selfplay_batch_284",
      "model_checkpoint": "position_evaluator_incremental_best.pth",
      "f1_score": 0.8801,
      "incremental_epochs": 10
    }
  ]
}
```

---

## Scenario-Aware Move Weighting (Future)

### The Insight

Different moves have different effectiveness in different time controls:

**Bullet (1+2)**: Favor simplification, quick tactics, forcing moves  
**Blitz (5+4)**: Balance between tactics and positional play  
**Rapid (15+10)**: Deep positional understanding, long-term plans

### Implementation Strategy

**Stage 1 Enhancement** (Future):
- Track which moves were played in which scenarios
- Learn scenario-specific evaluation adjustments
- Example: `eval_cp_bullet = eval_cp_base * scenario_weight["bullet"]`

**Stage 2 Integration**:
- Complexity scores already differentiate scenarios via `time_state`
- When Stage 2 predicts high complexity in bullet game, it learns to prioritize forcing moves
- When Stage 2 predicts low time pressure in rapid, it allows deeper positional analysis

**Cross-Reference Table** (future data structure):
```python
{
  "position_hash": "a1b2c3d4",
  "moves": {
    "e2e4": {
      "base_eval": 25,
      "played_in_scenarios": {
        "bullet": {"count": 120, "win_rate": 0.65},
        "blitz": {"count": 80, "win_rate": 0.58},
        "rapid": {"count": 40, "win_rate": 0.52}
      },
      "complexity_correlation": 0.72  # High complexity = more often played
    }
  }
}
```

---

## Cycle Execution Timeline

### Cycle 1 (Current)
- **Stage 1 v1.0**: Trained on 1.648M positions
- **Self-Play**: 284 games, 7,557 positions
- **Extraction**: ~4,000 balanced GOOD/BAD positions
- **Incremental Training**: 10 epochs on augmented dataset
- **Stage 1 v1.1**: Enhanced model
- **Stage 2 Training**: First training cycle
- **Status**: In progress (extraction phase)

### Cycle 2 (Planned)
- **Stage 2 Verification**: 100-200 games with trained Stage 2 model
- **Extraction**: ~1,000-2,000 new positions
- **Incremental Training**: 5 epochs (faster iteration)
- **Stage 1 v1.2**: Further enhanced
- **Status**: Not started

### Cycle N (Future State)
- **Automated Pipeline**: Self-play → extract → train → deploy
- **Continuous Learning**: Always improving without manual intervention
- **Dataset Size**: 2M+ positions, rich scenario diversity
- **Performance**: Approaching master-level play through self-improvement

---

## Research Questions

As we iterate through cycles, we want to answer:

1. **Learning Speed**: How many cycles to reach plateau in improvement?
2. **Data Efficiency**: What's the minimum positions per cycle for meaningful improvement?
3. **Scenario Specialization**: Do models naturally learn time-control-specific strategies?
4. **Catastrophic Forgetting**: At what dataset size does incremental training become unstable?
5. **Transfer Learning**: Can Stage 2 knowledge be distilled back into Stage 1 evaluation function?

---

## Implementation Checklist

### Immediate (Cycle 1)
- [x] Design continuous improvement architecture
- [ ] Extract Stage 1 labels from self-play (script created)
- [ ] Implement incremental training (script created)
- [ ] Run incremental training and validate performance
- [ ] Document dataset version 1.1 metadata
- [ ] Train Stage 2 model with improved Stage 1

### Short-Term (Cycle 2)
- [ ] Automate extraction pipeline
- [ ] Create dataset versioning system
- [ ] Implement color balance tracking
- [ ] Build scenario distribution analyzer
- [ ] Run Stage 2 verification games

### Long-Term (Cycle 3+)
- [ ] Multiple moves per position tracking
- [ ] Scenario-aware evaluation adjustments
- [ ] Automated cycle orchestration
- [ ] Performance plateau detection
- [ ] Transfer learning experiments (Stage 2 → Stage 1)

---

## Success Metrics

### Per-Cycle Validation
- ✅ No catastrophic forgetting (F1 delta ≥ -0.01)
- ✅ Dataset balance maintained (GOOD/BAD ratio 0.9-1.1)
- ✅ Color balance improving (White/Black ratio approaching 0.5)
- ✅ Scenario diversity maintained (all 8 scenarios represented)

### Long-Term Improvement
- 📈 Stage 1 F1 score trending upward over cycles
- 📈 Stage 2 complexity prediction accuracy improving
- 📈 Full engine ELO rating increasing
- 📈 Win rate vs baseline versions improving

---

## Conclusion

The continuous improvement cycle transforms V7P3R Chess AI from a **static trained model** into a **self-improving learning system**. Each cycle:

1. **Generates new training data** through self-play
2. **Enhances Stage 1** without starting over
3. **Improves Stage 2** with better move selection
4. **Validates scalability** of incremental learning
5. **Documents progress** with rich metadata

This is the **foundation for long-term autonomous improvement** — the AI teaching itself to play better chess through structured feedback loops.

> "Can this AI learn faster than I did?"  
> **Answer**: With 284 games per cycle, matching human tuning speed, and the ability to iterate continuously without fatigue — yes, it can.

---

**Next Steps**: Execute Phase 2 (Label Extraction) and Phase 4 (Incremental Training) to complete Cycle 1.
