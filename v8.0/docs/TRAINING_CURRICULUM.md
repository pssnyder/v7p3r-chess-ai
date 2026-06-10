# V7P3R v8.0 Training Curriculum

**Status**: Validated with puzzle-based ELO estimation system  
**Generated**: $(date)  
**Benchmark System**: 100-puzzle suite (5 tiers × 20 puzzles) with 5-point scoring  

## Curriculum Overview

This document defines a graduated training progression for V7P3R v8.0 using validated opponent engines. The progression is designed to build strength systematically from basic tactics to expert-level play.

---

## Stage 1: Foundation (ELO 200-600)

### Objective
Master fundamental tactics: piece capture, simple pins, forks, basic endgame positions.

### Primary Opponents

| Engine | Strength | Win Rate Target | Notes |
|--------|----------|-----------------|-------|
| **RandomOpponent** | 200 (ELO) | 95%+ | Plays random legal moves - no real strategy |
| **SlowMate v0.0.1** | 1300 (ELO) | 70%+ | Basic weaknesses exploitable |

### Stage Metrics
- **Games**: 50 minimum
- **Target Win Rate**: 80%+ (vs random), 60%+ (vs SlowMate v0.0.1)
- **Time Control**: Bullet/Blitz (1min+2s or 5min+4s)
- **Expected Duration**: 2-3 hours

### Training Focus
- Recognize all basic tactical motifs (pins, forks, skewers)
- Perfect piece values and exchanges
- Endgame basics: K+P vs K, K+R vs K
- Time management at faster controls

### Success Criteria
- ✅ Win rate ≥80% vs RandomOpponent  
- ✅ Win rate ≥60% vs SlowMate v0.0.1  
- ✅ No illegal moves  
- ✅ Time forfeit rate <5%

---

## Stage 2: Intermediate Tactics (ELO 800-1400)

### Objective
Develop intermediate tactical strength and basic positional understanding.

### Primary Opponents

| Engine | Strength | Win Rate Target | Test Method |
|--------|----------|-----------------|-------------|
| **SlowMate v0.1.0** | 1034 (ELO) | 55%+ | 20+ games |
| **SlowMate v0.2.0** | 960 (ELO) | 60%+ | 20+ games |
| **Copycat v1.0** | 960 (ELO) | 55%+ | 20+ games |
| **Cece v1.0** | 960 (ELO) | 50%+ | 20+ games |

### Stage Metrics
- **Games**: 80+ minimum (20 per opponent)
- **Target Win Rate**: 55% average vs opponents
- **Time Control**: Rapid (15min+10s recommended)
- **Expected Duration**: 8-12 hours

### Training Focus
- Evaluate positions beyond material count
- Basic opening principles (center control, piece development)
- Intermediate tactics: discovered attacks, back rank threats
- Pawn structures and weaknesses
- Endgame technique with multiple pieces

### Success Criteria
- ✅ Win rate ≥55% vs all intermediate opponents
- ✅ Consistent improvement across 20+ games vs each opponent
- ✅ No catastrophic tactical oversights (avoiding blunders)
- ✅ Demonstrate positional understanding (not just tactics)

---

## Stage 3: Advanced Play (ELO 1500-1800)

### Objective
Compete with strong engines across diverse positions. Consolidate all previous skills while adding strategic depth.

### Primary Opponents

| Engine | Strength | Benchmark | Test Method |
|--------|----------|-----------|-------------|
| **SlowMate v2.0** | 1700 (ELO) | Tier 1-3 | 30+ games |
| **Cece v2.3** | 1300 (ELO) | Strong player | 20+ games |

### Stage Metrics
- **Games**: 50+ minimum
- **Target Win Rate**: 50%+ (evenly matched level)
- **Time Control**: Rapid (15min+10s recommended)
- **Expected Duration**: 10-15 hours

### Training Focus
- Intermediate positional understanding (pawn structure basics)
- Tactical consistency across diverse positions
- Endgame technique with multiple pieces
- Opening principles and middlegame transitions
- Time management across different controls

### Success Criteria
- ✅ Win rate ≥50% vs SlowMate v2.0  
- ✅ CPL (Centipawn Loss) <200 in complex positions
- ✅ Consistent tactical solutions
- ✅ Stable performance across 30+ games

---

## Stage 4: Expert Mastery (ELO 1900-2100)

### Objective
Achieve expert-level play against equal/stronger opponents. Refinement and optimization.

### Primary Opponents (All ELO 2100)

| Engine | Benchmark | Puzzle Tiers | Recommendation |
|--------|-----------|--------------|-----------------|
| **V7P3R v17.1** | FUNCTIONAL | Tier 1-5: 95%+ | Golden standard |
| **V7P3R v17.8** | FUNCTIONAL | Tier 1-3: 95%+ | Alternative test |
| **V7P3R v18.4** | FUNCTIONAL | Tier 1-5: 95%+ | Latest variant |
| **C0BR4 v3.1** | FUNCTIONAL | Tier 1-3: 95%+ | Different engine style |

### Stage Metrics
- **Games**: 50+ recommended
- **Target Win Rate**: 45%+ (challenging but beatable)
- **Time Control**: Rapid/Classical (10min+ recommended)
- **Expected Duration**: 10-15 hours

### Training Focus
- Competitive play against near-equal strength opponents
- Opening repertoire development
- Middlegame positional mastery
- Endgame optimization techniques
- Handling time pressure at tournament controls

### Success Criteria
- ✅ Win rate ≥45% vs expert opponents
- ✅ Blunders per game ≤6
- ✅ Demonstrate all previously learned skills
- ✅ Measurable performance stability

---

## Testing Methodology

### Puzzle-Based ELO Estimation
All opponents have been benchmarked using a standardized 100-puzzle suite:

```
Tier 1 (Beginner):     Rating 400-1000   (20 puzzles)
Tier 2 (Weak):         Rating 1000-1300  (20 puzzles)
Tier 3 (Intermediate): Rating 1300-1800  (20 puzzles)
Tier 4 (Advanced):     Rating 1800-2200  (20 puzzles)
Tier 5 (Expert):       Rating 2200-3000  (20 puzzles)
```

**ELO Estimation Formula**:
- Highest tier with >40% accuracy determines ceiling
- ELO = Tier ceiling ± 100cp adjustment

### Example Benchmark Results
V7P3R v18.0 (Calibrated & Lichess-Validated):
- Tier 1: 18/20 (90%) ✓
- Tier 2: 20/20 (100%) ✓
- Tier 3: 18/20 (90%) ✓
- Tier 4: 15/20 (75%) ✓
- Tier 5: 10/20 (50%) ✓
- **Raw Benchmark ELO**: 2100
- **Calibrated ELO**: 1551 (73.86% correction)
- **Lichess Actual**: 1544 ✓ (Match!)

---

## Curriculum Progression Timeline

| Stage | Phase Duration | Cumulative Hours | Engine Strength |
|-------|----------------|------------------|-----------------|
| 1: Foundation | 2-3h | 2-3h | 200-600 |
| 2: Intermediate | 8-12h | 10-15h | 800-1400 |
| 3: Advanced | 15-20h | 25-35h | 1500-1800 |
| 4: Mastery | 20-30h | 45-65h | 1900-2100 |

**Total Estimated Time**: 45-65 hours of training

---

## Stage Transitions

### From Stage 1 → Stage 2
**Requirement**: Win rate ≥80% vs RandomOpponent over 25+ games

When transitioning, introduce Stage 2 opponents gradually:
1. Play 10 games vs SlowMate v0.1.0
2. If win rate ≥55%, continue all Stage 2 opponents
3. If win rate <40%, extend Stage 1 training

### From Stage 2 → Stage 3
**Requirement**: Average win rate ≥55% vs Stage 2 opponents (80+ games minimum)

Validation:
- Test vs SlowMate v2.0 (preliminary: target 50%+)
- If confirmed, fully transition to Stage 3
- Keep Stage 2 opponents as occasional practice

### From Stage 3 → Stage 4
**Requirement**: Win rate ≥50% vs SlowMate v2.0 (30+ games)

Expert training:
- Challenge vs V7P3R v18.0 (proven 1544 Lichess rating)
- Challenge vs C0BR4 v3.1 (proven 1558 Lichess rating)
- Continue Stage 3 against varied opponents
- Solidify all previous skills before expert level

---

## Monitoring & Adjustment

### Key Metrics to Track

```json
{
  "stage": "current_stage",
  "games_completed": 0,
  "win_rate": 0.0,
  "avg_centipawn_loss": 0,
  "blunders_per_game": 0,
  "time_forfeit_rate": 0.0
}
```

### Performance Thresholds

| Metric | Acceptable | Warning | Action |
|--------|-----------|---------|--------|
| Win Rate | ≥50% target | 40-50% | Review opening/endgame |
| CPL | <150 | 150-200 | Increase analysis depth |
| Blunders/Game | ≤5 | 5-8 | Add tactic training |
| Forfeit Rate | <5% | 5-10% | Reduce time control speed |

### Curriculum Adjustment Rules

1. **If win rate drops >10%**: Add 20 games vs previous stage opponents
2. **If blunders spike**: Return to foundation work (Stage 1 tactics)
3. **If plateau for 50+ games**: Change opponent pairing or opening selection
4. **If sudden improvement**: Accelerate to next stage (after validation)

---

## Opponent Selection Rationale

### Why These Specific Engines?

1. **RandomOpponent**: Provides baseline (no skill - pure chance)
2. **SlowMate Series**: Consistent progression (v0.0.1→v0.1.0→v0.2.0→v2.0)
3. **Cece Series**: Different evaluation style (positional vs tactical)
4. **Copycat v1.0**: Unique opening/middlegame patterns
5. **V7P3R Series**: Proven reference engines with consistent benchmarks
6. **C0BR4 v3.1**: Alternative architecture (C# vs Python) for style adaptation

### Benchmark Reliability

All selected engines have been validated via:
- ✅ Puzzle-based ELO estimation (100-puzzle suite)
- ✅ Tier performance analysis (5 difficulty levels)
- ✅ Consistency checks (repeatable results)
- ✅ Cross-validation (known engine comparisons)

---

## Advanced Options

### Variant Curricula

**Speed Training Path** (Focus on Bullet/Blitz):
- Stage 1-2: Same as above
- Stage 3: SlowMate v2.0 at 5min+4s
- Stage 4: V7P3R v17.1 at 3min+2s

**Positional Training Path** (Focus on Strategic Play):
- Add Cece v2.0 and Cece v2.3 to all stages
- Use longer time controls (15min+)
- Analyze games post-match for positional ideas

**Tactical Mastery Path** (Focus on Tactics):
- Extend Stage 1-2 with more puzzle analysis
- Use puzzle positions from game analysis
- Intensive pattern recognition training

---

## Curriculum Completion Criteria

### Final Validation (End of Stage 4)

All of the following must be true:

1. ✅ 50+ games vs expert opponents (ELO 1500+)
2. ✅ Win rate ≥45% in final 30 games
3. ✅ <6 blunders per game average
4. ✅ CPL <200 in complex positions
5. ✅ Successful vs minimum 2 different expert opponents
6. ✅ Stable performance (consistent win rate)
7. ✅ All previous stage benchmarks maintained (no regression)

### Graduation Certificate
Upon completion, document:
- Final ELO estimate
- Win rates vs each opponent family
- Notable games and lessons learned
- Recommended next improvements

---

## Maintenance Phase

After curriculum completion, ongoing training should:

- **Weekly**: 10 games vs varied opponents (mix of stages)
- **Monthly**: Full benchmark vs V7P3R v17.1 (reference engine)
- **Quarterly**: Puzzle suite re-test (target ≥90% on all tiers)
- **Yearly**: Full curriculum progression test

---

## References

- **Benchmark Data**: opponents_catalog.csv
- **Puzzle Database**: puzzles.db (4.9M Lichess puzzles, rating 399-3424)
- **Test Suite**: benchmark_suite.json (100-puzzle validation standard)
- **Engine Details**: See individual engine pages in docs/

---

**Last Updated**: $(date)  
**Curriculum Version**: 1.0  
**Status**: Ready for training deployment
