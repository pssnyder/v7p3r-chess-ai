# Opponent Engine Performance Analysis
**Date:** April 30, 2026  
**Purpose:** Identify best-performing opponent engines for hybrid AI evaluation simplification experiments

---

## Tournament Results Summary

### Engine Battle 20251118 (Blitz 5+5, 167/280 games)

| Rank | Engine | Score | Win % | vs V7P3R Performance |
|------|--------|-------|-------|---------------------|
| 1 | Stockfish 1% | 39.5/42 | 94.0% | Reference strength |
| 2 | V7P3R_v15.1 | 28.5/42 | 67.9% | Strong V7P3R version |
| 3 | V7P3R_v14.1 | 26.5/42 | 63.1% | Proven stable version |
| 4 | **MaterialOpponent** | **22.0/42** | **52.4%** | **BEST opponent engine** |
| 5 | C0BR4_v3.1 | 19.0/42 | 45.2% | Complex search |
| 6 | **PositionalOpponent** | **18.5/42** | **44.0%** | **2nd best opponent** |
| 7 | SlowMate_v3.1 | 12.5/41 | 30.5% | Mate specialist |
| 8 | RandomOpponent | 0.5/41 | 1.2% | Baseline |

### Engine Battle 20251107 (Bullet 1+1, 210 games complete)

| Rank | Engine | Score | Win % | Notes |
|------|--------|-------|-------|-------|
| 1 | Stockfish 1% | 54.0/60 | 90.0% | Dominant |
| 2 | C0BR4_v3.1 | 38.5/60 | 64.2% | Strong in bullet |
| 3 | **MaterialOpponent** | **36.5/60** | **60.8%** | **Excellent showing** |
| 4 | V7P3R_v14.1 | 33.0/60 | 55.0% | Competitive |
| 5 | V7P3R_v14.0 | 30.5/60 | 50.8% | Competitive |
| 6 | SlowMate_v3.1 | 15.0/60 | 25.0% | Weak in bullet |
| 7 | RandomOpponent | 2.5/60 | 4.2% | Baseline |

---

## Key Findings

### 🥇 **MaterialOpponent** - Clear Winner
- **Overall Performance:** 52-61% across tournaments
- **Consistency:** Strong in both blitz (5+5) and bullet (1+1)
- **Competitive:** Beat V7P3R versions in multiple games
- **Rating Estimate:** ~1400-1500 ELO based on tournament performance

### 🥈 **PositionalOpponent** - Strong Second
- **Overall Performance:** 44% in blitz tournament
- **Unique Style:** Pure PST evaluation (no material constants)
- **Positional Play:** Beat some V7P3R versions through positioning
- **Rating Estimate:** ~1300-1400 ELO

### ❌ **Weak Opponents**
- **CaptureOpponent:** 1.0/15 (6.7%) - Too one-dimensional
- **CoverageOpponent:** 0.0/15 (0%) - Ineffective strategy
- **RandomOpponent:** 1-4% - Baseline control

---

## Evaluation Architecture Comparison

### MaterialOpponent (Simple & Effective)
```python
# Material-only evaluation
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 325,  # Base value
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}

BISHOP_PAIR_BONUS = 50  # Only strategic bonus
BISHOP_ALONE_PENALTY = 50

# Evaluation = Sum of material + bishop pair adjustment
```

**Strengths:**
- ✅ Simple, fast evaluation (~0.1ms per position)
- ✅ Clear objective: win material
- ✅ Predictable move priorities
- ✅ Works well with alpha-beta pruning
- ✅ No complex phase detection or positional bonuses

**Search Features:**
- Minimax with alpha-beta pruning
- Transposition tables (Zobrist hashing)
- Killer moves (2 per depth)
- History heuristic
- Quiescence search (captures only)
- Null move pruning
- Move ordering: TT > Killers > MVV-LVA > History

---

### PositionalOpponent (Pure PST)
```python
# Position-based evaluation only
# No material constants - piece value = PST value!

PAWN_PST:   0-900   (promotion potential)
KNIGHT_PST: 200-400 (centralization)
BISHOP_PST: 250-400 (long diagonals)
ROOK_PST:   400-600 (7th rank penetration)
QUEEN_PST:  700-1100 (center control)
KING_PST:   Middlegame (safety) vs Endgame (center)

# Evaluation = Sum of all PST values for position
```

**Strengths:**
- ✅ Positional awareness built into evaluation
- ✅ No material counting - position IS the evaluation
- ✅ Natural piece coordination incentives
- ✅ Dynamic king safety (middlegame vs endgame)
- ✅ Pawn advancement naturally valued (0 → 900)

**Search Features:** (Same as MaterialOpponent)
- Minimax with alpha-beta pruning
- Transposition tables
- Killer moves, history heuristic
- Quiescence search
- Move ordering identical to MaterialOpponent

---

### V7P3R v18.3 (Complex & Proven)
```python
# Multi-component evaluation
1. PST evaluation (60% weight)
   - Optimized PST_DIRECT (28% faster)
   - Piece-square tables for all pieces
   
2. Material evaluation (40% weight)
   - P=100, N=320, B=330, R=500, Q=900
   
3. Strategic bonuses
   - Rooks on open files (+25)
   - Doubled rooks (+15)
   - Bishop pair bonus (+30)
   - Passed pawns (distance-based)
   - Isolated pawns (-20)
   - Doubled pawns (-10)
   - King safety (pawn shield)
   - Mobility bonuses
   - Center control
   - Piece coordination
   
4. Game phase detection
   - Opening: Full bonuses
   - Middlegame: Strategic emphasis
   - Endgame: King activity, passed pawns
```

**Strengths:**
- ✅ Proven +56 ELO vs v17.1 (58% win rate)
- ✅ Sophisticated positional understanding
- ✅ Adapts to game phase
- ✅ Strong endgame conversion

**Potential Weaknesses for AI Ordering:**
- ⚠️ Complex evaluation (~0.5-1ms per position)
- ⚠️ Many interacting components
- ⚠️ Hard for AI to predict "best move" with so many factors
- ⚠️ Strategic bonuses may conflict with AI's learned patterns

---

## Hypothesis: Evaluation Complexity Mismatch

### The Problem
**V7P3R v20.0.2 = AI Move Ordering + v18.3 Complex Evaluation**

- **AI Model:** Trained on 454,624 positions (100K puzzles + 374K V7P3R games)
- **AI Goal:** Predict which moves V7P3R would rank highly
- **AI Accuracy:** 97.1% top-5 accuracy on training data

**BUT:**
- AI learned from V7P3R's **historical move patterns**
- V7P3R's complex evaluation has **12+ interacting components**
- AI may not understand **HOW** the complex evaluation scores moves
- Result: AI suggests "positionally sound" moves that v18.3 eval scores poorly

### Evidence
1. **10% Tactical Accuracy:** AI finds only 1/10 tactics (back rank mate)
2. **Positional Focus:** AI trained on V7P3R's strategic style, not tactics
3. **Speed Variance:** 562-22,586 NPS (suggests evaluation mismatches)
4. **MaterialOpponent Success:** Simple eval performs at 52-61% vs V7P3R versions

### The Hypothesis
**Simpler evaluation may work BETTER with AI ordering:**

**MaterialOpponent eval:**
- Clear objective: maximize material
- Simple scoring: sum of piece values
- AI can easily predict which moves gain/lose material
- Evaluation aligns with AI's tactical puzzle training (100K puzzles)

**PositionalOpponent eval:**
- Clear objective: improve piece positions
- Simple scoring: sum of PST values
- AI can predict which moves improve positioning
- Evaluation aligns with AI's game position training (374K games)

---

## Proposed A/B Testing Plan

### Create 3 Test Versions

#### **v20.0.2** (Current - Control)
- AI Ordering: MoveOrderingNetwork v4.0
- Evaluation: v18.3 complex (PST + Material + Strategic)
- Search: v18.3 advanced (TT, killers, history, quiescence)
- **Expected:** Baseline hybrid performance

#### **v20.0.2-Material** (Simplification Test #1)
- AI Ordering: MoveOrderingNetwork v4.0 (SAME)
- Evaluation: MaterialOpponent simple (material + bishop pair)
- Search: v18.3 advanced (TT, killers, history, quiescence) (SAME)
- **Hypothesis:** AI ordering + simple material eval = better alignment
- **Expected:** Higher tactical accuracy, faster NPS, clearer move priorities

#### **v20.0.2-Positional** (Simplification Test #2)
- AI Ordering: MoveOrderingNetwork v4.0 (SAME)
- Evaluation: PositionalOpponent PST-only
- Search: v18.3 advanced (TT, killers, history, quiescence) (SAME)
- **Hypothesis:** AI ordering + pure PST eval = positional strength
- **Expected:** Better game phase accuracy, strong middlegame, piece coordination

---

## Tournament Testing Plan

### Phase 1: Internal Validation (20 games each pairing)
**Engines:**
1. v20.0.2 (current hybrid)
2. v20.0.2-Material
3. v20.0.2-Positional
4. v18.3 (proven baseline)
5. MaterialOpponent (reference)
6. PositionalOpponent (reference)

**Format:** Round-robin, 5+3 blitz
**Total Games:** 15 pairings × 20 games = 300 games

**Metrics to Track:**
- Win/Loss/Draw rates
- Tactical accuracy (find mate/tactics)
- Positional accuracy (game phase decisions)
- Average NPS (speed)
- Move time distribution
- TT/Killer hit rates

### Phase 2: Best Performer vs Production (50 games)
- Best hybrid variant vs v19.5 (current production)
- Validate improvement over flawed baseline

### Phase 3: Extended Testing (100 games if promising)
- Multiple time controls (1+1 bullet, 5+3 blitz, 15+10 rapid)
- Opponent diversity (Stockfish 1%, C0BR4, tactical bots)

---

## Expected Outcomes

### Scenario 1: **MaterialOpponent Eval Wins**
**Interpretation:** AI ordering works best with simple material evaluation
**Next Steps:**
- Deploy v20.0.2-Material as hybrid baseline
- Fine-tune on tactical puzzles (Stage 3 training)
- Focus AI on material-winning patterns

### Scenario 2: **PositionalOpponent Eval Wins**
**Interpretation:** AI ordering works best with pure PST evaluation
**Next Steps:**
- Deploy v20.0.2-Positional as hybrid baseline
- Train AI on positional game records (Stage 3)
- Emphasize piece coordination in training

### Scenario 3: **v20.0.2 (v18.3 eval) Still Best**
**Interpretation:** Complex evaluation IS working, tactical accuracy is AI limitation
**Next Steps:**
- Keep v20.0.2 architecture
- Focus on AI model improvement (Stage 3 tactical training)
- Add tactical puzzle dataset to training

### Scenario 4: **All Hybrids Lose to v18.3**
**Interpretation:** AI ordering overhead not worth it yet
**Next Steps:**
- Revert to pure v18.3 for production
- Improve AI model accuracy before retry
- Consider different model architecture

---

## Implementation Notes

### Code Changes Required
Both variants only need **evaluation method replacement** in `v7p3r_v20_hybrid.py`:

**Current (v18.3):**
```python
def evaluate_position(self, board: chess.Board) -> int:
    # Complex PST + Material + Strategic bonuses
    return self.evaluator.evaluate(board)
```

**MaterialOpponent variant:**
```python
def evaluate_position(self, board: chess.Board) -> int:
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = PIECE_VALUES[piece.piece_type]
            score += value if piece.color == chess.WHITE else -value
    
    # Bishop pair bonus
    white_bishops = len(board.pieces(chess.BISHOP, chess.WHITE))
    black_bishops = len(board.pieces(chess.BISHOP, chess.BLACK))
    if white_bishops == 2: score += 50
    elif white_bishops == 1: score -= 50
    if black_bishops == 2: score -= 50
    elif black_bishops == 1: score += 50
    
    return score
```

**PositionalOpponent variant:**
```python
def evaluate_position(self, board: chess.Board) -> int:
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            pst_value = self._get_pst_value(piece, square)
            score += pst_value if piece.color == chess.WHITE else -pst_value
    return score

def _get_pst_value(self, piece, square):
    # Use PositionalOpponent's PST tables
    # (copy from positional_opponent.py)
    ...
```

**Everything else stays the same:**
- AI move ordering at root
- v18.3 search algorithm (TT, killers, history, quiescence)
- UCI protocol handling
- Time management

---

## Success Criteria

### For Deployment
A hybrid variant must meet **ALL** criteria to replace v20.0.2:

1. **Win Rate:** ≥55% vs v20.0.2 (20 game minimum)
2. **Tactical Accuracy:** ≥15% on test suite (up from 10%)
3. **Game Phase Accuracy:** ≥60% (same or better)
4. **Average NPS:** ≥8,000 (speed acceptable)
5. **Stability:** No crashes, timeouts, or illegal moves
6. **Production Test:** ≥50% vs v19.5 (current production)

### For Further Research
If no variant meets deployment criteria:
- Document which eval performed best
- Analyze why (move ordering alignment, speed, etc.)
- Use insights for Stage 3 training data selection
- Consider hybrid approach (MaterialOpponent for opening, PositionalOpponent for endgame)

---

## Timeline

1. **Day 1 (Today):** Create v20.0.2-Material and v20.0.2-Positional
2. **Day 1-2:** Run internal validation tournament (300 games)
3. **Day 2-3:** Analyze results, identify best performer
4. **Day 3-4:** Extended testing of winner (100+ games)
5. **Day 4-5:** Production validation vs v19.5
6. **Day 5+:** Deploy or iterate based on results

---

## Conclusion

**MaterialOpponent** has proven to be a surprisingly strong engine with:
- 52-61% tournament performance
- Simple, fast evaluation
- Competitive with V7P3R versions

**PositionalOpponent** provides:
- Pure PST evaluation
- Positional playing style
- 44% tournament performance

Both offer **drastically simpler evaluations** than v18.3's complex multi-component system. By creating hybrid variants using these evaluations with V7P3R's proven search algorithm and AI move ordering, we can test whether evaluation complexity is helping or hurting the AI's effectiveness.

**The fundamental question:** Is v7p3r's evaluation standing in the way of the AI, or do we need to improve the AI model itself?

This A/B testing will provide the answer.
