# V6.1 Training Architecture - "Superhuman v7p3r" AI
## Focused, Streamlined Training for Tal-Inspired Tactical Chess

**Created**: 2026-05-26  
**Philosophy**: Train AI to play like "superhuman you" - YOUR actual tactical strengths + aspirational GM knowledge  
**Target**: Phase 1 - 1200 ELO (prevent blunders + tactical aggression)

---

## 🎯 Two-Stage Architecture (VALIDATED APPROACH)

### Stage 1: Position Evaluator (CURRENT FOCUS)
**Purpose**: Learn to recognize "good" vs "bad" positions according to YOUR tactical preferences

**Input**: 76-92 position features (F001-F092)
- Material balance, king safety, piece activity
- Pawn structure, center control, tactical patterns
- Rook placement, knight outposts, bishop pairs

**Output**: Binary classification
- `label = 1` (GOOD position - want to be here)
- `label = 0` (BAD position - avoid this)

**Model Architecture**:
- Graph-Augmented Policy Network
- Transposition attention (K=10 similar positions)
- Layers: [1024, 512, 256, 128]
- Weighted Binary Cross-Entropy Loss

**Training Objective**: Learn feature correlations that define YOUR style
- High checkmate threat = GOOD
- King safety asymmetry (yours safe, theirs exposed) = GOOD
- Tactical complexity with deterministic path = GOOD
- One-move blunders = BAD (weight 5.0x)

### Stage 2: Move Selector (FUTURE - POST STAGE 1)
**Purpose**: Given current position, select move that leads to "your style" position

**Input**: Current position features + candidate move features
**Output**: Move ranking/probability distribution
**Method**: Use Stage 1 evaluator to score positions after each legal move

**NOT YET IMPLEMENTED** - Stage 1 must work first

---

## 📊 Data Source Hierarchy

### Tier 0: Blunder Prevention (Weight 5.0x) - CRITICAL
**Source**: Positions before one-move blunders from your games
**Purpose**: Learn what NOT to do (prevent hanging pieces, avoid tactics against you)
**Label Strategy**: Position BEFORE blunder = BAD (0), similar position with correct move = GOOD (1)
**Implementation**: Extract from your losses and near-misses

### Tier 1: Tactical Mastery - YOUR Signature (Weight 3.0x)
**Sources**:
1. **Your Chess.com Tactical Wins** - `HumanTacticalGamesLoader`
   - File: `v7p3r_20250530.pgn` (100+ games)
   - Filter: Games with Bxf7+ king hunt patterns (weight 5.0x within tier)
   - Filter: Wins in under 25 moves with tactical themes
   - Label Strategy: Positions in winning sequences = GOOD (1)

2. **Mikhail Tal Master Games** - `TalGamesLoader` ⭐ NEW
   - File: `mikhail_tal_master_games.pgn` (110+ games)
   - Filter: Tal wins with sacrificial attacks
   - Extract: Positions with tactical complications + sound compensation
   - Label Strategy: Tal's positions in wins = GOOD (1), opponent positions = BAD (0)

### Tier 2: Pattern Recognition (Weight 2.5x)
**Sources**:
3. **Tactics Puzzles** - `TacticsLoader`
   - File: `csv_data_puzzles/*.csv` (861.5 MB)
   - Themes: Sacrifices, king hunts, mating attacks, forks, pins
   - Label Strategy: Solution positions = GOOD (1), pre-solution positions = context

### Tier 3: Repertoire Building (Weight 2.0x)
**Sources**:
4. **Opening Repertoire** - `OpeningPGNLoader`
   - Directory: `pgn_data_openings/` (120 PGN files)
   - Prioritize: 1.e4 openings, King's Gambit, Vienna, Italian
   - Gambit Threshold: -200cp (2 pawns) acceptable for initiative
   - Label Strategy: Final opening position (moves 12-15) evaluated, entire sequence labeled by outcome

### Tier 4: General Chess Knowledge (Weight 1.0x - BASELINE)
**Sources**:
5. **Lichess Database** - `LichessDBLoader`
   - File: `lichess_db_eval.jsonl` (millions of positions)
   - Filter: depth ≥ 15, pre-evaluated by Stockfish
   - Grade Thresholds: 1 (≥300cp), 2 (150-300), 3 (-150 to 150), 4 (-300 to -150), 5 (≤-300)
   - Label Strategy: Grades 1-2 = GOOD, Grades 4-5 = BAD, Grade 3 = contextual

6. **V7P3R Engine Wins** - `V7P3RGameLoader`
   - Files: Original `good_positions.jsonl` (5.7M) + `bad_positions.jsonl` (69k)
   - Additional: Engine vs Engine PGNs (510 games)
   - Purpose: Baseline engine self-play knowledge
   - Label Strategy: Pre-labeled in original dataset

### Tier 5: Endgame Mastery (Weight 1.5x)
**Sources**:
7. **Endgame Databases** - `EndgameLoader`
   - Directory: `pgn_data_endgames/` (theoretical endgames)
   - Filter: ≤10 pieces on board
   - Label Strategy: Winning side positions = GOOD, losing side = BAD

---

## 🔄 Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  RAW DATA SOURCES                                           │
├─────────────────────────────────────────────────────────────┤
│  1. v7p3r_20250530.pgn (Your Chess.com games)              │
│  2. mikhail_tal_master_games.pgn (Tal's games) ⭐ NEW      │
│  3. csv_data_puzzles/*.csv (Tactics)                        │
│  4. pgn_data_openings/ (120 PGN files)                     │
│  5. lichess_db_eval.jsonl (Millions pre-evaluated)         │
│  6. good_positions.jsonl + bad_positions.jsonl (5.7M+69k)  │
│  7. pgn_data_endgames/ (Theoretical endgames)              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  DATA LOADERS (Implement DataSourceLoader Interface)        │
├─────────────────────────────────────────────────────────────┤
│  • HumanTacticalGamesLoader (Tier 1, weight 3.0x)          │
│  • TalGamesLoader (Tier 1, weight 3.0x) ⭐ NEW             │
│  • TacticsLoader (Tier 2, weight 2.5x)                     │
│  • OpeningPGNLoader (Tier 3, weight 2.0x)                  │
│  • LichessDBLoader (Tier 4, weight 1.0x)                   │
│  • V7P3RGameLoader (Tier 4, weight 1.0x)                   │
│  • EndgameLoader (Tier 5, weight 1.5x)                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  POSITION EXTRACTION & FEATURE CALCULATION                  │
├─────────────────────────────────────────────────────────────┤
│  For each position:                                         │
│    • FEN → chess.Board()                                    │
│    • Calculate 76-92 features (F001-F092)                   │
│    • Assign label (GOOD=1, BAD=0)                          │
│    • Assign grade (1-5 for quality tiers)                   │
│    • Assign source (for tracking)                           │
│    • Apply tier weight multiplier                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  MULTI-SOURCE ORCHESTRATOR                                  │
├─────────────────────────────────────────────────────────────┤
│  MultiSourceDataLoader:                                     │
│    • Mix Ratios (UPDATED for Tal emphasis):                │
│      - Tal games: 20% (was 0%, NEW PRIORITY)               │
│      - Your games: 15% (was 0%, NEW)                       │
│      - Tactics: 15% (was 5%, INCREASED)                    │
│      - Openings: 15% (was 10%, INCREASED)                  │
│      - Lichess DB: 20% (was 70%, DECREASED)                │
│      - V7P3R engine: 10% (was 10%, MAINTAINED)             │
│      - Endgames: 5% (was 5%, MAINTAINED)                   │
│                                                             │
│    • Balance to 50:50 good/bad                             │
│    • Shuffle batches (seed=42)                             │
│    • Stream in 10k batches                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  TRAINING PIPELINE                                          │
├─────────────────────────────────────────────────────────────┤
│  train_policy.py (Stage 1):                                │
│    • Input: Position features (76-92 dimensions)            │
│    • Model: Graph-Augmented Policy Network                 │
│    • Loss: Weighted BCE (bad positions weighted 1.5x)      │
│    • Optimizer: Adam, LR=0.001                             │
│    • Epochs: 50-100                                        │
│    • Validation: 20% holdout set                           │
│    • Metrics: Accuracy, precision, recall, F1              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  TRAINED MODEL OUTPUT                                       │
├─────────────────────────────────────────────────────────────┤
│  v6.1_stage1_position_evaluator.pth                        │
│    • Can classify any position as GOOD (1) or BAD (0)      │
│    • Learned YOUR tactical preferences                     │
│    • Recognizes Tal-style complications                    │
│    • Avoids blunders (one-move mistakes)                   │
│    • Ready for Stage 2 move selection                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Data Mixing Strategy (REVISED FOR TAL EMPHASIS)

### Previous Mix (v6.1 initial):
```python
DEFAULT_MIX = {
    'lichess': 0.70,    # Too much engine-style play
    'v7p3r': 0.10,      # Baseline engine knowledge
    'openings': 0.10,   # Repertoire
    'tactics': 0.05,    # Too little tactical emphasis
    'endgames': 0.05    # Conversion skills
}
```

### **NEW Mix (v6.1 Tal-Inspired)**:
```python
TAL_INSPIRED_MIX = {
    'tal_games': 0.20,       # ⭐ NEW - GM tactical mastery
    'human_games': 0.15,     # ⭐ NEW - YOUR actual tactical style
    'tactics': 0.15,         # INCREASED - pattern recognition
    'openings': 0.15,        # INCREASED - aggressive repertoire
    'lichess': 0.20,         # DECREASED - general knowledge only
    'v7p3r': 0.10,           # MAINTAINED - engine baseline
    'endgames': 0.05         # MAINTAINED - conversion
}
```

### Rationale:
- **50% Tactical Focus** (Tal + Human + Tactics) - Learn aggressive chess
- **15% Repertoire** (Openings) - Build opening foundation
- **20% General Knowledge** (Lichess) - Don't overfit to tactics
- **15% Engine/Endgame** (V7P3R + Endgames) - Baseline competence

### Balance Target: **50:50 good/bad positions**
- Each source contributes balanced labels
- MultiSourceDataLoader ensures final 50:50 split
- Prevents model from predicting all good or all bad

---

## 🔧 Implementation Checklist

### ✅ Completed:
- [x] Base loader interface (`DataSourceLoader`)
- [x] V7P3R game loader (engine self-play)
- [x] Lichess DB loader (pre-evaluated positions)
- [x] Opening PGN loader (repertoire)
- [x] Tactics loader (puzzles)
- [x] Endgame loader (theoretical positions)
- [x] Multi-source orchestrator (mixing strategy)
- [x] Stockfish validator with SQLite cache
- [x] Feature extraction pipeline (76-92 features)
- [x] Study guide documenting preferences

### 🔄 In Progress:
- [ ] **TalGamesLoader** (extract Tal's tactical games) ⭐ IMMEDIATE
- [ ] **HumanTacticalGamesLoader** (extract YOUR tactical wins) ⭐ IMMEDIATE
- [ ] **Update MultiSourceDataLoader** (integrate new loaders, update mix ratios)
- [ ] **Blunder extractor** (positions before mistakes - Tier 0)
- [ ] **train_policy.py integration** (use MultiSourceDataLoader)
- [ ] **Feature correlation tracker** (which features define your style?)

### 📋 Planned (Stage 2):
- [ ] Move feature extractor (move → feature delta)
- [ ] Move selector model (Stage 2)
- [ ] Integration with V7P3R engine search
- [ ] Live game testing against engine baselines

---

## 📏 Success Metrics

### Stage 1 Validation (Position Evaluator):
**Target Metrics** (before moving to Stage 2):
- ✅ Accuracy ≥85% on validation set
- ✅ Precision ≥80% for GOOD positions (minimize false positives)
- ✅ Recall ≥80% for BAD positions (catch blunders)
- ✅ F1 Score ≥82% (balanced performance)
- ✅ Class balance maintained (50:50 in validation set)

### Qualitative Validation:
**Test Positions from YOUR Games**:
1. Load a Bxf7+ sacrifice position → Model should predict GOOD (1)
2. Load position before a blunder from your losses → Model should predict BAD (0)
3. Load Tal's sacrificial attack → Model should predict GOOD (1)
4. Load opponent's position when you won → Model should predict BAD (0)

### Performance Target (Live Play):
- **Phase 1 Goal**: 1200 ELO (prevent blunders + execute tactics)
- **Blunder Rate**: <3 per game (currently ~6-8)
- **Tactical Success**: ≥70% of Bxf7+ attempts succeed
- **Win Rate**: ≥55% against 1000-1200 rated opponents

---

## 🚀 Next Steps

### Immediate (This Session):
1. **Create TalGamesLoader** - Extract Tal's winning positions
2. **Create HumanTacticalGamesLoader** - Extract YOUR tactical wins
3. **Update MultiSourceDataLoader** - Integrate new loaders with TAL_INSPIRED_MIX ratios
4. **Test data pipeline** - Verify 50:50 balance, check feature extraction
5. **Document loader outputs** - Ensure compatibility with train_policy.py

### Week 1:
1. Run full training with new mix ratios
2. Validate Stage 1 model accuracy ≥85%
3. Test on qualitative positions (Tal games, your games)
4. Analyze feature correlations (which features define tactical style?)

### Week 2-3:
1. Implement Feature Correlation Tracker
2. Fine-tune weights based on feature importance
3. Add blunder prevention data (Tier 0)
4. Achieve Phase 1 performance target (1200 ELO)

### Month 2:
1. Design Stage 2 (Move Selector)
2. Extract move sequences from Tal/User games
3. Integrate with V7P3R engine search
4. Live tournament testing

---

## 💡 Key Insights

### Why Two Stages Works:
1. **Stage 1** learns WHAT positions you want (feature preferences)
2. **Stage 2** learns HOW to get there (move selection strategy)
3. Separation allows interpretability (we can see WHY the AI likes a position)

### Why This Isn't Overcomplicating:
- Stage 1 is JUST position evaluation (simpler than move prediction)
- We have TONS of position data (easier to train)
- Move prediction (Stage 2) requires move labels (harder to get)
- This approach lets us start with what we have (positions) and build up

### Tal's Philosophy in Data:
- **Chaos Creation**: Tactical positions from Tal/User games (GOOD labels)
- **Deterministic Navigation**: Feature-based evaluation guides through chaos
- **Opponent Confusion**: Complexity metrics in features
- **Your Path**: Feature correlations learned from YOUR winning patterns

---

**Document Status**: Ready for Implementation  
**Next Action**: Create TalGamesLoader and HumanTacticalGamesLoader
