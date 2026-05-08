# V7P3R AI Feature Set Definition
**Heuristic Observations for Neural Network Training**

---

## 🎯 Purpose

This document lists ALL heuristics from V7P3R v18.3 evaluation profiler (`eval_profile_results`).

**Your Task:** Mark which heuristics to convert into **binary/categorical features** for AI training.

**Philosophy:** 
- ❌ NOT weighted scores (let AI learn weights)
- ✅ Observable states (true/false, counts, categories)
- ✅ Measurements the AI can learn from

---

## 📋 Heuristic Categories

### **1. PERFORMANCE METRICS**
*Time and computational efficiency measurements*

**Time Performance:**
- `eval_time` - nullable - Total evaluation time for position, time spent by engine in search. 
- `reserved_time` - nullable - Time management calculation (amount of time reserved by engine for the search)
- `move_time` - nullable - Total time spent on this move
- `tempo_gain` - nullable - Reserved time minus actual move time (time saved)

**Code Performance:**
- `function_calls` - nullable - Total count of function calls for this position
- `node_count` - nullable - Total nodes explored in move tree
- `nodes_per_second` - nullable - Search speed (nodes/sec)
- `depth_reached` - nullable - Maximum depth reached during search
- `cutoff_count` - nullable - Beta cutoff count for pruned branches

**Function Performance:**
- `function_call_id` - nullable - Unique ID for code performance record
- `function_name` - nullable - Name of function being profiled
- `function_runtime` - nullable - Runtime data about specific function

---

### **2. MOVE ORDERING HEURISTICS**
*How moves are prioritized during search*

- `ordered_moves` - nullable - Complete ordered move list
- `transposition_moves` - nullable - Moves from transposition table lookup
- `capture_moves` - nullable - Capture moves (MVV-LVA algorithm)
- `check_moves` - nullable - Moves that give check
- `killer_moves` - nullable - Killer move heuristic matches
- `tactical_moves` - nullable - Moves with tactical bonus >= 20
- `quiet_moves` - nullable - Non-tactical moves (tactical bonus < 20)

---

### **3. POSITION CONTEXT**
*High-level position characteristics*

- `current_material` - required - Material balance calculation
- `piece_info` - required - Piece inventory (counts by type)
- `game_phase` - required - Current game phase (opening/middlegame/endgame)
- `tactical_flags` - nullable - Tactical pattern flags detected

---

### **4. FAST EVALUATOR**
*Core evaluation components (ACTIVE in v18.3)*

**Primary Scores:**
- `combined_score` - required - Perspective-based total evaluation
- `material_score` - required - Material count evaluation
- `pst_score` - nullable - Piece-square table score (position bonuses)

**Phase Detection:**
- `opening_phase` - Boolean: is position in opening?
- `endgame_phase` - Boolean: is position in endgame?

**Strategic Components:**
- `strategic_bonus` - Game phase-specific strategic adjustments
- `middlegame_bonus` - Middlegame-specific bonuses

**Middlegame Bonus Components:**
- Rooks on open files (+20cp)
- Rooks on semi-open files (+10cp)
- King pawn shield (+10cp per pawn)
- Passed pawns (+30cp)
- Doubled pawns (-20cp)

---

### **5. MODULAR EVALUATOR HEURISTICS**
*58 evaluation functions (many are PLACEHOLDERS in v18.3)*

#### **5.1 Material & Basic Evaluation**
- `material_counter` - Basic material counting (P=100, N=320, B=330, R=500, Q=900)
- `piece_square_tables` - PST positional bonuses

#### **5.2 Tactical Detection**
- `hanging_pieces` - nullable - Undefended pieces (captures without recapture)
- `capture_priority` - nullable - Recaptures and material-winning captures
- `check_threats` - nullable - Check-giving moves and mate threats
- `pins_forks_skewers` - nullable - Tactical pattern detection
- `see_evaluation` - nullable - Static Exchange Evaluation
- `trapped_pieces` - nullable - Pieces with no escape squares
- `back_rank_threats` - nullable - Back rank mate detection

#### **5.3 King Safety**
- `king_safety_basic` - nullable - Pawn shield and basic king exposure
- `king_safety_complex` - nullable - Attack patterns, tropism, storm detection
- `king_centralization` - nullable - King activity bonus in endgame

#### **5.4 Pawn Structure**
- `passed_pawns` - nullable - Passed pawn bonuses (distance, king proximity)
- `doubled_pawns` - nullable - Doubled/tripled pawn penalties
- `isolated_pawns` - nullable - Isolated pawn penalties
- `backward_pawns` - nullable - Backward pawn penalties
- `pawn_chains` - nullable - Connected pawn chain bonuses
- `pawn_structure_diff` - Pawn structure differential
- `opposition` - nullable - King opposition in pawn endgames
- `square_of_pawn` - nullable - Pawn race evaluation (rule of square)

#### **5.5 Piece-Specific Bonuses**
- `bishop_pair` - nullable - Bonus for having both bishops
- `knight_outposts` - nullable - Knights on strong outpost squares
- `rook_on_7th` - nullable - Rook on 7th rank bonus
- `rook_on_open_file` - nullable - Rook on open/semi-open file bonus
- `connected_rooks` - nullable - Connected rooks bonus
- `queen_mobility` - nullable - Queen activity and mobility

#### **5.6 Mobility & Activity**
- `piece_mobility` - nullable - Legal move count for all pieces (slow, accurate)
- `piece_activity` - nullable - Simplified mobility (attacked squares, no move gen)

#### **5.7 Positional Concepts**
- `center_control` - nullable - Control of central squares (e4, d4, e5, d5)
- `space_advantage` - nullable - Territorial control in opponent's half
- `development` - nullable - Piece development (off back rank)
- `move_tempo` - nullable - Tempo evaluation

#### **5.8 Endgame Evaluation**
- `endgame_tables` - nullable - Theoretical endgame knowledge (KQ vs K, etc.)
- `king_mobility` - nullable - King activity in endgame
- `losing_positions` - nullable - Zugzwang evaluation

#### **5.9 Move Safety & Validation**
- `move_safety_checker` - nullable - Pre-move validation
- `repetition_detector` - nullable - Threefold repetition avoidance

---

### **6. BITBOARD EVALUATOR HEURISTICS**
*Low-level bitboard-based calculations*

#### **6.1 Main Scoring**
- `optimized_score` - Optimized bitboard-based score calculation
- `bitboard_tactics` - Tactical detection via bitboards
- `eval_score` - Complete bitboard evaluation score

#### **6.2 Attack Calculations**
- `knight_attacks` - Knight attack bitboard
- `king_attacks` - King attack bitboard
- `w_pawn_attacks` - White pawn attack bitboard
- `b_pawn_attacks` - Black pawn attack bitboard

#### **6.3 Tactical Analysis**
- `tactical_bonus` - Bitboard-based tactical bonus
- `fork_analysis` - Fork detection via bitboards
- `pins_skewers` - Pin/skewer detection via bitboards

#### **6.4 Pawn Structure (Bitboard)**
- `pawn_structure` - Complete pawn structure evaluation
- `passed_pawn_masks` - Passed pawn detection masks
- `passed_count` - Count of passed pawns
- `passed_pawns` - Passed pawn evaluation (bitboard)
- `isolated_pawns` - Isolated pawn evaluation (bitboard)
- `doubled_pawns` - Doubled pawn evaluation (bitboard)
- `backward_pawns` - Backward pawn evaluation (bitboard)
- `connected_pawns` - Connected pawn evaluation (bitboard)
- `pawn_chains` - Pawn chain evaluation (bitboard)
- `pawn_storms` - Pawn storm evaluation (bitboard)
- `connected_passed_pawn` - Connected passed pawn detection
- `is_passed_pawn` - Check if specific pawn is passed
- `is_isolated_pawn` - Check if specific pawn is isolated
- `is_backward_pawn` - Check if specific pawn is backward
- `has_pawn_support` - Check if pawn has adjacent support
- `found_pawn_chains` - Find pawn chains in position
- `pawn_chain_length` - Count length of pawn chain

#### **6.5 Rook Placement (Bitboard)**
- `is_on_open_file` - Rook on open file check
- `is_on_semi_open_file` - Rook on semi-open file check
- `is_on_open_rank` - Rook on open rank check

#### **6.6 King Safety (Bitboard)**
- `king_safety` - Complete king safety evaluation
- `pawn_shelter` - King pawn shield evaluation
- `castling_rights` - Castling rights evaluation
- `enhanced_castle_score` - Enhanced castling evaluation
- `castled` - Has king castled check
- `has_castled` - Has king castled (duplicate?)
- `king_exposure` - King exposure to attacks
- `escape_squares` - King escape square evaluation
- `attack_zone` - Attack zone around king
- `enemy_pawn_storms` - Enemy pawn storm threats
- `king_activity` - King activity in endgame
- `enemy_attacks_near_king` - Count enemy attacks near king
- `is_safe_escape_square` - Check if king escape square is safe
- `is_square_attacked` - Check if square is attacked by enemy
- `pawn_attacks_square` - Check if pawn attacks square
- `king_zone` - Get king zone squares

#### **6.7 Material (Bitboard)**
- `material_count` - Material count via bitboards
- `bishop_pair` - Bishop pair bonus (bitboard)

---

### **7. MOVE SAFETY EVALUATOR**
*Move validation and safety checks (ACTIVE in v18.3)*

- `move_penalty` - Total safety penalty for move
- `hanging_piece_penalty` - Penalty for leaving pieces hanging
- `capture_penalty` - Penalty for unsafe captures
- `safe_moves` - List of moves passing safety checks

**Sub-Components:**
- `_is_piece_hanging` - Check if piece is undefended
- `_get_attackers` - Get pieces attacking square
- `_check_hanging_pieces` - Scan for hanging pieces after move
- `_check_immediate_captures` - Check for immediate recaptures

---

## 📊 Heuristic Statistics

**Total Heuristics Identified:** ~130+

**Categories:**
- Performance Metrics: 12
- Move Ordering: 7
- Position Context: 4
- Fast Evaluator: 10+
- Modular Evaluator: 35+ (20+ placeholders)
- Bitboard Evaluator: 50+
- Move Safety: 8

**Status:**
- ✅ **ACTIVE** in v18.3: ~30 heuristics (Fast Eval, Move Safety, Move Ordering)
- ⚠️ **AVAILABLE** but unused: ~50 heuristics (Bitboard Eval)
- ❌ nullable: ~40 heuristics (Modular Eval)

---

## 🎯 Next Step: Feature Selection

**Your Task:** Review each heuristic and decide:

1. **Include as Feature?** (Yes/No/Maybe)
2. **Feature Type:**
   - Binary (True/False)
   - Count (0, 1, 2, 3...)
   - Category (opening/middlegame/endgame)
   - Normalized Value (-1.0 to +1.0)
3. **Measurement Method:** How to extract from position?

**Example Selections:**

```markdown
## Selected Features

### Material & Position
- ✅ `has_material_advantage` - Binary - material_score > 0
- ✅ `material_advantage_category` - Category - (losing/disadvantage/even/advantage/winning)
- ✅ `game_phase` - Category - (opening/middlegame/endgame)
- ✅ `piece_count` - Count - Total pieces on board

### King Safety
- ✅ `king_has_pawn_shield` - Binary - 2+ pawns in front of king
- ✅ `king_is_castled` - Binary - Castling completed
- ✅ `king_under_attack` - Binary - Enemy pieces attacking king zone
- ✅ `king_escape_squares` - Count - 0-8 safe king moves

### Pawn Structure
- ✅ `has_passed_pawns` - Binary - Any passed pawns exist
- ✅ `passed_pawn_count` - Count - Number of passed pawns
- ✅ `has_doubled_pawns` - Binary - Doubled pawns exist
- ✅ `has_isolated_pawns` - Binary - Isolated pawns exist

### Piece Activity
- ✅ `rooks_on_open_files` - Count - 0-2 rooks on open files
- ✅ `has_bishop_pair` - Binary - Both bishops present
- ✅ `pieces_developed` - Count - Pieces off starting squares
```

---

## 📝 Instructions for Pat

1. **Review this list** - Check if all your heuristics are captured
2. **Mark selections** - Tell me which heuristics to convert to features
3. **Define measurements** - For each selected, specify how to calculate it
4. **Organize by priority** - Which features are most important?

Once you provide selections, I'll create:
1. **Feature extraction functions** for each selected heuristic
2. **Feature schema** for training dataset
3. **Feature extraction script** for PGN/puzzle/self-play data

---

## 💡 Design Principles Reminder

**You said:**
> "I want to turn those heuristics into calculated game observations that simply act as an input state not a prescribed and preweighted score for the position."

**This is correct!** Examples:

**❌ BAD (Prescribed Score):**
```python
rook_bonus = 20 if rook_on_open_file else 0
eval += rook_bonus
```

**✅ GOOD (Observable Feature):**
```python
features['rooks_on_open_files'] = count_rooks_on_open_files(board)
# Let AI learn if this matters and how much
```

---

## 🔧 Feature Specification Template

For each selected feature, we need to define:

| Attribute | Description | Example |
|-----------|-------------|---------|
| **Feature Name** | Unique identifier | `has_bishop_pair` |
| **Data Type** | Binary, Count, Category, Float | `Binary` |
| **Possible Values** | Valid value range | `true, false` |
| **Required?** | Must exist or can be null? | `Required` or `Optional` |
| **Default if Missing** | Value when unavailable | `false` |
| **Source Availability** | Which sources provide this? | `All` or `PGN only` |
| **Measurement Method** | How to calculate | `board.pieces(BISHOP, color).count() == 2` |
| **Dependencies** | Other features needed | `None` or `piece_count` |

---

## 📊 Feature Specification Worksheet

### **Category 1: Core Position Features**
*These should be available from ALL sources (PGN, Puzzles, Self-Play)*

#### F001: Position FEN
```yaml
Feature Name: position_fen
Data Type: String
Possible Values: Valid FEN string
Required: Yes
Default if Missing: ERROR (cannot proceed without FEN)
Source Availability: All sources
Measurement Method: board.fen()
Dependencies: None
Notes: Primary position identifier
```

#### F002: Game Phase
```yaml
Feature Name: game_phase
Data Type: Category
Possible Values: ["opening", "middlegame", "endgame", "late_endgame"]
Required: Yes
Default if Missing: Calculate from piece count
Source Availability: All sources (can calculate if missing)
Measurement Method: |
  phase_score = knights*1 + bishops*1 + rooks*2 + queens*4
  if phase_score >= 20: "opening"
  elif phase_score >= 10: "middlegame"
  elif phase_score >= 4: "endgame"
  else: "late_endgame"
Dependencies: piece_count
Notes: Can always be calculated from position
```

#### F003: Material Balance
```yaml
Feature Name: material_balance
Data Type: Integer
Possible Values: -9000 to +9000 (centipawns)
Required: Yes
Default if Missing: Calculate from position
Source Availability: All sources (can calculate if missing)
Measurement Method: white_material - black_material
Dependencies: material_count
Notes: Positive = White advantage, Negative = Black advantage
```

#### F004: Material Advantage Category
```yaml
Feature Name: material_advantage_category
Data Type: Category
Possible Values: ["losing", "disadvantage", "even", "advantage", "winning"]
Required: Yes
Default if Missing: Calculate from material_balance
Source Availability: All sources
Measurement Method: |
  if material_balance > 300: "winning"
  elif material_balance > 100: "advantage"
  elif material_balance > -100: "even"
  elif material_balance > -300: "disadvantage"
  else: "losing"
Dependencies: material_balance
Notes: Categorical version of material balance
```

#### F005: Total Piece Count
```yaml
Feature Name: piece_count
Data Type: Count
Possible Values: 2 to 32 (kings always present)
Required: Yes
Default if Missing: Calculate from position
Source Availability: All sources
Measurement Method: len(board.piece_map())
Dependencies: None
Notes: Total pieces on board (both colors)
```

---

### **Category 2: King Safety Features**

#### F010: King Has Castled
```yaml
Feature Name: king_has_castled
Data Type: Binary
Possible Values: [true, false]
Required: Optional
Default if Missing: false
Source Availability: All sources (can detect from position)
Measurement Method: Check king on g1/g8 or c1/c8 with no castling rights
Dependencies: None
Notes: May be ambiguous in some positions
```

#### F011: King Has Pawn Shield
```yaml
Feature Name: king_has_pawn_shield
Data Type: Binary
Possible Values: [true, false]
Required: Yes
Default if Missing: Calculate from position
Source Availability: All sources
Measurement Method: Count pawns on 3 squares in front of king >= 2
Dependencies: None
Notes: Protect king from frontal attacks
```

#### F012: King Under Attack
```yaml
Feature Name: king_under_attack
Data Type: Binary
Possible Values: [true, false]
Required: Yes
Default if Missing: Calculate from position
Source Availability: All sources
Measurement Method: board.is_check() or enemy pieces attacking king zone
Dependencies: None
Notes: Immediate danger indicator
```

#### F013: King Escape Squares
```yaml
Feature Name: king_escape_squares
Data Type: Count
Possible Values: 0 to 8
Required: Optional
Default if Missing: null
Source Availability: All sources (expensive to calculate)
Measurement Method: Count legal king moves to safe squares
Dependencies: None
Notes: Tactical safety measure, expensive calculation
```

---

### **Category 3: Pawn Structure Features**

#### F020: Has Passed Pawns
```yaml
Feature Name: has_passed_pawns
Data Type: Binary
Possible Values: [true, false]
Required: Optional
Default if Missing: null
Source Availability: All sources (expensive to calculate)
Measurement Method: Check if any pawn has no enemy pawns ahead or adjacent
Dependencies: None
Notes: Requires per-pawn analysis
```

#### F021: Passed Pawn Count
```yaml
Feature Name: passed_pawn_count
Data Type: Count
Possible Values: 0 to 8
Required: Optional
Default if Missing: null
Source Availability: All sources (if has_passed_pawns calculated)
Measurement Method: Count pawns with no enemy pawns ahead or adjacent
Dependencies: has_passed_pawns
Notes: Only calculate if has_passed_pawns is true
```

#### F022: Has Doubled Pawns
```yaml
Feature Name: has_doubled_pawns
Data Type: Binary
Possible Values: [true, false]
Required: Optional
Default if Missing: null
Source Availability: All sources
Measurement Method: Check if 2+ pawns on same file
Dependencies: None
Notes: Quick file-based check
```

#### F023: Has Isolated Pawns
```yaml
Feature Name: has_isolated_pawns
Data Type: Binary
Possible Values: [true, false]
Required: Optional
Default if Missing: null
Source Availability: All sources
Measurement Method: Check if any pawn has no friendly pawns on adjacent files
Dependencies: None
Notes: Structural weakness indicator
```

---

### **Category 4: Piece Activity Features**

#### F030: Rooks on Open Files
```yaml
Feature Name: rooks_on_open_files
Data Type: Count
Possible Values: 0 to 2
Required: Optional
Default if Missing: null
Source Availability: All sources
Measurement Method: Count rooks on files with no pawns
Dependencies: None
Notes: Active rook placement
```

#### F031: Rooks on Semi-Open Files
```yaml
Feature Name: rooks_on_semi_open_files
Data Type: Count
Possible Values: 0 to 2
Required: Optional
Default if Missing: null
Source Availability: All sources
Measurement Method: Count rooks on files with only enemy pawns
Dependencies: None
Notes: Semi-active rook placement
```

#### F032: Has Bishop Pair
```yaml
Feature Name: has_bishop_pair
Data Type: Binary
Possible Values: [true, false]
Required: Yes
Default if Missing: Calculate from position
Source Availability: All sources
Measurement Method: len(board.pieces(BISHOP, color)) == 2
Dependencies: None
Notes: Quick piece count check
```

#### F033: Pieces Developed
```yaml
Feature Name: pieces_developed
Data Type: Count
Possible Values: 0 to 8 (knights, bishops, queen, rooks)
Required: Optional
Default if Missing: null
Source Availability: All sources (opening phase only)
Measurement Method: Count minor pieces + queen off starting squares
Dependencies: game_phase
Notes: Only relevant in opening phase
```

---

### **Category 5: Tactical Features**

#### F040: Position Has Checks Available
```yaml
Feature Name: has_checks_available
Data Type: Binary
Possible Values: [true, false]
Required: Optional
Default if Missing: null
Source Availability: All sources (expensive)
Measurement Method: Any legal move with board.gives_check()
Dependencies: None
Notes: Tactical opportunity indicator
```

#### F041: Position Has Captures Available
```yaml
Feature Name: has_captures_available
Data Type: Binary
Possible Values: [true, false]
Required: Yes
Default if Missing: Calculate from position
Source Availability: All sources
Measurement Method: Any legal move with board.is_capture()
Dependencies: None
Notes: Material exchange opportunity
```

#### F042: Hanging Pieces Present
```yaml
Feature Name: has_hanging_pieces
Data Type: Binary
Possible Values: [true, false]
Required: Optional
Default if Missing: null
Source Availability: All sources (expensive)
Measurement Method: Any piece attacked without defense
Dependencies: None
Notes: Tactical vulnerability
```

---

### **Category 6: Move Context Features**
*These come from the move played, not just position*

#### F050: Move Type - Capture
```yaml
Feature Name: move_is_capture
Data Type: Binary
Possible Values: [true, false]
Required: Yes (if move data available)
Default if Missing: false
Source Availability: PGN, Puzzles, Self-Play (requires move data)
Measurement Method: board.is_capture(move)
Dependencies: move_played
Notes: From move, not position
```

#### F051: Move Type - Check
```yaml
Feature Name: move_is_check
Data Type: Binary
Possible Values: [true, false]
Required: Yes (if move data available)
Default if Missing: false
Source Availability: PGN, Puzzles, Self-Play (requires move data)
Measurement Method: board.gives_check(move)
Dependencies: move_played
Notes: From move, not position
```

#### F052: Move Type - Castling
```yaml
Feature Name: move_is_castling
Data Type: Binary
Possible Values: [true, false]
Required: Yes (if move data available)
Default if Missing: false
Source Availability: PGN, Puzzles, Self-Play (requires move data)
Measurement Method: board.is_castling(move)
Dependencies: move_played
Notes: From move, not position
```

#### F053: Piece Moved
```yaml
Feature Name: piece_moved
Data Type: Category
Possible Values: ["pawn", "knight", "bishop", "rook", "queen", "king"]
Required: Yes (if move data available)
Default if Missing: null
Source Availability: PGN, Puzzles, Self-Play (requires move data)
Measurement Method: board.piece_at(move.from_square).piece_type
Dependencies: move_played
Notes: From move, not position
```

---

### **Category 7: Source-Specific Features**
*Only available from certain data sources*

#### F060: Puzzle Theme
```yaml
Feature Name: puzzle_theme
Data Type: Category
Possible Values: ["mate", "fork", "pin", "skewer", "discovery", "sacrifice", "endgame", "unknown"]
Required: No (Puzzle data only)
Default if Missing: "unknown"
Source Availability: Puzzles only
Measurement Method: From puzzle database metadata
Dependencies: None
Notes: Not available in PGN or Self-Play sources
```

#### F061: Puzzle Difficulty
```yaml
Feature Name: puzzle_difficulty
Data Type: Category
Possible Values: ["easy", "medium", "hard", "expert"]
Required: No (Puzzle data only)
Default if Missing: null
Source Availability: Puzzles only
Measurement Method: From puzzle rating or database category
Dependencies: None
Notes: Not available in PGN or Self-Play sources
```

#### F062: Opponent ELO
```yaml
Feature Name: opponent_elo
Data Type: Integer
Possible Values: 800 to 3000
Required: No (PGN data only)
Default if Missing: null
Source Availability: PGN games only
Measurement Method: From PGN headers or game metadata
Dependencies: None
Notes: Not available in Puzzles or Self-Play
```

#### F063: Time Control
```yaml
Feature Name: time_control
Data Type: Category
Possible Values: ["bullet", "blitz", "rapid", "classical", "correspondence", "unknown"]
Required: No (PGN data only)
Default if Missing: "unknown"
Source Availability: PGN games only
Measurement Method: Parse time control string from PGN
Dependencies: None
Notes: Not available in Puzzles or Self-Play
```

---

## 🎯 Interactive Feature Selection Process

**Let's work through this together:**

### **Step 1: Review Core Features (F001-F005)**
These are **required from all sources**. Do you agree with:
- Data types (String, Category, Integer, Count)?
- Possible values?
- Measurement methods?

Any changes needed?

### **Step 2: Select King Safety Features**
Which ones do you want to include?
- ✅ F010: King Has Castled
- ✅ F011: King Has Pawn Shield
- ✅ F012: King Under Attack
- ⚠️ F013: King Escape Squares (expensive to calculate)

Your preferences?

### **Step 3: Select Pawn Structure Features**
Which ones are worth the calculation cost?
- ⚠️ F020: Has Passed Pawns (expensive)
- ⚠️ F021: Passed Pawn Count (if F020 calculated)
- ✅ F022: Has Doubled Pawns (cheap)
- ✅ F023: Has Isolated Pawns (cheap)

Your preferences?

### **Step 4: Continue Through Categories**
We'll systematically go through:
- Piece Activity Features (F030-F033)
- Tactical Features (F040-F042)
- Move Context Features (F050-F053)
- Source-Specific Features (F060-F063)

---

## 📋 Feature Selection Tracking

**Selected Features:** (We'll build this list together)
```
✅ REQUIRED from all sources:
- F001: position_fen
- F002: game_phase
- F003: material_balance
- F004: material_advantage_category
- F005: piece_count

⚠️ OPTIONAL (pending your approval):
- [ ] F010: king_has_castled
- [ ] F011: king_has_pawn_shield
- [ ] F012: king_under_attack
- [ ] F013: king_escape_squares
- [ ] ... (continue list)

❌ EXCLUDED (too expensive or not useful):
- (none yet)
```

---

Ready to work through the feature selection? Let's start with **Category 1: Core Position Features (F001-F005)** - do these look good to you?
