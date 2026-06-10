# V7P3R Human Study Guide
## AI Training Sources - "Superhuman v7p3r" Knowledge Base

**Purpose**: Define what positions, patterns, and playing styles the AI should learn to emulate "superhuman you" - the chess player you wish you could be.

**Philosophy**: Train the AI on YOUR actual tactical style - aggressive, sacrificial, chaos-creating chess inspired by Mikhail Tal. Complicate the opponent's perspective while maintaining a deterministic path through the complexity. "Deeper into the forest with only one path out."

**Performance Roadmap**:
- **Phase 1 (1200 ELO)**: Your current play style + prevent one-move blunders
- **Phase 2 (1600 ELO)**: Add sequential strategy (linking tactics into plans)
- **Phase 3 (1800 ELO)**: Integrate grandmaster knowledge + aspirational repertoire
- **Ultimate Goal (2400 ELO)**: Tal-level chaos mastery with perfect calculation

---

## 1. Human Games - Impressive Wins
*Games where you played well and want the AI to learn from*

### Wins to Study:
- **lichess_pgn_2021.02.04_v7p3r_vs_jfft.ZVEL8xeM.pgn** (WIN)
  - Opening: Horwitz Defense (London System approach)
  - Key lesson: Quick checkmate with Qxh7# after opponent weakened kingside
  - Result: 1-0 (White)
  
- **lichess_pgn_2021.01.16_HenrikofSweden_vs_v7p3r.0sYPB1Tj.pgn** (WIN)
  - Opening: Caro-Kann Defense
  - Key lesson: Sacrificed material to deliver Qxh2# checkmate
  - Result: 0-1 (Black)
  - Rating: 1198 (gained +16)

- **lichess_pgn_2025.10.21_v7p3r_vs_lichess_AI_level_3.1mofC31p.pgn** (WIN)
  - Opening: Chess960 variant
  - Key lesson: Endgame mastery, converted pawn to queen, delivered checkmate
  - Result: 1-0 (White)

- **lichess_pgn_2025.11.30_v7p3r_bot_vs_v7p3r.9i883UOF.pgn** (WIN vs ENGINE)
  - Opening: Queen's Gambit Accepted: Mannheim Variation
  - Key lesson: Beat V7P3R engine (1614 rating), gained +68 rating points
  - Result: 0-1 (Black, human v7p3r won)
  - **HIGHLY IMPORTANT**: This shows human intuition beating engine calculation

- **lichess_pgn_2025.12.05_v7p3r_vs_slowmate_bot.QJTFBLrr.pgn** (WIN)
  - Opening: Alekhine Defense
  - Result: 1-0 (White)

- **lichess_pgn_2026.04.19_v7p3r_vs_slowmate_bot.kGjzAOSx.pgn** (WIN)
  - Opening: Hippopotamus Defense
  - Result: 1-0 (White)
  - Rating: Gained +82 points (1031 → 1113)

- **v7p3r vs Coach1200_20251021.pgn** (WIN vs Coach)
  - Platform: Chess.com
  - Key lesson: Delivered Ra6# checkmate
  - Result: 1-0 (White)

- **v7p3r vs Bob the Cat (1350) 20251127.pgn** (WIN)
  - Platform: Chess.com vs 1350-rated bot
  - Result: 1-0 (White)

- **Florence_bot vs v7p3r chesscom_20251101.pgn** (WIN)
  - Key lesson: Delivered Re1# checkmate
  - Result: 0-1 (Black)

- **Coach1200_vs_v7p3r_20251023.pgn** (WIN)
  - Opening: Caro-Kann approach
  - Result: 0-1 (Black)

- **bentnoze and wartface bot vs v7p3r 20251107.pgn** (WIN)
  - Platform: Chess.com
  - Result: 0-1 (Black)

---

## 2. Human Games - Learning Experiences
*All human games for general pattern recognition*

### Draws (Positional Understanding):
- **lichess_pgn_2025.12.06_v7p3r_bot_vs_v7p3r.e68K458O.pgn** (DRAW)
  - Opening: Queen's Gambit Declined: Queen's Knight Variation
  - Key lesson: Held draw against engine
  - Result: 1/2-1/2

- **Coach1600 vs v7p3r 20251101.pgn** (DRAW)
  - Opening: Caro-Kann Defense
  - Result: 1/2-1/2

### Losses (What to Avoid):
- **lichess_pgn_2025.12.02_v7p3r_bot_vs_v7p3r.7BZt4kAK.pgn** (LOSS)
  - Opening: Queen's Gambit Declined: Marshall Defense
  - Key lesson: Learn what went wrong
  - Result: 1-0 (Engine won)

- **lichess_pgn_2025.10.29_v7p3r_vs_v7p3r_bot.SRG12jXg.pgn** (LOSS)
  - Opening: French Defense: Normal Variation
  - Result: 0-1 (Engine won)

- **v7p3r vs Bob the Cat 1350 20251126_0-1.pgn** (LOSS)
  - Result: 0-1

- **Abominable-Chessman-BOT_vs_v7p3r_2025.12.21.pgn** (LOSS)
  - Opening: Caro-Kann Defense: Two Knights Attack
  - Result: 1-0 (Engine won by resignation)

### Additional Games:
- **v7p3r_vs_pat314_2021.01.02.pgn**
- **Coach-Levy-Bot vs v7p3r 20251106.pgn** (DRAW)
- **lichess_pgn_2021.02.07_Ethan-C_vs_v7p3r.2yjfqtls.pgn**

**File Location**: `E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\game_records\v7p3r Human\`

---

## 3. V7P3R Engine Wins to Study
*Games where V7P3R engine demonstrated strong play*

### Source Directory:
`E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\game_records\Engine Battle 202512\`

### Filtering Criteria:
- **Result**: V7P3R won (1-0 as White or 0-1 as Black)
- **Versions to prioritize**:
  - v18.3 (current stable)
  - v17.8 (rapid improvement)
  - v17.7 (4-day stable deployment)
  - v17.1 (proven stable baseline)

### Extraction Method:
```python
# Filter for V7P3R wins only
if ("V7P3R" in game.headers["White"] and result == "1-0") or \
   ("V7P3R" in game.headers["Black"] and result == "0-1"):
    # Extract positions from winning games
```

---

## 4. Preferred Opening Repertoire
*Openings you want to master as "superhuman you"*

### Current Mastery (Weighted 3.0x - Your ACTUAL Style):

**As White - Open Aggressive Play**:
- **1.e4 King's Pawn Opening** (YOUR TRUE SIGNATURE)
  - Open positions, tactical opportunities
  - Italian Game, Scotch, King's Gambit variations
  - Source: Your Chess.com games (1.e4 in almost every game)
  
- **Bxf7+ Sacrificial Patterns**
  - "Fried Liver" style king hunts
  - Example: Monteiro97 game - `6. Bxf7+ Ke7 7. Ne6`
  - Sacrifice bishop to expose king, coordinate pieces for mate
  
- **Vienna Game**
  - Aggressive but sound
  - Fits your tactical style
  - PGN Files: `pgn_data_openings/vienna*.pgn`

**As Black - Flexible Tactical**:
- **Caro-Kann Defense** (SECONDARY SIGNATURE)
  - Solid opening → tactical middlegame transition
  - When you want solidity but maintain tactical chances
  - PGN Files: `pgn_data_openings/caro*.pgn`

- **1...e5 Open Games** (PRIMARY AS BLACK)
  - Accept tactical complications
  - Two Knights Defense, Italian Game as Black
  - Fits your aggressive style

- **French Defense**
  - Sharp counterplay
  - PGN Files: `pgn_data_openings/french*.pgn`

### Aspirational Openings (Weighted 2.0x):
*"Advanced repertoire to accelerate my learning"*

- **King's Gambit** (as White)
  - Ultimate Tal-style aggression
  - Sacrifice pawn for rapid development and attack
  - PGN Files: `pgn_data_openings/*gambit*.pgn`

- **Sicilian Defense** (as Black)
  - Sharp, tactical, asymmetric
  - Fits your complexity-creation style
  - PGN Files: `pgn_data_openings/sicilian*.pgn`

- **Queen's Gambit Accepted** (understanding, not primary)
  - Know it to face it, not necessarily to play
  - Tactical variations preferred over positional ones
  - PGN Files: `pgn_data_openings/*queen*gambit*.pgn`

### Opening Directory:
`E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\pgn_training_data\pgn_data_openings\`

**Total Files**: 120 PGN files

---

## 5. Tactical Patterns from YOUR Play
*"Smoke and mirrors" - Chaos creation with deterministic goals*

### Your Signature Tactics (Weighted 3.0x):

**Bxf7+ King Hunt Sequences**:
- Sacrifice bishop on f7 to expose enemy king
- Coordinate queen + knight for relentless attack
- Example from Monteiro97 win: `6. Bxf7+ Ke7 7. Ne6 Qd7 8. Qg4`
- Goal: Force king into open, deliver checkmate before material matters

**Piece Sacrifice for Attack**:
- Willing to sacrifice material for initiative
- Create complexity opponent can't calculate through
- You see the deterministic path; opponent sees chaos

**Quick Checkmate Patterns**:
- Several wins in under 20 moves
- Recognize mating nets early
- Examples: Back rank mates, bishop+queen coordination

**Your Weakness to AVOID** (Anti-pattern learning):
- **One-move blunders in tactical positions**
  - Train AI to do multi-move lookahead before committing
  - Verify no hanging pieces before playing move
  - This is THE difference between 1000 and 1200 ELO

---

## 6. Grandmaster Tactical Patterns to Learn

### Source:
`E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\pgn_training_data\csv_data_puzzles\`

**Size**: 861.5 MB (thousands of tactical puzzles)

### Pattern Categories (from Lichess puzzles):
- Pins
- Forks
- Skewers
- Discovered attacks
- Double attacks
- Back rank mates
- Smothered mates
- Deflection
- Decoy
- Zugzwang

---

## 7. Endgame Mastery

### Source:
`E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\pgn_training_data\pgn_data_endgames\`

**Files**: 2 PGN files (endgame theory)

### Priority Endgames:
- Rook + Bishop vs King (from v17.4 failure - must master this)
- Queen endgames (converting advantage)
- Pawn endgames (opposition, passed pawns)
- Knight vs Pawns
- Rook endgames

---

## 8. Features & Heuristics YOU Care About
*Chess concepts you naturally look for in a position*

### Positional Features (F001-F092):

#### **King Safety** (High Priority):
- F010: `white_king_castled` / `black_king_castled`
  - **Your preference**: ✅ YES (Your king) - Castle early for safety
  - **Opponent**: ❌ PREVENT - Stop opponent from castling when attacking
- F011: `white_king_has_pawn_shield` / `black_king_has_pawn_shield`
  - **Your preference**: ✅ YES (Your king) - Keep your king safe
  - **Opponent**: ❌ TARGET - Destroy opponent's pawn shield (h6, Bxf7+)
- F012: `white_king_under_attack` / `black_king_under_attack`
  - **Your preference**: ❌ AVOID (Your king) - Minimize your king danger
  - **Opponent**: ✅ CREATE (Opponent king) - Expose and hunt opponent's king

#### **Material & Piece Activity**:
- F003: `material_balance_cp`
  - **Your preference**: Positive balance preferred, but willing to sacrifice for attack
- F032: `bishop_pair`
  - **Your preference**: ✅ YES - London System keeps bishops active
- F005: `total_piece_count`
  - **Your preference**: Endgame competence (you convert wins in endgames)

#### **Pawn Structure**:
- F020: `white_has_passed_pawns` / `black_has_passed_pawns`
  - **Your preference**: ✅ YES - You understand pawn endgames
- F021: `white_isolated_pawns` / `black_isolated_pawns`
  - **Your preference**: ❌ AVOID - Caro-Kann aims for solid structure
- F022: `white_doubled_pawns` / `black_doubled_pawns`
  - **Your preference**: ❌ AVOID - Weak pawn structure

#### **Tactical Patterns** (YOUR STRENGTH):
- F040: `white_has_pinned_pieces` / `black_has_pinned_pieces`
  - **Your preference**: ✅ Create pins (offense), ❌ avoid being pinned (defense)
- F041: `white_has_forking_opportunities` / `black_has_forking_opportunities`
  - **Your preference**: ✅ YES - Knight forks, queen forks
- F042: `sacrifice_available` (NEW - should add this feature)
  - **Your preference**: ✅ STRONGLY YES - Bxf7+, piece sacs for attack
  - This is your SIGNATURE pattern
- F050: `white_has_checkmate_threat` / `black_has_checkmate_threat`
  - **Your preference**: ✅ CRITICAL - Many wins by quick mate
  - If you see mate, play for mate (ignore material)

#### **Center Control**:
- F070: `white_controls_center` / `black_controls_center`
  - **Your preference**: ✅ YES - Caro-Kann fights for center

#### **Development**:
- F002: `game_phase` (opening/middlegame/endgame)
  - **Your preference**: Fast development in opening, then strategic middlegame

---

## 9. Feature Preference Weighting
*How much each feature matters to "superhuman you"*

### Tier 0 - ABSOLUTE PRIORITY (Weight: 5.0):
- **Prevent one-move blunders** (hanging pieces, undefended pieces)
  - This is THE gap between 1000 and 1200 ELO
  - Multi-move lookahead before committing
  - "Don't hang pieces" > everything else

### Tier 1 - Critical (Weight: 3.0):
- **Checkmate threats** (you win MANY games this way)
- **King hunt opportunities** (exposed enemy king = attack)
- **Sacrifice for attack** (Bxf7+, piece sacs when king exposed)
- King safety for YOUR pieces (castled + pawn shield)

### Tier 2 - Important (Weight: 2.0):
- **Piece activity** (active pieces > passive pieces)
- **Development speed** (open games require fast development)
- **Initiative** (attacking > defending when kings exposed)
- Passed pawns (endgame conversion)

### Tier 3 - Moderate (Weight: 1.5):
- Knight activity
- Rook placement
- Pawn structure quality

### Tier 4 - Nice to Have (Weight: 1.0):
- Outposts
- Piece mobility
- Space advantage

---

## 10. Data Source Summary

### Primary Sources:
| Source | Path | Size | Weight | Purpose |
|--------|------|------|--------|---------|
| **Human Wins** | `v7p3r Human/` | 20+ games | 3.0x | Your best tactical play |
| **Human All Games** | `v7p3r_20250530.pgn` | 100+ games | 2.0x | Unbiased style discovery |
| **Preferred Openings** | `pgn_data_openings/` | e4, Vienna, Caro | 3.0x | Tactical openings |
| **Aspirational Openings** | `pgn_data_openings/` | ~30 files | 2.0x | Queen's Gambit, Sicilian |
| **V7P3R Wins** | `Engine Battle 202512/` | ~250 games | 1.5x | Engine mastery |
| **Tactics** | `csv_data_puzzles/` | 861.5 MB | 1.0x | Pattern recognition |
| **Endgames** | `pgn_data_endgames/` | 2 files | 1.0x | Conversion technique |
| **Grandmaster Games** | TBD | TBD | 0.5x | General patterns |

### Target Dataset Size:
- **Wins**: ~270 games (20 human + 250 engine)
- **Openings**: ~150 PGN files
- **Tactics**: Thousands of puzzles
- **Total positions**: ~50,000-100,000 curated positions

---

## 11. Training Philosophy - "Tal's Forest"

### Core Philosophy - "Deeper into the forest with only one path out":

**Mikhail Tal Inspiration**:
> "You must take your opponent into a deep dark forest where 2+2=5, and the path leading out is only wide enough for one." - Mikhail Tal

**Your Implementation**:
1. **Create complexity** - Open positions, tactical complications, sacrifices
2. **Opponent drowns in calculation** - Too many variations to calculate
3. **You have deterministic path** - Feature-based intuition guides you through
4. **Emerge with advantage** - While opponent still calculating, you're already winning

### What the AI Should Learn:
1. **"This is how I actually play"** - Aggressive tactics from your games
2. **"This is what works for me"** - Bxf7+ patterns, piece sacrifices
3. **"This is where I'm going"** - Deterministic goals through chaotic positions
4. **"This prevents my weakness"** - Multi-move verification (no blunders)

### What the AI Should NOT Learn:
- Positional closed games (not your style)
- Passive defensive play (you're an attacker)
- Material-first thinking (you sacrifice for initiative)
- Computer-style calculation (you use feature-based intuition)

### Stage 1 Goal:
**Build chaos navigation model**: "How similar is this position to the chaotic, tactical positions v7p3r creates and thrives in?"

**Key Questions**:
- Is the opponent's king exposed? (✅ Good for v7p3r)
- Are there tactical complications? (✅ Good for v7p3r)  
- Is there a clear path to checkmate? (✅ Good for v7p3r)
- Can I sacrifice for attack? (✅ Good for v7p3r)
- Will I hang a piece? (❌ Bad - prevent blunders)

### Stage 2 Goal:
**Deterministic navigation through chaos**: "Which move creates the complexity I can navigate, but opponent cannot?"

**Move Selection Logic**:
1. Check for checkmate (if yes, play it)
2. Check for blunders (if yes, don't play it)
3. Check for sacrifices that expose opponent king (weighted highly)
4. Select move that increases position similarity to "v7p3r tactical wins"
5. Complicate when ahead in tactical understanding, simplify when opponent adapting

---

## 12. Performance Milestones

### Phase 1: Prevent Blunders (Target: 1200 ELO)
**Implementation**:
- Multi-move lookahead (minimum 2 moves)
- Hanging piece detection
- Undefended piece verification
- "Is this move safe?" check before playing

**Success Metric**: <5% games lost to one-move blunders

### Phase 2: Sequential Strategy (Target: 1600 ELO)
**Implementation**:
- Link tactics into multi-move sequences
- "If I sacrifice here, then move here, then checkmate"
- Combination recognition (sac + sac + mate)
- Plan 3-5 moves ahead in forcing sequences

**Success Metric**: Win 70%+ games with tactical sequences

### Phase 3: Grandmaster Integration (Target: 1800 ELO)
**Implementation**:
- Add GM games with similar tactical style (Tal, Shirov, Kasparov)
- Advanced opening theory (King's Gambit, Open Sicilian)
- Endgame technique for converting advantages
- Pattern recognition from 100,000+ GM positions

**Success Metric**: Beat 1600-rated engines consistently

### Phase 4: Superhuman Mastery (Target: 2400 ELO)
**Implementation**:
- Perfect tactical calculation (no blunders)
- Deep strategic understanding
- Opening preparation to move 15+
- Endgame tablebase integration
- Chaos creation at will

**Success Metric**: Competitive with strong club players, beat weaker engines

---

## 13. Next Steps - Data Collection

### TODO: Add to This Document
- [ ] Review feature list F001-F092 and mark preferences (✅/❌/⚠️)
- [ ] Add Mikhail Tal games to training set (GM tactical inspiration)
- [ ] Add Shirov games (modern Tal-style player)
- [ ] Define "bad positions" (passive, closed, no tactics)
- [ ] Create "blunder prevention" test suite
- [ ] Add Chess.com full game history (v7p3r_20250530.pgn)
- [ ] Extract all Bxf7+ patterns from your games
- [ ] Catalog king hunt sequences that worked

### TODO: Implementation
- [ ] Create HumanGamesLoader for v7p3r_20250530.pgn
- [ ] Implement weighted mixing: Tactical games 3.0x, Learning games 1.0x
- [ ] Filter for games with Bxf7+ patterns (extract and weight 5.0x)
- [ ] Add "blunder detection" feature to prevent hanging pieces
- [ ] Build chaos navigation model (Stage 1): "Is this v7p3r-style chaos?"
- [ ] Test: "Would v7p3r create this position?" "Would v7p3r sacrifice here?"
- [ ] Create Tal games loader (GM tactical inspiration)

---

## Version History
- **v1.0** (2026-05-26): Initial study guide created with 20 human games cataloged
- **v2.0** (2026-05-26): MAJOR REVISION - Discovered actual tactical style via Chess.com history
  - Removed London System bias (doesn't match actual play)
  - Added Tal-inspired "chaos with deterministic path" philosophy
  - Emphasized 1.e4, Bxf7+ patterns, sacrificial play
  - Added performance roadmap: 1200 → 1600 → 1800 → 2400
  - Shifted from "aspirational positional" to "actual tactical aggression"
  - Key insight: User plays like Tal (complicate for opponent, navigate deterministically)
