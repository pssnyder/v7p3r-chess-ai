# V8.0 Opponent Pool Expansion Plan

**Purpose**: Test and integrate additional UCI opponents for diverse training  
**Status**: Draft - DO NOT execute until current 20-generation training completes  
**Created**: 2026-06-07

---

## Current Opponent Pool (5 Opponents)

| Opponent | Weight | Lichess ELO | Arena ELO | Status |
|----------|--------|-------------|-----------|--------|
| Random v1.0 | 10% | N/A | ~100 | ✅ Working |
| Material v1.0 | 15% | N/A | ~800 | ✅ Working |
| V7P3R v17.1 | 25% | 1487 (v7p3r_bot) | - | ✅ Working |
| V7P3R v17.8 | 25% | 1623 (v7p3r_bot) | - | ✅ Working |
| V7P3R v18.3 | 25% | 1661 (v7p3r_bot) | - | ✅ Working |
| **REMOVED**: Positional v1.0 | 0% | N/A | ~400 | ❌ Illegal moves |

**Critical Notes**:
- **ONLY Lichess bot ELOs are valid** (v7p3r_bot, c0br4_bot, slowmate_bot)
- **Arena ELOs are estimates only** - not reliable for strength comparison
- **v7p3r_bot max ELO**: 1600-1700 (confirmed Lichess play)
- **c0br4_bot max ELO**: 1500-1600 (confirmed Lichess play)
- **slowmate_bot max ELO**: 1200-1300 (confirmed Lichess play)
- Themed opponents (Random, Material, etc.) never played on Lichess
- Positional Opponent v1.0 removed due to frequent illegal moves

---

## Available UCI Engines for Testing

### Python UCI Engines (Easy Integration)

#### VPR Series (V7P3R variants)
- **VPR_v1.0** - `Tournament Engines\VPR\VPR_v1.0\src\vpr_uci.py`
- **VPR_v2.0** - `Tournament Engines\VPR\VPR_v2.0\src\vpr_uci.py`
- **VPR_v3.0** - `Tournament Engines\VPR\VPR_v3.0\src\vpr_uci.py`
- **VPR_v4.0** - `Tournament Engines\VPR\VPR_v4.0\src\vpr_uci.py`
- **VPR_v5.0** - `Tournament Engines\VPR\VPR_v5.0\src\vpr_uci.py`
- **VPR_v6.0** - `Tournament Engines\VPR\VPR_v6.0\src\vpr_uci.py`
- **VPR_v7.0** - `Tournament Engines\VPR\VPR_v7.0\src\vpr_uci.py`
- **VPR_v8.1** - `Tournament Engines\VPR\VPR_v8.1\src\vpr_uci.py`
- **VPR_v9.0** - `Tournament Engines\VPR\VPR_v9.0\src\vpr_uci.py`
- Lichess ELO: N/A (never deployed as bot)
- Arena ELO: Unknown, unreliable
- Status: **UNKNOWN FUNCTIONALITY** - may not work, needs testing
- **Caution**: User suspects VPR may not be functional

#### SlowMate
- **SlowMate_v3.2** - `Tournament Engines\SlowMate\SlowMate_v3.2\src\uci_main.py`
- **Lichess ELO**: 1200-1300 max (slowmate_bot confirmed)
- Arena ELO: N/A
- Status: **HIGH PRIORITY** (proven working, Lichess-tested, mate specialist)

#### V7P3R Additional Versions
- **v12.6** - `Tournament Engines\V7P3R\V7P3R_v12.6\src\v7p3r_uci.py` (Lichess: 1544)
- **v14.1** - Available if needed (Lichess: 1488)
- **v17.2** - `Tournament Engines\V7P3R\V7P3R_v17.2\src\v7p3r_uci.py` (Lichess: 1614)
- **v17.4** - `Tournament Engines\V7P3R\V7P3R_v17.4\src\v7p3r_uci.py` (Lichess: 1606, endgame issues)
- **v17.7** - `Tournament Engines\V7P3R\V7P3R_v17.7\src\v7p3r_uci.py` (Lichess: 1623)
- **v18.0** - `Tournament Engines\V7P3R\V7P3R_v18.0\src\v7p3r_uci.py` (Lichess: 1654)
- **v18.1** - `Tournament Engines\V7P3R\V7P3R_v18.1\src\v7p3r_uci.py`
- **v18.4** - `Tournament Engines\V7P3R\V7P3R_v18.4\src\v7p3r_uci.py` (Lichess: 1633→1614, rolled back)
- Status: **Low priority** (similar to existing v7p3r opponents)

### Compiled Executables (C# engines, requires testing)

#### C0BR4 Series
- **C0BR4_v2.9** - `Tournament Engines\C0BR4\C0BR4_ARCHIVE\v2.9_DEPLOYED\C0BR4_v2.9.exe`
- **C0BR4_v3.0** - `Tournament Engines\C0BR4\C0BR4_ARCHIVE\v3.0\C0BR4_v3.0.exe`
- **C0BR4_v3.1** - `Tournament Engines\C0BR4\C0BR4_v3.1\C0BR4_v3.1.exe`
- **Lichess ELO**: 1500-1600 max (c0br4_bot confirmed)
- Arena ELO: N/A
- Status: **HIGHEST PRIORITY** (proven working, Lichess-tested, C# bitboard engine
- **C0BR4_v3.4** - Check for executable
- Estimated ELO: Unknown (bitboard engine with alpha-beta, likely 1500-1800 range)
- Status: **High priority for testing** (different architecture, C# vs Python)

#### Copycat Series
- **Copycat_v1.0** - Check for UCI implementation
- **Copycat_v2.0** - Check for UCI implementation
- Estimated ELO: Unknown
- Status: **Low priority** (unknown capability)

### Downloaded Engines
- Lichess ELO: Unknown (not deployed as bot)
- Arena ELO: ~1800-2000 (unreliable estimate)
- Status: **Low priority** (no Lichess validation, may be too strong
- Status: **Medium priority** (strong opponent for diversity)

---

## Testing Protocol (AFTER Current Training)

### Phase 1: Pre-Flight Testing (1-2 hours)

**Goal**: Verify UCI protocol compatibility without disrupting training

**Script**: `v8.0/src/test_new_opponent.py` (to be created)

```python
"""
Test new opponent UCI compatibility before adding to pool
Usage: python test_new_opponent.py --path "path/to/engine" --name "Engine Name"
"""
```

**Tests**:
1. **UCI Handshake**: Send `uci`, expect `id name`, `id author`, `uciok`
2. **Ready Check**: Send `isready`, expect `readyok`
3. **New Game**: Send `ucinewgame`, `isready`, expect `readyok`
4. **Position Setup**: Send `position startpos`, `isready`, expect `readyok`
5. **Move Request**: Send `go movetime 3000`, expect `bestmove` within timeout
6. **Position with Moves**: Send `position startpos moves e2e4`, `go movetime 3000`, expect valid `bestmove`
7. **FEN Position**: Send `position fen <fen>`, `go movetime 3000`, expect valid `bestmove`
8. **10-Move Game**: Play 10 moves alternating, verify no illegal moves
9. **Cleanup**: Send `quit`, verify process terminates

**Pass Criteria**: All tests pass, 0 illegal moves in 10-move game

### Phase 2: Integration Testing (30 minutes)

**Goal**: Test opponent in opponent pool with 10 sample games

**Script**: `v8.0/src/test_opponent_integration.py` (to be created)

```python
"""
Play 10 games against new opponent using v8.0 Gen 20 network
Usage: python test_opponent_integration.py --opponent "C0BR4 v3.1"
"""
```

**Tests**:
1. Load latest v8.0 network (Gen 20)
2. Add test opponent to temporary pool (100% weight)
3. Play 10 games (5 as white, 5 as black)
4. Record: wins, draws, losses, illegal moves, avg moves/game, avg time/move
5. Report: game outcomes, termination reasons, any errors

**Pass Criteria**: 
- 0-1 illegal moves across 10 games (<10% rate)
- All games complete (no hangs or crashes)
- Avg time/move <5 seconds

### Phase 3: Production Pool Update

**Goal**: Add tested opponent to main opponent pool

**File**: `v8.0/src/opponent_manager.py`

**Steps**:
1. Add `OpponentConfig` entry with:
   - Accurate name from testing
   - Correct path (absolute)
   - Weight (10-25% based on strength)
   - Estimated ELO (from testing results + Lichess data if available)
   - Style category
2. Adjust weights of existing opponents to sum to 1.0
3. Test with `python opponent_manager.py` (built-in test mode)
4. Run 100-game training generation with new pool
5. Compare results vs previous generation

---

## ReIGHEST Priority (Proven Lichess Bots - Test First)

1. **C0BR4_v3.1** - Latest stable C0BR4, C# compiled
   - Path: `Tournament Engines\C0BR4\C0BR4_v3.1\C0BR4_v3.1.exe`
   - **Lichess ELO**: 1500-1600 max (c0br4_bot confirmed)
   - Rationale: **Proven working, Lichess-tested, C# architecture diversity**

2. **SlowMate_v3.2** - Mate-focused specialist
   - Path: `Tournament Engines\SlowMate\SlowMate_v3.2\src\uci_main.py`
   - **Lichess ELO**: 1200-1300 max (slowmate_bot confirmed)
   - Rationale: **Proven working, Lichess-tested, fills gap between Material and v7p3r**

### Medium Priority (Untested - Proceed with Caution)

3. **VPR_v8.1** - Latest VPR variant (IF IT WORKS)
   - Path: `Tournament Engines\VPR\VPR_v8.1\src\vpr_uci.py`
   - Lichess ELO: N/A (never deployed)
   - Arena ELO: Unknown
   - Rationale: **User suspects may not work - test carefully**
   - **Caution**: May be non-functional, verify UCI protocol first

### Low Priority (Not Worth Testing Now)

4. **V7P3R additional versions** - Already have 3 v7p3r opponents
   - v12.6 (1544), v17.2 (1614), v18.0 (1654), etc.
   - Rationale: Diminishing returns, pool already v7p3r-heavy

5. **Sunfish** - No Lichess validation
   - Arena ELO estimates unreliable
   - May be too strong or too weak
   - Rationale: Unknown strength, untested in real play
   - Lichess: 1544 ELO
   - Rationale: Already have 3 v7p3r variants, diminishing returns

---

## Updated Opponent Pool Proposals

### Proposal A: Balanced Strength Curve (6 opponents)

| Opponent | Estimated ELO | Weight | Rationale |
|----------|----Proven Lichess Bots (7 opponents) ⭐ RECOMMENDED

| Opponent | Lichess ELO | Weight | Rationale |
|----------|-------------|--------|-----------|
| Random v1.0 | N/A (~100) | 5% | Baseline weak |
| Material v1.0 | N/A (~800) | 10% | Tactical baseline |
| **SlowMate v3.2** | **1200-1300** | **20%** | **Lichess-proven, mate specialist** |
| V7P3R v17.1 | 1487 | 15% | v7p3r lower |
| **C0BR4 v3.1** | **1500-1600** | **20%** | **Lichess-proven, C# architecture** |
| V7P3R v17.8 | 1623 | 15% | v7p3r mid |
| V7P3R v18.3 | 1661 | 15% | v7p3r upper |

**Total**: 7 opponents, ALL with known strength (3 from Lichess, 2 from Arena estimates)  
**Strength Curve**: ~100 → ~800 → 1200-1300 → 1487 → 1500-1600 → 1623 → 1661  
**Architecture Mix**: 6 Python + 1 C# (C0BR4)  
**Style Mix**: Random, Tactical, Mate-focused, Balanced, Aggressive  

### Proposal B: Conservative (6 opponents)

| Opponent | Lichess ELO | Weight | Rationale |
|----------|-------------|--------|-----------|
| Random v1.0 | N/A (~100) | 10% | Baseline weak |
| Material v1.0 | N/A (~800) | 15% | Tactical baseline |
| **SlowMate v3.2** | **1200-1300** | **20%** | Lichess-proven |
| V7P3R v17.8 | 1623 | 25% | v7p3r mid (increased weight) |
| V7P3R v18.3 | 1661 | 30% | v7p3r upper (increased weight) |
| **SKIP**: C0BR4 | - | 0% | Save for later expansion |

**Total**: 6 opponents, minimal change from current 5  
**Rationale**: Add only SlowMate for now, test C0BR4 separately  

### Proposal C: All Three Proven Bots (8 opponents)

| Opponent | Lichess ELO | Weight | Rationale |
|----------|-------------|--------|-----------|
| Random v1.0 | N/A (~100) | 5% | Baseline weak |
| Material v1.0 | N/A (~800) | 10% | Tactical baseline |
| **SlowMate v3.2** | **1200-1300** | **15%** | Lichess-proven |
| V7P3R v17.1 | 1487 | 15% | v7p3r lower |
| **C0BR4 v3.1** | **1500-1600** | **15%** | Lichess-proven |
| V7P3R v17.8 | 1623 | 15% | v7p3r mid |
| V7P3R v18.3 | 1661 | 15% | v7p3r upper |
| **V7P3R v12.6** | 1544 | 10% | Historical diversity |

**Total**: 8 opponents, maximum Lichess-validated diversity  
**Rationale**: Add v12.6 for additional v7p3r datapoint between 1487 and 1623
## Post-Training Action Items

**After 20-generation training completes**:

1. ✅ Analyze Gen 20 results (win rates, patterns learned)
2. ✅ Creat**C0BR4_v3.1** (HIGHEST Priority - Lichess 1500-1600)
5. ✅ Test **SlowMate_v3.2** (HIGHEST Priority - Lichess 1200-1300)
6. 🔍 Test **VPR_v8.1** IF curious (may not work, low priority)
7. ⏳ Choose pool proposal (A=RECOMMENDED, B=Conservative, C=Maximum)
8. ⏳ Update `opponent_manager.py` with C0BR4 and SlowMate
9. ⏳ Run 5-generation validation training (500 games)
10. ⏳ Compare Gen 20 vs Gen 25 performance
9. ⏳ Run 5-generation validation training (500 games)
10. ⏳ Compare Gen 20 (current pool) vs Gen 25 (expanded pool)

---

## Notes

- **DO NOT modify opponent pool while current training is running**
- All testing must be done in isolation (separate scripts)
- Document actual ELO based on win rates vs known opponents
- Keep Random v1.0 for baseline comparison
- Prefer Python engines for easier debugging
- C# engines (C0BR4) require Windows execution
- Sunfish may be too strong - test last
- Update this document with test results as they complete

---

## Current Training Status

- **Generation**: X/20 (in progress)
- **Training Terminal**: [ID from current session]
- **Monitor Terminal**: [ID from current session]
- **Expected Completion**: ~3-4 hours from start
- **Do Not Disturb**: Let complete before testing new opponents
