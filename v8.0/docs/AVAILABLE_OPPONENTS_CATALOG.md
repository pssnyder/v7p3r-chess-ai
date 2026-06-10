# Available Training Opponents Catalog

Complete inventory of engines in `Tournament Engines/` folder for V8.0 curriculum design.

**Legend:**
- **Strength Range**: AI estimate based on features/complexity
- **Reference ELO**: Your chosen baseline for training
- **Status**: `UNTESTED` | `FUNCTIONAL` | `BROKEN` | `VERIFIED`
- **Priority**: Training priority based on strength gap needs

---

## WEAKEST TIER (Beginner Learning - ELO 100-500)

| Engine | Version | Path | Strength Range | Reference ELO | Status | Notes |
|--------|---------|------|----------------|---------------|--------|-------|
| **RandomOpponent** | v1.0 | `Opponents/RandomOpponent_v1.0/` | 50-150 | 100 | FUNCTIONAL | Pure random legal moves, instant responses |
| **V7P3R** | v4.1 | `V7P3R_ARCHIVE/V7P3R_v4.1.exe` | 150-300 | 200 | UNTESTED | Very early version, minimal evaluation |
| **V7P3R** | v4.2 | `V7P3R_ARCHIVE/V7P3R_v4.2.exe` | 150-300 | 220 | UNTESTED | Early iteration |
| **V7P3R** | v4.3 | `V7P3R_ARCHIVE/V7P3R_v4.3.exe` | 150-300 | 240 | UNTESTED | Early iteration |
| **V7P3R** | v5.0 | `V7P3R_ARCHIVE/V7P3R_v5.0.exe` | 200-350 | 260 | UNTESTED | Basic minimax added? |
| **V7P3R** | v5.1 | `V7P3R_ARCHIVE/V7P3R_v5.1.exe` | 200-350 | 280 | UNTESTED | Refinements |
| **SlowMate** | v0.0.0 | `SlowMate/SlowMate_v0.0.0.exe` | 100-250 | 150 | UNTESTED | Earliest SlowMate, mate-focused but weak |
| **SlowMate** | v0.0.1 | `SlowMate/SlowMate_v0.0.1.exe` | 100-250 | 170 | UNTESTED | Early iteration |
| **SlowMate** | v0.1.0 | `SlowMate/SlowMate_v0.1.0.exe` | 150-300 | 200 | UNTESTED | First minor release |
| **Cecilia** | v0.1.0 | `Cecilia/Cecilia_v0.1.0.exe` | 200-400 | 300 | UNTESTED | Experimental engine, unknown strength |

---

## WEAK TIER (Basic Tactics - ELO 500-900)

| Engine | Version | Path | Strength Range | Reference ELO | Status | Notes |
|--------|---------|------|----------------|---------------|--------|-------|
| **V7P3R** | v6.0 | `V7P3R_ARCHIVE/V7P3R_v6.0.exe` | 300-500 | 400 | UNTESTED | Version 6 series |
| **V7P3R** | v6.1 | `V7P3R_ARCHIVE/V7P3R_v6.1.exe` | 300-500 | 420 | UNTESTED | Refinements |
| **V7P3R** | v7.0 | `V7P3R_ARCHIVE/V7P3R_v7.0.exe` | 400-600 | 500 | UNTESTED | Version 7 series |
| **V7P3R** | v7.2 | `V7P3R_ARCHIVE/V7P3R_v7.2.exe` | 400-600 | 550 | UNTESTED | Refinements |
| **V7P3R** | v8.0 | `V7P3R_ARCHIVE/V7P3R_v8.0.exe` | 500-700 | 600 | UNTESTED | Version 8 series |
| **SlowMate** | v0.2.0 | `SlowMate/SlowMate_v0.2.0.exe` | 300-500 | 400 | UNTESTED | Early development |
| **SlowMate** | v0.3.0 | `SlowMate/SlowMate_v0.3.0.exe` | 400-600 | 500 | UNTESTED | Pre-release versions |
| **Cecilia** | v0.2.0 | `Cecilia/Cecilia_v0.2.0.exe` | 400-600 | 500 | UNTESTED | Unknown capability |
| **Cecilia** | v0.3.0 | `Cecilia/Cecilia_v0.3.0.exe` | 500-700 | 600 | UNTESTED | Unknown capability |
| **CaptureOpponent** | v1.0 | `Opponents/CaptureOpponent_v1.0/` | 300-500 | 400 | UNTESTED | Material-greedy bot |
| **MaterialOpponent** | v1.0 | `Opponents/MaterialOpponent_v1.0/` | 600-900 | 800 | FUNCTIONAL | Current training opponent |

---

## INTERMEDIATE TIER (Tactical Play - ELO 900-1300)

| Engine | Version | Path | Strength Range | Reference ELO | Status | Notes |
|--------|---------|------|----------------|---------------|--------|-------|
| **V7P3R** | v9.0 | `V7P3R_ARCHIVE/V7P3R_v9.0.exe` | 700-900 | 800 | UNTESTED | Version 9 series |
| **V7P3R** | v9.5 | `V7P3R_ARCHIVE/V7P3R_v9.5.exe` | 800-1000 | 900 | UNTESTED | Mid-version refinement |
| **V7P3R** | v10.0 | `V7P3R_ARCHIVE/V7P3R_v10.0.exe` | 900-1100 | 1000 | UNTESTED | Version 10 series start |
| **V7P3R** | v10.5 | `V7P3R_ARCHIVE/V7P3R_v10.5.exe` | 950-1150 | 1050 | UNTESTED | Mid v10 refinement |
| **V7P3R** | v11.0 | `V7P3R_ARCHIVE/V7P3R_v11.0.exe` | 1000-1200 | 1100 | UNTESTED | Version 11 series |
| **SlowMate** | v1.0 | `SlowMate/Slowmate_v1.0.exe` | 700-900 | 800 | UNTESTED | First major release |
| **SlowMate** | v2.0 | `SlowMate/SlowMate_v2.0.exe` | 900-1100 | 1000 | UNTESTED | Second major release |
| **SlowMate** | v3.0 | `SlowMate/SlowMate_v3.0.exe` | 1100-1300 | 1200 | UNTESTED | Third major release |
| **SlowMate** | v3.2 | `SlowMate/SlowMate_v3.2/` | 1200-1400 | 1250 | UNTESTED | Lichess deployed (1200-1300) |
| **VPR** | v1.0 | `VPR/VPR_v1.0/` | 600-900 | 750 | UNTESTED | First VPR version, may not work |
| **VPR** | v2.0 | `VPR/VPR_v2.0/` | 700-1000 | 850 | UNTESTED | Unknown status |
| **Cece** | v1.0 | `Cece/Cece_v1.0.exe` | 800-1100 | 950 | UNTESTED | Unknown lineage |
| **Cece** | v2.0 | `Cece/Cece_v2.0.exe` | 900-1200 | 1050 | UNTESTED | Version 2 series |
| **CaptureOpponent** | v2.0 | `Opponents/CaptureOpponent_v2.0/` | 500-800 | 650 | UNTESTED | Improved capture logic |
| **CoverageOpponent** | v1.0 | `Opponents/CoverageOpponent_v1.0/` | 600-900 | 750 | UNTESTED | Space control focused |
| **MaterialOpponent** | v2.0 | `Opponents/MaterialOpponent_v2.0/` | 800-1100 | 950 | UNTESTED | Enhanced material eval |

---

## ADVANCED TIER (Strong Tactical - ELO 1300-1600)

| Engine | Version | Path | Strength Range | Reference ELO | Status | Notes |
|--------|---------|------|----------------|---------------|--------|-------|
| **V7P3R** | v12.0 | `V7P3R_ARCHIVE/V7P3R_v12.0.exe` | 1100-1300 | 1200 | UNTESTED | First deployed version? |
| **V7P3R** | v12.6 | `V7P3R/V7P3R_v12.6/` | 1200-1400 | 1300 | UNTESTED | Last v12 version |
| **V7P3R** | v13.0 | `V7P3R_ARCHIVE/V7P3R_v13.0/` | 1300-1500 | 1400 | UNTESTED | Version 13 series |
| **V7P3R** | v14.1 | `V7P3R_ARCHIVE/V7P3R_v14.1/` | 1350-1550 | 1450 | UNTESTED | Stable v14 version |
| **VPR** | v3.0 | `VPR/VPR_v3.0/` | 900-1200 | 1050 | UNTESTED | Mid-series VPR |
| **VPR** | v4.0 | `VPR/VPR_v4.0/` | 1000-1300 | 1150 | UNTESTED | Unknown status |
| **VPR** | v5.0 | `VPR/VPR_v5.0/` | 1100-1400 | 1250 | UNTESTED | Unknown status |
| **C0BR4** | v2.9 | `C0BR4/C0BR4_v2.9/` | 1200-1400 | 1300 | UNTESTED | Pre-v3 C0BR4 |
| **C0BR4** | v3.1 | `C0BR4/C0BR4_v3.1/` | 1400-1600 | 1550 | UNTESTED | Lichess deployed (1500-1600) |
| **Cece** | v2.3 | `Cece/Cece_v2.3.exe` | 1100-1400 | 1250 | UNTESTED | Latest Cece |
| **Copycat** | v1.0 | `Copycat/Copycat_v1.0/` | 1000-1300 | 1150 | UNTESTED | Unknown concept |
| **Copycat** | v2.0 | `Copycat/Copycat_v2.0/` | 1200-1500 | 1350 | UNTESTED | Unknown concept |
| **PositionalOpponent** | v1.0 | `Opponents/PositionalOpponent_v1.0/` | 300-500 | 400 | BROKEN | Makes illegal moves |

---

## EXPERT TIER (Competition Level - ELO 1600-1900)

| Engine | Version | Path | Strength Range | Reference ELO | Status | Notes |
|--------|---------|------|----------------|---------------|--------|-------|
| **V7P3R** | v15.0 | `V7P3R_ARCHIVE/V7P3R_v15.0/` | 1400-1600 | 1500 | UNTESTED | Version 15 series |
| **V7P3R** | v16.0 | `V7P3R_ARCHIVE/V7P3R_v16.0/` | 1450-1650 | 1550 | UNTESTED | Version 16 series |
| **V7P3R** | v17.0 | `V7P3R_ARCHIVE/V7P3R_v17.0/` | 1500-1700 | 1600 | UNTESTED | First v17 |
| **V7P3R** | v17.1 | `V7P3R/V7P3R_v17.1/` | 1400-1600 | 1487 | FUNCTIONAL | Lichess deployed, current training opponent |
| **V7P3R** | v17.8 | `V7P3R/V7P3R_v17.8/` | 1550-1750 | 1623 | FUNCTIONAL | Lichess deployed, current training opponent |
| **V7P3R** | v18.0 | `V7P3R/V7P3R_v18.0/` | 1600-1800 | 1650 | UNTESTED | First v18 |
| **V7P3R** | v18.3 | `V7P3R/V7P3R_v18.3/` | 1600-1800 | 1661 | FUNCTIONAL | Lichess deployed, current training opponent |
| **V7P3R** | v18.4 | `V7P3R/V7P3R_v18.4/` | 1600-1800 | 1670 | UNTESTED | Latest deployed |
| **VPR** | v8.1 | `VPR/VPR_v8.1/` | 1400-1700 | 1550 | UNTESTED | Latest VPR, may not work |
| **C0BR4** | v3.4 | `C0BR4/C0BR4_v3.4/` | 1500-1700 | 1600 | UNTESTED | Latest C0BR4 |

---

## SUMMARY STATISTICS

**Total Engines Available**: ~90+ versions across 11 engine families

**By Strength Tier**:
- Weakest (100-500): 10 engines
- Weak (500-900): 10 engines  
- Intermediate (900-1300): 16 engines
- Advanced (1300-1600): 12 engines
- Expert (1600-1900): 10 engines

**By Status**:
- FUNCTIONAL: 4 (RandomOpponent v1.0, MaterialOpponent v1.0, V7P3R v17.1/v17.8/v18.3)
- BROKEN: 1 (PositionalOpponent v1.0)
- UNTESTED: 85+

**Key Families**:
- **V7P3R**: 60+ versions (v4.1 → v18.4) - Most comprehensive
- **SlowMate**: 17 versions (v0.0.0 → v3.2) - Mate specialist
- **C0BR4**: 5+ versions (v2.9 → v3.4) - C# architecture, Lichess proven
- **VPR**: 9 versions (v1.0 → v9.0) - Unknown functionality status
- **Cece/Cecilia**: 11 versions - Unknown lineage
- **Custom Opponents**: 11 scripted bots (Random, Material, Capture, Coverage, Positional)

---

## NEXT STEPS

1. **Validation Testing**: Test UNTESTED engines in priority order (weakest first)
2. **ELO Calibration**: Update Reference ELO based on head-to-head tournaments
3. **Curriculum Design**: Design scaled training regiment based on tested strengths
4. **Documentation**: Record which engines work, which are broken, actual observed ELO

**Testing Protocol**:
- 9-test UCI validation (does engine respond correctly?)
- 10-game integration test vs known baseline (does it play reasonable chess?)
- Status update: FUNCTIONAL, BROKEN, or BROKEN_UCI

---

## PROPOSED CURRICULUM (DRAFT - Pending Testing)

**Phase 1: Foundation (100 gens)**
- RandomOpponent v1.0: 80%
- V7P3R v4.1-v6.1: 20%
- Target: 80%+ win rate vs Random

**Phase 2: Basic Tactics (100 gens)**  
- V7P3R v6.0-v8.0: 50%
- SlowMate v0.1-v0.3: 30%
- MaterialOpponent v1.0: 20%
- Target: 60%+ win rate

**Phase 3: Tactical Development (100 gens)**
- V7P3R v9.0-v11.0: 50%
- SlowMate v1.0-v2.0: 30%
- MaterialOpponent v1.0: 20%
- Target: 50%+ win rate

**Phase 4: Intermediate Play (100 gens)**
- V7P3R v12.0-v14.0: 60%
- SlowMate v3.0-v3.2: 20%
- C0BR4 v2.9: 20%
- Target: 40%+ win rate

**Phase 5: Advanced Competition (200 gens)**
- V7P3R v15.0-v17.1: 50%
- V7P3R v17.8-v18.3: 30%
- C0BR4 v3.1-v3.4: 20%
- Target: 30%+ win rate

**Total: 600 generations, estimated 5-8 days of training**
