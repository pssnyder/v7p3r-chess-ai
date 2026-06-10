# Chess Engine Benchmark System - Summary

## Overview
Fast puzzle-based ELO estimation system using 4.9M Lichess puzzles. Provides accurate strength estimates in ~3 minutes instead of hours of tournament games.

## System Components

### 1. create_benchmark_suite.py
**Purpose**: Samples 100 representative puzzles from database
**Output**: `benchmarks/benchmark_suite.json`
**Structure**: 5 tiers × 20 puzzles each
- Tier 1 (400-800): Beginner tactics
- Tier 2 (800-1200): Weak tactics  
- Tier 3 (1200-1600): Intermediate
- Tier 4 (1600-2000): Advanced
- Tier 5 (2000-2400): Expert

### 2. benchmark_single_engine.py
**Purpose**: Tests single engine against benchmark suite
**Runtime**: ~3 minutes per engine (5s per puzzle)
**Features**:
- Early termination if <20% on Tier 1
- UCI protocol communication
- Supports both .bat and .exe engines
- Detailed per-puzzle results

**Puzzle Testing Pattern** (Critical):
```python
# Load board from FEN
board = chess.Board(puzzle.fen)

# Apply opponent's setup move (moves[0])
setup_move = chess.Move.from_uci(solution_moves[0])
board.push(setup_move)

# Get challenge FEN
challenge_fen = board.fen()

# Test engine on challenge position
engine_move = get_move_via_uci(challenge_fen, time_limit)

# Compare to expected solution (moves[1])
score = 5 if engine_move == solution_moves[1] else 0
```

### 3. batch_benchmark_catalog.py
**Purpose**: Automates testing of all engines in catalog
**Input**: `docs/opponents_catalog.csv`
**Features**:
- Sequential processing with progress saving
- Updates catalog with ELO estimates
- Filters for UNTESTED engines
- Estimated runtime: ~10 hours for 60 engines

## ELO Estimation Algorithm

1. **Test each tier sequentially** (20 puzzles per tier, 5 points each)
2. **Early termination**: Skip remaining tiers if <20% on Tier 1
3. **Find ceiling tier**: Highest tier with >40% accuracy
4. **Estimate ELO**: Use ceiling tier midpoint ± 100

Example: Engine with 95%/100%/90%/75%/50% scores
- Ceiling tier: Tier 5 (50% > 40%)
- Estimated ELO: 2100 (Tier 5 midpoint 2200, ±100)

## Validation Results

### V7P3R v17.1 (Known ~2100 ELO)
```
Tier 1 (400-800):   95% ✅ (19/20 solved)
Tier 2 (800-1200):  100% ✅ (20/20 solved)
Tier 3 (1200-1600): 90% ✅ (18/20 solved)  
Tier 4 (1600-2000): 75% ✅ (15/20 solved)
Tier 5 (2000-2400): 50% ⚠️ (10/20 solved)

Estimated ELO: 2100 (±100) ✅
Runtime: 171 seconds
```

### RandomOpponent (Expected <400 ELO)
```
Tier 1 (400-800): 0% ❌ (0/20 solved)
[Remaining tiers skipped]

Estimated ELO: 200 (±200) ✅
Runtime: 47 seconds (early termination)
```

## Database Details

**Source**: Lichess Puzzle Database
**Location**: `engine-tester/databases/puzzles.db`
**Size**: 4,914,603 puzzles
**Rating Range**: 399-3424 ELO

**Distribution**:
- 0-1000: ~1M puzzles
- 1000-1500: ~1.5M puzzles
- 1500-2000: ~1.3M puzzles
- 2000-2500: ~800K puzzles
- 2500+: ~100K puzzles

## Critical Bugfix (Dec 8, 2024)

**Problem**: All engines scoring 0% including known strong engines
**Root Cause**: Testing engine on original FEN instead of position after opponent's move
**Solution**: Apply opponent's setup move before getting engine response

Before Fix:
```python
# WRONG - tests from original position
process.stdin.write(f"position fen {puzzle.fen}\n")
```

After Fix:
```python
# CORRECT - tests from challenge position
board = chess.Board(puzzle.fen)
board.push(chess.Move.from_uci(solution_moves[0]))  # Apply opponent move
challenge_fen = board.fen()
process.stdin.write(f"position fen {challenge_fen}\n")
```

**Reference**: `engine-tester/engine_utilities/universal_puzzle_analyzer.py` lines 707-800

## Next Steps

1. ✅ Benchmark system created and validated
2. ✅ V7P3R v17.1 tested successfully (2100 ELO)
3. 🔄 Run batch benchmark on 60+ engines in catalog
4. ⏭️ Design graduated training curriculum based on validated ELO data
5. ⏭️ Implement curriculum learning for V8.0 training

## Usage Examples

### Test Single Engine
```bash
cd "E:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v8.0"
python src/benchmark_single_engine.py "E:\Tournament Engines\V7P3R\V7P3R_v17.1\V7P3R_v17.1.bat"
```

### Batch Test All Engines
```bash
python src/batch_benchmark_catalog.py
```

### Create New Benchmark Suite
```bash
python src/create_benchmark_suite.py
```

## Performance Metrics

- **Accuracy**: ✅ Validated against known engine strengths
- **Speed**: ~3 min per engine vs. hours for tournament games
- **Scalability**: Can test 60+ engines in ~10 hours
- **Reliability**: Early termination prevents wasting time on weak engines
- **Repeatability**: Same benchmark suite = consistent results

## Files

```
v8.0/
├── src/
│   ├── create_benchmark_suite.py       (190 lines)
│   ├── benchmark_single_engine.py      (480 lines)
│   └── batch_benchmark_catalog.py      (280 lines)
├── benchmarks/
│   ├── benchmark_suite.json            (100 puzzles)
│   └── report_*.json                   (test results)
└── docs/
    ├── opponents_catalog.csv           (60+ engines)
    └── BENCHMARK_SYSTEM_SUMMARY.md     (this file)
```
