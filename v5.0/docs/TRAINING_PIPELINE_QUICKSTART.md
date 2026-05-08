# V7P3R AI v5.0 - Training Pipeline Quick Start Guide

## 🎯 Overview

This guide walks you through creating a complete training dataset from V7P3R historical games using the 3-stage pipeline:

1. **Extract** positions from PGN files
2. **Calculate** heuristic features (binary observations)
3. **Grade** moves with Stockfish (0-5 quality scale)

## 📁 Prerequisites

### Required Software
- Python 3.8+
- Stockfish 16: `E:\Programming Stuff\Chess Engines\Tournament Engines\Stockfish\stockfish-windows-x86-64-avx2.exe`
- Required packages: `pip install python-chess`

### Data Sources
- **V7P3R PGN Games**: `E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\game_records\Lichess V7P3R Bot`
- **Puzzle Analysis Results**: JSON output from `universal_puzzle_analyzer.py` (1000+ puzzles recommended)

## 🚀 Quick Start (3 Commands)

### 1. Test Run (100 games, ~5 minutes)

```powershell
# Navigate to v5.0 directory
cd "E:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v5.0"

# Run test pipeline
python scripts/run_training_pipeline.py `
  --pgn-dir "E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\game_records\Lichess V7P3R Bot" `
  --output-dir "data/test_run" `
  --max-games 100 `
  --feature-set minimal `
  --stockfish-depth 15
```

**Expected Output**:
- ~2,000 positions extracted
- Features calculated (minimal set)
- Moves graded with Stockfish depth 15
- Final dataset: `data/test_run/stage3_graded/training_dataset_YYYYMMDD_HHMMSS.jsonl`

### 2. Standard Run (All games, standard features, ~2-3 hours)

```powershell
python scripts/run_training_pipeline.py `
  --pgn-dir "E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\game_records\Lichess V7P3R Bot" `
  --output-dir "data/training/v1_standard" `
  --feature-set standard `
  --stockfish-depth 20
```

**Expected Output**:
- ~100,000 positions (from 5,000+ games)
- Standard features (core + king safety + piece activity + move context)
- Stockfish depth 20 analysis
- Dataset size: ~100-200 MB

### 3. Production Run (All games, full features, ~4-6 hours)

```powershell
python scripts/run_training_pipeline.py `
  --pgn-dir "E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\game_records\Lichess V7P3R Bot" `
  --output-dir "data/training/v1_production" `
  --feature-set full `
  --stockfish-depth 20
```

**Expected Output**:
- ~100,000 positions
- Full features (includes expensive pawn structure and tactical features)
- Stockfish depth 20 analysis
- Dataset size: ~150-250 MB

## 📊 Pipeline Stages Explained

### Stage 1: PGN Position Extraction

**Purpose**: Extract all positions where V7P3R made a move

**Input**: PGN files
**Output**: `stage1_raw/positions_raw.jsonl`

**What it does**:
- Scans all PGN files in directory
- Identifies V7P3R games (white or black)
- Replays each game move-by-move
- Extracts FEN position BEFORE each v7p3r move
- Records move played (UCI + SAN)
- Adds game metadata (opponent, result, date, etc.)

**Example record** (stage 1):
```json
{
  "metadata": {
    "source": "v7p3r_pgn",
    "game_id": "2025.12.10_v7p3r_bot_vs_opponent",
    "position_id": "game123_15"
  },
  "position": {
    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "game_phase": "opening",
    "material_balance": 0,
    "in_check": false
  },
  "engine_decision": {
    "move_uci": "e2e4",
    "move_san": "e4",
    "is_capture": false,
    "is_check": false
  },
  "stockfish_analysis": null,
  "features": null
}
```

### Stage 2: Feature Calculation

**Purpose**: Add heuristic observations (binary/categorical features)

**Input**: `stage1_raw/positions_raw.jsonl`
**Output**: `stage2_features/positions_with_features.jsonl`

**What it does**:
- Parses FEN position
- Calculates configured features:
  - **Minimal**: Core position only (F001-F005)
  - **Standard**: Core + king safety + piece activity + move context
  - **Full**: All features including expensive calculations

**Feature Categories**:
- **Core Position** (F001-F005): Always calculated
  - Position FEN, game phase, material balance, piece count
- **King Safety** (F010-F013): Castling, pawn shield, under attack
- **Pawn Structure** (F020-F023): Passed, doubled, isolated pawns
- **Piece Activity** (F030-F033): Mobility, strong squares, bishop pair
- **Tactical** (F040-F042): Hanging pieces, attacked pieces
- **Move Context** (F050-F053): Capture, check, promotion, castling

**Example features block**:
```json
"features": {
  "F001_position_fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
  "F002_game_phase": "opening",
  "F003_material_balance_cp": 0,
  "F004_material_advantage_category": "equal",
  "F005_total_piece_count": 32,
  "F010_white_king_castled": false,
  "F010_black_king_castled": false,
  "F032_white_has_bishop_pair": true,
  "F032_black_has_bishop_pair": true,
  "F050_is_capture": false,
  "F051_is_check": false
}
```

### Stage 3: Stockfish Grading

**Purpose**: Analyze with Stockfish and grade move quality

**Input**: `stage2_features/positions_with_features.jsonl`
**Output**: `stage3_graded/training_dataset_YYYYMMDD_HHMMSS.jsonl`

**What it does**:
- Analyzes each position with Stockfish (depth 20, multipv 5)
- Compares v7p3r's move to Stockfish's top-5 moves
- Grades move on 0-5 scale:
  - **5 (Excellent)**: Move is #1 best
  - **4 (Good)**: Move is #2 best
  - **3 (Decent)**: Move is #3 best
  - **2 (Suboptimal)**: Move is #4 best
  - **1 (Poor)**: Move is #5 best
  - **0 (Blunder)**: Move not in top-5

**Example stockfish_analysis block**:
```json
"stockfish_analysis": {
  "stockfish_version": "16",
  "analysis_depth": 20,
  "top_moves": [
    {"rank": 1, "uci": "e2e4", "san": "e4", "eval_cp": 35, "pv": ["e2e4", "c7c5", "g1f3"]},
    {"rank": 2, "uci": "d2d4", "san": "d4", "eval_cp": 28, "pv": ["d2d4", "d7d5", "c2c4"]},
    {"rank": 3, "uci": "g1f3", "san": "Nf3", "eval_cp": 25, "pv": ["g1f3", "d7d5", "d2d4"]}
  ],
  "played_move_rank": 1,
  "move_quality_grade": 5,
  "eval_drop_cp": 0,
  "best_move_uci": "e2e4",
  "best_move_eval_cp": 35
}
```

## 🛠️ Advanced Usage

### Run Individual Stages

If you need to re-run specific stages:

#### Stage 1 Only: Extract Positions
```powershell
python scripts/extract_v7p3r_pgns.py `
  --pgn-dir "E:\...\Lichess V7P3R Bot" `
  --output "data/raw/positions.jsonl" `
  --max-games 500
```

#### Stage 2 Only: Calculate Features
```powershell
python scripts/calculate_features.py `
  --input "data/raw/positions.jsonl" `
  --output "data/features/positions_with_features.jsonl" `
  --feature-set standard
```

#### Stage 3 Only: Grade with Stockfish
```powershell
python scripts/grade_with_stockfish.py `
  --input "data/features/positions_with_features.jsonl" `
  --output "data/graded/training_dataset.jsonl" `
  --stockfish-path "C:\path\to\stockfish.exe" `
  --depth 20
```

### Stockfish Configuration

#### Faster Analysis (Lower Quality)
```powershell
--stockfish-depth 15 --stockfish-time-limit 5.0
```

#### Slower Analysis (Higher Quality)
```powershell
--stockfish-depth 25 --stockfish-time-limit 15.0
```

#### Custom Stockfish Path
```powershell
--stockfish-path "C:\Chess Engines\stockfish\stockfish-windows-x86-64-avx2.exe"
```

## 📈 Expected Performance

### Test Run (100 games)
- **Positions**: ~2,000
- **Time**: ~5-10 minutes
- **Size**: ~2-5 MB

### Standard Run (5,000 games)
- **Positions**: ~100,000
- **Time**: ~2-3 hours
- **Size**: ~100-200 MB

### Performance Breakdown
- **Stage 1** (Extraction): ~1,000 positions/sec (~2 minutes for 100k)
- **Stage 2** (Features - standard): ~500 positions/sec (~4 minutes for 100k)
- **Stage 2** (Features - full): ~200 positions/sec (~8 minutes for 100k)
- **Stage 3** (Stockfish depth 20): ~1-2 positions/sec (~15-30 hours for 100k!)

**Stockfish is the bottleneck!** Consider:
- Lower depth (15) for faster results: ~2-3 positions/sec (~10-15 hours)
- Parallel processing (future enhancement)

## 🔍 Inspecting Output

### View Pipeline Statistics
```powershell
cat data/training/v1_standard/pipeline_stats.json
```

### Sample Training Records
```powershell
# View first 5 records
Get-Content data/training/v1_standard/stage3_graded/training_dataset_*.jsonl | Select-Object -First 5
```

### Count Records
```powershell
(Get-Content data/training/v1_standard/stage3_graded/training_dataset_*.jsonl).Count
```

### Check File Size
```powershell
Get-ChildItem data/training/v1_standard/stage3_graded/*.jsonl | Select-Object Name, Length
```

## 🧩 Puzzle Data Ingestion

### Run Puzzle Analysis (Prerequisite)

First, run V7P3R v18.3 through the universal puzzle analyzer:

```powershell
cd "E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-tester"

python engine_utilities/universal_puzzle_analyzer.py `
  --engine "E:\Programming Stuff\Chess Engines\V7P3R Chess Engine\v7p3r-chess-engine\V7P3R_v18_current.bat" `
  --puzzles 1000 `
  --min-rating 800 `
  --max-rating 2500 `
  --time 20.0
```

This creates: `puzzle_results_v18_3_YYYYMMDD_HHMMSS.json`

### Extract Puzzle Positions

Once you have puzzle results, extract training positions:

```powershell
cd "E:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v5.0"

# Extract positions from puzzle results (Stockfish analysis already included!)
python scripts/extract_puzzle_results.py `
  --input "E:\path\to\puzzle_results_v18_3_20260506_123456.json" `
  --output "data/puzzles/positions_raw.jsonl" `
  --engine-version "18.3"

# Calculate features (same as PGN data)
python scripts/calculate_features.py `
  --input "data/puzzles/positions_raw.jsonl" `
  --output "data/puzzles/training_dataset.jsonl" `
  --feature-set standard
```

**BONUS**: Puzzle data already has Stockfish analysis! No need to run the grading step. ⚡

### Puzzle Data Advantages

- **Tactical Focus**: Positions with tactical themes (pins, forks, mates)
- **Difficulty Rated**: Puzzle ratings indicate position complexity
- **Pre-Graded**: Stockfish top-5 already included from puzzle analyzer
- **Theme Labels**: Additional metadata for themed training subsets

### Combine PGN + Puzzle Data

```powershell
# Combine both datasets
cat data/training/v1_standard/stage3_graded/training_dataset_*.jsonl, `
    data/puzzles/training_dataset.jsonl | `
  Set-Content data/combined/full_training_dataset.jsonl
```

**Expected Combined Dataset**:
- ~86,000 positions from PGNs (historical games)
- ~3,000-5,000 positions from puzzles (tactical positions)
- Total: ~90,000-91,000 diverse training examples

## 🎓 Next Steps

After creating your training dataset:

1. **Validate Dataset** - Check distribution of move grades, game phases, puzzle themes
2. **Build Model** - Create PyTorch dataset loader, define neural network architecture
3. **Train Model** - Supervised learning with Stockfish grades as targets
4. **Evaluate Model** - Test on held-out positions, compare to v7p3r baseline
5. **Deploy** - Wrap trained model in UCI interface

See `UNIFIED_TRAINING_DATASET.md` for dataset schema details.

## ❓ Troubleshooting

### "Stockfish not found"
```powershell
# Specify full path
--stockfish-path "C:\path\to\stockfish.exe"
```

### "No PGN files found"
- Check directory path is correct
- Verify `.pgn` extension (not `.txt` or `.zip`)

### "Module not found: chess"
```powershell
pip install python-chess
```

### Pipeline is slow
- Use `--feature-set minimal` for faster feature calculation
- Lower `--stockfish-depth` to 15 for 2x speedup
- Use `--max-games` to test with subset first

### Out of memory
- Process in batches with `--max-games`
- Combine batch outputs manually

## 📝 Batch Helper Scripts

See `run_test_pipeline.bat` and `run_production_pipeline.bat` for pre-configured commands.
