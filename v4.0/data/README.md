# Data Directory Structure

This directory contains training data for V7P3RAI v4.0 agents.

## Directory Organization

```
data/
├── puzzles/               # Stage 1: 4M puzzle library
│   └── 4M_puzzle_library/ # Lichess puzzle database (to be linked/copied)
│
├── historical_games/      # Stage 2: V7P3R historical games
│   └── v7p3r_pgns/        # PGN files from production games
│
├── opening_book/          # Stage 3: Opening theory database
│   └── master_games_db/   # Master-level opening games
│
└── tablebases/            # Stage 3: Endgame tablebases
    └── syzygy_6piece/     # 6-piece Syzygy tablebases (~150GB)
```

## Setup Instructions

### Stage 1: Puzzle Library

**Source**: Local puzzle database  
**Location**: `E:/Programming Stuff/Chess Engines/Chess PGNs/training_data/fen_data_lichess_puzzles_db/`

**Setup**:
```powershell
# Option 1: Create symbolic link (recommended)
cd data/puzzles
New-Item -ItemType SymbolicLink -Name "4M_puzzle_library" -Target "E:\Programming Stuff\Chess Engines\Chess PGNs\training_data\fen_data_lichess_puzzles_db"

# Option 2: Copy files (requires ~10GB space)
Copy-Item -Recurse "E:\Programming Stuff\Chess Engines\Chess PGNs\training_data\fen_data_lichess_puzzles_db\*" "data/puzzles/4M_puzzle_library/"
```

**Expected Format**:
- CSV files with columns: FEN, Moves, Rating, Themes, GameUrl
- Or: JSON files with puzzle objects

**Validation**:
```powershell
# Check puzzle count
Get-ChildItem data/puzzles/4M_puzzle_library/ -Recurse | Measure-Object
```

### Stage 2: Historical Games

**Source**: V7P3R production game records  
**Location**: `E:/Programming Stuff/Chess Engines/Chess Engine Playground/engine-metrics/raw_data/game_records/Lichess V7P3R Bot/`

**Setup**:
```powershell
# Create symbolic link
cd data/historical_games
New-Item -ItemType SymbolicLink -Name "v7p3r_pgns" -Target "E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\game_records\Lichess V7P3R Bot"
```

**Expected Format**:
- PGN files (standard chess game notation)
- One or more games per file
- Headers include: Event, Date, White, Black, Result

### Stage 3: Opening Book

**Source**: To be determined (Polyglot book or master game database)

**Options**:
1. Use existing Polyglot opening book
2. Generate from master game database
3. Download from online resources

**Setup** (once available):
```powershell
# Example: Copy Polyglot book
Copy-Item "path/to/book.bin" "data/opening_book/master_games_db/book.bin"
```

### Stage 3: Tablebases

**Source**: Syzygy Tablebases (https://syzygy-tables.info/)

**Size**: ~150GB for 6-piece tablebases  
**Optional**: Not required for initial training, but improves endgame play

**Setup** (optional):
```powershell
# Download and extract Syzygy tablebases
# Extract to: data/tablebases/syzygy_6piece/
```

## Data Validation

Run validation script to ensure all data is properly linked:
```powershell
python scripts/validate_data.py
```

Expected output:
```
✓ Puzzle library: 4,000,000 puzzles found
✓ Historical games: 5,234 games found
⚠ Opening book: Not configured (optional for Stage 1)
⚠ Tablebases: Not configured (optional for Stage 1)
```

## Notes

- **Stage 1** only requires puzzle library
- **Stage 2** requires historical games
- **Stage 3** requires opening book and tablebases (optional)
- Symbolic links save disk space but require admin privileges on Windows
- Copying files works without admin but uses more disk space
