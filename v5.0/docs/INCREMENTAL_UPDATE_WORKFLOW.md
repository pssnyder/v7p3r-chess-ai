# Incremental Dataset Update Workflow

This document describes how to add new training data to the existing V7P3R AI v5.0 dataset.

## Overview

When new puzzle analyses, game PGNs, or other data sources become available, use this workflow to incrementally update the master training dataset without starting from scratch.

## Workflow Steps

### 1. Extract New Positions

**For Puzzle Analysis Files:**
```bash
cd "E:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v5.0"

python scripts/extract_puzzle_results.py \
  --input "path/to/new_puzzle_analysis.json" \
  --output "data/puzzles/new_data_raw.jsonl"
```

**For PGN Game Files:**
```bash
python scripts/extract_v7p3r_pgns.py \
  --input "path/to/new_games.pgn" \
  --output "data/pgn/new_data_raw.jsonl"
```

### 2. Calculate Features

```bash
python scripts/calculate_features.py \
  --input "data/puzzles/new_data_raw.jsonl" \
  --output "data/puzzles/new_data_with_features.jsonl" \
  --feature-set standard
```

### 3. Grade with Stockfish (if needed)

**Puzzle data**: Already has Stockfish analysis, skip this step
**PGN data**: Requires grading

```bash
python scripts/grade_with_stockfish.py \
  --input "data/pgn/new_data_with_features.jsonl" \
  --output "data/pgn/new_data_graded.jsonl" \
  --stockfish-path "path/to/stockfish.exe" \
  --depth 15 \
  --time-limit 5.0
```

### 4. Add to Master Dataset

```bash
python scripts/update_master_dataset.py \
  --master "data/final/v7p3r_ai_v5_training_dataset_complete.jsonl" \
  --new "data/puzzles/new_data_with_features.jsonl" \
  --output "data/final/v7p3r_ai_v5_training_dataset_complete.jsonl" \
  --backup
```

This will:
- Create a timestamped backup of the current master dataset
- Append new positions to master dataset
- Remove any duplicates (by position_id)
- Update metadata

### 5. Regenerate Splits

```bash
python scripts/analyze_dataset.py \
  --input "data/final/v7p3r_ai_v5_training_dataset_complete.jsonl" \
  --output "data/analysis" \
  --create-splits \
  --train-ratio 0.8 \
  --val-ratio 0.1
```

This will:
- Regenerate stratified train/validation/test splits
- Update statistics and analysis reports
- Preserve stratification by move quality grade

### 6. Verify Update

```bash
python scripts/verify_dataset.py \
  --dataset "data/final/v7p3r_ai_v5_training_dataset_complete.jsonl"
```

This will:
- Count total positions
- Check for duplicates
- Validate JSON structure
- Verify all required fields present
- Compare grade distribution before/after

---

## Quick Update Commands

### Example: Adding 1000 New Puzzles (May 7, 2026)

```bash
cd "E:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v5.0"

# Step 1: Extract
python scripts/extract_puzzle_results.py \
  --input "E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\analysis_results\V7P3R_v18_3_enhanced_sequence_analysis_20260507_040307.json" \
  --output "data/puzzles/new_v18_3_1000puzzles_raw.jsonl"

# Step 2: Features
python scripts/calculate_features.py \
  --input "data/puzzles/new_v18_3_1000puzzles_raw.jsonl" \
  --output "data/puzzles/new_v18_3_1000puzzles_with_features.jsonl" \
  --feature-set standard

# Step 3: Add to master (Stockfish analysis already included in puzzles)
Get-Content "data/final/v7p3r_ai_v5_training_dataset_complete.jsonl", `
            "data/puzzles/new_v18_3_1000puzzles_with_features.jsonl" | `
  Set-Content "data/final/v7p3r_ai_v5_training_dataset_complete_updated.jsonl"

# Step 4: Backup old dataset
Move-Item "data/final/v7p3r_ai_v5_training_dataset_complete.jsonl" `
          "data/backups/v7p3r_ai_v5_training_dataset_$(Get-Date -Format 'yyyyMMdd_HHmmss').jsonl"

# Step 5: Replace with updated
Move-Item "data/final/v7p3r_ai_v5_training_dataset_complete_updated.jsonl" `
          "data/final/v7p3r_ai_v5_training_dataset_complete.jsonl"

# Step 6: Regenerate splits and stats
python scripts/analyze_dataset.py \
  --input "data/final/v7p3r_ai_v5_training_dataset_complete.jsonl" \
  --output "data/analysis" \
  --create-splits
```

---

## Best Practices

### Before Adding New Data

1. **Check for duplicates**: Ensure new data doesn't overlap with existing dataset
2. **Verify format**: Confirm new data follows unified schema
3. **Backup existing dataset**: Always create timestamped backup

### After Adding New Data

1. **Regenerate splits**: Always create new train/val/test splits with updated data
2. **Update statistics**: Run analysis script to get updated metrics
3. **Document changes**: Record what was added in `data/changelog.md`

### Data Quality Checks

- **Required fields present**: metadata, position, engine_decision, stockfish_analysis, features
- **No null values** in critical fields (FEN, move grades, etc.)
- **Consistent versioning**: Track which V7P3R version generated the data
- **Duplicate detection**: Check for duplicate position_ids

---

## Automation Script

Create `scripts/update_master_dataset.py` for automated incremental updates (see below).

---

## Tracking Dataset Versions

Maintain `data/final/DATASET_CHANGELOG.md`:

```markdown
# Dataset Changelog

## v1.1 - May 7, 2026
- Added 2,264 positions from V7P3R v18.3 1000-puzzle baseline analysis
- Total positions: 230,930 (was 228,666)
- Train/val/test splits regenerated

## v1.0 - May 7, 2026
- Initial dataset creation
- 210,054 PGN positions (Dec 29, 2025 - May 7, 2026)
- 18,612 puzzle positions (historical analyses v8.0-v18.4)
- Total: 228,666 positions
```

---

## Rollback Procedure

If something goes wrong:

```bash
# Restore from backup
$backup = Get-ChildItem "data/backups" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Copy-Item $backup.FullName "data/final/v7p3r_ai_v5_training_dataset_complete.jsonl"

# Regenerate splits from restored dataset
python scripts/analyze_dataset.py \
  --input "data/final/v7p3r_ai_v5_training_dataset_complete.jsonl" \
  --output "data/analysis" \
  --create-splits
```

---

## Future Enhancements

Planned features for `update_master_dataset.py`:

- [ ] Automatic duplicate detection and removal
- [ ] Validation of new data schema
- [ ] Automatic backup with timestamped versions
- [ ] Merge conflict resolution (if position_id exists)
- [ ] Statistics comparison (before/after)
- [ ] Automatic changelog entry generation
