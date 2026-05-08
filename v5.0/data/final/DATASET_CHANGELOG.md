# V7P3R AI v5.0 - Dataset Changelog

This document tracks all incremental updates to the master training dataset.

---

## v1.1 - May 7, 2026 @ 12:53 PM

**Update Type**: Incremental addition  
**Source**: V7P3R v18.3 1000-puzzle baseline analysis  
**Input File**: `V7P3R_v18_3_enhanced_sequence_analysis_20260507_040307.json`

### Changes
- ✅ Added **2,264 new positions** from 1,000 tactical puzzles
- Positions already included Stockfish depth-15 analysis (no re-grading needed)
- Features calculated using `standard` feature set

### Dataset Metrics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Total Positions** | 228,666 | 230,930 | +2,264 (+0.99%) |
| **PGN Positions** | 210,054 | 210,054 | - |
| **Puzzle Positions** | 18,612 | 20,876 | +2,264 |
| **v18.3 Coverage** | 210,054 | 212,318 | +2,264 |

### Split Distribution

| Split | Positions | Percentage |
|-------|-----------|------------|
| **Train** | 184,742 | 80.0% |
| **Validation** | 23,089 | 10.0% |
| **Test** | 23,099 | 10.0% |

### Move Quality Distribution

| Grade | Description | Positions | Percentage |
|-------|-------------|-----------|------------|
| 5 | Best move | 93,826 | 40.63% |
| 4 | 2nd best | 34,815 | 15.08% |
| 3 | 3rd best | 20,695 | 8.96% |
| 2 | 4th best | 14,357 | 6.22% |
| 1 | 5th best | 10,408 | 4.51% |
| 0 | Not in top-5 | 56,829 | 24.61% |

### Processing Steps
1. Extraction: `extract_puzzle_results.py` → 2,264 positions
2. Features: `calculate_features.py` with `standard` config
3. Combination: PowerShell append to master dataset
4. Backup: Created `dataset_backup_20260507_125235.jsonl`
5. Splits: Regenerated stratified train/val/test (seed=42)
6. Analysis: Updated statistics and reports

### Files Updated
- ✅ `data/final/v7p3r_ai_v5_training_dataset_complete.jsonl` (575.81 MB)
- ✅ `data/analysis/splits/train.jsonl` (184,742 positions)
- ✅ `data/analysis/splits/validation.jsonl` (23,089 positions)
- ✅ `data/analysis/splits/test.jsonl` (23,099 positions)
- ✅ `data/analysis/dataset_analysis.json`
- ✅ `data/analysis/dataset_analysis.md`

### Notes
- This update demonstrates the **incremental data addition workflow** documented in `INCREMENTAL_UPDATE_WORKFLOW.md`
- No duplicates detected (position_id validation passed)
- All positions validated with required fields present
- Backup preserved at `data/backups/dataset_backup_20260507_125235.jsonl`

---

## v1.0 - May 7, 2026 @ 10:59 AM

**Update Type**: Initial dataset creation  
**Sources**: PGN game files + historical puzzle analyses

### Initial Dataset Composition

**PGN Positions**: 210,054 positions
- **Source**: V7P3R Lichess Bot game history
- **Date Range**: December 29, 2025 - May 7, 2026
- **Games**: 5,736 total games
- **Processing**: 
  - Extraction: 45 seconds @ 4,669 games/sec
  - Features: 56 seconds @ 3,751 positions/sec
  - Stockfish Grading: 14.2 hours @ 4.1 positions/sec (depth 15)

**Puzzle Positions**: 18,612 positions
- **Source**: 31 historical puzzle analysis files
- **Version Range**: v8.0 through v18.4
- **Processing**: 
  - Extraction: Used existing Stockfish analysis from puzzle runs
  - Features: ~5 seconds @ 3,722 positions/sec

### Initial Metrics

| Metric | Value |
|--------|-------|
| **Total Positions** | 228,666 |
| **Dataset Size** | 548.31 MB |
| **V7P3R Best Moves (Grade 5)** | 40.14% |
| **Top-3 Performance** | 64.4% |
| **Opening Phase** | 80.5% |
| **Middlegame** | 15.8% |
| **Endgame** | 3.7% |

### Initial Splits

| Split | Positions | Percentage |
|-------|-----------|------------|
| **Train** | 182,930 | 80.0% |
| **Validation** | 22,864 | 10.0% |
| **Test** | 22,872 | 10.0% |

### Processing Pipeline
1. **PGN Extraction**: `extract_v7p3r_pgns.py`
2. **Puzzle Extraction**: `extract_puzzle_results.py`
3. **Feature Calculation**: `calculate_features.py` (standard config)
4. **Stockfish Grading**: `grade_with_stockfish.py` (depth 15, 5s limit)
5. **Combination**: Manual JSONL concatenation
6. **Analysis**: `analyze_dataset.py` with stratified splits

### Files Created
- ✅ `data/final/v7p3r_ai_v5_training_dataset_complete.jsonl`
- ✅ `data/analysis/splits/` (train/validation/test)
- ✅ `data/analysis/dataset_analysis.json`
- ✅ `data/analysis/dataset_analysis.md`
- ✅ Documentation: 
  - `DATASET_CREATION_SUMMARY.md`
  - `STOCKFISH_GRADING_SUMMARY.md`
  - `DATA_PROCESSING_PIPELINE.md`

---

## Version History Summary

| Version | Date | Total Positions | Change | Notes |
|---------|------|-----------------|--------|-------|
| **v1.1** | May 7, 2026 | 230,930 | +2,264 | Added 1000 puzzles from v18.3 |
| **v1.0** | May 7, 2026 | 228,666 | Initial | PGNs + historical puzzles |

---

## Future Planned Additions

### Short-term (v1.x)
- [ ] Additional puzzle analyses from ongoing v18.3 baseline runs
- [ ] Historical game data from earlier V7P3R versions (if available)
- [ ] Targeted tactical position sets (pins, forks, skewers, etc.)

### Medium-term (v2.x)
- [ ] V7P3R v18.4+ game data (when available)
- [ ] User-submitted interesting positions
- [ ] Tournament games against other engines
- [ ] Specific endgame tablebase positions (R+B vs K, Q vs R, etc.)

### Long-term (v3.x)
- [ ] Self-play generated positions (if/when reinforcement learning added)
- [ ] Lichess database positions matching V7P3R playing style
- [ ] Annotated master games (for strategic pattern learning)

---

## Maintenance Notes

### Backup Strategy
- **Before each update**: Create timestamped backup in `data/backups/`
- **Retention**: Keep last 5 backups, archive older versions
- **Verification**: Check file size and position count after each backup

### Quality Assurance
- **Duplicate Detection**: Validate no duplicate position_ids before merge
- **Schema Validation**: Ensure all required fields present
- **Grade Distribution**: Monitor for unexpected shifts in move quality
- **Feature Coverage**: Verify all positions have feature calculations

### Update Frequency
- **Puzzle Analyses**: Add every 1,000+ puzzle batch
- **Game Data**: Add every 2,000+ games or monthly (whichever first)
- **Manual Curation**: Add interesting positions as discovered

---

*Last Updated: May 7, 2026 @ 12:53 PM*  
*Maintained by: V7P3R AI Development Team*
