# Phase 0: Large-Scale Data Preparation Roadmap

**Status**: 🟢 PREREQUISITE TO PHASE 1  
**Duration**: 4 days (parallel work possible)  
**Input**: 120GB raw chess data  
**Output**: 30GB optimized, filtered, labeled training dataset  
**Bottleneck Solved**: I/O performance (1MB/sec → 500MB/sec)  

---

## Overview: Why Phase 0 is Critical

Your 120GB of data is currently **unusable for efficient training** without preprocessing:

| Format | Speed | Problem |
|--------|-------|---------|
| PGN (text) | 1 MB/sec | Full parsing overhead |
| JSONL | 5 MB/sec | JSON parsing, redundant text |
| Raw binary | 500 MB/sec | Fast but needs filtering |

**Solution**: Convert once to binary, apply filters, store on SSD

---

## Phase 0 Timeline: 4 Days

```
Day 1: Format Conversion (Parallel)
├─ PGNs → .bin format (Polyglot)
├─ JSONL → Binary positions (88-byte records)
└─ Puzzles → Tokenized concepts

Day 2: Filtering & Labeling (Parallel)
├─ Quiet position filtering (-30% positions)
├─ Evaluation balancing (50-50 pos/neg)
├─ Material distribution (40-60 imbal/balanced)
└─ Syzygy ground truth labeling (endgames)

Day 3: Dataset Construction
├─ Combine all sources
├─ Create train/val splits (90-10)
├─ Apply multi-task weights
└─ Create fast index

Day 4: Validation & Profiling
├─ Verify data integrity
├─ Profile I/O performance
├─ Generate statistics report
└─ Ready for Phase 1 training
```

---

## Detailed Workflow

### Day 1: Format Conversion

**Objective**: Convert all 120GB to optimized binary formats (500MB/sec speed)

#### Step 1.1: PGN Conversion (Parallel)
```bash
# Convert millions of PGNs to binary .bin format
python src/binary_format_converter.py \
  --input=raw_pgns/ \
  --output=binary_data/pgns.bin \
  --workers=8
```

**Input**: `raw_pgns/` (millions of PGN files)
**Output**: `pgns.bin` (~1.5GB)

**Conversion details**:
- Each game stored as moves (2 bytes per move)
- Move encoding: [from_square (8 bits)] [to_square (8 bits)]
- Game header: [game_id (4 bytes)] [move_count (2 bytes)]
- Example: 40-move game → ~80 bytes (vs ~5KB PGN text)

**Expected time**: 2-4 hours (depends on storage speed)
**CPU impact**: Moderate (move parsing)
**Disk I/O**: High sequential writes

#### Step 1.2: JSONL Conversion (Parallel)
```bash
# Convert evaluation JSON to binary positions
python src/binary_format_converter.py \
  --input=evaluations.jsonl \
  --output=binary_data/evals.bin \
  --chunk_size=100000
```

**Input**: `evaluations.jsonl` (95GB, FEN + eval pairs)
**Output**: `evals.bin` (~40GB)

**Binary record format** (88 bytes fixed):
```
Offset  Field              Type     Notes
0-7     fen_hash           uint64   MD5 hash of FEN (O(1) lookup)
8-9     eval               int16    Centipawns (-32K to +32K)
10      depth              uint8    Search depth
11-12   time_ms            uint16   Search time
13-15   wins/draws/losses  [u8×3]   WDL statistics
16      quiet_flag         bool     Quiet position
17-18   material_balance   int16    Material difference (cp)
19      phase              uint8    Opening/middle/endgame
20      piece_count        uint8    Total pieces
21-87   reserved           bytes    Future expansion
```

**Conversion details**:
- Parse each JSON line
- Calculate material balance (piece values)
- Determine game phase (piece count)
- Pack into fixed 88-byte record
- Write sequentially (no index lookup)

**Expected time**: 3-5 hours (JSON parsing overhead)
**CPU impact**: High (JSON parsing, material calculation)
**Disk I/O**: High sequential writes

#### Step 1.3: Puzzle Tokenization
```bash
# Tokenize Lichess puzzles into concept embeddings
python src/binary_format_converter.py \
  --input=lichess_puzzles.db \
  --output=binary_data/puzzles_tokenized.bin \
  --tokenizer=concepts
```

**Input**: Lichess puzzle database (4.9M positions)
**Output**: `puzzles_tokenized.bin` (~2GB)

**Concept tokenization**:
```
Traditional: FEN board → Network sees 4.9M unique positions (bad)
Tokenized: FEN board → [TACTIC_PIN, MATERIAL_UP_2, TEMPO] (good)

Concept tokens:
  1 = TACTIC_PIN         (piece pinned to king)
  2 = TACTIC_FORK        (one piece attacks multiple pieces)
  3 = TACTIC_SKEWER      (discovered attack)
  4 = TACTIC_BACK_RANK   (back rank mate threat)
  5 = MATERIAL_UP_1      (up 1 pawn)
  6 = MATERIAL_UP_2      (up 2 pawns)
  ... etc
  
Network learns: "Pinning patterns matter" not "This specific FEN"
Result: Better transfer learning to novel positions
```

**Expected time**: 30 minutes (pattern recognition)
**CPU impact**: Moderate
**Disk I/O**: Moderate

---

### Day 2: Filtering & Labeling

**Objective**: Apply filtering rules + Syzygy ground truth

#### Step 2.1: Quiet Position Filtering
```bash
# Remove tactical volatility
python src/position_filters.py \
  --input=binary_data/evals.bin \
  --output=binary_data/evals_quiet.bin \
  --filter=quiet_position
```

**Filtering criteria** (position is "quiet" if):
- ✅ No captures available
- ✅ No hanging pieces
- ✅ No checks possible
- ✅ No tactical threats pending
- ✅ No forks, pins in progress

**Implementation logic**:
```python
def is_quiet(board):
    # Check 1: Any captures available?
    if any(board.is_capture(move) for move in board.legal_moves):
        return False
    
    # Check 2: Position in check?
    if board.is_check():
        return False
    
    # Check 3: Hanging pieces?
    for square in board.squares:
        piece = board.piece_at(square)
        if piece and not is_defended(board, square):
            return False
    
    # Check 4: Threats after opponent move?
    for move in board.legal_moves:
        board.push(move)
        if has_captures_available(board):
            return False
        board.pop()
    
    return True  # Position is quiet
```

**Impact**: Removes ~30% of positions (the most chaotic ones)
**Benefit**: Network trains on stable evaluation tasks
**Output**: 30% smaller dataset (70GB → 49GB)

**Expected time**: 2-3 hours (position analysis)
**CPU impact**: High (board evaluation for each position)

#### Step 2.2: Evaluation Balancing
```bash
# Create 50-50 positive/negative split
python src/position_filters.py \
  --input=binary_data/evals_quiet.bin \
  --output=binary_data/evals_balanced.bin \
  --filter=evaluation_balance \
  --positive_ratio=0.5
```

**Goal**: 50% of positions favor white, 50% favor black

**Logic**:
```
For each position:
  if eval > 0 (white winning):   → positive_evals[]
  if eval < 0 (black winning):   → negative_evals[]

Balanced dataset:
  = positive_evals[:N/2] + negative_evals[:N/2]
```

**Impact**: Prevents trivial pattern learning
**Benefit**: Network learns full range of evaluations
**Constraint**: Also apply 40% material imbalance filter

#### Step 2.3: Material Distribution
```bash
# Apply 40% imbalanced / 60% balanced split
python src/position_filters.py \
  --input=binary_data/evals_balanced.bin \
  --output=binary_data/evals_distributed.bin \
  --filter=material_distribution \
  --imbalance_ratio=0.4
```

**Goal**: Represent both balanced and imbalanced positions

**Material imbalance criteria**:
- Imbalanced: Material difference >100cp (more than ~1 minor piece)
- Balanced: Material difference <100cp

**Distribution**:
```
40% positions with >100cp material difference
60% positions with <100cp material difference
```

**Impact**: Dataset contains diverse material situations
**Benefit**: Network learns both winning and defensive positions

#### Step 2.4: Syzygy Labeling
```bash
# Replace endgame evals with perfect Syzygy truth
python src/syzygy_integration.py \
  --input=binary_data/evals_distributed.bin \
  --output=binary_data/evals_syzygy.bin \
  --tablebase_path=/path/to/syzygy
```

**Process for each position**:
1. Check piece count
2. If ≤7 pieces: Query Syzygy WDL (ground truth)
3. Replace JSONL eval with perfect Syzygy eval
4. Store WDL statistics in record
5. Mark source as 'syzygy' vs 'original'

**Syzygy benefits**:
- **Endgame perfection**: Network learns perfect play
- **DTZ optimization**: 50-move rule awareness
- **WDL probabilities**: Win/draw/loss statistics
- **Ground truth labels**: Authoritative for ≤7 pieces

**Expected time**: 1-2 hours (tablebase probes)
**CPU impact**: Moderate (tablebase lookups)
**Prerequisite**: Download Syzygy tables (~180GB for all)
  - Minimal: 3-piece only (1KB, instant downloads)
  - Recommended: 3-5 piece (100MB, ~30 min download)
  - Full: 3-7 piece (300GB+, enterprise setups)

---

### Day 3: Dataset Construction

**Objective**: Create final training dataset with splits and weights

#### Step 3.1: Combine All Sources
```bash
# Merge PGNs + filtered evals + tokenized puzzles
python src/dataset_construction.py \
  --pgns=binary_data/pgns.bin \
  --evals=binary_data/evals_syzygy.bin \
  --puzzles=binary_data/puzzles_tokenized.bin \
  --output=binary_data/training_dataset.bin
```

**Merging strategy**:
- All evals included (filtering already applied)
- PGN games → convert to positions (board state each move)
- Puzzles → include with concept tokens
- Interleave to avoid batch imbalance

#### Step 3.2: Create Train/Val Splits
```bash
# 90-10 train/validation split
python src/dataset_construction.py \
  --input=binary_data/training_dataset.bin \
  --output_train=binary_data/train_dataset.bin \
  --output_val=binary_data/val_dataset.bin \
  --train_ratio=0.90
```

**Split strategy**:
- Random sampling (no time leakage)
- Balanced distribution (same ratio of quiet/balanced/imbalanced in both)
- Train: 90% (27GB)
- Val: 10% (3GB)

#### Step 3.3: Apply Multi-Task Weights
```bash
# Weight positions by task importance
python src/dataset_construction.py \
  --input=binary_data/train_dataset.bin \
  --output=binary_data/train_weighted.bin \
  --strategy=multi_task_learning
```

**Weighting scheme**:
```
For each position:
  strength_weight = 0.70  (70% priority)
  character_weight = 0.30 (30% priority)
  
  If position from personality engine:
    effective_weight = strength_weight + character_weight
  Else:
    effective_weight = strength_weight

Sampling: Positions with higher weight sampled more frequently
```

**Impact**: Network learns both generic strength + personal style

#### Step 3.4: Create Fast Index
```bash
# Create index for random access
python src/dataset_construction.py \
  --input=binary_data/train_weighted.bin \
  --output_index=binary_data/train_weighted.idx
```

**Index format**:
```
File: train_weighted.idx
[position_count (8 bytes)]
[offset_0 (8 bytes)]
[offset_1 (8 bytes)]
...
[offset_N (8 bytes)]

Allows O(1) random access to any position
```

---

### Day 4: Validation & Profiling

**Objective**: Verify data quality and profile I/O performance

#### Step 4.1: Data Integrity Check
```bash
# Verify all data is valid
python src/dataset_validation.py \
  --input=binary_data/train_weighted.bin \
  --validate_structure=true \
  --validate_chess=true
```

**Checks**:
- Record sizes correct (88 bytes)
- Evaluations in valid range (-32K to +32K)
- Material calculations correct
- Piece counts valid (0-32 pieces)
- Phase assignments correct
- Checksum verification

**Expected output**:
```
✅ 27,000,000 positions validated
   ✓ All record sizes: 88 bytes
   ✓ Evals in range: [-30000, 30000]
   ✓ Material balanced: OK
   ✓ Piece counts valid: OK
```

#### Step 4.2: I/O Performance Profile
```bash
# Measure loading speed
python src/dataset_profiling.py \
  --input=binary_data/train_weighted.bin \
  --test_samples=100000 \
  --batch_size=64
```

**Metrics to measure**:
- Sequential read speed (MB/sec)
- Random access latency (microseconds)
- Batch loading time (ms per 64 positions)
- Positions/sec throughput
- GPU transfer bandwidth

**Expected results**:
```
Sequential read: 200-400 MB/sec
Random access: 50-200 microseconds
Batch load (64): 2-5 milliseconds
Throughput: 12,000-32,000 positions/sec
GPU bandwidth: 8-10 GB/sec
```

#### Step 4.3: Statistics Report
```bash
# Generate comprehensive dataset stats
python src/dataset_analysis.py \
  --input=binary_data/train_weighted.bin \
  --output=reports/dataset_statistics.json
```

**Statistics to generate**:
```json
{
  "total_positions": 27000000,
  "evaluation_distribution": {
    "negative_50percent": true,
    "positive_50percent": true,
    "mean": 0.0,
    "std": 145.2
  },
  "material_distribution": {
    "imbalanced_40percent": true,
    "balanced_60percent": true
  },
  "phase_distribution": {
    "opening_percent": 35.2,
    "middlegame_percent": 45.1,
    "endgame_percent": 19.7
  },
  "quiet_positions_percent": 70.0,
  "syzygy_labeled_percent": 2.3,
  "file_size_gb": 27.0,
  "estimated_training_hours": 144.0
}
```

#### Step 4.4: Ready for Phase 1
```bash
# Final checklist
echo "Phase 0 Data Preparation Complete!"
echo "✅ 27GB optimized training dataset"
echo "✅ 90-10 train/val split"
echo "✅ All filtering applied"
echo "✅ Syzygy labeling complete"
echo "✅ I/O profiled and optimized"
echo ""
echo "Ready for Phase 1: Feature Expansion"
```

---

## Resource Requirements

### Hardware
- **Processor**: 8+ cores (parallel conversion)
- **RAM**: 16GB+ (buffered I/O)
- **Storage**: 120GB input + 30GB output = 150GB free space on SSD

### Software
```bash
pip install python-chess numpy struct
# Optional (for Syzygy):
pip install python-chess[syzygy]
```

### Time & Computation
| Step | Time | CPU | Disk I/O | Notes |
|------|------|-----|----------|-------|
| PGN conversion | 2-4 hr | High | High seq | Parallelizable |
| JSONL conversion | 3-5 hr | High | High seq | JSON parsing overhead |
| Puzzle tokenization | 30 min | Med | Med | Pattern recognition |
| Quiet filtering | 2-3 hr | High | High | Board analysis per pos |
| Evaluation balance | 30 min | Low | Med | In-memory sorting |
| Material distribution | 30 min | Low | Med | In-memory filtering |
| Syzygy labeling | 1-2 hr | Med | Med | Tablebase probes |
| **Total** | **~12-16 hr** | **Throughout** | **Throughout** | **Parallelizable** |

---

## Success Criteria for Phase 0

- [x] Format conversion speed: >50MB/sec sustained
- [ ] All 120GB converted to binary formats
- [ ] Quiet position filtering: ~30% reduction
- [ ] Evaluation balance: 50-50 pos/neg verified
- [ ] Material distribution: 40-60 imbalanced/balanced
- [ ] Syzygy labeling: Endgame positions ground-truthed
- [ ] Data integrity: 100% valid records
- [ ] I/O performance: >200MB/sec sequential read
- [ ] Random access: <200 microsecond latency
- [ ] Training dataset ready: 27GB optimized file

---

## Integration with Phase 1

**Phase 0 outputs** → **Phase 1 inputs**:
```
binary_data/train_weighted.bin   → Fast training loop
binary_data/val_weighted.bin     → Validation set
binary_data/train_weighted.idx   → O(1) random access
reports/dataset_statistics.json  → Metadata reference
```

**Phase 1 training loop**:
```python
from training_data_loader import OptimizedDataLoader

loader = OptimizedDataLoader(
    dataset_file='binary_data/train_weighted.bin',
    index_file='binary_data/train_weighted.idx'
)

for epoch in range(20):
    for batch in loader.batch_generator(batch_size=64):
        # Training with 6000+ features on clean data
        # 12,000+ positions/sec loading speed
        # Quiet positions only
        # Balanced evaluations
        # Multi-task learning weights applied
```

---

## Timeline Summary

```
Day 1: Conversion (12-14 hours)
  2-4 hr: PGN → binary
  3-5 hr: JSONL → binary
  30 min: Puzzles → tokenized
  6-8 hr: Storage I/O (parallel)

Day 2: Filtering (4-6 hours)
  2-3 hr: Quiet position filtering
  1 hr: Evaluation balancing
  1 hr: Material distribution
  1-2 hr: Syzygy labeling

Day 3: Construction (2-3 hours)
  30 min: Combine sources
  1 hr: Create splits
  30 min: Apply weights
  30 min: Create index

Day 4: Validation (2-3 hours)
  1 hr: Data integrity check
  1 hr: I/O performance profiling
  30 min: Statistics generation
  30 min: Final checklist

TOTAL: 12-16 productive hours (some can run in parallel)
TOTAL WALL TIME: 2-3 days (with overnight batch jobs)
```

---

## Next Steps

1. ✅ [ARCHITECTURE DOCUMENTED]
2. 🔜 **START CONVERSION** (Day 1)
   - Download Syzygy tables (if using)
   - Allocate storage space
   - Start PGN → binary conversion
3. 🔜 **APPLY FILTERS** (Day 2)
   - Quiet position analysis
   - Evaluation balancing
4. 🔜 **CONSTRUCT DATASET** (Day 3)
   - Merge sources
   - Create splits
5. 🔜 **VALIDATE & PROFILE** (Day 4)
   - Verify data
   - Profile I/O
   - **→ PHASE 1 READY**

---

## Archive

**Phase 0 is the unseen foundation** of all performance improvements in Phases 1-5.

Without clean, efficiently-loaded data:
- ❌ Neural network trains on garbage
- ❌ Training is slow (I/O bottlenecked)
- ❌ Model learns tactical positions it shouldn't
- ❌ Evaluation is imbalanced

With Phase 0 complete:
- ✅ 27GB optimized dataset (500MB/sec load speed)
- ✅ Tactical volatility removed (quiet positions only)
- ✅ Balanced evaluations (50-50 pos/neg)
- ✅ Perfect endgames (Syzygy ground truth)
- ✅ Fast training (12K+ positions/sec)
- ✅ **Foundation for 1B parameter scaling**

---

**Status**: 🟢 READY FOR EXECUTION  
**Created**: 2026-06-09  
**Next Phase**: Phase 1 Feature Expansion (2 weeks)  
**Timeline**: Phase 0 → Phase 1 → Phase 2-5 (integrated pipeline)
