# High-Performance Chess Engine: Modular Development Plan (Token-Optimized)

**🔴 CRITICAL**: This plan minimizes Copilot tool usage. Each sprint is **one file, one task, one goal**.

---

## Sprint 1: Data Serialization & Streaming (3-4 days)

**Goal**: Transform 120GB library into streaming binary format (27GB compressed).  
**Output**: One production module (binary_format_converter.py) ready for Phase 0 execution.

### Day 1.1: PGN → Binary Conversion

| File | Input | Output | Target Performance |
|------|-------|--------|---------------------|
| `src/binary_format_converter.py` | 120GB PGN files | pgns.bin (1.5GB) | >50 MB/sec |
| **Dependencies** | chess, struct, hashlib, tqdm | 88-byte records | 2-byte move encoding |

**Specific Module**: Create `binary_format_converter.py` with:
- Class `BinaryPositionRecord` (88-byte struct)
- Method `pgn_to_binary(pgn_path, output_path)`
- Method `jsonl_to_binary(jsonl_path, output_path)`
- **Validation**: Benchmark on small sample (1000 positions), expect >50 MB/sec

**No Copilot tool calls**: Read existing code comments, implement once, validate.

### Day 1.2-1.3: Filtering & Dataset Streaming

| File | Purpose | Output |
|------|---------|--------|
| `src/position_filters.py` | Filter quiet positions, balance evals, apply material distribution | filtered.bin (24GB) |
| `src/pytorch_dataset.py` | IterableDataset for streaming | Ready for Phase 1 |

**Validation**: Run on 1GB sample, verify no memory overflow.

**No Copilot tool calls**: Copy patterns from existing codebase, test locally.

---

## Sprint 2: Architecture & Feature Engineering (HalfKA) (3-5 days)

**Goal**: Implement HalfKA feature generator (Python) and validate on sample data.  
**Output**: One feature extraction module ready for Phase 0.5 → Phase 1 pipeline.

### Day 2.1-2.2: Python HalfKA Feature Generator

| File | Purpose | Output |
|------|---------|--------|
| `src/halfdka_features.py` | (Piece, Square, King) → index mapping | Feature indices (sparse) |
| **Sparse Mapping** | 45,056 features per side (vs 55 original) | King buckets: 8-32 zones |

**Implementation**: Single Python module with:
- Function `get_halfdka_index(piece, square, king_square)` → u16
- Function `get_active_features(board)` → List[u16]
- King bucket mapping (precomputed 32-zone table)
- **Validation**: Test on 1000 positions, verify ~30-32 features active per position

**Decision Gate**: If performance acceptable (<100ms per batch of 1000), proceed. Otherwise, implement C++ in Phase 0.5.

### Day 2.3-2.4: Perspective Accumulator Design

Create `src/accumulator_architecture.py`:
- Dual accumulators (white/black perspective)
- 1024-2048 neuron size (tunable)
- ClippedReLU activation (0, 1 bounded)
- **Test**: Verify perspective symmetry preserves evaluation

### Network Architecture

```
45K sparse inputs (HalfKA per perspective)
    ↓ [Incremental update: only 2-3 features change per move]
Perspective Accumulators (1024-2048 neurons each)
    ↓ [Cached white/black symmetry]
ClippedReLU (0, 1 bounded for INT8 later)
    ↓
Hidden layers (128 → 32 neurons)
    ↓ [Split into 3 heads]
    ├─ Strength (MSE, 70%)
    ├─ Character (CE, 20%)
    └─ WDL (CE, 10%)
```

---

## Sprint 3: Training Loop & Personality Preservation (2 weeks)

**Goal**: Working PyTorch training on HalfKA features with 3-signal loss.  
**Output**: Trained model checkpoint (v1) + ELO baseline measurement.

### Day 3.1-3.2: Multi-Signal Loss Function

Create `src/training_loss.py`:
```python
class MultiSignalLoss(nn.Module):
    def __init__(self, strength_weight=0.7, character_weight=0.2, wdl_weight=0.1):
        self.strength_loss = nn.MSELoss()  # vs Lichess evals
        self.character_loss = nn.CrossEntropyLoss()  # vs engine moves
        self.wdl_loss = nn.CrossEntropyLoss()  # vs Syzygy WDL
    
    def forward(self, predictions, evals, moves, wdls):
        loss_strength = self.strength_loss(predictions[:, 0], evals)
        loss_character = self.character_loss(predictions[:, 1:], moves)
        loss_wdl = self.wdl_loss(predictions[:, -3:], wdls)
        return (0.7 * loss_strength + 0.2 * loss_character + 0.1 * loss_wdl)
```

**Validation**: Verify all three losses are decreasing independently.

### Day 3.3-3.4: Training Loop

Create `src/train.py`:
- DataLoader from Phase 1 IterableDataset
- Gradient accumulation (16-32 steps)
- Learning rate scheduler (cosine annealing)
- Checkpoint saving every N steps
- **Stop condition**: ELO measurement every epoch (via tournament)

### Day 3.5-3.6: Monitoring & Validation

Use existing `monitoring_performance_tracker.py`:
- GPU memory usage
- Training throughput (pos/sec)
- Loss curves (separate plots for 3 signals)
- **Do NOT use Copilot**: Run locally, analyze logs, iterate.

---

## Sprint 4: Syzygy Integration & Production Scaling (1 week)

**Goal**: Perfect endgame labeling + INT8 quantization for production.  
**Output**: Production-ready model (v2) with Syzygy ground truth.

### Day 4.1-4.2: Syzygy Integration

Use existing `src/syzygy_integration.py`:
- Probe positions ≤7 pieces with Fathom API
- Replace JSONL evals with WDL/DTZ ground truth
- Track source (Syzygy vs eval) in metadata
- **Validation**: Check 5-piece endgame accuracy (should be 100%)

### Day 4.3-4.4: INT8 Quantization

Create `src/quantize_model.py`:
- Scale weights by constant factor (e.g., 600)
- Verify ClippedReLU preserves bounds
- Test inference on quantized model
- **Target**: <1% ELO loss vs FP32

### Day 4.5: Production Export

Export to ONNX + C++ loader (for Phase 4 C# integration):
- `models/v2_quantized.onnx`
- `src/onnx_loader.py` (PyTorch → ONNX)
- **Do NOT implement C++**: Keep for Phase 0.5 decision.

---

## Token-Efficiency Rules (Mandatory)

### ✅ What to Do
1. **Work file-by-file**: One module per day max
2. **Validate locally**: Run tests in terminal, don't ask Copilot
3. **Copy patterns**: Reuse code structure from existing codebase
4. **One tool call per task**: No exploratory searches
5. **Batch edits**: If multiple changes needed, use multi_replace

### ❌ What NOT to Do
1. **No semantic_search**: Search manually in existing files
2. **No file_search**: You know your codebase structure
3. **No runSubagent**: Do exploration yourself
4. **No multiple read_file calls**: Read once, understand, implement
5. **No ask questions**: Make decisions, implement, validate

### 📊 Token Budget Per Sprint
- **Sprint 1**: 1 file creation + 2 edits = 3 tool calls max
- **Sprint 2**: 2 file creations + testing = 4 tool calls max
- **Sprint 3**: 2 file creations + monitoring = 4 tool calls max
- **Sprint 4**: 2 file creations + export = 4 tool calls max

**Total**: ~15 tool calls for entire 4-week plan (vs 50+ if inefficient)