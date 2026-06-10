# V7P3R Chess AI v10.0 - Development Guide

Welcome to the v10.0 neural network training framework. This directory contains a **file-by-file** implementation roadmap for building a master-level chess engine.

## 🎯 Quick Start

### 1. Read the Files (In This Order)

1. **`.copilot-instructions.md`** - Rules for working efficiently with Copilot
2. **`ENVIRONMENT_SETUP.md`** - One-time terminal/VS Code setup (5 min)
3. **`V7P3R_vX_DEVELOPMENT PLAN.md`** - Daily task breakdown
4. **`TOKEN_EFFICIENCY_STRATEGY.md`** - How we're saving 75% of Copilot tokens

### 2. Activate Terminal (One Time)

```powershell
cd "e:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai"
.\.venv\Scripts\Activate.ps1
```

**Keep this terminal open for ALL work.**

### 3. Pick a Day and Start Coding

Example: **Day 1.1 of Sprint 1**

```
Task: Implement src/binary_format_converter.py
Methods: pgn_to_binary(), jsonl_to_binary()
Target: >50 MB/sec conversion speed
Test with: python -m pytest tests/test_binary_converter.py -v
```

Open the file, read existing patterns, code it, test locally. **No Copilot needed yet.**

---

## 📁 Directory Structure

```
v10.0/
├── src/                          # Source modules (implement one per day)
│   ├── __init__.py               # Package definition
│   ├── binary_format_converter.py # Sprint 1 Day 1: PGN → binary
│   ├── position_filters.py        # Sprint 1 Day 2: Filter positions
│   ├── pytorch_dataset.py         # Sprint 1 Day 3: Stream data
│   ├── halfdka_features.py        # Sprint 2 Day 1: 45K sparse features
│   ├── accumulator_architecture.py# Sprint 2 Day 3: NNUE architecture
│   ├── training_loss.py           # Sprint 3 Day 1: 3-signal loss
│   ├── train.py                   # Sprint 3 Day 3: Training loop
│   └── quantize_model.py          # Sprint 4 Day 3: INT8 quantization
│
├── tests/                        # Unit tests (test as you code)
│   ├── __init__.py
│   ├── test_binary_converter.py
│   ├── test_position_filters.py
│   ├── test_halfdka_features.py
│   ├── test_accumulator.py
│   ├── test_training_loss.py
│   ├── test_training.py
│   └── test_quantization.py
│
├── models/                       # Model checkpoints (generated during training)
│   ├── checkpoints/              # Training checkpoints
│   └── .gitkeep
│
├── data/                         # Binary datasets (generated in Sprint 1)
│   ├── filtered/                 # Filtered training data
│   └── .gitkeep
│
├── logs/                         # Training logs (generated in Sprint 3)
│   └── .gitkeep
│
├── .copilot-instructions.md      # Read FIRST - Copilot rules
├── ENVIRONMENT_SETUP.md          # Read SECOND - Terminal setup
├── V7P3R_vX_DEVELOPMENT PLAN.md  # Read THIRD - Daily roadmap
├── TOKEN_EFFICIENCY_STRATEGY.md  # Reference - Token savings
└── README.md                     # This file

```

---

## 📅 Four-Week Timeline

### **Week 1: Sprint 1 - Data Serialization**
- **Day 1.1**: Implement `binary_format_converter.py`
  - PGN → binary (88-byte records)
  - JSONL → binary (evaluations)
  - Target: >50 MB/sec

- **Day 1.2-1.3**: Implement `position_filters.py`
  - Filter quiet positions
  - Balance evaluations (50-50)
  - Apply material distribution

- **Day 1.3**: Implement `pytorch_dataset.py`
  - Stream 27GB binary data
  - IterableDataset pattern
  - No RAM overflow

**Exit condition**: 27GB filtered dataset ready

---

### **Week 2: Sprint 2 - Architecture**
- **Day 2.1-2.2**: Implement `halfdka_features.py`
  - Expand 55 → 45,056 features
  - King bucket mapping (weight sharing)
  - Sparse indexing

- **Day 2.3-2.4**: Implement `accumulator_architecture.py`
  - Dual accumulators (white/black)
  - Incremental updates (100x faster)
  - ClippedReLU activation

**Exit condition**: HalfKA feature generation tested

---

### **Week 3: Sprint 3 - Training**
- **Day 3.1-3.2**: Implement `training_loss.py`
  - Multi-signal loss (strength + character + WDL)
  - 70-20-10 weighting
  - Test loss computation

- **Day 3.3-3.4**: Implement `train.py`
  - Training loop orchestration
  - Gradient accumulation (16-32 steps)
  - Checkpoint management

- **Day 3.5-3.6**: Training execution
  - Phase 1: Binary data → HalfKA features
  - Phase 2: Feature training (1-2 weeks)
  - ELO measurements (after each epoch)

**Exit condition**: Training running, v1 model checkpoint saved

---

### **Week 4: Sprint 4 - Production**
- **Day 4.1-4.2**: Syzygy integration (existing code)
  - Replace eval with ground truth
  - 5-piece endgame accuracy

- **Day 4.3-4.4**: Implement `quantize_model.py`
  - INT8 conversion (4x model size reduction)
  - Inference speedup (2-4x)
  - <1% ELO loss

- **Day 4.5**: Export to ONNX
  - Cross-platform model format
  - Ready for C#/C++ integration
  - Production deployment

**Exit condition**: v2 quantized model exported, ready for Phase 1

---

## 🔧 Workflow (Every Day)

### Morning (10 minutes)

1. Open `.copilot-instructions.md` (rules reminder)
2. Read today's task from `V7P3R_vX_DEVELOPMENT PLAN.md`
3. Open VS Code to v10.0/
4. Activate terminal: `.\.venv\Scripts\Activate.ps1`

### Implementation (1-3 hours, NO Copilot)

1. Open existing module → understand patterns (5 min)
2. Create new module in `src/` (Ctrl+N)
3. Copy docstring + method signatures from task description
4. Write implementation (follow patterns)
5. Save (Ctrl+S, auto-formats)
6. See errors in Problems panel (Ctrl+Shift+M)
7. Fix errors locally (try yourself first)

### Testing (30 minutes)

```powershell
# Terminal: Run tests for the module
python -m pytest tests/test_binary_converter.py -v

# Result: Green ✅ or Red ❌
# If Red: See error, fix code, retry
```

### Before Asking Copilot (5 minutes)

- [ ] Ran tests locally (no syntax errors)
- [ ] Compared to existing code (is my structure similar?)
- [ ] Checked IDE problems (Ctrl+Shift+M shows issues)
- [ ] Tried for 15+ minutes (persistence!)

### Ask Copilot (Only If Stuck)

```
"I'm implementing pgn_to_binary() in src/binary_format_converter.py.
The test fails because [error message].
Here's my code: [paste function].
How do I fix this?"
```

**Result**: 1 replace_string_in_file, problem solved.

---

## 💡 Key Principles

| Principle | Why | Example |
|-----------|-----|---------|
| **One file per day** | Clarity, focus | "Day 1.1: Only binary_converter" |
| **Test locally first** | No token waste | `python -m pytest tests/` |
| **Copy patterns** | Consistency | Read 3 similar functions first |
| **Ask specific questions** | Efficient help | "Fix this [error]" not "Review code" |
| **Read dev plan first** | Know your task | "I'm on Day 2.1" |

---

## 📊 Success Metrics

### End of Week 1 ✅
- [ ] binary_format_converter.py working
- [ ] position_filters.py working
- [ ] pytorch_dataset.py working
- [ ] 27GB filtered dataset created
- [ ] <15 Copilot questions asked

### End of Week 2 ✅
- [ ] halfdka_features.py working
- [ ] accumulator_architecture.py working
- [ ] Feature generation tested
- [ ] Incremental updates working
- [ ] <15 Copilot questions asked

### End of Week 3 ✅
- [ ] training_loss.py working
- [ ] train.py working
- [ ] Training loop executing
- [ ] v1 model checkpoint saved
- [ ] ELO measurements recorded
- [ ] <15 Copilot questions asked

### End of Week 4 ✅
- [ ] Syzygy integration done
- [ ] quantize_model.py working
- [ ] v2 quantized model created
- [ ] ONNX export successful
- [ ] Ready for Phase 1
- [ ] <10 Copilot questions asked

**Total**: <55 tool calls (vs 200+ inefficient way) = 1,300+ tokens saved

---

## 🆘 Troubleshooting

### "Import error: module not found"
→ Check you're in v10.0/ directory when running tests
→ `cd "e:\Programming...\v10.0"` first

### "Copilot can't help without seeing the code"
→ Good! Read the code yourself, understand it, fix locally
→ Only ask if truly stuck 15+ minutes

### "Tests are failing"
→ Run locally first (you see the error)
→ Read error carefully (often self-explanatory)
→ Ask Copilot only if error is unclear

### "I finished early, what's next?"
→ Check .copilot-instructions.md for weekly checklist
→ Review your code (docstrings, type hints, tests)
→ Move to next day in dev plan

---

## 🚀 Next Step

1. **Read `.copilot-instructions.md`** (sections 1-3 only, 10 minutes)
2. **Read `ENVIRONMENT_SETUP.md`** (5-minute quick setup)
3. **Tomorrow**: Start Day 1.1 of Sprint 1

**Your first task**: Implement `src/binary_format_converter.py`

No Copilot needed. Just code. You've got this. 🎯

---

**Questions?** Check the instructions files above. They have answers.

**Status**: 🟢 **Framework complete. Ready to build.**
