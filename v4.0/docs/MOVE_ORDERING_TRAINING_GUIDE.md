# Move Ordering Training Pipeline - Quick Start Guide

## 🎯 Goal: Train a model to rank chess moves by quality

This pipeline trains a neural network on **4 million chess puzzles** enriched with **Stockfish's top-10 moves** to learn move ordering for V7P3R chess engine.

---

## 📋 Prerequisites

1. **Python environment** with PyTorch, python-chess installed
2. **Stockfish engine** for analyzing positions
3. **4M puzzle database** from Lichess (SQLite format)
4. **GPU recommended** (RTX 4070 Ti or better for reasonable training time)

---

## ⚡ Quick Test (15-30 minutes)

Test the entire pipeline on 1,000 puzzles:

```powershell
cd "e:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v4.0"
python scripts/quick_start_training.py
```

This will:
1. ✅ Preprocess 1,000 puzzles with Stockfish
2. ✅ Train model for 5 epochs
3. ✅ Validate top-k accuracy

**Expected results**: ~60-70% top-5 accuracy after 5 epochs on small dataset

---

## 🚀 Full Pipeline (3-5 days total)

### Step 1: Data Preprocessing (8-12 hours)

Convert 4M puzzles into Stockfish-enriched training dataset:

```powershell
python scripts/preprocess_puzzles_with_stockfish.py `
  --puzzle-db "E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-tester\databases\puzzles.db" `
  --stockfish "E:\Programming Stuff\Chess Engines\Tournament Engines\Stockfish\stockfish-windows-x86-64-avx2.exe" `
  --max-puzzles 4000000 `
  --rating-min 600 `
  --rating-max 2500 `
  --stockfish-time 1.0 `
  --top-n 10 `
  --output-dir data/preprocessed_puzzles
```

**What this does:**
- Loads puzzles from SQLite database
- For each position, runs Stockfish to find top-10 moves
- Saves enriched dataset with move rankings and scores
- Creates checkpoints every 5,000 puzzles (resumable if interrupted)

**Output:**
- `enriched_puzzles_<timestamp>.json` (~2-3 GB full dataset)
- `enriched_puzzles_compact_<timestamp>.json` (~1.5-2 GB compact version)
- `dataset_stats_<timestamp>.txt` (statistics)

**Hardware:**
- RTX 4070 Ti: ~8-10 hours
- CPU only: ~24-36 hours

---

### Step 2: Training (2-4 days)

Train move ordering network on enriched dataset:

```powershell
# Find the generated dataset file
$dataset = Get-ChildItem data/preprocessed_puzzles/enriched_puzzles_compact_*.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1

python scripts/train_move_ordering.py `
  --data-path $dataset.FullName `
  --batch-size 64 `
  --num-epochs 100 `
  --learning-rate 0.001 `
  --early-stopping-patience 10 `
  --num-workers 8 `
  --checkpoint-dir models/stage1_themes/full_4M
```

**Training configuration:**
- Batch size: 64 (reduce to 32 if GPU OOM)
- Effective batch: 64 × 1 = 64 (can increase with gradient accumulation)
- Optimizer: AdamW (lr=0.001, weight_decay=0.01)
- Scheduler: Cosine annealing with warm restarts
- Mixed precision: Enabled (automatic on CUDA)

**Expected performance:**
- Epoch time: ~20-30 minutes (RTX 4070 Ti, 3.2M training samples)
- Top-5 accuracy target: **>85%**
- Top-10 accuracy target: **>95%**
- Theme classification: **>90%** (multi-label)

**Checkpoints saved:**
- `latest_checkpoint.pt` - Most recent epoch
- `best_checkpoint.pt` - Best validation loss

**Monitoring:**
- Training progress: Live progress bars with metrics
- Validation: Every epoch with top-k accuracy
- Early stopping: Stops if no improvement for 10 epochs

---

### Step 3: Validation & Testing

Validate trained model on test set:

```powershell
python scripts/validate_agents.py `
  --model models/stage1_themes/full_4M/best_checkpoint.pt `
  --data-path $dataset.FullName `
  --batch-size 64
```

**Metrics:**
- Top-1 accuracy: How often best move is #1 prediction
- Top-5 accuracy: How often best move is in top-5
- Top-10 accuracy: How often best move is in top-10
- Theme accuracy: Multi-label classification accuracy
- Inference speed: Average time per position

---

## 📊 Expected Results

### After Full Training (4M puzzles, 100 epochs):

| Metric | Target | Baseline (random) |
|--------|--------|-------------------|
| Top-1 accuracy | >30% | 10% |
| Top-5 accuracy | >85% | 50% |
| Top-10 accuracy | >95% | 80% |
| Theme accuracy | >90% | 2% |
| Inference time | <5ms | N/A |

### By Rating Range:

| Rating | Top-5 Accuracy | Notes |
|--------|----------------|-------|
| 600-1000 | >90% | Tactical puzzles |
| 1000-1500 | >85% | Mixed tactics |
| 1500-2000 | >80% | Complex positions |
| 2000-2500 | >70% | Advanced tactics |

---

## 🔧 Troubleshooting

### GPU Out of Memory
```powershell
# Reduce batch size
python scripts/train_move_ordering.py --batch-size 32

# Or use gradient accumulation
python scripts/train_move_ordering.py --batch-size 32 --gradient-accumulation 2
```

### Slow Preprocessing
```powershell
# Reduce Stockfish time
python scripts/preprocess_puzzles_with_stockfish.py --stockfish-time 0.5

# Process in smaller chunks
python scripts/preprocess_puzzles_with_stockfish.py --max-puzzles 1000000
```

### Poor Accuracy
- Check if ChessStateExtractor is properly ported from v3.0
- Verify Stockfish is running correctly (test manually)
- Increase training time (more epochs)
- Increase model capacity (edit `move_ordering_network.py`)

### Checkpoint Recovery
```powershell
# Resume training from checkpoint
python scripts/train_move_ordering.py `
  --resume models/stage1_themes/full_4M/latest_checkpoint.pt
```

---

## 🎓 Understanding the Architecture

### Data Flow:
```
Raw Puzzle → Stockfish Analysis → Enriched Puzzle
  (FEN)         (top-10 moves)        (training sample)
     ↓               ↓                      ↓
Position Features  Move Encodings    Multi-task Targets
  (690-dim)        (from/to squares)  (rankings + themes)
```

### Model Architecture:
```
Position (690-dim)
    ↓
PositionEncoder (3-layer MLP)
    ↓
Position Embedding (512-dim) ───┬─→ Theme Classifier → Themes (50-dim)
    ↓                            │
    + Move Embeddings (64-dim)   │
    ↓                            │
Attention-based Ranking ←────────┘
    ↓
Move Scores (0-1 per move)
```

### Loss Functions:
1. **Move Ranking Loss**: Weighted MSE between predicted and target scores
   - Best move weighted 1.0
   - 2nd best weighted 0.8
   - Etc. (exponential decay)

2. **Theme Classification Loss**: Binary cross-entropy (multi-label)
   - 50 possible themes per position
   - Multiple themes can be active

**Combined Loss**: `0.7 × ranking_loss + 0.3 × theme_loss`

---

## 📁 Output Files

### After Preprocessing:
```
data/preprocessed_puzzles/
├── enriched_puzzles_20260418_120000.json         # Full dataset with metadata
├── enriched_puzzles_compact_20260418_120000.json # Compact version for training
├── dataset_stats_20260418_120000.txt             # Statistics report
└── checkpoint_*.json                              # Resumable checkpoints
```

### After Training:
```
models/stage1_themes/full_4M/
├── latest_checkpoint.pt    # Most recent epoch
├── best_checkpoint.pt      # Best validation loss
└── training_log.txt        # Training history
```

---

## 🔄 Next Steps (Stage 2-4)

Once Stage 1 completes with >85% top-5 accuracy:

1. **Stage 2**: Historical game analysis for move correction
2. **Stage 3**: Opening book and endgame tablebase integration
3. **Stage 4**: Reinforcement learning for middlegame tactics

---

## 💡 Tips for Success

1. **Start with quick test** (`quick_start_training.py`) to verify pipeline
2. **Monitor GPU usage** (`nvidia-smi`) during training
3. **Save checkpoints frequently** (every 5K puzzles during preprocessing)
4. **Use compact JSON** for training (loads faster than full JSON)
5. **Validate on held-out test set** before deploying

---

## 📞 Support

If issues arise:
1. Check logs in terminal output
2. Review dataset statistics file
3. Test with smaller subset (`--max-puzzles 1000`)
4. Verify Stockfish is working (`stockfish.exe` → `uci` → `isready`)

---

**Ready to start? Run `quick_start_training.py` to test the pipeline!** 🚀
