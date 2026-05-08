# Stage 2 Training Optimization Guide

## Production-Grade Configuration for V7P3R Primary Engine Deployment

This document details the advanced training optimizations implemented for Stage 2 corrective training, designed to achieve performance suitable for replacing V7P3R's traditional minimax alpha-beta search with AI-based move selection.

---

## 🎯 Performance Targets

### Baseline (Stage 1)
- **Top-5 Accuracy**: 86.6% (100K puzzles)
- **Top-10 Accuracy**: 100%
- **Training Data**: Tactical puzzles only

### Stage 2 Targets (Post-Correction)
- **Top-5 Accuracy**: ≥84% (maintain Stage 1, acceptable 2-3% regression for correction benefits)
- **Blunder Avoidance**: ≥90% (critical for deployment)
- **Top-1 Accuracy**: ≥65% (best move selected first)
- **Average Best Rank**: ≤1.5 (best move in top-2 on average)
- **ELO Impact**: +100-200 over baseline V7P3R

### V7P3R Primary Engine Deployment Criteria
- **Blunder Avoidance**: ≥95% (higher than human threshold)
- **Top-3 Accuracy**: ≥85% (strong tactical awareness)
- **Live Game Performance**: 60%+ win rate vs V7P3R v18.4 baseline
- **Blunders/Game**: <3.0 (lower than current ~5-6)
- **Time Efficiency**: <20ms inference per position
- **Consistency**: <10% game-to-game performance variance

---

## 🔧 Advanced Optimizations Implemented

### 1. Learning Rate Schedule: Warmup + Cosine Annealing

**What**: Linear warmup (10% of training) followed by cosine decay to near-zero.

**Why**: 
- **Warmup prevents instability** when fine-tuning from Stage 1 weights
- **Cosine decay** allows model to explore better local minima at end of training
- **Avoids abrupt learning rate changes** that can damage pre-trained features

**Implementation**:
```python
warmup_steps = total_steps * 0.1
lr(step) = {
    base_lr * (step / warmup_steps)                           if step < warmup_steps
    min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(π * progress))  otherwise
}
```

**Expected Benefit**: +2-5% validation accuracy, smoother convergence

---

### 2. Gradient Accumulation (Effective Batch Size 128)

**What**: Accumulate gradients over 4 batches before optimizer step.

**Why**:
- **Larger effective batch = more stable gradients** especially for diverse corrective data
- **CPU memory limitation**: Can't fit 128 examples in single batch
- **Better generalization**: Larger batches reduce noise in gradient estimates

**Implementation**:
```python
# Batch size 32 × 4 accumulation = effective batch 128
for i, batch in enumerate(train_loader):
    loss = compute_loss(batch) / gradient_accumulation_steps
    loss.backward()
    
    if (i + 1) % 4 == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Expected Benefit**: +3-7% blunder avoidance, reduced overfitting

---

### 3. Exponential Moving Average (EMA) of Parameters

**What**: Maintain shadow copy of weights as exponential moving average.

**Why**:
- **Smooths weight updates** reducing sensitivity to batch variation
- **Better generalization** especially on validation/test data
- **Common in production models** (Stable Diffusion, BERT fine-tuning)

**Implementation**:
```python
shadow[param] = 0.9995 * shadow[param] + 0.0005 * param
# Use shadow weights for inference, original for training
```

**Expected Benefit**: +1-3% validation metrics, more stable deployment

---

### 4. Blunder-Focused Loss Weighting (5x Penalty)

**What**: Amplify loss weight for blunder positions (eval_drop ≥300cp).

**Why**:
- **Blunders lose games** far more than inaccuracies
- **Historical data shows** V7P3R's losses primarily from 3-4 critical blunders
- **Standard weighting treats all mistakes equally** (suboptimal for real games)

**Implementation**:
```python
# Standard correction loss: weight=0.0 for blunders
# Optimized: weight=5.0 penalty instead
blunder_mask = (weights < 0.1)
weights[blunder_mask] = 5.0  # Amplified penalty
```

**Expected Benefit**: +15-25% blunder avoidance (most critical metric)

---

### 5. Margin-Based Ranking Loss

**What**: Ensure best move is ranked at least `margin` above all alternatives.

**Why**:
- **Simple MSE allows** best move to be only slightly higher
- **Margin forces clear separation** improving top-1 accuracy
- **Better reflects chess reality**: Best move should be obviously superior

**Implementation**:
```python
# Best move should be margin=0.1 above all others
ranking_loss = max(0, margin - (best_score - other_scores))
```

**Expected Benefit**: +5-10% top-1 accuracy, clearer move preferences

---

### 6. Label Smoothing (0.1)

**What**: Soften hard targets (0/1) to (0.05/0.95).

**Why**:
- **Prevents overconfidence** on training examples
- **Better calibration** for uncertainty in chess positions
- **Reduces overfitting** to specific Stockfish evaluations

**Implementation**:
```python
target = target * (1 - smoothing) + 0.5 * smoothing
# 1.0 → 0.95, 0.0 → 0.05
```

**Expected Benefit**: +2-4% generalization to unseen positions

---

### 7. Multi-Metric Early Stopping

**What**: Track both blunder avoidance AND validation loss for stopping.

**Why**:
- **Single metric can be misleading** (low loss but poor blunder avoidance)
- **Blunder avoidance is primary goal** for V7P3R deployment
- **Dual tracking ensures balance** between accuracy and safety

**Implementation**:
```python
# Improved if EITHER metric improves
if blunder_avoidance > best_blunder OR val_loss < best_loss:
    save_checkpoint()
    reset_patience()
```

**Expected Benefit**: Find better deployment checkpoint, avoid premature stopping

---

### 8. Comprehensive Validation Suite

**What**: Track 9 metrics (top-1/3/5/10, avg rank, blunder avoidance, losses).

**Why**:
- **Different metrics reveal different aspects** of model quality
- **Average rank** shows consistency beyond top-k
- **Multiple top-k** shows ranking quality at different cutoffs
- **Blunder avoidance** is deployment-critical

**Metrics**:
1. `val_loss`: Overall quality
2. `val_correction_loss`: Historical learning
3. `val_ranking_loss`: Move ordering quality
4. `val_top1_accuracy`: Best move ranked #1
5. `val_top3_accuracy`: Best move in top-3
6. `val_top5_accuracy`: Best move in top-5 (Stage 1 comparison)
7. `val_top10_accuracy`: Best move in top-10
8. `val_blunder_avoidance`: Avoid ranking blunders high (CRITICAL)
9. `val_avg_best_rank`: Average rank of best move

**Expected Benefit**: Better deployment decision, comprehensive model understanding

---

## 📊 Configuration Comparison

| Parameter | Original | Optimized | Rationale |
|-----------|----------|-----------|-----------|
| Learning Rate | 5e-5 | 1e-4 → 1e-7 | Warmup + decay for stability |
| LR Schedule | Cosine only | Warmup + Cosine | Prevent early instability |
| Effective Batch | 32 | 128 (32×4) | Gradient accumulation |
| Correction Weight | 2.0 | 3.0 | Emphasize correction over ranking |
| Blunder Weight | N/A | 5.0 | Focus on critical errors |
| Ranking Loss | MSE | Margin-based | Force clear separation |
| Label Smoothing | 0.0 | 0.1 | Better generalization |
| EMA | No | Yes (0.9995) | Stable predictions |
| Early Stopping | Val loss | Multi-metric | Blunder + loss |
| Validation Metrics | 3 | 9 | Comprehensive quality |
| Max Epochs | 50 | 100 | More search space with better stopping |
| Patience | 15 | 20 | Allow more exploration |

---

## 🚀 Expected Training Timeline

### Resource Estimates
- **Dataset**: 23,676 examples (11,838 negative + 11,838 positive)
- **Train/Val Split**: 21,308 train / 2,368 val
- **Effective Batch**: 128 (32 × 4 accumulation)
- **Steps/Epoch**: ~167 batches
- **Total Steps**: ~16,700 (100 epochs × 167)
- **Warmup Steps**: ~1,670 (10%)

### Time Estimates (CPU)
- **Per Batch Forward**: ~0.3s (32 examples, 690-dim features)
- **Per Batch Backward**: ~0.4s (gradient computation)
- **Per Epoch**: ~120s (167 batches × 0.7s)
- **Total Training**: ~3-4 hours (100 epochs, with early stopping likely ~40-60 epochs)

### Memory Usage
- **Model**: ~6.4 MB (1.6M parameters × 4 bytes)
- **Optimizer**: ~19.2 MB (3× model for AdamW states)
- **EMA**: ~6.4 MB (shadow parameters)
- **Batch Data**: ~85 MB (32 × 690 features × 10 moves × 4 bytes)
- **Total**: ~120 MB (well within CPU limits)

---

## 📈 Success Criteria & Decision Tree

### Validation Results Interpretation

#### Scenario 1: Excellent Performance (Deploy as Primary)
- ✅ Blunder Avoidance ≥95%
- ✅ Top-5 Accuracy ≥85%
- ✅ Top-1 Accuracy ≥70%
- ✅ Avg Best Rank ≤1.3

**Action**: 
1. Run 100-game tournament vs V7P3R v18.4
2. If win rate ≥60%, deploy as primary engine
3. Integrate into v7p3r.py as first-line move selector
4. Keep traditional search as fallback

#### Scenario 2: Good Performance (Hybrid Deployment)
- ✅ Blunder Avoidance ≥90%
- ✅ Top-5 Accuracy ≥82%
- ⚠️ Top-1 Accuracy 60-70%

**Action**:
1. Use AI for move candidate generation (top-5)
2. Traditional search evaluates AI candidates
3. Hybrid approach: AI narrows search, minimax validates

#### Scenario 3: Moderate Performance (Agent Layer Only)
- ⚠️ Blunder Avoidance 85-90%
- ⚠️ Top-5 Accuracy 78-82%
- ⚠️ Top-1 Accuracy 55-60%

**Action**:
1. Deploy as v7p3r-corrector agent (Stage 2 plan)
2. Use for historical position validation
3. Continue to Stage 3 (opening/endgame specialists)

#### Scenario 4: Needs Improvement (Iterate)
- ❌ Blunder Avoidance <85%
- ❌ Top-5 Accuracy <78%

**Action**:
1. Analyze failure modes (which positions fail?)
2. Consider more training data (expand from 1,693 to 5,000+ games)
3. Architectural improvements (deeper network, attention variants)
4. Revisit data quality (Stockfish analysis time, position selection)

---

## 🔍 Post-Training Validation Protocol

### Step 1: Checkpoint Evaluation
```bash
# Load best model
python scripts/evaluate_corrective_model.py \
    --model models/stage2_corrective_optimized/best_model.pt \
    --test-data data/stage2_positions/critical_positions.json \
    --output validation_results.json
```

**Metrics**:
- Blunder avoidance on held-out critical positions
- Top-k accuracy on held-out games
- Performance by position type (tactical, positional, endgame, opening)

### Step 2: Live Game Testing
```bash
# 50-game rapid tournament
python scripts/tournament_vs_baseline.py \
    --engine1 ai_corrected \
    --engine2 v7p3r_v18.4 \
    --games 50 \
    --time-control 5+4
```

**Metrics**:
- Win/Loss/Draw ratios
- Blunders per game
- Average centipawn loss (CPL)
- Time usage patterns
- Opening phase performance
- Endgame conversion rate

### Step 3: Position Replay Analysis
```bash
# Replay historical blunder positions
python scripts/replay_blunder_positions.py \
    --positions data/stage2_positions/critical_positions.json \
    --model models/stage2_corrective_optimized/best_model.pt \
    --compare-to v7p3r_v18.4
```

**Metrics**:
- Blunder repetition rate (should be <10%)
- Best move selection on historical failures
- Move preference shift from V7P3R baseline

---

## 🎓 Lessons from Training Optimizations

### From 100K Puzzle Training (Stage 1)
- **Progressive difficulty worked**: 1K→10K→100K scaling showed consistent improvement
- **Top-10 ceiling hit 100%**: Model can rank correctly, but top-1 needs work
- **Label quality matters**: Stockfish analysis time (0.5s) was sufficient
- **Early stopping patience=15 optimal**: More patience allowed best epoch at 95/100

### From Historical Game Analysis (Stage 2 Prep)
- **Quality > Quantity**: 1,693 losses more valuable than 5,000 mediocre games
- **Context matters**: Final 5 moves + tactical blunders captured 70%+ of critical errors
- **Color balance**: 47.5% White / 52.5% Black shows no bias
- **Top opponents**: R0bspierre, NexaStrat, HardRok exploited specific weaknesses

### From Corrective Dataset Generation
- **Dual learning pattern**: Negative (avoid) + Positive (exploit) = 50/50 split ideal
- **Classification helps**: Blunder/Mistake/Inaccuracy granularity allows weighted training
- **Stockfish multipv=10**: Top-10 moves sufficient, diminishing returns beyond
- **Position after mistake**: Simpler than board inversion, avoids illegal states

---

## 🔬 Experimental Variations (Future Work)

### If Initial Results Are Below Target

#### 1. Curriculum Learning
- **Phase 1 (10 epochs)**: Train only on blunders (weight=0.0)
- **Phase 2 (10 epochs)**: Add mistakes (weight<0.3)
- **Phase 3 (30 epochs)**: Full dataset with all classifications

#### 2. Data Augmentation
- **Horizontal mirroring**: Generate mirror positions for color symmetry
- **Tempo variations**: Same position with/without extra move
- **Partial game replays**: Extended sequences around critical positions

#### 3. Ensemble Methods
- **Train 3-5 models** with different random seeds
- **Majority voting** for move selection
- **Confidence weighting** based on agreement

#### 4. Architecture Enhancements
- **Residual connections** in position encoder
- **Multi-head attention** instead of single-head
- **Deeper networks**: 5-6 layers instead of 3
- **Larger hidden dims**: 768-1024 instead of 512

#### 5. Extended Training Data
- **Expand to 5,000+ historical games** (all losses, not just checkmates)
- **Include V7P3R draws in losing positions** (flagged positions)
- **Add opponent perspective wins** (learn from their best games)

---

## 📝 Training Command Reference

### Standard Training (Current)
```bash
python scripts/train_corrective_optimized.py
```

### Custom Configuration Examples

#### High-Performance Mode (Longer Training)
```bash
python scripts/train_corrective_optimized.py \
    --num-epochs 150 \
    --early-stopping-patience 30 \
    --learning-rate 2e-4 \
    --blunder-weight 7.0
```

#### Fast Iteration Mode (Quick Test)
```bash
python scripts/train_corrective_optimized.py \
    --num-epochs 20 \
    --early-stopping-patience 5 \
    --gradient-accumulation-steps 2 \
    --batch-size 64
```

#### Conservative Mode (Preserve Stage 1)
```bash
python scripts/train_corrective_optimized.py \
    --correction-weight 1.5 \
    --ranking-weight 2.0 \
    --learning-rate 5e-5
```

---

## ✅ Pre-Flight Checklist

Before starting training, verify:

- [x] Stage 1 model exists: `models/stage1_themes/best_model.pt`
- [x] Corrective dataset exists: `data/stage2_training/corrective_dataset.json`
- [x] Dataset validation passed: 23,676 examples, 50/50 split
- [x] PyTorch installed: `python -c "import torch; print(torch.__version__)"`
- [x] Disk space available: ~500 MB for checkpoints
- [x] Time allocated: 3-4 hours for full training
- [x] GPU/CPU target confirmed: CPU (no CUDA required)
- [x] Output directory writable: `models/stage2_corrective_optimized/`

**Ready to train!** Run: `run_stage2_training_optimized.bat`

---

## 🎯 Post-Training Next Steps

### Immediate Actions
1. Review training curves in `history` dict
2. Compare best vs latest vs EMA checkpoints
3. Run validation protocol (Step 1-3 above)
4. Document results in `STAGE_2_RESULTS.md`

### If Performance Targets Met
1. Integrate into V7P3R orchestrator
2. Create hybrid search option
3. Run 100-game tournament
4. Monitor production deployment
5. Proceed to Stage 3 (opening/endgame specialists)

### If Performance Needs Improvement
1. Analyze failure modes
2. Try experimental variations
3. Expand training data
4. Consider architectural changes
5. Re-run with adjusted hyperparameters

---

**Remember**: The goal is not just high accuracy, but **reliable blunder avoidance in real games**. A model with 80% top-5 but 98% blunder avoidance may outperform 90% top-5 with 85% blunder avoidance in tournament play.

**Focus**: Blunder Avoidance > Top-1 Accuracy > Top-5 Accuracy > Val Loss
