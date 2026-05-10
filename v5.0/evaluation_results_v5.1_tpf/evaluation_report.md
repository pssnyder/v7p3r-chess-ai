# V7P3R AI v5.0 - Evaluation Report

**Generated**: 2026-05-09 19:39:12

---

## Overall Performance

| Metric | Value |
|--------|-------|
| **Total Loss** | 1.4270 |
| **Policy Loss** | 1.4254 |
| **Value Loss** | 0.0158 |

---

## Policy Head Metrics (Move Quality)

| Metric | Value | Target |
|--------|-------|--------|
| **Exact Match Accuracy** | 45.15% | >50% |
| **Top-2 Accuracy** (±1 grade) | 65.38% | >75% |
| **Top-3 Accuracy** (±2 grades) | 76.92% | >85% |

### Per-Grade Performance

| Grade | Accuracy | Sample Count |
|-------|----------|--------------|
| 0 | 70.30% | 5,684 |
| 1 | 14.30% | 1,042 |
| 2 | 56.47% | 4,025 |
| 3 | 19.72% | 4,214 |
| 4 | 9.04% | 5,566 |
| 5 | 57.96% | 11,841 |

### Confusion Matrix

```
     Predicted Grade
     0      1      2      3      4      5
0    3996    649    179     18     49    793 
1     558    149     89      7     10    229 
2     624    211   2273    428     85    404 
3     847    242   1338    831    239    717 
4    1114    354   1167    765    503   1663 
5    1839    690   1126    712    611   6863 
```

---

## Value Head Metrics (Position Evaluation)

| Metric | Value | Target |
|--------|-------|--------|
| **MAE** (Mean Absolute Error) | 0.0745 | <0.15 |
| **RMSE** (Root Mean Squared Error) | 0.1978 | <0.20 |
| **Correlation** | 0.7019 | >0.80 |

**Note**: Value predictions are in [-1, 1] range (multiply by 10000 for centipawns)

---

## Interpretation

### Policy Head
- Model correctly predicts exact move quality **45.1%** of the time
- Within ±1 grade: **65.4%** (practical accuracy)
- Within ±2 grades: **76.9%** (near-miss tolerance)

### Value Head
- Average evaluation error: **745 centipawns**
- Position evaluation correlation: **0.702** (vs Stockfish)

---

## Baseline Comparison

| Metric | Baseline (Random) | Model | Improvement |
|--------|-------------------|-------|-------------|
| Policy Accuracy | 16.7% | 45.1% | 2.7x |
| Value MAE | ~0.30 | 0.074 | 4.0x better |

---

## Training Targets Status

❌ Policy Accuracy >50%
❌ Top-2 Accuracy >75%
✅ Value MAE <0.15

---

*End of Report*
