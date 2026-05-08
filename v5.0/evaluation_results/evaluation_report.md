# V7P3R AI v5.0 - Evaluation Report

**Generated**: 2026-05-07 21:18:12

---

## Overall Performance

| Metric | Value |
|--------|-------|
| **Total Loss** | 1.3666 |
| **Policy Loss** | 1.3645 |
| **Value Loss** | 0.0210 |

---

## Policy Head Metrics (Move Quality)

| Metric | Value | Target |
|--------|-------|--------|
| **Exact Match Accuracy** | 49.12% | >50% |
| **Top-2 Accuracy** (±1 grade) | 60.18% | >75% |
| **Top-3 Accuracy** (±2 grades) | 67.89% | >85% |

### Per-Grade Performance

| Grade | Accuracy | Sample Count |
|-------|----------|--------------|
| 0 | 78.68% | 5,684 |
| 1 | 0.00% | 1,042 |
| 2 | 0.00% | 1,437 |
| 3 | 0.10% | 2,070 |
| 4 | 5.54% | 3,482 |
| 5 | 71.18% | 9,384 |

### Confusion Matrix

```
     Predicted Grade
     0      1      2      3      4      5
0    4472      0      0      0     71   1141 
1     671      0      0      0     38    333 
2     879      0      0      0     89    469 
3    1151      0      0      2    107    810 
4    1621      0      0      1    193   1667 
5    2593      0      0      4    107   6680 
```

---

## Value Head Metrics (Position Evaluation)

| Metric | Value | Target |
|--------|-------|--------|
| **MAE** (Mean Absolute Error) | 0.0941 | <0.15 |
| **RMSE** (Root Mean Squared Error) | 0.2279 | <0.20 |
| **Correlation** | 0.6446 | >0.80 |

**Note**: Value predictions are in [-1, 1] range (multiply by 10000 for centipawns)

---

## Interpretation

### Policy Head
- Model correctly predicts exact move quality **49.1%** of the time
- Within ±1 grade: **60.2%** (practical accuracy)
- Within ±2 grades: **67.9%** (near-miss tolerance)

### Value Head
- Average evaluation error: **941 centipawns**
- Position evaluation correlation: **0.645** (vs Stockfish)

---

## Baseline Comparison

| Metric | Baseline (Random) | Model | Improvement |
|--------|-------------------|-------|-------------|
| Policy Accuracy | 16.7% | 49.1% | 2.9x |
| Value MAE | ~0.30 | 0.094 | 3.2x better |

---

## Training Targets Status

❌ Policy Accuracy >50%
❌ Top-2 Accuracy >75%
✅ Value MAE <0.15

---

*End of Report*
