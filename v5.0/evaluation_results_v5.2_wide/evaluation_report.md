# V7P3R AI v5.0 - Evaluation Report

**Generated**: 2026-05-10 19:56:26

---

## Overall Performance

| Metric | Value |
|--------|-------|
| **Total Loss** | 1.4255 |
| **Policy Loss** | 1.4240 |
| **Value Loss** | 0.0150 |

---

## Policy Head Metrics (Move Quality)

| Metric | Value | Target |
|--------|-------|--------|
| **Exact Match Accuracy** | 45.10% | >50% |
| **Top-2 Accuracy** (±1 grade) | 64.81% | >75% |
| **Top-3 Accuracy** (±2 grades) | 76.31% | >85% |

### Per-Grade Performance

| Grade | Accuracy | Sample Count |
|-------|----------|--------------|
| 0 | 74.49% | 5,684 |
| 1 | 13.92% | 1,042 |
| 2 | 55.30% | 4,025 |
| 3 | 20.55% | 4,214 |
| 4 | 6.95% | 5,566 |
| 5 | 56.93% | 11,841 |

### Confusion Matrix

```
     Predicted Grade
     0      1      2      3      4      5
0    4234    512    100     45     40    753 
1     592    145     74     11     12    208 
2     687    211   2226    445     64    392 
3     919    214   1349    866    186    680 
4    1218    327   1095    867    387   1672 
5    2146    617    992    872    473   6741 
```

---

## Value Head Metrics (Position Evaluation)

| Metric | Value | Target |
|--------|-------|--------|
| **MAE** (Mean Absolute Error) | 0.0744 | <0.15 |
| **RMSE** (Root Mean Squared Error) | 0.1928 | <0.20 |
| **Correlation** | 0.7201 | >0.80 |

**Note**: Value predictions are in [-1, 1] range (multiply by 10000 for centipawns)

---

## Interpretation

### Policy Head
- Model correctly predicts exact move quality **45.1%** of the time
- Within ±1 grade: **64.8%** (practical accuracy)
- Within ±2 grades: **76.3%** (near-miss tolerance)

### Value Head
- Average evaluation error: **744 centipawns**
- Position evaluation correlation: **0.720** (vs Stockfish)

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
