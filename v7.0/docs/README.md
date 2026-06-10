# V7P3R v7.0/v7.1 Documentation Index

## 🚀 Quick Start

**Want to start training right now?**

```bash
cd v7.0/src
python train_generational.py  # v7.1 generational training (recommended)
```

See [QUICK_START_V7.1.md](../QUICK_START_V7.1.md) for complete setup and configuration.

---

## 📚 Documentation Overview

### Training Guides

| Document | Purpose | Read If... |
|----------|---------|-----------|
| [QUICK_START_V7.1.md](../QUICK_START_V7.1.md) | Fast setup guide | You want to start training immediately |
| [V7.1_GENERATIONAL_TRAINING.md](V7.1_GENERATIONAL_TRAINING.md) | v7.1 complete spec | You need full technical details on v7.1 |
| [V7_TRAINING_WORKFLOW.md](V7_TRAINING_WORKFLOW.md) | Detailed workflow | You want to understand training stages |

### Architecture & Design

| Document | Purpose | Read If... |
|----------|---------|-----------|
| [V7_ARCHITECTURE.md](V7_ARCHITECTURE.md) | System architecture | You want to understand the neural network |
| [CHESS_AS_STORY_PHILOSOPHY.md](CHESS_AS_STORY_PHILOSOPHY.md) | Design philosophy | You want to know WHY we use phase-aware weighting |
| [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) | Implementation details | You're modifying the codebase |

### Specialized Topics

| Document | Purpose | Read If... |
|----------|---------|-----------|
| [V7_TABLEBASE_INTEGRATION.md](V7_TABLEBASE_INTEGRATION.md) | Endgame tablebases | You want perfect endgame play |
| [RETROGRADE_TRAP_SYSTEM.md](RETROGRADE_TRAP_SYSTEM.md) | Future feature | You're curious about planned features |

---

## 🎯 Which Version Should I Use?

### v7.1 Generational Training ⭐ **RECOMMENDED**

**Use this if:**
- ✅ You want meaningful win/loss metrics
- ✅ You need color-balanced evaluation
- ✅ You want better endgame conversion
- ✅ You're starting fresh

**Files:**
- `src/train_generational.py` - Main entry point
- `src/generational_trainer.py` - Core implementation

**Documentation:**
- [V7.1_GENERATIONAL_TRAINING.md](V7.1_GENERATIONAL_TRAINING.md)
- [QUICK_START_V7.1.md](../QUICK_START_V7.1.md)

### v7.0 Self-Play Training (Legacy)

**Use this if:**
- ❌ You need pure self-play (not recommended)
- ❌ You're reproducing old results

**Known Issues:**
- Win/loss metrics are meaningless (same model both sides)
- 78% of games hit max moves (poor endgame conversion)
- Massive color bias (0 White wins, 18 Black wins)

**Files:**
- `src/train_story_mode.py` - Main entry point
- `src/selfplay_trainer.py` - Core implementation

---

## 📊 Key Differences

| Metric | v7.0 (Legacy) | v7.1 (Current) |
|--------|---------------|----------------|
| **Opponent** | Same model | Previous best generation |
| **Win Metric** | % decisive games | % new beats old |
| **Color Balance** | Random | 50/50 enforced (3+3) |
| **Endgame SF** | 50% | **100%** ⬆️ |
| **Middlegame SF** | 10% | **20%** ⬆️ |
| **Max-Move Games** | 78% | ~30% (expected) |
| **White Win Rate** | 0% (bias!) | ~50% (expected) |
| **Training Style** | Self-play only | AlphaZero-style |

---

## 🏗️ Architecture Summary

### Neural Network
- **Input**: 51 features (comprehensive chess knowledge)
- **Hidden**: 3 layers (256→128→64 neurons)
- **Output**: 1 value (position evaluation)
- **Parameters**: 55,425 total

### Training Components
1. **Stockfish Oracle**: Evaluation supervisor (depth 15)
2. **Phase Manager**: Dynamic weight calculator
3. **Opening Book**: 12 aggressive opening lines
4. **Tablebase Oracle**: Perfect 5-piece endgame knowledge
5. **Personality Rewards**: Dark Forest Assassin style

### Weight Curve (v7.1)
```
Opening (1-10):        90% SF → Learn fundamentals
Early MG (11-20):      90% → 10% SF → Enter chaos
Deep MG (21-40):       20% SF → CONTROLLED CHAOS
Late MG (41-60):       20% → 80% SF → Return to precision
Endgame (61+):         100% SF → Perfect technique
Tablebase (≤7 pieces): 100% Perfect
```

---

## 🎓 Learning Path

### Beginner
1. Read [QUICK_START_V7.1.md](../QUICK_START_V7.1.md)
2. Run `python train_generational.py`
3. Observe first generation training

### Intermediate
1. Read [CHESS_AS_STORY_PHILOSOPHY.md](CHESS_AS_STORY_PHILOSOPHY.md)
2. Understand phase-aware weighting
3. Experiment with weight curve parameters

### Advanced
1. Read [V7_ARCHITECTURE.md](V7_ARCHITECTURE.md)
2. Read [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
3. Modify network architecture or training loop
4. Implement custom personality profiles

---

## 🔧 Common Tasks

### Start Training
```bash
cd v7.0/src
python train_generational.py
```

### Monitor Progress
```bash
cd ../training/v7_generational
cat generation_history.json  # View win/loss records
```

### Visualize Weight Curve
```bash
cd v7.0/src
python -c "from phase_manager import DynamicWeightCalculator; calc = DynamicWeightCalculator(); print(calc.visualize_weight_curve())"
```

### Load Best Model
```python
import torch
from network import create_v7_network

model, trainer = create_v7_network()
model.load_state_dict(torch.load("../training/v7_generational/best_model.pt"))
```

---

## 🐛 Troubleshooting

### Training Issues
- **Max-move games still high**: See [V7.1_GENERATIONAL_TRAINING.md](V7.1_GENERATIONAL_TRAINING.md) troubleshooting section
- **All generations rejected**: Increase self-play games (50 → 200)
- **Color bias persists**: Check evaluation match alternation logic

### Technical Issues
- **Import errors**: Ensure all dependencies installed (`pip install -r requirements.txt`)
- **Tablebase errors**: Check path in `train_generational.py` or disable
- **Memory errors**: Reduce batch size or buffer size

---

## 📈 Expected Results

### v7.1 Improvements (vs v7.0 baseline)
- **Max-move games**: 78% → ~30%
- **Color balance**: 0-100% → ~50-50%
- **Win rate trends**: Should increase across generations
- **Acceptance rate**: ~60-80% of generations

### Success Indicators
- ✅ Steady improvement (gen 5 > gen 3 > gen 1)
- ✅ Color-balanced wins (both White and Black)
- ✅ Decisive endgames (K+P vs K in <50 moves)
- ✅ Consistent training loss decrease

---

## 🚀 Next Steps

### After Successful v7.1 Training
1. Tournament test `best_model.pt` against Stockfish
2. Integrate into V7P3R chess engine
3. Train on specific openings (custom opening book)
4. Increase to 20-50 generations
5. Try different personality profiles

### Contributing
- Experiment with different weight curves
- Create custom personality profiles
- Implement new features (see [RETROGRADE_TRAP_SYSTEM.md](RETROGRADE_TRAP_SYSTEM.md))
- Share training results and insights

---

## 📝 Version History

- **v7.0** (June 2026): Initial "Chess as Story" implementation
- **v7.1** (June 2026): Generational training + revised weight curve ⭐ **Current**

---

## 🔗 Related Projects

- **V7P3R Chess Engine**: Production UCI engine (integrates trained models)
- **VPR Engine**: Alternative lightweight engine
- **Engine Tester**: Tournament testing framework

---

For questions or issues, refer to specific documentation files or check training logs in `training/v7_generational/`.
