# V7P3R Chess AI v2.0 🏆

A sophisticated chess engine using genetic algorithms and GPU-accelerated neural networks.

## Features

- 🧠 **GPU-Accelerated Neural Networks** (PyTorch/CUDA)
- 🧬 **Genetic Algorithm Training** 
- ⚔️ **Advanced Tactical Evaluation** (Bounty System)
- 🎯 **Tournament-Ready UCI Engine**
- 📊 **Cumulative Learning** across training sessions
- 🔥 **Aggressive Tactical Style** (1400-1800 ELO estimated)

## Project Structure

```
v7p3r-chess-ai/
├── src/                        # 📦 Main source code
│   ├── core/                   # ♟️ Core chess logic and AI
│   ├── models/                 # 🧠 Neural network architectures
│   ├── training/               # 🧬 Genetic algorithm trainers
│   ├── evaluation/             # ⚔️ Position evaluation systems
│   ├── tournament/             # 🏆 UCI engine packaging
│   ├── utils/                  # 🔧 Monitoring and utilities
│   └── tests/                  # 🧪 Tests and demos
├── config/                     # ⚙️ Configuration files
├── scripts/                    # 🎮 Game playing scripts
├── data/                       # 📊 Training data and analysis
├── models/                     # 💾 Trained neural networks
├── docs/                       # 📖 Documentation
├── tournament_packages/        # 📦 Generated tournament engines
└── README.md                   # 📝 This file
```

## Quick Start

### 1. Installation
```bash
git clone https://github.com/pssnyder/v7p3r-chess-ai.git
cd v7p3r-chess-ai
pip install -r requirements.txt
```

### 2. Play Against the AI
```bash
python scripts/play_game.py
```

### 3. Train a New Model
```bash
# Basic training session
python src/training/train_v7p3r_v2.py

# Extended tournament training (2-3 hours)
python src/utils/extended_training_session.py
```

### 4. Create Tournament Engine
```bash
# Package the latest model for Arena Chess
python src/tournament/package_tournament_engine.py
```

## Tournament Integration

Your V7P3R engine is UCI-compatible and ready for Arena Chess GUI:

1. **Run packaging script** to create tournament engine
2. **Install in Arena**: Engines → Install New Engine
3. **Select**: `v7p3r_tournament_engine.py` from the generated package
4. **Compete** against other engines!

## Training System

### Genetic Algorithm Features
- **Population-based evolution** with 32 individuals
- **GPU acceleration** for fast fitness evaluation
- **Incremental training** builds on previous best models
- **Bounty system** rewards tactical and positional play
- **Memory optimization** for large neural networks

### Training Configurations
- **Tactical**: High aggression, tactical focus
- **Balanced**: All-around play style  
- **Positional**: Strategic, long-term planning
- **Tournament**: Optimized for competitive play

## Model Architecture

- **Input**: 816-dimensional chess position encoding
- **Network**: LSTM with 256 hidden units, 3 layers
- **Output**: 128-dimensional move evaluation
- **Training**: Genetic algorithm with bounty-based fitness
- **Hardware**: Optimized for CUDA GPUs

## Performance

- **Estimated Rating**: 1400-1800 ELO
- **Playing Style**: Tactical and aggressive
- **Strengths**: Piece coordination, tactical patterns, mate threats
- **Best Time Controls**: 10+3 to 15+10 minutes

## Development

### Key Components

- **`src/core/chess_core.py`**: Core chess logic and board evaluation
- **`src/models/v7p3r_gpu_model.py`**: GPU-accelerated LSTM model
- **`src/training/v7p3r_gpu_genetic_trainer_clean.py`**: Main genetic trainer
- **`src/evaluation/v7p3r_bounty_system.py`**: Tactical evaluation system
- **`src/tournament/package_tournament_engine.py`**: UCI engine packager

### Recent Improvements (v2.0)
- ✅ GPU acceleration with PyTorch/CUDA
- ✅ Genetic algorithm training system
- ✅ Advanced bounty-based evaluation
- ✅ Incremental/cumulative training
- ✅ Tournament packaging automation
- ✅ Robust error handling and architecture validation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper testing
4. Submit a pull request

## License

This project is open source. See LICENSE file for details.

## Tournament Results

Track your engine's performance:

| Date | Tournament | Opponent | Time Control | Result | Rating |
|------|------------|----------|--------------|--------|--------|
| 2025-09-02 | Arena Test | Crafty | 10+3 | W | ~1500 |

---

**Built with ❤️ for competitive chess AI**

*Ready to dominate the Arena tournaments! 🏆♟️🚀*