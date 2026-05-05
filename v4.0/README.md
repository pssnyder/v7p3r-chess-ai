# V7P3RAI v4.0 - Multi-Agent Chess AI Enhancement Layer

**Status**: 🚧 In Development  
**Current Stage**: Stage 1 - Pattern Recognition & Move Ordering

## Overview

V7P3RAI v4.0 is an advanced multi-agent AI system designed to enhance the V7P3R Chess Engine with specialized neural network agents trained on 4,000,000 chess puzzles, historical game analysis, and reinforcement learning.

### Architecture

```
V7P3R Chess Engine (Orchestrator)
│
├── V7P3R-Themes Agent (Stage 1) - Pattern recognition & move ordering
├── V7P3R-Corrector Agent (Stage 2) - Historical move validation
├── V7P3R-Opening Agent (Stage 3) - Opening book mastery
├── V7P3R-Endgame Agent (Stage 3) - Tablebase & mate detection
└── V7P3R-Tactics Agent (Stage 4) - RL middlegame specialist
```

## Performance Targets

| Metric | Baseline (v18.4) | v4.0 Target |
|--------|------------------|-------------|
| **Actual ELO** | 1400 | 1800-2000 |
| **Lichess ELO** | 1600 | 2000-2200 |
| **Puzzle Accuracy** | N/A | 95%+ |
| **Opening Win %** | 50% | 75%+ |
| **Endgame Win %** | 50% | 85%+ |

## Installation

### Prerequisites
- Python 3.11+
- CUDA-capable GPU (RTX 4070 Ti recommended)
- 32GB RAM recommended
- 100GB free disk space (for puzzle database)

### Setup

```bash
# Clone repository
cd "e:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai"

# Navigate to v4.0 directory
cd v4.0

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Data Setup

```bash
# Stage 1: Link 4M puzzle library
# Ensure puzzles are available in data/puzzles/4M_puzzle_library/

# Stage 2: Extract historical games
python scripts/stage2_analyze_historical.py --source "../game_records/Lichess V7P3R Bot/"

# Stage 3: Download tablebases (optional, ~150GB)
# Download Syzygy 6-piece tablebases to data/tablebases/syzygy_6piece/
```

## Quick Start

### Stage 1: Train Themes Agent

```bash
# Train on 4M puzzle library
python scripts/stage1_train_themes.py --config config/training_config.json

# Validate agent
python scripts/validate_agents.py --agent themes --test-puzzles 1000

# Integrate into engine
python scripts/deploy_to_engine.py --stage 1
```

### Testing

```bash
# Run all tests
pytest tests/

# Test specific agent
pytest tests/test_themes_agent.py -v

# Integration test with engine
pytest tests/test_integration.py
```

## Project Structure

```
v4.0/
├── src/
│   ├── agents/          # Specialized AI agents
│   ├── core/            # Core utilities & orchestration
│   ├── engine_integration/  # V7P3R engine integration
│   ├── training/        # Training pipelines
│   ├── models/          # Neural network architectures
│   └── utils/           # Helper utilities
├── config/              # Configuration files
├── data/                # Training data (puzzles, games, etc.)
├── models/              # Trained model checkpoints
├── scripts/             # Training & deployment scripts
├── tests/               # Unit & integration tests
└── docs/                # Documentation
```

## Development Stages

### ✅ Stage 0: Setup (Current)
- [x] Directory structure created
- [x] Dependencies configured
- [ ] Data pipelines tested
- [ ] Integration with v7p3r-chess-engine verified

### 🚧 Stage 1: Pattern Recognition (In Progress)
- [ ] 4M puzzle database prepared
- [ ] Theme classifier trained
- [ ] Move ranking network trained
- [ ] Move ordering integration complete

### 📋 Stage 2: Historical Analysis (Planned)
- Historical game extraction
- Stockfish validation
- Corrector agent training
- Move validation integration

### 📋 Stage 3: Opening & Endgame (Planned)
- Opening book preparation
- Tablebase integration
- Specialized agents trained
- Perfect play in openings/endgames

### 📋 Stage 4: Middlegame Augmentation (Planned)
- Self-play data generation
- RL training pipeline
- NN evaluation replacement
- Production deployment

## Configuration

### Training Configuration

Edit `config/training_config.json`:

```json
{
  "stage1": {
    "puzzle_database": "data/puzzles/4M_puzzle_library/",
    "batch_size": 64,
    "learning_rate": 0.001,
    "epochs": 100,
    "gpu_id": 0
  }
}
```

### Agent Configuration

Edit `config/agent_config.json`:

```json
{
  "themes_agent": {
    "enabled": true,
    "model_path": "models/stage1_themes/final_model.pth",
    "inference_timeout_ms": 5,
    "fallback_enabled": true
  }
}
```

## Performance Monitoring

### Training Metrics
- Use TensorBoard: `tensorboard --logdir=models/stage1_themes/logs`
- Or W&B (optional): Set `WANDB_API_KEY` environment variable

### Validation
```bash
# Run validation suite
python scripts/validate_agents.py --all

# Specific stage validation
python scripts/validate_agents.py --stage 1 --detailed
```

## Integration with V7P3R Chess Engine

V4.0 agents integrate seamlessly with V7P3R Chess Engine v18.4+:

```python
from v7p3r_agent_orchestrator import AgentOrchestrator

# Initialize agents
orchestrator = AgentOrchestrator(config_path="config/agent_config.json")

# Engine integration
from v7p3r import V7P3REngine
engine = V7P3REngine()
engine.set_ai_orchestrator(orchestrator)

# Agents now enhance all engine decisions
move = engine.get_move(board, time_limit=3.0)
```

## Documentation

- [Master Plan](docs/V7P3RAI_V4.0_MASTER_PLAN.md) - Complete project vision
- [Stage 1 Guide](docs/STAGE1_IMPLEMENTATION.md) - Pattern recognition implementation
- [Agent Architecture](docs/AGENT_ARCHITECTURE.md) - Multi-agent design
- [Deployment Guide](docs/DEPLOYMENT_GUIDE.md) - Production deployment

## Troubleshooting

### GPU Not Detected
```bash
python -c "import torch; print(torch.cuda.is_available())"
# If False, check CUDA installation
```

### Puzzle Database Missing
```bash
# Verify puzzle database location
ls -la data/puzzles/4M_puzzle_library/
# Should contain puzzle files in PGN or CSV format
```

### Integration Test Failures
```bash
# Ensure v7p3r-chess-engine is in Python path
export PYTHONPATH="${PYTHONPATH}:../../v7p3r-chess-engine/src"
```

## Contributing

This is a personal research project, but feedback and suggestions are welcome!

## License

Proprietary - V7P3R Development Team

## Contact

For questions or support, open an issue in the v7p3r-chess-ai repository.

---

**Current Version**: 4.0.0-dev  
**Last Updated**: April 18, 2026  
**Next Milestone**: Stage 1 - 4M Puzzle Training Complete
