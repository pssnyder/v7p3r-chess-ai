# V7P3R Chess AI v2.0

Enhanced implementation with GPU acceleration, genetic algorithms, and tournament-ready packaging.

## Features
- GPU-accelerated training with CUDA support
- Genetic algorithm evolution with population-based learning
- Bounty-based fitness system for tactical evaluation
- Tournament engine packaging (UCI compatible)
- Multi-opponent training system
- Non-deterministic evaluation for continued learning

## Key Files
- `v7p3r_gpu_genetic_trainer_clean.py` - GPU genetic trainer
- `src/training/incremental_trainer.py` - Incremental training
- `scripts/enhanced_training_integration.py` - Enhanced training system
- `src/core/v7p3r_gpu_model.py` - GPU-optimized neural network

## Usage
```bash
# GPU genetic training
python v7p3r_gpu_genetic_trainer_clean.py

# Enhanced training with multi-opponents
python scripts/enhanced_training_integration.py

# Incremental training from best model
python -m src.training.incremental_trainer
```

## Models
GPU models are stored in `models/` with generation tracking.
Tournament packages in `tournament_packages/`.
