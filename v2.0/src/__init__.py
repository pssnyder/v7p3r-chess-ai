"""
V7P3R Chess AI v2.0 - Source Package
====================================

A sophisticated chess AI using genetic algorithms and neural networks.

Modules:
--------
- core: Core chess logic and AI implementations
- models: Neural network model architectures  
- training: Genetic algorithm trainers and training scripts
- evaluation: Position evaluation and bounty systems
- tournament: Tournament packaging and UCI engine creation
- utils: Monitoring, visualization, and utility scripts
- tests: Test files and demo scripts

Author: V7P3R Development Team
Version: 2.0
"""

__version__ = "2.0"
__author__ = "Pat Snyder"

# Make commonly used classes easily accessible
try:
    from .core.chess_core import BoardEvaluator
    from .models.v7p3r_gpu_model import V7P3RGPU_LSTM
    from .evaluation.v7p3r_bounty_system import ExtendedBountyEvaluator
    from .training.v7p3r_gpu_genetic_trainer_clean import V7P3RGPUGeneticTrainer
except ImportError:
    # Handle import errors gracefully during development
    pass
