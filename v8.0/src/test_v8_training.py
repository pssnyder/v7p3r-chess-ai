"""Quick smoke test of v8.0 training system"""

import torch
import logging
from train_v8 import V8GenerationalTrainer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Quick test: 1 generation, 5 games
config = {
    'num_generations': 1,
    'games_per_generation': 5,
    'batch_size': 16,
    'max_moves_per_game': 100,
    'tablebase_path': r'E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\pgn_training_data\pgn_data_endgames\3-4-5_pieces_Syzygy\3-4-5'
}

trainer = V8GenerationalTrainer(**config)
trainer.train()

print("\n✓ Smoke test passed! Ready for full training.")
