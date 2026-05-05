"""
PyTorch Dataset for Stage 2 Corrective Training

Loads dual-learning corrective examples (negative + positive)
with proper handling of move weights and example types.

Author: V7P3RAI Development Team
Date: 2026-04-24
"""

import json
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Dict, Tuple


class CorrectiveDataset(Dataset):
    """Dataset for corrective training with dual learning pattern."""
    
    def __init__(self, json_path: str):
        """
        Initialize corrective dataset.
        
        Args:
            json_path: Path to corrective_dataset.json
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.metadata = data['metadata']
        self.examples = data['examples']
        
        print(f"Loaded {len(self.examples)} corrective examples")
        print(f"  Negative (Avoid): {self.metadata['num_negative']}")
        print(f"  Positive (Exploit): {self.metadata['num_positive']}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        """
        Get a single training example.
        
        Returns dict with:
            position_features: (690,) tensor
            moves: (N, 3) tensor [from_sq, to_sq, promotion]
            move_weights: (N,) tensor - training weights for each move
            move_scores: (N,) tensor - normalized centipawn scores
            example_type: "negative" or "positive"
            move_classification: "blunder", "mistake", "inaccuracy", "good"
        """
        example = self.examples[idx]
        
        # Position features (690-dim)
        position_features = torch.tensor(
            example['position_features'], 
            dtype=torch.float32
        )
        
        # Moves: [(uci, san, promotion_str), ...]
        moves_data = example['moves']
        moves = []
        for uci, san, promo_str in moves_data:
            # Parse UCI move
            from_sq = self._square_from_uci(uci[:2])
            to_sq = self._square_from_uci(uci[2:4])
            promo = int(promo_str)
            moves.append([from_sq, to_sq, promo])
        
        moves_tensor = torch.tensor(moves, dtype=torch.long)
        
        # Move weights (training importance)
        move_weights = torch.tensor(
            example['move_weights'],
            dtype=torch.float32
        )
        
        # Move scores (normalized evaluations)
        move_scores = torch.tensor(
            example['move_scores'],
            dtype=torch.float32
        )
        # Normalize to [0, 1] range for training
        # Using tanh normalization: score / 1000 clamped to [-1, 1] then to [0, 1]
        move_scores = torch.tanh(move_scores / 1000.0) * 0.5 + 0.5
        
        return {
            'position_features': position_features,
            'moves': moves_tensor,
            'move_weights': move_weights,
            'move_scores': move_scores,
            'example_type': example['example_type'],
            'move_classification': example['move_classification'],
            'context': example['context']
        }
    
    def _square_from_uci(self, square_str: str) -> int:
        """Convert UCI square string (e.g., 'e2') to square index (0-63)."""
        file = ord(square_str[0]) - ord('a')
        rank = int(square_str[1]) - 1
        return rank * 8 + file


def custom_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function to handle variable-length move lists.
    
    Pads all move lists in the batch to the same length.
    """
    batch_size = len(batch)
    
    # Find max number of moves in this batch
    max_moves = max(item['moves'].shape[0] for item in batch)
    
    # Stack position features (fixed size)
    position_features = torch.stack([item['position_features'] for item in batch])
    
    # Pad moves, weights, scores to max_moves
    moves_padded = torch.zeros(batch_size, max_moves, 3, dtype=torch.long)
    weights_padded = torch.zeros(batch_size, max_moves, dtype=torch.float32)
    scores_padded = torch.zeros(batch_size, max_moves, dtype=torch.float32)
    move_masks = torch.zeros(batch_size, max_moves, dtype=torch.bool)
    
    for i, item in enumerate(batch):
        num_moves = item['moves'].shape[0]
        moves_padded[i, :num_moves] = item['moves']
        weights_padded[i, :num_moves] = item['move_weights']
        scores_padded[i, :num_moves] = item['move_scores']
        move_masks[i, :num_moves] = True
    
    # Collect metadata
    example_types = [item['example_type'] for item in batch]
    classifications = [item['move_classification'] for item in batch]
    contexts = [item['context'] for item in batch]
    
    return {
        'position_features': position_features,
        'moves': moves_padded,
        'move_weights': weights_padded,
        'move_scores': scores_padded,
        'move_masks': move_masks,  # Plural to match model expectation
        'example_types': example_types,
        'move_classifications': classifications,
        'contexts': contexts
    }
