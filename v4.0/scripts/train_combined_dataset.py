#!/usr/bin/env python3
"""
Stage 2.5: Continue training from Stage 1 with expanded dataset.

Strategy:
- Load Stage 1 checkpoint (86.6% top-5 on puzzles)
- Add historical game positions (V7P3R + opponent moves)
- Continue training with mixed dataset (curriculum learning)
- Preserve puzzle performance while learning game patterns
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import json
from pathlib import Path
import argparse
from typing import Dict, List, Any
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.move_ordering_network import MoveOrderingNetwork
from src.training.puzzle_dataset import MoveOrderingDataset, custom_collate_fn


class GamePositionDataset(Dataset):
    """Dataset for game positions (same format as puzzle dataset)."""
    
    def __init__(self, game_positions_path: str):
        """Load game positions from JSON."""
        print(f"Loading game positions from {game_positions_path}...")
        
        with open(game_positions_path, 'r') as f:
            data = json.load(f)
        
        self.positions = data['positions']
        self.metadata = data.get('metadata', {})
        
        print(f"Loaded {len(self.positions)} game positions")
        
        # Print phase distribution
        opening = sum(1 for p in self.positions if p['game_phase'] == 'opening')
        middlegame = sum(1 for p in self.positions if p['game_phase'] == 'middlegame')
        endgame = sum(1 for p in self.positions if p['game_phase'] == 'endgame')
        
        print(f"  Opening: {opening} ({opening/len(self.positions)*100:.1f}%)")
        print(f"  Middlegame: {middlegame} ({middlegame/len(self.positions)*100:.1f}%)")
        print(f"  Endgame: {endgame} ({endgame/len(self.positions)*100:.1f}%)")
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        """Get a single position (same format as puzzle dataset)."""
        position = self.positions[idx]
        
        # Convert to tensor format (same as MoveOrderingDataset)
        position_features = torch.tensor(position['position_features'], dtype=torch.float32)
        
        # Encode moves
        moves = []
        move_weights = []
        move_scores = []
        
        for move_info in position['top_moves']:
            # Parse UCI move to from_sq, to_sq, promotion
            uci = move_info['uci']
            from_sq = self._square_from_uci(uci[:2])
            to_sq = self._square_from_uci(uci[2:4])
            
            # Promotion
            if len(uci) == 5:
                promo_map = {'q': 1, 'r': 2, 'b': 3, 'n': 4}
                promotion = promo_map.get(uci[4], 0)
            else:
                promotion = 0
            
            moves.append([from_sq, to_sq, promotion])
            move_weights.append(move_info['weight'])
            
            # Normalize score to 0-1 range using tanh
            normalized_score = (torch.tanh(torch.tensor(move_info['score'] / 1000.0)) + 1) / 2
            move_scores.append(normalized_score.item())
        
        moves = torch.tensor(moves, dtype=torch.long)
        move_weights = torch.tensor(move_weights, dtype=torch.float32)
        move_scores = torch.tensor(move_scores, dtype=torch.float32)
        
        return {
            'position_features': position_features,
            'moves': moves,
            'move_weights': move_weights,
            'move_scores': move_scores,
            'themes': torch.zeros(57, dtype=torch.float32),  # No themes for game positions
            'rating': torch.tensor([1500.0 / 3000.0], dtype=torch.float32),  # Match puzzle format
            'puzzle_id': f'game_{idx}',  # Synthetic ID for game positions
            'game_phase': position['game_phase']  # For analysis
        }
    
    def _square_from_uci(self, square_str: str) -> int:
        """Convert UCI square string (e.g., 'e2') to 0-63 index."""
        file = ord(square_str[0]) - ord('a')  # 0-7
        rank = int(square_str[1]) - 1  # 0-7
        return rank * 8 + file


def load_stage1_checkpoint(model_path: str) -> tuple:
    """
    Load Stage 1 model checkpoint.
    
    Returns:
        (model, epoch, val_top5_accuracy)
    """
    print(f"\n📦 Loading Stage 1 checkpoint from {model_path}...")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create model
    model = MoveOrderingNetwork(num_themes=57)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    epoch = checkpoint.get('epoch', 'unknown')
    val_top5 = checkpoint.get('val_top5_accuracy', 'unknown')
    
    print(f"   ✅ Loaded from epoch {epoch}")
    print(f"   📊 Stage 1 Top-5 Accuracy: {val_top5}")
    
    return model, epoch, val_top5


def main():
    parser = argparse.ArgumentParser(
        description='Stage 2.5: Continue training with puzzles + game positions'
    )
    parser.add_argument('--stage1-checkpoint', type=str,
                       default='models/stage1_themes/best_checkpoint.pt',
                       help='Path to Stage 1 checkpoint')
    parser.add_argument('--puzzle-data', type=str,
                       default='data/preprocessed_puzzles/enriched_puzzles_compact_20260420_003909.json',
                       help='Path to puzzle dataset')
    parser.add_argument('--game-data', type=str,
                       default='data/stage2_games/historical_positions.json',
                       help='Path to game position dataset')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=5e-5,
                       help='Learning rate (lower for continued training)')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation split ratio')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    print("🚀 V7P3RAI Stage 2.5: Continued Training (Puzzles + Games)")
    print("=" * 70)
    print(f"📦 Stage 1 Checkpoint: {args.stage1_checkpoint}")
    print(f"📂 Puzzle Dataset: {args.puzzle_data}")
    print(f"📂 Game Dataset: {args.game_data}")
    print(f"⚙️  Batch Size: {args.batch_size}")
    print(f"📊 Epochs: {args.num_epochs}")
    print(f"🎯 Device: {args.device}")
    print("=" * 70)
    
    # Check if game data exists
    if not Path(args.game_data).exists():
        print(f"\n⚠️  Game position dataset not found: {args.game_data}")
        print("   Run extract_game_positions.py first to create this dataset")
        print("\n   For now, continuing with puzzles only (same as Stage 1)...")
        
        # Load puzzle dataset only
        print(f"\n📂 Loading puzzle dataset...")
        puzzle_dataset = MoveOrderingDataset(args.puzzle_data)
        combined_dataset = puzzle_dataset
        
    else:
        # Load both datasets
        print(f"\n📂 Loading puzzle dataset...")
        puzzle_dataset = MoveOrderingDataset(args.puzzle_data)
        
        print(f"\n📂 Loading game position dataset...")
        game_dataset = GamePositionDataset(args.game_data)
        
        # Combine datasets
        combined_dataset = ConcatDataset([puzzle_dataset, game_dataset])
        
        print(f"\n✅ Combined dataset: {len(combined_dataset)} positions")
        print(f"   Puzzles: {len(puzzle_dataset)} ({len(puzzle_dataset)/len(combined_dataset)*100:.1f}%)")
        print(f"   Games: {len(game_dataset)} ({len(game_dataset)/len(combined_dataset)*100:.1f}%)")
    
    # Split train/val
    total_size = len(combined_dataset)
    val_size = int(total_size * args.val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        combined_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\n   Train: {len(train_dataset):,} positions")
    print(f"   Val: {len(val_dataset):,} positions")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=0
    )
    
    # Load Stage 1 checkpoint
    model, start_epoch, stage1_top5 = load_stage1_checkpoint(args.stage1_checkpoint)
    model = model.to(args.device)
    
    # Continue training
    print(f"\n🚀 Continuing training from Stage 1...")
    print(f"   Starting from epoch {start_epoch}")
    print(f"   Target: Maintain {stage1_top5} top-5 accuracy on puzzles")
    print(f"   Goal: Learn game patterns while preserving puzzle knowledge")
    
    # Import training script
    from train_move_ordering import MoveOrderingTrainer
    
    trainer = MoveOrderingTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        device=args.device,
        checkpoint_dir='models/stage2_combined'
    )
    
    # Train
    history = trainer.train(
        num_epochs=args.num_epochs,
        early_stopping_patience=args.patience
    )
    
    print(f"\n✅ Stage 2.5 training complete!")
    print(f"   Best model: models/stage2_combined/best_checkpoint.pt")
    print(f"   Next step: Test against V7P3R baseline")


if __name__ == '__main__':
    main()
