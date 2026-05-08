"""
Move Ordering Puzzle Dataset

Loads Stockfish-enriched puzzles for move ordering training.

Each training sample contains:
- Position features (690-dim from ChessStateExtractor)
- Top-N candidate moves (encoded as from-to squares)
- Move weights (for weighted loss)
- Theme labels (multi-hot vector)
- Puzzle rating (difficulty)

Supports:
- Train/val/test splitting
- Efficient batching with DataLoader
- Optional data augmentation (board flipping)
- Memory-mapped loading for large datasets
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
import chess

# Import ChessStateExtractor (will use placeholder for now, port from v3.0 later)
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.core.chess_state_extractor import ChessStateExtractor


class MoveOrderingDataset(Dataset):
    """PyTorch dataset for move ordering training"""
    
    # Theme vocabulary (must match preprocessor)
    ALL_THEMES = [
        'mate', 'mateIn1', 'mateIn2', 'mateIn3', 'mateIn4', 'mateIn5',
        'pin', 'fork', 'skewer', 'discoveredAttack', 'doubleCheck',
        'hangingPiece', 'trappedPiece', 'defensiveMove', 'deflection',
        'attraction', 'clearance', 'interference', 'intermezzo',
        'sacrifice', 'endgame', 'middlegame', 'opening',
        'advancedPawn', 'attackingF2F7', 'capturingDefender', 'exposedKing',
        'kingsideAttack', 'queensideAttack', 'pawnEndgame', 'rookEndgame',
        'bishopEndgame', 'knightEndgame', 'queenEndgame', 'queenRookEndgame',
        'advantage', 'crushing', 'equality', 'quiet', 'zugzwang',
        'short', 'long', 'veryLong', 'master', 'masterVsMaster',
        'superGM', 'oneMove', 'promotion', 'underPromotion',
        'castling', 'enPassant', 'smotheredMate', 'backRankMate',
        'doubleBishopMate', 'dovetailMate', 'arabianMate', 'anastasiaMate'
    ]
    
    def __init__(self,
                 data_path: str,
                 split: str = 'train',
                 split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                 augment: bool = False,
                 max_samples: Optional[int] = None,
                 seed: int = 42):
        """
        Args:
            data_path: Path to enriched puzzle JSON file
            split: 'train', 'val', or 'test'
            split_ratios: (train, val, test) ratios
            augment: Enable data augmentation (horizontal flip)
            max_samples: Limit dataset size (for testing)
            seed: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.split = split
        self.augment = augment and split == 'train'
        self.seed = seed
        
        # Initialize feature extractor
        self.feature_extractor = ChessStateExtractor()
        
        # Load data
        print(f"📂 Loading dataset from {self.data_path}...")
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        all_puzzles = data['puzzles']
        self.metadata = data['metadata']
        
        print(f"   Total puzzles loaded: {len(all_puzzles):,}")
        
        # Apply max_samples limit if specified
        if max_samples is not None and max_samples < len(all_puzzles):
            random.seed(seed)
            all_puzzles = random.sample(all_puzzles, max_samples)
            print(f"   Limited to: {len(all_puzzles):,} samples")
        
        # Split data
        random.seed(seed)
        random.shuffle(all_puzzles)
        
        train_end = int(len(all_puzzles) * split_ratios[0])
        val_end = train_end + int(len(all_puzzles) * split_ratios[1])
        
        if split == 'train':
            self.puzzles = all_puzzles[:train_end]
        elif split == 'val':
            self.puzzles = all_puzzles[train_end:val_end]
        else:  # test
            self.puzzles = all_puzzles[val_end:]
        
        print(f"   {split.upper()} split: {len(self.puzzles):,} puzzles")
        
        # Build theme vocabulary
        self.theme_to_idx = {theme: idx for idx, theme in enumerate(self.ALL_THEMES)}
        self.num_themes = len(self.ALL_THEMES)
    
    def __len__(self) -> int:
        return len(self.puzzles)
    
    def encode_move(self, board: chess.Board, move_uci: str) -> torch.Tensor:
        """
        Encode a move as from-square + to-square indices
        Returns: [from_square (0-63), to_square (0-63), promotion (0-4)]
        
        Promotion encoding: 0=none, 1=queen, 2=rook, 3=bishop, 4=knight
        """
        try:
            move = chess.Move.from_uci(move_uci)
            
            from_square = move.from_square
            to_square = move.to_square
            
            # Encode promotion
            promotion = 0
            if move.promotion:
                if move.promotion == chess.QUEEN:
                    promotion = 1
                elif move.promotion == chess.ROOK:
                    promotion = 2
                elif move.promotion == chess.BISHOP:
                    promotion = 3
                elif move.promotion == chess.KNIGHT:
                    promotion = 4
            
            return torch.tensor([from_square, to_square, promotion], dtype=torch.long)
            
        except Exception as e:
            # Return invalid move encoding on error
            return torch.tensor([0, 0, 0], dtype=torch.long)
    
    def encode_themes(self, themes: List[str]) -> torch.Tensor:
        """Encode themes as multi-hot vector"""
        theme_vector = torch.zeros(self.num_themes, dtype=torch.float32)
        
        for theme in themes:
            if theme in self.theme_to_idx:
                theme_vector[self.theme_to_idx[theme]] = 1.0
        
        return theme_vector
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample
        
        Returns:
            {
                'position_features': Tensor (690,) - Position encoding
                'moves': Tensor (N, 3) - Encoded top-N moves
                'move_weights': Tensor (N,) - Training weights per move
                'move_scores': Tensor (N,) - Centipawn evaluations (normalized)
                'themes': Tensor (num_themes,) - Multi-hot theme vector
                'rating': Tensor (1,) - Puzzle difficulty
                'puzzle_id': str - Puzzle ID for tracking
            }
        """
        puzzle = self.puzzles[idx]
        
        # Parse board position
        board = chess.Board(puzzle['fen'])
        
        # Apply augmentation (flip board horizontally)
        if self.augment and random.random() < 0.5:
            board = board.mirror()
        
        # Extract position features (690-dim)
        position_features = self.feature_extractor.extract(board)
        position_features = torch.from_numpy(position_features).float()
        
        # Encode top-N moves
        top_moves = puzzle['top_moves']
        moves_encoded = []
        for move_uci in top_moves:
            moves_encoded.append(self.encode_move(board, move_uci))
        moves = torch.stack(moves_encoded)  # (N, 3)
        
        # Move weights for training (already calculated during preprocessing)
        move_weights = torch.tensor(puzzle['move_weights'], dtype=torch.float32)
        
        # Move scores (normalize to [-1, 1] range for stability)
        move_scores = torch.tensor(puzzle['move_scores'], dtype=torch.float32)
        move_scores = torch.tanh(move_scores / 1000.0)  # Soft clipping
        
        # Encode themes
        themes = self.encode_themes(puzzle['themes'])
        
        # Rating (normalize to [0, 1] range)
        rating = torch.tensor([puzzle['rating'] / 3000.0], dtype=torch.float32)
        
        return {
            'position_features': position_features,
            'moves': moves,
            'move_weights': move_weights,
            'move_scores': move_scores,
            'themes': themes,
            'rating': rating,
            'puzzle_id': puzzle['puzzle_id']
        }


def custom_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle variable-length move lists
    
    Pads move lists to max length in batch
    """
    # Find max number of moves in batch
    max_moves = max(sample['moves'].size(0) for sample in batch)
    batch_size = len(batch)
    
    # Initialize padded tensors
    position_features = torch.stack([s['position_features'] for s in batch])
    
    # Pad moves
    moves = torch.zeros(batch_size, max_moves, 3, dtype=torch.long)
    move_weights = torch.zeros(batch_size, max_moves, dtype=torch.float32)
    move_scores = torch.zeros(batch_size, max_moves, dtype=torch.float32)
    move_masks = torch.zeros(batch_size, max_moves, dtype=torch.bool)
    
    for i, sample in enumerate(batch):
        n_moves = sample['moves'].size(0)
        moves[i, :n_moves] = sample['moves']
        move_weights[i, :n_moves] = sample['move_weights']
        move_scores[i, :n_moves] = sample['move_scores']
        move_masks[i, :n_moves] = True
    
    themes = torch.stack([s['themes'] for s in batch])
    ratings = torch.stack([s['rating'] for s in batch])
    puzzle_ids = [s['puzzle_id'] for s in batch]
    
    return {
        'position_features': position_features,
        'moves': moves,
        'move_weights': move_weights,
        'move_scores': move_scores,
        'move_masks': move_masks,
        'themes': themes,
        'ratings': ratings,
        'puzzle_ids': puzzle_ids
    }


def test_dataset():
    """Test dataset loading and iteration"""
    print("🧪 Testing MoveOrderingDataset...")
    
    # Find most recent preprocessed file
    data_dir = Path("data/preprocessed_puzzles")
    json_files = list(data_dir.glob("enriched_puzzles_compact_*.json"))
    
    if not json_files:
        print("❌ No preprocessed data found. Run preprocess_puzzles_with_stockfish.py first.")
        return
    
    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    print(f"   Using: {latest_file.name}")
    
    # Create dataset
    dataset = MoveOrderingDataset(
        data_path=str(latest_file),
        split='train',
        max_samples=1000  # Test with small subset
    )
    
    print(f"   Dataset size: {len(dataset)}")
    
    # Test single sample
    sample = dataset[0]
    print(f"\n   Sample keys: {list(sample.keys())}")
    print(f"   Position features shape: {sample['position_features'].shape}")
    print(f"   Moves shape: {sample['moves'].shape}")
    print(f"   Move weights: {sample['move_weights']}")
    print(f"   Themes (first 5): {sample['themes'][:5]}")
    print(f"   Rating: {sample['rating'].item() * 3000:.0f}")
    
    # Test DataLoader with custom collate
    from torch.utils.data import DataLoader
    
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
    batch = next(iter(loader))
    
    print(f"\n   Batch shapes:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"     {key}: {value.shape}")
    
    print("\n✅ Dataset test passed!")


if __name__ == '__main__':
    test_dataset()
