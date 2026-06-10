"""HalfKA Sparse Feature Generator.

Expands from 55 hand-crafted features to 45,056 sparse HalfKA features.
Each feature represents a (Piece, Square, KingSquare) triplet.

SPRINT 2, DAY 1-2: Implement this module

Features:
    HalfKA (Half King-Aware) architecture:
    - 6 piece types × 64 squares × 32 king buckets = 12,288 features per side
    - Dual sides (white/black perspective) = 24,576 total
    - Some features inactive at any time (~30-32 active per position)
    - Sparse representation: only store active features

Classes:
    KingBucketMapper: King position → bucket index (weight sharing)
    HalfKAFeatureGenerator: Position → active feature indices

Methods (to implement):
    get_halfdka_index(piece, square, king_square) -> int
        Calculate feature index from piece, square, and king position
        Returns: 0-45055 (unique feature ID)
        
    get_active_features(board) -> List[int]
        Get all active features for current position
        Returns: List of feature indices
        
    get_feature_name(index) -> str
        Get human-readable name for feature
        Example: "White Pawn on e4 (King D3)"

Performance Requirements:
    - Index calculation: <1 microsecond per feature
    - Active feature extraction: <100 microseconds per position
    - Memory: King bucket table (32×32 = 1KB)

King Buckets:
    8×4 mapping of king squares to 32 zones (weight sharing)
    Reduces parameters while preserving king safety information

Test with: python -m pytest tests/test_halfdka_features.py -v
"""

import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class KingBucket:
    """King bucket configuration for weight sharing."""
    
    # Precomputed mapping: king_square (0-63) → bucket (0-31)
    KING_BUCKET_MAP = [
        # Initialize in __post_init__
    ]
    
    @staticmethod
    def get_bucket(king_square: int) -> int:
        """Get bucket index for king square.
        
        Args:
            king_square: 0-63 (a1 to h8)
            
        Returns:
            Bucket index 0-31 (for weight sharing)
            
        Mapping: 8×4 grid (4 zones per side)
            Bottom (pawns): 0-3
            Lower middle: 4-7
            Upper middle: 8-11
            Top (opposite end): 12-15
        """
        # TODO: SPRINT 2 DAY 1
        # Implement king bucket calculation
        # Reduces 64 squares to 32 zones for weight sharing
        pass


class HalfKAFeatureGenerator:
    """Generates HalfKA sparse features for a position."""
    
    # Feature space: 6 pieces × 64 squares × 32 king buckets = 12,288 per side
    PIECES_PER_SIDE = 6  # Pawn, Knight, Bishop, Rook, Queen, King
    SQUARES = 64
    KING_BUCKETS = 32
    FEATURES_PER_SIDE = PIECES_PER_SIDE * SQUARES * KING_BUCKETS
    
    def __init__(self):
        """Initialize feature generator."""
        self.king_bucket = KingBucket()
    
    def get_halfdka_index(self, piece_type: int, square: int, 
                         king_square: int, perspective: int = 0) -> int:
        """Calculate HalfKA feature index.
        
        Args:
            piece_type: 0-5 (Pawn, Knight, Bishop, Rook, Queen, King)
            square: 0-63 (piece position)
            king_square: 0-63 (king position)
            perspective: 0 (white) or 1 (black)
            
        Returns:
            Feature index 0-45055
            
        Formula:
            For white perspective:
                index = piece_type × 2048 + square × 32 + king_bucket
            Total: 6 × 64 × 32 = 12,288 per side
            Both sides: 24,576 (with perspective doubling)
            
        Example:
            # White pawn on e4, white king on d3
            idx = gen.get_halfdka_index(piece_type=0, square=12, king_square=11)
        """
        # TODO: SPRINT 2 DAY 1
        # 1. Get king bucket for king_square
        # 2. Calculate: piece_type × 2048 + square × 32 + bucket
        # 3. If perspective=1 (black), add 12288 (flip to black side)
        # 4. Return index
        pass
    
    def get_active_features(self, board) -> List[int]:
        """Get all active features for a position.
        
        Args:
            board: chess.Board object
            
        Returns:
            List of active feature indices (typically 30-32)
            
        Active features:
            - One per piece on the board (white + black)
            - Empty squares don't generate features
            - King always generates one feature per side
            
        Example:
            features = gen.get_active_features(board)
            print(f"Active features: {len(features)}")  # ~30
        """
        # TODO: SPRINT 2 DAY 1
        # 1. Get white king square
        # 2. Get black king square
        # 3. For each piece on board:
        #    a. Determine piece type (P/N/B/R/Q/K)
        #    b. Calculate HalfKA index
        #    c. Add to list
        # 4. Return list (should be 2 king + 30 other pieces max)
        pass
    
    def get_active_features_incremental(self, board, move) -> Tuple[List[int], List[int]]:
        """Get changed features for a move (incremental update).
        
        Args:
            board: chess.Board before move
            move: chess.Move to make
            
        Returns:
            (removed_indices, added_indices): Features that changed
            
        Incremental update rationale:
            Instead of recalculating all 30+ features per move,
            only recalculate affected pieces (typically 2-3):
            - Piece moving (removed from old square, added to new)
            - Captured piece (removed)
            - Promotions/castling (special cases)
            
        Impact: 100-1000x faster than recomputing all features
        
        Example:
            removed, added = gen.get_active_features_incremental(board, move)
            print(f"Changed: {len(removed)} removed, {len(added)} added")
        """
        # TODO: SPRINT 2 DAY 2
        # 1. Identify moving piece, captured piece, promotion
        # 2. Calculate removed features (old positions)
        # 3. Calculate added features (new positions)
        # 4. Return as (removed, added) tuples
        pass
    
    def get_feature_name(self, index: int) -> str:
        """Get human-readable name for feature index.
        
        Args:
            index: Feature index 0-45055
            
        Returns:
            String like "White Pawn on e4 (King D3)"
            
        Example:
            name = gen.get_feature_name(42)
            print(name)  # "White Bishop on f3 (King h1)"
        """
        # TODO: SPRINT 2 DAY 2
        # 1. Reverse HalfKA index calculation
        # 2. Extract piece type, square, king bucket
        # 3. Format as readable string
        pass


def load_piece_map() -> dict:
    """Load piece type mapping.
    
    Returns:
        {piece_type: name} dictionary
    """
    return {
        0: "Pawn",
        1: "Knight",
        2: "Bishop",
        3: "Rook",
        4: "Queen",
        5: "King",
    }


def load_square_names() -> dict:
    """Load algebraic notation for squares.
    
    Returns:
        {square_index (0-63): name ("a1"-"h8")}
    """
    files = "abcdefgh"
    ranks = "12345678"
    return {
        i * 8 + j: files[j] + ranks[i]
        for i in range(8)
        for j in range(8)
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    gen = HalfKAFeatureGenerator()
    
    # Example (when board available):
    # features = gen.get_active_features(board)
    # print(f"Active features: {features}")
    
    print("HalfKA feature generator module ready for implementation")
