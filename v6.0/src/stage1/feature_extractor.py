"""
V7P3R AI v6.1 - Fast Feature Extraction
Stage 1: Position Evaluator Features (19 dimensions)

Extracts lightweight features from FEN strings for rapid evaluation.

Features (19 total):
- Piece counts (12): white/black pawns, knights, bishops, rooks, queens, kings
- Material balance (1): centipawn advantage
- Castling rights (4): white/black kingside/queenside
- In check (1): boolean
- Mobility (2): white/black legal move counts (note: requires 2 board flips)

Author: Pat Snyder
Created: 2026-05-31
"""

import chess
import numpy as np
from typing import Optional, List


def extract_fast_features(fen: str) -> Optional[np.ndarray]:
    """
    Extract fast, simple features from a FEN string.
    
    Args:
        fen: FEN string representing board position
        
    Returns:
        numpy array of 19 features, or None if extraction fails
    """
    try:
        board = chess.Board(fen)
        
        features = []
        
        # Piece counts (12 features)
        for color in [chess.WHITE, chess.BLACK]:
            features.append(len(board.pieces(chess.PAWN, color)))
            features.append(len(board.pieces(chess.KNIGHT, color)))
            features.append(len(board.pieces(chess.BISHOP, color)))
            features.append(len(board.pieces(chess.ROOK, color)))
            features.append(len(board.pieces(chess.QUEEN, color)))
            features.append(len(board.pieces(chess.KING, color)))
        
        # Material balance (1 feature)
        piece_values = {
            chess.PAWN: 1, 
            chess.KNIGHT: 3, 
            chess.BISHOP: 3, 
            chess.ROOK: 5, 
            chess.QUEEN: 9
        }
        white_material = sum(
            len(board.pieces(pt, chess.WHITE)) * val 
            for pt, val in piece_values.items()
        )
        black_material = sum(
            len(board.pieces(pt, chess.BLACK)) * val 
            for pt, val in piece_values.items()
        )
        features.append(white_material - black_material)
        
        # Positional features (4 features)
        features.append(1 if board.turn == chess.WHITE else 0)  # Side to move
        features.append(1 if board.has_kingside_castling_rights(chess.WHITE) else 0)
        features.append(1 if board.has_queenside_castling_rights(chess.WHITE) else 0)
        features.append(1 if board.is_check() else 0)
        
        # Mobility (2 features - current side and opponent)
        current_mobility = board.legal_moves.count()
        board.turn = not board.turn  # Flip turn
        opponent_mobility = board.legal_moves.count()
        board.turn = not board.turn  # Restore
        
        features.append(current_mobility)
        features.append(opponent_mobility)
        
        return np.array(features, dtype=np.float32)
        
    except Exception as e:
        # Return None on any error (invalid FEN, etc.)
        return None


def extract_features_batch(fen_list: List[str]) -> np.ndarray:
    """
    Extract features from multiple FEN strings.
    
    Args:
        fen_list: List of FEN strings
        
    Returns:
        numpy array of shape (N, 19) where N is number of valid positions
        Invalid positions are skipped
    """
    features_list = []
    
    for fen in fen_list:
        features = extract_fast_features(fen)
        if features is not None:
            features_list.append(features)
    
    if not features_list:
        return np.array([]).reshape(0, 19)
    
    return np.array(features_list, dtype=np.float32)


# Feature dimension constant
FEATURE_DIM = 19

# Feature names for documentation
FEATURE_NAMES = [
    'white_pawns', 'white_knights', 'white_bishops', 'white_rooks', 
    'white_queens', 'white_kings',
    'black_pawns', 'black_knights', 'black_bishops', 'black_rooks',
    'black_queens', 'black_kings',
    'material_balance',
    'side_to_move',  # 1 = white, 0 = black
    'white_can_castle_kingside', 'white_can_castle_queenside',
    'is_in_check',
    'current_side_mobility', 'opponent_mobility',
]
