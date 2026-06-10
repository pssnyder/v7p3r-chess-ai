#!/usr/bin/env python3
"""
Enhanced Chess Board Feature Extraction

Expands from 55 hand-crafted features to 2000+ raw chess features
using piece-square basis, piece interactions, and bitwise operations.

This is Phase 1 of the neural network evolution plan.
Focus: Extract comprehensive features from raw board state.
"""

import numpy as np
import chess
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class ChessBoardFeatures:
    """Container for extracted chess board features"""
    piece_square_features: np.ndarray  # 768 features
    piece_interaction_features: np.ndarray  # Variable
    pawn_structure_features: np.ndarray  # 512 features
    king_safety_features: np.ndarray  # 512 features
    game_phase_features: np.ndarray  # 256 features
    
    def to_numpy(self) -> np.ndarray:
        """Concatenate all features into single array"""
        return np.concatenate([
            self.piece_square_features,
            self.piece_interaction_features,
            self.pawn_structure_features,
            self.king_safety_features,
            self.game_phase_features,
        ])
    
    def to_binary(self) -> np.ndarray:
        """Convert to binary representation for integer quantization"""
        return (self.to_numpy() > 0.5).astype(np.uint8)
    
    @property
    def total_features(self) -> int:
        """Total number of features"""
        return len(self.to_numpy())


class EnhancedBoardFeatureExtractor:
    """
    Extract 2000+ features from chess board state using bitwise operations.
    
    Feature categories:
    1. Piece-Square (768 features): Raw piece placement
    2. Piece Interactions (4096 features): Attacks, defenses, pins
    3. Pawn Structure (512 features): Pawn-specific patterns
    4. King Safety (512 features): King vulnerability patterns
    5. Game Phase (256 features): Opening/middlegame/endgame indicators
    """
    
    # Piece type to bit index mapping
    PIECE_TYPES = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }
    
    # Piece values for material count
    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,  # Don't count king
    }
    
    def __init__(self):
        """Initialize feature extractor"""
        self.reset_cache()
    
    def reset_cache(self):
        """Reset cached board states"""
        self._last_board = None
        self._last_features = None
    
    def extract_features(self, board: chess.Board) -> ChessBoardFeatures:
        """
        Extract all features from board state
        
        Args:
            board: python-chess Board object
        
        Returns:
            ChessBoardFeatures object with all feature arrays
        """
        # Piece-square features (768)
        ps_features = self._extract_piece_square_features(board)
        
        # Piece interaction features (4096)
        pi_features = self._extract_piece_interaction_features(board)
        
        # Pawn structure features (512)
        pawn_features = self._extract_pawn_structure_features(board)
        
        # King safety features (512)
        king_features = self._extract_king_safety_features(board)
        
        # Game phase features (256)
        phase_features = self._extract_game_phase_features(board)
        
        return ChessBoardFeatures(
            piece_square_features=ps_features,
            piece_interaction_features=pi_features,
            pawn_structure_features=pawn_features,
            king_safety_features=king_features,
            game_phase_features=phase_features,
        )
    
    def _extract_piece_square_features(self, board: chess.Board) -> np.ndarray:
        """
        Extract piece-square features (768)
        
        For each of 64 squares, indicate which piece (or none) occupies it.
        Binary encoding: 12 bits per square (6 white pieces + 6 black pieces)
        
        Returns:
            Array of 768 binary features (one-hot encoded per square)
        """
        features = np.zeros(768, dtype=np.float32)
        
        # For each square
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            
            if piece is None:
                continue
            
            # Determine feature index
            piece_type_idx = self.PIECE_TYPES[piece.type]
            
            # Add color offset (0-5 for white, 6-11 for black)
            if piece.color == chess.WHITE:
                feature_idx = square * 12 + piece_type_idx
            else:
                feature_idx = square * 12 + 6 + piece_type_idx
            
            features[feature_idx] = 1.0
        
        return features
    
    def _extract_piece_interaction_features(self, board: chess.Board) -> np.ndarray:
        """
        Extract piece interaction features (4096)
        
        For each square, which other squares can pieces on this square reach?
        Which pieces attack/defend this square?
        
        Returns:
            Array of piece interaction features
        """
        features = np.zeros(4096, dtype=np.float32)
        
        # For each square
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            
            if piece is None:
                continue
            
            # Get all squares this piece attacks
            attacks = board.attacks(square)
            
            # Mark each attacked square
            for target_square in chess.SQUARES:
                if target_square in attacks:
                    # Feature: piece at `square` attacks `target_square`
                    feature_idx = square * 64 + target_square
                    features[feature_idx] = 1.0
        
        return features
    
    def _extract_pawn_structure_features(self, board: chess.Board) -> np.ndarray:
        """
        Extract pawn structure features (512)
        
        Analyze pawn positions, chains, passed pawns, isolated pawns, doubled pawns.
        This is crucial for chess evaluation.
        
        Returns:
            Array of pawn-related features
        """
        features = np.zeros(512, dtype=np.float32)
        
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)
        
        # Extract pawn features for each file and rank
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            
            if piece is None or piece.type != chess.PAWN:
                continue
            
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            
            # Feature index based on pawn position
            feature_idx = square  # One feature per square for pawn presence
            features[feature_idx] = 1.0
            
            # Check if passed pawn (no enemy pawns ahead)
            is_passed = self._is_passed_pawn(board, square, piece.color)
            if is_passed:
                features[256 + feature_idx] = 1.0
            
            # Check if isolated pawn (no friendly pawns on adjacent files)
            is_isolated = self._is_isolated_pawn(board, square, piece.color)
            if is_isolated:
                features[256 + 64 + file] = 1.0
            
            # Check if doubled pawn
            is_doubled = self._is_doubled_pawn(board, square, piece.color)
            if is_doubled:
                features[256 + 64 + 8 + file] = 1.0
        
        return features
    
    def _extract_king_safety_features(self, board: chess.Board) -> np.ndarray:
        """
        Extract king safety features (512)
        
        Evaluate king vulnerability: exposed squares, attacked squares, escape squares.
        Highly important for endgame and attack assessment.
        
        Returns:
            Array of king safety features
        """
        features = np.zeros(512, dtype=np.float32)
        
        # White king safety
        white_king_square = board.king(chess.WHITE)
        if white_king_square is not None:
            self._extract_single_king_safety(board, white_king_square, chess.WHITE, features, 0)
        
        # Black king safety
        black_king_square = board.king(chess.BLACK)
        if black_king_square is not None:
            self._extract_single_king_safety(board, black_king_square, chess.BLACK, features, 256)
        
        return features
    
    def _extract_single_king_safety(self, board: chess.Board, king_square: int, 
                                    color: bool, features: np.ndarray, offset: int):
        """Extract safety features for a single king"""
        
        # King position
        features[offset + king_square] = 1.0
        
        # Escape squares (empty squares king can move to)
        escape_squares = board.attacks(king_square)
        for escape in escape_squares:
            if board.is_legal(chess.Move(king_square, escape)):
                features[offset + 64 + escape] = 1.0
        
        # Attacked squares around king (threat assessment)
        enemy_color = not color
        enemy_pieces = board.pieces(chess.PAWN, enemy_color)
        for pawn_sq in enemy_pieces:
            attacks = board.attacks(pawn_sq)
            for attacked in attacks:
                if attacked in escape_squares or attacked == king_square:
                    features[offset + 128 + attacked] = 1.0
        
        # Distance from center (kings near center are more exposed in endgame)
        file = chess.square_file(king_square)
        rank = chess.square_rank(king_square)
        center_distance = abs(3.5 - file) + abs(3.5 - rank)
        features[offset + 192 + int(center_distance)] = 1.0
    
    def _extract_game_phase_features(self, board: chess.Board) -> np.ndarray:
        """
        Extract game phase features (256)
        
        Indicate whether game is in opening, middlegame, or endgame.
        Important for model to adjust evaluation strategy.
        
        Returns:
            Array of game phase features
        """
        features = np.zeros(256, dtype=np.float32)
        
        # Calculate material on board (excluding pawns and kings)
        white_material = 0
        black_material = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
            
            if piece.type in [chess.PAWN, chess.KING]:
                continue
            
            value = self.PIECE_VALUES.get(piece.type, 0)
            
            if piece.color == chess.WHITE:
                white_material += value
            else:
                black_material += value
        
        total_material = white_material + black_material
        
        # Material balance
        material_balance = (white_material - black_material) / max(total_material, 1)
        features[0] = material_balance if material_balance > 0 else 0
        features[1] = -material_balance if material_balance < 0 else 0
        
        # Pawn count
        white_pawns = len(board.pieces(chess.PAWN, chess.WHITE))
        black_pawns = len(board.pieces(chess.PAWN, chess.BLACK))
        features[2] = white_pawns / 8.0
        features[3] = black_pawns / 8.0
        
        # Piece count
        white_pieces = 0
        black_pieces = 0
        for ptype in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            white_pieces += len(board.pieces(ptype, chess.WHITE))
            black_pieces += len(board.pieces(ptype, chess.BLACK))
        
        features[4] = white_pieces / 15.0
        features[5] = black_pieces / 15.0
        
        # Game phase classification
        if total_material > 30:
            features[6] = 1.0  # Opening/early middlegame
        elif total_material > 10:
            features[7] = 1.0  # Middlegame
        else:
            features[8] = 1.0  # Endgame
        
        # Development indicators (pieces moved from starting position)
        white_developed = self._count_developed_pieces(board, chess.WHITE)
        black_developed = self._count_developed_pieces(board, chess.BLACK)
        features[9] = white_developed / 10.0
        features[10] = black_developed / 10.0
        
        # Castling rights
        features[11] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
        features[12] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
        features[13] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
        features[14] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
        
        # Material by piece type (for special endgame patterns)
        for i, ptype in enumerate([chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]):
            features[15 + i] = len(board.pieces(ptype, chess.WHITE))
            features[19 + i] = len(board.pieces(ptype, chess.BLACK))
        
        return features
    
    @staticmethod
    def _is_passed_pawn(board: chess.Board, pawn_square: int, color: bool) -> bool:
        """Check if pawn at given square is passed"""
        file = chess.square_file(pawn_square)
        rank = chess.square_rank(pawn_square)
        
        enemy_color = not color
        enemy_pawns = board.pieces(chess.PAWN, enemy_color)
        
        # Check if any enemy pawns on same or adjacent files ahead of this pawn
        for enemy_pawn in enemy_pawns:
            enemy_file = chess.square_file(enemy_pawn)
            enemy_rank = chess.square_rank(enemy_pawn)
            
            if abs(enemy_file - file) <= 1:
                if color == chess.WHITE and enemy_rank > rank:
                    return False
                elif color == chess.BLACK and enemy_rank < rank:
                    return False
        
        return True
    
    @staticmethod
    def _is_isolated_pawn(board: chess.Board, pawn_square: int, color: bool) -> bool:
        """Check if pawn at given square is isolated"""
        file = chess.square_file(pawn_square)
        friendly_pawns = board.pieces(chess.PAWN, color)
        
        # Check if any friendly pawns on adjacent files
        for other_pawn in friendly_pawns:
            if other_pawn == pawn_square:
                continue
            
            other_file = chess.square_file(other_pawn)
            if abs(other_file - file) == 1:
                return False
        
        return True
    
    @staticmethod
    def _is_doubled_pawn(board: chess.Board, pawn_square: int, color: bool) -> bool:
        """Check if pawn at given square is doubled"""
        file = chess.square_file(pawn_square)
        friendly_pawns = board.pieces(chess.PAWN, color)
        
        # Count friendly pawns on same file
        pawns_on_file = sum(1 for p in friendly_pawns if chess.square_file(p) == file)
        
        return pawns_on_file > 1
    
    @staticmethod
    def _count_developed_pieces(board: chess.Board, color: bool) -> int:
        """Count how many pieces have moved from starting position"""
        developed = 0
        
        # Check if knights are developed (not on starting squares)
        knight_starting = {chess.B1, chess.G1} if color == chess.WHITE else {chess.B8, chess.G8}
        knights = board.pieces(chess.KNIGHT, color)
        for knight in knights:
            if knight not in knight_starting:
                developed += 1
        
        # Check if bishops are developed
        bishop_starting = {chess.C1, chess.F1} if color == chess.WHITE else {chess.C8, chess.F8}
        bishops = board.pieces(chess.BISHOP, color)
        for bishop in bishops:
            if bishop not in bishop_starting:
                developed += 1
        
        # Check if queen is developed (very rare in opening)
        queen = board.pieces(chess.QUEEN, color)
        if queen:
            queen_square = list(queen)[0]
            starting_queen = chess.D1 if color == chess.WHITE else chess.D8
            if queen_square != starting_queen:
                developed += 1
        
        # Check if rooks are developed (also rare in opening)
        rook_starting = {chess.A1, chess.H1} if color == chess.WHITE else {chess.A8, chess.H8}
        rooks = board.pieces(chess.ROOK, color)
        for rook in rooks:
            if rook not in rook_starting:
                developed += 1
        
        return developed


def demonstrate_feature_extraction():
    """Demonstrate the feature extraction system"""
    
    print("\n" + "="*80)
    print("🧠 ENHANCED CHESS BOARD FEATURE EXTRACTION - Phase 1")
    print("="*80)
    
    # Initialize extractor
    extractor = EnhancedBoardFeatureExtractor()
    
    # Test with starting position
    board = chess.Board()
    features = extractor.extract_features(board)
    
    print(f"\n📊 Starting Position Feature Extraction:")
    print(f"  Piece-Square Features:        {features.piece_square_features.sum():.0f} / {len(features.piece_square_features)}")
    print(f"  Piece Interaction Features:   {features.piece_interaction_features.sum():.0f} / {len(features.piece_interaction_features)}")
    print(f"  Pawn Structure Features:      {features.pawn_structure_features.sum():.0f} / {len(features.pawn_structure_features)}")
    print(f"  King Safety Features:         {features.king_safety_features.sum():.0f} / {len(features.king_safety_features)}")
    print(f"  Game Phase Features:          {features.game_phase_features.sum():.0f} / {len(features.game_phase_features)}")
    
    total_features = features.total_features
    print(f"\n✅ Total Features Extracted: {total_features}")
    print(f"   Improvement: {total_features/55:.1f}x more features than baseline (55)")
    print(f"   Memory Usage: {(total_features * 4) / 1024:.1f} KB (float32)")
    
    # Test with a middle game position
    moves = [
        "e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4",
        "f3d4", "g8f6", "b1c3", "a7a6", "f4e6", "e7e5"
    ]
    
    for move_uci in moves:
        board.push_uci(move_uci)
    
    features = extractor.extract_features(board)
    
    print(f"\n📊 Middle Game Position (Sicilian Defense):")
    print(f"  Piece-Square Features:        {features.piece_square_features.sum():.0f} / {len(features.piece_square_features)}")
    print(f"  Piece Interaction Features:   {features.piece_interaction_features.sum():.0f} / {len(features.piece_interaction_features)}")
    print(f"  Pawn Structure Features:      {features.pawn_structure_features.sum():.0f} / {len(features.pawn_structure_features)}")
    print(f"  King Safety Features:         {features.king_safety_features.sum():.0f} / {len(features.king_safety_features)}")
    print(f"  Game Phase Features:          {features.game_phase_features.sum():.0f} / {len(features.game_phase_features)}")
    
    print(f"\n✅ Total Features: {features.total_features}")
    print(f"   Memory: {(features.total_features * 4) / 1024:.1f} KB (float32)")
    print(f"   Bitwise: {(features.total_features * 1) / 1024:.1f} KB (int8)")
    
    print("\n" + "="*80)
    print("✨ Phase 1 Feature Extraction Complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    demonstrate_feature_extraction()
