"""
Chess State Feature Extractor
Extracts 690-dimensional feature vectors from chess positions

Simplified from v3.0 for move ordering tasks.
Feature groups:
- Material & piece counts (24 features)
- Piece-square tables (384 features: 6 types × 64 squares)
- Mobility & activity (40 features)
- King safety (30 features)
- Pawn structure (60 features)
- Tactical indicators (40 features)
- Positional features (40 features)
- Game phase indicators (20 features)
- Threat analysis (32 features)
- Special patterns (20 features)
Total: 690 features
"""

import chess
import numpy as np
from typing import List, Dict


class ChessStateExtractor:
    """
    Extract comprehensive features from chess positions
    Output: 690-dimensional feature vector as numpy array
    """
    
    def __init__(self):
        self.feature_dim = 690
    
    def extract(self, board: chess.Board) -> np.ndarray:
        """
        Extract features from position
        
        Args:
            board: Chess board state
            
        Returns:
            690-dimensional feature vector as numpy array
        """
        features = []
        
        # Material & piece counts (24 features)
        features.extend(self._extract_material_features(board))
        
        # Piece-square tables (384 features)
        features.extend(self._extract_pst_features(board))
        
        # Mobility (40 features)
        features.extend(self._extract_mobility_features(board))
        
        # King safety (30 features)
        features.extend(self._extract_king_safety_features(board))
        
        # Pawn structure (60 features)
        features.extend(self._extract_pawn_structure_features(board))
        
        # Tactical indicators (40 features)
        features.extend(self._extract_tactical_features(board))
        
        # Positional features (40 features)
        features.extend(self._extract_positional_features(board))
        
        # Game phase (20 features)
        features.extend(self._extract_game_phase_features(board))
        
        # Threat analysis (32 features)
        features.extend(self._extract_threat_features(board))
        
        # Special patterns (20 features)
        features.extend(self._extract_special_patterns(board))
        
        # Ensure exactly 690 features
        features = np.array(features, dtype=np.float32)
        if len(features) < self.feature_dim:
            features = np.pad(features, (0, self.feature_dim - len(features)))
        
        return features[:self.feature_dim]
    
    def extract_batch(self, boards: List[chess.Board]) -> np.ndarray:
        """
        Extract features from multiple positions
        
        Args:
            boards: List of chess boards
            
        Returns:
            (N, 690) numpy array of feature vectors
        """
        return np.vstack([self.extract(board) for board in boards])
    
    def _extract_material_features(self, board: chess.Board) -> List[float]:
        """Material balance and piece counts (24 features)"""
        features = []
        
        # Piece counts for each side (12 features)
        for color in [chess.WHITE, chess.BLACK]:
            features.append(len(board.pieces(chess.PAWN, color)) / 8.0)
            features.append(len(board.pieces(chess.KNIGHT, color)) / 10.0)
            features.append(len(board.pieces(chess.BISHOP, color)) / 10.0)
            features.append(len(board.pieces(chess.ROOK, color)) / 10.0)
            features.append(len(board.pieces(chess.QUEEN, color)) / 9.0)
            features.append(len(board.pieces(chess.KING, color)))
        
        # Material balance (6 features)
        white_material = (
            len(board.pieces(chess.PAWN, chess.WHITE)) * 1 +
            len(board.pieces(chess.KNIGHT, chess.WHITE)) * 3 +
            len(board.pieces(chess.BISHOP, chess.WHITE)) * 3 +
            len(board.pieces(chess.ROOK, chess.WHITE)) * 5 +
            len(board.pieces(chess.QUEEN, chess.WHITE)) * 9
        )
        black_material = (
            len(board.pieces(chess.PAWN, chess.BLACK)) * 1 +
            len(board.pieces(chess.KNIGHT, chess.BLACK)) * 3 +
            len(board.pieces(chess.BISHOP, chess.BLACK)) * 3 +
            len(board.pieces(chess.ROOK, chess.BLACK)) * 5 +
            len(board.pieces(chess.QUEEN, chess.BLACK)) * 9
        )
        
        features.append(white_material / 39.0)
        features.append(black_material / 39.0)
        features.append((white_material - black_material + 39) / 78.0)
        features.append(float(white_material > black_material))
        features.append(float(white_material < black_material))
        features.append(float(white_material == black_material))
        
        # Basic game state (6 features)
        features.append(float(board.turn == chess.WHITE))
        features.append(float(board.has_kingside_castling_rights(chess.WHITE)))
        features.append(float(board.has_queenside_castling_rights(chess.WHITE)))
        features.append(float(board.has_kingside_castling_rights(chess.BLACK)))
        features.append(float(board.has_queenside_castling_rights(chess.BLACK)))
        features.append(float(board.is_check()))
        
        return features
    
    def _extract_pst_features(self, board: chess.Board) -> List[float]:
        """Piece-square table features (384 features: 6 types × 64 squares)"""
        features = np.zeros(384, dtype=np.float32)
        
        piece_type_offset = {
            chess.PAWN: 0,
            chess.KNIGHT: 64,
            chess.BISHOP: 128,
            chess.ROOK: 192,
            chess.QUEEN: 256,
            chess.KING: 320
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                offset = piece_type_offset[piece.piece_type]
                if piece.color == chess.WHITE:
                    features[offset + square] = 1.0
                else:
                    features[offset + (63 - square)] = -1.0
        
        return features.tolist()
    
    def _extract_mobility_features(self, board: chess.Board) -> List[float]:
        """Mobility features (40 features)"""
        features = []
        
        # Legal moves count
        legal_moves = list(board.legal_moves)
        features.append(len(legal_moves) / 50.0)
        
        # Mobility by piece type (10 features)
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            white_mobility = sum(1 for move in legal_moves 
                               if board.piece_at(move.from_square) and 
                               board.piece_at(move.from_square).piece_type == piece_type and
                               board.piece_at(move.from_square).color == chess.WHITE)
            black_mobility = sum(1 for move in legal_moves 
                               if board.piece_at(move.from_square) and 
                               board.piece_at(move.from_square).piece_type == piece_type and
                               board.piece_at(move.from_square).color == chess.BLACK)
            features.append(white_mobility / 20.0)
            features.append(black_mobility / 20.0)
        
        # Center control (16 features: 4 center squares × 2 colors × 2 metrics)
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        for square in center_squares:
            white_attacks = len(board.attackers(chess.WHITE, square))
            black_attacks = len(board.attackers(chess.BLACK, square))
            features.append(white_attacks / 8.0)
            features.append(black_attacks / 8.0)
            piece = board.piece_at(square)
            features.append(float(piece is not None and piece.color == chess.WHITE))
            features.append(float(piece is not None and piece.color == chess.BLACK))
        
        # Padding to 40
        while len(features) < 40:
            features.append(0.0)
        
        return features[:40]
    
    def _extract_king_safety_features(self, board: chess.Board) -> List[float]:
        """King safety features (30 features)"""
        features = []
        
        for color in [chess.WHITE, chess.BLACK]:
            king_square = board.king(color)
            if king_square is None:
                features.extend([0.0] * 15)
                continue
            
            # King position
            features.append(chess.square_file(king_square) / 7.0)
            features.append(chess.square_rank(king_square) / 7.0)
            
            # Attackers around king
            king_zone = [sq for sq in chess.SQUARES if chess.square_distance(sq, king_square) <= 2]
            enemy_attackers = sum(len(board.attackers(not color, sq)) for sq in king_zone)
            features.append(enemy_attackers / 20.0)
            
            # Pawn shield
            pawn_shield = sum(1 for sq in king_zone if board.piece_at(sq) == chess.Piece(chess.PAWN, color))
            features.append(pawn_shield / 3.0)
            
            # Escape squares
            escape_squares = sum(1 for sq in king_zone if not board.is_attacked_by(not color, sq))
            features.append(escape_squares / 9.0)
            
            # Castling availability
            features.append(float(board.has_kingside_castling_rights(color)))
            features.append(float(board.has_queenside_castling_rights(color)))
            
            # Open files near king
            king_file = chess.square_file(king_square)
            open_files = 0
            for f in range(max(0, king_file - 1), min(8, king_file + 2)):
                if not any(board.piece_at(chess.square(f, r)) == chess.Piece(chess.PAWN, color) for r in range(8)):
                    open_files += 1
            features.append(open_files / 3.0)
            
            # Additional padding
            while len(features) % 15 != 0:
                features.append(0.0)
        
        return features[:30]
    
    def _extract_pawn_structure_features(self, board: chess.Board) -> List[float]:
        """Pawn structure features (60 features)"""
        features = []
        
        for color in [chess.WHITE, chess.BLACK]:
            pawns = board.pieces(chess.PAWN, color)
            
            # Isolated pawns (8 features: one per file)
            for file in range(8):
                file_pawns = [p for p in pawns if chess.square_file(p) == file]
                adj_files = [f for f in [file - 1, file + 1] if 0 <= f < 8]
                has_adjacent = any(chess.square_file(p) in adj_files for p in pawns)
                features.append(float(len(file_pawns) > 0 and not has_adjacent))
            
            # Doubled pawns (8 features: one per file)
            for file in range(8):
                file_pawns = [p for p in pawns if chess.square_file(p) == file]
                features.append(float(len(file_pawns) > 1))
            
            # Passed pawns (8 features: one per file)
            for file in range(8):
                file_pawns = [p for p in pawns if chess.square_file(p) == file]
                if file_pawns:
                    # Simplified: check if no enemy pawns ahead
                    rank = chess.square_rank(list(file_pawns)[0])
                    direction = 1 if color == chess.WHITE else -1
                    enemy_pawns = board.pieces(chess.PAWN, not color)
                    enemy_ahead = any(
                        chess.square_file(ep) in [file - 1, file, file + 1] and
                        (chess.square_rank(ep) > rank if color == chess.WHITE else chess.square_rank(ep) < rank)
                        for ep in enemy_pawns if 0 <= chess.square_file(ep) - file + 1 < 3
                    )
                    features.append(float(not enemy_ahead))
                else:
                    features.append(0.0)
            
            # Connected pawns count
            connected = sum(1 for p in pawns if any(
                chess.square_distance(p, p2) == 1 for p2 in pawns if p != p2
            ))
            features.append(connected / 8.0)
            
            # Backward pawns count
            features.append(0.0)  # Simplified
            
            # Pawn islands count
            features.append(0.0)  # Simplified
        
        return features[:60]
    
    def _extract_tactical_features(self, board: chess.Board) -> List[float]:
        """Tactical indicators (40 features)"""
        features = []
        
        # Pins, forks, skewers (simplified detection)
        for color in [chess.WHITE, chess.BLACK]:
            # Count hanging pieces
            hanging = 0
            for square in board.pieces(chess.PAWN, not color) | board.pieces(chess.KNIGHT, not color) | \
                         board.pieces(chess.BISHOP, not color) | board.pieces(chess.ROOK, not color) | \
                         board.pieces(chess.QUEEN, not color):
                if board.is_attacked_by(color, square) and not board.is_attacked_by(not color, square):
                    hanging += 1
            features.append(hanging / 16.0)
            
            # Pieces under attack
            attacked = sum(1 for sq in board.pieces(chess.PAWN, not color) | board.pieces(chess.KNIGHT, not color) |
                          board.pieces(chess.BISHOP, not color) | board.pieces(chess.ROOK, not color) |
                          board.pieces(chess.QUEEN, not color) if board.is_attacked_by(color, sq))
            features.append(attacked / 16.0)
        
        # Check threats
        features.append(float(board.is_check()))
        features.append(0.0)  # Discovered check threat (simplified)
        
        # Captures available
        captures = sum(1 for move in board.legal_moves if board.is_capture(move))
        features.append(captures / 20.0)
        
        # Padding to 40
        while len(features) < 40:
            features.append(0.0)
        
        return features[:40]
    
    def _extract_positional_features(self, board: chess.Board) -> List[float]:
        """Positional features (40 features)"""
        features = []
        
        # Space control
        for color in [chess.WHITE, chess.BLACK]:
            controlled_squares = sum(1 for sq in chess.SQUARES if board.is_attacked_by(color, sq))
            features.append(controlled_squares / 64.0)
        
        # Piece activity (centralization)
        for color in [chess.WHITE, chess.BLACK]:
            centrality = 0
            pieces = (board.pieces(chess.KNIGHT, color) | board.pieces(chess.BISHOP, color) |
                     board.pieces(chess.ROOK, color) | board.pieces(chess.QUEEN, color))
            for square in pieces:
                file, rank = chess.square_file(square), chess.square_rank(square)
                dist_from_center = abs(3.5 - file) + abs(3.5 - rank)
                centrality += (7 - dist_from_center) / 7.0
            features.append(centrality / 10.0 if pieces else 0.0)
        
        # Bishop pair
        features.append(float(len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2))
        features.append(float(len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2))
        
        # Rooks on open files
        for color in [chess.WHITE, chess.BLACK]:
            rooks_open = 0
            for rook_sq in board.pieces(chess.ROOK, color):
                file = chess.square_file(rook_sq)
                if not any(board.piece_at(chess.square(file, r)) and 
                          board.piece_at(chess.square(file, r)).piece_type == chess.PAWN 
                          for r in range(8)):
                    rooks_open += 1
            features.append(rooks_open / 2.0)
        
        # Padding to 40
        while len(features) < 40:
            features.append(0.0)
        
        return features[:40]
    
    def _extract_game_phase_features(self, board: chess.Board) -> List[float]:
        """Game phase indicators (20 features)"""
        features = []
        
        # Material-based phase
        total_material = sum(len(board.pieces(pt, c)) * v for c in [chess.WHITE, chess.BLACK]
                           for pt, v in [(chess.PAWN, 1), (chess.KNIGHT, 3), (chess.BISHOP, 3),
                                        (chess.ROOK, 5), (chess.QUEEN, 9)])
        opening_material = 78  # All pieces on board
        game_phase = 1.0 - (total_material / opening_material)
        features.append(game_phase)
        
        # Move number
        features.append(min(board.fullmove_number / 100.0, 1.0))
        
        # Queens on board
        features.append(float(len(board.pieces(chess.QUEEN, chess.WHITE)) > 0))
        features.append(float(len(board.pieces(chess.QUEEN, chess.BLACK)) > 0))
        
        # Development (pieces off back rank)
        for color in [chess.WHITE, chess.BLACK]:
            back_rank = 0 if color == chess.WHITE else 7
            developed = 0
            for pt in [chess.KNIGHT, chess.BISHOP]:
                for sq in board.pieces(pt, color):
                    if chess.square_rank(sq) != back_rank:
                        developed += 1
            features.append(developed / 4.0)
        
        # Castling status
        for color in [chess.WHITE, chess.BLACK]:
            king_sq = board.king(color)
            if king_sq:
                file = chess.square_file(king_sq)
                castled = file in [2, 6]  # C or G file
                features.append(float(castled))
            else:
                features.append(0.0)
        
        # Padding to 20
        while len(features) < 20:
            features.append(0.0)
        
        return features[:20]
    
    def _extract_threat_features(self, board: chess.Board) -> List[float]:
        """Threat analysis (32 features)"""
        features = []
        
        # Attacked squares count
        for color in [chess.WHITE, chess.BLACK]:
            attacked = sum(1 for sq in chess.SQUARES if board.is_attacked_by(color, sq))
            features.append(attacked / 64.0)
            
            # Attacked pieces value
            attacked_value = 0
            for sq in chess.SQUARES:
                piece = board.piece_at(sq)
                if piece and piece.color != color and board.is_attacked_by(color, sq):
                    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                             chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
                    attacked_value += values.get(piece.piece_type, 0)
            features.append(attacked_value / 39.0)
        
        # Defended squares
        for color in [chess.WHITE, chess.BLACK]:
            defended = sum(1 for sq in chess.SQUARES 
                          if (piece := board.piece_at(sq)) is not None 
                          and piece.color == color 
                          and board.is_attacked_by(color, sq))
            features.append(defended / 16.0)
        
        # Threats to win material
        for color in [chess.WHITE, chess.BLACK]:
            winning_captures = 0
            for move in board.legal_moves:
                if board.is_capture(move):
                    from_piece = board.piece_at(move.from_square)
                    to_piece = board.piece_at(move.to_square)
                    if from_piece and to_piece and from_piece.color == color:
                        values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                                 chess.ROOK: 5, chess.QUEEN: 9}
                        if values.get(to_piece.piece_type, 0) >= values.get(from_piece.piece_type, 0):
                            winning_captures += 1
            features.append(winning_captures / 10.0)
        
        # Padding to 32
        while len(features) < 32:
            features.append(0.0)
        
        return features[:32]
    
    def _extract_special_patterns(self, board: chess.Board) -> List[float]:
        """Special patterns (20 features)"""
        features = []
        
        # Checkmate / stalemate proximity
        features.append(float(board.is_checkmate()))
        features.append(float(board.is_stalemate()))
        features.append(float(board.is_insufficient_material()))
        features.append(float(board.can_claim_draw()))
        
        # Repetition
        features.append(float(board.is_repetition()))
        
        # Fifty-move rule proximity
        features.append(min(board.halfmove_clock / 50.0, 1.0))
        
        # En passant available
        features.append(float(board.has_legal_en_passant()))
        
        # Pawn endgame
        no_pieces = not any(board.pieces(pt, c) for c in [chess.WHITE, chess.BLACK]
                           for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
        features.append(float(no_pieces))
        
        # Padding to 20
        while len(features) < 20:
            features.append(0.0)
        
        return features[:20]


if __name__ == "__main__":
    extractor = ChessStateExtractor()
    board = chess.Board()
    
    features = extractor.extract(board)
    print(f"Feature vector length: {len(features)}")
    print(f"First 10 features: {features[:10]}")
