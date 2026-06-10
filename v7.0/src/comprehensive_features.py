"""
Comprehensive Feature Extractor for V7P3R v7.2

Combines ALL observable chess position features into a single feature vector:
- Stage 1 Fast Features (19 dims): piece counts, material, mobility, castling, check
- Heuristic Features (24 dims): bishop pair, passed pawns, pawn structure, king safety, piece activity
- Complexity Features (8 dims): forest darkness, tactical density, move diversity, game phase
- Temporal Features (4 dims): move urgency, time management, halfmove clock, previous inference time

Total: 55 dimensional feature vector representing complete position understanding

Philosophy: The model sees positions through features, not raw board state.
Training: Self-play with Stockfish oracle + custom personality rewards + time efficiency.
"""

import chess
import numpy as np
from typing import Dict, Tuple

# Piece values for material calculation
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}

class ComprehensiveFeatureExtractor:
    """
    Unified feature extraction combining all V6 research into single feature vector.
    
    Feature Groups:
    1. STAGE 1 FEATURES (19 dims) - Fast piece-centric features
    2. HEURISTIC FEATURES (24 dims) - Positional quality metrics
    3. COMPLEXITY FEATURES (8 dims) - Position difficulty and game phase
    4. TEMPORAL FEATURES (4 dims) - Time management and move urgency
    
    Total: 55 dimensions of pure chess knowledge + temporal awareness
    """
    
    def __init__(self):
        """Initialize feature extractor."""
        self.feature_names = self._build_feature_names()
        self.feature_count = len(self.feature_names)
    
    def _build_feature_names(self) -> list:
        """Build ordered list of feature names for interpretability."""
        names = []
        
        # Stage 1 Fast Features (19)
        for color_name in ['white', 'black']:
            for piece in ['pawns', 'knights', 'bishops', 'rooks', 'queens', 'kings']:
                names.append(f'{color_name}_{piece}_count')
        names.extend([
            'material_balance',
            'side_to_move',
            'white_kingside_castling',
            'white_queenside_castling',
            'in_check',
            'current_mobility',
            'opponent_mobility'
        ])
        
        # Heuristic Features (24)
        names.extend([
            'white_bishop_pair',
            'black_bishop_pair',
            'white_passed_pawns',
            'black_passed_pawns',
            'white_doubled_pawns',
            'black_doubled_pawns',
            'white_isolated_pawns',
            'black_isolated_pawns',
            'white_king_pawn_shield',
            'black_king_pawn_shield',
            'white_active_rooks',
            'black_active_rooks',
            'white_development_score',
            'black_development_score',
            'white_mobility_normalized',
            'black_mobility_normalized',
            # Relative advantages (from mover's perspective)
            'bishop_pair_advantage',
            'passed_pawns_advantage',
            'doubled_pawns_disadvantage',
            'isolated_pawns_disadvantage',
            'king_safety_advantage',
            'active_rooks_advantage',
            'development_advantage',
            'mobility_advantage'
        ])
        
        # Complexity Features (8)
        names.extend([
            'legal_moves_count',
            'captures_available',
            'checks_available',
            'piece_tension',  # Number of attacked pieces
            'center_control',
            'game_phase',  # 0=opening, 0.5=midgame, 1=endgame
            'move_diversity',  # Unique piece types that can move
            'forest_darkness_score'  # Custom V7P3R complexity metric
        ])
        
        # Temporal Features (4) - NEW in v7.2
        names.extend([
            'move_number_normalized',  # 0-1, where 1 = 100+ moves
            'halfmove_clock_normalized',  # 0-1, proximity to 50-move rule
            'urgency_score',  # Combined metric: simple position = high urgency to move fast
            'previous_inference_time'  # Feedback from last move (0 if first move)
        ])
        
        return names
    
    # ========================================
    # STAGE 1 FAST FEATURES (19 dims)
    # ========================================
    
    def extract_stage1_features(self, board: chess.Board) -> np.ndarray:
        """Extract Stage 1 fast features (original 19-dim from v1.0-v1.1)."""
        features = []
        
        # 12 piece counts (white pieces, then black pieces)
        for color in [chess.WHITE, chess.BLACK]:
            for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
                count = len(board.pieces(piece_type, color))
                features.append(count)
        
        # Material balance (white - black)
        white_material = sum(len(board.pieces(pt, chess.WHITE)) * PIECE_VALUES.get(pt, 0) 
                            for pt in PIECE_VALUES.keys() if pt != chess.KING)
        black_material = sum(len(board.pieces(pt, chess.BLACK)) * PIECE_VALUES.get(pt, 0) 
                            for pt in PIECE_VALUES.keys() if pt != chess.KING)
        features.append(white_material - black_material)
        
        # Side to move (1 for white, -1 for black)
        features.append(1 if board.turn == chess.WHITE else -1)
        
        # Castling rights (white kingside, white queenside)
        features.append(1 if board.has_kingside_castling_rights(chess.WHITE) else 0)
        features.append(1 if board.has_queenside_castling_rights(chess.WHITE) else 0)
        
        # In check
        features.append(1 if board.is_check() else 0)
        
        # Mobility (legal moves for current side and opponent)
        current_mobility = len(list(board.legal_moves))
        board.turn = not board.turn
        opponent_mobility = len(list(board.legal_moves))
        board.turn = not board.turn  # Restore original turn
        features.append(current_mobility)
        features.append(opponent_mobility)
        
        return np.array(features, dtype=np.float32)
    
    # ========================================
    # HEURISTIC FEATURES (24 dims)
    # ========================================
    
    def has_bishop_pair(self, board: chess.Board, color: chess.Color) -> int:
        """Returns 1 if color has both bishops, 0 otherwise."""
        return 1 if len(board.pieces(chess.BISHOP, color)) == 2 else 0
    
    def count_passed_pawns(self, board: chess.Board, color: chess.Color) -> int:
        """Count passed pawns for a color."""
        passed_pawns = 0
        for square in board.pieces(chess.PAWN, color):
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            is_passed = True
            direction = 1 if color == chess.WHITE else -1
            
            # Check files: current, left, right
            for f_offset in [-1, 0, 1]:
                check_file = file + f_offset
                if 0 <= check_file <= 7:
                    for r in range(rank + direction, 8 if color == chess.WHITE else -1, direction):
                        if board.piece_at(chess.square(check_file, r)) == chess.Piece(chess.PAWN, not color):
                            is_passed = False
                            break
                if not is_passed:
                    break
            if is_passed:
                passed_pawns += 1
        return passed_pawns
    
    def count_doubled_pawns(self, board: chess.Board, color: chess.Color) -> int:
        """Count doubled pawns for a color."""
        doubled_pawns = 0
        for file in range(8):
            pawn_count = sum(1 for rank in range(8) 
                           if board.piece_at(chess.square(file, rank)) == chess.Piece(chess.PAWN, color))
            if pawn_count > 1:
                doubled_pawns += pawn_count - 1
        return doubled_pawns
    
    def count_isolated_pawns(self, board: chess.Board, color: chess.Color) -> int:
        """Count isolated pawns for a color."""
        isolated_pawns = 0
        for square in board.pieces(chess.PAWN, color):
            file = chess.square_file(square)
            is_isolated = True
            
            # Check adjacent files
            for adj_file in [file - 1, file + 1]:
                if 0 <= adj_file <= 7:
                    if any(board.piece_at(chess.square(adj_file, r)) == chess.Piece(chess.PAWN, color) 
                          for r in range(8)):
                        is_isolated = False
                        break
            
            if is_isolated:
                isolated_pawns += 1
        return isolated_pawns
    
    def evaluate_king_pawn_shield(self, board: chess.Board, color: chess.Color) -> int:
        """Evaluate king's pawn shield strength."""
        king_square = board.king(color)
        if king_square is None:
            return 0
        
        shield_score = 0
        king_rank = chess.square_rank(king_square)
        king_file = chess.square_file(king_square)
        
        # Count pawns around king
        for file_offset in [-1, 0, 1]:
            for rank_offset in [-1, 0, 1]:
                check_file = king_file + file_offset
                check_rank = king_rank + rank_offset
                if 0 <= check_file <= 7 and 0 <= check_rank <= 7:
                    check_square = chess.square(check_file, check_rank)
                    if chess.square_distance(king_square, check_square) <= 2:
                        piece = board.piece_at(check_square)
                        if piece and piece.piece_type == chess.PAWN and piece.color == color:
                            shield_score += 1
        
        return shield_score
    
    def count_active_rooks(self, board: chess.Board, color: chess.Color) -> int:
        """Count active rooks (on open/semi-open files or 7th/8th rank)."""
        active_rooks = 0
        for square in board.pieces(chess.ROOK, color):
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            
            # Check if on 7th or 8th rank
            if (color == chess.WHITE and rank >= 6) or (color == chess.BLACK and rank <= 1):
                active_rooks += 1
                continue
            
            # Check for open/semi-open files
            is_semi_open = True
            for r in range(8):
                piece = board.piece_at(chess.square(file, r))
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    is_semi_open = False
                    break
            
            if is_semi_open:
                active_rooks += 1
        
        return active_rooks
    
    def count_development_score(self, board: chess.Board, color: chess.Color) -> int:
        """Count developed pieces (not on starting squares)."""
        development_score = 0
        starting_rank = 0 if color == chess.WHITE else 7
        
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.QUEEN]:
            for square in board.pieces(piece_type, color):
                if chess.square_rank(square) != starting_rank:
                    development_score += 1
        return development_score
    
    def evaluate_mobility_normalized(self, board: chess.Board) -> Tuple[float, float]:
        """Calculate normalized mobility (attacked squares / 64) for both sides."""
        white_attacked = set()
        black_attacked = set()
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                attacks = board.attacks(square)
                if piece.color == chess.WHITE:
                    white_attacked.update(attacks)
                else:
                    black_attacked.update(attacks)
        
        return len(white_attacked) / 64.0, len(black_attacked) / 64.0
    
    def extract_heuristic_features(self, board: chess.Board) -> np.ndarray:
        """Extract heuristic positional quality features (24 dims)."""
        features = []
        
        # Bishop pair
        white_bishop_pair = self.has_bishop_pair(board, chess.WHITE)
        black_bishop_pair = self.has_bishop_pair(board, chess.BLACK)
        features.extend([white_bishop_pair, black_bishop_pair])
        
        # Passed pawns
        white_passed = self.count_passed_pawns(board, chess.WHITE)
        black_passed = self.count_passed_pawns(board, chess.BLACK)
        features.extend([white_passed, black_passed])
        
        # Doubled pawns
        white_doubled = self.count_doubled_pawns(board, chess.WHITE)
        black_doubled = self.count_doubled_pawns(board, chess.BLACK)
        features.extend([white_doubled, black_doubled])
        
        # Isolated pawns
        white_isolated = self.count_isolated_pawns(board, chess.WHITE)
        black_isolated = self.count_isolated_pawns(board, chess.BLACK)
        features.extend([white_isolated, black_isolated])
        
        # King pawn shield
        white_shield = self.evaluate_king_pawn_shield(board, chess.WHITE)
        black_shield = self.evaluate_king_pawn_shield(board, chess.BLACK)
        features.extend([white_shield, black_shield])
        
        # Active rooks
        white_rooks = self.count_active_rooks(board, chess.WHITE)
        black_rooks = self.count_active_rooks(board, chess.BLACK)
        features.extend([white_rooks, black_rooks])
        
        # Development
        white_dev = self.count_development_score(board, chess.WHITE)
        black_dev = self.count_development_score(board, chess.BLACK)
        features.extend([white_dev, black_dev])
        
        # Mobility (normalized)
        white_mob, black_mob = self.evaluate_mobility_normalized(board)
        features.extend([white_mob, black_mob])
        
        # Relative advantages (from mover's perspective)
        perspective = 1 if board.turn == chess.WHITE else -1
        features.append((white_bishop_pair - black_bishop_pair) * perspective)
        features.append((white_passed - black_passed) * perspective)
        features.append((black_doubled - white_doubled) * perspective)  # More doubled pawns = worse
        features.append((black_isolated - white_isolated) * perspective)  # More isolated = worse
        features.append((white_shield - black_shield) * perspective)
        features.append((white_rooks - black_rooks) * perspective)
        features.append((white_dev - black_dev) * perspective)
        features.append((white_mob - black_mob) * perspective)
        
        return np.array(features, dtype=np.float32)
    
    # ========================================
    # COMPLEXITY FEATURES (8 dims)
    # ========================================
    
    def calculate_game_phase(self, board: chess.Board) -> float:
        """
        Calculate game phase: 0=opening, 0.5=midgame, 1=endgame
        Based on material remaining.
        """
        # Count non-pawn material
        total_material = 0
        for color in [chess.WHITE, chess.BLACK]:
            for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                total_material += len(board.pieces(piece_type, color)) * PIECE_VALUES[piece_type]
        
        # Starting material (excluding pawns): 2N + 2B + 2R + 1Q per side = 30 per side = 60 total
        starting_material = 60
        phase = 1.0 - (total_material / starting_material)
        return min(1.0, max(0.0, phase))
    
    def calculate_piece_tension(self, board: chess.Board) -> int:
        """Count number of pieces under attack."""
        tension = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                # Check if this piece is attacked by opponent
                if board.is_attacked_by(not piece.color, square):
                    tension += 1
        return tension
    
    def calculate_center_control(self, board: chess.Board) -> float:
        """
        Calculate center control score from mover's perspective.
        Center squares: e4, d4, e5, d5
        """
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        mover_control = 0
        opponent_control = 0
        
        for sq in center_squares:
            if board.is_attacked_by(board.turn, sq):
                mover_control += 1
            if board.is_attacked_by(not board.turn, sq):
                opponent_control += 1
        
        # Return normalized difference
        total = mover_control + opponent_control
        if total == 0:
            return 0.0
        return (mover_control - opponent_control) / total
    
    def calculate_move_diversity(self, board: chess.Board) -> int:
        """Count unique piece types that can move."""
        piece_types_moving = set()
        for move in board.legal_moves:
            piece = board.piece_at(move.from_square)
            if piece:
                piece_types_moving.add(piece.piece_type)
        return len(piece_types_moving)
    
    def calculate_forest_darkness(self, board: chess.Board) -> float:
        """
        V7P3R's custom complexity metric: "How dark is the forest?"
        Combines tactical density, piece tension, and move diversity.
        Higher = more complex/chaotic position (Tal would approve).
        """
        legal_moves = len(list(board.legal_moves))
        captures = sum(1 for m in board.legal_moves if board.is_capture(m))
        checks = sum(1 for m in board.legal_moves if board.gives_check(m))
        tension = self.calculate_piece_tension(board)
        diversity = self.calculate_move_diversity(board)
        
        # Normalize components
        legal_norm = min(legal_moves / 60.0, 1.0)  # Max ~60 moves
        capture_norm = min(captures / 15.0, 1.0)  # Max ~15 captures
        check_norm = min(checks / 5.0, 1.0)  # Max ~5 checks
        tension_norm = min(tension / 16.0, 1.0)  # Max ~16 pieces under attack
        diversity_norm = diversity / 6.0  # Max 6 piece types
        
        # Weighted combination (emphasize captures, checks, tension)
        darkness = (legal_norm * 0.2 + 
                   capture_norm * 0.3 + 
                   check_norm * 0.2 + 
                   tension_norm * 0.2 + 
                   diversity_norm * 0.1)
        
        return darkness
    
    def extract_complexity_features(self, board: chess.Board) -> np.ndarray:
        """Extract complexity and game phase features (8 dims)."""
        features = []
        
        legal_moves = list(board.legal_moves)
        features.append(len(legal_moves))
        features.append(sum(1 for m in legal_moves if board.is_capture(m)))
        features.append(sum(1 for m in legal_moves if board.gives_check(m)))
        features.append(self.calculate_piece_tension(board))
        features.append(self.calculate_center_control(board))
        features.append(self.calculate_game_phase(board))
        features.append(self.calculate_move_diversity(board))
        features.append(self.calculate_forest_darkness(board))
        
        return np.array(features, dtype=np.float32)
    
    # ========================================
    # TEMPORAL FEATURES (4 dims) - NEW v7.2
    # ========================================
    
    def extract_temporal_features(self, board: chess.Board, 
                                  move_number: int = 0, 
                                  previous_inference_ms: float = 0.0) -> np.ndarray:
        """
        Extract temporal features for time management learning.
        
        Args:
            board: Current position
            move_number: Move count (0 if unknown)
            previous_inference_ms: How long previous move took (milliseconds)
        
        Returns:
            4-dim array: [move_normalized, halfmove_normalized, urgency, prev_time_normalized]
        """
        features = []
        
        # 1. Move number normalized (0-1, where 1 = 100+ moves)
        move_norm = min(move_number / 100.0, 1.0)
        features.append(move_norm)
        
        # 2. Halfmove clock normalized (proximity to 50-move draw)
        # Higher values = approaching draw, should speed up or change plan
        halfmove_norm = min(board.halfmove_clock / 50.0, 1.0)
        features.append(halfmove_norm)
        
        # 3. Urgency score (simple positions should be decided quickly)
        # Based on: few legal moves + low complexity + late game = high urgency
        legal_moves_count = len(list(board.legal_moves))
        complexity_score = self.calculate_forest_darkness(board)
        game_phase = self.calculate_game_phase(board)
        
        # Urgency is HIGH when:
        # - Few legal moves (forced/simple)
        # - Low complexity (straightforward)
        # - Endgame (need to convert or rush)
        move_urgency = 1.0 - min(legal_moves_count / 40.0, 1.0)  # Few moves = high urgency
        complexity_urgency = 1.0 - complexity_score  # Simple = high urgency
        phase_urgency = game_phase  # Endgame = high urgency
        
        urgency = (move_urgency * 0.4 + complexity_urgency * 0.4 + phase_urgency * 0.2)
        features.append(urgency)
        
        # 4. Previous inference time (normalized to ~0-1, where 1 = 1000ms)
        # This creates feedback loop: network sees how long last move took
        prev_time_norm = min(previous_inference_ms / 1000.0, 1.0)
        features.append(prev_time_norm)
        
        return np.array(features, dtype=np.float32)
    
    # ========================================
    # UNIFIED EXTRACTION
    # ========================================
    
    def extract_all_features(self, board: chess.Board, 
                            move_number: int = 0,
                            previous_inference_ms: float = 0.0) -> np.ndarray:
        """
        Extract complete 55-dimensional feature vector.
        
        Args:
            board: Current position
            move_number: Move count for temporal features (default 0)
            previous_inference_ms: Previous move timing for feedback (default 0)
        
        Returns:
            numpy array of shape (55,) containing all features
        """
        stage1 = self.extract_stage1_features(board)  # 19 dims
        heuristic = self.extract_heuristic_features(board)  # 24 dims
        complexity = self.extract_complexity_features(board)  # 8 dims
        temporal = self.extract_temporal_features(board, move_number, previous_inference_ms)  # 4 dims
        
        # Concatenate all features
        all_features = np.concatenate([stage1, heuristic, complexity, temporal])
        
        assert len(all_features) == self.feature_count, \
            f"Feature count mismatch: expected {self.feature_count}, got {len(all_features)}"
        
        return all_features
    
    def extract_all_features_dict(self, board: chess.Board) -> Dict[str, float]:
        """
        Extract features as named dictionary for interpretability.
        
        Returns:
            Dictionary mapping feature names to values
        """
        features = self.extract_all_features(board)
        return dict(zip(self.feature_names, features))
    
    def get_feature_info(self) -> Dict[str, any]:
        """Get information about the feature set."""
        return {
            'total_features': self.feature_count,
            'stage1_features': 19,
            'heuristic_features': 24,
            'complexity_features': 8,
            'feature_names': self.feature_names,
            'description': 'Comprehensive feature set combining Stage 1, heuristics, and complexity metrics'
        }


# Convenience function for quick extraction
def extract_features(fen: str) -> np.ndarray:
    """Extract features from FEN string."""
    board = chess.Board(fen)
    extractor = ComprehensiveFeatureExtractor()
    return extractor.extract_all_features(board)


# Example usage and validation
if __name__ == "__main__":
    print("="*60)
    print("V7P3R v7.0 - COMPREHENSIVE FEATURE EXTRACTOR")
    print("="*60)
    
    extractor = ComprehensiveFeatureExtractor()
    info = extractor.get_feature_info()
    
    print(f"\n📊 Feature Set Summary:")
    print(f"  Total Features: {info['total_features']}")
    print(f"  - Stage 1 (Fast): {info['stage1_features']} dims")
    print(f"  - Heuristics: {info['heuristic_features']} dims")
    print(f"  - Complexity: {info['complexity_features']} dims")
    
    # Test on starting position
    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    board = chess.Board(starting_fen)
    
    print(f"\n🎯 Testing on starting position...")
    features = extractor.extract_all_features(board)
    print(f"  Feature vector shape: {features.shape}")
    print(f"  Feature vector dtype: {features.dtype}")
    
    # Show some key features
    features_dict = extractor.extract_all_features_dict(board)
    print(f"\n📈 Sample Features (Starting Position):")
    print(f"  white_pawns_count: {features_dict['white_pawns_count']}")
    print(f"  material_balance: {features_dict['material_balance']}")
    print(f"  current_mobility: {features_dict['current_mobility']}")
    print(f"  game_phase: {features_dict['game_phase']:.2f} (0=opening)")
    print(f"  forest_darkness_score: {features_dict['forest_darkness_score']:.2f}")
    print(f"  center_control: {features_dict['center_control']:.2f}")
    
    print(f"\n✅ Feature extraction working correctly!")
    print(f"\n📝 Next: Build V7 neural network + self-play training loop")
