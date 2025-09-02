# move_preparation.py
"""
V7P3R Chess AI 2.0 - Move Preparation and Ordering System
Optimizes move selection and ordering to improve neural network efficiency.
"""

import chess
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum


class MoveCategory(Enum):
    """Categories for move classification"""
    CAPTURE = "capture"
    CHECK = "check"
    PROMOTION = "promotion"
    CASTLING = "castling"
    TACTICAL = "tactical"
    POSITIONAL = "positional"
    QUIET = "quiet"
    DUBIOUS = "dubious"


@dataclass
class MoveScore:
    """Comprehensive move scoring for ordering"""
    mvv_lva_score: float = 0.0  # Most Valuable Victim - Least Valuable Attacker
    positional_score: float = 0.0
    tactical_score: float = 0.0
    history_score: float = 0.0
    killer_score: float = 0.0
    see_score: float = 0.0  # Static Exchange Evaluation
    total_score: float = 0.0
    category: MoveCategory = MoveCategory.QUIET
    
    def calculate_total(self) -> float:
        """Calculate total move score for ordering"""
        self.total_score = (
            self.mvv_lva_score * 1000 +  # Captures get highest priority
            self.tactical_score * 500 +   # Tactical moves (checks, threats)
            self.killer_score * 300 +     # Killer moves
            self.positional_score * 100 + # Positional improvements
            self.history_score * 50 +     # History heuristic
            self.see_score * 200          # Static exchange evaluation
        )
        return self.total_score


class MovePreparation:
    """Prepares and orders moves for optimal neural network processing"""
    
    def __init__(self):
        # Piece values for MVV-LVA and SEE
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        # Move ordering weights
        self.ordering_weights = {
            'capture': 1000,
            'promotion': 900,
            'check': 800,
            'castling': 700,
            'killer': 600,
            'history': 100,
            'positional': 50
        }
        
        # History table for move ordering (move -> score)
        self.history_table: Dict[str, float] = {}
        
        # Killer moves table (ply -> [move1, move2])
        self.killer_moves: Dict[int, List[chess.Move]] = {}
        
        # Center squares for positional evaluation
        self.center_squares = {chess.D4, chess.D5, chess.E4, chess.E5}
        self.extended_center = {
            chess.C3, chess.C4, chess.C5, chess.C6,
            chess.D3, chess.D6, chess.E3, chess.E6,
            chess.F3, chess.F4, chess.F5, chess.F6
        }
        
        # Bad squares and patterns to avoid
        self.edge_squares = set()
        for file in [0, 7]:  # a-file and h-file
            for rank in range(8):
                self.edge_squares.add(chess.square(file, rank))
        for rank in [0, 7]:  # 1st and 8th rank
            for file in range(8):
                self.edge_squares.add(chess.square(file, rank))
    
    def prepare_moves(self, board: chess.Board, max_moves: Optional[int] = None) -> List[Tuple[chess.Move, MoveScore]]:
        """
        Prepare and order moves for neural network processing
        Returns list of (move, score) tuples ordered by priority
        """
        legal_moves = list(board.legal_moves)
        
        # Score all moves
        scored_moves = []
        for move in legal_moves:
            score = self._score_move(board, move)
            scored_moves.append((move, score))
        
        # Sort by total score (descending)
        scored_moves.sort(key=lambda x: x[1].total_score, reverse=True)
        
        # Apply move filtering if requested
        if max_moves:
            scored_moves = self._filter_moves(scored_moves, max_moves)
        
        return scored_moves
    
    def _score_move(self, board: chess.Board, move: chess.Move) -> MoveScore:
        """Score a single move for ordering"""
        score = MoveScore()
        
        # 1. MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
        score.mvv_lva_score = self._calculate_mvv_lva(board, move)
        
        # 2. Positional score
        score.positional_score = self._calculate_positional_score(board, move)
        
        # 3. Tactical score (checks, threats, etc.)
        score.tactical_score = self._calculate_tactical_score(board, move)
        
        # 4. History score
        score.history_score = self._get_history_score(move)
        
        # 5. Killer move score
        score.killer_score = self._get_killer_score(move, 0)  # ply=0 for now
        
        # 6. Static Exchange Evaluation
        score.see_score = self._calculate_see(board, move)
        
        # 7. Categorize move
        score.category = self._categorize_move(board, move)
        
        # Calculate total score
        score.calculate_total()
        
        return score
    
    def _calculate_mvv_lva(self, board: chess.Board, move: chess.Move) -> float:
        """Calculate Most Valuable Victim - Least Valuable Attacker score"""
        captured_piece = board.piece_at(move.to_square)
        if not captured_piece:
            return 0.0
        
        moving_piece = board.piece_at(move.from_square)
        if not moving_piece:
            return 0.0
        
        victim_value = self.piece_values[captured_piece.piece_type]
        attacker_value = self.piece_values[moving_piece.piece_type]
        
        # MVV-LVA: prioritize capturing valuable pieces with less valuable pieces
        return victim_value - (attacker_value / 100.0)
    
    def _calculate_positional_score(self, board: chess.Board, move: chess.Move) -> float:
        """Calculate positional improvement score"""
        score = 0.0
        moving_piece = board.piece_at(move.from_square)
        
        if not moving_piece:
            return score
        
        # Center control bonus
        if move.to_square in self.center_squares:
            score += 20
        elif move.to_square in self.extended_center:
            score += 10
        
        # Avoid edge squares (except for specific pieces/situations)
        if move.to_square in self.edge_squares and moving_piece.piece_type not in [chess.ROOK, chess.KING]:
            score -= 10
        
        # Development bonus (moving pieces from back rank)
        if moving_piece.color == chess.WHITE:
            if chess.square_rank(move.from_square) == 0 and chess.square_rank(move.to_square) > 0:
                score += 15
        else:
            if chess.square_rank(move.from_square) == 7 and chess.square_rank(move.to_square) < 7:
                score += 15
        
        # Piece-specific positional bonuses
        if moving_piece.piece_type == chess.KNIGHT:
            # Knights to outposts
            if self._is_outpost(board, move.to_square, moving_piece.color):
                score += 25
        
        elif moving_piece.piece_type == chess.BISHOP:
            # Bishops on long diagonals
            if self._is_long_diagonal(move.to_square):
                score += 15
        
        elif moving_piece.piece_type == chess.ROOK:
            # Rooks on open files
            if self._is_open_file(board, move.to_square):
                score += 20
        
        return score
    
    def _calculate_tactical_score(self, board: chess.Board, move: chess.Move) -> float:
        """Calculate tactical value score"""
        score = 0.0
        
        # Make move temporarily to check tactics
        board_copy = board.copy()
        board_copy.push(move)
        
        # Check bonus
        if board_copy.is_check():
            score += 50
        
        # Checkmate bonus
        if board_copy.is_checkmate():
            score += 1000
        
        # Promotion bonus
        if move.promotion:
            promotion_piece_value = self.piece_values.get(move.promotion, 0)
            score += promotion_piece_value / 10
        
        # Castling bonus
        moving_piece = board.piece_at(move.from_square)
        if moving_piece and moving_piece.piece_type == chess.KING:
            if abs(move.from_square - move.to_square) == 2:  # Castling
                score += 40
        
        # Fork detection (simplified)
        if moving_piece:
            attacked_pieces = []
            for square in board_copy.attacks(move.to_square):
                piece = board_copy.piece_at(square)
                if piece and piece.color != moving_piece.color:
                    attacked_pieces.append(piece)
            
            if len(attacked_pieces) >= 2:
                score += 30  # Fork bonus
        
        board_copy.pop()
        return score
    
    def _get_history_score(self, move: chess.Move) -> float:
        """Get history heuristic score for move"""
        move_key = f"{move.from_square}-{move.to_square}"
        return self.history_table.get(move_key, 0.0)
    
    def _get_killer_score(self, move: chess.Move, ply: int) -> float:
        """Get killer move score"""
        if ply in self.killer_moves:
            if move in self.killer_moves[ply]:
                return 100.0
        return 0.0
    
    def _calculate_see(self, board: chess.Board, move: chess.Move) -> float:
        """
        Calculate Static Exchange Evaluation
        Simplified implementation - returns material difference if move is safe
        """
        if not board.piece_at(move.to_square):
            return 0.0  # No capture
        
        # Simplified SEE: check if capture is safe
        board_copy = board.copy()
        board_copy.push(move)
        
        # Check if the moved piece is now attacked
        attackers = list(board_copy.attackers(not board.turn, move.to_square))
        
        if attackers:
            # The piece might be recaptured
            moving_piece = board.piece_at(move.from_square)
            captured_piece = board.piece_at(move.to_square)
            
            if moving_piece and captured_piece:
                gain = self.piece_values[captured_piece.piece_type]
                loss = self.piece_values[moving_piece.piece_type]
                return gain - loss
        else:
            # Safe capture
            captured_piece = board.piece_at(move.to_square)
            if captured_piece:
                return self.piece_values[captured_piece.piece_type]
        
        return 0.0
    
    def _categorize_move(self, board: chess.Board, move: chess.Move) -> MoveCategory:
        """Categorize the move type"""
        # Check for capture
        if board.piece_at(move.to_square):
            return MoveCategory.CAPTURE
        
        # Check for promotion
        if move.promotion:
            return MoveCategory.PROMOTION
        
        # Check for castling
        moving_piece = board.piece_at(move.from_square)
        if moving_piece and moving_piece.piece_type == chess.KING:
            if abs(move.from_square - move.to_square) == 2:
                return MoveCategory.CASTLING
        
        # Check for check
        board_copy = board.copy()
        board_copy.push(move)
        if board_copy.is_check():
            board_copy.pop()
            return MoveCategory.CHECK
        board_copy.pop()
        
        # Default to quiet move
        return MoveCategory.QUIET
    
    def _filter_moves(self, scored_moves: List[Tuple[chess.Move, MoveScore]], max_moves: int) -> List[Tuple[chess.Move, MoveScore]]:
        """Filter moves to keep only the most promising ones"""
        if len(scored_moves) <= max_moves:
            return scored_moves
        
        # Always keep captures, checks, and promotions
        priority_moves = []
        other_moves = []
        
        for move, score in scored_moves:
            if score.category in [MoveCategory.CAPTURE, MoveCategory.CHECK, MoveCategory.PROMOTION]:
                priority_moves.append((move, score))
            else:
                other_moves.append((move, score))
        
        # Take priority moves first, then fill with best other moves
        result = priority_moves[:max_moves]
        remaining_slots = max_moves - len(result)
        
        if remaining_slots > 0:
            result.extend(other_moves[:remaining_slots])
        
        return result
    
    def update_history(self, move: chess.Move, depth: int, success: bool):
        """Update history table based on move success"""
        move_key = f"{move.from_square}-{move.to_square}"
        
        if success:
            # Increase history score
            self.history_table[move_key] = self.history_table.get(move_key, 0.0) + depth * depth
        else:
            # Decrease history score slightly
            self.history_table[move_key] = max(0.0, self.history_table.get(move_key, 0.0) - 1.0)
    
    def update_killers(self, move: chess.Move, ply: int):
        """Update killer moves table"""
        if ply not in self.killer_moves:
            self.killer_moves[ply] = []
        
        # Remove if already in list
        if move in self.killer_moves[ply]:
            self.killer_moves[ply].remove(move)
        
        # Add to front of list
        self.killer_moves[ply].insert(0, move)
        
        # Keep only top 2 killer moves per ply
        self.killer_moves[ply] = self.killer_moves[ply][:2]
    
    def _is_outpost(self, board: chess.Board, square: int, color: chess.Color) -> bool:
        """Check if square is an outpost for the given color"""
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        
        # Check adjacent files for enemy pawns that could attack this square
        for check_file in [file - 1, file + 1]:
            if 0 <= check_file <= 7:
                if color == chess.WHITE:
                    # For white, check ranks behind for black pawns
                    for check_rank in range(0, rank):
                        check_square = chess.square(check_file, check_rank)
                        piece = board.piece_at(check_square)
                        if piece and piece.piece_type == chess.PAWN and piece.color == chess.BLACK:
                            return False
                else:
                    # For black, check ranks ahead for white pawns
                    for check_rank in range(rank + 1, 8):
                        check_square = chess.square(check_file, check_rank)
                        piece = board.piece_at(check_square)
                        if piece and piece.piece_type == chess.PAWN and piece.color == chess.WHITE:
                            return False
        
        return True
    
    def _is_long_diagonal(self, square: int) -> bool:
        """Check if square is on a long diagonal"""
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        
        # Main diagonals: a1-h8 (file == rank) or a8-h1 (file + rank == 7)
        return file == rank or file + rank == 7
    
    def _is_open_file(self, board: chess.Board, square: int) -> bool:
        """Check if file is open (no pawns)"""
        file = chess.square_file(square)
        
        for rank in range(8):
            check_square = chess.square(file, rank)
            piece = board.piece_at(check_square)
            if piece and piece.piece_type == chess.PAWN:
                return False
        
        return True
    
    def get_move_priorities(self, board: chess.Board) -> Dict[MoveCategory, List[chess.Move]]:
        """Get moves organized by priority categories"""
        scored_moves = self.prepare_moves(board)
        
        categories = {category: [] for category in MoveCategory}
        
        for move, score in scored_moves:
            categories[score.category].append(move)
        
        return categories
    
    def analyze_position_complexity(self, board: chess.Board) -> Dict[str, float]:
        """Analyze position complexity to guide move preparation"""
        analysis = {
            'total_moves': len(list(board.legal_moves)),
            'captures': 0,
            'checks': 0,
            'tactical_density': 0.0,
            'complexity_score': 0.0
        }
        
        legal_moves = list(board.legal_moves)
        tactical_moves = 0
        
        for move in legal_moves:
            # Count captures
            if board.piece_at(move.to_square):
                analysis['captures'] += 1
                tactical_moves += 1
            
            # Count checks
            board_copy = board.copy()
            board_copy.push(move)
            if board_copy.is_check():
                analysis['checks'] += 1
                tactical_moves += 1
            board_copy.pop()
        
        # Calculate tactical density
        if legal_moves:
            analysis['tactical_density'] = tactical_moves / len(legal_moves)
        
        # Calculate complexity score
        analysis['complexity_score'] = (
            analysis['total_moves'] * 0.5 +
            analysis['captures'] * 2.0 +
            analysis['checks'] * 1.5 +
            analysis['tactical_density'] * 100
        )
        
        return analysis


class MovePreparationIntegration:
    """Integration layer between move preparation and neural network"""
    
    def __init__(self, move_prep: MovePreparation):
        self.move_prep = move_prep
        self.complexity_threshold = 50.0  # Adjust based on performance needs
    
    def prepare_for_network(self, board: chess.Board, max_candidates: int = 20) -> List[chess.Move]:
        """
        Prepare optimal move candidates for neural network evaluation
        
        Args:
            board: Current chess position
            max_candidates: Maximum number of moves to present to network
            
        Returns:
            List of moves ordered by priority
        """
        # Analyze position complexity
        complexity = self.move_prep.analyze_position_complexity(board)
        
        # Adjust max_candidates based on complexity
        if complexity['complexity_score'] > self.complexity_threshold:
            # High complexity - be more selective
            max_candidates = min(max_candidates, 15)
        else:
            # Lower complexity - can consider more moves
            max_candidates = min(max_candidates, 25)
        
        # Get scored and ordered moves
        scored_moves = self.move_prep.prepare_moves(board, max_candidates)
        
        # Extract just the moves in priority order
        return [move for move, score in scored_moves]
    
    def get_move_features(self, board: chess.Board, move: chess.Move) -> np.ndarray:
        """
        Extract move features for neural network input
        
        Returns:
            Feature vector for the move
        """
        score = self.move_prep._score_move(board, move)
        
        # Create feature vector
        features = np.array([
            score.mvv_lva_score / 1000.0,  # Normalize
            score.positional_score / 100.0,
            score.tactical_score / 100.0,
            score.history_score / 1000.0,
            score.killer_score / 100.0,
            score.see_score / 500.0,
            1.0 if score.category == MoveCategory.CAPTURE else 0.0,
            1.0 if score.category == MoveCategory.CHECK else 0.0,
            1.0 if score.category == MoveCategory.PROMOTION else 0.0,
            1.0 if score.category == MoveCategory.CASTLING else 0.0
        ])
        
        return features
    
    def feedback_move_quality(self, move: chess.Move, quality_score: float, depth: int):
        """
        Provide feedback on move quality to improve future ordering
        
        Args:
            move: The move that was evaluated
            quality_score: Quality score from neural network (0-1)
            depth: Search depth where move was evaluated
        """
        # Update history table based on quality
        success = quality_score > 0.6  # Threshold for "good" move
        self.move_prep.update_history(move, depth, success)
        
        # Update killer moves for very good moves
        if quality_score > 0.8:
            self.move_prep.update_killers(move, depth)
