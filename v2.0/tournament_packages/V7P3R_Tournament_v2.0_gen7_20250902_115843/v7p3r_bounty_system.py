# v7p3r_bounty_system.py
"""
V7P3R Chess AI 2.0 - Bounty-based Fitness System
Implements the gold bounty system for evaluating chess moves and AI performance.
"""

import chess
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass


@dataclass
class BountyScore:
    """Container for bounty score components"""
    center_control: float = 0.0
    piece_value: float = 0.0
    attack_defense: float = 0.0
    king_safety: float = 0.0
    tactical_patterns: float = 0.0
    mate_threats: float = 0.0
    piece_coordination: float = 0.0
    castling: float = 0.0
    positional: float = 0.0
    
    def total(self) -> float:
        """Calculate total bounty score"""
        return (
            self.center_control + self.piece_value + self.attack_defense +
            self.king_safety + self.tactical_patterns + self.mate_threats +
            self.piece_coordination + self.castling + self.positional
        )


class BountyEvaluator:
    """Evaluates chess moves using the bounty (gold) system"""
    
    def __init__(self):
        # Define board zones
        self.center_squares = {chess.D4, chess.D5, chess.E4, chess.E5}
        self.center_ring = {
            chess.C3, chess.C4, chess.C5, chess.C6,
            chess.D3, chess.D6, chess.E3, chess.E6,
            chess.F3, chess.F4, chess.F5, chess.F6
        }
        self.central_edge = self._get_central_edge_squares()
        self.outer_edges = self._get_outer_edge_squares()
        
        # Piece values for bounty calculation
        self.piece_values = {
            chess.KING: 100,
            chess.QUEEN: 9,
            chess.ROOK: 5,
            chess.BISHOP: 4,
            chess.KNIGHT: 3,
            chess.PAWN: 1
        }
        
        # Bounty rates
        self.bounty_rates = {
            'center_control': {
                'center_4': 3,
                'center_ring': 2,
                'central_edge': 1,
                'outer_edges': -1
            },
            'piece_values': self.piece_values,
            'attacks': {
                'equal_value': 1,
                'higher_value': 2,
                'lower_undefended': -5
            },
            'king_safety': {
                'near_king': 5,
                'check': 10,
                'checkmate': 1000
            },
            'tactical': {
                'pin': 5,
                'skewer': 5,
                'fork': 5,
                'defender_removal': 5,
                'deflection': 5
            },
            'mate_finding': {
                'mate_in_1': 100,
                'mate_in_2': 500,
                'mate_in_3_plus': 1000
            },
            'coordination': {
                'rook_same_rank_file': 5,
                'bishop_long_diagonal': 5
            },
            'castling': {
                'castling_move': 25,
                'losing_rights': -10
            }
        }
    
    def _get_central_edge_squares(self) -> Set[int]:
        """Get central edge squares"""
        return {
            chess.B2, chess.B3, chess.B4, chess.B5, chess.B6, chess.B7,
            chess.C2, chess.C7, chess.D2, chess.D7, chess.E2, chess.E7,
            chess.F2, chess.F7, chess.G2, chess.G3, chess.G4, chess.G5, chess.G6, chess.G7
        }
    
    def _get_outer_edge_squares(self) -> Set[int]:
        """Get outer edge squares"""
        edges = set()
        for rank in [0, 7]:  # First and last ranks
            for file in range(8):
                edges.add(chess.square(file, rank))
        for file in [0, 7]:  # First and last files
            for rank in range(8):
                edges.add(chess.square(file, rank))
        return edges
    
    def evaluate_move(self, board: chess.Board, move: chess.Move) -> BountyScore:
        """Evaluate a single move and return bounty score"""
        score = BountyScore()
        
        # Make the move temporarily
        original_board = board.copy()
        board.push(move)
        
        try:
            # 1. Center control evaluation
            score.center_control = self._evaluate_center_control(original_board, move)
            
            # 2. Piece value evaluation (after move)
            score.piece_value = self._evaluate_piece_values(board)
            
            # 3. Attack/defense relationships
            score.attack_defense = self._evaluate_attack_defense(original_board, move, board)
            
            # 4. King safety and checks
            score.king_safety = self._evaluate_king_safety(original_board, move, board)
            
            # 5. Tactical patterns
            score.tactical_patterns = self._evaluate_tactical_patterns(original_board, move, board)
            
            # 6. Mate threats
            score.mate_threats = self._evaluate_mate_threats(board)
            
            # 7. Piece coordination
            score.piece_coordination = self._evaluate_piece_coordination(original_board, move, board)
            
            # 8. Castling
            score.castling = self._evaluate_castling(original_board, move, board)
            
        finally:
            # Restore original position
            board.pop()
        
        return score
    
    def _evaluate_center_control(self, board: chess.Board, move: chess.Move) -> float:
        """Evaluate center control bounty - Rule 1"""
        gold = 0.0
        moved_piece = board.piece_at(move.from_square)
        
        if not moved_piece:
            return gold
        
        # Check what squares the piece controls after the move
        temp_board = board.copy()
        temp_board.push(move)
        
        controlled_squares = list(temp_board.attacks(move.to_square))
        
        for square in controlled_squares:
            if square in self.center_squares:
                gold += self.bounty_rates['center_control']['center_4']
            elif square in self.center_ring:
                gold += self.bounty_rates['center_control']['center_ring']
            elif square in self.central_edge:
                gold += self.bounty_rates['center_control']['central_edge']
            elif square in self.outer_edges:
                gold += self.bounty_rates['center_control']['outer_edges']
        
        return gold
    
    def _evaluate_piece_values(self, board: chess.Board) -> float:
        """Evaluate piece values after move - Rule 2"""
        gold = 0.0
        
        # Count pieces for the current player
        current_color = not board.turn  # Since we made the move, it's now opponent's turn
        
        for piece_type, value in self.piece_values.items():
            piece_count = len(list(board.pieces(piece_type, current_color)))
            gold += piece_count * value
        
        return gold
    
    def _evaluate_attack_defense(self, original_board: chess.Board, move: chess.Move, new_board: chess.Board) -> float:
        """Evaluate attack/defense relationships - Rule 3"""
        gold = 0.0
        moved_piece = original_board.piece_at(move.from_square)
        
        if not moved_piece:
            return gold
        
        # Check what the moved piece now attacks
        attacked_squares = list(new_board.attacks(move.to_square))
        
        for square in attacked_squares:
            attacked_piece = new_board.piece_at(square)
            if attacked_piece and attacked_piece.color != moved_piece.color:
                # Get piece values
                attacker_value = self.piece_values[moved_piece.piece_type]
                defender_value = self.piece_values[attacked_piece.piece_type]
                
                # Check if the attacked piece is defended
                is_defended = len(list(new_board.attackers(attacked_piece.color, square))) > 0
                
                if attacker_value == defender_value:
                    gold += self.bounty_rates['attacks']['equal_value']
                elif attacker_value < defender_value:
                    gold += self.bounty_rates['attacks']['higher_value']
                elif attacker_value > defender_value and not is_defended:
                    gold += self.bounty_rates['attacks']['lower_undefended']
        
        return gold
    
    def _evaluate_king_safety(self, original_board: chess.Board, move: chess.Move, new_board: chess.Board) -> float:
        """Evaluate king safety and checks - Rule 4"""
        gold = 0.0
        moved_piece = original_board.piece_at(move.from_square)
        
        if not moved_piece:
            return gold
        
        opponent_color = not moved_piece.color
        opponent_king = new_board.king(opponent_color)
        
        if opponent_king:
            # Check if move attacks squares near opponent king
            attacked_squares = list(new_board.attacks(move.to_square))
            
            for square in attacked_squares:
                if chess.square_distance(square, opponent_king) <= 2:
                    gold += self.bounty_rates['king_safety']['near_king']
            
            # Check for check
            if new_board.is_check():
                gold += self.bounty_rates['king_safety']['check']
            
            # Check for checkmate
            if new_board.is_checkmate():
                gold += self.bounty_rates['king_safety']['checkmate']
        
        return gold
    
    def _evaluate_tactical_patterns(self, original_board: chess.Board, move: chess.Move, new_board: chess.Board) -> float:
        """Evaluate tactical patterns - Rule 5"""
        gold = 0.0
        
        # This is a simplified version - full tactical analysis would be more complex
        moved_piece = original_board.piece_at(move.from_square)
        if not moved_piece:
            return gold
        
        # Check for pins (simplified)
        if self._creates_pin(original_board, move, new_board):
            gold += self.bounty_rates['tactical']['pin']
        
        # Check for forks (piece attacking 2+ enemy pieces)
        attacked_pieces = []
        for square in new_board.attacks(move.to_square):
            piece = new_board.piece_at(square)
            if piece and piece.color != moved_piece.color:
                attacked_pieces.append(piece)
        
        if len(attacked_pieces) >= 2:
            gold += self.bounty_rates['tactical']['fork']
        
        # Check for discovered attacks
        if self._creates_discovered_attack(original_board, move, new_board):
            gold += self.bounty_rates['tactical']['deflection']
        
        return gold
    
    def _evaluate_mate_threats(self, board: chess.Board) -> float:
        """Evaluate mate finding - Rule 6"""
        gold = 0.0
        
        if board.is_checkmate():
            # Try to determine mate distance (simplified)
            gold += self.bounty_rates['mate_finding']['mate_in_1']
        
        # For mate in 2+ detection, we would need deeper search
        # This is a placeholder for now
        
        return gold
    
    def _evaluate_piece_coordination(self, original_board: chess.Board, move: chess.Move, new_board: chess.Board) -> float:
        """Evaluate piece coordination - Rules 7 & 8"""
        gold = 0.0
        moved_piece = original_board.piece_at(move.from_square)
        
        if not moved_piece:
            return gold
        
        # Rule 7: Rook coordination
        if moved_piece.piece_type == chess.ROOK:
            rooks = list(new_board.pieces(chess.ROOK, moved_piece.color))
            if len(rooks) >= 2:
                rook_positions = rooks
                # Check if rooks are on same rank or file
                for i, rook1 in enumerate(rook_positions):
                    for rook2 in rook_positions[i+1:]:
                        if (chess.square_rank(rook1) == chess.square_rank(rook2) or 
                            chess.square_file(rook1) == chess.square_file(rook2)):
                            gold += self.bounty_rates['coordination']['rook_same_rank_file']
        
        # Rule 8: Bishop on long diagonal
        if moved_piece.piece_type == chess.BISHOP:
            # Check if bishop is on long diagonal (a1-h8 or a8-h1)
            file = chess.square_file(move.to_square)
            rank = chess.square_rank(move.to_square)
            
            # Main diagonals: file == rank or file + rank == 7
            if file == rank or file + rank == 7:
                gold += self.bounty_rates['coordination']['bishop_long_diagonal']
        
        return gold
    
    def _evaluate_castling(self, original_board: chess.Board, move: chess.Move, new_board: chess.Board) -> float:
        """Evaluate castling - Rule 9"""
        gold = 0.0
        moved_piece = original_board.piece_at(move.from_square)
        
        if not moved_piece:
            return gold
        
        # Check if this is a castling move
        if (moved_piece.piece_type == chess.KING and 
            abs(move.from_square - move.to_square) == 2):
            gold += self.bounty_rates['castling']['castling_move']
        
        # Check if move loses castling rights without castling
        original_rights = original_board.castling_rights
        new_rights = new_board.castling_rights
        
        if (original_rights != new_rights and 
            not (moved_piece.piece_type == chess.KING and abs(move.from_square - move.to_square) == 2)):
            gold += self.bounty_rates['castling']['losing_rights']
        
        return gold
    
    def _creates_pin(self, original_board: chess.Board, move: chess.Move, new_board: chess.Board) -> bool:
        """Check if move creates a pin (simplified)"""
        moved_piece = original_board.piece_at(move.from_square)
        if not moved_piece or moved_piece.piece_type not in [chess.ROOK, chess.BISHOP, chess.QUEEN]:
            return False
        
        opponent_color = not moved_piece.color
        opponent_king = new_board.king(opponent_color)
        
        if not opponent_king:
            return False
        
        # Check if there's exactly one piece between the moved piece and opponent king
        between_bb = chess.between(move.to_square, opponent_king)
        between_squares = [sq for sq in chess.SQUARES if between_bb & chess.BB_SQUARES[sq]]
        
        if len(between_squares) == 1:
            pinned_piece = new_board.piece_at(between_squares[0])
            return bool(pinned_piece and pinned_piece.color == opponent_color)
        
        return False
    
    def _creates_discovered_attack(self, original_board: chess.Board, move: chess.Move, new_board: chess.Board) -> bool:
        """Check if move creates a discovered attack (simplified)"""
        # This would require more sophisticated analysis
        # For now, return False as placeholder
        return False
    
    def evaluate_position(self, board: chess.Board) -> float:
        """Evaluate entire position for the current player"""
        total_gold = 0.0
        
        # Evaluate all legal moves and return average bounty
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return total_gold
        
        for move in legal_moves:
            move_score = self.evaluate_move(board, move)
            total_gold += move_score.total()
        
        return total_gold / len(legal_moves)
    
    def get_best_move_by_bounty(self, board: chess.Board) -> Tuple[Optional[chess.Move], float]:
        """Get the move with highest bounty score"""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, 0.0
        
        best_move = None
        best_score = float('-inf')
        
        for move in legal_moves:
            move_score = self.evaluate_move(board, move)
            total_score = move_score.total()
            
            if total_score > best_score:
                best_score = total_score
                best_move = move
        
        return best_move, best_score


# Additional bounty heuristics as suggested
class ExtendedBountyEvaluator(BountyEvaluator):
    """Extended bounty evaluator with additional heuristics"""
    
    def __init__(self):
        super().__init__()
        
        # Add more bounty types
        self.bounty_rates.update({
            'pawn_structure': {
                'pawn_storm': 3,
                'passed_pawn': 4,
                'isolated_pawn': -2,
                'doubled_pawn': -1,
                'backward_pawn': -1
            },
            'piece_activity': {
                'active_piece': 2,
                'trapped_piece': -3,
                'outpost': 3
            },
            'space_control': {
                'space_advantage': 1,
                'territory_expansion': 2
            },
            'tempo': {
                'gaining_tempo': 2,
                'losing_tempo': -1,
                'forcing_move': 3
            }
        })
    
    def evaluate_move(self, board: chess.Board, move: chess.Move) -> BountyScore:
        """Extended move evaluation with additional heuristics"""
        score = super().evaluate_move(board, move)
        
        # Add extended evaluations
        original_board = board.copy()
        board.push(move)
        
        try:
            # Pawn structure bonuses
            score.positional += self._evaluate_pawn_structure(original_board, move, board)
            
            # Piece activity
            score.positional += self._evaluate_piece_activity(original_board, move, board)
            
            # Space control
            score.positional += self._evaluate_space_control(original_board, move, board)
            
            # Tempo considerations
            score.positional += self._evaluate_tempo(original_board, move, board)
            
        finally:
            board.pop()
        
        return score
    
    def _evaluate_pawn_structure(self, original_board: chess.Board, move: chess.Move, new_board: chess.Board) -> float:
        """Evaluate pawn structure improvements"""
        gold = 0.0
        moved_piece = original_board.piece_at(move.from_square)
        
        if moved_piece and moved_piece.piece_type == chess.PAWN:
            # Check for passed pawns
            if self._is_passed_pawn(new_board, move.to_square, moved_piece.color):
                gold += self.bounty_rates['pawn_structure']['passed_pawn']
            
            # Check for pawn storms (advancing towards enemy king)
            opponent_king = new_board.king(not moved_piece.color)
            if opponent_king:
                pawn_rank = chess.square_rank(move.to_square)
                king_rank = chess.square_rank(opponent_king)
                
                # Bonus for advancing towards enemy king
                if moved_piece.color == chess.WHITE and pawn_rank > chess.square_rank(move.from_square):
                    if pawn_rank > king_rank - 3:
                        gold += self.bounty_rates['pawn_structure']['pawn_storm']
                elif moved_piece.color == chess.BLACK and pawn_rank < chess.square_rank(move.from_square):
                    if pawn_rank < king_rank + 3:
                        gold += self.bounty_rates['pawn_structure']['pawn_storm']
        
        return gold
    
    def _evaluate_piece_activity(self, original_board: chess.Board, move: chess.Move, new_board: chess.Board) -> float:
        """Evaluate piece activity improvements"""
        gold = 0.0
        moved_piece = original_board.piece_at(move.from_square)
        
        if moved_piece:
            # Count squares the piece can attack before and after
            old_attacks = len(list(original_board.attacks(move.from_square)))
            new_attacks = len(list(new_board.attacks(move.to_square)))
            
            if new_attacks > old_attacks:
                gold += self.bounty_rates['piece_activity']['active_piece']
        
        return gold
    
    def _evaluate_space_control(self, original_board: chess.Board, move: chess.Move, new_board: chess.Board) -> float:
        """Evaluate space control improvements"""
        gold = 0.0
        
        # Simple space evaluation - count controlled squares
        moved_piece = original_board.piece_at(move.from_square)
        if moved_piece:
            # Bonus for advancing pieces into opponent territory
            if moved_piece.color == chess.WHITE:
                if chess.square_rank(move.to_square) > chess.square_rank(move.from_square):
                    gold += self.bounty_rates['space_control']['territory_expansion']
            else:
                if chess.square_rank(move.to_square) < chess.square_rank(move.from_square):
                    gold += self.bounty_rates['space_control']['territory_expansion']
        
        return gold
    
    def _evaluate_tempo(self, original_board: chess.Board, move: chess.Move, new_board: chess.Board) -> float:
        """Evaluate tempo gains/losses"""
        gold = 0.0
        
        # Check if move creates threats that opponent must respond to
        if new_board.is_check():
            gold += self.bounty_rates['tempo']['forcing_move']
        
        # Check if move attacks valuable pieces
        moved_piece = original_board.piece_at(move.from_square)
        if moved_piece:
            attacked_squares = list(new_board.attacks(move.to_square))
            for square in attacked_squares:
                piece = new_board.piece_at(square)
                if piece and piece.color != moved_piece.color:
                    if piece.piece_type in [chess.QUEEN, chess.ROOK]:
                        gold += self.bounty_rates['tempo']['forcing_move']
        
        return gold
    
    def _is_passed_pawn(self, board: chess.Board, square: int, color: chess.Color) -> bool:
        """Check if pawn is passed"""
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        
        # Check files (current and adjacent) for enemy pawns in front
        for check_file in [file - 1, file, file + 1]:
            if 0 <= check_file <= 7:
                if color == chess.WHITE:
                    # Check ranks ahead for white
                    for check_rank in range(rank + 1, 8):
                        check_square = chess.square(check_file, check_rank)
                        piece = board.piece_at(check_square)
                        if piece and piece.piece_type == chess.PAWN and piece.color == chess.BLACK:
                            return False
                else:
                    # Check ranks ahead for black
                    for check_rank in range(0, rank):
                        check_square = chess.square(check_file, check_rank)
                        piece = board.piece_at(check_square)
                        if piece and piece.piece_type == chess.PAWN and piece.color == chess.WHITE:
                            return False
        
        return True
