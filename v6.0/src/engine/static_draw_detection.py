#!/usr/bin/env python3
"""
Static Draw Detection Module
V7P3R AI v6.1 - Stage 2 Engine Integration

Based on V7P3R v18.6.3 simplified repetition handling.
Implements draw detection with V7P3R personality (avoid draws when ahead).

Author: Pat Snyder
Created: 2026-05-31
"""

import chess
from typing import Optional, List


class StaticDrawDetector:
    """
    Fast draw detection with V7P3R personality integration.
    
    Features:
    - Threefold repetition detection (O(1) hash lookup)
    - 50-move rule tracking
    - Insufficient material detection
    - Personality-aware draw rejection (prefer fighting when ahead)
    
    Philosophy:
    "Blaze of glory over boring draw" - reject draws when eval >50cp
    """
    
    def __init__(self, repetition_eval_threshold: int = 50):
        """
        Initialize draw detector.
        
        Args:
            repetition_eval_threshold: Centipawn threshold for rejecting threefold
                                      (reject repetition if eval > threshold)
        """
        self.repetition_eval_threshold = repetition_eval_threshold
        
        # Constants
        self.FIFTY_MOVE_THRESHOLD = 50  # Half-moves
        self.INSUFFICIENT_MATERIAL_DRAW = True
        
    def would_cause_threefold(
        self, 
        board: chess.Board, 
        move: chess.Move
    ) -> bool:
        """
        Check if move would cause threefold repetition.
        
        Uses O(1) hash lookup via python-chess's built-in repetition tracking.
        
        Args:
            board: Current board position
            move: Candidate move to check
            
        Returns:
            True if move causes threefold repetition, else False
        """
        board.push(move)
        is_threefold = board.is_repetition(2)  # Position occurred 2+ times before
        board.pop()
        
        return is_threefold
    
    def should_reject_threefold(
        self, 
        board: chess.Board, 
        move: chess.Move, 
        current_eval_cp: int
    ) -> bool:
        """
        Determine if threefold repetition should be rejected based on evaluation.
        
        V7P3R Personality: Avoid draws when ahead (eval > threshold).
        
        Args:
            board: Current board position
            move: Candidate move that causes threefold
            current_eval_cp: Current position evaluation (centipawns)
            
        Returns:
            True if should reject threefold (prefer fighting), else False
        """
        if not self.would_cause_threefold(board, move):
            return False  # Not a threefold, no rejection needed
        
        # Reject threefold if we're ahead by more than threshold
        return current_eval_cp > self.repetition_eval_threshold
    
    def check_fifty_move_rule(self, board: chess.Board) -> bool:
        """
        Check if 50-move rule is approaching or reached.
        
        Args:
            board: Current board position
            
        Returns:
            True if 50-move rule reached (draw claimable), else False
        """
        return board.halfmove_clock >= self.FIFTY_MOVE_THRESHOLD
    
    def moves_until_fifty_move_draw(self, board: chess.Board) -> int:
        """
        Calculate moves remaining until 50-move rule forces draw.
        
        Args:
            board: Current board position
            
        Returns:
            Number of half-moves until 50-move draw (0 if already reached)
        """
        return max(0, self.FIFTY_MOVE_THRESHOLD - board.halfmove_clock)
    
    def should_force_pawn_move_or_capture(self, board: chess.Board) -> bool:
        """
        Determine if engine should prioritize pawn moves or captures to reset 50-move clock.
        
        Strategy: If within 10 moves of 50-move draw, strongly prefer pawn moves/captures.
        
        Args:
            board: Current board position
            
        Returns:
            True if should prioritize clock-resetting moves, else False
        """
        moves_remaining = self.moves_until_fifty_move_draw(board)
        return moves_remaining <= 10 and moves_remaining > 0
    
    def check_insufficient_material(self, board: chess.Board) -> bool:
        """
        Check if position has insufficient material for checkmate.
        
        Uses python-chess's built-in insufficient material detection:
        - K vs K
        - K+B vs K
        - K+N vs K
        - K+B vs K+B (same color bishops)
        
        Args:
            board: Current board position
            
        Returns:
            True if insufficient material (draw), else False
        """
        return board.is_insufficient_material()
    
    def is_draw_position(self, board: chess.Board) -> bool:
        """
        Comprehensive draw detection.
        
        Checks:
        - Stalemate
        - Threefold repetition
        - 50-move rule
        - Insufficient material
        
        Args:
            board: Current board position
            
        Returns:
            True if position is drawn, else False
        """
        # Stalemate
        if board.is_stalemate():
            return True
        
        # Threefold repetition (actual threefold, not would-cause)
        if board.is_repetition(2):
            return True
        
        # 50-move rule
        if self.check_fifty_move_rule(board):
            return True
        
        # Insufficient material
        if self.check_insufficient_material(board):
            return True
        
        return False
    
    def get_draw_type(self, board: chess.Board) -> Optional[str]:
        """
        Identify type of draw if position is drawn.
        
        Args:
            board: Current board position
            
        Returns:
            String describing draw type, or None if not drawn
        """
        if board.is_stalemate():
            return "stalemate"
        
        if board.is_repetition(2):
            return "threefold_repetition"
        
        if self.check_fifty_move_rule(board):
            return "fifty_move_rule"
        
        if self.check_insufficient_material(board):
            return "insufficient_material"
        
        return None
    
    def filter_draw_causing_moves(
        self, 
        board: chess.Board, 
        moves: List[chess.Move], 
        current_eval_cp: int
    ) -> List[chess.Move]:
        """
        Filter out moves that cause draws when we should avoid them.
        
        V7P3R Personality: Only filter if eval > threshold (we're ahead).
        
        Args:
            board: Current board position
            moves: List of candidate moves
            current_eval_cp: Current position evaluation (centipawns)
            
        Returns:
            Filtered list of moves (draw-causing moves removed if eval > threshold)
        """
        if current_eval_cp <= self.repetition_eval_threshold:
            return moves  # Accept draws when behind or equal
        
        # Filter out threefold-causing moves
        filtered = [
            move for move in moves 
            if not self.would_cause_threefold(board, move)
        ]
        
        # If filtering removes all moves, return original list (avoid empty move list)
        return filtered if filtered else moves


# Example usage and testing
if __name__ == "__main__":
    print("Testing Static Draw Detector...")
    print("=" * 60)
    
    detector = StaticDrawDetector(repetition_eval_threshold=50)
    
    # Test position 1: Stalemate
    print("\nTest 1: Stalemate detection")
    board = chess.Board("7k/8/6Q1/8/8/8/8/K7 b - - 0 1")
    print(f"Position: {board.fen()}")
    is_draw = detector.is_draw_position(board)
    draw_type = detector.get_draw_type(board)
    print(f"Is draw: {is_draw}, Type: {draw_type}")
    if is_draw and draw_type == "stalemate":
        print("✓ Correctly identified stalemate")
    else:
        print("✗ Failed to identify stalemate")
    
    # Test position 2: Insufficient material (K+B vs K)
    print("\nTest 2: Insufficient material (K+B vs K)")
    board = chess.Board("7k/8/8/8/8/8/3B4/K7 w - - 0 1")
    print(f"Position: {board.fen()}")
    is_draw = detector.is_draw_position(board)
    draw_type = detector.get_draw_type(board)
    print(f"Is draw: {is_draw}, Type: {draw_type}")
    if is_draw and draw_type == "insufficient_material":
        print("✓ Correctly identified insufficient material")
    else:
        print("✗ Failed to identify insufficient material")
    
    # Test position 3: 50-move rule approaching
    print("\nTest 3: 50-move rule approaching")
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 45 1")
    print(f"Position: {board.fen()}")
    print(f"Half-move clock: {board.halfmove_clock}")
    moves_remaining = detector.moves_until_fifty_move_draw(board)
    should_force = detector.should_force_pawn_move_or_capture(board)
    print(f"Moves until 50-move draw: {moves_remaining}")
    print(f"Should force pawn/capture: {should_force}")
    if moves_remaining == 5 and should_force:
        print("✓ Correctly identified approaching 50-move draw")
    else:
        print("✗ Calculation error")
    
    # Test position 4: Threefold repetition (would-cause check)
    print("\nTest 4: Threefold repetition detection")
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    
    # Create repetition by repeating knight moves
    moves_sequence = [
        chess.Move.from_uci("g1f3"),  # Nf3
        chess.Move.from_uci("g8f6"),  # Nf6
        chess.Move.from_uci("f3g1"),  # Ng1
        chess.Move.from_uci("f6g8"),  # Ng8
        chess.Move.from_uci("g1f3"),  # Nf3
        chess.Move.from_uci("g8f6"),  # Nf6
        chess.Move.from_uci("f3g1"),  # Ng1 (would cause threefold)
    ]
    
    for i, move in enumerate(moves_sequence[:-1]):
        board.push(move)
    
    print(f"After {len(moves_sequence)-1} moves (repetition setup)")
    last_move = moves_sequence[-1]
    would_repeat = detector.would_cause_threefold(board, last_move)
    print(f"Would move {last_move} cause threefold? {would_repeat}")
    
    # Test rejection with different evals
    should_reject_ahead = detector.should_reject_threefold(board, last_move, current_eval_cp=100)
    should_reject_equal = detector.should_reject_threefold(board, last_move, current_eval_cp=30)
    
    print(f"Should reject at +100cp: {should_reject_ahead} (expected True)")
    print(f"Should reject at +30cp: {should_reject_equal} (expected False)")
    
    if would_repeat and should_reject_ahead and not should_reject_equal:
        print("✓ Threefold detection and rejection logic working")
    else:
        print("✗ Threefold detection failed")
    
    # Test position 5: Filter draw-causing moves
    print("\nTest 5: Filter draw-causing moves")
    legal_moves = list(board.legal_moves)
    print(f"Legal moves: {len(legal_moves)}")
    
    filtered_ahead = detector.filter_draw_causing_moves(board, legal_moves, current_eval_cp=100)
    filtered_equal = detector.filter_draw_causing_moves(board, legal_moves, current_eval_cp=30)
    
    print(f"Filtered moves (ahead +100cp): {len(filtered_ahead)}")
    print(f"Filtered moves (equal +30cp): {len(filtered_equal)}")
    
    if len(filtered_ahead) < len(legal_moves) and len(filtered_equal) == len(legal_moves):
        print("✓ Filtering works correctly based on eval")
    else:
        print("✗ Filtering logic error")
    
    print("\n" + "=" * 60)
    print("Static Draw Detector tests complete!")
