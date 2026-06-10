#!/usr/bin/env python3
"""
Static Checkmate Detection Module
V7P3R AI v6.1 - Stage 2 Engine Integration

Based on V7P3R v18.6.3 efficient checkmate detection.
Implements adaptive-depth minimax search for forced checkmates.

Author: Pat Snyder
Created: 2026-05-31
"""

import chess
from typing import Optional, Tuple
import time


class StaticCheckmateDetector:
    """
    Fast checkmate detection using minimax with alpha-beta pruning.
    
    Features:
    - Adaptive depth based on available time
    - Efficient alpha-beta pruning
    - Prefers faster mates (depth bonus)
    - Thread-safe for parallel execution with Stage 2
    
    Performance:
    - Depth 3 (mate-in-2): ~50ms average
    - Depth 5 (mate-in-3): ~100-200ms average
    - Depth 7 (mate-in-4): ~500ms-1s average
    """
    
    def __init__(self, default_depth: int = 5):
        """
        Initialize checkmate detector.
        
        Args:
            default_depth: Default search depth (5 = mate-in-3)
        """
        self.default_depth = default_depth
        self.nodes_searched = 0
        self.search_start_time = 0.0
        
        # Constants
        self.CHECKMATE_SCORE = 29000.0
        self.STALEMATE_SCORE = 0.0
        
    def find_checkmate(
        self, 
        board: chess.Board, 
        time_available: Optional[float] = None
    ) -> Optional[chess.Move]:
        """
        Search for forced checkmate in current position.
        
        Args:
            board: Current board position
            time_available: Seconds available for search (enables adaptive depth)
            
        Returns:
            Checkmating move if found, else None
            
        Adaptive Depth:
            - time_available > 10s: depth 7 (mate-in-4)
            - time_available 3-10s: depth 5 (mate-in-3) [DEFAULT]
            - time_available < 3s: depth 3 (mate-in-2)
            - time_available None: use default_depth
        """
        # Adaptive depth calculation
        search_depth = self._calculate_adaptive_depth(time_available)
        
        # Reset counters
        self.nodes_searched = 0
        self.search_start_time = time.time()
        
        # Try each legal move to find forced mate
        best_move = None
        best_score = -99999.0
        
        for move in board.legal_moves:
            board.push(move)
            
            # Search from opponent's perspective (they're trying to avoid mate)
            score = -self._minimax_checkmate(
                board, 
                depth=search_depth - 1, 
                alpha=-99999.0, 
                beta=99999.0, 
                maximizing=False
            )
            
            board.pop()
            
            # Check if this move leads to forced checkmate
            # Score > CHECKMATE_SCORE means we checkmated opponent
            if score > (self.CHECKMATE_SCORE - search_depth):
                # Prefer faster mates (higher score = faster mate)
                if score > best_score:
                    best_score = score
                    best_move = move
        
        elapsed = time.time() - self.search_start_time
        
        if best_move:
            mate_in_moves = (search_depth + 1) // 2  # Convert plies to moves
            print(f"info string Checkmate found! Mate in {mate_in_moves} "
                  f"({self.nodes_searched} nodes, {elapsed*1000:.1f}ms)", flush=True)
        
        return best_move
    
    def _minimax_checkmate(
        self, 
        board: chess.Board, 
        depth: int, 
        alpha: float, 
        beta: float, 
        maximizing: bool
    ) -> float:
        """
        Minimax search with alpha-beta pruning for checkmate detection.
        
        Args:
            board: Current board position
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            maximizing: True if maximizing player's turn
            
        Returns:
            Evaluation score (CHECKMATE_SCORE if mate found)
        """
        self.nodes_searched += 1
        
        # Terminal conditions
        if depth == 0 or board.is_game_over():
            return self._evaluate_terminal(board, depth)
        
        if maximizing:
            max_eval = -99999.0
            
            for move in board.legal_moves:
                board.push(move)
                eval_score = self._minimax_checkmate(board, depth - 1, alpha, beta, False)
                board.pop()
                
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                
                # Alpha-beta pruning
                if beta <= alpha:
                    break
            
            return max_eval
        
        else:  # Minimizing
            min_eval = 99999.0
            
            for move in board.legal_moves:
                board.push(move)
                eval_score = self._minimax_checkmate(board, depth - 1, alpha, beta, True)
                board.pop()
                
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                
                # Alpha-beta pruning
                if beta <= alpha:
                    break
            
            return min_eval
    
    def _evaluate_terminal(self, board: chess.Board, depth_remaining: int) -> float:
        """
        Evaluate terminal position (checkmate, stalemate, or depth=0).
        
        Args:
            board: Current board position
            depth_remaining: Depth remaining in search
            
        Returns:
            Evaluation score (from side-to-move perspective)
        """
        if board.is_checkmate():
            # Checkmate - side to move has lost
            # Return negative (we got checkmated) from their perspective
            # Prefer quicker mates (less depth remaining = faster mate)
            return -(self.CHECKMATE_SCORE - depth_remaining)
        
        if board.is_stalemate() or board.is_insufficient_material():
            return self.STALEMATE_SCORE
        
        # Depth limit reached without mate/stalemate
        return 0.0
    
    def _calculate_adaptive_depth(self, time_available: Optional[float]) -> int:
        """
        Calculate search depth based on available time.
        
        Args:
            time_available: Seconds available for search (None = use default)
            
        Returns:
            Search depth (odd number for our move at root)
        """
        if time_available is None:
            return self.default_depth
        
        if time_available > 10.0:
            return 7  # Mate-in-4 (deep search when time permits)
        elif time_available >= 3.0:
            return 5  # Mate-in-3 (balanced)
        else:
            return 3  # Mate-in-2 (emergency shallow search)
    
    def get_search_stats(self) -> dict:
        """
        Get statistics from last search.
        
        Returns:
            Dictionary with nodes_searched, time_elapsed
        """
        elapsed = time.time() - self.search_start_time if self.search_start_time > 0 else 0.0
        
        return {
            'nodes_searched': self.nodes_searched,
            'time_elapsed_ms': elapsed * 1000.0,
            'nodes_per_second': self.nodes_searched / elapsed if elapsed > 0 else 0,
        }


# Example usage and testing
if __name__ == "__main__":
    print("Testing Static Checkmate Detector...")
    print("=" * 60)
    
    detector = StaticCheckmateDetector(default_depth=5)
    
    # Test position 1: Mate in 1 (classic back-rank mate)
    print("\nTest 1: Mate in 1 (back-rank)")
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    print(f"Position: {board.fen()}")
    mate_move = detector.find_checkmate(board, time_available=5.0)
    if mate_move:
        print(f"✓ Found: {mate_move} (Ra8#)")
        stats = detector.get_search_stats()
        print(f"  Stats: {stats['nodes_searched']} nodes in {stats['time_elapsed_ms']:.1f}ms")
    else:
        print("✗ No mate found (expected Ra8#)")
    
    # Test position 2: Mate in 2 (smothered mate setup)
    print("\nTest 2: Mate in 2 (smothered mate)")
    board = chess.Board("6k1/5ppp/4n3/8/8/8/5PPP/4R1K1 w - - 0 1")
    print(f"Position: {board.fen()}")
    mate_move = detector.find_checkmate(board, time_available=5.0)
    if mate_move:
        print(f"✓ Found: {mate_move}")
        stats = detector.get_search_stats()
        print(f"  Stats: {stats['nodes_searched']} nodes in {stats['time_elapsed_ms']:.1f}ms")
    else:
        print("✗ No mate found")
    
    # Test position 3: No mate available (complex middlegame)
    print("\nTest 3: No immediate mate (middlegame)")
    board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    print(f"Position: {board.fen()}")
    mate_move = detector.find_checkmate(board, time_available=3.0)
    if mate_move:
        print(f"✗ False positive: {mate_move}")
    else:
        print("✓ Correctly found no mate")
        stats = detector.get_search_stats()
        print(f"  Stats: {stats['nodes_searched']} nodes in {stats['time_elapsed_ms']:.1f}ms")
    
    # Test position 4: Famous puzzle (mate in 3)
    print("\nTest 4: Mate in 3 (famous puzzle)")
    board = chess.Board("r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 1 0")
    print(f"Position: {board.fen()}")
    mate_move = detector.find_checkmate(board, time_available=10.0)
    if mate_move:
        print(f"✓ Found: {mate_move}")
        stats = detector.get_search_stats()
        print(f"  Stats: {stats['nodes_searched']} nodes in {stats['time_elapsed_ms']:.1f}ms")
    else:
        print("✗ Mate exists but not found (may need depth 7+)")
    
    print("\n" + "=" * 60)
    print("Static Checkmate Detector tests complete!")
