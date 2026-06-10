"""
V7P3R v7.0 - Stockfish Oracle Integration

Provides objective chess evaluation from Stockfish engine.
Used as training signal for V7 network: "Is this position winning/losing?"

Oracle Responsibilities:
- Position evaluation (centipawn scores)
- Best move suggestions
- Win probability estimation
- Training label generation

Philosophy: Stockfish teaches "good chess", personality rewards add "V7P3R style"
"""

import chess
import chess.engine
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import numpy as np
from dataclasses import dataclass
import time


@dataclass
class StockfishEvaluation:
    """Container for Stockfish evaluation results."""
    score_cp: Optional[int]  # Centipawn score (None if mate)
    score_mate: Optional[int]  # Mate in N moves (None if not mate)
    best_move: Optional[chess.Move]
    pv: List[chess.Move]  # Principal variation
    depth: int
    time_ms: int
    nodes: int
    
    @property
    def normalized_score(self) -> float:
        """
        Normalize score to [-1, 1] range for neural network training.
        
        Returns:
            -1 (losing) to +1 (winning) from current player's perspective
        """
        if self.score_mate is not None:
            # Mate scores: +1 if we're mating, -1 if we're getting mated
            return 1.0 if self.score_mate > 0 else -1.0
        
        if self.score_cp is None:
            return 0.0
        
        # Sigmoid-like normalization: cp / (1 + |cp|/400)
        # Maps [-inf, inf] to approximately [-1, 1]
        # 400cp ≈ 0.5, 800cp ≈ 0.67, 1200cp ≈ 0.75
        abs_cp = abs(self.score_cp)
        normalized = self.score_cp / (1.0 + abs_cp / 400.0)
        
        # Clamp to [-1, 1] for safety
        return max(-1.0, min(1.0, normalized / 400.0))
    
    @property
    def win_probability(self) -> float:
        """
        Convert score to win probability using logistic function.
        
        Returns:
            Win probability [0, 1] for current player
        """
        if self.score_mate is not None:
            return 1.0 if self.score_mate > 0 else 0.0
        
        if self.score_cp is None:
            return 0.5
        
        # Logistic function: P(win) = 1 / (1 + 10^(-cp/400))
        # Based on chess engine evaluation conventions
        return 1.0 / (1.0 + 10 ** (-self.score_cp / 400.0))


class StockfishOracle:
    """
    Stockfish integration for training signal generation.
    
    Manages engine lifecycle, evaluation queries, and training label creation.
    Designed for self-play training with efficient batch evaluation.
    """
    
    def __init__(
        self,
        stockfish_path: Optional[str] = None,
        depth: int = 15,
        time_limit_ms: int = 1000,
        threads: int = 1,
        hash_mb: int = 128
    ):
        """
        Initialize Stockfish oracle.
        
        Args:
            stockfish_path: Path to Stockfish executable (auto-detect if None)
            depth: Search depth for evaluations
            time_limit_ms: Time limit per evaluation in milliseconds
            threads: Number of CPU threads
            hash_mb: Hash table size in MB
        """
        self.stockfish_path = self._find_stockfish(stockfish_path)
        self.depth = depth
        self.time_limit_ms = time_limit_ms
        self.threads = threads
        self.hash_mb = hash_mb
        
        self.engine: Optional[chess.engine.SimpleEngine] = None
        self._evaluation_count = 0
        self._total_time_ms = 0
    
    def _find_stockfish(self, provided_path: Optional[str]) -> str:
        """
        Find Stockfish executable.
        
        Args:
            provided_path: User-provided path (takes precedence)
        
        Returns:
            Path to Stockfish executable
        
        Raises:
            FileNotFoundError: If Stockfish not found
        """
        if provided_path and Path(provided_path).exists():
            return provided_path
        
        # Common Stockfish locations
        possible_paths = [
            # Windows
            r"E:\Programming Stuff\Chess Engines\Tournament Engines\Stockfish\stockfish-windows-x86-64-avx2.exe",
            r"C:\Program Files\Stockfish\stockfish.exe",
            r"C:\stockfish\stockfish.exe",
            # Linux/Mac
            "/usr/local/bin/stockfish",
            "/usr/bin/stockfish",
            "./stockfish",
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return path
        
        raise FileNotFoundError(
            "Stockfish executable not found. Please provide path explicitly.\n"
            f"Searched locations: {possible_paths}"
        )
    
    def start(self):
        """Start Stockfish engine."""
        if self.engine is not None:
            return  # Already started
        
        self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        
        # Configure engine
        self.engine.configure({
            'Threads': self.threads,
            'Hash': self.hash_mb
        })
        
        print(f"[OK] Stockfish oracle started: {self.stockfish_path}")
        print(f"  Depth: {self.depth}, Time: {self.time_limit_ms}ms, "
              f"Threads: {self.threads}, Hash: {self.hash_mb}MB")
    
    def stop(self):
        """Stop Stockfish engine."""
        if self.engine is not None:
            self.engine.quit()
            self.engine = None
            print(f"[OK] Stockfish oracle stopped")
            print(f"  Total evaluations: {self._evaluation_count}")
            if self._evaluation_count > 0:
                avg_time = self._total_time_ms / self._evaluation_count
                print(f"  Average time: {avg_time:.1f}ms")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
    
    def evaluate(
        self,
        board: chess.Board,
        depth: Optional[int] = None,
        time_limit_ms: Optional[int] = None
    ) -> StockfishEvaluation:
        """
        Evaluate a position.
        
        Args:
            board: Chess position to evaluate
            depth: Search depth (uses default if None)
            time_limit_ms: Time limit in ms (uses default if None)
        
        Returns:
            StockfishEvaluation with score and best move
        """
        if self.engine is None:
            raise RuntimeError("Stockfish engine not started. Call start() first.")
        
        start_time = time.time()
        
        # Use provided limits or defaults
        search_depth = depth if depth is not None else self.depth
        search_time = time_limit_ms if time_limit_ms is not None else self.time_limit_ms
        
        # Perform analysis
        limit = chess.engine.Limit(
            depth=search_depth,
            time=search_time / 1000.0  # Convert to seconds
        )
        
        info = self.engine.analyse(board, limit)
        
        # Extract score
        score = info.get('score')
        if score is None:
            # Fallback: use evaluation from white's perspective
            score = chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE)
        
        # Convert to current player's perspective
        pov_score = score.relative
        
        score_cp = pov_score.score() if not pov_score.is_mate() else None
        score_mate = pov_score.mate() if pov_score.is_mate() else None
        
        # Extract best move and PV
        best_move = info.get('pv', [None])[0] if 'pv' in info else None
        pv = info.get('pv', [])
        
        # Track statistics
        elapsed_ms = int((time.time() - start_time) * 1000)
        self._evaluation_count += 1
        self._total_time_ms += elapsed_ms
        
        return StockfishEvaluation(
            score_cp=score_cp,
            score_mate=score_mate,
            best_move=best_move,
            pv=pv,
            depth=info.get('depth', search_depth),
            time_ms=elapsed_ms,
            nodes=info.get('nodes', 0)
        )
    
    def evaluate_batch(
        self,
        boards: List[chess.Board],
        depth: Optional[int] = None,
        time_limit_ms: Optional[int] = None
    ) -> List[StockfishEvaluation]:
        """
        Evaluate multiple positions (sequential for now, could parallelize).
        
        Args:
            boards: List of positions to evaluate
            depth: Search depth
            time_limit_ms: Time limit per position
        
        Returns:
            List of evaluations
        """
        return [
            self.evaluate(board, depth, time_limit_ms)
            for board in boards
        ]
    
    def get_training_label(
        self,
        board: chess.Board,
        game_result: Optional[str] = None,
        result_weight: float = 0.3
    ) -> float:
        """
        Generate training label combining Stockfish eval + game outcome.
        
        Args:
            board: Position to evaluate
            game_result: Game result ('1-0', '0-1', '1/2-1/2', or None)
            result_weight: How much to weight game result vs Stockfish eval
        
        Returns:
            Training label in [-1, 1] range
        """
        # Get Stockfish evaluation
        eval_result = self.evaluate(board)
        stockfish_label = eval_result.normalized_score
        
        # If game result available, blend it
        if game_result is not None:
            # Convert game result to label from current player's perspective
            if game_result == '1-0':
                outcome_label = 1.0 if board.turn == chess.WHITE else -1.0
            elif game_result == '0-1':
                outcome_label = -1.0 if board.turn == chess.WHITE else 1.0
            else:  # Draw
                outcome_label = 0.0
            
            # Weighted combination
            final_label = (
                stockfish_label * (1.0 - result_weight) +
                outcome_label * result_weight
            )
        else:
            final_label = stockfish_label
        
        return final_label
    
    def get_statistics(self) -> Dict:
        """Get oracle usage statistics."""
        avg_time = (
            self._total_time_ms / self._evaluation_count
            if self._evaluation_count > 0
            else 0.0
        )
        
        return {
            'evaluation_count': self._evaluation_count,
            'total_time_ms': self._total_time_ms,
            'average_time_ms': avg_time,
            'depth': self.depth,
            'time_limit_ms': self.time_limit_ms
        }


# Convenience function
def create_oracle(
    stockfish_path: Optional[str] = None,
    depth: int = 15,
    time_limit_ms: int = 1000
) -> StockfishOracle:
    """
    Create and start Stockfish oracle.
    
    Args:
        stockfish_path: Path to Stockfish (auto-detect if None)
        depth: Search depth
        time_limit_ms: Time limit per evaluation
    
    Returns:
        Started StockfishOracle instance
    """
    oracle = StockfishOracle(stockfish_path, depth, time_limit_ms)
    oracle.start()
    return oracle


# Example usage and validation
if __name__ == "__main__":
    print("="*60)
    print("V7P3R v7.0 - STOCKFISH ORACLE")
    print("="*60)
    
    # Create oracle with context manager
    try:
        with StockfishOracle(depth=12, time_limit_ms=500) as oracle:
            # Test starting position
            print("\n🧪 Testing Starting Position...")
            board = chess.Board()
            eval_result = oracle.evaluate(board)
            
            print(f"  FEN: {board.fen()}")
            print(f"  Score: {eval_result.score_cp}cp")
            print(f"  Best move: {eval_result.best_move}")
            print(f"  Normalized: {eval_result.normalized_score:.3f}")
            print(f"  Win probability: {eval_result.win_probability:.1%}")
            print(f"  Depth: {eval_result.depth}")
            print(f"  Time: {eval_result.time_ms}ms")
            
            # Test after e4
            print("\n🧪 Testing After 1.e4...")
            board.push_san('e4')
            eval_result = oracle.evaluate(board)
            
            print(f"  FEN: {board.fen()}")
            print(f"  Score: {eval_result.score_cp}cp")
            print(f"  Best move: {eval_result.best_move}")
            print(f"  Normalized: {eval_result.normalized_score:.3f}")
            
            # Test training label generation
            print("\n🧪 Testing Training Label Generation...")
            label = oracle.get_training_label(board, game_result='1-0')
            print(f"  Training label (with game result): {label:.3f}")
            
            # Test batch evaluation
            print("\n🧪 Testing Batch Evaluation...")
            test_boards = [chess.Board() for _ in range(3)]
            test_boards[1].push_san('e4')
            test_boards[2].push_san('d4')
            
            batch_results = oracle.evaluate_batch(test_boards)
            for i, result in enumerate(batch_results):
                print(f"  Position {i+1}: {result.score_cp}cp, "
                      f"best: {result.best_move}")
            
            # Show statistics
            print("\n📊 Oracle Statistics:")
            stats = oracle.get_statistics()
            for key, value in stats.items():
                print(f"  {key}: {value}")
            
            print(f"\n✅ Stockfish oracle validated!")
            
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nTo use Stockfish oracle, ensure Stockfish is installed:")
        print("  Download: https://stockfishchess.org/download/")
        print("  Then provide path when creating oracle:")
        print("  oracle = StockfishOracle(stockfish_path='path/to/stockfish')")
    
    print(f"\n📝 Next Steps:")
    print(f"  1. Build self-play trainer (src/v7/selfplay_trainer.py)")
    print(f"  2. Define personality rewards (src/v7/personality_rewards.py)")
    print(f"  3. Integrate: features → network → oracle → training")
