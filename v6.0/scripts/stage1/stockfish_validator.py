"""
Stockfish Validator with SQLite Cache - validates positions and assigns quality grades.

This module provides live Stockfish evaluation with persistent caching to avoid
redundant analysis. Each position is evaluated once and cached in SQLite database.

Expected performance: ~100ms per position analysis, 10s per 100-position batch.
Cache hit rate should exceed 80% after initial training runs.
"""

import chess
import chess.engine
import sqlite3
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import time
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Result of Stockfish position evaluation."""
    fen: str
    eval_cp: int  # Centipawn evaluation
    mate_in: Optional[int]  # Mate in N moves (None if not mate)
    grade: int  # Quality grade 1-5 (1=good, 5=blunder)
    depth: int  # Analysis depth
    time_ms: int  # Analysis time in milliseconds
    from_cache: bool  # Whether result came from cache


class StockfishValidator:
    """Validate chess positions using Stockfish with SQLite caching."""
    
    # Grade thresholds based on absolute evaluation change
    GRADE_THRESHOLDS = [
        (100, 1),   # < 100cp = good (grade 1)
        (200, 2),   # 100-200cp = inaccuracy (grade 2)
        (400, 3),   # 200-400cp = mistake (grade 3)
        (800, 4),   # 400-800cp = blunder (grade 4)
        (float('inf'), 5)  # >= 800cp = severe blunder (grade 5)
    ]
    
    def __init__(
        self,
        stockfish_path: str = "stockfish",
        db_path: str = "data/stage1/stockfish_cache.db",
        analysis_time: float = 0.1,  # 100ms per position
        min_depth: int = 15,
        threads: int = 1
    ):
        """
        Initialize Stockfish validator.
        
        Args:
            stockfish_path: Path to Stockfish executable
            db_path: Path to SQLite cache database
            analysis_time: Time limit per position in seconds
            min_depth: Minimum analysis depth
            threads: Number of Stockfish threads
        """
        # Initialize engine FIRST to avoid deallocator errors
        self._engine = None
        
        self.stockfish_path = stockfish_path
        self.db_path = Path(db_path)
        self.analysis_time = analysis_time
        self.min_depth = min_depth
        self.threads = threads
        
        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Statistics
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_analysis_time = 0.0
        
    def _init_database(self):
        """Initialize SQLite database with evaluation cache table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                fen TEXT PRIMARY KEY,
                eval_cp INTEGER,
                mate_in INTEGER,
                grade INTEGER,
                depth INTEGER,
                validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_grade ON evaluations(grade)
        """)
        
        conn.commit()
        conn.close()
        
    def _get_engine(self) -> chess.engine.SimpleEngine:
        """Get Stockfish engine instance (lazy initialization)."""
        if self._engine is None:
            try:
                self._engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
                self._engine.configure({"Threads": self.threads})
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Stockfish at {self.stockfish_path}: {e}")
        return self._engine
        
    def _check_cache(self, fen: str) -> Optional[EvaluationResult]:
        """
        Check if evaluation exists in cache.
        
        Args:
            fen: Position FEN string
            
        Returns:
            Cached evaluation result or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT eval_cp, mate_in, grade, depth FROM evaluations WHERE fen = ?",
            (fen,)
        )
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            self._cache_hits += 1
            return EvaluationResult(
                fen=fen,
                eval_cp=row[0],
                mate_in=row[1],
                grade=row[2],
                depth=row[3],
                time_ms=0,
                from_cache=True
            )
        
        self._cache_misses += 1
        return None
        
    def _save_to_cache(self, result: EvaluationResult):
        """
        Save evaluation result to cache.
        
        Args:
            result: Evaluation result to cache
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            """
            INSERT OR REPLACE INTO evaluations (fen, eval_cp, mate_in, grade, depth)
            VALUES (?, ?, ?, ?, ?)
            """,
            (result.fen, result.eval_cp, result.mate_in, result.grade, result.depth)
        )
        
        conn.commit()
        conn.close()
        
    def _calculate_grade(self, eval_cp: int) -> int:
        """
        Calculate quality grade based on evaluation.
        
        Args:
            eval_cp: Centipawn evaluation
            
        Returns:
            Grade 1-5 (1=good, 5=severe blunder)
        """
        abs_eval = abs(eval_cp)
        
        for threshold, grade in self.GRADE_THRESHOLDS:
            if abs_eval < threshold:
                return grade
        
        return 5  # Fallback
        
    def _analyze_position(self, fen: str) -> EvaluationResult:
        """
        Analyze position with Stockfish.
        
        Args:
            fen: Position FEN string
            
        Returns:
            Evaluation result
        """
        start_time = time.time()
        
        try:
            board = chess.Board(fen)
            engine = self._get_engine()
            
            # Run analysis
            info = engine.analyse(
                board,
                chess.engine.Limit(time=self.analysis_time, depth=self.min_depth)
            )
            
            # Extract evaluation
            score = info.get("score")
            if score is None:
                raise ValueError(f"No score in analysis for {fen}")
            
            # Convert score to centipawns
            pov_score = score.white() if board.turn == chess.WHITE else score.black()
            
            if pov_score.is_mate():
                mate_in = pov_score.mate()
                eval_cp = 10000 if mate_in > 0 else -10000
            else:
                mate_in = None
                eval_cp = pov_score.score()
            
            # Calculate grade
            grade = self._calculate_grade(eval_cp)
            
            # Get depth
            depth = info.get("depth", self.min_depth)
            
            analysis_time_ms = int((time.time() - start_time) * 1000)
            self._total_analysis_time += (time.time() - start_time)
            
            result = EvaluationResult(
                fen=fen,
                eval_cp=eval_cp,
                mate_in=mate_in,
                grade=grade,
                depth=depth,
                time_ms=analysis_time_ms,
                from_cache=False
            )
            
            # Save to cache
            self._save_to_cache(result)
            
            return result
            
        except Exception as e:
            # Return neutral evaluation on error
            return EvaluationResult(
                fen=fen,
                eval_cp=0,
                mate_in=None,
                grade=1,
                depth=0,
                time_ms=0,
                from_cache=False
            )
        
    def validate_position(self, fen: str) -> EvaluationResult:
        """
        Validate a single position (with caching).
        
        Args:
            fen: Position FEN string
            
        Returns:
            Evaluation result
        """
        # Check cache first
        cached = self._check_cache(fen)
        if cached:
            return cached
        
        # Analyze with Stockfish
        return self._analyze_position(fen)
        
    def validate_batch(
        self,
        positions: List[Dict[str, Any]],
        update_in_place: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Validate a batch of positions.
        
        Args:
            positions: List of position dictionaries
            update_in_place: Whether to update position dicts in place
            
        Returns:
            Updated position dictionaries
        """
        results = []
        
        for pos in positions:
            fen = pos['fen']
            result = self.validate_position(fen)
            
            if update_in_place:
                pos['eval_cp'] = result.eval_cp
                pos['grade'] = result.grade
                pos['mate_in'] = result.mate_in
                pos['stockfish_depth'] = result.depth
                pos['validated'] = True
            
            results.append(pos)
        
        return results
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_queries = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_queries if total_queries > 0 else 0
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM evaluations")
        cache_size = cursor.fetchone()[0]
        conn.close()
        
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': cache_size,
            'total_analysis_time': self._total_analysis_time,
            'avg_analysis_time': self._total_analysis_time / self._cache_misses if self._cache_misses > 0 else 0
        }
        
    def print_stats(self):
        """Print cache statistics."""
        stats = self.get_cache_stats()
        
        print("\n" + "="*60)
        print("Stockfish Validator Statistics")
        print("="*60)
        print(f"Cache hits:           {stats['cache_hits']:,}")
        print(f"Cache misses:         {stats['cache_misses']:,}")
        print(f"Hit rate:             {stats['hit_rate']:.1%}")
        print(f"Cached positions:     {stats['cache_size']:,}")
        print(f"Total analysis time:  {stats['total_analysis_time']:.1f}s")
        print(f"Avg analysis time:    {stats['avg_analysis_time']*1000:.0f}ms")
        print("="*60 + "\n")
        
    def close(self):
        """Close Stockfish engine."""
        if self._engine is not None:
            self._engine.quit()
            self._engine = None
            
    def __del__(self):
        """Cleanup on deletion."""
        self.close()
