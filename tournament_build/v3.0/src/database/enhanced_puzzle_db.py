"""
Enhanced Puzzle Database for V7P3R AI Training

This database system tracks AI performance on individual puzzles over time,
enabling sophisticated learning with historical context and regression detection.

Features:
- AI performance tracking (solved count, scores, timing)
- Stockfish grading history for move quality analysis
- Dataset splits (train/test/validation)
- Regression detection (when AI starts failing previously solved puzzles)
- Rich metadata for enhanced neural network features
"""

import sqlite3
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class EnhancedPuzzleDatabase:
    """
    Enhanced puzzle database with AI performance tracking and learning analytics
    """
    
    def __init__(self, db_path: str = "v7p3r_puzzle_trainer.db"):
        self.db_path = Path(db_path)
        self.connection = None
        self._init_database()
    
    def _init_database(self):
        """Initialize database with enhanced schema for AI tracking"""
        self.connection = sqlite3.connect(str(self.db_path))
        self.connection.row_factory = sqlite3.Row  # Enable dict-like access
        
        # Create enhanced tables
        self._create_tables()
        self._create_indices()
        
        logger.info(f"Enhanced puzzle database initialized: {self.db_path}")
    
    def _create_tables(self):
        """Create database tables with AI performance tracking"""
        cursor = self.connection.cursor()
        
        # Main puzzles table (enhanced)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS puzzles (
                id TEXT PRIMARY KEY,
                fen TEXT NOT NULL,
                moves TEXT NOT NULL,
                rating INTEGER,
                themes TEXT,
                popularity INTEGER DEFAULT 0,
                nb_plays INTEGER DEFAULT 0,
                
                -- Dataset classification
                dataset_split TEXT DEFAULT 'train',  -- 'train', 'test', 'validation'
                difficulty_tier TEXT DEFAULT 'medium',  -- 'easy', 'medium', 'hard', 'expert'
                
                -- AI Performance Tracking
                ai_encounter_count INTEGER DEFAULT 0,
                ai_solved_count INTEGER DEFAULT 0,
                ai_best_score INTEGER DEFAULT 0,
                ai_average_score REAL DEFAULT 0.0,
                ai_first_solved_date TEXT,
                ai_last_solved_date TEXT,
                ai_last_encounter_date TEXT,
                ai_consecutive_fails INTEGER DEFAULT 0,
                ai_regression_detected BOOLEAN DEFAULT 0,
                
                -- Timing and Analysis
                ai_average_solve_time REAL DEFAULT 0.0,
                ai_fastest_solve_time REAL DEFAULT 0.0,
                stockfish_baseline_score INTEGER DEFAULT 0,
                
                -- Metadata
                created_date TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_date TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # AI performance history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_performance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                puzzle_id TEXT NOT NULL,
                encounter_date TEXT NOT NULL,
                ai_move TEXT,
                expected_move TEXT,
                ai_score INTEGER,
                ai_rank INTEGER,
                found_solution BOOLEAN,
                solve_time REAL,
                stockfish_top_moves TEXT,  -- JSON array
                ai_move_quality TEXT,  -- 'excellent', 'good', 'fair', 'poor', 'blunder'
                learning_context TEXT,  -- 'training', 'validation', 'testing'
                model_version TEXT,
                session_id TEXT,
                
                FOREIGN KEY (puzzle_id) REFERENCES puzzles (id)
            )
        """)
        
        # Stockfish grading reference table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stockfish_gradings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fen TEXT NOT NULL,
                move TEXT NOT NULL,
                stockfish_score INTEGER,
                stockfish_rank INTEGER,
                evaluation_depth INTEGER DEFAULT 15,
                evaluation_time REAL,
                evaluation_date TEXT DEFAULT CURRENT_TIMESTAMP,
                
                UNIQUE(fen, move, evaluation_depth)
            )
        """)
        
        # Training sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_sessions (
                session_id TEXT PRIMARY KEY,
                start_time TEXT NOT NULL,
                end_time TEXT,
                total_puzzles INTEGER DEFAULT 0,
                puzzles_solved INTEGER DEFAULT 0,
                average_score REAL DEFAULT 0.0,
                model_version TEXT,
                training_config TEXT,  -- JSON config
                performance_summary TEXT,  -- JSON summary
                notes TEXT
            )
        """)
        
        self.connection.commit()
    
    def _create_indices(self):
        """Create database indices for performance"""
        cursor = self.connection.cursor()
        
        indices = [
            "CREATE INDEX IF NOT EXISTS idx_puzzles_rating ON puzzles(rating)",
            "CREATE INDEX IF NOT EXISTS idx_puzzles_themes ON puzzles(themes)",
            "CREATE INDEX IF NOT EXISTS idx_puzzles_dataset ON puzzles(dataset_split)",
            "CREATE INDEX IF NOT EXISTS idx_puzzles_difficulty ON puzzles(difficulty_tier)",
            "CREATE INDEX IF NOT EXISTS idx_puzzles_ai_score ON puzzles(ai_best_score)",
            "CREATE INDEX IF NOT EXISTS idx_performance_puzzle ON ai_performance_history(puzzle_id)",
            "CREATE INDEX IF NOT EXISTS idx_performance_date ON ai_performance_history(encounter_date)",
            "CREATE INDEX IF NOT EXISTS idx_stockfish_fen ON stockfish_gradings(fen)",
            "CREATE INDEX IF NOT EXISTS idx_stockfish_move ON stockfish_gradings(move)"
        ]
        
        for index_sql in indices:
            cursor.execute(index_sql)
        
        self.connection.commit()
    
    def import_original_puzzles(self, original_db_path: str, limit: Optional[int] = None):
        """Import puzzles from original database"""
        try:
            original_conn = sqlite3.connect(original_db_path)
            original_conn.row_factory = sqlite3.Row
            
            cursor = original_conn.cursor()
            
            # Get puzzles from original database
            query = "SELECT * FROM puzzles"
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query)
            puzzles = cursor.fetchall()
            
            imported_count = 0
            for puzzle in puzzles:
                # Classify puzzle difficulty based on rating
                difficulty_tier = self._classify_difficulty(puzzle['rating'] or 1500)
                
                # Insert with enhanced fields
                self.connection.execute("""
                    INSERT OR REPLACE INTO puzzles 
                    (id, fen, moves, rating, themes, popularity, nb_plays, difficulty_tier)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    puzzle['id'],
                    puzzle['fen'],
                    puzzle['moves'],
                    puzzle['rating'],
                    puzzle['themes'],
                    puzzle.get('popularity', 0),
                    puzzle.get('nb_plays', 0),
                    difficulty_tier
                ))
                imported_count += 1
            
            self.connection.commit()
            original_conn.close()
            
            logger.info(f"Imported {imported_count} puzzles from {original_db_path}")
            return imported_count
            
        except Exception as e:
            logger.error(f"Error importing puzzles: {e}")
            return 0
    
    def _classify_difficulty(self, rating: int) -> str:
        """Classify puzzle difficulty based on rating"""
        if rating < 1300:
            return 'easy'
        elif rating < 1600:
            return 'medium'
        elif rating < 2000:
            return 'hard'
        else:
            return 'expert'
    
    def assign_dataset_splits(self, train_ratio: float = 0.7, test_ratio: float = 0.2, validation_ratio: float = 0.1):
        """Assign puzzles to train/test/validation datasets"""
        import random
        
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM puzzles ORDER BY RANDOM()")
        puzzle_ids = [row[0] for row in cursor.fetchall()]
        
        total = len(puzzle_ids)
        train_end = int(total * train_ratio)
        test_end = train_end + int(total * test_ratio)
        
        # Assign splits
        for i, puzzle_id in enumerate(puzzle_ids):
            if i < train_end:
                split = 'train'
            elif i < test_end:
                split = 'test'
            else:
                split = 'validation'
            
            cursor.execute(
                "UPDATE puzzles SET dataset_split = ? WHERE id = ?",
                (split, puzzle_id)
            )
        
        self.connection.commit()
        
        train_count = train_end
        test_count = test_end - train_end
        val_count = total - test_end
        
        logger.info(f"Dataset splits assigned: {train_count} train, {test_count} test, {val_count} validation")
        return train_count, test_count, val_count
    
    def get_puzzles_for_training(self, 
                               dataset: str = 'train',
                               difficulty_tier: Optional[str] = None,
                               themes: Optional[List[str]] = None,
                               rating_min: int = 1000,
                               rating_max: int = 2500,
                               limit: int = 1000,
                               exclude_recently_solved: bool = True) -> List[Dict]:
        """Get puzzles for training with filtering options"""
        
        cursor = self.connection.cursor()
        
        conditions = ["dataset_split = ?"]
        params = [dataset]
        
        if difficulty_tier:
            conditions.append("difficulty_tier = ?")
            params.append(difficulty_tier)
        
        if rating_min:
            conditions.append("rating >= ?")
            params.append(rating_min)
        
        if rating_max:
            conditions.append("rating <= ?")
            params.append(rating_max)
        
        if themes:
            theme_conditions = []
            for theme in themes:
                theme_conditions.append("themes LIKE ?")
                params.append(f"%{theme}%")
            conditions.append(f"({' OR '.join(theme_conditions)})")
        
        if exclude_recently_solved:
            # Exclude puzzles solved in last 24 hours to ensure variety
            conditions.append("""
                (ai_last_solved_date IS NULL OR 
                 datetime(ai_last_solved_date) < datetime('now', '-1 day'))
            """)
        
        query = f"""
            SELECT * FROM puzzles 
            WHERE {' AND '.join(conditions)}
            ORDER BY 
                ai_encounter_count ASC,  -- Prefer less encountered puzzles
                RANDOM()
            LIMIT ?
        """
        params.append(limit)
        
        cursor.execute(query, params)
        puzzles = [dict(row) for row in cursor.fetchall()]
        
        logger.info(f"Retrieved {len(puzzles)} puzzles for {dataset} dataset")
        return puzzles
    
    def record_ai_encounter(self, 
                          puzzle_id: str,
                          ai_move: str,
                          expected_move: str,
                          ai_score: int,
                          ai_rank: int,
                          found_solution: bool,
                          solve_time: float,
                          stockfish_top_moves: List[Tuple[str, int]],
                          learning_context: str = 'training',
                          model_version: str = 'v3.0',
                          session_id: str = None) -> None:
        """Record AI encounter with a puzzle"""
        
        cursor = self.connection.cursor()
        encounter_date = datetime.now().isoformat()
        
        # Determine move quality based on score and rank
        move_quality = self._assess_move_quality(ai_score, ai_rank, found_solution)
        
        # Insert performance history record
        cursor.execute("""
            INSERT INTO ai_performance_history
            (puzzle_id, encounter_date, ai_move, expected_move, ai_score, ai_rank,
             found_solution, solve_time, stockfish_top_moves, ai_move_quality,
             learning_context, model_version, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            puzzle_id, encounter_date, ai_move, expected_move, ai_score, ai_rank,
            found_solution, solve_time, json.dumps(stockfish_top_moves), move_quality,
            learning_context, model_version, session_id
        ))
        
        # Update puzzle statistics
        self._update_puzzle_stats(puzzle_id, ai_score, found_solution, solve_time, encounter_date)
        
        self.connection.commit()
    
    def _assess_move_quality(self, score: int, rank: int, found_solution: bool) -> str:
        """Assess move quality based on performance metrics"""
        if found_solution and score == 5:
            return 'excellent'
        elif score >= 4 or (found_solution and score >= 3):
            return 'good'
        elif score >= 2:
            return 'fair'
        elif score >= 1:
            return 'poor'
        else:
            return 'blunder'
    
    def _update_puzzle_stats(self, puzzle_id: str, ai_score: int, found_solution: bool, solve_time: float, encounter_date: str):
        """Update puzzle statistics with new encounter data"""
        cursor = self.connection.cursor()
        
        # Get current stats
        cursor.execute("""
            SELECT ai_encounter_count, ai_solved_count, ai_best_score, ai_average_score,
                   ai_first_solved_date, ai_last_solved_date, ai_consecutive_fails,
                   ai_average_solve_time, ai_fastest_solve_time
            FROM puzzles WHERE id = ?
        """, (puzzle_id,))
        
        row = cursor.fetchone()
        if not row:
            return
        
        # Calculate updated statistics
        encounter_count = row[0] + 1
        solved_count = row[1] + (1 if found_solution else 0)
        best_score = max(row[2], ai_score)
        
        # Calculate new average score
        if encounter_count == 1:
            avg_score = ai_score
        else:
            avg_score = ((row[3] * (encounter_count - 1)) + ai_score) / encounter_count
        
        # Update solve dates
        first_solved = row[4] if row[4] else (encounter_date if found_solution else None)
        last_solved = encounter_date if found_solution else row[5]
        
        # Track consecutive failures for regression detection
        if found_solution:
            consecutive_fails = 0
        else:
            consecutive_fails = row[6] + 1
        
        # Update solve time statistics
        if solve_time > 0:
            if row[7] == 0:  # First timing record
                avg_solve_time = solve_time
                fastest_time = solve_time
            else:
                avg_solve_time = ((row[7] * (encounter_count - 1)) + solve_time) / encounter_count
                fastest_time = min(row[8], solve_time) if row[8] > 0 else solve_time
        else:
            avg_solve_time = row[7]
            fastest_time = row[8]
        
        # Detect regression (failing puzzles previously solved consistently)
        regression_detected = (consecutive_fails >= 3 and solved_count >= 3)
        
        # Update puzzle record
        cursor.execute("""
            UPDATE puzzles SET
                ai_encounter_count = ?,
                ai_solved_count = ?,
                ai_best_score = ?,
                ai_average_score = ?,
                ai_first_solved_date = ?,
                ai_last_solved_date = ?,
                ai_last_encounter_date = ?,
                ai_consecutive_fails = ?,
                ai_regression_detected = ?,
                ai_average_solve_time = ?,
                ai_fastest_solve_time = ?,
                updated_date = ?
            WHERE id = ?
        """, (
            encounter_count, solved_count, best_score, avg_score,
            first_solved, last_solved, encounter_date, consecutive_fails,
            regression_detected, avg_solve_time, fastest_time,
            encounter_date, puzzle_id
        ))
    
    def get_puzzle_performance_features(self, puzzle_id: str) -> Dict:
        """Get puzzle performance features for neural network input"""
        cursor = self.connection.cursor()
        
        # Get puzzle stats
        cursor.execute("""
            SELECT ai_encounter_count, ai_solved_count, ai_best_score, ai_average_score,
                   ai_consecutive_fails, ai_regression_detected, ai_average_solve_time,
                   difficulty_tier, rating
            FROM puzzles WHERE id = ?
        """, (puzzle_id,))
        
        puzzle_row = cursor.fetchone()
        if not puzzle_row:
            return self._get_default_performance_features()
        
        # Get recent performance history (last 5 encounters)
        cursor.execute("""
            SELECT ai_score, ai_rank, found_solution, solve_time, ai_move_quality
            FROM ai_performance_history 
            WHERE puzzle_id = ?
            ORDER BY encounter_date DESC
            LIMIT 5
        """, (puzzle_id,))
        
        recent_history = cursor.fetchall()
        
        # Compute performance features
        features = {
            # Basic encounter statistics
            'encounter_count': puzzle_row[0],
            'solved_count': puzzle_row[1],
            'solve_rate': puzzle_row[1] / max(1, puzzle_row[0]),  # Avoid division by zero
            'best_score': puzzle_row[2],
            'average_score': puzzle_row[3],
            'consecutive_fails': puzzle_row[4],
            'regression_detected': int(puzzle_row[5]),
            'average_solve_time': puzzle_row[6],
            
            # Difficulty context
            'difficulty_tier_encoded': self._encode_difficulty(puzzle_row[7]),
            'rating_normalized': (puzzle_row[8] - 1000) / 2000.0 if puzzle_row[8] else 0.5,
            
            # Recent performance trends
            'recent_performance_trend': self._calculate_performance_trend(recent_history),
            'recent_avg_score': self._calculate_recent_average(recent_history, 'score'),
            'recent_avg_time': self._calculate_recent_average(recent_history, 'time'),
            'last_encounter_success': int(recent_history[0][2]) if recent_history else 0,
            
            # Move quality indicators
            'excellent_moves': len([h for h in recent_history if h[4] == 'excellent']),
            'blunder_moves': len([h for h in recent_history if h[4] == 'blunder']),
        }
        
        return features
    
    def _get_default_performance_features(self) -> Dict:
        """Get default performance features for new puzzles"""
        return {
            'encounter_count': 0,
            'solved_count': 0,
            'solve_rate': 0.0,
            'best_score': 0,
            'average_score': 0.0,
            'consecutive_fails': 0,
            'regression_detected': 0,
            'average_solve_time': 0.0,
            'difficulty_tier_encoded': 0.5,  # Medium difficulty default
            'rating_normalized': 0.5,
            'recent_performance_trend': 0.0,
            'recent_avg_score': 0.0,
            'recent_avg_time': 0.0,
            'last_encounter_success': 0,
            'excellent_moves': 0,
            'blunder_moves': 0,
        }
    
    def _encode_difficulty(self, difficulty_tier: str) -> float:
        """Encode difficulty tier as numerical value"""
        mapping = {'easy': 0.0, 'medium': 0.33, 'hard': 0.67, 'expert': 1.0}
        return mapping.get(difficulty_tier, 0.5)
    
    def _calculate_performance_trend(self, history: List) -> float:
        """Calculate performance trend from recent history"""
        if len(history) < 2:
            return 0.0
        
        scores = [h[0] for h in reversed(history)]  # Chronological order
        
        # Simple linear trend calculation
        n = len(scores)
        x_sum = sum(range(n))
        y_sum = sum(scores)
        xy_sum = sum(i * score for i, score in enumerate(scores))
        x2_sum = sum(i * i for i in range(n))
        
        # Linear regression slope
        denominator = n * x2_sum - x_sum * x_sum
        if denominator == 0:
            return 0.0
        
        slope = (n * xy_sum - x_sum * y_sum) / denominator
        return slope / 5.0  # Normalize to -1 to 1 range
    
    def _calculate_recent_average(self, history: List, metric: str) -> float:
        """Calculate recent average for a specific metric"""
        if not history:
            return 0.0
        
        if metric == 'score':
            values = [h[0] for h in history]
        elif metric == 'time':
            values = [h[3] for h in history if h[3] > 0]
        else:
            return 0.0
        
        return sum(values) / len(values) if values else 0.0
    
    def store_stockfish_grading(self, fen: str, move: str, score: int, rank: int, depth: int = 15):
        """Store Stockfish grading for move quality reference"""
        cursor = self.connection.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO stockfish_gradings
            (fen, move, stockfish_score, stockfish_rank, evaluation_depth)
            VALUES (?, ?, ?, ?, ?)
        """, (fen, move, score, rank, depth))
        
        self.connection.commit()
    
    def get_stockfish_grading(self, fen: str, move: str) -> Optional[Dict]:
        """Get stored Stockfish grading for a position/move"""
        cursor = self.connection.cursor()
        
        cursor.execute("""
            SELECT stockfish_score, stockfish_rank, evaluation_depth, evaluation_date
            FROM stockfish_gradings
            WHERE fen = ? AND move = ?
            ORDER BY evaluation_depth DESC, evaluation_date DESC
            LIMIT 1
        """, (fen, move))
        
        row = cursor.fetchone()
        if row:
            return {
                'score': row[0],
                'rank': row[1],
                'depth': row[2],
                'date': row[3]
            }
        return None
    
    def start_training_session(self, session_id: str, model_version: str, training_config: Dict) -> None:
        """Start a new training session"""
        cursor = self.connection.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO training_sessions
            (session_id, start_time, model_version, training_config)
            VALUES (?, ?, ?, ?)
        """, (session_id, datetime.now().isoformat(), model_version, json.dumps(training_config)))
        
        self.connection.commit()
    
    def end_training_session(self, session_id: str, performance_summary: Dict, notes: str = "") -> None:
        """End a training session with results"""
        cursor = self.connection.cursor()
        
        cursor.execute("""
            UPDATE training_sessions SET
                end_time = ?,
                total_puzzles = ?,
                puzzles_solved = ?,
                average_score = ?,
                performance_summary = ?,
                notes = ?
            WHERE session_id = ?
        """, (
            datetime.now().isoformat(),
            performance_summary.get('total_puzzles', 0),
            performance_summary.get('puzzles_solved', 0),
            performance_summary.get('average_score', 0.0),
            json.dumps(performance_summary),
            notes,
            session_id
        ))
        
        self.connection.commit()
    
    def get_regression_puzzles(self, limit: int = 50) -> List[Dict]:
        """Get puzzles where AI performance has regressed"""
        cursor = self.connection.cursor()
        
        cursor.execute("""
            SELECT * FROM puzzles
            WHERE ai_regression_detected = 1
            ORDER BY ai_consecutive_fails DESC, ai_last_encounter_date DESC
            LIMIT ?
        """, (limit,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_performance_analytics(self) -> Dict:
        """Get comprehensive performance analytics"""
        cursor = self.connection.cursor()
        
        # Overall statistics
        cursor.execute("""
            SELECT 
                COUNT(*) as total_puzzles,
                COUNT(CASE WHEN ai_encounter_count > 0 THEN 1 END) as encountered_puzzles,
                COUNT(CASE WHEN ai_solved_count > 0 THEN 1 END) as solved_puzzles,
                AVG(ai_average_score) as overall_avg_score,
                COUNT(CASE WHEN ai_regression_detected = 1 THEN 1 END) as regression_puzzles
            FROM puzzles
        """)
        
        stats = dict(cursor.fetchone())
        
        # Performance by difficulty
        cursor.execute("""
            SELECT 
                difficulty_tier,
                COUNT(*) as count,
                AVG(ai_average_score) as avg_score,
                AVG(ai_average_solve_time) as avg_time
            FROM puzzles
            WHERE ai_encounter_count > 0
            GROUP BY difficulty_tier
        """)
        
        difficulty_stats = {row[0]: {'count': row[1], 'avg_score': row[2], 'avg_time': row[3]} 
                          for row in cursor.fetchall()}
        
        return {
            'overall_stats': stats,
            'difficulty_performance': difficulty_stats,
            'database_path': str(self.db_path),
            'last_updated': datetime.now().isoformat()
        }
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
