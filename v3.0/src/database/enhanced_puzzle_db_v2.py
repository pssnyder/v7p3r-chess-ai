"""
Enhanced Puzzle Database Schema V2

Improvements over V1:
1. **Stockfish Move Grading**: Store the actual Stockfish evaluation of AI's chosen move
2. **Learning Velocity**: Track how quickly AI improves on similar puzzles  
3. **Session Context**: Performance context when puzzle was encountered
4. **Temporal Patterns**: Time-based performance analytics
5. **Progressive Difficulty**: Track appropriate difficulty progression
6. **Theme Mastery**: Theme-specific performance tracking
7. **Move Quality Trends**: Historical progression of move quality
8. **Efficiency Metrics**: Learning efficiency and retention analysis

Key New Fields:
- ai_move_stockfish_score: Stockfish evaluation of AI's actual move
- ai_move_stockfish_rank: Where AI's move ranked in Stockfish top moves
- session_performance_context: AI's overall performance in this session
- time_since_last_encounter: Gap between encounters (learning retention)
- learning_velocity: Rate of improvement on similar puzzles
- difficulty_progression: Whether puzzle difficulty is appropriate
- theme_mastery_level: AI's current mastery of puzzle themes
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EnhancedPuzzleDatabaseV2:
    """Enhanced puzzle database with comprehensive AI performance tracking"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = sqlite3.connect(db_path)
        self.connection.row_factory = sqlite3.Row
        self._create_enhanced_schema()
        self._create_enhanced_indices()
        self._create_analytics_views()
    
    def _create_enhanced_schema(self):
        """Create enhanced database schema with new tracking fields"""
        cursor = self.connection.cursor()
        
        # Enhanced puzzles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS puzzles_v2 (
                id TEXT PRIMARY KEY,
                fen TEXT NOT NULL,
                moves TEXT NOT NULL,
                rating INTEGER,
                themes TEXT,
                popularity INTEGER DEFAULT 0,
                nb_plays INTEGER DEFAULT 0,
                
                -- Dataset classification
                dataset_split TEXT DEFAULT 'train',
                difficulty_tier TEXT DEFAULT 'medium',
                
                -- Basic AI Performance Tracking
                ai_encounter_count INTEGER DEFAULT 0,
                ai_solved_count INTEGER DEFAULT 0,
                ai_best_score INTEGER DEFAULT 0,
                ai_average_score REAL DEFAULT 0.0,
                ai_first_solved_date TEXT,
                ai_last_solved_date TEXT,
                ai_last_encounter_date TEXT,
                ai_consecutive_fails INTEGER DEFAULT 0,
                ai_regression_detected BOOLEAN DEFAULT 0,
                
                -- NEW: Enhanced Performance Metrics
                ai_best_stockfish_score INTEGER DEFAULT NULL,  -- Best Stockfish score AI achieved on this puzzle
                ai_latest_stockfish_score INTEGER DEFAULT NULL,  -- Most recent Stockfish score
                ai_stockfish_score_trend REAL DEFAULT 0.0,  -- Trend in Stockfish scores (positive = improving)
                ai_learning_velocity REAL DEFAULT 0.0,  -- Rate of improvement (points per encounter)
                ai_mastery_level TEXT DEFAULT 'novice',  -- 'novice', 'learning', 'competent', 'proficient', 'expert'
                ai_stability_score REAL DEFAULT 0.0,  -- Consistency of performance (0-1)
                
                -- NEW: Temporal and Context Analytics
                ai_average_solve_time REAL DEFAULT 0.0,
                ai_fastest_solve_time REAL DEFAULT 0.0,
                ai_solve_time_trend REAL DEFAULT 0.0,  -- Trend in solve times (negative = getting faster)
                ai_session_context_avg REAL DEFAULT 0.0,  -- Average session performance when encountering
                ai_time_between_encounters_avg REAL DEFAULT 0.0,  -- Average time between encounters
                ai_optimal_revisit_interval REAL DEFAULT 0.0,  -- Predicted optimal time for revisiting
                
                -- NEW: Difficulty and Theme Analytics
                difficulty_appropriateness REAL DEFAULT 0.5,  -- How appropriate difficulty is (0-1)
                theme_complexity_score REAL DEFAULT 0.0,  -- Complexity based on theme combinations
                prerequisite_themes_mastered BOOLEAN DEFAULT 0,  -- Whether prerequisite themes are mastered
                
                -- Metadata
                created_date TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_date TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Enhanced AI performance history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_performance_history_v2 (
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
                ai_move_quality TEXT,
                learning_context TEXT,
                model_version TEXT,
                session_id TEXT,
                
                -- NEW: Enhanced Move Analysis
                ai_move_stockfish_score INTEGER DEFAULT NULL,  -- Stockfish evaluation of AI's move
                ai_move_stockfish_rank INTEGER DEFAULT NULL,  -- Rank of AI's move in Stockfish top moves
                ai_move_stockfish_evaluation TEXT DEFAULT NULL,  -- JSON: full Stockfish evaluation
                move_improvement_from_last REAL DEFAULT 0.0,  -- Improvement from last encounter
                
                -- NEW: Session and Temporal Context
                session_performance_context REAL DEFAULT 0.0,  -- AI's session performance when this occurred
                time_since_last_encounter REAL DEFAULT 0.0,  -- Hours since last encounter with this puzzle
                cumulative_exposure_time REAL DEFAULT 0.0,  -- Total time spent on this puzzle type
                session_puzzle_number INTEGER DEFAULT 0,  -- Which puzzle in the session this was
                fatigue_indicator REAL DEFAULT 0.0,  -- Estimated fatigue level (0-1)
                
                -- NEW: Learning Context
                similar_puzzles_recent_performance REAL DEFAULT 0.0,  -- Performance on similar recent puzzles
                theme_confidence_at_encounter REAL DEFAULT 0.0,  -- AI's confidence in puzzle themes
                difficulty_match_score REAL DEFAULT 0.0,  -- How well difficulty matched AI's level
                
                FOREIGN KEY (puzzle_id) REFERENCES puzzles_v2 (id)
            )
        """)
        
        # Enhanced Stockfish grading table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stockfish_evaluations_v2 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fen TEXT NOT NULL,
                move TEXT NOT NULL,
                stockfish_score INTEGER,
                stockfish_centipawn_eval INTEGER DEFAULT NULL,  -- Centipawn evaluation
                stockfish_mate_in INTEGER DEFAULT NULL,  -- Mate in X moves (if applicable)
                stockfish_rank INTEGER,
                evaluation_depth INTEGER DEFAULT 15,
                evaluation_time REAL,
                evaluation_date TEXT DEFAULT CURRENT_TIMESTAMP,
                engine_version TEXT DEFAULT 'stockfish_15',
                
                -- NEW: Advanced Evaluation Metrics
                position_complexity REAL DEFAULT 0.0,  -- Positional complexity score
                tactical_themes TEXT DEFAULT NULL,  -- JSON array of detected tactical themes
                move_category TEXT DEFAULT 'unknown',  -- 'forcing', 'positional', 'defensive', etc.
                alternative_quality REAL DEFAULT 0.0,  -- Quality of other available moves
                
                UNIQUE(fen, move, evaluation_depth, engine_version)
            )
        """)
        
        # NEW: Theme mastery tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS theme_mastery (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                theme TEXT NOT NULL,
                model_version TEXT NOT NULL,
                
                -- Mastery Metrics
                puzzles_encountered INTEGER DEFAULT 0,
                puzzles_solved INTEGER DEFAULT 0,
                average_score REAL DEFAULT 0.0,
                mastery_level TEXT DEFAULT 'novice',
                confidence_score REAL DEFAULT 0.0,  -- 0-1 confidence in this theme
                
                -- Learning Analytics
                first_encounter_date TEXT,
                mastery_achieved_date TEXT DEFAULT NULL,
                regression_count INTEGER DEFAULT 0,
                learning_velocity REAL DEFAULT 0.0,
                
                -- Context
                difficulty_range_min INTEGER DEFAULT 1200,
                difficulty_range_max INTEGER DEFAULT 1800,
                prerequisite_themes TEXT DEFAULT NULL,  -- JSON array
                
                last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
                
                UNIQUE(theme, model_version)
            )
        """)
        
        # NEW: Learning efficiency analytics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_efficiency (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT NOT NULL,
                analysis_date TEXT NOT NULL,
                
                -- Efficiency Metrics
                puzzles_per_hour REAL DEFAULT 0.0,
                accuracy_improvement_rate REAL DEFAULT 0.0,  -- Improvement per 100 puzzles
                time_to_competency REAL DEFAULT 0.0,  -- Hours to reach competency on new themes
                retention_rate REAL DEFAULT 0.0,  -- How well performance is retained over time
                
                -- Performance Analytics
                average_session_length REAL DEFAULT 0.0,
                optimal_session_length REAL DEFAULT 0.0,
                fatigue_impact_score REAL DEFAULT 0.0,
                best_performance_time_of_day TEXT DEFAULT NULL,
                
                -- Difficulty Progression
                difficulty_progression_rate REAL DEFAULT 0.0,
                optimal_difficulty_increase REAL DEFAULT 0.0,
                challenge_preference_score REAL DEFAULT 0.0,  -- Preference for challenging puzzles
                
                -- Theme Learning
                theme_transfer_efficiency REAL DEFAULT 0.0,  -- How well learning transfers between themes
                weak_theme_count INTEGER DEFAULT 0,
                strong_theme_count INTEGER DEFAULT 0,
                
                UNIQUE(model_version, analysis_date)
            )
        """)
        
        # Enhanced training sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_sessions_v2 (
                session_id TEXT PRIMARY KEY,
                start_time TEXT NOT NULL,
                end_time TEXT,
                total_puzzles INTEGER DEFAULT 0,
                puzzles_solved INTEGER DEFAULT 0,
                average_score REAL DEFAULT 0.0,
                model_version TEXT,
                training_config TEXT,  -- JSON config
                performance_summary TEXT,  -- JSON summary
                notes TEXT,
                
                -- NEW: Session Analytics
                session_efficiency REAL DEFAULT 0.0,  -- Puzzles solved per minute
                learning_momentum REAL DEFAULT 0.0,  -- Performance improvement during session
                fatigue_detected BOOLEAN DEFAULT 0,
                peak_performance_time TEXT DEFAULT NULL,
                session_quality_score REAL DEFAULT 0.0,  -- Overall session quality (0-1)
                
                -- NEW: Contextual Information
                previous_session_gap REAL DEFAULT 0.0,  -- Hours since last session
                model_state_before TEXT DEFAULT NULL,  -- JSON: model performance before session
                model_state_after TEXT DEFAULT NULL,  -- JSON: model performance after session
                environmental_factors TEXT DEFAULT NULL  -- JSON: factors that might affect performance
            )
        """)
        
        self.connection.commit()
    
    def _create_enhanced_indices(self):
        """Create enhanced indices for performance"""
        cursor = self.connection.cursor()
        
        indices = [
            # Basic indices
            "CREATE INDEX IF NOT EXISTS idx_puzzles_v2_rating ON puzzles_v2(rating)",
            "CREATE INDEX IF NOT EXISTS idx_puzzles_v2_themes ON puzzles_v2(themes)",
            "CREATE INDEX IF NOT EXISTS idx_puzzles_v2_difficulty ON puzzles_v2(difficulty_tier)",
            "CREATE INDEX IF NOT EXISTS idx_puzzles_v2_dataset ON puzzles_v2(dataset_split)",
            
            # Performance indices
            "CREATE INDEX IF NOT EXISTS idx_puzzles_v2_mastery ON puzzles_v2(ai_mastery_level)",
            "CREATE INDEX IF NOT EXISTS idx_puzzles_v2_regression ON puzzles_v2(ai_regression_detected)",
            "CREATE INDEX IF NOT EXISTS idx_puzzles_v2_last_encounter ON puzzles_v2(ai_last_encounter_date)",
            
            # History indices
            "CREATE INDEX IF NOT EXISTS idx_history_v2_puzzle ON ai_performance_history_v2(puzzle_id)",
            "CREATE INDEX IF NOT EXISTS idx_history_v2_date ON ai_performance_history_v2(encounter_date)",
            "CREATE INDEX IF NOT EXISTS idx_history_v2_session ON ai_performance_history_v2(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_history_v2_model ON ai_performance_history_v2(model_version)",
            
            # Stockfish indices
            "CREATE INDEX IF NOT EXISTS idx_stockfish_v2_fen ON stockfish_evaluations_v2(fen)",
            "CREATE INDEX IF NOT EXISTS idx_stockfish_v2_position ON stockfish_evaluations_v2(fen, move)",
            
            # Theme mastery indices
            "CREATE INDEX IF NOT EXISTS idx_theme_mastery_theme ON theme_mastery(theme)",
            "CREATE INDEX IF NOT EXISTS idx_theme_mastery_model ON theme_mastery(model_version)",
            "CREATE INDEX IF NOT EXISTS idx_theme_mastery_level ON theme_mastery(mastery_level)",
            
            # Session indices
            "CREATE INDEX IF NOT EXISTS idx_sessions_v2_start ON training_sessions_v2(start_time)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_v2_model ON training_sessions_v2(model_version)",
        ]
        
        for index in indices:
            cursor.execute(index)
        
        self.connection.commit()
    
    def _create_analytics_views(self):
        """Create views for common analytics queries"""
        cursor = self.connection.cursor()
        
        # Learning progress view
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS learning_progress_summary AS
            SELECT 
                model_version,
                COUNT(*) as total_puzzles_encountered,
                SUM(CASE WHEN ai_solved_count > 0 THEN 1 ELSE 0 END) as puzzles_solved,
                AVG(ai_average_score) as overall_average_score,
                AVG(ai_learning_velocity) as average_learning_velocity,
                COUNT(CASE WHEN ai_mastery_level IN ('proficient', 'expert') THEN 1 END) as mastered_puzzles,
                COUNT(CASE WHEN ai_regression_detected = 1 THEN 1 END) as regression_puzzles
            FROM puzzles_v2 
            WHERE ai_encounter_count > 0
            GROUP BY model_version
        """)
        
        # Theme mastery summary view
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS theme_mastery_summary AS
            SELECT 
                theme,
                model_version,
                mastery_level,
                puzzles_solved * 100.0 / NULLIF(puzzles_encountered, 0) as solve_rate,
                confidence_score,
                learning_velocity,
                regression_count
            FROM theme_mastery
            ORDER BY confidence_score DESC, average_score DESC
        """)
        
        # Recent performance trends view
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS recent_performance_trends AS
            SELECT 
                DATE(encounter_date) as date,
                COUNT(*) as puzzles_attempted,
                SUM(CASE WHEN found_solution = 1 THEN 1 ELSE 0 END) as puzzles_solved,
                AVG(ai_score) as average_score,
                AVG(ai_move_stockfish_score) as average_stockfish_score,
                AVG(solve_time) as average_solve_time,
                AVG(session_performance_context) as average_session_context
            FROM ai_performance_history_v2 
            WHERE encounter_date >= date('now', '-30 days')
            GROUP BY DATE(encounter_date)
            ORDER BY date DESC
        """)
        
        self.connection.commit()
    
    def record_enhanced_ai_encounter(self, 
                                   puzzle_id: str,
                                   ai_move: str,
                                   expected_move: str,
                                   ai_score: int,
                                   ai_rank: int,
                                   found_solution: bool,
                                   solve_time: float,
                                   stockfish_top_moves: List[Tuple[str, int]],
                                   ai_move_stockfish_evaluation: Dict,  # NEW
                                   session_context: Dict,  # NEW
                                   learning_context: str = 'training',
                                   model_version: str = 'v3.0',
                                   session_id: Optional[str] = None) -> None:
        """Record AI encounter with enhanced analytics"""
        
        cursor = self.connection.cursor()
        encounter_date = datetime.now().isoformat()
        
        # Extract enhanced data from parameters
        ai_move_stockfish_score = ai_move_stockfish_evaluation.get('score', None)
        ai_move_stockfish_rank = ai_move_stockfish_evaluation.get('rank', None)
        session_performance_context = session_context.get('average_performance', 0.0)
        session_puzzle_number = session_context.get('puzzle_number', 0)
        fatigue_indicator = session_context.get('fatigue_estimate', 0.0)
        
        # Calculate time since last encounter
        time_since_last = self._calculate_time_since_last_encounter(puzzle_id)
        
        # Get similar puzzles recent performance
        similar_performance = self._get_similar_puzzles_performance(puzzle_id, model_version)
        
        # Assess move quality
        move_quality = self._assess_enhanced_move_quality(ai_score, ai_rank, found_solution, ai_move_stockfish_score)
        
        # Calculate improvement from last encounter
        improvement = self._calculate_improvement_from_last(puzzle_id, ai_score)
        
        # Insert enhanced performance history record
        cursor.execute("""
            INSERT INTO ai_performance_history_v2
            (puzzle_id, encounter_date, ai_move, expected_move, ai_score, ai_rank,
             found_solution, solve_time, stockfish_top_moves, ai_move_quality,
             learning_context, model_version, session_id,
             ai_move_stockfish_score, ai_move_stockfish_rank, ai_move_stockfish_evaluation,
             move_improvement_from_last, session_performance_context, time_since_last_encounter,
             cumulative_exposure_time, session_puzzle_number, fatigue_indicator, 
             similar_puzzles_recent_performance, theme_confidence_at_encounter, difficulty_match_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            puzzle_id, encounter_date, ai_move, expected_move, ai_score, ai_rank,
            found_solution, solve_time, json.dumps(stockfish_top_moves), move_quality,
            learning_context, model_version, session_id,
            ai_move_stockfish_score, ai_move_stockfish_rank, json.dumps(ai_move_stockfish_evaluation),
            improvement, session_performance_context, time_since_last,
            0.0,  # cumulative_exposure_time - default value
            session_puzzle_number, fatigue_indicator, similar_performance,
            0.0,  # theme_confidence_at_encounter - default value 
            0.0   # difficulty_match_score - default value
        ))
        
        # Update enhanced puzzle statistics
        self._update_enhanced_puzzle_stats(puzzle_id, ai_score, found_solution, solve_time, 
                                         encounter_date, ai_move_stockfish_score, session_context)
        
        # Update theme mastery
        self._update_theme_mastery(puzzle_id, found_solution, ai_score, model_version)
        
        self.connection.commit()
    
    def _calculate_time_since_last_encounter(self, puzzle_id: str) -> float:
        """Calculate hours since last encounter with this puzzle"""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT encounter_date FROM ai_performance_history_v2 
            WHERE puzzle_id = ? 
            ORDER BY encounter_date DESC 
            LIMIT 1
        """, (puzzle_id,))
        
        row = cursor.fetchone()
        if row:
            last_encounter = datetime.fromisoformat(row[0])
            return (datetime.now() - last_encounter).total_seconds() / 3600.0
        return 0.0
    
    def _get_similar_puzzles_performance(self, puzzle_id: str, model_version: str) -> float:
        """Get recent performance on puzzles with similar themes"""
        cursor = self.connection.cursor()
        
        # Get themes for this puzzle
        cursor.execute("SELECT themes FROM puzzles_v2 WHERE id = ?", (puzzle_id,))
        puzzle_row = cursor.fetchone()
        if not puzzle_row or not puzzle_row[0]:
            return 0.0
        
        themes = puzzle_row[0].split()
        if not themes:
            return 0.0
        
        # Find recent performance on similar puzzles
        placeholders = ','.join(['?' for _ in themes])
        cursor.execute(f"""
            SELECT AVG(h.ai_score)
            FROM ai_performance_history_v2 h
            JOIN puzzles_v2 p ON h.puzzle_id = p.id
            WHERE h.model_version = ? 
            AND h.encounter_date >= date('now', '-7 days')
            AND (p.themes LIKE '%' || ? || '%' OR {' OR '.join([f"p.themes LIKE '%' || ? || '%'" for _ in themes[1:]])})
        """, [model_version] + themes + themes[1:])
        
        result = cursor.fetchone()
        return result[0] if result and result[0] else 0.0
    
    def _assess_enhanced_move_quality(self, ai_score: int, ai_rank: int, found_solution: bool, stockfish_score: Optional[int]) -> str:
        """Enhanced move quality assessment including Stockfish evaluation"""
        if found_solution and ai_score == 5:
            return 'excellent'
        elif stockfish_score and stockfish_score >= 4:
            return 'very_good'
        elif ai_score >= 4 or (found_solution and ai_score >= 3):
            return 'good'
        elif ai_score >= 2 or (stockfish_score and stockfish_score >= 2):
            return 'fair'
        elif ai_score >= 1 or (stockfish_score and stockfish_score >= 1):
            return 'poor'
        else:
            return 'blunder'
    
    def _calculate_improvement_from_last(self, puzzle_id: str, current_score: int) -> float:
        """Calculate improvement from last encounter"""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT ai_score FROM ai_performance_history_v2 
            WHERE puzzle_id = ? 
            ORDER BY encounter_date DESC 
            LIMIT 1
        """, (puzzle_id,))
        
        row = cursor.fetchone()
        if row:
            return current_score - row[0]
        return 0.0
    
    def _update_enhanced_puzzle_stats(self, puzzle_id: str, ai_score: int, found_solution: bool, 
                                    solve_time: float, encounter_date: str, ai_move_stockfish_score: Optional[int],
                                    session_context: Dict):
        """Update puzzle statistics with enhanced metrics"""
        cursor = self.connection.cursor()
        
        # Get current stats
        cursor.execute("""
            SELECT ai_encounter_count, ai_solved_count, ai_best_score, ai_average_score,
                   ai_first_solved_date, ai_last_solved_date, ai_consecutive_fails,
                   ai_average_solve_time, ai_fastest_solve_time, ai_best_stockfish_score,
                   ai_latest_stockfish_score, ai_learning_velocity, ai_stability_score
            FROM puzzles_v2 WHERE id = ?
        """, (puzzle_id,))
        
        row = cursor.fetchone()
        if not row:
            # Initialize new puzzle
            cursor.execute("""
                INSERT INTO puzzles_v2 (id, ai_encounter_count, ai_solved_count, ai_best_score, 
                                      ai_average_score, ai_last_encounter_date)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (puzzle_id, 1, 1 if found_solution else 0, ai_score, ai_score, encounter_date))
            return
        
        # Calculate enhanced metrics
        encounter_count = row[0] + 1
        solved_count = row[1] + (1 if found_solution else 0)
        best_score = max(row[2], ai_score)
        avg_score = ((row[3] * row[0]) + ai_score) / encounter_count
        
        # Enhanced metrics
        best_stockfish_score = max(row[9] or 0, ai_move_stockfish_score or 0) if ai_move_stockfish_score else row[9]
        latest_stockfish_score = ai_move_stockfish_score
        
        # Calculate learning velocity (improvement rate)
        learning_velocity = self._calculate_learning_velocity(puzzle_id, ai_score)
        
        # Calculate stability score (consistency of performance)
        stability_score = self._calculate_stability_score(puzzle_id)
        
        # Assess mastery level
        mastery_level = self._assess_mastery_level(encounter_count, solved_count, avg_score, stability_score)
        
        # Update with enhanced metrics
        cursor.execute("""
            UPDATE puzzles_v2 SET
                ai_encounter_count = ?, ai_solved_count = ?, ai_best_score = ?, ai_average_score = ?,
                ai_last_encounter_date = ?, ai_best_stockfish_score = ?, ai_latest_stockfish_score = ?,
                ai_learning_velocity = ?, ai_stability_score = ?, ai_mastery_level = ?,
                updated_date = ?
            WHERE id = ?
        """, (
            encounter_count, solved_count, best_score, avg_score, encounter_date,
            best_stockfish_score, latest_stockfish_score, learning_velocity, stability_score,
            mastery_level, encounter_date, puzzle_id
        ))
    
    def _calculate_learning_velocity(self, puzzle_id: str, current_score: int) -> float:
        """Calculate learning velocity (improvement rate per encounter)"""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT ai_score FROM ai_performance_history_v2 
            WHERE puzzle_id = ? 
            ORDER BY encounter_date ASC
            LIMIT 10
        """, (puzzle_id,))
        
        scores = [row[0] for row in cursor.fetchall()]
        scores.append(current_score)
        
        if len(scores) < 2:
            return 0.0
        
        # Simple linear regression to find improvement trend
        n = len(scores)
        x_sum = sum(range(n))
        y_sum = sum(scores)
        xy_sum = sum(i * score for i, score in enumerate(scores))
        x2_sum = sum(i * i for i in range(n))
        
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum) if (n * x2_sum - x_sum * x_sum) != 0 else 0
        return slope
    
    def _calculate_stability_score(self, puzzle_id: str) -> float:
        """Calculate stability score (consistency of performance)"""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT ai_score FROM ai_performance_history_v2 
            WHERE puzzle_id = ? 
            ORDER BY encounter_date DESC
            LIMIT 10
        """, (puzzle_id,))
        
        scores = [row[0] for row in cursor.fetchall()]
        
        if len(scores) < 2:
            return 0.0
        
        # Calculate coefficient of variation (lower = more stable)
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        std_dev = variance ** 0.5
        
        if mean_score == 0:
            return 0.0
        
        cv = std_dev / mean_score
        # Convert to stability score (1 = perfectly stable, 0 = very unstable)
        return max(0, 1 - cv)
    
    def _assess_mastery_level(self, encounter_count: int, solved_count: int, avg_score: float, stability_score: float) -> str:
        """Assess AI's mastery level for this puzzle"""
        if encounter_count < 3:
            return 'novice'
        
        solve_rate = solved_count / encounter_count
        
        if avg_score >= 4.5 and solve_rate >= 0.8 and stability_score >= 0.8:
            return 'expert'
        elif avg_score >= 4.0 and solve_rate >= 0.7 and stability_score >= 0.6:
            return 'proficient'
        elif avg_score >= 3.0 and solve_rate >= 0.5:
            return 'competent'
        elif avg_score >= 2.0:
            return 'learning'
        else:
            return 'novice'
    
    def _update_theme_mastery(self, puzzle_id: str, found_solution: bool, ai_score: int, model_version: str):
        """Update theme mastery tracking"""
        cursor = self.connection.cursor()
        
        # Get puzzle themes
        cursor.execute("SELECT themes FROM puzzles_v2 WHERE id = ?", (puzzle_id,))
        puzzle_row = cursor.fetchone()
        if not puzzle_row or not puzzle_row[0]:
            return
        
        themes = puzzle_row[0].split()
        encounter_date = datetime.now().isoformat()
        
        for theme in themes:
            # Update or insert theme mastery record
            cursor.execute("""
                INSERT OR IGNORE INTO theme_mastery 
                (theme, model_version, first_encounter_date) 
                VALUES (?, ?, ?)
            """, (theme, model_version, encounter_date))
            
            # Get current theme stats
            cursor.execute("""
                SELECT puzzles_encountered, puzzles_solved, average_score 
                FROM theme_mastery 
                WHERE theme = ? AND model_version = ?
            """, (theme, model_version))
            
            row = cursor.fetchone()
            if row:
                new_encountered = row[0] + 1
                new_solved = row[1] + (1 if found_solution else 0)
                new_avg_score = ((row[2] * row[0]) + ai_score) / new_encountered
                
                # Calculate confidence score
                confidence = min(1.0, (new_solved / max(1, new_encountered)) * (new_avg_score / 5.0))
                
                # Determine mastery level
                if confidence >= 0.8 and new_avg_score >= 4.0:
                    mastery_level = 'expert'
                elif confidence >= 0.6 and new_avg_score >= 3.5:
                    mastery_level = 'proficient'
                elif confidence >= 0.4 and new_avg_score >= 2.5:
                    mastery_level = 'competent'
                elif new_encountered >= 5:
                    mastery_level = 'learning'
                else:
                    mastery_level = 'novice'
                
                cursor.execute("""
                    UPDATE theme_mastery SET
                        puzzles_encountered = ?, puzzles_solved = ?, average_score = ?,
                        confidence_score = ?, mastery_level = ?, last_updated = ?
                    WHERE theme = ? AND model_version = ?
                """, (new_encountered, new_solved, new_avg_score, confidence, mastery_level,
                      encounter_date, theme, model_version))
    
    def get_enhanced_analytics(self, model_version: str = 'v3.0') -> Dict:
        """Get comprehensive enhanced analytics"""
        cursor = self.connection.cursor()
        
        # Basic performance metrics
        cursor.execute("""
            SELECT COUNT(*) as total_puzzles, 
                   AVG(ai_average_score) as avg_score,
                   AVG(ai_learning_velocity) as avg_learning_velocity,
                   AVG(ai_stability_score) as avg_stability
            FROM puzzles_v2 WHERE ai_encounter_count > 0
        """)
        basic_stats = dict(cursor.fetchone())
        
        # Theme mastery summary
        cursor.execute("""
            SELECT theme, mastery_level, confidence_score, average_score
            FROM theme_mastery 
            WHERE model_version = ?
            ORDER BY confidence_score DESC
        """, (model_version,))
        theme_mastery = [dict(row) for row in cursor.fetchall()]
        
        # Recent performance trends
        cursor.execute("""
            SELECT date, puzzles_attempted, puzzles_solved, average_score, average_stockfish_score
            FROM recent_performance_trends
            LIMIT 14
        """)
        recent_trends = [dict(row) for row in cursor.fetchall()]
        
        # Learning efficiency metrics
        cursor.execute("""
            SELECT * FROM learning_efficiency 
            WHERE model_version = ?
            ORDER BY analysis_date DESC
            LIMIT 1
        """, (model_version,))
        efficiency_row = cursor.fetchone()
        efficiency_metrics = dict(efficiency_row) if efficiency_row else {}
        
        return {
            'basic_performance': basic_stats,
            'theme_mastery': theme_mastery,
            'recent_trends': recent_trends,
            'learning_efficiency': efficiency_metrics,
            'analysis_date': datetime.now().isoformat()
        }
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
