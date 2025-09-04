"""
Enhanced Puzzle Trainer V2 - Uses Enhanced Database Schema V2

This trainer leverages the new V2 database schema for comprehensive learning analytics:
- Stockfish grading of AI moves
- Learning velocity tracking
- Theme mastery analysis
- Session context awareness
- Temporal learning patterns
"""

import os
import sys
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from tqdm import tqdm
import logging

# Custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ai.thinking_brain import ThinkingBrain, PositionMemory
from core.chess_state import ChessStateExtractor
from core.neural_features import NeuralFeatureConverter
from database.enhanced_puzzle_db_v2 import EnhancedPuzzleDatabaseV2
from training.puzzle_trainer import PuzzleTrainer

logger = logging.getLogger(__name__)


class EnhancedPuzzleTrainerV2(PuzzleTrainer):
    """
    Enhanced puzzle trainer V2 with comprehensive analytics and learning intelligence
    """
    
    def __init__(self, 
                 thinking_brain: ThinkingBrain,
                 puzzle_db_path: str = "data/v7p3rai_puzzle_training_v2.db",
                 stockfish_path: Optional[str] = None,
                 save_directory: str = "models/enhanced_puzzle_training_v2",
                 memory_config: Optional[Dict] = None,
                 model_version: str = "v3.0"):
        
        # Set default Stockfish path to the new engine location
        if stockfish_path is None:
            stockfish_path = "v3.0/stockfish/stockfish-windows-x86-64-avx2.exe"
        
        # Initialize V2 enhanced database
        self.enhanced_db_v2 = EnhancedPuzzleDatabaseV2(puzzle_db_path)
        
        # Initialize parent class but override some components
        super().__init__(
            thinking_brain=thinking_brain,
            stockfish_path=stockfish_path,
            puzzle_db_path=puzzle_db_path,
            save_directory=save_directory,
            memory_config=memory_config
        )
        
        self.model_version = model_version
        self.current_session_id = None
        self.session_context = {
            'puzzle_number': 0,
            'session_start_time': None,
            'performance_scores': [],
            'fatigue_estimate': 0.0,
            'average_performance': 0.0
        }
        
        # Enhanced training statistics with V2 metrics
        self.enhanced_stats_v2 = {
            'stockfish_graded_moves': 0,
            'moves_in_stockfish_top_3': 0,
            'average_stockfish_score': 0.0,
            'learning_velocity_improvements': 0,
            'theme_mastery_gains': 0,
            'regression_recoveries': 0,
            'optimal_timing_hits': 0,
            'session_efficiency': 0.0
        }
        
        logger.info("EnhancedPuzzleTrainerV2 initialized with V2 database schema")
    
    def train_enhanced_v2(self, 
                         num_puzzles: int = 1000,
                         target_themes: Optional[List[str]] = None,
                         difficulty_adaptation: bool = True,
                         intelligent_selection: bool = True,
                         spaced_repetition: bool = True,
                         checkpoint_interval: int = 100,
                         save_progress: bool = True) -> Dict:
        """
        Enhanced V2 training with intelligent puzzle selection and adaptive difficulty
        
        Args:
            num_puzzles: Number of puzzles to train on
            target_themes: Specific themes to focus on (None = auto-select weak themes)
            difficulty_adaptation: Automatically adjust difficulty based on performance
            intelligent_selection: Use analytics to select optimal puzzles
            spaced_repetition: Include puzzles ready for optimal revisiting
            checkpoint_interval: Save model every N puzzles
            save_progress: Save training progress to database
        """
        
        # Start enhanced training session
        self.current_session_id = str(uuid.uuid4())
        self.session_context['session_start_time'] = datetime.now()
        self.session_context['puzzle_number'] = 0
        
        training_config = {
            'num_puzzles': num_puzzles,
            'target_themes': target_themes,
            'difficulty_adaptation': difficulty_adaptation,
            'intelligent_selection': intelligent_selection,
            'spaced_repetition': spaced_repetition,
            'model_version': self.model_version
        }
        
        # Record session start in V2 database
        try:
            self.enhanced_db_v2.connection.execute("""
                INSERT INTO training_sessions_v2 (
                    session_id, start_time, model_version, training_config
                ) VALUES (?, ?, ?, ?)
            """, (self.current_session_id, self.session_context['session_start_time'].isoformat(),
                  self.model_version, json.dumps(training_config, cls=DateTimeEncoder)))
            self.enhanced_db_v2.connection.commit()
        except Exception as e:
            logger.warning(f"Could not record session start: {e} - continuing without database recording")
        
        logger.info(f"🧩 Starting enhanced V2 puzzle training session: {self.current_session_id}")
        
        # Get intelligent puzzle selection
        puzzles = self._get_intelligent_puzzle_selection(
            num_puzzles, target_themes, difficulty_adaptation, 
            intelligent_selection, spaced_repetition
        )
        
        if not puzzles:
            logger.error("No puzzles found matching criteria")
            return {}
        
        logger.info(f"Selected {len(puzzles)} puzzles using intelligent selection")
        
        # Reset memory for training session
        self.memory.start_new_game()
        
        # Enhanced training loop with V2 analytics
        results = []
        with tqdm(total=len(puzzles), desc="Enhanced V2 Puzzle Training") as pbar:
            for i, puzzle in enumerate(puzzles):
                self.session_context['puzzle_number'] = i + 1
                
                result = self._train_on_puzzle_v2(puzzle)
                
                if result:
                    results.append(result)
                    self._update_session_context(result)
                    self._update_enhanced_stats_v2(result, puzzle)
                    
                    # Update progress bar with V2 metrics
                    self._update_progress_bar_v2(pbar)
                
                pbar.update(1)
                
                # Checkpoint saving
                if save_progress and (i + 1) % checkpoint_interval == 0:
                    self._save_enhanced_checkpoint_v2(i + 1, results)
        
        # End training session with V2 analytics
        performance_summary = self._generate_enhanced_report_v2(results)
        
        # Update session record in database
        end_time = datetime.now()
        session_duration = (end_time - self.session_context['session_start_time']).total_seconds() / 3600.0
        session_efficiency = len(results) / session_duration if session_duration > 0 else 0
        
        self.enhanced_db_v2.connection.execute("""
            UPDATE training_sessions_v2 SET
                end_time = ?, total_puzzles = ?, puzzles_solved = ?,
                average_score = ?, performance_summary = ?, session_efficiency = ?,
                learning_momentum = ?, session_quality_score = ?
            WHERE session_id = ?
        """, (
            end_time.isoformat(), len(results), 
            sum(1 for r in results if r['found_solution']),
            sum(r['ai_score'] for r in results) / len(results) if results else 0,
            json.dumps(performance_summary, cls=DateTimeEncoder), session_efficiency,
            self.enhanced_stats_v2.get('learning_velocity_improvements', 0),
            self._calculate_session_quality_score(results),
            self.current_session_id
        ))
        self.enhanced_db_v2.connection.commit()
        
        # Final save
        if save_progress:
            self._save_enhanced_final_results_v2(results)
        
        logger.info("🎉 Enhanced V2 puzzle training completed!")
        return performance_summary
    
    def _get_intelligent_puzzle_selection(self, num_puzzles: int, target_themes: Optional[List[str]],
                                        difficulty_adaptation: bool, intelligent_selection: bool,
                                        spaced_repetition: bool) -> List[Dict]:
        """Use V2 analytics to intelligently select puzzles for training"""
        
        cursor = self.enhanced_db_v2.connection.cursor()
        puzzles = []
        
        # If no target themes specified, auto-select weak themes
        if not target_themes and intelligent_selection:
            cursor.execute("""
                SELECT theme FROM theme_mastery 
                WHERE model_version = ? AND confidence_score < 0.6
                ORDER BY confidence_score ASC, learning_velocity DESC
                LIMIT 5
            """, (self.model_version,))
            weak_themes = [row[0] for row in cursor.fetchall()]
            target_themes = weak_themes[:3] if weak_themes else None
            if target_themes:
                logger.info(f"Auto-selected weak themes for focus: {target_themes}")
        
        # Get spaced repetition puzzles (ready for optimal revisiting)
        spaced_rep_count = 0
        if spaced_repetition:
            cursor.execute("""
                SELECT * FROM puzzles_v2 
                WHERE ai_encounter_count > 0 
                AND datetime('now') > datetime(ai_last_encounter_date, '+' || ai_optimal_revisit_interval || ' hours')
                AND ai_mastery_level IN ('learning', 'competent')
                ORDER BY 
                    CASE WHEN ai_regression_detected = 1 THEN 0 ELSE 1 END,
                    (datetime('now') - datetime(ai_last_encounter_date)) DESC
                LIMIT ?
            """, (num_puzzles // 4,))  # 25% spaced repetition
            
            spaced_puzzles = [dict(row) for row in cursor.fetchall()]
            puzzles.extend(spaced_puzzles)
            spaced_rep_count = len(spaced_puzzles)
            logger.info(f"Added {spaced_rep_count} spaced repetition puzzles")
        
        # Get regression recovery puzzles (high priority)
        cursor.execute("""
            SELECT * FROM puzzles_v2 
            WHERE ai_regression_detected = 1
            ORDER BY ai_consecutive_fails DESC, ai_last_encounter_date ASC
            LIMIT ?
        """, (num_puzzles // 10,))  # 10% regression recovery
        
        regression_puzzles = [dict(row) for row in cursor.fetchall()]
        puzzles.extend(regression_puzzles)
        logger.info(f"Added {len(regression_puzzles)} regression recovery puzzles")
        
        # Fill remaining with intelligently selected new/practice puzzles
        remaining_needed = num_puzzles - len(puzzles)
        
        if remaining_needed > 0:
            # Build dynamic query based on preferences
            conditions = ["ai_encounter_count < 3"]  # Prefer less encountered puzzles
            params = []
            
            if target_themes:
                theme_conditions = []
                for theme in target_themes:
                    theme_conditions.append("themes LIKE ?")
                    params.append(f"%{theme}%")
                conditions.append(f"({' OR '.join(theme_conditions)})")
            
            if difficulty_adaptation:
                # Get current performance level and adjust difficulty
                current_level = self._estimate_current_difficulty_level()
                difficulty_range = self._get_appropriate_difficulty_range(current_level)
                conditions.append("rating BETWEEN ? AND ?")
                params.extend(difficulty_range)
            
            # Exclude already selected puzzles
            if puzzles:
                puzzle_ids = [p['id'] for p in puzzles]
                placeholders = ','.join('?' for _ in puzzle_ids)
                conditions.append(f"id NOT IN ({placeholders})")
                params.extend(puzzle_ids)
            
            query = f"""
                SELECT * FROM puzzles_v2 
                WHERE {' AND '.join(conditions)}
                ORDER BY 
                    difficulty_appropriateness DESC,
                    ai_encounter_count ASC,
                    rating ASC,
                    RANDOM()
                LIMIT ?
            """
            params.append(remaining_needed)
            
            cursor.execute(query, params)
            additional_puzzles = [dict(row) for row in cursor.fetchall()]
            puzzles.extend(additional_puzzles)
            logger.info(f"Added {len(additional_puzzles)} intelligently selected puzzles")
        
        # Shuffle while preserving high-priority puzzles at the front
        import random
        if len(puzzles) > spaced_rep_count + len(regression_puzzles):
            # Keep spaced repetition and regression puzzles at front, shuffle the rest
            high_priority = puzzles[:spaced_rep_count + len(regression_puzzles)]
            regular_puzzles = puzzles[spaced_rep_count + len(regression_puzzles):]
            random.shuffle(regular_puzzles)
            puzzles = high_priority + regular_puzzles
        
        return puzzles[:num_puzzles]
    
    def _train_on_puzzle_v2(self, puzzle: Dict) -> Optional[Dict]:
        """Train on a single puzzle with V2 enhanced analytics"""
        try:
            puzzle_id = puzzle['id']
            
            # Convert dict to puzzle object format for compatibility
            class PuzzleObj:
                def __init__(self, puzzle_dict):
                    for key, value in puzzle_dict.items():
                        setattr(self, key, value)
            
            puzzle_obj = PuzzleObj(puzzle)
            
            # Get challenge position
            challenge_fen, expected_move, is_valid, context_info = self.get_challenge_position(puzzle_obj)
            
            if not is_valid:
                return None
            
            # Get Stockfish evaluation and store it
            stockfish_moves = self.get_stockfish_evaluation(challenge_fen, 5)
            ai_move_stockfish_eval = None
            
            # Extract chess features
            import chess
            board = chess.Board(challenge_fen)
            current_state = self.state_extractor.extract_state(board)
            position_features = self.feature_converter.convert_to_features(current_state, device=str(self.thinking_brain.device))
            
            # Get legal moves
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return None
            
            # AI analyzes position
            start_time = time.time()
            ai_candidates, ai_probabilities = self.memory.process_position(
                position_features, legal_moves, top_k=5
            )
            analysis_time = time.time() - start_time
            
            if not ai_candidates:
                return None
            
            ai_move = str(ai_candidates[0])
            
            # Score AI performance against expected move
            score, rank = self.score_ai_move(ai_move, stockfish_moves)
            found_solution = ai_move == expected_move
            
            # Get Stockfish evaluation of AI's actual move
            if stockfish_moves:
                for sf_move, sf_score in stockfish_moves:
                    if sf_move == ai_move:
                        ai_move_stockfish_eval = {
                            'score': sf_score,
                            'rank': next((i for i, (m, s) in enumerate(stockfish_moves, 1) if m == ai_move), len(stockfish_moves) + 1),
                            'centipawn_eval': None,  # Could be enhanced with actual centipawn eval
                            'move_category': 'stockfish_top_moves'
                        }
                        break
                
                # If AI move not in top Stockfish moves, try to get its evaluation
                if not ai_move_stockfish_eval:
                    try:
                        ai_move_eval = self._get_move_stockfish_evaluation(challenge_fen, ai_move)
                        if ai_move_eval:
                            ai_move_stockfish_eval = ai_move_eval
                    except Exception as e:
                        logger.debug(f"Could not get direct Stockfish evaluation for move {ai_move}: {e}")
                        # Create a basic evaluation based on our scoring
                        ai_move_stockfish_eval = {
                            'score': score,  # Use our existing score
                            'rank': rank if rank > 0 else 10,
                            'centipawn_eval': None,
                            'move_category': 'estimated'
                        }
            
            # Record enhanced encounter in V2 database
            try:
                self.enhanced_db_v2.record_enhanced_ai_encounter(
                    puzzle_id=puzzle_id,
                    ai_move=ai_move,
                    expected_move=expected_move,
                    ai_score=score,
                    ai_rank=rank,
                    found_solution=found_solution,
                    solve_time=analysis_time,
                    stockfish_top_moves=stockfish_moves,
                    ai_move_stockfish_evaluation=ai_move_stockfish_eval or {},
                    session_context=self.session_context,
                    learning_context='training_v2',
                    model_version=self.model_version,
                    session_id=self.current_session_id
                )
            except Exception as e:
                logger.debug(f"Could not record enhanced encounter in V2 database: {e}")
                # Continue without enhanced recording - non-critical for training
            
            # Neural network training
            if found_solution or score > 0:
                self._train_neural_network(
                    position_features, legal_moves, expected_move, score, found_solution
                )
            
            # Update memory with outcome-based weighting
            outcome_score = 1.0 if found_solution else (score / 5.0)
            self.memory.finalize_game_memory(outcome_score)
            
            # Create enhanced result record
            result = {
                'puzzle_id': puzzle_id,
                'challenge_fen': challenge_fen,
                'expected_move': expected_move,
                'ai_move': ai_move,
                'ai_score': score,
                'ai_rank': rank,
                'found_solution': found_solution,
                'stockfish_moves': stockfish_moves,
                'ai_move_stockfish_eval': ai_move_stockfish_eval,
                'analysis_time': analysis_time,
                'themes': puzzle.get('themes', '').split() if puzzle.get('themes') else [],
                'rating': puzzle.get('rating', 0),
                'difficulty_tier': puzzle.get('difficulty_tier', 'medium'),
                'session_context': self.session_context.copy(),
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced V2 puzzle training: {e}")
            return None
    
    def _get_move_stockfish_evaluation(self, fen: str, move: str) -> Optional[Dict]:
        """Get Stockfish evaluation for a specific move"""
        try:
            if not os.path.exists(self.stockfish_path):
                # Create a basic evaluation structure
                return {
                    'score': 0,  # Unknown score
                    'rank': 10,  # Low rank for unknown moves
                    'centipawn_eval': None,
                    'mate_in': None,
                    'move_category': 'not_evaluated'
                }
            
            import chess.engine
            
            # Use Stockfish engine directly
            with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
                board = chess.Board(fen)
                
                # Get top moves with multi-PV
                multipv_info = engine.analyse(board, chess.engine.Limit(depth=15), multipv=10)
                
                for i, analysis in enumerate(multipv_info):
                    best_move = analysis.get('pv', [None])[0]
                    if best_move and str(best_move) == move:
                        score_info = analysis.get('score')
                        
                        if score_info:
                            # Convert score to our 0-5 scale
                            if score_info.is_mate():
                                mate_moves = score_info.relative.mate()
                                score = 5 if mate_moves and mate_moves > 0 else 0
                                mate_in = mate_moves
                                cp_eval = None
                            else:
                                cp_eval = score_info.relative.score()
                                if cp_eval is None:
                                    cp_eval = 0
                                # Convert centipawn to 0-5 scale
                                score = min(5, max(0, (cp_eval + 500) / 200))
                                mate_in = None
                        else:
                            score = 2.5  # Default score
                            cp_eval = None
                            mate_in = None
                        
                        return {
                            'score': score,
                            'rank': i + 1,
                            'centipawn_eval': cp_eval,
                            'mate_in': mate_in,
                            'move_category': 'stockfish_evaluated'
                        }
                
                # If move not in top 10, it's probably not great
                return {
                    'score': 1,  # Low score
                    'rank': 15,  # Beyond top moves
                    'centipawn_eval': None,
                    'mate_in': None,
                    'move_category': 'poor_move'
                }
            
        except Exception as e:
            logger.debug(f"Error in Stockfish move evaluation for {move}: {e}")
            # Return fallback evaluation
            return {
                'score': 0,
                'rank': 10,
                'centipawn_eval': None,
                'mate_in': None,
                'move_category': 'evaluation_failed'
            }
    
    def _update_session_context(self, result: Dict):
        """Update session context with latest result"""
        self.session_context['performance_scores'].append(result['ai_score'])
        
        # Keep only last 20 scores for moving average
        if len(self.session_context['performance_scores']) > 20:
            self.session_context['performance_scores'] = self.session_context['performance_scores'][-20:]
        
        # Calculate average performance
        if self.session_context['performance_scores']:
            self.session_context['average_performance'] = sum(self.session_context['performance_scores']) / len(self.session_context['performance_scores'])
        
        # Estimate fatigue (decreasing performance over time)
        if len(self.session_context['performance_scores']) >= 10:
            recent_avg = sum(self.session_context['performance_scores'][-5:]) / 5
            earlier_avg = sum(self.session_context['performance_scores'][-10:-5]) / 5
            fatigue_indicator = max(0, earlier_avg - recent_avg) / 5.0  # 0-1 scale
            self.session_context['fatigue_estimate'] = fatigue_indicator
    
    def _update_enhanced_stats_v2(self, result: Dict, puzzle: Dict):
        """Update enhanced V2 training statistics"""
        # Update parent stats
        self.training_stats['puzzles_solved'] += 1
        self.training_stats['total_score'] += result['ai_score']
        if result['found_solution']:
            self.training_stats['perfect_solutions'] += 1
        if result['ai_rank'] > 0:
            self.training_stats['top5_hits'] += 1
        
        # Update V2 enhanced stats
        if result.get('ai_move_stockfish_eval'):
            self.enhanced_stats_v2['stockfish_graded_moves'] += 1
            sf_score = result['ai_move_stockfish_eval'].get('score', 0)
            sf_rank = result['ai_move_stockfish_eval'].get('rank', 10)
            
            # Track total Stockfish score for average
            current_total = self.enhanced_stats_v2['average_stockfish_score'] * (self.enhanced_stats_v2['stockfish_graded_moves'] - 1)
            self.enhanced_stats_v2['average_stockfish_score'] = (current_total + sf_score) / self.enhanced_stats_v2['stockfish_graded_moves']
            
            if sf_rank <= 3:
                self.enhanced_stats_v2['moves_in_stockfish_top_3'] += 1
        
        # Track other V2 metrics
        if puzzle.get('ai_regression_detected') and result['found_solution']:
            self.enhanced_stats_v2['regression_recoveries'] += 1
        
        # Calculate session efficiency
        session_duration = (datetime.now() - self.session_context['session_start_time']).total_seconds() / 3600.0
        if session_duration > 0:
            self.enhanced_stats_v2['session_efficiency'] = self.training_stats['puzzles_solved'] / session_duration
    
    def _update_progress_bar_v2(self, pbar):
        """Update progress bar with V2 enhanced metrics"""
        if self.training_stats['puzzles_solved'] > 0:
            avg_score = self.training_stats['total_score'] / self.training_stats['puzzles_solved']
            solution_rate = (self.training_stats['perfect_solutions'] / self.training_stats['puzzles_solved']) * 100
            
            # V2 specific metrics
            sf_graded = self.enhanced_stats_v2['stockfish_graded_moves']
            avg_sf_score = self.enhanced_stats_v2['average_stockfish_score']
            top3_rate = (self.enhanced_stats_v2['moves_in_stockfish_top_3'] / max(1, sf_graded)) * 100
            fatigue = self.session_context['fatigue_estimate'] * 100
            
            pbar.set_postfix({
                'Score': f"{avg_score:.2f}/5",
                'Solutions': f"{solution_rate:.1f}%", 
                'SF Score': f"{avg_sf_score:.2f}/5",
                'Top3': f"{top3_rate:.1f}%",
                'Fatigue': f"{fatigue:.0f}%"
            })
    
    def _generate_enhanced_report_v2(self, results: List[Dict]) -> Dict:
        """Generate comprehensive V2 enhanced training report"""
        basic_report = self._generate_training_report(results)
        
        # Enhanced V2 analytics
        v2_analytics = self.enhanced_db_v2.get_enhanced_analytics(self.model_version)
        
        enhanced_report = basic_report.copy()
        enhanced_report.update({
            'enhanced_stats_v2': self.enhanced_stats_v2,
            'session_context': self.session_context,
            'v2_analytics': v2_analytics,
            'session_id': self.current_session_id,
            'model_version': self.model_version,
            'training_methodology': 'enhanced_v2_puzzle_training'
        })
        
        return enhanced_report
    
    def _estimate_current_difficulty_level(self) -> int:
        """Estimate current AI difficulty level based on recent performance"""
        cursor = self.enhanced_db_v2.connection.cursor()
        cursor.execute("""
            SELECT AVG(p.rating) as avg_rating, AVG(h.ai_score) as avg_score
            FROM ai_performance_history_v2 h
            JOIN puzzles_v2 p ON h.puzzle_id = p.id
            WHERE h.model_version = ? 
            AND h.encounter_date >= date('now', '-7 days')
            AND h.ai_score > 0
        """, (self.model_version,))
        
        row = cursor.fetchone()
        if row and row[0] and row[1]:
            avg_rating = row[0]
            avg_score = row[1]
            
            # Adjust based on performance
            if avg_score >= 4.0:
                return int(avg_rating + 100)  # Increase difficulty
            elif avg_score <= 2.0:
                return int(avg_rating - 100)  # Decrease difficulty
            else:
                return int(avg_rating)  # Maintain difficulty
        
        return 1400  # Default difficulty level
    
    def _get_appropriate_difficulty_range(self, current_level: int) -> Tuple[int, int]:
        """Get appropriate difficulty range for current AI level"""
        range_size = 200
        return (current_level - range_size // 2, current_level + range_size // 2)
    
    def _calculate_session_quality_score(self, results: List[Dict]) -> float:
        """Calculate overall session quality score"""
        if not results:
            return 0.0
        
        # Factors: solution rate, improvement trend, consistency
        solution_rate = sum(1 for r in results if r['found_solution']) / len(results)
        avg_score = sum(r['ai_score'] for r in results) / len(results) / 5.0
        
        # Improvement trend (are we getting better during the session?)
        if len(results) >= 10:
            first_half_avg = sum(r['ai_score'] for r in results[:len(results)//2]) / (len(results)//2)
            second_half_avg = sum(r['ai_score'] for r in results[len(results)//2:]) / (len(results) - len(results)//2)
            improvement_factor = max(0, (second_half_avg - first_half_avg) / 5.0 + 1.0)
        else:
            improvement_factor = 1.0
        
        # Fatigue penalty
        fatigue_penalty = 1.0 - self.session_context['fatigue_estimate']
        
        quality_score = (solution_rate * 0.4 + avg_score * 0.4 + improvement_factor * 0.1) * fatigue_penalty
        return min(1.0, quality_score)
    
    def _save_enhanced_checkpoint_v2(self, puzzle_count: int, results: List[Dict]):
        """Save enhanced V2 checkpoint with comprehensive analytics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = self.save_directory / f"v7p3r_enhanced_v2_{puzzle_count}puzzles_{timestamp}.pkl"
        self.thinking_brain.save_model(str(model_path))
        
        # Save enhanced V2 progress
        progress_path = self.save_directory / f"enhanced_v2_progress_{puzzle_count}_{timestamp}.json"
        progress_data = {
            'puzzle_count': puzzle_count,
            'session_id': self.current_session_id,
            'training_stats': self.training_stats,
            'enhanced_stats_v2': self.enhanced_stats_v2,
            'session_context': self.session_context,
            'v2_analytics': self.enhanced_db_v2.get_enhanced_analytics(self.model_version),
            'recent_results': results[-50:],
            'timestamp': timestamp
        }
        
        with open(progress_path, 'w') as f:
            json.dump(progress_data, f, indent=2, cls=DateTimeEncoder)
        
        logger.info(f"Enhanced V2 checkpoint saved: {model_path}")
    
    def _save_enhanced_final_results_v2(self, results: List[Dict]):
        """Save final enhanced V2 results with comprehensive analytics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save final model
        final_model_path = self.save_directory / f"v7p3r_enhanced_v2_final_{timestamp}.pkl"
        self.thinking_brain.save_model(str(final_model_path))
        
        # Save comprehensive V2 results
        final_results_path = self.save_directory / f"enhanced_v2_training_complete_{timestamp}.json"
        complete_data = {
            'session_id': self.current_session_id,
            'training_stats': self.training_stats,
            'enhanced_stats_v2': self.enhanced_stats_v2,
            'session_context': self.session_context,
            'all_results': results,
            'final_report': self._generate_enhanced_report_v2(results),
            'comprehensive_analytics': self.enhanced_db_v2.get_enhanced_analytics(self.model_version),
            'model_path': str(final_model_path),
            'model_version': self.model_version,
            'training_methodology': 'enhanced_v2_puzzle_training',
            'timestamp': timestamp
        }
        
        with open(final_results_path, 'w') as f:
            json.dump(complete_data, f, indent=2, cls=DateTimeEncoder)
        
        logger.info(f"Enhanced V2 training results saved: {final_results_path}")
        logger.info(f"Enhanced V2 final model saved: {final_model_path}")
    
    def get_comprehensive_analytics(self) -> Dict:
        """Get comprehensive V2 analytics dashboard"""
        return self.enhanced_db_v2.get_enhanced_analytics(self.model_version)
    
    def close(self):
        """Clean up resources"""
        if self.enhanced_db_v2:
            self.enhanced_db_v2.close()
