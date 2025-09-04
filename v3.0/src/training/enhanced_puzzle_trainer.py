"""
Enhanced Puzzle Trainer with Database Integration

This enhanced version integrates with the puzzle performance database to provide:
- Persistent learning with performance tracking
- Regression detection and remedial training
- Rich historical context for neural network features
- Dataset management (train/test/validation splits)
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

from ai.thinking_brain import ThinkingBrain, PositionMemory
from core.chess_state import ChessStateExtractor
from core.neural_features import NeuralFeatureConverter
from database.enhanced_puzzle_db import EnhancedPuzzleDatabase
from training.puzzle_trainer import PuzzleTrainer

logger = logging.getLogger(__name__)


class EnhancedPuzzleTrainer(PuzzleTrainer):
    """
    Enhanced puzzle trainer with database integration for persistent learning
    """
    
    def __init__(self, 
                 thinking_brain: ThinkingBrain,
                 puzzle_db_path: str = "v7p3r_puzzle_trainer.db",
                 stockfish_path: Optional[str] = None,
                 save_directory: str = "models/enhanced_puzzle_training",
                 memory_config: Optional[Dict] = None,
                 model_version: str = "v3.0"):
        
        # Initialize enhanced database
        self.enhanced_db = EnhancedPuzzleDatabase(puzzle_db_path)
        
        # Initialize enhanced feature converter with database
        self.enhanced_feature_converter = NeuralFeatureConverter(enhanced_puzzle_db=self.enhanced_db)
        
        # Initialize parent class but override some components
        super().__init__(
            thinking_brain=thinking_brain,
            stockfish_path=stockfish_path,
            puzzle_db_path=puzzle_db_path,
            save_directory=save_directory,
            memory_config=memory_config
        )
        
        # Replace feature converter with enhanced version
        self.feature_converter = self.enhanced_feature_converter
        
        self.model_version = model_version
        self.current_session_id = None
        
        # Enhanced training statistics
        self.enhanced_stats = {
            'regression_puzzles_encountered': 0,
            'new_puzzles_solved_first_try': 0,
            'improvement_on_repeat_puzzles': 0,
            'performance_trend_by_session': [],
            'best_themes': [],
            'worst_themes': []
        }
        
        logger.info("EnhancedPuzzleTrainer initialized with database integration")
    
    def setup_database(self, original_puzzle_db_path: str, import_limit: Optional[int] = None):
        """Setup the enhanced database by importing from original and creating splits"""
        logger.info("Setting up enhanced puzzle database...")
        
        # Import puzzles from original database
        if os.path.exists(original_puzzle_db_path):
            imported_count = self.enhanced_db.import_original_puzzles(original_puzzle_db_path, import_limit)
            logger.info(f"Imported {imported_count} puzzles from original database")
        else:
            logger.warning(f"Original puzzle database not found: {original_puzzle_db_path}")
        
        # Create dataset splits
        train_count, test_count, val_count = self.enhanced_db.assign_dataset_splits()
        logger.info(f"Dataset splits: {train_count} train, {test_count} test, {val_count} validation")
        
        return train_count + test_count + val_count
    
    def train_enhanced(self, 
                      num_puzzles: int = 1000,
                      dataset: str = 'train',
                      difficulty_tier: Optional[str] = None,
                      themes_filter: Optional[List[str]] = None,
                      rating_min: int = 1200,
                      rating_max: int = 1800,
                      checkpoint_interval: int = 100,
                      include_regression_training: bool = True,
                      save_progress: bool = True) -> Dict:
        """
        Enhanced training with database integration and performance tracking
        
        Args:
            num_puzzles: Number of puzzles to train on
            dataset: 'train', 'test', or 'validation'
            difficulty_tier: 'easy', 'medium', 'hard', 'expert' or None
            themes_filter: Specific themes to focus on
            rating_min/max: Puzzle rating range
            checkpoint_interval: Save model every N puzzles
            include_regression_training: Include remedial training on regressed puzzles
            save_progress: Save training progress to database
        """
        
        # Start training session
        self.current_session_id = str(uuid.uuid4())
        training_config = {
            'num_puzzles': num_puzzles,
            'dataset': dataset,
            'difficulty_tier': difficulty_tier,
            'themes_filter': themes_filter,
            'rating_range': [rating_min, rating_max],
            'model_version': self.model_version
        }
        
        self.enhanced_db.start_training_session(
            self.current_session_id, 
            self.model_version, 
            training_config
        )
        
        logger.info(f"🧩 Starting enhanced puzzle training session: {self.current_session_id}")
        logger.info(f"📊 Dataset: {dataset}, Target: {num_puzzles} puzzles")
        
        # Get puzzles for training
        puzzles = self.enhanced_db.get_puzzles_for_training(
            dataset=dataset,
            difficulty_tier=difficulty_tier,
            themes=themes_filter,
            rating_min=rating_min,
            rating_max=rating_max,
            limit=num_puzzles,
            exclude_recently_solved=True
        )
        
        if not puzzles:
            logger.error("No puzzles found matching criteria")
            return {}
        
        logger.info(f"Loaded {len(puzzles)} puzzles for training")
        
        # Add regression puzzles if enabled
        if include_regression_training:
            regression_puzzles = self.enhanced_db.get_regression_puzzles(limit=num_puzzles // 10)
            if regression_puzzles:
                puzzles.extend(regression_puzzles)
                logger.info(f"Added {len(regression_puzzles)} regression puzzles for remedial training")
        
        # Reset memory for training session
        self.memory.start_new_game()
        
        # Training loop with enhanced tracking
        results = []
        with tqdm(total=len(puzzles), desc="Enhanced Puzzle Training") as pbar:
            for i, puzzle in enumerate(puzzles):
                result = self._train_on_puzzle_enhanced(puzzle)
                
                if result:
                    results.append(result)
                    self._update_enhanced_stats(result, puzzle)
                    
                    # Update progress bar with enhanced metrics
                    self._update_progress_bar(pbar)
                
                pbar.update(1)
                
                # Checkpoint saving
                if save_progress and (i + 1) % checkpoint_interval == 0:
                    self._save_enhanced_checkpoint(i + 1, results)
        
        # End training session
        performance_summary = self._generate_enhanced_report(results)
        self.enhanced_db.end_training_session(
            self.current_session_id, 
            performance_summary,
            f"Enhanced training completed with {len(results)} puzzles"
        )
        
        # Final save
        if save_progress:
            self._save_enhanced_final_results(results)
        
        logger.info("🎉 Enhanced puzzle training completed!")
        return performance_summary
    
    def _train_on_puzzle_enhanced(self, puzzle: Dict) -> Optional[Dict]:
        """Train on a single puzzle with enhanced database tracking"""
        try:
            puzzle_id = puzzle['id']
            
            # Get challenge position
            challenge_fen, expected_move, is_valid, context_info = self.get_challenge_position(puzzle)
            
            if not is_valid:
                return None
            
            # Get historical performance for this puzzle
            puzzle_performance = self.enhanced_db.get_puzzle_performance_features(puzzle_id)
            
            # Get Stockfish evaluation and store it
            stockfish_moves = self.get_stockfish_evaluation(challenge_fen, 5)
            stockfish_grading = None
            
            if stockfish_moves:
                # Store Stockfish grading for expected move
                for rank, (move, score) in enumerate(stockfish_moves, 1):
                    self.enhanced_db.store_stockfish_grading(challenge_fen, move, score, rank)
                    if move == expected_move:
                        stockfish_grading = {'score': score, 'rank': rank, 'depth': 15}
            
            # Extract enhanced features including puzzle performance
            import chess
            board = chess.Board(challenge_fen)
            current_state = self.state_extractor.extract_state(board)
            position_features = self.enhanced_feature_converter.convert_to_features_enhanced(
                current_state, 
                device=str(self.thinking_brain.device),
                puzzle_id=puzzle_id,
                stockfish_grading=stockfish_grading
            )
            
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
            
            # Score AI performance
            score, rank = self.score_ai_move(ai_move, stockfish_moves)
            found_solution = ai_move == expected_move
            
            # Record encounter in database
            self.enhanced_db.record_ai_encounter(
                puzzle_id=puzzle_id,
                ai_move=ai_move,
                expected_move=expected_move,
                ai_score=score,
                ai_rank=rank,
                found_solution=found_solution,
                solve_time=analysis_time,
                stockfish_top_moves=stockfish_moves,
                learning_context='training',
                model_version=self.model_version,
                session_id=self.current_session_id or "unknown_session"
            )
            
            # Neural network training with enhanced features
            if found_solution or score > 0:  # Only train on meaningful results
                self._train_neural_network(
                    position_features, legal_moves, expected_move, score, found_solution
                )
            
            # Update memory with outcome-based weighting
            outcome_score = 1.0 if found_solution else (score / 5.0)
            self.memory.finalize_game_memory(outcome_score)
            
            # Create result record
            result = {
                'puzzle_id': puzzle_id,
                'challenge_fen': challenge_fen,
                'expected_move': expected_move,
                'ai_move': ai_move,
                'ai_score': score,
                'ai_rank': rank,
                'found_solution': found_solution,
                'stockfish_moves': stockfish_moves,
                'analysis_time': analysis_time,
                'themes': puzzle.get('themes', '').split() if puzzle.get('themes') else [],
                'rating': puzzle.get('rating', 0),
                'difficulty_tier': puzzle.get('difficulty_tier', 'medium'),
                'historical_performance': puzzle_performance,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced puzzle training: {e}")
            return None
    
    def _update_enhanced_stats(self, result: Dict, puzzle: Dict):
        """Update enhanced training statistics"""
        # Update parent stats
        self.training_stats['puzzles_solved'] += 1
        self.training_stats['total_score'] += result['ai_score']
        if result['found_solution']:
            self.training_stats['perfect_solutions'] += 1
        if result['ai_rank'] > 0:
            self.training_stats['top5_hits'] += 1
        
        # Update enhanced stats
        if puzzle.get('ai_regression_detected'):
            self.enhanced_stats['regression_puzzles_encountered'] += 1
        
        if puzzle.get('ai_encounter_count', 0) == 0 and result['found_solution']:
            self.enhanced_stats['new_puzzles_solved_first_try'] += 1
        
        if puzzle.get('ai_encounter_count', 0) > 0 and result['ai_score'] > puzzle.get('ai_best_score', 0):
            self.enhanced_stats['improvement_on_repeat_puzzles'] += 1
    
    def _update_progress_bar(self, pbar):
        """Update progress bar with enhanced metrics"""
        if self.training_stats['puzzles_solved'] > 0:
            avg_score = self.training_stats['total_score'] / self.training_stats['puzzles_solved']
            solution_rate = (self.training_stats['perfect_solutions'] / self.training_stats['puzzles_solved']) * 100
            first_try_rate = (self.enhanced_stats['new_puzzles_solved_first_try'] / max(1, self.training_stats['puzzles_solved'])) * 100
            
            pbar.set_postfix({
                'Avg Score': f"{avg_score:.2f}/5.0",
                'Solutions': f"{solution_rate:.1f}%",
                'First Try': f"{first_try_rate:.1f}%",
                'Regressions': self.enhanced_stats['regression_puzzles_encountered']
            })
    
    def _generate_enhanced_report(self, results: List[Dict]) -> Dict:
        """Generate enhanced training report with database analytics"""
        basic_report = self._generate_training_report(results)
        
        # Add enhanced analytics
        enhanced_report = basic_report.copy()
        enhanced_report.update({
            'enhanced_stats': self.enhanced_stats,
            'database_analytics': self.enhanced_db.get_performance_analytics(),
            'session_id': self.current_session_id,
            'model_version': self.model_version
        })
        
        return enhanced_report
    
    def _save_enhanced_checkpoint(self, puzzle_count: int, results: List[Dict]):
        """Save enhanced checkpoint with database state"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = self.save_directory / f"v7p3r_enhanced_puzzle_{puzzle_count}puzzles_{timestamp}.pkl"
        self.thinking_brain.save_model(str(model_path))
        
        # Save enhanced progress including database analytics
        progress_path = self.save_directory / f"enhanced_progress_{puzzle_count}_{timestamp}.json"
        progress_data = {
            'puzzle_count': puzzle_count,
            'session_id': self.current_session_id,
            'training_stats': self.training_stats,
            'enhanced_stats': self.enhanced_stats,
            'database_analytics': self.enhanced_db.get_performance_analytics(),
            'recent_results': results[-50:],
            'timestamp': timestamp
        }
        
        with open(progress_path, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        logger.info(f"Enhanced checkpoint saved: {model_path}")
    
    def _save_enhanced_final_results(self, results: List[Dict]):
        """Save final enhanced results with comprehensive analytics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save final model
        final_model_path = self.save_directory / f"v7p3r_enhanced_final_{timestamp}.pkl"
        self.thinking_brain.save_model(str(final_model_path))
        
        # Save comprehensive results
        final_results_path = self.save_directory / f"enhanced_training_complete_{timestamp}.json"
        complete_data = {
            'session_id': self.current_session_id,
            'training_stats': self.training_stats,
            'enhanced_stats': self.enhanced_stats,
            'all_results': results,
            'final_report': self._generate_enhanced_report(results),
            'database_analytics': self.enhanced_db.get_performance_analytics(),
            'model_path': str(final_model_path),
            'model_version': self.model_version,
            'timestamp': timestamp
        }
        
        with open(final_results_path, 'w') as f:
            json.dump(complete_data, f, indent=2)
        
        logger.info(f"Enhanced training results saved: {final_results_path}")
        logger.info(f"Enhanced final model saved: {final_model_path}")
    
    def run_validation_test(self, num_puzzles: int = 200) -> Dict:
        """Run validation test on held-out puzzle set"""
        logger.info("🧪 Running validation test on held-out puzzles...")
        
        validation_results = self.train_enhanced(
            num_puzzles=num_puzzles,
            dataset='validation',  # Use validation set
            save_progress=False,   # Don't save during validation
            include_regression_training=False  # Pure validation
        )
        
        logger.info("✅ Validation test completed")
        return validation_results
    
    def analyze_regression_patterns(self) -> Dict:
        """Analyze patterns in puzzles where AI performance has regressed"""
        regression_puzzles = self.enhanced_db.get_regression_puzzles(limit=100)
        
        if not regression_puzzles:
            return {'message': 'No regression puzzles found'}
        
        # Analyze regression patterns
        themes_analysis = {}
        difficulty_analysis = {}
        
        for puzzle in regression_puzzles:
            # Theme analysis
            themes = puzzle.get('themes', '').split() if puzzle.get('themes') else []
            for theme in themes:
                if theme not in themes_analysis:
                    themes_analysis[theme] = 0
                themes_analysis[theme] += 1
            
            # Difficulty analysis
            difficulty = puzzle.get('difficulty_tier', 'unknown')
            if difficulty not in difficulty_analysis:
                difficulty_analysis[difficulty] = 0
            difficulty_analysis[difficulty] += 1
        
        analysis = {
            'total_regression_puzzles': len(regression_puzzles),
            'most_problematic_themes': sorted(themes_analysis.items(), key=lambda x: x[1], reverse=True)[:5],
            'difficulty_distribution': difficulty_analysis,
            'recommendations': self._generate_regression_recommendations(themes_analysis, difficulty_analysis)
        }
        
        logger.info(f"Regression analysis: {len(regression_puzzles)} problematic puzzles identified")
        return analysis
    
    def _generate_regression_recommendations(self, themes_analysis: Dict, difficulty_analysis: Dict) -> List[str]:
        """Generate training recommendations based on regression analysis"""
        recommendations = []
        
        # Theme-based recommendations
        if themes_analysis:
            worst_theme = max(themes_analysis.items(), key=lambda x: x[1])[0]
            recommendations.append(f"Focus additional training on '{worst_theme}' theme puzzles")
        
        # Difficulty-based recommendations
        if difficulty_analysis:
            if difficulty_analysis.get('expert', 0) > len(difficulty_analysis) * 0.3:
                recommendations.append("Consider reducing expert-level puzzle ratio in training")
            if difficulty_analysis.get('easy', 0) > len(difficulty_analysis) * 0.4:
                recommendations.append("Review basic tactical understanding - many easy puzzle regressions")
        
        return recommendations
    
    def close(self):
        """Clean up resources"""
        if self.enhanced_db:
            self.enhanced_db.close()
