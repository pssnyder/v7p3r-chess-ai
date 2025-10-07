"""
V7P3R Hybrid Training Integration V3.0
=====================================

Connects the enhanced puzzle training system with the V3 two-brain architecture,
using your custom ChessState feature extraction for tactical pattern learning.

Key Integration Points:
1. Enhanced ChessState extraction for puzzle positions
2. Thinking Brain integration for move candidate generation  
3. Gameplay Brain integration for tactical validation
4. Performance correlation with puzzle ELO ratings
5. Model expansion capabilities for future features
"""

import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

# Import V3 components
sys.path.append(str(Path(__file__).parent / "v3.0" / "src"))
from ai.thinking_brain import ThinkingBrain, PositionMemory
from ai.gameplay_brain import GameplayBrain, MoveCandidate
from core.chess_state import ChessStateExtractor
from core.neural_features import NeuralFeatureConverter

# Import existing enhanced trainer
from v3.0.src.training.enhanced_puzzle_trainer_v2 import EnhancedPuzzleTrainerV2

logger = logging.getLogger(__name__)

class V3HybridPuzzleTrainer:
    """
    Hybrid V3 trainer that combines:
    - Enhanced puzzle training methodology  
    - V3 two-brain architecture (Thinking + Gameplay)
    - Custom ChessState feature extraction
    - ELO-correlated performance validation
    """
    
    def __init__(self,
                 thinking_brain: ThinkingBrain,
                 gameplay_brain: GameplayBrain,
                 puzzle_db_path: str = "v3.0/data/v7p3rai_puzzle_training_v2.db",
                 stockfish_path: Optional[str] = None,
                 save_directory: str = "v3.0/models/hybrid_training",
                 model_version: str = "v3.0_hybrid"):
        
        self.thinking_brain = thinking_brain
        self.gameplay_brain = gameplay_brain
        self.model_version = model_version
        
        # Initialize enhanced components
        self.chess_state_extractor = ChessStateExtractor()
        self.feature_converter = NeuralFeatureConverter()
        
        # Enhanced database for puzzle access
        from database.enhanced_puzzle_db_v2 import EnhancedPuzzleDatabaseV2
        self.puzzle_db = EnhancedPuzzleDatabaseV2(puzzle_db_path)
        
        # Training statistics
        self.training_stats = {
            'puzzles_solved': 0,
            'average_thinking_brain_accuracy': 0.0,
            'average_gameplay_brain_accuracy': 0.0,
            'combined_accuracy': 0.0,
            'elo_estimate': 0.0,
            'feature_importance': {},
            'tactical_pattern_mastery': {}
        }
        
        # Session context for enhanced monitoring
        self.session_context = {
            'session_id': None,
            'start_time': None,
            'current_puzzle': 0,
            'performance_history': [],
            'brain_coordination_score': 0.0
        }
        
        logger.info("🧠 V3 Hybrid Puzzle Trainer initialized")
        logger.info(f"   Thinking Brain: {self.thinking_brain.num_layers} layer GRU")
        logger.info(f"   Gameplay Brain: GA with {self.gameplay_brain.population_size} candidates")
        logger.info(f"   ChessState features: Enhanced metadata extraction")
    
    def train_hybrid_session(self,
                           num_puzzles: Optional[int] = None,
                           training_hours: Optional[float] = None,
                           target_themes: Optional[List[str]] = None,
                           excluded_themes: Optional[List[str]] = None,
                           max_rating: Optional[int] = None,
                           min_rating: Optional[int] = None,
                           difficulty_progression: bool = True,
                           brain_coordination_training: bool = True) -> Dict:
        """
        Run a hybrid training session using V3 architecture with puzzle-based learning
        
        Args:
            num_puzzles: Fixed number of puzzles (if not time-based)
            training_hours: Training duration in hours (if time-based)
            target_themes: Specific tactical themes to focus on
            excluded_themes: Themes to exclude (e.g., ['long', 'mate'])
            max_rating/min_rating: Puzzle difficulty range
            difficulty_progression: Automatically increase difficulty as AI improves
            brain_coordination_training: Train coordination between the two brains
        """
        
        # Initialize session
        import uuid
        self.session_context['session_id'] = str(uuid.uuid4())
        self.session_context['start_time'] = datetime.now()
        self.session_context['current_puzzle'] = 0
        
        logger.info("🚀 Starting V3 Hybrid Training Session")
        logger.info(f"   Session ID: {self.session_context['session_id'][:8]}...")
        logger.info(f"   Architecture: Two-Brain (Thinking + Gameplay)")
        logger.info(f"   Training Method: Puzzle-based tactical learning")
        
        # Determine training mode
        if training_hours:
            results = self._train_timed_hybrid(
                training_hours, target_themes, excluded_themes, 
                max_rating, min_rating, difficulty_progression, brain_coordination_training
            )
        else:
            num_puzzles = num_puzzles or 1000
            results = self._train_fixed_hybrid(
                num_puzzles, target_themes, excluded_themes,
                max_rating, min_rating, difficulty_progression, brain_coordination_training
            )
        
        # Generate comprehensive report
        session_report = self._generate_hybrid_report(results)
        
        logger.info("🏁 V3 Hybrid Training Session Complete!")
        logger.info(f"   Puzzles solved: {len(results)}")
        logger.info(f"   Combined accuracy: {session_report['combined_accuracy']:.1f}%")
        logger.info(f"   Estimated ELO: {session_report['elo_estimate']:.0f}")
        logger.info(f"   Brain coordination: {session_report['brain_coordination_score']:.2f}")
        
        return session_report
    
    def _train_on_hybrid_puzzle(self, puzzle: Dict) -> Optional[Dict]:
        """
        Train both brains on a single puzzle using V3 architecture
        
        Process:
        1. Extract enhanced ChessState features
        2. Thinking Brain generates move candidates
        3. Gameplay Brain validates candidates tactically  
        4. Compare combined result against puzzle solution
        5. Train both brains based on performance
        """
        
        try:
            import chess
            
            puzzle_id = puzzle['id']
            puzzle_fen = puzzle['fen']
            expected_moves = puzzle['moves'].split()
            puzzle_rating = puzzle.get('rating', 1200)
            puzzle_themes = puzzle.get('themes', '').split()
            
            # Step 1: Extract enhanced ChessState features
            board = chess.Board(puzzle_fen)
            chess_state = self.chess_state_extractor.extract_state(board)
            feature_vector = self.feature_converter.convert_to_features(chess_state, 
                                                                       device=str(self.thinking_brain.device))
            
            # Step 2: Thinking Brain generates move candidates
            thinking_start = time.time()
            
            # Get legal moves for position
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return None
            
            # Process through Thinking Brain
            with torch.no_grad():
                move_candidates, candidate_scores = self.thinking_brain.generate_move_candidates(
                    feature_vector, legal_moves, top_k=5
                )
            
            thinking_time = time.time() - thinking_start
            
            # Step 3: Gameplay Brain tactical validation
            gameplay_start = time.time()
            
            # Convert to MoveCandidate objects for GA
            ga_population = [
                MoveCandidate(move=move, fitness=score) 
                for move, score in zip(move_candidates, candidate_scores)
            ]
            
            # Run genetic algorithm tactical simulation
            best_move, ga_fitness = self.gameplay_brain.evaluate_tactical_position(
                board, ga_population, simulation_depth=3
            )
            
            gameplay_time = time.time() - gameplay_start
            
            # Step 4: Evaluate performance against puzzle solution
            expected_move_str = expected_moves[0] if expected_moves else None
            expected_move = None
            
            if expected_move_str:
                try:
                    expected_move = chess.Move.from_uci(expected_move_str)
                except:
                    return None
            
            # Thinking Brain accuracy
            thinking_correct = expected_move in move_candidates if expected_move else False
            thinking_rank = move_candidates.index(expected_move) + 1 if thinking_correct else len(move_candidates) + 1
            
            # Gameplay Brain accuracy  
            gameplay_correct = best_move == expected_move if expected_move else False
            
            # Combined system accuracy
            combined_correct = gameplay_correct
            
            # Calculate performance scores
            thinking_score = max(0, (6 - thinking_rank) / 5.0)  # 0-1 score based on rank
            gameplay_score = 1.0 if gameplay_correct else 0.0
            combined_score = 1.0 if combined_correct else 0.0
            
            # Step 5: Train both brains based on performance
            if expected_move:
                # Train Thinking Brain
                self._train_thinking_brain(feature_vector, legal_moves, expected_move, thinking_score)
                
                # Train Gameplay Brain (through reward signal)
                self._train_gameplay_brain(board, ga_population, expected_move, gameplay_score)
            
            # Create result record
            result = {
                'puzzle_id': puzzle_id,
                'puzzle_rating': puzzle_rating,
                'puzzle_themes': puzzle_themes,
                'thinking_brain': {
                    'candidates': [str(m) for m in move_candidates],
                    'scores': candidate_scores.tolist() if hasattr(candidate_scores, 'tolist') else candidate_scores,
                    'correct': thinking_correct,
                    'rank': thinking_rank,
                    'score': thinking_score,
                    'time': thinking_time
                },
                'gameplay_brain': {
                    'selected_move': str(best_move),
                    'fitness': ga_fitness,
                    'correct': gameplay_correct,
                    'score': gameplay_score,
                    'time': gameplay_time
                },
                'combined': {
                    'final_move': str(best_move),
                    'correct': combined_correct,
                    'score': combined_score,
                    'total_time': thinking_time + gameplay_time
                },
                'expected_move': str(expected_move) if expected_move else None,
                'chess_state_features': {
                    'feature_count': len(feature_vector),
                    'board_features': chess_state.get('Board_Features', {}),
                    'piece_count': len(chess_state.get('Pieces', []))
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Update session context
            self.session_context['current_puzzle'] += 1
            self.session_context['performance_history'].append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error training on hybrid puzzle {puzzle_id}: {e}")
            return None
    
    def _train_thinking_brain(self, feature_vector, legal_moves, expected_move, performance_score):
        """Train the Thinking Brain based on puzzle performance"""
        # This would implement the GRU training logic
        # For now, placeholder - would need actual loss calculation and backprop
        pass
    
    def _train_gameplay_brain(self, board, population, expected_move, performance_score):
        """Train the Gameplay Brain based on tactical accuracy"""
        # This would implement GA parameter adaptation based on performance
        # For now, placeholder - would adjust mutation rates, selection pressure, etc.
        pass
    
    def _generate_hybrid_report(self, results: List[Dict]) -> Dict:
        """Generate comprehensive report for hybrid training session"""
        
        if not results:
            return {}
        
        # Thinking Brain statistics
        thinking_correct = sum(1 for r in results if r['thinking_brain']['correct'])
        thinking_accuracy = thinking_correct / len(results) * 100
        avg_thinking_rank = sum(r['thinking_brain']['rank'] for r in results) / len(results)
        avg_thinking_time = sum(r['thinking_brain']['time'] for r in results) / len(results)
        
        # Gameplay Brain statistics  
        gameplay_correct = sum(1 for r in results if r['gameplay_brain']['correct'])
        gameplay_accuracy = gameplay_correct / len(results) * 100
        avg_gameplay_time = sum(r['gameplay_brain']['time'] for r in results) / len(results)
        
        # Combined system statistics
        combined_correct = sum(1 for r in results if r['combined']['correct'])
        combined_accuracy = combined_correct / len(results) * 100
        avg_total_time = sum(r['combined']['total_time'] for r in results) / len(results)
        
        # ELO estimation based on puzzle ratings and accuracy
        puzzle_ratings = [r['puzzle_rating'] for r in results if r['puzzle_rating']]
        avg_puzzle_rating = sum(puzzle_ratings) / len(puzzle_ratings) if puzzle_ratings else 1200
        
        # Simple ELO estimation: base on puzzle difficulty and accuracy
        elo_estimate = avg_puzzle_rating * (combined_accuracy / 100) + (combined_accuracy - 50) * 10
        elo_estimate = max(600, min(2400, elo_estimate))  # Clamp to reasonable range
        
        # Brain coordination score (how well they work together)
        coordination_score = combined_accuracy / max(thinking_accuracy, gameplay_accuracy) if max(thinking_accuracy, gameplay_accuracy) > 0 else 0
        
        return {
            'session_id': self.session_context['session_id'],
            'total_puzzles': len(results),
            'thinking_brain': {
                'accuracy': thinking_accuracy,
                'average_rank': avg_thinking_rank,
                'average_time': avg_thinking_time,
                'correct_count': thinking_correct
            },
            'gameplay_brain': {
                'accuracy': gameplay_accuracy,
                'average_time': avg_gameplay_time,
                'correct_count': gameplay_correct
            },
            'combined_system': {
                'accuracy': combined_accuracy,
                'average_time': avg_total_time,
                'correct_count': combined_correct
            },
            'performance_metrics': {
                'elo_estimate': elo_estimate,
                'brain_coordination_score': coordination_score,
                'average_puzzle_rating': avg_puzzle_rating,
                'tactical_efficiency': combined_accuracy / (avg_total_time * 1000)  # accuracy per millisecond
            },
            'training_insights': {
                'thinking_brain_strength': thinking_accuracy,
                'gameplay_brain_strength': gameplay_accuracy,
                'integration_quality': coordination_score,
                'time_efficiency': avg_total_time < 2.0  # Under 2 seconds is good
            }
        }
    
    def get_current_status(self) -> Dict:
        """Get current training status for monitoring"""
        return {
            'session_active': self.session_context['session_id'] is not None,
            'current_puzzle': self.session_context['current_puzzle'],
            'session_duration': (datetime.now() - self.session_context['start_time']).total_seconds() / 3600.0 if self.session_context['start_time'] else 0,
            'recent_performance': self.session_context['performance_history'][-10:] if self.session_context['performance_history'] else [],
            'training_stats': self.training_stats
        }

if __name__ == "__main__":
    print("V7P3R Hybrid Training Integration V3.0")
    print("Combines puzzle training with two-brain architecture")