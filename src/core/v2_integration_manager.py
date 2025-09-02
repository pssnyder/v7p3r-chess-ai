# v2_integration_manager.py
"""
V7P3R Chess AI 2.0 - Integration Manager
Properly integrates improved bounty system with genetic algorithm + RNN architecture.
This is the CORRECT V2.0 training - real-time learning with bounty rewards during gameplay.
"""

import os
import json
import time
import chess
from typing import Dict, List, Optional, Any
from datetime import datetime

# Import V2.0 components
from evaluation.v7p3r_bounty_system import ExtendedBountyEvaluator
from core.performance_optimizer import PerformanceOptimizer
from core.move_preparation import MovePreparation


class V2IntegrationManager:
    """
    Manages the integration of V2.0 components for real-time training:
    - Enhanced bounty system with offensive/defensive/outcome evaluation
    - Move preparation and ordering for efficiency
    - Performance optimization for move selection
    - Integration with genetic algorithm + RNN training
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        
        print("Initializing V7P3R Chess AI 2.0 Integration Manager...")
        
        # Initialize core V2.0 systems
        self.bounty_evaluator = ExtendedBountyEvaluator()
        self.performance_optimizer = PerformanceOptimizer()
        self.move_preparation = MovePreparation()
        
        # Training statistics
        self.stats = {
            'games_played': 0,
            'moves_evaluated': 0,
            'total_bounty_rewards': 0.0,
            'average_game_length': 0.0,
            'performance_metrics': [],
            'training_start_time': None
        }
        
        # Configure integration settings
        self._configure_integration()
        
        print("✓ V2.0 Integration Manager initialized successfully")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for V2.0 integration"""
        return {
            'max_moves_per_position': 15,
            'move_evaluation_time_limit': 0.02,  # 20ms per position
            'bounty_weight_offensive': 0.8,
            'bounty_weight_defensive': 1.0,      # Higher weight for defensive balance
            'bounty_weight_outcome': 0.9,
            'neural_network_weight': 0.4,
            'move_prep_weight': 0.1,
            'enable_move_filtering': True,
            'enable_performance_optimization': True,
            'enable_caching': True,
            'detailed_logging': False
        }
    
    def _configure_integration(self):
        """Configure integration between systems"""
        # Configure performance optimizer for real-time training
        self.performance_optimizer.config.update({
            'max_candidates_opening': self.config['max_moves_per_position'],
            'max_candidates_middlegame': self.config['max_moves_per_position'],
            'max_candidates_endgame': self.config['max_moves_per_position'] + 5,
            'enable_caching': self.config['enable_caching'],
            'bounty_weight': 0.6,
            'move_order_weight': 0.4
        })
        
        print("✓ Integration configuration applied")
    
    def evaluate_move_for_training(self, board: chess.Board, move: chess.Move, 
                                 neural_network_value: float = 0.0) -> Dict[str, float]:
        """
        Comprehensive move evaluation for V2.0 training.
        Combines all evaluation systems for real-time learning.
        
        Args:
            board: Current chess position
            move: Move to evaluate
            neural_network_value: Value from RNN (if available)
            
        Returns:
            Dictionary with all evaluation components
        """
        start_time = time.time()
        
        try:
            # 1. Enhanced bounty evaluation
            bounty_score = self.bounty_evaluator.evaluate_move(board, move)
            
            # 2. Move preparation features
            move_features = self.performance_optimizer.get_neural_network_features(board, move)
            
            # 3. Calculate combined training reward
            training_reward = self._calculate_training_reward(
                bounty_score, move_features, neural_network_value
            )
            
            # 4. Generate detailed evaluation
            evaluation = {
                'total_reward': training_reward,
                'bounty_total': bounty_score.total(),
                'bounty_offensive': bounty_score.offensive_total(),
                'bounty_defensive': bounty_score.defensive_total(),
                'bounty_outcome': bounty_score.outcome_total(),
                'neural_network_value': neural_network_value,
                'move_quality_score': move_features[:5].sum() if len(move_features) >= 5 else 0.0,
                'evaluation_time': time.time() - start_time,
                'components': {
                    'center_control': bounty_score.center_control,
                    'piece_value': bounty_score.piece_value,
                    'attack_patterns': bounty_score.attack_patterns,
                    'defensive_measures': bounty_score.defensive_measures,
                    'piece_protection': bounty_score.piece_protection,
                    'king_safety': bounty_score.king_safety,
                    'tactical_patterns': bounty_score.tactical_patterns,
                    'counter_threats': bounty_score.counter_threats,
                    'mate_threats': bounty_score.mate_threats,
                    'piece_coordination': bounty_score.piece_coordination,
                    'defensive_coordination': bounty_score.defensive_coordination,
                    'castling': bounty_score.castling,
                    'material_balance': bounty_score.material_balance,
                    'positional_advantage': bounty_score.positional_advantage,
                    'game_phase_bonus': bounty_score.game_phase_bonus,
                    'initiative': bounty_score.initiative,
                    'positional': bounty_score.positional
                }
            }
            
            # Update statistics
            self._update_move_stats(evaluation)
            
            return evaluation
            
        except Exception as e:
            print(f"Error in move evaluation: {e}")
            return {'total_reward': 0.0, 'error': str(e)}
    
    def select_move_for_training(self, board: chess.Board, 
                               neural_network_predictions: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Select best move for V2.0 training using all integrated systems.
        This is what the genetic algorithm should call during gameplay.
        
        Args:
            board: Current chess position
            neural_network_predictions: Optional NN predictions for moves
            
        Returns:
            Dictionary with selected move and evaluation details
        """
        if board.is_game_over():
            return {'move': None, 'evaluation': None}
        
        start_time = time.time()
        
        try:
            # 1. Get optimized move candidates using performance optimizer
            if self.config['enable_performance_optimization']:
                optimized_moves = self.performance_optimizer.optimize_move_selection(
                    board, time_limit=self.config['move_evaluation_time_limit']
                )
                candidate_moves = [move for move, score in optimized_moves[:self.config['max_moves_per_position']]]
            else:
                candidate_moves = list(board.legal_moves)
            
            if not candidate_moves:
                return {'move': None, 'evaluation': None}
            
            # 2. Evaluate each candidate move
            move_evaluations = []
            
            for move in candidate_moves:
                # Get neural network prediction if available
                nn_value = 0.0
                if neural_network_predictions and move.uci() in neural_network_predictions:
                    nn_value = neural_network_predictions[move.uci()]
                
                # Get comprehensive evaluation
                evaluation = self.evaluate_move_for_training(board, move, nn_value)
                
                move_evaluations.append({
                    'move': move,
                    'evaluation': evaluation
                })
            
            # 3. Select best move based on total reward
            best_move_data = max(move_evaluations, key=lambda x: x['evaluation']['total_reward'])
            
            # 4. Prepare result
            result = {
                'move': best_move_data['move'],
                'evaluation': best_move_data['evaluation'],
                'all_evaluations': move_evaluations,
                'selection_time': time.time() - start_time,
                'candidates_considered': len(candidate_moves)
            }
            
            # 5. Update statistics
            self._update_selection_stats(result)
            
            if self.config['detailed_logging']:
                self._log_move_selection(board, result)
            
            return result
            
        except Exception as e:
            print(f"Error in move selection: {e}")
            # Fallback to random legal move
            legal_moves = list(board.legal_moves)
            if legal_moves:
                return {
                    'move': legal_moves[0],
                    'evaluation': {'total_reward': 0.0, 'error': str(e)},
                    'fallback': True
                }
            return {'move': None, 'evaluation': None, 'error': str(e)}
    
    def _calculate_training_reward(self, bounty_score, move_features, neural_network_value) -> float:
        """Calculate training reward for genetic algorithm"""
        # V2.0 enhanced reward calculation with balanced offensive/defensive components
        reward = (
            neural_network_value * self.config['neural_network_weight'] +
            bounty_score.offensive_total() * self.config['bounty_weight_offensive'] +
            bounty_score.defensive_total() * self.config['bounty_weight_defensive'] +  # Higher weight for balance
            bounty_score.outcome_total() * self.config['bounty_weight_outcome'] +
            (move_features[:5].sum() if len(move_features) >= 5 else 0.0) * self.config['move_prep_weight']
        )
        
        return reward
    
    def evaluate_game_outcome(self, board: chess.Board, moves_played: List[chess.Move], 
                            game_result: str) -> Dict[str, float]:
        """
        Evaluate complete game outcome for genetic algorithm fitness.
        This provides the final fitness score for the genetic algorithm.
        
        Args:
            board: Final board position
            moves_played: List of moves played during the game
            game_result: Game result ('1-0', '0-1', '1/2-1/2')
            
        Returns:
            Game evaluation for genetic algorithm
        """
        try:
            # Base outcome rewards
            outcome_rewards = {
                '1-0': 100.0,    # Win as white
                '0-1': -100.0,   # Loss as white (or win as black)
                '1/2-1/2': 10.0  # Draw
            }
            
            base_reward = outcome_rewards.get(game_result, 0.0)
            
            # Game length bonus (encourage reasonable game lengths)
            game_length = len(moves_played)
            length_bonus = 0.0
            if 20 <= game_length <= 80:
                length_bonus = min(game_length - 20, 30)  # Bonus for games 20-50 moves
            elif game_length > 80:
                length_bonus = -10  # Penalty for very long games
            
            # Calculate average move quality during the game
            total_move_quality = 0.0
            moves_evaluated = 0
            
            temp_board = chess.Board()
            for move in moves_played[:50]:  # Evaluate first 50 moves max
                if not temp_board.is_legal(move):
                    break
                    
                move_eval = self.evaluate_move_for_training(temp_board, move)
                total_move_quality += move_eval.get('total_reward', 0.0)
                moves_evaluated += 1
                temp_board.push(move)
            
            average_move_quality = total_move_quality / max(moves_evaluated, 1)
            
            # Final game evaluation
            final_evaluation = {
                'base_outcome_reward': base_reward,
                'game_length_bonus': length_bonus,
                'average_move_quality': average_move_quality,
                'total_game_reward': base_reward + length_bonus + average_move_quality * 0.1,
                'game_length': game_length,
                'moves_evaluated': moves_evaluated
            }
            
            # Update game statistics
            self._update_game_stats(final_evaluation)
            
            return final_evaluation
            
        except Exception as e:
            print(f"Error evaluating game outcome: {e}")
            return {'total_game_reward': 0.0, 'error': str(e)}
    
    def _update_move_stats(self, evaluation: Dict):
        """Update move evaluation statistics"""
        self.stats['moves_evaluated'] += 1
        self.stats['total_bounty_rewards'] += evaluation.get('bounty_total', 0.0)
    
    def _update_selection_stats(self, result: Dict):
        """Update move selection statistics"""
        if 'evaluation' in result and result['evaluation']:
            self.stats['performance_metrics'].append({
                'timestamp': datetime.now().isoformat(),
                'selection_time': result.get('selection_time', 0.0),
                'candidates_considered': result.get('candidates_considered', 0),
                'reward': result['evaluation'].get('total_reward', 0.0)
            })
    
    def _update_game_stats(self, evaluation: Dict):
        """Update game statistics"""
        self.stats['games_played'] += 1
        
        # Update average game length
        if self.stats['games_played'] == 1:
            self.stats['average_game_length'] = evaluation.get('game_length', 0.0)
        else:
            self.stats['average_game_length'] = (
                (self.stats['average_game_length'] * (self.stats['games_played'] - 1) + 
                 evaluation.get('game_length', 0.0)) / self.stats['games_played']
            )
    
    def _log_move_selection(self, board: chess.Board, result: Dict):
        """Log detailed move selection information"""
        move = result['move']
        evaluation = result['evaluation']
        
        print(f"Move selected: {move.uci()} (reward: {evaluation['total_reward']:.3f})")
        print(f"  Bounty - Offensive: {evaluation['bounty_offensive']:.2f}, "
              f"Defensive: {evaluation['bounty_defensive']:.2f}, "
              f"Outcome: {evaluation['bounty_outcome']:.2f}")
        print(f"  NN Value: {evaluation['neural_network_value']:.3f}")
    
    def get_training_statistics(self) -> Dict:
        """Get comprehensive training statistics"""
        current_time = time.time()
        if self.stats['training_start_time']:
            training_duration = current_time - self.stats['training_start_time']
        else:
            training_duration = 0.0
        
        stats = self.stats.copy()
        stats.update({
            'training_duration': training_duration,
            'moves_per_second': self.stats['moves_evaluated'] / max(training_duration, 1),
            'games_per_hour': self.stats['games_played'] / max(training_duration / 3600, 1/3600),
            'average_bounty_reward': self.stats['total_bounty_rewards'] / max(self.stats['moves_evaluated'], 1)
        })
        
        return stats
    
    def start_training_session(self):
        """Mark the start of a training session"""
        self.stats['training_start_time'] = time.time()
        print("V2.0 training session started")
    
    def save_integration_report(self, filepath: str):
        """Save integration and performance report"""
        report = {
            'v7p3r_version': '2.0',
            'integration_manager': {
                'config': self.config,
                'statistics': self.get_training_statistics()
            },
            'bounty_system': {
                'type': 'ExtendedBountyEvaluator',
                'bounty_rates': self.bounty_evaluator.bounty_rates,
                'piece_values': self.bounty_evaluator.piece_values
            },
            'performance_optimizer': {
                'config': self.performance_optimizer.config,
                'cache_sizes': {
                    'position_cache': len(self.performance_optimizer.position_cache),
                    'evaluation_cache': len(self.performance_optimizer.evaluation_cache)
                }
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"V2.0 integration report saved to {filepath}")


def test_v2_integration():
    """Test the V2.0 integration system"""
    print("Testing V7P3R Chess AI 2.0 Integration...")
    print("=" * 50)
    
    # Initialize integration manager
    manager = V2IntegrationManager()
    manager.start_training_session()
    
    # Test position
    board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2")
    
    print(f"Test position: {board.fen()}")
    
    # Test move selection
    print("\n1. Testing move selection for training...")
    result = manager.select_move_for_training(board)
    
    if result['move']:
        print(f"Selected move: {result['move'].uci()}")
        print(f"Total reward: {result['evaluation']['total_reward']:.3f}")
        print(f"Bounty breakdown:")
        print(f"  Offensive: {result['evaluation']['bounty_offensive']:.2f}")
        print(f"  Defensive: {result['evaluation']['bounty_defensive']:.2f}")
        print(f"  Outcome: {result['evaluation']['bounty_outcome']:.2f}")
        print(f"Candidates considered: {result['candidates_considered']}")
        print(f"Selection time: {result['selection_time']:.4f}s")
    
    # Test individual move evaluation
    print(f"\n2. Testing individual move evaluation...")
    test_move = chess.Move.from_uci("e4d5")
    move_eval = manager.evaluate_move_for_training(board, test_move, neural_network_value=0.5)
    
    print(f"Move {test_move.uci()} evaluation:")
    print(f"  Total reward: {move_eval['total_reward']:.3f}")
    print(f"  Bounty components: {len(move_eval['components'])} factors")
    
    # Test game outcome evaluation
    print(f"\n3. Testing game outcome evaluation...")
    test_moves = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d7d5"), chess.Move.from_uci("e4d5")]
    game_eval = manager.evaluate_game_outcome(board, test_moves, "1-0")
    
    print(f"Game outcome evaluation:")
    print(f"  Total game reward: {game_eval['total_game_reward']:.2f}")
    print(f"  Average move quality: {game_eval['average_move_quality']:.3f}")
    
    # Get final statistics
    stats = manager.get_training_statistics()
    print(f"\n4. Training statistics:")
    print(f"  Moves evaluated: {stats['moves_evaluated']}")
    print(f"  Games played: {stats['games_played']}")
    print(f"  Moves per second: {stats['moves_per_second']:.1f}")
    
    # Save report
    os.makedirs("reports", exist_ok=True)
    report_file = f"reports/v2_integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    manager.save_integration_report(report_file)
    
    print(f"\nV2.0 integration test completed successfully!")
    print(f"Report saved: {report_file}")


if __name__ == "__main__":
    test_v2_integration()
