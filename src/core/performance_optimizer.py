# performance_optimizer.py
"""
V7P3R Chess AI 2.0 - Performance Optimization System
Integrates bounty system with move preparation for optimal neural network efficiency.
"""

import chess
import numpy as np
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import json

from evaluation.v7p3r_bounty_system import ExtendedBountyEvaluator, BountyScore
from core.move_preparation import MovePreparation, MovePreparationIntegration, MoveScore


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization tracking"""
    moves_evaluated: int = 0
    evaluation_time: float = 0.0
    moves_per_second: float = 0.0
    accuracy_score: float = 0.0
    efficiency_ratio: float = 0.0
    memory_usage: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'moves_evaluated': self.moves_evaluated,
            'evaluation_time': self.evaluation_time,
            'moves_per_second': self.moves_per_second,
            'accuracy_score': self.accuracy_score,
            'efficiency_ratio': self.efficiency_ratio,
            'memory_usage': self.memory_usage
        }


class PerformanceOptimizer:
    """
    Optimizes chess AI performance by integrating bounty evaluation 
    with intelligent move preparation and ordering
    """
    
    def __init__(self):
        self.bounty_evaluator = ExtendedBountyEvaluator()
        self.move_preparation = MovePreparation()
        self.move_integration = MovePreparationIntegration(self.move_preparation)
        
        # Performance configuration
        self.config = {
            'max_candidates_opening': 25,
            'max_candidates_middlegame': 20,
            'max_candidates_endgame': 30,
            'complexity_threshold_high': 80.0,
            'complexity_threshold_low': 20.0,
            'bounty_weight': 0.6,
            'move_order_weight': 0.4,
            'enable_pruning': True,
            'enable_caching': True
        }
        
        # Caching for performance
        self.position_cache: Dict[str, Tuple[List[chess.Move], float]] = {}
        self.evaluation_cache: Dict[str, BountyScore] = {}
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.performance_history: List[PerformanceMetrics] = []
    
    def optimize_move_selection(self, board: chess.Board, time_limit: Optional[float] = None) -> List[Tuple[chess.Move, float]]:
        """
        Optimize move selection by combining bounty evaluation with move preparation
        
        Args:
            board: Current chess position
            time_limit: Optional time limit for evaluation (seconds)
            
        Returns:
            List of (move, combined_score) tuples ordered by quality
        """
        start_time = time.time()
        
        # Check cache first
        position_key = self._get_position_key(board)
        if self.config['enable_caching'] and position_key in self.position_cache:
            cached_moves, cached_time = self.position_cache[position_key]
            if time.time() - cached_time < 60:  # Cache valid for 1 minute
                return [(move, 0.0) for move in cached_moves]
        
        # Determine game phase and adjust parameters
        game_phase = self._determine_game_phase(board)
        max_candidates = self._get_max_candidates(game_phase)
        
        # Analyze position complexity
        complexity = self.move_preparation.analyze_position_complexity(board)
        
        # Adjust parameters based on complexity and time constraints
        if complexity['complexity_score'] > self.config['complexity_threshold_high']:
            max_candidates = max(10, max_candidates // 2)  # More selective in complex positions
        elif complexity['complexity_score'] < self.config['complexity_threshold_low']:
            max_candidates = min(max_candidates * 2, 40)  # Consider more moves in simple positions
        
        # Get prepared moves from move ordering system
        prepared_moves = self.move_integration.prepare_for_network(board, max_candidates)
        
        # Evaluate moves with bounty system
        evaluated_moves = []
        moves_evaluated = 0
        
        for move in prepared_moves:
            # Check time limit
            if time_limit and (time.time() - start_time) > time_limit:
                break
            
            # Get move preparation score
            move_score = self.move_preparation._score_move(board, move)
            
            # Get bounty evaluation
            bounty_score = self._get_bounty_evaluation(board, move)
            
            # Combine scores
            combined_score = self._combine_scores(move_score, bounty_score)
            
            evaluated_moves.append((move, combined_score))
            moves_evaluated += 1
        
        # Sort by combined score
        evaluated_moves.sort(key=lambda x: x[1], reverse=True)
        
        # Update performance metrics
        elapsed_time = time.time() - start_time
        self._update_metrics(moves_evaluated, elapsed_time, len(evaluated_moves))
        
        # Cache result
        if self.config['enable_caching']:
            self.position_cache[position_key] = ([move for move, _ in evaluated_moves], time.time())
        
        return evaluated_moves
    
    def _get_bounty_evaluation(self, board: chess.Board, move: chess.Move) -> BountyScore:
        """Get bounty evaluation with caching"""
        if self.config['enable_caching']:
            move_key = f"{self._get_position_key(board)}_{move.from_square}_{move.to_square}"
            if move_key in self.evaluation_cache:
                return self.evaluation_cache[move_key]
        
        bounty_score = self.bounty_evaluator.evaluate_move(board, move)
        
        if self.config['enable_caching']:
            self.evaluation_cache[move_key] = bounty_score
        
        return bounty_score
    
    def _combine_scores(self, move_score: MoveScore, bounty_score: BountyScore) -> float:
        """Combine move preparation score with bounty score"""
        # Normalize scores
        normalized_move_score = move_score.total_score / 1000.0  # Normalize to 0-10 range
        normalized_bounty_score = bounty_score.total() / 100.0   # Normalize to 0-5 range
        
        # Weighted combination
        combined = (
            normalized_move_score * self.config['move_order_weight'] +
            normalized_bounty_score * self.config['bounty_weight']
        )
        
        return combined
    
    def _determine_game_phase(self, board: chess.Board) -> str:
        """Determine current game phase"""
        # Count total material
        total_material = 0
        for color in [chess.WHITE, chess.BLACK]:
            for piece_type, value in self.bounty_evaluator.piece_values.items():
                if piece_type != chess.KING:
                    count = len(list(board.pieces(piece_type, color)))
                    total_material += count * value
        
        if total_material > 60:
            return "opening"
        elif total_material > 20:
            return "middlegame"
        else:
            return "endgame"
    
    def _get_max_candidates(self, game_phase: str) -> int:
        """Get maximum candidates based on game phase"""
        return {
            "opening": self.config['max_candidates_opening'],
            "middlegame": self.config['max_candidates_middlegame'],
            "endgame": self.config['max_candidates_endgame']
        }.get(game_phase, 20)
    
    def _get_position_key(self, board: chess.Board) -> str:
        """Generate a unique key for position caching"""
        return f"{board.fen().split(' ')[0]}_{board.turn}_{board.castling_rights}"
    
    def _update_metrics(self, moves_evaluated: int, elapsed_time: float, total_candidates: int):
        """Update performance metrics"""
        self.metrics.moves_evaluated = moves_evaluated
        self.metrics.evaluation_time = elapsed_time
        self.metrics.moves_per_second = moves_evaluated / elapsed_time if elapsed_time > 0 else 0
        self.metrics.efficiency_ratio = moves_evaluated / max(total_candidates, 1)
        
        # Add to history
        self.performance_history.append(PerformanceMetrics(
            moves_evaluated=moves_evaluated,
            evaluation_time=elapsed_time,
            moves_per_second=self.metrics.moves_per_second,
            efficiency_ratio=self.metrics.efficiency_ratio
        ))
    
    def analyze_performance(self, board: chess.Board, iterations: int = 100) -> Dict:
        """
        Analyze performance across multiple evaluations
        
        Args:
            board: Test position
            iterations: Number of test iterations
            
        Returns:
            Performance analysis results
        """
        print(f"Running performance analysis with {iterations} iterations...")
        
        times = []
        moves_per_iteration = []
        
        for i in range(iterations):
            start_time = time.time()
            evaluated_moves = self.optimize_move_selection(board)
            elapsed = time.time() - start_time
            
            times.append(elapsed)
            moves_per_iteration.append(len(evaluated_moves))
            
            if (i + 1) % 20 == 0:
                print(f"Completed {i + 1}/{iterations} iterations...")
        
        analysis = {
            'average_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'std_time': np.std(times),
            'average_moves_evaluated': np.mean(moves_per_iteration),
            'average_moves_per_second': np.mean([m/t for m, t in zip(moves_per_iteration, times) if t > 0]),
            'total_iterations': iterations,
            'cache_hit_rate': len(self.position_cache) / iterations if iterations > 0 else 0,
            'configuration': self.config.copy()
        }
        
        return analysis
    
    def optimize_configuration(self, board: chess.Board, target_moves_per_second: float = 1000) -> Dict:
        """
        Optimize configuration parameters for target performance
        
        Args:
            board: Test position
            target_moves_per_second: Target evaluation speed
            
        Returns:
            Optimized configuration
        """
        print(f"Optimizing configuration for target {target_moves_per_second} moves/second...")
        
        best_config = self.config.copy()
        best_performance = 0.0
        
        # Test different parameter combinations
        test_configs = [
            {'max_candidates_middlegame': 15, 'bounty_weight': 0.7, 'move_order_weight': 0.3},
            {'max_candidates_middlegame': 20, 'bounty_weight': 0.6, 'move_order_weight': 0.4},
            {'max_candidates_middlegame': 25, 'bounty_weight': 0.5, 'move_order_weight': 0.5},
            {'max_candidates_middlegame': 30, 'bounty_weight': 0.4, 'move_order_weight': 0.6},
        ]
        
        for test_config in test_configs:
            # Update configuration
            original_config = self.config.copy()
            self.config.update(test_config)
            
            # Test performance
            analysis = self.analyze_performance(board, iterations=50)
            performance_score = analysis['average_moves_per_second']
            
            print(f"Config {test_config}: {performance_score:.1f} moves/second")
            
            if performance_score > best_performance and performance_score >= target_moves_per_second:
                best_performance = performance_score
                best_config.update(test_config)
            
            # Restore original config
            self.config = original_config
        
        # Apply best configuration
        self.config = best_config
        print(f"Best configuration: {best_config}")
        print(f"Achieved {best_performance:.1f} moves/second")
        
        return best_config
    
    def get_neural_network_features(self, board: chess.Board, move: chess.Move) -> np.ndarray:
        """
        Generate comprehensive features for neural network input
        
        Args:
            board: Current position
            move: Move to evaluate
            
        Returns:
            Feature vector combining move preparation and bounty features
        """
        # Get move preparation features
        move_features = self.move_integration.get_move_features(board, move)
        
        # Get bounty evaluation
        bounty_score = self._get_bounty_evaluation(board, move)
        
        # Create bounty features
        bounty_features = np.array([
            bounty_score.center_control / 10.0,
            bounty_score.piece_value / 100.0,
            bounty_score.attack_patterns / 20.0,
            bounty_score.defensive_measures / 20.0,
            bounty_score.piece_protection / 15.0,
            bounty_score.king_safety / 50.0,
            bounty_score.tactical_patterns / 25.0,
            bounty_score.counter_threats / 15.0,
            bounty_score.mate_threats / 100.0,
            bounty_score.piece_coordination / 20.0,
            bounty_score.defensive_coordination / 15.0,
            bounty_score.castling / 30.0,
            bounty_score.material_balance / 50.0,
            bounty_score.positional_advantage / 20.0,
            bounty_score.game_phase_bonus / 10.0,
            bounty_score.initiative / 15.0,
            bounty_score.positional / 20.0
        ])
        
        # Combine features
        combined_features = np.concatenate([move_features, bounty_features])
        
        return combined_features
    
    def save_performance_report(self, filename: str):
        """Save performance analysis to file"""
        report = {
            'configuration': self.config,
            'current_metrics': self.metrics.to_dict(),
            'performance_history': [m.to_dict() for m in self.performance_history[-100:]],  # Last 100 entries
            'cache_statistics': {
                'position_cache_size': len(self.position_cache),
                'evaluation_cache_size': len(self.evaluation_cache)
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Performance report saved to {filename}")
    
    def clear_caches(self):
        """Clear all caches to free memory"""
        self.position_cache.clear()
        self.evaluation_cache.clear()
        print("Caches cleared")


def main():
    """Test the performance optimization system"""
    print("V7P3R Chess AI 2.0 - Performance Optimization System Test")
    print("=" * 60)
    
    # Initialize optimizer
    optimizer = PerformanceOptimizer()
    
    # Test positions
    test_positions = [
        chess.Board(),  # Starting position
        chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),  # After 1.e4
        chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"),  # After 1.e4 d5
    ]
    
    for i, board in enumerate(test_positions):
        print(f"\nTesting position {i + 1}: {board.fen()[:30]}...")
        
        # Optimize move selection
        start_time = time.time()
        optimized_moves = optimizer.optimize_move_selection(board, time_limit=1.0)
        elapsed = time.time() - start_time
        
        print(f"Evaluated {len(optimized_moves)} moves in {elapsed:.3f}s")
        print(f"Top 5 moves: {[str(move) for move, score in optimized_moves[:5]]}")
        
        # Get neural network features for best move
        if optimized_moves:
            best_move, best_score = optimized_moves[0]
            features = optimizer.get_neural_network_features(board, best_move)
            print(f"Feature vector length: {len(features)}")
            print(f"Best move {best_move} score: {best_score:.3f}")
    
    # Performance analysis
    print(f"\nRunning comprehensive performance analysis...")
    analysis = optimizer.analyze_performance(chess.Board(), iterations=50)
    
    print(f"\nPerformance Analysis Results:")
    print(f"Average time per evaluation: {analysis['average_time']:.4f}s")
    print(f"Average moves per second: {analysis['average_moves_per_second']:.1f}")
    print(f"Cache hit rate: {analysis['cache_hit_rate']:.2%}")
    
    # Save performance report
    optimizer.save_performance_report("reports/performance_optimization_test.json")
    
    print(f"\nPerformance optimization system test completed successfully!")


if __name__ == "__main__":
    main()
