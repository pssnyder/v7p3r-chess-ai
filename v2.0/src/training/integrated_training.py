# integrated_training.py
"""
V7P3R Chess AI 2.0 - Integrated Training System
Combines the improved bounty system, move preparation, and performance optimization
for enhanced neural network training.
"""

import os
import time
import json
import chess
import chess.pgn
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# Import our enhanced systems
import sys
sys.path.append('src')
sys.path.append('src/evaluation')
sys.path.append('src/core')
sys.path.append('src/training')

from evaluation.v7p3r_bounty_system import ExtendedBountyEvaluator
from core.performance_optimizer import PerformanceOptimizer
from core.move_preparation import MovePreparation, MovePreparationIntegration


class IntegratedTrainingSystem:
    """
    Integrated training system that combines all our enhancements:
    - Improved bounty system with offensive/defensive/outcome evaluation
    - Move preparation and ordering for efficiency
    - Performance optimization with caching
    - Smart move filtering and feature extraction
    """
    
    def __init__(self):
        print("Initializing V7P3R Chess AI 2.0 Integrated Training System...")
        
        # Initialize all systems
        self.bounty_evaluator = ExtendedBountyEvaluator()
        self.performance_optimizer = PerformanceOptimizer()
        self.move_preparation = MovePreparation()
        
        # Training configuration
        self.config = {
            'max_positions_per_game': 50,
            'max_moves_per_position': 15,
            'min_game_length': 20,
            'max_game_length': 150,
            'complexity_threshold': 30.0,
            'time_per_position': 0.05,  # 50ms per position
            'batch_size': 100,
            'save_interval': 1000,
            'performance_analysis_interval': 500
        }
        
        # Statistics tracking
        self.stats = {
            'positions_processed': 0,
            'moves_evaluated': 0,
            'games_processed': 0,
            'total_training_time': 0.0,
            'performance_metrics': []
        }
        
        print("✓ All systems initialized successfully")
    
    def process_pgn_file(self, pgn_file: str, output_file: str, max_games: Optional[int] = None) -> Dict:
        """
        Process a PGN file to generate enhanced training data
        
        Args:
            pgn_file: Path to PGN file
            output_file: Path for output training data
            max_games: Maximum number of games to process
            
        Returns:
            Processing statistics
        """
        print(f"Processing PGN file: {pgn_file}")
        start_time = time.time()
        
        training_data = []
        games_processed = 0
        
        try:
            with open(pgn_file, 'r') as f:
                while True:
                    if max_games and games_processed >= max_games:
                        break
                    
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    
                    # Process game
                    game_data = self._process_game(game)
                    if game_data:
                        training_data.extend(game_data)
                        games_processed += 1
                        
                        if games_processed % 100 == 0:
                            print(f"Processed {games_processed} games, {len(training_data)} training samples")
                            
                        # Save periodically
                        if len(training_data) >= self.config['save_interval']:
                            self._save_training_data(training_data, output_file, append=games_processed > 100)
                            training_data = []
                
                # Save remaining data
                if training_data:
                    self._save_training_data(training_data, output_file, append=games_processed > 100)
        
        except FileNotFoundError:
            print(f"Error: PGN file {pgn_file} not found")
            return {}
        except Exception as e:
            print(f"Error processing PGN file: {e}")
            return {}
        
        # Calculate statistics
        total_time = time.time() - start_time
        self.stats['total_training_time'] += total_time
        self.stats['games_processed'] += games_processed
        
        processing_stats = {
            'games_processed': games_processed,
            'positions_processed': self.stats['positions_processed'],
            'moves_evaluated': self.stats['moves_evaluated'],
            'processing_time': total_time,
            'positions_per_second': self.stats['positions_processed'] / total_time if total_time > 0 else 0,
            'moves_per_second': self.stats['moves_evaluated'] / total_time if total_time > 0 else 0
        }
        
        print(f"PGN processing complete:")
        print(f"  Games processed: {games_processed}")
        print(f"  Positions processed: {self.stats['positions_processed']}")
        print(f"  Moves evaluated: {self.stats['moves_evaluated']}")
        print(f"  Processing time: {total_time:.2f}s")
        print(f"  Performance: {processing_stats['positions_per_second']:.1f} positions/s")
        
        return processing_stats
    
    def _process_game(self, game: chess.pgn.Game) -> List[Dict]:
        """Process a single game to extract training data"""
        game_data = []
        
        # Get game result
        result = game.headers.get('Result', '*')
        game_outcome = self._parse_game_result(result)
        
        # Check game length
        board = game.board()
        moves = list(game.mainline_moves())
        
        if len(moves) < self.config['min_game_length'] or len(moves) > self.config['max_game_length']:
            return game_data
        
        # Process positions
        positions_in_game = 0
        
        for move_idx, move in enumerate(moves):
            if positions_in_game >= self.config['max_positions_per_game']:
                break
            
            # Skip very early opening moves (first 6 moves)
            if move_idx < 6:
                board.push(move)
                continue
            
            try:
                # Analyze position complexity
                complexity = self.move_preparation.analyze_position_complexity(board)
                
                # Skip positions that are too simple or too complex
                if (complexity['complexity_score'] < 5.0 or 
                    complexity['complexity_score'] > self.config['complexity_threshold']):
                    board.push(move)
                    continue
                
                # Generate training data for this position
                position_data = self._generate_position_data(
                    board, move, game_outcome, move_idx, len(moves)
                )
                
                if position_data:
                    game_data.extend(position_data)
                    positions_in_game += 1
                    self.stats['positions_processed'] += 1
                
                board.push(move)
                
            except Exception as e:
                print(f"Error processing move {move_idx}: {e}")
                break
        
        return game_data
    
    def _generate_position_data(self, board: chess.Board, played_move: chess.Move,
                              game_outcome: float, move_idx: int, total_moves: int) -> List[Dict]:
        """Generate training data for a single position"""
        position_data = []
        
        try:
            # Get optimized move candidates
            start_time = time.time()
            optimized_moves = self.performance_optimizer.optimize_move_selection(
                board, time_limit=self.config['time_per_position']
            )
            
            if not optimized_moves:
                return position_data
            
            # Limit moves to evaluate
            moves_to_evaluate = min(len(optimized_moves), self.config['max_moves_per_position'])
            
            for rank, (move, optimizer_score) in enumerate(optimized_moves[:moves_to_evaluate]):
                # Get comprehensive features
                move_features = self.performance_optimizer.get_neural_network_features(board, move)
                
                # Get bounty evaluation
                bounty_score = self.bounty_evaluator.evaluate_move(board, move)
                
                # Calculate target values
                move_quality = self._calculate_move_quality(
                    move, played_move, optimizer_score, rank, game_outcome, move_idx, total_moves
                )
                
                # Create training sample
                sample = {
                    'position_fen': board.fen(),
                    'move_uci': move.uci(),
                    'move_features': move_features.tolist(),
                    'bounty_score': {
                        'total': bounty_score.total(),
                        'offensive': bounty_score.offensive_total(),
                        'defensive': bounty_score.defensive_total(),
                        'outcome': bounty_score.outcome_total(),
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
                    },
                    'optimizer_score': optimizer_score,
                    'move_quality': move_quality,
                    'move_rank': rank,
                    'was_played': move == played_move,
                    'game_outcome': game_outcome,
                    'move_progress': move_idx / max(total_moves, 1),
                    'timestamp': datetime.now().isoformat()
                }
                
                position_data.append(sample)
                self.stats['moves_evaluated'] += 1
            
            elapsed = time.time() - start_time
            
            # Performance analysis
            if self.stats['positions_processed'] % self.config['performance_analysis_interval'] == 0:
                self._analyze_performance(elapsed, moves_to_evaluate)
                
        except Exception as e:
            print(f"Error generating position data: {e}")
        
        return position_data
    
    def _calculate_move_quality(self, move: chess.Move, played_move: chess.Move,
                              optimizer_score: float, rank: int, game_outcome: float,
                              move_idx: int, total_moves: int) -> float:
        """Calculate move quality score (0-1) based on multiple factors"""
        
        # Base quality starts at neutral
        quality = 0.5
        
        # Strong bonus if this is the move that was actually played
        if move == played_move:
            quality += 0.25
        
        # Ranking bonus (top moves get higher quality)
        rank_bonus = max(0, (15 - rank) / 15 * 0.2)
        quality += rank_bonus
        
        # Optimizer score bonus
        score_bonus = min(optimizer_score / 100.0, 0.15)
        quality += score_bonus
        
        # Game outcome influence (stronger for later moves)
        move_progress = move_idx / max(total_moves, 1)
        if game_outcome != 0:  # Only for decisive games
            outcome_influence = game_outcome * move_progress * 0.1
            quality += outcome_influence
        
        # Ensure quality is in valid range
        return max(0.1, min(0.9, quality))
    
    def _parse_game_result(self, result: str) -> float:
        """Parse game result to numeric value"""
        if result == '1-0':
            return 1.0  # White wins
        elif result == '0-1':
            return -1.0  # Black wins
        else:
            return 0.0  # Draw
    
    def _save_training_data(self, data: List[Dict], output_file: str, append: bool = False):
        """Save training data to file"""
        mode = 'a' if append else 'w'
        
        with open(output_file, mode) as f:
            for sample in data:
                f.write(json.dumps(sample) + '\n')
        
        if not append:
            print(f"Training data saved to {output_file}")
        else:
            print(f"Training data appended to {output_file} ({len(data)} samples)")
    
    def _analyze_performance(self, last_position_time: float, moves_evaluated: int):
        """Analyze and report performance metrics"""
        current_metrics = {
            'timestamp': datetime.now().isoformat(),
            'positions_processed': self.stats['positions_processed'],
            'moves_evaluated': self.stats['moves_evaluated'],
            'last_position_time': last_position_time,
            'moves_per_position': moves_evaluated,
            'cache_sizes': {
                'position_cache': len(self.performance_optimizer.position_cache),
                'evaluation_cache': len(self.performance_optimizer.evaluation_cache)
            }
        }
        
        self.stats['performance_metrics'].append(current_metrics)
        
        # Print performance update
        if self.stats['total_training_time'] > 0:
            overall_rate = self.stats['positions_processed'] / self.stats['total_training_time']
            print(f"Performance update: {self.stats['positions_processed']} positions, "
                  f"{overall_rate:.1f} pos/sec, caches: {current_metrics['cache_sizes']}")
    
    def generate_performance_report(self, output_file: str):
        """Generate comprehensive performance report"""
        report = {
            'system_info': {
                'version': 'V7P3R Chess AI 2.0',
                'timestamp': datetime.now().isoformat(),
                'configuration': self.config
            },
            'statistics': self.stats,
            'performance_optimization': self.performance_optimizer.config,
            'bounty_system_config': {
                'bounty_rates': self.bounty_evaluator.bounty_rates,
                'piece_values': self.bounty_evaluator.piece_values
            },
            'move_preparation_config': {
                'ordering_weights': self.move_preparation.ordering_weights,
                'piece_values': self.move_preparation.piece_values
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Performance report saved to {output_file}")
    
    def run_training_session(self, pgn_files: List[str], output_dir: str, 
                           max_games_per_file: Optional[int] = None) -> Dict:
        """
        Run a complete training session on multiple PGN files
        
        Args:
            pgn_files: List of PGN file paths
            output_dir: Directory for output files
            max_games_per_file: Maximum games to process per file
            
        Returns:
            Session statistics
        """
        print(f"Starting V7P3R Chess AI 2.0 training session...")
        print(f"Files to process: {len(pgn_files)}")
        print(f"Output directory: {output_dir}")
        print("=" * 60)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        session_start = time.time()
        session_stats = {
            'files_processed': 0,
            'total_games': 0,
            'total_positions': 0,
            'total_moves': 0,
            'processing_time': 0.0,
            'file_results': []
        }
        
        for file_idx, pgn_file in enumerate(pgn_files):
            print(f"\nProcessing file {file_idx + 1}/{len(pgn_files)}: {pgn_file}")
            
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(pgn_file))[0]
            output_file = os.path.join(output_dir, f"training_data_{base_name}.jsonl")
            
            # Process file
            file_stats = self.process_pgn_file(pgn_file, output_file, max_games_per_file)
            
            if file_stats:
                session_stats['files_processed'] += 1
                session_stats['total_games'] += file_stats['games_processed']
                session_stats['total_positions'] += file_stats['positions_processed']
                session_stats['total_moves'] += file_stats['moves_evaluated']
                session_stats['file_results'].append({
                    'file': pgn_file,
                    'output': output_file,
                    'stats': file_stats
                })
        
        # Calculate session totals
        session_stats['processing_time'] = time.time() - session_start
        
        # Generate comprehensive report
        report_file = os.path.join(output_dir, f"training_session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        session_stats['report_file'] = report_file
        
        with open(report_file, 'w') as f:
            json.dump(session_stats, f, indent=2)
        
        # Generate performance report
        perf_report_file = os.path.join(output_dir, f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        self.generate_performance_report(perf_report_file)
        
        # Print session summary
        print(f"\n" + "=" * 60)
        print(f"Training session completed!")
        print(f"Files processed: {session_stats['files_processed']}")
        print(f"Total games: {session_stats['total_games']}")
        print(f"Total positions: {session_stats['total_positions']}")
        print(f"Total moves evaluated: {session_stats['total_moves']}")
        print(f"Session time: {session_stats['processing_time']:.2f}s")
        
        if session_stats['processing_time'] > 0:
            print(f"Performance: {session_stats['total_positions'] / session_stats['processing_time']:.1f} positions/sec")
        
        print(f"Session report: {report_file}")
        print(f"Performance report: {perf_report_file}")
        
        return session_stats


def main():
    """Main training execution"""
    print("V7P3R Chess AI 2.0 - Integrated Training System")
    print("=" * 60)
    
    # Initialize training system
    trainer = IntegratedTrainingSystem()
    
    # Test with available data
    data_dir = "data"
    output_dir = "training_output"
    
    # Look for PGN files
    pgn_files = []
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith('.pgn'):
                pgn_files.append(os.path.join(data_dir, file))
    
    if not pgn_files:
        print(f"No PGN files found in {data_dir}")
        print("Creating test training data...")
        
        # Create test data with the existing system
        test_board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2")
        
        optimized_moves = trainer.performance_optimizer.optimize_move_selection(test_board)
        print(f"Test position analysis:")
        print(f"  FEN: {test_board.fen()}")
        print(f"  Top moves: {[str(move) for move, score in optimized_moves[:5]]}")
        
        # Generate test training data
        test_data = trainer._generate_position_data(test_board, optimized_moves[0][0], 1.0, 10, 50)
        print(f"  Generated {len(test_data)} training samples")
        
        # Save test data
        os.makedirs(output_dir, exist_ok=True)
        test_file = os.path.join(output_dir, "test_training_data.jsonl")
        trainer._save_training_data(test_data, test_file)
        
        # Generate performance report
        perf_report = os.path.join(output_dir, "test_performance_report.json")
        trainer.generate_performance_report(perf_report)
        
    else:
        print(f"Found {len(pgn_files)} PGN files:")
        for file in pgn_files:
            print(f"  {file}")
        
        # Run training session
        session_stats = trainer.run_training_session(
            pgn_files, 
            output_dir, 
            max_games_per_file=100  # Limit for testing
        )
    
    print(f"\nIntegrated training system test completed!")


if __name__ == "__main__":
    main()
