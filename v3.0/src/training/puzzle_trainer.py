"""
V7P3R AI Puzzle-Based Training System

Revolutionary approach to chess AI training using tactical puzzles instead of self-play.
This system provides:
- Immediate feedback with known correct answers
- Quality grading via Stockfish comparison (1-5 points)
- Targeted learning on specific tactical themes
- Much faster iteration than full game self-play
- Scalable access to millions of puzzle positions

Training Process:
1. Load puzzles from database with specific themes/ratings
2. Present challenge position to AI (after opponent's setup move)
3. AI analyzes and returns best move using ThinkingBrain + memory
4. Compare AI move to puzzle solution and Stockfish top 5 moves
5. Train neural network using puzzle solution as target and quality score as fitness
6. Update position memory with outcome-based weighting
"""

import os
import sys
import json
import time
import chess
import chess.engine
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from tqdm import tqdm
import logging

try:
    import torch
except ImportError:
    print("Warning: PyTorch not available")
    torch = None

# Add puzzle database to path
current_dir = Path(__file__).parent.parent.parent
puzzle_db_path = Path("s:/Maker Stuff/Programming/Chess Engines/Chess Engine Playground/engine-tester/chess-puzzle-challenger/src")
if puzzle_db_path.exists():
    sys.path.append(str(puzzle_db_path))

try:
    from database import PuzzleDatabase, Puzzle
except ImportError:
    print("Warning: Puzzle database not available. Falling back to synthetic puzzles.")
    PuzzleDatabase = None
    Puzzle = None

from ai.thinking_brain import ThinkingBrain, PositionMemory
from core.chess_state import ChessStateExtractor
from core.neural_features import NeuralFeatureConverter

logger = logging.getLogger(__name__)


class PuzzleTrainer:
    """
    Advanced puzzle-based training system for V7P3R AI
    
    Uses tactical puzzles as training data with immediate feedback and quality scoring
    """
    
    def __init__(self, 
                 thinking_brain: ThinkingBrain,
                 stockfish_path: Optional[str] = None,
                 puzzle_db_path: Optional[str] = None,
                 save_directory: str = "models",
                 memory_config: Optional[Dict] = None):
        
        self.thinking_brain = thinking_brain
        self.save_directory = Path(save_directory)
        self.save_directory.mkdir(exist_ok=True)
        
        # Initialize memory system
        self.memory = PositionMemory(thinking_brain, memory_config)
        
        # Puzzle database
        default_puzzle_db = "s:/Maker Stuff/Programming/Chess Engines/Chess Engine Playground/engine-tester/chess-puzzle-challenger/puzzles.db"
        self.puzzle_db_path = puzzle_db_path or default_puzzle_db
        
        # Stockfish for move quality evaluation
        default_stockfish = "s:/Maker Stuff/Programming/Chess Engines/Chess Engine Playground/engine-tester/engines/Stockfish/stockfish-windows-x86-64-avx2.exe"
        self.stockfish_path = stockfish_path or default_stockfish
        
        # Feature converters
        self.state_extractor = ChessStateExtractor()
        self.feature_converter = NeuralFeatureConverter()
        
        # Training progress tracking
        self.training_stats = {
            'puzzles_solved': 0,
            'total_score': 0,
            'perfect_solutions': 0,
            'top5_hits': 0,
            'theme_performance': {},
            'learning_curve': [],
            'checkpoint_history': []
        }
        
        # Optimizer for neural network training (if torch available)
        if torch:
            self.optimizer = torch.optim.Adam(
                self.thinking_brain.parameters(), 
                lr=0.001, 
                weight_decay=1e-5
            )
            self.loss_function = torch.nn.CrossEntropyLoss()
        else:
            self.optimizer = None
            self.loss_function = None
        
        logger.info("PuzzleTrainer initialized - Revolutionary puzzle-based training system")
    
    def load_puzzles(self, 
                     num_puzzles: int = 1000,
                     rating_min: int = 1200,
                     rating_max: int = 1800,
                     themes_filter: Optional[List[str]] = None) -> List[Any]:
        """Load puzzles from database with filtering"""
        
        if not PuzzleDatabase or not os.path.exists(self.puzzle_db_path):
            logger.warning("Puzzle database not available, generating synthetic puzzles")
            return self._generate_synthetic_puzzles(num_puzzles)
        
        try:
            db = PuzzleDatabase(self.puzzle_db_path)
            
            # Default tactical themes for chess AI training
            if not themes_filter:
                themes_filter = [
                    'pin', 'fork', 'skewer', 'discovery', 'deflection',
                    'mate', 'mateIn1', 'mateIn2', 'mateIn3',
                    'attackingF2F7', 'hangingPiece', 'trapped',
                    'endgame', 'promotion', 'castling'
                ]
            
            puzzles = db.query_puzzles(
                themes=themes_filter,
                min_rating=rating_min,
                max_rating=rating_max,
                quantity=num_puzzles
            )
            
            logger.info(f"Loaded {len(puzzles)} puzzles (rating {rating_min}-{rating_max})")
            return puzzles
            
        except Exception as e:
            logger.error(f"Error loading puzzles: {e}")
            return self._generate_synthetic_puzzles(num_puzzles)
    
    def _generate_synthetic_puzzles(self, num_puzzles: int) -> List:
        """Generate synthetic puzzle-like positions when database unavailable"""
        logger.info(f"Generating {num_puzzles} synthetic tactical positions")
        
        synthetic_puzzles = []
        
        # Common tactical patterns
        tactical_fens = [
            # Fork patterns
            "rnbqkbnr/pppp1ppp/8/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR b KQkq - 2 2",
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 4",
            # Pin patterns  
            "rnbqkb1r/ppp2ppp/3p1n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 4",
            # Discovery patterns
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        ]
        
        for i in range(num_puzzles):
            fen = tactical_fens[i % len(tactical_fens)]
            
            # Create mock puzzle object
            puzzle = type('Puzzle', (), {
                'id': f"synthetic_{i}",
                'fen': fen,
                'moves': "e5e4 f3e5",  # Simple tactical sequence
                'rating': 1400 + (i % 600),  # Spread ratings 1400-2000
                'themes': "fork pin discovery"
            })()
            
            synthetic_puzzles.append(puzzle)
        
        return synthetic_puzzles
    
    def get_challenge_position(self, puzzle) -> Tuple[str, str, bool, str]:
        """
        Extract the actual challenge position from puzzle
        Returns: (challenge_fen, expected_move, is_valid, context_info)
        """
        try:
            board = chess.Board(puzzle.fen)
            expected_moves = puzzle.moves.split() if puzzle.moves else []
            
            if len(expected_moves) < 2:
                return puzzle.fen, "unknown", False, "Insufficient moves in solution"
            
            # Play opponent's setup move
            opponent_move_text = expected_moves[0]
            try:
                opponent_move = chess.Move.from_uci(opponent_move_text)
                if opponent_move not in board.legal_moves:
                    raise ValueError("Move not legal")
            except:
                try:
                    opponent_move = board.parse_san(opponent_move_text)
                except:
                    return puzzle.fen, "unknown", False, f"Cannot parse opponent move: {opponent_move_text}"
            
            # Apply opponent's move to get challenge position
            board.push(opponent_move)
            challenge_fen = board.fen()
            
            # Expected AI response
            expected_move_text = expected_moves[1]
            try:
                expected_move = chess.Move.from_uci(expected_move_text)
                if expected_move not in board.legal_moves:
                    raise ValueError("Move not legal")
                expected_move_uci = str(expected_move)
            except:
                try:
                    expected_move = board.parse_san(expected_move_text)
                    expected_move_uci = str(expected_move)
                except:
                    return challenge_fen, expected_move_text, False, f"Cannot parse expected move: {expected_move_text}"
            
            turn_info = f"{'White' if board.turn else 'Black'} to move"
            return challenge_fen, expected_move_uci, True, turn_info
            
        except Exception as e:
            return puzzle.fen, "unknown", False, f"Error processing puzzle: {e}"
    
    def get_stockfish_evaluation(self, fen: str, num_moves: int = 5) -> List[Tuple[str, int]]:
        """Get Stockfish's top moves with scores for move quality evaluation"""
        try:
            if not os.path.exists(self.stockfish_path):
                logger.warning("Stockfish not found, using simplified evaluation")
                return self._get_simplified_evaluation(fen, num_moves)
            
            with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
                board = chess.Board(fen)
                
                result = engine.analyse(
                    board, 
                    chess.engine.Limit(time=1.0),  # Quick analysis
                    multipv=num_moves
                )
                
                moves_with_scores = []
                for analysis in result:
                    if 'pv' in analysis and analysis['pv']:
                        move = analysis['pv'][0]
                        score = analysis.get('score', chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE))
                        
                        if score.is_mate():
                            mate_in = score.white().mate()
                            if mate_in is not None:
                                cp_score = 10000 - abs(mate_in) * 100 if mate_in > 0 else -10000 + abs(mate_in) * 100
                            else:
                                cp_score = 0
                        else:
                            cp_score = score.white().score()
                        
                        moves_with_scores.append((str(move), cp_score))
                
                return moves_with_scores
                
        except Exception as e:
            logger.warning(f"Stockfish evaluation failed: {e}")
            return self._get_simplified_evaluation(fen, num_moves)
    
    def _get_simplified_evaluation(self, fen: str, num_moves: int) -> List[Tuple[str, int]]:
        """Simplified move evaluation when Stockfish unavailable"""
        try:
            board = chess.Board(fen)
            legal_moves = list(board.legal_moves)
            
            # Simple heuristic scoring
            moves_with_scores = []
            for move in legal_moves[:num_moves]:
                board.push(move)
                
                # Basic evaluation: material + positional
                score = 0
                for square in chess.SQUARES:
                    piece = board.piece_at(square)
                    if piece:
                        piece_value = {
                            chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
                            chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0
                        }.get(piece.piece_type, 0)
                        
                        if piece.color == chess.WHITE:
                            score += piece_value
                        else:
                            score -= piece_value
                
                board.pop()
                moves_with_scores.append((str(move), score))
            
            # Sort by score (best first)
            moves_with_scores.sort(key=lambda x: x[1], reverse=True)
            return moves_with_scores
            
        except Exception as e:
            logger.error(f"Simplified evaluation failed: {e}")
            return []
    
    def score_ai_move(self, ai_move: str, stockfish_moves: List[Tuple[str, int]]) -> Tuple[int, int]:
        """
        Score AI move based on Stockfish ranking
        Returns: (score, rank) where score is 0-5 points
        """
        if not ai_move or not stockfish_moves:
            return 0, 0
        
        for rank, (sf_move, _) in enumerate(stockfish_moves, 1):
            if ai_move == sf_move:
                score = 6 - rank  # 5pts for 1st, 4pts for 2nd, ..., 1pt for 5th
                return score, rank
        
        return 0, 0  # Not in top 5
    
    def train_on_puzzle(self, puzzle) -> Optional[Dict]:
        """
        Train the AI on a single puzzle with neural network learning
        Returns puzzle analysis result
        """
        # Get challenge position
        challenge_fen, expected_move, is_valid, context_info = self.get_challenge_position(puzzle)
        
        if not is_valid:
            return None
        
        try:
            # Extract position features
            board = chess.Board(challenge_fen)
            current_state = self.state_extractor.extract_state(board)
            position_features = self.feature_converter.convert_to_features(
                current_state, device=str(self.thinking_brain.device)
            )
            
            # Get legal moves
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return None
            
            # AI analyzes position using enhanced memory
            start_time = time.time()
            ai_candidates, ai_probabilities = self.memory.process_position(
                position_features, legal_moves, top_k=5
            )
            analysis_time = time.time() - start_time
            
            if not ai_candidates:
                return None
            
            ai_move = str(ai_candidates[0])  # Best move from AI
            
            # Get Stockfish evaluation for quality scoring
            stockfish_moves = self.get_stockfish_evaluation(challenge_fen, 5)
            
            # Score AI performance
            score, rank = self.score_ai_move(ai_move, stockfish_moves)
            found_solution = ai_move == expected_move
            
            # Neural network training
            self._train_neural_network(
                position_features, legal_moves, expected_move, score, found_solution
            )
            
            # Update memory with outcome-based weighting
            outcome_score = 1.0 if found_solution else (score / 5.0)  # 0.0 to 1.0
            self.memory.finalize_game_memory(outcome_score)
            
            # Record training statistics
            self.training_stats['puzzles_solved'] += 1
            self.training_stats['total_score'] += score
            if found_solution:
                self.training_stats['perfect_solutions'] += 1
            if rank > 0:
                self.training_stats['top5_hits'] += 1
            
            # Theme performance tracking
            themes = puzzle.themes.split() if hasattr(puzzle, 'themes') and puzzle.themes else ['general']
            for theme in themes:
                if theme not in self.training_stats['theme_performance']:
                    self.training_stats['theme_performance'][theme] = {'total': 0, 'score_sum': 0}
                self.training_stats['theme_performance'][theme]['total'] += 1
                self.training_stats['theme_performance'][theme]['score_sum'] += score
            
            result = {
                'puzzle_id': getattr(puzzle, 'id', 'unknown'),
                'challenge_fen': challenge_fen,
                'expected_move': expected_move,
                'ai_move': ai_move,
                'ai_score': score,
                'ai_rank': rank,
                'found_solution': found_solution,
                'stockfish_moves': stockfish_moves,
                'analysis_time': analysis_time,
                'themes': themes,
                'rating': getattr(puzzle, 'rating', 0),
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error training on puzzle: {e}")
            return None
    
    def _train_neural_network(self, position_features: Any, 
                             legal_moves: List[chess.Move], 
                             expected_move: str, 
                             quality_score: int,
                             found_solution: bool):
        """Train the neural network using puzzle feedback"""
        if not torch or not self.optimizer or not self.loss_function:
            return  # Skip neural training if torch not available
            
        try:
            # Create target tensor for expected move
            target_index = None
            for i, move in enumerate(legal_moves):
                if str(move) == expected_move:
                    target_index = i
                    break
            
            if target_index is None:
                return  # Expected move not in legal moves
            
            # Forward pass through thinking brain
            self.thinking_brain.train()
            self.optimizer.zero_grad()
            
            # Get move predictions
            move_logits, _ = self.thinking_brain.forward(position_features)
            
            # Limit to legal moves only
            if len(legal_moves) < move_logits.size(-1):
                move_logits = move_logits[:, :len(legal_moves)]
            
            # Create target with quality-based weighting
            target = torch.tensor([target_index], device=self.thinking_brain.device)
            
            # Calculate loss with quality weighting
            loss = self.loss_function(move_logits, target)
            
            # Quality bonus: reduce loss for high-quality moves
            quality_weight = 1.0 + (quality_score / 5.0)  # 1.0 to 2.0
            if found_solution:
                quality_weight *= 1.5  # Extra bonus for finding exact solution
            
            weighted_loss = loss / quality_weight
            
            # Backward pass and optimization
            weighted_loss.backward()
            
            # Gradient clipping for stability
            if torch.nn:
                torch.nn.utils.clip_grad_norm_(self.thinking_brain.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Record learning curve
            self.training_stats['learning_curve'].append({
                'loss': weighted_loss.item(),
                'quality_score': quality_score,
                'found_solution': found_solution,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Neural network training error: {e}")
    
    def train(self, 
              num_puzzles: int = 1000,
              rating_min: int = 1200,
              rating_max: int = 1800,
              themes_filter: Optional[List[str]] = None,
              checkpoint_interval: int = 100,
              save_progress: bool = True) -> Dict:
        """
        Main training loop using puzzle-based learning
        
        Args:
            num_puzzles: Number of puzzles to train on
            rating_min/max: Puzzle difficulty range
            themes_filter: Specific tactical themes to focus on
            checkpoint_interval: Save model every N puzzles
            save_progress: Whether to save checkpoints and progress
        """
        
        logger.info(f"🧩 Starting puzzle-based training on {num_puzzles} puzzles")
        logger.info(f"Rating range: {rating_min}-{rating_max}")
        logger.info(f"Themes: {themes_filter or 'All tactical themes'}")
        
        # Load puzzles
        puzzles = self.load_puzzles(num_puzzles, rating_min, rating_max, themes_filter)
        if not puzzles:
            logger.error("No puzzles loaded, aborting training")
            return {}
        
        # Reset memory for training session
        self.memory.start_new_game()
        
        # Training loop with progress bar
        results = []
        with tqdm(total=len(puzzles), desc="Puzzle Training") as pbar:
            for i, puzzle in enumerate(puzzles):
                result = self.train_on_puzzle(puzzle)
                
                if result:
                    results.append(result)
                    
                    # Update progress bar
                    avg_score = self.training_stats['total_score'] / max(1, self.training_stats['puzzles_solved'])
                    solution_rate = (self.training_stats['perfect_solutions'] / max(1, self.training_stats['puzzles_solved'])) * 100
                    
                    pbar.set_postfix({
                        'Avg Score': f"{avg_score:.2f}/5.0",
                        'Solutions': f"{solution_rate:.1f}%",
                        'Top5 Rate': f"{(self.training_stats['top5_hits'] / max(1, self.training_stats['puzzles_solved'])) * 100:.1f}%"
                    })
                
                pbar.update(1)
                
                # Checkpoint saving
                if save_progress and (i + 1) % checkpoint_interval == 0:
                    self._save_checkpoint(i + 1, results)
        
        # Final report and save
        if save_progress:
            self._save_final_results(results)
        
        logger.info("🎉 Puzzle-based training completed!")
        return self._generate_training_report(results)
    
    def _save_checkpoint(self, puzzle_count: int, results: List[Dict]):
        """Save training checkpoint"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = self.save_directory / f"v7p3r_puzzle_trained_{puzzle_count}puzzles_{timestamp}.pkl"
        self.thinking_brain.save_model(str(model_path))
        
        # Save training progress
        progress_path = self.save_directory / f"puzzle_training_progress_{puzzle_count}_{timestamp}.json"
        progress_data = {
            'puzzle_count': puzzle_count,
            'training_stats': self.training_stats,
            'recent_results': results[-50:],  # Last 50 results
            'timestamp': timestamp
        }
        
        with open(progress_path, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        self.training_stats['checkpoint_history'].append({
            'puzzle_count': puzzle_count,
            'model_path': str(model_path),
            'progress_path': str(progress_path),
            'timestamp': timestamp
        })
        
        logger.info(f"Checkpoint saved at {puzzle_count} puzzles: {model_path}")
    
    def _save_final_results(self, results: List[Dict]):
        """Save final training results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save final model
        final_model_path = self.save_directory / f"v7p3r_puzzle_trained_final_{timestamp}.pkl"
        self.thinking_brain.save_model(str(final_model_path))
        
        # Save complete analysis
        final_results_path = self.save_directory / f"puzzle_training_complete_{timestamp}.json"
        complete_data = {
            'training_stats': self.training_stats,
            'all_results': results,
            'final_report': self._generate_training_report(results),
            'model_path': str(final_model_path),
            'timestamp': timestamp
        }
        
        with open(final_results_path, 'w') as f:
            json.dump(complete_data, f, indent=2)
        
        logger.info(f"Final results saved: {final_results_path}")
        logger.info(f"Final model saved: {final_model_path}")
    
    def _generate_training_report(self, results: List[Dict]) -> Dict:
        """Generate comprehensive training performance report"""
        if not results:
            return {}
        
        total_puzzles = len(results)
        total_score = sum(r['ai_score'] for r in results)
        perfect_solutions = sum(1 for r in results if r['found_solution'])
        top5_hits = sum(1 for r in results if r['ai_rank'] > 0)
        
        # Score distribution
        score_dist = {i: 0 for i in range(6)}
        for result in results:
            score_dist[result['ai_score']] += 1
        
        # Theme performance
        theme_performance = {}
        for result in results:
            for theme in result['themes']:
                if theme not in theme_performance:
                    theme_performance[theme] = {'total': 0, 'score_sum': 0, 'solutions': 0}
                theme_performance[theme]['total'] += 1
                theme_performance[theme]['score_sum'] += result['ai_score']
                if result['found_solution']:
                    theme_performance[theme]['solutions'] += 1
        
        for theme in theme_performance:
            data = theme_performance[theme]
            data['avg_score'] = data['score_sum'] / data['total']
            data['solution_rate'] = (data['solutions'] / data['total']) * 100
        
        report = {
            'total_puzzles': total_puzzles,
            'total_score': total_score,
            'average_score': total_score / total_puzzles,
            'percentage_score': (total_score / (total_puzzles * 5)) * 100,
            'perfect_solutions': perfect_solutions,
            'solution_rate': (perfect_solutions / total_puzzles) * 100,
            'top5_hits': top5_hits,
            'top5_rate': (top5_hits / total_puzzles) * 100,
            'score_distribution': score_dist,
            'theme_performance': theme_performance,
            'learning_progress': self.training_stats['learning_curve'][-100:],  # Last 100 points
            'timestamp': datetime.now().isoformat()
        }
        
        return report
    
    def print_training_report(self, report: Dict):
        """Print formatted training report"""
        print("\n" + "=" * 80)
        print("V7P3R AI PUZZLE-BASED TRAINING REPORT")
        print("=" * 80)
        
        print(f"Total Puzzles Trained: {report['total_puzzles']}")
        print(f"Average Score: {report['average_score']:.2f}/5.0 ({report['percentage_score']:.1f}%)")
        print(f"Perfect Solutions: {report['perfect_solutions']}/{report['total_puzzles']} ({report['solution_rate']:.1f}%)")
        print(f"Top-5 Hit Rate: {report['top5_hits']}/{report['total_puzzles']} ({report['top5_rate']:.1f}%)")
        
        print("\nScore Distribution:")
        for score in range(5, -1, -1):
            count = report['score_distribution'][score]
            percentage = (count / report['total_puzzles']) * 100
            bar = "█" * int(percentage / 2)
            print(f"  {score} pts: {count:3d} ({percentage:4.1f}%) {bar}")
        
        print("\nTop Theme Performance:")
        theme_items = list(report['theme_performance'].items())
        theme_items.sort(key=lambda x: x[1]['avg_score'], reverse=True)
        
        for theme, data in theme_items[:10]:
            avg_score = data['avg_score']
            solution_rate = data['solution_rate']
            count = data['total']
            print(f"  {theme:15s}: {avg_score:.2f}/5.0 | Solutions: {solution_rate:4.1f}% | Count: {count}")
