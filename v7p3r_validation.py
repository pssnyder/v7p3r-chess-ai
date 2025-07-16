# v7p3r_validation.py
"""Model Validation Module for V7P3R Chess AI
Handles validation of the V7P3R Chess AI models to ensure they meet performance and accuracy standards.
"""

import chess
import pickle
import os
import time
import json
import random
from datetime import datetime
from chess_core import ChessConfig, GameState, BoardEvaluator
from v7p3r_ai import V7P3RAI
from stockfish_handler import StockfishHandler


class V7P3RValidator:
    """Validator for V7P3R Chess AI models"""
    
    def __init__(self, config=None):
        self.config = config if config else ChessConfig()
        self.v7p3r_config = self.config.get_v7p3r_config()
        self.stockfish_config = self.config.get_stockfish_config()
        
        # Load V7P3R AI
        self.v7p3r = V7P3RAI(self.config)
        
        # Load Stockfish
        self.stockfish = StockfishHandler(self.config)
        
        # Validation results
        self.results_dir = "validation_results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
    
    def validate(self, num_games=10, elo_levels=None):
        """Validate V7P3R model against Stockfish at various levels
        
        Args:
            num_games: Number of games to play at each ELO level
            elo_levels: List of ELO levels to test against, defaults to [400, 800, 1200, 1600]
            
        Returns:
            Dictionary with validation results
        """
        print("Starting validation...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.results_dir, f"validation_{timestamp}.json")
        
        # Use default ELO levels if none provided
        if elo_levels is None:
            elo_levels = [400, 800, 1200, 1600]
        
        # Ensure at least one ELO level is tested
        if not elo_levels:
            elo_levels = [400]
        
        # Initialize overall results
        overall_results = {
            "games_played": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "win_rate": 0,
            "timestamp": timestamp,
            "model_path": self.v7p3r_config.get("model_path", ""),
            "by_elo": {}
        }
        
        for elo in elo_levels:
            print(f"Testing against Stockfish ELO {elo}...")
            
            # Set Stockfish ELO
            self.stockfish.set_elo(elo)
            
            # Play games
            white_wins = 0
            black_wins = 0
            draws = 0
            
            for game_idx in range(num_games):
                # Play game as white
                print(f"Game {game_idx+1}/{num_games*2} (V7P3R as White)")
                result = self._play_game(v7p3r_is_white=True)
                
                if result == "1-0":
                    white_wins += 1
                elif result == "0-1":
                    black_wins += 1
                else:
                    draws += 1
                
                # Play game as black
                print(f"Game {game_idx+1+num_games}/{num_games*2} (V7P3R as Black)")
                result = self._play_game(v7p3r_is_white=False)
                
                if result == "1-0":
                    white_wins += 1
                elif result == "0-1":
                    black_wins += 1
                else:
                    draws += 1
            
            # Calculate win rate
            v7p3r_wins = 0
            stockfish_wins = 0
            
            # When V7P3R plays as white, it wins when the result is "1-0"
            # When V7P3R plays as black, it wins when the result is "0-1"
            # We played num_games as white and num_games as black
            v7p3r_as_white_wins = white_wins
            v7p3r_as_black_wins = black_wins
            
            v7p3r_wins = v7p3r_as_white_wins + v7p3r_as_black_wins
            stockfish_wins = (num_games * 2) - v7p3r_wins - draws
            
            win_rate = v7p3r_wins / (num_games * 2)
            
            # Store results for this ELO level
            elo_results = {
                "games_played": num_games * 2,
                "v7p3r_wins": v7p3r_wins,
                "stockfish_wins": stockfish_wins,
                "draws": draws,
                "win_rate": win_rate
            }
            
            # Add to overall results
            overall_results["by_elo"][str(elo)] = elo_results
            overall_results["games_played"] += num_games * 2
            overall_results["wins"] += v7p3r_wins
            overall_results["draws"] += draws
            overall_results["losses"] += stockfish_wins
            
            print(f"Results against Stockfish ELO {elo}:")
            print(f"Win rate: {win_rate:.2%}")
            print(f"V7P3R wins: {v7p3r_wins}, Stockfish wins: {stockfish_wins}, Draws: {draws}")
        
        # Calculate overall win rate
        if overall_results["games_played"] > 0:
            overall_results["win_rate"] = overall_results["wins"] / overall_results["games_played"]
        
        # Save results
        with open(results_file, 'w') as f:
            json.dump(overall_results, f, indent=4)
        
        print(f"Validation results saved to {results_file}")
        print(f"Overall win rate: {overall_results['win_rate']:.2%}")
        
        return overall_results
        
        for elo in elo_levels:
            print(f"Testing against Stockfish ELO {elo}...")
            
            # Set Stockfish ELO
            self.stockfish.set_elo(elo)
            
            # Play games
            white_wins = 0
            black_wins = 0
            draws = 0
            
            for game_idx in range(num_games):
                # Play game as white
                print(f"Game {game_idx+1}/{num_games*2} (V7P3R as White)")
                result = self._play_game(v7p3r_is_white=True)
                
                if result == "1-0":
                    white_wins += 1
                elif result == "0-1":
                    black_wins += 1
                else:
                    draws += 1
                
                # Play game as black
                print(f"Game {game_idx+1+num_games}/{num_games*2} (V7P3R as Black)")
                result = self._play_game(v7p3r_is_white=False)
                
                if result == "1-0":
                    white_wins += 1
                elif result == "0-1":
                    black_wins += 1
                else:
                    draws += 1
            
            # Calculate win rate
            v7p3r_wins = 0
            stockfish_wins = 0
            
            # When V7P3R plays as white, it wins when the result is "1-0"
            # When V7P3R plays as black, it wins when the result is "0-1"
            # We played num_games as white and num_games as black
            v7p3r_as_white_wins = white_wins
            v7p3r_as_black_wins = black_wins
            
            v7p3r_wins = v7p3r_as_white_wins + v7p3r_as_black_wins
            stockfish_wins = (num_games * 2) - v7p3r_wins - draws
            
            win_rate = v7p3r_wins / (num_games * 2)
            
            # Store results
            results[str(elo)] = {
                "games_played": num_games * 2,
                "v7p3r_wins": v7p3r_wins,
                "stockfish_wins": stockfish_wins,
                "draws": draws,
                "win_rate": win_rate
            }
            
            print(f"Results against Stockfish ELO {elo}:")
            print(f"Win rate: {win_rate:.2%}")
            print(f"V7P3R wins: {v7p3r_wins}, Stockfish wins: {stockfish_wins}, Draws: {draws}")
        
        # Save results
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Validation results saved to {results_file}")
        
        return results
    
    def _play_game(self, v7p3r_is_white=True):
        """Play a game between V7P3R and Stockfish"""
        game_state = GameState(self.config)
        
        # Set player names in headers
        if v7p3r_is_white:
            game_state.game.headers["White"] = "V7P3R"
            game_state.game.headers["Black"] = "Stockfish"
        else:
            game_state.game.headers["White"] = "Stockfish"
            game_state.game.headers["Black"] = "V7P3R"
        
        # Game loop
        move_count = 0
        while not game_state.is_game_over() and move_count < 200:
            board = game_state.get_board_state()
            
            # Determine who's turn it is
            is_v7p3r_turn = (board.turn == chess.WHITE and v7p3r_is_white) or \
                           (board.turn == chess.BLACK and not v7p3r_is_white)
            
            if is_v7p3r_turn:
                # V7P3R's turn
                move = self.v7p3r.get_move(board)
            else:
                # Stockfish's turn
                move = self.stockfish.get_move(board)
            
            # Make move
            if move:
                game_state.make_move(move)
            else:
                # No legal moves
                break
            
            move_count += 1
        
        # Get result
        result = game_state.get_result()
        
        return result
    
    def analyze_opening_play(self, num_openings=5):
        """Analyze V7P3R's play in common openings"""
        print("Analyzing opening play...")
        
        # Common opening sequences (first 4 moves in algebraic notation)
        common_openings = {
            "Ruy Lopez": ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"],
            "Italian Game": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"],
            "Sicilian Defense": ["e2e4", "c7c5"],
            "French Defense": ["e2e4", "e7e6", "d2d4", "d7d5"],
            "Queen's Gambit": ["d2d4", "d7d5", "c2c4"]
        }
        
        results = {}
        
        # Test each opening
        for opening_name, moves in list(common_openings.items())[:num_openings]:
            print(f"Testing {opening_name}...")
            
            # Initialize game
            game_state = GameState(self.config)
            
            # Play opening moves
            for move_uci in moves:
                move = chess.Move.from_uci(move_uci)
                if move in game_state.get_board_state().legal_moves:
                    game_state.make_move(move)
                else:
                    print(f"Invalid move in opening: {move_uci}")
                    break
            
            # Let V7P3R make the next move
            board = game_state.get_board_state()
            v7p3r_move = self.v7p3r.get_move(board)
            
            if v7p3r_move:
                # Check if the move is in a list of good responses
                # (This would require a database of good responses to openings)
                quality_assessment = "Unknown"  # Replace with actual assessment
                
                results[opening_name] = {
                    "v7p3r_move": v7p3r_move.uci(),
                    "quality": quality_assessment
                }
            else:
                results[opening_name] = {
                    "v7p3r_move": None,
                    "quality": "No move found"
                }
        
        return results
    
    def benchmark_decision_time(self, num_positions=10):
        """Benchmark the time it takes for V7P3R to make decisions"""
        print("Benchmarking decision time...")
        
        # Generate or load test positions
        positions = self._get_test_positions(num_positions)
        
        times = []
        
        for i, fen in enumerate(positions):
            board = chess.Board(fen)
            
            # Measure time to make a decision
            start_time = time.time()
            move = self.v7p3r.get_move(board)
            end_time = time.time()
            
            decision_time = end_time - start_time
            times.append(decision_time)
            
            print(f"Position {i+1}: {decision_time:.4f} seconds")
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)
        
        print(f"Average decision time: {avg_time:.4f} seconds")
        print(f"Max decision time: {max_time:.4f} seconds")
        print(f"Min decision time: {min_time:.4f} seconds")
        
        return {
            "average_time": avg_time,
            "max_time": max_time,
            "min_time": min_time,
            "position_times": times
        }
    
    def _get_test_positions(self, num_positions):
        """Get test positions for benchmarking"""
        # You could load these from a file or generate them
        # For now, we'll use a few standard positions plus some random ones
        
        standard_positions = [
            # Starting position
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            
            # Middle game positions
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "r1bqk2r/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w kq - 0 7",
            
            # Endgame positions
            "8/8/8/8/3k4/8/3K4/8 w - - 0 1",
            "8/8/8/8/3k4/8/3KP3/8 w - - 0 1"
        ]
        
        # Generate additional random positions by playing random games
        additional_positions = []
        for _ in range(max(0, num_positions - len(standard_positions))):
            game_state = GameState(self.config)
            
            # Make random moves
            for _ in range(random.randint(10, 40)):
                legal_moves = list(game_state.get_legal_moves())
                if legal_moves:
                    move = random.choice(legal_moves)
                    game_state.make_move(move)
                else:
                    break
            
            additional_positions.append(game_state.get_fen())
        
        return standard_positions + additional_positions


if __name__ == "__main__":
    # Initialize config
    config = ChessConfig()
    
    # Create and run validator
    validator = V7P3RValidator(config)
    
    # Run validation
    validator.validate(num_games=5)
    
    # Analyze openings
    validator.analyze_opening_play()
    
    # Benchmark decision time
    validator.benchmark_decision_time()