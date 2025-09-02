# play_game.py
"""Play Game Module for V7P3R Chess AI
Handles the main game loop and player interactions between V7P3R Chess AI, Stockfish, and the chess core."""

import os
import sys
import chess
import time
import json
from chess_core import ChessConfig, GameState
from v7p3r_ai import V7P3RAI
from stockfish_handler import StockfishHandler


class ChessGameManager:
    """Manages chess games between different engines"""
    
    def __init__(self, config=None):
        self.config = config if config else ChessConfig()
        self.game_config = self.config.get_game_config()
        
        # Load players
        self.white_player_name = self.game_config.get("white_player", "v7p3r")
        self.black_player_name = self.game_config.get("black_player", "stockfish")
        
        # Initialize engines
        self.v7p3r = V7P3RAI(self.config)
        self.stockfish = StockfishHandler(self.config)
        
        # Game state
        self.game_state = None
        
        # Statistics
        self.stats = {
            "games_played": 0,
            "white_wins": 0,
            "black_wins": 0,
            "draws": 0,
            "v7p3r_wins": 0,
            "v7p3r_losses": 0,
            "v7p3r_draws": 0
        }
    
    def start_games(self, game_count=None):
        """Start playing the specified number of games"""
        if game_count is None:
            game_count = self.game_config.get("game_count", 1)
        
        print(f"Starting {game_count} games:")
        print(f"White: {self.white_player_name}, Black: {self.black_player_name}")
        
        for game_idx in range(game_count):
            print(f"\nGame {game_idx + 1}/{game_count}")
            result = self.play_game()
            
            # Update statistics
            self.stats["games_played"] += 1
            
            if result == "1-0":
                self.stats["white_wins"] += 1
                if self.white_player_name == "v7p3r":
                    self.stats["v7p3r_wins"] += 1
                else:
                    self.stats["v7p3r_losses"] += 1
                    
                print(f"Game {game_idx + 1} result: White wins")
                
            elif result == "0-1":
                self.stats["black_wins"] += 1
                if self.black_player_name == "v7p3r":
                    self.stats["v7p3r_wins"] += 1
                else:
                    self.stats["v7p3r_losses"] += 1
                    
                print(f"Game {game_idx + 1} result: Black wins")
                
            else:  # Draw
                self.stats["draws"] += 1
                self.stats["v7p3r_draws"] += 1
                print(f"Game {game_idx + 1} result: Draw")
        
        # Print final statistics
        self.print_stats()
    
    def play_game(self):
        """Play a single game"""
        # Initialize new game
        self.game_state = GameState(self.config)
        
        # Set player names in headers
        self.game_state.game.headers["White"] = self.white_player_name
        self.game_state.game.headers["Black"] = self.black_player_name
        
        # Game loop
        move_count = 0
        last_move_time = time.time()
        
        while not self.game_state.is_game_over() and move_count < 200:
            # Get current board state
            board = self.game_state.get_board_state()
            
            # Determine whose turn it is
            current_player = self.white_player_name if board.turn == chess.WHITE else self.black_player_name
            
            # Get move based on player
            move = None
            start_time = time.time()
            
            if current_player == "v7p3r":
                move = self.v7p3r.get_move(board)
            elif current_player == "stockfish":
                move = self.stockfish.get_move(board)
            else:
                print(f"Unknown player: {current_player}")
                break
            
            # Calculate thinking time
            thinking_time = time.time() - start_time
            
            # Make the move
            if move:
                self.game_state.make_move(move)
                move_count += 1
                
                # Print move info
                move_str = board.san(move)
                print(f"Move {move_count}: {current_player} plays {move_str} ({thinking_time:.2f}s)")
                
                # Enforce minimum move delay for viewing
                time_since_last_move = time.time() - last_move_time
                if time_since_last_move < 0.5:  # Minimum 0.5 second between moves for viewing
                    time.sleep(0.5 - time_since_last_move)
                
                last_move_time = time.time()
            else:
                print(f"No valid move found for {current_player}")
                break
        
        # Game ended
        result = self.game_state.get_result()
        
        # Save the final PGN
        if result:  # Only set result if not None
            self.game_state.game.headers["Result"] = result
        else:
            self.game_state.game.headers["Result"] = "*"  # Unknown result
            
        self.game_state._save_pgn("active_game.pgn")
        
        # Save to game history
        self._save_game_history(result)
        
        return result
    
    def print_stats(self):
        """Print game statistics"""
        print("\n===== Game Statistics =====")
        print(f"Games played: {self.stats['games_played']}")
        print(f"White wins: {self.stats['white_wins']}")
        print(f"Black wins: {self.stats['black_wins']}")
        print(f"Draws: {self.stats['draws']}")
        
        if self.stats['games_played'] > 0:
            v7p3r_win_rate = self.stats['v7p3r_wins'] / self.stats['games_played'] * 100
            print(f"\nV7P3R Statistics:")
            print(f"Wins: {self.stats['v7p3r_wins']} ({v7p3r_win_rate:.1f}%)")
            print(f"Losses: {self.stats['v7p3r_losses']}")
            print(f"Draws: {self.stats['v7p3r_draws']}")
    
    def _save_game_history(self, result):
        """Save game to history file"""
        if not self.game_state:
            return
            
        history_dir = "game_history"
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)
        
        # Generate filename based on date and time
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{history_dir}/game_{timestamp}.pgn"
        
        # Save PGN
        with open(filename, 'w') as f:
            print(self.game_state.game, file=f, end="\n\n")


def main():
    """Main function to run the game"""
    # Initialize config
    config = ChessConfig()
    
    # Create game manager
    manager = ChessGameManager(config)
    
    # Start games
    manager.start_games()


if __name__ == "__main__":
    main()