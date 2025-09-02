# analyze_personal_games.py
"""Analyze Personal Game History for V7P3R Chess AI
Analyzes personal game history to extract insights about playing style, patterns,
and preferences to inform the AI's behavior.
"""

import os
import sys
import chess
import chess.pgn
import json
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np
from personal_style_analyzer import PersonalStyleAnalyzer
from chess_core import ChessConfig


class PersonalGameAnalyzer:
    """Analyze personal chess games to extract insights"""
    
    def __init__(self, pgn_path="data/v7p3r_games.pgn"):
        self.pgn_path = pgn_path
        self.stats = {
            "total_games": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "win_as_white": 0,
            "win_as_black": 0,
            "checkmate_wins": 0,
            "resigned_wins": 0,
            "time_wins": 0,
            "average_moves": 0,
            "opening_stats": defaultdict(int),
            "move_patterns": defaultdict(int),
            "piece_activity": defaultdict(int),
            "capture_stats": defaultdict(int)
        }
        
        # Load personal style analyzer
        self.config = ChessConfig()
        self.personal_style = PersonalStyleAnalyzer(self.config)
        
        # Make sure data directory exists
        os.makedirs("data/analysis", exist_ok=True)
    
    def analyze(self):
        """Run comprehensive analysis on personal games"""
        print(f"Analyzing games from {self.pgn_path}...")
        
        # Check if file exists
        if not os.path.exists(self.pgn_path):
            print(f"Error: PGN file not found at {self.pgn_path}")
            return
        
        total_moves = 0
        move_counts = []
        
        # Open PGN file
        with open(self.pgn_path) as pgn_file:
            game_count = 0
            
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                
                game_count += 1
                self.stats["total_games"] += 1
                
                # Analyze game
                result = game.headers.get("Result", "*")
                white_player = game.headers.get("White", "").lower()
                black_player = game.headers.get("Black", "").lower()
                
                # Determine if player is white or black
                player_is_white = "v7p3r" in white_player or "snyder" in white_player
                player_is_black = "v7p3r" in black_player or "snyder" in black_player
                
                # Record game result
                if result == "1-0" and player_is_white:
                    self.stats["wins"] += 1
                    self.stats["win_as_white"] += 1
                elif result == "0-1" and player_is_black:
                    self.stats["wins"] += 1
                    self.stats["win_as_black"] += 1
                elif result == "1/2-1/2":
                    self.stats["draws"] += 1
                elif result == "1-0" and player_is_black:
                    self.stats["losses"] += 1
                elif result == "0-1" and player_is_white:
                    self.stats["losses"] += 1
                
                # Check termination type
                termination = game.headers.get("Termination", "").lower()
                if "mate" in termination or "checkmate" in termination:
                    if (result == "1-0" and player_is_white) or (result == "0-1" and player_is_black):
                        self.stats["checkmate_wins"] += 1
                elif "resign" in termination:
                    if (result == "1-0" and player_is_white) or (result == "0-1" and player_is_black):
                        self.stats["resigned_wins"] += 1
                elif "time" in termination:
                    if (result == "1-0" and player_is_white) or (result == "0-1" and player_is_black):
                        self.stats["time_wins"] += 1
                
                # Count moves
                move_count = self._count_game_moves(game)
                move_counts.append(move_count)
                total_moves += move_count
                
                # Record opening
                opening = game.headers.get("Opening", "Unknown")
                self.stats["opening_stats"][opening] += 1
                
                # Analyze individual moves
                self._analyze_game_moves(game, player_is_white, player_is_black)
                
                if game_count % 10 == 0:
                    print(f"Analyzed {game_count} games...")
        
        # Calculate average moves per game
        if self.stats["total_games"] > 0:
            self.stats["average_moves"] = total_moves / self.stats["total_games"]
        
        # Generate visualizations
        self._generate_visualizations(move_counts)
        
        # Save analysis results
        self._save_results()
        
        print(f"Analysis complete! Processed {self.stats['total_games']} games.")
        self._print_summary()
    
    def _count_game_moves(self, game):
        """Count the number of moves in a game"""
        move_count = 0
        node = game
        while node.variations:
            move_count += 1
            node = node.variation(0)
        return move_count
    
    def _analyze_game_moves(self, game, player_is_white, player_is_black):
        """Analyze individual moves in a game"""
        board = game.board()
        node = game
        
        while node.variations:
            next_node = node.variation(0)
            move = next_node.move
            
            # Only analyze player's moves
            player_move = (board.turn == chess.WHITE and player_is_white) or \
                          (board.turn == chess.BLACK and player_is_black)
            
            if player_move:
                # Piece movement stats
                piece = board.piece_at(move.from_square)
                if piece:
                    piece_name = chess.piece_name(piece.piece_type)
                    self.stats["piece_activity"][piece_name] += 1
                
                # Capture stats
                if board.is_capture(move):
                    captured_piece = board.piece_at(move.to_square)
                    if captured_piece:
                        captured_name = chess.piece_name(captured_piece.piece_type)
                        self.stats["capture_stats"][f"captured_{captured_name}"] += 1
                        
                        if piece:
                            exchange = f"{piece_name}_takes_{captured_name}"
                            self.stats["capture_stats"][exchange] += 1
                
                # Check stats
                board.push(move)
                if board.is_check():
                    self.stats["move_patterns"]["gives_check"] += 1
                board.pop()
                
                # Special moves
                if board.is_castling(move):
                    if move.to_square == chess.G1 or move.to_square == chess.G8:
                        self.stats["move_patterns"]["kingside_castle"] += 1
                    else:
                        self.stats["move_patterns"]["queenside_castle"] += 1
                
                if move.promotion:
                    promotion_piece = chess.piece_name(move.promotion)
                    self.stats["move_patterns"][f"promotion_to_{promotion_piece}"] += 1
                
                if board.is_en_passant(move):
                    self.stats["move_patterns"]["en_passant"] += 1
            
            # Make the move
            board.push(move)
            node = next_node
    
    def _generate_visualizations(self, move_counts):
        """Generate visualizations from analysis data"""
        try:
            # Set up plot style
            plt.style.use('ggplot')
            
            # Game length distribution
            plt.figure(figsize=(10, 6))
            plt.hist(move_counts, bins=20, alpha=0.7, color='blue')
            plt.title('Distribution of Game Lengths')
            plt.xlabel('Number of Moves')
            plt.ylabel('Number of Games')
            plt.savefig('data/analysis/game_length_distribution.png')
            
            # Win-loss pie chart
            plt.figure(figsize=(8, 8))
            labels = ['Wins', 'Draws', 'Losses']
            sizes = [self.stats["wins"], self.stats["draws"], self.stats["losses"]]
            colors = ['green', 'gray', 'red']
            explode = (0.1, 0, 0)  # explode the 1st slice (Wins)
            plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True)
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            plt.title('Game Results')
            plt.savefig('data/analysis/game_results_pie.png')
            
            # Piece activity bar chart
            plt.figure(figsize=(10, 6))
            pieces = list(self.stats["piece_activity"].keys())
            activity = [self.stats["piece_activity"][p] for p in pieces]
            plt.bar(pieces, activity, color='purple')
            plt.title('Piece Activity')
            plt.xlabel('Piece Type')
            plt.ylabel('Number of Moves')
            plt.savefig('data/analysis/piece_activity.png')
            
            # Win types bar chart
            plt.figure(figsize=(10, 6))
            win_types = ['Checkmate', 'Resignation', 'Time']
            win_counts = [self.stats["checkmate_wins"], self.stats["resigned_wins"], self.stats["time_wins"]]
            plt.bar(win_types, win_counts, color='green')
            plt.title('Types of Wins')
            plt.xlabel('Win Type')
            plt.ylabel('Number of Games')
            plt.savefig('data/analysis/win_types.png')
            
            # Top openings bar chart (limited to top 10)
            plt.figure(figsize=(12, 8))
            openings = list(self.stats["opening_stats"].keys())
            counts = [self.stats["opening_stats"][o] for o in openings]
            
            # Sort and limit to top 10
            sorted_indices = np.argsort(counts)[::-1][:10]
            top_openings = [openings[i] for i in sorted_indices]
            top_counts = [counts[i] for i in sorted_indices]
            
            plt.barh(top_openings, top_counts, color='blue')
            plt.title('Top 10 Openings')
            plt.xlabel('Number of Games')
            plt.tight_layout()
            plt.savefig('data/analysis/top_openings.png')
            
            print("Visualizations generated and saved to data/analysis/ directory")
            
        except Exception as e:
            print(f"Error generating visualizations: {e}")
    
    def _save_results(self):
        """Save analysis results to JSON file"""
        # Convert defaultdicts to regular dicts for JSON serialization
        serializable_stats = dict(self.stats)
        serializable_stats["opening_stats"] = dict(serializable_stats["opening_stats"])
        serializable_stats["move_patterns"] = dict(serializable_stats["move_patterns"])
        serializable_stats["piece_activity"] = dict(serializable_stats["piece_activity"])
        serializable_stats["capture_stats"] = dict(serializable_stats["capture_stats"])
        
        with open('data/analysis/personal_game_stats.json', 'w') as f:
            json.dump(serializable_stats, f, indent=4)
        
        print("Analysis results saved to data/analysis/personal_game_stats.json")
    
    def _print_summary(self):
        """Print a summary of the analysis results"""
        win_rate = self.stats["wins"] / self.stats["total_games"] * 100 if self.stats["total_games"] > 0 else 0
        
        print("\n===== Personal Game Analysis Summary =====")
        print(f"Total Games: {self.stats['total_games']}")
        print(f"Wins: {self.stats['wins']} ({win_rate:.1f}%)")
        print(f"Draws: {self.stats['draws']}")
        print(f"Losses: {self.stats['losses']}")
        print(f"Win as White: {self.stats['win_as_white']}")
        print(f"Win as Black: {self.stats['win_as_black']}")
        print(f"Average Moves per Game: {self.stats['average_moves']:.1f}")
        print(f"Checkmate Wins: {self.stats['checkmate_wins']}")
        
        print("\nTop 5 Openings:")
        sorted_openings = sorted(self.stats["opening_stats"].items(), key=lambda x: x[1], reverse=True)
        for opening, count in sorted_openings[:5]:
            print(f"  - {opening}: {count} games")
        
        print("\nPiece Activity:")
        for piece, count in self.stats["piece_activity"].items():
            print(f"  - {piece.capitalize()}: {count} moves")
        
        print("\nSpecial Move Patterns:")
        patterns = self.stats["move_patterns"]
        print(f"  - Giving Check: {patterns.get('gives_check', 0)}")
        print(f"  - Kingside Castle: {patterns.get('kingside_castle', 0)}")
        print(f"  - Queenside Castle: {patterns.get('queenside_castle', 0)}")
        print(f"  - En Passant: {patterns.get('en_passant', 0)}")
        
        print("\nThe personal style analyzer detected:")
        print(f"  - {len(self.personal_style.opening_moves)} unique opening positions")
        print(f"  - {len(self.personal_style.checkmate_patterns)} checkmate patterns")
        print(f"  - {len(self.personal_style.winning_sequences)} winning sequences")
        print("\nMore detailed statistics available in data/analysis/personal_game_stats.json")
        print("Visualizations available in data/analysis/ directory")


if __name__ == "__main__":
    pgn_path = "data/v7p3r_games.pgn"
    if len(sys.argv) > 1:
        pgn_path = sys.argv[1]
    
    analyzer = PersonalGameAnalyzer(pgn_path)
    analyzer.analyze()
