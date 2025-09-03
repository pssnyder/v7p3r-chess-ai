# personal_style_analyzer.py
"""Personal Style Analyzer for V7P3R Chess AI
Analyzes the player's historical games to extract playing style, patterns, and preferences.
Used to give the AI a more personalized and human-like playing style.
"""

import chess
import chess.pgn
import os
import json
from collections import defaultdict
import numpy as np


class PersonalStyleAnalyzer:
    """Analyzes personal chess games to extract playing style and patterns"""
    
    def __init__(self, config=None):
        from chess_core import ChessConfig
        self.config = config if config else ChessConfig()
        self.v7p3r_config = self.config.get_v7p3r_config()
        
        # Personal style settings
        self.personal_pgn_path = self.v7p3r_config.get("personal_pgn_path", "data/v7p3r_games.pgn")
        self.style_weights = self.v7p3r_config.get("personal_style_weights", {})
        
        # Data structures to store analysis results
        self.opening_moves = defaultdict(list)  # FEN -> list of moves with statistics
        self.middlegame_patterns = defaultdict(list)
        self.endgame_patterns = defaultdict(list)
        self.checkmate_patterns = []
        self.winning_sequences = []
        self.position_evaluations = {}
        
        # Analysis results path
        self.analysis_path = "data/personal_style_analysis.json"
        
        # Load analysis if available, otherwise analyze games
        if os.path.exists(self.analysis_path):
            self._load_analysis()
        elif os.path.exists(self.personal_pgn_path):
            self.analyze_games()
    
    def _load_analysis(self):
        """Load pre-computed analysis from file"""
        try:
            with open(self.analysis_path, 'r') as f:
                analysis = json.load(f)
                
                # Convert from JSON format back to our data structures
                self.opening_moves = defaultdict(list, analysis.get("opening_moves", {}))
                self.middlegame_patterns = defaultdict(list, analysis.get("middlegame_patterns", {}))
                self.endgame_patterns = defaultdict(list, analysis.get("endgame_patterns", {}))
                self.checkmate_patterns = analysis.get("checkmate_patterns", [])
                self.winning_sequences = analysis.get("winning_sequences", [])
                self.position_evaluations = analysis.get("position_evaluations", {})
                
                print(f"Loaded personal style analysis from {self.analysis_path}")
                print(f"Analyzed {len(self.opening_moves)} opening positions")
                print(f"Found {len(self.checkmate_patterns)} checkmate patterns")
        except Exception as e:
            print(f"Error loading analysis: {e}")
            self.analyze_games()
    
    def _save_analysis(self):
        """Save analysis to file"""
        try:
            # Convert defaultdicts to regular dicts for JSON serialization
            analysis = {
                "opening_moves": dict(self.opening_moves),
                "middlegame_patterns": dict(self.middlegame_patterns),
                "endgame_patterns": dict(self.endgame_patterns),
                "checkmate_patterns": self.checkmate_patterns,
                "winning_sequences": self.winning_sequences,
                "position_evaluations": self.position_evaluations
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.analysis_path), exist_ok=True)
            
            with open(self.analysis_path, 'w') as f:
                json.dump(analysis, f, indent=2)
                
            print(f"Saved personal style analysis to {self.analysis_path}")
        except Exception as e:
            print(f"Error saving analysis: {e}")
    
    def analyze_games(self):
        """Analyze all games in the personal PGN file"""
        if not os.path.exists(self.personal_pgn_path):
            print(f"Personal PGN file not found: {self.personal_pgn_path}")
            return
        
        print(f"Analyzing personal games from {self.personal_pgn_path}...")
        
        game_count = 0
        checkmate_count = 0
        win_count = 0
        
        try:
            # Ensure data directory exists
            os.makedirs("data", exist_ok=True)
            
            # Reset analysis structures
            self.opening_moves = defaultdict(list)
            self.middlegame_patterns = defaultdict(list)
            self.endgame_patterns = defaultdict(list)
            self.checkmate_patterns = []
            self.winning_sequences = []
            self.position_evaluations = {}
            with open(self.personal_pgn_path) as pgn_file:
                while True:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break
                    
                    game_count += 1
                    if game_count % 10 == 0:
                        print(f"Analyzed {game_count} games...")
                    
                    # Get player color
                    player_is_white = self._is_player_white(game)
                    
                    # Extract result
                    result = game.headers.get("Result", "*")
                    player_won = (result == "1-0" and player_is_white) or (result == "0-1" and not player_is_white)
                    
                    # Check if game ended in checkmate
                    termination = game.headers.get("Termination", "")
                    ended_in_checkmate = "mate" in termination.lower() or "checkmate" in termination.lower()
                    
                    if player_won:
                        win_count += 1
                        
                    if ended_in_checkmate and player_won:
                        checkmate_count += 1
                    
                    # Analyze game
                    self._analyze_game(game, player_is_white, player_won, ended_in_checkmate)
            
            print(f"Analysis complete. Processed {game_count} games.")
            print(f"Found {win_count} wins, {checkmate_count} checkmate wins.")
            print(f"Extracted {len(self.opening_moves)} unique opening positions.")
            print(f"Identified {len(self.checkmate_patterns)} checkmate patterns.")
            
            # Save analysis
            self._save_analysis()
            
        except Exception as e:
            print(f"Error analyzing games: {e}")
    
    def _is_player_white(self, game):
        """Determine if the player was white in this game"""
        # This assumes your name/handle is consistent in the PGN files
        # Adjust the pattern matching logic as needed
        white_player = game.headers.get("White", "").lower()
        return "v7p3r" in white_player or "snyder" in white_player
    
    def _analyze_game(self, game, player_is_white, player_won, ended_in_checkmate):
        """Analyze a single game"""
        # Initialize board
        board = game.board()
        
        # Track moves and positions
        move_number = 0
        positions = []
        
        # Analyze each move
        node = game
        while node.variations:
            move_number += 1
            next_node = node.variation(0)
            move = next_node.move
            
            # Get position FEN (without move counters)
            fen_parts = board.fen().split(" ")
            position_fen = " ".join(fen_parts[0:4])
            
            # Store position
            positions.append({
                "fen": position_fen,
                "move": move.uci(),
                "move_number": move_number,
                "player_move": (board.turn == chess.WHITE) == player_is_white
            })
            
            # Check if this is an opening move (first 10 moves)
            if move_number <= 10:
                self._analyze_opening_move(board, move, player_is_white, player_won)
            
            # Check if this is a middlegame move
            elif 11 <= move_number <= 30:
                self._analyze_middlegame_move(board, move, player_is_white, player_won)
            
            # Check if this is an endgame move
            else:
                self._analyze_endgame_move(board, move, player_is_white, player_won)
            
            # Make the move on the board
            board.push(move)
            
            # Update node
            node = next_node
        
        # If game ended in checkmate by the player, analyze the final sequence
        if ended_in_checkmate and player_won:
            self._analyze_checkmate_sequence(positions)
        
        # If player won, analyze winning sequences
        if player_won:
            self._analyze_winning_sequence(positions)
    
    def _analyze_opening_move(self, board, move, player_is_white, player_won):
        """Analyze an opening move"""
        # Only analyze the player's moves
        if (board.turn == chess.WHITE) != player_is_white:
            return
            
        # Get position FEN (without move counters)
        fen_parts = board.fen().split(" ")
        position_fen = " ".join(fen_parts[0:4])
        
        # Add move to opening database
        move_uci = move.uci()
        
        # Check if move exists in database
        move_exists = False
        for entry in self.opening_moves[position_fen]:
            if entry["move"] == move_uci:
                # Update statistics
                entry["count"] += 1
                if player_won:
                    entry["wins"] += 1
                
                move_exists = True
                break
        
        # Add new move if not found
        if not move_exists:
            self.opening_moves[position_fen].append({
                "move": move_uci,
                "count": 1,
                "wins": 1 if player_won else 0
            })
    
    def _analyze_middlegame_move(self, board, move, player_is_white, player_won):
        """Analyze a middlegame move"""
        # Similar to opening, but for middlegame
        if (board.turn == chess.WHITE) != player_is_white:
            return
            
        # In middlegame, we might want to focus on tactical patterns
        # This is a simplified version - you could add more sophisticated pattern recognition
        
        # Capture moves
        is_capture = board.is_capture(move)
        
        # Check moves
        board.push(move)
        gives_check = board.is_check()
        board.pop()
        
        # Get position FEN
        fen_parts = board.fen().split(" ")
        position_fen = " ".join(fen_parts[0:4])
        
        # Store pattern
        pattern_type = "normal"
        if is_capture and gives_check:
            pattern_type = "capture_check"
        elif is_capture:
            pattern_type = "capture"
        elif gives_check:
            pattern_type = "check"
        
        # Add to database
        move_uci = move.uci()
        pattern_key = f"{position_fen}_{pattern_type}"
        
        # Check if pattern exists
        pattern_exists = False
        for entry in self.middlegame_patterns[pattern_key]:
            if entry["move"] == move_uci:
                entry["count"] += 1
                if player_won:
                    entry["wins"] += 1
                
                pattern_exists = True
                break
        
        # Add new pattern if not found
        if not pattern_exists:
            self.middlegame_patterns[pattern_key].append({
                "move": move_uci,
                "pattern_type": pattern_type,
                "count": 1,
                "wins": 1 if player_won else 0
            })
    
    def _analyze_endgame_move(self, board, move, player_is_white, player_won):
        """Analyze an endgame move"""
        # Similar to middlegame, but with endgame-specific patterns
        if (board.turn == chess.WHITE) != player_is_white:
            return
        
        # In endgame, pawn advancement and king activity are often key
        move_uci = move.uci()
        piece = board.piece_at(move.from_square)
        
        # Classify move type
        move_type = "normal"
        
        if piece and piece.piece_type == chess.PAWN:
            # Check if it's a pawn push to 7th/2nd rank
            to_rank = chess.square_rank(move.to_square)
            if (piece.color == chess.WHITE and to_rank == 6) or \
               (piece.color == chess.BLACK and to_rank == 1):
                move_type = "pawn_advance"
        
        elif piece and piece.piece_type == chess.KING:
            move_type = "king_move"
        
        # Check if it's a promotion
        if move.promotion:
            move_type = "promotion"
        
        # Get position FEN
        fen_parts = board.fen().split(" ")
        position_fen = " ".join(fen_parts[0:4])
        
        # Add to database
        pattern_key = f"{position_fen}_{move_type}"
        
        # Check if pattern exists
        pattern_exists = False
        for entry in self.endgame_patterns[pattern_key]:
            if entry["move"] == move_uci:
                entry["count"] += 1
                if player_won:
                    entry["wins"] += 1
                
                pattern_exists = True
                break
        
        # Add new pattern if not found
        if not pattern_exists:
            self.endgame_patterns[pattern_key].append({
                "move": move_uci,
                "move_type": move_type,
                "count": 1,
                "wins": 1 if player_won else 0
            })
    
    def _analyze_checkmate_sequence(self, positions):
        """Analyze a sequence leading to checkmate"""
        # Extract the last few moves before checkmate (player's moves only)
        player_positions = [p for p in positions if p["player_move"]]
        checkmate_sequence = player_positions[-5:] if len(player_positions) >= 5 else player_positions
        
        # Add to checkmate patterns
        self.checkmate_patterns.append({
            "sequence": [pos["move"] for pos in checkmate_sequence],
            "positions": [pos["fen"] for pos in checkmate_sequence]
        })
    
    def _analyze_winning_sequence(self, positions):
        """Analyze a sequence leading to victory"""
        # Extract key positions from the winning game (player's moves only)
        player_positions = [p for p in positions if p["player_move"]]
        
        # For simplicity, we'll take positions from the last third of the game
        start_idx = max(0, int(len(player_positions) * 2/3))
        winning_sequence = player_positions[start_idx:]
        
        # Add to winning sequences
        self.winning_sequences.append({
            "sequence": [pos["move"] for pos in winning_sequence],
            "positions": [pos["fen"] for pos in winning_sequence]
        })
    
    def get_personal_move(self, board, phase="opening"):
        """Get a move from the player's historical games for the current position"""
        # Get position FEN (without move counters)
        fen_parts = board.fen().split(" ")
        position_fen = " ".join(fen_parts[0:4])
        
        if phase == "opening":
            # Check if position exists in opening database
            if position_fen in self.opening_moves:
                moves = self.opening_moves[position_fen]
                if moves:
                    # Sort by win rate and count
                    sorted_moves = sorted(moves, key=lambda x: (x["wins"] / max(x["count"], 1), x["count"]), reverse=True)
                    
                    # Apply influence weight
                    influence = self.style_weights.get("opening_style_influence", 0.8)
                    
                    # Return best move if it passes the influence threshold
                    if random.random() < influence:
                        best_move = sorted_moves[0]["move"]
                        return chess.Move.from_uci(best_move)
        
        elif phase == "middlegame":
            # Try different pattern types
            for pattern_type in ["capture_check", "check", "capture", "normal"]:
                pattern_key = f"{position_fen}_{pattern_type}"
                if pattern_key in self.middlegame_patterns:
                    moves = self.middlegame_patterns[pattern_key]
                    if moves:
                        sorted_moves = sorted(moves, key=lambda x: (x["wins"] / max(x["count"], 1), x["count"]), reverse=True)
                        
                        influence = self.style_weights.get("middlegame_style_influence", 0.6)
                        if random.random() < influence:
                            best_move = sorted_moves[0]["move"]
                            return chess.Move.from_uci(best_move)
        
        elif phase == "endgame":
            # Try different move types
            for move_type in ["promotion", "pawn_advance", "king_move", "normal"]:
                pattern_key = f"{position_fen}_{move_type}"
                if pattern_key in self.endgame_patterns:
                    moves = self.endgame_patterns[pattern_key]
                    if moves:
                        sorted_moves = sorted(moves, key=lambda x: (x["wins"] / max(x["count"], 1), x["count"]), reverse=True)
                        
                        influence = self.style_weights.get("endgame_style_influence", 0.7)
                        if random.random() < influence:
                            best_move = sorted_moves[0]["move"]
                            return chess.Move.from_uci(best_move)
        
        # If no move found or influence check failed, return None
        return None
    
    def detect_game_phase(self, board):
        """Detect the current game phase"""
        # Simple heuristic based on piece count and move number
        piece_count = sum(1 for _ in board.piece_map())
        
        if piece_count >= 26 or board.fullmove_number <= 10:
            return "opening"
        elif piece_count >= 10:
            return "middlegame"
        else:
            return "endgame"
    
    def check_for_checkmate_pattern(self, board):
        """Check if the current position matches a known checkmate pattern"""
        # This would need a more sophisticated pattern matching algorithm
        # For simplicity, we'll just check if we're a few moves away from a known checkmate position
        
        # Get position FEN
        fen_parts = board.fen().split(" ")
        position_fen = " ".join(fen_parts[0:4])
        
        # Check each checkmate pattern
        for pattern in self.checkmate_patterns:
            if position_fen in pattern["positions"]:
                # Find the move that follows this position in the pattern
                idx = pattern["positions"].index(position_fen)
                if idx < len(pattern["sequence"]) - 1:
                    next_move = pattern["sequence"][idx + 1]
                    # Verify the move is legal
                    try:
                        move = chess.Move.from_uci(next_move)
                        if move in board.legal_moves:
                            return move
                    except ValueError:
                        continue
        
        return None
    
    def check_for_winning_sequence(self, board):
        """Check if the current position matches a known winning sequence"""
        # Get position FEN
        fen_parts = board.fen().split(" ")
        position_fen = " ".join(fen_parts[0:4])
        
        # Check each winning sequence
        for sequence in self.winning_sequences:
            # Check for both formats (new and legacy)
            if "position" in sequence:
                pos = sequence["position"]
            elif "positions" in sequence and len(sequence["positions"]) > 0:
                pos = sequence["positions"][0]
            else:
                continue
                
            if self._position_match(board, pos):
                # Check if the next move in the sequence is legal
                moves_key = "moves" if "moves" in sequence else "sequence"
                if moves_key in sequence and len(sequence[moves_key]) > 0:
                    try:
                        move = chess.Move.from_uci(sequence[moves_key][0])
                        if move in board.legal_moves:
                            return move
                    except ValueError:
                        continue
        
        return None
    
    def _position_match(self, board, pattern_position):
        """Check if the current position matches a pattern position with some flexibility"""
        # For simplicity, we'll just check if key pieces are in the same place
        # A more sophisticated implementation would use a similarity metric
        
        try:
            pattern_board = chess.Board(pattern_position)
            
            # Check key pieces (kings, queens, rooks)
            key_pieces = [chess.KING, chess.QUEEN, chess.ROOK]
            
            for piece_type in key_pieces:
                # Check white pieces
                pattern_squares = pattern_board.pieces(piece_type, chess.WHITE)
                current_squares = board.pieces(piece_type, chess.WHITE)
                
                if pattern_squares != current_squares:
                    return False
                
                # Check black pieces
                pattern_squares = pattern_board.pieces(piece_type, chess.BLACK)
                current_squares = board.pieces(piece_type, chess.BLACK)
                
                if pattern_squares != current_squares:
                    return False
            
            return True
        except ValueError:
            return False


# Add ability to import random
import random
