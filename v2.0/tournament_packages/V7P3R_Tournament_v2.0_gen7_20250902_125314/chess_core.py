# chess_core.py
"""Core Chess Module for V7P3R Chess AI
Handles all python chess logic and game state management.
"""

import chess
import chess.pgn
import json
import os
import numpy as np
from datetime import datetime


class ChessConfig:
    """Configuration handler for the chess engine"""
    
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self):
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return self._get_default_config()
            
    def _get_default_config(self):
        """Return default configuration if file cannot be loaded"""
        return {
            "game_config": {
                "game_count": 1,
                "white_player": "v7p3r",
                "black_player": "stockfish"
            },
            "v7p3r_config": {
                "engine_id": "v7p3r_chess_ai_dev",
                "name": "v7p3r_ai",
                "version": "0.0.1",
                "elo_rating": 0
            },
            "stockfish_config": {
                "engine_id": "stockfish_python_release",
                "name": "stockfish",
                "version": "17.1",
                "elo_rating": 400
            }
        }
    
    def get_game_config(self):
        """Get game configuration"""
        return self.config.get("game_config", {})
    
    def get_v7p3r_config(self):
        """Get V7P3R AI configuration"""
        return self.config.get("v7p3r_config", {})
    
    def get_stockfish_config(self):
        """Get Stockfish configuration"""
        return self.config.get("stockfish_config", {})
    
    def get_training_config(self):
        """Get training configuration"""
        return self.config.get("training_config", {})


class GameState:
    """Game state manager for chess games"""
    
    def __init__(self, config=None):
        self.config = config if config else ChessConfig()
        self.board = chess.Board()
        self.move_history = []
        self.game = chess.pgn.Game()
        self.game.headers["Event"] = "V7P3R AI Training Game"
        self.game.headers["Site"] = "Local"
        self.game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        self.game.headers["Round"] = "1"
        
        game_config = self.config.get_game_config()
        self.game.headers["White"] = game_config.get("white_player", "v7p3r")
        self.game.headers["Black"] = game_config.get("black_player", "stockfish")
        
        self.node = self.game
    
    def make_move(self, move):
        """Make a move on the board and update PGN"""
        if move in self.board.legal_moves:
            # Save move to history
            self.move_history.append(move)
            
            # Make move on board
            self.board.push(move)
            
            # Update PGN
            self.node = self.node.add_variation(move)
            
            # Save PGN after each move
            self._save_pgn()
            
            return True
        else:
            print(f"Illegal move attempted: {move}")
            return False
    
    def _save_pgn(self, pgn_path="active_game.pgn"):
        """Save the current game to PGN file"""
        try:
            with open(pgn_path, 'w') as f:
                print(self.game, file=f, end="\n\n")
        except Exception as e:
            print(f"Error saving PGN: {e}")
    
    def get_legal_moves(self):
        """Get all legal moves in the current position"""
        return list(self.board.legal_moves)
    
    def is_game_over(self):
        """Check if the game is over"""
        return self.board.is_game_over()
    
    def get_result(self):
        """Get the result of the game"""
        if not self.is_game_over():
            return None
        
        if self.board.is_checkmate():
            return "1-0" if self.board.turn == chess.BLACK else "0-1"
        else:
            return "1/2-1/2"  # Draw
    
    def get_board_state(self):
        """Get the current board state"""
        return self.board
    
    def get_fen(self):
        """Get the FEN string of the current position"""
        return self.board.fen()
    
    def reset(self):
        """Reset the game state"""
        self.board = chess.Board()
        self.move_history = []
        self.game = chess.pgn.Game()
        self.game.headers["Event"] = "V7P3R AI Training Game"
        self.game.headers["Site"] = "Local"
        self.game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        self.game.headers["Round"] = "1"
        
        game_config = self.config.get_game_config()
        self.game.headers["White"] = game_config.get("white_player", "v7p3r")
        self.game.headers["Black"] = game_config.get("black_player", "stockfish")
        
        self.node = self.game


class BoardEvaluator:
    """Chess board evaluator for V7P3R AI"""
    
    # Piece values (centipawn)
    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    
    # Piece-square tables for positional evaluation
    # These tables give bonuses/penalties for piece positions
    # Pawns
    PAWN_TABLE = [
        0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5,  5, 10, 25, 25, 10,  5,  5,
        0,  0,  0, 20, 20,  0,  0,  0,
        5, -5,-10,  0,  0,-10, -5,  5,
        5, 10, 10,-20,-20, 10, 10,  5,
        0,  0,  0,  0,  0,  0,  0,  0
    ]
    
    # Knights
    KNIGHT_TABLE = [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50
    ]
    
    # Bishops
    BISHOP_TABLE = [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5,  5,  5,  5,  5,-10,
        -10,  0,  5,  0,  0,  5,  0,-10,
        -20,-10,-10,-10,-10,-10,-10,-20
    ]
    
    # Rooks
    ROOK_TABLE = [
        0,  0,  0,  0,  0,  0,  0,  0,
        5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        0,  0,  0,  5,  5,  0,  0,  0
    ]
    
    # Queens
    QUEEN_TABLE = [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
        -5,  0,  5,  5,  5,  5,  0, -5,
        0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20
    ]
    
    # Kings - Middlegame
    KING_MIDDLE_TABLE = [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
        20, 20,  0,  0,  0,  0, 20, 20,
        20, 30, 10,  0,  0, 10, 30, 20
    ]
    
    # Kings - Endgame
    KING_END_TABLE = [
        -50,-40,-30,-20,-20,-30,-40,-50,
        -30,-20,-10,  0,  0,-10,-20,-30,
        -30,-10, 20, 30, 30, 20,-10,-30,
        -30,-10, 30, 40, 40, 30,-10,-30,
        -30,-10, 30, 40, 40, 30,-10,-30,
        -30,-10, 20, 30, 30, 20,-10,-30,
        -30,-30,  0,  0,  0,  0,-30,-30,
        -50,-30,-30,-30,-30,-30,-30,-50
    ]
    
    def __init__(self, config=None):
        self.config = config if config else ChessConfig()
        
    def evaluate(self, board, perspective=chess.WHITE):
        """Evaluate the board position from the given perspective"""
        if board.is_checkmate():
            return -10000 if board.turn == perspective else 10000
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        # Material score
        material_score = self._evaluate_material(board)
        
        # Positional score
        positional_score = self._evaluate_position(board)
        
        # Mobility (number of legal moves)
        mobility_score = self._evaluate_mobility(board)
        
        # King safety
        king_safety_score = self._evaluate_king_safety(board)
        
        # Pawn structure
        pawn_structure_score = self._evaluate_pawn_structure(board)
        
        # Sum all scores
        total_score = (
            material_score + 
            positional_score + 
            mobility_score + 
            king_safety_score + 
            pawn_structure_score
        )
        
        # Return from perspective
        multiplier = 1 if perspective == chess.WHITE else -1
        return total_score * multiplier
    
    def _evaluate_material(self, board):
        """Evaluate material balance on the board"""
        score = 0
        
        # Count material for each side
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.PIECE_VALUES[piece.piece_type]
                if piece.color == chess.WHITE:
                    score += value
                else:
                    score -= value
        
        return score
    
    def _evaluate_position(self, board):
        """Evaluate piece positioning using piece-square tables"""
        score = 0
        
        # Is it endgame?
        is_endgame = self._is_endgame(board)
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if not piece:
                continue
                
            # Get position value from appropriate table
            if piece.piece_type == chess.PAWN:
                table_value = self.PAWN_TABLE[self._mirror_square(square) if piece.color == chess.BLACK else square]
            elif piece.piece_type == chess.KNIGHT:
                table_value = self.KNIGHT_TABLE[self._mirror_square(square) if piece.color == chess.BLACK else square]
            elif piece.piece_type == chess.BISHOP:
                table_value = self.BISHOP_TABLE[self._mirror_square(square) if piece.color == chess.BLACK else square]
            elif piece.piece_type == chess.ROOK:
                table_value = self.ROOK_TABLE[self._mirror_square(square) if piece.color == chess.BLACK else square]
            elif piece.piece_type == chess.QUEEN:
                table_value = self.QUEEN_TABLE[self._mirror_square(square) if piece.color == chess.BLACK else square]
            elif piece.piece_type == chess.KING:
                if is_endgame:
                    table_value = self.KING_END_TABLE[self._mirror_square(square) if piece.color == chess.BLACK else square]
                else:
                    table_value = self.KING_MIDDLE_TABLE[self._mirror_square(square) if piece.color == chess.BLACK else square]
            
            # Add/subtract based on piece color
            if piece.color == chess.WHITE:
                score += table_value
            else:
                score -= table_value
        
        return score
    
    def _evaluate_mobility(self, board):
        """Evaluate piece mobility (number of legal moves)"""
        original_turn = board.turn
        
        # Count white moves
        board.turn = chess.WHITE
        white_moves = len(list(board.legal_moves))
        
        # Count black moves
        board.turn = chess.BLACK
        black_moves = len(list(board.legal_moves))
        
        # Restore original turn
        board.turn = original_turn
        
        return (white_moves - black_moves) * 5  # Mobility weight
    
    def _evaluate_king_safety(self, board):
        """Evaluate king safety"""
        score = 0
        
        # Check if kings are castled
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)
        
        # Castling rights
        if board.has_castling_rights(chess.WHITE):
            score += 30
        if board.has_castling_rights(chess.BLACK):
            score -= 30
        
        # King attackers
        white_king_attackers = len(list(board.attackers(chess.BLACK, white_king_square)))
        black_king_attackers = len(list(board.attackers(chess.WHITE, black_king_square)))
        
        score -= white_king_attackers * 50  # Penalty for white king being attacked
        score += black_king_attackers * 50  # Bonus for attacking black king
        
        return score
    
    def _evaluate_pawn_structure(self, board):
        """Evaluate pawn structure (doubled/isolated pawns)"""
        score = 0
        
        # Evaluate doubled pawns
        for file_idx in range(8):
            file_mask = chess.BB_FILES[file_idx]
            white_pawns_on_file = bin(board.pieces(chess.PAWN, chess.WHITE) & file_mask).count('1')
            black_pawns_on_file = bin(board.pieces(chess.PAWN, chess.BLACK) & file_mask).count('1')
            
            # Penalty for doubled pawns
            if white_pawns_on_file > 1:
                score -= 20 * (white_pawns_on_file - 1)
            if black_pawns_on_file > 1:
                score += 20 * (black_pawns_on_file - 1)
        
        # Evaluate isolated pawns
        for file_idx in range(8):
            adjacent_files = 0
            if file_idx > 0:
                adjacent_files |= chess.BB_FILES[file_idx - 1]
            if file_idx < 7:
                adjacent_files |= chess.BB_FILES[file_idx + 1]
            
            # White isolated pawns
            if (board.pieces(chess.PAWN, chess.WHITE) & chess.BB_FILES[file_idx]) and not (board.pieces(chess.PAWN, chess.WHITE) & adjacent_files):
                score -= 10
            
            # Black isolated pawns
            if (board.pieces(chess.PAWN, chess.BLACK) & chess.BB_FILES[file_idx]) and not (board.pieces(chess.PAWN, chess.BLACK) & adjacent_files):
                score += 10
        
        return score
    
    def _is_endgame(self, board):
        """Determine if the position is an endgame"""
        queens = len(list(board.pieces(chess.QUEEN, chess.WHITE))) + len(list(board.pieces(chess.QUEEN, chess.BLACK)))
        minors = (
            len(list(board.pieces(chess.KNIGHT, chess.WHITE))) +
            len(list(board.pieces(chess.BISHOP, chess.WHITE))) +
            len(list(board.pieces(chess.KNIGHT, chess.BLACK))) +
            len(list(board.pieces(chess.BISHOP, chess.BLACK)))
        )
        
        return queens == 0 or (queens <= 2 and minors <= 4)
    
    def _mirror_square(self, square):
        """Mirror a square vertically (for black's perspective)"""
        return square ^ 56  # XOR with 56 flips the rank


class RewardCalculator:
    """Calculate rewards for reinforcement learning"""
    
    def __init__(self, config=None):
        self.config = config if config else ChessConfig()
        self.training_config = self.config.get_training_config()
        self.rewards = self.training_config.get("rewards", {})
        self.penalties = self.training_config.get("penalties", {})
        
    def calculate_reward(self, state, action, next_state):
        """Calculate reward for a state-action-next_state transition"""
        board = state.get_board_state()
        next_board = next_state.get_board_state()
        
        reward = 0
        
        # Check if game ended
        if next_state.is_game_over():
            if next_board.is_checkmate():
                # If we checkmate the opponent
                if next_board.turn == chess.BLACK:  # It's black's turn, so white won
                    reward += self.rewards.get("checkmate", 100.0)
                else:  # We got checkmated
                    reward += self.penalties.get("checkmate_against", -100.0)
            elif next_board.is_stalemate():
                reward += self.rewards.get("stalemate", -10.0)
            else:  # Draw
                reward += self.rewards.get("draw", -5.0)
            
            return reward
        
        # Check
        if next_board.is_check():
            if next_board.turn == chess.WHITE:  # We gave check to black
                reward += self.rewards.get("check", 5.0)
            else:  # We're in check
                reward += self.penalties.get("in_check", -3.0)
        
        # Capture rewards
        if action and board.piece_at(action.to_square):
            captured_piece = board.piece_at(action.to_square)
            piece_names = {
                chess.PAWN: "pawn", 
                chess.KNIGHT: "knight", 
                chess.BISHOP: "bishop", 
                chess.ROOK: "rook", 
                chess.QUEEN: "queen"
            }
            
            # Check if the capture is safe (no immediate recapture)
            is_safe = True
            board.push(action)
            attackers = board.attackers(not board.turn, action.to_square)
            if attackers:
                is_safe = False
            board.pop()
            
            if is_safe:
                reward += self.rewards.get("safe_capture", {}).get(
                    piece_names[captured_piece.piece_type], 1.0
                )
            else:
                reward += self.rewards.get("unsafe_capture", {}).get(
                    piece_names[captured_piece.piece_type], -0.5
                )
        
        # Castling reward
        if action and board.piece_at(action.from_square) and board.piece_at(action.from_square).piece_type == chess.KING:
            if action.from_square == chess.E1 and action.to_square in [chess.G1, chess.C1]:
                reward += self.rewards.get("castling", 3.0)
        
        # Losing castling rights penalty
        if board.has_castling_rights(chess.WHITE) and not next_board.has_castling_rights(chess.WHITE):
            # Check if we actually castled
            if not (action and board.piece_at(action.from_square) and 
                    board.piece_at(action.from_square).piece_type == chess.KING and
                    action.from_square == chess.E1 and action.to_square in [chess.G1, chess.C1]):
                reward += self.penalties.get("losing_castling_rights", -1.0)
        
        # Pawn promotion reward
        if action and action.promotion:
            if action.promotion == chess.QUEEN:
                reward += self.rewards.get("pawn_promotion", 8.0)
            else:
                reward += self.rewards.get("pawn_promotion", 8.0) / 2
        
        # En passant capture
        if action and board.is_en_passant(action):
            reward += self.rewards.get("en_passant_capture", 1.2)
        
        return reward


class FeatureExtractor:
    """Extract features from a chess board for machine learning"""
    
    def __init__(self):
        pass
    
    def extract_features(self, board):
        """Extract features from a board state"""
        # Board representation as a 8x8x12 tensor
        # 12 channels: 6 piece types for each color
        features = np.zeros((8, 8, 12), dtype=np.float32)
        
        # Fill features array
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Convert square index to rank and file (0-7)
                rank = square // 8
                file = square % 8
                
                # Determine feature index based on piece type and color
                piece_idx = piece.piece_type - 1  # 0-5
                if piece.color == chess.BLACK:
                    piece_idx += 6  # 6-11 for black pieces
                
                # Set feature value
                features[rank, file, piece_idx] = 1.0
        
        # Flatten for traditional ML models
        flat_features = features.flatten()
        
        # Add additional features
        additional_features = np.array([
            board.turn,  # 1 for white, 0 for black
            int(board.has_kingside_castling_rights(chess.WHITE)),
            int(board.has_queenside_castling_rights(chess.WHITE)),
            int(board.has_kingside_castling_rights(chess.BLACK)),
            int(board.has_queenside_castling_rights(chess.BLACK)),
            int(board.is_check()),
            int(board.is_checkmate()),
            int(board.is_stalemate()),
            int(board.is_insufficient_material()),
            int(board.is_seventyfive_moves()),
            int(board.is_fivefold_repetition()),
            int(board.has_legal_en_passant()),
            board.fullmove_number,
            board.halfmove_clock
        ], dtype=np.float32)
        
        # Combine flattened board features with additional features
        combined_features = np.concatenate((flat_features, additional_features))
        
        return combined_features