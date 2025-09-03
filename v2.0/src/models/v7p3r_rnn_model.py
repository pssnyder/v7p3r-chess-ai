# v7p3r_rnn_model.py
"""
V7P3R Chess AI 2.0 - Recurrent Neural Network Model
A powerful RNN-based chess AI using LSTM for move prediction and evaluation.
"""

import numpy as np
import chess
import random
from typing import List, Tuple, Optional, Dict, Any
import json
import pickle
import os


class LSTMCell:
    """Custom LSTM cell implementation for chess move prediction"""
    
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights using Xavier initialization
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * np.sqrt(2.0 / (input_size + hidden_size))
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * np.sqrt(2.0 / (input_size + hidden_size))
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * np.sqrt(2.0 / (input_size + hidden_size))
        self.Wg = np.random.randn(hidden_size, input_size + hidden_size) * np.sqrt(2.0 / (input_size + hidden_size))
        
        # Bias terms
        self.bf = np.zeros(hidden_size)
        self.bi = np.zeros(hidden_size)
        self.bo = np.zeros(hidden_size)
        self.bg = np.zeros(hidden_size)
        
        # Hidden and cell states
        self.hidden_state = np.zeros(hidden_size)
        self.cell_state = np.zeros(hidden_size)
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through LSTM cell"""
        # Concatenate input and previous hidden state
        combined = np.concatenate([x, self.hidden_state])
        
        # Forget gate
        f_gate = self.sigmoid(np.dot(self.Wf, combined) + self.bf)
        
        # Input gate
        i_gate = self.sigmoid(np.dot(self.Wi, combined) + self.bi)
        
        # Output gate
        o_gate = self.sigmoid(np.dot(self.Wo, combined) + self.bo)
        
        # Candidate values
        g_gate = np.tanh(np.dot(self.Wg, combined) + self.bg)
        
        # Update cell state
        self.cell_state = f_gate * self.cell_state + i_gate * g_gate
        
        # Update hidden state
        self.hidden_state = o_gate * np.tanh(self.cell_state)
        
        return self.hidden_state, self.cell_state
    
    def reset_state(self):
        """Reset hidden and cell states"""
        self.hidden_state = np.zeros(self.hidden_size)
        self.cell_state = np.zeros(self.hidden_size)
    
    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function with numerical stability"""
        return np.where(x >= 0, 
                       1 / (1 + np.exp(-x)), 
                       np.exp(x) / (1 + np.exp(x)))


class V7P3RNeuralNetwork:
    """V7P3R 2.0 RNN-based Chess AI Model"""
    
    def __init__(self, 
                 input_size: int = 816,  # Updated to actual feature size
                 hidden_size: int = 512,  # LSTM hidden units
                 num_layers: int = 2,     # Number of LSTM layers
                 output_size: int = 1,    # Move evaluation score
                 sequence_length: int = 64):  # Game history length
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.sequence_length = sequence_length
        
        # Create LSTM layers
        self.lstm_layers = []
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.lstm_layers.append(LSTMCell(layer_input_size, hidden_size))
        
        # Output layer weights
        self.W_out = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b_out = np.zeros(output_size)
        
        # Store sequence for training
        self.sequence_memory = []
        
        # Model parameters for genetic algorithm
        self.genome = self._extract_genome()
    
    def forward(self, x: np.ndarray) -> float:
        """Forward pass through the network"""
        # Pass through LSTM layers
        current_input = x
        for layer in self.lstm_layers:
            hidden_state, _ = layer.forward(current_input)
            current_input = hidden_state
        
        # Output layer
        output = np.dot(self.W_out, current_input) + self.b_out
        
        # Store in sequence memory
        self.sequence_memory.append(x)
        if len(self.sequence_memory) > self.sequence_length:
            self.sequence_memory.pop(0)
        
        return float(output[0])
    
    def reset_memory(self):
        """Reset LSTM memory and sequence"""
        for layer in self.lstm_layers:
            layer.reset_state()
        self.sequence_memory = []
    
    def _extract_genome(self) -> np.ndarray:
        """Extract all model parameters as a genome for genetic algorithm"""
        genome_parts = []
        
        # LSTM layer parameters
        for layer in self.lstm_layers:
            genome_parts.extend([
                layer.Wf.flatten(),
                layer.Wi.flatten(),
                layer.Wo.flatten(),
                layer.Wg.flatten(),
                layer.bf,
                layer.bi,
                layer.bo,
                layer.bg
            ])
        
        # Output layer parameters
        genome_parts.extend([
            self.W_out.flatten(),
            self.b_out
        ])
        
        return np.concatenate(genome_parts)
    
    def _load_genome(self, genome: np.ndarray):
        """Load parameters from genome"""
        idx = 0
        
        # Load LSTM layer parameters
        for layer in self.lstm_layers:
            # Weight matrices
            wf_size = layer.Wf.size
            layer.Wf = genome[idx:idx + wf_size].reshape(layer.Wf.shape)
            idx += wf_size
            
            wi_size = layer.Wi.size
            layer.Wi = genome[idx:idx + wi_size].reshape(layer.Wi.shape)
            idx += wi_size
            
            wo_size = layer.Wo.size
            layer.Wo = genome[idx:idx + wo_size].reshape(layer.Wo.shape)
            idx += wo_size
            
            wg_size = layer.Wg.size
            layer.Wg = genome[idx:idx + wg_size].reshape(layer.Wg.shape)
            idx += wg_size
            
            # Bias vectors
            layer.bf = genome[idx:idx + self.hidden_size]
            idx += self.hidden_size
            
            layer.bi = genome[idx:idx + self.hidden_size]
            idx += self.hidden_size
            
            layer.bo = genome[idx:idx + self.hidden_size]
            idx += self.hidden_size
            
            layer.bg = genome[idx:idx + self.hidden_size]
            idx += self.hidden_size
        
        # Load output layer parameters
        w_out_size = self.W_out.size
        self.W_out = genome[idx:idx + w_out_size].reshape(self.W_out.shape)
        idx += w_out_size
        
        self.b_out = genome[idx:idx + self.output_size]
        
        # Update genome reference
        self.genome = genome.copy()
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.01) -> 'V7P3RNeuralNetwork':
        """Create a mutated copy of this network"""
        new_network = V7P3RNeuralNetwork(
            self.input_size, self.hidden_size, self.num_layers, 
            self.output_size, self.sequence_length
        )
        
        # Copy current genome
        new_genome = self.genome.copy()
        
        # Apply mutations
        mutation_mask = np.random.random(new_genome.shape) < mutation_rate
        mutations = np.random.normal(0, mutation_strength, new_genome.shape)
        new_genome[mutation_mask] += mutations[mutation_mask]
        
        # Load mutated genome
        new_network._load_genome(new_genome)
        
        return new_network
    
    def crossover(self, other: 'V7P3RNeuralNetwork', crossover_rate: float = 0.5) -> 'V7P3RNeuralNetwork':
        """Create offspring through crossover with another network"""
        new_network = V7P3RNeuralNetwork(
            self.input_size, self.hidden_size, self.num_layers,
            self.output_size, self.sequence_length
        )
        
        # Create child genome through crossover
        crossover_mask = np.random.random(self.genome.shape) < crossover_rate
        child_genome = np.where(crossover_mask, self.genome, other.genome)
        
        # Load child genome
        new_network._load_genome(child_genome)
        
        return new_network
    
    def save_model(self, filepath: str):
        """Save model to file"""
        model_data = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.output_size,
            'sequence_length': self.sequence_length,
            'genome': self.genome.tolist()
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'V7P3RNeuralNetwork':
        """Load model from file"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        network = cls(
            model_data['input_size'],
            model_data['hidden_size'],
            model_data['num_layers'],
            model_data['output_size'],
            model_data['sequence_length']
        )
        
        network._load_genome(np.array(model_data['genome']))
        
        return network


class ChessFeatureExtractor:
    """Advanced feature extractor for V7P3R 2.0"""
    
    def __init__(self):
        # Piece values for feature extraction
        self.piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3.25,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100
        }
        
        # Board zones for positional features
        self.center_squares = {chess.D4, chess.D5, chess.E4, chess.E5}
        self.center_ring = {
            chess.C3, chess.C4, chess.C5, chess.C6,
            chess.D3, chess.D6, chess.E3, chess.E6,
            chess.F3, chess.F4, chess.F5, chess.F6
        }
        self.central_edge = {
            chess.B2, chess.B3, chess.B4, chess.B5, chess.B6, chess.B7,
            chess.C2, chess.C7, chess.D2, chess.D7, chess.E2, chess.E7,
            chess.F2, chess.F7, chess.G2, chess.G3, chess.G4, chess.G5, chess.G6, chess.G7
        }
    
    def extract_features(self, board: chess.Board, move_history: Optional[List[chess.Move]] = None) -> np.ndarray:
        """Extract comprehensive features from board state"""
        features = []
        
        # 1. Board representation (8x8x12 = 768 features)
        board_features = self._extract_board_features(board)
        features.extend(board_features)
        
        # 2. Game state features (14 features)
        game_state = self._extract_game_state_features(board)
        features.extend(game_state)
        
        # 3. Positional features (32 features)
        positional = self._extract_positional_features(board)
        features.extend(positional)
        
        # 4. Tactical features (16 features)
        tactical = self._extract_tactical_features(board)
        features.extend(tactical)
        
        # 5. Move history features (8 features)
        history = self._extract_history_features(board, move_history)
        features.extend(history)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_board_features(self, board: chess.Board) -> List[float]:
        """Extract 8x8x12 board representation"""
        features = np.zeros((8, 8, 12), dtype=np.float32)
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank = square // 8
                file = square % 8
                
                # 6 piece types for each color
                piece_idx = piece.piece_type - 1
                if piece.color == chess.BLACK:
                    piece_idx += 6
                
                features[rank, file, piece_idx] = 1.0
        
        return features.flatten().tolist()
    
    def _extract_game_state_features(self, board: chess.Board) -> List[float]:
        """Extract game state metadata"""
        return [
            float(board.turn),  # Current player
            float(board.has_kingside_castling_rights(chess.WHITE)),
            float(board.has_queenside_castling_rights(chess.WHITE)),
            float(board.has_kingside_castling_rights(chess.BLACK)),
            float(board.has_queenside_castling_rights(chess.BLACK)),
            float(board.is_check()),
            float(board.is_checkmate()),
            float(board.is_stalemate()),
            float(board.is_insufficient_material()),
            float(board.has_legal_en_passant()),
            float(board.fullmove_number) / 100.0,  # Normalized
            float(board.halfmove_clock) / 50.0,   # Normalized
            float(len(list(board.legal_moves))) / 50.0,  # Mobility normalized
            float(self._calculate_material_balance(board)) / 39.0  # Material balance normalized
        ]
    
    def _extract_positional_features(self, board: chess.Board) -> List[float]:
        """Extract positional features"""
        features = []
        
        # Center control
        white_center_control = 0
        black_center_control = 0
        
        for square in self.center_squares:
            white_attackers = len(list(board.attackers(chess.WHITE, square)))
            black_attackers = len(list(board.attackers(chess.BLACK, square)))
            white_center_control += white_attackers
            black_center_control += black_attackers
        
        features.extend([
            white_center_control / 16.0,  # Normalized
            black_center_control / 16.0
        ])
        
        # Piece development
        features.extend(self._calculate_piece_development(board))
        
        # King safety
        features.extend(self._calculate_king_safety(board))
        
        # Pawn structure
        features.extend(self._calculate_pawn_structure(board))
        
        return features
    
    def _extract_tactical_features(self, board: chess.Board) -> List[float]:
        """Extract tactical pattern features"""
        features = []
        
        # Pins, skewers, forks detection
        pins = self._detect_pins(board)
        skewers = self._detect_skewers(board)
        forks = self._detect_forks(board)
        
        features.extend([
            len(pins[chess.WHITE]) / 8.0,   # White pins normalized
            len(pins[chess.BLACK]) / 8.0,   # Black pins normalized
            len(skewers[chess.WHITE]) / 8.0, # White skewers normalized
            len(skewers[chess.BLACK]) / 8.0, # Black skewers normalized
            len(forks[chess.WHITE]) / 8.0,   # White forks normalized
            len(forks[chess.BLACK]) / 8.0    # Black forks normalized
        ])
        
        # Attack features
        white_attacks = 0
        black_attacks = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                attackers = list(board.attackers(not piece.color, square))
                if piece.color == chess.WHITE:
                    black_attacks += len(attackers)
                else:
                    white_attacks += len(attackers)
        
        features.extend([
            white_attacks / 64.0,  # Normalized attack counts
            black_attacks / 64.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Reserved for future tactical features
        ])
        
        return features
    
    def _extract_history_features(self, board: chess.Board, move_history: Optional[List[chess.Move]]) -> List[float]:
        """Extract features from move history"""
        features = [0.0] * 8  # Initialize with zeros
        
        if move_history and len(move_history) > 0:
            recent_moves = move_history[-8:]  # Last 8 moves
            
            for i, move in enumerate(recent_moves):
                # Simple encoding of recent move patterns
                features[i] = float(move.from_square + move.to_square) / 128.0  # Normalized
        
        return features
    
    def _calculate_material_balance(self, board: chess.Board) -> float:
        """Calculate material balance"""
        balance = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    balance += value
                else:
                    balance -= value
        
        return balance
    
    def _calculate_piece_development(self, board: chess.Board) -> List[float]:
        """Calculate piece development features"""
        white_developed = 0
        black_developed = 0
        
        # Check if knights and bishops are developed
        starting_squares = {
            chess.B1: chess.KNIGHT, chess.G1: chess.KNIGHT,
            chess.C1: chess.BISHOP, chess.F1: chess.BISHOP,
            chess.B8: chess.KNIGHT, chess.G8: chess.KNIGHT,
            chess.C8: chess.BISHOP, chess.F8: chess.BISHOP
        }
        
        for square, piece_type in starting_squares.items():
            piece = board.piece_at(square)
            if not piece or piece.piece_type != piece_type:
                if square in [chess.B1, chess.G1, chess.C1, chess.F1]:
                    white_developed += 1
                else:
                    black_developed += 1
        
        return [white_developed / 4.0, black_developed / 4.0]
    
    def _calculate_king_safety(self, board: chess.Board) -> List[float]:
        """Calculate king safety features"""
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)
        
        white_safety = 0
        black_safety = 0
        
        if white_king:
            # Count attackers near white king
            for square in chess.SQUARES:
                if chess.square_distance(square, white_king) <= 2:
                    black_safety += len(list(board.attackers(chess.BLACK, square)))
        
        if black_king:
            # Count attackers near black king
            for square in chess.SQUARES:
                if chess.square_distance(square, black_king) <= 2:
                    white_safety += len(list(board.attackers(chess.WHITE, square)))
        
        return [white_safety / 20.0, black_safety / 20.0]  # Normalized
    
    def _calculate_pawn_structure(self, board: chess.Board) -> List[float]:
        """Calculate pawn structure features"""
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)
        
        # Count doubled pawns
        white_doubled = 0
        black_doubled = 0
        
        for file_idx in range(8):
            file_mask = chess.BB_FILES[file_idx]
            white_on_file = bin(white_pawns & file_mask).count('1')
            black_on_file = bin(black_pawns & file_mask).count('1')
            
            if white_on_file > 1:
                white_doubled += white_on_file - 1
            if black_on_file > 1:
                black_doubled += black_on_file - 1
        
        # Count isolated pawns
        white_isolated = 0
        black_isolated = 0
        
        for file_idx in range(8):
            adjacent_files = 0
            if file_idx > 0:
                adjacent_files |= chess.BB_FILES[file_idx - 1]
            if file_idx < 7:
                adjacent_files |= chess.BB_FILES[file_idx + 1]
            
            current_file = chess.BB_FILES[file_idx]
            
            if (white_pawns & current_file) and not (white_pawns & adjacent_files):
                white_isolated += 1
            if (black_pawns & current_file) and not (black_pawns & adjacent_files):
                black_isolated += 1
        
        return [
            white_doubled / 8.0, black_doubled / 8.0,
            white_isolated / 8.0, black_isolated / 8.0
        ]
    
    def _detect_pins(self, board: chess.Board) -> Dict[chess.Color, List]:
        """Detect pinned pieces"""
        pins = {chess.WHITE: [], chess.BLACK: []}
        
        for color in [chess.WHITE, chess.BLACK]:
            king_square = board.king(color)
            if not king_square:
                continue
                
            # Check for pins by sliding pieces
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.color != color and piece.piece_type in [chess.ROOK, chess.BISHOP, chess.QUEEN]:
                    # Check if piece can attack king through another piece
                    between_bb = chess.between(square, king_square)
                    between_squares = [sq for sq in chess.SQUARES if between_bb & chess.BB_SQUARES[sq]]
                    
                    if len(between_squares) == 1:
                        pinned_square = between_squares[0]
                        pinned_piece = board.piece_at(pinned_square)
                        if pinned_piece and pinned_piece.color == color:
                            pins[color].append((pinned_square, square))
        
        return pins
    
    def _detect_skewers(self, board: chess.Board) -> Dict[chess.Color, List]:
        """Detect skewer patterns"""
        skewers = {chess.WHITE: [], chess.BLACK: []}
        
        # Simplified skewer detection for now
        # This would need more sophisticated ray-casting logic
        return skewers
    
    def _detect_forks(self, board: chess.Board) -> Dict[chess.Color, List]:
        """Detect fork patterns"""
        forks = {chess.WHITE: [], chess.BLACK: []}
        
        for color in [chess.WHITE, chess.BLACK]:
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.color == color:
                    # Count enemy pieces this piece attacks
                    attacked_pieces = []
                    for attacked_square in board.attacks(square):
                        attacked_piece = board.piece_at(attacked_square)
                        if attacked_piece and attacked_piece.color != color:
                            attacked_pieces.append(attacked_square)
                    
                    # Fork if attacking 2+ pieces
                    if len(attacked_pieces) >= 2:
                        forks[color].append((square, attacked_pieces))
        
        return forks
