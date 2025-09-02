# v7p3r_gpu_model.py
"""
V7P3R Chess AI 2.0 - GPU-Accelerated Neural Network
CUDA-enabled LSTM model for high-performance chess AI training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess
import random
from typing import List, Tuple, Optional, Dict, Any
import json
import os
from pathlib import Path


class V7P3RGPU_LSTM(nn.Module):
    """GPU-accelerated LSTM network for V7P3R Chess AI 2.0"""
    
    def __init__(self, 
                 input_size: int = 816,
                 hidden_size: int = 512,
                 num_layers: int = 2,
                 output_size: int = 1,
                 dropout: float = 0.1,
                 device: str = 'cuda'):
        super(V7P3RGPU_LSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.device = device
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bias=True
        )
        
        # Output layers with residual connection
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, output_size)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 4)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Move to device
        self.to(device)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                if 'bias_ih' in name:
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1)
    
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through the network"""
        # x shape: (batch_size, sequence_length, input_size)
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Take the last output from the sequence
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Feed-forward layers
        out = F.relu(self.fc1(last_output))
        out = self.bn1(out)
        out = self.dropout(out)
        
        out = F.relu(self.fc2(out))
        out = self.bn2(out)
        out = self.dropout(out)
        
        out = self.fc3(out)  # Final output
        
        return out, hidden
    
    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        return h0, c0
    
    def get_genome(self) -> torch.Tensor:
        """Extract model parameters as a genome tensor"""
        params = []
        for param in self.parameters():
            params.append(param.data.flatten())
        return torch.cat(params)
    
    def set_genome(self, genome: torch.Tensor):
        """Set model parameters from genome tensor"""
        idx = 0
        for param in self.parameters():
            param_size = param.numel()
            param.data = genome[idx:idx + param_size].view(param.shape)
            idx += param_size
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.02) -> 'V7P3RGPU_LSTM':
        """Create a mutated copy of this network"""
        new_model = V7P3RGPU_LSTM(
            self.input_size, self.hidden_size, self.num_layers, 
            self.output_size, device=self.device
        )
        
        # Copy parameters
        new_model.load_state_dict(self.state_dict())
        
        # Apply mutations
        with torch.no_grad():
            for param in new_model.parameters():
                mutation_mask = torch.rand_like(param) < mutation_rate
                mutations = torch.normal(0, mutation_strength, param.shape, device=self.device)
                param.data[mutation_mask] += mutations[mutation_mask]
        
        return new_model
    
    def crossover(self, other: 'V7P3RGPU_LSTM', crossover_rate: float = 0.5) -> 'V7P3RGPU_LSTM':
        """Create offspring through crossover"""
        child = V7P3RGPU_LSTM(
            self.input_size, self.hidden_size, self.num_layers,
            self.output_size, device=self.device
        )
        
        # Crossover parameters
        with torch.no_grad():
            for child_param, parent1_param, parent2_param in zip(
                child.parameters(), self.parameters(), other.parameters()
            ):
                crossover_mask = torch.rand_like(parent1_param) < crossover_rate
                child_param.data = torch.where(crossover_mask, parent1_param.data, parent2_param.data)
        
        return child
    
    def save_model(self, filepath: str):
        """Save model to file"""
        model_data = {
            'state_dict': self.state_dict(),
            'config': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'output_size': self.output_size
            }
        }
        torch.save(model_data, filepath)
    
    @classmethod
    def load_model(cls, filepath: str, device: str = 'cuda') -> 'V7P3RGPU_LSTM':
        """Load model from file"""
        checkpoint = torch.load(filepath, map_location=device)
        config = checkpoint['config']
        
        model = cls(
            config['input_size'],
            config['hidden_size'], 
            config['num_layers'],
            config['output_size'],
            device=device
        )
        
        model.load_state_dict(checkpoint['state_dict'])
        return model


class GPUChessFeatureExtractor:
    """GPU-optimized feature extractor"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
        # Pre-computed piece values
        self.piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3.25,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100
        }
        
        # Board zones
        self.center_squares = {chess.D4, chess.D5, chess.E4, chess.E5}
        self.center_ring = {
            chess.C3, chess.C4, chess.C5, chess.C6,
            chess.D3, chess.D6, chess.E3, chess.E6,
            chess.F3, chess.F4, chess.F5, chess.F6
        }
    
    def to(self, device):
        """Move to device (compatibility with PyTorch models)"""
        self.device = device
        return self
    
    def extract_features(self, board: chess.Board, 
                        move_history: Optional[List[chess.Move]] = None) -> List[float]:
        """Extract features for a single board (compatible with existing interface)"""
        features_tensor = self._extract_single_features(board, move_history)
        return features_tensor.flatten().tolist()
    
    def extract_features_batch(self, boards: List[chess.Board], 
                              move_histories: Optional[List[List[chess.Move]]] = None) -> torch.Tensor:
        """Extract features for a batch of boards (GPU-optimized)"""
        batch_size = len(boards)
        features_list = []
        
        for i, board in enumerate(boards):
            move_history = move_histories[i] if move_histories else None
            features = self._extract_single_features(board, move_history)
            features_list.append(features)
        
        # Stack into batch tensor
        batch_features = torch.stack(features_list)
        return batch_features.to(self.device)
    
    def _extract_single_features(self, board: chess.Board, 
                                move_history: Optional[List[chess.Move]] = None) -> torch.Tensor:
        """Extract features for a single board"""
        features = []
        
        # 1. Board representation (8x8x12 = 768 features)
        board_tensor = torch.zeros(8, 8, 12)
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank = square // 8
                file = square % 8
                piece_idx = piece.piece_type - 1
                if piece.color == chess.BLACK:
                    piece_idx += 6
                board_tensor[rank, file, piece_idx] = 1.0
        
        features.extend(board_tensor.flatten().tolist())
        
        # 2. Game state features
        game_state = [
            float(board.turn),
            float(board.has_kingside_castling_rights(chess.WHITE)),
            float(board.has_queenside_castling_rights(chess.WHITE)),
            float(board.has_kingside_castling_rights(chess.BLACK)),
            float(board.has_queenside_castling_rights(chess.BLACK)),
            float(board.is_check()),
            float(board.is_checkmate()),
            float(board.is_stalemate()),
            float(board.is_insufficient_material()),
            float(board.has_legal_en_passant()),
            float(board.fullmove_number) / 100.0,
            float(board.halfmove_clock) / 50.0,
            float(len(list(board.legal_moves))) / 50.0,
            float(self._calculate_material_balance(board)) / 39.0
        ]
        features.extend(game_state)
        
        # 3. Positional features (simplified for GPU efficiency)
        positional = self._extract_positional_features_fast(board)
        features.extend(positional)
        
        # 4. Move history (last 8 moves encoded)
        history_features = [0.0] * 8
        if move_history:
            recent_moves = move_history[-8:]
            for i, move in enumerate(recent_moves):
                history_features[i] = float(move.from_square + move.to_square) / 128.0
        features.extend(history_features)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _extract_positional_features_fast(self, board: chess.Board) -> List[float]:
        """Fast positional feature extraction"""
        features = []
        
        # Center control (simplified)
        white_center = 0
        black_center = 0
        for square in self.center_squares:
            white_center += len(list(board.attackers(chess.WHITE, square)))
            black_center += len(list(board.attackers(chess.BLACK, square)))
        
        features.extend([white_center / 8.0, black_center / 8.0])
        
        # King safety (simplified)
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)
        
        white_king_attackers = 0
        black_king_attackers = 0
        
        if white_king:
            black_king_attackers = len(list(board.attackers(chess.BLACK, white_king)))
        if black_king:
            white_king_attackers = len(list(board.attackers(chess.WHITE, black_king)))
        
        features.extend([white_king_attackers / 8.0, black_king_attackers / 8.0])
        
        # Pad to expected size
        while len(features) < 26:  # Adjust based on expected feature count
            features.append(0.0)
        
        return features[:26]  # Ensure consistent size
    
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


class V7P3RGPU_AI:
    """GPU-accelerated V7P3R Chess AI"""
    
    def __init__(self, model: Optional[V7P3RGPU_LSTM] = None, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        if self.device == 'cpu' and device == 'cuda':
            print("CUDA not available, falling back to CPU")
        
        self.model = model if model else V7P3RGPU_LSTM(device=self.device)
        self.feature_extractor = GPUChessFeatureExtractor(self.device)
        self.move_history: List[chess.Move] = []
        self.hidden_state = None
        
        # Set model to evaluation mode
        self.model.eval()
    
    def get_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Get best move using GPU acceleration"""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        # Evaluate all moves in parallel on GPU
        move_scores = self._evaluate_moves_gpu(board, legal_moves)
        
        # Find best move
        best_idx = torch.argmax(move_scores).item()
        best_move = legal_moves[best_idx]
        
        # Update move history
        self.move_history.append(best_move)
        if len(self.move_history) > 64:
            self.move_history.pop(0)
        
        return best_move
    
    def _evaluate_moves_gpu(self, board: chess.Board, moves: List[chess.Move]) -> torch.Tensor:
        """Evaluate moves using GPU batch processing"""
        batch_boards = []
        batch_histories = []
        
        # Create board positions for each move
        for move in moves:
            temp_board = board.copy()
            temp_board.push(move)
            batch_boards.append(temp_board)
            batch_histories.append(self.move_history + [move])
        
        # Extract features in batch
        with torch.no_grad():
            batch_features = self.feature_extractor.extract_features_batch(
                batch_boards, batch_histories
            )
            
            # Add sequence dimension (batch_size, seq_len=1, features)
            batch_features = batch_features.unsqueeze(1)
            
            # Forward pass
            outputs, _ = self.model(batch_features)
            
            # Return flattened scores
            return outputs.flatten()
    
    def reset_game(self):
        """Reset for new game"""
        self.move_history = []
        self.hidden_state = None
    
    def save_model(self, filepath: str):
        """Save the model"""
        self.model.save_model(filepath)
    
    @classmethod
    def load_model(cls, filepath: str, device: str = 'cuda') -> 'V7P3RGPU_AI':
        """Load model from file"""
        model = V7P3RGPU_LSTM.load_model(filepath, device)
        return cls(model, device)


def check_gpu_availability():
    """Check GPU availability and capabilities"""
    print("GPU Availability Check:")
    print("=" * 40)
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ CUDA available with {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Test GPU performance
        print("\n🧪 GPU Performance Test:")
        device = torch.device('cuda:0')
        
        # Create test tensors
        test_size = (1000, 1000)
        a = torch.randn(test_size, device=device)
        b = torch.randn(test_size, device=device)
        
        # Time matrix multiplication
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        c = torch.matmul(a, b)
        end_time.record()
        
        torch.cuda.synchronize()
        gpu_time = start_time.elapsed_time(end_time)
        
        print(f"  Matrix multiplication (1000x1000): {gpu_time:.2f}ms")
        
        # Memory usage
        memory_allocated = torch.cuda.memory_allocated(0) / 1e9
        memory_cached = torch.cuda.memory_reserved(0) / 1e9
        print(f"  Memory allocated: {memory_allocated:.2f} GB")
        print(f"  Memory cached: {memory_cached:.2f} GB")
        
        return True, device
    else:
        print("❌ CUDA not available")
        print("  Falling back to CPU")
        return False, torch.device('cpu')


if __name__ == "__main__":
    # Test GPU functionality
    gpu_available, device = check_gpu_availability()
    
    if gpu_available:
        print("\n🚀 Testing V7P3R GPU Model:")
        
        # Create GPU model
        model = V7P3RGPU_LSTM(device=device)
        ai = V7P3RGPU_AI(model, device)
        
        # Test with a chess position
        board = chess.Board()
        
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        move = ai.get_move(board)
        end_time.record()
        
        torch.cuda.synchronize()
        inference_time = start_time.elapsed_time(end_time)
        
        print(f"  Best move: {move.uci() if move else 'None'}")
        print(f"  Inference time: {inference_time:.2f}ms")
        print("  ✅ GPU acceleration working!")
    else:
        print("  Consider installing CUDA for GPU acceleration")
