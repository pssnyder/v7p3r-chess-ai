# neural_network_integration.py
"""
V7P3R Chess AI 2.0 - Neural Network Integration
Integrates the improved bounty system and move preparation with neural network training.
"""

import torch
import torch.nn as nn
import numpy as np
import chess
from typing import List, Tuple, Dict, Optional
import time
from dataclasses import dataclass

from core.performance_optimizer import PerformanceOptimizer
from evaluation.v7p3r_bounty_system import ExtendedBountyEvaluator


@dataclass
class TrainingData:
    """Training data for neural network"""
    position_features: np.ndarray  # Board position features
    move_features: np.ndarray      # Move features from optimizer
    bounty_score: float           # Target bounty score
    move_quality: float           # Move quality (0-1)
    game_outcome: float           # Game outcome (-1, 0, 1)


class EnhancedNeuralNetwork(nn.Module):
    """
    Enhanced neural network that processes both position and move features
    """
    
    def __init__(self, 
                 position_size: int = 773,  # Standard chess position encoding
                 move_features_size: int = 27,  # From performance optimizer
                 hidden_size: int = 512):
        super().__init__()
        
        # Position encoding network
        self.position_encoder = nn.Sequential(
            nn.Linear(position_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Move features network
        self.move_encoder = nn.Sequential(
            nn.Linear(move_features_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, hidden_size // 4),
            nn.ReLU()
        )
        
        # Combined evaluation network
        combined_size = hidden_size // 2 + hidden_size // 4
        self.evaluator = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Tanh()  # Output between -1 and 1
        )
        
        # Move quality predictor (auxiliary task)
        self.quality_predictor = nn.Sequential(
            nn.Linear(combined_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
    
    def forward(self, position_features: torch.Tensor, move_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Returns:
            Tuple of (position_evaluation, move_quality)
        """
        # Encode position and move features
        pos_encoded = self.position_encoder(position_features)
        move_encoded = self.move_encoder(move_features)
        
        # Combine features
        combined = torch.cat([pos_encoded, move_encoded], dim=-1)
        
        # Generate outputs
        evaluation = self.evaluator(combined)
        quality = self.quality_predictor(combined)
        
        return evaluation, quality


class NeuralNetworkTrainer:
    """
    Enhanced trainer that integrates bounty system and move preparation
    """
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.optimizer_system = PerformanceOptimizer()
        self.bounty_evaluator = ExtendedBountyEvaluator()
        
        # Initialize neural network
        self.model = EnhancedNeuralNetwork().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5)
        
        # Loss functions
        self.evaluation_loss = nn.MSELoss()
        self.quality_loss = nn.BCELoss()
        
        # Training configuration
        self.config = {
            'batch_size': 32,
            'moves_per_position': 10,
            'quality_weight': 0.3,
            'evaluation_weight': 0.7,
            'use_move_ordering': True,
            'max_positions_per_batch': 100
        }
        
        print(f"Neural Network Trainer initialized on {device}")
    
    def prepare_training_data(self, games: List[str], max_positions: Optional[int] = None) -> List[TrainingData]:
        """
        Prepare training data from PGN games
        
        Args:
            games: List of PGN game strings
            max_positions: Maximum positions to extract
            
        Returns:
            List of training data
        """
        training_data = []
        positions_processed = 0
        
        print(f"Preparing training data from {len(games)} games...")
        
        for game_idx, pgn_string in enumerate(games):
            if max_positions and positions_processed >= max_positions:
                break
            
            try:
                # Parse game
                board = chess.Board()
                moves = self._parse_pgn_moves(pgn_string)
                game_outcome = self._extract_game_outcome(pgn_string)
                
                # Process each position in the game
                for move_idx, move_san in enumerate(moves):
                    if max_positions and positions_processed >= max_positions:
                        break
                    
                    try:
                        move = board.parse_san(move_san)
                        
                        # Generate training data for this position
                        position_data = self._generate_position_training_data(
                            board, move, game_outcome, move_idx, len(moves)
                        )
                        
                        training_data.extend(position_data)
                        positions_processed += 1
                        
                        # Make the move
                        board.push(move)
                        
                    except (chess.InvalidMoveError, chess.IllegalMoveError):
                        continue
                
                if (game_idx + 1) % 100 == 0:
                    print(f"Processed {game_idx + 1}/{len(games)} games, {positions_processed} positions")
                
            except Exception as e:
                print(f"Error processing game {game_idx}: {e}")
                continue
        
        print(f"Training data preparation complete: {len(training_data)} samples from {positions_processed} positions")
        return training_data
    
    def _generate_position_training_data(self, board: chess.Board, played_move: chess.Move, 
                                       game_outcome: float, move_idx: int, total_moves: int) -> List[TrainingData]:
        """Generate training data for a single position"""
        position_data = []
        
        # Get position features (simplified - would need proper chess encoding)
        position_features = self._encode_position(board)
        
        # Get top moves from optimizer
        optimized_moves = self.optimizer_system.optimize_move_selection(board, time_limit=0.1)
        
        # Limit number of moves to consider
        moves_to_consider = min(len(optimized_moves), self.config['moves_per_position'])
        
        for i, (move, optimizer_score) in enumerate(optimized_moves[:moves_to_consider]):
            # Get move features from optimizer
            move_features = self.optimizer_system.get_neural_network_features(board, move)
            
            # Calculate bounty score
            bounty_score = self.bounty_evaluator.evaluate_move(board, move)
            normalized_bounty = bounty_score.total() / 100.0  # Normalize
            
            # Calculate move quality based on various factors
            move_quality = self._calculate_move_quality(
                move, played_move, optimizer_score, i, game_outcome, move_idx, total_moves
            )
            
            # Create training sample
            training_sample = TrainingData(
                position_features=position_features,
                move_features=move_features,
                bounty_score=normalized_bounty,
                move_quality=move_quality,
                game_outcome=game_outcome
            )
            
            position_data.append(training_sample)
        
        return position_data
    
    def _calculate_move_quality(self, move: chess.Move, played_move: chess.Move, 
                              optimizer_score: float, rank: int, game_outcome: float,
                              move_idx: int, total_moves: int) -> float:
        """Calculate move quality score (0-1)"""
        base_quality = 0.5
        
        # Bonus if this is the move that was actually played
        if move == played_move:
            base_quality += 0.3
        
        # Bonus based on optimizer ranking
        rank_bonus = max(0, (10 - rank) / 10 * 0.2)
        base_quality += rank_bonus
        
        # Bonus based on optimizer score
        score_bonus = min(optimizer_score / 50.0, 0.2)  # Cap at 0.2
        base_quality += score_bonus
        
        # Game outcome influence (stronger for later moves)
        move_progress = move_idx / max(total_moves, 1)
        outcome_influence = game_outcome * move_progress * 0.1
        base_quality += outcome_influence
        
        # Ensure quality is in [0, 1] range
        return max(0.0, min(1.0, base_quality))
    
    def train_batch(self, training_data: List[TrainingData]) -> Dict[str, float]:
        """Train on a batch of data"""
        if len(training_data) < self.config['batch_size']:
            return {}
        
        # Sample batch
        batch_indices = np.random.choice(len(training_data), self.config['batch_size'], replace=False)
        batch_data = [training_data[i] for i in batch_indices]
        
        # Prepare tensors
        position_features = torch.tensor(
            np.stack([data.position_features for data in batch_data]), 
            dtype=torch.float32, device=self.device
        )
        move_features = torch.tensor(
            np.stack([data.move_features for data in batch_data]), 
            dtype=torch.float32, device=self.device
        )
        bounty_targets = torch.tensor(
            [data.bounty_score for data in batch_data], 
            dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        quality_targets = torch.tensor(
            [data.move_quality for data in batch_data], 
            dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        
        # Forward pass
        self.optimizer.zero_grad()
        evaluation_pred, quality_pred = self.model(position_features, move_features)
        
        # Calculate losses
        eval_loss = self.evaluation_loss(evaluation_pred, bounty_targets)
        qual_loss = self.quality_loss(quality_pred, quality_targets)
        
        # Combined loss
        total_loss = (
            self.config['evaluation_weight'] * eval_loss +
            self.config['quality_weight'] * qual_loss
        )
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'evaluation_loss': eval_loss.item(),
            'quality_loss': qual_loss.item()
        }
    
    def evaluate_position(self, board: chess.Board) -> Dict[str, float]:
        """Evaluate a position using the trained model"""
        self.model.eval()
        
        with torch.no_grad():
            # Get optimized moves
            optimized_moves = self.optimizer_system.optimize_move_selection(board, time_limit=0.5)
            
            if not optimized_moves:
                return {'evaluation': 0.0, 'best_move': None}
            
            best_move = None
            best_evaluation = float('-inf')
            move_evaluations = []
            
            # Evaluate each move with the neural network
            for move, optimizer_score in optimized_moves[:10]:  # Top 10 moves
                position_features = self._encode_position(board)
                move_features = self.optimizer_system.get_neural_network_features(board, move)
                
                # Convert to tensors
                pos_tensor = torch.tensor(position_features, dtype=torch.float32, device=self.device).unsqueeze(0)
                move_tensor = torch.tensor(move_features, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                # Get model prediction
                evaluation, quality = self.model(pos_tensor, move_tensor)
                
                evaluation_score = evaluation.item()
                quality_score = quality.item()
                
                # Combine with optimizer score
                combined_score = (
                    evaluation_score * 0.5 +
                    quality_score * 0.3 +
                    optimizer_score / 50.0 * 0.2
                )
                
                move_evaluations.append({
                    'move': move,
                    'evaluation': evaluation_score,
                    'quality': quality_score,
                    'optimizer_score': optimizer_score,
                    'combined_score': combined_score
                })
                
                if combined_score > best_evaluation:
                    best_evaluation = combined_score
                    best_move = move
        
        self.model.train()
        
        return {
            'best_move': best_move,
            'best_evaluation': best_evaluation,
            'move_evaluations': move_evaluations
        }
    
    def _encode_position(self, board: chess.Board) -> np.ndarray:
        """
        Encode chess position as feature vector
        Simplified implementation - would need proper chess encoding
        """
        # This is a simplified encoding - in practice, would use proper chess encoding
        features = np.zeros(773)  # Standard size for chess position encoding
        
        # Basic piece placement encoding
        piece_map = {
            chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3,
            chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_type = piece_map[piece.piece_type]
                color_offset = 0 if piece.color == chess.WHITE else 6
                features[square * 12 + piece_type - 1 + color_offset] = 1
        
        # Add basic game state features
        features[768] = 1 if board.turn == chess.WHITE else 0
        features[769] = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
        features[770] = 1 if board.has_queenside_castling_rights(chess.WHITE) else 0
        features[771] = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0
        features[772] = 1 if board.has_queenside_castling_rights(chess.BLACK) else 0
        
        return features
    
    def _parse_pgn_moves(self, pgn_string: str) -> List[str]:
        """Parse moves from PGN string"""
        # Simplified PGN parsing - would need proper PGN parser
        moves = []
        lines = pgn_string.strip().split('\n')
        
        for line in lines:
            if line.startswith('1.'):
                # Extract moves from game notation
                tokens = line.split()
                for token in tokens:
                    if not token[0].isdigit() and token not in ['1-0', '0-1', '1/2-1/2']:
                        moves.append(token)
        
        return moves
    
    def _extract_game_outcome(self, pgn_string: str) -> float:
        """Extract game outcome from PGN"""
        if '1-0' in pgn_string:
            return 1.0  # White wins
        elif '0-1' in pgn_string:
            return -1.0  # Black wins
        else:
            return 0.0  # Draw
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config.update(checkpoint.get('config', {}))
        print(f"Model loaded from {filepath}")


def main():
    """Test the neural network integration"""
    print("V7P3R Chess AI 2.0 - Neural Network Integration Test")
    print("=" * 60)
    
    # Initialize trainer
    trainer = NeuralNetworkTrainer()
    
    # Test position evaluation
    test_board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2")
    
    print("\nTesting position evaluation...")
    start_time = time.time()
    result = trainer.evaluate_position(test_board)
    elapsed = time.time() - start_time
    
    print(f"Position evaluated in {elapsed:.3f}s")
    print(f"Best move: {result['best_move']}")
    print(f"Best evaluation: {result['best_evaluation']:.3f}")
    
    if result['move_evaluations']:
        print("\nTop 5 move evaluations:")
        for i, move_eval in enumerate(result['move_evaluations'][:5]):
            print(f"{i+1}. {move_eval['move']}: eval={move_eval['evaluation']:.3f}, "
                  f"quality={move_eval['quality']:.3f}, combined={move_eval['combined_score']:.3f}")
    
    # Test training data preparation (with dummy data)
    print("\nTesting training data preparation...")
    dummy_games = [
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0",
        "1. d4 d5 2. c4 e6 3. Nc3 Nf6 0-1"
    ]
    
    training_data = trainer.prepare_training_data(dummy_games, max_positions=5)
    print(f"Generated {len(training_data)} training samples")
    
    # Test batch training
    if len(training_data) >= trainer.config['batch_size']:
        print("\nTesting batch training...")
        losses = trainer.train_batch(training_data)
        print(f"Training losses: {losses}")
    
    print("\nNeural network integration test completed successfully!")


if __name__ == "__main__":
    main()
