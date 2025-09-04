"""
V7P3R AI v3.0 - "Thinking Brain" GRU Neural Network
==================================================

The core neural network that processes chess positions as "frames" and generates
move candidates through pure pattern discovery. No human chess knowledge - only
mathematical relationships learned through self-play.

Architecture:
- 8-layer GRU (Gated Recurrent Unit)
- 256 neurons per layer
- CUDA GPU accelerated
- Input: ChessState feature vector (~690 dimensions)
- Output: Move candidate probabilities (~4096 possible moves)
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
import chess
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThinkingBrain(nn.Module):
    """
    GRU-based neural network that learns chess patterns autonomously
    
    The "Thinking Brain" processes chess positions and generates move candidates
    based on patterns discovered through self-play training.
    """
    
    def __init__(
        self,
        input_size: int = 690,           # ChessState feature vector size
        hidden_size: int = 256,          # Neurons per GRU layer
        num_layers: int = 8,             # Number of GRU layers
        output_size: int = 4096,         # Maximum possible chess moves
        dropout: float = 0.1,            # Dropout for regularization
        device: Optional[str] = None
    ):
        super(ThinkingBrain, self).__init__()
        
        # Auto-detect CUDA if available
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # Input processing layer
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        
        # Main GRU layers for sequence processing
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Output processing layers
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
        
        # Move to GPU if available
        self.to(self.device)
        
        logger.info(f"ThinkingBrain initialized on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(
        self, 
        position_features: torch.Tensor, 
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Thinking Brain
        
        Args:
            position_features: Chess position features [batch_size, input_size]
            hidden_state: Previous GRU hidden state [num_layers, batch_size, hidden_size]
            
        Returns:
            move_logits: Raw move candidate scores [batch_size, output_size]
            new_hidden_state: Updated GRU hidden state
        """
        batch_size = position_features.size(0)
        
        # Project input features to hidden dimension
        x = self.input_projection(position_features)
        x = self.input_norm(x)
        x = F.relu(x)
        
        # Add sequence dimension for GRU (single time step)
        x = x.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # Process through GRU layers
        gru_output, new_hidden_state = self.gru(x, hidden_state)
        
        # Remove sequence dimension
        gru_output = gru_output.squeeze(1)  # [batch_size, hidden_size]
        
        # Generate move candidate logits
        move_logits = self.output_projection(gru_output)
        
        return move_logits, new_hidden_state
    
    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """Initialize hidden state for new game"""
        return torch.zeros(
            self.num_layers, 
            batch_size, 
            self.hidden_size,
            device=self.device,
            dtype=torch.float32
        )
    
    def generate_move_candidates(
        self, 
        position_features: torch.Tensor,
        legal_moves: List[chess.Move],
        hidden_state: Optional[torch.Tensor] = None,
        top_k: int = 10
    ) -> Tuple[List[chess.Move], torch.Tensor, torch.Tensor]:
        """
        Generate top-K move candidates for a chess position
        
        Args:
            position_features: Chess position features
            legal_moves: List of legal moves in current position
            hidden_state: Previous hidden state
            top_k: Number of top candidates to return
            
        Returns:
            top_moves: List of top-K move candidates
            move_probabilities: Probabilities for top moves
            new_hidden_state: Updated hidden state
        """
        self.eval()
        
        with torch.no_grad():
            # Forward pass
            move_logits, new_hidden_state = self.forward(position_features, hidden_state)
            
            # Convert to probabilities
            move_probs = F.softmax(move_logits, dim=-1)
            
            # Filter to only legal moves
            legal_move_indices = [self._move_to_index(move) for move in legal_moves]
            legal_move_probs = move_probs[0, legal_move_indices]  # Assuming batch_size=1
            
            # Get top-K legal moves
            top_k = min(top_k, len(legal_moves))
            top_indices = torch.topk(legal_move_probs, top_k).indices
            
            top_moves = [legal_moves[idx] for idx in top_indices]
            top_probabilities = legal_move_probs[top_indices]
            
            return top_moves, top_probabilities, new_hidden_state
    
    def _move_to_index(self, move: chess.Move) -> int:
        """
        Convert chess move to index in output vector
        
        Simple encoding: from_square * 64 + to_square
        For promotions, add offset based on promotion piece
        """
        base_index = move.from_square * 64 + move.to_square
        
        if move.promotion:
            # Add promotion piece offset (knight=1, bishop=2, rook=3, queen=4)
            promotion_offset = move.promotion - 1  # chess pieces are 1-indexed
            base_index += 4096 + (promotion_offset * 64 * 64)
        
        return min(base_index, self.output_size - 1)
    
    def _index_to_move(self, index: int) -> chess.Move:
        """Convert index back to chess move"""
        if index >= 4096:
            # Promotion move
            promotion_index = index - 4096
            promotion_piece = (promotion_index // (64 * 64)) + 1
            base_index = promotion_index % (64 * 64)
            from_square = base_index // 64
            to_square = base_index % 64
            return chess.Move(from_square, to_square, promotion=promotion_piece)
        else:
            # Regular move
            from_square = index // 64
            to_square = index % 64
            return chess.Move(from_square, to_square)
    
    def save_model(self, filepath: str):
        """Save model state to file"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.output_size,
            'device': str(self.device)
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str, device: Optional[str] = None):
        """Load model from file"""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        model = cls(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers'],
            output_size=checkpoint['output_size'],
            device=device
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {filepath}")
        return model


class PositionMemory:
    """
    Manages hidden state memory across moves in a game with dynamic pruning
    
    The GRU maintains memory of previous positions but dynamically prunes
    less relevant history based on game status trends and outcomes.
    """
    
    def __init__(self, model: ThinkingBrain, memory_config: Optional[Dict] = None):
        self.model = model
        self.hidden_state = None
        self.move_history = []
        self.position_history = []
        
        # Enhanced memory management
        self.position_evaluations = []  # Track position evaluations over time
        self.memory_weights = []        # Weight importance of each position
        self.game_status_trend = []     # Track improving/declining game status
        
        # Memory management parameters
        config = memory_config or {}
        self.max_memory_positions = config.get('max_memory_positions', 20)
        self.memory_decay_factor = config.get('memory_decay_factor', 0.95)
        self.status_trend_window = config.get('status_trend_window', 5)
        self.pruning_threshold = config.get('pruning_threshold', 0.3)
        self.critical_memory_boost = config.get('critical_memory_boost', 1.5)
    
    def start_new_game(self):
        """Reset memory for new game"""
        self.hidden_state = None
        self.move_history.clear()
        self.position_history.clear()
        self.position_evaluations.clear()
        self.memory_weights.clear()
        self.game_status_trend.clear()
        logger.debug("Started new game - memory reset")
    
    def _evaluate_position_status(self, position_features: torch.Tensor) -> float:
        """
        Evaluate current position status (simplified evaluation)
        Returns a score between -1.0 (very bad) and 1.0 (very good)
        """
        # Use a simple heuristic based on piece values and position
        # This is a placeholder - in practice you might use a more sophisticated evaluator
        features = position_features.cpu().numpy().flatten()
        
        # Simple material balance + positional factors
        material_balance = features[32:64].sum() - features[64:96].sum()  # Approximate material
        positional_factors = features[96:128].mean()  # Approximate positional features
        
        # Normalize to [-1, 1] range
        status = np.tanh(material_balance * 0.1 + positional_factors * 0.5)
        return float(status)
    
    def _update_memory_weights(self, current_status: float):
        """Update memory weights based on game status trends"""
        self.game_status_trend.append(current_status)
        
        # Keep only recent trend history
        if len(self.game_status_trend) > self.status_trend_window:
            self.game_status_trend.pop(0)
        
        # Calculate trend direction (improving vs declining)
        if len(self.game_status_trend) >= 2:
            recent_trend = np.mean(self.game_status_trend[-3:]) if len(self.game_status_trend) >= 3 else self.game_status_trend[-1]
            earlier_trend = np.mean(self.game_status_trend[:-2]) if len(self.game_status_trend) >= 3 else self.game_status_trend[0]
            trend_direction = recent_trend - earlier_trend
            
            # Boost memory weights for positions that led to improvement
            if trend_direction > 0:  # Game status improving
                # Boost recent memories that contributed to improvement
                boost_range = min(3, len(self.memory_weights))
                for i in range(1, boost_range + 1):
                    if len(self.memory_weights) >= i:
                        self.memory_weights[-i] *= self.critical_memory_boost
            else:  # Game status declining
                # Decay memory weights for positions that led to decline
                decay_range = min(2, len(self.memory_weights))
                for i in range(1, decay_range + 1):
                    if len(self.memory_weights) >= i:
                        self.memory_weights[-i] *= self.memory_decay_factor
    
    def _prune_irrelevant_memory(self):
        """Prune memory positions with low weights to maintain efficiency"""
        if len(self.position_history) <= self.max_memory_positions:
            return
        
        # Calculate retention threshold
        if len(self.memory_weights) > 0:
            weight_threshold = np.percentile(self.memory_weights, self.pruning_threshold * 100)
            
            # Keep positions with weights above threshold and recent positions
            keep_indices = []
            for i, weight in enumerate(self.memory_weights):
                # Always keep recent positions (last 5)
                if i >= len(self.memory_weights) - 5:
                    keep_indices.append(i)
                # Keep positions with high importance weights
                elif weight >= weight_threshold:
                    keep_indices.append(i)
            
            # Prune less important positions
            if len(keep_indices) < len(self.position_history):
                self.position_history = [self.position_history[i] for i in keep_indices]
                self.move_history = [self.move_history[i] for i in keep_indices if i < len(self.move_history)]
                self.position_evaluations = [self.position_evaluations[i] for i in keep_indices]
                self.memory_weights = [self.memory_weights[i] for i in keep_indices]
                
                logger.debug(f"Pruned memory: kept {len(keep_indices)} of {len(self.memory_weights)} positions")
    
    def process_position(
        self, 
        position_features: torch.Tensor,
        legal_moves: List[chess.Move],
        top_k: int = 10
    ) -> Tuple[List[chess.Move], torch.Tensor]:
        """
        Process position with enhanced memory management
        
        Args:
            position_features: Current position features
            legal_moves: Legal moves in current position
            top_k: Number of candidates to generate
            
        Returns:
            top_moves: Generated move candidates
            probabilities: Move probabilities
        """
        # Initialize hidden state for first move of game
        if self.hidden_state is None:
            self.hidden_state = self.model.init_hidden(batch_size=1)
        
        # Evaluate current position status
        current_status = self._evaluate_position_status(position_features)
        
        # Generate candidates with memory
        top_moves, probabilities, new_hidden_state = self.model.generate_move_candidates(
            position_features, legal_moves, self.hidden_state, top_k
        )
        
        # Update memory with enhanced tracking
        self.hidden_state = new_hidden_state
        self.position_history.append(position_features.cpu().numpy())
        self.position_evaluations.append(current_status)
        
        # Initialize memory weight (will be updated based on outcomes)
        initial_weight = 1.0
        self.memory_weights.append(initial_weight)
        
        # Update memory weights based on trends
        self._update_memory_weights(current_status)
        
        # Prune irrelevant memory to maintain efficiency
        self._prune_irrelevant_memory()
        
        return top_moves, probabilities
    
    def record_move(self, move: chess.Move):
        """Record the move that was actually played"""
        self.move_history.append(move)
    
    def finalize_game_memory(self, game_outcome: float):
        """
        Finalize memory weights based on game outcome
        
        Args:
            game_outcome: 1.0=win, 0.5=draw, 0.0=loss
        """
        if len(self.memory_weights) == 0:
            return
        
        # Boost all memory weights for winning games
        outcome_multiplier = 1.0
        if game_outcome >= 0.9:  # Win
            outcome_multiplier = self.critical_memory_boost
        elif game_outcome <= 0.1:  # Loss - reduce memory importance
            outcome_multiplier = self.memory_decay_factor
        # Draw maintains current weights
        
        # Apply outcome-based weighting
        self.memory_weights = [w * outcome_multiplier for w in self.memory_weights]
        
        # Boost memory for critical moments (significant status changes)
        if len(self.position_evaluations) >= 2:
            for i in range(1, len(self.position_evaluations)):
                status_change = abs(self.position_evaluations[i] - self.position_evaluations[i-1])
                if status_change > 0.3:  # Significant position change
                    if i < len(self.memory_weights):
                        self.memory_weights[i] *= (1.0 + status_change)
        
        logger.debug(f"Finalized game memory: outcome={game_outcome:.2f}, "
                    f"avg_weight={np.mean(self.memory_weights):.3f}, "
                    f"positions_kept={len(self.position_history)}")
    
    def get_game_memory(self) -> Dict:
        """Get complete game memory for training"""
        return {
            'moves': self.move_history.copy(),
            'positions': self.position_history.copy(),
            'final_hidden_state': self.hidden_state.cpu().numpy() if self.hidden_state is not None else None
        }


# Training utilities
class ThinkingBrainTrainer:
    """Handles training of the Thinking Brain through self-play"""
    
    def __init__(
        self, 
        model: ThinkingBrain,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Training metrics
        self.training_losses = []
        self.games_trained = 0
    
    def train_on_game(
        self, 
        game_positions: List[torch.Tensor],
        game_moves: List[chess.Move],
        game_outcome: float  # 1.0=win, 0.5=draw, 0.0=loss
    ):
        """
        Train the model on a completed game
        
        Args:
            game_positions: List of position features from the game
            game_moves: List of moves played in the game
            game_outcome: Final game result for the side that played
        """
        self.model.train()
        
        total_loss = 0.0
        hidden_state = self.model.init_hidden(batch_size=1)
        
        for position_features, move_played in zip(game_positions, game_moves):
            # Forward pass
            move_logits, hidden_state = self.model.forward(position_features, hidden_state)
            
            # Convert move to target index
            target_index = self.model._move_to_index(move_played)
            target = torch.tensor([target_index], device=self.model.device)
            
            # Calculate loss with outcome weighting
            loss = self.criterion(move_logits, target) * game_outcome
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(game_positions) if game_positions else 0
        self.training_losses.append(avg_loss)
        self.games_trained += 1
        
        if self.games_trained % 100 == 0:
            logger.info(f"Trained on {self.games_trained} games, avg loss: {avg_loss:.4f}")
        
        return avg_loss


if __name__ == "__main__":
    # Test the Thinking Brain
    logger.info("Testing Thinking Brain...")
    
    # Create model
    brain = ThinkingBrain()
    
    # Test with dummy data
    dummy_features = torch.randn(1, 690).to(brain.device)
    dummy_moves = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")]
    
    # Generate candidates
    candidates, probs, hidden = brain.generate_move_candidates(
        dummy_features, dummy_moves, top_k=2
    )
    
    logger.info(f"Generated {len(candidates)} move candidates")
    logger.info(f"Model memory usage: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    
    # Test memory system
    memory = PositionMemory(brain)
    memory.start_new_game()
    
    moves, probs = memory.process_position(dummy_features, dummy_moves)
    logger.info(f"Memory system working: {len(moves)} candidates generated")
    
    logger.info("Thinking Brain test completed! 🧠✅")
