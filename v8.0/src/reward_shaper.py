"""
V7P3R v8.0 - Reward Shaper

Learns which feature patterns correlate with winning games.
Discovers optimal feature weights through meta-learning instead of hand-coding.

KEY INNOVATION: Model discovers what's "good" vs "bad" through win/loss experience,
not through human-engineered reward functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class RewardShaper(nn.Module):
    """
    Meta-learner that discovers which evaluation features matter most.
    
    Learns position-dependent reward weights:
    - Opening: Prioritize development, center control
    - Middlegame: Prioritize tactics, king safety
    - Endgame: Prioritize pawn promotion, tablebase conversion
    
    NO HAND-CODED REWARDS - Pure correlation discovery through win/loss.
    """
    
    def __init__(self, feature_dim: int = 55, num_feature_groups: int = 10):
        """
        Args:
            feature_dim: Number of input features (55 from ComprehensiveFeatureExtractor)
            num_feature_groups: Number of feature groups to learn weights for
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_feature_groups = num_feature_groups
        
        # Deep network for complex pattern recognition
        # Architecture: 55 → 256 → 128 → 64 → 10 weights
        self.fc1 = nn.Linear(feature_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        
        # Output: Feature group weights (sum to 1.0)
        self.weight_head = nn.Linear(64, num_feature_groups)
        
        # Feature group definitions (which raw features belong to which group)
        self.feature_groups = self._define_feature_groups()
        
    def _define_feature_groups(self) -> Dict[str, List[int]]:
        """
        Define which raw features belong to which conceptual groups.
        
        Groups correspond to chess concepts:
        - Material: Piece counts, material balance
        - Mobility: Legal moves, piece activity
        - King Safety: King pawn shield, king tropism
        - Pawn Structure: Passed pawns, doubled pawns, isolated pawns
        - Complexity: Forest darkness, tactical density
        - Development: Development score, castling
        - Center Control: Center pawns, center pieces
        - Piece Coordination: Rook/knight/bishop placement
        - Endgame: Pawn promotion potential, opposition
        - Temporal: Move urgency, time pressure
        
        Returns:
            Dictionary mapping group name -> list of feature indices
        """
        return {
            'material': [0, 1, 2, 3, 4, 5, 6],  # Piece counts + material balance
            'mobility': [7, 19, 20, 21],  # Legal moves, attacked squares
            'king_safety': [22, 23, 24],  # King pawn shield, king tropism
            'pawn_structure': [25, 26, 27, 28, 29],  # Passed, doubled, isolated pawns
            'complexity': [43, 44, 45, 46],  # Darkness, tactical density, diversity
            'development': [30, 31, 8, 9],  # Development, castling
            'center_control': [32, 33],  # Center pawns/pieces
            'piece_coordination': [34, 35, 36, 37, 38],  # Piece placement
            'endgame_patterns': [47, 48, 49, 50],  # Promotion, opposition
            'temporal_urgency': [51, 52, 53, 54]  # Time management
        }
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: Features → Feature group weights
        
        Args:
            features: (batch_size, 55) position features
        
        Returns:
            weights: (batch_size, 10) feature group weights (sum to 1.0)
            weighted_value: (batch_size, 1) position evaluation using learned weights
        """
        batch_size = features.shape[0]
        
        # Deep feature extraction
        x = self.fc1(features)
        if batch_size > 1:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        if batch_size > 1:
            x = self.bn3(x)
        x = F.relu(x)
        
        # Output: Feature group weights (softmax ensures they sum to 1.0)
        weights = torch.softmax(self.weight_head(x), dim=-1)
        
        # Calculate weighted value using these weights
        weighted_value = self._apply_weights(features, weights)
        
        return weights, weighted_value
    
    def _apply_weights(self, features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Apply learned weights to feature groups to get position evaluation
        
        Args:
            features: (batch_size, 55) raw features
            weights: (batch_size, 10) feature group weights
        
        Returns:
            value: (batch_size, 1) weighted position evaluation
        """
        batch_size = features.shape[0]
        group_values = []
        
        # Calculate value contribution from each feature group
        for i, (group_name, indices) in enumerate(self.feature_groups.items()):
            # Average features in this group
            group_features = features[:, indices]
            group_value = group_features.mean(dim=1, keepdim=True)  # (batch, 1)
            group_values.append(group_value)
        
        # Stack: (batch, num_groups)
        group_values = torch.cat(group_values, dim=1)
        
        # Apply learned weights: (batch, 10) * (batch, 10) -> (batch, 1)
        weighted_value = (group_values * weights).sum(dim=1, keepdim=True)
        
        # Tanh to bound between -1 and +1
        weighted_value = torch.tanh(weighted_value)
        
        return weighted_value
    
    def interpret_weights(self, weights: torch.Tensor) -> Dict[str, float]:
        """
        Convert weight tensor to interpretable dictionary
        
        Args:
            weights: (10,) feature group weights
        
        Returns:
            Dictionary mapping group name -> weight value
        """
        weights_np = weights.detach().cpu().numpy()
        
        interpretation = {}
        for i, group_name in enumerate(self.feature_groups.keys()):
            interpretation[group_name] = float(weights_np[i])
        
        return interpretation


class RewardShapingTrainer:
    """
    Trains reward shaper through meta-learning.
    
    Updates shaper based on which feature patterns led to wins vs losses.
    """
    
    def __init__(self, shaper: RewardShaper, lr: float = 0.001):
        """
        Args:
            shaper: RewardShaper network
            lr: Learning rate
        """
        self.shaper = shaper
        self.optimizer = torch.optim.Adam(shaper.parameters(), lr=lr)
        self.loss_history = []
    
    def train_on_game(self, 
                      game_features: List[np.ndarray],
                      game_result: float) -> float:
        """
        Train shaper on a completed game
        
        Args:
            game_features: List of feature vectors from game (each 55-dim)
            game_result: +1.0 (win), 0.0 (draw), -1.0 (loss)
        
        Returns:
            loss: Training loss for this game
        """
        if len(game_features) == 0:
            return 0.0
        
        # Convert to tensor (num_positions, 55)
        features_tensor = torch.tensor(np.array(game_features), dtype=torch.float32)
        
        # Get shaper predictions
        weights, predicted_values = self.shaper(features_tensor)
        
        # Target: All positions in winning game should evaluate positively
        # Positions in losing game should evaluate negatively
        # This teaches shaper which feature patterns correlate with wins
        target_values = torch.full_like(predicted_values, game_result)
        
        # Loss: MSE between predicted values and game result
        loss = F.mse_loss(predicted_values, target_values)
        
        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        
        return loss_value
    
    def train_on_batch(self,
                       batch_features: torch.Tensor,
                       batch_results: torch.Tensor) -> float:
        """
        Train on batch of game experiences
        
        Args:
            batch_features: (batch_size, 55) features
            batch_results: (batch_size, 1) game results
        
        Returns:
            loss: Training loss
        """
        weights, predicted_values = self.shaper(batch_features)
        
        loss = F.mse_loss(predicted_values, batch_results)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def get_average_loss(self, last_n: int = 100) -> float:
        """Get average loss over last N games"""
        if len(self.loss_history) == 0:
            return 0.0
        recent = self.loss_history[-last_n:]
        return sum(recent) / len(recent)


def visualize_learned_weights(shaper: RewardShaper, 
                              sample_features: torch.Tensor,
                              position_desc: str = "Sample Position"):
    """
    Visualize what the shaper has learned for a given position
    
    Args:
        shaper: Trained RewardShaper
        sample_features: (55,) feature vector
        position_desc: Description of position
    """
    with torch.no_grad():
        features = sample_features.unsqueeze(0)  # Add batch dimension
        weights, value = shaper(features)
        
        interpretation = shaper.interpret_weights(weights.squeeze())
    
    print("\n" + "="*60)
    print(f"LEARNED WEIGHTS: {position_desc}")
    print("="*60)
    print(f"Position Evaluation: {value.item():+.3f}")
    print("\nFeature Group Importances:")
    
    # Sort by weight (highest first)
    sorted_groups = sorted(interpretation.items(), key=lambda x: -x[1])
    
    for group_name, weight in sorted_groups:
        bar_length = int(weight * 50)  # Scale to 50 chars max
        bar = "█" * bar_length
        print(f"  {group_name:20s} {weight:.3f} {bar}")
    
    print("="*60)


def test_reward_shaper():
    """Test reward shaper functionality"""
    print("Testing Reward Shaper...")
    
    # Create shaper
    shaper = RewardShaper(feature_dim=55, num_feature_groups=10)
    trainer = RewardShapingTrainer(shaper, lr=0.001)
    
    print(f"✓ Created shaper with {sum(p.numel() for p in shaper.parameters())} parameters")
    
    # Test forward pass
    sample_features = torch.randn(8, 55)  # Batch of 8 positions
    weights, values = shaper(sample_features)
    
    print(f"✓ Forward pass: {sample_features.shape} → weights {weights.shape}, values {values.shape}")
    
    # Test training on fake game
    fake_game_features = [np.random.randn(55) for _ in range(50)]
    fake_result = 1.0  # Win
    
    loss = trainer.train_on_game(fake_game_features, fake_result)
    print(f"✓ Training on fake game: loss = {loss:.4f}")
    
    # Visualize learned weights
    visualize_learned_weights(shaper, sample_features[0], "Opening Position")
    
    print("\n✓ Reward shaper tests passed!")


if __name__ == '__main__':
    test_reward_shaper()
