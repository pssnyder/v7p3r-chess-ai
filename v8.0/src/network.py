"""
V7P3R v8.0 - Simplified Value Network

MUCH SIMPLER than v7.0 - No personality, no phase-aware logic.
Just: Features → Value estimation

The reward shaper learns the complexity, not hand-coded rules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class V8ValueNetwork(nn.Module):
    """
    Simplified value network for v8.0
    
    Input: 55-dim features from ComprehensiveFeatureExtractor
    Output: Position evaluation (-1 to +1)
    
    NO personality rewards, NO phase-aware logic
    Pure feature → value mapping learned through self-play
    """
    
    def __init__(self, input_dim: int = 55, dropout_rate: float = 0.3):
        """
        Args:
            input_dim: Number of input features (55 for v8.0)
            dropout_rate: Dropout probability for regularization
        """
        super().__init__()
        
        self.input_dim = input_dim
        
        # Simple architecture: 55 → 256 → 128 → 64 → 1
        self.input_layer = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.hidden1 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.hidden2 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.value_head = nn.Linear(64, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: (batch_size, 55) feature tensor
        
        Returns:
            value: (batch_size, 1) position evaluation (-1 to +1)
        """
        batch_size = x.shape[0]
        
        # Input layer
        x = self.input_layer(x)
        if batch_size > 1:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Hidden layer 1
        x = self.hidden1(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Hidden layer 2
        x = self.hidden2(x)
        if batch_size > 1:
            x = self.bn3(x)
        x = F.relu(x)
        
        # Value head
        value = self.value_head(x)
        value = torch.tanh(value)  # Bound between -1 and +1
        
        return value


class V8NetworkTrainer:
    """
    Trainer for V8ValueNetwork
    
    Trains on pure win/loss outcomes from self-play
    """
    
    def __init__(self, network: V8ValueNetwork, lr: float = 0.001, device: str = 'cpu'):
        """
        Args:
            network: V8ValueNetwork to train
            lr: Learning rate
            device: 'cpu' or 'cuda'
        """
        self.network = network.to(device)
        self.device = device
        
        self.optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        self.training_history = []
    
    def train_on_batch(self, 
                       features_batch: torch.Tensor,
                       targets_batch: torch.Tensor) -> float:
        """
        Train on batch of experiences
        
        Args:
            features_batch: (batch_size, 55) features
            targets_batch: (batch_size, 1) target values (game results)
        
        Returns:
            loss: Training loss
        """
        features_batch = features_batch.to(self.device)
        targets_batch = targets_batch.to(self.device)
        
        # Forward pass
        predictions = self.network(features_batch)
        
        # Calculate loss
        loss = self.loss_fn(predictions, targets_batch)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        loss_value = loss.item()
        self.training_history.append(loss_value)
        
        return loss_value
    
    def get_average_loss(self, last_n: int = 100) -> float:
        """Get average loss over last N batches"""
        if len(self.training_history) == 0:
            return 0.0
        recent = self.training_history[-last_n:]
        return sum(recent) / len(recent)


def create_v8_network(input_dim: int = 55, 
                      dropout_rate: float = 0.3,
                      lr: float = 0.001,
                      device: str = 'cpu') -> tuple:
    """
    Create V8 network and trainer
    
    Args:
        input_dim: Number of input features
        dropout_rate: Dropout probability
        lr: Learning rate
        device: 'cpu' or 'cuda'
    
    Returns:
        (network, trainer): Tuple of network and trainer
    """
    network = V8ValueNetwork(input_dim=input_dim, dropout_rate=dropout_rate)
    trainer = V8NetworkTrainer(network, lr=lr, device=device)
    
    return network, trainer


def count_parameters(network: nn.Module) -> int:
    """Count trainable parameters in network"""
    return sum(p.numel() for p in network.parameters() if p.requires_grad)


def test_v8_network():
    """Test V8 network functionality"""
    print("Testing V8 Value Network...")
    
    # Create network
    network, trainer = create_v8_network(input_dim=55, lr=0.001)
    
    print(f"✓ Created network with {count_parameters(network):,} parameters")
    print(f"  Architecture: 55 → 256 → 128 → 64 → 1")
    
    # Test forward pass
    batch_size = 32
    test_features = torch.randn(batch_size, 55)
    
    with torch.no_grad():
        predictions = network(test_features)
    
    print(f"✓ Forward pass: {test_features.shape} → {predictions.shape}")
    print(f"  Value range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    
    # Test training
    test_targets = torch.randn(batch_size, 1)
    loss = trainer.train_on_batch(test_features, test_targets)
    
    print(f"✓ Training step: loss = {loss:.4f}")
    
    print("\n✓ V8 network tests passed!")


if __name__ == '__main__':
    test_v8_network()
