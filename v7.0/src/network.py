"""
V7P3R v7.0 - Neural Network Architecture

Single unified value network that learns chess through comprehensive features.
Takes 51-dimensional feature vector, outputs position evaluation [-1, 1].

Architecture:
- Input: 51 features (Stage 1 + Heuristics + Complexity)
- Hidden: [256, 128, 64] with BatchNorm, ReLU, Dropout
- Output: 1 value (Tanh) - position quality from mover's perspective

Training:
- Self-play with Stockfish oracle
- Custom personality rewards
- Experience replay
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import json

class V7ValueNetwork(nn.Module):
    """
    Value network for position evaluation.
    
    Architecture optimized for feature-based chess learning:
    - Deeper than Stage 1 (3 hidden layers vs 2)
    - Batch normalization for training stability
    - Dropout for generalization
    - Tanh output for bounded values [-1, 1]
    """
    
    def __init__(self, input_dim: int = 55, dropout_rate: float = 0.3):
        """
        Initialize V7 value network.
        
        Args:
            input_dim: Number of input features (default 55: 51 positional + 4 temporal)
            dropout_rate: Dropout probability for regularization
        """
        super(V7ValueNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, 256)
        self.input_bn = nn.BatchNorm1d(256)
        
        # Hidden layers
        self.hidden1 = nn.Linear(256, 128)
        self.hidden1_bn = nn.BatchNorm1d(128)
        
        self.hidden2 = nn.Linear(128, 64)
        self.hidden2_bn = nn.BatchNorm1d(64)
        
        # Output layer (value head)
        self.value_head = nn.Linear(64, 1)
        
        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.tanh = nn.Tanh()
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input features (batch_size, 51)
        
        Returns:
            Position value (batch_size, 1) in range [-1, 1]
        """
        # Input layer
        x = self.input_layer(x)
        x = self.input_bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Hidden layer 1
        x = self.hidden1(x)
        x = self.hidden1_bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Hidden layer 2
        x = self.hidden2(x)
        x = self.hidden2_bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Value head (output)
        x = self.value_head(x)
        x = self.tanh(x)  # Bounded [-1, 1]
        
        return x
    
    def predict(self, features: np.ndarray) -> float:
        """
        Predict position value from numpy features (inference mode).
        
        Args:
            features: Feature vector (51,) or (batch_size, 51)
        
        Returns:
            Position value in [-1, 1]
        """
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            # Handle single sample or batch
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            x = torch.FloatTensor(features)
            value = self.forward(x)
            
            # Return scalar if single sample
            if value.shape[0] == 1:
                return value.item()
            else:
                return value.cpu().numpy().flatten()
    
    def get_config(self) -> Dict:
        """Get network configuration."""
        return {
            'architecture': 'V7ValueNetwork',
            'input_dim': self.input_dim,
            'hidden_dims': [256, 128, 64],
            'output_dim': 1,
            'dropout_rate': self.dropout_rate,
            'activation': 'ReLU',
            'output_activation': 'Tanh',
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class V7Trainer:
    """
    Training manager for V7 value network.
    
    Handles:
    - Training loop with experience replay
    - Checkpoint saving/loading
    - Learning rate scheduling
    - Training metrics tracking
    """
    
    def __init__(
        self,
        network: V7ValueNetwork,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        device: str = 'cpu'
    ):
        """
        Initialize trainer.
        
        Args:
            network: V7ValueNetwork to train
            learning_rate: Initial learning rate
            weight_decay: L2 regularization strength
            device: 'cpu' or 'cuda'
        """
        self.network = network
        self.device = torch.device(device)
        self.network.to(self.device)
        
        # Optimizer and loss
        self.optimizer = optim.Adam(
            network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()  # Mean squared error for value prediction
        
        # Learning rate scheduler (reduce on plateau)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch': []
        }
    
    def train_epoch(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        batch_size: int = 256,
        shuffle: bool = True
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            features: Training features (N, 51)
            targets: Target values (N,) in [-1, 1]
            batch_size: Batch size for training
            shuffle: Whether to shuffle data
        
        Returns:
            Average training loss
        """
        self.network.train()
        
        # Convert to tensors
        X = torch.FloatTensor(features).to(self.device)
        y = torch.FloatTensor(targets).reshape(-1, 1).to(self.device)
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
        
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_X, batch_y in dataloader:
            # Forward pass
            predictions = self.network(batch_X)
            loss = self.criterion(predictions, batch_y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        return epoch_loss / num_batches
    
    def validate(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        batch_size: int = 256
    ) -> float:
        """
        Validate on held-out data.
        
        Args:
            features: Validation features (N, 51)
            targets: Target values (N,)
            batch_size: Batch size for validation
        
        Returns:
            Validation loss
        """
        self.network.eval()
        
        X = torch.FloatTensor(features).to(self.device)
        y = torch.FloatTensor(targets).reshape(-1, 1).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                predictions = self.network(batch_X)
                loss = self.criterion(predictions, batch_y)
                val_loss += loss.item()
                num_batches += 1
        
        return val_loss / num_batches
    
    def fit(
        self,
        train_features: np.ndarray,
        train_targets: np.ndarray,
        val_features: Optional[np.ndarray] = None,
        val_targets: Optional[np.ndarray] = None,
        epochs: int = 20,
        batch_size: int = 256,
        save_path: Optional[str] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Full training loop with validation and checkpointing.
        
        Args:
            train_features: Training features (N, 51)
            train_targets: Training targets (N,)
            val_features: Validation features (optional)
            val_targets: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            save_path: Path to save best model
            verbose: Print progress
        
        Returns:
            Training history dictionary
        """
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(
                train_features,
                train_targets,
                batch_size=batch_size
            )
            
            # Validation
            if val_features is not None and val_targets is not None:
                val_loss = self.validate(val_features, val_targets, batch_size)
                self.scheduler.step(val_loss)  # Update learning rate
                
                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Save best model
                if val_loss < best_val_loss and save_path is not None:
                    best_val_loss = val_loss
                    self.save_checkpoint(save_path, epoch, val_loss)
                
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f} - "
                          f"Val Loss: {val_loss:.4f} - "
                          f"LR: {current_lr:.6f}")
                
                # Update history
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['learning_rate'].append(current_lr)
                self.history['epoch'].append(epoch + 1)
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")
                
                self.history['train_loss'].append(train_loss)
                self.history['epoch'].append(epoch + 1)
        
        return self.history
    
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        val_loss: float
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.network.get_config(),
            'history': self.history
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> Dict:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        return checkpoint


def create_v7_network(
    input_dim: int = 55,  # Updated for v7.2 temporal features
    dropout_rate: float = 0.3,
    device: str = 'cpu'
) -> Tuple[V7ValueNetwork, V7Trainer]:
    """
    Create V7 network and trainer (convenience function).
    
    Args:
        input_dim: Number of input features (55 in v7.2: 51 positional + 4 temporal)
        dropout_rate: Dropout probability
        device: 'cpu' or 'cuda'
    
    Returns:
        (network, trainer) tuple
    """
    network = V7ValueNetwork(input_dim, dropout_rate)
    trainer = V7Trainer(network, device=device)
    return network, trainer


# Example usage and validation
if __name__ == "__main__":
    print("="*60)
    print("V7P3R v7.0 - NEURAL NETWORK ARCHITECTURE")
    print("="*60)
    
    # Create network
    network, trainer = create_v7_network()
    config = network.get_config()
    
    print(f"\n📊 Network Configuration:")
    print(f"  Architecture: {config['architecture']}")
    print(f"  Input Dimension: {config['input_dim']}")
    print(f"  Hidden Layers: {config['hidden_dims']}")
    print(f"  Output Dimension: {config['output_dim']}")
    print(f"  Dropout Rate: {config['dropout_rate']}")
    print(f"  Total Parameters: {config['total_parameters']:,}")
    print(f"  Trainable Parameters: {config['trainable_parameters']:,}")
    
    # Test forward pass
    print(f"\n🧪 Testing Forward Pass...")
    test_features = np.random.randn(10, 51).astype(np.float32)
    test_output = network.predict(test_features)
    
    print(f"  Input shape: {test_features.shape}")
    print(f"  Output shape: {test_output.shape}")
    print(f"  Output range: [{test_output.min():.3f}, {test_output.max():.3f}]")
    print(f"  Expected range: [-1.0, 1.0]")
    
    # Test single prediction
    single_features = np.random.randn(51).astype(np.float32)
    single_output = network.predict(single_features)
    print(f"\n  Single prediction: {single_output:.4f}")
    
    # Test training step
    print(f"\n🎯 Testing Training Step...")
    dummy_features = np.random.randn(100, 51).astype(np.float32)
    dummy_targets = np.random.randn(100).astype(np.float32)
    dummy_targets = np.tanh(dummy_targets)  # Bound to [-1, 1]
    
    loss = trainer.train_epoch(dummy_features, dummy_targets, batch_size=32)
    print(f"  Training loss (random data): {loss:.4f}")
    
    print(f"\n✅ Network architecture validated!")
    print(f"\n📝 Next Steps:")
    print(f"  1. Integrate Stockfish oracle (src/v7/stockfish_oracle.py)")
    print(f"  2. Build self-play trainer (src/v7/selfplay_trainer.py)")
    print(f"  3. Define personality rewards (src/v7/personality_rewards.py)")
    print(f"  4. Generate initial training data from self-play")
