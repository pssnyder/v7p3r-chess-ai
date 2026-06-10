"""
V7P3R AI v6.1 - Position Evaluator Model
Stage 1: GOOD vs BAD Position Classifier

Simple feed-forward neural network with BatchNorm + Dropout.

Architecture:
- Input: 19 features
- Hidden: [512, 256, 128] with BatchNorm + ReLU + Dropout(0.3)
- Output: 1 (Sigmoid) - probability of "GOOD" position

Performance (trained on 1.648M positions):
- F1 Score: 87.76%
- Accuracy: 88.31%
- Precision: 92.08%
- Recall: 83.82%

Author: Pat Snyder
Created: 2026-05-31
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Union

from .feature_extractor import extract_fast_features, FEATURE_DIM


class PositionEvaluator(nn.Module):
    """
    Simple neural network for position evaluation (GOOD vs BAD classifier).
    
    Outputs probability that a position is "GOOD" (winning/favorable).
    """
    
    def __init__(
        self, 
        input_dim: int = FEATURE_DIM, 
        hidden_dims: list = [512, 256, 128], 
        dropout: float = 0.3
    ):
        """
        Initialize Position Evaluator network.
        
        Args:
            input_dim: Number of input features (default 19)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size,) with probabilities
        """
        return self.network(x).squeeze()
    
    def predict_probability(
        self, 
        features: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Predict probability that position is GOOD.
        
        Args:
            features: Feature array/tensor of shape (input_dim,)
            
        Returns:
            Probability between 0 and 1
        """
        self.eval()
        
        with torch.no_grad():
            if isinstance(features, np.ndarray):
                features = torch.FloatTensor(features)
            
            # Add batch dimension if needed
            if features.dim() == 1:
                features = features.unsqueeze(0)
            
            prob = self.forward(features)
            
            # Return scalar
            if prob.numel() == 1:
                return prob.item()
            else:
                return prob[0].item()
    
    def evaluate_fen(self, fen: str) -> Optional[float]:
        """
        Evaluate FEN string directly.
        
        Args:
            fen: FEN string
            
        Returns:
            Probability (0-1) or None if feature extraction fails
        """
        features = extract_fast_features(fen)
        
        if features is None:
            return None
        
        return self.predict_probability(features)
    
    def save(self, path: Union[str, Path]):
        """
        Save model to disk.
        
        Args:
            path: Path to save model file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout,
        }, path)
        
        print(f"Model saved to {path}")
    
    @staticmethod
    def load(
        path: Union[str, Path], 
        device: str = 'cpu'
    ) -> 'PositionEvaluator':
        """
        Load model from disk.
        
        Args:
            path: Path to model file
            device: Device to load model on ('cpu' or 'cuda')
            
        Returns:
            Loaded PositionEvaluator model
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # PyTorch 2.6+ requires weights_only=False for models with custom objects
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        # Extract architecture from checkpoint
        # Handle both old format (direct keys) and new format (config dict)
        if 'config' in checkpoint:
            config = checkpoint['config']
            input_dim = FEATURE_DIM  # Always 19 for Stage 1
            hidden_dims = config.get('hidden_dims', [512, 256, 128])
            dropout = config.get('dropout', 0.3)
        else:
            input_dim = checkpoint.get('input_dim', FEATURE_DIM)
            hidden_dims = checkpoint.get('hidden_dims', [512, 256, 128])
            dropout = checkpoint.get('dropout', 0.3)
        
        # Create model with saved architecture
        model = PositionEvaluator(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        return model


# Example usage
if __name__ == "__main__":
    # Test model creation
    model = PositionEvaluator()
    print(f"Model architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test inference on starting position
    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    prob = model.evaluate_fen(starting_fen)
    print(f"\nStarting position evaluation (untrained): {prob:.4f}")
