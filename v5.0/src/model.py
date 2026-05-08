"""
V7P3R AI v5.0 - Neural Network Model
Dual-head architecture for move quality classification and position evaluation

Architecture designed for future expansion:
- Deep network (256→256→128→64) accommodates feature growth
- Residual connections enable smooth gradient flow
- Batch normalization for stable training
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual block with batch normalization
    
    Implements: out = ReLU(BatchNorm(Linear(x))) + projection(x)
    
    The skip connection (projection) allows gradients to flow directly through
    the network, preventing vanishing gradients in deep architectures.
    """
    
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super(ResidualBlock, self).__init__()
        
        # Main transformation path
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Projection layer for dimension mismatch
        # (e.g., 256 → 128 needs projection to add residual)
        self.projection = None
        if in_dim != out_dim:
            self.projection = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        identity = x
        
        # Main path: Linear → BatchNorm → ReLU → Dropout
        out = self.linear(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Residual connection (skip connection)
        if self.projection is not None:
            identity = self.projection(identity)
        
        out = out + identity  # Element-wise addition
        return out


class V7P3R_AI_v5(nn.Module):
    """
    V7P3R AI v5.0 - Dual-head neural network
    
    Learns chess move quality and position evaluation from V7P3R game history.
    Uses supervised learning with Stockfish-graded positions.
    
    Architecture:
        Input (26 features) 
          → Shared Embedding (256→256→128→64 with residuals)
          → Policy Head (6-class move quality)
          → Value Head (position evaluation)
    
    Key Features:
        - Residual connections for deep gradient flow
        - Batch normalization for stable training
        - Designed for feature expansion (26 → 40+ features)
        - Dual-head output for multi-task learning
    """
    
    def __init__(self, 
                 input_dim=26,
                 shared_dims=[256, 256, 128, 64],
                 policy_hidden=64,
                 value_hidden=32,
                 dropout=0.3,
                 use_residuals=True):
        """
        Initialize V7P3R AI v5.0 model
        
        Args:
            input_dim: Number of input features (default: 26, expandable to 40+)
            shared_dims: Hidden layer dimensions for shared embedding
            policy_hidden: Hidden layer size for policy head
            value_hidden: Hidden layer size for value head
            dropout: Dropout probability for regularization
            use_residuals: Enable residual connections (recommended: True)
        """
        super(V7P3R_AI_v5, self).__init__()
        
        self.use_residuals = use_residuals
        self.input_dim = input_dim
        self.shared_dims = shared_dims
        
        # Initial projection to first shared dimension
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, shared_dims[0]),
            nn.BatchNorm1d(shared_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Shared embedding network
        if use_residuals:
            # Use residual blocks for better gradient flow
            self.shared_blocks = nn.ModuleList([
                ResidualBlock(shared_dims[i], shared_dims[i+1], dropout)
                for i in range(len(shared_dims) - 1)
            ])
        else:
            # Fallback to sequential (for ablation studies)
            layers = []
            for i in range(len(shared_dims) - 1):
                layers.extend([
                    nn.Linear(shared_dims[i], shared_dims[i+1]),
                    nn.BatchNorm1d(shared_dims[i+1]),
                    nn.ReLU(),
                    nn.Dropout(dropout if i < len(shared_dims) - 2 else dropout * 0.7)
                ])
            self.shared_sequential = nn.Sequential(*layers)
        
        # Policy head - Move quality classification (0-5 grades)
        self.policy = nn.Sequential(
            nn.Linear(shared_dims[-1], policy_hidden),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(policy_hidden, 6)  # 6 classes: grades 0-5
        )
        
        # Value head - Position evaluation regression
        self.value = nn.Sequential(
            nn.Linear(shared_dims[-1], value_hidden),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(value_hidden, 1),
            nn.Tanh()  # Bound output to [-1, 1]
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using He initialization for ReLU"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through dual-head network
        
        Args:
            x: Input features (batch_size, input_dim)
               - 26 features currently (expandable to 40+)
               - Preprocessed: normalized numericals + one-hot categoricals
        
        Returns:
            policy_logits: (batch_size, 6) - unnormalized class scores
                          Apply softmax for probabilities
            value: (batch_size, 1) - position evaluation in [-1, 1]
                   -1 = Black winning, 0 = equal, +1 = White winning
        """
        # Project input to shared dimension
        x = self.input_proj(x)
        
        # Shared embedding with residual connections
        if self.use_residuals:
            for block in self.shared_blocks:
                x = block(x)
        else:
            x = self.shared_sequential(x)
        
        # Dual heads
        policy_logits = self.policy(x)  # Raw scores for CrossEntropyLoss
        value = self.value(x)            # Bounded by Tanh [-1, 1]
        
        return policy_logits, value
    
    def predict_move_quality(self, x):
        """
        Predict move quality grade with probabilities
        
        Args:
            x: Input features (batch_size, input_dim)
        
        Returns:
            grades: (batch_size,) - predicted grades (0-5)
            probs: (batch_size, 6) - probability distribution over grades
        """
        policy_logits, _ = self.forward(x)
        probs = torch.softmax(policy_logits, dim=1)
        grades = torch.argmax(probs, dim=1)
        return grades, probs
    
    def predict_position_eval(self, x):
        """
        Predict position evaluation
        
        Args:
            x: Input features (batch_size, input_dim)
        
        Returns:
            eval_cp: (batch_size,) - position evaluation in centipawns
                     Denormalized from [-1, 1] to centipawns
        """
        _, value = self.forward(x)
        # Denormalize from [-1, 1] to centipawns
        eval_cp = value.squeeze() * 10000  # Range: [-10000, 10000] cp
        return eval_cp
    
    def count_parameters(self):
        """Count total and trainable parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'non_trainable': total - trainable
        }
    
    def get_model_summary(self):
        """Get detailed model summary"""
        params = self.count_parameters()
        return {
            'model_name': 'V7P3R_AI_v5',
            'input_dim': self.input_dim,
            'shared_dims': self.shared_dims,
            'use_residuals': self.use_residuals,
            'total_params': params['total'],
            'trainable_params': params['trainable'],
            'param_size_mb': params['total'] * 4 / (1024 * 1024),  # Assuming float32
        }


def create_model(config):
    """
    Factory function to create model from config dictionary
    
    Args:
        config: Dictionary with model hyperparameters
                {
                    'input_dim': 26,
                    'shared_dims': [256, 256, 128, 64],
                    'policy_hidden': 64,
                    'value_hidden': 32,
                    'dropout': 0.3,
                    'use_residuals': True
                }
    
    Returns:
        model: V7P3R_AI_v5 instance
    """
    model = V7P3R_AI_v5(
        input_dim=config.get('input_dim', 26),
        shared_dims=config.get('shared_dims', [256, 256, 128, 64]),
        policy_hidden=config.get('policy_hidden', 64),
        value_hidden=config.get('value_hidden', 32),
        dropout=config.get('dropout', 0.3),
        use_residuals=config.get('use_residuals', True)
    )
    return model


if __name__ == '__main__':
    # Test model instantiation
    print("=" * 80)
    print("V7P3R AI v5.0 - Model Test")
    print("=" * 80)
    
    # Create model
    model = V7P3R_AI_v5(
        input_dim=26,
        shared_dims=[256, 256, 128, 64],
        policy_hidden=64,
        value_hidden=32,
        dropout=0.3,
        use_residuals=True
    )
    
    # Print summary
    summary = model.get_model_summary()
    print(f"\nModel: {summary['model_name']}")
    print(f"Input dimension: {summary['input_dim']}")
    print(f"Shared layers: {summary['shared_dims']}")
    print(f"Residual connections: {summary['use_residuals']}")
    print(f"\nTotal parameters: {summary['total_params']:,}")
    print(f"Trainable parameters: {summary['trainable_params']:,}")
    print(f"Model size: {summary['param_size_mb']:.2f} MB")
    
    # Test forward pass
    print("\n" + "-" * 80)
    print("Testing forward pass...")
    
    batch_size = 4
    x = torch.randn(batch_size, 26)  # Random input
    
    policy_logits, value = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Policy logits shape: {policy_logits.shape}")
    print(f"Value shape: {value.shape}")
    
    # Test prediction methods
    grades, probs = model.predict_move_quality(x)
    eval_cp = model.predict_position_eval(x)
    
    print(f"\nPredicted grades: {grades}")
    print(f"Grade probabilities shape: {probs.shape}")
    print(f"Position evaluations (cp): {eval_cp}")
    
    print("\n" + "=" * 80)
    print("✅ Model test complete!")
    print("=" * 80)
