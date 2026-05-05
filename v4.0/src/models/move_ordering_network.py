"""
Move Ordering Neural Network

Focused architecture for learning to rank chess moves based on tactical quality.

Architecture:
- Input: Position features (690-dim) + Move encoding (3-dim)
- Output: Move quality score (0-1 for ranking)

Multi-task training:
1. Move Ranking: Learn to rank top-N moves by quality
2. Theme Classification: Identify positional themes (50 classes)

Design principles:
- Fast inference (<5ms per position)
- Transformer-style attention for move comparison
- Shared feature backbone for efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class PositionEncoder(nn.Module):
    """Encodes raw position features into rich representation"""
    
    def __init__(self, input_dim: int = 690, hidden_dim: int = 512, num_layers: int = 3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Multi-layer encoder with residual connections
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            current_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, position_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            position_features: (batch_size, 690)
        Returns:
            encoded: (batch_size, hidden_dim)
        """
        return self.encoder(position_features)


class MoveEncoder(nn.Module):
    """Encodes move information (from/to squares, promotion)"""
    
    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        
        # Embeddings for squares (0-63) and promotion (0-4)
        self.from_square_embedding = nn.Embedding(64, embedding_dim)
        self.to_square_embedding = nn.Embedding(64, embedding_dim)
        self.promotion_embedding = nn.Embedding(5, embedding_dim // 4)
        
        # Combine embeddings
        self.combiner = nn.Linear(embedding_dim * 2 + embedding_dim // 4, embedding_dim)
    
    def forward(self, moves: torch.Tensor) -> torch.Tensor:
        """
        Args:
            moves: (batch_size, max_moves, 3) - [from_square, to_square, promotion]
        Returns:
            move_embeddings: (batch_size, max_moves, embedding_dim)
        """
        from_sq = self.from_square_embedding(moves[:, :, 0])
        to_sq = self.to_square_embedding(moves[:, :, 1])
        promo = self.promotion_embedding(moves[:, :, 2])
        
        # Concatenate and combine
        combined = torch.cat([from_sq, to_sq, promo], dim=-1)
        return self.combiner(combined)


class MoveRankingHead(nn.Module):
    """Ranks moves based on position and move features"""
    
    def __init__(self, position_dim: int = 512, move_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        
        # Attention mechanism to compare moves in context of position
        self.query = nn.Linear(move_dim, hidden_dim)
        self.key = nn.Linear(position_dim, hidden_dim)
        self.value = nn.Linear(position_dim, hidden_dim)
        
        # Final scoring network
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim + move_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1] for ranking
        )
    
    def forward(self, position_encoding: torch.Tensor, move_embeddings: torch.Tensor,
                move_masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            position_encoding: (batch_size, position_dim)
            move_embeddings: (batch_size, max_moves, move_dim)
            move_masks: (batch_size, max_moves) - True for valid moves
        Returns:
            move_scores: (batch_size, max_moves) - Quality scores [0, 1]
        """
        batch_size, max_moves, move_dim = move_embeddings.shape
        
        # Compute attention: how well does each move fit the position?
        queries = self.query(move_embeddings)  # (batch, max_moves, hidden)
        
        # Expand position for attention
        position_expanded = position_encoding.unsqueeze(1).expand(-1, max_moves, -1)
        keys = self.key(position_expanded)
        values = self.value(position_expanded)
        
        # Attention scores
        attention_scores = (queries * keys).sum(dim=-1) / (keys.size(-1) ** 0.5)
        attention_scores = attention_scores.masked_fill(~move_masks, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention
        attended = attention_weights.unsqueeze(-1) * values
        
        # Combine with move embeddings for final scoring
        combined = torch.cat([attended, move_embeddings], dim=-1)
        move_scores = self.scorer(combined).squeeze(-1)
        
        # Mask invalid moves
        move_scores = move_scores.masked_fill(~move_masks, 0.0)
        
        return move_scores


class ThemeClassificationHead(nn.Module):
    """Multi-label classification for puzzle themes"""
    
    def __init__(self, position_dim: int = 512, num_themes: int = 57):  # Updated to 57 themes
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(position_dim, 384),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_themes),
            nn.Sigmoid()  # Multi-label
        )
    
    def forward(self, position_encoding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            position_encoding: (batch_size, position_dim)
        Returns:
            theme_probs: (batch_size, num_themes)
        """
        return self.classifier(position_encoding)


class MoveOrderingNetwork(nn.Module):
    """
    Complete move ordering network with multi-task learning
    
    Tasks:
    1. Move Ranking: Score each candidate move
    2. Theme Classification: Identify positional themes
    """
    
    def __init__(self,
                 position_dim: int = 690,
                 position_hidden: int = 512,
                 move_embedding_dim: int = 64,
                 num_themes: int = 57):  # Updated to match dataset's 57 themes
        super().__init__()
        
        self.position_encoder = PositionEncoder(position_dim, position_hidden)
        self.move_encoder = MoveEncoder(move_embedding_dim)
        self.ranking_head = MoveRankingHead(position_hidden, move_embedding_dim)
        self.theme_head = ThemeClassificationHead(position_hidden, num_themes)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: Dict containing:
                - position_features: (batch_size, 690)
                - moves: (batch_size, max_moves, 3)
                - move_masks: (batch_size, max_moves)
        
        Returns:
            Dict containing:
                - move_scores: (batch_size, max_moves) - Predicted move quality
                - theme_probs: (batch_size, num_themes) - Theme probabilities
        """
        # Encode position
        position_encoding = self.position_encoder(batch['position_features'])
        
        # Encode moves
        move_embeddings = self.move_encoder(batch['moves'])
        
        # Predict move scores
        move_scores = self.ranking_head(
            position_encoding,
            move_embeddings,
            batch['move_masks']
        )
        
        # Predict themes
        theme_probs = self.theme_head(position_encoding)
        
        return {
            'move_scores': move_scores,
            'theme_probs': theme_probs
        }
    
    def get_top_k_moves(self, batch: Dict[str, torch.Tensor], k: int = 5) -> torch.Tensor:
        """
        Get top-k move indices after scoring
        
        Args:
            batch: Input batch
            k: Number of top moves to return
        
        Returns:
            top_k_indices: (batch_size, k) - Indices of top-k moves
        """
        with torch.no_grad():
            outputs = self.forward(batch)
            move_scores = outputs['move_scores']
            
            # Get top-k indices (higher score = better move)
            top_k_indices = torch.topk(move_scores, k, dim=-1).indices
            
            return top_k_indices


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """Test model forward pass"""
    print("🧪 Testing MoveOrderingNetwork...")
    
    batch_size = 4
    max_moves = 10
    
    # Create dummy batch with valid move encodings
    from_squares = torch.randint(0, 64, (batch_size, max_moves, 1))
    to_squares = torch.randint(0, 64, (batch_size, max_moves, 1))
    promotions = torch.randint(0, 5, (batch_size, max_moves, 1))  # 0-4 for promotion
    
    batch = {
        'position_features': torch.randn(batch_size, 690),
        'moves': torch.cat([from_squares, to_squares, promotions], dim=-1),
        'move_masks': torch.ones(batch_size, max_moves, dtype=torch.bool)
    }
    
    # Create model
    model = MoveOrderingNetwork()
    
    print(f"   Total parameters: {count_parameters(model):,}")
    
    # Forward pass
    outputs = model(batch)
    
    print(f"   Move scores shape: {outputs['move_scores'].shape}")
    print(f"   Theme probs shape: {outputs['theme_probs'].shape}")
    print(f"   Sample move scores: {outputs['move_scores'][0]}")
    
    # Test top-k selection
    top_k = model.get_top_k_moves(batch, k=5)
    print(f"   Top-5 move indices: {top_k[0]}")
    
    print("✅ Model test passed!")


if __name__ == '__main__':
    test_model()
