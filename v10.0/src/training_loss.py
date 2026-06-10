"""Multi-Signal Training Loss Function.

Combines three training signals with different weights:
1. Strength (70%): MSE loss vs Lichess/GM evaluations
2. Character (20%): Cross-entropy vs historical engine moves
3. WDL (10%): Cross-entropy vs Syzygy win/draw/loss

SPRINT 3, DAY 1-2: Implement this module

Classes:
    StrengthLoss: MSE loss for evaluation prediction
    CharacterLoss: Cross-entropy for move distribution
    WDLLoss: Cross-entropy for endgame correctness
    MultiSignalLoss: Combined weighted loss

Methods (to implement):
    strength_loss(predictions, target_evals) -> Tensor
        MSE between predicted and target evaluations
        Target: learn accurate evaluation
        
    character_loss(predictions, target_moves) -> Tensor
        Cross-entropy between predicted and target moves
        Target: learn engine personality
        
    wdl_loss(predictions, target_wdl) -> Tensor
        Cross-entropy between predicted and Syzygy WDL
        Target: perfect endgame knowledge
        
    forward(predictions, evals, moves, wdls) -> Tensor
        Compute all three losses and combine:
        loss = 0.7 * strength + 0.2 * character + 0.1 * wdl

Weight Rationale:
    Strength (70%): Main training signal (evaluation accuracy)
    Character (20%): Secondary signal (preserve personality)
    WDL (10%): Tertiary signal (endgame perfection)
    Weights optimized empirically (can tune in Phase 3)

Test with: python -m pytest tests/test_training_loss.py -v
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class StrengthLoss(nn.Module):
    """Mean Squared Error loss for evaluation prediction.
    
    Measures how well the network predicts position evaluations.
    Target: minimize difference between predicted and actual eval.
    """
    
    def __init__(self, scale: float = 1.0):
        """Initialize strength loss.
        
        Args:
            scale: Scaling factor for gradients (default 1.0)
        """
        super().__init__()
        self.scale = scale
        self.mse = nn.MSELoss()
    
    def forward(self, predictions: torch.Tensor, 
               target_evals: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss.
        
        Args:
            predictions: Model's predicted evaluations (batch_size,)
            target_evals: Ground truth evaluations (batch_size,)
            
        Returns:
            Loss value (scalar)
            
        Example:
            pred_evals = model(positions)[:, 0]  # First output head
            loss = strength_loss(pred_evals, target_evals)
        """
        # TODO: SPRINT 3 DAY 1
        # return self.mse(predictions, target_evals) * self.scale
        pass


class CharacterLoss(nn.Module):
    """Cross-entropy loss for move distribution.
    
    Measures how well the network learns engine personality.
    Target: predict the moves the historical engine would make.
    """
    
    def __init__(self, num_moves: int = 1880, scale: float = 1.0):
        """Initialize character loss.
        
        Args:
            num_moves: Number of possible moves (default 1880)
            scale: Scaling factor for gradients (default 1.0)
        """
        super().__init__()
        self.num_moves = num_moves
        self.scale = scale
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, predictions: torch.Tensor,
               target_moves: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss for moves.
        
        Args:
            predictions: Model's move logits (batch_size, num_moves)
            target_moves: Ground truth move indices (batch_size,)
            
        Returns:
            Loss value (scalar)
            
        Example:
            move_logits = model(positions)[:, 1:num_moves+1]  # Character head
            loss = character_loss(move_logits, target_moves)
        """
        # TODO: SPRINT 3 DAY 1
        # return self.ce(predictions, target_moves) * self.scale
        pass


class WDLLoss(nn.Module):
    """Cross-entropy loss for win/draw/loss prediction.
    
    Measures endgame correctness using Syzygy ground truth.
    Target: predict the true win/draw/loss outcome.
    """
    
    def __init__(self, scale: float = 1.0):
        """Initialize WDL loss.
        
        Args:
            scale: Scaling factor for gradients (default 1.0)
        """
        super().__init__()
        self.scale = scale
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, predictions: torch.Tensor,
               target_wdl: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss for WDL.
        
        Args:
            predictions: Model's WDL logits (batch_size, 3)
            target_wdl: Ground truth WDL labels (batch_size,)
            
        Returns:
            Loss value (scalar)
            
        Example:
            wdl_logits = model(positions)[:, -3:]  # WDL head
            loss = wdl_loss(wdl_logits, target_wdl)
        """
        # TODO: SPRINT 3 DAY 1
        # return self.ce(predictions, target_wdl) * self.scale
        pass


class MultiSignalLoss(nn.Module):
    """Combined loss from three training signals.
    
    Loss = 0.7 * strength + 0.2 * character + 0.1 * wdl
    
    Each signal trains a different aspect:
        - Strength: Evaluation accuracy (core skill)
        - Character: Move selection (personality)
        - WDL: Endgame perfection (ground truth)
    """
    
    def __init__(self, 
                 strength_weight: float = 0.7,
                 character_weight: float = 0.2,
                 wdl_weight: float = 0.1,
                 num_moves: int = 1880):
        """Initialize multi-signal loss.
        
        Args:
            strength_weight: Weight for strength loss (default 0.7)
            character_weight: Weight for character loss (default 0.2)
            wdl_weight: Weight for WDL loss (default 0.1)
            num_moves: Number of possible moves (default 1880)
            
        Weights should sum to ~1.0 (can normalize if needed)
        """
        super().__init__()
        self.strength_weight = strength_weight
        self.character_weight = character_weight
        self.wdl_weight = wdl_weight
        
        # Initialize individual losses
        self.strength_loss = StrengthLoss(scale=strength_weight)
        self.character_loss = CharacterLoss(num_moves, scale=character_weight)
        self.wdl_loss = WDLLoss(scale=wdl_weight)
        
        # Track individual loss values for monitoring
        self.last_strength = 0.0
        self.last_character = 0.0
        self.last_wdl = 0.0
    
    def forward(self, model_outputs: Dict[str, torch.Tensor],
               targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss from multiple signals.
        
        Args:
            model_outputs: Dictionary with keys:
                - strength: (batch_size, 1) predicted evals
                - character: (batch_size, 1880) move logits
                - wdl: (batch_size, 3) WDL logits
                
            targets: Dictionary with keys:
                - evals: (batch_size,) target evaluations
                - moves: (batch_size,) target move indices
                - wdl: (batch_size,) target WDL labels (0/1/2)
        
        Returns:
            (total_loss, loss_dict): Combined loss + individual components
            
        Loss calculation:
            strength_loss = MSE(pred_eval, target_eval)
            character_loss = CE(move_logits, target_moves)
            wdl_loss = CE(wdl_logits, target_wdl)
            total = 0.7 * strength + 0.2 * character + 0.1 * wdl
        
        Example:
            outputs = model(positions)
            targets = {
                'evals': batch_evals,
                'moves': batch_moves,
                'wdl': batch_wdl
            }
            loss, metrics = loss_fn(outputs, targets)
            print(f"Strength: {metrics['strength']}, "
                  f"Character: {metrics['character']}, "
                  f"WDL: {metrics['wdl']}")
        """
        # TODO: SPRINT 3 DAY 1
        # 1. Compute strength loss
        # 2. Compute character loss
        # 3. Compute WDL loss
        # 4. Combine: total = 0.7*strength + 0.2*character + 0.1*wdl
        # 5. Store individual values for monitoring
        # 6. Return (total_loss, {'strength': s, 'character': c, 'wdl': w})
        pass
    
    def get_loss_components(self) -> Dict[str, float]:
        """Get last computed loss components for monitoring.
        
        Returns:
            Dictionary with individual loss values
            
        Usage:
            for batch in data_loader:
                loss, _ = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                
                # Log metrics
                metrics = loss_fn.get_loss_components()
                wandb.log(metrics)
        """
        return {
            'strength': self.last_strength,
            'character': self.last_character,
            'wdl': self.last_wdl,
            'total': (self.strength_weight * self.last_strength +
                     self.character_weight * self.last_character +
                     self.wdl_weight * self.last_wdl)
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    loss_fn = MultiSignalLoss(
        strength_weight=0.7,
        character_weight=0.2,
        wdl_weight=0.1
    )
    
    # Dummy batch
    batch_size = 32
    num_moves = 1880
    
    outputs = {
        'strength': torch.randn(batch_size, 1),
        'character': torch.randn(batch_size, num_moves),
        'wdl': torch.randn(batch_size, 3)
    }
    
    targets = {
        'evals': torch.randn(batch_size),
        'moves': torch.randint(0, num_moves, (batch_size,)),
        'wdl': torch.randint(0, 3, (batch_size,))
    }
    
    # loss, metrics = loss_fn(outputs, targets)
    # print(f"Loss: {loss.item():.4f}")
    # print(f"Metrics: {metrics}")
    
    print("Multi-signal loss module ready for implementation")
