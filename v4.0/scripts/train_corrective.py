"""
Stage 2: Corrective Training Script

Fine-tunes Stage 1 model with historical failure correction.
Uses dual-learning pattern: avoid V7P3R's mistakes + exploit opponent patterns.

Author: V7P3RAI Development Team
Date: 2026-04-24
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import json
import sys
from datetime import datetime
from typing import Dict, Tuple
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.move_ordering_network import MoveOrderingNetwork
from src.training.corrective_dataset import CorrectiveDataset, custom_collate_fn


class CorrectiveTrainer:
    """Trainer for Stage 2 corrective learning."""
    
    def __init__(
        self,
        model: MoveOrderingNetwork,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 5e-5,
        device: str = 'cpu',
        correction_weight: float = 2.0,
        ranking_weight: float = 1.0
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.correction_weight = correction_weight
        self.ranking_weight = ranking_weight
        
        # Optimizer (lower LR for fine-tuning)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler (cosine annealing)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=50,
            eta_min=1e-6
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss(reduction='none')
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_correction_loss': [],
            'val_correction_loss': [],
            'train_ranking_loss': [],
            'val_ranking_loss': [],
            'val_top5_accuracy': [],
            'val_blunder_avoidance': []
        }
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
    
    def compute_correction_loss(self, batch: Dict) -> torch.Tensor:
        """
        Compute weighted correction loss.
        
        Heavily penalizes historical bad moves, rewards best moves.
        Uses move_weights from dataset (0.0-1.0 per move).
        """
        # Forward pass
        output = self.model(batch)
        move_scores = output['move_scores']
        
        # Target scores from Stockfish analysis (normalized 0-1)
        target_scores = batch['move_scores']
        
        # Move importance weights (0.0 for bad moves, 1.0 for best)
        importance_weights = batch['move_weights']
        
        # Mask for valid moves (handle variable lengths)
        mask = batch['move_masks']
        
        # MSE loss per move
        loss_per_move = self.mse_loss(move_scores, target_scores)
        
        # Apply importance weighting
        weighted_loss = loss_per_move * importance_weights
        
        # Apply mask and average
        masked_loss = weighted_loss * mask.float()
        total_loss = masked_loss.sum() / mask.float().sum()
        
        return total_loss
    
    def compute_ranking_loss(self, batch: Dict) -> torch.Tensor:
        """
        Compute move ranking loss (preserve Stage 1 ability).
        
        Similar to Stage 1 training - rank moves by quality.
        """
        output = self.model(batch)
        move_scores = output['move_scores']
        
        target_scores = batch['move_scores']
        importance_weights = batch['move_weights']
        mask = batch['move_masks']
        
        # Standard weighted MSE
        loss_per_move = self.mse_loss(move_scores, target_scores)
        weighted_loss = loss_per_move * importance_weights
        masked_loss = weighted_loss * mask.float()
        
        return masked_loss.sum() / mask.float().sum()
    
    def compute_top_k_accuracy(self, batch: Dict, k: int = 5) -> float:
        """
        Compute top-K accuracy (best move in top K predictions).
        """
        with torch.no_grad():
            output = self.model(batch)
            move_scores = output['move_scores']
            
            # Find best move (highest weight = best move)
            weights = batch['move_weights']
            mask = batch['move_masks']
            
            # Mask out invalid moves
            masked_weights = weights.clone()
            masked_weights[~mask] = -float('inf')
            
            # Get index of best move per example
            best_move_idx = masked_weights.argmax(dim=1)
            
            # Get top-k predicted moves
            masked_scores = move_scores.clone()
            masked_scores[~mask] = -float('inf')
            _, top_k_indices = masked_scores.topk(k, dim=1)
            
            # Check if best move is in top-k
            correct = 0
            for i in range(len(best_move_idx)):
                if best_move_idx[i] in top_k_indices[i]:
                    correct += 1
            
            return correct / len(best_move_idx)
    
    def compute_blunder_avoidance(self, batch: Dict) -> float:
        """
        Compute blunder avoidance rate.
        
        Checks if model ranks blunder moves (weight=0.0) low.
        """
        with torch.no_grad():
            output = self.model(batch)
            move_scores = output['move_scores']
            
            weights = batch['move_weights']
            mask = batch['move_masks']
            
            # Find blunder moves (weight < 0.3)
            blunder_mask = (weights < 0.3) & mask
            
            if blunder_mask.sum() == 0:
                return 1.0  # No blunders in batch
            
            # For each example with a blunder, check if it's NOT top-ranked
            avoided = 0
            total_blunders = 0
            
            for i in range(len(weights)):
                example_blunders = blunder_mask[i]
                if not example_blunders.any():
                    continue
                
                # Get predicted scores for this example
                example_scores = move_scores[i]
                example_mask = mask[i]
                
                # Mask invalid moves
                masked_scores = example_scores.clone()
                masked_scores[~example_mask] = -float('inf')
                
                # Top predicted move
                top_move_idx = masked_scores.argmax()
                
                # Check if top move is NOT a blunder
                if not example_blunders[top_move_idx]:
                    avoided += 1
                
                total_blunders += 1
            
            return avoided / total_blunders if total_blunders > 0 else 1.0
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_correction_loss = 0.0
        total_ranking_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            correction_loss = self.compute_correction_loss(batch)
            ranking_loss = self.compute_ranking_loss(batch)
            
            # Combined loss
            loss = (self.correction_weight * correction_loss + 
                   self.ranking_weight * ranking_loss)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_correction_loss += correction_loss.item()
            total_ranking_loss += ranking_loss.item()
            num_batches += 1
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}: "
                      f"Loss={loss.item():.4f}, "
                      f"Correction={correction_loss.item():.4f}, "
                      f"Ranking={ranking_loss.item():.4f}")
        
        return {
            'train_loss': total_loss / num_batches,
            'train_correction_loss': total_correction_loss / num_batches,
            'train_ranking_loss': total_ranking_loss / num_batches
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        total_correction_loss = 0.0
        total_ranking_loss = 0.0
        total_top5_acc = 0.0
        total_blunder_avoid = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Compute losses
                correction_loss = self.compute_correction_loss(batch)
                ranking_loss = self.compute_ranking_loss(batch)
                loss = (self.correction_weight * correction_loss + 
                       self.ranking_weight * ranking_loss)
                
                # Compute metrics
                top5_acc = self.compute_top_k_accuracy(batch, k=5)
                blunder_avoid = self.compute_blunder_avoidance(batch)
                
                total_loss += loss.item()
                total_correction_loss += correction_loss.item()
                total_ranking_loss += ranking_loss.item()
                total_top5_acc += top5_acc
                total_blunder_avoid += blunder_avoid
                num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches,
            'val_correction_loss': total_correction_loss / num_batches,
            'val_ranking_loss': total_ranking_loss / num_batches,
            'val_top5_accuracy': total_top5_acc / num_batches,
            'val_blunder_avoidance': total_blunder_avoid / num_batches
        }
    
    def train(self, num_epochs: int, early_stopping_patience: int = 10):
        """Full training loop."""
        print(f"\n{'='*60}")
        print(f"Starting Stage 2 Corrective Training")
        print(f"{'='*60}")
        print(f"Epochs: {num_epochs}")
        print(f"Patience: {early_stopping_patience}")
        print(f"Correction Weight: {self.correction_weight}")
        print(f"Ranking Weight: {self.ranking_weight}")
        print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            print(f"\n[Epoch {epoch+1}/{num_epochs}]")
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Update history
            for key, value in {**train_metrics, **val_metrics}.items():
                self.history[key].append(value)
            
            # Learning rate step
            self.scheduler.step()
            
            # Print epoch summary
            print(f"\n  Summary:")
            print(f"    Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"    Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"    Val Top-5 Accuracy: {val_metrics['val_top5_accuracy']*100:.1f}%")
            print(f"    Val Blunder Avoidance: {val_metrics['val_blunder_avoidance']*100:.1f}%")
            print(f"    LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping check
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.best_epoch = epoch
                self.patience_counter = 0
                print(f"    ✅ New best model! (val_loss: {self.best_val_loss:.4f})")
                self.save_checkpoint('best_model.pt')
            else:
                self.patience_counter += 1
                print(f"    No improvement ({self.patience_counter}/{early_stopping_patience})")
                
                if self.patience_counter >= early_stopping_patience:
                    print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
                    print(f"    Best epoch was {self.best_epoch+1} with val_loss {self.best_val_loss:.4f}")
                    break
            
            # Save latest checkpoint
            self.save_checkpoint('latest_model.pt')
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Epoch: {self.best_epoch+1}")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_dir = Path('models/stage2_corrective')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch
        }
        
        torch.save(checkpoint, checkpoint_dir / filename)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Stage 2: Corrective Training')
    parser.add_argument('--data-path', type=str,
                       default='data/stage2_training/corrective_dataset.json',
                       help='Path to corrective dataset')
    parser.add_argument('--stage1-model', type=str,
                       default='models/stage1_themes/best_checkpoint.pt',
                       help='Path to Stage 1 best model')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=5e-5,
                       help='Learning rate (lower for fine-tuning)')
    parser.add_argument('--correction-weight', type=float, default=2.0,
                       help='Weight for correction loss')
    parser.add_argument('--ranking-weight', type=float, default=1.0,
                       help='Weight for ranking loss')
    parser.add_argument('--early-stopping-patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation split ratio')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')
    
    args = parser.parse_args()
    
    print("🚀 V7P3RAI Stage 2: Corrective Training")
    print("=" * 60)
    print(f"📂 Corrective Dataset: {args.data_path}")
    print(f"📦 Stage 1 Model: {args.stage1_model}")
    print(f"⚙️  Batch Size: {args.batch_size}")
    print(f"📊 Epochs: {args.num_epochs}")
    print(f"🎯 Device: {args.device}")
    print("=" * 60)
    
    # Load dataset
    print("\n📂 Loading corrective dataset...")
    dataset = CorrectiveDataset(args.data_path)
    
    # Split into train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"   Train: {train_size} examples")
    print(f"   Val: {val_size} examples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=0
    )
    
    # Load Stage 1 model
    print(f"\n📦 Loading Stage 1 model from {args.stage1_model}...")
    model = MoveOrderingNetwork(num_themes=57)
    
    checkpoint = torch.load(args.stage1_model, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"   ✅ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"   Stage 1 Top-5 Accuracy: {checkpoint.get('val_top5_accuracy', 'unknown')}")
    
    # Create trainer
    trainer = CorrectiveTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        device=args.device,
        correction_weight=args.correction_weight,
        ranking_weight=args.ranking_weight
    )
    
    # Train
    trainer.train(args.num_epochs, args.early_stopping_patience)
    
    print("\n✅ Stage 2 training complete!")
    print(f"   Best model saved to: models/stage2_corrective/best_model.pt")


if __name__ == '__main__':
    main()

