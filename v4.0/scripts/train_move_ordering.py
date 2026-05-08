#!/usr/bin/env python3
"""
Move Ordering Training Script

Trains the MoveOrderingNetwork on Stockfish-enriched puzzles.

Training objectives:
1. Move Ranking Loss: Learn to score moves by quality (weighted MSE)
2. Theme Classification Loss: Identify positional themes (BCE)

Features:
- Multi-GPU support (DataParallel)
- Mixed precision training (AMP) for speed
- Gradient accumulation for large effective batch sizes
- TensorBoard logging
- Model checkpointing
- Early stopping
- Learning rate scheduling
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.training.puzzle_dataset import MoveOrderingDataset, custom_collate_fn
from src.models.move_ordering_network import MoveOrderingNetwork, count_parameters


class MoveOrderingTrainer:
    """Trainer for move ordering network"""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device,
                 learning_rate: float = 1e-3,
                 checkpoint_dir: str = "models/stage1_themes",
                 use_amp: bool = True,
                 gradient_accumulation_steps: int = 1):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Optimizer and scheduler
        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        
        # Mixed precision
        self.scaler = GradScaler() if use_amp else None
        
        # Loss weights
        self.ranking_loss_weight = 1.0
        self.theme_loss_weight = 0.5
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        
        print(f"🚀 Trainer initialized")
        print(f"   Device: {device}")
        print(f"   Parameters: {count_parameters(model):,}")
        print(f"   Mixed precision: {use_amp}")
        print(f"   Gradient accumulation: {gradient_accumulation_steps} steps")
    
    def compute_ranking_loss(self, pred_scores: torch.Tensor, target_scores: torch.Tensor,
                            move_weights: torch.Tensor, move_masks: torch.Tensor) -> torch.Tensor:
        """
        Weighted MSE loss for move ranking
        
        Args:
            pred_scores: (batch, max_moves) - Predicted move scores
            target_scores: (batch, max_moves) - Target move scores (normalized)
            move_weights: (batch, max_moves) - Importance weights
            move_masks: (batch, max_moves) - Valid move masks
        """
        # Compute squared error
        squared_error = (pred_scores - target_scores) ** 2
        
        # Weight by importance
        weighted_error = squared_error * move_weights
        
        # Mask invalid moves
        weighted_error = weighted_error * move_masks.float()
        
        # Average over valid moves
        loss = weighted_error.sum() / move_masks.float().sum().clamp(min=1.0)
        
        return loss
    
    def compute_theme_loss(self, pred_themes: torch.Tensor, target_themes: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy loss for multi-label theme classification"""
        return F.binary_cross_entropy(pred_themes, target_themes)
    
    def compute_total_loss(self, outputs: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        """Compute combined loss"""
        # Ranking loss
        ranking_loss = self.compute_ranking_loss(
            outputs['move_scores'],
            batch['move_scores'],
            batch['move_weights'],
            batch['move_masks']
        )
        
        # Theme classification loss
        theme_loss = self.compute_theme_loss(
            outputs['theme_probs'],
            batch['themes']
        )
        
        # Combined loss
        total_loss = (self.ranking_loss_weight * ranking_loss +
                     self.theme_loss_weight * theme_loss)
        
        return {
            'total_loss': total_loss,
            'ranking_loss': ranking_loss,
            'theme_loss': theme_loss
        }
    
    def compute_top_k_accuracy(self, pred_scores: torch.Tensor, target_scores: torch.Tensor,
                               move_masks: torch.Tensor, k: int = 5) -> float:
        """
        Compute top-k accuracy: How often is the best move in predicted top-k?
        """
        batch_size = pred_scores.size(0)
        correct = 0
        
        for i in range(batch_size):
            valid_mask = move_masks[i]
            if not valid_mask.any():
                continue
            
            # Get predicted top-k indices
            pred_top_k = torch.topk(pred_scores[i][valid_mask], min(k, valid_mask.sum())).indices
            
            # Get actual best move index (highest target score)
            target_best = torch.argmax(target_scores[i][valid_mask])
            
            # Check if best move is in predicted top-k
            if target_best in pred_top_k:
                correct += 1
        
        return correct / batch_size if batch_size > 0 else 0.0
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_ranking_loss = 0.0
        total_theme_loss = 0.0
        total_top5_acc = 0.0
        total_top10_acc = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(batch)
                    losses = self.compute_total_loss(outputs, batch)
                    loss = losses['total_loss'] / self.gradient_accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.global_step += 1
            else:
                outputs = self.model(batch)
                losses = self.compute_total_loss(outputs, batch)
                loss = losses['total_loss'] / self.gradient_accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
            
            # Compute metrics
            with torch.no_grad():
                top5_acc = self.compute_top_k_accuracy(
                    outputs['move_scores'], batch['move_scores'], batch['move_masks'], k=5
                )
                top10_acc = self.compute_top_k_accuracy(
                    outputs['move_scores'], batch['move_scores'], batch['move_masks'], k=10
                )
            
            # Update metrics
            total_loss += losses['total_loss'].item()
            total_ranking_loss += losses['ranking_loss'].item()
            total_theme_loss += losses['theme_loss'].item()
            total_top5_acc += top5_acc
            total_top10_acc += top10_acc
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss/num_batches:.4f}",
                'top5': f"{total_top5_acc/num_batches:.3f}",
                'top10': f"{total_top10_acc/num_batches:.3f}"
            })
        
        return {
            'loss': total_loss / num_batches,
            'ranking_loss': total_ranking_loss / num_batches,
            'theme_loss': total_theme_loss / num_batches,
            'top5_accuracy': total_top5_acc / num_batches,
            'top10_accuracy': total_top10_acc / num_batches
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set"""
        self.model.eval()
        
        total_loss = 0.0
        total_ranking_loss = 0.0
        total_theme_loss = 0.0
        total_top5_acc = 0.0
        total_top10_acc = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(batch)
            losses = self.compute_total_loss(outputs, batch)
            
            # Compute metrics
            top5_acc = self.compute_top_k_accuracy(
                outputs['move_scores'], batch['move_scores'], batch['move_masks'], k=5
            )
            top10_acc = self.compute_top_k_accuracy(
                outputs['move_scores'], batch['move_scores'], batch['move_masks'], k=10
            )
            
            # Update metrics
            total_loss += losses['total_loss'].item()
            total_ranking_loss += losses['ranking_loss'].item()
            total_theme_loss += losses['theme_loss'].item()
            total_top5_acc += top5_acc
            total_top10_acc += top10_acc
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'ranking_loss': total_ranking_loss / num_batches,
            'theme_loss': total_theme_loss / num_batches,
            'top5_accuracy': total_top5_acc / num_batches,
            'top10_accuracy': total_top10_acc / num_batches
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_checkpoint.pt'
            torch.save(checkpoint, best_path)
            print(f"   💾 Best model saved (val_loss: {self.best_val_loss:.4f})")
    
    def train(self, num_epochs: int, early_stopping_patience: int = 5):
        """Main training loop"""
        print(f"\n🎯 Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics)
            
            # Validate
            val_metrics = self.validate()
            self.val_losses.append(val_metrics)
            
            # Update learning rate
            self.scheduler.step()
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch} Summary:")
            print(f"   Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Top-5: {train_metrics['top5_accuracy']:.3f}, "
                  f"Top-10: {train_metrics['top10_accuracy']:.3f}")
            print(f"   Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Top-5: {val_metrics['top5_accuracy']:.3f}, "
                  f"Top-10: {val_metrics['top10_accuracy']:.3f}")
            print(f"   LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                print(f"\n⏹️  Early stopping triggered (patience: {early_stopping_patience})")
                break
        
        print(f"\n✅ Training complete!")
        print(f"   Best val loss: {self.best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train move ordering network")
    parser.add_argument('--data-path', type=str, required=True,
                       help="Path to preprocessed puzzle JSON")
    parser.add_argument('--checkpoint-dir', type=str, default="models/stage1_themes",
                       help="Directory to save checkpoints")
    parser.add_argument('--batch-size', type=int, default=64,
                       help="Batch size")
    parser.add_argument('--num-epochs', type=int, default=100,
                       help="Number of epochs")
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                       help="Early stopping patience")
    parser.add_argument('--num-workers', type=int, default=4,
                       help="DataLoader workers")
    parser.add_argument('--gradient-accumulation', type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument('--max-samples', type=int, default=None,
                       help="Limit dataset size (for testing)")
    parser.add_argument('--no-amp', action='store_true',
                       help="Disable mixed precision training")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create datasets
    print(f"\n📚 Loading datasets...")
    train_dataset = MoveOrderingDataset(
        data_path=args.data_path,
        split='train',
        augment=True,
        max_samples=args.max_samples
    )
    
    val_dataset = MoveOrderingDataset(
        data_path=args.data_path,
        split='val',
        augment=False,
        max_samples=args.max_samples // 10 if args.max_samples else None
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    print(f"\n🧠 Creating model...")
    model = MoveOrderingNetwork()
    
    # Create trainer
    trainer = MoveOrderingTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        use_amp=not args.no_amp,
        gradient_accumulation_steps=args.gradient_accumulation
    )
    
    # Train
    trainer.train(
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping_patience
    )


if __name__ == '__main__':
    main()
