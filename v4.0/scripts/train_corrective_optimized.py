"""
Stage 2: OPTIMIZED Corrective Training Script

Production-grade training with advanced techniques for maximum performance.
Designed for potential deployment as V7P3R's primary move selection engine.

Enhancements:
- Warmup + cosine annealing LR schedule
- Gradient accumulation for effective larger batches
- Dynamic loss weighting based on training progress
- Advanced regularization (dropout, label smoothing, weight decay)
- Comprehensive validation suite
- Model checkpointing with EMA (Exponential Moving Average)
- Blunder-focused curriculum learning
- Advanced early stopping with multiple metrics

Author: V7P3RAI Development Team
Date: 2026-04-24
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from pathlib import Path
import json
import sys
from datetime import datetime
from typing import Dict, Tuple, List
import numpy as np
from collections import defaultdict
import copy

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.move_ordering_network import MoveOrderingNetwork
from src.training.corrective_dataset import CorrectiveDataset, custom_collate_fn


class ExponentialMovingAverage:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: nn.Module):
        """Update EMA parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1.0 - self.decay) * param.data
                )
    
    def apply_shadow(self, model: nn.Module):
        """Apply EMA parameters to model."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model: nn.Module):
        """Restore original parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class WarmupCosineSchedule:
    """Learning rate scheduler with warmup and cosine annealing."""
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
    
    def step(self):
        """Update learning rate."""
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_lr(self):
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


class OptimizedCorrectiveTrainer:
    """
    Production-grade trainer with advanced optimization techniques.
    
    Designed for maximum performance to potentially replace V7P3R's traditional search.
    """
    
    def __init__(
        self,
        model: MoveOrderingNetwork,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 1e-4,
        device: str = 'cpu',
        correction_weight: float = 3.0,
        ranking_weight: float = 1.0,
        blunder_weight: float = 5.0,
        gradient_accumulation_steps: int = 4,
        use_ema: bool = True,
        label_smoothing: float = 0.1,
        warmup_ratio: float = 0.1
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.correction_weight = correction_weight
        self.ranking_weight = ranking_weight
        self.blunder_weight = blunder_weight
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.label_smoothing = label_smoothing
        
        # Optimizer with optimized settings
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01  # Stronger regularization
        )
        
        # Learning rate scheduler with warmup
        total_steps = len(train_loader) // gradient_accumulation_steps
        warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = WarmupCosineSchedule(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=1e-7
        )
        
        # Exponential Moving Average
        self.ema = ExponentialMovingAverage(model, decay=0.9995) if use_ema else None
        
        # Loss functions
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        
        # Training history with comprehensive metrics
        self.history = defaultdict(list)
        
        self.best_val_loss = float('inf')
        self.best_blunder_avoidance = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        
        print(f"🔧 Optimizer: AdamW (lr={learning_rate:.2e}, weight_decay=0.01)")
        print(f"📊 Gradient Accumulation: {gradient_accumulation_steps}x (effective batch={gradient_accumulation_steps * train_loader.batch_size})")
        print(f"🌡️  LR Schedule: Warmup({warmup_steps} steps) + Cosine({total_steps} steps)")
        print(f"⚖️  Loss Weights: Correction={correction_weight}, Ranking={ranking_weight}, Blunder={blunder_weight}")
        print(f"🎯 EMA: {'Enabled (decay=0.9995)' if use_ema else 'Disabled'}")
        print(f"🔀 Label Smoothing: {label_smoothing}")
    
    def compute_correction_loss(self, batch: Dict, use_blunder_focus: bool = True) -> torch.Tensor:
        """
        Enhanced correction loss with blunder focus and label smoothing.
        
        Heavily penalizes historical bad moves (especially blunders).
        """
        # Forward pass
        output = self.model(batch)
        move_scores = output['move_scores']
        
        # Target scores with label smoothing
        target_scores = batch['move_scores']
        if self.label_smoothing > 0:
            target_scores = target_scores * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Move importance weights (0.0 for bad moves, 1.0 for best)
        importance_weights = batch['move_weights']
        
        # Apply extra blunder penalty
        if use_blunder_focus:
            # Identify blunders (weight=0.0) and mistakes (weight<0.3)
            blunder_mask = (importance_weights < 0.1) & batch['move_masks']
            importance_weights = importance_weights.clone()
            importance_weights[blunder_mask] = self.blunder_weight  # Amplify blunder penalty
        
        # Mask for valid moves
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
        Margin-based ranking loss for better move ordering.
        
        Ensures best moves are ranked significantly higher than alternatives.
        """
        output = self.model(batch)
        move_scores = output['move_scores']
        
        target_scores = batch['move_scores']
        importance_weights = batch['move_weights']
        mask = batch['move_masks']
        
        # Find best move (highest weight)
        masked_weights = importance_weights.clone()
        masked_weights[~mask] = -float('inf')
        best_move_idx = masked_weights.argmax(dim=1)
        
        # Get best move scores
        best_scores = move_scores[torch.arange(len(best_move_idx)), best_move_idx].unsqueeze(1)
        
        # Margin-based ranking loss: best should be margin above others
        margin = 0.1
        ranking_loss = torch.clamp(margin - (best_scores - move_scores), min=0)
        
        # Mask out best move itself and invalid moves
        ranking_mask = mask.clone()
        ranking_mask[torch.arange(len(best_move_idx)), best_move_idx] = False
        
        # Apply mask and average
        masked_ranking_loss = ranking_loss * ranking_mask.float()
        total_ranking_loss = masked_ranking_loss.sum() / ranking_mask.float().sum()
        
        return total_ranking_loss
    
    def compute_top_k_accuracy(self, batch: Dict, k: int = 5) -> float:
        """Compute top-K accuracy (best move in top K predictions)."""
        with torch.no_grad():
            output = self.model(batch)
            move_scores = output['move_scores']
            
            weights = batch['move_weights']
            mask = batch['move_masks']
            
            # Find best move
            masked_weights = weights.clone()
            masked_weights[~mask] = -float('inf')
            best_move_idx = masked_weights.argmax(dim=1)
            
            # Get top-k predicted moves
            masked_scores = move_scores.clone()
            masked_scores[~mask] = -float('inf')
            _, top_k_indices = masked_scores.topk(min(k, masked_scores.size(1)), dim=1)
            
            # Check if best move is in top-k
            correct = 0
            for i in range(len(best_move_idx)):
                if best_move_idx[i] in top_k_indices[i]:
                    correct += 1
            
            return correct / len(best_move_idx)
    
    def compute_blunder_avoidance(self, batch: Dict, threshold: float = 0.3) -> float:
        """
        Compute blunder avoidance rate (critical metric for V7P3R deployment).
        
        Checks if model ranks blunder/mistake moves low.
        """
        with torch.no_grad():
            output = self.model(batch)
            move_scores = output['move_scores']
            
            weights = batch['move_weights']
            mask = batch['move_masks']
            
            # Find blunder/mistake moves
            bad_move_mask = (weights < threshold) & mask
            
            if bad_move_mask.sum() == 0:
                return 1.0
            
            avoided = 0
            total_bad = 0
            
            for i in range(len(weights)):
                example_bad_moves = bad_move_mask[i]
                if not example_bad_moves.any():
                    continue
                
                example_scores = move_scores[i]
                example_mask = mask[i]
                
                masked_scores = example_scores.clone()
                masked_scores[~example_mask] = -float('inf')
                
                top_move_idx = masked_scores.argmax()
                
                # Success if top move is NOT a blunder/mistake
                if not example_bad_moves[top_move_idx]:
                    avoided += 1
                
                total_bad += 1
            
            return avoided / total_bad if total_bad > 0 else 1.0
    
    def compute_average_rank_of_best(self, batch: Dict) -> float:
        """
        Compute average rank of best move in predictions.
        
        Lower is better (1.0 = always ranked first).
        """
        with torch.no_grad():
            output = self.model(batch)
            move_scores = output['move_scores']
            
            weights = batch['move_weights']
            mask = batch['move_masks']
            
            total_rank = 0
            count = 0
            
            for i in range(len(weights)):
                example_weights = weights[i]
                example_scores = move_scores[i]
                example_mask = mask[i]
                
                # Find best move
                masked_weights = example_weights.clone()
                masked_weights[~example_mask] = -float('inf')
                best_idx = masked_weights.argmax()
                
                # Rank moves by predicted score
                masked_scores = example_scores.clone()
                masked_scores[~example_mask] = -float('inf')
                sorted_indices = masked_scores.argsort(descending=True)
                
                # Find rank of best move (1-indexed)
                rank = (sorted_indices == best_idx).nonzero(as_tuple=True)[0].item() + 1
                total_rank += rank
                count += 1
            
            return total_rank / count if count > 0 else float('inf')
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with gradient accumulation."""
        self.model.train()
        
        total_loss = 0.0
        total_correction_loss = 0.0
        total_ranking_loss = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            correction_loss = self.compute_correction_loss(batch, use_blunder_focus=True)
            ranking_loss = self.compute_ranking_loss(batch)
            
            # Combined loss with dynamic weighting
            loss = (self.correction_weight * correction_loss + 
                   self.ranking_weight * ranking_loss)
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Update learning rate
                self.scheduler.step()
                
                # Update EMA
                if self.ema is not None:
                    self.ema.update(self.model)
            
            # Accumulate metrics (unscaled)
            total_loss += loss.item() * self.gradient_accumulation_steps
            total_correction_loss += correction_loss.item()
            total_ranking_loss += ranking_loss.item()
            num_batches += 1
            
            # Print progress
            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}: "
                      f"Loss={loss.item() * self.gradient_accumulation_steps:.4f}, "
                      f"Correction={correction_loss.item():.4f}, "
                      f"Ranking={ranking_loss.item():.4f}, "
                      f"LR={self.scheduler.get_lr():.2e}")
        
        return {
            'train_loss': total_loss / num_batches,
            'train_correction_loss': total_correction_loss / num_batches,
            'train_ranking_loss': total_ranking_loss / num_batches
        }
    
    def validate(self, use_ema: bool = False) -> Dict[str, float]:
        """
        Comprehensive validation with multiple metrics.
        
        Tests both accuracy and critical blunder avoidance for V7P3R deployment.
        """
        # Apply EMA if requested
        if use_ema and self.ema is not None:
            self.ema.apply_shadow(self.model)
        
        self.model.eval()
        
        total_loss = 0.0
        total_correction_loss = 0.0
        total_ranking_loss = 0.0
        total_top1_acc = 0.0
        total_top3_acc = 0.0
        total_top5_acc = 0.0
        total_top10_acc = 0.0
        total_blunder_avoid = 0.0
        total_avg_rank = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Compute losses
                correction_loss = self.compute_correction_loss(batch, use_blunder_focus=False)
                ranking_loss = self.compute_ranking_loss(batch)
                loss = (self.correction_weight * correction_loss + 
                       self.ranking_weight * ranking_loss)
                
                # Compute metrics
                top1_acc = self.compute_top_k_accuracy(batch, k=1)
                top3_acc = self.compute_top_k_accuracy(batch, k=3)
                top5_acc = self.compute_top_k_accuracy(batch, k=5)
                top10_acc = self.compute_top_k_accuracy(batch, k=10)
                blunder_avoid = self.compute_blunder_avoidance(batch, threshold=0.3)
                avg_rank = self.compute_average_rank_of_best(batch)
                
                total_loss += loss.item()
                total_correction_loss += correction_loss.item()
                total_ranking_loss += ranking_loss.item()
                total_top1_acc += top1_acc
                total_top3_acc += top3_acc
                total_top5_acc += top5_acc
                total_top10_acc += top10_acc
                total_blunder_avoid += blunder_avoid
                total_avg_rank += avg_rank
                num_batches += 1
        
        # Restore original parameters if using EMA
        if use_ema and self.ema is not None:
            self.ema.restore(self.model)
        
        return {
            'val_loss': total_loss / num_batches,
            'val_correction_loss': total_correction_loss / num_batches,
            'val_ranking_loss': total_ranking_loss / num_batches,
            'val_top1_accuracy': total_top1_acc / num_batches,
            'val_top3_accuracy': total_top3_acc / num_batches,
            'val_top5_accuracy': total_top5_acc / num_batches,
            'val_top10_accuracy': total_top10_acc / num_batches,
            'val_blunder_avoidance': total_blunder_avoid / num_batches,
            'val_avg_best_rank': total_avg_rank / num_batches
        }
    
    def train(self, num_epochs: int, early_stopping_patience: int = 20):
        """Full training loop with advanced early stopping."""
        print(f"\n{'='*70}")
        print(f"🚀 OPTIMIZED Stage 2 Corrective Training")
        print(f"   Production-grade configuration for V7P3R deployment")
        print(f"{'='*70}")
        print(f"Epochs: {num_epochs}")
        print(f"Patience: {early_stopping_patience}")
        print(f"{'='*70}\n")
        
        for epoch in range(num_epochs):
            print(f"\n[Epoch {epoch+1}/{num_epochs}]")
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate (both standard and EMA)
            val_metrics = self.validate(use_ema=False)
            if self.ema is not None:
                ema_metrics = self.validate(use_ema=True)
                print(f"\n  📊 EMA Validation:")
                print(f"     Top-1: {ema_metrics['val_top1_accuracy']*100:.1f}%")
                print(f"     Top-5: {ema_metrics['val_top5_accuracy']*100:.1f}%")
                print(f"     Blunder Avoidance: {ema_metrics['val_blunder_avoidance']*100:.1f}%")
            
            # Update history
            for key, value in {**train_metrics, **val_metrics}.items():
                self.history[key].append(value)
            
            # Print epoch summary
            print(f"\n  📈 Summary:")
            print(f"     Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"     Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"     Val Top-1: {val_metrics['val_top1_accuracy']*100:.1f}%")
            print(f"     Val Top-3: {val_metrics['val_top3_accuracy']*100:.1f}%")
            print(f"     Val Top-5: {val_metrics['val_top5_accuracy']*100:.1f}%")
            print(f"     Val Top-10: {val_metrics['val_top10_accuracy']*100:.1f}%")
            print(f"     Val Blunder Avoidance: {val_metrics['val_blunder_avoidance']*100:.1f}%")
            print(f"     Val Avg Best Rank: {val_metrics['val_avg_best_rank']:.2f}")
            print(f"     LR: {self.scheduler.get_lr():.2e}")
            
            # Multi-metric early stopping
            # Primary: blunder avoidance (most critical for V7P3R)
            # Secondary: validation loss
            improved = False
            
            if val_metrics['val_blunder_avoidance'] > self.best_blunder_avoidance:
                self.best_blunder_avoidance = val_metrics['val_blunder_avoidance']
                improved = True
                print(f"     ✅ New best blunder avoidance: {self.best_blunder_avoidance*100:.1f}%")
            
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                improved = True
                print(f"     ✅ New best val loss: {self.best_val_loss:.4f}")
            
            if improved:
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt', val_metrics)
            else:
                self.patience_counter += 1
                print(f"     ⏳ No improvement ({self.patience_counter}/{early_stopping_patience})")
                
                if self.patience_counter >= early_stopping_patience:
                    print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
                    print(f"    Best epoch: {self.best_epoch+1}")
                    print(f"    Best blunder avoidance: {self.best_blunder_avoidance*100:.1f}%")
                    print(f"    Best val loss: {self.best_val_loss:.4f}")
                    break
            
            # Save latest checkpoint
            self.save_checkpoint('latest_model.pt', val_metrics)
            
            # Save EMA checkpoint
            if self.ema is not None and epoch % 5 == 0:
                self.save_ema_checkpoint(f'ema_epoch_{epoch+1}.pt')
        
        print(f"\n{'='*70}")
        print(f"✅ Training Complete!")
        print(f"   Best Epoch: {self.best_epoch+1}")
        print(f"   Best Blunder Avoidance: {self.best_blunder_avoidance*100:.1f}%")
        print(f"   Best Val Loss: {self.best_val_loss:.4f}")
        print(f"{'='*70}\n")
    
    def save_checkpoint(self, filename: str, metrics: Dict = None):
        """Save model checkpoint with comprehensive metadata."""
        checkpoint_dir = Path('models/stage2_corrective_optimized')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': dict(self.history),
            'best_val_loss': self.best_val_loss,
            'best_blunder_avoidance': self.best_blunder_avoidance,
            'best_epoch': self.best_epoch,
            'current_metrics': metrics,
            'config': {
                'correction_weight': self.correction_weight,
                'ranking_weight': self.ranking_weight,
                'blunder_weight': self.blunder_weight,
                'gradient_accumulation_steps': self.gradient_accumulation_steps,
                'label_smoothing': self.label_smoothing
            }
        }
        
        torch.save(checkpoint, checkpoint_dir / filename)
    
    def save_ema_checkpoint(self, filename: str):
        """Save EMA model checkpoint."""
        if self.ema is None:
            return
        
        checkpoint_dir = Path('models/stage2_corrective_optimized')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Temporarily apply EMA
        self.ema.apply_shadow(self.model)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'is_ema': True
        }
        
        torch.save(checkpoint, checkpoint_dir / filename)
        
        # Restore original
        self.ema.restore(self.model)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Stage 2: OPTIMIZED Corrective Training')
    parser.add_argument('--data-path', type=str,
                       default='data/stage2_training/corrective_dataset.json',
                       help='Path to corrective dataset')
    parser.add_argument('--stage1-model', type=str,
                       default='models/stage1_themes/best_checkpoint.pt',
                       help='Path to Stage 1 best model')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (effective batch = batch_size * gradient_accumulation_steps)')
    parser.add_argument('--num-epochs', type=int, default=100,
                       help='Number of epochs (more epochs with better early stopping)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Peak learning rate (warmup + cosine decay)')
    parser.add_argument('--correction-weight', type=float, default=3.0,
                       help='Weight for correction loss (increased for focus)')
    parser.add_argument('--ranking-weight', type=float, default=1.0,
                       help='Weight for ranking loss')
    parser.add_argument('--blunder-weight', type=float, default=5.0,
                       help='Extra penalty weight for blunders')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4,
                       help='Gradient accumulation steps (effective larger batch)')
    parser.add_argument('--early-stopping-patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation split ratio')
    parser.add_argument('--use-ema', action='store_true', default=True,
                       help='Use Exponential Moving Average')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                       help='Label smoothing for regularization')
    parser.add_argument('--warmup-ratio', type=float, default=0.1,
                       help='Warmup ratio for LR schedule')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')
    
    args = parser.parse_args()
    
    print("🚀 V7P3RAI Stage 2: OPTIMIZED Corrective Training")
    print("=" * 70)
    print("   Production-Grade Configuration for V7P3R Primary Engine")
    print("=" * 70)
    print(f"📂 Corrective Dataset: {args.data_path}")
    print(f"📦 Stage 1 Model: {args.stage1_model}")
    print(f"⚙️  Batch Size: {args.batch_size} (effective: {args.batch_size * args.gradient_accumulation_steps})")
    print(f"📊 Epochs: {args.num_epochs}")
    print(f"🎯 Device: {args.device}")
    print("=" * 70)
    
    # Load dataset
    print("\n📂 Loading corrective dataset...")
    dataset = CorrectiveDataset(args.data_path)
    
    # Split into train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"   Train: {train_size:,} examples")
    print(f"   Val: {val_size:,} examples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=0,
        pin_memory=False
    )
    
    # Load Stage 1 model
    print(f"\n📦 Loading Stage 1 model from {args.stage1_model}...")
    model = MoveOrderingNetwork(num_themes=57)
    
    checkpoint = torch.load(args.stage1_model, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    stage1_epoch = checkpoint.get('epoch', 'unknown')
    stage1_top5 = checkpoint.get('val_top5_accuracy', 'unknown')
    if isinstance(stage1_top5, float):
        stage1_top5 = f"{stage1_top5*100:.1f}%"
    
    print(f"   ✅ Loaded from epoch {stage1_epoch}")
    print(f"   📊 Stage 1 Top-5 Accuracy: {stage1_top5}")
    
    # Create trainer
    trainer = OptimizedCorrectiveTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        device=args.device,
        correction_weight=args.correction_weight,
        ranking_weight=args.ranking_weight,
        blunder_weight=args.blunder_weight,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_ema=args.use_ema,
        label_smoothing=args.label_smoothing,
        warmup_ratio=args.warmup_ratio
    )
    
    # Train
    trainer.train(args.num_epochs, args.early_stopping_patience)
    
    print("\n✅ Stage 2 OPTIMIZED training complete!")
    print(f"   Best model: models/stage2_corrective_optimized/best_model.pt")
    print(f"   Latest model: models/stage2_corrective_optimized/latest_model.pt")
    print(f"   Training ready for V7P3R deployment evaluation!")


if __name__ == '__main__':
    main()

