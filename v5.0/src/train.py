"""
V7P3R AI v5.0 - Training Script

Trains the dual-head neural network on preprocessed position data.

Features:
- Dual-head loss (policy + value)
- Early stopping with patience
- Model checkpointing (save best)
- Metrics tracking and logging
- Resume training capability
- Learning rate scheduling

Usage:
    python src/train.py --config configs/training_config.yaml
    python src/train.py --resume checkpoints/best_model.pth
"""

import argparse
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from datetime import datetime
import time
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from model import V7P3R_AI_v5, create_model
from dataset import V7P3RDataset, create_dataloaders


class Trainer:
    """Training manager for V7P3R AI v5.0"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Create model
        self.model = create_model(config['model']).to(self.device)
        print(f"\nModel parameters: {self.model.count_parameters()['total']:,}")
        
        # Class weights for handling imbalanced grades
        # Grade distribution: 0=24.6%, 1=4.5%, 2=6.2%, 3=9.0%, 4=15.1%, 5=40.6%
        # Weight rare classes more to force model to learn all grades
        class_weights = torch.tensor([1.0, 5.0, 3.5, 2.5, 1.8, 1.0]).to(self.device)
        
        # Loss functions
        self.policy_criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.value_criterion = nn.HuberLoss(delta=config['training']['huber_delta'])
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config['training']['lr_factor'],
            patience=config['training']['lr_patience'],
            min_lr=config['training']['min_lr']
        )
        
        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.history = {
            'train_loss': [],
            'train_policy_loss': [],
            'train_value_loss': [],
            'train_policy_acc': [],
            'val_loss': [],
            'val_policy_loss': [],
            'val_value_loss': [],
            'val_policy_acc': [],
            'val_value_mae': [],
            'learning_rates': []
        }
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config['training']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint for resuming training"""
        print(f"\nLoading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        print(f"Resumed from epoch {checkpoint['epoch']}")
        print(f"Best val loss: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"  💾 Saved best model (val_loss: {val_loss:.4f})")
        
        # Save periodic checkpoints
        if (epoch + 1) % self.config['training']['save_every'] == 0:
            epoch_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save(checkpoint, epoch_path)
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        policy_correct = 0
        total_samples = 0
        
        policy_weight = self.config['training']['policy_weight']
        value_weight = self.config['training']['value_weight']
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            features = batch['features'].to(self.device)
            policy_targets = batch['policy_target'].to(self.device)
            value_targets = batch['value_target'].to(self.device)
            
            # Forward pass
            policy_logits, value_preds = self.model(features)
            
            # Compute losses
            policy_loss = self.policy_criterion(policy_logits, policy_targets)
            value_loss = self.value_criterion(value_preds, value_targets)
            
            loss = policy_weight * policy_loss + value_weight * value_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['grad_clip']
            )
            
            self.optimizer.step()
            
            # Accumulate metrics
            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            total_policy_loss += policy_loss.item() * batch_size
            total_value_loss += value_loss.item() * batch_size
            policy_correct += (policy_logits.argmax(1) == policy_targets).sum().item()
            total_samples += batch_size
            
            # Print progress
            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)}: "
                      f"Loss={loss.item():.4f}, "
                      f"Policy Acc={policy_correct/total_samples:.3f}")
        
        return {
            'loss': total_loss / total_samples,
            'policy_loss': total_policy_loss / total_samples,
            'value_loss': total_value_loss / total_samples,
            'policy_accuracy': policy_correct / total_samples
        }
    
    def validate(self, val_loader):
        """Validate on validation set"""
        self.model.eval()
        
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        policy_correct = 0
        value_mae = 0
        total_samples = 0
        
        policy_weight = self.config['training']['policy_weight']
        value_weight = self.config['training']['value_weight']
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(self.device)
                policy_targets = batch['policy_target'].to(self.device)
                value_targets = batch['value_target'].to(self.device)
                
                policy_logits, value_preds = self.model(features)
                
                policy_loss = self.policy_criterion(policy_logits, policy_targets)
                value_loss = self.value_criterion(value_preds, value_targets)
                
                loss = policy_weight * policy_loss + value_weight * value_loss
                
                batch_size = features.size(0)
                total_loss += loss.item() * batch_size
                total_policy_loss += policy_loss.item() * batch_size
                total_value_loss += value_loss.item() * batch_size
                policy_correct += (policy_logits.argmax(1) == policy_targets).sum().item()
                value_mae += torch.abs(value_preds - value_targets).sum().item()
                total_samples += batch_size
        
        return {
            'loss': total_loss / total_samples,
            'policy_loss': total_policy_loss / total_samples,
            'value_loss': total_value_loss / total_samples,
            'policy_accuracy': policy_correct / total_samples,
            'value_mae': value_mae / total_samples
        }
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        print("\n" + "=" * 80)
        print("Starting Training")
        print("=" * 80)
        
        num_epochs = self.config['training']['epochs']
        early_stopping_patience = self.config['training']['early_stopping_patience']
        
        for epoch in range(self.start_epoch, num_epochs):
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 80)
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_policy_loss'].append(train_metrics['policy_loss'])
            self.history['train_value_loss'].append(train_metrics['value_loss'])
            self.history['train_policy_acc'].append(train_metrics['policy_accuracy'])
            
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_policy_loss'].append(val_metrics['policy_loss'])
            self.history['val_value_loss'].append(val_metrics['value_loss'])
            self.history['val_policy_acc'].append(val_metrics['policy_accuracy'])
            self.history['val_value_mae'].append(val_metrics['value_mae'])
            self.history['learning_rates'].append(current_lr)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\n📊 Epoch {epoch+1} Summary ({epoch_time:.1f}s):")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Policy Acc: {train_metrics['policy_accuracy']:.3f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Policy Acc: {val_metrics['policy_accuracy']:.3f}, "
                  f"Value MAE: {val_metrics['value_mae']:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # Check for improvement
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics['loss'], is_best)
            
            # Early stopping
            if self.epochs_without_improvement >= early_stopping_patience:
                print(f"\n⚠️ Early stopping triggered (patience: {early_stopping_patience})")
                print(f"Best val loss: {self.best_val_loss:.4f}")
                break
        
        # Save training history
        self._save_history()
        
        print("\n" + "=" * 80)
        print("✅ Training Complete!")
        print("=" * 80)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Model saved to: {self.checkpoint_dir / 'best_model.pth'}")
    
    def _save_history(self):
        """Save training history to JSON"""
        history_path = self.checkpoint_dir / 'training_history.json'
        
        history_data = {
            'config': self.config,
            'metrics': self.history,
            'best_val_loss': self.best_val_loss,
            'total_epochs': len(self.history['train_loss'])
        }
        
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        print(f"\n💾 Training history saved to: {history_path}")


def load_config(config_path):
    """Load training configuration from YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_preprocessed_data(data_dir):
    """Load preprocessed NumPy arrays"""
    data_dir = Path(data_dir)
    
    print("Loading preprocessed data...")
    
    X_train = np.load(data_dir / 'X_train.npy')
    y_train_policy = np.load(data_dir / 'y_train_policy.npy')
    y_train_value = np.load(data_dir / 'y_train_value.npy')
    
    X_val = np.load(data_dir / 'X_val.npy')
    y_val_policy = np.load(data_dir / 'y_val_policy.npy')
    y_val_value = np.load(data_dir / 'y_val_value.npy')
    
    X_test = np.load(data_dir / 'X_test.npy')
    y_test_policy = np.load(data_dir / 'y_test_policy.npy')
    y_test_value = np.load(data_dir / 'y_test_value.npy')
    
    print(f"  Train: {X_train.shape[0]:,} positions")
    print(f"  Val:   {X_val.shape[0]:,} positions")
    print(f"  Test:  {X_test.shape[0]:,} positions")
    
    return (X_train, {'policy': y_train_policy, 'value': y_train_value},
            X_val, {'policy': y_val_policy, 'value': y_val_value},
            X_test, {'policy': y_test_policy, 'value': y_test_value})


def main():
    parser = argparse.ArgumentParser(description='Train V7P3R AI v5.0')
    parser.add_argument('--config', type=str, 
                        default='configs/training_config.yaml',
                        help='Path to training config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--data-dir', type=str,
                        default='data/preprocessed',
                        help='Directory with preprocessed data')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("V7P3R AI v5.0 - Training Pipeline")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load config
    config = load_config(args.config)
    print(f"\nConfig loaded from: {args.config}")
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_preprocessed_data(args.data_dir)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    # Create trainer
    trainer = Trainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(train_loader, val_loader)
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
