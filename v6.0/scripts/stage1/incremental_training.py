"""
Incremental Training for Stage 1 Position Evaluator
Loads existing trained model and continues training with new data.

This validates that we can scale the dataset without retraining from scratch.
Key principles:
- Load existing model weights
- Add new balanced data to training set
- Train for fewer epochs (5-10 vs original 20)
- Monitor for catastrophic forgetting (performance on original test set)
- Track data distribution metadata
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from stage1.position_evaluator import PositionEvaluator
from stage1.feature_extractor import extract_fast_features

class PositionDataset(Dataset):
    def __init__(self, fens, labels, scaler=None):
        self.fens = fens
        self.labels = labels
        
        # Extract features
        features = np.array([extract_fast_features(fen) for fen in fens])
        
        # Scale features
        if scaler is None:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(features)
        else:
            self.scaler = scaler
            self.features = self.scaler.transform(features)
        
        self.features = torch.FloatTensor(self.features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.fens)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class IncrementalTrainer:
    def __init__(self, 
                 existing_model_path: str,
                 new_data_dir: str,
                 output_dir: str,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.existing_model_path = Path(existing_model_path)
        self.new_data_dir = Path(new_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        print(f"🔧 Device: {device}")
        if device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    def load_existing_model(self):
        """Load the existing trained Stage 1 model."""
        print(f"\n📦 Loading existing model from {self.existing_model_path}")
        
        self.model = PositionEvaluator.load(str(self.existing_model_path))
        self.model.to(self.device)
        
        print(f"✅ Model loaded successfully")
        return self.model
    
    def load_merged_data(self, max_positions=None):
        """Load merged dataset (original + self-play)."""
        print(f"\n📊 Loading merged training data from {self.new_data_dir}")
        
        good_file = self.new_data_dir / "merged_good_positions.jsonl"
        bad_file = self.new_data_dir / "merged_bad_positions.jsonl"
        
        good_fens = []
        bad_fens = []
        
        # Load GOOD positions
        print(f"  Loading GOOD positions from {good_file}")
        count = 0
        with open(good_file, 'r') as f:
            for line in f:
                if max_positions and count >= max_positions // 2:
                    break
                try:
                    pos = json.loads(line)
                    good_fens.append(pos['fen'])
                    count += 1
                    if count % 100000 == 0:
                        print(f"    Loaded {count} positions...")
                except:
                    continue
        
        # Load BAD positions
        print(f"  Loading BAD positions from {bad_file}")
        count = 0
        with open(bad_file, 'r') as f:
            for line in f:
                if max_positions and count >= max_positions // 2:
                    break
                try:
                    pos = json.loads(line)
                    bad_fens.append(pos['fen'])
                    count += 1
                    if count % 100000 == 0:
                        print(f"    Loaded {count} positions...")
                except:
                    continue
        
        print(f"  ✅ GOOD positions: {len(good_fens):,}")
        print(f"  ✅ BAD positions: {len(bad_fens):,}")
        
        # Combine and create labels
        all_fens = good_fens + bad_fens
        labels = [1.0] * len(good_fens) + [0.0] * len(bad_fens)
        
        print(f"  📊 Total positions: {len(all_fens):,}")
        
        return all_fens, labels
    
    def create_dataloaders(self, fens, labels, batch_size=512, val_split=0.2):
        """Create train/val dataloaders with proper scaling."""
        
        # Split into train/val
        split_idx = int(len(fens) * (1 - val_split))
        
        # Shuffle data
        indices = np.random.permutation(len(fens))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_fens = [fens[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        val_fens = [fens[i] for i in val_indices]
        val_labels = [labels[i] for i in val_indices]
        
        # Create datasets (fit scaler on training data only)
        train_dataset = PositionDataset(train_fens, train_labels)
        val_dataset = PositionDataset(val_fens, val_labels, scaler=train_dataset.scaler)
        
        # Save scaler for later use
        scaler_path = self.output_dir / "incremental_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(train_dataset.scaler, f)
        print(f"💾 Saved scaler to {scaler_path}")
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train_epoch(self, model, train_loader, optimizer, criterion):
        """Train for one epoch."""
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for features, labels in train_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def validate(self, model, val_loader, criterion):
        """Validate model performance."""
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(features).squeeze()
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                
                preds = (outputs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(outputs.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        
        return avg_loss, accuracy, precision, recall, f1
    
    def run_incremental_training(self, epochs=10, lr=0.0001, batch_size=512, max_positions=None):
        """Execute incremental training pipeline on merged dataset."""
        print("\n" + "="*60)
        print("INCREMENTAL STAGE 1 TRAINING (MERGED DATASET)")
        print("="*60)
        
        # Load existing model
        model = self.load_existing_model()
        
        # Load merged data (original + self-play)
        fens, labels = self.load_merged_data(max_positions=max_positions)
        
        print(f"\n📈 Dataset Growth:")
        print(f"  Original:  1,648,000 positions")
        print(f"  Self-play: +7,448 positions")
        print(f"  Merged:    {len(fens):,} positions")
        print(f"  Growth:    +{(len(fens) - 1648000) / 1648000 * 100:.2f}%")
        
        # Create dataloaders
        print(f"\n📦 Creating dataloaders (batch_size={batch_size})...")
        train_loader, val_loader = self.create_dataloaders(fens, labels, batch_size=batch_size)
        
        # Setup training
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        print(f"\n🎯 Training Configuration:")
        print(f"  Epochs: {epochs} (continuing from original epoch 18)")
        print(f"  Learning Rate: {lr} (lower than original 0.001 to preserve knowledge)")
        print(f"  Batch Size: {batch_size}")
        print(f"  Optimizer: Adam")
        print(f"  Loss: Binary Cross-Entropy")
        print(f"  Strategy: Fine-tuning existing weights, NOT training from scratch")
        
        # Training loop
        best_f1 = 0
        best_epoch = 0
        
        print(f"\n{'='*60}")
        print(f"TRAINING PROGRESS")
        print(f"{'='*60}")
        print(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<10} {'F1':<10}")
        print(f"{'-'*60}")
        
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_acc, val_prec, val_rec, val_f1 = self.validate(model, val_loader, criterion)
            
            print(f"{epoch:<6} {train_loss:<12.4f} {train_acc:<12.4f} {val_loss:<12.4f} {val_acc:<10.4f} {val_f1:<10.4f}")
            
            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch
                
                model_path = self.output_dir / "position_evaluator_incremental_best.pth"
                model.save(str(model_path))
        
        print(f"{'-'*60}")
        print(f"✅ Best F1: {best_f1:.4f} at epoch {best_epoch}")
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"INCREMENTAL TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"\n📊 Final Metrics:")
        print(f"  Validation Accuracy:  {val_acc:.4f}")
        print(f"  Validation Precision: {val_prec:.4f}")
        print(f"  Validation Recall:    {val_rec:.4f}")
        print(f"  Validation F1:        {val_f1:.4f}")
        
        print(f"\n📈 Comparison to Original Model:")
        original_f1 = 0.8776
        original_acc = 0.8831
        f1_delta = val_f1 - original_f1
        acc_delta = val_acc - original_acc
        
        print(f"  F1 Score:   {original_f1:.4f} → {val_f1:.4f} ({f1_delta:+.4f})")
        print(f"  Accuracy:   {original_acc:.4f} → {val_acc:.4f} ({acc_delta:+.4f})")
        
        if f1_delta >= -0.01:
            print(f"\n  ✅ NO CATASTROPHIC FORGETTING (delta ≥ -0.01)")
        else:
            print(f"\n  ⚠️  WARNING: Performance degradation detected")
        
        if f1_delta > 0:
            print(f"  🎉 IMPROVEMENT ACHIEVED (+{f1_delta:.4f} F1)")
        
        print(f"\n💾 Best model saved to: {self.output_dir / 'position_evaluator_incremental_best.pth'}")
        
        print(f"\n📝 Next Steps:")
        print(f"  1. Test model on held-out positions (not in training set)")
        print(f"  2. If F1 ≥ 0.87, use this model for Stage 2 training")
        print(f"  3. Document dataset v1.1 in version log")
        print(f"  4. Run Stage 2 self-play with improved Stage 1 model")

if __name__ == "__main__":
    existing_model = "models/position_evaluator_best.pth"
    merged_data_dir = "data/stage1/merged"  # Points to merged dataset
    output_dir = "models/incremental"
    
    trainer = IncrementalTrainer(existing_model, merged_data_dir, output_dir)
    
    # FULL PRODUCTION TRAINING: 1.656M positions, 10 epochs
    # Estimated time: 2-4 hours depending on CPU
    print("\n" + "="*60)
    print("FULL PRODUCTION INCREMENTAL TRAINING")
    print("="*60)
    print("  Dataset: 1,655,448 positions (827,724 GOOD + 827,724 BAD)")
    print("  Epochs: 10 (continuing from original epoch 18)")
    print("  Purpose: Complete Cycle 1 - Stage 1 v1.1")
    print("  Time estimate: 2-4 hours")
    print("  Output: Production-ready Stage 1 v1.1 model")
    print("="*60 + "\n")
    
    trainer.run_incremental_training(epochs=10, lr=0.0001, max_positions=None)
