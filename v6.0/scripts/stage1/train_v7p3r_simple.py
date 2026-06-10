"""
V7P3R AI v6.1 - Simple Training Script
Uses only V7P3R positions with pre-calculated features for fast training
"""

import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Training Configuration
CONFIG = {
    'epochs': 10,
    'batch_size': 256,
    'learning_rate': 0.001,
    'hidden_dims': [1024, 512, 256, 128],
    'dropout': 0.3,
    'train_val_split': 0.8,
    'random_seed': 42,
    'max_positions': 50000,  # Load up to 50k positions
}

# Set random seeds
random.seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])
torch.manual_seed(CONFIG['random_seed'])

class PositionDataset(Dataset):
    """PyTorch Dataset for chess positions"""
    def __init__(self, features, labels, weights=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.weights = torch.FloatTensor(weights) if weights is not None else torch.ones(len(labels))
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.weights[idx]

class PositionEvaluator(nn.Module):
    """Graph-Augmented Policy Network for position evaluation"""
    def __init__(self, input_dim, hidden_dims, dropout=0.3):
        super().__init__()
        
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
    
    def forward(self, x):
        return self.network(x).squeeze()

def load_v7p3r_positions(max_positions=50000):
    """Load positions from V7P3R bad_positions.jsonl with numeric features"""
    print("\n🔄 Loading V7P3R positions...")
    
    data_path = project_root / "data" / "stage1" / "bad_positions.jsonl"
    
    if not data_path.exists():
        raise FileNotFoundError(f"V7P3R data not found: {data_path}")
    
    all_features = []
    all_labels = []
    all_weights = []
    
    positions_loaded = 0
    positions_skipped = 0
    
    with open(data_path, 'r') as f:
        for line in f:
            if positions_loaded >= max_positions:
                break
            
            try:
                record = json.loads(line)
                
                # Check if position has numeric features
                if 'features' in record and record['features']:
                    features = record['features']
                    
                    # Validate all features are numeric
                    if all(isinstance(f, (int, float)) for f in features):
                        all_features.append(features)
                        all_labels.append(record.get('label', 0))
                        all_weights.append(record.get('weight', 1.0))
                        positions_loaded += 1
                    else:
                        positions_skipped += 1
                else:
                    positions_skipped += 1
                    
            except Exception as e:
                positions_skipped += 1
                continue
    
    print(f"  ✅ Loaded {positions_loaded:,} positions")
    print(f"  ⚠  Skipped {positions_skipped:,} positions (missing/invalid features)")
    
    if positions_loaded == 0:
        raise ValueError("No valid positions loaded!")
    
    return np.array(all_features), np.array(all_labels), np.array(all_weights)

def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for features, labels, weights in train_loader:
        features = features.to(device)
        labels = labels.to(device)
        weights = weights.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        
        # Weighted loss
        loss = criterion(outputs, labels)
        loss = (loss * weights).mean()
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Track predictions
        preds = (outputs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels, weights in val_loader:
            features = features.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss = (loss * weights).mean()
            
            total_loss += loss.item()
            
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1

def main():
    print("=" * 70)
    print("V7P3R AI v6.1 - SIMPLE TRAINING")
    print("Stage 1: Position Evaluator")
    print("=" * 70)
    
    print(f"\n📋 Training Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    # Load data
    print("\n🚀 Loading V7P3R training data...")
    features, labels, weights = load_v7p3r_positions(CONFIG['max_positions'])
    
    # Check feature vector consistency
    print(f"\n🔍 Checking feature vectors...")
    feature_lengths = [len(f) for f in features]
    unique_lengths = set(feature_lengths)
    print(f"  Feature vector lengths: {unique_lengths}")
    
    if len(unique_lengths) > 1:
        # Filter to most common length
        from collections import Counter
        most_common_length = Counter(feature_lengths).most_common(1)[0][0]
        print(f"  Filtering to length {most_common_length}...")
        
        mask = [len(f) == most_common_length for f in features]
        features = features[mask]
        labels = labels[mask]
        weights = weights[mask]
        
        print(f"  ✅ Retained {len(features):,} positions")
    
    input_dim = len(features[0])
    print(f"  Input dimension: {input_dim}")
    
    # Normalize features
    print(f"\n📊 Normalizing features...")
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # Split data
    print(f"\n✂️  Splitting into train/validation...")
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        features, labels, weights,
        test_size=1 - CONFIG['train_val_split'],
        random_state=CONFIG['random_seed'],
        stratify=labels
    )
    
    print(f"  Training set: {len(X_train):,} positions")
    print(f"  Validation set: {len(X_val):,} positions")
    
    # Create datasets and dataloaders
    train_dataset = PositionDataset(X_train, y_train, w_train)
    val_dataset = PositionDataset(X_val, y_val, w_val)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    
    # Create model
    print(f"\n🏗️  Building model...")
    model = PositionEvaluator(
        input_dim=input_dim,
        hidden_dims=CONFIG['hidden_dims'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss(reduction='none')  # No reduction for weighted loss
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Training loop
    print(f"\n🎯 Starting training for {CONFIG['epochs']} epochs...")
    print("=" * 70)
    
    best_val_f1 = 0
    
    for epoch in range(CONFIG['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(model, val_loader, criterion, device)
        
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
              f"Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_path = project_root / "models" / "position_evaluator_best.pth"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': scaler,
                'config': CONFIG,
                'val_metrics': {
                    'accuracy': val_acc,
                    'precision': val_prec,
                    'recall': val_rec,
                    'f1': val_f1,
                }
            }, save_path)
            print(f"  💾 Saved best model (F1: {best_val_f1:.4f})")
    
    print("\n" + "=" * 70)
    print(f"✅ Training complete!")
    print(f"🏆 Best validation F1: {best_val_f1:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
