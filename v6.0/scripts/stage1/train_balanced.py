"""
V7P3R AI v6.1 - Balanced Training
Stage 1: Position Evaluator (GOOD + BAD positions)
Uses fast feature extraction with balanced 50/50 dataset
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
import chess

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Training Configuration
CONFIG = {
    'epochs': 20,
    'batch_size': 512,
    'learning_rate': 0.001,
    'hidden_dims': [512, 256, 128],
    'dropout': 0.3,
    'train_val_split': 0.8,
    'random_seed': 42,
    'max_positions': 1_648_000,  # Total: 824k good + 824k bad (balanced from massive dataset)
}

# Set random seeds
random.seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])
torch.manual_seed(CONFIG['random_seed'])

def extract_fast_features(fen):
    """Extract fast, simple features from a FEN string"""
    try:
        board = chess.Board(fen)
        
        features = []
        
        # Piece counts (12 features)
        for color in [chess.WHITE, chess.BLACK]:
            features.append(len(board.pieces(chess.PAWN, color)))
            features.append(len(board.pieces(chess.KNIGHT, color)))
            features.append(len(board.pieces(chess.BISHOP, color)))
            features.append(len(board.pieces(chess.ROOK, color)))
            features.append(len(board.pieces(chess.QUEEN, color)))
            features.append(len(board.pieces(chess.KING, color)))
        
        # Material balance (1 feature)
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                       chess.ROOK: 5, chess.QUEEN: 9}
        white_material = sum(len(board.pieces(pt, chess.WHITE)) * val 
                            for pt, val in piece_values.items())
        black_material = sum(len(board.pieces(pt, chess.BLACK)) * val 
                            for pt, val in piece_values.items())
        features.append(white_material - black_material)
        
        # Positional features (4 features)
        features.append(1 if board.turn == chess.WHITE else 0)  # Side to move
        features.append(board.has_kingside_castling_rights(chess.WHITE))
        features.append(board.has_queenside_castling_rights(chess.WHITE))
        features.append(board.is_check())
        
        # Mobility (2 features)
        features.append(board.legal_moves.count())
        board.turn = not board.turn
        features.append(board.legal_moves.count())
        
        return features
        
    except Exception as e:
        return None

def load_balanced_positions(max_positions=50000):
    """Load balanced dataset: 50% good, 50% bad positions"""
    
    # Calculate how many of each to load
    positions_per_class = max_positions // 2
    
    # Paths to data files
    data_dir = project_root / 'data' / 'stage1'
    good_path = data_dir / 'good_positions.jsonl'
    bad_path = data_dir / 'bad_positions_massive.jsonl'  # Massive 824k bad positions
    
    all_features = []
    all_labels = []
    all_weights = []
    
    # Load GOOD positions (label=1)
    print(f"\n🔄 Loading GOOD positions from {good_path}...")
    if good_path.exists():
        with open(good_path, 'r') as f:
            count = 0
            skipped = 0
            for line in f:
                if count >= positions_per_class:
                    break
                    
                try:
                    record = json.loads(line.strip())
                    fen = record.get('fen')
                    
                    if not fen:
                        skipped += 1
                        continue
                    
                    features = extract_fast_features(fen)
                    if features is None or len(features) == 0:
                        skipped += 1
                        continue
                    
                    all_features.append(features)
                    all_labels.append(1)  # GOOD = 1
                    all_weights.append(record.get('weight', 1.0))
                    count += 1
                    
                    if count % 10000 == 0:
                        print(f"  Processed {count} good positions...")
                        
                except Exception as e:
                    skipped += 1
                    continue
        
        print(f"  ✅ Loaded {count} GOOD positions")
        print(f"  ⚠  Skipped {skipped} positions")
    else:
        print(f"  ⚠️  File not found: {good_path}")
    
    # Load BAD positions (label=0)
    print(f"\n🔄 Loading BAD positions from {bad_path}...")
    if bad_path.exists():
        with open(bad_path, 'r') as f:
            count = 0
            skipped = 0
            for line in f:
                if count >= positions_per_class:
                    break
                    
                try:
                    record = json.loads(line.strip())
                    fen = record.get('fen')
                    
                    if not fen:
                        skipped += 1
                        continue
                    
                    features = extract_fast_features(fen)
                    if features is None or len(features) == 0:
                        skipped += 1
                        continue
                    
                    all_features.append(features)
                    all_labels.append(0)  # BAD = 0
                    all_weights.append(record.get('weight', 1.0))
                    count += 1
                    
                    if count % 10000 == 0:
                        print(f"  Processed {count} bad positions...")
                        
                except Exception as e:
                    skipped += 1
                    continue
        
        print(f"  ✅ Loaded {count} BAD positions")
        print(f"  ⚠  Skipped {skipped} positions")
    else:
        print(f"  ⚠️  File not found: {bad_path}")
    
    # Convert to numpy arrays
    features_array = np.array(all_features, dtype=np.float32)
    labels_array = np.array(all_labels, dtype=np.float32)
    weights_array = np.array(all_weights, dtype=np.float32)
    
    print(f"\n📊 Dataset Summary:")
    print(f"  Total positions: {len(labels_array)}")
    print(f"  GOOD positions (label=1): {np.sum(labels_array == 1)}")
    print(f"  BAD positions (label=0): {np.sum(labels_array == 0)}")
    print(f"  Class balance: {np.sum(labels_array == 1) / len(labels_array) * 100:.1f}% good")
    
    return features_array, labels_array, weights_array

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
    """Simple neural network for position evaluation"""
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

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for features, labels, weights in dataloader:
        features = features.to(device)
        labels = labels.to(device)
        weights = weights.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        
        # Weighted BCE loss
        loss_per_sample = criterion(outputs, labels)
        loss = (loss_per_sample * weights).mean()
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = (outputs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels, weights in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            
            outputs = model(features)
            
            # Weighted BCE loss
            loss_per_sample = criterion(outputs, labels)
            loss = (loss_per_sample * weights).mean()
            
            total_loss += loss.item()
            
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1

def main():
    """Main training function"""
    
    print("=" * 70)
    print("V7P3R AI v6.1 - BALANCED TRAINING")
    print("Stage 1: Position Evaluator (GOOD + BAD)")
    print("=" * 70)
    
    print("\n📋 Training Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    # Load data
    print("\n🚀 Loading balanced training data...")
    features, labels, weights = load_balanced_positions(CONFIG['max_positions'])
    
    if len(features) == 0:
        print("\n❌ No data loaded! Exiting...")
        return
    
    print(f"\n✓ Feature dimension: {features.shape[1]}")
    
    # Normalize features
    print("\n📊 Normalizing features...")
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # Split data
    print("\n✂️  Splitting into train/validation...")
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        features, labels, weights,
        test_size=1-CONFIG['train_val_split'],
        random_state=CONFIG['random_seed'],
        stratify=labels  # Maintain class balance in split
    )
    
    print(f"  Training set: {len(X_train)} positions")
    print(f"    - GOOD: {np.sum(y_train == 1)}")
    print(f"    - BAD: {np.sum(y_train == 0)}")
    print(f"  Validation set: {len(X_val)} positions")
    print(f"    - GOOD: {np.sum(y_val == 1)}")
    print(f"    - BAD: {np.sum(y_val == 0)}")
    
    # Create datasets
    train_dataset = PositionDataset(X_train, y_train, w_train)
    val_dataset = PositionDataset(X_val, y_val, w_val)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    
    # Build model
    print("\n🏗️  Building model...")
    input_dim = features.shape[1]
    model = PositionEvaluator(input_dim, CONFIG['hidden_dims'], CONFIG['dropout']).to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Training loop
    print(f"\n🎯 Starting training for {CONFIG['epochs']} epochs...")
    print("=" * 70)
    
    best_f1 = 0
    best_epoch = 0
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        print(f"\nEpoch {epoch}/{CONFIG['epochs']}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(model, val_loader, criterion, device)
        
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            
            models_dir = project_root / 'models'
            models_dir.mkdir(exist_ok=True)
            
            save_path = models_dir / 'position_evaluator_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler': scaler,
                'config': CONFIG,
                'val_f1': val_f1,
            }, save_path)
    
    print("\n" + "=" * 70)
    print("✅ Training complete!")
    print(f"🏆 Best validation F1: {best_f1:.4f} (epoch {best_epoch})")
    print("=" * 70)

if __name__ == '__main__':
    main()
