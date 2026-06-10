"""
Stage 1 v2.0 Training Script (Heuristic-Based Labels)

Trains Position Evaluator with heuristic sentiment-based GOOD/BAD labels
instead of game outcome-based labels.

Key Differences from v1.0:
- Labels based on weighted_sentiment_delta (objective positional heuristics)
- 300k positions (vs 1.648M) - smaller but higher quality
- Same 19-feature architecture (features unchanged)
- Trains from scratch (not incremental)

Philosophy: "What would a chess player look at first?" not "Who won the game?"
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from pathlib import Path
import chess

# Configuration
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "stage1" / "heuristic_labeled"
MODEL_DIR = Path(__file__).parent.parent.parent / "models"
GOOD_POSITIONS_PATH = DATA_DIR / "heuristic_good_positions.jsonl"
BAD_POSITIONS_PATH = DATA_DIR / "heuristic_bad_positions.jsonl"

BATCH_SIZE = 512
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Feature extraction (identical to v1.0)
def extract_fast_features(fen: str) -> np.ndarray:
    """Extract 19-dimensional fast features from FEN."""
    board = chess.Board(fen)
    features = []
    
    # 12 piece counts (white pieces, then black pieces)
    for color in [chess.WHITE, chess.BLACK]:
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
            count = len(board.pieces(piece_type, color))
            features.append(count)
    
    # Material balance (white - black)
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    white_material = sum(len(board.pieces(pt, chess.WHITE)) * piece_values.get(pt, 0) for pt in piece_values.keys())
    black_material = sum(len(board.pieces(pt, chess.BLACK)) * piece_values.get(pt, 0) for pt in piece_values.keys())
    features.append(white_material - black_material)
    
    # Side to move (1 for white, -1 for black)
    features.append(1 if board.turn == chess.WHITE else -1)
    
    # Castling rights (white kingside, white queenside)
    features.append(1 if board.has_kingside_castling_rights(chess.WHITE) else 0)
    features.append(1 if board.has_queenside_castling_rights(chess.WHITE) else 0)
    
    # In check
    features.append(1 if board.is_check() else 0)
    
    # Mobility (legal moves for current side and opponent)
    current_mobility = len(list(board.legal_moves))
    board.turn = not board.turn
    opponent_mobility = len(list(board.legal_moves))
    features.append(current_mobility)
    features.append(opponent_mobility)
    
    return np.array(features, dtype=np.float32)

# Dataset class
class PositionDataset(Dataset):
    def __init__(self, positions, labels):
        self.positions = positions
        self.labels = labels
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        return torch.tensor(self.positions[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# Model architecture (identical to v1.0)
class PositionEvaluator(nn.Module):
    def __init__(self, input_size=19):
        super(PositionEvaluator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

def load_positions_from_jsonl(file_path):
    """Load positions from JSONL file."""
    positions = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            positions.append(data['fen'])
    return positions

def load_balanced_positions():
    """Load balanced GOOD/BAD positions."""
    print("📊 Loading heuristic-labeled positions...")
    
    # Load GOOD positions
    good_fens = load_positions_from_jsonl(GOOD_POSITIONS_PATH)
    print(f"  ✅ GOOD positions: {len(good_fens):,}")
    
    # Load BAD positions
    bad_fens = load_positions_from_jsonl(BAD_POSITIONS_PATH)
    print(f"  ✅ BAD positions: {len(bad_fens):,}")
    
    # Combine and create labels (1 for GOOD, 0 for BAD)
    all_fens = good_fens + bad_fens
    labels = [1] * len(good_fens) + [0] * len(bad_fens)
    
    print(f"  📊 Total positions: {len(all_fens):,}")
    print(f"  ⚖️  Balance: {len(good_fens)/(len(good_fens)+len(bad_fens))*100:.1f}% GOOD, {len(bad_fens)/(len(good_fens)+len(bad_fens))*100:.1f}% BAD")
    
    return all_fens, labels

def create_dataloaders(fens, labels):
    """Create train/validation dataloaders."""
    print("\n📦 Creating dataloaders...")
    
    # Extract features
    print("  Extracting features from FENs...")
    features = []
    for i, fen in enumerate(fens):
        if (i + 1) % 50000 == 0:
            print(f"    Processed {i+1:,}/{len(fens):,} positions...")
        features.append(extract_fast_features(fen))
    features = np.array(features)
    labels = np.array(labels)
    
    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Save scaler
    scaler_path = MODEL_DIR / "scaler_v2.0.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  💾 Saved scaler to {scaler_path}")
    
    # Create datasets and dataloaders
    train_dataset = PositionDataset(X_train, y_train)
    val_dataset = PositionDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"  ✅ Train set: {len(train_dataset):,} positions")
    print(f"  ✅ Val set: {len(val_dataset):,} positions")
    
    return train_loader, val_loader

def train_epoch(model, train_loader, criterion, optimizer):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE).unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct += (predicted == targets).sum().item()
        total += targets.size(0)
    
    return total_loss / len(train_loader), correct / total

def validate(model, val_loader):
    """Validate model."""
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0
    
    criterion = nn.BCELoss()
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE).unsqueeze(1)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            predicted = (outputs > 0.5).float()
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, zero_division=0)
    recall = recall_score(all_targets, all_predictions, zero_division=0)
    f1 = f1_score(all_targets, all_predictions, zero_division=0)
    
    return total_loss / len(val_loader), accuracy, precision, recall, f1

def main():
    """Main training pipeline."""
    print("="*60)
    print("STAGE 1 v2.0 TRAINING (HEURISTIC-BASED LABELS)")
    print("="*60)
    print(f"🔧 Device: {DEVICE}")
    print(f"📊 Data source: Heuristic sentiment-based labels")
    print(f"🎯 Training from scratch (NOT incremental)")
    
    # Load data
    fens, labels = load_balanced_positions()
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(fens, labels)
    
    # Create model
    model = PositionEvaluator().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\n🎯 Training Configuration:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Optimizer: Adam")
    print(f"  Loss: Binary Cross-Entropy")
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING PROGRESS")
    print("="*60)
    print(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<10} {'F1':<10}")
    print("-" * 60)
    
    best_f1 = 0
    best_epoch = 0
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_precision, val_recall, val_f1 = validate(model, val_loader)
        
        print(f"{epoch+1:<6} {train_loss:<12.4f} {train_acc:<12.4f} {val_loss:<12.4f} {val_acc:<10.4f} {val_f1:<10.4f}")
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            model_path = MODEL_DIR / "position_evaluator_v2.0.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': val_f1,
                'accuracy': val_acc,
                'precision': val_precision,
                'recall': val_recall
            }, model_path)
            print(f"Model saved to {model_path}")
    
    print("-" * 60)
    print(f"✅ Best F1: {best_f1:.4f} at epoch {best_epoch}")
    
    # Final evaluation
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    # Load best model and evaluate
    checkpoint = torch.load(MODEL_DIR / "position_evaluator_v2.0.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    val_loss, val_acc, val_precision, val_recall, val_f1 = validate(model, val_loader)
    
    print(f"\n📊 Final Metrics (Best Model):")
    print(f"  Validation Accuracy:  {val_acc:.4f}")
    print(f"  Validation Precision: {val_precision:.4f}")
    print(f"  Validation Recall:    {val_recall:.4f}")
    print(f"  Validation F1:        {val_f1:.4f}")
    
    print(f"\n📈 Comparison to v1.1 (Outcome-Based Labels):")
    v1_1_f1 = 0.8957
    v1_1_acc = 0.8946
    print(f"  F1 Score:   {v1_1_f1:.4f} → {val_f1:.4f} ({val_f1 - v1_1_f1:+.4f})")
    print(f"  Accuracy:   {v1_1_acc:.4f} → {val_acc:.4f} ({val_acc - v1_1_acc:+.4f})")
    
    if val_f1 > v1_1_f1:
        print(f"\n  🎉 IMPROVEMENT: v2.0 outperforms v1.1!")
    elif val_f1 > v1_1_f1 - 0.02:
        print(f"\n  ✅ ACCEPTABLE: v2.0 performance within 2% of v1.1")
    else:
        print(f"\n  ⚠️  WARNING: v2.0 underperforms v1.1 by {v1_1_f1 - val_f1:.4f}")
    
    print(f"\n💾 Model saved to: {MODEL_DIR / 'position_evaluator_v2.0.pth'}")
    print(f"💾 Scaler saved to: {MODEL_DIR / 'scaler_v2.0.pkl'}")
    
    print(f"\n📝 Next Steps:")
    print(f"  1. Test v2.0 on held-out positions not in training set")
    print(f"  2. Run self-play with v2.0 vs v1.1 (50 games each)")
    print(f"  3. Compare move selection behavior")
    print(f"  4. If v2.0 performs well, use for Stage 2 training")
    print(f"  5. Document findings in version log")
    
    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()
