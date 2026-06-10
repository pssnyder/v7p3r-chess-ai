"""
V7P3R AI v6.1 - Tal-Inspired Training Script

Train Stage 1 Position Evaluator with Tal-inspired tactical focus.

Data Sources (Tal-Inspired Mix):
- 25% Tal games (GM chaos mastery)
- 19% Your games (Bxf7+ king hunts - 5.0x weighted)
- 19% Tactics puzzles (pattern recognition)
- 19% Opening repertoire (aggressive openings)
- 12% V7P3R engine (baseline knowledge)
- 6% Endgames (conversion skills)

Architecture:
- Input: 76-92 position features
- Graph-Augmented Policy Network
- Transposition attention (K=10 similar positions)
- Hidden layers: [1024, 512, 256, 128]
- Output: Binary classification (GOOD=1, BAD=0)

Training Strategy:
- 10 epochs initial test
- Weighted BCE loss (bad positions 1.5x weight)
- Adam optimizer (LR=0.001)
- 80/20 train/val split
- Batch size: 256
- Validation every epoch
"""

import sys
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler

# Add v6.0 directory to path
v6_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(v6_root))

from scripts.stage1.data_sources.multi_source_loader import MultiSourceDataLoader


class PositionDataset(Dataset):
    """PyTorch dataset for position features."""
    
    def __init__(self, features, labels, weights=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.weights = torch.FloatTensor(weights) if weights is not None else torch.ones(len(labels))
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'label': self.labels[idx],
            'weight': self.weights[idx]
        }


class GraphAugmentedPolicyNetwork(nn.Module):
    """
    Graph-Augmented Neural Network for Position Evaluation.
    
    Simplified from original v6.0 - focuses on core position evaluation
    without full transposition graph (will add back in future iterations).
    """
    
    def __init__(self, input_dim, hidden_dims=[1024, 512, 256, 128], dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, features):
        return self.network(features).squeeze()


def load_tal_inspired_data(batch_size=10000, max_batches=50):
    """
    Load training data using Tal-inspired multi-source mixing.
    
    Args:
        batch_size: Positions per batch from each source
        max_batches: Maximum batches to load (memory limit)
        
    Returns:
        features, labels, weights, source_info
    """
    print("\n" + "="*70)
    print("LOADING TAL-INSPIRED TRAINING DATA")
    print("="*70)
    
    # Define data paths
    base_path = Path("E:/Programming Stuff/Chess Engines/Chess Engine Playground/engine-metrics/raw_data")
    
    lichess_db = base_path / "pgn_training_data/json_data_lichess_evaluations_db/lichess_db_eval.jsonl"
    v7p3r_bad = Path("E:/Programming Stuff/Chess Engines/V7P3R Chess AI/v7p3r-chess-ai/v6.0/data/stage1/bad_positions.jsonl")
    openings_dir = base_path / "pgn_training_data/pgn_data_openings"
    tactics_csv = base_path / "pgn_training_data/csv_data_puzzles"
    endgame_dir = base_path / "pgn_training_data/pgn_data_endgames"
    tal_games = base_path / "pgn_training_data/pgn_data_general/mikhail_tal_master_games.pgn"
    human_games = base_path / "game_records/v7p3r Human/v7p3r_20250530.pgn"
    
    # Initialize multi-source loader
    print("\n🎯 Initializing Tal-Inspired Multi-Source Loader...")
    print("-" * 70)
    
    loader = MultiSourceDataLoader(
        lichess_db_path=str(lichess_db),
        v7p3r_bad_positions=str(v7p3r_bad),
        opening_pgn_dir=str(openings_dir),
        tactics_csv_path=str(tactics_csv),
        endgame_pgn_dir=str(endgame_dir),
        tal_games_pgn=str(tal_games),
        human_games_pgn=str(human_games),
        use_tal_mix=True,  # Use TAL_INSPIRED_MIX ratios
        seed=42,
        shuffle=True
    )
    
    print("\n📊 Mixing Ratios:")
    for source, ratio in sorted(loader.mix_ratios.items(), key=lambda x: -x[1]):
        if ratio > 0:
            print(f"  {source:15s}: {ratio:5.1%}")
    
    # Load data in batches
    print(f"\n🔄 Loading {max_batches} batches of {batch_size:,} positions each...")
    print("-" * 70)
    
    all_features = []
    all_labels = []
    all_weights = []
    all_sources = []
    
    from scripts.utils.calculate_features import FeatureCalculator, FeatureConfig
    feature_calc = FeatureCalculator(FeatureConfig())
    
    for batch_idx in range(max_batches):
        batch = loader.load_batch(
            size=batch_size,
            target_balance={0: 0.5, 1: 0.5}  # 50:50 good/bad
        )
        
        if not batch:
            print(f"  ⚠ No more data at batch {batch_idx}")
            break
        
        batch_features = []
        batch_labels = []
        batch_weights = []
        batch_sources = []
        
        errors_logged = 0
        max_errors_to_log = 5
        
        for pos in batch:
            # Use pre-existing features when available, otherwise skip
            try:
                # Check if position already has features
                if 'features' in pos and pos['features']:
                    features_list = pos['features']
                    
                    # Validate that all features are numeric
                    if not all(isinstance(f, (int, float)) for f in features_list):
                        if errors_logged < max_errors_to_log:
                            non_numeric = [f for f in features_list if not isinstance(f, (int, float))]
                            print(f"    Warning: Non-numeric features in {pos['fen'][:30]}...")
                            print(f"      Sample non-numeric types: {[type(f).__name__ for f in non_numeric[:2]]}")
                            errors_logged += 1
                        continue
                    
                    batch_features.append(features_list)
                    batch_labels.append(pos['label'])
                    batch_weights.append(pos.get('weight', 1.0))
                    batch_sources.append(pos.get('source', 'unknown'))
                else:
                    # Skip positions without pre-calculated features
                    # (Feature calculation is too slow for real-time use)
                    continue
                    
            except Exception as e:
                # Log first few errors to understand what's failing
                if errors_logged < max_errors_to_log:
                    print(f"    Warning: Failed to process position - {type(e).__name__}: {str(e)[:80]}")
                    errors_logged += 1
                continue
        
        if batch_features:
            all_features.extend(batch_features)
            all_labels.extend(batch_labels)
            all_weights.extend(batch_weights)
            all_sources.extend(batch_sources)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Loaded {batch_idx + 1} batches ({len(all_features):,} positions)...")
    
    print(f"\n✅ Loaded {len(all_features):,} total positions")
    
    # Check feature consistency and filter
    print(f"\n🔍 Checking feature vector consistency...")
    feature_lengths = [len(f) for f in all_features]
    from collections import Counter
    length_counts = Counter(feature_lengths)
    print(f"  Feature length distribution: {dict(length_counts)}")
    
    if not length_counts:
        print("\n❌ ERROR: No valid positions loaded! All feature calculations failed.")
        print("This usually means:")
        print("  1. FeatureCalculator is incompatible with the data format")
        print("  2. Required dependencies are missing")
        print("  3. Positions lack necessary data for feature calculation")
        return None, None, None, None
    
    # Use most common feature length
    target_length = length_counts.most_common(1)[0][0]
    print(f"  Target feature length: {target_length}")
    
    # Filter to only positions with correct feature length
    filtered_features = []
    filtered_labels = []
    filtered_weights = []
    filtered_sources = []
    
    for i, (feat, label, weight, source) in enumerate(zip(all_features, all_labels, all_weights, all_sources)):
        if len(feat) == target_length:
            filtered_features.append(feat)
            filtered_labels.append(label)
            filtered_weights.append(weight)
            filtered_sources.append(source)
    
    print(f"  ✓ Kept {len(filtered_features):,} / {len(all_features):,} positions ({100*len(filtered_features)/len(all_features):.1f}%)")
    
    # Double-check all features are exactly target_length and are lists/arrays
    print(f"\n🔍 Validating filtered features...")
    validated_features = []
    validated_labels = []
    validated_weights = []
    validated_sources = []
    
    for i, (feat, label, weight, source) in enumerate(zip(filtered_features, filtered_labels, filtered_weights, filtered_sources)):
        # Convert to list if needed and validate length
        if isinstance(feat, np.ndarray):
            feat_list = feat.tolist()
        else:
            feat_list = list(feat)
        
        if len(feat_list) == target_length:
            # Ensure all elements are floats
            try:
                feat_array = np.array(feat_list, dtype=np.float32)
                if feat_array.shape == (target_length,):
                    validated_features.append(feat_array)
                    validated_labels.append(label)
                    validated_weights.append(weight)
                    validated_sources.append(source)
            except (ValueError, TypeError) as e:
                print(f"  Warning: Skipping position {i} due to conversion error: {e}")
    
    print(f"  ✓ Validated {len(validated_features):,} / {len(filtered_features):,} positions")

    # Convert to numpy arrays
    features = np.array(validated_features, dtype=np.float32)
    labels = np.array(validated_labels, dtype=np.float32)
    weights = np.array(validated_weights, dtype=np.float32)
    weights = weights * (len(weights) / weights.sum())
    
    # Statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total positions: {len(labels):,}")
    print(f"  Feature dimensions: {features.shape[1]}")
    print(f"  Good positions: {np.sum(labels):,.0f} ({np.mean(labels)*100:.1f}%)")
    print(f"  Bad positions: {np.sum(1-labels):,.0f} ({(1-np.mean(labels))*100:.1f}%)")
    print(f"  Average weight: {np.mean(weights):.2f}")
    print(f"  Max weight: {np.max(weights):.2f} (Bxf7+ patterns)")
    
    # Source distribution
    from collections import Counter
    source_counts = Counter(filtered_sources)
    print(f"\n📊 Source Distribution:")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        pct = (count / len(filtered_sources)) * 100
        print(f"  {source:20s}: {count:6,} ({pct:5.1f}%)")
    
    return features, labels, weights, filtered_sources


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in train_loader:
        features = batch['features'].to(device)
        labels = batch['label'].to(device)
        weights = batch['weight'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        
        # Weighted loss
        loss = criterion(outputs, labels)
        loss = (loss * weights).mean()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Collect predictions
        preds = (outputs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            weights = batch['weight'].to(device)
            
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
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return avg_loss, accuracy, precision, recall, f1, cm


def main():
    """Main training loop."""
    print("\n" + "="*70)
    print("V7P3R AI v6.1 - TAL-INSPIRED TRAINING")
    print("Stage 1: Position Evaluator")
    print("="*70)
    
    # Configuration
    config = {
        'epochs': 10,
        'batch_size': 256,
        'learning_rate': 0.001,
        'hidden_dims': [1024, 512, 256, 128],
        'dropout': 0.3,
        'train_val_split': 0.8,
        'random_seed': 42
    }
    
    print(f"\n📋 Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Set random seeds
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    # Load data
    print("\n🚀 Loading data for initial training run (starting with smaller dataset)...")
    features, labels, weights, sources = load_tal_inspired_data(
        batch_size=5000,  # Smaller batch size for faster loading
        max_batches=10   # 50k positions for initial 10-epoch test
    )
    
    # Normalize features
    print(f"\n📊 Normalizing features...")
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # Train/val split
    print(f"\n📊 Splitting into train/val ({config['train_val_split']*100:.0f}/{(1-config['train_val_split'])*100:.0f})...")
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        features, labels, weights,
        train_size=config['train_val_split'],
        random_state=config['random_seed'],
        stratify=labels
    )
    
    print(f"  Train: {len(X_train):,} positions")
    print(f"  Val:   {len(X_val):,} positions")
    
    # Create datasets
    train_dataset = PositionDataset(X_train, y_train, w_train)
    val_dataset = PositionDataset(X_val, y_val, w_val)
    
    train_loader = TorchDataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    val_loader = TorchDataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Initialize model
    input_dim = features.shape[1]
    print(f"\n🧠 Initializing Graph-Augmented Policy Network...")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dims: {config['hidden_dims']}")
    print(f"  Dropout: {config['dropout']}")
    
    model = GraphAugmentedPolicyNetwork(
        input_dim=input_dim,
        hidden_dims=config['hidden_dims'],
        dropout=config['dropout']
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = nn.BCELoss(reduction='none')  # Will apply weights manually
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    best_val_acc = 0
    best_epoch = 0
    training_history = []
    
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_prec, val_rec, val_f1, cm = validate(
            model, val_loader, criterion, device
        )
        
        epoch_time = time.time() - epoch_start
        
        # Print results
        print(f"\nEpoch {epoch+1}/{config['epochs']} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Prec: {train_prec:.4f} | Rec: {train_rec:.4f} | F1: {train_f1:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f}")
        print(f"  Confusion Matrix (Val):")
        print(f"    TN: {cm[0,0]:6,}  FP: {cm[0,1]:6,}")
        print(f"    FN: {cm[1,0]:6,}  TP: {cm[1,1]:6,}")
        
        # Save history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'train_prec': float(train_prec),
            'train_rec': float(train_rec),
            'train_f1': float(train_f1),
            'val_loss': float(val_loss),
            'val_acc': float(val_acc),
            'val_prec': float(val_prec),
            'val_rec': float(val_rec),
            'val_f1': float(val_f1),
            'time': epoch_time
        })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            
            # Save checkpoint
            checkpoint_path = Path("E:/Programming Stuff/Chess Engines/V7P3R Chess AI/v7p3r-chess-ai/v6.0/models/stage1")
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, checkpoint_path / f'v6.1_tal_inspired_epoch{epoch+1}.pth')
            
            print(f"  ✅ New best! Saved checkpoint (val_acc: {val_acc:.4f})")
    
    total_time = time.time() - start_time
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\n⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"🏆 Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    
    # Save training history
    history_path = checkpoint_path / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\n💾 Saved training history to: {history_path}")
    
    print("\n🎯 Next Steps:")
    print("  1. Review training curves (accuracy, loss)")
    print("  2. Test on Bxf7+ positions (qualitative validation)")
    print("  3. If accuracy ≥85%, proceed to Stage 2 (Move Selector)")
    print("  4. If accuracy <85%, train more epochs or tune hyperparameters")
    
    print("\n🚀 Your AI is learning to recognize tactical positions like Tal!")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
