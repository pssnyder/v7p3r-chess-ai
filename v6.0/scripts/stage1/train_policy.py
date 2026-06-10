"""
V7P3R AI v6.0 - Stage 1 Training Script

Graph-Augmented Neural Network for Binary Move Classification (Good vs Bad)

Architecture:
- Input: 325D position features + transposition graph structure
- Transposition Attention: Attend to K=10 similar positions
- Hidden Layers: [1024, 512, 256, 128] with dropout & batch norm
- Output: Binary classification (sigmoid)

Loss:
- Weighted BCE (handles 82:1 imbalance)
- Graph Regularization (enforce prediction consistency on neighbors)
"""

import json
import sys
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import datetime
import time


class DataLoader:
    """Load and preprocess filtered dataset + transposition graph."""
    
    def __init__(self, good_path: str, bad_path: str, graph_path: str, config: dict):
        self.good_path = Path(good_path)
        self.bad_path = Path(bad_path)
        self.graph_path = Path(graph_path)
        self.config = config
        
        self.scaler = StandardScaler()
        self.graph = None
        self.zobrist_to_idx = {}  # Map Zobrist hash → dataset index
        self.feature_names = None  # Consistent feature ordering
        
    def load(self, max_samples: int = None):
        """Load and preprocess complete dataset."""
        print("=" * 70)
        print("LOADING DATASET - V7P3R AI v6.0")
        print("=" * 70)
        
        # Load transposition graph
        print("\n📊 Loading transposition graph...")
        with open(self.graph_path, 'rb') as f:
            self.graph = pickle.load(f)
        print(f"✅ Loaded graph: {len(self.graph):,} nodes")
        
        # Discover feature names from first few records
        if self.feature_names is None:
            print("\n📊 Discovering feature schema...")
            self.feature_names = self._discover_features([self.good_path, self.bad_path], sample_size=1000)
            print(f"✅ Found {len(self.feature_names)} numeric features")
        
        # Load positions
        print("\n📊 Loading positions...")
        good_data = self._load_positions(self.good_path, label=1, max_samples=max_samples)
        bad_data = self._load_positions(self.bad_path, label=0, max_samples=max_samples)
        
        print(f"✅ Loaded {len(good_data['labels']):,} good + {len(bad_data['labels']):,} bad positions")
        
        # Combine datasets
        all_features = np.vstack([good_data['features'], bad_data['features']])
        all_labels = np.concatenate([good_data['labels'], bad_data['labels']])
        all_hashes = good_data['zobrist_hashes'] + bad_data['zobrist_hashes']
        
        print(f"\n📊 Total dataset: {len(all_labels):,} positions")
        print(f"   Good: {np.sum(all_labels):,} ({np.mean(all_labels)*100:.1f}%)")
        print(f"   Bad:  {len(all_labels) - np.sum(all_labels):,} ({(1-np.mean(all_labels))*100:.1f}%)")
        
        # Build Zobrist hash → index mapping
        print("\n📊 Building position index...")
        for idx, zobrist_hash in enumerate(all_hashes):
            self.zobrist_to_idx[zobrist_hash] = idx
        
        # Normalize features
        print("\n📊 Normalizing features...")
        all_features = self.scaler.fit_transform(all_features)
        
        # Train/val/test split
        print("\n📊 Splitting dataset (80/10/10)...")
        X_temp, X_test, y_temp, y_test, hash_temp, hash_test = train_test_split(
            all_features, all_labels, all_hashes,
            test_size=0.1, random_state=42, stratify=all_labels
        )
        
        X_train, X_val, y_train, y_val, hash_train, hash_val = train_test_split(
            X_temp, y_temp, hash_temp,
            test_size=0.111,  # 0.111 of 90% = 10% of total
            random_state=42, stratify=y_temp
        )
        
        print(f"✅ Train: {len(y_train):,} positions")
        print(f"✅ Val:   {len(y_val):,} positions")
        print(f"✅ Test:  {len(y_test):,} positions")
        
        return {
            'train': (X_train, y_train, hash_train),
            'val': (X_val, y_val, hash_val),
            'test': (X_test, y_test, hash_test),
            'feature_dim': all_features.shape[1],
        }
    
    def _discover_features(self, filepaths: list, sample_size: int = 1000):
        """Discover all numeric feature names from sample of data."""
        all_feature_names = set()
        
        for filepath in filepaths:
            count = 0
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if count >= sample_size:
                        break
                    
                    record = json.loads(line)
                    features = record.get('features', {})
                    
                    # Add numeric feature names
                    for name, val in features.items():
                        # Skip string features
                        if isinstance(val, str):
                            continue
                        
                        # Skip non-numeric
                        if isinstance(val, (bool, int, float)):
                            all_feature_names.add(name)
                    
                    count += 1
        
        # Return sorted list for consistent ordering
        return sorted(all_feature_names)
    
    def _load_positions(self, filepath: Path, label: int, max_samples: int = None):
        """Load positions from JSONL file."""
        features_list = []
        labels_list = []
        zobrist_hashes = []
        
        count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if max_samples and count >= max_samples:
                    break
                
                record = json.loads(line)
                features = record.get('features', {})
                zobrist_hash = record.get('zobrist_hash', None)
                
                # Extract feature vector (assume ordered)
                feature_vector = self._features_to_vector(features)
                
                features_list.append(feature_vector)
                labels_list.append(label)
                zobrist_hashes.append(zobrist_hash)
                
                count += 1
                
                if count % 500000 == 0:
                    print(f"  Loaded: {count:,} positions...")
        
        print(f"✅ Loaded {count:,} positions from {filepath.name}")
        
        return {
            'features': np.array(features_list, dtype=np.float32),
            'labels': np.array(labels_list, dtype=np.float32),
            'zobrist_hashes': zobrist_hashes,
        }
    
    def _features_to_vector(self, features: dict) -> np.ndarray:
        """Convert feature dict to fixed-size vector using consistent feature names."""
        if self.feature_names is None:
            raise ValueError("feature_names not initialized. Call _discover_features() first.")
        
        vector = []
        for name in self.feature_names:
            val = features.get(name, 0)  # Default to 0 if feature missing
            
            # Convert boolean to int
            if isinstance(val, bool):
                val = int(val)
            
            # Convert to float
            try:
                vector.append(float(val))
            except (ValueError, TypeError):
                vector.append(0.0)  # Use 0 for non-convertible values
        
        return np.array(vector, dtype=np.float32)
    
    def get_neighbor_indices(self, zobrist_hashes: list) -> list:
        """
        Get graph neighbor indices for each position.
        
        Returns:
            List of neighbor index lists (or None if no neighbors)
        """
        neighbor_indices = []
        
        for zobrist_hash in zobrist_hashes:
            if zobrist_hash in self.graph:
                # Get neighbor zobrist hashes from graph
                neighbor_hashes = list(self.graph[zobrist_hash]['neighbors'])
                
                # Map to dataset indices
                neighbor_idxs = []
                for neighbor_hash in neighbor_hashes[:10]:  # K=10 neighbors
                    if neighbor_hash in self.zobrist_to_idx:
                        neighbor_idxs.append(self.zobrist_to_idx[neighbor_hash])
                
                neighbor_indices.append(neighbor_idxs if neighbor_idxs else None)
            else:
                neighbor_indices.append(None)
        
        return neighbor_indices


class ChessDataset(Dataset):
    """PyTorch Dataset for chess positions."""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels).unsqueeze(1)  # (N,) -> (N, 1)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class GraphAugmentedPolicyNetwork(nn.Module):
    """
    Graph-Augmented Neural Network for binary move classification.
    
    Architecture:
    - Position Embedding: 325 → 512
    - [If graph neighbors] Transposition Attention over K neighbors
    - Hidden Layers: [512/1024, 512, 256, 128] with dropout + batch norm
    - Output: Binary classification (sigmoid)
    """
    
    def __init__(self, input_dim: int, config: dict):
        super().__init__()
        
        self.input_dim = input_dim
        self.config = config
        self.use_graph = config.get('use_graph_attention', True)
        
        # Position embedding layer
        self.position_embedding = nn.Linear(input_dim, 512)
        self.position_bn = nn.BatchNorm1d(512)
        self.position_relu = nn.ReLU()
        
        # Transposition attention (if graph neighbors available)
        if self.use_graph:
            self.attention_query = nn.Linear(512, 512)
            self.attention_key = nn.Linear(512, 512)
            self.attention_value = nn.Linear(512, 512)
        
        # Hidden layers
        self.hidden1 = nn.Linear(512, 512)
        self.hidden1_bn = nn.BatchNorm1d(512)
        self.hidden1_dropout = nn.Dropout(0.3)
        self.hidden1_relu = nn.ReLU()
        
        self.hidden2 = nn.Linear(512, 256)
        self.hidden2_bn = nn.BatchNorm1d(256)
        self.hidden2_dropout = nn.Dropout(0.3)
        self.hidden2_relu = nn.ReLU()
        
        self.hidden3 = nn.Linear(256, 128)
        self.hidden3_dropout = nn.Dropout(0.3)
        self.hidden3_relu = nn.ReLU()
        
        # Output layer (no sigmoid here - using BCEWithLogitsLoss)
        self.output_layer = nn.Linear(128, 1)
    
    def forward(self, features, neighbor_features=None):
        """Forward pass."""
        # Position embedding
        x = self.position_embedding(features)
        x = self.position_bn(x)
        x = self.position_relu(x)
        
        # Transposition attention (if neighbors available)
        if self.use_graph and neighbor_features is not None:
            # Compute attention over neighbors
            query = self.attention_query(x)  # (batch, 512)
            
            # neighbor_features shape: (batch, K, 512) or None
            # For simplicity, mean pooling over neighbors
            attended = torch.mean(neighbor_features, dim=1)  # (batch, 512)
            
            # Concatenate position + attended neighbors
            x = torch.cat([x, attended], dim=1)  # (batch, 1024)
        
        # Hidden layers
        x = self.hidden1(x)
        x = self.hidden1_bn(x)
        x = self.hidden1_relu(x)
        x = self.hidden1_dropout(x)
        
        x = self.hidden2(x)
        x = self.hidden2_bn(x)
        x = self.hidden2_relu(x)
        x = self.hidden2_dropout(x)
        
        x = self.hidden3(x)
        x = self.hidden3_relu(x)
        x = self.hidden3_dropout(x)
        
        # Output (logits - no sigmoid, using BCEWithLogitsLoss)
        output = self.output_layer(x)
        
        return output


class Trainer:
    """Training orchestrator."""
    
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }
        
    def train(self, data: dict, data_loader: DataLoader):
        """Execute full training pipeline."""
        print("\n" + "=" * 70)
        print("TRAINING - V7P3R AI v6.0")
        print("=" * 70)
        print(f"Device: {self.device}")
        
        X_train, y_train, hash_train = data['train']
        X_val, y_val, hash_val = data['val']
        
        # Calculate class weights
        pos_count = np.sum(y_train)
        neg_count = len(y_train) - pos_count
        pos_weight = len(y_train) / (2 * pos_count) if pos_count > 0 else 1.0
        neg_weight = len(y_train) / (2 * neg_count) if neg_count > 0 else 1.0
        
        print(f"\n📊 Class weights:")
        print(f"   Good (1): {pos_weight:.4f}")
        print(f"   Bad (0):  {neg_weight:.4f}")
        
        # Create datasets and loaders
        train_dataset = ChessDataset(X_train, y_train)
        val_dataset = ChessDataset(X_val, y_val)
        
        train_loader = TorchDataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 2048),
            shuffle=True,
            num_workers=0  # Windows compatibility
        )
        
        val_loader = TorchDataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 2048),
            shuffle=False,
            num_workers=0
        )
        
        # Build model
        print("\n📊 Building model...")
        self.model = GraphAugmentedPolicyNetwork(
            input_dim=data['feature_dim'],
            config=self.config
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.001)
        )
        
        # Loss function with class weights
        # Note: BCEWithLogitsLoss expects pos_weight for positive class only
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight / neg_weight]).to(self.device)
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        print("✅ Model compiled")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config.get('patience', 10)
        
        print(f"\n📊 Training for up to {self.config.get('epochs', 100)} epochs...")
        print(f"   Early stopping patience: {patience}")
        
        for epoch in range(self.config.get('epochs', 100)):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self._train_epoch(train_loader, criterion)
            
            # Validate
            val_loss, val_acc = self._validate_epoch(val_loader, criterion)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            epoch_time = time.time() - epoch_start
            
            print(f"Epoch {epoch+1}/{self.config.get('epochs', 100)} "
                  f"- {epoch_time:.1f}s - "
                  f"loss: {train_loss:.4f} - "
                  f"acc: {train_acc:.4f} - "
                  f"val_loss: {val_loss:.4f} - "
                  f"val_acc: {val_acc:.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                output_dir = Path(self.config['output_dir'])
                output_dir.mkdir(parents=True, exist_ok=True)
                best_model_path = output_dir / 'stage1_policy_best.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': self.config,
                }, best_model_path)
                print(f"   ✅ Saved best model (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"\n⏹️  Early stopping triggered (patience={patience})")
                    print(f"   Best val_loss: {best_val_loss:.4f}")
                    break
        
        # Load best model
        print(f"\n✅ Training complete!")
        print(f"   Best val_loss: {best_val_loss:.4f}")
        print(f"   Loading best checkpoint...")
        
        best_checkpoint = torch.load(best_model_path)
        self.model.load_state_dict(best_checkpoint['model_state_dict'])
        
        return self.history
    
    def _train_epoch(self, train_loader, criterion):
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for features, labels in train_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item() * features.size(0)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            total_correct += (predictions == labels).sum().item()
            total_samples += features.size(0)
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def _validate_epoch(self, val_loader, criterion):
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item() * features.size(0)
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                total_correct += (predictions == labels).sum().item()
                total_samples += features.size(0)
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc


def main():
    """Main training pipeline."""
    # Configuration
    config = {
        'epochs': 100,
        'batch_size': 2048,
        'learning_rate': 0.001,
        'patience': 10,
        'use_graph_attention': False,  # Start simple, add later
        'output_dir': 'models/stage1',
    }
    
    # Paths
    base_path = Path(__file__).parent.parent.parent
    good_path = base_path / "data" / "stage1" / "good_positions.jsonl"
    bad_path = base_path / "data" / "stage1" / "bad_positions.jsonl"
    graph_path = base_path / "data" / "stage1" / "transposition_graph.pkl"
    
    # Create output directory
    output_dir = base_path / config['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    loader = DataLoader(str(good_path), str(bad_path), str(graph_path), config)
    data = loader.load(max_samples=None)  # Use full dataset
    
    # Train
    trainer = Trainer(config)
    history = trainer.train(data, loader)
    
    # Save final model
    final_model_path = output_dir / 'stage1_policy_final.pt'
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'config': config,
        'history': history,
        'feature_dim': data['feature_dim'],
    }, final_model_path)
    print(f"\n✅ Model saved to: {final_model_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
