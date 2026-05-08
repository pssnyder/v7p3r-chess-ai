"""
V7P3R AI v5.0 - PyTorch Dataset Classes
Handles loading and preprocessing of training data
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path


class V7P3RDataset(Dataset):
    """
    PyTorch dataset for V7P3R training data
    
    Loads preprocessed position data with features and Stockfish grades.
    Returns batches for dual-head training (policy + value).
    """
    
    def __init__(self, X, policy_targets, value_targets):
        """
        Initialize dataset
        
        Args:
            X: Input features (N, 26) - preprocessed and normalized
            policy_targets: Move quality grades (N,) - integers 0-5
            value_targets: Position evaluations (N,) - floats in [-1, 1]
        """
        self.X = torch.FloatTensor(X)
        self.policy = torch.LongTensor(policy_targets)
        self.value = torch.FloatTensor(value_targets).reshape(-1, 1)
        
        assert len(self.X) == len(self.policy) == len(self.value), \
            "Feature and target lengths must match"
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'features': self.X[idx],
            'policy_target': self.policy[idx],
            'value_target': self.value[idx]
        }


class V7P3RDatasetFromJSONL(Dataset):
    """
    PyTorch dataset that loads directly from JSONL files
    
    Performs on-the-fly preprocessing (slower but more memory efficient).
    Use V7P3RDataset with preprocessed arrays for faster training.
    """
    
    def __init__(self, jsonl_path, feature_config=None):
        """
        Initialize dataset from JSONL file
        
        Args:
            jsonl_path: Path to JSONL file with training data
            feature_config: Optional feature extraction configuration
        """
        self.jsonl_path = Path(jsonl_path)
        self.feature_config = feature_config or {}
        
        # Load all records into memory
        self.records = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.records.append(json.loads(line))
        
        print(f"Loaded {len(self.records)} positions from {jsonl_path}")
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        record = self.records[idx]
        
        # Extract features (implement preprocessing here)
        features = self._extract_features(record)
        
        # Extract targets
        stockfish = record['stockfish_analysis']
        policy_target = stockfish['move_quality_grade']
        value_target = np.clip(stockfish['best_move_eval'], -10000, 10000) / 10000.0
        
        return {
            'features': torch.FloatTensor(features),
            'policy_target': torch.LongTensor([policy_target])[0],
            'value_target': torch.FloatTensor([value_target])
        }
    
    def _extract_features(self, record):
        """Extract and preprocess features from record"""
        # Implement feature extraction logic here
        # For now, placeholder - should match preprocessing pipeline
        raise NotImplementedError("Use preprocessed arrays instead")


def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, 
                        batch_size=256, num_workers=4, pin_memory=True):
    """
    Create train/val/test dataloaders from preprocessed arrays
    
    Args:
        X_train, y_train: Training features and targets
        X_val, y_val: Validation features and targets
        X_test, y_test: Test features and targets
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = V7P3RDataset(
        X_train, 
        y_train['policy'], 
        y_train['value']
    )
    
    val_dataset = V7P3RDataset(
        X_val, 
        y_val['policy'], 
        y_val['value']
    )
    
    test_dataset = V7P3RDataset(
        X_test, 
        y_test['policy'], 
        y_test['value']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"Created dataloaders:")
    print(f"  Train: {len(train_dataset):,} positions, {len(train_loader):,} batches")
    print(f"  Val:   {len(val_dataset):,} positions, {len(val_loader):,} batches")
    print(f"  Test:  {len(test_dataset):,} positions, {len(test_loader):,} batches")
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test dataset creation
    print("=" * 80)
    print("V7P3R AI v5.0 - Dataset Test")
    print("=" * 80)
    
    # Create dummy data
    n_samples = 1000
    X = np.random.randn(n_samples, 26).astype(np.float32)
    policy_targets = np.random.randint(0, 6, n_samples)
    value_targets = np.random.uniform(-1, 1, n_samples).astype(np.float32)
    
    # Create dataset
    dataset = V7P3RDataset(X, policy_targets, value_targets)
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test dataloader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"Batches: {len(loader)}")
    
    # Get one batch
    batch = next(iter(loader))
    
    print(f"\nBatch shapes:")
    print(f"  Features: {batch['features'].shape}")
    print(f"  Policy targets: {batch['policy_target'].shape}")
    print(f"  Value targets: {batch['value_target'].shape}")
    
    print("\n" + "=" * 80)
    print("✅ Dataset test complete!")
    print("=" * 80)
