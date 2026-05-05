"""Quick test of corrective dataset loading."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.corrective_dataset import CorrectiveDataset, custom_collate_fn
from torch.utils.data import DataLoader

print("🧪 Testing Corrective Dataset Loading\n")

# Load dataset
print("Loading dataset...")
dataset = CorrectiveDataset('data/stage2_training/corrective_dataset.json')

print(f"\n✅ Dataset loaded successfully!")
print(f"   Total examples: {len(dataset)}")

# Test single example
print("\n📦 Testing single example retrieval...")
example = dataset[0]
print(f"   Position features: {example['position_features'].shape}")
print(f"   Moves: {example['moves'].shape}")
print(f"   Move weights: {example['move_weights'].shape}")
print(f"   Move scores: {example['move_scores'].shape}")
print(f"   Example type: {example['example_type']}")
print(f"   Classification: {example['move_classification']}")

# Test data loader with collate
print("\n📊 Testing DataLoader with batch collation...")
loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=custom_collate_fn,
    num_workers=0
)

batch = next(iter(loader))
print(f"   Batch position features: {batch['position_features'].shape}")
print(f"   Batch moves (padded): {batch['moves'].shape}")
print(f"   Batch move masks: {batch['move_masks'].shape}")
print(f"   Example types in batch: {len(batch['example_types'])}")

print("\n✅ All tests passed! Dataset ready for training.")
