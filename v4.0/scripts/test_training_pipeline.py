"""Test complete training pipeline with model forward pass."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.corrective_dataset import CorrectiveDataset, custom_collate_fn
from src.models.move_ordering_network import MoveOrderingNetwork
from torch.utils.data import DataLoader
import torch

print("🧪 Testing Complete Training Pipeline\n")

# Load dataset
print("📂 Loading dataset...")
dataset = CorrectiveDataset('data/stage2_training/corrective_dataset.json')
print(f"   ✅ {len(dataset)} examples loaded\n")

# Create data loader
print("📊 Creating data loader...")
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=custom_collate_fn,
    num_workers=0
)
batch = next(iter(loader))
print(f"   ✅ Batch created: {batch['position_features'].shape[0]} examples\n")

# Create model
print("🤖 Creating model...")
model = MoveOrderingNetwork(num_themes=57)
total_params = sum(p.numel() for p in model.parameters())
print(f"   ✅ Model created: {total_params:,} parameters\n")

# Test forward pass
print("⚡ Testing forward pass...")
output = model(batch)
print(f"   ✅ Forward pass successful!")
print(f"   Move scores shape: {output['move_scores'].shape}")
print(f"   Theme probs shape: {output['theme_probs'].shape}\n")

# Test loss computation
print("📉 Testing loss computation...")
from torch import nn
mse_loss = nn.MSELoss(reduction='none')
move_scores = output['move_scores']
target_scores = batch['move_scores']
importance_weights = batch['move_weights']
mask = batch['move_masks']

loss_per_move = mse_loss(move_scores, target_scores)
weighted_loss = loss_per_move * importance_weights
masked_loss = weighted_loss * mask.float()
total_loss = masked_loss.sum() / mask.float().sum()
print(f"   ✅ Loss computed: {total_loss.item():.4f}\n")

print("✅ All pipeline tests passed!")
print("   Ready for training!")
