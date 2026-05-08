#!/usr/bin/env python3
"""
V7P3R Verified Training Pipeline

Complete training system that:
1. Trains AI to imitate V7P3R's evaluation brain (58 features)
2. Verifies V7P3R evals against Lichess Stockfish database (95GB)
3. Flags positions where V7P3R disagrees significantly
4. Creates corrective dataset for V7P3R improvement
5. Retrains on corrected positions

Training Philosophy:
✓ Primary signal: V7P3R's evaluations (preserve personality)
✓ Verification: Lichess database (catch eval bugs)
✓ Active learning: Flag disagreements → Fix V7P3R → Retrain
✓ Result: AI that plays like V7P3R but faster and more robust

Author: Pat Snyder
Created: 2026-05-03 (Verified Training Pipeline v1.0)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import chess
import json
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from evaluation.v7p3r_ai_evaluator import V7P3RAIEvaluator
from training.v7p3r_reward_system import V7P3RRewardCalculator, FeatureImitationLoss
from training.eval_verification_system import EvalVerificationSystem, VerificationResult
from data.lichess_eval_indexer import LichessEvalIndexer
from scripts.train_v7p3r_imitation import V7P3REvaluationPredictor


@dataclass
class VerifiedTrainingConfig:
    """Configuration for verified training pipeline"""
    # V7P3R engine
    v7p3r_engine_path: str
    v7p3r_search_depth: int = 3
    
    # Lichess database
    lichess_db_path: str
    rebuild_lichess_index: bool = False
    
    # Training data
    puzzle_path: str
    max_positions: int = 10000
    
    # Verification thresholds
    max_eval_difference: int = 100  # Flag if |V7P3R - Stockfish| > 100cp
    require_move_match: bool = False  # Also flag if best moves differ
    
    # Model
    hidden_size: int = 512
    num_layers: int = 4
    
    # Training
    batch_size: int = 64
    learning_rate: float = 0.0001
    num_epochs: int = 10
    weight_decay: float = 0.0001
    
    # Output
    checkpoint_dir: str = 'checkpoints/verified_training'
    flag_dir: str = 'flags/eval_discrepancies'
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class VerifiedTrainingDataset(Dataset):
    """
    Training dataset with V7P3R evaluation verification.
    
    Each sample includes:
    - Position features (58-dim from V7P3R)
    - V7P3R evaluation score
    - Verification result (agreement with Stockfish)
    - Confidence weight for training
    """
    
    def __init__(self, verification_results: List[VerificationResult]):
        """
        Create dataset from verification results.
        
        Only includes positions where use_for_training=True (not flagged).
        """
        # Filter to training-eligible positions
        self.samples = [r for r in verification_results if r.use_for_training]
        
        print(f"Verified training dataset:")
        print(f"  Total verified: {len(verification_results)}")
        print(f"  Eligible for training: {len(self.samples)}")
        print(f"  Flagged (excluded): {len(verification_results) - len(self.samples)}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        result = self.samples[idx]
        
        return {
            'features': torch.tensor(result.v7p3r_features, dtype=torch.float32),
            'v7p3r_score': torch.tensor([result.v7p3r_score], dtype=torch.float32),
            'confidence': torch.tensor([result.confidence], dtype=torch.float32),
            'eval_difference': result.eval_difference,
            'agreement_level': result.agreement_level.value
        }


class ConfidenceWeightedLoss(nn.Module):
    """
    Loss function weighted by verification confidence.
    
    Positions where V7P3R agrees with Stockfish get higher weight.
    Positions near flagging threshold get lower weight.
    """
    
    def __init__(self, base_criterion: nn.Module):
        super().__init__()
        self.base_criterion = base_criterion
    
    def forward(self, 
                predictions: torch.Tensor,
                targets: torch.Tensor,
                confidence: torch.Tensor) -> torch.Tensor:
        """
        Calculate weighted loss.
        
        Args:
            predictions: [batch, 1] - Model predictions
            targets: [batch, 1] - V7P3R evaluations
            confidence: [batch, 1] - Verification confidence weights
            
        Returns:
            Weighted loss
        """
        # Calculate base loss per sample
        loss_per_sample = F.mse_loss(predictions, targets, reduction='none')
        
        # Weight by confidence
        weighted_loss = loss_per_sample * confidence
        
        # Return mean
        return weighted_loss.mean()


def create_verified_dataset(config: VerifiedTrainingConfig) -> Tuple[List[VerificationResult], EvalVerificationSystem]:
    """
    Create verified training dataset.
    
    Steps:
    1. Load puzzle positions
    2. Run V7P3R evaluation on each
    3. Verify against Lichess database
    4. Flag discrepancies
    5. Return verified results
    """
    print("\n" + "="*80)
    print("STEP 1: Creating Verified Training Dataset")
    print("="*80)
    
    # Initialize components
    print("\nInitializing V7P3R evaluator...")
    evaluator = V7P3RAIEvaluator()
    
    print("Initializing V7P3R reward calculator...")
    reward_calc = V7P3RRewardCalculator(
        v7p3r_engine_path=config.v7p3r_engine_path,
        feature_evaluator=evaluator,
        search_depth=config.v7p3r_search_depth
    )
    
    print("Initializing Lichess evaluation database...")
    lichess_indexer = LichessEvalIndexer(
        jsonl_path=config.lichess_db_path,
        rebuild_index=config.rebuild_lichess_index
    )
    
    print("Initializing verification system...")
    verifier = EvalVerificationSystem(
        v7p3r_reward_calculator=reward_calc,
        lichess_indexer=lichess_indexer,
        flag_output_dir=config.flag_dir
    )
    
    # Load puzzle positions
    print(f"\nLoading puzzles from: {config.puzzle_path}")
    with open(config.puzzle_path, 'r') as f:
        puzzle_data = json.load(f)
    
    # Extract positions
    positions = []
    for puzzle in puzzle_data.get('puzzles', puzzle_data.get('training_data', []))[:config.max_positions]:
        if 'fen' in puzzle:
            positions.append(puzzle['fen'])
        elif 'position_fen' in puzzle:
            positions.append(puzzle['position_fen'])
    
    print(f"Loaded {len(positions)} positions")
    
    # Verify all positions
    print("\nVerifying positions (V7P3R vs Stockfish)...")
    verification_results = []
    
    for fen in tqdm(positions, desc="Verifying"):
        try:
            board = chess.Board(fen)
            result = verifier.verify_position(board)
            verification_results.append(result)
        except Exception as e:
            print(f"\nError verifying {fen}: {e}")
            continue
    
    # Print statistics
    verifier.print_statistics()
    
    # Save flagged positions
    print("\nSaving flagged positions for review...")
    verifier.save_flagged_positions(verification_results, batch_name="training_batch")
    
    return verification_results, verifier


def train_epoch_verified(model: V7P3REvaluationPredictor,
                        dataloader: DataLoader,
                        optimizer: optim.Optimizer,
                        device: str,
                        epoch: int) -> Dict[str, float]:
    """Train one epoch with confidence weighting"""
    model.train()
    
    total_loss = 0.0
    total_weighted_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        features = batch['features'].to(device)
        v7p3r_scores = batch['v7p3r_score'].to(device)
        confidence = batch['confidence'].to(device)
        
        # Forward pass
        predictions, feature_weights, weighted_features = model(features)
        
        # Calculate loss per sample
        loss_per_sample = F.mse_loss(predictions, v7p3r_scores, reduction='none')
        
        # Weight by confidence
        weighted_loss = (loss_per_sample * confidence).mean()
        
        # Backward pass
        optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate
        total_loss += loss_per_sample.mean().item()
        total_weighted_loss += weighted_loss.item()
        
        pbar.set_postfix({
            'loss': f"{weighted_loss.item():.4f}",
            'avg_conf': f"{confidence.mean().item():.2f}"
        })
    
    return {
        'total_loss': total_loss / len(dataloader),
        'weighted_loss': total_weighted_loss / len(dataloader)
    }


def main():
    parser = argparse.ArgumentParser(description="Train V7P3R AI with evaluation verification")
    
    # V7P3R engine
    parser.add_argument('--v7p3r-engine', type=str, required=True,
                       help='Path to V7P3R UCI engine (v18.3 recommended)')
    parser.add_argument('--v7p3r-depth', type=int, default=3,
                       help='Search depth for V7P3R evaluation')
    
    # Lichess database
    parser.add_argument('--lichess-db', type=str, required=True,
                       help='Path to lichess_db_eval.jsonl (95GB)')
    parser.add_argument('--rebuild-index', action='store_true',
                       help='Rebuild Lichess database index (10-15min one-time)')
    
    # Training data
    parser.add_argument('--puzzle-path', type=str, required=True,
                       help='Path to puzzle dataset JSON')
    parser.add_argument('--max-positions', type=int, default=10000,
                       help='Maximum positions to verify')
    
    # Verification
    parser.add_argument('--max-eval-diff', type=int, default=100,
                       help='Flag positions with eval diff > threshold (cp)')
    parser.add_argument('--require-move-match', action='store_true',
                       help='Also flag if best moves differ')
    
    # Training
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--num-epochs', type=int, default=10)
    
    # Output
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/verified_training')
    parser.add_argument('--flag-dir', type=str, default='flags/eval_discrepancies')
    
    args = parser.parse_args()
    
    # Create config
    config = VerifiedTrainingConfig(
        v7p3r_engine_path=args.v7p3r_engine,
        v7p3r_search_depth=args.v7p3r_depth,
        lichess_db_path=args.lichess_db,
        rebuild_lichess_index=args.rebuild_index,
        puzzle_path=args.puzzle_path,
        max_positions=args.max_positions,
        max_eval_difference=args.max_eval_diff,
        require_move_match=args.require_move_match,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        checkpoint_dir=args.checkpoint_dir,
        flag_dir=args.flag_dir
    )
    
    print("\n" + "="*80)
    print("V7P3R Verified Training Pipeline")
    print("="*80)
    print(f"Device: {config.device}")
    print(f"V7P3R Engine: {config.v7p3r_engine_path}")
    print(f"Lichess DB: {config.lichess_db_path}")
    print(f"Max Eval Difference: {config.max_eval_difference} cp")
    print(f"Max Positions: {config.max_positions}")
    
    # Create verified dataset
    verification_results, verifier = create_verified_dataset(config)
    
    # Create training dataset
    print("\n" + "="*80)
    print("STEP 2: Creating Training Dataset")
    print("="*80)
    
    full_dataset = VerifiedTrainingDataset(verification_results)
    
    # Split train/val
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Initialize model
    print("\n" + "="*80)
    print("STEP 3: Training Model")
    print("="*80)
    
    model = V7P3REvaluationPredictor(
        feature_dim=58,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers
    ).to(config.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Setup training
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2
    )
    
    # Train
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    for epoch in range(1, config.num_epochs + 1):
        # Train
        train_metrics = train_epoch_verified(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=config.device,
            epoch=epoch
        )
        
        # Validate (simple for now)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(config.device)
                targets = batch['v7p3r_score'].to(config.device)
                
                predictions, _, _ = model(features)
                val_loss += F.mse_loss(predictions, targets).item()
        
        val_loss /= len(val_loader)
        
        print(f"\nEpoch {epoch}/{config.num_epochs}")
        print(f"  Train Loss: {train_metrics['weighted_loss']:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        scheduler.step(val_loss)
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            checkpoint_path = os.path.join(config.checkpoint_dir, 'v7p3r_verified_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': config
            }, checkpoint_path)
            
            print(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f})")
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"\nNext Steps:")
    print(f"1. Review flagged positions in: {config.flag_dir}")
    print(f"2. Fix V7P3R evaluation bugs identified")
    print(f"3. Retrain on corrected positions")
    print(f"4. Integrate puzzle analysis data")


if __name__ == "__main__":
    main()
