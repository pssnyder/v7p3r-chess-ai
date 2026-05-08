#!/usr/bin/env python3
"""
V7P3R AI Reward System - Reinforcement Learning Training Module

Converts V7P3R evaluation features into training rewards for move ordering network.
This module implements imitation learning where the AI learns to mimic V7P3R's
evaluation brain by matching feature weights.

Training Paradigm:
1. Generate chess position
2. Extract V7P3R features (58-dimensional vector)
3. Get V7P3R's actual evaluation score
4. Train AI to predict score from features
5. Result: AI learns implicit feature weights of V7P3R

Author: Pat Snyder
Created: 2026-05-03 (V7P3R AI Reward System v1.0)
"""

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import subprocess
import json
import tempfile
import os

from v7p3r_ai_evaluator import V7P3RAIEvaluator, EvaluationFeatures


@dataclass
class TrainingReward:
    """
    Reward signal for reinforcement learning training.
    
    Combines V7P3R's actual evaluation with feature-based decomposition
    to provide rich training signal.
    """
    # V7P3R's actual evaluation score (ground truth)
    v7p3r_score: float  # Centipawns
    
    # 58-dimensional feature vector
    features: np.ndarray  # EvaluationFeatures.to_array()
    
    # Move-specific rewards
    move_quality: float  # How good is this move? [-1, 1]
    move_rank: int       # Rank among legal moves (1 = best)
    
    # Position context
    is_tactical: bool    # Tactical position flag
    is_endgame: bool     # Endgame position flag
    material_balance: float  # Material difference
    
    # Training metadata
    position_fen: str    # Position for debugging
    move_uci: str        # Move for debugging


class V7P3RRewardCalculator:
    """
    Converts V7P3R evaluations into training rewards for the AI.
    
    This class runs V7P3R engine to get ground truth evaluations,
    extracts features using V7P3RAIEvaluator, and creates training
    rewards that teach the AI to mimic V7P3R's decision-making.
    """
    
    def __init__(self, 
                 v7p3r_engine_path: str,
                 feature_evaluator: V7P3RAIEvaluator,
                 search_depth: int = 3):
        """
        Initialize reward calculator.
        
        Args:
            v7p3r_engine_path: Path to V7P3R UCI engine (v18.3 recommended)
            feature_evaluator: V7P3RAIEvaluator for feature extraction
            search_depth: Search depth for V7P3R evaluation (default 3)
        """
        self.v7p3r_path = v7p3r_engine_path
        self.evaluator = feature_evaluator
        self.search_depth = search_depth
        
        # Verify V7P3R engine exists
        if not os.path.exists(v7p3r_engine_path):
            raise FileNotFoundError(f"V7P3R engine not found: {v7p3r_engine_path}")
    
    def get_v7p3r_evaluation(self, board: chess.Board, depth: int = None) -> int:
        """
        Get V7P3R's evaluation of a position.
        
        Runs V7P3R engine in UCI mode to get actual evaluation score.
        This serves as ground truth for training the AI.
        
        Args:
            board: Chess position to evaluate
            depth: Search depth (uses self.search_depth if None)
            
        Returns:
            Evaluation score in centipawns (from White's perspective)
        """
        if depth is None:
            depth = self.search_depth
        
        try:
            # Run V7P3R engine via UCI
            process = subprocess.Popen(
                [self.v7p3r_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Send UCI commands
            commands = [
                "uci\n",
                f"position fen {board.fen()}\n",
                f"go depth {depth}\n",
                "quit\n"
            ]
            
            stdout, stderr = process.communicate(''.join(commands), timeout=10)
            
            # Parse evaluation from UCI output
            score = self._parse_uci_score(stdout)
            
            return score
            
        except subprocess.TimeoutExpired:
            process.kill()
            return 0  # Fallback
        except Exception as e:
            print(f"Error running V7P3R: {e}")
            return 0  # Fallback
    
    def _parse_uci_score(self, uci_output: str) -> int:
        """
        Parse evaluation score from UCI output.
        
        Args:
            uci_output: Raw UCI output from V7P3R
            
        Returns:
            Evaluation score in centipawns
        """
        # Look for "score cp" in UCI output
        for line in uci_output.split('\n'):
            if 'score cp' in line:
                # Example: "info depth 3 score cp 35 nodes 1234 ..."
                parts = line.split()
                try:
                    cp_idx = parts.index('cp')
                    score = int(parts[cp_idx + 1])
                    return score
                except (ValueError, IndexError):
                    continue
            elif 'score mate' in line:
                # Mate score - convert to large centipawn value
                parts = line.split()
                try:
                    mate_idx = parts.index('mate')
                    mate_in = int(parts[mate_idx + 1])
                    # Positive mate_in = we're mating, negative = we're getting mated
                    return 10000 if mate_in > 0 else -10000
                except (ValueError, IndexError):
                    continue
        
        return 0  # Default if no score found
    
    def calculate_move_rewards(self, board: chess.Board) -> List[TrainingReward]:
        """
        Calculate training rewards for all legal moves in a position.
        
        This is the main method for generating training data. For each legal move:
        1. Make move on board
        2. Get V7P3R's evaluation of resulting position
        3. Extract 58 features from resulting position
        4. Create training reward with all data
        
        Args:
            board: Chess position to analyze
            
        Returns:
            List of TrainingReward objects (one per legal move)
        """
        legal_moves = list(board.legal_moves)
        rewards = []
        
        # Evaluate all resulting positions
        move_evaluations = []
        for move in legal_moves:
            board.push(move)
            
            # Get V7P3R's evaluation (negate for opponent's perspective)
            v7p3r_score = -self.get_v7p3r_evaluation(board)
            
            # Extract features
            features = self.evaluator.extract_features(board)
            
            move_evaluations.append((move, v7p3r_score, features))
            board.pop()
        
        # Sort by evaluation (best moves first)
        move_evaluations.sort(key=lambda x: x[1], reverse=True)
        
        # Create rewards with move rankings
        for rank, (move, score, features) in enumerate(move_evaluations, start=1):
            # Calculate move quality [-1, 1] based on rank
            best_score = move_evaluations[0][1]
            worst_score = move_evaluations[-1][1]
            
            if best_score == worst_score:
                move_quality = 0.0
            else:
                # Normalize score to [-1, 1] range
                move_quality = (score - worst_score) / (best_score - worst_score)
                move_quality = 2.0 * move_quality - 1.0  # Scale to [-1, 1]
            
            # Detect position characteristics
            board.push(move)
            is_tactical = board.is_check() or len(list(board.legal_moves)) < 10
            is_endgame = self._is_endgame(board)
            material_balance = features.material_diff
            board.pop()
            
            reward = TrainingReward(
                v7p3r_score=score,
                features=features.to_array(),
                move_quality=move_quality,
                move_rank=rank,
                is_tactical=is_tactical,
                is_endgame=is_endgame,
                material_balance=material_balance,
                position_fen=board.fen(),
                move_uci=move.uci()
            )
            
            rewards.append(reward)
        
        return rewards
    
    def _is_endgame(self, board: chess.Board) -> bool:
        """Detect if position is endgame"""
        # Count non-pawn, non-king pieces
        pieces = 0
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            pieces += len(board.pieces(piece_type, chess.WHITE))
            pieces += len(board.pieces(piece_type, chess.BLACK))
        
        return pieces <= 6
    
    def create_training_dataset(self, 
                                positions: List[str],
                                output_path: str,
                                max_positions: int = 10000):
        """
        Create full training dataset from position FENs.
        
        This generates a dataset that the move ordering network can train on
        to learn V7P3R's evaluation patterns.
        
        Args:
            positions: List of FEN strings
            output_path: Path to save training dataset (JSON)
            max_positions: Maximum positions to process
        """
        dataset = {
            'metadata': {
                'v7p3r_version': 'v18.3',
                'feature_count': 58,
                'search_depth': self.search_depth,
                'positions_analyzed': min(len(positions), max_positions)
            },
            'training_data': []
        }
        
        positions_to_process = positions[:max_positions]
        
        print(f"Creating training dataset from {len(positions_to_process)} positions...")
        
        for i, fen in enumerate(positions_to_process):
            if i % 100 == 0:
                print(f"Processing position {i}/{len(positions_to_process)}...")
            
            try:
                board = chess.Board(fen)
                rewards = self.calculate_move_rewards(board)
                
                # Add all moves to training data
                for reward in rewards:
                    dataset['training_data'].append({
                        'position_fen': reward.position_fen,
                        'move_uci': reward.move_uci,
                        'v7p3r_score': reward.v7p3r_score,
                        'move_quality': reward.move_quality,
                        'move_rank': reward.move_rank,
                        'features': reward.features.tolist(),
                        'is_tactical': reward.is_tactical,
                        'is_endgame': reward.is_endgame,
                        'material_balance': reward.material_balance
                    })
            
            except Exception as e:
                print(f"Error processing position {fen}: {e}")
                continue
        
        # Save dataset
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"\nDataset saved to {output_path}")
        print(f"Total training samples: {len(dataset['training_data'])}")


class FeatureImitationLoss(nn.Module):
    """
    Custom loss function for imitation learning with V7P3R features.
    
    This loss combines:
    1. Evaluation error: (AI_eval - V7P3R_eval)²
    2. Move ranking error: How well does AI rank moves like V7P3R?
    3. Feature consistency: Do AI's implicit features match V7P3R's explicit features?
    """
    
    def __init__(self, 
                 eval_weight: float = 1.0,
                 ranking_weight: float = 0.5,
                 feature_weight: float = 0.3):
        """
        Initialize imitation loss.
        
        Args:
            eval_weight: Weight for evaluation error
            ranking_weight: Weight for move ranking error
            feature_weight: Weight for feature consistency
        """
        super().__init__()
        self.eval_weight = eval_weight
        self.ranking_weight = ranking_weight
        self.feature_weight = feature_weight
    
    def forward(self, 
                ai_predictions: torch.Tensor,      # [batch, 1] - AI's evaluation
                v7p3r_scores: torch.Tensor,        # [batch, 1] - V7P3R's evaluation
                ai_features: torch.Tensor,         # [batch, 58] - AI's learned features
                v7p3r_features: torch.Tensor,      # [batch, 58] - V7P3R's features
                move_qualities: torch.Tensor       # [batch, 1] - Move quality labels
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate imitation loss.
        
        Returns:
            Total loss and dictionary of component losses
        """
        # 1. Evaluation error (MSE between AI and V7P3R scores)
        eval_loss = F.mse_loss(ai_predictions, v7p3r_scores)
        
        # 2. Ranking error (how well does AI rank moves?)
        # Convert scores to rankings and compare
        ranking_loss = F.mse_loss(
            torch.sigmoid(ai_predictions),
            torch.sigmoid(v7p3r_scores)
        )
        
        # 3. Feature consistency (cosine similarity of feature vectors)
        feature_similarity = F.cosine_similarity(ai_features, v7p3r_features, dim=1)
        feature_loss = 1.0 - feature_similarity.mean()  # Minimize dissimilarity
        
        # Combine losses
        total_loss = (
            self.eval_weight * eval_loss +
            self.ranking_weight * ranking_loss +
            self.feature_weight * feature_loss
        )
        
        # Return loss and components for logging
        loss_components = {
            'total': total_loss.item(),
            'evaluation': eval_loss.item(),
            'ranking': ranking_loss.item(),
            'feature_consistency': feature_loss.item()
        }
        
        return total_loss, loss_components


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Example: Calculate rewards for a position
    
    # Initialize components
    evaluator = V7P3RAIEvaluator()
    
    # Path to V7P3R v18.3 engine (highest achiever)
    v7p3r_path = r"e:\Programming Stuff\Chess Engines\V7P3R Chess Engine\v7p3r-chess-engine\lichess\engines\V7P3R_v18.3_20251229\v7p3r_uci.py"
    
    # Check if engine exists
    if os.path.exists(v7p3r_path):
        reward_calc = V7P3RRewardCalculator(
            v7p3r_engine_path=v7p3r_path,
            feature_evaluator=evaluator,
            search_depth=3
        )
        
        # Test position (Sicilian Defense)
        board = chess.Board("r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
        
        print("Calculating move rewards for Sicilian Defense position...")
        rewards = reward_calc.calculate_move_rewards(board)
        
        print(f"\nTop 5 moves (V7P3R's perspective):")
        for i, reward in enumerate(rewards[:5], 1):
            print(f"{i}. {reward.move_uci}: {reward.v7p3r_score:+5d} cp (quality: {reward.move_quality:+.3f})")
        
        print(f"\nFeature vector size: {len(rewards[0].features)}")
        print(f"Material balance: {rewards[0].material_balance:.3f}")
        print(f"Is tactical: {rewards[0].is_tactical}")
        print(f"Is endgame: {rewards[0].is_endgame}")
    else:
        print(f"V7P3R engine not found at: {v7p3r_path}")
        print("Please update path to your V7P3R installation")
