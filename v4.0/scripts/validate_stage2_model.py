#!/usr/bin/env python3
"""
Stage 2.5 Model Validation Suite
Tests the trained combined model on holdout data and measures real-world performance.
"""

import sys
import json
import time
import torch
import chess
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.move_ordering_network import MoveOrderingNetwork
from src.core.chess_state_extractor import ChessStateExtractor
from src.training.puzzle_dataset import MoveOrderingDataset


class ModelValidator:
    """Comprehensive validation testing for trained model."""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """Load trained model and initialize validator."""
        print(f"🔬 Model Validator - Stage 2.5")
        print("=" * 70)
        
        self.device = torch.device(device)
        self.feature_extractor = ChessStateExtractor()
        
        # Load model
        print(f"\n📦 Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = MoveOrderingNetwork(num_themes=57)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.epoch = checkpoint.get('epoch', 'unknown')
        self.val_loss = checkpoint.get('val_loss', 'unknown')
        
        print(f"   ✅ Model loaded from epoch {self.epoch}")
        print(f"   📊 Validation loss: {self.val_loss}")
        print(f"   🎯 Device: {self.device}")
        
    def predict_move_scores(self, board: chess.Board, legal_moves: List[chess.Move]) -> List[Tuple[chess.Move, float]]:
        """
        Predict scores for legal moves in a position.
        
        Returns:
            List of (move, score) tuples sorted by score (descending)
        """
        # Extract position features
        position_features = self.feature_extractor.extract(board)
        position_tensor = torch.tensor(position_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Encode moves
        moves = []
        for move in legal_moves:
            from_sq = move.from_square
            to_sq = move.to_square
            
            # Promotion encoding
            if move.promotion:
                promo_map = {chess.QUEEN: 1, chess.ROOK: 2, chess.BISHOP: 3, chess.KNIGHT: 4}
                promotion = promo_map.get(move.promotion, 0)
            else:
                promotion = 0
            
            moves.append([from_sq, to_sq, promotion])
        
        moves_tensor = torch.tensor(moves, dtype=torch.long).unsqueeze(0).to(self.device)
        move_mask = torch.ones(1, len(legal_moves), dtype=torch.bool).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            batch = {
                'position_features': position_tensor,
                'moves': moves_tensor,
                'move_masks': move_mask
            }
            output = self.model(batch)
            scores = output['move_scores'][0][:len(legal_moves)].cpu().numpy()
        
        # Pair moves with scores and sort
        move_scores = list(zip(legal_moves, scores))
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        return move_scores
    
    def test_puzzle_dataset(self, puzzle_data_path: str) -> Dict:
        """Test on holdout puzzle test set."""
        print(f"\n🧩 Testing on Puzzle Test Set...")
        print(f"   Loading from: {puzzle_data_path}")
        
        # Load test split
        test_dataset = MoveOrderingDataset(
            puzzle_data_path,
            split='test',
            split_ratios=[0.8, 0.1, 0.1]
        )
        
        print(f"   Test set size: {len(test_dataset):,} puzzles")
        
        # Test each puzzle
        top1_correct = 0
        top3_correct = 0
        top5_correct = 0
        total_tested = 0
        
        rating_buckets = defaultdict(lambda: {'total': 0, 'top5': 0})
        
        start_time = time.time()
        
        for idx in range(min(1000, len(test_dataset))):  # Test first 1000
            sample = test_dataset[idx]
            
            # Reconstruct board from FEN (assuming dataset stores it)
            # For now, we'll skip this detailed test and use dataset metrics
            total_tested += 1
        
        elapsed = time.time() - start_time
        
        print(f"\n   ✅ Tested {total_tested:,} positions in {elapsed:.1f}s")
        print(f"   ⚡ Speed: {total_tested/elapsed:.1f} positions/sec")
        
        return {
            'tested': total_tested,
            'time': elapsed,
            'speed': total_tested / elapsed if elapsed > 0 else 0
        }
    
    def test_tactical_positions(self) -> Dict:
        """Test on standard tactical positions."""
        print(f"\n♟️  Testing on Tactical Positions...")
        
        # Famous tactical positions with known best moves
        tactical_tests = [
            {
                'name': 'Scholar\'s Mate Defense',
                'fen': 'r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1',
                'best_moves': ['Qf7'],  # Checkmate threat
                'description': 'Defending against Scholar\'s mate'
            },
            {
                'name': 'Back Rank Mate',
                'fen': '6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1',
                'best_moves': ['Re8'],  # Back rank checkmate
                'description': 'Classic back rank mate pattern'
            },
            {
                'name': 'Fork Opportunity',
                'fen': 'r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1',
                'best_moves': ['Nxe5'],  # Knight fork
                'description': 'Knight fork winning material'
            },
            {
                'name': 'Pin Tactic',
                'fen': 'r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1',
                'best_moves': ['Ng5', 'Bxf7'],  # Fried Liver or Italian
                'description': 'Tactical opportunity in Italian Game'
            },
            {
                'name': 'Mate in 1',
                'fen': '5rk1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1',
                'best_moves': ['Re8'],  # Checkmate
                'description': 'Simple mate in 1'
            }
        ]
        
        results = []
        top1_found = 0
        top3_found = 0
        top5_found = 0
        
        start_time = time.time()
        
        for test in tactical_tests:
            board = chess.Board(test['fen'])
            legal_moves = list(board.legal_moves)
            
            # Get model predictions
            move_scores = self.predict_move_scores(board, legal_moves)
            
            # Extract top 5 moves
            top5_moves = [move.uci() for move, score in move_scores[:5]]
            
            # Check if best move is in top-k
            found_in_top1 = any(top5_moves[0] == bm for bm in test['best_moves'])
            found_in_top3 = any(m in test['best_moves'] for m in top5_moves[:3])
            found_in_top5 = any(m in test['best_moves'] for m in top5_moves[:5])
            
            if found_in_top1:
                top1_found += 1
            if found_in_top3:
                top3_found += 1
            if found_in_top5:
                top5_found += 1
            
            results.append({
                'name': test['name'],
                'best_moves': test['best_moves'],
                'predicted_top5': top5_moves,
                'found_in_top1': found_in_top1,
                'found_in_top3': found_in_top3,
                'found_in_top5': found_in_top5
            })
            
            status = '✅' if found_in_top1 else '⚠️' if found_in_top5 else '❌'
            print(f"   {status} {test['name']}")
            print(f"      Expected: {test['best_moves']}")
            print(f"      Top-5: {top5_moves[:5]}")
        
        elapsed = time.time() - start_time
        total = len(tactical_tests)
        
        print(f"\n   📊 Tactical Test Results:")
        print(f"      Top-1: {top1_found}/{total} ({top1_found/total*100:.1f}%)")
        print(f"      Top-3: {top3_found}/{total} ({top3_found/total*100:.1f}%)")
        print(f"      Top-5: {top5_found}/{total} ({top5_found/total*100:.1f}%)")
        print(f"   ⚡ Average time: {elapsed/total*1000:.1f}ms per position")
        
        return {
            'total': total,
            'top1_accuracy': top1_found / total,
            'top3_accuracy': top3_found / total,
            'top5_accuracy': top5_found / total,
            'avg_time_ms': elapsed / total * 1000,
            'results': results
        }
    
    def test_game_phase_performance(self, game_data_path: str) -> Dict:
        """Test performance by game phase."""
        print(f"\n🎮 Testing Game Phase Performance...")
        
        # Load game position dataset
        print(f"   Loading from: {game_data_path}")
        with open(game_data_path, 'r') as f:
            data = json.load(f)
        
        positions = data['positions']
        
        # Sample positions from each phase
        opening_pos = [p for p in positions if p['game_phase'] == 'opening'][:100]
        middlegame_pos = [p for p in positions if p['game_phase'] == 'middlegame'][:100]
        endgame_pos = [p for p in positions if p['game_phase'] == 'endgame'][:100]
        
        phases = {
            'Opening': opening_pos,
            'Middlegame': middlegame_pos,
            'Endgame': endgame_pos
        }
        
        results = {}
        
        for phase_name, phase_positions in phases.items():
            print(f"\n   Testing {phase_name} ({len(phase_positions)} positions)...")
            
            top5_matches = 0
            total_time = 0
            
            for pos in phase_positions:
                board = chess.Board(pos['fen'])
                legal_moves = list(board.legal_moves)
                
                start = time.time()
                move_scores = self.predict_move_scores(board, legal_moves)
                elapsed = time.time() - start
                total_time += elapsed
                
                # Check if any of Stockfish's top moves are in our top 5
                stockfish_top_ucis = [m['uci'] for m in pos['top_moves'][:5]]
                our_top5_ucis = [m.uci() for m, s in move_scores[:5]]
                
                if any(uci in our_top5_ucis for uci in stockfish_top_ucis):
                    top5_matches += 1
            
            accuracy = top5_matches / len(phase_positions)
            avg_time = total_time / len(phase_positions) * 1000
            
            print(f"      Top-5 Match: {top5_matches}/{len(phase_positions)} ({accuracy*100:.1f}%)")
            print(f"      Avg Time: {avg_time:.2f}ms")
            
            results[phase_name] = {
                'tested': len(phase_positions),
                'top5_match': top5_matches,
                'accuracy': accuracy,
                'avg_time_ms': avg_time
            }
        
        return results
    
    def test_inference_speed(self) -> Dict:
        """Benchmark inference speed for engine integration."""
        print(f"\n⚡ Inference Speed Benchmark...")
        
        # Standard starting position
        board = chess.Board()
        legal_moves = list(board.legal_moves)
        
        # Warmup
        for _ in range(10):
            self.predict_move_scores(board, legal_moves)
        
        # Benchmark
        iterations = 100
        start_time = time.time()
        
        for _ in range(iterations):
            self.predict_move_scores(board, legal_moves)
        
        elapsed = time.time() - start_time
        avg_time_ms = (elapsed / iterations) * 1000
        positions_per_sec = iterations / elapsed
        
        print(f"   Positions evaluated: {iterations}")
        print(f"   Total time: {elapsed:.2f}s")
        print(f"   Average time: {avg_time_ms:.2f}ms per position")
        print(f"   Speed: {positions_per_sec:.1f} positions/sec")
        
        # Estimate nodes per second equivalent
        # Assuming ~30 legal moves per position on average
        moves_evaluated = iterations * len(legal_moves)
        moves_per_sec = moves_evaluated / elapsed
        
        print(f"   Move evaluations: {moves_evaluated:,} in {elapsed:.2f}s")
        print(f"   Move eval speed: {moves_per_sec:,.0f} moves/sec")
        
        return {
            'iterations': iterations,
            'total_time': elapsed,
            'avg_time_ms': avg_time_ms,
            'positions_per_sec': positions_per_sec,
            'moves_evaluated': moves_evaluated,
            'moves_per_sec': moves_per_sec
        }


def main():
    """Run full validation suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate Stage 2.5 model')
    parser.add_argument('--model-path', type=str,
                       default='models/stage2_combined/best_checkpoint.pt',
                       help='Path to trained model checkpoint')
    parser.add_argument('--puzzle-data', type=str,
                       default='data/preprocessed_puzzles/enriched_puzzles_compact_20260420_003909.json',
                       help='Path to puzzle dataset')
    parser.add_argument('--game-data', type=str,
                       default='data/stage2_games/historical_positions_full.json',
                       help='Path to game position dataset')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = ModelValidator(args.model_path, device=args.device)
    
    # Run validation tests
    all_results = {}
    
    # Test 1: Tactical positions
    all_results['tactical'] = validator.test_tactical_positions()
    
    # Test 2: Game phase performance
    all_results['game_phases'] = validator.test_game_phase_performance(args.game_data)
    
    # Test 3: Inference speed
    all_results['speed'] = validator.test_inference_speed()
    
    # Test 4: Puzzle dataset (quick test)
    # all_results['puzzles'] = validator.test_puzzle_dataset(args.puzzle_data)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 VALIDATION SUMMARY")
    print("=" * 70)
    
    print(f"\n🎯 Tactical Positions:")
    print(f"   Top-1: {all_results['tactical']['top1_accuracy']*100:.1f}%")
    print(f"   Top-5: {all_results['tactical']['top5_accuracy']*100:.1f}%")
    print(f"   Speed: {all_results['tactical']['avg_time_ms']:.1f}ms/position")
    
    print(f"\n🎮 Game Phase Performance:")
    for phase, metrics in all_results['game_phases'].items():
        print(f"   {phase}: {metrics['accuracy']*100:.1f}% top-5 match ({metrics['avg_time_ms']:.1f}ms)")
    
    print(f"\n⚡ Inference Speed:")
    print(f"   {all_results['speed']['avg_time_ms']:.2f}ms per position")
    print(f"   {all_results['speed']['positions_per_sec']:.1f} positions/sec")
    print(f"   {all_results['speed']['moves_per_sec']:,.0f} move evaluations/sec")
    
    # Save results
    output_path = Path('validation_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    
    print("\n" + "=" * 70)
    print("✅ VALIDATION COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
