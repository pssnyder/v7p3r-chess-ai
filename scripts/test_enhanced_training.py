#!/usr/bin/env python3
"""
Test the enhanced genetic trainer components
"""

import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from training.enhanced_genetic_trainer import GameLikeEvaluator
from training.multi_opponent_system import OpponentManager
import chess

def test_game_like_evaluator():
    """Test the game-like evaluation system"""
    print("=== Testing Game-Like Evaluator ===")
    
    evaluator = GameLikeEvaluator()
    board = chess.Board()
    
    # Test piece performance evaluation
    scores = evaluator.evaluate_piece_performance(board, chess.WHITE)
    
    print("Starting position piece scores:")
    for metric, score in scores.items():
        print(f"  {metric}: {score:.2f}")
    
    # Test after some moves
    moves = ["e4", "e5", "Nf3", "Nc6", "Bb5"]
    for move_san in moves:
        try:
            move = board.parse_san(move_san)
            board.push(move)
        except:
            break
    
    scores_after = evaluator.evaluate_piece_performance(board, chess.WHITE)
    print(f"\nAfter moves {' '.join(moves)}:")
    for metric, score in scores_after.items():
        change = score - scores[metric]
        print(f"  {metric}: {score:.2f} ({change:+.2f})")

def test_opponent_variety():
    """Test opponent variety"""
    print("\n=== Testing Opponent Variety ===")
    
    manager = OpponentManager()
    board = chess.Board()
    
    print("Testing 10 moves from different opponents:")
    for i in range(10):
        opponent = manager.get_random_opponent()
        move = opponent.get_move(board)
        if move:
            print(f"  {i+1:2d}. {opponent.name:>15}: {board.san(move)}")
            board.push(move)
        else:
            break
    
    manager.cleanup()

def test_genetic_parameters():
    """Test genetic algorithm parameter handling"""
    print("\n=== Testing Genetic Parameters ===")
    
    # Test parameter combinations
    configs = [
        {"pop": 16, "mut": 0.1, "cross": 0.7, "elite": 0.1},
        {"pop": 32, "mut": 0.2, "cross": 0.8, "elite": 0.15},
        {"pop": 64, "mut": 0.3, "cross": 0.9, "elite": 0.2},
    ]
    
    for i, config in enumerate(configs):
        print(f"Config {i+1}:")
        print(f"  Population: {config['pop']}")
        print(f"  Mutation Rate: {config['mut']}")
        print(f"  Crossover Rate: {config['cross']}")
        print(f"  Elite Count: {int(config['pop'] * config['elite'])}")
        print(f"  Breeding Pool: {config['pop'] - int(config['pop'] * config['elite'])}")

def show_training_plan():
    """Show the complete training plan"""
    print("\n" + "="*60)
    print("ENHANCED V7P3R TRAINING SYSTEM - IMPLEMENTATION PLAN")
    print("="*60)
    
    print("""
🎮 GAME-LIKE APPROACH:
   • Pieces as video game characters with special abilities
   • Knights: Cavalry units (mobility = power)
   • Bishops: Archers (vision = range)
   • Rooks: Tanks (file/rank control)
   • Queen: Hero unit (activity + safety)
   • King: Commander (safety in opening, activity in endgame)
   • Pawns: Army foundation (advancement + formation)

🎯 OPPONENT DIVERSITY:
   • 40% Random opponents (0-75 skill) - Tactics training
   • 30% Weak engines (Stockfish depth 1-3) - Positional basics
   • 20% Medium opponents - Challenge and improvement
   • 10% Strong opponents - Defensive training

🧬 GENETIC ALGORITHM ENHANCEMENTS:
   • Mutation rate: 0.1-0.3 (configurable exploration)
   • Crossover rate: 0.7-0.9 (genetic mixing)
   • Elite preservation: 10-20% (keep best individuals)
   • Population diversity tracking
   • Temperature-based move selection

⚡ NON-DETERMINISTIC EVALUATION:
   • Random temperature for move selection
   • Exploration moves (10% random)
   • Randomized evaluation factors (0.8-1.2x)
   • Diverse starting positions (20% random openings)
   • Opponent variety prevents overfitting

📊 EVALUATION COMPONENTS:
   • Bounty system: 30% (tactical rewards)
   • Game-like metrics: 20% (piece performance)
   • Neural network: 50% (learned patterns)
   • Outcome bonuses: Win +1000, Draw +200, Loss -200
   • Exploration bonus: Longer games rewarded

🚀 EXPECTED IMPROVEMENTS:
   • Break training plateaus through variety
   • Learn robust strategies vs diverse opponents
   • Develop tactical creativity vs random players
   • Build defensive skills vs strong engines
   • Maintain genetic diversity through randomness
""")

if __name__ == "__main__":
    print("Enhanced V7P3R Genetic Training System - Component Tests")
    print("=" * 60)
    
    test_game_like_evaluator()
    test_opponent_variety()
    test_genetic_parameters()
    show_training_plan()
