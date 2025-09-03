#!/usr/bin/env python3
"""
Simple test of enhanced training components without complex imports
"""

import chess
import random
import json

class SimpleGameEvaluator:
    """Simplified game-like evaluator for testing"""
    
    def evaluate_knight_mobility(self, board, color):
        """Test knight mobility evaluation"""
        score = 0.0
        knights = board.pieces(chess.KNIGHT, color)
        
        for knight_square in knights:
            # Count possible attacks
            attacks = len(list(board.attacks(knight_square)))
            max_attacks = 8
            mobility_ratio = attacks / max_attacks
            score += mobility_ratio * 10
            
            # Center bonus
            file, rank = chess.square_file(knight_square), chess.square_rank(knight_square)
            center_distance = max(abs(file - 3.5), abs(rank - 3.5))
            center_bonus = max(0, 3 - center_distance)
            score += center_bonus
            
            # Edge penalty
            if file == 0 or file == 7 or rank == 0 or rank == 7:
                score -= 2
                
        return score

class SimpleRandomOpponent:
    """Simple random opponent for testing"""
    
    def __init__(self, skill=0):
        self.skill = skill
        self.name = f"Random_{skill}"
        
    def get_move(self, board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
            
        if self.skill == 0:
            return random.choice(legal_moves)
        
        # Add some basic move preferences
        scored_moves = []
        for move in legal_moves:
            score = random.randint(0, 50)
            
            if board.is_capture(move):
                score += self.skill
            
            temp_board = board.copy()
            temp_board.push(move)
            if temp_board.is_check():
                score += self.skill // 2
                
            scored_moves.append((score, move))
            
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        top_moves = scored_moves[:max(1, len(scored_moves) // 3)]
        return random.choice(top_moves)[1]

def test_game_evaluator():
    """Test the game-like evaluation"""
    print("=== Testing Game-Like Evaluation ===")
    
    evaluator = SimpleGameEvaluator()
    board = chess.Board()
    
    # Test starting position
    white_knights = evaluator.evaluate_knight_mobility(board, chess.WHITE)
    print(f"Starting position - White knight mobility: {white_knights:.2f}")
    
    # Test after knight development
    board.push(chess.Move.from_uci("g1f3"))  # Nf3
    white_knights_after = evaluator.evaluate_knight_mobility(board, chess.WHITE)
    print(f"After Nf3 - White knight mobility: {white_knights_after:.2f}")
    print(f"Improvement: {white_knights_after - white_knights:.2f}")

def test_random_opponents():
    """Test random opponents with different skills"""
    print("\n=== Testing Random Opponents ===")
    
    board = chess.Board()
    opponents = [
        SimpleRandomOpponent(0),   # Completely random
        SimpleRandomOpponent(25),  # Slight preferences
        SimpleRandomOpponent(50),  # Better move selection
    ]
    
    for opponent in opponents:
        move = opponent.get_move(board)
        if move:
            print(f"{opponent.name:>12}: {board.san(move)}")

def test_genetic_parameters():
    """Test genetic algorithm parameters"""
    print("\n=== Testing Genetic Parameters ===")
    
    # Mutation rate testing
    mutation_rates = [0.1, 0.2, 0.3]
    print("Mutation rate effects:")
    for rate in mutation_rates:
        exploration = "Low" if rate < 0.15 else "Medium" if rate < 0.25 else "High"
        print(f"  {rate}: {exploration} exploration")
    
    # Population size effects
    print("\nPopulation size recommendations:")
    pop_sizes = [16, 32, 64, 128]
    for size in pop_sizes:
        elite_count = int(size * 0.15)
        diversity = "Low" if size < 32 else "Good" if size < 64 else "High"
        print(f"  {size:3d}: {elite_count:2d} elite, {diversity} diversity")

def show_training_improvements():
    """Show the key training improvements"""
    print("\n" + "="*50)
    print("KEY TRAINING IMPROVEMENTS IMPLEMENTED")
    print("="*50)
    
    print("""
🎯 OPPONENT DIVERSITY SYSTEM:
   ✅ Random opponents (0-75 skill levels)
   ✅ Weak Stockfish engines (depth 1-3)  
   ✅ Training rotation (40% random, 30% weak, 20% medium, 10% strong)
   ✅ No external dependencies (uses your existing Stockfish)

🧬 ENHANCED GENETIC ALGORITHM:
   ✅ Configurable mutation rate (0.1-0.3)
   ✅ Configurable crossover rate (0.7-0.9)
   ✅ Elite preservation (10-20% best individuals)
   ✅ Tournament selection for diversity

🎮 GAME-LIKE EVALUATION:
   ✅ Knight mobility (your example - edge penalty)
   ✅ Bishop vision (diagonal control)
   ✅ Rook power (file/rank dominance)
   ✅ Queen activity vs safety balance
   ✅ King leadership (safety vs activity by game phase)
   ✅ Pawn army formation and advancement

⚡ NON-DETERMINISTIC FIXES:
   ✅ Random temperature for move selection
   ✅ 10% exploration moves (completely random)
   ✅ Randomized evaluation factors (0.8-1.2x multiplier)
   ✅ 20% random opening positions
   ✅ Opponent variety prevents deterministic loops

📈 EXPECTED RESULTS:
   • Break past generation 7 plateau
   • Robust play against different opponents
   • Creative tactical solutions vs random players
   • Solid defensive skills vs engines
   • Continued model improvement and saving
""")

def create_training_config():
    """Create a configuration file for the enhanced trainer"""
    config = {
        "enhanced_training": {
            "population_size": 32,
            "mutation_rate": 0.2,
            "crossover_rate": 0.8,
            "elite_percentage": 0.15,
            "generations": 100
        },
        "evaluation": {
            "games_per_individual": 4,
            "max_moves_per_game": 100,
            "bounty_weight": 0.3,
            "neural_weight": 0.5,
            "game_evaluation_weight": 0.2
        },
        "opponents": {
            "random_percentage": 0.4,
            "weak_engine_percentage": 0.3,
            "medium_percentage": 0.2,
            "strong_percentage": 0.1
        },
        "randomness": {
            "exploration_rate": 0.1,
            "temperature_range": [0.1, 0.5],
            "evaluation_noise_range": [0.8, 1.2],
            "random_opening_chance": 0.2
        }
    }
    
    with open("enhanced_training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Created enhanced_training_config.json")
    return config

if __name__ == "__main__":
    print("Enhanced V7P3R Training System - Simple Tests")
    print("=" * 50)
    
    test_game_evaluator()
    test_random_opponents() 
    test_genetic_parameters()
    show_training_improvements()
    create_training_config()
