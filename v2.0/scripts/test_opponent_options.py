#!/usr/bin/env python3
"""
Test available opponent options for training
"""

import chess
import chess.engine
import random
import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class RandomPlayer:
    """Simple random move player"""
    
    def __init__(self, skill_level=0):
        self.skill_level = skill_level  # 0 = completely random, 100 = slightly better
        
    def get_move(self, board):
        """Get a move from this player"""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
            
        if self.skill_level == 0:
            # Completely random
            return random.choice(legal_moves)
        else:
            # Add some basic move ordering (captures, checks, etc.)
            scored_moves = []
            for move in legal_moves:
                score = random.randint(0, 100)  # Base randomness
                
                # Slight preferences for "good" moves
                if board.is_capture(move):
                    score += self.skill_level // 4
                    
                temp_board = board.copy()
                temp_board.push(move)
                if temp_board.is_check():
                    score += self.skill_level // 2
                    
                scored_moves.append((score, move))
                
            # Sort by score and add randomness
            scored_moves.sort(key=lambda x: x[0], reverse=True)
            top_moves = scored_moves[:max(1, len(scored_moves) // 3)]
            return random.choice(top_moves)[1]

class WeakEngine:
    """Wrapper for a weakened engine (like limited depth Stockfish)"""
    
    def __init__(self, engine_path="stockfish.exe", depth=1, time_limit=0.1):
        self.engine_path = engine_path
        self.depth = depth
        self.time_limit = time_limit
        self.engine = None
        
    def start(self):
        """Start the engine"""
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
            return True
        except:
            return False
            
    def stop(self):
        """Stop the engine"""
        if self.engine:
            self.engine.quit()
            self.engine = None
            
    def get_move(self, board):
        """Get a move from the weak engine"""
        if not self.engine:
            return None
            
        try:
            limit = chess.engine.Limit(depth=self.depth, time=self.time_limit)
            result = self.engine.play(board, limit)
            return result.move
        except:
            return None

def test_random_player():
    """Test the random player"""
    print("=== Testing Random Player ===")
    
    board = chess.Board()
    random_player = RandomPlayer(skill_level=20)
    
    for i in range(5):
        move = random_player.get_move(board)
        if move:
            print(f"Random move {i+1}: {board.san(move)}")
            board.push(move)
        else:
            break
            
    print(f"Board after 5 random moves:\n{board}")

def test_weak_engine():
    """Test the weak engine"""
    print("\n=== Testing Weak Engine ===")
    
    weak_engine = WeakEngine(depth=1, time_limit=0.05)  # Very weak settings
    
    if weak_engine.start():
        print("✅ Weak engine started successfully")
        
        board = chess.Board()
        for i in range(3):
            move = weak_engine.get_move(board)
            if move:
                print(f"Weak engine move {i+1}: {board.san(move)}")
                board.push(move)
            else:
                break
                
        weak_engine.stop()
        print(f"Board after 3 weak engine moves:\n{board}")
    else:
        print("❌ Could not start weak engine (Stockfish not available)")

def show_opponent_plan():
    """Show the planned opponent system"""
    print("\n=== OPPONENT SYSTEM PLAN ===")
    print("""
1. RANDOM OPPONENTS:
   - Level 0: Completely random moves
   - Level 25: Slightly prefer captures/checks
   - Level 50: Basic move ordering with randomness
   - Level 75: Decent move selection with variation

2. WEAK ENGINES:
   - Stockfish Depth 1 (0.05s): Very weak, ~400 ELO
   - Stockfish Depth 2 (0.1s): Weak, ~800 ELO  
   - Stockfish Depth 3 (0.2s): Beginner, ~1200 ELO

3. TRAINING ROTATION:
   - 40% Random opponents (variety and tactics training)
   - 30% Weak engines (positional understanding)
   - 20% Self-play (strategic depth)
   - 10% Strong engines (defensive training)

4. GENETIC ALGORITHM IMPROVEMENTS:
   - Add mutation rate parameter (0.1-0.3)
   - Add crossover rate parameter (0.7-0.9)
   - Add elite preservation (top 10% survive)
   - Add population diversity tracking

5. GAME-LIKE EVALUATION:
   - Piece mobility scores (knights on edge penalty)
   - Attack/defense ratios per piece
   - Space control metrics
   - Tactical pattern recognition
   """)

if __name__ == "__main__":
    print("Testing Opponent Options for V7P3R Training")
    print("=" * 50)
    
    test_random_player()
    test_weak_engine()
    show_opponent_plan()
