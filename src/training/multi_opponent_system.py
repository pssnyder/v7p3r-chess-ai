#!/usr/bin/env python3
"""
Multi-Opponent Training System for V7P3R Chess AI
Implements various opponents with different skill levels and styles
"""

import chess
import chess.engine
import random
import threading
import time
from typing import Optional, List, Tuple
import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class RandomOpponent:
    """Configurable random opponent with skill levels"""
    
    def __init__(self, skill_level: int = 0, name: str = "Random"):
        """
        Initialize random opponent
        skill_level: 0 = completely random, 100 = decent move selection
        """
        self.skill_level = min(100, max(0, skill_level))
        self.name = f"{name}_L{skill_level}"
        
    def get_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Get a move from this opponent"""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
            
        if self.skill_level == 0:
            # Completely random
            return random.choice(legal_moves)
        
        # Add intelligence based on skill level
        scored_moves = []
        for move in legal_moves:
            score = random.randint(0, 50)  # Base randomness
            
            # Bonus for captures
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                if captured_piece:
                    piece_values = {
                        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
                    }
                    capture_bonus = piece_values.get(captured_piece.piece_type, 0) * 10
                    score += int(capture_bonus * self.skill_level / 100)
            
            # Bonus for checks
            temp_board = board.copy()
            temp_board.push(move)
            if temp_board.is_check():
                score += int(20 * self.skill_level / 100)
                
            # Bonus for center moves
            center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
            if move.to_square in center_squares:
                score += int(5 * self.skill_level / 100)
                
            # Penalty for moving into attacks (higher skill avoids this)
            if self.skill_level > 30:
                if temp_board.is_attacked_by(not board.turn, move.to_square):
                    score -= int(15 * self.skill_level / 100)
            
            scored_moves.append((score, move))
        
        # Select from top moves based on skill level
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        
        # Choose from top percentage based on skill
        top_percent = max(10, 100 - self.skill_level)  # Higher skill = consider fewer moves
        num_top_moves = max(1, len(scored_moves) * top_percent // 100)
        top_moves = scored_moves[:num_top_moves]
        
        return random.choice(top_moves)[1]

class WeakEngineOpponent:
    """Opponent using weakened external engine (Stockfish)"""
    
    def __init__(self, engine_path: str = "stockfish.exe", depth: int = 1, 
                 time_limit: float = 0.1, name: str = "WeakStock"):
        self.engine_path = engine_path
        self.depth = depth
        self.time_limit = time_limit
        self.name = f"{name}_D{depth}T{int(time_limit*1000)}ms"
        self.engine = None
        self._lock = threading.Lock()
        
    def start(self) -> bool:
        """Start the engine"""
        with self._lock:
            try:
                if self.engine:
                    return True
                self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
                return True
            except Exception as e:
                print(f"Failed to start {self.name}: {e}")
                return False
                
    def stop(self):
        """Stop the engine"""
        with self._lock:
            if self.engine:
                try:
                    self.engine.quit()
                except:
                    pass
                self.engine = None
                
    def get_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Get a move from the weak engine"""
        if not self.engine and not self.start():
            return None
            
        try:
            with self._lock:
                if self.engine:  # Additional check
                    limit = chess.engine.Limit(depth=self.depth, time=self.time_limit)
                    result = self.engine.play(board, limit)
                    return result.move
                return None
        except Exception as e:
            print(f"Engine error in {self.name}: {e}")
            return None

class OpponentManager:
    """Manages multiple opponents for training variety"""
    
    def __init__(self, stockfish_path: str = "stockfish.exe"):
        self.stockfish_path = stockfish_path
        self.opponents = []
        self._setup_opponents()
        
    def _setup_opponents(self):
        """Setup all available opponents"""
        # Random opponents with different skill levels
        self.opponents.extend([
            RandomOpponent(0, "Chaos"),      # Completely random
            RandomOpponent(25, "Novice"),    # Slight preferences
            RandomOpponent(50, "Amateur"),   # Basic tactics
            RandomOpponent(75, "Decent"),    # Good move selection
        ])
        
        # Weak engine opponents (if Stockfish available)
        weak_engines = [
            WeakEngineOpponent(self.stockfish_path, 1, 0.05, "Minimal"),  # ~400 ELO
            WeakEngineOpponent(self.stockfish_path, 2, 0.1, "Weak"),      # ~800 ELO
            WeakEngineOpponent(self.stockfish_path, 3, 0.2, "Basic"),     # ~1200 ELO
        ]
        
        # Test which engines work
        for engine in weak_engines:
            if engine.start():
                self.opponents.append(engine)
                print(f"✅ Added opponent: {engine.name}")
            else:
                print(f"❌ Failed to add: {engine.name}")
        
    def get_random_opponent(self, exclude_types: Optional[List[str]] = None):
        """Get a random opponent, optionally excluding certain types"""
        available = self.opponents
        
        if exclude_types:
            available = [opp for opp in self.opponents 
                        if not any(ex in opp.name for ex in exclude_types)]
        
        if not available:
            # Fallback to simplest random opponent
            return RandomOpponent(0, "Fallback")
            
        return random.choice(available)
        
    def get_opponent_by_strength(self, strength: str):
        """Get opponent by strength category"""
        strength_map = {
            'random': [opp for opp in self.opponents if 'Chaos' in opp.name or 'Novice' in opp.name],
            'weak': [opp for opp in self.opponents if 'Amateur' in opp.name or 'Minimal' in opp.name],
            'medium': [opp for opp in self.opponents if 'Decent' in opp.name or 'Weak' in opp.name],
            'strong': [opp for opp in self.opponents if 'Basic' in opp.name],
        }
        
        candidates = strength_map.get(strength, self.opponents)
        return random.choice(candidates) if candidates else RandomOpponent(0)
        
    def get_training_rotation(self) -> List[Tuple[str, float]]:
        """Get recommended training rotation with percentages"""
        return [
            ('random', 0.4),   # 40% random for variety and tactics
            ('weak', 0.3),     # 30% weak engines for basics
            ('medium', 0.2),   # 20% medium for challenge
            ('strong', 0.1),   # 10% strong for defense training
        ]
        
    def cleanup(self):
        """Cleanup all engine opponents"""
        for opponent in self.opponents:
            if hasattr(opponent, 'stop'):
                opponent.stop()

def test_opponent_system():
    """Test the opponent system"""
    print("=== Testing Multi-Opponent System ===")
    
    manager = OpponentManager()
    
    print(f"\nAvailable opponents: {len(manager.opponents)}")
    for opp in manager.opponents:
        print(f"  - {opp.name}")
    
    # Test random selection
    print(f"\nRandom opponent: {manager.get_random_opponent().name}")
    print(f"Weak opponent: {manager.get_opponent_by_strength('weak').name}")
    print(f"Strong opponent: {manager.get_opponent_by_strength('strong').name}")
    
    # Test training rotation
    print(f"\nTraining rotation:")
    for strength, percentage in manager.get_training_rotation():
        opp = manager.get_opponent_by_strength(strength)
        print(f"  {percentage*100:>3.0f}% - {strength:>6} ({opp.name})")
    
    # Test a few moves
    print(f"\nTesting moves from different opponents:")
    board = chess.Board()
    
    for i in range(3):
        opp = manager.get_random_opponent()
        move = opp.get_move(board)
        if move:
            print(f"  {opp.name:>15}: {board.san(move)}")
            board.push(move)
    
    manager.cleanup()
    print(f"\nFinal position:\n{board}")

if __name__ == "__main__":
    test_opponent_system()
