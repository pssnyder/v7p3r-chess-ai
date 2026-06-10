"""
Opponent Manager for V8.0 Training

Manages a pool of UCI chess engines to provide opponent diversity during training.
Rotates through opponents, tracks statistics, and handles UCI communication.
"""

import subprocess
import threading
import queue
import time
import chess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import random


@dataclass
class OpponentConfig:
    """Configuration for a single opponent engine"""
    name: str
    path: str  # Path to .bat or .py file
    weight: float = 1.0  # Relative selection probability
    estimated_elo: int = 1200  # Rough estimate
    style: str = "balanced"  # balanced, aggressive, positional, tactical, random


class UCIEngine:
    """
    UCI engine subprocess wrapper
    
    Handles launching, communication, and cleanup of external UCI engines.
    """
    
    def __init__(self, engine_path: str, timeout: float = 30.0):
        """
        Initialize UCI engine
        
        Args:
            engine_path: Path to engine executable (.bat, .py, .exe)
            timeout: Command timeout in seconds
        """
        self.engine_path = engine_path
        self.timeout = timeout
        self.process = None
        self.ready = False
        
        # Determine launch command
        if engine_path.endswith('.bat'):
            # Windows batch file
            self.launch_cmd = [engine_path]
        elif engine_path.endswith('.py'):
            # Python script
            self.launch_cmd = ['python', engine_path]
        elif engine_path.endswith('.exe'):
            # Executable
            self.launch_cmd = [engine_path]
        else:
            raise ValueError(f"Unknown engine type: {engine_path}")
    
    def start(self):
        """Launch engine process"""
        try:
            self.process = subprocess.Popen(
                self.launch_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Send uci command
            self.send_command("uci")
            
            # Wait for uciok
            start_time = time.time()
            while time.time() - start_time < 10.0:
                line = self.read_line(timeout=1.0)
                if line and "uciok" in line:
                    self.ready = True
                    break
            
            if not self.ready:
                raise RuntimeError("Engine did not respond to uci command")
            
            # Send isready
            self.send_command("isready")
            response = self.wait_for("readyok", timeout=5.0)
            if not response:
                raise RuntimeError("Engine did not respond to isready command")
        
        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Failed to start engine {self.engine_path}: {e}")
    
    def send_command(self, command: str):
        """Send command to engine"""
        if self.process and self.process.stdin:
            try:
                self.process.stdin.write(command + "\n")
                self.process.stdin.flush()
            except:
                pass
    
    def read_line(self, timeout: float = 1.0) -> Optional[str]:
        """Read one line from engine (with timeout)"""
        if not self.process or not self.process.stdout:
            return None
        
        # Simple timeout-based read (not perfect but works)
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Check if process is still alive
                if self.process.poll() is not None:
                    return None
                
                # Try to read with short timeout
                line = self.process.stdout.readline()
                if line:
                    return line.strip()
            except:
                break
        
        return None
    
    def wait_for(self, keyword: str, timeout: float = 5.0) -> Optional[str]:
        """Wait for a line containing keyword"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            line = self.read_line(timeout=1.0)
            if line and keyword in line:
                return line
        return None
    
    def get_move(self, fen: str, movetime_ms: int = 3000, move_history: List = None) -> Optional['chess.Move']:
        """
        Get move from engine for given position
        
        Args:
            fen: FEN string of position
            movetime_ms: Time to think in milliseconds
            move_history: List of chess.Move objects (for position startpos moves)
        
        Returns:
            chess.Move or None
        """
        try:
            # Set position - prefer move history over FEN for better compatibility
            if move_history is not None and len(move_history) == 0:
                # Starting position
                self.send_command("position startpos")
            elif move_history is not None:
                # Position with move history
                moves_uci = " ".join([move.uci() for move in move_history])
                self.send_command(f"position startpos moves {moves_uci}")
            else:
                # Fallback to FEN
                self.send_command(f"position fen {fen}")
            
            # Request move
            self.send_command(f"go movetime {movetime_ms}")
            
            # Wait for bestmove
            response = self.wait_for("bestmove", timeout=movetime_ms/1000.0 + 5.0)
            
            if response and "bestmove" in response:
                parts = response.split()
                bestmove_idx = parts.index("bestmove")
                if bestmove_idx + 1 < len(parts):
                    move_uci = parts[bestmove_idx + 1]
                    try:
                        return chess.Move.from_uci(move_uci)
                    except:
                        return None
        except:
            pass
        
        return None
    
    def cleanup(self):
        """Clean up engine process"""
        if self.process:
            try:
                self.send_command("quit")
                self.process.wait(timeout=2.0)
            except:
                pass
            
            try:
                if self.process.poll() is None:
                    self.process.terminate()
                    self.process.wait(timeout=1.0)
            except:
                pass
            
            try:
                if self.process.poll() is None:
                    self.process.kill()
            except:
                pass
            
            self.process = None
            self.ready = False


class OpponentPool:
    """
    Manages a pool of opponent engines for training
    
    Handles rotation, statistics, and lifecycle management.
    """
    
    def __init__(self, opponents: List[OpponentConfig]):
        """
        Initialize opponent pool
        
        Args:
            opponents: List of opponent configurations
        """
        self.opponents = opponents
        self.current_engine = None
        self.current_opponent_idx = 0
        
        # Statistics
        self.stats = {
            opp.name: {
                'games_played': 0,
                'wins': 0,
                'draws': 0,
                'losses': 0,
                'avg_moves': 0.0,
                'total_moves': 0
            }
            for opp in opponents
        }
        
        # Normalize weights
        total_weight = sum(opp.weight for opp in opponents)
        for opp in self.opponents:
            opp.weight /= total_weight
    
    def get_next_opponent(self, strategy: str = "weighted_random") -> OpponentConfig:
        """
        Select next opponent
        
        Args:
            strategy: Selection strategy
                - "round_robin": Cycle through in order
                - "weighted_random": Random weighted by config
                - "weakest_first": Prioritize weakest opponents
        
        Returns:
            OpponentConfig
        """
        if strategy == "round_robin":
            opponent = self.opponents[self.current_opponent_idx]
            self.current_opponent_idx = (self.current_opponent_idx + 1) % len(self.opponents)
            return opponent
        
        elif strategy == "weighted_random":
            weights = [opp.weight for opp in self.opponents]
            return random.choices(self.opponents, weights=weights, k=1)[0]
        
        elif strategy == "weakest_first":
            # Sort by estimated ELO
            sorted_opps = sorted(self.opponents, key=lambda x: x.estimated_elo)
            return sorted_opps[self.current_opponent_idx % len(sorted_opps)]
        
        else:
            return random.choice(self.opponents)
    
    def launch_opponent(self, opponent: OpponentConfig) -> UCIEngine:
        """
        Launch an opponent engine
        
        Args:
            opponent: Opponent configuration
        
        Returns:
            UCIEngine instance
        """
        engine = UCIEngine(opponent.path, timeout=30.0)
        engine.start()
        return engine
    
    def record_game(self, opponent_name: str, result: str, num_moves: int):
        """
        Record game statistics
        
        Args:
            opponent_name: Name of opponent
            result: "1-0" (win), "1/2-1/2" (draw), "0-1" (loss) from v8's perspective
            num_moves: Number of moves in game
        """
        if opponent_name in self.stats:
            stats = self.stats[opponent_name]
            stats['games_played'] += 1
            stats['total_moves'] += num_moves
            stats['avg_moves'] = stats['total_moves'] / stats['games_played']
            
            if result == "1-0":
                stats['wins'] += 1
            elif result == "1/2-1/2":
                stats['draws'] += 1
            elif result == "0-1":
                stats['losses'] += 1
    
    def get_win_rate(self, opponent_name: str) -> float:
        """Get win rate against specific opponent"""
        if opponent_name in self.stats:
            stats = self.stats[opponent_name]
            total = stats['games_played']
            if total > 0:
                return stats['wins'] / total
        return 0.0
    
    def print_summary(self):
        """Print statistics summary"""
        print("\n" + "="*70)
        print("OPPONENT TRAINING STATISTICS")
        print("="*70)
        
        for opp in self.opponents:
            stats = self.stats[opp.name]
            games = stats['games_played']
            
            if games > 0:
                win_rate = stats['wins'] / games * 100
                draw_rate = stats['draws'] / games * 100
                loss_rate = stats['losses'] / games * 100
                
                print(f"\n{opp.name} (Est. ELO: {opp.estimated_elo}, Style: {opp.style})")
                print(f"  Games: {games}")
                print(f"  Record: {stats['wins']}W - {stats['draws']}D - {stats['losses']}L")
                print(f"  Win Rate: {win_rate:.1f}%")
                print(f"  Avg Moves: {stats['avg_moves']:.1f}")
        
        print("="*70)


def create_opponent_pool() -> OpponentPool:
    """
    Create default opponent pool for v8.0 training
    
    Returns:
        OpponentPool with configured opponents
    """
    # Base path to tournament engines
    base_path = Path(r"E:\Programming Stuff\Chess Engines\Tournament Engines")
    
    opponents = [
        # Baseline opponents (should dominate these)
        OpponentConfig(
            name="Random Opponent v1.0",
            path=str(base_path / "Opponents" / "RandomOpponent_v1.0" / "random_opponent.py"),
            weight=0.1,
            estimated_elo=600,
            style="random"
        ),
        
        OpponentConfig(
            name="Material Opponent v1.0",
            path=str(base_path / "Opponents" / "MaterialOpponent_v1.0" / "material_opponent.py"),
            weight=0.15,
            estimated_elo=1100,
            style="tactical"
        ),
        
        # NOTE: Positional Opponent v1.0 removed - was making illegal moves
        
        # V7P3R versions (main training targets)
        OpponentConfig(
            name="V7P3R v17.1",
            path=str(base_path / "V7P3R" / "V7P3R_v17.1" / "src" / "v7p3r_uci.py"),
            weight=0.25,  # Increased from 0.2 to compensate for removed opponent
            estimated_elo=1700,
            style="balanced"
        ),
        
        OpponentConfig(
            name="V7P3R v17.8",
            path=str(base_path / "V7P3R" / "V7P3R_v17.8" / "src" / "v7p3r_uci.py"),
            weight=0.25,  # Increased from 0.2
            estimated_elo=1800,
            style="aggressive"
        ),
        
        OpponentConfig(
            name="V7P3R v18.3",
            path=str(base_path / "V7P3R" / "V7P3R_v18.3" / "src" / "v7p3r_uci.py"),
            weight=0.25,  # Increased from 0.2
            estimated_elo=1850,
            style="balanced"
        ),
    ]
    
    return OpponentPool(opponents)


if __name__ == "__main__":
    # Test opponent pool
    print("Testing Opponent Pool...")
    
    pool = create_opponent_pool()
    
    print(f"\nConfigured {len(pool.opponents)} opponents:")
    for opp in pool.opponents:
        print(f"  - {opp.name} (ELO: {opp.estimated_elo}, Weight: {opp.weight:.2f})")
    
    print("\nTesting opponent selection (10 samples):")
    for i in range(10):
        opp = pool.get_next_opponent(strategy="weighted_random")
        print(f"  {i+1}. {opp.name}")
