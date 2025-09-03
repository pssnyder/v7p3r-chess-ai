#!/usr/bin/env python3
"""
V7P3R Chess AI v2.0 Tournament Engine
UCI-compatible chess engine for tournament play
"""

import sys
import os
import threading
import time
from pathlib import Path

# Add the engine directory to Python path
engine_dir = Path(__file__).parent
sys.path.insert(0, str(engine_dir))

try:
    import torch
    from v7p3r_tournament_ai import V7P3RTournamentAI
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"Error: Required dependencies not available: {e}")
    print("Please ensure PyTorch and python-chess are installed")
    sys.exit(1)

class V7P3RTournamentEngine:
    """Tournament UCI Engine"""
    
    def __init__(self):
        self.ai = None
        self.model_loaded = False
        self.debug = False
        
    def load_model(self):
        """Load the tournament model"""
        try:
            model_path = engine_dir / "v7p3r_tournament_model.pth"
            if not model_path.exists():
                print("Error: Tournament model not found!")
                return False
                
            self.ai = V7P3RTournamentAI()
            self.ai.load_tournament_model(str(model_path))
            self.model_loaded = True
            print("Tournament model loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def uci_loop(self):
        """Main UCI command loop"""
        
        while True:
            try:
                line = input().strip()
                
                if self.debug:
                    print(f"Debug: Received command: {line}")
                
                if line == "uci":
                    self.handle_uci()
                elif line == "isready":
                    self.handle_isready()
                elif line == "quit":
                    break
                elif line.startswith("position"):
                    self.handle_position(line)
                elif line.startswith("go"):
                    self.handle_go(line)
                elif line == "ucinewgame":
                    self.handle_newgame()
                elif line.startswith("setoption"):
                    self.handle_setoption(line)
                elif line == "stop":
                    self.handle_stop()
                else:
                    if self.debug:
                        print(f"Debug: Unknown command: {line}")
                        
            except EOFError:
                break
            except Exception as e:
                if self.debug:
                    print(f"Debug: Error in UCI loop: {e}")
    
    def handle_uci(self):
        """Handle UCI identification"""
        print("id name V7P3R Chess AI v2.0")
        print("id author V7P3R Development Team")
        print("option name Debug type check default false")
        print("option name Threads type spin default 4 min 1 max 16")
        print("option name Time_Management type check default true")
        print("uciok")
    
    def handle_isready(self):
        """Handle isready command"""
        if not self.model_loaded:
            self.load_model()
        print("readyok")
    
    def handle_position(self, line):
        """Handle position command"""
        if not self.model_loaded:
            self.load_model()
            
        if self.ai:
            self.ai.set_position(line)
    
    def handle_go(self, line):
        """Handle go command"""
        if not self.model_loaded or not self.ai:
            print("bestmove e2e4")  # Fallback move
            return
            
        try:
            best_move = self.ai.get_best_move(line)
            print(f"bestmove {best_move}")
        except Exception as e:
            if self.debug:
                print(f"Debug: Error getting best move: {e}")
            print("bestmove e2e4")  # Fallback move
    
    def handle_newgame(self):
        """Handle new game"""
        if self.ai:
            self.ai.new_game()
    
    def handle_setoption(self, line):
        """Handle setoption command"""
        if "name Debug" in line and "value true" in line:
            self.debug = True
            print("Debug mode enabled")
        elif "name Debug" in line and "value false" in line:
            self.debug = False
    
    def handle_stop(self):
        """Handle stop command"""
        if self.ai:
            self.ai.stop_search()

def main():
    """Main entry point"""
    engine = V7P3RTournamentEngine()
    engine.uci_loop()

if __name__ == "__main__":
    main()
