"""
V7P3RAI V3.0 UCI Interface
=========================
Tournament-ready UCI chess engine wrapper for V7P3R AI V3.0
"""

import sys
import threading
import time
from typing import Optional, Dict, Any
from pathlib import Path

# Add V3.0 system paths
sys.path.insert(0, str(Path(__file__).parent / "v3.0" / "src"))

class V7P3RAIUCI:
    """UCI interface for V7P3R AI V3.0 tournament engine"""
    
    def __init__(self):
        self.engine_name = "V7P3RAI v3.0"
        self.author = "V7P3R Team"
        self.thinking_brain = None
        self.gameplay_brain = None
        self.current_position = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self.move_history = []
        self.is_thinking = False
        self.stop_thinking = False
        
        # UCI options
        self.options = {
            "Hash": {"type": "spin", "default": 128, "min": 1, "max": 1024},
            "Threads": {"type": "spin", "default": 1, "min": 1, "max": 8},
            "CUDA_Enabled": {"type": "check", "default": True},
            "Puzzle_Mode": {"type": "check", "default": True},
            "Aggression_Level": {"type": "spin", "default": 5, "min": 1, "max": 10},
            "Time_Management": {"type": "combo", "default": "Balanced", "var": ["Conservative", "Balanced", "Aggressive"]}
        }
        
        self.setup_ai_systems()
    
    def setup_ai_systems(self):
        """Initialize the V7P3R two-brain system"""
        try:
            # Import and initialize thinking brain
            from ai.thinking_brain import ThinkingBrain
            from ai.gameplay_brain import GameplayBrain
            
            self.thinking_brain = ThinkingBrain()
            self.gameplay_brain = GameplayBrain()
            
            # Load trained models
            self.load_trained_models()
            
        except Exception as e:
            self.log_error(f"Failed to initialize AI systems: {e}")
    
    def load_trained_models(self):
        """Load the intensively trained V3.0 models"""
        try:
            # Load thinking brain weights
            model_path = Path("models/v7p3r_model.pkl")
            if model_path.exists():
                self.thinking_brain.load_model(str(model_path))
                self.log_info("Thinking brain model loaded successfully")
            
            # Load gameplay brain configuration
            # (Genetic algorithm parameters from training)
            
        except Exception as e:
            self.log_error(f"Failed to load trained models: {e}")
    
    def log_info(self, message: str):
        """Log information message"""
        with open("logs/engine_log.txt", "a") as f:
            f.write(f"[INFO] {time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    
    def log_error(self, message: str):
        """Log error message"""
        with open("logs/engine_log.txt", "a") as f:
            f.write(f"[ERROR] {time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    
    def send_response(self, response: str):
        """Send response to UCI GUI"""
        print(response, flush=True)
        self.log_info(f"Sent: {response}")
    
    def handle_uci(self):
        """Handle UCI command"""
        self.send_response(f"id name {self.engine_name}")
        self.send_response(f"id author {self.author}")
        
        # Send options
        for name, config in self.options.items():
            if config["type"] == "spin":
                self.send_response(f"option name {name} type spin default {config['default']} min {config['min']} max {config['max']}")
            elif config["type"] == "check":
                self.send_response(f"option name {name} type check default {'true' if config['default'] else 'false'}")
            elif config["type"] == "combo":
                var_str = " ".join([f"var {v}" for v in config['var']])
                self.send_response(f"option name {name} type combo default {config['default']} {var_str}")
        
        self.send_response("uciok")
    
    def handle_isready(self):
        """Handle isready command"""
        # Verify AI systems are ready
        if self.thinking_brain and self.gameplay_brain:
            self.send_response("readyok")
        else:
            # Try to reinitialize
            self.setup_ai_systems()
            if self.thinking_brain and self.gameplay_brain:
                self.send_response("readyok")
            else:
                self.send_response("readyok")  # Respond anyway to avoid hanging
    
    def handle_position(self, args: list):
        """Handle position command"""
        if not args:
            return
        
        if args[0] == "startpos":
            self.current_position = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            self.move_history = []
            
            # Check for moves
            if len(args) > 1 and args[1] == "moves":
                self.move_history = args[2:]
        
        elif args[0] == "fen":
            # Find where FEN ends and moves begin
            fen_parts = []
            moves_start = -1
            
            for i, arg in enumerate(args[1:], 1):
                if arg == "moves":
                    moves_start = i + 1
                    break
                fen_parts.append(arg)
            
            self.current_position = " ".join(fen_parts)
            
            if moves_start > 0:
                self.move_history = args[moves_start:]
            else:
                self.move_history = []
    
    def handle_go(self, args: list):
        """Handle go command"""
        # Parse go arguments
        go_params = self.parse_go_args(args)
        
        # Start thinking in separate thread
        self.stop_thinking = False
        thinking_thread = threading.Thread(target=self.think_and_respond, args=(go_params,))
        thinking_thread.daemon = True
        thinking_thread.start()
    
    def parse_go_args(self, args: list) -> Dict[str, Any]:
        """Parse go command arguments"""
        params = {}
        i = 0
        
        while i < len(args):
            arg = args[i]
            
            if arg == "depth" and i + 1 < len(args):
                params["depth"] = int(args[i + 1])
                i += 2
            elif arg == "movetime" and i + 1 < len(args):
                params["movetime"] = int(args[i + 1])
                i += 2
            elif arg == "wtime" and i + 1 < len(args):
                params["wtime"] = int(args[i + 1])
                i += 2
            elif arg == "btime" and i + 1 < len(args):
                params["btime"] = int(args[i + 1])
                i += 2
            elif arg == "winc" and i + 1 < len(args):
                params["winc"] = int(args[i + 1])
                i += 2
            elif arg == "binc" and i + 1 < len(args):
                params["binc"] = int(args[i + 1])
                i += 2
            elif arg == "infinite":
                params["infinite"] = True
                i += 1
            else:
                i += 1
        
        return params
    
    def think_and_respond(self, go_params: Dict[str, Any]):
        """Think and send best move"""
        self.is_thinking = True
        
        try:
            # Calculate thinking time
            think_time = self.calculate_think_time(go_params)
            
            # Get best move from V7P3R AI
            best_move = self.get_best_move(think_time)
            
            if not self.stop_thinking:
                self.send_response(f"bestmove {best_move}")
                
        except Exception as e:
            self.log_error(f"Error during thinking: {e}")
            # Send a basic move to avoid hanging
            self.send_response("bestmove e2e4")
        
        finally:
            self.is_thinking = False
    
    def calculate_think_time(self, go_params: Dict[str, Any]) -> float:
        """Calculate how long to think"""
        if "movetime" in go_params:
            return go_params["movetime"] / 1000.0  # Convert to seconds
        
        if "depth" in go_params:
            # Use depth-based time (deeper = more time)
            return min(go_params["depth"] * 0.5, 10.0)
        
        if "wtime" in go_params or "btime" in go_params:
            # Simple time management
            remaining_time = go_params.get("wtime", go_params.get("btime", 30000))
            increment = go_params.get("winc", go_params.get("binc", 0))
            
            # Use 1/40 of remaining time plus increment
            think_time = (remaining_time / 40000.0) + (increment / 1000.0)
            return max(0.1, min(think_time, 30.0))
        
        # Default thinking time
        return 2.0
    
    def get_best_move(self, think_time: float) -> str:
        """Get best move from V7P3R AI system"""
        try:
            # Convert position and moves to board state
            from chess_core import ChessGame
            
            game = ChessGame()
            
            # Apply move history
            for move_str in self.move_history:
                try:
                    game.make_move_from_string(move_str)
                except:
                    self.log_error(f"Invalid move in history: {move_str}")
                    break
            
            # Get current board state
            current_fen = game.get_fen()
            
            # Use V7P3R two-brain system to find best move
            if self.thinking_brain and self.gameplay_brain:
                # Thinking brain analysis
                position_evaluation = self.thinking_brain.evaluate_position(current_fen)
                
                # Gameplay brain move generation
                candidate_moves = self.gameplay_brain.generate_moves(current_fen)
                
                # Combine analysis (simplified for UCI)
                if candidate_moves:
                    best_move = candidate_moves[0]  # Take top candidate
                    
                    # Validate move format (e.g., "e2e4", "e7e8q")
                    if len(best_move) >= 4:
                        return best_move
            
            # Fallback: use basic chess logic
            legal_moves = game.get_legal_moves()
            if legal_moves:
                return legal_moves[0]  # First legal move
            
            return "0000"  # Null move if no legal moves
            
        except Exception as e:
            self.log_error(f"Error getting best move: {e}")
            return "e2e4"  # Safe fallback
    
    def handle_stop(self):
        """Handle stop command"""
        self.stop_thinking = True
        # The thinking thread will send bestmove when it detects stop
    
    def handle_quit(self):
        """Handle quit command"""
        self.stop_thinking = True
        sys.exit(0)
    
    def handle_setoption(self, args: list):
        """Handle setoption command"""
        if len(args) >= 4 and args[0] == "name" and args[2] == "value":
            option_name = args[1]
            option_value = args[3]
            
            if option_name in self.options:
                self.log_info(f"Setting option {option_name} = {option_value}")
                # Apply option changes here
                # For now, just log them
    
    def run(self):
        """Main UCI loop"""
        self.log_info("V7P3RAI v3.0 UCI engine started")
        
        while True:
            try:
                line = input().strip()
                if not line:
                    continue
                
                self.log_info(f"Received: {line}")
                parts = line.split()
                command = parts[0].lower()
                args = parts[1:] if len(parts) > 1 else []
                
                if command == "uci":
                    self.handle_uci()
                elif command == "isready":
                    self.handle_isready()
                elif command == "position":
                    self.handle_position(args)
                elif command == "go":
                    self.handle_go(args)
                elif command == "stop":
                    self.handle_stop()
                elif command == "quit":
                    self.handle_quit()
                elif command == "setoption":
                    self.handle_setoption(args)
                else:
                    self.log_info(f"Unknown command: {command}")
                    
            except EOFError:
                break
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.log_error(f"Error in main loop: {e}")

def main():
    """Entry point for V7P3RAI UCI engine"""
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Start UCI engine
    engine = V7P3RAIUCI()
    engine.run()

if __name__ == "__main__":
    main()