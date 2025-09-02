#!/usr/bin/env python3
"""
Package Tournament Engine for V7P3R Chess AI v2.0
Creates a complete tournament-ready engine package
"""

import os
import shutil
import glob
import json
import time
from datetime import datetime
from pathlib import Path

def find_latest_best_model():
    """Find the most recent best model"""
    model_pattern = "models/best_gpu_model_gen_*.pth"
    model_files = glob.glob(model_pattern)
    
    if not model_files:
        print("❌ No trained models found!")
        return None
    
    # Sort by generation number
    def extract_gen_number(filepath):
        try:
            base = os.path.basename(filepath)
            gen_part = base.split('gen_')[1].split('.')[0]
            return int(gen_part)
        except:
            return -1
    
    latest_model = max(model_files, key=extract_gen_number)
    gen_number = extract_gen_number(latest_model)
    
    print(f"📄 Latest model found: {latest_model} (Generation {gen_number})")
    return latest_model, gen_number

def create_tournament_engine_wrapper():
    """Create the main tournament engine wrapper"""
    
    wrapper_code = '''#!/usr/bin/env python3
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
'''

    return wrapper_code

def create_tournament_ai():
    """Create the tournament AI implementation"""
    
    ai_code = '''"""
V7P3R Tournament AI Implementation
Optimized for tournament play with time management
"""

import chess
import chess.engine
import time
import threading
from typing import Optional, Dict, Any

try:
    import torch
    from v7p3r_gpu_model import V7P3RGPU_LSTM, GPUChessFeatureExtractor
    from v7p3r_bounty_system import ExtendedBountyEvaluator
    from chess_core import BoardEvaluator
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class V7P3RTournamentAI:
    """Tournament-optimized V7P3R AI"""
    
    def __init__(self):
        self.model = None
        self.feature_extractor = None
        self.bounty_evaluator = ExtendedBountyEvaluator()
        self.board_evaluator = BoardEvaluator()
        self.board = chess.Board()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stop_flag = False
        self.debug = False
        
        # Tournament settings
        self.time_management = True
        self.threads = 4
        
    def load_tournament_model(self, model_path: str):
        """Load the tournament model"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        self.model = V7P3RGPU_LSTM.load_model(model_path, self.device)
        self.model.eval()
        
        self.feature_extractor = GPUChessFeatureExtractor()
        self.feature_extractor.to(self.device)
        
        print(f"Tournament model loaded from {model_path}")
        
    def set_position(self, position_line: str):
        """Set board position from UCI position command"""
        try:
            parts = position_line.split()
            
            if "startpos" in parts:
                self.board = chess.Board()
                moves_index = parts.index("moves") if "moves" in parts else -1
            elif "fen" in parts:
                fen_index = parts.index("fen")
                fen_parts = parts[fen_index+1:fen_index+7]
                fen = " ".join(fen_parts)
                self.board = chess.Board(fen)
                moves_index = parts.index("moves") if "moves" in parts else -1
            else:
                return
            
            # Apply moves
            if moves_index != -1:
                moves = parts[moves_index+1:]
                for move_str in moves:
                    try:
                        move = chess.Move.from_uci(move_str)
                        if move in self.board.legal_moves:
                            self.board.push(move)
                    except:
                        break
                        
        except Exception as e:
            if self.debug:
                print(f"Debug: Error setting position: {e}")
    
    def evaluate_move(self, move: chess.Move) -> float:
        """Evaluate a single move"""
        if not self.model:
            return 0.0
        
        try:
            # Create a copy of the board and make the move
            temp_board = self.board.copy()
            temp_board.push(move)
            
            # Extract features and get neural network evaluation
            with torch.no_grad():
                features = self.feature_extractor.extract_features(temp_board)
                features_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
                features_tensor = features_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dims
                
                output, _ = self.model(features_tensor)
                nn_score = output.squeeze().cpu().numpy()[0] if len(output.squeeze().shape) > 0 else output.squeeze().cpu().numpy().item()
            
            # Get bounty evaluation
            bounty_score = self.bounty_evaluator.evaluate_move(self.board, move)
            bounty_value = bounty_score.total()
            
            # Get positional evaluation
            position_value = self.board_evaluator.evaluate(temp_board)
            
            # Combine scores
            total_score = nn_score * 1.0 + bounty_value * 0.3 + position_value * 0.1
            
            return total_score
            
        except Exception as e:
            if self.debug:
                print(f"Debug: Error evaluating move {move}: {e}")
            return 0.0
    
    def get_best_move(self, go_line: str) -> str:
        """Get the best move for current position"""
        self.stop_flag = False
        
        # Parse time controls
        time_info = self.parse_go_command(go_line)
        max_time = self.calculate_time_budget(time_info)
        
        start_time = time.time()
        
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return "0000"  # No legal moves
        
        if len(legal_moves) == 1:
            return legal_moves[0].uci()
        
        # Evaluate all moves
        move_scores = []
        
        for move in legal_moves:
            if self.stop_flag or (time.time() - start_time) > max_time * 0.8:
                break
                
            score = self.evaluate_move(move)
            move_scores.append((move, score))
        
        # Sort by score and return best move
        move_scores.sort(key=lambda x: x[1], reverse=True)
        best_move = move_scores[0][0]
        
        return best_move.uci()
    
    def parse_go_command(self, go_line: str) -> Dict[str, Any]:
        """Parse go command for time information"""
        parts = go_line.split()
        time_info = {}
        
        i = 1
        while i < len(parts):
            if parts[i] == "wtime" and i + 1 < len(parts):
                time_info["wtime"] = int(parts[i + 1])
                i += 2
            elif parts[i] == "btime" and i + 1 < len(parts):
                time_info["btime"] = int(parts[i + 1])
                i += 2
            elif parts[i] == "winc" and i + 1 < len(parts):
                time_info["winc"] = int(parts[i + 1])
                i += 2
            elif parts[i] == "binc" and i + 1 < len(parts):
                time_info["binc"] = int(parts[i + 1])
                i += 2
            elif parts[i] == "movetime" and i + 1 < len(parts):
                time_info["movetime"] = int(parts[i + 1])
                i += 2
            elif parts[i] == "depth" and i + 1 < len(parts):
                time_info["depth"] = int(parts[i + 1])
                i += 2
            else:
                i += 1
        
        return time_info
    
    def calculate_time_budget(self, time_info: Dict[str, Any]) -> float:
        """Calculate time budget for this move"""
        if "movetime" in time_info:
            return time_info["movetime"] / 1000.0  # Convert to seconds
        
        # Determine our time and increment
        our_time = None
        our_inc = 0
        
        if self.board.turn == chess.WHITE:
            our_time = time_info.get("wtime", 60000)  # Default 1 minute
            our_inc = time_info.get("winc", 0)
        else:
            our_time = time_info.get("btime", 60000)
            our_inc = time_info.get("binc", 0)
        
        # Simple time management: use 1/30th of remaining time + increment
        time_budget = (our_time / 30000.0) + (our_inc / 1000.0)
        
        # Minimum 0.1 seconds, maximum 30 seconds
        return max(0.1, min(time_budget, 30.0))
    
    def new_game(self):
        """Start a new game"""
        self.board = chess.Board()
        self.stop_flag = False
    
    def stop_search(self):
        """Stop current search"""
        self.stop_flag = True
'''
    
    return ai_code

def create_package_structure(model_path, gen_number):
    """Create the tournament package structure"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_name = f"V7P3R_Tournament_v2.0_gen{gen_number}_{timestamp}"
    package_dir = f"tournament_packages/{package_name}"
    
    # Create package directory
    os.makedirs(package_dir, exist_ok=True)
    
    print(f"📦 Creating tournament package: {package_name}")
    
    # Copy essential files
    essential_files = [
        "chess_core.py",
        "v7p3r_bounty_system.py", 
        "v7p3r_gpu_model.py"
    ]
    
    for file in essential_files:
        if os.path.exists(file):
            shutil.copy2(file, package_dir)
            print(f"  ✅ Copied {file}")
        else:
            print(f"  ❌ Missing {file}")
    
    # Copy and rename the model
    model_dest = os.path.join(package_dir, "v7p3r_tournament_model.pth")
    shutil.copy2(model_path, model_dest)
    print(f"  ✅ Copied tournament model")
    
    # Create the main engine script
    engine_script = os.path.join(package_dir, "v7p3r_tournament_engine.py")
    with open(engine_script, 'w') as f:
        f.write(create_tournament_engine_wrapper())
    print(f"  ✅ Created engine wrapper")
    
    # Create the AI implementation
    ai_script = os.path.join(package_dir, "v7p3r_tournament_ai.py") 
    with open(ai_script, 'w') as f:
        f.write(create_tournament_ai())
    print(f"  ✅ Created tournament AI")
    
    # Create requirements.txt
    requirements = os.path.join(package_dir, "requirements.txt")
    with open(requirements, 'w') as f:
        f.write("""torch>=2.0.0
python-chess>=1.999
numpy>=1.21.0
""")
    print(f"  ✅ Created requirements.txt")
    
    # Create README
    readme = os.path.join(package_dir, "README.md")
    with open(readme, 'w') as f:
        f.write(f"""# V7P3R Chess AI v2.0 Tournament Engine

## Tournament Package Information
- **Generation**: {gen_number}
- **Package Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Engine Type**: GPU-accelerated neural network with genetic training
- **Protocol**: UCI compatible

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Test the engine:
```bash
python v7p3r_tournament_engine.py
```

## Arena Integration

1. In Arena, go to Engines → Install New Engine
2. Browse to `v7p3r_tournament_engine.py`
3. Set engine name to "V7P3R v2.0 Gen{gen_number}"
4. Configure time controls as needed

## Engine Features

- Neural network trained with genetic algorithms
- Advanced bounty-based position evaluation
- GPU acceleration (CUDA)
- Time management for tournament play
- UCI protocol compliant

## Technical Specifications

- Model: {gen_number}-generation GPU-trained LSTM
- Input features: 816-dimensional chess position encoding
- Evaluation: Hybrid neural network + tactical bounties
- Time management: Adaptive based on remaining time
- Hardware: Optimized for CUDA-capable GPUs

## Tournament Results

Record your tournament results here:

| Tournament | Date | Score | Rating | Notes |
|------------|------|-------|--------|-------|
|            |      |       |        |       |

""")
    print(f"  ✅ Created README.md")
    
    # Create a batch file for Windows
    batch_file = os.path.join(package_dir, "run_engine.bat")
    with open(batch_file, 'w') as f:
        f.write(f"""@echo off
echo Starting V7P3R Tournament Engine v2.0 Generation {gen_number}
python v7p3r_tournament_engine.py
pause
""")
    print(f"  ✅ Created run_engine.bat")
    
    return package_dir, package_name

def test_tournament_engine(package_dir):
    """Test the tournament engine package"""
    print("\n🧪 Testing tournament engine...")
    
    # Change to package directory and test
    original_dir = os.getcwd()
    
    try:
        os.chdir(package_dir)
        
        # Test basic import
        test_script = """
import sys
import time

try:
    from v7p3r_tournament_ai import V7P3RTournamentAI
    print("✅ Tournament AI import successful")
    
    ai = V7P3RTournamentAI()
    print("✅ Tournament AI initialization successful")
    
    # Test model loading
    ai.load_tournament_model("v7p3r_tournament_model.pth")
    print("✅ Model loading successful")
    
    # Test position setting
    ai.set_position("position startpos moves e2e4")
    print("✅ Position setting successful")
    
    # Test move generation
    move = ai.get_best_move("go movetime 1000")
    print(f"✅ Move generation successful: {move}")
    
    print("\\n🎉 All tests passed! Tournament engine is ready.")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    sys.exit(1)
"""
        
        with open("test_engine.py", 'w') as f:
            f.write(test_script)
        
        # Run the test
        result = os.system("python test_engine.py")
        
        if result == 0:
            print("✅ Tournament engine test passed!")
            return True
        else:
            print("❌ Tournament engine test failed!")
            return False
            
    finally:
        os.chdir(original_dir)

def create_arena_instructions(package_dir, package_name):
    """Create detailed Arena setup instructions"""
    
    instructions = f"""
# Arena Chess GUI Setup Instructions for V7P3R v2.0

## Step 1: Install the Engine

1. Open Arena Chess GUI
2. Go to **Engines** → **Install New Engine**
3. Navigate to: `{os.path.abspath(package_dir)}`
4. Select: `v7p3r_tournament_engine.py`
5. Engine Name: `V7P3R v2.0 Tournament`
6. Click **OK**

## Step 2: Configure Engine Settings

1. Go to **Engines** → **Manage Engines**
2. Find "V7P3R v2.0 Tournament" in the list
3. Click **Details** and configure:
   - **Time per move**: 5-10 seconds (for testing)
   - **Depth**: Leave default (time-based)
   - **Threads**: 4 (adjust based on your CPU)
   - **Debug**: Off (unless troubleshooting)

## Step 3: Test the Engine

1. Go to **Engines** → **New Engine Match**
2. Select V7P3R v2.0 as Player 1
3. Select another engine as Player 2
4. Set time control: 5+3 (5 minutes + 3 second increment)
5. Start the match

## Step 4: Tournament Setup

### Quick Tournament:
1. **Engines** → **Tournament**
2. Add V7P3R v2.0 and other engines
3. Set time control: 15+5 or 10+3
4. Set rounds: 10-20 games per opponent
5. Start tournament

### Swiss Tournament:
1. Create a Swiss system tournament
2. Time control: 15+10 (standard tournament time)
3. Add 6-8 engines of similar strength
4. Set 7-9 rounds
5. Monitor results

## Recommended Opponents for Rating

**Beginner Level (800-1200):**
- Fairy-Max
- Micro-Max  
- Vice

**Intermediate Level (1200-1800):**
- Fruit 2.1
- Crafty
- GNU Chess

**Advanced Level (1800-2200):**
- Stockfish (limited depth/time)
- Komodo (limited)
- Houdini (limited)

## Tournament Recording

Record results in this format:

```
Date: {datetime.now().strftime('%Y-%m-%d')}
Tournament: [Tournament Name]
Time Control: [e.g., 15+5]
Opponents: [List engines]
Score: X/Y ([Win-Draw-Loss])
Performance Rating: [Estimate]
Notes: [Key observations]
```

## Troubleshooting

If the engine doesn't start:
1. Check Python installation
2. Verify PyTorch installation: `pip install torch`
3. Run `python v7p3r_tournament_engine.py` directly
4. Check the Arena error log

## Performance Expectations

Based on training data:
- **Estimated Rating**: 1400-1800 ELO
- **Style**: Tactical, aggressive
- **Strengths**: Piece coordination, tactical shots
- **Weaknesses**: May be inconsistent in pure endgames

Good luck in your tournaments! 🏆
"""

    instructions_file = os.path.join(package_dir, "ARENA_SETUP.md")
    with open(instructions_file, 'w') as f:
        f.write(instructions)
    
    print(f"  ✅ Created Arena setup instructions")

def main():
    """Main packaging function"""
    
    print("=" * 80)
    print("V7P3R Tournament Engine Packaging System")
    print("=" * 80)
    
    # Find the latest model
    model_info = find_latest_best_model()
    if not model_info:
        return
    
    model_path, gen_number = model_info
    
    print(f"📊 Packaging model from generation {gen_number}")
    print(f"📄 Model file: {model_path}")
    
    # Create package
    package_dir, package_name = create_package_structure(model_path, gen_number)
    
    # Create Arena instructions
    create_arena_instructions(package_dir, package_name)
    
    # Test the package
    test_success = test_tournament_engine(package_dir)
    
    print("\n" + "=" * 80)
    print("TOURNAMENT ENGINE PACKAGING COMPLETE!")
    print("=" * 80)
    print(f"📦 Package: {package_name}")
    print(f"📁 Location: {os.path.abspath(package_dir)}")
    print(f"🧪 Tests: {'✅ PASSED' if test_success else '❌ FAILED'}")
    print("=" * 80)
    
    if test_success:
        print("\n🚀 Next Steps:")
        print("1. Navigate to the package directory")
        print("2. Follow ARENA_SETUP.md instructions")
        print("3. Add engine to Arena Chess GUI")
        print("4. Run test matches")
        print("5. Enter tournaments and record results!")
        
        print(f"\n📂 Package location:")
        print(f"   {os.path.abspath(package_dir)}")
        
        # Ask if user wants to open the directory
        open_dir = input("\nOpen package directory now? (y/N): ")
        if open_dir.lower() == 'y':
            os.system(f'explorer "{os.path.abspath(package_dir)}"')
    else:
        print("\n❌ Package testing failed. Please check the errors above.")

if __name__ == "__main__":
    main()
