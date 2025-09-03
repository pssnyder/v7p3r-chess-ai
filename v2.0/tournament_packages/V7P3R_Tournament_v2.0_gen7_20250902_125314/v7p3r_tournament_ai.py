"""
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
