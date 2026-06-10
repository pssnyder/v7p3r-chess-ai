#!/usr/bin/env python3
"""
V7P3R AI v8.0 UCI Interface - Pure Learned Neural Network Engine

UCI-compatible interface for the v8.0 neural network trained through
pure self-play with learned reward shaping and opening book meta-actions.

Key Features:
- Neural network position evaluation (no hand-coded heuristics)
- Learned feature importance (mobility-focused as discovered)
- Opening book integration (100 variations)
- Tablebase oracle for perfect endgames
- Fast inference (52k+ positions/sec batch throughput)
"""

import sys
import time
import chess
import torch
import numpy as np
from pathlib import Path

# Import v8.0 components
from network import V8ValueNetwork
from comprehensive_features import ComprehensiveFeatureExtractor
from opening_selector import OpeningSelector
try:
    from tablebase_oracle import TablebaseOracle
    TABLEBASE_AVAILABLE = True
except:
    TABLEBASE_AVAILABLE = False


class V8NeuralEngine:
    """Neural network chess engine core"""
    
    def __init__(self, model_path='../training/v8_generational/gen_0010_value_network.pt'):
        """Initialize neural engine"""
        # Device selection
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load neural network
        self.network = V8ValueNetwork(input_dim=55).to(self.device)
        
        if Path(model_path).exists():
            self.network.load_state_dict(torch.load(model_path, map_location=self.device))
            self.network.eval()
            self.model_loaded = True
            self.model_path = model_path
        else:
            self.model_loaded = False
            self.model_path = None
        
        # Feature extractor
        self.feature_extractor = ComprehensiveFeatureExtractor()
        
        # Opening book
        try:
            self.opening_selector = OpeningSelector('opening_book.json')
            self.opening_book_loaded = True
        except:
            self.opening_selector = None
            self.opening_book_loaded = False
        
        # Tablebase oracle (optional)
        if TABLEBASE_AVAILABLE:
            tablebase_path = r'E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\pgn_training_data\pgn_data_endgames\3-4-5_pieces_Syzygy\3-4-5'
            if Path(tablebase_path).exists():
                self.tablebase = TablebaseOracle(tablebase_path)
                self.tablebase_loaded = True
            else:
                self.tablebase = None
                self.tablebase_loaded = False
        else:
            self.tablebase = None
            self.tablebase_loaded = False
        
        # Engine state
        self.move_number = 0
        self.last_inference_time_ms = 0.0
        
        # UCI options (default values)
        self.temperature = 0.1  # Low = deterministic, High = creative
        self.use_opening_book = True
        self.opening_book_depth = 10
        self.use_tablebase = True
        self.debug_mode = False
    
    def new_game(self):
        """Reset for new game"""
        self.move_number = 0
        self.last_inference_time_ms = 0.0
    
    def evaluate_position(self, board):
        """
        Evaluate position using neural network
        
        Returns:
            float: Evaluation in range [-1.0, +1.0]
        """
        features = self.feature_extractor.extract_all_features(
            board,
            move_number=self.move_number,
            previous_inference_ms=self.last_inference_time_ms
        )
        
        features_tensor = torch.tensor([features], dtype=torch.float32).to(self.device)
        
        start_time = time.time()
        with torch.no_grad():
            value = self.network(features_tensor).item()
        
        self.last_inference_time_ms = (time.time() - start_time) * 1000.0
        
        return value
    
    def search(self, board, time_limit=3.0):
        """
        Search for best move
        
        Args:
            board: Current position
            time_limit: Time limit in seconds (for future use)
        
        Returns:
            chess.Move: Best move found
        """
        # Check tablebase first (perfect knowledge)
        if self.use_tablebase and self.tablebase_loaded and self.tablebase.is_available(board):
            tb_move = self.tablebase.get_best_move(board)
            if tb_move:
                if self.debug_mode:
                    print("info string Using tablebase move", flush=True)
                return tb_move
        
        # Get all legal moves
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            return None
        
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        # Evaluate each move
        move_scores = []
        
        for move in legal_moves:
            board.push(move)
            
            # Evaluate from opponent's perspective (flip sign)
            score = -self.evaluate_position(board)
            
            board.pop()
            
            move_scores.append((move, score))
        
        # Select move based on temperature
        if self.temperature < 0.01:
            # Deterministic: pick best move
            best_move, best_score = max(move_scores, key=lambda x: x[1])
            
            if self.debug_mode:
                print(f"info string Best move: {best_move.uci()} (eval: {best_score:+.3f})", flush=True)
            
            return best_move
        else:
            # Stochastic: sample based on softmax probabilities
            scores = np.array([score for _, score in move_scores])
            
            # Apply softmax with temperature
            exp_scores = np.exp(scores / self.temperature)
            probs = exp_scores / np.sum(exp_scores)
            
            chosen_idx = np.random.choice(len(legal_moves), p=probs)
            chosen_move = move_scores[chosen_idx][0]
            
            if self.debug_mode:
                print(f"info string Sampled move: {chosen_move.uci()} (prob: {probs[chosen_idx]:.3f})", flush=True)
            
            return chosen_move


def main():
    """UCI interface main loop"""
    engine = V8NeuralEngine()
    board = chess.Board()
    
    while True:
        try:
            line = input().strip()
            if not line:
                continue
            
            parts = line.split()
            command = parts[0]
            
            if command == "quit":
                break
            
            elif command == "uci":
                print("id name V7P3R-AI v8.0")
                print("id author Pat Snyder")
                
                # Report model status
                if engine.model_loaded:
                    print(f"info string Model loaded: {engine.model_path}")
                else:
                    print("info string WARNING: No model loaded, using random weights")
                
                if engine.opening_book_loaded:
                    print(f"info string Opening book: {engine.opening_selector.num_openings} variations")
                else:
                    print("info string Opening book: Not loaded")
                
                if engine.tablebase_loaded:
                    print("info string Tablebase: 5-piece Syzygy available")
                else:
                    print("info string Tablebase: Not available")
                
                print(f"info string Device: {engine.device}")
                print(f"info string Network: V8ValueNetwork (56,449 parameters)")
                print("info string Training: 10 generations, 1000 games, pure self-play")
                
                # UCI options
                print("option name Temperature type spin default 10 min 0 max 100")
                print("option name UseOpeningBook type check default true")
                print("option name OpeningBookDepth type spin default 10 min 0 max 50")
                print("option name UseTablebase type check default true")
                print("option name Debug type check default false")
                
                print("uciok", flush=True)
            
            elif command == "setoption":
                if len(parts) >= 4 and parts[1] == "name":
                    option_name = " ".join(parts[2:parts.index("value")]) if "value" in parts else parts[2]
                    option_value = " ".join(parts[parts.index("value")+1:]) if "value" in parts else None
                    
                    if option_name == "Temperature" and option_value:
                        try:
                            engine.temperature = int(option_value) / 100.0
                            print(f"info string Temperature set to {engine.temperature:.2f}", flush=True)
                        except:
                            pass
                    
                    elif option_name == "UseOpeningBook" and option_value:
                        engine.use_opening_book = (option_value.lower() == "true")
                        print(f"info string Opening book {'enabled' if engine.use_opening_book else 'disabled'}", flush=True)
                    
                    elif option_name == "OpeningBookDepth" and option_value:
                        try:
                            engine.opening_book_depth = int(option_value)
                            print(f"info string Opening book depth set to {engine.opening_book_depth}", flush=True)
                        except:
                            pass
                    
                    elif option_name == "UseTablebase" and option_value:
                        engine.use_tablebase = (option_value.lower() == "true")
                        print(f"info string Tablebase {'enabled' if engine.use_tablebase else 'disabled'}", flush=True)
                    
                    elif option_name == "Debug" and option_value:
                        engine.debug_mode = (option_value.lower() == "true")
                        print(f"info string Debug mode {'enabled' if engine.debug_mode else 'disabled'}", flush=True)
            
            elif command == "isready":
                print("readyok", flush=True)
            
            elif command == "ucinewgame":
                board = chess.Board()
                engine.new_game()
            
            elif command == "position":
                if len(parts) > 1:
                    if parts[1] == "startpos":
                        board = chess.Board()
                        move_start = 2
                        if len(parts) > 2 and parts[2] == "moves":
                            move_start = 3
                    elif parts[1] == "fen":
                        fen_parts = parts[2:8]  # FEN has 6 parts
                        fen = " ".join(fen_parts)
                        board = chess.Board(fen)
                        move_start = 8
                        if len(parts) > 8 and parts[8] == "moves":
                            move_start = 9
                    
                    # Apply moves
                    if len(parts) > move_start:
                        for move_uci in parts[move_start:]:
                            try:
                                move = chess.Move.from_uci(move_uci)
                                if board.is_legal(move):
                                    board.push(move)
                                    engine.move_number += 1
                                else:
                                    break
                            except:
                                break
            
            elif command == "go":
                # Parse time controls
                time_limit = 3.0  # Default
                
                for i, part in enumerate(parts):
                    if part == "movetime" and i + 1 < len(parts):
                        try:
                            time_limit = int(parts[i + 1]) / 1000.0
                        except:
                            pass
                    
                    elif part == "wtime" and i + 1 < len(parts):
                        if board.turn == chess.WHITE:
                            try:
                                remaining_time = int(parts[i + 1]) / 1000.0
                                increment = 0.0
                                
                                # Check for increment
                                for j, p in enumerate(parts):
                                    if p == "winc" and j + 1 < len(parts):
                                        try:
                                            increment = int(parts[j + 1]) / 1000.0
                                        except:
                                            pass
                                
                                # Simple time management
                                # Neural network is FAST, so we can think a bit longer
                                moves_played = len(board.move_stack)
                                
                                if moves_played < 10:
                                    time_factor = 30.0  # Quick in opening
                                elif moves_played < 25:
                                    time_factor = 20.0  # More time in middlegame
                                elif moves_played < 40:
                                    time_factor = 15.0  # Even more in critical phase
                                else:
                                    time_factor = 20.0  # Moderate in endgame
                                
                                # Increment awareness
                                if increment > 0.5:
                                    effective_time = remaining_time + (increment * 10)
                                    calculated_time = effective_time / time_factor
                                else:
                                    calculated_time = remaining_time / time_factor
                                
                                # Hard caps based on remaining time
                                if remaining_time > 180:
                                    time_limit = min(calculated_time, 20.0)
                                elif remaining_time > 120:
                                    time_limit = min(calculated_time, 15.0)
                                elif remaining_time > 60:
                                    time_limit = min(calculated_time, 10.0)
                                elif remaining_time > 30:
                                    time_limit = min(calculated_time, 5.0)
                                else:
                                    time_limit = min(calculated_time, 2.0)
                                
                                # Safety cap
                                time_limit = min(time_limit, 30.0)
                            except:
                                pass
                    
                    elif part == "btime" and i + 1 < len(parts):
                        if board.turn == chess.BLACK:
                            try:
                                remaining_time = int(parts[i + 1]) / 1000.0
                                increment = 0.0
                                
                                # Check for increment
                                for j, p in enumerate(parts):
                                    if p == "binc" and j + 1 < len(parts):
                                        try:
                                            increment = int(parts[j + 1]) / 1000.0
                                        except:
                                            pass
                                
                                # Simple time management (same as white)
                                moves_played = len(board.move_stack)
                                
                                if moves_played < 10:
                                    time_factor = 30.0
                                elif moves_played < 25:
                                    time_factor = 20.0
                                elif moves_played < 40:
                                    time_factor = 15.0
                                else:
                                    time_factor = 20.0
                                
                                # Increment awareness
                                if increment > 0.5:
                                    effective_time = remaining_time + (increment * 10)
                                    calculated_time = effective_time / time_factor
                                else:
                                    calculated_time = remaining_time / time_factor
                                
                                # Hard caps based on remaining time
                                if remaining_time > 180:
                                    time_limit = min(calculated_time, 20.0)
                                elif remaining_time > 120:
                                    time_limit = min(calculated_time, 15.0)
                                elif remaining_time > 60:
                                    time_limit = min(calculated_time, 10.0)
                                elif remaining_time > 30:
                                    time_limit = min(calculated_time, 5.0)
                                else:
                                    time_limit = min(calculated_time, 2.0)
                                
                                # Safety cap
                                time_limit = min(time_limit, 30.0)
                            except:
                                pass
                
                # Search for best move
                best_move = engine.search(board, time_limit)
                
                if best_move:
                    print(f"bestmove {best_move.uci()}", flush=True)
                else:
                    # No legal moves (checkmate or stalemate)
                    print("bestmove 0000", flush=True)
        
        except (EOFError, KeyboardInterrupt):
            break
        except Exception as e:
            if engine.debug_mode:
                print(f"info string Error: {e}", flush=True)
            # Continue on errors during development


if __name__ == "__main__":
    main()
