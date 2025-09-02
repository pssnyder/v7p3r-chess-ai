# v7p3r_ai_v2.py
"""
V7P3R Chess AI 2.0 - Main AI Implementation
Integrates RNN model, bounty system, and genetic training for powerful chess AI.
"""

import chess
import chess.engine
import numpy as np
import sys
import os
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from v7p3r_rnn_model import V7P3RNeuralNetwork, ChessFeatureExtractor
from v7p3r_bounty_system import ExtendedBountyEvaluator, BountyScore
from chess_core import ChessConfig


class V7P3RAI_v2:
    """V7P3R Chess AI 2.0 - RNN-based with bounty evaluation"""
    
    def __init__(self, model_path: Optional[str] = None, config_path: str = "config.json"):
        self.config = ChessConfig(config_path)
        self.feature_extractor = ChessFeatureExtractor()
        self.bounty_evaluator = ExtendedBountyEvaluator()
        
        # Load or create neural network
        if model_path and os.path.exists(model_path):
            self.network = V7P3RNeuralNetwork.load_model(model_path)
            print(f"Loaded RNN model from {model_path}")
        else:
            self.network = V7P3RNeuralNetwork()
            print("Created new RNN model")
        
        # Game state tracking
        self.move_history: List[chess.Move] = []
        self.position_history: List[str] = []
        self.game_phase = "opening"
        
        # Performance settings
        v7p3r_config = self.config.get_v7p3r_config()
        self.search_depth = v7p3r_config.get("search_depth", 3)
        self.time_per_move = v7p3r_config.get("time_per_move", 5.0)
        self.use_bounty_guidance = v7p3r_config.get("use_bounty_guidance", True)
        self.bounty_weight = v7p3r_config.get("bounty_weight", 0.3)
        self.rnn_weight = v7p3r_config.get("rnn_weight", 0.7)
        
        print(f"V7P3R AI 2.0 initialized - RNN weight: {self.rnn_weight}, Bounty weight: {self.bounty_weight}")
    
    def get_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Get the best move using RNN + bounty evaluation"""
        if board.is_game_over():
            return None
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        if len(legal_moves) == 1:
            move = legal_moves[0]
            self._update_game_state(board, move)
            return move
        
        # Update game phase
        self._update_game_phase(board)
        
        # Get best move using hybrid evaluation
        best_move = self._evaluate_moves(board, legal_moves)
        
        if best_move:
            self._update_game_state(board, best_move)
        
        return best_move
    
    def _evaluate_moves(self, board: chess.Board, legal_moves: List[chess.Move]) -> Optional[chess.Move]:
        """Evaluate all legal moves using RNN + bounty system"""
        move_evaluations = []
        
        # Extract current position features
        current_features = self.feature_extractor.extract_features(board, self.move_history)
        
        for move in legal_moves:
            # Create position after move
            temp_board = board.copy()
            temp_board.push(move)
            
            # Get RNN evaluation
            move_features = self.feature_extractor.extract_features(
                temp_board, self.move_history + [move]
            )
            rnn_score = self.network.forward(move_features)
            
            # Get bounty evaluation
            bounty_score = 0.0
            if self.use_bounty_guidance:
                bounty = self.bounty_evaluator.evaluate_move(board, move)
                bounty_score = bounty.total()
            
            # Combine scores based on game phase
            combined_score = self._combine_scores(rnn_score, bounty_score)
            
            move_evaluations.append((move, combined_score, rnn_score, bounty_score))
        
        # Sort by combined score
        move_evaluations.sort(key=lambda x: x[1], reverse=True)
        
        # Add some exploration in non-critical positions
        if not board.is_check() and len(move_evaluations) > 3:
            # 10% chance to pick from top 3 moves instead of just the best
            if np.random.random() < 0.1:
                top_moves = move_evaluations[:3]
                selected = np.random.choice(len(top_moves))
                return top_moves[selected][0]
        
        return move_evaluations[0][0] if move_evaluations else None
    
    def _combine_scores(self, rnn_score: float, bounty_score: float) -> float:
        """Combine RNN and bounty scores based on game phase"""
        if self.game_phase == "opening":
            # In opening, favor bounty system for known good principles
            return 0.4 * rnn_score + 0.6 * bounty_score
        elif self.game_phase == "middlegame":
            # In middlegame, balance both systems
            return self.rnn_weight * rnn_score + self.bounty_weight * bounty_score
        else:  # endgame
            # In endgame, favor RNN for precise calculation
            return 0.8 * rnn_score + 0.2 * bounty_score
    
    def _update_game_phase(self, board: chess.Board):
        """Update current game phase based on position"""
        # Count material
        total_pieces = len([sq for sq in chess.SQUARES if board.piece_at(sq)])
        queens = len(list(board.pieces(chess.QUEEN, chess.WHITE))) + len(list(board.pieces(chess.QUEEN, chess.BLACK)))
        
        if len(self.move_history) < 15:
            self.game_phase = "opening"
        elif total_pieces <= 12 or queens == 0:
            self.game_phase = "endgame"
        else:
            self.game_phase = "middlegame"
    
    def _update_game_state(self, board: chess.Board, move: chess.Move):
        """Update internal game state after move"""
        self.move_history.append(move)
        self.position_history.append(board.fen())
        
        # Keep history manageable
        if len(self.move_history) > 100:
            self.move_history = self.move_history[-80:]
            self.position_history = self.position_history[-80:]
    
    def reset_game(self):
        """Reset for new game"""
        self.network.reset_memory()
        self.move_history = []
        self.position_history = []
        self.game_phase = "opening"
    
    def get_position_evaluation(self, board: chess.Board) -> Dict[str, Any]:
        """Get detailed position evaluation"""
        features = self.feature_extractor.extract_features(board, self.move_history)
        rnn_eval = self.network.forward(features)
        bounty_eval = self.bounty_evaluator.evaluate_position(board)
        
        return {
            'rnn_evaluation': float(rnn_eval),
            'bounty_evaluation': float(bounty_eval),
            'combined_evaluation': self._combine_scores(rnn_eval, bounty_eval),
            'game_phase': self.game_phase,
            'moves_played': len(self.move_history)
        }
    
    def analyze_move(self, board: chess.Board, move: chess.Move) -> Dict[str, Any]:
        """Analyze a specific move in detail"""
        if move not in board.legal_moves:
            return {'error': 'Illegal move'}
        
        # Get bounty breakdown
        bounty = self.bounty_evaluator.evaluate_move(board, move)
        
        # Get RNN evaluation
        temp_board = board.copy()
        temp_board.push(move)
        move_features = self.feature_extractor.extract_features(
            temp_board, self.move_history + [move]
        )
        rnn_score = self.network.forward(move_features)
        
        return {
            'move': move.uci(),
            'rnn_score': float(rnn_score),
            'bounty_breakdown': {
                'center_control': bounty.center_control,
                'piece_value': bounty.piece_value,
                'attack_defense': bounty.attack_defense,
                'king_safety': bounty.king_safety,
                'tactical_patterns': bounty.tactical_patterns,
                'mate_threats': bounty.mate_threats,
                'piece_coordination': bounty.piece_coordination,
                'castling': bounty.castling,
                'positional': bounty.positional,
                'total': bounty.total()
            },
            'combined_score': self._combine_scores(rnn_score, bounty.total()),
            'game_phase': self.game_phase
        }


class UCIHandler_v2:
    """UCI protocol handler for V7P3R Chess AI 2.0"""
    
    def __init__(self, ai: V7P3RAI_v2):
        self.ai = ai
        self.board = chess.Board()
        self.running = True
        self.debug = False
    
    def run(self):
        """Main UCI loop"""
        while self.running:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                
                line = line.strip()
                self._handle_command(line)
                
            except Exception as e:
                if self.debug:
                    print(f"info string UCI error: {e}")
                sys.stdout.flush()
    
    def _handle_command(self, line: str):
        """Handle UCI command"""
        if line == "uci":
            print("id name V7P3R Chess AI 2.0")
            print("id author pssnyder")
            print("option name Debug type check default false")
            print("option name BountyWeight type spin default 30 min 0 max 100")
            print("option name SearchDepth type spin default 3 min 1 max 6")
            print("uciok")
            
        elif line == "isready":
            print("readyok")
            
        elif line.startswith("setoption"):
            self._handle_option(line)
            
        elif line == "ucinewgame":
            self.board.reset()
            self.ai.reset_game()
            
        elif line.startswith("position"):
            self._handle_position(line)
            
        elif line.startswith("go"):
            self._handle_go(line)
            
        elif line == "quit":
            self.running = False
            
        elif line == "stop":
            pass  # Handle stop if needed
        
        sys.stdout.flush()
    
    def _handle_option(self, line: str):
        """Handle setoption command"""
        parts = line.split()
        if "name" in parts and "value" in parts:
            name_idx = parts.index("name") + 1
            value_idx = parts.index("value") + 1
            
            if name_idx < len(parts) and value_idx < len(parts):
                option_name = parts[name_idx]
                option_value = parts[value_idx]
                
                if option_name == "Debug":
                    self.debug = option_value.lower() == "true"
                elif option_name == "BountyWeight":
                    weight = int(option_value) / 100.0
                    self.ai.bounty_weight = weight
                    self.ai.rnn_weight = 1.0 - weight
                elif option_name == "SearchDepth":
                    self.ai.search_depth = int(option_value)
    
    def _handle_position(self, line: str):
        """Handle position command"""
        parts = line.split()
        
        if "startpos" in parts:
            self.board.reset()
            self.ai.reset_game()
            moves_start = parts.index("startpos") + 1
        elif "fen" in parts:
            fen_start = parts.index("fen") + 1
            fen = " ".join(parts[fen_start:fen_start + 6])
            self.board.set_fen(fen)
            self.ai.reset_game()
            moves_start = fen_start + 6
        else:
            return
        
        if "moves" in parts:
            moves_start = parts.index("moves") + 1
            for move_str in parts[moves_start:]:
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in self.board.legal_moves:
                        self.ai._update_game_state(self.board, move)
                        self.board.push(move)
                except:
                    break
    
    def _handle_go(self, line: str):
        """Handle go command"""
        try:
            move = self.ai.get_move(self.board)
            if move:
                print(f"bestmove {move.uci()}")
                
                if self.debug:
                    analysis = self.ai.analyze_move(self.board, move)
                    print(f"info string RNN: {analysis['rnn_score']:.2f}, "
                          f"Bounty: {analysis['bounty_breakdown']['total']:.2f}, "
                          f"Phase: {analysis['game_phase']}")
            else:
                print("bestmove 0000")
        except Exception as e:
            if self.debug:
                print(f"info string Error: {e}")
            print("bestmove 0000")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="V7P3R Chess AI 2.0")
    parser.add_argument("--model", type=str, help="Path to trained model")
    parser.add_argument("--uci", action="store_true", help="Run in UCI mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--analyze", type=str, help="Analyze position from FEN")
    
    args = parser.parse_args()
    
    # Find model path
    model_path = args.model
    if not model_path:
        # Look for latest trained model
        models_dir = Path("models/genetic")
        if models_dir.exists():
            model_files = list(models_dir.glob("v7p3r_2.0_final.json"))
            if not model_files:
                model_files = list(models_dir.glob("best_gen_*.json"))
                if model_files:
                    model_files.sort()
                    model_path = str(model_files[-1])
            else:
                model_path = str(model_files[0])
    
    # Create AI
    ai = V7P3RAI_v2(model_path)
    
    if args.analyze:
        # Analyze specific position
        board = chess.Board(args.analyze)
        evaluation = ai.get_position_evaluation(board)
        print(f"Position evaluation: {evaluation}")
        
        # Show best moves
        legal_moves = list(board.legal_moves)[:5]  # Top 5 moves
        print("\nTop moves:")
        for move in legal_moves:
            analysis = ai.analyze_move(board, move)
            print(f"{move.uci()}: {analysis['combined_score']:.2f} "
                  f"(RNN: {analysis['rnn_score']:.2f}, "
                  f"Bounty: {analysis['bounty_breakdown']['total']:.2f})")
    
    elif args.uci or len(sys.argv) == 1:
        # UCI mode
        uci = UCIHandler_v2(ai)
        uci.debug = args.debug
        uci.run()
    
    else:
        print("V7P3R Chess AI 2.0")
        print("Use --uci for UCI mode or --analyze <fen> for position analysis")


if __name__ == "__main__":
    main()
