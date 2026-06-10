"""
UCI Game Executor for V8.0 Training

Executes games between v8.0 neural network and external UCI opponents.
Collects positions and outcomes for training.
"""

import chess
import time
import torch
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from network import V8ValueNetwork
from comprehensive_features import ComprehensiveFeatureExtractor
from opponent_manager import UCIEngine, OpponentConfig
try:
    from tablebase_oracle import TablebaseOracle
    TABLEBASE_AVAILABLE = True
except:
    TABLEBASE_AVAILABLE = False


@dataclass
class GameResult:
    """Result of a single game"""
    moves: List[chess.Move]
    result: str  # "1-0", "1/2-1/2", "0-1"
    termination: str  # "checkmate", "stalemate", "insufficient", "max_moves", "tablebase"
    num_moves: int
    v8_color: chess.Color  # Which color v8 played
    positions: List[chess.Board]  # All positions from v8's perspective
    outcomes: List[float]  # Final result from v8's perspective for each position


class UCIGameExecutor:
    """
    Executes games between v8.0 and UCI opponents
    
    Collects experience for training while playing fast games.
    """
    
    def __init__(
        self,
        v8_network: V8ValueNetwork,
        feature_extractor: ComprehensiveFeatureExtractor,
        tablebase: Optional[TablebaseOracle] = None,
        device: str = 'cpu'
    ):
        """
        Initialize game executor
        
        Args:
            v8_network: Trained v8 neural network
            feature_extractor: Feature extraction utility
            tablebase: Optional tablebase oracle
            device: 'cpu' or 'cuda'
        """
        self.network = v8_network
        self.network.eval()
        self.feature_extractor = feature_extractor
        self.tablebase = tablebase
        self.device = device
        
        # Game parameters
        self.max_moves = 200
        self.movetime_ms = 3000  # 3 seconds per move (fast training)
        self.temperature = 0.3  # Some randomness for exploration
    
    def evaluate_position(self, board: chess.Board, move_number: int, prev_inference_ms: float) -> float:
        """Evaluate position using v8 network"""
        features = self.feature_extractor.extract_all_features(
            board,
            move_number=move_number,
            previous_inference_ms=prev_inference_ms
        )
        
        # Convert to numpy array first for better performance
        features_array = np.array([features], dtype=np.float32)
        features_tensor = torch.from_numpy(features_array).to(self.device)
        
        with torch.no_grad():
            value = self.network(features_tensor).item()
        
        return value
    
    def select_move(
        self,
        board: chess.Board,
        move_number: int,
        prev_inference_ms: float
    ) -> Tuple[chess.Move, float]:
        """
        Select move for v8
        
        Returns:
            (move, inference_time_ms)
        """
        # Check tablebase first
        if self.tablebase and self.tablebase.is_available(board):
            tb_move = self.tablebase.get_best_move(board)
            if tb_move:
                return tb_move, 0.0
        
        # Get legal moves
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            return None, 0.0
        
        if len(legal_moves) == 1:
            return legal_moves[0], 0.0
        
        # Evaluate all moves
        move_scores = []
        start_time = time.time()
        
        for move in legal_moves:
            board.push(move)
            score = -self.evaluate_position(board, move_number, prev_inference_ms)
            board.pop()
            move_scores.append((move, score))
        
        inference_time_ms = (time.time() - start_time) * 1000.0
        
        # Select with temperature
        if self.temperature < 0.01:
            best_move, _ = max(move_scores, key=lambda x: x[1])
            return best_move, inference_time_ms
        else:
            import numpy as np
            
            scores = np.array([score for _, score in move_scores])
            exp_scores = np.exp(scores / self.temperature)
            probs = exp_scores / np.sum(exp_scores)
            
            chosen_idx = np.random.choice(len(legal_moves), p=probs)
            chosen_move = move_scores[chosen_idx][0]
            
            return chosen_move, inference_time_ms
    
    def play_game(
        self,
        opponent_engine: UCIEngine,
        v8_plays_white: bool,
        opponent_name: str = "Opponent"
    ) -> GameResult:
        """
        Play a single game
        
        Args:
            opponent_engine: UCI engine opponent
            v8_plays_white: True if v8 plays white
            opponent_name: Name for logging
        
        Returns:
            GameResult with experience data
        """
        board = chess.Board()
        moves = []
        v8_positions = []  # Positions where v8 moved
        move_number = 0
        prev_inference_ms = 0.0
        
        # Initialize opponent for new game
        opponent_engine.send_command("ucinewgame")
        opponent_engine.send_command("isready")
        opponent_engine.wait_for("readyok", timeout=2.0)
        
        # Play game
        while not board.is_game_over() and len(moves) < self.max_moves:
            # Check tablebase termination
            if self.tablebase and self.tablebase.is_available(board):
                # Reached tablebase position - stop here
                termination = "tablebase"
                result = board.result(claim_draw=True) if board.result(claim_draw=True) != "*" else "1/2-1/2"
                break
            
            # Determine whose turn
            is_v8_turn = (board.turn == chess.WHITE) == v8_plays_white
            
            if is_v8_turn:
                # V8's turn
                # Save position before move
                v8_positions.append(board.copy())
                
                move, inference_ms = self.select_move(board, move_number, prev_inference_ms)
                prev_inference_ms = inference_ms
                
                if move is None:
                    break
            else:
                # Opponent's turn - pass move history for better UCI compatibility
                move = opponent_engine.get_move(board.fen(), self.movetime_ms, move_history=moves)
                
                if move is None or move not in board.legal_moves:
                    # Opponent failed to provide legal move
                    # Award win to v8
                    result = "1-0" if v8_plays_white else "0-1"
                    termination = "opponent_illegal_move"
                    break
            
            # Make move
            board.push(move)
            moves.append(move)
            move_number += 1
        
        # Determine result and termination
        if board.is_game_over():
            result = board.result(claim_draw=True)
            
            if board.is_checkmate():
                termination = "checkmate"
            elif board.is_stalemate():
                termination = "stalemate"
            elif board.is_insufficient_material():
                termination = "insufficient"
            else:
                termination = "draw_rule"
        elif len(moves) >= self.max_moves:
            result = "1/2-1/2"
            termination = "max_moves"
        else:
            # Game ended for other reason (tablebase, illegal move handled above)
            pass
        
        # Convert result to outcome value for v8's positions
        outcomes = []
        for pos in v8_positions:
            if result == "1-0":
                outcome = +1.0 if v8_plays_white else -1.0
            elif result == "0-1":
                outcome = -1.0 if v8_plays_white else +1.0
            else:
                outcome = 0.0
            
            # Flip outcome if position is from black's perspective
            if not v8_plays_white:
                outcome = -outcome
            
            outcomes.append(outcome)
        
        return GameResult(
            moves=moves,
            result=result,
            termination=termination,
            num_moves=len(moves),
            v8_color=chess.WHITE if v8_plays_white else chess.BLACK,
            positions=v8_positions,
            outcomes=outcomes
        )
    
    def play_game_pair(
        self,
        opponent_engine: UCIEngine,
        opponent_name: str = "Opponent"
    ) -> Tuple[GameResult, GameResult]:
        """
        Play 2 games (one as white, one as black)
        
        Args:
            opponent_engine: UCI opponent
            opponent_name: Name for logging
        
        Returns:
            (white_game, black_game) results
        """
        # Game 1: v8 plays white
        white_game = self.play_game(opponent_engine, v8_plays_white=True, opponent_name=opponent_name)
        
        # Game 2: v8 plays black
        black_game = self.play_game(opponent_engine, v8_plays_white=False, opponent_name=opponent_name)
        
        return white_game, black_game


if __name__ == "__main__":
    # Test game executor
    print("Testing UCI Game Executor...")
    
    # Load v8 network
    network = V8ValueNetwork(input_dim=55)
    try:
        network.load_state_dict(torch.load('../training/v8_generational/gen_0010_value_network.pt'))
        print("✓ Loaded Gen 10 network")
    except:
        print("⚠ Using random network (Gen 10 not found)")
    
    network.eval()
    
    # Create executor
    from opponent_manager import create_opponent_pool
    
    feature_extractor = ComprehensiveFeatureExtractor()
    executor = UCIGameExecutor(network, feature_extractor, device='cpu')
    
    print(f"✓ Game executor initialized")
    print(f"  Max moves: {executor.max_moves}")
    print(f"  Move time: {executor.movetime_ms}ms")
    print(f"  Temperature: {executor.temperature}")
