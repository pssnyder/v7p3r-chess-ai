"""
V7P3R v8.0 - Pure Self-Play Trainer

PURE SELF-PLAY - NO STOCKFISH, NO ORACLES
Model learns entirely from its own experience: wins, draws, losses.

Speed: 100-1000x faster than v7.0 (no Stockfish evaluation overhead)
Training signal: Pure win/loss outcomes + tablebase perfection
"""

import chess
import torch
import numpy as np
import time
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


@dataclass
class GameExperience:
    """Single position experience during self-play"""
    features: np.ndarray  # 55-dim feature vector
    reward: float  # Game outcome: +1 win, 0 draw, -1 loss
    move_number: int
    opening_id: int  # Which opening was used


@dataclass
class GameResult:
    """Complete game result with statistics"""
    result: str  # "1-0", "0-1", "1/2-1/2"
    num_moves: int
    opening_id: int
    opening_name: str
    experiences: List[GameExperience]
    tablebase_finish: bool
    game_duration_sec: float


class PureSelfPlayGame:
    """
    Pure self-play game execution - NO external oracles
    
    Speed optimizations:
    - No Stockfish evaluation (saves 30-50ms per move)
    - No complex validation logic
    - Simple temperature-based move selection
    - Tablebase for perfect endgames only
    """
    
    def __init__(self,
                 value_network,
                 feature_extractor,
                 tablebase_oracle=None,
                 max_moves: int = 200,
                 temperature: float = 0.3):
        """
        Args:
            value_network: Neural network for position evaluation
            feature_extractor: ComprehensiveFeatureExtractor (55-dim)
            tablebase_oracle: Optional tablebase for endgames
            max_moves: Maximum moves before draw
            temperature: Move selection randomness (lower = more deterministic)
        """
        self.value_network = value_network
        self.extractor = feature_extractor
        self.tablebase_oracle = tablebase_oracle
        self.max_moves = max_moves
        self.temperature = temperature
        
        self.device = next(value_network.parameters()).device
    
    def select_move(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Select move using ONLY neural network evaluation
        
        NO STOCKFISH - Pure model inference
        
        Args:
            board: Current position
        
        Returns:
            Selected move (or None if no legal moves)
        """
        # First check tablebase (perfect endgames)
        if self.tablebase_oracle and self.tablebase_oracle.is_available(board):
            tb_move = self.tablebase_oracle.get_best_move(board)
            if tb_move is not None:
                return tb_move
        
        # Get legal moves
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        # If only one move, play it
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        # Evaluate all legal moves using neural network
        move_evaluations = []
        
        for move in legal_moves:
            # Make move temporarily
            board.push(move)
            
            # Extract features
            features = self.extractor.extract_all_features(
                board,
                move_number=board.ply() // 2,
                previous_inference_ms=0.0
            )
            
            # Evaluate with network
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                value = self.value_network(features_tensor).item()
            
            # Negate if opponent's turn (we want OUR evaluation)
            if board.turn == chess.BLACK:
                value = -value
            
            move_evaluations.append((move, value))
            
            # Undo move
            board.pop()
        
        # Sort by evaluation (best first)
        move_evaluations.sort(key=lambda x: -x[1])
        
        # Temperature-based selection
        if self.temperature > 0 and len(move_evaluations) > 1:
            # Softmax with temperature
            values = np.array([val for _, val in move_evaluations])
            values = values / self.temperature
            
            # Shift to prevent overflow
            values = values - values.max()
            exp_values = np.exp(values)
            probs = exp_values / exp_values.sum()
            
            # Sample
            idx = np.random.choice(len(move_evaluations), p=probs)
            selected_move = move_evaluations[idx][0]
        else:
            # Greedy: best move
            selected_move = move_evaluations[0][0]
        
        return selected_move
    
    def play_game(self, opening_id: int, opening_selector) -> GameResult:
        """
        Play complete self-play game
        
        Workflow:
        1. Execute opening macro (10-20 moves instantly)
        2. Play from there using neural network
        3. Stop at tablebase position OR max moves OR checkmate/stalemate
        4. Assign reward based on outcome
        
        Args:
            opening_id: Which opening to use
            opening_selector: OpeningSelector instance
        
        Returns:
            GameResult with experiences and outcome
        """
        start_time = time.time()
        
        # Initialize board
        board = chess.Board()
        
        # Execute opening macro
        board, opening_ply, opening_name = opening_selector.execute_opening(board, opening_id)
        
        # Play game
        experiences = []
        tablebase_finish = False
        previous_inference_ms = 0.0
        
        for move_num in range(opening_ply // 2, self.max_moves):
            # Check game over conditions
            if board.is_checkmate():
                result_str = "0-1" if board.turn == chess.WHITE else "1-0"
                break
            
            if board.is_stalemate() or board.is_insufficient_material():
                result_str = "1/2-1/2"
                break
            
            if board.halfmove_clock >= 100:  # 50-move rule
                result_str = "1/2-1/2"
                break
            
            if board.is_repetition(3):
                result_str = "1/2-1/2"
                break
            
            # Check tablebase
            if self.tablebase_oracle and self.tablebase_oracle.is_available(board):
                # Tablebase position reached - declare result
                wdl = self.tablebase_oracle.probe_wdl(board)
                if wdl > 0:
                    result_str = "1-0"  # White wins
                elif wdl < 0:
                    result_str = "0-1"  # Black wins
                else:
                    result_str = "1/2-1/2"  # Draw
                tablebase_finish = True
                break
            
            # Extract features BEFORE move
            inference_start = time.time()
            features = self.extractor.extract_all_features(
                board,
                move_number=move_num,
                previous_inference_ms=previous_inference_ms
            )
            
            # Select and make move
            move = self.select_move(board)
            inference_ms = (time.time() - inference_start) * 1000
            previous_inference_ms = inference_ms
            
            if move is None:
                # No legal moves (shouldn't happen, but safety)
                result_str = "1/2-1/2"
                break
            
            board.push(move)
            
            # Store experience (reward assigned later based on game outcome)
            experiences.append(GameExperience(
                features=features,
                reward=0.0,  # Will be updated with game result
                move_number=move_num,
                opening_id=opening_id
            ))
        
        else:
            # Max moves reached
            result_str = "1/2-1/2"
        
        # Calculate game duration
        game_duration = time.time() - start_time
        
        # Assign rewards based on result
        if result_str == "1-0":
            # White won
            for exp in experiences:
                exp.reward = 1.0  # Winning positions get +1
        elif result_str == "0-1":
            # Black won
            for exp in experiences:
                exp.reward = -1.0  # Losing positions get -1
        else:
            # Draw
            for exp in experiences:
                exp.reward = 0.0  # Drawn positions get 0
        
        # Create result
        return GameResult(
            result=result_str,
            num_moves=board.ply() // 2,
            opening_id=opening_id,
            opening_name=opening_name,
            experiences=experiences,
            tablebase_finish=tablebase_finish,
            game_duration_sec=game_duration
        )


def test_pure_selfplay():
    """Test pure self-play game functionality"""
    print("Testing Pure Self-Play...")
    
    # Import dependencies
    from comprehensive_features import ComprehensiveFeatureExtractor
    from opening_selector import OpeningSelector
    
    # Create minimal network for testing
    class DummyNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(55, 1)
        
        def forward(self, x):
            return torch.tanh(self.fc(x))
    
    network = DummyNetwork()
    extractor = ComprehensiveFeatureExtractor()
    selector = OpeningSelector()
    
    # Create self-play game
    game = PureSelfPlayGame(
        value_network=network,
        feature_extractor=extractor,
        tablebase_oracle=None,  # No tablebase for quick test
        max_moves=50,  # Short game
        temperature=0.5
    )
    
    print(f"✓ Created self-play game (max {game.max_moves} moves)")
    
    # Play test game
    print("\nPlaying test game...")
    opening_id = selector.random_opening()
    
    result = game.play_game(opening_id, selector)
    
    print(f"\n✓ Game completed!")
    print(f"  Result: {result.result}")
    print(f"  Moves: {result.num_moves}")
    print(f"  Opening: {result.opening_name}")
    print(f"  Experiences: {len(result.experiences)}")
    print(f"  Duration: {result.game_duration_sec:.2f}s")
    print(f"  Speed: {result.num_moves / result.game_duration_sec:.1f} moves/sec")
    
    print("\n✓ Pure self-play tests passed!")


if __name__ == '__main__':
    test_pure_selfplay()
