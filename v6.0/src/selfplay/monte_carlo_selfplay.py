#!/usr/bin/env python3
"""
Monte Carlo Self-Play Engine
V7P3R AI v6.1 - Stage 2 Training Data Generation

Generates diverse chess games using Stage 1 evaluator to create training data
for Stage 2 complexity and time management learning.

Core Philosophy: Teach the model to predict EFFORT required per position.

Author: Pat Snyder
Created: 2026-05-31
"""

import chess
import chess.pgn
import random
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.stage1.position_evaluator import PositionEvaluator
from src.stage1.feature_extractor import extract_fast_features
from src.engine.static_checkmate import StaticCheckmateDetector
from src.engine.static_draw_detection import StaticDrawDetector


class TimeScenario:
    """
    Represents a time control scenario for diverse training.
    
    Scenarios cover:
    - Early game (abundant time)
    - Midgame (moderate time)
    - Endgame (time pressure)
    - Emergency (severe time pressure)
    """
    
    def __init__(
        self, 
        name: str,
        initial_time: float,
        increment: float,
        target_moves_remaining: int,
        description: str
    ):
        self.name = name
        self.initial_time = initial_time
        self.increment = increment
        self.target_moves_remaining = target_moves_remaining
        self.description = description
    
    def __repr__(self):
        return f"TimeScenario({self.name}: {self.initial_time}+{self.increment})"


# Predefined time scenarios for diverse training
TIME_SCENARIOS = {
    'bullet_early': TimeScenario(
        name='bullet_early',
        initial_time=60.0,  # 1 minute
        increment=2.0,
        target_moves_remaining=30,
        description='Early bullet game - moderate time pressure'
    ),
    'bullet_midgame': TimeScenario(
        name='bullet_midgame',
        initial_time=30.0,  # 30 seconds
        increment=2.0,
        target_moves_remaining=15,
        description='Midgame bullet - increasing pressure'
    ),
    'bullet_endgame': TimeScenario(
        name='bullet_endgame',
        initial_time=8.0,  # 8 seconds (emergency)
        increment=2.0,
        target_moves_remaining=5,
        description='Bullet endgame - severe time pressure'
    ),
    'blitz_early': TimeScenario(
        name='blitz_early',
        initial_time=300.0,  # 5 minutes
        increment=4.0,
        target_moves_remaining=40,
        description='Early blitz - abundant time'
    ),
    'blitz_midgame': TimeScenario(
        name='blitz_midgame',
        initial_time=120.0,  # 2 minutes
        increment=4.0,
        target_moves_remaining=20,
        description='Midgame blitz - moderate pressure'
    ),
    'blitz_endgame': TimeScenario(
        name='blitz_endgame',
        initial_time=25.0,  # 25 seconds
        increment=4.0,
        target_moves_remaining=8,
        description='Blitz endgame - time pressure'
    ),
    'rapid_early': TimeScenario(
        name='rapid_early',
        initial_time=900.0,  # 15 minutes
        increment=10.0,
        target_moves_remaining=50,
        description='Early rapid - deep calculation time'
    ),
    'rapid_midgame': TimeScenario(
        name='rapid_midgame',
        initial_time=600.0,  # 10 minutes
        increment=10.0,
        target_moves_remaining=25,
        description='Midgame rapid - comfortable time'
    ),
}


class PositionComplexityAnalyzer:
    """
    Analyzes position complexity for ground truth labeling.
    
    Calculates "forest darkness" - how complex/tactical the position is.
    """
    
    @staticmethod
    def analyze_position(board: chess.Board) -> Dict[str, float]:
        """
        Calculate position complexity metrics.
        
        Returns:
            Dictionary with complexity features
        """
        legal_moves = list(board.legal_moves)
        legal_moves_count = len(legal_moves)
        
        # Count move types
        capture_moves = [m for m in legal_moves if board.is_capture(m)]
        check_moves = []
        for move in legal_moves:
            board.push(move)
            if board.is_check():
                check_moves.append(move)
            board.pop()
        
        # Count tactical features
        pieces_under_attack = 0
        pieces_undefended = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                # Check if attacked
                if board.is_attacked_by(not board.turn, square):
                    pieces_under_attack += 1
                    # Check if defended
                    if not board.is_attacked_by(board.turn, square):
                        pieces_undefended += 1
        
        # Calculate branching factor (1-ply look-ahead)
        branching_factors = []
        for move in legal_moves[:10]:  # Sample first 10 moves
            board.push(move)
            branching_factors.append(len(list(board.legal_moves)))
            board.pop()
        
        branching_factor_1ply = sum(branching_factors) / len(branching_factors) if branching_factors else 0.0
        
        # Forest darkness score (Tal complexity metric)
        forest_darkness = (
            0.3 * (legal_moves_count / 280.0) +  # Normalized move count
            0.2 * (len(capture_moves) / max(legal_moves_count, 1)) +  # Capture ratio
            0.15 * min(len(check_moves) / max(legal_moves_count, 1), 1.0) +  # Check ratio
            0.15 * (pieces_under_attack / 16.0) +  # Attack pressure
            0.2 * (branching_factor_1ply / 40.0)  # Branching complexity
        )
        
        return {
            'legal_moves_count': legal_moves_count,
            'capture_moves_count': len(capture_moves),
            'check_moves_count': len(check_moves),
            'pieces_under_attack': pieces_under_attack,
            'pieces_undefended': pieces_undefended,
            'branching_factor_1ply': branching_factor_1ply,
            'forest_darkness_score': min(forest_darkness, 1.0),  # Cap at 1.0
            'tactical_density': pieces_under_attack + len(check_moves),
        }


class MonteCarloSelfPlay:
    """
    Self-play engine for generating Stage 2 training data.
    
    Key Features:
    - Uses Stage 1 evaluator for move selection
    - Varies time scenarios (bullet/blitz/rapid, early/mid/endgame)
    - Records effort metrics (nodes searched, time spent)
    - Generates diverse position complexities
    - Labels positions with ground truth complexity and time allocation
    """
    
    def __init__(
        self,
        stage1_model_path: Path,
        output_dir: Path,
        device: str = 'cpu'
    ):
        """
        Initialize self-play engine.
        
        Args:
            stage1_model_path: Path to trained Stage 1 model (.pth)
            output_dir: Directory for saving game data
            device: 'cpu' or 'cuda'
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Stage 1 model
        print(f"Loading Stage 1 model from {stage1_model_path}...")
        self.stage1_model = PositionEvaluator.load(stage1_model_path, device=device)
        self.stage1_model.eval()
        
        # Initialize static modules
        self.checkmate_detector = StaticCheckmateDetector(default_depth=5)
        self.draw_detector = StaticDrawDetector(repetition_eval_threshold=50)
        
        # Position complexity analyzer
        self.complexity_analyzer = PositionComplexityAnalyzer()
        
        # Game statistics
        self.games_played = 0
        self.positions_recorded = 0
        
    def play_game(
        self,
        time_scenario: TimeScenario,
        starting_fen: Optional[str] = None,
        max_moves: int = 150,
        resignation_threshold_cp: int = 800,
        resignation_move_count: int = 5
    ) -> Dict:
        """
        Play a single self-play game.
        
        Args:
            time_scenario: Time control scenario for this game
            starting_fen: Starting position (None = standard opening)
            max_moves: Maximum moves before draw
            resignation_threshold_cp: Resign if down this many centipawns
            resignation_move_count: Consecutive moves down before resignation
            
        Returns:
            Dictionary with game data and recorded positions
        """
        # Initialize board
        if starting_fen:
            board = chess.Board(starting_fen)
        else:
            board = chess.Board()
        
        # Initialize time clocks
        time_white = time_scenario.initial_time
        time_black = time_scenario.initial_time
        increment = time_scenario.increment
        
        # Game metadata
        game_id = f"selfplay_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
        positions_data = []
        move_number = 0
        consecutive_losing_moves = 0
        last_eval_cp = 0.0
        
        # PGN for game record
        game_pgn = chess.pgn.Game()
        game_pgn.headers["Event"] = "V7P3R AI Self-Play"
        game_pgn.headers["White"] = "V7P3R_Stage1"
        game_pgn.headers["Black"] = "V7P3R_Stage1"
        game_pgn.headers["TimeControl"] = f"{time_scenario.initial_time}+{increment}"
        game_pgn.headers["Scenario"] = time_scenario.name
        
        node = game_pgn
        
        while not board.is_game_over() and move_number < max_moves:
            move_number += 1
            
            # Current time state
            time_remaining = time_white if board.turn == chess.WHITE else time_black
            
            # Check for draw
            if self.draw_detector.is_draw_position(board):
                break
            
            # Calculate time budget
            time_budget = self._calculate_time_budget(
                time_remaining,
                increment,
                time_scenario.target_moves_remaining - (move_number // 2)
            )
            
            # Record position BEFORE move
            position_data = self._record_position(
                board=board.copy(),
                game_id=game_id,
                move_number=move_number,
                time_white=time_white,
                time_black=time_black,
                increment=increment,
                time_budget=time_budget
            )
            
            # Make move decision (this is where effort is spent)
            move_start_time = time.time()
            nodes_searched = 0
            
            # 1. Check for checkmate
            mate_move = self.checkmate_detector.find_checkmate(board, time_available=time_budget)
            nodes_searched += self.checkmate_detector.nodes_searched
            
            if mate_move:
                best_move = mate_move
                eval_cp = 2900.0  # Checkmate score
            else:
                # 2. Use Stage 1 to select move
                best_move, eval_cp, move_nodes = self._select_move_with_stage1(
                    board, 
                    time_budget
                )
                nodes_searched += move_nodes
            
            move_time_spent = time.time() - move_start_time
            
            # Update time clocks
            if board.turn == chess.WHITE:
                time_white -= move_time_spent
                time_white += increment
            else:
                time_black -= move_time_spent
                time_black += increment
            
            # Complete position data with actual effort metrics
            position_data['move_played'] = best_move.uci()
            position_data['time_spent'] = move_time_spent
            position_data['nodes_searched'] = nodes_searched
            position_data['eval_cp'] = eval_cp
            
            # Calculate labels for this position
            labels = self._generate_labels(
                position_data=position_data,
                time_budget=time_budget,
                time_spent=move_time_spent,
                nodes_searched=nodes_searched
            )
            position_data['labels'] = labels
            
            positions_data.append(position_data)
            
            # Make move on board
            board.push(best_move)
            node = node.add_variation(best_move)
            
            # Check resignation condition
            if eval_cp < -resignation_threshold_cp:
                consecutive_losing_moves += 1
                if consecutive_losing_moves >= resignation_move_count:
                    result = "0-1" if board.turn == chess.WHITE else "1-0"
                    game_pgn.headers["Result"] = result
                    game_pgn.headers["Termination"] = "resignation"
                    break
            else:
                consecutive_losing_moves = 0
            
            last_eval_cp = eval_cp
            
            # Check time forfeit
            if time_white <= 0:
                game_pgn.headers["Result"] = "0-1"
                game_pgn.headers["Termination"] = "time forfeit"
                break
            if time_black <= 0:
                game_pgn.headers["Result"] = "1-0"
                game_pgn.headers["Termination"] = "time forfeit"
                break
        
        # Game ended - determine result
        if board.is_game_over():
            result = board.result()
            game_pgn.headers["Result"] = result
            
            if board.is_checkmate():
                game_pgn.headers["Termination"] = "checkmate"
            elif board.is_stalemate():
                game_pgn.headers["Termination"] = "stalemate"
            elif board.is_insufficient_material():
                game_pgn.headers["Termination"] = "insufficient material"
            elif board.can_claim_fifty_moves():
                game_pgn.headers["Termination"] = "50-move rule"
            elif board.is_repetition(2):
                game_pgn.headers["Termination"] = "threefold repetition"
        elif move_number >= max_moves:
            game_pgn.headers["Result"] = "1/2-1/2"
            game_pgn.headers["Termination"] = "maximum moves"
        
        # Update position labels with game outcome
        game_result = game_pgn.headers["Result"]
        for pos_data in positions_data:
            pos_data['game_result'] = game_result
        
        self.games_played += 1
        self.positions_recorded += len(positions_data)
        
        return {
            'game_id': game_id,
            'pgn': str(game_pgn),
            'positions': positions_data,
            'result': game_result,
            'moves': move_number,
            'scenario': time_scenario.name,
        }
    
    def _record_position(
        self,
        board: chess.Board,
        game_id: str,
        move_number: int,
        time_white: float,
        time_black: float,
        increment: float,
        time_budget: float
    ) -> Dict:
        """
        Record position state for training data.
        
        Returns:
            Dictionary with position features and metadata
        """
        fen = board.fen()
        
        # Extract Stage 1 fast features (19-dim)
        stage1_features = extract_fast_features(fen)
        
        # Analyze position complexity
        complexity_metrics = self.complexity_analyzer.analyze_position(board)
        
        # Time state
        time_state = {
            'time_white': time_white,
            'time_black': time_black,
            'increment': increment,
            'time_budget': time_budget,
            'time_remaining': time_white if board.turn == chess.WHITE else time_black,
        }
        
        return {
            'game_id': game_id,
            'position_id': f"{game_id}_move_{move_number}",
            'fen': fen,
            'move_number': move_number,
            'side_to_move': 'white' if board.turn == chess.WHITE else 'black',
            'stage1_features': stage1_features.tolist(),
            'complexity_metrics': complexity_metrics,
            'time_state': time_state,
        }
    
    def _select_move_with_stage1(
        self,
        board: chess.Board,
        time_budget: float
    ) -> Tuple[chess.Move, float, int]:
        """
        Select move using Stage 1 evaluator.
        
        Returns:
            (best_move, eval_cp, nodes_searched)
        """
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        # Evaluate all legal moves with Stage 1
        move_evaluations = []
        nodes_searched = len(legal_moves)  # Count each move evaluation as 1 node
        
        for move in legal_moves:
            board.push(move)
            fen = board.fen()
            features = extract_fast_features(fen)
            prob_good = self.stage1_model.predict_probability(features)
            board.pop()
            
            move_evaluations.append({
                'move': move,
                'prob_good': prob_good,
            })
        
        # Select move with highest Stage 1 probability
        # (with small random noise to avoid determinism)
        best_eval = max(move_evaluations, key=lambda x: x['prob_good'] + random.uniform(0, 0.05))
        
        # Convert probability to centipawn estimate
        eval_cp = (best_eval['prob_good'] - 0.5) * 200.0  # Rough conversion
        
        return best_eval['move'], eval_cp, nodes_searched
    
    def _calculate_time_budget(
        self,
        time_remaining: float,
        increment: float,
        moves_remaining_estimate: int
    ) -> float:
        """
        Calculate time budget for this move.
        
        Simple heuristic: time_remaining / moves_remaining + increment
        """
        if moves_remaining_estimate <= 0:
            moves_remaining_estimate = 10
        
        base_budget = time_remaining / moves_remaining_estimate
        total_budget = base_budget + (increment * 0.8)  # Use 80% of increment
        
        # Keep 10% reserve
        reserve = time_remaining * 0.1
        max_budget = time_remaining - reserve
        
        return min(total_budget, max_budget, time_remaining - 1.0)
    
    def _generate_labels(
        self,
        position_data: Dict,
        time_budget: float,
        time_spent: float,
        nodes_searched: int
    ) -> Dict:
        """
        Generate ground truth labels for Stage 2 training.
        
        Labels:
        - complexity_score: 0-10 scale (from forest_darkness + branching)
        - time_allocation: fraction of budget used (0-1)
        - processing_tick_count: actual nodes searched
        """
        complexity_metrics = position_data['complexity_metrics']
        
        # Complexity score (0-10 scale)
        complexity_score = (
            complexity_metrics['forest_darkness_score'] * 5.0 +  # 0-5 from darkness
            (complexity_metrics['branching_factor_1ply'] / 40.0) * 3.0 +  # 0-3 from branching
            (complexity_metrics['tactical_density'] / 10.0) * 2.0  # 0-2 from tactics
        )
        complexity_score = min(complexity_score, 10.0)
        
        # Time allocation (fraction of budget used)
        time_allocation = min(time_spent / time_budget, 1.0) if time_budget > 0 else 0.0
        
        return {
            'complexity_score': complexity_score,
            'time_allocation': time_allocation,
            'processing_tick_count': nodes_searched,
            'effort_metric': nodes_searched / max(time_spent, 0.001),  # Nodes per second
        }
    
    def save_game_data(self, game_data: Dict, format: str = 'jsonl'):
        """
        Save game data to disk.
        
        Args:
            game_data: Game data dictionary from play_game()
            format: 'jsonl' or 'pgn'
        """
        if format == 'jsonl':
            # Save positions as JSONL
            positions_file = self.output_dir / f"{game_data['game_id']}_positions.jsonl"
            with open(positions_file, 'w') as f:
                for position in game_data['positions']:
                    f.write(json.dumps(position) + '\n')
        
        if format == 'pgn':
            # Save PGN
            pgn_file = self.output_dir / f"{game_data['game_id']}.pgn"
            with open(pgn_file, 'w') as f:
                f.write(game_data['pgn'])
        
        print(f"Saved game {game_data['game_id']}: "
              f"{game_data['result']} ({game_data['moves']} moves, "
              f"{len(game_data['positions'])} positions)")


# Example usage
if __name__ == "__main__":
    print("Monte Carlo Self-Play Engine - Test Run")
    print("=" * 60)
    
    # Initialize self-play engine
    model_path = Path("models/position_evaluator_best.pth")
    output_dir = Path("data/stage2/selfplay_games")
    
    if not model_path.exists():
        print(f"ERROR: Stage 1 model not found at {model_path}")
        print("Please train Stage 1 model first:")
        print("  python scripts/stage1/train_balanced.py")
        sys.exit(1)
    
    selfplay = MonteCarloSelfPlay(
        stage1_model_path=model_path,
        output_dir=output_dir,
        device='cpu'
    )
    
    # Play test game with blitz scenario
    print("\nPlaying test game (blitz midgame scenario)...")
    game_data = selfplay.play_game(
        time_scenario=TIME_SCENARIOS['blitz_midgame'],
        max_moves=50  # Short test game
    )
    
    # Save game data
    selfplay.save_game_data(game_data, format='jsonl')
    selfplay.save_game_data(game_data, format='pgn')
    
    # Print statistics
    print("\n" + "=" * 60)
    print(f"Test game complete!")
    print(f"  Result: {game_data['result']}")
    print(f"  Moves: {game_data['moves']}")
    print(f"  Positions recorded: {len(game_data['positions'])}")
    print(f"  Output: {output_dir}")
    
    # Show sample position data
    if game_data['positions']:
        print("\nSample position data (move 5):")
        if len(game_data['positions']) >= 5:
            sample = game_data['positions'][4]
            print(f"  FEN: {sample['fen']}")
            print(f"  Complexity score: {sample['labels']['complexity_score']:.2f}")
            print(f"  Time allocation: {sample['labels']['time_allocation']:.2f}")
            print(f"  Processing ticks: {sample['labels']['processing_tick_count']}")
            print(f"  Forest darkness: {sample['complexity_metrics']['forest_darkness_score']:.3f}")
