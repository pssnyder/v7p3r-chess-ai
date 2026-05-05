#!/usr/bin/env python3
"""
Extract V7P3R heuristic evaluations from historical games.

ARCHITECTURE:
1. Load V7P3R game history (PGN files)
2. For each position:
   - Extract move V7P3R actually played
   - Get Stockfish's top-5 moves (ground truth)
   - Decision:
     * IF V7P3R move in top-5 → Extract heuristic breakdown, mark as "correct"
     * ELSE → Flag as "wrong", provide corrective signal

3. Heuristic extraction (only for correct moves):
   - Material, PST, pawn structure, king safety, bishop pair, etc.
   - Teach AI "this is why V7P3R made this good move"

4. Wrong move handling:
   - Mark as "uncertain" or "correction needed"
   - Optionally provide Stockfish's best move
   - Don't teach AI wrong patterns
"""

import argparse
import chess
import chess.pgn
import chess.engine
import sys
import os
import json
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, asdict
from pathlib import Path
from tqdm import tqdm

# Add V7P3R source to path for heuristic extraction (local working copy)
V7P3R_SRC = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'v7p3r_engine')
sys.path.insert(0, V7P3R_SRC)

from v7p3r_modular_eval import ModularEvaluator
from v7p3r_fast_evaluator import V7P3RFastEvaluator
from v7p3r_eval_selector import EvaluationProfile, EvaluationProfileSelector
from v7p3r_position_context import PositionContextCalculator


@dataclass
class HeuristicBreakdown:
    """Individual V7P3R heuristic evaluations"""
    material: float
    pst: float
    pawn_structure: float
    bishop_pair: float
    king_safety_basic: float
    # Add more as needed
    total_score: float


@dataclass
class GamePosition:
    """Training example from V7P3R game history"""
    fen: str
    game_id: str
    move_number: int
    v7p3r_played: str  # Move V7P3R actually played in the game
    stockfish_top5: List[Tuple[str, int]]  # [(move, score), ...]
    stockfish_best: str
    stockfish_eval: int
    
    # Decision outcome
    v7p3r_correct: bool  # Is V7P3R's move in Stockfish top-5?
    in_top5_rank: Optional[int]  # If correct, what rank? (1=best, 2=2nd best, etc.)
    
    # Heuristic breakdown (only if v7p3r_correct=True)
    heuristics: Optional[HeuristicBreakdown] = None
    
    # Correction info (only if v7p3r_correct=False)
    needs_correction: bool = False
    eval_difference: Optional[int] = None  # If we compute V7P3R's eval for this position


class GameHistoryExtractor:
    """Extract training data from V7P3R's game history"""
    
    def __init__(self, stockfish_path: str):
        self.stockfish_path = stockfish_path
        
        # Verify Stockfish exists
        if not os.path.exists(stockfish_path):
            raise FileNotFoundError(f"Stockfish not found: {stockfish_path}")
        
        # Initialize Stockfish
        print(f"Starting Stockfish engine...")
        self.stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        print(f"✅ Stockfish ready")
        
        # Initialize V7P3R evaluator for heuristic extraction
        print(f"Loading V7P3R evaluator...")
        self.fast_evaluator = V7P3RFastEvaluator()
        self.v7p3r_evaluator = ModularEvaluator(self.fast_evaluator)
        self.context_calculator = PositionContextCalculator()
        self.profile_selector = EvaluationProfileSelector()
        print(f"✅ V7P3R evaluator ready")
    
    def get_stockfish_top5(self, fen: str, time_limit: float = 0.5) -> Tuple[List[Tuple[str, int]], str, int]:
        """
        Get Stockfish's top 5 moves with evaluations (ground truth)
        
        Returns:
            (top5_moves, best_move, best_eval)
            top5_moves: [(move, cp_score), ...]
        """
        try:
            board = chess.Board(fen)
            
            # MultiPV analysis (top 5)
            result = self.stockfish.analyse(
                board,
                chess.engine.Limit(time=time_limit),
                multipv=5
            )
            
            moves_with_scores = []
            for analysis in result:
                if 'pv' in analysis and analysis['pv']:
                    move = analysis['pv'][0].uci()
                    score = analysis['score'].white().score(mate_score=10000)
                    moves_with_scores.append((move, score))
            
            if moves_with_scores:
                best_move = moves_with_scores[0][0]
                best_eval = moves_with_scores[0][1]
                return (moves_with_scores, best_move, best_eval)
            
            return ([], None, None)
            
        except Exception as e:
            print(f"⚠️  Stockfish error: {e}")
            return ([], None, None)
    
    def extract_heuristics(self, fen: str) -> HeuristicBreakdown:
        """
        Extract individual heuristic evaluations from V7P3R
        
        This is the core value: teaching AI why V7P3R makes good moves
        """
        board = chess.Board(fen)
        
        # Extract individual heuristics using fast evaluator methods
        material = self.fast_evaluator.evaluate_material(board)
        pst = self.fast_evaluator.evaluate_pst(board)
        strategic = self.fast_evaluator.evaluate_strategic(board)
        
        # ModularEvaluator methods
        bishop_pair = self.v7p3r_evaluator._evaluate_bishop_pair(board)
        king_safety = self.v7p3r_evaluator._evaluate_king_safety_basic(board)
        
        # Get total score using COMPREHENSIVE profile with context
        context = self.context_calculator.calculate(board)
        profile = self.profile_selector._build_comprehensive_profile(context)
        total = self.v7p3r_evaluator.evaluate_with_profile(
            board,
            profile,
            context
        )
        
        return HeuristicBreakdown(
            material=material,
            pst=pst,
            pawn_structure=strategic,  # Strategic includes pawn structure
            bishop_pair=bishop_pair,
            king_safety_basic=king_safety,
            total_score=total
        )
    
    def extract_positions_from_pgn(self, 
                                   pgn_path: str,
                                   max_positions: int = 100,
                                   sample_every_n_moves: int = 5) -> List[GamePosition]:
        """
        Extract positions from V7P3R game PGN
        
        Args:
            pgn_path: Path to PGN file
            max_positions: Maximum positions to extract
            sample_every_n_moves: Sample every Nth move (avoid consecutive positions)
        
        Returns:
            List of GamePosition with heuristic breakdowns
        """
        positions = []
        
        with open(pgn_path) as pgn_file:
            game_num = 0
            
            while len(positions) < max_positions:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                
                game_num += 1
                game_id = f"{Path(pgn_path).stem}_game{game_num}"
                
                board = game.board()
                move_num = 0
                
                for node in game.mainline():
                    move_num += 1
                    
                    # Sample every Nth move
                    if move_num % sample_every_n_moves != 0:
                        board.push(node.move)
                        continue
                    
                    # Skip opening (first 8 moves)
                    if move_num < 8:
                        board.push(node.move)
                        continue
                    
                    # Get FEN before the move
                    fen = board.fen()
                    v7p3r_played = node.move.uci()
                    
                    # Get Stockfish's top-5 moves (ground truth)
                    sf_top5, sf_best, sf_eval = self.get_stockfish_top5(fen)
                    
                    if not sf_top5:
                        board.push(node.move)
                        continue
                    
                    # Check if V7P3R's move is correct (in top-5)
                    sf_moves = [m for m, s in sf_top5]
                    v7p3r_correct = v7p3r_played in sf_moves
                    in_top5_rank = (sf_moves.index(v7p3r_played) + 1) if v7p3r_correct else None
                    
                    # Extract heuristics ONLY if V7P3R was correct
                    heuristics = None
                    if v7p3r_correct:
                        heuristics = self.extract_heuristics(fen)
                    
                    # Create position
                    position = GamePosition(
                        fen=fen,
                        game_id=game_id,
                        move_number=move_num,
                        v7p3r_played=v7p3r_played,
                        stockfish_top5=sf_top5,
                        stockfish_best=sf_best,
                        stockfish_eval=sf_eval,
                        v7p3r_correct=v7p3r_correct,
                        in_top5_rank=in_top5_rank,
                        heuristics=heuristics,
                        needs_correction=not v7p3r_correct
                    )
                    
                    positions.append(position)
                    
                    if len(positions) >= max_positions:
                        break
                    
                    board.push(node.move)
        
        return positions
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'stockfish'):
            self.stockfish.quit()


def create_training_examples(positions: List[GamePosition], output_path: str, corrections_path: str):
    """
    Create training examples from extracted positions
    
    Two outputs:
    1. Correct moves (with heuristic breakdowns) → Train AI on V7P3R's logic
    2. Wrong moves (corrections needed) → Flag uncertain positions
    """
    correct_examples = []
    correction_examples = []
    
    for pos in positions:
        # Common fields
        example = {
            'fen': pos.fen,
            'game_id': pos.game_id,
            'move_number': pos.move_number,
            'v7p3r_played': pos.v7p3r_played,
            'stockfish_best': pos.stockfish_best,
            'stockfish_top5': pos.stockfish_top5,
            'stockfish_eval': pos.stockfish_eval
        }
        
        if pos.v7p3r_correct:
            # V7P3R was correct - teach AI these heuristics
            example.update({
                'in_top5_rank': pos.in_top5_rank,
                'heuristics': asdict(pos.heuristics) if pos.heuristics else None,
                'confidence': 1.0 if pos.in_top5_rank == 1 else 0.8  # Highest confidence for #1 moves
            })
            correct_examples.append(example)
        else:
            # V7P3R was wrong - flag for correction
            example.update({
                'needs_correction': True,
                'v7p3r_wrong': True
            })
            correction_examples.append(example)
    
    # Save correct examples (for training)
    with open(output_path, 'w') as f:
        json.dump(correct_examples, f, indent=2)
    
    # Save corrections (for analysis/feedback)
    with open(corrections_path, 'w') as f:
        json.dump(correction_examples, f, indent=2)
    
    # Statistics
    total = len(positions)
    correct = len(correct_examples)
    wrong = len(correction_examples)
    
    print()
    print("="*80)
    print("Training Data Statistics:")
    print("="*80)
    print(f"Total positions: {total}")
    if total > 0:
        print(f"V7P3R correct (in top-5): {correct} ({correct/total*100:.1f}%)")
        print(f"V7P3R wrong (not in top-5): {wrong} ({wrong/total*100:.1f}%)")
    print()
    
    if correct > 0:
        rank_distribution = {}
        for ex in correct_examples:
            rank = ex['in_top5_rank']
            rank_distribution[rank] = rank_distribution.get(rank, 0) + 1
        
        print("Rank distribution (for correct moves):")
        for rank in sorted(rank_distribution.keys()):
            count = rank_distribution[rank]
            print(f"  Rank #{rank}: {count} ({count/correct*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Extract V7P3R heuristics from game history")
    parser.add_argument('--game-dir', required=True, help="Directory with V7P3R game PGNs")
    parser.add_argument('--stockfish', default=r"E:\Programming Stuff\Chess Engines\Tournament Engines\Stockfish\stockfish-windows-x86-64-avx2.exe", help="Stockfish path")
    parser.add_argument('--output', default='data/v7p3r_heuristics_training.json', help="Output JSON for correct moves")
    parser.add_argument('--corrections', default='data/v7p3r_corrections.json', help="Output JSON for wrong moves")
    parser.add_argument('--max-positions', type=int, default=100, help="Max positions to extract")
    parser.add_argument('--sample-every', type=int, default=5, help="Sample every Nth move")
    
    args = parser.parse_args()
    
    # Find most recent PGN file
    game_dir = Path(args.game_dir)
    pgn_files = sorted(game_dir.glob('*.pgn'), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not pgn_files:
        print(f"❌ No PGN files found in {game_dir}")
        return
    
    # Use only the most recent file
    latest_pgn = pgn_files[0]
    print(f"📂 Using most recent PGN: {latest_pgn.name}")
    print(f"📅 Last modified: {latest_pgn.stat().st_mtime}")
    print(f"🎯 Extracting up to {args.max_positions} positions")
    print()
    
    # Initialize extractor
    extractor = GameHistoryExtractor(args.stockfish)
    
    # Extract from latest game file
    try:
        all_positions = extractor.extract_positions_from_pgn(
            str(latest_pgn),
            max_positions=args.max_positions,
            sample_every_n_moves=args.sample_every
        )
        print(f"✅ Extracted {len(all_positions)} positions")
    except Exception as e:
        print(f"❌ Error processing {latest_pgn.name}: {e}")
        import traceback
        traceback.print_exc()
        extractor.close()
        return
    
    # Create training examples
    create_training_examples(all_positions, args.output, args.corrections)
    
    print(f"✅ Saved training data to {args.output}")
    print(f"✅ Saved corrections to {args.corrections}")
    
    # Cleanup
    extractor.close()


if __name__ == "__main__":
    main()
