#!/usr/bin/env python3
"""
V7P3R Evaluation AI Training - Historical Games

Simple, efficient training pipeline:
1. Load V7P3R bot game history (PGN files)
2. Extract positions from actual games
3. Get V7P3R's evaluation for each position (UCI)
4. Use Stockfish to verify/correct evaluations (top-5 moves)
5. Train neural network to learn V7P3R's eval patterns + corrections

Two-Stage AI Integration:
- Move Ordering AI: Provides candidate moves (pattern-based)
- Evaluation AI: Selects best from candidates (learned from V7P3R games)

Author: Pat Snyder
Created: 2026-05-04 (Simplified Training v2.0)
"""

import os
import sys
import json
import chess
import chess.pgn
import chess.engine
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, asdict
import subprocess
from tqdm import tqdm
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class MultiDepthEval:
    """V7P3R evaluation at different depths"""
    depth: int
    best_move: str
    eval_score: int
    agrees_with_stockfish: bool
    agrees_with_shallow: bool  # Agrees with depth-2
    

@dataclass
class GamePosition:
    """Position extracted from V7P3R game history with multi-depth analysis"""
    fen: str
    game_id: str
    move_number: int
    v7p3r_played: str  # Move V7P3R actually played
    
    # Multi-depth V7P3R evaluations
    v7p3r_depth2: Optional[MultiDepthEval]  # Fast heuristic
    v7p3r_depth5: Optional[MultiDepthEval]  # Tactical verification
    v7p3r_depth10: Optional[MultiDepthEval]  # Full strategic
    
    # Stockfish ground truth
    stockfish_top5: List[Tuple[str, int]]  # [(move, cp_score), ...]
    stockfish_best: str
    stockfish_eval: int
    
    # Analysis flags
    shallow_is_best: bool  # Depth-2 matches Stockfish (good heuristic)
    deep_overthinks: bool  # Depth-10 differs from correct depth-2
    consistent_across_depths: bool  # All depths agree on move
    needs_correction: bool  # Move selection wrong at all depths


@dataclass
class TrainingExample:
    """Training data for Evaluation AI"""
    fen: str
    candidate_moves: List[str]  # From Move Ordering AI (or legal moves)
    v7p3r_eval: int
    stockfish_eval: int
    correct_move: str  # Stockfish best or V7P3R if close
    use_v7p3r: bool  # True if V7P3R eval is accurate
    confidence: float  # 1.0 if perfect match, decreasing with difference


class V7P3RGameExtractor:
    """Extract training positions from V7P3R bot game history"""
    
    def __init__(self,
                 v7p3r_engine_path: str,
                 stockfish_path: str,
                 eval_difference_threshold: int = 100):
        """
        Args:
            v7p3r_engine_path: Path to V7P3R UCI engine
            stockfish_path: Path to Stockfish engine
            eval_difference_threshold: Flag positions with eval diff > this (cp)
        """
        self.v7p3r_engine_path = v7p3r_engine_path
        self.stockfish_path = stockfish_path
        self.threshold = eval_difference_threshold
        
        # Verify engines exist
        if not os.path.exists(v7p3r_engine_path):
            raise FileNotFoundError(f"V7P3R engine not found: {v7p3r_engine_path}")
        if not os.path.exists(stockfish_path):
            raise FileNotFoundError(f"Stockfish not found: {stockfish_path}")
        
        # Initialize Stockfish (keep alive for performance)
        print(f"Starting Stockfish engine...")
        self.stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        print(f"✅ Stockfish ready")
    
    def get_v7p3r_eval_multi_depth(self, fen: str) -> Tuple[Optional[MultiDepthEval], Optional[MultiDepthEval], Optional[MultiDepthEval]]:
        """
        Get V7P3R evaluation at multiple depths
        
        Returns:
            (depth2_eval, depth5_eval, depth10_eval)
        """
        depths = [2, 5, 10]
        evals = []
        
        for depth in depths:
            result = self._get_v7p3r_eval_single(fen, depth)
            if result:
                best_move, eval_score = result
                evals.append(MultiDepthEval(
                    depth=depth,
                    best_move=best_move,
                    eval_score=eval_score,
                    agrees_with_stockfish=False,  # Will be set later
                    agrees_with_shallow=False  # Will be set later
                ))
            else:
                evals.append(None)
        
        return tuple(evals)
    
    def _get_v7p3r_eval_single(self, fen: str, depth: int) -> Optional[Tuple[str, int]]:
        """
        Get V7P3R's best move and eval at a specific depth via UCI
        
        **V18.3 MODULAR EVALUATION**:
        - depth 1-4: FAST mode (Material + PST only)
        - depth 5-8: TACTICAL mode (includes strategic modules)
        - depth 9+: COMPREHENSIVE mode (all evaluation components)
        
        Args:
            fen: FEN string
            depth: Search depth
            
        Returns:
            (best_move, eval_score) or None if error
        """
        try:
            # Run V7P3R UCI engine
            process = subprocess.Popen(
                [sys.executable, self.v7p3r_engine_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # UCI protocol
            commands = [
                "uci",
                f"position fen {fen}",
                f"go depth {depth}",
                "quit"
            ]
            
            stdout, stderr = process.communicate("\n".join(commands), timeout=30)
            
            best_move = None
            eval_score = None
            
            # Parse best move and eval from UCI output
            for line in stdout.split('\n'):
                # Get best move from "bestmove" line
                if line.startswith('bestmove'):
                    parts = line.split()
                    if len(parts) >= 2:
                        best_move = parts[1]
                
                # Get eval from last "info score" line at this depth
                if f'depth {depth}' in line and 'score' in line:
                    parts = line.split()
                    if 'cp' in parts:
                        cp_idx = parts.index('cp')
                        if cp_idx + 1 < len(parts):
                            eval_score = int(parts[cp_idx + 1])
                    elif 'mate' in parts:
                        mate_idx = parts.index('mate')
                        if mate_idx + 1 < len(parts):
                            mate_in = int(parts[mate_idx + 1])
                            eval_score = 10000 - abs(mate_in) * 100 if mate_in > 0 else -10000 + abs(mate_in) * 100
            
            if best_move and eval_score is not None:
                return (best_move, eval_score)
            
            return None
            
        except Exception as e:
            print(f"⚠️  Error getting V7P3R eval at depth {depth}: {e}")
            return None
    
    def get_stockfish_top5(self, fen: str, time_limit: float = 0.5) -> Tuple[List[Tuple[str, int]], str, int]:
        """
        Get Stockfish's top 5 moves with evaluations
        
        Args:
            fen: FEN string
            time_limit: Time per position (seconds)
            
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
                    move = str(analysis['pv'][0])
                    score = analysis.get('score', chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE))
                    
                    # Convert to centipawns
                    if score.is_mate():
                        mate_in = score.white().mate()
                        if mate_in is not None:
                            cp_score = 10000 - abs(mate_in) * 100 if mate_in > 0 else -10000 + abs(mate_in) * 100
                        else:
                            cp_score = 0
                    else:
                        cp_score = score.white().score() if score.white().score() is not None else 0
                    
                    moves_with_scores.append((move, cp_score))
            
            if moves_with_scores:
                best_move = moves_with_scores[0][0]
                best_eval = moves_with_scores[0][1]
                return moves_with_scores, best_move, best_eval
            
            return [], "", 0
            
        except Exception as e:
            print(f"⚠️  Error getting Stockfish eval: {e}")
            return [], "", 0
    
    def extract_positions_from_pgn(self, 
                                   pgn_file: str,
                                   max_positions: Optional[int] = None,
                                   sample_every_n_moves: int = 3) -> List[GamePosition]:
        """
        Extract training positions from PGN file
        
        Args:
            pgn_file: Path to PGN file
            max_positions: Max positions to extract (None = all)
            sample_every_n_moves: Extract every Nth move (avoid correlated positions)
            
        Returns:
            List of GamePosition objects
        """
        positions = []
        
        with open(pgn_file) as f:
            game_count = 0
            
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                
                game_count += 1
                game_id = f"{os.path.basename(pgn_file)}_{game_count}"
                
                board = game.board()
                move_num = 0
                
                for move in game.mainline_moves():
                    move_num += 1
                    
                    # Sample every Nth move
                    if move_num % sample_every_n_moves != 0:
                        board.push(move)
                        continue
                    
                    fen = board.fen()
                    v7p3r_played = str(move)
                    
                    # Get V7P3R multi-depth eval
                    depth2, depth5, depth10 = self.get_v7p3r_eval_multi_depth(fen)
                    
                    # Get Stockfish top 5
                    sf_top5, sf_best, sf_eval = self.get_stockfish_top5(fen)
                    
                    if depth2 and sf_top5:
                        # Set agreement flags
                        if depth2:
                            depth2.agrees_with_stockfish = (depth2.best_move == sf_best)
                            depth2.agrees_with_shallow = True  # Is shallow
                        
                        if depth5:
                            depth5.agrees_with_stockfish = (depth5.best_move == sf_best)
                            depth5.agrees_with_shallow = (depth5.best_move == depth2.best_move)
                        
                        if depth10:
                            depth10.agrees_with_stockfish = (depth10.best_move == sf_best)
                            depth10.agrees_with_shallow = (depth10.best_move == depth2.best_move)
                        
                        # Analyze patterns
                        shallow_is_best = depth2.agrees_with_stockfish if depth2 else False
                        deep_overthinks = (depth10 and not depth10.agrees_with_shallow and depth2.agrees_with_stockfish)
                        consistent = (depth2 and depth5 and depth10 and 
                                    depth2.best_move == depth5.best_move == depth10.best_move)
                        needs_correction = not (depth2.agrees_with_stockfish or 
                                              (depth5 and depth5.agrees_with_stockfish) or
                                              (depth10 and depth10.agrees_with_stockfish))
                        
                        positions.append(GamePosition(
                            fen=fen,
                            game_id=game_id,
                            move_number=move_num,
                            v7p3r_played=v7p3r_played,
                            v7p3r_depth2=depth2,
                            v7p3r_depth5=depth5,
                            v7p3r_depth10=depth10,
                            stockfish_top5=sf_top5,
                            stockfish_best=sf_best,
                            stockfish_eval=sf_eval,
                            shallow_is_best=shallow_is_best,
                            deep_overthinks=deep_overthinks,
                            consistent_across_depths=consistent,
                            needs_correction=needs_correction
                        ))
                        
                        if max_positions and len(positions) >= max_positions:
                            return positions
                    
                    board.push(move)
        
        return positions
    
    def create_training_examples(self, positions: List[GamePosition]) -> List[TrainingExample]:
        """
        Convert GamePositions to TrainingExamples
        
        **Move-Based Strategy** (not centipawn-based):
        - Shallow correct: Use depth-2, confidence 1.0 (fast heuristic works!)
        - Overthinking: Use depth-2, confidence 0.8 (teach AI not to overthink)
        - Consistent + correct: Use any depth, confidence 1.0 (all agree)
        - Needs correction: Use Stockfish, confidence 1.0 (learn from mistake)
        """
        examples = []
        
        for pos in positions:
            # Get candidate moves (top 5 from Stockfish as proxy for Move Ordering AI)
            candidate_moves = [move for move, score in pos.stockfish_top5]
            
            # Determine which depth to learn from based on move correctness
            if pos.shallow_is_best:
                # Depth-2 got it right! Use fast heuristic eval
                use_v7p3r = True
                confidence = 1.0
                correct_move = pos.v7p3r_depth2.best_move
                selected_eval = pos.v7p3r_depth2.eval_score
                learning_signal = "shallow_correct"
                
            elif pos.deep_overthinks:
                # Depth-2 was right, depth-10 changed mind → overthinking!
                # Teach AI to trust fast eval
                use_v7p3r = True
                confidence = 0.8
                correct_move = pos.v7p3r_depth2.best_move  # Use shallow
                selected_eval = pos.v7p3r_depth2.eval_score
                learning_signal = "avoid_overthinking"
                
            elif pos.consistent_across_depths:
                # All depths agree on move
                if pos.v7p3r_depth10 and pos.v7p3r_depth10.agrees_with_stockfish:
                    # Consistent AND correct
                    use_v7p3r = True
                    confidence = 1.0
                    correct_move = pos.v7p3r_depth10.best_move
                    selected_eval = pos.v7p3r_depth10.eval_score
                    learning_signal = "consistent_correct"
                else:
                    # Consistent but wrong - need correction
                    use_v7p3r = False
                    confidence = 1.0
                    correct_move = pos.stockfish_best
                    selected_eval = pos.stockfish_eval
                    learning_signal = "consistent_wrong"
                    
            else:
                # Inconsistent and needs correction
                use_v7p3r = False
                confidence = 1.0
                correct_move = pos.stockfish_best
                selected_eval = pos.stockfish_eval
                learning_signal = "correction_needed"
            
            examples.append(TrainingExample(
                fen=pos.fen,
                candidate_moves=candidate_moves,
                v7p3r_eval=selected_eval,
                stockfish_eval=pos.stockfish_eval,
                correct_move=correct_move,
                use_v7p3r=use_v7p3r,
                confidence=confidence
            ))
        
        return examples
    
    def save_training_data(self, examples: List[TrainingExample], output_path: str):
        """Save training examples to JSON"""
        data = [asdict(ex) for ex in examples]
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Saved {len(examples)} training examples to {output_path}")
    
    def close(self):
        """Cleanup engines"""
        if hasattr(self, 'stockfish'):
            self.stockfish.quit()


def main():
    """Extract training data from V7P3R game history"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract training data from V7P3R games")
    parser.add_argument('--v7p3r-engine', type=str, required=True,
                       help="Path to V7P3R UCI engine")
    parser.add_argument('--stockfish-path', type=str,
                       default=r"E:\Programming Stuff\Chess Engines\Tournament Engines\Stockfish\stockfish-windows-x86-64-avx2.exe",
                       help="Path to Stockfish engine")
    parser.add_argument('--game-dir', type=str, required=True,
                       help="Directory containing PGN files")
    parser.add_argument('--output', type=str, default='data/eval_training_games.json',
                       help="Output file for training data")
    parser.add_argument('--max-positions', type=int, default=5000,
                       help="Maximum positions to extract")
    parser.add_argument('--eval-threshold', type=int, default=100,
                       help="Flag positions with eval diff > this (cp)")
    parser.add_argument('--sample-every', type=int, default=3,
                       help="Sample every Nth move from games")
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = V7P3RGameExtractor(
        v7p3r_engine_path=args.v7p3r_engine,
        stockfish_path=args.stockfish_path,
        eval_difference_threshold=args.eval_threshold
    )
    
    try:
        # Find all PGN files
        game_dir = Path(args.game_dir)
        pgn_files = list(game_dir.glob("*.pgn"))
        
        if not pgn_files:
            print(f"❌ No PGN files found in {game_dir}")
            return
        
        print(f"\n📂 Found {len(pgn_files)} PGN files")
        print(f"🎯 Extracting up to {args.max_positions} positions")
        print(f"📊 Evaluation threshold: {args.eval_threshold}cp")
        print()
        
        # Extract positions from all games
        all_positions = []
        positions_per_file = args.max_positions // len(pgn_files)
        
        for pgn_file in tqdm(pgn_files, desc="Processing PGN files"):
            positions = extractor.extract_positions_from_pgn(
                str(pgn_file),
                max_positions=positions_per_file,
                sample_every_n_moves=args.sample_every
            )
            all_positions.extend(positions)
            
            if len(all_positions) >= args.max_positions:
                all_positions = all_positions[:args.max_positions]
                break
        
        print(f"\n✅ Extracted {len(all_positions)} positions")
        
        # Analyze multi-depth patterns
        shallow_best = sum(1 for p in all_positions if p.shallow_is_best)
        overthinks = sum(1 for p in all_positions if p.deep_overthinks)
        consistent = sum(1 for p in all_positions if p.consistent_across_depths)
        needs_correction = sum(1 for p in all_positions if p.needs_correction)
        
        print(f"\n📊 Multi-Depth Analysis:")
        print(f"   Shallow correct (depth 2 = Stockfish): {shallow_best} ({shallow_best/len(all_positions)*100:.1f}%)")
        print(f"   Overthinks (depth 10 ≠ correct depth 2): {overthinks} ({overthinks/len(all_positions)*100:.1f}%)")
        print(f"   Consistent across depths: {consistent} ({consistent/len(all_positions)*100:.1f}%)")
        print(f"   Needs correction (all depths wrong): {needs_correction} ({needs_correction/len(all_positions)*100:.1f}%)")
        
        # Create training examples
        print(f"\n🔄 Creating training examples...")
        examples = extractor.create_training_examples(all_positions)
        
        # Save
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        extractor.save_training_data(examples, str(output_path))
        
        # Save corrections separately
        corrections = [pos for pos in all_positions if pos.needs_correction]
        if corrections:
            corrections_path = output_path.parent / f"corrections_{output_path.name}"
            corrections_data = [asdict(pos) for pos in corrections]
            with open(corrections_path, 'w') as f:
                json.dump(corrections_data, f, indent=2)
            print(f"✅ Saved {len(corrections)} corrections to {corrections_path}")
        
    finally:
        extractor.close()


if __name__ == "__main__":
    main()
