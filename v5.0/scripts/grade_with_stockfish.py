"""
V7P3R AI v5.0 - Stockfish Move Grader
======================================
Analyzes positions with Stockfish and grades move quality.

Purpose:
- Read JSONL positions (with features calculated)
- Run Stockfish analysis on each position (depth 20, multipv 5)
- Grade v7p3r's move on 0-5 scale based on rank in top-5
- Add 'stockfish_analysis' block to each record
- Output fully analyzed JSONL ready for training

Grading Scale:
- 5 (Excellent): Move is #1 best move
- 4 (Good): Move is #2 best move
- 3 (Decent): Move is #3 best move
- 2 (Suboptimal): Move is #4 best move
- 1 (Poor): Move is #5 best move
- 0 (Blunder): Move not in top-5

Usage:
    python scripts/grade_with_stockfish.py --input positions_with_features.jsonl --output training_dataset.jsonl
    python scripts/grade_with_stockfish.py --input data.jsonl --output graded.jsonl --stockfish-path ./stockfish --depth 20
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import subprocess
import time

import chess
import chess.engine


class StockfishGrader:
    """Grade chess moves using Stockfish engine analysis."""
    
    def __init__(self, stockfish_path: str, depth: int = 20, multipv: int = 5, 
                 time_limit: float = 10.0):
        """
        Initialize Stockfish grader.
        
        Args:
            stockfish_path: Path to Stockfish executable
            depth: Analysis depth (default: 20)
            multipv: Number of variations to analyze (default: 5)
            time_limit: Maximum time per position in seconds (default: 10.0)
        """
        self.stockfish_path = stockfish_path
        self.depth = depth
        self.multipv = multipv
        self.time_limit = time_limit
        
        self.positions_graded = 0
        self.analysis_errors = 0
        
        # Initialize engine
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            logging.info(f"Stockfish engine initialized: {stockfish_path}")
            logging.info(f"Analysis settings: depth={depth}, multipv={multipv}, time_limit={time_limit}s")
        except Exception as e:
            logging.error(f"Failed to initialize Stockfish: {e}")
            raise
    
    def __del__(self):
        """Cleanup engine on deletion."""
        if hasattr(self, 'engine'):
            try:
                self.engine.quit()
            except:
                pass
    
    def process_file(self, input_path: Path, output_path: Path) -> None:
        """
        Process JSONL file and add Stockfish analysis to each record.
        
        Args:
            input_path: Input JSONL file (positions with features)
            output_path: Output JSONL file (positions with Stockfish analysis)
        """
        logging.info(f"Processing: {input_path}")
        logging.info(f"Output: {output_path}")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, start=1):
                try:
                    record = json.loads(line)
                    
                    # Analyze position and grade move
                    stockfish_analysis = self.analyze_and_grade(record)
                    
                    # Add stockfish_analysis block to record
                    record['stockfish_analysis'] = stockfish_analysis
                    
                    # Write analyzed record
                    outfile.write(json.dumps(record) + '\n')
                    
                    self.positions_graded += 1
                    
                    if self.positions_graded % 100 == 0:
                        elapsed = time.time() - start_time
                        rate = self.positions_graded / elapsed
                        logging.info(f"Graded {self.positions_graded} positions "
                                   f"({rate:.1f} pos/sec, {self.analysis_errors} errors)")
                
                except Exception as e:
                    logging.error(f"Error processing line {line_num}: {e}")
                    self.analysis_errors += 1
                    continue
        
        elapsed = time.time() - start_time
        logging.info(f"Complete! Graded {self.positions_graded} positions in {elapsed:.1f}s "
                    f"({self.positions_graded/elapsed:.1f} pos/sec)")
        logging.info(f"Analysis errors: {self.analysis_errors}")
    
    def analyze_and_grade(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze position with Stockfish and grade the move.
        
        Args:
            record: Position record from JSONL
        
        Returns:
            Stockfish analysis dictionary with top-5 moves and quality grade
        """
        # Parse position
        fen = record['position']['fen']
        board = chess.Board(fen)
        
        # Get the move that was played
        move_uci = record['engine_decision']['move_uci']
        played_move = chess.Move.from_uci(move_uci)
        
        try:
            # Analyze with Stockfish
            analysis = self.engine.analyse(
                board,
                chess.engine.Limit(depth=self.depth, time=self.time_limit),
                multipv=self.multipv
            )
            
            # Extract top-5 moves
            top_moves = self._extract_top_moves(analysis, board)
            
            # Grade the played move
            grade, rank = self._grade_move(played_move, top_moves)
            
            # Calculate evaluation drop
            eval_drop = self._calculate_eval_drop(played_move, top_moves, board)
            
            # Build stockfish_analysis block
            stockfish_analysis = {
                "stockfish_version": "16",  # Update as needed
                "analysis_depth": self.depth,
                "top_moves": top_moves,
                "played_move_rank": rank,
                "move_quality_grade": grade,
                "eval_drop_cp": eval_drop,
                "best_move_uci": top_moves[0]['uci'] if top_moves else None,
                "best_move_eval_cp": top_moves[0]['eval_cp'] if top_moves else None,
            }
            
            return stockfish_analysis
        
        except Exception as e:
            logging.error(f"Stockfish analysis failed for position {fen}: {e}")
            return self._create_error_analysis()
    
    def _extract_top_moves(self, analysis: List, board: chess.Board) -> List[Dict[str, Any]]:
        """
        Extract top-5 moves from Stockfish analysis.
        
        Args:
            analysis: Stockfish analysis result (list of info dicts)
            board: Chess board
        
        Returns:
            List of top move dictionaries
        """
        top_moves = []
        
        for i, info in enumerate(analysis[:self.multipv], start=1):
            # Get best move from PV
            pv = info.get('pv', [])
            if not pv:
                continue
            
            move = pv[0]
            
            # Get evaluation
            score = info.get('score')
            if score:
                # Convert score to centipawns (from perspective of side to move)
                relative_score = score.relative
                if relative_score.is_mate():
                    # Mate score: convert to large CP value
                    mate_in = relative_score.mate()
                    eval_cp = 10000 - abs(mate_in) * 100 if mate_in > 0 else -10000 + abs(mate_in) * 100
                else:
                    eval_cp = relative_score.score(mate_score=10000)
            else:
                eval_cp = 0
            
            top_moves.append({
                "rank": i,
                "uci": move.uci(),
                "san": board.san(move),
                "eval_cp": eval_cp,
                "pv": [m.uci() for m in pv[:5]],  # First 5 moves of PV
            })
        
        return top_moves
    
    def _grade_move(self, played_move: chess.Move, top_moves: List[Dict[str, Any]]) -> Tuple[int, Optional[int]]:
        """
        Grade the played move based on its rank in top-5.
        
        Args:
            played_move: Move that was played
            top_moves: List of top moves from Stockfish
        
        Returns:
            Tuple of (grade 0-5, rank in top-5 or None)
        """
        # Find rank of played move
        for top_move in top_moves:
            if top_move['uci'] == played_move.uci():
                rank = top_move['rank']
                # Grade: 6 - rank (so rank 1 = grade 5, rank 5 = grade 1)
                grade = 6 - rank
                return grade, rank
        
        # Move not in top-5 = blunder
        return 0, None
    
    def _calculate_eval_drop(self, played_move: chess.Move, top_moves: List[Dict[str, Any]], 
                            board: chess.Board) -> Optional[int]:
        """
        Calculate evaluation drop from best move to played move.
        
        Args:
            played_move: Move that was played
            top_moves: List of top moves from Stockfish
            board: Chess board
        
        Returns:
            Evaluation drop in centipawns (positive = loss, negative = gain)
        """
        if not top_moves:
            return None
        
        best_eval = top_moves[0]['eval_cp']
        
        # Find eval of played move
        played_eval = None
        for top_move in top_moves:
            if top_move['uci'] == played_move.uci():
                played_eval = top_move['eval_cp']
                break
        
        if played_eval is None:
            # Move not in top-5, need to analyze it separately
            # For now, return None (could enhance to do separate analysis)
            return None
        
        # Calculate drop (positive = worse for player to move)
        if board.turn == chess.WHITE:
            eval_drop = best_eval - played_eval
        else:
            eval_drop = played_eval - best_eval
        
        return eval_drop
    
    def _create_error_analysis(self) -> Dict[str, Any]:
        """Create placeholder analysis for positions that failed."""
        return {
            "stockfish_version": "16",
            "analysis_depth": self.depth,
            "top_moves": [],
            "played_move_rank": None,
            "move_quality_grade": None,
            "eval_drop_cp": None,
            "best_move_uci": None,
            "best_move_eval_cp": None,
            "analysis_failed": True,
        }


def main():
    """Main entry point for Stockfish grading."""
    parser = argparse.ArgumentParser(
        description="Grade chess moves using Stockfish analysis"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL file (positions with features)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file (training dataset with Stockfish grades)"
    )
    parser.add_argument(
        "--stockfish-path",
        type=str,
        default="stockfish",
        help="Path to Stockfish executable (default: stockfish)"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=20,
        help="Stockfish analysis depth (default: 20)"
    )
    parser.add_argument(
        "--multipv",
        type=int,
        default=5,
        help="Number of variations to analyze (default: 5)"
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=10.0,
        help="Maximum time per position in seconds (default: 10.0)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create grader
    grader = StockfishGrader(
        stockfish_path=args.stockfish_path,
        depth=args.depth,
        multipv=args.multipv,
        time_limit=args.time_limit
    )
    
    # Process file
    grader.process_file(args.input, args.output)


if __name__ == "__main__":
    main()
