"""
V7P3R AI v5.0 - Puzzle Analysis Results Extractor
==================================================
Extracts positions from Universal Puzzle Analyzer results for training dataset.

Purpose:
- Read puzzle analysis JSON from universal_puzzle_analyzer.py
- Extract each position where v7p3r was challenged
- Convert to unified training format
- BONUS: Stockfish analysis already included (no need to reanalyze!)
- Add puzzle-specific metadata (theme, difficulty, expected move)

Input Format (from universal_puzzle_analyzer.py):
{
  "analysis_results": [
    {
      "puzzle_id": "00008",
      "rating": 1928,
      "themes": ["crushing", "hangingPiece", "long", "middlegame"],
      "position_analyses": [
        {
          "challenge_fen": "fen string",
          "expected_move": "solution move",
          "engine_move": "v7p3r's move",
          "stockfish_top_moves": [["e2e4", 35], ["d2d4", 28], ...],
          "engine_stockfish_rank": 1,
          "time_analysis": {...}
        }
      ]
    }
  ]
}

Output Format:
- JSONL file in unified training dataset format
- Each line = one position where v7p3r was challenged
- Features block populated separately (run calculate_features.py after)
- Stockfish analysis already complete (derived from puzzle results)

Usage:
    python scripts/extract_puzzle_results.py --input puzzle_analysis_v18_3.json --output data/puzzles/positions.jsonl
    python scripts/extract_puzzle_results.py --input results.json --output positions.jsonl --engine-version "18.3"
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import chess


class PuzzleResultsExtractor:
    """Extract training positions from puzzle analysis results."""
    
    def __init__(self, output_path: Path, engine_version: str = "18.3"):
        """
        Initialize extractor.
        
        Args:
            output_path: Path to output JSONL file
            engine_version: V7P3R version used in puzzle analysis
        """
        self.output_path = output_path
        self.engine_version = engine_version
        self.positions_extracted = 0
        self.puzzles_processed = 0
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Puzzle Results Extractor initialized - output: {output_path}")
    
    def extract_from_file(self, puzzle_results_file: Path) -> None:
        """
        Extract positions from puzzle analysis JSON file.
        
        Args:
            puzzle_results_file: Path to JSON results from universal_puzzle_analyzer.py
        """
        logging.info(f"Processing puzzle results: {puzzle_results_file}")
        
        # Load puzzle analysis results
        with open(puzzle_results_file, 'r', encoding='utf-8') as f:
            puzzle_data = json.load(f)
        
        analysis_results = puzzle_data.get('analysis_results', [])
        total_puzzles = len(analysis_results)
        
        logging.info(f"Found {total_puzzles} puzzle analyses")
        
        # Process each puzzle
        with open(self.output_path, 'w', encoding='utf-8') as outfile:
            for puzzle_result in analysis_results:
                positions = self._extract_from_puzzle(puzzle_result, puzzle_results_file.name)
                
                # Write positions to output
                for position_record in positions:
                    outfile.write(json.dumps(position_record) + '\n')
                
                self.puzzles_processed += 1
                
                if self.puzzles_processed % 100 == 0:
                    logging.info(f"Progress: {self.puzzles_processed}/{total_puzzles} puzzles, "
                               f"{self.positions_extracted} positions extracted")
        
        self._log_summary(total_puzzles)
    
    def _extract_from_puzzle(self, puzzle_result: Dict[str, Any], source_file: str) -> List[Dict[str, Any]]:
        """
        Extract all positions from a single puzzle analysis.
        
        Args:
            puzzle_result: Puzzle analysis result dictionary
            source_file: Name of source JSON file
        
        Returns:
            List of position records
        """
        puzzle_id = puzzle_result.get('puzzle_id', 'unknown')
        puzzle_rating = puzzle_result.get('rating', 0)
        puzzle_themes = puzzle_result.get('themes', [])
        position_analyses = puzzle_result.get('position_analyses', [])
        
        if not position_analyses:
            logging.warning(f"Puzzle {puzzle_id} has no position analyses")
            return []
        
        positions = []
        
        for pos_idx, pos_analysis in enumerate(position_analyses, start=1):
            # Create position record
            position_record = self._create_position_record(
                pos_analysis=pos_analysis,
                puzzle_id=puzzle_id,
                puzzle_rating=puzzle_rating,
                puzzle_themes=puzzle_themes,
                position_index=pos_idx,
                source_file=source_file
            )
            
            positions.append(position_record)
            self.positions_extracted += 1
        
        return positions
    
    def _create_position_record(self, pos_analysis: Dict[str, Any], puzzle_id: str,
                                puzzle_rating: int, puzzle_themes: List[str],
                                position_index: int, source_file: str) -> Dict[str, Any]:
        """
        Create a training record for a single puzzle position.
        
        Follows UNIFIED_TRAINING_DATASET.md schema.
        """
        # Parse position
        fen = pos_analysis.get('challenge_fen', '')
        board = chess.Board(fen)
        
        # Get moves
        engine_move_uci = pos_analysis.get('engine_move', '')
        expected_move_uci = pos_analysis.get('expected_move', '')
        
        # Parse UCI moves
        try:
            engine_move = chess.Move.from_uci(engine_move_uci) if engine_move_uci else None
            expected_move = chess.Move.from_uci(expected_move_uci) if expected_move_uci else None
        except:
            logging.warning(f"Invalid move in puzzle {puzzle_id}: engine={engine_move_uci}, expected={expected_move_uci}")
            engine_move = None
            expected_move = None
        
        # Calculate position properties
        material_count = self._count_material(board)
        game_phase = self._calculate_game_phase(material_count)
        material_balance = self._calculate_material_balance(board)
        
        # Get Stockfish analysis from puzzle results
        stockfish_top_moves = pos_analysis.get('stockfish_top_moves', [])
        engine_rank = pos_analysis.get('engine_stockfish_rank', None)
        
        # Convert rank to grade (0-5 scale)
        if engine_rank is not None and 1 <= engine_rank <= 5:
            grade = 6 - engine_rank  # Rank 1 = Grade 5, Rank 5 = Grade 1
        elif engine_rank == 0:
            grade = 0  # Not in top 5 = Grade 0
        else:
            grade = None
        
        # Create record following unified schema
        record = {
            # METADATA BLOCK
            "metadata": {
                "source": "puzzle_analysis",
                "source_file": source_file,
                "puzzle_id": puzzle_id,
                "position_id": f"{puzzle_id}_pos{position_index}",
                "extraction_timestamp": datetime.now().isoformat(),
                "v7p3r_version": self.engine_version,
                "puzzle_metadata": {
                    "rating": puzzle_rating,
                    "themes": puzzle_themes,
                    "expected_move": expected_move_uci,
                    "position_in_sequence": position_index,
                },
            },
            
            # POSITION BLOCK
            "position": {
                "fen": fen,
                "move_number": position_index,
                "side_to_move": "white" if board.turn == chess.WHITE else "black",
                "game_phase": game_phase,
                "material_count": material_count,
                "material_balance": material_balance,
                "in_check": board.is_check(),
                "castling_rights": board.castling_rights,
                "en_passant_square": board.ep_square,
            },
            
            # ENGINE DECISION BLOCK
            "engine_decision": {
                "move_uci": engine_move_uci,
                "move_san": board.san(engine_move) if engine_move else None,
                "is_capture": board.is_capture(engine_move) if engine_move else False,
                "is_check": board.gives_check(engine_move) if engine_move else False,
                "is_castling": board.is_castling(engine_move) if engine_move else False,
                "is_en_passant": board.is_en_passant(engine_move) if engine_move else False,
                "promotion": engine_move.promotion if (engine_move and engine_move.promotion) else None,
                # Time analysis from puzzle results
                "time_ms": pos_analysis.get('time_analysis', {}).get('actual_time_ms'),
                "v7p3r_eval_cp": None,  # Not available in puzzle results
                "search_depth": None,
                "nodes_searched": None,
            },
            
            # STOCKFISH ANALYSIS BLOCK (already available from puzzle results!)
            "stockfish_analysis": self._create_stockfish_analysis(
                stockfish_top_moves=stockfish_top_moves,
                engine_move=engine_move_uci,
                grade=grade,
                rank=engine_rank,
                board=board
            ),
            
            # FEATURES BLOCK (populated by feature calculator)
            "features": None,
        }
        
        return record
    
    def _create_stockfish_analysis(self, stockfish_top_moves: List[List], 
                                   engine_move: str, grade: Optional[int],
                                   rank: Optional[int], board: chess.Board) -> Dict[str, Any]:
        """
        Create stockfish_analysis block from puzzle results.
        
        Args:
            stockfish_top_moves: List of [move_uci, eval_cp] pairs
            engine_move: Move played by engine
            grade: Move quality grade (0-5)
            rank: Rank in top-5 (1-5 or None)
            board: Chess board for SAN conversion
        """
        # Convert stockfish moves to proper format
        top_moves = []
        for i, move_data in enumerate(stockfish_top_moves[:5], start=1):
            if len(move_data) >= 2:
                move_uci = move_data[0]
                eval_cp = move_data[1]
                
                try:
                    move = chess.Move.from_uci(move_uci)
                    san = board.san(move)
                except:
                    san = move_uci  # Fallback to UCI if SAN conversion fails
                
                top_moves.append({
                    "rank": i,
                    "uci": move_uci,
                    "san": san,
                    "eval_cp": eval_cp,
                    "pv": [move_uci],  # Only first move available from puzzle results
                })
        
        # Calculate eval drop
        eval_drop = None
        if top_moves and grade is not None and rank is not None and rank > 0:
            best_eval = top_moves[0]['eval_cp']
            played_eval = top_moves[rank - 1]['eval_cp'] if rank <= len(top_moves) else None
            
            if played_eval is not None:
                # Calculate drop (positive = worse for player to move)
                if board.turn == chess.WHITE:
                    eval_drop = best_eval - played_eval
                else:
                    eval_drop = played_eval - best_eval
        
        return {
            "stockfish_version": "16",  # From universal_puzzle_analyzer.py
            "analysis_depth": 20,  # Default depth used in puzzle analyzer
            "top_moves": top_moves,
            "played_move_rank": rank,
            "move_quality_grade": grade,
            "eval_drop_cp": eval_drop,
            "best_move_uci": top_moves[0]['uci'] if top_moves else None,
            "best_move_eval_cp": top_moves[0]['eval_cp'] if top_moves else None,
        }
    
    def _count_material(self, board: chess.Board) -> int:
        """Count total material on board (pawn units)."""
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0,
        }
        
        total = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                total += piece_values.get(piece.piece_type, 0)
        
        return total
    
    def _calculate_game_phase(self, material_count: int) -> str:
        """
        Determine game phase based on material.
        
        Opening: >28 material
        Middlegame: 14-28 material
        Endgame: <14 material
        """
        if material_count > 28:
            return "opening"
        elif material_count >= 14:
            return "middlegame"
        else:
            return "endgame"
    
    def _calculate_material_balance(self, board: chess.Board) -> int:
        """Calculate material balance (positive = white ahead) in centipawns."""
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 300,
            chess.BISHOP: 300,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0,
        }
        
        white_material = 0
        black_material = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values.get(piece.piece_type, 0)
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value
        
        return white_material - black_material
    
    def _log_summary(self, total_puzzles: int) -> None:
        """Log extraction summary statistics."""
        logging.info("=" * 60)
        logging.info("PUZZLE EXTRACTION COMPLETE")
        logging.info(f"Puzzles processed: {self.puzzles_processed}/{total_puzzles}")
        logging.info(f"Positions extracted: {self.positions_extracted}")
        logging.info(f"Avg positions/puzzle: {self.positions_extracted/max(self.puzzles_processed,1):.1f}")
        logging.info(f"Output file: {self.output_path}")
        logging.info(f"File size: {self.output_path.stat().st_size / 1024:.2f} KB")
        logging.info("=" * 60)


def main():
    """Main entry point for puzzle results extraction."""
    parser = argparse.ArgumentParser(
        description="Extract training positions from puzzle analysis results"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSON file from universal_puzzle_analyzer.py"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--engine-version",
        type=str,
        default="18.3",
        help="V7P3R engine version used in analysis (default: 18.3)"
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
    
    # Validate input file exists
    if not args.input.exists():
        logging.error(f"Input file not found: {args.input}")
        return 1
    
    # Create extractor
    extractor = PuzzleResultsExtractor(
        output_path=args.output,
        engine_version=args.engine_version
    )
    
    # Extract positions
    extractor.extract_from_file(args.input)
    
    logging.info(f"\n✅ Puzzle positions ready for feature calculation!")
    logging.info(f"Next step: python scripts/calculate_features.py --input {args.output} --output positions_with_features.jsonl")
    
    return 0


if __name__ == "__main__":
    exit(main())
