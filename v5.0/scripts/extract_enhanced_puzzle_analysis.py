"""
Extract training data from enhanced puzzle sequence analysis JSON files.

This script handles the enhanced analysis format with comprehensive Stockfish comparisons
and converts it to the unified training dataset format.

Input Format: Enhanced sequence analysis JSON (from universal_puzzle_analyzer.py v2+)
Output Format: Unified JSONL training dataset

Usage:
    python scripts/extract_enhanced_puzzle_analysis.py \
        --input "path/to/V7P3R_v18_3_enhanced_sequence_analysis_*.json" \
        --output "data/puzzles/enhanced_puzzles_extracted.jsonl"
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedPuzzleExtractor:
    """Extract training positions from enhanced puzzle analysis JSON."""
    
    def __init__(self, input_file: str, output_file: str):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'puzzles_processed': 0,
            'positions_extracted': 0,
            'perfect_sequences': 0,
            'positions_with_stockfish': 0,
            'positions_without_stockfish': 0,
            'extraction_errors': 0
        }
        
    def extract(self):
        """Extract all positions from enhanced puzzle analysis file."""
        logger.info(f"Loading enhanced puzzle analysis from {self.input_file}")
        
        with open(self.input_file, 'r') as f:
            data = json.load(f)
        
        analysis_results = data.get('analysis_results', [])
        metadata = data.get('metadata', {})
        
        logger.info(f"Found {len(analysis_results)} puzzles in analysis")
        
        with open(self.output_file, 'w') as out_f:
            for puzzle in analysis_results:
                try:
                    positions = self._extract_puzzle_positions(puzzle, metadata)
                    for position in positions:
                        out_f.write(json.dumps(position) + '\n')
                        self.stats['positions_extracted'] += 1
                    
                    self.stats['puzzles_processed'] += 1
                    
                    if puzzle.get('puzzle_solved', False):
                        self.stats['perfect_sequences'] += 1
                    
                    if self.stats['puzzles_processed'] % 100 == 0:
                        logger.info(f"Processed {self.stats['puzzles_processed']} puzzles, "
                                  f"extracted {self.stats['positions_extracted']} positions")
                
                except Exception as e:
                    logger.error(f"Error processing puzzle {puzzle.get('puzzle_id', 'unknown')}: {e}")
                    self.stats['extraction_errors'] += 1
        
        logger.info(f"Extraction complete! Extracted {self.stats['positions_extracted']} positions "
                   f"from {self.stats['puzzles_processed']} puzzles")
        
        return self.stats
    
    def _extract_puzzle_positions(self, puzzle: Dict[str, Any], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract training records from a single puzzle's positions."""
        positions = []
        
        puzzle_id = puzzle.get('puzzle_id', 'unknown')
        rating = puzzle.get('rating', 0)
        themes = puzzle.get('themes', [])
        position_sequence = puzzle.get('position_sequence', [])
        
        for pos_data in position_sequence:
            try:
                position_record = self._create_position_record(
                    puzzle_id=puzzle_id,
                    rating=rating,
                    themes=themes,
                    pos_data=pos_data,
                    puzzle=puzzle,
                    metadata=metadata
                )
                positions.append(position_record)
            except Exception as e:
                logger.warning(f"Error creating record for puzzle {puzzle_id}, "
                             f"position {pos_data.get('position_number', '?')}: {e}")
        
        return positions
    
    def _create_position_record(self, puzzle_id: str, rating: int, themes: List[str],
                                pos_data: Dict[str, Any], puzzle: Dict[str, Any],
                                metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create a unified training record for a single position."""
        
        # Extract position details
        challenge_fen = pos_data.get('challenge_fen', '')
        position_number = pos_data.get('position_number', 1)
        turn_info = pos_data.get('turn_info', {})
        
        # Extract engine decision
        engine_move = pos_data.get('engine_move', '')
        expected_move = pos_data.get('expected_move', '')
        engine_found_solution = pos_data.get('engine_found_solution', False)
        
        # Extract Stockfish analysis
        stockfish_top_moves = pos_data.get('stockfish_top_moves', [])
        engine_stockfish_rank = pos_data.get('engine_stockfish_rank', None)
        
        # Extract time data
        engine_time_seconds = pos_data.get('engine_time_seconds', 0.0)
        suggested_time = pos_data.get('suggested_time', 0.0)
        
        # Create unified record following UNIFIED_TRAINING_DATASET.md schema
        record = {
            'metadata': {
                'source': 'enhanced_puzzle_analysis',
                'source_file': self.input_file.name,
                'puzzle_id': puzzle_id,
                'position_id': f"{puzzle_id}_pos{position_number}",
                'extraction_timestamp': datetime.now().isoformat(),
                'v7p3r_version': metadata.get('engine_name', 'v18.3').replace('V7P3R ', '').replace('v', ''),
                'puzzle_metadata': {
                    'rating': rating,
                    'themes': themes,
                    'position_number': position_number,
                    'total_positions': len(puzzle.get('position_sequence', [])),
                    'puzzle_solved': puzzle.get('puzzle_solved', False),
                    'sequence_accuracy': puzzle.get('sequence_accuracy', 0.0),
                    'puzzle_url': puzzle.get('puzzle_url', '')
                }
            },
            
            'position': self._create_position_block(challenge_fen, turn_info),
            
            'engine_decision': {
                'move_uci': self._convert_to_uci(engine_move, challenge_fen),
                'move_san': engine_move,
                'is_capture': self._is_capture_move(engine_move),
                'is_check': self._is_check_move(engine_move),
                'is_castling': engine_move in ['O-O', 'O-O-O'],
                'is_en_passant': False,  # Could parse from move
                'promotion': self._get_promotion(engine_move),
                'v7p3r_eval_cp': None,
                'search_depth': None,
                'nodes_searched': None,
                'time_ms': int(engine_time_seconds * 1000) if engine_time_seconds else None
            },
            
            'stockfish_analysis': self._create_stockfish_analysis(
                stockfish_top_moves=stockfish_top_moves,
                engine_move=engine_move,
                expected_move=expected_move,
                engine_stockfish_rank=engine_stockfish_rank,
                engine_found_solution=engine_found_solution
            ),
            
            'features': {}  # Will be populated by calculate_features.py
        }
        
        return record
    
    def _create_position_block(self, fen: str, turn_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create position block from FEN and turn info."""
        import chess
        
        try:
            board = chess.Board(fen)
            
            # Determine game phase
            piece_count = len(board.piece_map())
            if piece_count >= 28:
                phase = 'opening'
            elif piece_count >= 12:
                phase = 'middlegame'
            else:
                phase = 'endgame'
            
            # Calculate material balance (simplified)
            piece_values = {'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 0}
            material = 0
            for piece in board.piece_map().values():
                value = piece_values.get(piece.symbol().upper(), 0)
                material += value if piece.color == chess.WHITE else -value
            
            return {
                'fen': fen,
                'move_number': turn_info.get('move_number', 1),
                'side_to_move': 'white' if board.turn == chess.WHITE else 'black',
                'game_phase': phase,
                'material_count': piece_count,
                'material_balance': material,
                'in_check': board.is_check(),
                'castling_rights': board.castling_rights,
                'en_passant_square': board.ep_square if board.ep_square else None
            }
        except Exception as e:
            logger.warning(f"Error parsing FEN {fen}: {e}")
            return {
                'fen': fen,
                'move_number': 1,
                'side_to_move': 'white',
                'game_phase': 'unknown',
                'material_count': 0,
                'material_balance': 0,
                'in_check': False,
                'castling_rights': 0,
                'en_passant_square': None
            }
    
    def _create_stockfish_analysis(self, stockfish_top_moves: List[List], engine_move: str,
                                   expected_move: str, engine_stockfish_rank: Any,
                                   engine_found_solution: bool) -> Dict[str, Any]:
        """Create Stockfish analysis block from enhanced puzzle data."""
        
        if not stockfish_top_moves:
            self.stats['positions_without_stockfish'] += 1
            return None
        
        self.stats['positions_with_stockfish'] += 1
        
        # Parse Stockfish top moves (format: [move, eval_cp] or [move, "mate in X"])
        top_moves = []
        for rank, move_data in enumerate(stockfish_top_moves[:5], 1):
            if len(move_data) >= 2:
                move_uci = move_data[0]
                eval_str = move_data[1]
                
                # Parse evaluation
                eval_cp = None
                eval_mate = None
                if isinstance(eval_str, (int, float)):
                    eval_cp = int(eval_str)
                elif isinstance(eval_str, str) and 'mate' in eval_str.lower():
                    # Extract mate distance
                    try:
                        mate_dist = int(eval_str.split()[-1])
                        eval_mate = mate_dist
                    except:
                        eval_cp = 0
                else:
                    try:
                        eval_cp = int(eval_str)
                    except:
                        eval_cp = 0
                
                top_moves.append({
                    'rank': rank,
                    'uci': move_uci,
                    'san': move_uci,  # Would need board to convert properly
                    'eval_cp': eval_cp,
                    'eval_mate': eval_mate,
                    'pv': []  # Not available in this format
                })
        
        # Determine move quality grade (0-5 scale)
        grade = self._calculate_move_grade(engine_stockfish_rank, engine_found_solution)
        
        # Best move details
        best_move_uci = top_moves[0]['uci'] if top_moves else None
        best_move_eval_cp = top_moves[0]['eval_cp'] if top_moves else None
        best_move_eval_mate = top_moves[0]['eval_mate'] if top_moves else None
        
        # Eval drop (if engine move is in top 5)
        eval_drop_cp = 0
        if engine_stockfish_rank and isinstance(engine_stockfish_rank, int) and engine_stockfish_rank <= 5:
            try:
                engine_eval = top_moves[engine_stockfish_rank - 1]['eval_cp']
                best_eval = best_move_eval_cp
                if engine_eval is not None and best_eval is not None:
                    eval_drop_cp = abs(best_eval - engine_eval)
            except:
                eval_drop_cp = 0
        
        return {
            'stockfish_version': '16',  # Assuming Stockfish 16
            'analysis_depth': 20,  # Enhanced analysis typically uses depth 20
            'top_moves': top_moves,
            'played_move_rank': engine_stockfish_rank if isinstance(engine_stockfish_rank, int) else None,
            'move_quality_grade': grade,
            'eval_drop_cp': eval_drop_cp,
            'best_move_uci': best_move_uci,
            'best_move_eval_cp': best_move_eval_cp,
            'best_move_eval_mate': best_move_eval_mate
        }
    
    def _calculate_move_grade(self, rank: Any, found_solution: bool) -> int:
        """Calculate 0-5 move quality grade from Stockfish ranking."""
        if found_solution and rank == 1:
            return 5  # Best move
        elif rank == 1:
            return 5
        elif rank == 2:
            return 4
        elif rank == 3:
            return 3
        elif rank == 4:
            return 2
        elif rank == 5:
            return 1
        else:
            return 0  # Not in top 5
    
    def _convert_to_uci(self, san_move: str, fen: str) -> str:
        """Convert SAN move to UCI notation."""
        try:
            import chess
            board = chess.Board(fen)
            move = board.parse_san(san_move)
            return move.uci()
        except:
            return san_move  # Return as-is if conversion fails
    
    def _is_capture_move(self, move: str) -> bool:
        """Check if move is a capture (simple SAN parsing)."""
        return 'x' in move
    
    def _is_check_move(self, move: str) -> bool:
        """Check if move gives check."""
        return '+' in move or '#' in move
    
    def _get_promotion(self, move: str) -> str:
        """Extract promotion piece from SAN move."""
        if '=' in move:
            promo = move.split('=')[1][0].lower()
            return promo if promo in ['q', 'r', 'b', 'n'] else None
        return None
    
    def save_stats(self):
        """Save extraction statistics."""
        stats_file = self.output_file.parent / f"{self.output_file.stem}_stats.json"
        
        stats = {
            **self.stats,
            'extraction_timestamp': datetime.now().isoformat(),
            'input_file': str(self.input_file),
            'output_file': str(self.output_file),
            'stockfish_coverage': f"{self.stats['positions_with_stockfish']}/{self.stats['positions_extracted']}"
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Statistics saved to {stats_file}")
        
        return stats


def main():
    parser = argparse.ArgumentParser(description='Extract training data from enhanced puzzle analysis')
    parser.add_argument('--input', required=True, help='Input enhanced analysis JSON file')
    parser.add_argument('--output', required=True, help='Output JSONL file')
    
    args = parser.parse_args()
    
    extractor = EnhancedPuzzleExtractor(args.input, args.output)
    stats = extractor.extract()
    extractor.save_stats()
    
    # Print summary
    print("\n" + "="*60)
    print("ENHANCED PUZZLE EXTRACTION COMPLETE")
    print("="*60)
    print(f"Puzzles processed: {stats['puzzles_processed']}")
    print(f"Positions extracted: {stats['positions_extracted']}")
    print(f"Perfect sequences: {stats['perfect_sequences']}")
    print(f"Positions with Stockfish: {stats['positions_with_stockfish']}")
    print(f"Extraction errors: {stats['extraction_errors']}")
    print("="*60)


if __name__ == '__main__':
    main()
