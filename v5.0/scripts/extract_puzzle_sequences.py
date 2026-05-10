"""
V7P3R AI v5.1 - Puzzle Sequence Extractor
==========================================
Extracts multi-move sequences from puzzle database with temporal features.

Puzzles provide the PERFECT data for temporal learning:
- Correct multi-move PVs (solution lines)
- Forced sequences with clear tactical themes
- Historical context can be properly populated

Process:
1. Load puzzle database (Lichess format JSONL)
2. For each puzzle:
   - Parse starting FEN and solution moves
   - Generate position sequence with temporal context
   - Calculate features for each position with history
3. Output enhanced JSONL with temporal features

Input Format (Lichess puzzles):
{
  "PuzzleId": "00008",
  "FEN": "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24",
  "Moves": "f2g3 e6e7 b2b1 b3c1 b1c1 h6c1",
  "Rating": 1678,
  "RatingDeviation": 74,
  "Popularity": 88,
  "NbPlays": 807,
  "Themes": "crushing hangingPiece long middlegame",
  "GameUrl": "https://lichess.org/PGRExT9w/black#48"
}

Output Format (Enhanced JSONL):
{
  "fen": "...",
  "previous_fen": "...",
  "move_uci": "f2g3",
  "sequence_id": "puzzle_00008_line_1",
  "sequence_index": 1,
  "has_history": 1,
  "puzzle_id": "00008",
  "puzzle_rating": 1678,
  "puzzle_themes": ["crushing", "hangingPiece", "long"],
  "features": {
    // F000-F114: Current features
    // F200-F220: Temporal features (properly populated)
  }
}

Usage:
    python scripts/extract_puzzle_sequences.py \\
        --input data/puzzles/lichess_db_puzzle.jsonl \\
        --output data/puzzles/puzzle_sequences_with_features.jsonl \\
        --limit 20000
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import chess

# Import our feature calculators
from calculate_features import FeatureCalculator, FeatureConfig
from temporal_feature_calculator import TemporalFeatureCalculator


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_puzzles(filepath: Path, limit: int = None) -> List[Dict]:
    """Load puzzles from JSONL file."""
    puzzles = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            puzzles.append(json.loads(line))
    return puzzles


def extract_sequence_from_puzzle(
    puzzle: Dict,
    base_calculator: FeatureCalculator,
    temporal_calculator: TemporalFeatureCalculator,
    logger: logging.Logger
) -> List[Dict]:
    """
    Extract training positions from a single puzzle with temporal context.
    
    Args:
        puzzle: Puzzle data with FEN and Moves
        base_calculator: Calculator for current features
        temporal_calculator: Calculator for temporal features
        logger: Logger instance
        
    Returns:
        List of position records with features
    """
    try:
        board = chess.Board(puzzle['FEN'])
        moves = puzzle['Moves'].split()
        puzzle_id = puzzle['PuzzleId']
        
        positions = []
        previous_fen = None
        previous_move = None
        
        for seq_idx, move_uci in enumerate(moves):
            current_fen = board.fen()
            
            # Calculate features with temporal context
            try:
                features = temporal_calculator.calculate_temporal_features(
                    current_fen=current_fen,
                    previous_fen=previous_fen,
                    last_move_uci=previous_move,
                    sequence_index=seq_idx,
                    stockfish_eval=0.0  # Puzzles don't have evals (winning for player)
                )
                
                # Create position record
                position = {
                    'fen': current_fen,
                    'previous_fen': previous_fen,
                    'move_uci': move_uci,
                    'sequence_id': f"puzzle_{puzzle_id}_line_1",
                    'sequence_index': seq_idx,
                    'has_history': 1 if previous_fen else 0,
                    'puzzle_id': puzzle_id,
                    'puzzle_rating': puzzle.get('Rating', 0),
                    'puzzle_themes': puzzle.get('Themes', '').split(),
                    'source': 'lichess_puzzle',
                    'features': features
                }
                
                positions.append(position)
                
                # Make move and update for next iteration
                move = chess.Move.from_uci(move_uci)
                board.push(move)
                previous_fen = current_fen
                previous_move = move_uci
                
            except Exception as e:
                logger.warning(f"Feature calculation failed for puzzle {puzzle_id} move {seq_idx}: {e}")
                continue
        
        return positions
        
    except Exception as e:
        logger.error(f"Failed to process puzzle {puzzle.get('PuzzleId', 'unknown')}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Extract multi-move sequences from puzzle database with temporal features"
    )
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input puzzle JSONL file (Lichess format)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output JSONL with position sequences and features'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of puzzles to process (default: all)'
    )
    parser.add_argument(
        '--feature-set',
        choices=['minimal', 'standard', 'full'],
        default='standard',
        help='Feature set to calculate (default: standard)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate input
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize feature calculators
    logger.info("Initializing feature calculators...")
    feature_config = FeatureConfig.from_preset(args.feature_set)
    base_calculator = FeatureCalculator(feature_config)
    temporal_calculator = TemporalFeatureCalculator(base_calculator)
    
    # Load puzzles
    logger.info(f"Loading puzzles from {args.input}...")
    puzzles = load_puzzles(args.input, limit=args.limit)
    logger.info(f"Loaded {len(puzzles)} puzzles")
    
    # Process puzzles
    logger.info("Extracting sequences with temporal features...")
    total_positions = 0
    processed_puzzles = 0
    
    with open(args.output, 'w', encoding='utf-8') as outfile:
        for i, puzzle in enumerate(puzzles):
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i+1}/{len(puzzles)} puzzles ({total_positions} positions)")
                # Clear cache periodically to avoid memory issues
                if (i + 1) % 1000 == 0:
                    temporal_calculator.clear_cache()
            
            positions = extract_sequence_from_puzzle(
                puzzle, base_calculator, temporal_calculator, logger
            )
            
            if positions:
                for pos in positions:
                    outfile.write(json.dumps(pos) + '\n')
                total_positions += len(positions)
                processed_puzzles += 1
    
    # Final statistics
    logger.info("=" * 60)
    logger.info("Puzzle Sequence Extraction Complete")
    logger.info("=" * 60)
    logger.info(f"Puzzles processed: {processed_puzzles}/{len(puzzles)}")
    logger.info(f"Total positions extracted: {total_positions}")
    logger.info(f"Average positions per puzzle: {total_positions/processed_puzzles:.1f}")
    logger.info(f"Output written to: {args.output}")
    logger.info("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
