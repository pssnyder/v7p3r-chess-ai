"""
V7P3R AI v5.3 - Lichess Puzzle Extractor (CSV → Training Data)
================================================================
Extracts puzzle solutions from 4M Lichess puzzle database (CSV format)

Key Insight:
- Puzzle SOLUTIONS are correct moves by definition (grade 0-2)
- Don't need Stockfish grading - Lichess already validated these
- Multi-move sequences provide temporal context
- Rating indicates difficulty (use for stratification)

Process:
1. Load puzzles from CSV (PuzzleId, FEN, Moves, Rating, Themes)
2. For each puzzle:
   - Parse solution moves
   - Generate position sequence
   - Calculate features (current + temporal)
   - Assign grade 0 to puzzle solutions (correct by definition)
3. Output JSONL with features ready for preprocessing

Output Format:
{
  "fen": "...",
  "previous_fen": "...",
  "move_uci": "e6e7",
  "stockfish_analysis": {
    "grade": 0,  # Puzzle solutions are correct moves
    "evaluation": 300,
    "best_move": "e6e7",
    "pv": ["e6e7", "b2b1", ...]
  },
  "source": "lichess_puzzle",
  "puzzle_id": "00008",
  "puzzle_rating": 1928,
  "puzzle_themes": "crushing hangingPiece long middlegame",
  "sequence_id": "puzzle_00008",
  "has_history": 1,
  "features": {...}  # F000-F220
}
"""

import pandas as pd
import chess
import json
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import sys

# Import feature calculators
sys.path.insert(0, str(Path(__file__).parent))
from calculate_features import FeatureCalculator, FeatureConfig
from temporal_feature_calculator import TemporalFeatureCalculator


class LichessPuzzleExtractor:
    """Extract training data from Lichess puzzle database"""
    
    def __init__(self):
        # Create feature config with all features enabled
        config = FeatureConfig.from_preset("full")
        self.feature_calc = FeatureCalculator(config)
        self.temporal_calc = TemporalFeatureCalculator(self.feature_calc)
        
    def extract_puzzle_sequence(self, puzzle_row: pd.Series) -> List[Dict]:
        """
        Extract training positions from a single puzzle
        
        Args:
            puzzle_row: Pandas Series with columns: PuzzleId, FEN, Moves, Rating, Themes
        
        Returns:
            List of position dictionaries with features
        """
        puzzle_id = puzzle_row['PuzzleId']
        start_fen = puzzle_row['FEN']
        moves_str = puzzle_row['Moves']
        rating = puzzle_row['Rating']
        themes = puzzle_row['Themes'] if pd.notna(puzzle_row['Themes']) else ""
        
        # Parse moves
        moves = moves_str.split()
        
        # Create board
        try:
            board = chess.Board(start_fen)
        except Exception as e:
            print(f"Error parsing FEN for puzzle {puzzle_id}: {e}")
            return []
        
        positions = []
        previous_fen = None
        
        # Extract each move in the sequence
        for move_idx, move_uci in enumerate(moves):
            try:
                move = chess.Move.from_uci(move_uci)
                
                if move not in board.legal_moves:
                    print(f"Illegal move {move_uci} in puzzle {puzzle_id}")
                    break
                
                # Current position before move
                current_fen = board.fen()
                
                # Calculate features with temporal context
                if previous_fen:
                    features = self.temporal_calc.calculate_temporal_features(
                        current_fen=current_fen,
                        previous_fen=previous_fen,
                        last_move_uci=move_uci
                    )
                    has_history = 1
                else:
                    # First move - no history
                    features = self.feature_calc.calculate_features_from_fen(
                        fen=current_fen,
                        move_uci=move_uci
                    )
                    has_history = 0
                
                # Puzzle solutions are correct moves (grade 0)
                # Odd-indexed moves are player moves (the ones we want to learn)
                # Even-indexed moves are opponent setup moves
                if move_idx % 2 == 1:  # Player move (solution)
                    grade = 0  # Correct move by definition
                else:  # Opponent move (setup)
                    grade = None  # Don't use for training
                
                position = {
                    "fen": current_fen,
                    "previous_fen": previous_fen,
                    "move_uci": move_uci,
                    "stockfish_analysis": {
                        "grade": grade,
                        "evaluation": 0,  # Placeholder
                        "best_move": move_uci,
                        "pv": moves[move_idx:]  # Remaining solution
                    },
                    "source": "lichess_puzzle",
                    "puzzle_id": puzzle_id,
                    "puzzle_rating": int(rating) if pd.notna(rating) else None,
                    "puzzle_themes": themes,
                    "sequence_id": f"puzzle_{puzzle_id}",
                    "sequence_index": move_idx,
                    "has_history": has_history,
                    "features": features
                }
                
                # Only add player moves (odd indices) to training data
                if move_idx % 2 == 1:
                    positions.append(position)
                
                # Make move and update for next iteration
                board.push(move)
                previous_fen = current_fen
                
            except Exception as e:
                print(f"Error processing move {move_uci} in puzzle {puzzle_id}: {e}")
                break
        
        return positions


def extract_puzzles(
    csv_path: str,
    output_path: str,
    num_puzzles: int = None,
    rating_min: int = 1500,
    rating_max: int = 2500,
    skip_existing: bool = True,
    checkpoint_interval: int = 100000,
    resume: bool = True
):
    """
    Extract puzzles from Lichess CSV database
    
    Args:
        csv_path: Path to lichess_db_puzzle.csv
        output_path: Output JSONL file
        num_puzzles: Number of puzzles to extract (None = all filtered)
        rating_min: Minimum puzzle rating
        rating_max: Maximum puzzle rating
        skip_existing: Skip if output file exists
        checkpoint_interval: Save checkpoint every N puzzles
        resume: Resume from last checkpoint if available
    """
    output_file = Path(output_path)
    checkpoint_file = output_file.with_suffix('.checkpoint')
    
    if skip_existing and output_file.exists() and not resume:
        print(f"Output file {output_path} already exists. Skipping.")
        return
    
    # Resume from checkpoint
    processed_ids = set()
    if resume and output_file.exists():
        print(f"Resuming from existing output file...")
        with open(output_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                processed_ids.add(data['puzzle_id'])
        print(f"Already processed: {len(processed_ids):,} puzzles")
    
    print(f"\n{'='*80}")
    print(f"Lichess Puzzle Extraction v5.3 - FULL DATASET")
    print(f"{'='*80}")
    print(f"Input: {csv_path}")
    print(f"Output: {output_path}")
    print(f"Target puzzles: {'ALL FILTERED' if num_puzzles is None else f'{num_puzzles:,}'}")
    print(f"Rating range: {rating_min}-{rating_max}")
    print(f"Checkpoint interval: {checkpoint_interval:,} puzzles")
    print(f"{'='*80}\n")
    
    # Load CSV in chunks for memory efficiency
    print("Loading puzzle database...")
    
    # Read entire CSV with dtype specification
    df = pd.read_csv(
        csv_path,
        dtype={
            'PuzzleId': str,
            'FEN': str,
            'Moves': str,
            'Rating': 'Int64',
            'Themes': str
        }
    )
    
    print(f"Total puzzles in database: {len(df):,}")
    
    # Filter by rating
    df_filtered = df[
        (df['Rating'] >= rating_min) & 
        (df['Rating'] <= rating_max)
    ].copy()
    
    print(f"Puzzles in rating range: {len(df_filtered):,}")
    
    # Filter out already processed puzzles
    if processed_ids:
        df_filtered = df_filtered[~df_filtered['PuzzleId'].isin(processed_ids)].copy()
        print(f"Remaining after resume: {len(df_filtered):,}")
    
    # Sample if num_puzzles specified
    if num_puzzles is not None and len(df_filtered) > num_puzzles:
        df_sample = df_filtered.sample(n=num_puzzles, random_state=42)
    else:
        df_sample = df_filtered
    
    print(f"Processing {len(df_sample):,} puzzles...\n")
    
    # Create extractor
    extractor = LichessPuzzleExtractor()
    
    # Process puzzles
    total_positions = 0
    puzzles_processed = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Open in append mode if resuming
    mode = 'a' if (resume and output_file.exists()) else 'w'
    
    with open(output_file, mode) as f:
        for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
            try:
                positions = extractor.extract_puzzle_sequence(row)
                
                for pos in positions:
                    f.write(json.dumps(pos) + '\n')
                    total_positions += 1
                
                puzzles_processed += 1
                
                # Save incrementally every 1000 puzzles
                if puzzles_processed % 1000 == 0:
                    f.flush()
                
                # Checkpoint every N puzzles
                if puzzles_processed % checkpoint_interval == 0:
                    print(f"\n✅ Checkpoint: {puzzles_processed:,} puzzles, {total_positions:,} positions")
                    f.flush()
                    
            except Exception as e:
                print(f"\nError processing puzzle {row['PuzzleId']}: {e}")
                continue
    
    print(f"\n{'='*80}")
    print(f"✅ Extraction Complete!")
    print(f"{'='*80}")
    print(f"Puzzles processed: {len(df_sample):,}")
    print(f"Positions extracted: {total_positions:,}")
    print(f"Avg positions/puzzle: {total_positions/len(df_sample):.1f}")
    print(f"Output: {output_path}")
    print(f"{'='*80}\n")
    
    # Summary statistics
    print("📊 Grade Distribution:")
    with open(output_file, 'r') as f:
        grades = []
        for line in f:
            data = json.loads(line)
            grade = data['stockfish_analysis'].get('grade')
            if grade is not None:
                grades.append(grade)
    
    if grades:
        from collections import Counter
        grade_counts = Counter(grades)
        
        for g in range(6):
            count = grade_counts.get(g, 0)
            pct = (count / len(grades) * 100)
            bar = '█' * int(pct / 2)
            print(f"  Grade {g}: {count:6,} ({pct:5.1f}%) {bar}")
        
        good_moves = sum(grade_counts[g] for g in range(3))
        print(f"\n🎯 Good Moves (Grades 0-2): {good_moves:,} ({good_moves/len(grades)*100:.1f}%)")
    else:
        print("  ⚠️  No positions with grades extracted!")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Configuration
    CSV_PATH = "E:/Programming Stuff/Chess Engines/Chess Engine Playground/engine-metrics/raw_data/pgn_training_data/csv_data_puzzles/lichess_db_puzzle.csv"
    OUTPUT_PATH = "E:/Programming Stuff/Chess Engines/V7P3R Chess AI/v7p3r-chess-ai/v5.0/data/puzzles/lichess_puzzles_v5.3_full.jsonl"
    
    NUM_PUZZLES = None  # Process ALL filtered puzzles (2.1M)
    RATING_MIN = 1500
    RATING_MAX = 2500
    CHECKPOINT_INTERVAL = 100000  # Checkpoint every 100k puzzles
    
    extract_puzzles(
        csv_path=CSV_PATH,
        output_path=OUTPUT_PATH,
        num_puzzles=NUM_PUZZLES,
        rating_min=RATING_MIN,
        rating_max=RATING_MAX,
        skip_existing=False,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        resume=True
    )
