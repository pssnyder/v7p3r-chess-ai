"""
V7P3R Multi-Engine Puzzle Solver
Runs multiple V7P3R versions through Lichess puzzles to collect engine-specific move data

Strategy:
- V7P3R engines (18.0, 18.3, 18.4, 17.1.1) solve puzzles
- Record their move choices vs puzzle solutions
- Stockfish grades all moves (puzzle solutions + V7P3R alternatives)
- Creates V7P3R-character training data
"""

import chess
import chess.engine
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys
from tqdm import tqdm
from datetime import datetime

# V7P3R engine configurations
V7P3R_ENGINES = {
    "v18.5": {
        "path": "E:/Programming Stuff/Chess Engines/V7P3R Chess Engine/v7p3r-chess-engine/lichess/engines/V7P3R_v18.5_20260510/src/v7p3r_uci.py",
        "type": "python"
    },
    "v18.3": {
        "path": "E:/Programming Stuff/Chess Engines/V7P3R Chess Engine/v7p3r-chess-engine/lichess/engines/V7P3R_v18.3_20251229/src/v7p3r_uci.py",
        "type": "python"
    },
    "v18.0": {
        "path": "E:/Programming Stuff/Chess Engines/V7P3R Chess Engine/v7p3r-chess-engine/lichess/engines/V7P3R_v18.0_20251220/src/v7p3r_uci.py",
        "type": "python"
    },
    "v17.1.1": {
        "path": "E:/Programming Stuff/Chess Engines/V7P3R Chess Engine/v7p3r-chess-engine/lichess/engines/V7P3R_v17.1.1_20251121/src/v7p3r_uci.py",
        "type": "python"
    }
}

# Stockfish for grading
STOCKFISH_PATH = "E:/Programming Stuff/Chess Engines/Tournament Engines/Stockfish/stockfish-windows-x86-64-avx2.exe"


class PuzzleSolver:
    """Solve puzzles with V7P3R engines and collect move data"""
    
    def __init__(self, engine_config: Dict, stockfish_path: str):
        self.engine_name = None
        self.engine = None
        self.stockfish = None
        self.engine_config = engine_config
        self.stockfish_path = stockfish_path
        
    def __enter__(self):
        # Start V7P3R engine
        if self.engine_config["type"] == "python":
            self.engine = chess.engine.SimpleEngine.popen_uci(
                ["python", self.engine_config["path"]]
            )
        else:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_config["path"])
        
        # Start Stockfish for grading
        self.stockfish = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.engine:
            self.engine.quit()
        if self.stockfish:
            self.stockfish.quit()
    
    def solve_puzzle(self, puzzle_data: Dict, time_limit: float = 5.0) -> Dict:
        """
        Solve single puzzle with V7P3R engine
        
        Returns:
            {
                'puzzle_id': str,
                'fen': str,
                'rating': int,
                'themes': str,
                'puzzle_moves': list,
                'engine_move': str,
                'engine_matches_puzzle': bool,
                'stockfish_grade': int,  # 0-5 grade for engine move
                'puzzle_solution_grade': int,  # 0-5 grade for puzzle solution
                'engine_eval': int,  # Centipawns
                'puzzle_eval': int  # Centipawns
            }
        """
        board = chess.Board(puzzle_data['FEN'])
        puzzle_moves = puzzle_data['Moves'].split()
        
        # First move is usually opponent's move (puzzle setup)
        if puzzle_moves:
            opponent_move = chess.Move.from_uci(puzzle_moves[0])
            board.push(opponent_move)
        
        # Get V7P3R's move
        try:
            result = self.engine.play(board, chess.engine.Limit(time=time_limit))
            engine_move_uci = result.move.uci()
        except Exception as e:
            print(f"Engine error on puzzle {puzzle_data['PuzzleId']}: {e}")
            return None
        
        # Check if engine matches puzzle solution
        puzzle_solution = puzzle_moves[1] if len(puzzle_moves) > 1 else None
        engine_matches = (engine_move_uci == puzzle_solution)
        
        # Grade both moves with Stockfish
        stockfish_depth = 20
        
        # Grade engine move
        test_board = board.copy()
        test_board.push(chess.Move.from_uci(engine_move_uci))
        engine_eval_info = self.stockfish.analyse(test_board, chess.engine.Limit(depth=stockfish_depth))
        engine_eval = engine_eval_info['score'].relative.score(mate_score=10000)
        
        # Grade puzzle solution
        if puzzle_solution:
            test_board = board.copy()
            test_board.push(chess.Move.from_uci(puzzle_solution))
            puzzle_eval_info = self.stockfish.analyse(test_board, chess.engine.Limit(depth=stockfish_depth))
            puzzle_eval = puzzle_eval_info['score'].relative.score(mate_score=10000)
        else:
            puzzle_eval = None
        
        # Get all legal moves and evaluate top 6 for grading
        legal_moves = list(board.legal_moves)
        move_evals = []
        
        for move in legal_moves[:20]:  # Evaluate top 20 moves (time constraint)
            test_board = board.copy()
            test_board.push(move)
            eval_info = self.stockfish.analyse(test_board, chess.engine.Limit(depth=15))
            score = eval_info['score'].relative.score(mate_score=10000)
            move_evals.append((move.uci(), score))
        
        # Sort by score (best first)
        move_evals.sort(key=lambda x: x[1] if x[1] is not None else -999999, reverse=True)
        
        # Assign grades (0 = best, 5 = 6th best or worse)
        def get_grade(move_uci, sorted_evals):
            for i, (m, _) in enumerate(sorted_evals[:6]):
                if m == move_uci:
                    return i
            return 5  # Worse than 6th best
        
        engine_grade = get_grade(engine_move_uci, move_evals)
        puzzle_grade = get_grade(puzzle_solution, move_evals) if puzzle_solution else None
        
        return {
            'puzzle_id': puzzle_data['PuzzleId'],
            'fen': puzzle_data['FEN'],
            'rating': puzzle_data['Rating'],
            'themes': puzzle_data['Themes'],
            'puzzle_moves': puzzle_moves,
            'engine_move': engine_move_uci,
            'engine_matches_puzzle': engine_matches,
            'stockfish_grade': engine_grade,
            'puzzle_solution_grade': puzzle_grade,
            'engine_eval': engine_eval,
            'puzzle_eval': puzzle_eval,
            'top_6_moves': move_evals[:6]
        }


def process_puzzles(
    puzzle_csv: str,
    engine_name: str,
    engine_config: Dict,
    output_dir: str,
    puzzle_subset: pd.DataFrame,
    engine_index: int,
    total_engines: int
):
    """
    Process puzzles with a specific V7P3R engine
    
    Args:
        puzzle_csv: Path to Lichess puzzle database
        engine_name: Name of V7P3R version (e.g., "v18.5")
        engine_config: Engine configuration dict
        output_dir: Output directory for results
        puzzle_subset: Pre-allocated DataFrame of puzzles for this engine
        engine_index: Index of this engine (0-based)
        total_engines: Total number of engines
    """
    print(f"\n{'='*80}")
    print(f"V7P3R {engine_name} Puzzle Solver")
    print(f"{'='*80}")
    print(f"Engine {engine_index + 1} of {total_engines}")
    print(f"Assigned puzzles: {len(puzzle_subset):,}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")
    
    df_sample = puzzle_subset
    
    print(f"Processing {len(df_sample):,} unique puzzles for {engine_name}...\n")
    
    # Solve puzzles
    results = []
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    with PuzzleSolver(engine_config, STOCKFISH_PATH) as solver:
        for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
            puzzle_data = row.to_dict()
            
            try:
                result = solver.solve_puzzle(puzzle_data, time_limit=3.0)
                if result:
                    result['engine_version'] = engine_name
                    result['timestamp'] = datetime.now().isoformat()
                    results.append(result)
                    
                    # Save incrementally every 100 puzzles
                    if len(results) % 100 == 0:
                        output_file = f"{output_dir}/{engine_name}_puzzle_results_{len(results)}.jsonl"
                        with open(output_file, 'w') as f:
                            for r in results[-100:]:
                                f.write(json.dumps(r) + '\n')
                        
            except Exception as e:
                print(f"\nError on puzzle {puzzle_data['PuzzleId']}: {e}")
                continue
    
    # Save final results
    output_file = f"{output_dir}/{engine_name}_puzzle_results_final.jsonl"
    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    
    print(f"\n{'='*80}")
    print(f"✅ Completed! Solved {len(results):,} puzzles")
    print(f"Output: {output_file}")
    print(f"{'='*80}\n")
    
    # Print summary statistics
    print_summary(results)


def print_summary(results: List[Dict]):
    """Print summary statistics for puzzle solving results"""
    if not results:
        return
    
    total = len(results)
    matches = sum(1 for r in results if r['engine_matches_puzzle'])
    
    # Grade distribution for engine moves
    grade_counts = [0] * 6
    for r in results:
        grade_counts[r['stockfish_grade']] += 1
    
    # Calculate % of good moves (grades 0-2)
    good_moves = sum(grade_counts[:3])
    good_move_pct = (good_moves / total) * 100
    
    print("\n📊 Summary Statistics")
    print(f"{'='*60}")
    print(f"Total puzzles solved: {total:,}")
    print(f"Exact puzzle matches: {matches:,} ({matches/total*100:.1f}%)")
    print(f"\nMove Quality Distribution:")
    for grade in range(6):
        pct = (grade_counts[grade] / total) * 100
        bar = '█' * int(pct / 2)
        print(f"  Grade {grade}: {grade_counts[grade]:5,} ({pct:5.1f}%) {bar}")
    
    print(f"\n🎯 Good Moves (Grades 0-2): {good_moves:,} ({good_move_pct:.1f}%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Configuration
    PUZZLE_CSV = "E:/Programming Stuff/Chess Engines/Chess Engine Playground/engine-metrics/raw_data/pgn_training_data/csv_data_puzzles/lichess_db_puzzle.csv"
    OUTPUT_DIR = "E:/Programming Stuff/Chess Engines/V7P3R Chess AI/v7p3r-chess-ai/v5.0/data/multi_engine_puzzles"
    
    # Total puzzles to cover across ALL engines (each engine gets unique subset)
    TOTAL_PUZZLES = 4000  # Start with 4k total (1k per engine)
    RATING_RANGE = (1500, 2500)
    
    print(f"\n{'='*80}")
    print(f"V7P3R Multi-Engine Puzzle Solver - UNIQUE PUZZLE DISTRIBUTION")
    print(f"{'='*80}")
    print(f"Total puzzles to process: {TOTAL_PUZZLES:,}")
    print(f"Number of engines: {len(V7P3R_ENGINES)}")
    print(f"Puzzles per engine: {TOTAL_PUZZLES // len(V7P3R_ENGINES):,}")
    print(f"Rating range: {RATING_RANGE[0]}-{RATING_RANGE[1]}")
    print(f"{'='*80}\n")
    
    # Load and split puzzles ONCE
    print("Loading puzzle database...")
    df = pd.read_csv(PUZZLE_CSV)
    print(f"Total puzzles available: {len(df):,}")
    
    # Filter by rating
    df_filtered = df[(df['Rating'] >= RATING_RANGE[0]) & (df['Rating'] <= RATING_RANGE[1])]
    print(f"Puzzles in rating range: {len(df_filtered):,}")
    
    # Sample total puzzles needed
    if len(df_filtered) > TOTAL_PUZZLES:
        df_sample = df_filtered.sample(n=TOTAL_PUZZLES, random_state=42)
    else:
        df_sample = df_filtered
    
    # Shuffle for random distribution
    df_sample = df_sample.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    print(f"Selected {len(df_sample):,} puzzles to distribute\n")
    
    # Split puzzles across engines (each gets unique subset)
    num_engines = len(V7P3R_ENGINES)
    puzzles_per_engine = len(df_sample) // num_engines
    
    print("Puzzle Distribution:")
    print("-" * 60)
    
    engine_subsets = {}
    for idx, engine_name in enumerate(V7P3R_ENGINES.keys()):
        start_idx = idx * puzzles_per_engine
        if idx == num_engines - 1:
            # Last engine gets any remainder
            end_idx = len(df_sample)
        else:
            end_idx = (idx + 1) * puzzles_per_engine
        
        engine_subsets[engine_name] = df_sample.iloc[start_idx:end_idx].copy()
        
        avg_rating = engine_subsets[engine_name]['Rating'].mean()
        print(f"  {engine_name}: {len(engine_subsets[engine_name]):,} puzzles (avg rating: {avg_rating:.0f})")
    
    print("-" * 60)
    print(f"  TOTAL: {sum(len(subset) for subset in engine_subsets.values()):,} unique puzzles\n")
    
    # Process each V7P3R version with its unique puzzle subset
    for idx, (engine_name, engine_config) in enumerate(V7P3R_ENGINES.items()):
        try:
            process_puzzles(
                puzzle_csv=PUZZLE_CSV,
                engine_name=engine_name,
                engine_config=engine_config,
                output_dir=OUTPUT_DIR,
                puzzle_subset=engine_subsets[engine_name],
                engine_index=idx,
                total_engines=num_engines
            )
        except Exception as e:
            print(f"❌ Error processing {engine_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n🎉 All engines completed!")
    print(f"Total unique puzzles processed: {TOTAL_PUZZLES:,}")
    print(f"Expected unique positions: ~{TOTAL_PUZZLES * 3:,} (assuming ~3 moves per puzzle)")
