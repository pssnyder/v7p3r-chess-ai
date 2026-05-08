#!/usr/bin/env python3
"""
Stockfish-Enriched Puzzle Preprocessor

Converts raw puzzle database into ML-friendly training dataset by:
1. Loading puzzles from SQLite database
2. Using Stockfish to find top-N moves (default: top-10) with evaluations
3. Extracting positional themes and puzzle metadata
4. Saving enriched data in efficient format (PyTorch tensors + JSON metadata)

This creates a training dataset where each puzzle includes:
- Starting FEN
- Top-N best moves ranked by Stockfish
- Move scores (centipawns)
- Move ranking weights (5pt, 4pt, 3pt, 2pt, 1pt for top-5)
- Puzzle themes (multi-hot encoded)
- Puzzle rating (difficulty)

Output format optimized for:
- Fast loading during training
- Memory-efficient batching
- Multi-task learning (theme classification + move ranking)
"""

import sys
import os
import json
import sqlite3
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
import chess
import chess.engine
import numpy as np

# Add engine-tester utilities to path
ENGINE_TESTER_PATH = r"E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-tester"
sys.path.append(ENGINE_TESTER_PATH)
sys.path.append(os.path.join(ENGINE_TESTER_PATH, 'databases'))

try:
    from database import PuzzleDatabase, Puzzle
except ImportError:
    print("ERROR: Could not import PuzzleDatabase from engine-tester")
    print(f"Make sure engine-tester is at: {ENGINE_TESTER_PATH}")
    sys.exit(1)


@dataclass
class EnrichedPuzzle:
    """Puzzle enriched with Stockfish top-N moves and rankings"""
    puzzle_id: str
    fen: str
    rating: int
    themes: List[str]
    
    # Top-N moves from Stockfish (ranked best to worst)
    top_moves: List[str]  # UCI format moves
    move_scores: List[int]  # Centipawn evaluations
    move_weights: List[float]  # Training weights (1.0 for best, decreasing)
    
    # Original puzzle solution
    solution_moves: List[str]
    solution_first_move: str
    
    # Puzzle metadata
    popularity: int
    num_plays: int
    game_url: str


class StockfishPuzzlePreprocessor:
    """Preprocesses puzzles using Stockfish analysis for move ordering training"""
    
    # Theme mapping for multi-hot encoding
    ALL_THEMES = [
        'mate', 'mateIn1', 'mateIn2', 'mateIn3', 'mateIn4', 'mateIn5',
        'pin', 'fork', 'skewer', 'discoveredAttack', 'doubleCheck',
        'hangingPiece', 'trappedPiece', 'defensiveMove', 'deflection',
        'attraction', 'clearance', 'interference', 'intermezzo',
        'sacrifice', 'endgame', 'middlegame', 'opening',
        'advancedPawn', 'attackingF2F7', 'capturingDefender', 'exposedKing',
        'kingsideAttack', 'queensideAttack', 'pawnEndgame', 'rookEndgame',
        'bishopEndgame', 'knightEndgame', 'queenEndgame', 'queenRookEndgame',
        'advantage', 'crushing', 'equality', 'quiet', 'zugzwang',
        'short', 'long', 'veryLong', 'master', 'masterVsMaster',
        'superGM', 'oneMove', 'promotion', 'underPromotion',
        'castling', 'enPassant', 'smotheredMate', 'backRankMate',
        'doubleBishopMate', 'dovetailMate', 'arabianMate', 'anastasiaMate'
    ]
    
    def __init__(self,
                 puzzle_db_path: str,
                 stockfish_path: str = r"E:\Programming Stuff\Chess Engines\Tournament Engines\Stockfish\stockfish-windows-x86-64-avx2.exe",
                 output_dir: str = "data/preprocessed_puzzles",
                 top_n_moves: int = 10,
                 stockfish_time: float = 1.0):
        
        self.puzzle_db_path = puzzle_db_path
        self.stockfish_path = stockfish_path
        self.output_dir = Path(output_dir)
        self.top_n_moves = top_n_moves
        self.stockfish_time = stockfish_time
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify files exist
        if not os.path.exists(puzzle_db_path):
            raise FileNotFoundError(f"Puzzle database not found: {puzzle_db_path}")
        if not os.path.exists(stockfish_path):
            raise FileNotFoundError(f"Stockfish not found: {stockfish_path}")
        
        print(f"🚀 Initializing Stockfish Puzzle Preprocessor")
        print(f"   Puzzle DB: {puzzle_db_path}")
        print(f"   Stockfish: {stockfish_path}")
        print(f"   Output: {self.output_dir}")
        print(f"   Top-N moves: {top_n_moves}")
        print(f"   Stockfish time per position: {stockfish_time}s")
        
        # Initialize Stockfish engine once (PERFORMANCE FIX)
        print(f"   Starting Stockfish engine...")
        self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        print(f"   ✅ Stockfish ready")
    
    def get_stockfish_top_moves(self, fen: str, num_moves: int = 10) -> List[Tuple[str, int]]:
        """Get Stockfish's top N moves with centipawn scores"""
        try:
            board = chess.Board(fen)
            
            # Use multipv to get multiple best moves (reuse existing engine)
            result = self.engine.analyse(
                board,
                chess.engine.Limit(time=self.stockfish_time),
                multipv=num_moves
            )
            
            moves_with_scores = []
            for analysis in result:
                if 'pv' in analysis and analysis['pv']:
                    move = analysis['pv'][0]
                    score = analysis.get('score', chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE))
                    
                    # Convert score to centipawns from white's perspective
                    if score.is_mate():
                        mate_in = score.white().mate()
                        if mate_in is not None:
                            # Mate scores: +10000 for mate, decreasing by distance
                            cp_score = 10000 - abs(mate_in) * 100 if mate_in > 0 else -10000 + abs(mate_in) * 100
                        else:
                            cp_score = 0
                    else:
                        cp_score = score.white().score() if score.white().score() is not None else 0
                    
                    moves_with_scores.append((str(move), cp_score))
            
            return moves_with_scores
                
        except Exception as e:
            print(f"⚠️  Error analyzing position: {e}")
            return []
    
    def calculate_move_weights(self, num_moves: int) -> List[float]:
        """
        Calculate training weights for top-N moves
        Uses exponential decay: 1.0 for best, decreasing
        
        Top-5 get standard weights: [1.0, 0.8, 0.6, 0.4, 0.2]
        Beyond top-5: Continue exponential decay
        """
        weights = []
        for i in range(num_moves):
            if i < 5:
                # Standard top-5 weights
                weight = 1.0 - (i * 0.2)
            else:
                # Exponential decay for moves 6-10
                weight = max(0.1, 0.2 * (0.7 ** (i - 4)))
            weights.append(weight)
        
        return weights
    
    def parse_themes(self, theme_string: str) -> List[str]:
        """Parse space-separated theme string into list"""
        if not theme_string or theme_string.strip() == '':
            return []
        return [t.strip() for t in theme_string.split() if t.strip()]
    
    def enrich_puzzle(self, puzzle: Puzzle) -> Optional[EnrichedPuzzle]:
        """Enrich a single puzzle with Stockfish analysis"""
        try:
            # Parse solution moves
            solution_moves = puzzle.moves.split() if puzzle.moves else []
            if not solution_moves:
                return None
            
            solution_first_move = solution_moves[0]
            
            # Get top-N moves from Stockfish
            stockfish_moves = self.get_stockfish_top_moves(puzzle.fen, self.top_n_moves)
            
            if not stockfish_moves:
                return None
            
            # Extract moves and scores
            top_moves = [move for move, _ in stockfish_moves]
            move_scores = [score for _, score in stockfish_moves]
            
            # Calculate training weights
            move_weights = self.calculate_move_weights(len(top_moves))
            
            # Parse themes
            themes = self.parse_themes(puzzle.themes)
            
            return EnrichedPuzzle(
                puzzle_id=puzzle.id,
                fen=puzzle.fen,
                rating=puzzle.rating or 1500,
                themes=themes,
                top_moves=top_moves,
                move_scores=move_scores,
                move_weights=move_weights,
                solution_moves=solution_moves,
                solution_first_move=solution_first_move,
                popularity=puzzle.popularity or 0,
                num_plays=puzzle.num_plays or 0,
                game_url=puzzle.game_url or ''
            )
            
        except Exception as e:
            print(f"⚠️  Error enriching puzzle {puzzle.id}: {e}")
            return None
    
    def preprocess_batch(self, 
                        rating_min: int = 600,
                        rating_max: int = 2500,
                        max_puzzles: Optional[int] = None,
                        batch_size: int = 1000,
                        checkpoint_interval: int = 5000):
        """
        Preprocess puzzles in batches and save results
        
        Args:
            rating_min: Minimum puzzle rating
            rating_max: Maximum puzzle rating
            max_puzzles: Maximum puzzles to process (None = all)
            batch_size: Puzzles to load from DB at once
            checkpoint_interval: Save checkpoint every N puzzles
        """
        print(f"\n📊 Starting puzzle preprocessing...")
        print(f"   Rating range: {rating_min}-{rating_max}")
        print(f"   Max puzzles: {max_puzzles or 'All'}")
        
        db = PuzzleDatabase(self.puzzle_db_path)
        
        # Count total puzzles in range
        conn = sqlite3.connect(self.puzzle_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM puzzles WHERE rating >= ? AND rating <= ?",
            (rating_min, rating_max)
        )
        total_available = cursor.fetchone()[0]
        conn.close()
        
        total_to_process = min(total_available, max_puzzles) if max_puzzles else total_available
        print(f"   Total puzzles available: {total_available:,}")
        print(f"   Will process: {total_to_process:,}")
        
        enriched_puzzles = []
        processed_count = 0
        skipped_count = 0
        
        # Create progress bar
        pbar = tqdm(total=total_to_process, desc="Enriching puzzles", unit="puzzle")
        
        # Process in batches
        offset = 0
        while processed_count < total_to_process:
            # Query batch from database
            puzzles = db.query_puzzles(
                min_rating=rating_min,
                max_rating=rating_max,
                quantity=batch_size
            )
            
            if not puzzles:
                break
            
            # Enrich each puzzle in batch
            for puzzle in puzzles:
                if processed_count >= total_to_process:
                    break
                
                enriched = self.enrich_puzzle(puzzle)
                if enriched:
                    enriched_puzzles.append(enriched)
                else:
                    skipped_count += 1
                
                processed_count += 1
                pbar.update(1)
                
                # Save checkpoint periodically
                if len(enriched_puzzles) % checkpoint_interval == 0 and len(enriched_puzzles) > 0:
                    self.save_checkpoint(enriched_puzzles, processed_count, total_to_process)
            
            offset += batch_size
        
        pbar.close()
        
        # Save final dataset
        print(f"\n💾 Saving final dataset...")
        self.save_dataset(enriched_puzzles, processed_count, total_to_process)
        
        print(f"\n✅ Preprocessing complete!")
        print(f"   Processed: {processed_count:,} puzzles")
        print(f"   Enriched: {len(enriched_puzzles):,} puzzles")
        print(f"   Skipped: {skipped_count:,} puzzles")
        print(f"   Success rate: {len(enriched_puzzles)/processed_count*100:.1f}%")
    
    def cleanup(self):
        """Close Stockfish engine and cleanup resources"""
        if hasattr(self, 'engine'):
            try:
                self.engine.quit()
                print(f"   ✅ Stockfish engine closed")
            except Exception as e:
                print(f"   ⚠️  Error closing engine: {e}")
    
    def save_checkpoint(self, enriched_puzzles: List[EnrichedPuzzle], processed: int, total: int):
        """Save intermediate checkpoint"""
        checkpoint_file = self.output_dir / f"checkpoint_{processed}_{total}.json"
        
        data = {
            'metadata': {
                'processed': processed,
                'total': total,
                'enriched_count': len(enriched_puzzles),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'top_n_moves': self.top_n_moves
            },
            'puzzles': [asdict(p) for p in enriched_puzzles]
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"   📁 Checkpoint saved: {checkpoint_file.name}")
    
    def save_dataset(self, enriched_puzzles: List[EnrichedPuzzle], processed: int, total: int):
        """Save final dataset in multiple formats"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # Save full JSON (for inspection and debugging)
        json_file = self.output_dir / f"enriched_puzzles_{timestamp}.json"
        data = {
            'metadata': {
                'processed': processed,
                'total': total,
                'enriched_count': len(enriched_puzzles),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'top_n_moves': self.top_n_moves,
                'theme_vocabulary': self.ALL_THEMES
            },
            'puzzles': [asdict(p) for p in enriched_puzzles]
        }
        
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"   ✓ JSON saved: {json_file.name} ({json_file.stat().st_size / 1024 / 1024:.1f} MB)")
        
        # Save compact format (for fast loading during training)
        compact_file = self.output_dir / f"enriched_puzzles_compact_{timestamp}.json"
        compact_data = {
            'metadata': data['metadata'],
            'puzzles': [asdict(p) for p in enriched_puzzles]  # No indentation = smaller file
        }
        
        with open(compact_file, 'w') as f:
            json.dump(compact_data, f)
        
        print(f"   ✓ Compact JSON saved: {compact_file.name} ({compact_file.stat().st_size / 1024 / 1024:.1f} MB)")
        
        # Save summary statistics
        self.save_statistics(enriched_puzzles, timestamp)
    
    def save_statistics(self, enriched_puzzles: List[EnrichedPuzzle], timestamp: str):
        """Generate and save dataset statistics"""
        stats_file = self.output_dir / f"dataset_stats_{timestamp}.txt"
        
        # Calculate statistics
        ratings = [p.rating for p in enriched_puzzles]
        theme_counts = {}
        for p in enriched_puzzles:
            for theme in p.themes:
                theme_counts[theme] = theme_counts.get(theme, 0) + 1
        
        with open(stats_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("ENRICHED PUZZLE DATASET STATISTICS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Total puzzles: {len(enriched_puzzles):,}\n")
            f.write(f"Top-N moves per puzzle: {self.top_n_moves}\n\n")
            
            if ratings:
                f.write("Rating Distribution:\n")
                f.write(f"  Min: {min(ratings)}\n")
                f.write(f"  Max: {max(ratings)}\n")
                f.write(f"  Mean: {np.mean(ratings):.0f}\n")
                f.write(f"  Median: {np.median(ratings):.0f}\n\n")
            else:
                f.write("Rating Distribution: N/A (no puzzles processed)\n\n")
            
            if theme_counts:
                f.write("Top 20 Themes:\n")
                sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
                for theme, count in sorted_themes[:20]:
                    f.write(f"  {theme:20s}: {count:6,} ({count/len(enriched_puzzles)*100:5.1f}%)\n")
            else:
                f.write("Themes: N/A (no puzzles processed)\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        print(f"   ✓ Statistics saved: {stats_file.name}")


def main():
    """Main preprocessing pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess puzzles with Stockfish analysis")
    parser.add_argument('--puzzle-db', type=str, 
                       default=r"E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-tester\databases\puzzles.db",
                       help="Path to puzzle database")
    parser.add_argument('--stockfish', type=str,
                       default=r"E:\Programming Stuff\Chess Engines\Tournament Engines\Stockfish\stockfish-windows-x86-64-avx2.exe",
                       help="Path to Stockfish engine")
    parser.add_argument('--output-dir', type=str,
                       default="data/preprocessed_puzzles",
                       help="Output directory for enriched dataset")
    parser.add_argument('--top-n', type=int, default=10,
                       help="Number of top moves to extract")
    parser.add_argument('--stockfish-time', type=float, default=1.0,
                       help="Stockfish analysis time per position (seconds)")
    parser.add_argument('--rating-min', type=int, default=600,
                       help="Minimum puzzle rating")
    parser.add_argument('--rating-max', type=int, default=2500,
                       help="Maximum puzzle rating")
    parser.add_argument('--max-puzzles', type=int, default=None,
                       help="Maximum puzzles to process (None = all)")
    parser.add_argument('--batch-size', type=int, default=1000,
                       help="Batch size for database queries")
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = StockfishPuzzlePreprocessor(
        puzzle_db_path=args.puzzle_db,
        stockfish_path=args.stockfish,
        output_dir=args.output_dir,
        top_n_moves=args.top_n,
        stockfish_time=args.stockfish_time
    )
    
    try:
        # Run preprocessing
        preprocessor.preprocess_batch(
            rating_min=args.rating_min,
            rating_max=args.rating_max,
            max_puzzles=args.max_puzzles,
            batch_size=args.batch_size
        )
    finally:
        # Always cleanup engine
        preprocessor.cleanup()


if __name__ == '__main__':
    main()
