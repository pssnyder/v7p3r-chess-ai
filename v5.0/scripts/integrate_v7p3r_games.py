"""
Extract positions from V7P3R Lichess games for v5.3 training.

This script processes V7P3R PGN game files to extract training positions
with full 325-feature calculation (including temporal features).

Similar to C0BR4 integration but for V7P3R bot games.
"""

import chess
import chess.engine
import chess.pgn
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.calculate_features import FeatureCalculator, FeatureConfig
from scripts.temporal_feature_calculator import TemporalFeatureCalculator


class V7P3RGameIntegrator:
    """Extract and process positions from V7P3R Lichess games."""
    
    def __init__(self, stockfish_path: str, output_file: str):
        """Initialize with Stockfish path and output file."""
        self.stockfish_path = stockfish_path
        self.output_file = output_file
        
        # Initialize feature calculators
        config = FeatureConfig.from_preset("full")
        self.feature_calc = FeatureCalculator(config)
        self.temporal_calc = TemporalFeatureCalculator(self.feature_calc)
        
        # Initialize Stockfish engine
        print(f"Initializing Stockfish: {stockfish_path}")
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        
        # Statistics
        self.games_processed = 0
        self.positions_extracted = 0
        self.positions_analyzed = 0
        
    def __del__(self):
        """Clean up Stockfish engine."""
        if hasattr(self, 'engine'):
            self.engine.quit()
    
    def analyze_position(self, board: chess.Board) -> dict:
        """
        Analyze position with Stockfish to get top 6 candidate moves and grades.
        
        Uses time-based limit (0.5s) for speed instead of depth-based.
        """
        # Get top 6 moves from Stockfish
        info = self.engine.analyse(
            board,
            chess.engine.Limit(time=0.5),  # Fast time-based limit
            multipv=6
        )
        
        candidates = []
        for i, pv_info in enumerate(info):
            move = pv_info['pv'][0]
            score = pv_info['score'].relative
            
            # Convert score to centipawns
            if score.is_mate():
                cp = 10000 if score.mate() > 0 else -10000
            else:
                cp = score.score()
            
            candidates.append({
                'move': move.uci(),
                'score': cp,
                'grade': i  # Grade 0-5 based on rank
            })
        
        return {
            'candidates': candidates,
            'top_moves': [c['move'] for c in candidates]
        }
    
    def should_extract_position(self, board: chess.Board, move_num: int, last_move: chess.Move = None) -> bool:
        """
        Decide if position should be extracted.
        
        Strategy: Extract every 5th move + tactical moments
        (captures, checks, castling, promotions)
        """
        # Every 5th move
        if move_num % 5 == 0:
            return True
        
        # Tactical moments if we have last move
        if last_move:
            # Captures
            if board.is_capture(last_move):
                return True
            
            # Checks
            if board.is_check():
                return True
            
            # Castling
            if board.is_castling(last_move):
                return True
            
            # Promotions
            if last_move.promotion:
                return True
        
        return False
    
    def extract_game_positions(self, game: chess.pgn.Game) -> list:
        """
        Extract positions from a game with Stockfish analysis and feature calculation.
        
        Returns list of position records with features and grades.
        """
        positions = []
        board = game.board()
        move_num = 0
        
        # Track previous position for temporal features
        previous_fen = None
        last_move_uci = None
        
        for move in game.mainline_moves():
            move_num += 1
            
            # Check if we should extract this position
            if not self.should_extract_position(board, move_num, move):
                # Still update for temporal context
                previous_fen = board.fen()
                last_move_uci = move.uci()
                board.push(move)
                continue
            
            # Analyze position to get candidate moves
            analysis = self.analyze_position(board)
            self.positions_analyzed += 1
            
            # Find grade of actual move played
            actual_move_uci = move.uci()
            grade = 5  # Default to worst grade
            
            for candidate in analysis['candidates']:
                if candidate['move'] == actual_move_uci:
                    grade = candidate['grade']
                    break
            
            # Get current position FEN
            current_fen = board.fen()
            
            # Calculate features with temporal context
            if previous_fen and last_move_uci:
                # Has history - calculate temporal features
                features = self.temporal_calc.calculate_temporal_features(
                    current_fen=current_fen,
                    previous_fen=previous_fen,
                    last_move_uci=last_move_uci,
                    sequence_index=move_num,
                    stockfish_eval=analysis['candidates'][0]['score'] if analysis['candidates'] else 0
                )
            else:
                # First move - use base features with null sentinels
                features = self.temporal_calc.calculate_temporal_features(
                    current_fen=current_fen,
                    previous_fen=None,
                    last_move_uci=None
                )
            
            # Create position record
            record = {
                'fen': current_fen,
                'move': actual_move_uci,
                'grade': grade,
                'features': features,
                'metadata': {
                    'move_number': move_num,
                    'source': 'v7p3r_lichess',
                    'white': game.headers.get('White', 'Unknown'),
                    'black': game.headers.get('Black', 'Unknown'),
                    'result': game.headers.get('Result', '*'),
                    'date': game.headers.get('UTCDate', ''),
                    'game_id': game.headers.get('GameId', '')
                }
            }
            
            positions.append(record)
            self.positions_extracted += 1
            
            # Update for next iteration
            previous_fen = current_fen
            last_move_uci = actual_move_uci
            board.push(move)
        
        return positions
    
    def process_pgn_file(self, pgn_path: str, output_handle):
        """
        Process a PGN file and extract positions.
        
        Writes directly to output file handle for memory efficiency.
        """
        print(f"\n📂 Processing: {os.path.basename(pgn_path)}")
        
        game_count = 0
        
        with open(pgn_path, 'r') as pgn_file:
            while True:
                try:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break
                    
                    game_count += 1
                    self.games_processed += 1
                    
                    # Extract positions from game
                    positions = self.extract_game_positions(game)
                    
                    # Write positions to output
                    for position in positions:
                        output_handle.write(json.dumps(position) + '\n')
                    
                    # Progress update every 10 games
                    if game_count % 10 == 0:
                        print(f"  Games: {game_count}, Positions: {self.positions_extracted}")
                
                except KeyboardInterrupt:
                    print("\n⚠️  Interrupted by user")
                    print(f"   Progress: {game_count} games, {self.positions_extracted} positions")
                    raise
                
                except Exception as e:
                    print(f"  ⚠️  Error processing game {game_count}: {e}")
                    continue
        
        print(f"  ✅ Processed: {game_count} games, {self.positions_extracted - (self.positions_extracted - len(positions) * game_count)} positions")
        return game_count


def main():
    """Main integration workflow for V7P3R games."""
    
    print("="*60)
    print("V7P3R Game Integration for v5.3 - Single File Mode")
    print("="*60)
    
    # Paths
    base_dir = Path(__file__).parent.parent
    
    # V7P3R PGN file (comprehensive export)
    TARGET_PGN = "E:/Programming Stuff/Chess Engines/Chess Engine Playground/engine-metrics/raw_data/game_records/Lichess V7P3R Bot/lichess_v7p3r_bot_2026-05-18.pgn"
    
    # Output file
    OUTPUT_FILE = base_dir / "data" / "games" / "v7p3r_games_v5.3.jsonl"
    
    # Stockfish path
    STOCKFISH_PATH = base_dir.parent / "stockfish.exe"
    
    print(f"\nSource: {os.path.basename(TARGET_PGN)}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Analysis: 0.5s time limit per position")
    
    # Verify files exist
    if not os.path.exists(TARGET_PGN):
        print(f"\n❌ ERROR: PGN file not found: {TARGET_PGN}")
        return 1
    
    if not os.path.exists(STOCKFISH_PATH):
        print(f"\n❌ ERROR: Stockfish not found: {STOCKFISH_PATH}")
        return 1
    
    # Create output directory
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Initialize integrator
    integrator = V7P3RGameIntegrator(STOCKFISH_PATH, str(OUTPUT_FILE))
    
    print("\n" + "="*60)
    print("Starting extraction...")
    print("="*60)
    
    start_time = datetime.now()
    
    try:
        # Process PGN file
        with open(OUTPUT_FILE, 'w') as output_handle:
            integrator.process_pgn_file(TARGET_PGN, output_handle)
        
        # Summary
        elapsed = datetime.now() - start_time
        
        print("\n" + "="*60)
        print("Integration Complete!")
        print("="*60)
        print(f"Games processed: {integrator.games_processed:,}")
        print(f"Positions extracted: {integrator.positions_extracted:,}")
        print(f"Positions analyzed: {integrator.positions_analyzed:,}")
        print(f"Avg positions/game: {integrator.positions_extracted / integrator.games_processed:.1f}")
        print(f"Time elapsed: {elapsed}")
        print(f"Output: {OUTPUT_FILE.name}")
        print()
        
        return 0
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Extraction interrupted by user")
        print(f"Partial results saved to: {OUTPUT_FILE}")
        return 1
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
