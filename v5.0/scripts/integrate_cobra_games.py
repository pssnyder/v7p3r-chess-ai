"""
C0BR4 Game Integration Script for v5.3
======================================
Extracts positions from C0BR4 Lichess games for training data expansion.

Process:
1. Load all C0BR4 PGN files from archive
2. Extract critical positions (every 5 moves + tactical moments)
3. Analyze with Stockfish (depth 20)
4. Calculate features (F000-F114 current + F200-F220 temporal)
5. Output to JSONL format

Expected output: ~200-250k positions from 10k+ games
"""

import chess
import chess.pgn
import chess.engine
import json
import os
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any
import sys

# Add scripts directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from calculate_features import FeatureCalculator, FeatureConfig
from temporal_feature_calculator import TemporalFeatureCalculator


class CobraGameIntegrator:
    """Integrates C0BR4 game data into v5.3 training dataset."""
    
    def __init__(self, stockfish_path: str, output_file: str):
        """
        Initialize integrator.
        
        Args:
            stockfish_path: Path to Stockfish executable
            output_file: Output JSONL file path
        """
        self.stockfish_path = stockfish_path
        self.output_file = output_file
        
        # Initialize feature calculators
        config = FeatureConfig.from_preset("full")
        self.feature_calc = FeatureCalculator(config)
        self.temporal_calc = TemporalFeatureCalculator(self.feature_calc)
        
        # Stockfish engine (will be initialized per-process)
        self.engine = None
        
        # Statistics
        self.games_processed = 0
        self.positions_extracted = 0
        self.errors = []
    
    def find_pgn_files(self, base_dir: str) -> List[str]:
        """Find all PGN files in C0BR4 game directories."""
        pgn_files = []
        
        # Main directory PGN files
        main_dir = Path(base_dir)
        for file in main_dir.glob("*.pgn"):
            pgn_files.append(str(file))
        
        # Archive directory PGN files
        archive_dir = main_dir / "c0br4_bot-archive"
        if archive_dir.exists():
            for file in archive_dir.glob("*.pgn"):
                pgn_files.append(str(file))
        
        return sorted(pgn_files)
    
    def should_extract_position(self, board: chess.Board, move_num: int, 
                               last_move: chess.Move) -> bool:
        """
        Determine if position should be extracted.
        
        Extract on:
        - Every 5th move
        - Captures
        - Checks
        - Castling
        - Promotions
        """
        # Every 5th move
        if move_num % 5 == 0:
            return True
        
        # Tactical moments
        if last_move:
            # Capture
            if board.is_capture(last_move):
                return True
            
            # Check
            if board.is_check():
                return True
            
            # Promotion
            if last_move.promotion:
                return True
            
            # Castling
            if board.is_castling(last_move):
                return True
        
        return False
    
    def analyze_position(self, board: chess.Board) -> Dict[str, Any]:
        """
        Analyze position with Stockfish (fast time-based limit).
        
        Returns:
            dict: Analysis including top 6 moves with scores
        """
        if not self.engine:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        
        try:
            # Get top 6 candidate moves with TIME LIMIT (not depth)
            # Use 0.5 seconds per position for speed
            info = self.engine.analyse(
                board,
                chess.engine.Limit(time=0.5),
                multipv=6
            )
            
            # Extract moves and scores
            candidates = []
            for idx, pv_info in enumerate(info):
                if 'pv' in pv_info and len(pv_info['pv']) > 0:
                    move = pv_info['pv'][0]
                    score = pv_info.get('score', chess.engine.Score(0, 0))
                    
                    # Convert score to centipawns
                    if score.is_mate():
                        cp = 10000 if score.mate() > 0 else -10000
                    else:
                        cp = score.score(mate_score=10000)
                    
                    candidates.append({
                        'move': move.uci(),
                        'score': cp,
                        'grade': idx  # 0=best, 1=2nd best, etc.
                    })
            
            return {
                'candidates': candidates,
                'best_move': candidates[0]['move'] if candidates else None,
                'best_score': candidates[0]['score'] if candidates else 0
            }
            
        except Exception as e:
            return {
                'candidates': [],
                'best_move': None,
                'best_score': 0,
                'error': str(e)
            }
    
    def extract_game_positions(self, game: chess.pgn.Game) -> List[Dict[str, Any]]:
        """Extract positions from a single game."""
        positions = []
        board = game.board()
        move_sequence = list(game.mainline_moves())
        
        if len(move_sequence) < 10:
            return positions  # Skip very short games
        
        previous_fen = None
        
        for move_num, move in enumerate(move_sequence, start=1):
            # Check if we should extract this position
            if self.should_extract_position(board, move_num, move):
                current_fen = board.fen()
                
                # Analyze position to get actual move grade
                analysis = self.analyze_position(board)
                
                # Find what grade the actual game move gets
                actual_move_uci = move.uci()
                grade = 5  # Default: not in top 6
                
                for candidate in analysis['candidates']:
                    if candidate['move'] == actual_move_uci:
                        grade = candidate['grade']
                        break
                
                # Calculate features with temporal context
                try:
                    if previous_fen:
                        features = self.temporal_calc.calculate_temporal_features(
                            current_fen=current_fen,
                            previous_fen=previous_fen,
                            last_move_uci=actual_move_uci
                        )
                        has_history = 1
                    else:
                        features = self.feature_calc.calculate_features_from_fen(
                            fen=current_fen,
                            move_uci=actual_move_uci
                        )
                        has_history = 0
                    
                    # Create position record
                    position = {
                        'position': {'fen': current_fen},
                        'engine_decision': {'move': actual_move_uci},
                        'stockfish_analysis': {
                            'best_move': analysis['best_move'],
                            'grade': grade,
                            'top_moves': analysis['candidates']
                        },
                        'features': features,
                        'has_history': has_history,
                        'source': 'c0br4_game',
                        'move_number': move_num
                    }
                    
                    positions.append(position)
                    previous_fen = current_fen
                    
                except Exception as e:
                    self.errors.append(f"Feature calc error move {move_num}: {e}")
            
            # Make the move on the board
            board.push(move)
        
        return positions
    
    def process_pgn_file(self, pgn_path: str, output_handle) -> int:
        """Process all games in a PGN file."""
        positions_from_file = 0
        
        try:
            with open(pgn_path, 'r', encoding='utf-8') as pgn:
                game_count = 0
                while True:
                    game = chess.pgn.read_game(pgn)
                    if game is None:
                        break
                    
                    game_count += 1
                    
                    try:
                        positions = self.extract_game_positions(game)
                        
                        # Write positions to output
                        for pos in positions:
                            output_handle.write(json.dumps(pos) + '\n')
                            positions_from_file += 1
                        
                        self.games_processed += 1
                        self.positions_extracted += positions_from_file
                        
                        # Progress update every 10 games
                        if game_count % 10 == 0:
                            print(f"  Games: {game_count}, Positions: {positions_from_file:,}")
                        
                    except KeyboardInterrupt:
                        print(f"\n⚠️  Interrupted! Saved {positions_from_file:,} positions from {game_count} games")
                        raise
                    except Exception as e:
                        self.errors.append(f"Game {game_count} processing error: {e}")
                        continue
        
        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.errors.append(f"File error {pgn_path}: {e}")
        
        return positions_from_file
    
    def integrate_all_games(self, base_dir: str):
        """Process all C0BR4 PGN files."""
        print("=" * 80)
        print("C0BR4 Game Integration for v5.3")
        print("=" * 80)
        print(f"Source: {base_dir}")
        print(f"Output: {self.output_file}")
        print(f"Stockfish: {self.stockfish_path}")
        print("=" * 80)
        print()
        
        # Find all PGN files
        pgn_files = self.find_pgn_files(base_dir)
        print(f"Found {len(pgn_files)} PGN files")
        print()
        
        # Process files with progress bar
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        with open(self.output_file, 'w', encoding='utf-8') as out:
            for pgn_file in tqdm(pgn_files, desc="Processing PGN files"):
                file_positions = self.process_pgn_file(pgn_file, out)
                self.positions_extracted += file_positions
                
                # Progress update every 10 files
                if len(pgn_files) % 10 == 0:
                    tqdm.write(f"✅ {self.games_processed:,} games, {self.positions_extracted:,} positions")
        
        # Close Stockfish engine
        if self.engine:
            self.engine.quit()
        
        # Print summary
        print()
        print("=" * 80)
        print("✅ Integration Complete!")
        print("=" * 80)
        print(f"PGN files processed: {len(pgn_files)}")
        print(f"Games processed: {self.games_processed:,}")
        print(f"Positions extracted: {self.positions_extracted:,}")
        print(f"Avg positions/game: {self.positions_extracted / self.games_processed:.1f}")
        print(f"Output: {self.output_file}")
        
        if self.errors:
            print(f"\n⚠️  Errors: {len(self.errors)}")
            print("First 5 errors:")
            for error in self.errors[:5]:
                print(f"  - {error}")
        
        print("=" * 80)
        print()


def main():
    """Main entry point."""
    
    # Configuration - TARGET SPECIFIC FILE
    TARGET_PGN = "E:/Programming Stuff/Chess Engines/Chess Engine Playground/engine-metrics/raw_data/game_records/Lichess C0BR4 Bot/lichess_c0br4_bot_2026-05-11.pgn"
    OUTPUT_FILE = "E:/Programming Stuff/Chess Engines/V7P3R Chess AI/v7p3r-chess-ai/v5.0/data/games/c0br4_games_v5.3.jsonl"
    STOCKFISH_PATH = "E:/Programming Stuff/Chess Engines/Tournament Engines/Stockfish/stockfish-windows-x86-64-avx2.exe"
    
    # Verify files exist
    if not os.path.exists(STOCKFISH_PATH):
        print(f"❌ Stockfish not found at: {STOCKFISH_PATH}")
        return
    
    if not os.path.exists(TARGET_PGN):
        print(f"❌ PGN file not found at: {TARGET_PGN}")
        return
    
    # Create integrator
    integrator = CobraGameIntegrator(STOCKFISH_PATH, OUTPUT_FILE)
    
    print("=" * 80)
    print("C0BR4 Game Integration for v5.3 - Single File Mode")
    print("=" * 80)
    print(f"Source: {TARGET_PGN}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Analysis: 0.5s time limit per position")
    print("=" * 80)
    print()
    
    # Process the single file
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
        positions = integrator.process_pgn_file(TARGET_PGN, out)
        print(f"\n✅ Processed: {integrator.games_processed:,} games, {positions:,} positions")
    
    # Close engine
    if integrator.engine:
        integrator.engine.quit()
    
    # Summary
    print()
    print("=" * 80)
    print("✅ Integration Complete!")
    print("=" * 80)
    print(f"Games processed: {integrator.games_processed:,}")
    print(f"Positions extracted: {integrator.positions_extracted:,}")
    print(f"Avg positions/game: {integrator.positions_extracted / integrator.games_processed:.1f}" if integrator.games_processed > 0 else "No games processed")
    print(f"Output: {OUTPUT_FILE}")
    
    if integrator.errors:
        print(f"\n⚠️  Errors: {len(integrator.errors)}")
        print("First 5 errors:")
        for error in integrator.errors[:5]:
            print(f"  - {error}")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
