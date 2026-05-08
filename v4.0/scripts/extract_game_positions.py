#!/usr/bin/env python3
"""
Extract training positions from all V7P3R historical games.

Strategy: Learn from ALL moves (V7P3R + opponents) with Stockfish enrichment.
Focus: Best move learning, not avoidance. Game phase aware.

Game Phase Classification:
- Opening: Moves 1-20
- Middlegame: Moves 21-50
- Endgame: Moves 51+
"""

import chess
import chess.pgn
import chess.engine
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import sys
from tqdm import tqdm
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.core.chess_state_extractor import ChessStateExtractor


@dataclass
class GamePosition:
    """Single position from a game."""
    game_id: str
    position_index: int
    ply: int  # Half-move number (1-based)
    fen: str
    move_played_uci: str
    move_played_san: str
    game_phase: str  # 'opening', 'middlegame', 'endgame'
    player: str  # Who played this move
    opponent: str
    result: str  # Game result from player's perspective
    
    # Stockfish analysis
    top_moves: List[Dict[str, Any]]  # [{uci, san, score, weight}, ...]
    position_features: List[float]  # 690-dim features
    
    # Metadata
    game_date: str
    time_control: str


class GamePositionExtractor:
    """Extract positions from PGN games with Stockfish analysis."""
    
    # Game phase thresholds
    OPENING_END = 20  # Moves 1-20
    MIDDLEGAME_END = 50  # Moves 21-50
    # Endgame: 51+
    
    def __init__(self, stockfish_path: str, analysis_time: float = 0.5, num_top_moves: int = 5):
        """
        Args:
            stockfish_path: Path to Stockfish executable
            analysis_time: Time in seconds for Stockfish analysis per position
            num_top_moves: Number of top moves to extract from Stockfish
        """
        self.stockfish_path = stockfish_path
        self.analysis_time = analysis_time
        self.num_top_moves = num_top_moves
        self.engine: Optional[chess.engine.SimpleEngine] = None
        self.feature_extractor = ChessStateExtractor()
        
    def initialize_stockfish(self):
        """Start Stockfish engine once for reuse."""
        print("   Starting Stockfish engine...")
        self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        print("   ✅ Stockfish ready")
    
    def cleanup(self):
        """Close Stockfish engine."""
        if self.engine:
            self.engine.quit()
            print("   ✅ Stockfish engine closed")
    
    def classify_game_phase(self, ply: int) -> str:
        """
        Classify game phase based on move number.
        
        Args:
            ply: Half-move number (1-based)
            
        Returns:
            'opening', 'middlegame', or 'endgame'
        """
        move_number = (ply + 1) // 2  # Convert ply to full move number
        
        if move_number <= self.OPENING_END:
            return 'opening'
        elif move_number <= self.MIDDLEGAME_END:
            return 'middlegame'
        else:
            return 'endgame'
    
    def get_stockfish_top_moves(self, board: chess.Board) -> List[Dict[str, Any]]:
        """
        Get top N moves from Stockfish with evaluations.
        
        Returns:
            List of dicts with keys: uci, san, score, weight
        """
        if not self.engine:
            raise RuntimeError("Stockfish engine not initialized")
        
        # Analyze position
        info = self.engine.analyse(
            board,
            chess.engine.Limit(time=self.analysis_time),
            multipv=self.num_top_moves
        )
        
        # Extract moves and scores
        moves = []
        for i, variation in enumerate(info):
            if 'pv' not in variation or not variation['pv']:
                continue
            
            best_move = variation['pv'][0]
            
            # Get score (convert mate to centipawns)
            score_info = variation.get('score')
            if score_info:
                if score_info.is_mate():
                    # Convert mate distance to large centipawn value
                    mate_in = score_info.relative.mate()
                    score = 10000 if mate_in > 0 else -10000
                else:
                    score = score_info.relative.score()
            else:
                score = 0
            
            # Calculate weight (exponential decay)
            # Top move: 1.0, 2nd: 0.8, 3rd: 0.6, 4th: 0.4, 5th: 0.2, rest: 0.1
            if i == 0:
                weight = 1.0
            elif i < 5:
                weight = 1.0 - (i * 0.2)
            else:
                weight = 0.1
            
            moves.append({
                'uci': best_move.uci(),
                'san': board.san(best_move),
                'score': score,
                'weight': weight
            })
        
        return moves
    
    def extract_from_game(self, game: chess.pgn.Game, game_id: str) -> List[GamePosition]:
        """
        Extract all positions from a single game.
        
        Args:
            game: python-chess Game object
            game_id: Unique identifier for the game
            
        Returns:
            List of GamePosition objects
        """
        positions = []
        
        # Extract game metadata
        headers = game.headers
        white_player = headers.get('White', 'Unknown')
        black_player = headers.get('Black', 'Unknown')
        result = headers.get('Result', '*')
        game_date = headers.get('Date', 'Unknown')
        time_control = headers.get('TimeControl', 'Unknown')
        
        # Traverse game
        board = game.board()
        node = game
        
        ply = 0
        position_index = 0
        
        while node.variations:
            node = node.variation(0)
            move = node.move
            ply += 1
            
            # Determine player and opponent
            if board.turn == chess.WHITE:
                player = white_player
                opponent = black_player
                player_result = result  # From White's perspective
            else:
                player = black_player
                opponent = white_player
                # Flip result for Black's perspective
                if result == '1-0':
                    player_result = '0-1'
                elif result == '0-1':
                    player_result = '1-0'
                else:
                    player_result = result
            
            # Get current position (before move)
            fen = board.fen()
            game_phase = self.classify_game_phase(ply)
            
            # Get Stockfish analysis
            try:
                top_moves = self.get_stockfish_top_moves(board)
            except Exception as e:
                print(f"      ⚠️  Stockfish analysis failed for ply {ply}: {e}")
                top_moves = []
            
            # Extract position features
            try:
                features = self.feature_extractor.extract(board)
                features_list = features.tolist()
            except Exception as e:
                print(f"      ⚠️  Feature extraction failed for ply {ply}: {e}")
                features_list = [0.0] * 690
            
            # Create position object
            position = GamePosition(
                game_id=game_id,
                position_index=position_index,
                ply=ply,
                fen=fen,
                move_played_uci=move.uci(),
                move_played_san=board.san(move),
                game_phase=game_phase,
                player=player,
                opponent=opponent,
                result=player_result,
                top_moves=top_moves,
                position_features=features_list,
                game_date=game_date,
                time_control=time_control
            )
            
            positions.append(position)
            position_index += 1
            
            # Make move
            board.push(move)
        
        return positions
    
    def process_pgn_file(self, pgn_path: Path, max_games: Optional[int] = None) -> List[GamePosition]:
        """
        Process a single PGN file.
        
        Args:
            pgn_path: Path to PGN file
            max_games: Maximum number of games to process (None = all)
            
        Returns:
            List of all positions from all games
        """
        all_positions = []
        
        print(f"\n📂 Processing: {pgn_path.name}")
        
        with open(pgn_path) as pgn_file:
            game_count = 0
            
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                
                game_count += 1
                if max_games and game_count > max_games:
                    break
                
                # Create game ID
                game_id = f"{pgn_path.stem}_game{game_count}"
                
                # Extract positions
                try:
                    positions = self.extract_from_game(game, game_id)
                    all_positions.extend(positions)
                    
                    if game_count % 10 == 0:
                        print(f"   Processed {game_count} games, {len(all_positions)} positions...")
                
                except Exception as e:
                    print(f"   ⚠️  Error processing game {game_count}: {e}")
                    continue
        
        print(f"   ✅ Processed {game_count} games, {len(all_positions)} total positions")
        return all_positions
    
    def process_all_pgns(self, pgn_dir: Path, max_games_per_file: Optional[int] = None) -> List[GamePosition]:
        """
        Process all PGN files in directory.
        
        Args:
            pgn_dir: Directory containing PGN files
            max_games_per_file: Max games per file (None = all)
            
        Returns:
            List of all positions from all games
        """
        pgn_files = sorted(pgn_dir.glob('*.pgn'))
        
        if not pgn_files:
            raise FileNotFoundError(f"No PGN files found in {pgn_dir}")
        
        print(f"📁 Found {len(pgn_files)} PGN files")
        
        all_positions = []
        
        for pgn_file in pgn_files:
            positions = self.process_pgn_file(pgn_file, max_games_per_file)
            all_positions.extend(positions)
        
        return all_positions
    
    def save_dataset(self, positions: List[GamePosition], output_path: Path):
        """Save positions to JSON file."""
        # Convert to dict format
        data = {
            'metadata': {
                'total_positions': len(positions),
                'extraction_date': datetime.now().isoformat(),
                'stockfish_analysis_time': self.analysis_time,
                'game_phase_thresholds': {
                    'opening_end': self.OPENING_END,
                    'middlegame_end': self.MIDDLEGAME_END
                }
            },
            'positions': [asdict(pos) for pos in positions]
        }
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving dataset to {output_path}...")
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Get file size
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"   ✅ Saved {len(positions)} positions ({size_mb:.1f} MB)")
    
    def print_statistics(self, positions: List[GamePosition]):
        """Print dataset statistics."""
        total = len(positions)
        
        # Phase distribution
        opening = sum(1 for p in positions if p.game_phase == 'opening')
        middlegame = sum(1 for p in positions if p.game_phase == 'middlegame')
        endgame = sum(1 for p in positions if p.game_phase == 'endgame')
        
        # Player distribution (V7P3R vs opponents)
        v7p3r_positions = sum(1 for p in positions if 'v7p3r' in p.player.lower())
        opponent_positions = total - v7p3r_positions
        
        # Result distribution
        wins = sum(1 for p in positions if p.result in ['1-0', '0-1'] and '1' in p.result.split('-')[0] if p.result.startswith('1'))
        losses = sum(1 for p in positions if p.result in ['1-0', '0-1'] and '0' in p.result.split('-')[0] if p.result.startswith('0'))
        draws = sum(1 for p in positions if p.result == '1/2-1/2')
        
        print(f"\n📊 Dataset Statistics")
        print(f"   Total positions: {total:,}")
        print(f"\n   Game Phase Distribution:")
        print(f"      Opening (moves 1-20): {opening:,} ({opening/total*100:.1f}%)")
        print(f"      Middlegame (21-50): {middlegame:,} ({middlegame/total*100:.1f}%)")
        print(f"      Endgame (51+): {endgame:,} ({endgame/total*100:.1f}%)")
        print(f"\n   Player Distribution:")
        print(f"      V7P3R positions: {v7p3r_positions:,} ({v7p3r_positions/total*100:.1f}%)")
        print(f"      Opponent positions: {opponent_positions:,} ({opponent_positions/total*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Extract training positions from V7P3R historical games'
    )
    parser.add_argument('--pgn-file', type=str,
                       default='E:/Programming Stuff/Chess Engines/Chess Engine Playground/engine-metrics/raw_data/game_records/Lichess V7P3R Bot/lichess_v7p3r_bot_2026-04-09.pgn',
                       help='Path to PGN file')
    parser.add_argument('--output', type=str,
                       default='data/stage2_games/historical_positions.json',
                       help='Output JSON file')
    parser.add_argument('--stockfish-path', type=str,
                       default='E:/Programming Stuff/Chess Engines/Tournament Engines/Stockfish/stockfish-windows-x86-64-avx2.exe',
                       help='Path to Stockfish executable')
    parser.add_argument('--analysis-time', type=float, default=0.5,
                       help='Stockfish analysis time per position (seconds)')
    parser.add_argument('--max-games', type=int, default=None,
                       help='Maximum number of games to process (None = all)')
    parser.add_argument('--num-top-moves', type=int, default=5,
                       help='Number of top moves to extract from Stockfish')
    
    args = parser.parse_args()
    
    print("🚀 V7P3R Historical Game Position Extractor")
    print("=" * 70)
    print(f"📂 PGN File: {args.pgn_file}")
    print(f"💾 Output: {args.output}")
    print(f"⚙️  Stockfish: {args.stockfish_path}")
    print(f"⏱️  Analysis time: {args.analysis_time}s per position")
    print(f"🎯 Top moves: {args.num_top_moves}")
    print("\n📝 Strategy:")
    print("   - Extract ALL positions from V7P3R's games")
    print("   - Stockfish analyzes each position (best moves)")
    print("   - Train on what SHOULD have been played")
    print("   - Real positions V7P3R will face on Lichess")
    print("=" * 70)
    
    # Create extractor
    extractor = GamePositionExtractor(args.stockfish_path, args.analysis_time, args.num_top_moves)
    
    try:
        # Initialize Stockfish
        extractor.initialize_stockfish()
        
        # Process single PGN file
        pgn_file = Path(args.pgn_file)
        if not pgn_file.exists():
            raise FileNotFoundError(f"PGN file not found: {pgn_file}")
        
        positions = extractor.process_pgn_file(pgn_file, args.max_games)
        
        # Print statistics
        extractor.print_statistics(positions)
        
        # Save dataset
        output_path = Path(args.output)
        extractor.save_dataset(positions, output_path)
        
        print(f"\n✅ Position extraction complete!")
        print(f"   {len(positions):,} positions ready for training")
        print(f"   Next step: Train combined model (puzzles + games)")
        print(f"   Command: python scripts/train_combined_dataset.py")
        
    finally:
        # Cleanup
        extractor.cleanup()


if __name__ == '__main__':
    main()
