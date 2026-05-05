"""
Stage 2: Parse V7P3R Historical Games

Filters PGN files for games where V7P3R lost by checkmate or resignation.
Extracts metadata and prepares for critical position analysis.

Usage:
    python scripts/parse_historical_games.py --pgn-dir "path/to/pgn/files" --output data/stage2_games/v7p3r_losses.json
"""

import os
import json
import chess.pgn
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse


@dataclass
class GameMetadata:
    """Metadata for a V7P3R losing game."""
    game_id: str
    date: str
    white: str
    black: str
    result: str
    termination: str
    time_control: str
    v7p3r_color: str  # "white" or "black"
    opponent: str
    moves: List[str]  # SAN notation
    ply_count: int
    pgn_file: str
    
    def to_dict(self):
        return asdict(self)


class HistoricalGameParser:
    """Parse PGN files and filter for V7P3R losses."""
    
    def __init__(self, pgn_directory: str):
        self.pgn_directory = Path(pgn_directory)
        self.games: List[GameMetadata] = []
        
    def parse_all_pgns(self) -> List[GameMetadata]:
        """Parse all PGN files in directory and subdirectories."""
        pgn_files = list(self.pgn_directory.rglob("*.pgn"))
        print(f"📂 Found {len(pgn_files)} PGN files")
        
        total_games = 0
        losing_games = 0
        
        for pgn_file in pgn_files:
            print(f"   Parsing: {pgn_file.name}...", end="")
            file_games, file_losses = self._parse_pgn_file(pgn_file)
            total_games += file_games
            losing_games += file_losses
            print(f" {file_losses}/{file_games} losses")
        
        print(f"\n📊 Summary:")
        print(f"   Total games parsed: {total_games}")
        print(f"   V7P3R losses: {losing_games}")
        print(f"   Win rate: {((total_games - losing_games) / total_games * 100):.1f}%")
        
        return self.games
    
    def _parse_pgn_file(self, pgn_file: Path) -> tuple:
        """Parse single PGN file and extract losing games."""
        file_games = 0
        file_losses = 0
        
        try:
            with open(pgn_file, 'r', encoding='utf-8') as f:
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    
                    file_games += 1
                    metadata = self._extract_game_metadata(game, pgn_file.name)
                    
                    if metadata is not None:
                        self.games.append(metadata)
                        file_losses += 1
        
        except Exception as e:
            print(f"\n   ⚠️  Error parsing {pgn_file.name}: {e}")
        
        return file_games, file_losses
    
    def _extract_game_metadata(self, game: chess.pgn.Game, pgn_filename: str) -> Optional[GameMetadata]:
        """
        Extract metadata if game is a V7P3R loss by checkmate/resignation.
        
        Returns None if not a relevant game.
        """
        headers = game.headers
        
        # Get player names
        white = headers.get("White", "").lower()
        black = headers.get("Black", "").lower()
        
        # Check if V7P3R played
        v7p3r_color = None
        opponent = None
        
        if "v7p3r" in white:
            v7p3r_color = "white"
            opponent = headers.get("Black", "Unknown")
        elif "v7p3r" in black:
            v7p3r_color = "black"
            opponent = headers.get("White", "Unknown")
        else:
            return None  # V7P3R didn't play
        
        # Get result
        result = headers.get("Result", "*")
        
        # Check if V7P3R lost
        v7p3r_lost = (
            (result == "0-1" and v7p3r_color == "white") or
            (result == "1-0" and v7p3r_color == "black")
        )
        
        if not v7p3r_lost:
            return None  # Not a loss
        
        # Get termination reason
        termination = headers.get("Termination", "Unknown")
        
        # Filter: Only checkmate or resignation
        # Exclude time forfeits, abandoned games, etc.
        if "time" in termination.lower() or "forfeit" in termination.lower():
            return None  # Time-related loss (no chess lesson)
        
        if termination.lower() not in ["normal", "abandoned"] and "resign" not in termination.lower():
            # Keep "Normal" (checkmate), "abandoned" (resignation in some formats), and explicit "resign"
            # Filter out draws, stalemates, etc.
            if result not in ["0-1", "1-0"]:
                return None
        
        # Extract moves
        moves = []
        board = game.board()
        for move in game.mainline_moves():
            moves.append(board.san(move))
            board.push(move)
        
        # Create metadata
        return GameMetadata(
            game_id=headers.get("Site", "").split("/")[-1] if "Site" in headers else f"{pgn_filename}_{len(self.games)}",
            date=headers.get("UTCDate", headers.get("Date", "Unknown")),
            white=headers.get("White", "Unknown"),
            black=headers.get("Black", "Unknown"),
            result=result,
            termination=termination,
            time_control=headers.get("TimeControl", "Unknown"),
            v7p3r_color=v7p3r_color,
            opponent=opponent,
            moves=moves,
            ply_count=len(moves),
            pgn_file=pgn_filename
        )
    
    def save_to_json(self, output_path: str):
        """Save filtered games to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "total_losses": len(self.games),
                "source_directory": str(self.pgn_directory)
            },
            "games": [game.to_dict() for game in self.games]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Saved {len(self.games)} losing games to: {output_path}")
        
        # Print statistics
        self._print_statistics()
    
    def _print_statistics(self):
        """Print detailed statistics about losing games."""
        if not self.games:
            return
        
        print(f"\n📈 Loss Statistics:")
        
        # Color distribution
        white_losses = sum(1 for g in self.games if g.v7p3r_color == "white")
        black_losses = sum(1 for g in self.games if g.v7p3r_color == "black")
        print(f"   As White: {white_losses} ({white_losses/len(self.games)*100:.1f}%)")
        print(f"   As Black: {black_losses} ({black_losses/len(self.games)*100:.1f}%)")
        
        # Termination types
        terminations = {}
        for game in self.games:
            term = game.termination
            terminations[term] = terminations.get(term, 0) + 1
        
        print(f"\n   Termination Reasons:")
        for term, count in sorted(terminations.items(), key=lambda x: -x[1]):
            print(f"      {term}: {count} ({count/len(self.games)*100:.1f}%)")
        
        # Game length distribution
        avg_ply = sum(g.ply_count for g in self.games) / len(self.games)
        short_games = sum(1 for g in self.games if g.ply_count < 30)
        medium_games = sum(1 for g in self.games if 30 <= g.ply_count < 60)
        long_games = sum(1 for g in self.games if g.ply_count >= 60)
        
        print(f"\n   Game Length:")
        print(f"      Average: {avg_ply:.1f} plies")
        print(f"      Short (<30): {short_games} ({short_games/len(self.games)*100:.1f}%)")
        print(f"      Medium (30-59): {medium_games} ({medium_games/len(self.games)*100:.1f}%)")
        print(f"      Long (60+): {long_games} ({long_games/len(self.games)*100:.1f}%)")
        
        # Top opponents
        opponents = {}
        for game in self.games:
            opp = game.opponent
            opponents[opp] = opponents.get(opp, 0) + 1
        
        print(f"\n   Top Opponents (Most Wins Against V7P3R):")
        for opp, count in sorted(opponents.items(), key=lambda x: -x[1])[:5]:
            print(f"      {opp}: {count} wins")


def main():
    parser = argparse.ArgumentParser(description="Parse V7P3R historical losing games")
    parser.add_argument(
        "--pgn-dir",
        type=str,
        default=r"E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\game_records\Lichess V7P3R Bot",
        help="Directory containing PGN files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/stage2_games/v7p3r_losses.json",
        help="Output JSON file path"
    )
    
    args = parser.parse_args()
    
    print("🚀 V7P3R Historical Game Parser (Stage 2)")
    print("=" * 60)
    print(f"📂 PGN Directory: {args.pgn_dir}")
    print(f"💾 Output: {args.output}")
    print("=" * 60)
    
    parser = HistoricalGameParser(args.pgn_dir)
    games = parser.parse_all_pgns()
    
    if games:
        parser.save_to_json(args.output)
        print(f"\n✅ Parsing complete! {len(games)} losing games extracted.")
    else:
        print(f"\n⚠️  No losing games found!")


if __name__ == "__main__":
    main()
