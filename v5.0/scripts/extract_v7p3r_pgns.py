"""
V7P3R AI v5.0 - PGN Position Extractor
======================================
Extracts positions from V7P3R historical games for training dataset.

Purpose:
- Read PGN files from Lichess V7P3R Bot games
- Replay each game move-by-move
- Extract FEN position BEFORE each v7p3r move
- Record move played (UCI + SAN)
- Output to JSONL in unified training format

Output Format:
- One JSONL file per PGN source
- Each line = one position where v7p3r moved
- Follows UNIFIED_TRAINING_DATASET.md schema (metadata + position + engine_decision blocks)
- Features block populated by separate feature calculator
- Stockfish analysis added by separate grading script

Usage:
    python scripts/extract_v7p3r_pgns.py --pgn-dir "path/to/pgns" --output data/raw/v7p3r_positions.jsonl
    python scripts/extract_v7p3r_pgns.py --pgn-file "specific.pgn" --output positions.jsonl --max-games 100
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Iterator
from datetime import datetime

import chess
import chess.pgn


class PGNPositionExtractor:
    """Extract training positions from V7P3R PGN game files."""
    
    def __init__(self, output_path: Path, player_name: str = "v7p3r_bot"):
        """
        Initialize extractor.
        
        Args:
            output_path: Path to output JSONL file
            player_name: Name of the engine in PGN files (default: v7p3r_bot)
        """
        self.output_path = output_path
        self.player_name = player_name.lower()
        self.positions_extracted = 0
        self.games_processed = 0
        self.games_skipped = 0
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"PGN Extractor initialized - output: {output_path}")
    
    def extract_from_directory(self, pgn_dir: Path, max_games: Optional[int] = None, 
                               recursive: bool = True) -> None:
        """
        Extract positions from all PGN files in a directory.
        
        Args:
            pgn_dir: Directory containing PGN files
            max_games: Maximum games to process (None = unlimited)
            recursive: Search subdirectories for PGN files
        """
        logging.info(f"Scanning directory: {pgn_dir}")
        
        # Find all PGN files
        pattern = "**/*.pgn" if recursive else "*.pgn"
        pgn_files = sorted(pgn_dir.glob(pattern))
        
        if not pgn_files:
            logging.warning(f"No PGN files found in {pgn_dir}")
            return
        
        logging.info(f"Found {len(pgn_files)} PGN files")
        
        # Process each file
        games_remaining = max_games
        with open(self.output_path, 'w', encoding='utf-8') as outfile:
            for pgn_file in pgn_files:
                if games_remaining is not None and games_remaining <= 0:
                    logging.info(f"Reached max_games limit ({max_games})")
                    break
                
                logging.info(f"Processing: {pgn_file.name}")
                
                games_from_file = self._extract_from_file(pgn_file, outfile, games_remaining)
                
                if games_remaining is not None:
                    games_remaining -= games_from_file
        
        self._log_summary()
    
    def extract_from_file(self, pgn_file: Path, max_games: Optional[int] = None) -> None:
        """
        Extract positions from a single PGN file.
        
        Args:
            pgn_file: Path to PGN file
            max_games: Maximum games to process (None = unlimited)
        """
        logging.info(f"Processing single file: {pgn_file}")
        
        with open(self.output_path, 'w', encoding='utf-8') as outfile:
            self._extract_from_file(pgn_file, outfile, max_games)
        
        self._log_summary()
    
    def _extract_from_file(self, pgn_file: Path, outfile, max_games: Optional[int]) -> int:
        """
        Extract positions from PGN file and write to output stream.
        
        Returns:
            Number of games processed from this file
        """
        games_from_file = 0
        
        try:
            with open(pgn_file, 'r', encoding='utf-8') as pgn:
                while True:
                    if max_games is not None and games_from_file >= max_games:
                        break
                    
                    game = chess.pgn.read_game(pgn)
                    if game is None:
                        break
                    
                    # Process game and extract positions
                    positions = self._extract_from_game(game, pgn_file.name)
                    
                    if positions:
                        # Write positions to output
                        for position_record in positions:
                            outfile.write(json.dumps(position_record) + '\n')
                        
                        self.games_processed += 1
                        games_from_file += 1
                    else:
                        self.games_skipped += 1
                    
                    # Progress logging
                    if (self.games_processed + self.games_skipped) % 100 == 0:
                        logging.info(f"Progress: {self.games_processed} games processed, "
                                   f"{self.positions_extracted} positions extracted")
        
        except Exception as e:
            logging.error(f"Error processing {pgn_file}: {e}")
        
        return games_from_file
    
    def _extract_from_game(self, game: chess.pgn.Game, source_file: str) -> list[Dict[str, Any]]:
        """
        Extract all positions where v7p3r moved from a single game.
        
        Args:
            game: python-chess Game object
            source_file: Name of source PGN file
        
        Returns:
            List of position records (empty if v7p3r not in game)
        """
        # Determine if v7p3r is white or black
        white = game.headers.get("White", "").lower()
        black = game.headers.get("Black", "").lower()
        
        v7p3r_color = None
        if self.player_name in white:
            v7p3r_color = chess.WHITE
        elif self.player_name in black:
            v7p3r_color = chess.BLACK
        else:
            # V7P3R not in this game
            return []
        
        # Extract game metadata
        game_metadata = {
            "white": game.headers.get("White", "Unknown"),
            "black": game.headers.get("Black", "Unknown"),
            "result": game.headers.get("Result", "*"),
            "date": game.headers.get("Date", ""),
            "event": game.headers.get("Event", ""),
            "site": game.headers.get("Site", ""),
            "time_control": game.headers.get("TimeControl", ""),
            "eco": game.headers.get("ECO", ""),
        }
        
        # Replay game and extract positions
        positions = []
        board = game.board()
        
        for move_num, move in enumerate(game.mainline_moves(), start=1):
            # Check if it's v7p3r's turn
            if board.turn == v7p3r_color:
                # This is a position where v7p3r is about to move
                position_record = self._create_position_record(
                    board=board,
                    move=move,
                    move_num=move_num,
                    v7p3r_color=v7p3r_color,
                    game_metadata=game_metadata,
                    source_file=source_file
                )
                positions.append(position_record)
                self.positions_extracted += 1
            
            # Make the move
            board.push(move)
        
        return positions
    
    def _create_position_record(self, board: chess.Board, move: chess.Move, 
                                move_num: int, v7p3r_color: chess.Color,
                                game_metadata: Dict[str, Any], source_file: str) -> Dict[str, Any]:
        """
        Create a training record for a single position.
        
        Follows UNIFIED_TRAINING_DATASET.md schema.
        """
        # Calculate material count
        material_count = self._count_material(board)
        
        # Determine game phase (simple heuristic based on material)
        game_phase = self._calculate_game_phase(material_count)
        
        # Calculate material balance
        material_balance = self._calculate_material_balance(board)
        
        # Check tactical flags
        in_check = board.is_check()
        
        # Create record following unified schema
        record = {
            # METADATA BLOCK
            "metadata": {
                "source": "v7p3r_pgn",
                "source_file": source_file,
                "game_id": f"{game_metadata['date']}_{game_metadata['white']}_vs_{game_metadata['black']}",
                "position_id": f"{source_file}_{move_num}",
                "extraction_timestamp": datetime.now().isoformat(),
                "v7p3r_version": "18.3",  # Update as needed
                "game_metadata": game_metadata,
            },
            
            # POSITION BLOCK
            "position": {
                "fen": board.fen(),
                "move_number": move_num,
                "side_to_move": "white" if v7p3r_color == chess.WHITE else "black",
                "game_phase": game_phase,
                "material_count": material_count,
                "material_balance": material_balance,
                "in_check": in_check,
                "castling_rights": board.castling_rights,
                "en_passant_square": board.ep_square,
            },
            
            # ENGINE DECISION BLOCK
            "engine_decision": {
                "move_uci": move.uci(),
                "move_san": board.san(move),
                "is_capture": board.is_capture(move),
                "is_check": board.gives_check(move),
                "is_castling": board.is_castling(move),
                "is_en_passant": board.is_en_passant(move),
                "promotion": move.promotion if move.promotion else None,
                # Evaluation details added later by profiler if available
                "v7p3r_eval_cp": None,
                "search_depth": None,
                "nodes_searched": None,
                "time_ms": None,
            },
            
            # STOCKFISH ANALYSIS BLOCK (populated by grading script)
            "stockfish_analysis": None,
            
            # FEATURES BLOCK (populated by feature calculator)
            "features": None,
        }
        
        return record
    
    def _count_material(self, board: chess.Board) -> int:
        """Count total material on board."""
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
        """Calculate material balance (positive = white ahead)."""
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
    
    def _log_summary(self) -> None:
        """Log extraction summary statistics."""
        logging.info("=" * 60)
        logging.info("EXTRACTION COMPLETE")
        logging.info(f"Games processed: {self.games_processed}")
        logging.info(f"Games skipped (no v7p3r): {self.games_skipped}")
        logging.info(f"Positions extracted: {self.positions_extracted}")
        logging.info(f"Output file: {self.output_path}")
        logging.info(f"File size: {self.output_path.stat().st_size / 1024:.2f} KB")
        logging.info("=" * 60)


def main():
    """Main entry point for PGN extraction."""
    parser = argparse.ArgumentParser(
        description="Extract training positions from V7P3R PGN game files"
    )
    parser.add_argument(
        "--pgn-dir",
        type=Path,
        help="Directory containing PGN files (searches recursively)"
    )
    parser.add_argument(
        "--pgn-file",
        type=Path,
        help="Single PGN file to process"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--player-name",
        type=str,
        default="v7p3r_bot",
        help="Engine player name in PGN files (default: v7p3r_bot)"
    )
    parser.add_argument(
        "--max-games",
        type=int,
        help="Maximum number of games to process"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search subdirectories"
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
    
    # Validate arguments
    if not args.pgn_dir and not args.pgn_file:
        parser.error("Must specify either --pgn-dir or --pgn-file")
    
    if args.pgn_dir and args.pgn_file:
        parser.error("Cannot specify both --pgn-dir and --pgn-file")
    
    # Create extractor
    extractor = PGNPositionExtractor(
        output_path=args.output,
        player_name=args.player_name
    )
    
    # Extract positions
    if args.pgn_dir:
        extractor.extract_from_directory(
            pgn_dir=args.pgn_dir,
            max_games=args.max_games,
            recursive=not args.no_recursive
        )
    else:
        extractor.extract_from_file(
            pgn_file=args.pgn_file,
            max_games=args.max_games
        )


if __name__ == "__main__":
    main()
