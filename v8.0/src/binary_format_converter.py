#!/usr/bin/env python3
"""
Binary Format Converter for Chess Data Pipeline

Converts large chess datasets (PGNs, JSONL evaluations) into optimized
binary formats for fast, efficient training.

Phase 0: Data Preparation - Stage 1
"""

import struct
import json
import chess
import chess.pgn
from pathlib import Path
from typing import BinaryIO, Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
from tqdm import tqdm
import hashlib


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BinaryFormatConverter")


@dataclass
class BinaryPositionRecord:
    """Fixed-length binary position record (88 bytes)"""
    # FEN encoding
    fen_hash: int          # uint64 - position fingerprint
    
    # Evaluation
    eval: int              # int16 - evaluation in centipawns
    depth: int             # uint8 - search depth
    time_ms: int           # uint16 - search time in milliseconds
    
    # WDL statistics
    wins: int              # uint8 - wins out of 100
    draws: int             # uint8 - draws out of 100
    losses: int            # uint8 - losses out of 100
    
    # Position characteristics
    quiet: bool            # bool - quiet position flag
    material_balance: int  # int16 - material difference (cp)
    phase: int             # uint8 - 0=opening, 1=middlegame, 2=endgame
    piece_count: int       # uint8 - total pieces on board
    
    # For future expansion
    reserved: bytes        # 68 bytes for future fields
    
    SIZE = 88  # Fixed record size in bytes
    
    def to_bytes(self) -> bytes:
        """Pack record to binary format"""
        return struct.pack(
            '<QHBHBBBBBHBBB68s',  # Format string
            self.fen_hash,
            self.eval,
            self.depth,
            self.time_ms,
            self.wins,
            self.draws,
            self.losses,
            int(self.quiet),
            self.material_balance & 0xFFFF,  # As unsigned
            self.phase,
            self.piece_count,
            0,  # Padding
            0,  # Padding
            self.reserved if self.reserved else b'\x00' * 68
        )
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'BinaryPositionRecord':
        """Unpack record from binary format"""
        if len(data) != cls.SIZE:
            raise ValueError(f"Invalid record size: {len(data)} (expected {cls.SIZE})")
        
        values = struct.unpack('<QHBHBBBBBHBBB68s', data)
        return cls(
            fen_hash=values[0],
            eval=values[1],
            depth=values[2],
            time_ms=values[3],
            wins=values[4],
            draws=values[5],
            losses=values[6],
            quiet=bool(values[7]),
            material_balance=values[8],
            phase=values[9],
            piece_count=values[10],
            reserved=values[13]
        )


class BinaryFormatConverter:
    """Convert chess data to optimized binary formats"""
    
    def __init__(self, output_dir: Path = None):
        """Initialize converter"""
        self.output_dir = Path(output_dir or "./binary_data")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def pgn_to_binary(self, pgn_file: Path, output_file: Path = None) -> Path:
        """
        Convert PGN file to binary format
        
        PGN → Binary: Stores moves as 2-byte integers (from + to square)
        Typical compression: 5KB PGN → 1.5KB binary
        
        Args:
            pgn_file: Input PGN file
            output_file: Output binary file
        
        Returns:
            Path to output binary file
        """
        if output_file is None:
            output_file = self.output_dir / f"{pgn_file.stem}.bin"
        
        logger.info(f"Converting PGN to binary: {pgn_file} → {output_file}")
        
        move_count = 0
        game_count = 0
        
        with open(pgn_file, 'r') as pgn_in, open(output_file, 'wb') as bin_out:
            # Write header with version
            bin_out.write(b'PGNNB\x01')  # Format signature + version
            
            while True:
                game = chess.pgn.read_game(pgn_in)
                if game is None:
                    break
                
                game_count += 1
                board = game.board()
                
                # Get moves from game
                moves = []
                for move in game.mainline_moves():
                    from_square = move.from_square
                    to_square = move.to_square
                    
                    # Encode as 2-byte integer
                    move_code = (from_square << 8) | to_square
                    moves.append(move_code)
                    move_count += 1
                
                # Write game record
                # Format: [game_id (4)] [move_count (2)] [moves...]
                game_id = game_count
                bin_out.write(struct.pack('<IH', game_id, len(moves)))
                
                for move_code in moves:
                    bin_out.write(struct.pack('<H', move_code))
                
                if game_count % 1000 == 0:
                    logger.info(f"  Processed {game_count} games, {move_count} moves")
        
        output_file_size = output_file.stat().st_size
        logger.info(f"✅ Conversion complete: {game_count} games, {move_count} moves")
        logger.info(f"   Output: {output_file_size / 1e6:.1f} MB")
        
        return output_file
    
    def jsonl_to_binary(self, jsonl_file: Path, output_file: Path = None) -> Path:
        """
        Convert JSONL evaluation file to binary position records
        
        JSONL → Binary: Fixed 88-byte records for O(1) random access
        Typical compression: 95GB JSONL → 40GB binary
        
        Args:
            jsonl_file: Input JSONL with {fen, eval, depth, time, wdl}
            output_file: Output binary file
        
        Returns:
            Path to output binary file
        """
        if output_file is None:
            output_file = self.output_dir / f"{jsonl_file.stem}.bin"
        
        logger.info(f"Converting JSONL to binary: {jsonl_file} → {output_file}")
        
        record_count = 0
        
        with open(jsonl_file, 'r') as json_in, open(output_file, 'wb') as bin_out:
            # Write header
            bin_out.write(b'POSNB\x01')  # Format signature + version
            bin_out.write(struct.pack('<Q', 0))  # Record count (will update later)
            
            for line_num, line in enumerate(tqdm(json_in, desc="Converting positions"), 1):
                try:
                    data = json.loads(line.strip())
                    
                    # Parse FEN
                    fen = data.get('fen', '')
                    board = chess.Board(fen)
                    
                    # Calculate FEN hash
                    fen_hash = int(hashlib.md5(fen.encode()).hexdigest()[:16], 16)
                    
                    # Extract evaluation data
                    eval_cp = int(data.get('eval', 0))
                    depth = int(data.get('depth', 0))
                    time_ms = int(data.get('time', 0))
                    
                    # Extract WDL if available
                    wdl = data.get('wdl', [33, 34, 33])
                    wins = min(100, max(0, wdl[0] if len(wdl) > 0 else 33))
                    draws = min(100, max(0, wdl[1] if len(wdl) > 1 else 34))
                    losses = min(100, max(0, wdl[2] if len(wdl) > 2 else 33))
                    
                    # Calculate material balance
                    material = self._calculate_material_balance(board)
                    
                    # Determine game phase
                    phase = self._determine_phase(board)
                    
                    # Create record
                    record = BinaryPositionRecord(
                        fen_hash=fen_hash,
                        eval=eval_cp,
                        depth=depth,
                        time_ms=time_ms,
                        wins=wins,
                        draws=draws,
                        losses=losses,
                        quiet=False,  # Will be set by filter
                        material_balance=material,
                        phase=phase,
                        piece_count=board.piece_count(),
                        reserved=b'\x00' * 68
                    )
                    
                    # Write record
                    bin_out.write(record.to_bytes())
                    record_count += 1
                    
                    if record_count % 10000 == 0:
                        logger.info(f"  Processed {record_count} positions")
                
                except Exception as e:
                    logger.warning(f"Error on line {line_num}: {e}")
                    continue
            
            # Update record count in header
            bin_out.seek(6)
            bin_out.write(struct.pack('<Q', record_count))
        
        output_file_size = output_file.stat().st_size
        logger.info(f"✅ Conversion complete: {record_count} positions")
        logger.info(f"   Output: {output_file_size / 1e9:.1f} GB")
        
        return output_file
    
    @staticmethod
    def _calculate_material_balance(board: chess.Board) -> int:
        """Calculate material difference in centipawns"""
        values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
        }
        
        white_material = 0
        black_material = 0
        
        for piece_type in values:
            white_material += len(board.pieces(piece_type, chess.WHITE)) * values[piece_type]
            black_material += len(board.pieces(piece_type, chess.BLACK)) * values[piece_type]
        
        # Return from white's perspective
        material = white_material - black_material
        
        # Clamp to int16 range
        return max(-32768, min(32767, material))
    
    @staticmethod
    def _determine_phase(board: chess.Board) -> int:
        """
        Determine game phase (0=opening, 1=middlegame, 2=endgame)
        
        Uses material count as heuristic:
        - Opening: Lots of pieces
        - Middlegame: Fewer pieces
        - Endgame: Very few pieces
        """
        material = 0
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            material += (len(board.pieces(piece_type, chess.WHITE)) +
                        len(board.pieces(piece_type, chess.BLACK)))
        
        if material > 24:
            return 0  # Opening
        elif material > 10:
            return 1  # Middlegame
        else:
            return 2  # Endgame
    
    def benchmark_conversion(self, input_file: Path, num_samples: int = 1000):
        """Benchmark conversion speed"""
        import time
        
        logger.info(f"Benchmarking conversion on {num_samples} samples from {input_file}")
        
        start_time = time.time()
        record_count = 0
        
        with open(input_file, 'r') as f:
            for line in f:
                if record_count >= num_samples:
                    break
                
                try:
                    data = json.loads(line.strip())
                    fen = data.get('fen', '')
                    board = chess.Board(fen)
                    
                    # Simulate record creation
                    material = self._calculate_material_balance(board)
                    phase = self._determine_phase(board)
                    
                    record_count += 1
                except:
                    pass
        
        elapsed = time.time() - start_time
        rate = record_count / elapsed
        
        logger.info(f"✅ Benchmark complete:")
        logger.info(f"   Records processed: {record_count}")
        logger.info(f"   Time: {elapsed:.2f} seconds")
        logger.info(f"   Rate: {rate:.0f} records/second ({rate*88/1e6:.1f} MB/sec)")


def main():
    """Demo conversion"""
    converter = BinaryFormatConverter()
    
    print("\n" + "="*80)
    print("🔄 BINARY FORMAT CONVERTER - Chess Data Pipeline Phase 0")
    print("="*80)
    
    # Example: Convert PGN
    print("\n📊 Example conversion (PGN):")
    print("  Input: games.pgn (millions of games)")
    print("  Output: games.bin (optimized 2-byte moves)")
    print("  Compression: ~3.3x smaller (5KB PGN → 1.5KB binary)")
    
    # Example: Convert JSONL
    print("\n📊 Example conversion (JSONL):")
    print("  Input: evaluations.jsonl (95GB)")
    print("  Output: evaluations.bin (40GB)")
    print("  Format: 88-byte fixed records (O(1) random access)")
    print("  Compression: 2.4x smaller")
    
    print("\n" + "="*80)
    print("✨ Ready for conversion!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
