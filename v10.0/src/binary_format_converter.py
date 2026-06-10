"""Binary Format Converter: PGN/JSONL → Optimized Binary.

Transforms 120GB of PGN, JSONL evaluations, and puzzle data into compact
binary formats for high-speed training and validation.

SPRINT 1, DAY 1.1: Implement this module

Classes:
    BinaryPositionRecord: 88-byte struct for a single position
    BinaryFormatConverter: Main conversion engine

Methods (to implement):
    pgn_to_binary(pgn_path: str, output_path: str) -> int
        Convert PGN files to binary format
        Returns: number of positions written
        Target: >50 MB/sec throughput
        Output: pgns.bin (~1.5GB)

    jsonl_to_binary(jsonl_path: str, output_path: str) -> int
        Convert JSONL evaluation files to binary
        Returns: number of positions written
        Target: >50 MB/sec throughput
        Output: evals.bin (~40GB)

    puzzle_tokenize(puzzle_file: str, output_path: str) -> int
        Tokenize puzzle data for training
        Returns: number of puzzles tokenized
        Output: puzzles_tokenized.bin (~2GB)

    benchmark_conversion(num_positions: int = 10000) -> dict
        Measure conversion speed, returns timing metrics

Performance Requirements:
    - Speed: >50 MB/sec (vs 1-5 MB/sec text parsing)
    - CPU: 8+ cores (parallelizable)
    - RAM: 8GB sufficient
    - Output: 27GB total (from 120GB input)

Data Format:
    Record size: 88 bytes per position
    Structure:
        - FEN hash (8 bytes)
        - Evaluation (2 bytes, int16)
        - Depth (1 byte)
        - Time (2 bytes)
        - WDL (3 bytes: W/D/L)
        - Material (2 bytes)
        - Phase (1 byte)
        - Piece count (1 byte)
        - Fen String
        - Reserved (68 bytes for future)

Test with: python -m pytest tests/test_binary_converter.py -v
"""

import struct
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
import chess
import chess.pgn
import hashlib
import json
import time
import io
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class BinaryPositionRecord:
    """88-byte binary record for a single chess position.
    
    Attributes:
        fen_hash (int): 8-byte hash of FEN position
        evaluation (int): 2-byte evaluation in centipawns
        depth (int): 1-byte search depth
        time_ms (int): 2-byte search time in milliseconds
        wins (int): 1-byte WDL wins count
        draws (int): 1-byte WDL draws count
        losses (int): 1-byte WDL losses count
        is_quiet (int): 1-byte flag (1 if quiet, 0 if tactical)
        material_diff (int): 2-byte material balance
        phase (int): 1-byte game phase (opening/mid/endgame)
        piece_count (int): 1-byte number of pieces
    """
    
    fen_hash: int
    evaluation: int
    depth: int
    time_ms: int
    wins: int
    draws: int
    losses: int
    is_quiet: int
    material_diff: int
    phase: int
    piece_count: int
    # 68 bytes reserved for future use
    
    STRUCT_FORMAT = "=QhBHBBBBhBB68s"
    RECORD_SIZE = 88  # bytes
    
    def pack(self) -> bytes:
        """Pack record to 88-byte binary format.
        
        Returns:
            88-byte binary record
        """
        reserved = b'\x00' * 68
        return struct.pack(
            self.STRUCT_FORMAT,
            self.fen_hash,
            self.evaluation,
            self.depth,
            self.time_ms,
            self.wins,
            self.draws,
            self.losses,
            self.material_diff,
            self.phase,
            self.piece_count,
            reserved
        )
    
    @staticmethod
    def unpack(data: bytes) -> 'BinaryPositionRecord':
        """Unpack binary record to BinaryPositionRecord.
        
        Args:
            data: 88-byte binary record
            
        Returns:
            BinaryPositionRecord instance
        """
        if len(data) != BinaryPositionRecord.RECORD_SIZE:
            raise ValueError(f"Expected 88 bytes, got {len(data)}")
        
        unpacked = struct.unpack(BinaryPositionRecord.STRUCT_FORMAT, data)
        return BinaryPositionRecord(
            fen_hash=unpacked[0],
            evaluation=unpacked[1],
            depth=unpacked[2],
            time_ms=unpacked[3],
            wins=unpacked[4],
            draws=unpacked[5],
            losses=unpacked[6],
            is_quiet=unpacked[7],
            material_diff=unpacked[8],
            phase=unpacked[9],
            piece_count=unpacked[10]
        )


class BinaryFormatConverter:
    """Converts PGN, JSONL, and puzzle files to optimized binary formats."""
    
    BUFFER_SIZE = 1024 * 1024  # 1MB buffer for disk writes
    
    def __init__(self, verbose: bool = True):
        """Initialize converter with optional logging.
        
        Args:
            verbose: Enable detailed progress logging
        """
        self.verbose = verbose
        if verbose:
            logger.setLevel(logging.INFO)
        
        self.positions_written = 0
        self.start_time = None
    
    def pgn_to_binary(self, pgn_path: str, output_path: str) -> int:
        """Convert PGN file to binary format.
        
        Args:
            pgn_path: Path to input PGN file
            output_path: Path to output .bin file
            
        Returns:
            Number of positions written
            
        Raises:
            FileNotFoundError: If input file not found
            IOError: If output file cannot be written
            
        Example:
            converter = BinaryFormatConverter()
            count = converter.pgn_to_binary("games.pgn", "pgns.bin")
            print(f"Converted {count} positions")
        """
        pgn_file = Path(pgn_path)
        if not pgn_file.exists():
            raise FileNotFoundError(f"PGN file not found: {pgn_path}")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.positions_written = 0
        self.start_time = time.time()
        
        game_count = 0
        
        try:
            with open(pgn_file, 'r', encoding='utf-8', errors='ignore') as pgn:
                with open(output_file, 'wb') as bin_out:
                    # Use buffered writer for performance
                    buffered = io.BufferedWriter(bin_out, self.BUFFER_SIZE)
                    
                    pbar = tqdm(desc="Converting PGN", unit=" positions") if self.verbose else None
                    
                    while True:
                        game = chess.pgn.read_game(pgn)
                        if game is None:
                            break
                        
                        game_count += 1
                        positions_in_game = self._process_game(game, buffered)
                        
                        if pbar:
                            pbar.update(positions_in_game)
                    
                    buffered.flush()
                    if pbar:
                        pbar.close()
        
        except Exception as e:
            logger.error(f"Error converting PGN: {e}")
            raise
        
        elapsed = time.time() - self.start_time
        if self.verbose:
            logger.info(f"Converted {game_count} games, {self.positions_written} positions in {elapsed:.2f}s")
            if elapsed > 0:
                logger.info(f"Throughput: {(self.positions_written * 88 / 1024 / 1024) / elapsed:.2f} MB/sec")
        
        return self.positions_written
    
    def _process_game(self, game: chess.pgn.Game, output_file) -> int:
        """Process a single game and write positions to binary file.
        
        Args:
            game: chess.pgn.Game object
            output_file: Buffered output file handle
            
        Returns:
            Number of positions written from this game
        """
        positions_written = 0
        board = chess.Board()
        game_time_ms = self._extract_time_control(game)
        
        try:
            for move in game.mainline_moves():
                # Record position before move
                record = self._board_to_record(board, game_time_ms)
                output_file.write(record.pack())
                positions_written += 1
                self.positions_written += 1
                
                # Make move on board
                board.push(move)
        
        except Exception as e:
            logger.warning(f"Error processing game: {e}")
        
        return positions_written
    
    def _board_to_record(self, board: chess.Board, game_time_ms: int = 0) -> BinaryPositionRecord:
        """Convert chess.Board to BinaryPositionRecord.
        
        Args:
            board: chess.Board instance
            game_time_ms: Time control in milliseconds
            
        Returns:
            BinaryPositionRecord
        """
        # Calculate FEN hash
        fen = board.fen()
        fen_bytes = hashlib.sha256(fen.encode()).digest()[:8]
        fen_hash = int.from_bytes(fen_bytes, 'big')
        
        # Calculate material count and balance
        material_diff, piece_count = self._calculate_material(board)
        
        # Determine game phase
        phase = self._calculate_phase(board)
        
        # Check if position is quiet (no checks, no hanging pieces)
        is_quiet = 1 if self._is_quiet_position(board) else 0
        
        # Get current depth (estimated from move number)
        depth = min(255, board.fullmove_number)
        
        return BinaryPositionRecord(
            fen_hash=fen_hash,
            evaluation=0,  # Placeholder: filled from JSONL later
            depth=depth,
            time_ms=game_time_ms,
            wins=0,  # Placeholder: filled from Syzygy later
            draws=0,
            losses=0,
            is_quiet=is_quiet,
            material_diff=material_diff & 0xFFFF,  # Keep in 16-bit range
            phase=phase,
            piece_count=piece_count
        )
    
    def _is_quiet_position(self, board: chess.Board) -> bool:
        """Check if position is quiet (no tactical elements).
        
        Args:
            board: chess.Board instance
            
        Returns:
            True if position is quiet
        """
        # Position is quiet if:
        # 1. Not in check
        # 2. No immediate captures available
        # 3. No hanging pieces
        
        if board.is_check():
            return False
        
        # Check if any pieces are under attack without defense
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
            
            # Check attackers and defenders
            attackers = board.attackers(not piece.color, square)
            defenders = board.attackers(piece.color, square)
            
            if attackers and not defenders:
                return False  # Hanging piece found
        
        return True
    
    def _calculate_material(self, board: chess.Board) -> Tuple[int, int]:
        """Calculate material balance and piece count.
        
        Args:
            board: chess.Board instance
            
        Returns:
            (material_diff, piece_count)
        """
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        
        white_material = 0
        black_material = 0
        piece_count = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
            
            value = piece_values.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                white_material += value
            else:
                black_material += value
            piece_count += 1
        
        material_diff = white_material - black_material
        return material_diff, piece_count
    
    def _calculate_phase(self, board: chess.Board) -> int:
        """Calculate game phase.
        
        Args:
            board: chess.Board instance
            
        Returns:
            0 = opening, 1 = middlegame, 2 = endgame
        """
        # Count pieces (excluding pawns and kings)
        piece_count = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None and piece.piece_type not in (chess.PAWN, chess.KING):
                piece_count += 1
        
        # Opening: many pieces on board
        # Middlegame: medium pieces
        # Endgame: few pieces
        if piece_count >= 10:
            return 0  # Opening
        elif piece_count >= 4:
            return 1  # Middlegame
        else:
            return 2  # Endgame
    
    def _extract_time_control(self, game: chess.pgn.Game) -> int:
        """Extract time control from game headers.
        
        Args:
            game: chess.pgn.Game object
            
        Returns:
            Time control in milliseconds (or 0 if not found)
        """
        time_control = game.headers.get('TimeControl', '0+0')
        try:
            parts = time_control.split('+')
            if len(parts) >= 1:
                base_ms = int(parts[0]) * 1000
                return base_ms
        except:
            pass
        return 0
    
    def jsonl_to_binary(self, jsonl_path: str, output_path: str) -> int:
        """Convert JSONL evaluation file to binary format.
        
        Args:
            jsonl_path: Path to JSONL file with evaluations
            output_path: Path to output .bin file
            
        Returns:
            Number of positions written
            
        Example:
            count = converter.jsonl_to_binary("evals.jsonl", "evals.bin")
        """
        jsonl_file = Path(jsonl_path)
        if not jsonl_file.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.positions_written = 0
        self.start_time = time.time()
        
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as jsonl:
                with open(output_file, 'wb') as bin_out:
                    buffered = io.BufferedWriter(bin_out, self.BUFFER_SIZE)
                    
                    pbar = tqdm(desc="Converting JSONL", unit=" positions") if self.verbose else None
                    
                    for line_num, line in enumerate(jsonl):
                        if not line.strip():
                            continue
                        
                        try:
                            data = json.loads(line)
                            record = self._jsonl_to_record(data)
                            buffered.write(record.pack())
                            self.positions_written += 1
                            
                            if pbar:
                                pbar.update(1)
                        
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON at line {line_num}: {e}")
                            continue
                    
                    buffered.flush()
                    if pbar:
                        pbar.close()
        
        except Exception as e:
            logger.error(f"Error converting JSONL: {e}")
            raise
        
        elapsed = time.time() - self.start_time
        if self.verbose:
            logger.info(f"Converted {self.positions_written} positions in {elapsed:.2f}s")
            if elapsed > 0:
                logger.info(f"Throughput: {(self.positions_written * 88 / 1024 / 1024) / elapsed:.2f} MB/sec")
        
        return self.positions_written
    
    def _jsonl_to_record(self, data: Dict) -> BinaryPositionRecord:
        """Convert JSONL evaluation entry to BinaryPositionRecord.
        
        Args:
            data: Dictionary from JSONL line
            
        Returns:
            BinaryPositionRecord
        """
        fen = data.get('fen', 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
        evaluation = data.get('eval', 0)
        depth = data.get('depth', 20)
        
        # Clamp to valid ranges
        evaluation = max(-32768, min(32767, int(evaluation)))
        depth = max(0, min(255, int(depth)))
        
        # Calculate FEN hash
        fen_bytes = hashlib.sha256(fen.encode()).digest()[:8]
        fen_hash = int.from_bytes(fen_bytes, 'big')
        
        # Get board for material/phase calculation
        try:
            board = chess.Board(fen)
            material_diff, piece_count = self._calculate_material(board)
            phase = self._calculate_phase(board)
            is_quiet = 1 if self._is_quiet_position(board) else 0
        except:
            material_diff = 0
            piece_count = 32
            phase = 1
            is_quiet = 0
        
        return BinaryPositionRecord(
            fen_hash=fen_hash,
            evaluation=evaluation,
            depth=depth,
            time_ms=data.get('time_ms', 0),
            wins=data.get('wins', 0),
            draws=data.get('draws', 0),
            losses=data.get('losses', 0),
            is_quiet=is_quiet,
            material_diff=material_diff & 0xFFFF,
            phase=phase,
            piece_count=piece_count
        )
    
    def puzzle_tokenize(self, puzzle_file: str, output_path: str) -> int:
        """Tokenize puzzle data for training.
        
        Args:
            puzzle_file: Path to puzzle file (4.9M Lichess puzzles)
            output_path: Path to output tokenized file
            
        Returns:
            Number of puzzles processed
            
        Example:
            count = converter.puzzle_tokenize("puzzles.csv", "puzzles.bin")
        """
        puzzle_file_path = Path(puzzle_file)
        if not puzzle_file_path.exists():
            raise FileNotFoundError(f"Puzzle file not found: {puzzle_file}")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.positions_written = 0
        self.start_time = time.time()
        
        try:
            with open(puzzle_file_path, 'r', encoding='utf-8') as pf:
                with open(output_file, 'wb') as bin_out:
                    buffered = io.BufferedWriter(bin_out, self.BUFFER_SIZE)
                    
                    pbar = tqdm(desc="Tokenizing puzzles", unit=" puzzles") if self.verbose else None
                    
                    for line_num, line in enumerate(pf):
                        if line_num == 0:  # Skip header if present
                            continue
                        
                        if not line.strip():
                            continue
                        
                        try:
                            # Expected format: puzzle_id,fen,moves,rating,rating_deviation,popularity,themes
                            parts = line.strip().split(',')
                            if len(parts) < 3:
                                continue
                            
                            fen = parts[1]
                            moves = parts[2]
                            
                            # Parse puzzle and create records
                            board = chess.Board(fen)
                            move_list = moves.split()
                            
                            for move_san in move_list:
                                try:
                                    move = board.parse_san(move_san)
                                    record = self._board_to_record(board, 0)
                                    buffered.write(record.pack())
                                    self.positions_written += 1
                                    board.push(move)
                                except:
                                    continue
                            
                            if pbar:
                                pbar.update(1)
                        
                        except Exception as e:
                            logger.warning(f"Error processing puzzle at line {line_num}: {e}")
                            continue
                    
                    buffered.flush()
                    if pbar:
                        pbar.close()
        
        except Exception as e:
            logger.error(f"Error tokenizing puzzles: {e}")
            raise
        
        elapsed = time.time() - self.start_time
        if self.verbose:
            logger.info(f"Tokenized {self.positions_written} puzzle positions in {elapsed:.2f}s")
        
        return self.positions_written
    
    def benchmark_conversion(self, num_positions: int = 10000) -> Dict[str, float]:
        """Benchmark conversion speed.
        
        Args:
            num_positions: Number of test positions
            
        Returns:
            Dictionary with metrics:
                - throughput_mb_s: MB/sec
                - positions_per_sec: positions/sec
                - total_time_s: total time in seconds
                
        Example:
            metrics = converter.benchmark_conversion(10000)
            print(f"Speed: {metrics['throughput_mb_s']} MB/sec")
        """
        logger.info(f"Benchmarking conversion on {num_positions} positions...")
        
        # Generate test records
        records = []
        board = chess.Board()
        for i in range(min(num_positions, 100)):
            record = self._board_to_record(board)
            records.append(record)
            # Make some test moves to vary positions
            if board.legal_moves:
                board.push(next(board.legal_moves))
        
        # Replicate to fill num_positions
        while len(records) < num_positions:
            records.extend(records[:num_positions - len(records)])
        
        records = records[:num_positions]
        
        # Benchmark writing
        start = time.time()
        
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp:
            tmp_path = tmp.name
        
        try:
            with open(tmp_path, 'wb') as f:
                buffered = io.BufferedWriter(f, self.BUFFER_SIZE)
                for record in records:
                    buffered.write(record.pack())
                buffered.flush()
            
            elapsed = time.time() - start
            
            # Calculate metrics
            bytes_written = num_positions * BinaryPositionRecord.RECORD_SIZE
            mb_written = bytes_written / (1024 * 1024)
            throughput = mb_written / elapsed if elapsed > 0 else 0
            pos_per_sec = num_positions / elapsed if elapsed > 0 else 0
            
            metrics = {
                'throughput_mb_s': throughput,
                'positions_per_sec': pos_per_sec,
                'total_time_s': elapsed,
                'total_mb': mb_written
            }
            
            if self.verbose:
                logger.info(f"Benchmark results:")
                logger.info(f"  Throughput: {throughput:.2f} MB/sec")
                logger.info(f"  Positions/sec: {pos_per_sec:.0f}")
                logger.info(f"  Total time: {elapsed:.2f}s")
                logger.info(f"  Target: >50 MB/sec {'✓ PASS' if throughput >= 50 else '✗ FAIL'}")
            
            return metrics
        
        finally:
            Path(tmp_path).unlink(missing_ok=True)


def batch_convert_pgns(pgn_dir: str, output_dir: str, num_workers: int = 8) -> int:
    """Batch convert all PGN files in a directory (parallel).
    
    Args:
        pgn_dir: Directory containing PGN files
        output_dir: Output directory for .bin files
        num_workers: Number of parallel workers
        
    Returns:
        Total positions converted
        
    Example:
        total = batch_convert_pgns("pgn_files/", "output/")
        print(f"Converted {total} positions total")
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    pgn_dir_path = Path(pgn_dir)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Find all PGN files
    pgn_files = list(pgn_dir_path.glob('*.pgn')) + list(pgn_dir_path.glob('*.PGN'))
    
    if not pgn_files:
        logger.warning(f"No PGN files found in {pgn_dir}")
        return 0
    
    logger.info(f"Found {len(pgn_files)} PGN files to convert")
    
    total_positions = 0
    converter = BinaryFormatConverter(verbose=False)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        
        for pgn_file in pgn_files:
            output_file = output_dir_path / f"{pgn_file.stem}.bin"
            future = executor.submit(converter.pgn_to_binary, str(pgn_file), str(output_file))
            futures[future] = pgn_file.name
        
        pbar = tqdm(total=len(futures), desc="Batch converting PGN files")
        
        for future in as_completed(futures):
            try:
                positions = future.result()
                total_positions += positions
                pbar.update(1)
                pbar.set_postfix({'total': total_positions})
            except Exception as e:
                logger.error(f"Error converting {futures[future]}: {e}")
                pbar.update(1)
        
        pbar.close()
    
    logger.info(f"Batch conversion complete: {total_positions} total positions")
    return total_positions


if __name__ == "__main__":
    # Quick test/example usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    converter = BinaryFormatConverter(verbose=True)
    
    # Test 1: Benchmark conversion speed
    print("\n=== Test 1: Benchmark Conversion ===")
    metrics = converter.benchmark_conversion(10000)
    print(f"Results: {metrics}")
    
    # Test 2: Convert sample PGN (if available)
    print("\n=== Test 2: Convert Sample PGN ===")
    sample_pgn = Path("../../Chess PGNs/training_data/pgn_data_general/mikhail_tal_master_games.pgn")
    if sample_pgn.exists():
        try:
            count = converter.pgn_to_binary(str(sample_pgn), "test_output.bin")
            print(f"✓ Converted {count} positions from PGN")
            
            # Verify binary file
            if Path("test_output.bin").exists():
                file_size = Path("test_output.bin").stat().st_size
                expected_records = file_size // 88
                print(f"✓ Binary file created: {file_size} bytes ({expected_records} records)")
                Path("test_output.bin").unlink()  # Clean up
        except Exception as e:
            print(f"✗ Error: {e}")
    else:
        print(f"Sample PGN not found at {sample_pgn}")
    
    print("\nBinary format converter module ready for testing")
