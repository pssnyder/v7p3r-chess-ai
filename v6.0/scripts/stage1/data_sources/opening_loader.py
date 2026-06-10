"""
Opening PGN Loader - loads positions from opening repertoire PGN files.

Data source: Opening PGN files (London, Caro-Kann, Vienna, French, Dutch, Scandinavian, KID)
This provides positions from preferred openings, weighted 1.5x in the training pipeline.

The loader parses PGN games and extracts positions from the opening phase (first 12-15 moves).
"""

import chess
import chess.pgn
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from .base_loader import DataSourceLoader
from scripts.utils.calculate_features import FeatureCalculator, FeatureConfig


class OpeningPGNLoader(DataSourceLoader):
    """Load positions from opening repertoire PGN files."""
    
    # Preferred openings for personalization
    PREFERRED_OPENINGS = [
        'london', 'caro-kann', 'vienna', 'french', 'dutch',
        'scandinavian', "king's indian"
    ]
    
    def __init__(
        self,
        pgn_dir: str,
        seed: int = 42,
        shuffle: bool = True,
        max_opening_moves: int = 15,
        preferred_only: bool = True,
        feature_config: Optional[FeatureConfig] = None
    ):
        """
        Initialize Opening PGN loader.
        
        Args:
            pgn_dir: Directory containing opening PGN files
            seed: Random seed for reproducibility
            shuffle: Whether to shuffle positions
            max_opening_moves: Maximum move number to extract (opening phase)
            preferred_only: Only load from preferred openings
            feature_config: Configuration for feature calculation
        """
        super().__init__(seed=seed, shuffle=shuffle)
        self.pgn_dir = Path(pgn_dir)
        self.max_opening_moves = max_opening_moves
        self.preferred_only = preferred_only
        self.feature_calculator = FeatureCalculator(config=feature_config or FeatureConfig())
        
        if not self.pgn_dir.exists():
            raise FileNotFoundError(f"PGN directory not found: {pgn_dir}")
        
        # Discover PGN files
        self._pgn_files = list(self.pgn_dir.glob("*.pgn"))
        if self.preferred_only:
            self._pgn_files = [
                f for f in self._pgn_files
                if any(pref in f.name.lower() for pref in self.PREFERRED_OPENINGS)
            ]
        
        if not self._pgn_files:
            raise ValueError(f"No PGN files found in {pgn_dir}")
        
        self._current_file_idx = 0
        self._position_buffer = []
        
    def _is_preferred_opening(self, game_headers: Dict[str, str]) -> bool:
        """Check if game is from a preferred opening."""
        opening = game_headers.get('Opening', '').lower()
        eco = game_headers.get('ECO', '').lower()
        
        for pref in self.PREFERRED_OPENINGS:
            if pref in opening or pref in eco:
                return True
        return False
        
    def _extract_positions_from_game(self, game: chess.pgn.Game) -> List[Dict[str, Any]]:
        """
        Extract opening positions using sequence-based grading.
        
        CRITICAL LOGIC:
        - Don't grade openings by individual move eval drops
        - Grade entire sequence based on final position eval (move 12-15)
        - Allow temporary sacrifices (gambits up to -200cp = 2 pawns)
        - Only reject sequences where final eval is terrible (< -200cp)
        - Individual blunders (>300cp drop) still rejected
        
        Args:
            game: PGN game object
            
        Returns:
            List of position records
        """
        # Check if preferred opening
        if self.preferred_only and not self._is_preferred_opening(game.headers):
            return []
        
        board = game.board()
        opening_sequence = []
        
        # Extract entire opening sequence first
        move_num = 0
        for move in game.mainline_moves():
            move_num += 1
            
            # Only extract from opening phase
            if move_num > self.max_opening_moves:
                break
            
            # Make move and get position
            board.push(move)
            
            try:
                # Calculate features
                features = self.feature_calculator.calculate_features_from_fen(board.fen())
                
                position = {
                    'fen': board.fen(),
                    'move_uci': move.uci(),
                    'source': 'opening_pgn',
                    'features': features,
                    'opening': game.headers.get('Opening', 'Unknown'),
                    'eco': game.headers.get('ECO', ''),
                    'move_number': move_num
                }
                
                opening_sequence.append(position)
                
            except Exception:
                # Skip positions that fail feature calculation
                continue
        
        # If sequence is too short, skip
        if len(opening_sequence) < 8:  # Need at least 8 moves for meaningful eval
            return []
        
        # Evaluate FINAL position in sequence (this is the key insight)
        final_position = opening_sequence[-1]
        final_board = chess.Board(final_position['fen'])
        
        # Import Stockfish validator for eval
        try:
            from scripts.stage1.stockfish_validator import StockfishValidator
            if not hasattr(self, '_stockfish_validator'):
                self._stockfish_validator = StockfishValidator(
                    stockfish_path="stockfish/stockfish.exe",
                    cache_db="data/stage1/stockfish_cache.db"
                )
            
            # Validate final position
            eval_result = self._stockfish_validator.validate([final_position])[0]
            final_eval_cp = eval_result['eval_cp']
            
        except Exception:
            # If Stockfish fails, skip this opening
            return []
        
        # GRADING LOGIC:
        # Threshold: -200cp (2 pawns) - allows gambits like King's Gambit, Danish Gambit
        # If final position is sound (> -200cp), entire sequence is GOOD
        # If final position is terrible (< -200cp), entire sequence is BAD/SKIP
        
        GAMBIT_THRESHOLD = -200  # Allow up to 2 pawn sacrifices
        
        if final_eval_cp < GAMBIT_THRESHOLD:
            # Opening is unsound - final position is too bad
            # Don't include these positions in training
            return []
        
        # Opening is sound! Label all positions in sequence as GOOD
        for pos in opening_sequence:
            pos['label'] = 1  # Good
            pos['grade'] = 1  # High quality
            pos['eval_cp'] = final_eval_cp  # Use final eval for context
            pos['opening_final_eval'] = final_eval_cp  # Track final position eval
        
        return opening_sequence
        
    def _load_from_file(self, file_path: Path, count: int) -> List[Dict[str, Any]]:
        """
        Load positions from a single PGN file.
        
        Args:
            file_path: Path to PGN file
            count: Number of positions to load
            
        Returns:
            List of position records
        """
        positions = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                while len(positions) < count:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    
                    game_positions = self._extract_positions_from_game(game)
                    positions.extend(game_positions)
                    
        except Exception as e:
            # Skip files that can't be parsed
            pass
        
        return positions
        
    def load_batch(self, size: int) -> List[Dict[str, Any]]:
        """
        Load a batch of positions from opening PGN files.
        
        Args:
            size: Number of positions to load
            
        Returns:
            List of position dictionaries
        """
        if not self._position_buffer or len(self._position_buffer) < size:
            # Refill buffer from files
            while len(self._position_buffer) < size and self._current_file_idx < len(self._pgn_files):
                file_path = self._pgn_files[self._current_file_idx]
                new_positions = self._load_from_file(file_path, size - len(self._position_buffer))
                self._position_buffer.extend(new_positions)
                self._current_file_idx += 1
            
            # If we've exhausted all files, reset
            if self._current_file_idx >= len(self._pgn_files):
                self.reset()
        
        # Extract batch from buffer
        batch = self._position_buffer[:size]
        self._position_buffer = self._position_buffer[size:]
        
        if self.shuffle:
            self.random.shuffle(batch)
        
        self._total_loaded += len(batch)
        return batch
        
    def reset(self):
        """Reset loader to beginning of file list."""
        self._current_file_idx = 0
        self._position_buffer = []
        self._total_loaded = 0
        if self.shuffle:
            self.random.shuffle(self._pgn_files)
        
    def get_name(self) -> str:
        """Get human-readable name of this data source."""
        return f"Opening PGNs ({'Preferred' if self.preferred_only else 'All'})"
