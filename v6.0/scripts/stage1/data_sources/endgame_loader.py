"""
Endgame Loader - loads endgame positions from PGN files and databases.

Data source: Endgame PGN files (R+B vs K, Q vs R, pawn endgames, etc.)
This provides critical endgame positions for learning conversion techniques.
"""

import chess
import chess.pgn
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from .base_loader import DataSourceLoader
from scripts.utils.calculate_features import FeatureCalculator, FeatureConfig


class EndgameLoader(DataSourceLoader):
    """Load endgame positions from PGN files."""
    
    # Piece count threshold for endgame (total pieces including kings)
    MAX_PIECE_COUNT = 10
    
    def __init__(
        self,
        pgn_dir: str,
        seed: int = 42,
        shuffle: bool = True,
        max_pieces: int = 10,
        feature_config: Optional[FeatureConfig] = None
    ):
        """
        Initialize endgame loader.
        
        Args:
            pgn_dir: Directory containing endgame PGN files
            seed: Random seed for reproducibility
            shuffle: Whether to shuffle positions
            max_pieces: Maximum total pieces for endgame (including kings)
            feature_config: Configuration for feature calculation
        """
        super().__init__(seed=seed, shuffle=shuffle)
        self.pgn_dir = Path(pgn_dir)
        self.max_pieces = max_pieces
        self.feature_calculator = FeatureCalculator(config=feature_config or FeatureConfig())
        
        if not self.pgn_dir.exists():
            raise FileNotFoundError(f"PGN directory not found: {pgn_dir}")
        
        # Discover PGN files
        self._pgn_files = list(self.pgn_dir.glob("*.pgn"))
        if not self._pgn_files:
            raise ValueError(f"No PGN files found in {pgn_dir}")
        
        self._current_file_idx = 0
        self._position_buffer = []
        
    def _count_pieces(self, board: chess.Board) -> int:
        """Count total pieces on board."""
        return len(board.piece_map())
        
    def _is_endgame(self, board: chess.Board) -> bool:
        """Check if position is an endgame."""
        return self._count_pieces(board) <= self.max_pieces
        
    def _extract_endgame_positions(self, game: chess.pgn.Game) -> List[Dict[str, Any]]:
        """
        Extract endgame positions from a game.
        
        Args:
            game: PGN game object
            
        Returns:
            List of position records
        """
        positions = []
        board = game.board()
        
        for node in game.mainline():
            move = node.move
            board.push(move)
            
            # Check if we've entered endgame
            if not self._is_endgame(board):
                continue
            
            try:
                # Calculate features
                features = self.feature_calculator.calculate_features_from_fen(board.fen())
                
                # Endgame positions are critical for learning conversion
                # Label as good if from a won game, otherwise let validation determine
                result = game.headers.get('Result', '*')
                
                if result == '1-0':
                    label = 1 if board.turn == chess.WHITE else 0
                elif result == '0-1':
                    label = 1 if board.turn == chess.BLACK else 0
                else:
                    label = 1  # Default to good, Stockfish will validate
                
                position = {
                    'fen': board.fen(),
                    'move_uci': move.uci(),
                    'label': label,
                    'source': 'endgame',
                    'features': features,
                    'eval_cp': 0,  # Placeholder
                    'grade': 1,  # Placeholder
                    'piece_count': self._count_pieces(board),
                    'result': result
                }
                
                positions.append(position)
                
            except Exception:
                # Skip positions that fail feature calculation
                continue
        
        return positions
        
    def _load_from_file(self, file_path: Path, count: int) -> List[Dict[str, Any]]:
        """
        Load endgame positions from a single PGN file.
        
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
                    
                    game_positions = self._extract_endgame_positions(game)
                    positions.extend(game_positions)
                    
        except Exception as e:
            # Skip files that can't be parsed
            pass
        
        return positions
        
    def load_batch(self, size: int) -> List[Dict[str, Any]]:
        """
        Load a batch of endgame positions.
        
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
            
            # If exhausted all files, reset
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
        return "Endgame Positions"
