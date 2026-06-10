"""
Tactics Loader - loads tactical puzzle positions from CSV files.

Data source: CSV puzzle files with FEN positions and evaluations
Format: fen,evaluation,difficulty,themes,etc.

This provides tactical positions (pins, forks, skewers, etc.) which are important
for learning pattern recognition in complex positions.
"""

import csv
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from .base_loader import DataSourceLoader
from scripts.utils.calculate_features import FeatureCalculator, FeatureConfig


class TacticsLoader(DataSourceLoader):
    """Load tactical puzzle positions from CSV files."""
    
    def __init__(
        self,
        csv_path: str,
        seed: int = 42,
        shuffle: bool = True,
        min_difficulty: int = 1,
        max_difficulty: int = 5,
        feature_config: Optional[FeatureConfig] = None
    ):
        """
        Initialize tactics loader.
        
        Args:
            csv_path: Path to CSV file or directory containing CSV files
            seed: Random seed for reproducibility
            shuffle: Whether to shuffle positions
            min_difficulty: Minimum puzzle difficulty (1-5)
            max_difficulty: Maximum puzzle difficulty (1-5)
            feature_config: Configuration for feature calculation
        """
        super().__init__(seed=seed, shuffle=shuffle)
        self.csv_path = Path(csv_path)
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.feature_calculator = FeatureCalculator(config=feature_config or FeatureConfig())
        
        # Discover CSV files
        if self.csv_path.is_file():
            self._csv_files = [self.csv_path]
        elif self.csv_path.is_dir():
            self._csv_files = list(self.csv_path.glob("*.csv"))
        else:
            raise FileNotFoundError(f"CSV path not found: {csv_path}")
        
        if not self._csv_files:
            raise ValueError(f"No CSV files found at {csv_path}")
        
        self._position_buffer = []
        self._current_file_idx = 0
        
    def _parse_difficulty(self, row: Dict[str, str]) -> int:
        """Extract difficulty rating from puzzle data."""
        # Try different possible column names
        for col in ['difficulty', 'rating', 'puzzle_rating', 'eval']:
            if col in row:
                try:
                    # Convert rating to 1-5 scale
                    rating = float(row[col])
                    if rating < 1000:
                        return 1
                    elif rating < 1500:
                        return 2
                    elif rating < 2000:
                        return 3
                    elif rating < 2500:
                        return 4
                    else:
                        return 5
                except (ValueError, TypeError):
                    pass
        return 3  # Default medium difficulty
        
    def _parse_themes(self, row: Dict[str, str]) -> List[str]:
        """Extract tactical themes from puzzle data."""
        for col in ['themes', 'tags', 'theme', 'type']:
            if col in row and row[col]:
                # Themes might be space or comma separated
                themes_str = row[col].replace(',', ' ')
                return [t.strip().lower() for t in themes_str.split() if t.strip()]
        return []
        
    def _load_from_csv(self, file_path: Path, count: int) -> List[Dict[str, Any]]:
        """
        Load positions from a CSV file.
        
        Args:
            file_path: Path to CSV file
            count: Number of positions to load
            
        Returns:
            List of position records
        """
        positions = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Try to detect delimiter
                sample = f.read(1024)
                f.seek(0)
                
                delimiter = ',' if ',' in sample else '\t'
                reader = csv.DictReader(f, delimiter=delimiter)
                
                for row in reader:
                    if len(positions) >= count:
                        break
                    
                    # Extract FEN
                    fen = None
                    for col in ['fen', 'FEN', 'position']:
                        if col in row and row[col]:
                            fen = row[col].strip()
                            break
                    
                    if not fen:
                        continue
                    
                    # Get difficulty
                    difficulty = self._parse_difficulty(row)
                    if difficulty < self.min_difficulty or difficulty > self.max_difficulty:
                        continue
                    
                    # Get themes
                    themes = self._parse_themes(row)
                    
                    try:
                        # Calculate features
                        features = self.feature_calculator.calculate_features_from_fen(fen)
                        
                        # Tactical puzzles are positions where there's a forcing continuation
                        # We'll label them as "good" positions to find (the tactical shot)
                        # and let Stockfish validation determine actual quality
                        position = {
                            'fen': fen,
                            'label': 1,  # Assume good tactical position
                            'source': 'tactics',
                            'features': features,
                            'eval_cp': 0,  # Placeholder
                            'grade': 1,  # Placeholder
                            'difficulty': difficulty,
                            'themes': themes
                        }
                        
                        positions.append(position)
                        
                    except Exception:
                        # Skip positions that fail feature calculation
                        continue
                        
        except Exception as e:
            # Skip files that can't be parsed
            pass
        
        return positions
        
    def load_batch(self, size: int) -> List[Dict[str, Any]]:
        """
        Load a batch of tactical positions.
        
        Args:
            size: Number of positions to load
            
        Returns:
            List of position dictionaries
        """
        if not self._position_buffer or len(self._position_buffer) < size:
            # Refill buffer from files
            while len(self._position_buffer) < size and self._current_file_idx < len(self._csv_files):
                file_path = self._csv_files[self._current_file_idx]
                new_positions = self._load_from_csv(file_path, size - len(self._position_buffer))
                self._position_buffer.extend(new_positions)
                self._current_file_idx += 1
            
            # If exhausted all files, reset
            if self._current_file_idx >= len(self._csv_files):
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
            self.random.shuffle(self._csv_files)
        
    def get_name(self) -> str:
        """Get human-readable name of this data source."""
        return "Tactical Puzzles"
