"""
Lichess Database Loader - loads pre-evaluated positions from Lichess DB.

Data source: lichess_db_eval.jsonl (millions of positions with Stockfish evaluations)
Expected format: {"fen": "...", "evals": [{"pvs": [...], "cp": 123, "mate": null, "depth": 20, "knodes": 1234}]}

This is the primary data source (70% of training data) as positions are already
validated by Stockfish and include evaluation scores.
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from .base_loader import DataSourceLoader
from scripts.utils.calculate_features import FeatureCalculator, FeatureConfig


class LichessDBLoader(DataSourceLoader):
    """Load positions from Lichess evaluation database."""
    
    def __init__(
        self,
        db_path: str,
        seed: int = 42,
        shuffle: bool = True,
        min_depth: int = 15,
        feature_config: Optional[FeatureConfig] = None
    ):
        """
        Initialize Lichess DB loader.
        
        Args:
            db_path: Path to lichess_db_eval.jsonl file
            seed: Random seed for reproducibility
            shuffle: Whether to shuffle positions
            min_depth: Minimum Stockfish analysis depth (default 15)
            feature_config: Configuration for feature calculation
        """
        super().__init__(seed=seed, shuffle=shuffle)
        self.db_path = Path(db_path)
        self.min_depth = min_depth
        self.feature_calculator = FeatureCalculator(config=feature_config or FeatureConfig())
        
        # Initialize file handle before any potential errors
        self._file_handle = None
        self._eof = False
        self._position_buffer = []
        
        if not self.db_path.exists():
            raise FileNotFoundError(f"Lichess DB not found: {db_path}")
        
    def _open_file(self):
        """Open the JSONL file for streaming."""
        if self._file_handle is None:
            self._file_handle = open(self.db_path, 'r', encoding='utf-8')
            
    def _close_file(self):
        """Close the JSONL file."""
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
            
    def _read_positions(self, count: int) -> List[Dict[str, Any]]:
        """
        Read positions from file until we have enough valid ones.
        
        Args:
            count: Number of positions to read
            
        Returns:
            List of position records
        """
        self._open_file()
        positions = []
        
        while len(positions) < count:
            if self._eof:
                # Reached end of file, reset for continuous streaming
                self.reset()
                self._open_file()
                
            line = self._file_handle.readline()
            if not line:
                self._eof = True
                continue
                
            try:
                record = json.loads(line.strip())
                
                # Validate record has required fields
                if 'fen' not in record or 'evals' not in record:
                    continue
                
                # Get the best evaluation (highest depth)
                evals = sorted(record['evals'], key=lambda e: e.get('depth', 0), reverse=True)
                if not evals or evals[0].get('depth', 0) < self.min_depth:
                    continue
                
                best_eval = evals[0]
                
                # Extract centipawn evaluation
                if best_eval.get('mate') is not None:
                    # Convert mate score to centipawn equivalent
                    mate_in = best_eval['mate']
                    eval_cp = 10000 if mate_in > 0 else -10000
                else:
                    eval_cp = best_eval.get('cp', 0)
                
                # Calculate features
                try:
                    features = self.feature_calculator.calculate_features_from_fen(record['fen'])
                except Exception as e:
                    # Skip positions that fail feature calculation
                    continue
                
                # Determine label and grade based on eval
                # Good positions: |eval| < 100cp (grade 1)
                # Bad positions: |eval| >= 100cp (grades 2-5 based on magnitude)
                abs_eval = abs(eval_cp)
                if abs_eval < 100:
                    label = 1  # Good position
                    grade = 1
                elif abs_eval < 200:
                    label = 0  # Bad position
                    grade = 2
                elif abs_eval < 400:
                    label = 0
                    grade = 3
                elif abs_eval < 800:
                    label = 0
                    grade = 4
                else:
                    label = 0
                    grade = 5
                
                position = {
                    'fen': record['fen'],
                    'label': label,
                    'source': 'lichess_db',
                    'features': features,
                    'eval_cp': eval_cp,
                    'grade': grade,
                    'depth': best_eval.get('depth', 0),
                    'knodes': best_eval.get('knodes', 0)
                }
                
                positions.append(position)
                
            except json.JSONDecodeError:
                continue
            except Exception as e:
                # Log and skip problematic records
                continue
                
        return positions
        
    def load_batch(self, size: int) -> List[Dict[str, Any]]:
        """
        Load a batch of positions from Lichess DB.
        
        Args:
            size: Number of positions to load
            
        Returns:
            List of position dictionaries
        """
        positions = self._read_positions(size)
        
        if self.shuffle:
            self.random.shuffle(positions)
        
        self._total_loaded += len(positions)
        return positions
        
    def reset(self):
        """Reset loader to beginning of file."""
        self._close_file()
        self._eof = False
        self._total_loaded = 0
        
    def get_name(self) -> str:
        """Get human-readable name of this data source."""
        return "Lichess Database"
    
    def __del__(self):
        """Cleanup file handle on deletion."""
        self._close_file()
