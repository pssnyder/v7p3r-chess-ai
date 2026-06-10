"""
Base abstract class for data source loaders.

All data loaders must inherit from DataSourceLoader and implement load_batch().
This ensures consistent interface for the MultiSourceDataLoader orchestrator.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import random


class DataSourceLoader(ABC):
    """Abstract base class for chess position data loaders."""
    
    def __init__(self, seed: int = 42, shuffle: bool = True):
        """
        Initialize data source loader.
        
        Args:
            seed: Random seed for reproducibility
            shuffle: Whether to shuffle data on load
        """
        self.seed = seed
        self.shuffle = shuffle
        self.random = random.Random(seed)
        self._total_loaded = 0
        
    @abstractmethod
    def load_batch(self, size: int) -> List[Dict[str, Any]]:
        """
        Load a batch of positions from this data source.
        
        Args:
            size: Number of positions to load
            
        Returns:
            List of position dictionaries with format:
            {
                'fen': str,
                'move_uci': str (optional),
                'label': int (0=bad, 1=good),
                'source': str,
                'features': Dict[str, float],
                'eval_cp': int (centipawn evaluation),
                'grade': int (1-5, where 5 is worst blunder)
            }
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset loader to beginning of data source."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get human-readable name of this data source."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about loaded data.
        
        Returns:
            Dictionary with loader statistics
        """
        return {
            'name': self.get_name(),
            'total_loaded': self._total_loaded,
            'seed': self.seed,
            'shuffle': self.shuffle
        }
