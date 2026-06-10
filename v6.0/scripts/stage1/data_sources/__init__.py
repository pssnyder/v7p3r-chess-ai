"""
Multi-source data loaders for V6.1 training pipeline.

This package provides modular data loaders for different chess data sources:
- LichessDBLoader: Pre-evaluated positions from Lichess database
- OpeningPGNLoader: Opening positions from PGN files
- V7P3RGameLoader: Positions from V7P3R engine battles
- TacticsLoader: Tactical puzzles and training positions
- EndgameLoader: Endgame tablebase positions

The MultiSourceDataLoader orchestrates mixing data from multiple sources
with configurable proportions (default: 70% Lichess, 10% V7P3R, 10% openings, 5% tactics, 5% endgames).
"""

from .base_loader import DataSourceLoader
from .lichess_loader import LichessDBLoader
from .opening_loader import OpeningPGNLoader
from .v7p3r_loader import V7P3RGameLoader
from .tactics_loader import TacticsLoader
from .endgame_loader import EndgameLoader
from .multi_source_loader import MultiSourceDataLoader

__all__ = [
    'DataSourceLoader',
    'LichessDBLoader',
    'OpeningPGNLoader',
    'V7P3RGameLoader',
    'TacticsLoader',
    'EndgameLoader',
    'MultiSourceDataLoader',
]
