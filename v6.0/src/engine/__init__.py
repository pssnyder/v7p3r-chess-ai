"""
V7P3R AI v6.1 - Engine Package
Static checkmate search and draw detection modules
"""

from .static_checkmate import StaticCheckmateDetector
from .static_draw_detection import StaticDrawDetector

__all__ = ['StaticCheckmateDetector', 'StaticDrawDetector']
