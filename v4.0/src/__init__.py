"""
V7P3RAI v4.0 - Main Package
Multi-Agent Chess AI Enhancement Layer
"""

__version__ = "4.0.0"
__author__ = "V7P3R Development Team"
__description__ = "Multi-Agent Chess AI Enhancement Layer for V7P3R Chess Engine"

from .agents import (
    V7P3RThemesAgent,
    V7P3RCorrectorAgent,
    V7P3ROpeningAgent,
    V7P3REndgameAgent,
    V7P3RTacticsAgent,
)

from .core import (
    AgentOrchestrator,
    AgentMessage,
    ChessStateExtractor,
)

__all__ = [
    "V7P3RThemesAgent",
    "V7P3RCorrectorAgent",
    "V7P3ROpeningAgent",
    "V7P3REndgameAgent",
    "V7P3RTacticsAgent",
    "AgentOrchestrator",
    "AgentMessage",
    "ChessStateExtractor",
]
