"""
Agent Module
Specialized AI agents for different chess tasks
"""

from .v7p3r_themes_agent import V7P3RThemesAgent
from .v7p3r_corrector_agent import V7P3RCorrectorAgent
from .v7p3r_opening_agent import V7P3ROpeningAgent
from .v7p3r_endgame_agent import V7P3REndgameAgent
from .v7p3r_tactics_agent import V7P3RTacticsAgent

__all__ = [
    "V7P3RThemesAgent",
    "V7P3RCorrectorAgent",
    "V7P3ROpeningAgent",
    "V7P3REndgameAgent",
    "V7P3RTacticsAgent",
]
