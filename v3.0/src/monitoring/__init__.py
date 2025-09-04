"""
V7P3R AI v3.0 - Monitoring Module
=================================

Real-time visual monitoring and data collection system for AI training.

This module provides:
- Real-time chess board visualization with AI attention heatmap
- Move candidate visualization with intensity-based arrows
- Training data collection and session analytics
- Integration with AI training components

Key Components:
- visual_monitor: Real-time pygame-based chess board display
- integration: Data collection and AI component integration
"""

from .visual_monitor import (
    ChessBoardVisualizer,
    TrainingMonitor,
    SquareAttention,
    MoveVisualization
)

from .integration import (
    IntegratedTrainingMonitor,
    MonitoringDataCollector,
    MonitoringEvent,
    create_training_monitor
)

__all__ = [
    # Visual monitoring
    'ChessBoardVisualizer',
    'TrainingMonitor',
    'SquareAttention',
    'MoveVisualization',
    
    # Integration
    'IntegratedTrainingMonitor', 
    'MonitoringDataCollector',
    'MonitoringEvent',
    'create_training_monitor'
]

# Version info
__version__ = "3.0.0"
__author__ = "V7P3R AI Development Team"
