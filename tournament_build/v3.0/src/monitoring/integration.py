"""
V7P3R AI v3.0 - Training Monitor Integration
===========================================

Integration layer that connects the visual monitoring system
with the self-play training process. Handles data flow between
the AI components and the visual display.
"""

import time
import threading
from typing import List, Dict, Any, Optional
import chess
import logging
from dataclasses import dataclass
import json
from pathlib import Path

from .visual_monitor import TrainingMonitor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MonitoringEvent:
    """Event data for monitoring updates"""
    timestamp: float
    event_type: str  # 'position_update', 'thinking_update', 'move_made', 'game_end'
    data: Dict[str, Any]


class MonitoringDataCollector:
    """
    Collects and formats data from AI components for visual monitoring
    """
    
    def __init__(self, save_data: bool = True):
        self.save_data = save_data
        self.events = []
        self.session_start = time.time()
        
        # Data aggregation
        self.total_positions = 0
        self.total_thinking_updates = 0
        self.total_moves_made = 0
        
    def record_position_update(self, board: chess.Board, game_id: str = ""):
        """Record a position update event"""
        event = MonitoringEvent(
            timestamp=time.time(),
            event_type='position_update',
            data={
                'fen': board.fen(),
                'game_id': game_id,
                'move_number': board.fullmove_number,
                'turn': 'white' if board.turn else 'black'
            }
        )
        
        if self.save_data:
            self.events.append(event)
        
        self.total_positions += 1
    
    def record_thinking_update(
        self, 
        move_candidates: List[chess.Move], 
        probabilities: List[float],
        thinking_time: float,
        ai_type: str = "unknown"
    ):
        """Record AI thinking process update"""
        event = MonitoringEvent(
            timestamp=time.time(),
            event_type='thinking_update',
            data={
                'candidates': [str(move) for move in move_candidates],
                'probabilities': probabilities,
                'thinking_time': thinking_time,
                'ai_type': ai_type,
                'num_candidates': len(move_candidates)
            }
        )
        
        if self.save_data:
            self.events.append(event)
        
        self.total_thinking_updates += 1
    
    def record_move_made(self, move: chess.Move, move_source: str, evaluation: float = 0.0):
        """Record when a move is actually made"""
        event = MonitoringEvent(
            timestamp=time.time(),
            event_type='move_made',
            data={
                'move': str(move),
                'source': move_source,  # 'thinking_brain', 'gameplay_brain', etc.
                'evaluation': evaluation,
                'from_square': chess.square_name(move.from_square),
                'to_square': chess.square_name(move.to_square)
            }
        )
        
        if self.save_data:
            self.events.append(event)
        
        self.total_moves_made += 1
    
    def record_game_end(self, result: str, reason: str, total_moves: int):
        """Record game completion"""
        event = MonitoringEvent(
            timestamp=time.time(),
            event_type='game_end',
            data={
                'result': result,
                'reason': reason,
                'total_moves': total_moves,
                'game_duration': time.time() - self.session_start
            }
        )
        
        if self.save_data:
            self.events.append(event)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics for the current monitoring session"""
        session_duration = time.time() - self.session_start
        
        return {
            'session_duration': session_duration,
            'total_events': len(self.events),
            'total_positions': self.total_positions,
            'total_thinking_updates': self.total_thinking_updates,
            'total_moves_made': self.total_moves_made,
            'events_per_second': len(self.events) / max(session_duration, 0.1),
            'start_time': self.session_start
        }
    
    def save_session_data(self, output_dir: Path):
        """Save collected monitoring data to disk"""
        if not self.save_data or not self.events:
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"monitoring_session_{timestamp}.json"
        filepath = output_dir / filename
        
        # Convert events to serializable format
        events_data = []
        for event in self.events:
            events_data.append({
                'timestamp': event.timestamp,
                'event_type': event.event_type,
                'data': event.data
            })
        
        # Create session summary
        session_data = {
            'session_info': self.get_session_stats(),
            'events': events_data
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        logger.info(f"Monitoring session data saved to {filepath}")
        logger.info(f"Session stats: {self.get_session_stats()}")


class IntegratedTrainingMonitor:
    """
    Integrated monitoring system that combines visual display with data collection
    """
    
    def __init__(
        self, 
        enable_visual: bool = True, 
        save_data: bool = True,
        output_dir: Optional[Path] = None
    ):
        self.enable_visual = enable_visual
        self.save_data = save_data
        self.output_dir = output_dir or Path("monitoring_data")
        
        # Initialize components
        self.visual_monitor = TrainingMonitor(enable_visual=enable_visual) if enable_visual else None
        self.data_collector = MonitoringDataCollector(save_data=save_data)
        
        # State tracking
        self.current_game_id = ""
        self.is_active = False
        self.current_player_turn = None  # Track whose turn it is for visual updates
        self.visual_update_counter = 0   # Alternate visual updates
        
        logger.info(f"Integrated Training Monitor initialized (visual={enable_visual}, data={save_data})")
    
    def start_monitoring(self, game_id: str = ""):
        """Start the integrated monitoring system"""
        self.current_game_id = game_id
        self.is_active = True
        
        if self.visual_monitor:
            self.visual_monitor.start_visual_monitoring()
            logger.info("Visual monitoring started")
        
        logger.info(f"Integrated monitoring started for game: {game_id}")
    
    def update_position(self, board: chess.Board):
        """Update the current chess position across all monitors"""
        if not self.is_active:
            return
        
        # Update visual monitor
        if self.visual_monitor:
            self.visual_monitor.update_position(board)
        
        # Record data
        self.data_collector.record_position_update(board, self.current_game_id)
    
    def update_thinking_brain_activity(
        self, 
        move_candidates: List[chess.Move], 
        probabilities: List[float],
        thinking_time: float = 0.0,
        current_player: Optional[str] = None
    ):
        """Update monitoring with Thinking Brain activity"""
        if not self.is_active:
            return
        
        # Debug logging for training probabilities
        if move_candidates:
            max_prob = max(probabilities) if probabilities else 0
            logger.debug(f"Thinking brain ({current_player}): {len(move_candidates)} candidates, max_prob: {max_prob:.4f}, probs: {probabilities[:3]}...")
        
        # Update visual monitor (only for current player to avoid conflicts)
        if self.visual_monitor and self.should_show_visuals_for_player(current_player):
            # Scale probabilities for better visibility during training
            scaled_probs = self._scale_probabilities_for_training(probabilities)
            self.visual_monitor.update_ai_attention(move_candidates, scaled_probs, thinking_time)
        
        # Record data
        self.data_collector.record_thinking_update(
            move_candidates, probabilities, thinking_time, "thinking_brain"
        )
    
    def update_gameplay_brain_activity(
        self, 
        move_candidates: List[chess.Move], 
        fitness_scores: List[float],
        selection_time: float = 0.0
    ):
        """Update monitoring with Gameplay Brain activity"""
        if not self.is_active:
            return
        
        # Convert fitness scores to probabilities for visualization
        if fitness_scores:
            max_fitness = max(fitness_scores)
            min_fitness = min(fitness_scores)
            if max_fitness > min_fitness:
                # Normalize to 0-1 range
                probabilities = [(score - min_fitness) / (max_fitness - min_fitness) 
                               for score in fitness_scores]
            else:
                probabilities = [1.0 / len(fitness_scores)] * len(fitness_scores)
        else:
            probabilities = []
        
        # Update visual monitor
        if self.visual_monitor:
            self.visual_monitor.update_ai_attention(move_candidates, probabilities, selection_time)
        
        # Record data
        self.data_collector.record_thinking_update(
            move_candidates, probabilities, selection_time, "gameplay_brain"
        )
    
    def record_move_made(self, move: chess.Move, move_source: str, evaluation: float = 0.0):
        """Record when a move is actually made"""
        if not self.is_active:
            return
        
        self.data_collector.record_move_made(move, move_source, evaluation)
    
    def record_game_end(self, result: str, reason: str, total_moves: int):
        """Record game completion"""
        if not self.is_active:
            return
        
        self.data_collector.record_game_end(result, reason, total_moves)
        logger.info(f"Game ended: {result} ({reason}) after {total_moves} moves")
    
    def should_show_visuals_for_player(self, current_player: Optional[str]) -> bool:
        """
        Determine if visuals should be shown for the current player
        This prevents both players from updating visuals simultaneously
        """
        if not current_player:
            return True  # If no player specified, show visuals
        
        # Only show visuals for white player, or alternate between players
        # You can modify this logic as needed
        return current_player == "white"
    
    def _scale_probabilities_for_training(self, probabilities: List[float]) -> List[float]:
        """
        Scale probabilities to make them more visible during training
        Training probabilities are often much lower than test values
        """
        if not probabilities:
            return probabilities
        
        # Find the maximum probability
        max_prob = max(probabilities)
        
        # If all probabilities are very low, scale them up
        if max_prob < 0.1:
            # Scale up by 10x but cap at 1.0
            scaled = [min(p * 10.0, 1.0) for p in probabilities]
        elif max_prob < 0.5:
            # Scale up by 2x but cap at 1.0
            scaled = [min(p * 2.0, 1.0) for p in probabilities]
        else:
            # Use original probabilities
            scaled = probabilities
        
        return scaled

    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        return self.data_collector.get_session_stats()
    
    def stop_monitoring(self):
        """Stop all monitoring and save data"""
        if not self.is_active:
            return
        
        self.is_active = False
        
        # Stop visual monitoring
        if self.visual_monitor:
            self.visual_monitor.stop_monitoring()
            logger.info("Visual monitoring stopped")
        
        # Save collected data
        if self.save_data:
            self.data_collector.save_session_data(self.output_dir)
        
        # Print session summary
        stats = self.get_session_stats()
        logger.info(f"Monitoring session completed. Stats: {stats}")


def create_training_monitor(config: Dict[str, Any]) -> IntegratedTrainingMonitor:
    """
    Factory function to create a training monitor from configuration
    
    Args:
        config: Configuration dictionary with monitoring settings
    
    Returns:
        Configured IntegratedTrainingMonitor instance
    """
    # Extract monitoring configuration
    monitoring_config = config.get('monitoring', {})
    
    enable_visual = monitoring_config.get('enable_visual', True)
    save_data = monitoring_config.get('save_data', True)
    output_dir = Path(monitoring_config.get('output_dir', 'monitoring_data'))
    
    monitor = IntegratedTrainingMonitor(
        enable_visual=enable_visual,
        save_data=save_data,
        output_dir=output_dir
    )
    
    logger.info("Training monitor created from configuration")
    return monitor


def test_integrated_monitor():
    """Test the integrated monitoring system"""
    logger.info("Testing Integrated Training Monitor...")
    
    # Create monitor
    monitor = IntegratedTrainingMonitor(
        enable_visual=True,
        save_data=True,
        output_dir=Path("test_monitoring")
    )
    
    # Start monitoring
    monitor.start_monitoring("test_game_001")
    
    # Simulate a game
    board = chess.Board()
    monitor.update_position(board)
    
    import random
    
    for move_num in range(20):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        
        # Simulate Thinking Brain activity
        thinking_candidates = random.sample(legal_moves, min(3, len(legal_moves)))
        thinking_probs = [random.random() for _ in thinking_candidates]
        thinking_time = random.uniform(0.1, 2.0)
        
        monitor.update_thinking_brain_activity(
            thinking_candidates, thinking_probs, thinking_time
        )
        
        time.sleep(0.1)  # Brief pause to see thinking
        
        # Simulate Gameplay Brain activity
        gameplay_candidates = random.sample(legal_moves, min(5, len(legal_moves)))
        gameplay_fitness = [random.uniform(0.1, 1.0) for _ in gameplay_candidates]
        selection_time = random.uniform(0.05, 0.5)
        
        monitor.update_gameplay_brain_activity(
            gameplay_candidates, gameplay_fitness, selection_time
        )
        
        time.sleep(0.1)  # Brief pause to see selection
        
        # Make a move
        selected_move = random.choice(legal_moves)
        monitor.record_move_made(selected_move, "gameplay_brain", random.uniform(-1.0, 1.0))
        
        board.push(selected_move)
        monitor.update_position(board)
        
        time.sleep(0.2)  # Pause between moves
    
    # End game
    monitor.record_game_end("1-0", "Test completed", board.fullmove_number)
    
    # Show stats
    stats = monitor.get_session_stats()
    logger.info(f"Test session stats: {stats}")
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    logger.info("Integrated monitoring test completed!")


if __name__ == "__main__":
    test_integrated_monitor()
