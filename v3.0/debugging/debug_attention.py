"""
Debug script to test visual monitoring attention updates
"""

import sys
import time
import chess
import random
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from monitoring.visual_monitor import ChessBoardVisualizer
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)


def test_attention_debug():
    """Test attention updates with detailed debugging"""
    print("🔍 Debug Test: Visual Attention Updates")
    
    visualizer = ChessBoardVisualizer()
    
    # Set up a test position
    board = chess.Board()
    visualizer.update_position(board)
    
    # Start monitoring in background
    import threading
    monitor_thread = threading.Thread(target=visualizer.start_monitoring, daemon=True)
    monitor_thread.start()
    
    # Test with known values
    legal_moves = list(board.legal_moves)
    test_moves = legal_moves[:3]  # Just first 3 moves
    test_probs = [0.8, 0.6, 0.4]  # Clear probability values
    
    print(f"Testing with moves: {[str(m) for m in test_moves]}")
    print(f"Testing with probabilities: {test_probs}")
    
    # Update attention multiple times
    for i in range(10):
        print(f"\nUpdate {i+1}:")
        visualizer.update_attention(test_moves, test_probs, 1.0)
        
        # Check square attention levels
        for move, prob in zip(test_moves, test_probs):
            from_heat = visualizer.square_attention[move.from_square].heat_level
            to_heat = visualizer.square_attention[move.to_square].heat_level
            print(f"  {move}: from_square heat={from_heat:.3f}, to_square heat={to_heat:.3f}")
        
        # Check move visualizations
        print(f"  Active arrows: {len(visualizer.move_visualizations)}")
        for move_str, viz in visualizer.move_visualizations.items():
            print(f"    {move_str}: intensity={viz.intensity:.3f}")
        
        time.sleep(1)
    
    print("\nTest completed - check the visual display!")
    time.sleep(10)  # Let user see results
    visualizer.stop_monitoring()


if __name__ == "__main__":
    test_attention_debug()
