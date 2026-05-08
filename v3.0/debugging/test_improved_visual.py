"""
Test the improved visual monitoring system with better graphics and behavior
"""

import sys
import time
import chess
import random
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from monitoring.visual_monitor import ChessBoardVisualizer


def test_improved_visualizer():
    """Test the improved chess board visualizer"""
    print("🎯 Testing Improved Chess Board Visualizer...")
    print("   Features tested:")
    print("   ✓ Real piece images from images/ directory")
    print("   ✓ Darker, more visible heat colors")
    print("   ✓ Faster arrow fade (less clutter)")
    print("   ✓ Last move highlighting")
    print("   ✓ Heat persistence with move-based fading")
    print("   ✓ Arrow reset on each move")
    print()
    print("   Controls: ESC = Exit, R = Reset heatmap")
    print("   Watch how the AI attention evolves!")
    
    visualizer = ChessBoardVisualizer()
    
    # Set up a test position
    board = chess.Board()
    visualizer.update_position(board)
    
    # Start monitoring in background
    import threading
    monitor_thread = threading.Thread(target=visualizer.start_monitoring, daemon=True)
    monitor_thread.start()
    
    # Simulate AI thinking patterns with realistic game flow
    move_number = 0
    
    for i in range(200):  # Longer test to see persistence
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            board = chess.Board()  # Reset if no moves
            legal_moves = list(board.legal_moves)
            visualizer.update_position(board)
            move_number = 0
        
        # Simulate intense AI thinking on some squares
        num_candidates = min(4, len(legal_moves))  # Fewer candidates for less clutter
        candidates = random.sample(legal_moves, num_candidates)
        
        # Create more realistic probability distributions
        probabilities = []
        for j in range(num_candidates):
            if j == 0:
                probabilities.append(random.uniform(0.6, 0.9))  # Strong preference
            elif j == 1:
                probabilities.append(random.uniform(0.3, 0.6))  # Medium
            else:
                probabilities.append(random.uniform(0.1, 0.4))  # Lower
        
        thinking_time = random.uniform(0.3, 2.0)
        
        # Update attention
        visualizer.update_attention(candidates, probabilities, thinking_time)
        
        time.sleep(0.25)  # Slightly longer pause to see effects
        
        # Make a move every 8-12 iterations (more realistic)
        if i % random.randint(8, 12) == 0 and legal_moves:
            move = random.choice(legal_moves)
            board.push(move)
            visualizer.update_position(board)  # This will reset arrows
            move_number += 1
            print(f"   Move {move_number}: {move}")
            
            # Brief pause after making a move
            time.sleep(0.5)
    
    print("   Test completed! Close the window to continue.")
    monitor_thread.join(timeout=2.0)


def main():
    """Run the improved visual test"""
    print("🎮 V7P3R AI v3.0 - Improved Visual Monitor Test")
    print("=" * 50)
    
    try:
        test_improved_visualizer()
        
        print("\n" + "=" * 50)
        print("✅ Improved visual monitoring test completed!")
        print("\nKey improvements:")
        print("• Real piece images loaded from images/ directory")
        print("• Darker, more visible heat colors with better opacity")
        print("• Arrows fade much faster to reduce clutter")
        print("• Last move highlighting in green")
        print("• Heat persists throughout game, fading after 8+ moves")
        print("• Arrow reset after each move")
        print("\nReady for training integration!")
        
    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
