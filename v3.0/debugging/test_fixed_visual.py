"""
Quick test to verify attention and arrow fixes
"""

import sys
import time
import chess
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from monitoring.visual_monitor import ChessBoardVisualizer
import logging

# Set to INFO level to see key messages
logging.basicConfig(level=logging.INFO)


def test_fixed_attention():
    """Test the fixed attention system"""
    print("🔧 Testing Fixed Attention System")
    
    visualizer = ChessBoardVisualizer()
    
    # Set up a test position
    board = chess.Board()
    visualizer.update_position(board)
    
    # Start monitoring in background
    import threading
    monitor_thread = threading.Thread(target=visualizer.start_monitoring, daemon=True)
    monitor_thread.start()
    
    print("   Visual display should show:")
    print("   ✓ Real chess piece images")
    print("   ✓ Arrows for move candidates")
    print("   ✓ Heat map on considered squares")
    print("   ✓ Last move highlighting")
    print("\n   Making moves and updating attention...")
    
    # Test with realistic patterns
    for round_num in range(5):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            board = chess.Board()
            legal_moves = list(board.legal_moves)
            visualizer.update_position(board)
        
        # Select top candidates
        candidates = legal_moves[:4]
        # Use realistic probability distribution
        probabilities = [0.7, 0.5, 0.3, 0.2][:len(candidates)]
        
        print(f"\n   Round {round_num + 1}:")
        print(f"     Considering: {[str(m) for m in candidates]}")
        print(f"     Probabilities: {probabilities}")
        
        # Multiple attention updates (simulating AI thinking)
        for i in range(3):
            visualizer.update_attention(candidates, probabilities, 1.0)
            time.sleep(0.3)
        
        # Make a move
        if candidates:
            selected_move = candidates[0]  # Pick the highest probability move
            board.push(selected_move)
            visualizer.update_position(board)
            print(f"     Made move: {selected_move}")
            
        time.sleep(2)  # Pause to see the effects
    
    print("\n   Test completed! You should see:")
    print("   ✓ Arrows appearing during consideration")
    print("   ✓ Heat building up on squares")
    print("   ✓ Green highlighting of last moves")
    print("   ✓ Arrows resetting after moves")
    print("\n   Press ESC in the visual window to exit")
    
    # Let user observe for a while
    time.sleep(30)
    visualizer.stop_monitoring()


if __name__ == "__main__":
    test_fixed_attention()
