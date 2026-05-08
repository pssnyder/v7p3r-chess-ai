"""
V7P3R AI v3.0 - Visual Monitoring Test
=====================================

Test script for the visual monitoring system.
Tests both the standalone visualizer and integrated training monitor.
"""

import sys
import time
import chess
import random
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from monitoring.visual_monitor import ChessBoardVisualizer, TrainingMonitor
from monitoring.integration import IntegratedTrainingMonitor


def test_basic_visualizer():
    """Test the basic chess board visualizer"""
    print("🎯 Testing Basic Chess Board Visualizer...")
    
    visualizer = ChessBoardVisualizer()
    
    # Set up a test position
    board = chess.Board()
    visualizer.update_position(board)
    
    # Simulate AI thinking for a few seconds
    print("   Starting visualization - press ESC to exit")
    print("   Watch the board show AI attention patterns!")
    
    # Start monitoring in background
    import threading
    monitor_thread = threading.Thread(target=visualizer.start_monitoring, daemon=True)
    monitor_thread.start()
    
    # Simulate AI thinking patterns
    for i in range(100):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            board = chess.Board()  # Reset if no moves
            legal_moves = list(board.legal_moves)
            visualizer.update_position(board)
        
        # Generate random move candidates with probabilities
        num_candidates = min(5, len(legal_moves))
        candidates = random.sample(legal_moves, num_candidates)
        probabilities = [random.random() for _ in candidates]
        thinking_time = random.uniform(0.1, 1.5)
        
        # Update attention
        visualizer.update_attention(candidates, probabilities, thinking_time)
        
        time.sleep(0.2)  # Pause to see the effect
        
        # Occasionally make a move
        if i % 15 == 0 and legal_moves:
            move = random.choice(legal_moves)
            board.push(move)
            visualizer.update_position(board)
    
    print("   Test completed! Close the window to continue.")
    monitor_thread.join(timeout=1.0)


def test_training_monitor():
    """Test the training monitor interface"""
    print("🎯 Testing Training Monitor Interface...")
    
    monitor = TrainingMonitor(enable_visual=True)
    monitor.start_visual_monitoring()
    
    # Simulate training with different positions
    board = chess.Board()
    monitor.update_position(board)
    
    for i in range(50):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            board = chess.Board()
            legal_moves = list(board.legal_moves)
            monitor.update_position(board)
        
        # Simulate AI attention
        candidates = random.sample(legal_moves, min(3, len(legal_moves)))
        probabilities = [random.random() for _ in candidates]
        thinking_time = random.uniform(0.1, 2.0)
        
        monitor.update_ai_attention(candidates, probabilities, thinking_time)
        
        time.sleep(0.3)
        
        # Make a move every few iterations
        if i % 10 == 0:
            move = random.choice(legal_moves)
            board.push(move)
            monitor.update_position(board)
    
    monitor.stop_monitoring()
    print("   Training monitor test completed!")


def test_integrated_monitor():
    """Test the integrated monitoring system"""
    print("🎯 Testing Integrated Training Monitor...")
    
    monitor = IntegratedTrainingMonitor(
        enable_visual=True,
        save_data=True,
        output_dir=Path("test_monitoring_output")
    )
    
    # Start monitoring
    monitor.start_monitoring("test_session")
    
    # Simulate a training session
    board = chess.Board()
    monitor.update_position(board)
    
    for move_num in range(30):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        
        # Simulate Thinking Brain activity
        thinking_candidates = random.sample(legal_moves, min(4, len(legal_moves)))
        thinking_probs = [random.random() for _ in thinking_candidates]
        thinking_time = random.uniform(0.2, 1.5)
        
        monitor.update_thinking_brain_activity(
            thinking_candidates, thinking_probs, thinking_time
        )
        
        time.sleep(0.1)
        
        # Simulate Gameplay Brain activity
        gameplay_candidates = random.sample(legal_moves, min(6, len(legal_moves)))
        gameplay_fitness = [random.uniform(0.1, 1.0) for _ in gameplay_candidates]
        selection_time = random.uniform(0.05, 0.3)
        
        monitor.update_gameplay_brain_activity(
            gameplay_candidates, gameplay_fitness, selection_time
        )
        
        time.sleep(0.1)
        
        # Make a move
        selected_move = random.choice(legal_moves)
        monitor.record_move_made(selected_move, "test_brain", random.uniform(-0.5, 0.5))
        
        board.push(selected_move)
        monitor.update_position(board)
        
        time.sleep(0.2)
    
    # End game
    monitor.record_game_end("Test result", "Test completed", move_num)
    
    # Get and print stats
    stats = monitor.get_session_stats()
    print(f"   Session stats: {stats}")
    
    # Stop monitoring
    monitor.stop_monitoring()
    print("   Integrated monitor test completed!")


def main():
    """Run all visual monitoring tests"""
    print("🎮 V7P3R AI v3.0 - Visual Monitoring Test Suite")
    print("=" * 50)
    
    try:
        # Test 1: Basic visualizer
        test_basic_visualizer()
        
        print("\n" + "=" * 50)
        
        # Test 2: Training monitor interface
        test_training_monitor()
        
        print("\n" + "=" * 50)
        
        # Test 3: Integrated monitor
        test_integrated_monitor()
        
        print("\n" + "=" * 50)
        print("✅ All visual monitoring tests completed successfully!")
        print("\nThe visual monitoring system is ready for training integration.")
        print("To enable during training, use: python main_trainer.py --visual")
        
    except KeyboardInterrupt:
        print("\n❌ Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
