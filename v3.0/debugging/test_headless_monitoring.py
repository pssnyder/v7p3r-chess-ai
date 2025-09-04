"""
V7P3R AI v3.0 - Headless Monitoring Test
========================================

Test script for the monitoring system without visual display.
Tests the data collection and integration components.
"""

import sys
import time
import chess
import random
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from monitoring.integration import IntegratedTrainingMonitor, MonitoringDataCollector


def test_data_collector():
    """Test the monitoring data collector"""
    print("🎯 Testing Monitoring Data Collector...")
    
    collector = MonitoringDataCollector(save_data=True)
    
    # Simulate training events
    board = chess.Board()
    collector.record_position_update(board, "test_game_001")
    
    # Simulate thinking updates
    legal_moves = list(board.legal_moves)
    candidates = legal_moves[:5]
    probabilities = [random.random() for _ in candidates]
    
    collector.record_thinking_update(candidates, probabilities, 1.2, "thinking_brain")
    collector.record_thinking_update(candidates, probabilities, 0.8, "gameplay_brain")
    
    # Simulate moves
    selected_move = random.choice(legal_moves)
    collector.record_move_made(selected_move, "gameplay_brain", 0.75)
    
    # End game
    collector.record_game_end("White wins", "Checkmate", 42)
    
    # Get stats
    stats = collector.get_session_stats()
    print(f"   ✅ Collector stats: {stats}")
    
    # Save data
    output_dir = Path("test_monitoring_output")
    collector.save_session_data(output_dir)
    print(f"   ✅ Data saved to {output_dir}")


def test_headless_integrated_monitor():
    """Test the integrated monitor without visual display"""
    print("🎯 Testing Integrated Monitor (Headless)...")
    
    monitor = IntegratedTrainingMonitor(
        enable_visual=False,  # Headless mode
        save_data=True,
        output_dir=Path("test_monitoring_headless")
    )
    
    # Start monitoring
    monitor.start_monitoring("headless_test_session")
    
    # Simulate multiple games
    for game_num in range(3):
        print(f"   Simulating game {game_num + 1}...")
        
        board = chess.Board()
        monitor.update_position(board)
        
        for move_num in range(20):
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            
            # Simulate Thinking Brain activity
            thinking_candidates = random.sample(legal_moves, min(3, len(legal_moves)))
            thinking_probs = [random.random() for _ in thinking_candidates]
            thinking_time = random.uniform(0.1, 1.0)
            
            monitor.update_thinking_brain_activity(
                thinking_candidates, thinking_probs, thinking_time
            )
            
            # Simulate Gameplay Brain activity
            gameplay_candidates = random.sample(legal_moves, min(5, len(legal_moves)))
            gameplay_fitness = [random.uniform(0.2, 1.0) for _ in gameplay_candidates]
            selection_time = random.uniform(0.05, 0.2)
            
            monitor.update_gameplay_brain_activity(
                gameplay_candidates, gameplay_fitness, selection_time
            )
            
            # Make a move
            selected_move = random.choice(legal_moves)
            evaluation = random.uniform(-0.5, 0.5)
            monitor.record_move_made(selected_move, "gameplay_brain", evaluation)
            
            board.push(selected_move)
            monitor.update_position(board)
        
        # End game
        results = ["White wins", "Black wins", "Draw"]
        reasons = ["Checkmate", "Resignation", "Stalemate", "Time"]
        
        monitor.record_game_end(
            random.choice(results), 
            random.choice(reasons), 
            move_num
        )
    
    # Get final stats
    stats = monitor.get_session_stats()
    print(f"   ✅ Session stats: {stats}")
    
    # Stop monitoring
    monitor.stop_monitoring()
    print("   ✅ Headless integrated monitor test completed!")


def test_monitoring_integration():
    """Test monitoring integration with mock AI components"""
    print("🎯 Testing Monitoring Integration...")
    
    # Test creating monitor from config
    from monitoring.integration import create_training_monitor
    
    config = {
        'monitoring': {
            'enable_visual': False,
            'save_data': True,
            'output_dir': 'test_config_monitor'
        }
    }
    
    monitor = create_training_monitor(config)
    print("   ✅ Monitor created from configuration")
    
    # Test basic functionality
    monitor.start_monitoring("config_test")
    
    board = chess.Board()
    monitor.update_position(board)
    
    # Simulate some AI activity
    legal_moves = list(board.legal_moves)
    candidates = legal_moves[:3]
    probabilities = [0.4, 0.3, 0.3]
    
    monitor.update_thinking_brain_activity(candidates, probabilities, 0.5)
    
    fitness_scores = [0.8, 0.6, 0.7]
    monitor.update_gameplay_brain_activity(candidates, fitness_scores, 0.2)
    
    move = candidates[0]
    monitor.record_move_made(move, "test_brain", 0.8)
    
    monitor.record_game_end("Test complete", "Integration test", 1)
    
    stats = monitor.get_session_stats()
    print(f"   ✅ Integration test stats: {stats}")
    
    monitor.stop_monitoring()
    print("   ✅ Monitoring integration test completed!")


def test_performance():
    """Test monitoring system performance"""
    print("🎯 Testing Monitoring Performance...")
    
    monitor = IntegratedTrainingMonitor(
        enable_visual=False,
        save_data=False  # Skip file I/O for pure performance test
    )
    
    monitor.start_monitoring("performance_test")
    
    start_time = time.time()
    num_operations = 1000
    
    board = chess.Board()
    legal_moves = list(board.legal_moves)
    
    for i in range(num_operations):
        # Update position
        monitor.update_position(board)
        
        # Update thinking brain
        candidates = random.sample(legal_moves, min(3, len(legal_moves)))
        probabilities = [random.random() for _ in candidates]
        monitor.update_thinking_brain_activity(candidates, probabilities, 0.1)
        
        # Update gameplay brain
        fitness_scores = [random.random() for _ in candidates]
        monitor.update_gameplay_brain_activity(candidates, fitness_scores, 0.05)
        
        # Record move
        move = random.choice(candidates)
        monitor.record_move_made(move, "test", random.random())
    
    end_time = time.time()
    duration = end_time - start_time
    operations_per_second = num_operations / duration
    
    print(f"   ✅ Performance: {operations_per_second:.1f} operations/second")
    print(f"   ✅ Total time: {duration:.2f} seconds for {num_operations} operations")
    
    stats = monitor.get_session_stats()
    print(f"   ✅ Final stats: {stats}")
    
    monitor.stop_monitoring()


def main():
    """Run all headless monitoring tests"""
    print("🎮 V7P3R AI v3.0 - Headless Monitoring Test Suite")
    print("=" * 55)
    
    try:
        # Test 1: Data collector
        test_data_collector()
        
        print("\n" + "=" * 55)
        
        # Test 2: Headless integrated monitor
        test_headless_integrated_monitor()
        
        print("\n" + "=" * 55)
        
        # Test 3: Monitoring integration
        test_monitoring_integration()
        
        print("\n" + "=" * 55)
        
        # Test 4: Performance test
        test_performance()
        
        print("\n" + "=" * 55)
        print("✅ All headless monitoring tests completed successfully!")
        print("\nThe monitoring system is ready for integration with training!")
        print("Visual monitoring can be enabled with: python main_trainer.py --visual")
        print("For headless training with data collection: python main_trainer.py --no-visual")
        
    except KeyboardInterrupt:
        print("\n❌ Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
