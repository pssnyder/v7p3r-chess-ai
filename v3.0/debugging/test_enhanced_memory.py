"""
Test the enhanced memory management system
"""

import sys
import time
import chess
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from ai.thinking_brain import ThinkingBrain, PositionMemory
from core.chess_state import ChessStateExtractor
from core.neural_features import NeuralFeatureConverter


def test_enhanced_memory():
    """Test the enhanced memory system with dynamic pruning"""
    print("🧠 Testing Enhanced Memory Management System")
    print("=" * 50)
    
    # Initialize components
    thinking_brain = ThinkingBrain()
    memory_config = {
        'max_memory_positions': 8,      # Small for testing
        'memory_decay_factor': 0.8,     # Aggressive decay for testing
        'status_trend_window': 3,       # Short window
        'pruning_threshold': 0.3,       # Aggressive pruning
        'critical_memory_boost': 1.5    # Noticeable boost
    }
    
    memory = PositionMemory(thinking_brain, memory_config)
    feature_converter = NeuralFeatureConverter()
    
    print(f"✅ Memory initialized with config: {memory_config}")
    
    # Start a test game
    board = chess.Board()
    memory.start_new_game()
    
    print("\n🎮 Simulating game with memory tracking...")
    
    moves_played = []
    for move_num in range(15):  # Play several moves
        if not list(board.legal_moves):
            break
            
        # Extract position features
        state_extractor = ChessStateExtractor()
        current_state = state_extractor.extract_state(board)
        position_features = feature_converter.convert_to_features(
            current_state, device=str(thinking_brain.device)
        )
        
        # Process position through memory
        legal_moves = list(board.legal_moves)
        candidates, probabilities = memory.process_position(
            position_features, legal_moves, top_k=3
        )
        
        # Make a move
        move = candidates[0] if candidates else legal_moves[0]
        board.push(move)
        memory.record_move(move)
        moves_played.append(move)
        
        print(f"Move {move_num + 1}: {move}")
        print(f"  Memory positions: {len(memory.position_history)}")
        print(f"  Memory weights: {len(memory.memory_weights)}")
        if memory.memory_weights:
            print(f"  Avg weight: {sum(memory.memory_weights) / len(memory.memory_weights):.3f}")
        print(f"  Status trend: {memory.game_status_trend[-3:] if len(memory.game_status_trend) >= 3 else memory.game_status_trend}")
        
        time.sleep(0.1)  # Brief pause
    
    # Test game finalization
    print(f"\n🏁 Finalizing memory with different outcomes...")
    
    # Test win scenario
    memory_win = PositionMemory(thinking_brain, memory_config)
    memory_win.position_history = memory.position_history.copy()
    memory_win.memory_weights = memory.memory_weights.copy()
    memory_win.position_evaluations = memory.position_evaluations.copy()
    
    print(f"Before win finalization: avg_weight = {sum(memory_win.memory_weights) / len(memory_win.memory_weights):.3f}")
    memory_win.finalize_game_memory(1.0)  # Win
    print(f"After win finalization: avg_weight = {sum(memory_win.memory_weights) / len(memory_win.memory_weights):.3f}")
    
    # Test loss scenario
    memory_loss = PositionMemory(thinking_brain, memory_config)
    memory_loss.position_history = memory.position_history.copy()
    memory_loss.memory_weights = memory.memory_weights.copy()
    memory_loss.position_evaluations = memory.position_evaluations.copy()
    
    print(f"Before loss finalization: avg_weight = {sum(memory_loss.memory_weights) / len(memory_loss.memory_weights):.3f}")
    memory_loss.finalize_game_memory(0.0)  # Loss
    print(f"After loss finalization: avg_weight = {sum(memory_loss.memory_weights) / len(memory_loss.memory_weights):.3f}")
    
    print("\n✅ Enhanced memory system test completed!")
    print("\nKey features demonstrated:")
    print("• Dynamic memory pruning based on position importance")
    print("• Game status trend tracking")
    print("• Outcome-based memory weight adjustment")
    print("• Memory size management with configurable limits")


if __name__ == "__main__":
    test_enhanced_memory()
