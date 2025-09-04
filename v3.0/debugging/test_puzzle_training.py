"""
Test the puzzle-based training system for V7P3R AI

This test demonstrates the revolutionary new training approach using tactical puzzles
instead of slow self-play games.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from ai.thinking_brain import ThinkingBrain
from training.puzzle_trainer import PuzzleTrainer


def test_puzzle_training():
    """Test the puzzle-based training system"""
    print("🧩 Testing V7P3R AI Puzzle-Based Training System")
    print("=" * 60)
    
    # Initialize components
    print("Initializing ThinkingBrain...")
    thinking_brain = ThinkingBrain()
    
    memory_config = {
        'max_memory_positions': 15,
        'memory_decay_factor': 0.9,
        'status_trend_window': 5,
        'pruning_threshold': 0.4,
        'critical_memory_boost': 1.3
    }
    
    print("Initializing PuzzleTrainer...")
    trainer = PuzzleTrainer(
        thinking_brain=thinking_brain,
        memory_config=memory_config,
        save_directory="models/puzzle_training"
    )
    
    print(f"✅ Puzzle trainer initialized successfully!")
    print(f"📁 Models will be saved to: {trainer.save_directory}")
    
    # Test small training run
    print("\n🎯 Starting small puzzle training test (20 puzzles)...")
    
    results = trainer.train(
        num_puzzles=20,           # Small test run
        rating_min=1200,          # Easy-medium puzzles
        rating_max=1600,          
        themes_filter=['pin', 'fork', 'mate'],  # Focus on basic tactics
        checkpoint_interval=10,   # Save every 10 puzzles
        save_progress=True
    )
    
    if results:
        print("\n📊 Training Results Summary:")
        trainer.print_training_report(results)
        
        print(f"\n✅ Test completed successfully!")
        print(f"📈 Average score: {results['average_score']:.2f}/5.0")
        print(f"🎯 Solution rate: {results['solution_rate']:.1f}%")
        print(f"⭐ Top-5 hit rate: {results['top5_rate']:.1f}%")
        
        # Show improvement over time
        if len(trainer.training_stats['learning_curve']) > 5:
            recent_scores = [point['quality_score'] for point in trainer.training_stats['learning_curve'][-5:]]
            early_scores = [point['quality_score'] for point in trainer.training_stats['learning_curve'][:5]]
            
            recent_avg = sum(recent_scores) / len(recent_scores)
            early_avg = sum(early_scores) / len(early_scores)
            
            print(f"📈 Learning progress: {early_avg:.1f} → {recent_avg:.1f} (+{recent_avg - early_avg:.1f})")
    else:
        print("❌ Training failed to return results")
    
    print("\n" + "=" * 60)
    print("🎉 Puzzle training test completed!")
    

if __name__ == "__main__":
    test_puzzle_training()
