"""
V7P3R AI Puzzle-Based Training Pipeline

Revolutionary training approach using tactical puzzles instead of slow self-play.
This provides much faster, more targeted learning with immediate feedback.

Usage:
    python puzzle_main_trainer.py --puzzles 1000 --themes pin,fork,mate --rating-min 1200 --rating-max 1800
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from ai.thinking_brain import ThinkingBrain
from training.puzzle_trainer import PuzzleTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main puzzle training execution"""
    parser = argparse.ArgumentParser(description="V7P3R AI Puzzle-Based Training Pipeline")
    parser.add_argument('--puzzles', type=int, default=500, help='Number of puzzles to train on')
    parser.add_argument('--rating-min', type=int, default=1200, help='Minimum puzzle rating')
    parser.add_argument('--rating-max', type=int, default=1800, help='Maximum puzzle rating')
    parser.add_argument('--themes', type=str, help='Comma-separated list of themes (e.g., pin,fork,mate)')
    parser.add_argument('--checkpoint-interval', type=int, default=100, help='Save model every N puzzles')
    parser.add_argument('--quick-test', action='store_true', help='Run quick test with 20 puzzles')
    parser.add_argument('--model-dir', type=str, default='models/puzzle_training', help='Model save directory')
    
    args = parser.parse_args()
    
    # Parse themes filter
    themes_filter = None
    if args.themes:
        themes_filter = [theme.strip() for theme in args.themes.split(',')]
    
    # Quick test mode
    if args.quick_test:
        args.puzzles = 20
        args.checkpoint_interval = 10
        logger.info("🧪 Quick test mode: 20 puzzles")
    
    print("🧩 V7P3R AI Puzzle-Based Training Pipeline")
    print("=" * 60)
    print(f"Revolutionary training approach using {args.puzzles} tactical puzzles")
    print(f"Rating range: {args.rating_min}-{args.rating_max}")
    print(f"Themes: {themes_filter or 'All tactical themes'}")
    print(f"Model directory: {args.model_dir}")
    print("=" * 60)
    
    try:
        # Initialize ThinkingBrain
        logger.info("Initializing V7P3R ThinkingBrain...")
        thinking_brain = ThinkingBrain()
        
        # Enhanced memory configuration for puzzle training
        memory_config = {
            'max_memory_positions': 20,      # More memory for complex positions
            'memory_decay_factor': 0.92,     # Slower decay for puzzle positions
            'status_trend_window': 7,        # Longer trend analysis
            'pruning_threshold': 0.35,       # Moderate pruning
            'critical_memory_boost': 1.4     # Good boost for critical positions
        }
        
        # Initialize PuzzleTrainer
        logger.info("Initializing PuzzleTrainer...")
        trainer = PuzzleTrainer(
            thinking_brain=thinking_brain,
            memory_config=memory_config,
            save_directory=args.model_dir
        )
        
        logger.info("✅ All systems initialized successfully!")
        
        # Start puzzle-based training
        logger.info("🚀 Starting puzzle-based training...")
        print(f"🎯 Training on {args.puzzles} puzzles with immediate feedback and quality scoring")
        
        start_time = datetime.now()
        
        results = trainer.train(
            num_puzzles=args.puzzles,
            rating_min=args.rating_min,
            rating_max=args.rating_max,
            themes_filter=themes_filter,
            checkpoint_interval=args.checkpoint_interval,
            save_progress=True
        )
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        if results:
            print("\n" + "=" * 80)
            print("🎉 PUZZLE TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            
            # Print comprehensive report
            trainer.print_training_report(results)
            
            print(f"\n⏱️ Total training time: {training_duration}")
            print(f"🚀 Average time per puzzle: {training_duration.total_seconds() / args.puzzles:.1f}s")
            
            # Compare to traditional self-play
            equivalent_games = args.puzzles // 20  # Estimate games equivalent
            print(f"📊 This training is equivalent to ~{equivalent_games} traditional self-play games")
            print(f"⚡ But completed in a fraction of the time with immediate feedback!")
            
            # Show best performing themes
            if results.get('theme_performance'):
                best_themes = sorted(
                    results['theme_performance'].items(), 
                    key=lambda x: x[1]['avg_score'], 
                    reverse=True
                )[:3]
                
                print(f"\n🎯 Best performing themes:")
                for theme, data in best_themes:
                    print(f"   {theme}: {data['avg_score']:.2f}/5.0 ({data['solution_rate']:.1f}% solutions)")
        
        else:
            logger.error("❌ Training failed to produce results")
            return 1
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
