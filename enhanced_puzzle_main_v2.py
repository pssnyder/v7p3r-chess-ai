"""
Enhanced Puzzle Training V2 Main Entry Point

This script demonstrates the enhanced V2 puzzle-based training system with comprehensive analytics.
Us        # Show training results
        logger.info("=" * 60)
        logger.info("ENHANCED V2 TRAINING COMPLETED")
        logger.info("=" * 60)
        
        if results:
            # Debug: Check what's actually in results
            print(f"DEBUG: Results type: {type(results)}")
            print(f"DEBUG: Results keys: {list(results.keys()) if isinstance(results, dict) else 'Not a dict'}")
            print(f"DEBUG: total_puzzles value: {results.get('total_puzzles', 'KEY NOT FOUND')}")
            
            # The training method returns the final report directly, not wrapped
            total_puzzles = results.get('total_puzzles', 0)
            logger.info(f"✅ Puzzles processed: {total_puzzles}")
            logger.info(f"🎯 Perfect solutions: {results.get('perfect_solutions', 0)}")
            logger.info(f"📊 Average score: {results.get('average_score', 0):.2f}/5.0")
            logger.info(f"🔝 Top-5 hits: {results.get('top5_rate', 0):.1f}%")2 database schema for advanced learning intelligence.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add v3.0/src to Python path
v3_src = Path(__file__).parent / "v3.0" / "src"
sys.path.insert(0, str(v3_src))

from ai.thinking_brain import ThinkingBrain
from training.enhanced_puzzle_trainer_v2 import EnhancedPuzzleTrainerV2


def setup_logging(debug=False):
    """Setup logging configuration"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('enhanced_puzzle_training_v2.log')
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Enhanced V7P3R Chess AI Puzzle Training V2")
    parser.add_argument('--quick-test', action='store_true', help='Run quick test with 20 puzzles')
    parser.add_argument('--num-puzzles', type=int, default=100, help='Number of puzzles to train on')
    parser.add_argument('--target-themes', nargs='+', help='Specific themes to focus on')
    parser.add_argument('--excluded-themes', nargs='+', help='Themes to exclude (e.g., long veryLong)')
    parser.add_argument('--max-rating', type=int, help='Maximum puzzle rating (e.g., 800)')
    parser.add_argument('--min-rating', type=int, help='Minimum puzzle rating (e.g., 600)')
    parser.add_argument('--randomized', action='store_true', help='Randomize puzzle selection within filters')
    parser.add_argument('--difficulty-adaptation', action='store_true', default=True, help='Enable adaptive difficulty')
    parser.add_argument('--intelligent-selection', action='store_true', default=True, help='Use intelligent puzzle selection')
    parser.add_argument('--spaced-repetition', action='store_true', default=True, help='Include spaced repetition')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--model-path', type=str, help='Path to existing model to load')
    parser.add_argument('--stockfish-path', type=str, help='Path to Stockfish engine')
    parser.add_argument('--analytics-only', action='store_true', help='Show analytics without training')
    
    args = parser.parse_args()
    
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    try:
        # Check if V2 database exists
        v2_db_path = "v3.0/data/v7p3rai_puzzle_training_v2.db"
        if not os.path.exists(v2_db_path):
            logger.error(f"V2 database not found: {v2_db_path}")
            logger.info("Please run the migration script first:")
            logger.info("python v3.0/scripts/migrate_database_v2.py v3.0/data/v7p3rai_puzzle_training.db")
            return
        
        # Initialize or load ThinkingBrain
        if args.model_path and os.path.exists(args.model_path):
            logger.info(f"Loading existing model from {args.model_path}")
            brain = ThinkingBrain.load_model(args.model_path)
        else:
            logger.info("Initializing new ThinkingBrain model")
            # Use a more flexible input size that will be adjusted based on actual features
            brain = ThinkingBrain(
                input_size=690,  # Adjust to match actual feature vector size
                hidden_size=512,
                num_layers=3,
                dropout=0.2
            )
        
        # Initialize Enhanced Puzzle Trainer V2
        stockfish_path = args.stockfish_path or "v3.0/stockfish.exe"
        
        trainer = EnhancedPuzzleTrainerV2(
            thinking_brain=brain,
            puzzle_db_path=v2_db_path,
            stockfish_path=stockfish_path,
            save_directory="v3.0/models/enhanced_puzzle_training_v2",
            model_version="v3.0"
        )
        
        # Show analytics if requested
        if args.analytics_only:
            logger.info("=" * 60)
            logger.info("ENHANCED V2 ANALYTICS DASHBOARD")
            logger.info("=" * 60)
            
            analytics = trainer.get_comprehensive_analytics()
            
            # Basic performance
            basic_perf = analytics.get('basic_performance', {})
            logger.info(f"📊 Total puzzles encountered: {basic_perf.get('total_puzzles', 0)}")
            logger.info(f"🎯 Average score: {basic_perf.get('avg_score') or 0:.2f}/5.0")
            logger.info(f"📈 Average learning velocity: {basic_perf.get('avg_learning_velocity') or 0:.3f}")
            logger.info(f"🎯 Average stability: {basic_perf.get('avg_stability') or 0:.2f}")
            
            # Theme mastery
            theme_mastery = analytics.get('theme_mastery', [])
            if theme_mastery:
                logger.info("\n🎨 THEME MASTERY:")
                for theme in theme_mastery[:10]:  # Top 10 themes
                    logger.info(f"   {theme['theme']}: {theme['mastery_level']} "
                              f"(confidence: {theme['confidence_score']:.2f}, "
                              f"avg score: {theme['average_score']:.2f})")
            
            # Recent trends
            recent_trends = analytics.get('recent_trends', [])
            if recent_trends:
                logger.info("\n📈 RECENT PERFORMANCE TRENDS:")
                for trend in recent_trends[:7]:  # Last 7 days
                    logger.info(f"   {trend['date']}: {trend['puzzles_solved']}/{trend['puzzles_attempted']} solved "
                              f"(avg score: {trend.get('average_score') or 0:.2f})")
            
            # Learning efficiency
            efficiency = analytics.get('learning_efficiency', {})
            if efficiency:
                logger.info(f"\n⚡ LEARNING EFFICIENCY:")
                logger.info(f"   Puzzles per hour: {efficiency.get('puzzles_per_hour') or 0:.1f}")
                logger.info(f"   Accuracy improvement rate: {efficiency.get('accuracy_improvement_rate') or 0:.3f}")
                logger.info(f"   Retention rate: {efficiency.get('retention_rate') or 0:.2f}")
            
            logger.info("=" * 60)
            trainer.close()
            return
        
        # Determine training parameters
        if args.quick_test:
            num_puzzles = 20
            logger.info("🚀 Running quick V2 test with 20 puzzles")
        else:
            num_puzzles = args.num_puzzles
            logger.info(f"🧩 Starting enhanced V2 puzzle training with {num_puzzles} puzzles")
        
        # Show training configuration
        logger.info("🔧 TRAINING CONFIGURATION:")
        logger.info(f"   Target themes: {args.target_themes or 'Auto-selected based on weaknesses'}")
        logger.info(f"   Excluded themes: {args.excluded_themes or 'None'}")
        logger.info(f"   Rating range: {args.min_rating or 'No min'} - {args.max_rating or 'No max'}")
        logger.info(f"   Difficulty adaptation: {args.difficulty_adaptation}")
        logger.info(f"   Intelligent selection: {args.intelligent_selection}")
        logger.info(f"   Spaced repetition: {args.spaced_repetition}")
        
        # Run enhanced V2 training
        results = trainer.train_enhanced_v2(
            num_puzzles=num_puzzles,
            target_themes=args.target_themes,
            excluded_themes=args.excluded_themes,
            max_rating=args.max_rating,
            min_rating=args.min_rating,
            difficulty_adaptation=args.difficulty_adaptation,
            intelligent_selection=args.intelligent_selection,
            spaced_repetition=args.spaced_repetition,
            checkpoint_interval=25 if args.quick_test else 50,
            save_progress=True
        )
        
        # Print comprehensive summary
        logger.info("=" * 60)
        logger.info("ENHANCED V2 TRAINING COMPLETED")
        logger.info("=" * 60)
        
        if results:
            # Basic metrics from top-level results structure
            total_puzzles = results.get('total_puzzles', 0)
            logger.info(f"✅ Puzzles processed: {total_puzzles}")
            logger.info(f"🎯 Perfect solutions: {results.get('perfect_solutions', 0)}")
            logger.info(f"📊 Average score: {results.get('average_score', 0):.2f}/5.0")
            logger.info(f"🔝 Top-5 hits: {results.get('top5_rate', 0):.1f}%")
            
            # Enhanced V2 metrics
            enhanced_stats = results.get('enhanced_stats_v2', {})
            logger.info(f"\n🧠 ENHANCED V2 METRICS:")
            logger.info(f"   Stockfish graded moves: {enhanced_stats.get('stockfish_graded_moves', 0)}")
            logger.info(f"   Average Stockfish score: {enhanced_stats.get('average_stockfish_score', 0):.2f}/5.0")
            logger.info(f"   Moves in Stockfish top-5: {enhanced_stats.get('moves_in_stockfish_top_5', 0)}")
            logger.info(f"   Regression recoveries: {enhanced_stats.get('regression_recoveries', 0)}")
            logger.info(f"   Session efficiency: {enhanced_stats.get('session_efficiency', 0):.1f} puzzles/hour")
            
            # Session context
            session_context = results.get('session_context', {})
            logger.info(f"\n📈 SESSION ANALYSIS:")
            logger.info(f"   Average performance: {session_context.get('average_performance', 0):.2f}")
            logger.info(f"   Fatigue estimate: {session_context.get('fatigue_estimate', 0) * 100:.1f}%")
            logger.info(f"   Puzzles completed: {session_context.get('puzzle_number', 0)}")
            
            # V2 analytics summary
            v2_analytics = results.get('v2_analytics', {})
            if v2_analytics:
                basic_perf = v2_analytics.get('basic_performance', {})
                logger.info(f"\n🗄️  DATABASE ANALYTICS:")
                logger.info(f"   Total puzzles in database: {basic_perf.get('total_puzzles', 0)}")
                logger.info(f"   Overall average score: {basic_perf.get('avg_score') or 0:.2f}")
                logger.info(f"   Overall learning velocity: {basic_perf.get('avg_learning_velocity') or 0:.3f}")
        
        logger.info("=" * 60)
        logger.info("Enhanced V2 puzzle training session completed successfully! 🎉")
        logger.info("Use --analytics-only to view comprehensive analytics dashboard")
        
        # Cleanup
        trainer.close()
        
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        if 'trainer' in locals():
            trainer.close()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        if 'trainer' in locals():
            trainer.close()


if __name__ == "__main__":
    main()
