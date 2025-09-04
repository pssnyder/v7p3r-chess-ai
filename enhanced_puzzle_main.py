"""
Enhanced Puzzle Training Main Entry Point

This script demonstrates the enhanced puzzle-based training system with database integration.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add v3.0 directory to Python path
v3_root = Path(__file__).parent / "v3.0"
sys.path.insert(0, str(v3_root))

from src.ai.thinking_brain import ThinkingBrain
from src.training.enhanced_puzzle_trainer import EnhancedPuzzleTrainer


def setup_logging(debug=False):
    """Setup logging configuration"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('enhanced_puzzle_training.log')
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Enhanced V7P3R Chess AI Puzzle Training")
    parser.add_argument('--quick-test', action='store_true', help='Run quick test with 10 puzzles')
    parser.add_argument('--setup-db', action='store_true', help='Setup database from original puzzle data')
    parser.add_argument('--num-puzzles', type=int, default=100, help='Number of puzzles to train on')
    parser.add_argument('--dataset', choices=['train', 'test', 'validation'], default='train', help='Dataset to use')
    parser.add_argument('--difficulty', choices=['easy', 'medium', 'hard', 'expert'], help='Difficulty tier')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--model-path', type=str, help='Path to existing model to load')
    parser.add_argument('--stockfish-path', type=str, help='Path to Stockfish engine')
    
    args = parser.parse_args()
    
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize ThinkingBrain
        if args.model_path and os.path.exists(args.model_path):
            logger.info(f"Loading existing model from {args.model_path}")
            brain = ThinkingBrain.load_model(args.model_path)
        else:
            logger.info("Initializing new ThinkingBrain model")
            brain = ThinkingBrain(
                input_size=1024,  # Will be updated based on actual features
                hidden_size=512,
                num_layers=3,
                dropout=0.2
            )
        
        # Initialize Enhanced Puzzle Trainer
        stockfish_path = args.stockfish_path or "v3.0/stockfish.exe"
        enhanced_puzzle_db_path = "v3.0/data/v7p3r_enhanced_puzzles.db"
        
        trainer = EnhancedPuzzleTrainer(
            thinking_brain=brain,
            puzzle_db_path=enhanced_puzzle_db_path,
            stockfish_path=stockfish_path,
            save_directory="models/enhanced_puzzle_training",
            model_version="v3.0"
        )
        
        # Setup database if requested
        if args.setup_db:
            logger.info("Setting up enhanced puzzle database...")
            original_db_path = "v3.0/data/v7p3rai_puzzle_training.db"
            total_puzzles = trainer.setup_database(original_db_path, import_limit=1000 if args.quick_test else None)
            logger.info(f"Database setup complete: {total_puzzles} puzzles available")
        
        # Determine training parameters
        if args.quick_test:
            num_puzzles = 10
            logger.info("🚀 Running quick test with 10 puzzles")
        else:
            num_puzzles = args.num_puzzles
            logger.info(f"🧩 Starting enhanced puzzle training with {num_puzzles} puzzles")
        
        # Run enhanced training
        results = trainer.train_enhanced(
            num_puzzles=num_puzzles,
            dataset=args.dataset,
            difficulty_tier=args.difficulty,
            rating_min=1200,
            rating_max=1800,
            checkpoint_interval=50 if args.quick_test else 100,
            include_regression_training=not args.quick_test,
            save_progress=True
        )
        
        # Print summary
        logger.info("=" * 60)
        logger.info("ENHANCED TRAINING COMPLETED")
        logger.info("=" * 60)
        
        if results:
            logger.info(f"✅ Puzzles processed: {results.get('puzzles_solved', 0)}")
            logger.info(f"🎯 Perfect solutions: {results.get('perfect_solutions', 0)}")
            logger.info(f"📊 Average score: {results.get('average_score', 0):.2f}/5.0")
            logger.info(f"🔝 Top-5 hits: {results.get('top5_accuracy', 0):.1f}%")
            
            # Enhanced metrics
            enhanced_stats = results.get('enhanced_stats', {})
            logger.info(f"🆕 New puzzles solved first try: {enhanced_stats.get('new_puzzles_solved_first_try', 0)}")
            logger.info(f"📈 Improvements on repeats: {enhanced_stats.get('improvement_on_repeat_puzzles', 0)}")
            logger.info(f"⚠️  Regression puzzles encountered: {enhanced_stats.get('regression_puzzles_encountered', 0)}")
            
            # Database analytics
            db_analytics = results.get('database_analytics', {})
            if db_analytics:
                logger.info(f"🗄️  Total puzzle encounters: {db_analytics.get('total_encounters', 0)}")
                logger.info(f"📚 Unique puzzles seen: {db_analytics.get('unique_puzzles_encountered', 0)}")
        
        # Run validation test if not in quick test mode
        if not args.quick_test:
            logger.info("\n🧪 Running validation test...")
            validation_results = trainer.run_validation_test(num_puzzles=min(50, num_puzzles // 4))
            
            if validation_results:
                val_accuracy = validation_results.get('solution_accuracy', 0)
                logger.info(f"✅ Validation accuracy: {val_accuracy:.1f}%")
        
        # Analyze regression patterns
        logger.info("\n🔍 Analyzing regression patterns...")
        regression_analysis = trainer.analyze_regression_patterns()
        
        if regression_analysis.get('total_regression_puzzles', 0) > 0:
            logger.info(f"⚠️  Found {regression_analysis['total_regression_puzzles']} regression puzzles")
            
            recommendations = regression_analysis.get('recommendations', [])
            if recommendations:
                logger.info("💡 Training recommendations:")
                for rec in recommendations:
                    logger.info(f"   • {rec}")
        else:
            logger.info("✅ No significant regression patterns detected")
        
        logger.info("=" * 60)
        logger.info("Enhanced puzzle training session completed successfully! 🎉")
        
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
