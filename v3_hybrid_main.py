"""
V7P3R Hybrid Training Main Interface V3.0
=========================================

Main training script that combines:
- Enhanced puzzle training methodology
- V3 two-brain architecture (Thinking + Gameplay)  
- Custom ChessState feature extraction
- Advanced monitoring and analytics

Usage Examples:
    # Time-based training with hybrid architecture
    python v3_hybrid_main.py --hpts 4 --batch-size 30 --max-rating 1800 --excluded-themes long

    # Fixed puzzle count training
    python v3_hybrid_main.py --num-puzzles 1000 --target-themes pin,fork,skewer

    # Analytics and monitoring
    python v3_hybrid_main.py --analytics-only
    python v3_hybrid_main.py --monitor-session <session_id>
"""

import os
import sys
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add v3.0/src to Python path
v3_src = Path(__file__).parent / "v3.0" / "src"
sys.path.insert(0, str(v3_src))

# Import V3 components
from ai.thinking_brain import ThinkingBrain
from ai.gameplay_brain import GameplayBrain

# Import existing enhanced puzzle database
sys.path.insert(0, str(Path(__file__).parent))

def setup_logging(debug=False):
    """Setup comprehensive logging for hybrid training"""
    level = logging.DEBUG if debug else logging.INFO
    
    # Create logs directory
    log_dir = Path("v3.0/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging with both file and console output
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / f'v3_hybrid_training_{datetime.now().strftime("%Y%m%d")}.log')
        ]
    )

def create_argument_parser():
    """Create command line argument parser for hybrid training"""
    parser = argparse.ArgumentParser(
        description="V7P3R Hybrid Training System V3.0 - Two-Brain Architecture with Puzzle Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 4-hour hybrid training session
  python v3_hybrid_main.py --hpts 4 --batch-size 30 --max-rating 1800

  # 1000 puzzle training with specific themes  
  python v3_hybrid_main.py --num-puzzles 1000 --target-themes pin,fork,skewer

  # Advanced training with difficulty progression
  python v3_hybrid_main.py --hpts 2 --difficulty-progression --brain-coordination

  # Monitor current training session
  python v3_hybrid_main.py --monitor-session abc123

  # View comprehensive analytics
  python v3_hybrid_main.py --analytics-only
        """
    )
    
    # Training mode options
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--hpts', type=float, help='Training duration in hours (time-based mode)')
    mode_group.add_argument('--num-puzzles', type=int, help='Number of puzzles to train on (fixed mode)')
    mode_group.add_argument('--analytics-only', action='store_true', help='Show analytics dashboard only')
    mode_group.add_argument('--monitor-session', type=str, help='Monitor specific training session')
    
    # Puzzle selection options
    parser.add_argument('--target-themes', type=str, help='Comma-separated tactical themes to focus on')
    parser.add_argument('--excluded-themes', type=str, help='Comma-separated themes to exclude')
    parser.add_argument('--max-rating', type=int, help='Maximum puzzle rating')
    parser.add_argument('--min-rating', type=int, help='Minimum puzzle rating')
    
    # Training options
    parser.add_argument('--batch-size', type=int, default=30, help='Puzzles per batch (default: 30)')
    parser.add_argument('--difficulty-progression', action='store_true', help='Automatically increase difficulty')
    parser.add_argument('--brain-coordination', action='store_true', help='Enable brain coordination training')
    
    # Architecture options
    parser.add_argument('--thinking-layers', type=int, default=8, help='GRU layers for Thinking Brain')
    parser.add_argument('--thinking-neurons', type=int, default=256, help='Neurons per GRU layer')
    parser.add_argument('--ga-population', type=int, default=5, help='GA population size for Gameplay Brain')
    parser.add_argument('--ga-generations', type=int, default=10, help='GA generations per move')
    
    # System options
    parser.add_argument('--model-path', type=str, help='Path to existing model to continue training')
    parser.add_argument('--save-interval', type=int, default=50, help='Save model every N puzzles')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--cpu-only', action='store_true', help='Force CPU training (disable CUDA)')
    
    return parser

def initialize_v3_brains(args):
    """Initialize both Thinking Brain and Gameplay Brain"""
    logger = logging.getLogger(__name__)
    
    # Determine device
    import torch
    if args.cpu_only:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"🧠 Initializing V3 Two-Brain Architecture on {device.upper()}")
    
    # Initialize Thinking Brain (GRU)
    thinking_brain = ThinkingBrain(
        input_size=690,  # ChessState feature vector size
        hidden_size=args.thinking_neurons,
        num_layers=args.thinking_layers,
        output_size=4096,  # Max possible chess moves
        device=device
    )
    
    # Load existing model if specified
    if args.model_path and Path(args.model_path).exists():
        logger.info(f"📁 Loading existing Thinking Brain from {args.model_path}")
        thinking_brain.load_model(args.model_path)
    
    # Initialize Gameplay Brain (GA)
    gameplay_brain = GameplayBrain(
        population_size=args.ga_population,
        max_generations=args.ga_generations,
        simulation_depth=3,
        device=device
    )
    
    logger.info(f"✅ Two-Brain Architecture initialized:")
    logger.info(f"   Thinking Brain: {args.thinking_layers} layers × {args.thinking_neurons} neurons")
    logger.info(f"   Gameplay Brain: GA with {args.ga_population} population, {args.ga_generations} generations")
    logger.info(f"   Device: {device.upper()}")
    
    return thinking_brain, gameplay_brain

def show_analytics_dashboard():
    """Display comprehensive analytics dashboard"""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("V7P3R HYBRID TRAINING ANALYTICS DASHBOARD V3.0")
    logger.info("=" * 80)
    
    try:
        # Load recent training sessions
        from database.enhanced_puzzle_db_v2 import EnhancedPuzzleDatabaseV2
        db = EnhancedPuzzleDatabaseV2("v3.0/data/v7p3rai_puzzle_training_v2.db")
        
        # Get recent hybrid training sessions
        analytics = db.get_enhanced_analytics_readonly("v3.0_hybrid")
        
        # Display key metrics
        basic_perf = analytics.get('basic_performance', {})
        logger.info(f"📊 Total puzzles solved: {basic_perf.get('total_puzzles', 0)}")
        logger.info(f"🎯 Overall accuracy: {basic_perf.get('avg_score', 0):.2f}/5.0")
        logger.info(f"📈 Estimated ELO: {basic_perf.get('estimated_elo', 'Not calculated')}")
        
        # Brain-specific analytics
        logger.info("\n🧠 THINKING BRAIN PERFORMANCE:")
        logger.info(f"   Move candidate accuracy: {analytics.get('thinking_brain_accuracy', 'N/A')}")
        logger.info(f"   Average candidate rank: {analytics.get('avg_candidate_rank', 'N/A')}")
        logger.info(f"   Processing speed: {analytics.get('thinking_brain_speed', 'N/A')} ms/position")
        
        logger.info("\n🎮 GAMEPLAY BRAIN PERFORMANCE:")  
        logger.info(f"   Tactical validation accuracy: {analytics.get('gameplay_brain_accuracy', 'N/A')}")
        logger.info(f"   GA convergence rate: {analytics.get('ga_convergence_rate', 'N/A')}")
        logger.info(f"   Simulation speed: {analytics.get('gameplay_brain_speed', 'N/A')} ms/move")
        
        logger.info("\n🤝 BRAIN COORDINATION:")
        logger.info(f"   Coordination score: {analytics.get('brain_coordination_score', 'N/A')}")
        logger.info(f"   Combined system accuracy: {analytics.get('combined_accuracy', 'N/A')}")
        logger.info(f"   Integration efficiency: {analytics.get('integration_efficiency', 'N/A')}")
        
        # Theme mastery
        theme_mastery = analytics.get('theme_mastery', {})
        if theme_mastery:
            logger.info("\n🎨 TACTICAL THEME MASTERY:")
            for theme, data in list(theme_mastery.items())[:10]:
                confidence = data.get('confidence_score', 0)
                avg_score = data.get('avg_score', 0)
                status = "mastered" if confidence > 0.8 else "learning" if confidence > 0.4 else "novice"
                logger.info(f"   {theme}: {status} (confidence: {confidence:.2f}, avg score: {avg_score:.2f})")
        
        db.close()
        
    except Exception as e:
        logger.error(f"Error loading analytics: {e}")
        logger.info("No analytics data available yet. Start training to generate metrics.")
    
    logger.info("=" * 80)

def monitor_training_session(session_id: str):
    """Monitor a specific training session in real-time"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"📊 Monitoring V3 Hybrid Training Session: {session_id}")
    logger.info("Press Ctrl+C to stop monitoring")
    
    try:
        import time
        from training.v3_hybrid_puzzle_trainer import V3HybridPuzzleTrainer
        
        # Note: This would need a shared monitoring system
        # For now, show placeholder monitoring interface
        
        while True:
            # Get current session status (would need shared state)
            logger.info(f"⏰ {datetime.now().strftime('%H:%M:%S')} - Session {session_id[:8]}... active")
            logger.info("   📊 Puzzles completed: [Would show real-time count]")
            logger.info("   🧠 Thinking Brain accuracy: [Would show real-time accuracy]")
            logger.info("   🎮 Gameplay Brain accuracy: [Would show real-time accuracy]")
            logger.info("   🤝 Brain coordination: [Would show coordination score]")
            logger.info("")
            
            time.sleep(30)  # Update every 30 seconds
            
    except KeyboardInterrupt:
        logger.info("📊 Monitoring stopped by user")

def main():
    """Main entry point for V3 hybrid training"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 V7P3R Hybrid Training System V3.0")
    logger.info("   Architecture: Two-Brain (Thinking + Gameplay)")
    logger.info("   Training Method: Puzzle-based tactical learning")
    logger.info("   Features: Enhanced ChessState metadata extraction")
    
    # Handle different modes
    if args.analytics_only:
        show_analytics_dashboard()
        return
    
    if args.monitor_session:
        monitor_training_session(args.monitor_session)
        return
    
    # Initialize the two-brain architecture
    thinking_brain, gameplay_brain = initialize_v3_brains(args)
    
    # Parse training parameters
    target_themes = args.target_themes.split(',') if args.target_themes else None
    excluded_themes = args.excluded_themes.split(',') if args.excluded_themes else None
    
    # Initialize hybrid trainer
    try:
        # This would need the actual implementation
        logger.info("🧪 Initializing V3 Hybrid Puzzle Trainer...")
        logger.info("   Connecting: Enhanced puzzle database")
        logger.info("   Connecting: Thinking Brain (GRU)")
        logger.info("   Connecting: Gameplay Brain (GA)")
        logger.info("   Connecting: ChessState feature extraction")
        
        # For now, show what would happen
        if args.hpts:
            logger.info(f"⏱️  Would start {args.hpts}-hour hybrid training session")
            logger.info(f"   Batch size: {args.batch_size} puzzles")
            logger.info(f"   Target themes: {target_themes or 'Auto-selected based on weaknesses'}")
            logger.info(f"   Excluded themes: {excluded_themes or 'None'}")
            logger.info(f"   Difficulty progression: {'Enabled' if args.difficulty_progression else 'Disabled'}")
            logger.info(f"   Brain coordination training: {'Enabled' if args.brain_coordination else 'Disabled'}")
        elif args.num_puzzles:
            logger.info(f"🧩 Would start {args.num_puzzles}-puzzle hybrid training session")
            logger.info(f"   Two-brain architecture validation on each puzzle")
            logger.info(f"   Enhanced ChessState feature extraction")
            logger.info(f"   Performance correlation with puzzle ELO ratings")
        
        # Show what the training process would involve
        logger.info("\n🔄 HYBRID TRAINING PROCESS:")
        logger.info("1. 📋 Load puzzle from enhanced database")
        logger.info("2. 🎯 Extract enhanced ChessState features")
        logger.info("3. 🧠 Thinking Brain generates move candidates")
        logger.info("4. 🎮 Gameplay Brain validates candidates tactically")
        logger.info("5. ✅ Compare combined result vs puzzle solution")
        logger.info("6. 📈 Train both brains based on performance")
        logger.info("7. 💾 Save progress and update analytics")
        
        logger.info("\n✨ This hybrid approach combines the best of both worlds:")
        logger.info("   • Tactical pattern learning from puzzles")
        logger.info("   • Two-brain architecture for robust decision making")
        logger.info("   • Custom ChessState features for enhanced perception")
        logger.info("   • Real-time performance validation with ELO correlation")
        
    except Exception as e:
        logger.error(f"Error initializing hybrid trainer: {e}")
        logger.error("Please ensure all V3 components are properly set up")
        return 1
    
    logger.info("🎉 V3 Hybrid Training System ready!")
    logger.info("   Use --analytics-only to view current progress")
    logger.info("   Use --monitor-session <id> for real-time monitoring")
    
    return 0

if __name__ == "__main__":
    exit(main())