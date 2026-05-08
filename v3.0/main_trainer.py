"""
V7P3R AI v3.0 - Main Training Pipeline
======================================

Complete end-to-end training pipeline that integrates all components:
- Thinking Brain (GRU Neural Network)
- Gameplay Brain (Genetic Algorithm) 
- Self-Play Training System
- Bounty/Reward System
- Performance Monitoring

This is the main entry point for training the autonomous chess AI.
"""

import os
import sys
import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
import argparse
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import our AI systems
from ai.thinking_brain import ThinkingBrain, PositionMemory, ThinkingBrainTrainer
from ai.gameplay_brain import GameplayBrain
from training.self_play import SelfPlayTrainer, TrainingProgress
from core.chess_state import ChessStateExtractor
from core.neural_features import NeuralFeatureConverter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class V7P3RAI_TrainingPipeline:
    """
    Complete training pipeline for V7P3R AI v3.0
    
    Orchestrates the entire autonomous learning process from initialization
    through thousands of self-play games to a trained chess engine.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the training pipeline"""
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize AI components
        self.thinking_brain = None
        self.gameplay_brain = None
        self.trainer = None
        
        # Directories
        self.model_dir = Path(self.config.get('model_directory', 'models'))
        self.model_dir.mkdir(exist_ok=True)
        
        logger.info(f"V7P3R AI v3.0 Training Pipeline initialized on device: {self.device}")
    
    def _load_config(self, config_path: str) -> dict:
        """Load training configuration"""
        default_config = {
            # Thinking Brain (GRU) Configuration
            'thinking_brain': {
                'input_size': 690,
                'hidden_size': 256,
                'num_layers': 8,
                'output_size': 4096,
                'dropout': 0.1,
                'learning_rate': 0.001,
                'weight_decay': 1e-5
            },
            
            # Gameplay Brain (GA) Configuration  
            'gameplay_brain': {
                'population_size': 15,
                'simulation_depth': 3,
                'generations': 8,
                'mutation_rate': 0.3,
                'crossover_rate': 0.7,
                'time_limit': 1.5,
                'parallel_simulations': 4
            },
            
            # Training Configuration
            'training': {
                'target_games': 10000,
                'games_per_checkpoint': 1000,
                'max_game_length': 150,
                'resume_from_checkpoint': True,
                'enable_progress_tracking': True
            },
            
            # Visual Monitoring Configuration (Phase 5)
            'monitoring': {
                'enable_visual': False,     # Set to True to enable real-time visualization
                'save_data': True,          # Save monitoring data to files
                'output_dir': 'monitoring_data'  # Directory for monitoring output
            },
            
            # Directories
            'model_directory': 'models',
            'log_directory': 'logs'
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # Merge with defaults
                self._deep_merge(default_config, user_config)
        
        return default_config
    
    def _deep_merge(self, base_dict: dict, update_dict: dict):
        """Deep merge configuration dictionaries"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def initialize_ai_systems(self):
        """Initialize all AI components"""
        logger.info("Initializing AI systems...")
        
        # Initialize Thinking Brain (GRU Neural Network)
        thinking_config = self.config['thinking_brain']
        self.thinking_brain = ThinkingBrain(
            input_size=thinking_config['input_size'],
            hidden_size=thinking_config['hidden_size'],
            num_layers=thinking_config['num_layers'],
            output_size=thinking_config['output_size'],
            dropout=thinking_config['dropout'],
            device=str(self.device)
        )
        
        # Initialize Gameplay Brain (Genetic Algorithm)
        gameplay_config = self.config['gameplay_brain']
        self.gameplay_brain = GameplayBrain(
            population_size=gameplay_config['population_size'],
            simulation_depth=gameplay_config['simulation_depth'],
            generations=gameplay_config['generations'],
            mutation_rate=gameplay_config['mutation_rate'],
            crossover_rate=gameplay_config['crossover_rate'],
            time_limit=gameplay_config['time_limit'],
            parallel_simulations=gameplay_config['parallel_simulations']
        )
        
        # Initialize Self-Play Trainer
        training_config = self.config['training']
        monitoring_config = self.config.get('monitoring', {})
        
        self.trainer = SelfPlayTrainer(
            thinking_brain=self.thinking_brain,
            gameplay_brain=self.gameplay_brain,
            save_directory=str(self.model_dir),
            games_per_checkpoint=training_config['games_per_checkpoint'],
            max_game_length=training_config['max_game_length'],
            enable_progress_tracking=training_config['enable_progress_tracking'],
            enable_visual_monitoring=monitoring_config.get('enable_visual', False),
            monitoring_config=monitoring_config
        )
        
        logger.info("✅ All AI systems initialized successfully!")
    
    def run_training(self):
        """Execute the complete training pipeline"""
        training_config = self.config['training']
        
        logger.info("🚀 Starting V7P3R AI v3.0 autonomous training...")
        logger.info(f"Target: {training_config['target_games']} games")
        logger.info(f"Checkpoints every: {training_config['games_per_checkpoint']} games")
        
        # Run self-play training
        self.trainer.train(
            target_games=training_config['target_games'],
            resume_from_checkpoint=training_config['resume_from_checkpoint']
        )
        
        logger.info("🎉 Training completed successfully!")
        
        # Generate final report
        self._generate_training_report()
    
    def _generate_training_report(self):
        """Generate comprehensive training report"""
        progress = self.trainer.progress
        
        report = {
            'training_summary': {
                'total_games_played': progress.games_played,
                'total_training_time': progress.total_training_time,
                'wins': progress.wins,
                'draws': progress.draws,
                'losses': progress.losses,
                'win_rate': progress.wins / progress.games_played if progress.games_played > 0 else 0,
                'draw_rate': progress.draws / progress.games_played if progress.games_played > 0 else 0,
                'avg_game_length': progress.avg_game_length,
                'avg_move_time': progress.avg_move_time,
                'final_model_loss': progress.latest_model_loss
            },
            'performance_milestones': progress.performance_milestones,
            'thinking_brain_stats': {
                'total_parameters': sum(p.numel() for p in self.thinking_brain.parameters()),
                'device': str(self.thinking_brain.device),
                'architecture': {
                    'input_size': self.thinking_brain.input_size,
                    'hidden_size': self.thinking_brain.hidden_size,
                    'num_layers': self.thinking_brain.num_layers,
                    'output_size': self.thinking_brain.output_size
                }
            },
            'gameplay_brain_stats': self.gameplay_brain.get_performance_stats(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save report
        report_path = self.model_dir / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        logger.info("📊 Training Report Summary:")
        logger.info(f"Games Played: {report['training_summary']['total_games_played']}")
        logger.info(f"Win Rate: {report['training_summary']['win_rate']:.3f}")
        logger.info(f"Draw Rate: {report['training_summary']['draw_rate']:.3f}")
        logger.info(f"Avg Game Length: {report['training_summary']['avg_game_length']:.1f}")
        logger.info(f"Final Model Loss: {report['training_summary']['final_model_loss']:.4f}")
        logger.info(f"Total Parameters: {report['thinking_brain_stats']['total_parameters']:,}")
        logger.info(f"Report saved to: {report_path}")
    
    def validate_systems(self):
        """Quick validation of all systems"""
        logger.info("🔍 Validating AI systems...")
        
        # Test ChessState extraction
        from core.chess_state import ChessStateExtractor
        import chess
        
        extractor = ChessStateExtractor()
        board = chess.Board()
        state = extractor.extract_state(board)
        logger.info(f"✅ ChessState extraction: {len([p for p in state.pieces if p.type > 0])} pieces")
        
        # Test feature conversion
        converter = NeuralFeatureConverter()
        features = converter.convert_to_features(state, device=str(self.device))
        logger.info(f"✅ Feature conversion: {features.shape} tensor on {features.device}")
        
        # Test Thinking Brain
        legal_moves = list(board.legal_moves)
        candidates, probs, hidden = self.thinking_brain.generate_move_candidates(
            features, legal_moves[:5], top_k=3
        )
        logger.info(f"✅ Thinking Brain: Generated {len(candidates)} move candidates")
        
        # Test Gameplay Brain  
        best_move, analysis = self.gameplay_brain.select_best_move(
            board, candidates[:3]
        )
        logger.info(f"✅ Gameplay Brain: Selected {best_move} in {analysis['total_time']:.2f}s")
        
        logger.info("🎯 All systems validated successfully!")


def create_training_config():
    """Create default training configuration file"""
    config = {
        "thinking_brain": {
            "input_size": 690,
            "hidden_size": 256,
            "num_layers": 8,
            "output_size": 4096,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "weight_decay": 1e-5
        },
        "gameplay_brain": {
            "population_size": 15,
            "simulation_depth": 3,
            "generations": 8,
            "mutation_rate": 0.3,
            "crossover_rate": 0.7,
            "time_limit": 1.5,
            "parallel_simulations": 4
        },
        "training": {
            "target_games": 10000,
            "games_per_checkpoint": 1000,
            "max_game_length": 150,
            "resume_from_checkpoint": True,
            "enable_progress_tracking": True
        },
        "monitoring": {
            "enable_visual": False,
            "save_data": True,
            "output_dir": "monitoring_data"
        },
        "model_directory": "models",
        "log_directory": "logs"
    }
    
    with open("training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Created training_config.json with default settings")


def main():
    """Main entry point for V7P3R AI v3.0 training"""
    parser = argparse.ArgumentParser(description="V7P3R AI v3.0 Training Pipeline")
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--validate-only', action='store_true', help='Only validate systems')
    parser.add_argument('--create-config', action='store_true', help='Create default config file')
    parser.add_argument('--quick-test', action='store_true', help='Run quick test with 10 games')
    parser.add_argument('--visual', action='store_true', help='Enable visual monitoring (real-time board display)')
    parser.add_argument('--no-visual', action='store_true', help='Disable visual monitoring (headless mode)')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_training_config()
        return
    
    # Initialize pipeline
    config_path = args.config
    if not config_path and Path("training_config.json").exists():
        config_path = "training_config.json"
        logger.info("Using training_config.json")
    pipeline = V7P3RAI_TrainingPipeline(config_path=config_path)
    
    # Override visual monitoring setting from command line
    if args.visual:
        pipeline.config['monitoring']['enable_visual'] = True
        logger.info("Visual monitoring enabled via command line")
    elif args.no_visual:
        pipeline.config['monitoring']['enable_visual'] = False
        logger.info("Visual monitoring disabled via command line")
    
    pipeline.initialize_ai_systems()
    
    if args.validate_only:
        pipeline.validate_systems()
        return
    
    if args.quick_test:
        # Override config for quick test
        pipeline.config['training']['target_games'] = 10
        pipeline.config['training']['games_per_checkpoint'] = 5
        pipeline.config['gameplay_brain']['population_size'] = 5
        pipeline.config['gameplay_brain']['generations'] = 3
        pipeline.config['gameplay_brain']['time_limit'] = 0.5
        logger.info("🧪 Running quick test with 10 games...")
    
    # Validate systems before training
    pipeline.validate_systems()
    
    # Run training
    pipeline.run_training()


if __name__ == "__main__":
    main()
