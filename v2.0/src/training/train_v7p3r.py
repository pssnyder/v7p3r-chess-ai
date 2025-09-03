# train_v7p3r.py
"""
Training Workflow for V7P3R Chess AI
This script orchestrates the complete training workflow:
1. Analyze personal games
2. Train the V7P3R model
3. Validate against Stockfish
4. Generate training reports
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

# Import V7P3R components
from analyze_personal_games import PersonalGameAnalyzer
from personal_style_analyzer import PersonalStyleAnalyzer
from v7p3r_training import V7P3RTrainer
from v7p3r_validation import V7P3RValidator
from chess_core import ChessConfig


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="V7P3R Chess AI Training Workflow")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze personal games, don't train")
    parser.add_argument("--train-only", action="store_true", help="Only train model, don't analyze")
    parser.add_argument("--validate-only", action="store_true", help="Only validate model, don't train")
    parser.add_argument("--episodes", type=int, default=None, help="Number of training episodes")
    parser.add_argument("--no-reports", action="store_true", help="Skip generating reports")
    parser.add_argument("--pgn-path", type=str, default="data/v7p3r_games.pgn", help="Path to PGN file with personal games")
    
    return parser.parse_args()


def setup_directories():
    """Ensure all necessary directories exist"""
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/analysis", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)


def analyze_personal_games(pgn_path):
    """Analyze personal games to extract style patterns"""
    print("\n=== ANALYZING PERSONAL GAMES ===")
    analyzer = PersonalGameAnalyzer(pgn_path)
    analyzer.analyze()
    
    # Ensure personal style analyzer is refreshed with latest data
    config = ChessConfig()
    personal_style = PersonalStyleAnalyzer(config)
    
    # Force re-analysis if needed
    if not personal_style.opening_moves:
        print("Refreshing personal style analysis...")
        personal_style.analyze_games()
    
    return personal_style


def train_model(config, episodes=None):
    """Train the V7P3R AI model"""
    print("\n=== TRAINING V7P3R MODEL ===")
    trainer = V7P3RTrainer(config)
    
    # Override episodes if specified
    if episodes:
        trainer.episodes = episodes
        print(f"Training for {episodes} episodes...")
    
    start_time = time.time()
    trainer.train()
    duration = time.time() - start_time
    
    print(f"\nTraining completed in {duration:.2f} seconds")
    print(f"Model saved to {trainer.model_path}")
    
    return trainer.model_path


def validate_model(config, model_path=None):
    """Validate the V7P3R AI model against Stockfish"""
    print("\n=== VALIDATING V7P3R MODEL ===")
    validator = V7P3RValidator(config)
    
    # Override model path if specified
    if model_path:
        v7p3r_config = config.get_v7p3r_config()
        v7p3r_config["model_path"] = model_path
    
    start_time = time.time()
    results = validator.validate()
    duration = time.time() - start_time
    
    print(f"\nValidation completed in {duration:.2f} seconds")
    
    return results


def generate_reports(personal_style, validation_results, config):
    """Generate training reports and summaries"""
    print("\n=== GENERATING REPORTS ===")
    
    # Create timestamp for report filenames
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Save validation results
    results_path = f"reports/validation_results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    # Generate a summary report
    report_path = f"reports/training_summary_{timestamp}.md"
    
    v7p3r_config = config.get_v7p3r_config()
    training_config = config.get_training_config()
    
    with open(report_path, 'w') as f:
        f.write(f"# V7P3R Chess AI Training Summary\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## Model Configuration\n\n")
        f.write(f"- Model Path: {v7p3r_config.get('model_path')}\n")
        f.write(f"- Search Depth: {v7p3r_config.get('search_depth')}\n")
        f.write(f"- Using Personal History: {v7p3r_config.get('use_personal_history')}\n\n")
        
        f.write(f"## Personal Style Analysis\n\n")
        f.write(f"- Opening Positions: {len(personal_style.opening_moves)}\n")
        f.write(f"- Middlegame Patterns: {len(personal_style.middlegame_patterns)}\n")
        f.write(f"- Endgame Patterns: {len(personal_style.endgame_patterns)}\n")
        f.write(f"- Checkmate Patterns: {len(personal_style.checkmate_patterns)}\n")
        f.write(f"- Winning Sequences: {len(personal_style.winning_sequences)}\n\n")
        
        f.write(f"## Training Parameters\n\n")
        f.write(f"- Learning Rate: {training_config.get('learning_rate')}\n")
        f.write(f"- Discount Factor: {training_config.get('discount_factor')}\n")
        f.write(f"- Exploration Rate: {training_config.get('exploration_rate')}\n")
        f.write(f"- Batch Size: {training_config.get('batch_size')}\n")
        f.write(f"- Episodes: {training_config.get('episodes')}\n\n")
        
        f.write(f"## Validation Results\n\n")
        f.write(f"- Games Played: {validation_results.get('games_played', 0)}\n")
        f.write(f"- Wins: {validation_results.get('wins', 0)}\n")
        f.write(f"- Draws: {validation_results.get('draws', 0)}\n")
        f.write(f"- Losses: {validation_results.get('losses', 0)}\n")
        f.write(f"- Win Rate: {validation_results.get('win_rate', 0)*100:.2f}%\n\n")
        
        f.write(f"## Style Influence\n\n")
        style_weights = v7p3r_config.get("personal_style_weights", {})
        f.write(f"- Opening Style Influence: {style_weights.get('opening_style_influence', 0.8)}\n")
        f.write(f"- Middlegame Style Influence: {style_weights.get('middlegame_style_influence', 0.6)}\n")
        f.write(f"- Endgame Style Influence: {style_weights.get('endgame_style_influence', 0.7)}\n")
        f.write(f"- Winning Patterns Influence: {style_weights.get('winning_patterns_influence', 0.9)}\n")
        f.write(f"- Checkmate Patterns Influence: {style_weights.get('checkmate_patterns_influence', 1.0)}\n")
    
    print(f"Summary report saved to {report_path}")
    print(f"Validation results saved to {results_path}")


def main():
    """Main training workflow function"""
    args = parse_arguments()
    setup_directories()
    config = ChessConfig()
    
    # Step 1: Analyze personal games
    if not args.train_only and not args.validate_only:
        personal_style = analyze_personal_games(args.pgn_path)
    else:
        personal_style = PersonalStyleAnalyzer(config)
    
    # Step 2: Train model
    model_path = None
    if not args.analyze_only and not args.validate_only:
        model_path = train_model(config, args.episodes)
    
    # Step 3: Validate model
    validation_results = {}
    if not args.analyze_only and not args.train_only:
        validation_results = validate_model(config, model_path)
    
    # Step 4: Generate reports
    if not args.no_reports and not args.analyze_only:
        generate_reports(personal_style, validation_results, config)
    
    print("\n=== WORKFLOW COMPLETED ===")


if __name__ == "__main__":
    main()
