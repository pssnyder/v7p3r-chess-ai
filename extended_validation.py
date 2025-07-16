#!/usr/bin/env python
# extended_validation.py
"""
Extended validation script for V7P3R Chess AI
This script runs a comprehensive validation of the trained model:
1. Validates against Stockfish at multiple ELO levels
2. Simulates games with longer time per move for quality analysis
3. Generates detailed reports on performance
"""

import os
import sys
import time
import json
import argparse
import subprocess
from datetime import datetime

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="V7P3R Extended Validation")
    parser.add_argument("--elo-levels", type=str, default="400,800,1200,1600",
                        help="Comma-separated list of Stockfish ELO levels to test against")
    parser.add_argument("--games-per-level", type=int, default=5,
                        help="Number of games to play against each ELO level")
    parser.add_argument("--time-per-move", type=float, default=0.5,
                        help="Time per move in seconds")
    parser.add_argument("--output-dir", type=str, default="reports/extended_validation",
                        help="Directory to save validation results")
    
    return parser.parse_args()

def setup_directories(output_dir):
    """Ensure output directories exist"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "pgn"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)

def run_validation_against_elo(elo, games, time_per_move, output_dir):
    """Run validation against specific ELO level"""
    print(f"\n=== Validating against Stockfish ELO {elo} ===")
    
    # Update config for validation
    config_file = "config.json"
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Store original ELO
    original_elo = config["stockfish_config"]["elo_rating"]
    
    # Update config with test ELO
    config["stockfish_config"]["elo_rating"] = elo
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Run validation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    validation_command = f"python v7p3r_validation.py --games {games} --save-report"
    subprocess.run(validation_command, shell=True)
    
    # Run simulation for detailed game analysis
    simulation_command = f"python simulate_games.py --games {games} --elo {elo} --time-per-move {time_per_move} --verbose --save-dir {os.path.join(output_dir, 'pgn', f'elo_{elo}')}"
    subprocess.run(simulation_command, shell=True)
    
    # Restore original config
    config["stockfish_config"]["elo_rating"] = original_elo
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Copy last validation results to output directory
    latest_validation = max([f for f in os.listdir("reports") if f.startswith("validation_results_")], 
                            key=lambda x: os.path.getmtime(os.path.join("reports", x)))
    
    with open(os.path.join("reports", latest_validation), 'r') as f:
        validation_results = json.load(f)
    
    # Save results with ELO level included
    results_file = os.path.join(output_dir, "results", f"validation_elo_{elo}_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(validation_results, f, indent=4)
    
    return validation_results

def compile_overall_results(results_by_elo, output_dir):
    """Compile results from all ELO levels into a single report"""
    print("\n=== Compiling Overall Results ===")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    overall_results = {
        "timestamp": timestamp,
        "elo_levels": list(results_by_elo.keys()),
        "summary": {
            "total_games": sum(r.get("games_played", 0) for r in results_by_elo.values()),
            "total_wins": sum(r.get("wins", 0) for r in results_by_elo.values()),
            "total_draws": sum(r.get("draws", 0) for r in results_by_elo.values()),
            "total_losses": sum(r.get("losses", 0) for r in results_by_elo.values())
        },
        "by_elo": results_by_elo
    }
    
    # Calculate overall win rate
    total_games = overall_results["summary"]["total_games"]
    if total_games > 0:
        overall_results["summary"]["overall_win_rate"] = (
            overall_results["summary"]["total_wins"] / total_games
        )
    else:
        overall_results["summary"]["overall_win_rate"] = 0
    
    # Save overall results
    overall_file = os.path.join(output_dir, f"extended_validation_results_{timestamp}.json")
    with open(overall_file, 'w') as f:
        json.dump(overall_results, f, indent=4)
    
    # Generate markdown report
    report_file = os.path.join(output_dir, f"extended_validation_report_{timestamp}.md")
    with open(report_file, 'w') as f:
        f.write(f"# V7P3R Extended Validation Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## Overall Summary\n\n")
        f.write(f"- Total Games: {overall_results['summary']['total_games']}\n")
        f.write(f"- Total Wins: {overall_results['summary']['total_wins']}\n")
        f.write(f"- Total Draws: {overall_results['summary']['total_draws']}\n")
        f.write(f"- Total Losses: {overall_results['summary']['total_losses']}\n")
        f.write(f"- Overall Win Rate: {overall_results['summary']['overall_win_rate']*100:.2f}%\n\n")
        
        f.write(f"## Results by ELO Level\n\n")
        for elo, results in results_by_elo.items():
            win_rate = results.get("win_rate", 0) * 100
            f.write(f"### Stockfish ELO {elo}\n\n")
            f.write(f"- Games Played: {results.get('games_played', 0)}\n")
            f.write(f"- Wins: {results.get('wins', 0)}\n")
            f.write(f"- Draws: {results.get('draws', 0)}\n")
            f.write(f"- Losses: {results.get('losses', 0)}\n")
            f.write(f"- Win Rate: {win_rate:.2f}%\n\n")
    
    print(f"Extended validation report saved to: {report_file}")
    print(f"Extended validation results saved to: {overall_file}")
    
    return overall_results

def main():
    """Main function"""
    args = parse_arguments()
    output_dir = args.output_dir
    
    # Setup directories
    setup_directories(output_dir)
    
    # Parse ELO levels
    elo_levels = [int(elo) for elo in args.elo_levels.split(",")]
    
    # Run validation against each ELO level
    results_by_elo = {}
    for elo in elo_levels:
        results = run_validation_against_elo(
            elo, 
            args.games_per_level, 
            args.time_per_move, 
            output_dir
        )
        results_by_elo[str(elo)] = results
    
    # Compile overall results
    compile_overall_results(results_by_elo, output_dir)
    
    print("\n=== Extended Validation Complete ===")


if __name__ == "__main__":
    main()
