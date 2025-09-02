#!/usr/bin/env python
# visualize_training.py
"""
Visualization tool for V7P3R training progress
This script creates visualizations of the training process based on saved model
performance data.
"""

import os
import sys
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="V7P3R Training Visualizer")
    parser.add_argument("--reports-dir", type=str, default="reports",
                        help="Directory containing training reports")
    parser.add_argument("--output-dir", type=str, default="reports/visualizations",
                        help="Directory to save visualizations")
    
    return parser.parse_args()


def setup_directories(output_dir):
    """Ensure output directories exist"""
    os.makedirs(output_dir, exist_ok=True)


def collect_training_data(reports_dir):
    """Collect training data from report files"""
    data = {
        "timestamps": [],
        "win_rates": [],
        "episode_counts": [],
        "rewards": []
    }
    
    # Find all training summary files
    summary_files = [f for f in os.listdir(reports_dir) if f.startswith("training_summary_")]
    
    if not summary_files:
        return None
    
    # Sort by timestamp
    summary_files.sort()
    
    for summary_file in summary_files:
        file_path = os.path.join(reports_dir, summary_file)
        
        try:
            # Parse timestamp from filename
            timestamp_str = summary_file.replace("training_summary_", "").replace(".md", "")
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d-%H%M%S")
            data["timestamps"].append(timestamp)
            
            # Extract data from markdown file
            with open(file_path, 'r') as f:
                content = f.read()
                
                # Extract episode count
                episode_match = content.find("Episodes: ")
                if episode_match != -1:
                    line = content[episode_match:].split("\n")[0]
                    episodes = int(line.replace("Episodes: ", ""))
                    data["episode_counts"].append(episodes)
                else:
                    data["episode_counts"].append(None)
                
                # Extract win rate
                win_rate_match = content.find("Win Rate: ")
                if win_rate_match != -1:
                    line = content[win_rate_match:].split("\n")[0]
                    win_rate = float(line.replace("Win Rate: ", "").replace("%", "")) / 100
                    data["win_rates"].append(win_rate)
                else:
                    data["win_rates"].append(0)
        
        except Exception as e:
            print(f"Error processing file {summary_file}: {e}")
    
    return data


def collect_validation_results(reports_dir):
    """Collect validation results from result files"""
    data = {
        "timestamps": [],
        "elo_levels": [],
        "win_rates": []
    }
    
    # Find all validation results files
    results_files = [f for f in os.listdir(reports_dir) if f.startswith("validation_results_")]
    
    if not results_files:
        return None
    
    # Sort by timestamp
    results_files.sort()
    
    for results_file in results_files:
        file_path = os.path.join(reports_dir, results_file)
        
        try:
            # Parse timestamp from filename
            timestamp_str = results_file.replace("validation_results_", "").replace(".json", "")
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d-%H%M%S")
            
            # Extract data from JSON file
            with open(file_path, 'r') as f:
                results = json.load(f)
                
                elo = results.get("stockfish_elo", 400)
                win_rate = results.get("win_rate", 0)
                
                data["timestamps"].append(timestamp)
                data["elo_levels"].append(elo)
                data["win_rates"].append(win_rate)
        
        except Exception as e:
            print(f"Error processing file {results_file}: {e}")
    
    return data


def create_win_rate_chart(data, output_dir):
    """Create win rate chart over time"""
    if not data or not data["timestamps"]:
        print("No training data available for win rate chart")
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(data["timestamps"], [rate * 100 for rate in data["win_rates"]], 'b-o', linewidth=2)
    plt.title("V7P3R Win Rate Over Time", fontsize=16)
    plt.xlabel("Training Time", fontsize=12)
    plt.ylabel("Win Rate (%)", fontsize=12)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, "win_rate_over_time.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Win rate chart saved to {output_path}")


def create_elo_performance_chart(data, output_dir):
    """Create ELO performance chart"""
    if not data or not data["elo_levels"]:
        print("No validation data available for ELO performance chart")
        return
    
    # Group by ELO level
    elo_levels = sorted(set(data["elo_levels"]))
    win_rates_by_elo = {elo: [] for elo in elo_levels}
    
    for i, elo in enumerate(data["elo_levels"]):
        win_rates_by_elo[elo].append(data["win_rates"][i] * 100)
    
    # Calculate average win rate for each ELO level
    avg_win_rates = [np.mean(win_rates_by_elo[elo]) for elo in elo_levels]
    
    plt.figure(figsize=(10, 6))
    plt.bar(elo_levels, avg_win_rates, color='blue', alpha=0.7)
    plt.title("V7P3R Performance Against Different ELO Levels", fontsize=16)
    plt.xlabel("Stockfish ELO", fontsize=12)
    plt.ylabel("Win Rate (%)", fontsize=12)
    plt.grid(True, axis='y')
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, "elo_performance.png")
    plt.savefig(output_path)
    plt.close()
    print(f"ELO performance chart saved to {output_path}")


def main():
    """Main function"""
    args = parse_arguments()
    
    # Setup directories
    setup_directories(args.output_dir)
    
    # Collect data
    training_data = collect_training_data(args.reports_dir)
    validation_data = collect_validation_results(args.reports_dir)
    
    # Create visualizations
    if training_data:
        create_win_rate_chart(training_data, args.output_dir)
    else:
        print("No training data found")
    
    if validation_data:
        create_elo_performance_chart(validation_data, args.output_dir)
    else:
        print("No validation data found")
    
    print("Visualization complete")


if __name__ == "__main__":
    main()
