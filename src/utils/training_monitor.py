# training_monitor.py
"""
V7P3R Chess AI 2.0 - Training Monitor
Real-time monitoring of genetic algorithm training progress and bounty system effectiveness.
"""

import json
import time
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import argparse


class TrainingMonitor:
    """Monitor and analyze training progress"""
    
    def __init__(self, logs_dir: str = "logs/genetic"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
    def watch_training(self, check_interval: int = 30):
        """Watch training progress in real-time"""
        print("V7P3R 2.0 Training Monitor Started")
        print("=" * 50)
        
        last_generation = -1
        
        while True:
            try:
                # Find latest generation log
                log_files = list(self.logs_dir.glob("gen_*_stats.json"))
                if not log_files:
                    print("Waiting for training to start...")
                    time.sleep(check_interval)
                    continue
                
                # Get latest generation
                latest_file = max(log_files, key=lambda x: int(x.stem.split('_')[1]))
                generation = int(latest_file.stem.split('_')[1])
                
                if generation > last_generation:
                    self.analyze_generation(latest_file)
                    last_generation = generation
                
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                print("\nMonitoring stopped by user")
                break
            except Exception as e:
                print(f"Monitor error: {e}")
                time.sleep(check_interval)
    
    def analyze_generation(self, log_file: Path):
        """Analyze a single generation's results"""
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)
            
            generation = data['generation']
            best_fitness = data['best_fitness']
            avg_fitness = data['average_fitness']
            
            print(f"\nGeneration {generation:3d} | "
                  f"Best: {best_fitness:6.1f} | "
                  f"Avg: {avg_fitness:6.1f} | "
                  f"Improvement: {best_fitness - avg_fitness:5.1f}")
            
            # Analyze top performers
            if 'population_stats' in data:
                top_performers = data['population_stats'][:3]
                print("Top 3 performers:")
                for i, performer in enumerate(top_performers):
                    print(f"  {i+1}. ID:{performer['genome_id']:2d} | "
                          f"Fitness:{performer['total_bounty']:6.1f} | "
                          f"W/D/L:{performer['wins']}/{performer['draws']}/{performer['losses']} | "
                          f"Avg moves:{performer['average_game_length']:.0f}")
            
            # Check for convergence
            self.check_convergence()
            
        except Exception as e:
            print(f"Error analyzing generation: {e}")
    
    def check_convergence(self):
        """Check if training is converging"""
        log_files = sorted(list(self.logs_dir.glob("gen_*_stats.json")), 
                          key=lambda x: int(x.stem.split('_')[1]))
        
        if len(log_files) < 10:
            return
        
        # Get last 10 generations
        recent_fitness = []
        for log_file in log_files[-10:]:
            try:
                with open(log_file, 'r') as f:
                    data = json.load(f)
                recent_fitness.append(data['best_fitness'])
            except:
                continue
        
        if len(recent_fitness) >= 10:
            # Check improvement rate
            early_avg = np.mean(recent_fitness[:5])
            recent_avg = np.mean(recent_fitness[-5:])
            improvement = recent_avg - early_avg
            
            if improvement < 2.0:  # Less than 2 fitness improvement
                print("  ⚠️  Warning: Training may be converging (low improvement)")
            elif improvement > 20.0:
                print("  🚀 Excellent progress! Strong improvement detected")
    
    def generate_report(self):
        """Generate comprehensive training report"""
        log_files = sorted(list(self.logs_dir.glob("gen_*_stats.json")), 
                          key=lambda x: int(x.stem.split('_')[1]))
        
        if not log_files:
            print("No training data found")
            return
        
        generations = []
        best_fitness = []
        avg_fitness = []
        bounty_stats = []
        
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    data = json.load(f)
                
                generations.append(data['generation'])
                best_fitness.append(data['best_fitness'])
                avg_fitness.append(data['average_fitness'])
                
                # Collect bounty statistics
                if 'population_stats' in data and data['population_stats']:
                    top_performer = data['population_stats'][0]
                    bounty_stats.append({
                        'generation': data['generation'],
                        'total_bounty': top_performer['total_bounty'],
                        'wins': top_performer['wins'],
                        'avg_length': top_performer['average_game_length']
                    })
                    
            except Exception as e:
                print(f"Error reading {log_file}: {e}")
        
        # Create plots
        self.plot_training_progress(generations, best_fitness, avg_fitness)
        self.plot_bounty_analysis(bounty_stats)
        
        # Generate summary
        self.print_training_summary(generations, best_fitness, avg_fitness, bounty_stats)
    
    def plot_training_progress(self, generations: List[int], best_fitness: List[float], avg_fitness: List[float]):
        """Plot fitness progress over generations"""
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(generations, best_fitness, 'b-', label='Best Fitness', linewidth=2)
        plt.plot(generations, avg_fitness, 'r-', label='Average Fitness', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('V7P3R 2.0 Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Improvement rate
        plt.subplot(2, 2, 2)
        if len(best_fitness) > 1:
            improvement = np.diff(best_fitness)
            plt.plot(generations[1:], improvement, 'g-', linewidth=2)
            plt.xlabel('Generation')
            plt.ylabel('Fitness Improvement')
            plt.title('Generation-to-Generation Improvement')
            plt.grid(True, alpha=0.3)
        
        # Moving average
        plt.subplot(2, 2, 3)
        if len(best_fitness) >= 5:
            window = min(5, len(best_fitness) // 4)
            moving_avg = np.convolve(best_fitness, np.ones(window)/window, mode='valid')
            plt.plot(generations[:len(moving_avg)], moving_avg, 'purple', linewidth=2)
            plt.xlabel('Generation')
            plt.ylabel('Moving Average Fitness')
            plt.title(f'Fitness Trend (Window: {window})')
            plt.grid(True, alpha=0.3)
        
        # Fitness distribution
        plt.subplot(2, 2, 4)
        plt.hist(best_fitness, bins=20, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('Fitness Value')
        plt.ylabel('Frequency')
        plt.title('Fitness Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.logs_dir / 'training_progress.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_bounty_analysis(self, bounty_stats: List[Dict]):
        """Plot bounty system effectiveness"""
        if not bounty_stats:
            return
        
        generations = [stat['generation'] for stat in bounty_stats]
        bounties = [stat['total_bounty'] for stat in bounty_stats]
        wins = [stat['wins'] for stat in bounty_stats]
        game_lengths = [stat['avg_length'] for stat in bounty_stats]
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 3, 1)
        plt.plot(generations, bounties, 'gold', linewidth=2, marker='o')
        plt.xlabel('Generation')
        plt.ylabel('Total Bounty')
        plt.title('Bounty Accumulation')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(generations, wins, 'green', linewidth=2, marker='s')
        plt.xlabel('Generation')
        plt.ylabel('Wins')
        plt.title('Win Rate Progress')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.plot(generations, game_lengths, 'red', linewidth=2, marker='^')
        plt.xlabel('Generation')
        plt.ylabel('Average Game Length')
        plt.title('Game Length Trend')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.logs_dir / 'bounty_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_training_summary(self, generations: List[int], best_fitness: List[float], 
                              avg_fitness: List[float], bounty_stats: List[Dict]):
        """Print comprehensive training summary"""
        print("\n" + "=" * 60)
        print("V7P3R 2.0 TRAINING SUMMARY")
        print("=" * 60)
        
        if not generations:
            print("No training data available")
            return
        
        # Basic statistics
        total_generations = len(generations)
        final_best = best_fitness[-1] if best_fitness else 0
        final_avg = avg_fitness[-1] if avg_fitness else 0
        initial_best = best_fitness[0] if best_fitness else 0
        
        print(f"Total Generations: {total_generations}")
        print(f"Initial Best Fitness: {initial_best:.2f}")
        print(f"Final Best Fitness: {final_best:.2f}")
        print(f"Total Improvement: {final_best - initial_best:.2f}")
        print(f"Final Average Fitness: {final_avg:.2f}")
        
        # Progress analysis
        if len(best_fitness) > 1:
            improvement_rate = (final_best - initial_best) / total_generations
            print(f"Average Improvement per Generation: {improvement_rate:.2f}")
        
        # Bounty analysis
        if bounty_stats:
            final_bounty = bounty_stats[-1]['total_bounty']
            initial_bounty = bounty_stats[0]['total_bounty']
            print(f"\nBounty System Analysis:")
            print(f"Initial Bounty: {initial_bounty:.1f}")
            print(f"Final Bounty: {final_bounty:.1f}")
            print(f"Bounty Improvement: {final_bounty - initial_bounty:.1f}")
            
            final_wins = bounty_stats[-1]['wins']
            print(f"Final Win Count: {final_wins}")
            
            avg_game_length = np.mean([stat['avg_length'] for stat in bounty_stats])
            print(f"Average Game Length: {avg_game_length:.1f} moves")
        
        # Recommendations
        print(f"\nRecommendations:")
        if final_best < 30:
            print("- Consider increasing mutation rate or population size")
            print("- Training may need more exploration")
        elif final_best < 70:
            print("- Good progress! Consider longer training")
            print("- May benefit from reduced mutation rate")
        else:
            print("- Excellent results! Consider production training")
            print("- Ready for longer, more intensive training")
        
        if bounty_stats:
            recent_bounty = np.mean([stat['total_bounty'] for stat in bounty_stats[-5:]])
            if recent_bounty < 50:
                print("- Bounty system may need tuning")
                print("- Consider adjusting bounty weights")
        
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="V7P3R 2.0 Training Monitor")
    parser.add_argument('--watch', action='store_true', help='Watch training in real-time')
    parser.add_argument('--report', action='store_true', help='Generate training report')
    parser.add_argument('--logs-dir', type=str, default='logs/genetic', help='Logs directory')
    parser.add_argument('--interval', type=int, default=30, help='Check interval for watching (seconds)')
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(args.logs_dir)
    
    if args.watch:
        monitor.watch_training(args.interval)
    elif args.report:
        monitor.generate_report()
    else:
        print("V7P3R 2.0 Training Monitor")
        print("Use --watch to monitor training or --report to generate analysis")


if __name__ == "__main__":
    main()
