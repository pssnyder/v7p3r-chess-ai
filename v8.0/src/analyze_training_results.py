"""
V8.0 Training Results Analysis
Analyzes generation 1-18 training data to identify learning patterns, bottlenecks, and optimization opportunities.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class TrainingAnalyzer:
    """Analyzes V8.0 opponent-based training results"""
    
    def __init__(self, training_dir: str = '../training/v8_opponent_training'):
        self.training_dir = Path(training_dir)
        self.generations_data = []
        self.num_generations = 0
        
    def load_all_generations(self):
        """Load all generation stats files"""
        stats_files = sorted(self.training_dir.glob('gen_*_stats.json'))
        
        for stats_file in stats_files:
            with open(stats_file, 'r') as f:
                data = json.load(f)
                self.generations_data.append(data)
        
        self.num_generations = len(self.generations_data)
        logging.info(f"Loaded {self.num_generations} generations of training data")
    
    def calculate_win_rates(self) -> Tuple[List[float], Dict[str, List[float]]]:
        """Calculate overall and per-opponent win rates across generations"""
        overall_win_rates = []
        opponent_win_rates = defaultdict(list)
        
        for gen_data in self.generations_data:
            total_games = 0
            total_wins = 0
            
            for opponent_name, stats in gen_data['opponent_stats'].items():
                games = stats['games_played']
                wins = stats['wins']
                
                total_games += games
                total_wins += wins
                
                # Per-opponent win rate
                win_rate = (wins / games * 100) if games > 0 else 0
                opponent_win_rates[opponent_name].append(win_rate)
            
            # Overall win rate
            overall_rate = (total_wins / total_games * 100) if total_games > 0 else 0
            overall_win_rates.append(overall_rate)
        
        return overall_win_rates, dict(opponent_win_rates)
    
    def calculate_draw_rates(self) -> List[float]:
        """Calculate draw rate across generations"""
        draw_rates = []
        
        for gen_data in self.generations_data:
            total_games = 0
            total_draws = 0
            
            for opponent_name, stats in gen_data['opponent_stats'].items():
                total_games += stats['games_played']
                total_draws += stats['draws']
            
            draw_rate = (total_draws / total_games * 100) if total_games > 0 else 0
            draw_rates.append(draw_rate)
        
        return draw_rates
    
    def calculate_avg_game_lengths(self) -> Tuple[List[float], Dict[str, List[float]]]:
        """Calculate average game length across generations"""
        overall_avg_moves = []
        opponent_avg_moves = defaultdict(list)
        
        for gen_data in self.generations_data:
            total_moves = 0
            total_games = 0
            
            for opponent_name, stats in gen_data['opponent_stats'].items():
                games = stats['games_played']
                moves = stats['total_moves']
                
                total_moves += moves
                total_games += games
                
                # Per-opponent avg moves
                avg_moves = moves / games if games > 0 else 0
                opponent_avg_moves[opponent_name].append(avg_moves)
            
            # Overall avg moves
            avg = total_moves / total_games if total_games > 0 else 0
            overall_avg_moves.append(avg)
        
        return overall_avg_moves, dict(opponent_avg_moves)
    
    def analyze_opponent_distribution(self) -> Dict[str, List[int]]:
        """Analyze how many games were played against each opponent per generation"""
        opponent_games = defaultdict(list)
        
        for gen_data in self.generations_data:
            for opponent_name, stats in gen_data['opponent_stats'].items():
                opponent_games[opponent_name].append(stats['games_played'])
        
        return dict(opponent_games)
    
    def detect_plateaus(self, metric_values: List[float], window_size: int = 5) -> List[int]:
        """Detect generations where metric plateaued (no improvement over window)"""
        plateaus = []
        
        for i in range(window_size, len(metric_values)):
            window = metric_values[i-window_size:i]
            current = metric_values[i]
            
            # Check if no improvement over window mean
            window_mean = np.mean(window)
            if abs(current - window_mean) < 0.5:  # Less than 0.5% change
                plateaus.append(i + 1)  # 1-indexed generation number
        
        return plateaus
    
    def calculate_learning_velocity(self, metric_values: List[float]) -> List[float]:
        """Calculate rate of change (velocity) for a metric"""
        velocities = [0]  # First generation has no velocity
        
        for i in range(1, len(metric_values)):
            velocity = metric_values[i] - metric_values[i-1]
            velocities.append(velocity)
        
        return velocities
    
    def generate_visualizations(self):
        """Generate comprehensive training analysis visualizations"""
        overall_win_rates, opponent_win_rates = self.calculate_win_rates()
        draw_rates = self.calculate_draw_rates()
        overall_avg_moves, opponent_avg_moves = self.calculate_avg_game_lengths()
        opponent_games = self.analyze_opponent_distribution()
        
        generations = list(range(1, self.num_generations + 1))
        
        # Create figure with 6 subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Overall Win Rate Over Time
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(generations, overall_win_rates, 'b-o', linewidth=2, markersize=6)
        ax1.axhline(y=5, color='r', linestyle='--', alpha=0.5, label='5% baseline')
        ax1.set_xlabel('Generation', fontsize=12)
        ax1.set_ylabel('Win Rate (%)', fontsize=12)
        ax1.set_title('Overall Win Rate Progression', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add plateau detection
        plateaus = self.detect_plateaus(overall_win_rates)
        if plateaus:
            ax1.scatter([p for p in plateaus], [overall_win_rates[p-1] for p in plateaus],
                       color='red', s=100, marker='x', label='Plateau detected', zorder=5)
        
        # 2. Win Rate by Opponent
        ax2 = plt.subplot(2, 3, 2)
        for opponent_name, win_rates in opponent_win_rates.items():
            ax2.plot(generations, win_rates, '-o', label=opponent_name, linewidth=2, markersize=4)
        ax2.set_xlabel('Generation', fontsize=12)
        ax2.set_ylabel('Win Rate (%)', fontsize=12)
        ax2.set_title('Win Rate by Opponent', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=8, loc='best')
        ax2.grid(True, alpha=0.3)
        
        # 3. Draw Rate Over Time
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(generations, draw_rates, 'g-o', linewidth=2, markersize=6)
        ax3.set_xlabel('Generation', fontsize=12)
        ax3.set_ylabel('Draw Rate (%)', fontsize=12)
        ax3.set_title('Draw Rate Progression', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Average Game Length
        ax4 = plt.subplot(2, 3, 4)
        ax4.plot(generations, overall_avg_moves, 'm-o', linewidth=2, markersize=6)
        ax4.set_xlabel('Generation', fontsize=12)
        ax4.set_ylabel('Average Moves per Game', fontsize=12)
        ax4.set_title('Game Length Progression', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Learning Velocity (Win Rate Change)
        ax5 = plt.subplot(2, 3, 5)
        velocities = self.calculate_learning_velocity(overall_win_rates)
        ax5.bar(generations, velocities, color=['green' if v > 0 else 'red' for v in velocities])
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax5.set_xlabel('Generation', fontsize=12)
        ax5.set_ylabel('Win Rate Change (%)', fontsize=12)
        ax5.set_title('Learning Velocity (Generation-to-Generation Change)', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Opponent Game Distribution
        ax6 = plt.subplot(2, 3, 6)
        bottom = np.zeros(self.num_generations)
        for opponent_name, games in opponent_games.items():
            ax6.bar(generations, games, bottom=bottom, label=opponent_name)
            bottom += np.array(games)
        ax6.set_xlabel('Generation', fontsize=12)
        ax6.set_ylabel('Games Played', fontsize=12)
        ax6.set_title('Opponent Distribution per Generation', fontsize=14, fontweight='bold')
        ax6.legend(fontsize=8, loc='best')
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save figure
        output_path = Path('../training/training_analysis_gen1-18.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved visualization to {output_path}")
        
        plt.close()  # Close instead of show for non-interactive mode
    
    def generate_diagnostic_report(self) -> str:
        """Generate comprehensive diagnostic report"""
        overall_win_rates, opponent_win_rates = self.calculate_win_rates()
        draw_rates = self.calculate_draw_rates()
        overall_avg_moves, opponent_avg_moves = self.calculate_avg_game_lengths()
        velocities = self.calculate_learning_velocity(overall_win_rates)
        
        report = []
        report.append("=" * 80)
        report.append("V8.0 TRAINING DIAGNOSTIC REPORT (Generations 1-18)")
        report.append("=" * 80)
        report.append("")
        
        # Overall Performance Summary
        report.append("## OVERALL PERFORMANCE")
        report.append(f"Total Generations: {self.num_generations}")
        report.append(f"Total Games Played: {self.num_generations * 100}")
        report.append(f"")
        report.append(f"Win Rate - Gen 1:  {overall_win_rates[0]:.2f}%")
        report.append(f"Win Rate - Gen 18: {overall_win_rates[-1]:.2f}%")
        report.append(f"Total Improvement:  {overall_win_rates[-1] - overall_win_rates[0]:+.2f}%")
        report.append(f"")
        report.append(f"Draw Rate - Gen 1:  {draw_rates[0]:.2f}%")
        report.append(f"Draw Rate - Gen 18: {draw_rates[-1]:.2f}%")
        report.append(f"")
        report.append(f"Avg Game Length - Gen 1:  {overall_avg_moves[0]:.1f} moves")
        report.append(f"Avg Game Length - Gen 18: {overall_avg_moves[-1]:.1f} moves")
        report.append("")
        
        # Learning Dynamics
        report.append("## LEARNING DYNAMICS")
        report.append("")
        
        # Identify best/worst generations
        best_gen = np.argmax(overall_win_rates) + 1
        worst_gen = np.argmin(overall_win_rates) + 1
        best_velocity_gen = np.argmax(velocities) + 1
        worst_velocity_gen = np.argmin(velocities) + 1
        
        report.append(f"Best Performance:    Gen {best_gen} ({overall_win_rates[best_gen-1]:.2f}% win rate)")
        report.append(f"Worst Performance:   Gen {worst_gen} ({overall_win_rates[worst_gen-1]:.2f}% win rate)")
        report.append(f"")
        report.append(f"Fastest Learning:    Gen {best_velocity_gen} ({velocities[best_velocity_gen-1]:+.2f}% improvement)")
        report.append(f"Biggest Regression:  Gen {worst_velocity_gen} ({velocities[worst_velocity_gen-1]:+.2f}% change)")
        report.append(f"")
        
        # Plateau detection
        plateaus = self.detect_plateaus(overall_win_rates)
        if plateaus:
            report.append(f"⚠️  PLATEAUS DETECTED: Generations {', '.join(map(str, plateaus))}")
            report.append(f"   (No significant improvement over previous 5 generations)")
        else:
            report.append("✓  No prolonged plateaus detected")
        report.append("")
        
        # Per-Opponent Performance
        report.append("## PER-OPPONENT ANALYSIS")
        report.append("")
        
        for opponent_name, win_rates in opponent_win_rates.items():
            gen1_rate = win_rates[0]
            gen18_rate = win_rates[-1]
            improvement = gen18_rate - gen1_rate
            
            report.append(f"{opponent_name}:")
            report.append(f"  Gen 1 Win Rate:  {gen1_rate:.2f}%")
            report.append(f"  Gen 18 Win Rate: {gen18_rate:.2f}%")
            report.append(f"  Improvement:     {improvement:+.2f}%")
            
            # Identify trend
            if improvement > 5:
                report.append(f"  Status: ✓ Learning effectively")
            elif improvement > 0:
                report.append(f"  Status: → Slow learning")
            else:
                report.append(f"  Status: ✗ No learning / regression")
            report.append("")
        
        # Training Efficiency
        report.append("## TRAINING EFFICIENCY")
        report.append("")
        
        total_time_hours = sum([gen['generation_time_sec'] for gen in self.generations_data]) / 3600
        avg_games_per_hour = np.mean([gen['games_per_hour'] for gen in self.generations_data])
        
        report.append(f"Total Training Time: {total_time_hours:.2f} hours")
        report.append(f"Avg Speed: {avg_games_per_hour:.1f} games/hour")
        report.append(f"Games per % Win Rate Improvement: {self.num_generations * 100 / max(overall_win_rates[-1] - overall_win_rates[0], 0.001):.0f} games")
        report.append("")
        
        # Critical Insights
        report.append("## CRITICAL INSIGHTS")
        report.append("")
        
        # Insight 1: Overall learning rate
        if overall_win_rates[-1] < 10:
            report.append("🔴 CRITICAL: Win rate < 10% after 1800 games")
            report.append("   → Learning is extremely slow, fundamental architecture issues likely")
            report.append("")
        
        # Insight 2: Opponent difficulty mismatch
        avg_random_wr = np.mean(opponent_win_rates.get('Random Opponent v1.0', [0]))
        avg_v18_wr = np.mean(opponent_win_rates.get('V7P3R v18.3', [0]))
        
        if avg_random_wr < 20:
            report.append("🔴 CRITICAL: Cannot beat Random Opponent reliably (<20% avg win rate)")
            report.append("   → Opponent pool may be too difficult for initial learning")
            report.append("")
        
        # Insight 3: Plateau analysis
        if len(plateaus) > 5:
            report.append(f"🔴 WARNING: {len(plateaus)} plateau generations detected")
            report.append("   → Learning stagnating, may need curriculum adjustment or exploration")
            report.append("")
        
        # Insight 4: Draw rate analysis
        if draw_rates[-1] > 30:
            report.append(f"⚠️  High draw rate: {draw_rates[-1]:.1f}%")
            report.append("   → Network may be learning to avoid losing rather than winning")
            report.append("")
        
        # Recommendations
        report.append("## RECOMMENDATIONS")
        report.append("")
        
        if overall_win_rates[-1] < 10:
            report.append("1. REDUCE OPPONENT DIFFICULTY:")
            report.append("   - Increase weight on Random/Material opponents (easier baseline)")
            report.append("   - Add weaker intermediate opponents (500-1000 ELO)")
            report.append("   - Consider curriculum learning (start easy, increase difficulty)")
            report.append("")
        
        if avg_random_wr < 50:
            report.append("2. INCREASE EXPLORATION:")
            report.append("   - Raise temperature from 0.3 to 0.5-0.7 (more varied moves)")
            report.append("   - Add epsilon-greedy exploration (10-20% random moves early)")
            report.append("")
        
        if len(velocities) > 5 and np.std(velocities[5:]) < 1.0:
            report.append("3. ACCELERATE LEARNING:")
            report.append("   - Increase batch size (512 → 1024) for more stable gradients")
            report.append("   - Adjust learning rate (try 1e-3 or adaptive scheduler)")
            report.append("   - Add experience replay with priority sampling")
            report.append("")
        
        report.append("4. DIAGNOSTIC ACTIONS:")
        report.append("   - Analyze Gen 5/10/15 learned weights (are patterns meaningful?)")
        report.append("   - Check feature importance (which of 55 features matter most?)")
        report.append("   - Test Gen 18 network in isolation (can it beat Random Opponent?)")
        report.append("   - Compare Gen 1 vs Gen 18 move selection on tactical puzzles")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_report(self, report: str):
        """Save diagnostic report to file"""
        output_path = Path('../training/TRAINING_DIAGNOSTIC_REPORT.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logging.info(f"Saved diagnostic report to {output_path}")
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        self.load_all_generations()
        
        logging.info("Generating visualizations...")
        self.generate_visualizations()
        
        logging.info("Generating diagnostic report...")
        report = self.generate_diagnostic_report()
        
        # Save to file (print may fail on Windows console with Unicode)
        self.save_report(report)
        
        logging.info("\n✓ Analysis complete! Check training/ folder for results.")


if __name__ == '__main__':
    analyzer = TrainingAnalyzer()
    analyzer.run_full_analysis()
