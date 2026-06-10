"""
Real-Time Training Monitor for V8.0 Opponent-Based Training

Watches training directory and displays live metrics:
- Win rates per opponent
- Games per hour
- Average game length
- Experience buffer size
- Generation progress
"""

import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import defaultdict
import numpy as np


class TrainingMonitor:
    """Real-time training metrics monitor"""
    
    def __init__(self, training_dir='../training/v8_opponent_training'):
        self.training_dir = Path(training_dir)
        self.training_dir.mkdir(exist_ok=True, parents=True)
        
        # Data storage
        self.generations = []
        self.games_per_hour = []
        self.avg_game_length = []
        self.win_rates = defaultdict(list)
        self.opponent_names = []
        
        # Figure setup
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle('V8.0 Opponent Training - Real-Time Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        # Last generation checked
        self.last_generation = 0
    
    def load_generation_stats(self, gen_num):
        """Load statistics for a specific generation"""
        stats_file = self.training_dir / f"gen_{gen_num:04d}_stats.json"
        
        if not stats_file.exists():
            return None
        
        try:
            with open(stats_file, 'r') as f:
                return json.load(f)
        except:
            return None
    
    def update_data(self):
        """Check for new generation data"""
        # Check for new generations
        stats_files = sorted(self.training_dir.glob("gen_*_stats.json"))
        
        if not stats_files:
            return False
        
        latest_gen = int(stats_files[-1].stem.split('_')[1])
        
        if latest_gen <= self.last_generation:
            return False
        
        # Load new data
        for gen in range(self.last_generation + 1, latest_gen + 1):
            stats = self.load_generation_stats(gen)
            
            if stats:
                self.generations.append(stats['generation'])
                self.games_per_hour.append(stats.get('games_per_hour', 0))
                
                # Extract opponent stats
                opponent_stats = stats.get('opponent_stats', {})
                for opp_name, opp_data in opponent_stats.items():
                    if opp_name not in self.opponent_names:
                        self.opponent_names.append(opp_name)
                    
                    games = opp_data.get('games_played', 0)
                    if games > 0:
                        wins = opp_data.get('wins', 0)
                        win_rate = wins / games * 100
                        self.win_rates[opp_name].append(win_rate)
                    else:
                        self.win_rates[opp_name].append(0)
        
        self.last_generation = latest_gen
        return True
    
    def update_plots(self, frame):
        """Update all plots"""
        # Check for new data
        has_new_data = self.update_data()
        
        if not has_new_data and len(self.generations) == 0:
            # No data yet
            for ax in self.axes.flat:
                ax.clear()
                ax.text(0.5, 0.5, 'Waiting for training data...', 
                       ha='center', va='center', fontsize=12)
            return
        
        if not self.generations:
            return
        
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Plot 1: Games per Hour
        ax1 = self.axes[0, 0]
        ax1.plot(self.generations, self.games_per_hour, 'b-', linewidth=2, marker='o')
        ax1.set_xlabel('Generation', fontweight='bold')
        ax1.set_ylabel('Games/Hour', fontweight='bold')
        ax1.set_title('Training Speed', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=300, color='g', linestyle='--', alpha=0.5, label='Target: 300/hr')
        ax1.legend()
        
        # Plot 2: Win Rates by Opponent
        ax2 = self.axes[0, 1]
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.opponent_names)))
        
        for idx, opp_name in enumerate(self.opponent_names):
            if opp_name in self.win_rates and len(self.win_rates[opp_name]) > 0:
                # Pad win rates to match generations
                padded_rates = [0] * (len(self.generations) - len(self.win_rates[opp_name])) + self.win_rates[opp_name]
                ax2.plot(self.generations, padded_rates, 
                        marker='o', linewidth=2, label=opp_name, color=colors[idx])
        
        ax2.set_xlabel('Generation', fontweight='bold')
        ax2.set_ylabel('Win Rate (%)', fontweight='bold')
        ax2.set_title('Win Rates vs Opponents', fontweight='bold')
        ax2.set_ylim(-5, 105)
        ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% target')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=8)
        
        # Plot 3: Current Status Table
        ax3 = self.axes[1, 0]
        ax3.axis('off')
        
        latest_gen = self.generations[-1]
        latest_stats = self.load_generation_stats(latest_gen)
        
        if latest_stats:
            opponent_stats = latest_stats.get('opponent_stats', {})
            
            # Create table data
            table_data = []
            table_data.append(['Opponent', 'Games', 'W-D-L', 'Win%'])
            
            for opp_name in self.opponent_names:
                if opp_name in opponent_stats:
                    opp = opponent_stats[opp_name]
                    games = opp.get('games_played', 0)
                    wins = opp.get('wins', 0)
                    draws = opp.get('draws', 0)
                    losses = opp.get('losses', 0)
                    win_pct = (wins / games * 100) if games > 0 else 0
                    
                    short_name = opp_name.replace('Opponent', 'Opp').replace('v1.0', '').strip()[:20]
                    record = f"{wins}-{draws}-{losses}"
                    table_data.append([short_name, str(games), record, f"{win_pct:.1f}%"])
            
            table = ax3.table(cellText=table_data, cellLoc='left', loc='center',
                            colWidths=[0.4, 0.15, 0.25, 0.2])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            
            # Style header row
            for i in range(4):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            ax3.set_title(f'Generation {latest_gen} Statistics', fontweight='bold', pad=20)
        
        # Plot 4: Progress Indicator
        ax4 = self.axes[1, 1]
        ax4.axis('off')
        
        # Progress bar
        total_generations = 20
        current_gen = self.generations[-1] if self.generations else 0
        progress = current_gen / total_generations
        
        # Draw progress bar
        bar_width = 0.8
        bar_height = 0.1
        bar_x = 0.1
        bar_y = 0.7
        
        # Background
        rect_bg = plt.Rectangle((bar_x, bar_y), bar_width, bar_height, 
                               facecolor='lightgray', edgecolor='black', linewidth=2)
        ax4.add_patch(rect_bg)
        
        # Progress fill
        rect_fill = plt.Rectangle((bar_x, bar_y), bar_width * progress, bar_height,
                                 facecolor='#4CAF50', edgecolor='black', linewidth=2)
        ax4.add_patch(rect_fill)
        
        # Text
        ax4.text(0.5, bar_y + bar_height + 0.1, 
                f'Generation {current_gen}/{total_generations}',
                ha='center', fontsize=14, fontweight='bold')
        ax4.text(0.5, bar_y + bar_height/2, 
                f'{progress*100:.1f}%',
                ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Stats summary
        if latest_stats:
            total_games = sum(opp.get('games_played', 0) 
                            for opp in opponent_stats.values())
            total_wins = sum(opp.get('wins', 0) 
                           for opp in opponent_stats.values())
            overall_win_rate = (total_wins / total_games * 100) if total_games > 0 else 0
            
            ax4.text(0.5, 0.4, f'Total Games: {total_games}', 
                    ha='center', fontsize=11)
            ax4.text(0.5, 0.3, f'Overall Win Rate: {overall_win_rate:.1f}%', 
                    ha='center', fontsize=11)
            ax4.text(0.5, 0.2, f'Speed: {self.games_per_hour[-1]:.0f} games/hr', 
                    ha='center', fontsize=11)
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.set_title('Training Progress', fontweight='bold', pad=20)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    def run(self):
        """Run the monitoring dashboard"""
        print("="*70)
        print("V8.0 TRAINING MONITOR - Real-Time Dashboard")
        print("="*70)
        print(f"Watching: {self.training_dir}")
        print("Waiting for training data...")
        print("\nPress Ctrl+C to stop monitoring")
        print("="*70)
        
        # Create animation (updates every 2 seconds)
        anim = animation.FuncAnimation(self.fig, self.update_plots, 
                                      interval=2000, cache_frame_data=False)
        
        plt.show()


if __name__ == "__main__":
    monitor = TrainingMonitor()
    monitor.run()
