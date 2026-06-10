#!/usr/bin/env python3
"""
Batch Self-Play Runner
V7P3R AI v6.1 - Stage 2 Training Data Generation

Generates 284 diverse self-play games for Stage 2 training.
Distributes games across time scenarios to create diverse training data.

Target: Match human learning efficiency (284 games median benchmark)

Supports parallel processing with multiprocessing.Pool for faster generation.

Author: Pat Snyder
Created: 2026-05-31
"""

import sys
from pathlib import Path
import random
from typing import List, Dict
import json
from datetime import datetime
import multiprocessing as mp
from functools import partial

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.selfplay.monte_carlo_selfplay import MonteCarloSelfPlay, TIME_SCENARIOS


def play_single_game(args):
    """
    Worker function to play a single game (for parallel processing).
    
    Args:
        args: Tuple of (game_num, scenario_name, stage1_model_path, output_dir, max_moves)
        
    Returns:
        Tuple of (game_num, game_data, scenario_name)
    """
    game_num, scenario_name, stage1_model_path, output_dir, max_moves = args
    
    # Initialize self-play engine (each worker gets its own instance)
    selfplay = MonteCarloSelfPlay(
        stage1_model_path=stage1_model_path,
        output_dir=output_dir,
        device='cpu'
    )
    
    scenario = TIME_SCENARIOS[scenario_name]
    
    # Play game
    game_data = selfplay.play_game(
        time_scenario=scenario,
        max_moves=max_moves,
        resignation_threshold_cp=800,
        resignation_move_count=5
    )
    
    # Save game data
    selfplay.save_game_data(game_data, format='jsonl')
    
    return game_num, game_data, scenario_name


class BatchSelfPlayRunner:
    """
    Runs batch of self-play games with scenario distribution.
    
    Distribution Strategy:
    - 60% Blitz (170 games): 40% early, 35% midgame, 25% endgame
    - 20% Bullet (57 games): 30% early, 35% midgame, 35% endgame  
    - 20% Rapid (57 games): 50% early, 30% midgame, 20% endgame
    
    Total: 284 games (historical median benchmark)
    """
    
    def __init__(
        self,
        stage1_model_path: Path,
        output_dir: Path,
        target_games: int = 284
    ):
        """
        Initialize batch runner.
        
        Args:
            stage1_model_path: Path to trained Stage 1 model
            output_dir: Output directory for game data
            target_games: Total games to generate (default 284)
        """
        self.target_games = target_games
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stage1_model_path = stage1_model_path
        
        # Initialize self-play engine (for sequential mode)
        print(f"Initializing self-play engine...")
        self.selfplay = MonteCarloSelfPlay(
            stage1_model_path=stage1_model_path,
            output_dir=output_dir,
            device='cpu'
        )
        
        # Calculate scenario distribution
        self.scenario_distribution = self._calculate_scenario_distribution()
        
        # Statistics
        self.stats = {
            'total_games': 0,
            'total_positions': 0,
            'games_by_scenario': {},
            'games_by_result': {'1-0': 0, '0-1': 0, '1/2-1/2': 0},
            'avg_moves_per_game': 0.0,
            'avg_positions_per_game': 0.0,
        }
        
    def _calculate_scenario_distribution(self) -> List[str]:
        """
        Calculate scenario distribution for target games.
        
        Returns:
            List of scenario names (length = target_games)
        """
        # Distribution percentages
        blitz_games = int(self.target_games * 0.60)  # 60% blitz
        bullet_games = int(self.target_games * 0.20)  # 20% bullet
        rapid_games = self.target_games - blitz_games - bullet_games  # Remaining rapid
        
        scenarios = []
        
        # Blitz distribution (early/mid/endgame)
        scenarios.extend(['blitz_early'] * int(blitz_games * 0.40))
        scenarios.extend(['blitz_midgame'] * int(blitz_games * 0.35))
        scenarios.extend(['blitz_endgame'] * (blitz_games - int(blitz_games * 0.40) - int(blitz_games * 0.35)))
        
        # Bullet distribution
        scenarios.extend(['bullet_early'] * int(bullet_games * 0.30))
        scenarios.extend(['bullet_midgame'] * int(bullet_games * 0.35))
        scenarios.extend(['bullet_endgame'] * (bullet_games - int(bullet_games * 0.30) - int(bullet_games * 0.35)))
        
        # Rapid distribution
        scenarios.extend(['rapid_early'] * int(rapid_games * 0.50))
        scenarios.extend(['rapid_midgame'] * (rapid_games - int(rapid_games * 0.50)))
        
        # Shuffle for variety
        random.shuffle(scenarios)
        
        return scenarios
    
    def run_batch(
        self,
        max_moves_per_game: int = 150,
        save_interval: int = 10,
        resume: bool = True,
        workers: int = 1
    ):
        """
        Run batch of self-play games.
        
        Args:
            max_moves_per_game: Maximum moves per game before draw
            save_interval: Save progress every N games
            resume: Resume from previous run if available
            workers: Number of parallel workers (1 = sequential, >1 = parallel)
        """
        if workers > 1:
            self.run_batch_parallel(
                max_moves_per_game=max_moves_per_game,
                save_interval=save_interval,
                resume=resume,
                workers=workers
            )
        else:
            self.run_batch_sequential(
                max_moves_per_game=max_moves_per_game,
                save_interval=save_interval,
                resume=resume
            )
    
    def run_batch_parallel(
        self,
        max_moves_per_game: int = 150,
        save_interval: int = 10,
        resume: bool = True,
        workers: int = 4
    ):
        """
        Run batch of self-play games in parallel using multiprocessing.
        
        Args:
            max_moves_per_game: Maximum moves per game before draw
            save_interval: Save progress every N games
            resume: Resume from previous run if available
            workers: Number of parallel worker processes
        """
        print("=" * 70)
        print("V7P3R AI Stage 2 Training Data Generation (PARALLEL)")
        print("=" * 70)
        print(f"Target games: {self.target_games}")
        print(f"Workers: {workers}")
        print(f"Scenario distribution:")
        scenario_counts = {}
        for scenario in self.scenario_distribution:
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        for scenario, count in sorted(scenario_counts.items()):
            pct = (count / self.target_games) * 100
            print(f"  {scenario:20s}: {count:3d} games ({pct:5.1f}%)")
        print(f"Output directory: {self.output_dir}")
        print("=" * 70)
        
        # Resume check
        start_game = 0
        if resume:
            progress_file = self.output_dir / "batch_progress.json"
            if progress_file.exists():
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                start_game = progress.get('games_completed', 0)
                if start_game > 0:
                    print(f"\n✓ Resuming from game {start_game + 1}")
                    self.stats = progress.get('stats', self.stats)
        
        # Prepare work items
        work_items = [
            (
                game_num,
                self.scenario_distribution[game_num],
                self.stage1_model_path,
                self.output_dir,
                max_moves_per_game
            )
            for game_num in range(start_game, self.target_games)
        ]
        
        # Process games in parallel
        print(f"\nStarting parallel processing with {workers} workers...")
        completed_count = start_game
        
        with mp.Pool(processes=workers) as pool:
            # Use imap_unordered for progress updates
            for game_num, game_data, scenario_name in pool.imap_unordered(
                play_single_game, work_items
            ):
                completed_count += 1
                
                # Update statistics
                self._update_stats(game_data, scenario_name)
                
                print(f"[{completed_count}/{self.target_games}] "
                      f"Game {game_num + 1} complete ({scenario_name}): "
                      f"{game_data['result']} ({game_data['moves']} moves)")
                
                # Save progress periodically
                if completed_count % save_interval == 0:
                    self._save_progress(completed_count)
                    self._print_stats()
        
        # Final save
        self._save_progress(self.target_games)
        self._save_final_report()
        
        print("\n" + "=" * 70)
        print("✓ Parallel batch self-play complete!")
        self._print_stats()
        print("=" * 70)
    
    def run_batch_sequential(
        self,
        max_moves_per_game: int = 150,
        save_interval: int = 10,
        resume: bool = True
    ):
        """
        Run batch of self-play games sequentially (single-threaded).
        
        Args:
            max_moves_per_game: Maximum moves per game before draw
            save_interval: Save progress every N games
            resume: Resume from previous run if available
        """
        print("=" * 70)
        print("V7P3R AI Stage 2 Training Data Generation (SEQUENTIAL)")
        print("=" * 70)
        print(f"Target games: {self.target_games}")
        print(f"Scenario distribution:")
        scenario_counts = {}
        for scenario in self.scenario_distribution:
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        for scenario, count in sorted(scenario_counts.items()):
            pct = (count / self.target_games) * 100
            print(f"  {scenario:20s}: {count:3d} games ({pct:5.1f}%)")
        print(f"Output directory: {self.output_dir}")
        print("=" * 70)
        
        # Resume check
        start_game = 0
        if resume:
            progress_file = self.output_dir / "batch_progress.json"
            if progress_file.exists():
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                start_game = progress.get('games_completed', 0)
                if start_game > 0:
                    print(f"\n✓ Resuming from game {start_game + 1}")
                    self.stats = progress.get('stats', self.stats)
        
        # Play games
        for game_num in range(start_game, self.target_games):
            scenario_name = self.scenario_distribution[game_num]
            scenario = TIME_SCENARIOS[scenario_name]
            
            print(f"\n[{game_num + 1}/{self.target_games}] "
                  f"Playing game ({scenario_name})...")
            
            try:
                # Play game
                game_data = self.selfplay.play_game(
                    time_scenario=scenario,
                    max_moves=max_moves_per_game,
                    resignation_threshold_cp=800,
                    resignation_move_count=5
                )
                
                # Save game data
                self.selfplay.save_game_data(game_data, format='jsonl')
                
                # Update statistics
                self._update_stats(game_data, scenario_name)
                
                # Save progress
                if (game_num + 1) % save_interval == 0:
                    self._save_progress(game_num + 1)
                    self._print_stats()
                
            except Exception as e:
                print(f"  ERROR in game {game_num + 1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Final save
        self._save_progress(self.target_games)
        self._save_final_report()
        
        print("\n" + "=" * 70)
        print("✓ Batch self-play complete!")
        self._print_stats()
        print("=" * 70)
    
    def _update_stats(self, game_data: Dict, scenario_name: str):
        """Update running statistics."""
        self.stats['total_games'] += 1
        self.stats['total_positions'] += len(game_data['positions'])
        
        # By scenario
        if scenario_name not in self.stats['games_by_scenario']:
            self.stats['games_by_scenario'][scenario_name] = 0
        self.stats['games_by_scenario'][scenario_name] += 1
        
        # By result
        result = game_data['result']
        if result in self.stats['games_by_result']:
            self.stats['games_by_result'][result] += 1
        
        # Averages
        self.stats['avg_moves_per_game'] = (
            (self.stats['avg_moves_per_game'] * (self.stats['total_games'] - 1) + game_data['moves'])
            / self.stats['total_games']
        )
        self.stats['avg_positions_per_game'] = (
            self.stats['total_positions'] / self.stats['total_games']
        )
    
    def _save_progress(self, games_completed: int):
        """Save progress checkpoint."""
        progress_file = self.output_dir / "batch_progress.json"
        progress = {
            'games_completed': games_completed,
            'target_games': self.target_games,
            'timestamp': datetime.now().isoformat(),
            'stats': self.stats,
        }
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def _save_final_report(self):
        """Save final batch report."""
        report_file = self.output_dir / "batch_report.json"
        report = {
            'timestamp': datetime.now().isoformat(),
            'target_games': self.target_games,
            'stats': self.stats,
            'scenario_distribution': {
                scenario: count 
                for scenario, count in sorted(
                    self.stats['games_by_scenario'].items()
                )
            },
        }
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Also save as markdown
        report_md = self.output_dir / "BATCH_REPORT.md"
        with open(report_md, 'w') as f:
            f.write(f"# Stage 2 Self-Play Batch Report\n\n")
            f.write(f"**Generated**: {report['timestamp']}\n\n")
            f.write(f"## Summary\n\n")
            f.write(f"- **Total Games**: {self.stats['total_games']}\n")
            f.write(f"- **Total Positions**: {self.stats['total_positions']}\n")
            f.write(f"- **Avg Moves/Game**: {self.stats['avg_moves_per_game']:.1f}\n")
            f.write(f"- **Avg Positions/Game**: {self.stats['avg_positions_per_game']:.1f}\n\n")
            f.write(f"## Results Distribution\n\n")
            for result, count in self.stats['games_by_result'].items():
                pct = (count / self.stats['total_games']) * 100
                f.write(f"- **{result}**: {count} games ({pct:.1f}%)\n")
            f.write(f"\n## Scenario Distribution\n\n")
            for scenario, count in sorted(self.stats['games_by_scenario'].items()):
                pct = (count / self.stats['total_games']) * 100
                f.write(f"- **{scenario}**: {count} games ({pct:.1f}%)\n")
    
    def _print_stats(self):
        """Print current statistics."""
        print(f"\n  Statistics:")
        print(f"    Games: {self.stats['total_games']}")
        print(f"    Positions: {self.stats['total_positions']}")
        print(f"    Avg moves/game: {self.stats['avg_moves_per_game']:.1f}")
        print(f"    Results: W={self.stats['games_by_result']['1-0']} "
              f"B={self.stats['games_by_result']['0-1']} "
              f"D={self.stats['games_by_result']['1/2-1/2']}")


# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Stage 2 self-play training data")
    parser.add_argument(
        '--model',
        type=Path,
        default=Path('models/position_evaluator_best.pth'),
        help='Path to Stage 1 model'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/stage2/selfplay_batch_284'),
        help='Output directory'
    )
    parser.add_argument(
        '--games',
        type=int,
        default=284,
        help='Number of games to generate (default: 284 historical benchmark)'
    )
    parser.add_argument(
        '--max-moves',
        type=int,
        default=150,
        help='Maximum moves per game'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Do not resume from previous run'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of parallel workers (1=sequential, 4=recommended for parallel)'
    )
    
    args = parser.parse_args()
    
    # Validate Stage 1 model exists
    if not args.model.exists():
        print(f"ERROR: Stage 1 model not found at {args.model}")
        print("Please train Stage 1 model first:")
        print("  python scripts/stage1/train_balanced.py")
        sys.exit(1)
    
    # Validate worker count
    if args.workers < 1:
        print(f"ERROR: Workers must be >= 1, got {args.workers}")
        sys.exit(1)
    
    if args.workers > mp.cpu_count():
        print(f"WARNING: {args.workers} workers requested but only {mp.cpu_count()} CPUs available")
        print(f"Recommend using {mp.cpu_count()} or fewer workers")
    
    # Performance estimates
    if args.workers == 1:
        est_time = "~9-10 hours"
    elif args.workers <= 4:
        est_time = f"~{10/args.workers:.1f}-{12/args.workers:.1f} hours"
    else:
        est_time = f"~{10/args.workers:.1f}-{12/args.workers:.1f} hours"
    
    print(f"\nEstimated completion time: {est_time}")
    print(f"Mode: {'PARALLEL' if args.workers > 1 else 'SEQUENTIAL'}")
    if args.workers > 1:
        print(f"Workers: {args.workers}")
    print()
    
    # Run batch
    runner = BatchSelfPlayRunner(
        stage1_model_path=args.model,
        output_dir=args.output,
        target_games=args.games
    )
    
    runner.run_batch(
        max_moves_per_game=args.max_moves,
        save_interval=10,
        resume=not args.no_resume,
        workers=args.workers
    )
    
    print("\n✓ Self-play data generation complete!")
    print(f"\nNext steps:")
    print(f"  1. Review batch report: {args.output}/BATCH_REPORT.md")
    print(f"  2. Verify feature compatibility: python scripts/stage2/verify_compatibility.py")
    print(f"  3. Train Stage 2 model: python scripts/stage2/train_stage2.py")
    
    if args.workers == 1:
        print(f"\n💡 TIP: Use --workers 4 for ~4x faster generation!")
        print(f"   Example: python {sys.argv[0]} --workers 4")
