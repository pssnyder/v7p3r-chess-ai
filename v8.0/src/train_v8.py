"""
V7P3R v8.0 - Main Training Script

PURE LEARNED ARCHITECTURE
- No hand-coded reward weights
- No Stockfish oracle (except optional validation)
- No complex personality/phase logic
- Just: Features + Opening Book + Tablebase + Self-Play + Win/Loss Learning

Expected speed: 100-1000 games per hour (vs v7.0's 6-10 games per hour)
"""

import torch
import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

# V8.0 imports
from network import create_v8_network, count_parameters
from reward_shaper import RewardShaper, RewardShapingTrainer, visualize_learned_weights
from opening_selector import OpeningSelector, OpeningDiversityTracker
from pure_selfplay_trainer import PureSelfPlayGame, GameResult
from comprehensive_features import ComprehensiveFeatureExtractor
from tablebase_oracle import TablebaseOracle

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class V8GenerationalTrainer:
    """
    V8.0 generational training with pure self-play
    
    Key differences from v7.0:
    - NO Stockfish oracle (pure self-play)
    - Opening book with learnable preferences
    - Reward shaper learns feature importance
    - 100-1000x faster training
    """
    
    def __init__(self,
                 num_generations: int = 10,
                 games_per_generation: int = 100,
                 batch_size: int = 256,
                 max_moves_per_game: int = 200,
                 tablebase_path: Optional[str] = None):
        """
        Args:
            num_generations: Number of training generations
            games_per_generation: Self-play games per generation
            batch_size: Training batch size
            max_moves_per_game: Max moves before draw
            tablebase_path: Path to Syzygy tablebases (optional)
        """
        self.num_generations = num_generations
        self.games_per_generation = games_per_generation
        self.batch_size = batch_size
        self.max_moves_per_game = max_moves_per_game
        
        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Using device: {self.device}")
        
        # Initialize components
        self.feature_extractor = ComprehensiveFeatureExtractor()
        
        # Networks
        self.value_network, self.value_trainer = create_v8_network(
            input_dim=55,
            lr=0.001,
            device=self.device
        )
        
        self.reward_shaper = RewardShaper(feature_dim=55, num_feature_groups=10).to(self.device)
        self.shaper_trainer = RewardShapingTrainer(self.reward_shaper, lr=0.001)
        
        logging.info(f"Value network: {count_parameters(self.value_network):,} parameters")
        logging.info(f"Reward shaper: {count_parameters(self.reward_shaper):,} parameters")
        
        # Opening book
        self.opening_selector = OpeningSelector('opening_book.json')
        self.opening_tracker = OpeningDiversityTracker(self.opening_selector.num_openings)
        
        # Tablebase (optional)
        if tablebase_path and Path(tablebase_path).exists():
            self.tablebase_oracle = TablebaseOracle(tablebase_path)
            logging.info(f"Tablebase loaded from {tablebase_path}")
        else:
            self.tablebase_oracle = None
            logging.info("No tablebase - endgames will play to mate/draw")
        
        # Training state
        self.current_generation = 0
        self.experience_buffer = []
        self.generation_results = []
        
        # Statistics
        self.stats = {
            'generation_times': [],
            'games_per_hour': [],
            'average_game_length': [],
            'win_rates': [],
            'tablebase_usage': []
        }
    
    def play_selfplay_generation(self, generation_num: int) -> List[GameResult]:
        """
        Play self-play games for one generation
        
        Args:
            generation_num: Current generation number
        
        Returns:
            List of GameResult objects
        """
        logging.info(f"\nGENERATION {generation_num}/{self.num_generations}")
        logging.info("="*60)
        
        # Create self-play game engine
        selfplay_game = PureSelfPlayGame(
            value_network=self.value_network,
            feature_extractor=self.feature_extractor,
            tablebase_oracle=self.tablebase_oracle,
            max_moves=self.max_moves_per_game,
            temperature=0.3
        )
        
        results = []
        gen_start_time = time.time()
        
        # Play games
        for game_num in range(1, self.games_per_generation + 1):
            # Select opening (with decreasing exploration)
            epsilon = max(0.1, 0.5 - (generation_num * 0.05))  # Decrease over time
            opening_id = self.opening_selector.random_opening() if np.random.random() < epsilon else self.opening_selector.random_opening()
            
            # Play game
            result = selfplay_game.play_game(opening_id, self.opening_selector)
            results.append(result)
            
            # Track opening usage
            self.opening_tracker.record_game(opening_id, result.result)
            
            # Collect experiences
            self.experience_buffer.extend(result.experiences)
            
            # Print progress
            if game_num % 10 == 0:
                avg_moves = np.mean([r.num_moves for r in results[-10:]])
                avg_time = np.mean([r.game_duration_sec for r in results[-10:]])
                games_per_sec = 10 / sum(r.game_duration_sec for r in results[-10:])
                
                logging.info(f"  Games {game_num}/{self.games_per_generation}: "
                           f"{avg_moves:.0f} moves/game, {avg_time:.2f}s/game, "
                           f"{games_per_sec:.1f} games/sec")
        
        # Generation statistics
        gen_duration = time.time() - gen_start_time
        games_per_hour = (self.games_per_generation / gen_duration) * 3600
        
        wins = sum(1 for r in results if r.result == "1-0")
        draws = sum(1 for r in results if r.result == "1/2-1/2")
        losses = sum(1 for r in results if r.result == "0-1")
        tablebase_finishes = sum(1 for r in results if r.tablebase_finish)
        
        logging.info(f"\nGeneration {generation_num} Summary:")
        logging.info(f"  Games: {self.games_per_generation}")
        logging.info(f"  Results: {wins}W - {draws}D - {losses}L")
        logging.info(f"  Tablebase finishes: {tablebase_finishes} ({tablebase_finishes/self.games_per_generation*100:.1f}%)")
        logging.info(f"  Avg moves/game: {np.mean([r.num_moves for r in results]):.1f}")
        logging.info(f"  Duration: {gen_duration/60:.1f} min")
        logging.info(f"  Speed: {games_per_hour:.0f} games/hour")
        logging.info(f"  Experience buffer: {len(self.experience_buffer)} positions")
        
        # Update stats
        self.stats['generation_times'].append(gen_duration)
        self.stats['games_per_hour'].append(games_per_hour)
        self.stats['average_game_length'].append(np.mean([r.num_moves for r in results]))
        self.stats['win_rates'].append(wins / self.games_per_generation)
        self.stats['tablebase_usage'].append(tablebase_finishes / self.games_per_generation)
        
        return results
    
    def train_networks(self, generation_num: int):
        """
        Train value network and reward shaper on collected experiences
        
        Args:
            generation_num: Current generation number
        """
        if len(self.experience_buffer) == 0:
            logging.warning("No experiences to train on!")
            return
        
        logging.info(f"\nTraining networks on {len(self.experience_buffer)} experiences...")
        
        # Prepare training data
        features_list = []
        rewards_list = []
        
        for exp in self.experience_buffer:
            features_list.append(exp.features)
            rewards_list.append([exp.reward])
        
        features_array = np.array(features_list, dtype=np.float32)
        rewards_array = np.array(rewards_list, dtype=np.float32)
        
        # Convert to tensors
        features_tensor = torch.tensor(features_array)
        rewards_tensor = torch.tensor(rewards_array)
        
        # Training loop
        num_epochs = 3
        num_batches = len(features_tensor) // self.batch_size
        
        for epoch in range(num_epochs):
            # Shuffle data
            indices = torch.randperm(len(features_tensor))
            features_shuffled = features_tensor[indices]
            rewards_shuffled = rewards_tensor[indices]
            
            epoch_losses_value = []
            epoch_losses_shaper = []
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = start_idx + self.batch_size
                
                batch_features = features_shuffled[start_idx:end_idx]
                batch_rewards = rewards_shuffled[start_idx:end_idx]
                
                # Train value network
                value_loss = self.value_trainer.train_on_batch(batch_features, batch_rewards)
                epoch_losses_value.append(value_loss)
                
                # Train reward shaper
                shaper_loss = self.shaper_trainer.train_on_batch(batch_features, batch_rewards)
                epoch_losses_shaper.append(shaper_loss)
            
            avg_value_loss = np.mean(epoch_losses_value)
            avg_shaper_loss = np.mean(epoch_losses_shaper)
            
            logging.info(f"  Epoch {epoch+1}/{num_epochs}: "
                       f"Value loss={avg_value_loss:.4f}, Shaper loss={avg_shaper_loss:.4f}")
        
        # Clear buffer after training
        self.experience_buffer = []
        
        logging.info("✓ Network training complete")
    
    def visualize_learned_patterns(self, generation_num: int):
        """Show what the reward shaper has learned"""
        logging.info(f"\nLEARNED PATTERNS (Generation {generation_num}):")
        
        # Create sample positions (opening, middlegame, endgame)
        sample_features = [
            (torch.randn(55), "Random Opening"),
            (torch.randn(55), "Random Middlegame"),
            (torch.randn(55), "Random Endgame")
        ]
        
        for features, desc in sample_features:
            visualize_learned_weights(self.reward_shaper, features, desc)
    
    def save_generation(self, generation_num: int):
        """Save networks and statistics"""
        save_dir = Path('../training/v8_generational')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save value network
        value_path = save_dir / f'gen_{generation_num:04d}_value_network.pt'
        torch.save(self.value_network.state_dict(), value_path)
        
        # Save reward shaper
        shaper_path = save_dir / f'gen_{generation_num:04d}_reward_shaper.pt'
        torch.save(self.reward_shaper.state_dict(), shaper_path)
        
        # Save statistics
        stats_path = save_dir / f'gen_{generation_num:04d}_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logging.info(f"✓ Saved generation {generation_num} to {save_dir}")
    
    def train(self):
        """Run complete training workflow"""
        logging.info("\n" + "="*60)
        logging.info("V7P3R v8.0 - PURE LEARNED TRAINING")
        logging.info("="*60)
        logging.info(f"Generations: {self.num_generations}")
        logging.info(f"Games/generation: {self.games_per_generation}")
        logging.info(f"Openings: {self.opening_selector.num_openings}")
        logging.info(f"Tablebase: {'Yes' if self.tablebase_oracle else 'No'}")
        logging.info("="*60)
        
        overall_start = time.time()
        
        for gen in range(1, self.num_generations + 1):
            # Play self-play games
            results = self.play_selfplay_generation(gen)
            
            # Train networks
            self.train_networks(gen)
            
            # Visualize learned patterns every 3 generations
            if gen % 3 == 0:
                self.visualize_learned_patterns(gen)
            
            # Show opening statistics every 5 generations
            if gen % 5 == 0:
                self.opening_tracker.print_summary(self.opening_selector, top_k=10)
            
            # Save checkpoint
            self.save_generation(gen)
            
            self.current_generation = gen
        
        # Final summary
        overall_duration = time.time() - overall_start
        total_games = self.num_generations * self.games_per_generation
        
        logging.info("\n" + "="*60)
        logging.info("TRAINING COMPLETE!")
        logging.info("="*60)
        logging.info(f"Total time: {overall_duration/3600:.1f} hours")
        logging.info(f"Total games: {total_games}")
        logging.info(f"Overall speed: {total_games / (overall_duration/3600):.0f} games/hour")
        logging.info(f"Avg generation time: {np.mean(self.stats['generation_times'])/60:.1f} min")
        logging.info("="*60)
        
        # Final opening summary
        self.opening_tracker.print_summary(self.opening_selector, top_k=20)


def main():
    """Run V8.0 training"""
    
    # Configuration
    config = {
        'num_generations': 10,
        'games_per_generation': 100,
        'batch_size': 256,
        'max_moves_per_game': 200,
        'tablebase_path': r'E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\pgn_training_data\pgn_data_endgames\3-4-5_pieces_Syzygy\3-4-5'
    }
    
    # Create trainer
    trainer = V8GenerationalTrainer(**config)
    
    # Run training
    trainer.train()


if __name__ == '__main__':
    main()
