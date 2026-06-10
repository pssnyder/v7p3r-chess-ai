"""
V7P3R v8.0 - Opponent-Based Training Script

ENHANCED LEARNING ARCHITECTURE
- Train against diverse UCI opponents (no self-play echo chamber)
- Learn to exploit weaknesses in historical v7p3r versions
- Faster convergence through opponent diversity
- Same speed as pure self-play (no Stockfish analysis)

Target opponents:
- Themed opponents: Random, Material, Positional (baseline skills)
- V7P3R historical: v17.1, v17.8, v18.3 (progressive difficulty)

Goals:
- Beat all v7p3r versions from v18+ backwards
- Reach tablebase positions in 20-30 moves
"""

import torch
import logging
import json
import time
import chess
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

# V8.0 imports
from network import create_v8_network, count_parameters
from reward_shaper import RewardShaper, RewardShapingTrainer, visualize_learned_weights
from opening_selector import OpeningSelector, OpeningDiversityTracker
from comprehensive_features import ComprehensiveFeatureExtractor
from opponent_manager import create_opponent_pool, OpponentPool, UCIEngine
from uci_game_executor import UCIGameExecutor, GameResult
try:
    from tablebase_oracle import TablebaseOracle
    TABLEBASE_AVAILABLE = True
except:
    TABLEBASE_AVAILABLE = False
    logging.warning("Tablebase oracle not available")

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class V8OpponentTrainer:
    """
    V8.0 training against diverse UCI opponents
    
    Advantages over pure self-play:
    - Learns to exploit specific weaknesses
    - Avoids echo chamber effect
    - More robust feature learning
    - Faster convergence to strong play
    """
    
    def __init__(self,
                 num_generations: int = 20,
                 games_per_generation: int = 100,
                 batch_size: int = 512,
                 max_moves_per_game: int = 200,
                 opponent_pool: Optional[OpponentPool] = None,
                 tablebase_path: Optional[str] = None):
        """
        Args:
            num_generations: Number of training generations
            games_per_generation: Games vs opponents per generation
            batch_size: Training batch size
            max_moves_per_game: Max moves before draw
            opponent_pool: Pool of UCI opponents (created if None)
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
        logging.info(f"Opening book: {self.opening_selector.num_openings} variations")
        
        # Tablebase (optional)
        if TABLEBASE_AVAILABLE and tablebase_path and Path(tablebase_path).exists():
            self.tablebase_oracle = TablebaseOracle(tablebase_path)
            logging.info(f"Tablebase loaded from {tablebase_path}")
        else:
            self.tablebase_oracle = None
            logging.info("No tablebase - endgames will play to mate/draw")
        
        # Opponent pool
        if opponent_pool is None:
            self.opponent_pool = create_opponent_pool()
            logging.info(f"Created default opponent pool: {len(self.opponent_pool.opponents)} opponents")
        else:
            self.opponent_pool = opponent_pool
        
        # Game executor
        self.game_executor = UCIGameExecutor(
            v8_network=self.value_network,
            feature_extractor=self.feature_extractor,
            tablebase=self.tablebase_oracle,
            device=self.device
        )
        
        # Training state
        self.current_generation = 0
        self.experience_buffer = []
        self.generation_results = []
        
        # Statistics
        self.stats = {
            'generation_times': [],
            'games_per_hour': [],
            'average_game_length': [],
            'win_rates_by_opponent': {},
            'tablebase_usage': []
        }
    
    def play_opponent_generation(self, generation_num: int) -> List[GameResult]:
        """
        Play games against opponents for one generation
        
        Args:
            generation_num: Current generation number
        
        Returns:
            List of GameResult objects
        """
        logging.info(f"\nGENERATION {generation_num}/{self.num_generations}")
        logging.info("="*60)
        
        results = []
        gen_start_time = time.time()
        
        # Determine how many games per opponent (roughly)
        games_played = 0
        target_games = self.games_per_generation
        
        # Play games against rotating opponents
        while games_played < target_games:
            # Select opponent
            opponent_config = self.opponent_pool.get_next_opponent(strategy="weighted_random")
            
            logging.info(f"\n  Games {games_played+1}-{min(games_played+2, target_games)}: "
                        f"vs {opponent_config.name} (ELO: {opponent_config.estimated_elo})")
            
            # Launch opponent engine
            try:
                opponent_engine = self.opponent_pool.launch_opponent(opponent_config)
                
                # Play 2 games (one as white, one as black)
                try:
                    white_game, black_game = self.game_executor.play_game_pair(
                        opponent_engine,
                        opponent_name=opponent_config.name
                    )
                    
                    # Record results
                    for game in [white_game, black_game]:
                        results.append(game)
                        games_played += 1
                        
                        # Collect training experience
                        for i, pos in enumerate(game.positions):
                            # Extract features
                            features = self.feature_extractor.extract_all_features(
                                pos,
                                move_number=i,
                                previous_inference_ms=0.0
                            )
                            
                            # Store experience
                            from dataclasses import dataclass
                            @dataclass
                            class Experience:
                                features: np.ndarray
                                reward: float
                            
                            self.experience_buffer.append(
                                Experience(
                                    features=features,
                                    reward=game.outcomes[i]
                                )
                            )
                        
                        # Record game statistics
                        if game.v8_color == chess.WHITE:
                            result_str = game.result
                        else:
                            # Flip result for black perspective
                            if game.result == "1-0":
                                result_str = "0-1"
                            elif game.result == "0-1":
                                result_str = "1-0"
                            else:
                                result_str = game.result
                        
                        self.opponent_pool.record_game(
                            opponent_config.name,
                            result_str,
                            game.num_moves
                        )
                    
                    # Log game results
                    white_result_emoji = "✓" if white_game.result == "1-0" else ("=" if white_game.result == "1/2-1/2" else "✗")
                    black_result_emoji = "✓" if black_game.result == "0-1" else ("=" if black_game.result == "1/2-1/2" else "✗")
                    
                    logging.info(f"    As White: {white_result_emoji} {white_game.result} ({white_game.num_moves} moves, {white_game.termination})")
                    logging.info(f"    As Black: {black_result_emoji} {black_game.result} ({black_game.num_moves} moves, {black_game.termination})")
                
                except Exception as e:
                    logging.error(f"    Game execution error: {e}")
                
                finally:
                    # Clean up opponent engine
                    opponent_engine.cleanup()
            
            except Exception as e:
                logging.error(f"    Failed to launch {opponent_config.name}: {e}")
                # Skip this opponent and continue
                continue
            
            # Progress update every 10 games
            if games_played % 10 == 0 and games_played > 0:
                recent_results = results[-10:]
                avg_moves = np.mean([r.num_moves for r in recent_results])
                elapsed = time.time() - gen_start_time
                games_per_sec = games_played / elapsed if elapsed > 0 else 0
                
                logging.info(f"\n  Progress: {games_played}/{target_games} games")
                logging.info(f"    Avg moves: {avg_moves:.0f}, Speed: {games_per_sec:.2f} games/sec")
        
        # Generation statistics
        gen_duration = time.time() - gen_start_time
        games_per_hour = (len(results) / gen_duration) * 3600 if gen_duration > 0 else 0
        
        # Calculate win/draw/loss from v8's perspective
        v8_wins = sum(1 for r in results if (r.result == "1-0" and r.v8_color == chess.WHITE) or (r.result == "0-1" and r.v8_color == chess.BLACK))
        v8_draws = sum(1 for r in results if r.result == "1/2-1/2")
        v8_losses = sum(1 for r in results if (r.result == "0-1" and r.v8_color == chess.WHITE) or (r.result == "1-0" and r.v8_color == chess.BLACK))
        
        tablebase_finishes = sum(1 for r in results if r.termination == "tablebase")
        
        logging.info(f"\nGeneration {generation_num} Summary:")
        logging.info(f"  Games: {len(results)}")
        logging.info(f"  V8 Results: {v8_wins}W - {v8_draws}D - {v8_losses}L")
        logging.info(f"  Win Rate: {v8_wins/len(results)*100:.1f}%")
        logging.info(f"  Tablebase finishes: {tablebase_finishes} ({tablebase_finishes/len(results)*100:.1f}%)")
        logging.info(f"  Avg moves/game: {np.mean([r.num_moves for r in results]):.1f}")
        logging.info(f"  Duration: {gen_duration/60:.1f} min")
        logging.info(f"  Speed: {games_per_hour:.0f} games/hour")
        logging.info(f"  Experience buffer: {len(self.experience_buffer)} positions")
        
        # Update stats
        self.stats['generation_times'].append(gen_duration)
        self.stats['games_per_hour'].append(games_per_hour)
        self.stats['average_game_length'].append(np.mean([r.num_moves for r in results]))
        self.stats['tablebase_usage'].append(tablebase_finishes / len(results) if len(results) > 0 else 0)
        
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
        num_batches = max(1, len(features_tensor) // self.batch_size)
        
        for epoch in range(num_epochs):
            # Shuffle data
            indices = torch.randperm(len(features_tensor))
            features_shuffled = features_tensor[indices]
            rewards_shuffled = rewards_tensor[indices]
            
            epoch_losses_value = []
            epoch_losses_shaper = []
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(features_tensor))
                
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
    
    def save_generation(self, generation_num: int):
        """Save networks and statistics for current generation"""
        save_dir = Path('../training/v8_opponent_training')
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Save networks
        value_path = save_dir / f"gen_{generation_num:04d}_value_network.pt"
        shaper_path = save_dir / f"gen_{generation_num:04d}_reward_shaper.pt"
        
        torch.save(self.value_network.state_dict(), value_path)
        torch.save(self.reward_shaper.state_dict(), shaper_path)
        
        # Save statistics
        stats_path = save_dir / f"gen_{generation_num:04d}_stats.json"
        stats_data = {
            'generation': generation_num,
            'games_played': len(self.generation_results[-1]) if self.generation_results else 0,
            'generation_time_sec': self.stats['generation_times'][-1] if self.stats['generation_times'] else 0,
            'games_per_hour': self.stats['games_per_hour'][-1] if self.stats['games_per_hour'] else 0,
            'opponent_stats': self.opponent_pool.stats
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        logging.info(f"  Saved generation {generation_num} to {save_dir}")
    
    def find_last_generation(self) -> int:
        """Find the last saved generation checkpoint"""
        save_dir = Path('../training/v8_opponent_training')
        if not save_dir.exists():
            return 0
        
        # Look for value network checkpoints
        checkpoints = list(save_dir.glob('gen_*_value_network.pt'))
        if not checkpoints:
            return 0
        
        # Extract generation numbers
        gen_nums = []
        for cp in checkpoints:
            try:
                gen_num = int(cp.stem.split('_')[1])
                gen_nums.append(gen_num)
            except (IndexError, ValueError):
                continue
        
        return max(gen_nums) if gen_nums else 0
    
    def load_generation(self, generation_num: int):
        """Load networks from a saved generation"""
        save_dir = Path('../training/v8_opponent_training')
        
        value_path = save_dir / f"gen_{generation_num:04d}_value_network.pt"
        shaper_path = save_dir / f"gen_{generation_num:04d}_reward_shaper.pt"
        
        self.value_network.load_state_dict(torch.load(value_path, map_location=self.device))
        self.reward_shaper.load_state_dict(torch.load(shaper_path, map_location=self.device))
        
        logging.info(f"  Loaded networks from generation {generation_num}")
    
    def train(self):
        """Run full training process"""
        logging.info("\n" + "="*70)
        logging.info("V7P3R v8.0 OPPONENT-BASED TRAINING")
        logging.info("="*70)
        logging.info(f"Generations: {self.num_generations}")
        logging.info(f"Games/generation: {self.games_per_generation}")
        logging.info(f"Batch size: {self.batch_size}")
        logging.info(f"Opponents: {len(self.opponent_pool.opponents)}")
        
        # Check for existing checkpoints to resume
        last_gen = self.find_last_generation()
        start_gen = 1
        
        if last_gen > 0:
            logging.info(f"\nFound checkpoint at generation {last_gen}")
            logging.info("Resuming training from generation {}".format(last_gen + 1))
            self.load_generation(last_gen)
            start_gen = last_gen + 1
        else:
            logging.info("\nStarting fresh training from generation 1")
        
        total_start_time = time.time()
        
        for gen in range(start_gen, self.num_generations + 1):
            self.current_generation = gen
            
            # Play games against opponents
            gen_results = self.play_opponent_generation(gen)
            self.generation_results.append(gen_results)
            
            # Train networks
            self.train_networks(gen)
            
            # Visualize learned patterns
            if gen % 5 == 0:
                # Create sample features from starting position for visualization
                sample_board = chess.Board()
                sample_features = self.feature_extractor.extract_all_features(sample_board)
                sample_features_tensor = torch.FloatTensor(sample_features).to(self.device)
                visualize_learned_weights(self.reward_shaper, sample_features_tensor, f"Gen {gen} Learned Weights")
            
            # Save generation
            self.save_generation(gen)
            
            # Print opponent statistics every 5 generations
            if gen % 5 == 0:
                self.opponent_pool.print_summary()
        
        total_duration = time.time() - total_start_time
        
        logging.info("\n" + "="*70)
        logging.info("TRAINING COMPLETE")
        logging.info("="*70)
        logging.info(f"Total time: {total_duration/3600:.2f} hours")
        logging.info(f"Total games: {sum(len(r) for r in self.generation_results)}")
        logging.info(f"Average speed: {(sum(len(r) for r in self.generation_results) / total_duration) * 3600:.0f} games/hour")
        
        # Final opponent statistics
        self.opponent_pool.print_summary()


if __name__ == "__main__":
    # Set tablebase path
    tablebase_path = r'E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\pgn_training_data\pgn_data_endgames\3-4-5_pieces_Syzygy\3-4-5'
    
    # Create trainer
    trainer = V8OpponentTrainer(
        num_generations=20,
        games_per_generation=100,
        batch_size=512,
        max_moves_per_game=200,
        tablebase_path=tablebase_path
    )
    
    # Run training
    trainer.train()
