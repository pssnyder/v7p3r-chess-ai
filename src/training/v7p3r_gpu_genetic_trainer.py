"""
GPU-Accelerated Genetic Algorithm Trainer for V7P3R Chess AI v2.0
Combines genetic algorithm optimization with PyTorch/CUDA acceleration
"""

import os
import json
import time
import random
import threading
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not available. GPU acceleration disabled.")
    TORCH_AVAILABLE = False

import chess
import chess.engine
import numpy as np

# Import local modules
from chess_core import BoardEvaluator
from v7p3r_bounty_system import ExtendedBountyEvaluator
from v7p3r_rnn_model import V7P3RNeuralNetwork
from v7p3r_gpu_model import V7P3RGPU_LSTM, GPUChessFeatureExtractor

class V7P3RGPUGeneticTrainer:
    """
    GPU-accelerated genetic algorithm trainer for V7P3R Chess AI.
    Uses PyTorch/CUDA for neural network operations and batch processing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.generation = 0
        self.best_fitness = -float('inf')
        self.fitness_history = []
        self.training_start_time = None
        
        # GPU/CPU setup
        self.device = torch.device('cuda' if torch.cuda.is_available() and TORCH_AVAILABLE else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize bounty system
        self.bounty_system = ExtendedBountyEvaluator()
        
        # Initialize board evaluator
        self.board_evaluator = BoardEvaluator()
        
        # Feature extractor
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available - GPU acceleration requires PyTorch/CUDA")
        
        self.feature_extractor = GPUChessFeatureExtractor()
        self.feature_extractor.to(self.device)
        
        # Population storage
        self.population = []
        self.population_fitness = []
        
        # Training statistics
        self.stats = {
            'generations_trained': 0,
            'total_games_played': 0,
            'average_fitness_history': [],
            'best_fitness_history': [],
            'training_time': 0.0
        }
    
    def get_game_phase(self, board: chess.Board) -> str:
        """Determine current game phase"""
        # Count remaining pieces
        piece_count = len(board.piece_map())
        
        if piece_count >= 20:
            return "opening"
        elif piece_count >= 10:
            return "middlegame"
        else:
            return "endgame"
    
    def evaluate_board(self, board: chess.Board) -> float:
        """Evaluate board position"""
        return self.board_evaluator.evaluate(board)
    
    def create_random_population(self, population_size: int) -> List[V7P3RGPU_LSTM]:
        """Create initial random population of GPU models"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available - cannot create GPU population")
        
        print(f"Creating random population of {population_size} GPU models...")
        
        population = []
        for i in range(population_size):
            model = V7P3RGPU_LSTM(
                input_size=self.config['model']['input_size'],
                hidden_size=self.config['model']['hidden_size'],
                num_layers=self.config['model']['num_layers'],
                output_size=self.config['model']['output_size'],
                device=self.device
            )
            model.to(self.device)
            population.append(model)
            
            if (i + 1) % 10 == 0:
                print(f"Created {i + 1}/{population_size} models...")
        
        return population
    
    def evaluate_individual_gpu(self, model: V7P3RGPU_LSTM, num_games: int = 5) -> float:
        """Evaluate individual using GPU acceleration"""
        if not TORCH_AVAILABLE:
            print("PyTorch not available, cannot use GPU evaluation")
            return -1000.0
        
        total_score = 0.0
        games_played = 0
        
        with torch.no_grad():
            model.eval()
            
            for _ in range(num_games):
                try:
                    score = self.play_evaluation_game_gpu(model)
                    total_score += score
                    games_played += 1
                except Exception as e:
                    print(f"Game evaluation error: {e}")
                    continue
        
        if games_played == 0:
            return -1000.0
        
        return total_score / games_played
    
    def evaluate_individual_cpu(self, model, num_games: int = 5) -> float:
        """Fallback CPU evaluation for compatibility"""
        total_score = 0.0
        games_played = 0
        
        for _ in range(num_games):
            try:
                score = self.play_evaluation_game_cpu(model)
                total_score += score
                games_played += 1
            except Exception as e:
                print(f"Game evaluation error: {e}")
                continue
        
        if games_played == 0:
            return -1000.0
        
        return total_score / games_played
    
    def play_evaluation_game_gpu(self, model: V7P3RGPU_LSTM) -> float:
        """Play evaluation game using GPU model"""
        board = chess.Board()
        total_reward = 0.0
        move_count = 0
        max_moves = self.config.get('max_moves_per_game', 200)
        
        # Batch process moves for efficiency
        move_features = []
        move_rewards = []
        
        while not board.is_game_over() and move_count < max_moves:
            # Get legal moves
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            
            # Extract current position features
            try:
                features = self.feature_extractor.extract_features(board)
                features_tensor = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                # Get model prediction
                output, _ = model(features_tensor)
                move_values = output.squeeze().cpu().numpy()
                
                # Select best move
                best_move_idx = 0
                best_value = -float('inf')
                
                for i, move in enumerate(legal_moves):
                    move_idx = min(i, len(move_values) - 1)
                    value = move_values[move_idx]
                    
                    if value > best_value:
                        best_value = value
                        best_move_idx = i
                
                chosen_move = legal_moves[best_move_idx]
                
                # Apply move and calculate reward
                board.push(chosen_move)
                
                # Calculate bounty reward
                bounty_score = self.bounty_system.evaluate_move(board, chosen_move)
                bounty_reward = bounty_score.total()
                
                # Calculate position reward
                position_reward = self.evaluate_board(board)
                
                total_reward += bounty_reward + position_reward * 0.1
                move_count += 1
                
            except Exception as e:
                print(f"Move evaluation error: {e}")
                # Random fallback
                chosen_move = random.choice(legal_moves)
                board.push(chosen_move)
                move_count += 1
        
        # Game outcome bonus
        result = board.result()
        if result == "1-0":  # White wins
            total_reward += 100
        elif result == "0-1":  # Black wins
            total_reward -= 100
        elif result == "1/2-1/2":  # Draw
            total_reward += 10
        
        # Encourage longer games (within reason)
        if move_count > 20:
            total_reward += min(move_count - 20, 30)
        
        return total_reward
    
    def evaluate_population_parallel(self, population: List[V7P3RGPU_LSTM], num_games_per_individual: int = 5) -> List[float]:
        """Fallback CPU game evaluation"""
        board = chess.Board()
        total_reward = 0.0
        move_count = 0
        max_moves = self.config.get('max_moves_per_game', 200)
        
        while not board.is_game_over() and move_count < max_moves:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            
            try:
                # Extract features (CPU version)
                features = extract_feature_vector(board)
                features = np.array(features, dtype=np.float32).reshape(1, 1, -1)
                
                # Get model prediction
                output = model.forward(features)
                move_values = output[0, -1, :]  # Last timestep output
                
                # Select best move
                best_move_idx = 0
                best_value = -float('inf')
                
                for i, move in enumerate(legal_moves):
                    move_idx = min(i, len(move_values) - 1)
                    value = move_values[move_idx]
                    
                    if value > best_value:
                        best_value = value
                        best_move_idx = i
                
                chosen_move = legal_moves[best_move_idx]
                board.push(chosen_move)
                
                # Calculate rewards
                bounty_score = self.bounty_system.evaluate_move(board, chosen_move)
                bounty_reward = bounty_score.total()
                position_reward = self.evaluate_board(board)
                
                total_reward += bounty_reward + position_reward * 0.1
                move_count += 1
                
            except Exception as e:
                print(f"Move evaluation error: {e}")
                chosen_move = random.choice(legal_moves)
                board.push(chosen_move)
                move_count += 1
        
        # Game outcome bonus
        result = board.result()
        if result == "1-0":
            total_reward += 100
        elif result == "0-1":
            total_reward -= 100
        elif result == "1/2-1/2":
            total_reward += 10
        
        if move_count > 20:
            total_reward += min(move_count - 20, 30)
        
        return total_reward
    
    def evaluate_population_parallel(self, population: List, num_games_per_individual: int = 5) -> List[float]:
        """Evaluate entire population in parallel"""
        print(f"Evaluating population of {len(population)} individuals...")
        
        fitness_scores = []
        max_workers = min(self.config.get('max_parallel_evaluations', 4), len(population))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit evaluation tasks
            future_to_idx = {}
            for i, individual in enumerate(population):
                if TORCH_AVAILABLE and isinstance(individual, V7P3RGPU_LSTM):
                    future = executor.submit(self.evaluate_individual_gpu, individual, num_games_per_individual)
                else:
                    future = executor.submit(self.evaluate_individual_cpu, individual, num_games_per_individual)
                future_to_idx[future] = i
            
            # Collect results
            fitness_scores = [0.0] * len(population)
            completed = 0
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    fitness = future.result()
                    fitness_scores[idx] = fitness
                    completed += 1
                    
                    if completed % 5 == 0 or completed == len(population):
                        print(f"Evaluated {completed}/{len(population)} individuals...")
                        
                except Exception as e:
                    print(f"Evaluation failed for individual {idx}: {e}")
                    fitness_scores[idx] = -1000.0
        
        return fitness_scores
    
    def selection_tournament(self, population: List, fitness_scores: List[float], tournament_size: int = 3) -> List:
        """Tournament selection for breeding"""
        selected = []
        pop_size = len(population)
        
        for _ in range(pop_size):
            # Select random individuals for tournament
            tournament_indices = random.sample(range(pop_size), min(tournament_size, pop_size))
            
            # Find best individual in tournament
            best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
            selected.append(population[best_idx])
        
        return selected
    
    def crossover_and_mutation(self, parents: List, mutation_rate: float = 0.1) -> List:
        """Create offspring through crossover and mutation"""
        offspring = []
        
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[(i + 1) % len(parents)]
            
            try:
                if TORCH_AVAILABLE and isinstance(parent1, V7P3RGPU_LSTM):
                    # GPU crossover
                    child1 = parent1.crossover(parent2)
                    child2 = parent2.crossover(parent1)
                    
                    # Mutation
                    if random.random() < mutation_rate:
                        child1.mutate()
                    if random.random() < mutation_rate:
                        child2.mutate()
                else:
                    # CPU crossover (simplified)
                    child1 = parent1  # Placeholder - implement CPU crossover if needed
                    child2 = parent2
                
                offspring.extend([child1, child2])
                
            except Exception as e:
                print(f"Crossover error: {e}")
                # Fallback to parents
                offspring.extend([parent1, parent2])
        
        return offspring[:len(parents)]  # Maintain population size
    
    def train_generation(self) -> Dict[str, Any]:
        """Train one generation"""
        generation_start = time.time()
        
        # Evaluate population
        fitness_scores = self.evaluate_population_parallel(
            self.population, 
            self.config.get('games_per_evaluation', 5)
        )
        
        # Update statistics
        avg_fitness = np.mean(fitness_scores)
        max_fitness = np.max(fitness_scores)
        min_fitness = np.min(fitness_scores)
        
        self.fitness_history.append(avg_fitness)
        
        if max_fitness > self.best_fitness:
            self.best_fitness = max_fitness
            best_idx = np.argmax(fitness_scores)
            
            # Save best model
            best_model = self.population[best_idx]
            model_path = f"models/best_gpu_model_gen_{self.generation}.pkl"
            os.makedirs("models", exist_ok=True)
            
            if TORCH_AVAILABLE and isinstance(best_model, V7P3RGPU_LSTM):
                best_model.save_model(model_path.replace('.pkl', '.pth'))
            
            print(f"New best fitness: {max_fitness:.2f} (saved to {model_path})")
        
        # Selection and breeding
        selected_parents = self.selection_tournament(
            self.population, fitness_scores,
            self.config.get('tournament_size', 3)
        )
        
        # Create next generation
        self.population = self.crossover_and_mutation(
            selected_parents, 
            self.config.get('mutation_rate', 0.1)
        )
        
        generation_time = time.time() - generation_start
        
        # Generation report
        report = {
            'generation': self.generation,
            'avg_fitness': avg_fitness,
            'max_fitness': max_fitness,
            'min_fitness': min_fitness,
            'best_fitness_so_far': self.best_fitness,
            'generation_time': generation_time,
            'population_size': len(self.population)
        }
        
        print(f"Generation {self.generation}: Avg={avg_fitness:.2f}, Max={max_fitness:.2f}, Best={self.best_fitness:.2f}, Time={generation_time:.1f}s")
        
        self.generation += 1
        self.stats['generations_trained'] += 1
        self.stats['total_games_played'] += len(self.population) * self.config.get('games_per_evaluation', 5)
        self.stats['average_fitness_history'].append(avg_fitness)
        self.stats['best_fitness_history'].append(max_fitness)
        
        return report
    
    def train(self, num_generations: int) -> Dict[str, Any]:
        """Main training loop"""
        print(f"Starting GPU-accelerated genetic training for {num_generations} generations...")
        print(f"Population size: {self.config['population_size']}")
        print(f"Device: {self.device}")
        
        self.training_start_time = time.time()
        
        # Initialize population if needed
        if not self.population:
            self.population = self.create_random_population(self.config['population_size'])
        
        generation_reports = []
        
        try:
            for gen in range(num_generations):
                report = self.train_generation()
                generation_reports.append(report)
                
                # Save progress periodically
                if (gen + 1) % 10 == 0:
                    self.save_training_progress(f"training_progress_gen_{self.generation-1}.json")
                
                # Early stopping check
                if len(self.fitness_history) >= 20:
                    recent_improvement = max(self.fitness_history[-10:]) - max(self.fitness_history[-20:-10])
                    if recent_improvement < 1.0:
                        print(f"Early stopping: No significant improvement in last 10 generations")
                        break
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        
        training_time = time.time() - self.training_start_time
        self.stats['training_time'] = training_time
        
        # Final report
        final_report = {
            'training_summary': {
                'total_generations': len(generation_reports),
                'total_training_time': training_time,
                'best_fitness_achieved': self.best_fitness,
                'final_avg_fitness': self.fitness_history[-1] if self.fitness_history else 0,
                'device_used': str(self.device)
            },
            'generation_reports': generation_reports,
            'fitness_history': self.fitness_history,
            'config_used': self.config,
            'statistics': self.stats
        }
        
        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"reports/gpu_training_report_{timestamp}.json"
        os.makedirs("reports", exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        print(f"\nTraining completed!")
        print(f"Best fitness achieved: {self.best_fitness:.2f}")
        print(f"Total training time: {training_time:.1f} seconds")
        print(f"Report saved to: {report_path}")
        
        return final_report
    
    def save_training_progress(self, filepath: str):
        """Save current training progress"""
        progress_data = {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'fitness_history': self.fitness_history,
            'stats': self.stats,
            'config': self.config
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    def load_training_progress(self, filepath: str):
        """Load previous training progress"""
        with open(filepath, 'r') as f:
            progress_data = json.load(f)
        
        self.generation = progress_data['generation']
        self.best_fitness = progress_data['best_fitness']
        self.fitness_history = progress_data['fitness_history']
        self.stats = progress_data['stats']
        
        print(f"Loaded training progress from generation {self.generation}")

def main():
    """Test GPU genetic trainer"""
    # Test configuration
    config = {
        'population_size': 20,
        'games_per_evaluation': 3,
        'max_parallel_evaluations': 4,
        'tournament_size': 3,
        'mutation_rate': 0.15,
        'max_moves_per_game': 150,
        'model': {
            'input_size': 816,
            'hidden_size': 128,
            'num_layers': 2,
            'output_size': 64
        },
        'bounty_config': {
            'center_control_weight': 2.0,
            'piece_development_weight': 1.5,
            'king_safety_weight': 3.0,
            'tactical_weight': 4.0
        }
    }
    
    print("Initializing GPU Genetic Trainer...")
    trainer = V7P3RGPUGeneticTrainer(config)
    
    # Quick training test
    print("Running 5-generation test...")
    report = trainer.train(5)
    
    print(f"Test completed. Best fitness: {report['training_summary']['best_fitness_achieved']:.2f}")

if __name__ == "__main__":
    main()
