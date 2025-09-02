"""
Clean GPU-Accelerated Genetic Algorithm Trainer for V7P3R Chess AI v2.0
GPU-only implementation with PyTorch/CUDA acceleration
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
    print("Error: PyTorch not available. GPU acceleration requires PyTorch/CUDA.")
    TORCH_AVAILABLE = False
    raise ImportError("PyTorch required for GPU acceleration")

import chess
import chess.engine
import numpy as np

# Import local modules
from chess_core import BoardEvaluator
from v7p3r_bounty_system import ExtendedBountyEvaluator
from v7p3r_gpu_model import V7P3RGPU_LSTM, GPUChessFeatureExtractor

class V7P3RGPUGeneticTrainer:
    """
    GPU-accelerated genetic algorithm trainer for V7P3R Chess AI.
    Uses PyTorch/CUDA for neural network operations and batch processing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available - GPU acceleration requires PyTorch/CUDA")
        
        self.config = config
        self.generation = 0
        self.best_fitness = -float('inf')
        self.fitness_history = []
        self.training_start_time = None
        
        # GPU setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
        
        # Initialize evaluators
        self.bounty_system = ExtendedBountyEvaluator()
        self.board_evaluator = BoardEvaluator()
        
        # Performance optimizer for move preparation
        from core.performance_optimizer import PerformanceOptimizer
        self.performance_optimizer = PerformanceOptimizer()
        
        # Feature extractor
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
    
    def validate_population_architecture(self) -> None:
        """Ensure all models in population have consistent architecture"""
        if not self.population:
            return
        
        reference_model = self.population[0]
        ref_input = reference_model.input_size
        ref_hidden = reference_model.hidden_size
        ref_layers = reference_model.num_layers
        ref_output = reference_model.output_size
        
        print(f"Validating population architecture: {ref_input}->{ref_hidden}x{ref_layers}->{ref_output}")
        
        # Check for mismatched architectures
        mismatched_indices = []
        for i, model in enumerate(self.population):
            if (model.input_size != ref_input or
                model.hidden_size != ref_hidden or
                model.num_layers != ref_layers or
                model.output_size != ref_output):
                mismatched_indices.append(i)
        
        if mismatched_indices:
            print(f"Found {len(mismatched_indices)} models with mismatched architectures. Recreating...")
            
            # Replace mismatched models with new random models of correct architecture
            for i in mismatched_indices:
                self.population[i] = V7P3RGPU_LSTM(
                    ref_input, ref_hidden, ref_layers, ref_output,
                    device=self.device
                )
            
            print(f"Replaced {len(mismatched_indices)} mismatched models")
    
    def get_game_phase(self, board: chess.Board) -> str:
        """Determine current game phase"""
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
            
            # Ensure optimal memory layout for new models
            model.lstm.flatten_parameters()
            
            population.append(model)
            
            if (i + 1) % 10 == 0:
                print(f"Created {i + 1}/{population_size} models...")
        
        return population
    
    def evaluate_individual_gpu(self, model: V7P3RGPU_LSTM, num_games: int = 5) -> float:
        """Evaluate individual using GPU acceleration"""
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
    
    def play_evaluation_game_gpu(self, model: V7P3RGPU_LSTM) -> float:
        """Play evaluation game using GPU model"""
        board = chess.Board()
        total_reward = 0.0
        move_count = 0
        max_moves = self.config.get('max_moves_per_game', 200)
        
        while not board.is_game_over() and move_count < max_moves:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            
            try:
                # Use performance optimizer for smart move preparation
                try:
                    optimized_moves = self.performance_optimizer.optimize_move_selection(board, time_limit=0.02)
                    if optimized_moves:
                        # Use pre-filtered and ordered moves from performance optimizer
                        candidate_moves = [move for move, score in optimized_moves[:min(10, len(optimized_moves))]]
                    else:
                        candidate_moves = legal_moves
                except:
                    # Fallback if performance optimizer fails
                    candidate_moves = legal_moves
                
                # Enhanced move selection with multiple evaluation methods
                best_move = None
                best_combined_score = -float('inf')
                
                # Extract current position features for model
                features = self.feature_extractor.extract_features(board)
                features_tensor = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
                
                for move in candidate_moves:
                    # Make move on copy to get resulting position
                    board_copy = board.copy()
                    board_copy.push(move)
                    
                    # Get features for resulting position
                    next_features = self.feature_extractor.extract_features(board_copy)
                    next_features_tensor = torch.tensor(next_features, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
                    
                    # Get model prediction for resulting position
                    output, _ = model(next_features_tensor)
                    nn_value = output.squeeze().cpu().item() if output.numel() > 0 else 0.0
                    
                    # Get enhanced bounty evaluation
                    try:
                        bounty_score = self.bounty_system.evaluate_move(board, move)
                        
                        # Get move preparation features for additional context
                        move_features = self.performance_optimizer.get_neural_network_features(board, move)
                        
                        # Combined evaluation with multiple factors (V2.0 enhancement)
                        combined_score = (
                            nn_value * 0.4 +                        # Neural network prediction
                            bounty_score.total() * 0.003 +          # Total bounty (scaled)
                            bounty_score.offensive_total() * 0.002 + # Offensive component
                            bounty_score.defensive_total() * 0.003 + # Defensive component (weighted higher for balance)
                            bounty_score.outcome_total() * 0.004 +   # Outcome-based rewards
                            move_features[:5].sum() * 0.1 if len(move_features) >= 5 else 0  # Move preparation features
                        )
                    except Exception as e:
                        # Fallback to just neural network value
                        combined_score = nn_value
                    
                    if combined_score > best_combined_score:
                        best_combined_score = combined_score
                        best_move = move
                
                # Fallback if no move was selected
                if best_move is None:
                    best_move = legal_moves[0]
                
                chosen_move = best_move
                
                # Calculate comprehensive reward BEFORE making the move
                try:
                    bounty_score = self.bounty_system.evaluate_move(board, chosen_move)
                    
                    # Enhanced reward calculation with all bounty components (V2.0)
                    bounty_reward = (
                        bounty_score.offensive_total() * 0.8 +      # Offensive play
                        bounty_score.defensive_total() * 1.0 +      # Defensive play (important for balance!)
                        bounty_score.outcome_total() * 0.9 +        # Outcome-based rewards
                        bounty_score.center_control * 0.3 +         # Specific strategic components
                        bounty_score.king_safety * 0.5 +
                        bounty_score.tactical_patterns * 0.7 +
                        bounty_score.material_balance * 0.4 +
                        bounty_score.initiative * 0.6
                    )
                    
                except Exception as e:
                    print(f"Enhanced bounty evaluation error: {e}")
                    bounty_reward = 0.0
                
                board.push(chosen_move)
                
                # Calculate position reward
                try:
                    position_reward = self.evaluate_board(board)
                except Exception as e:
                    print(f"Position evaluation error: {e}")
                    position_reward = 0.0
                
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
        """Evaluate entire population in parallel"""
        print(f"Evaluating population of {len(population)} individuals...")
        
        fitness_scores = []
        max_workers = min(self.config.get('max_parallel_evaluations', 4), len(population))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit evaluation tasks
            future_to_idx = {}
            for i, individual in enumerate(population):
                future = executor.submit(self.evaluate_individual_gpu, individual, num_games_per_individual)
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
    
    def selection_tournament(self, population: List[V7P3RGPU_LSTM], fitness_scores: List[float], tournament_size: int = 3) -> List[V7P3RGPU_LSTM]:
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
    
    def crossover_and_mutation(self, parents: List[V7P3RGPU_LSTM], mutation_rate: float = 0.1) -> List[V7P3RGPU_LSTM]:
        """Create offspring through crossover and mutation"""
        offspring = []
        
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[(i + 1) % len(parents)]
            
            try:
                # Check if parents are compatible for crossover
                if (parent1.input_size == parent2.input_size and
                    parent1.hidden_size == parent2.hidden_size and
                    parent1.num_layers == parent2.num_layers and
                    parent1.output_size == parent2.output_size):
                    # GPU crossover with compatible architectures
                    child1 = parent1.crossover(parent2)
                    child2 = parent2.crossover(parent1)
                else:
                    # If architectures are incompatible, create mutated copies
                    child1 = V7P3RGPU_LSTM(
                        parent1.input_size, parent1.hidden_size, parent1.num_layers,
                        parent1.output_size, device=parent1.device
                    )
                    child1.load_state_dict(parent1.state_dict())
                    child1.mutate(mutation_rate=0.15)
                    
                    child2 = V7P3RGPU_LSTM(
                        parent2.input_size, parent2.hidden_size, parent2.num_layers,
                        parent2.output_size, device=parent2.device
                    )
                    child2.load_state_dict(parent2.state_dict())
                    child2.mutate(mutation_rate=0.15)
                
                # Additional mutation
                if random.random() < mutation_rate:
                    child1.mutate()
                if random.random() < mutation_rate:
                    child2.mutate()
                
                offspring.extend([child1, child2])
                
            except Exception as e:
                # More detailed error logging for debugging
                print(f"Crossover error between models with shapes:")
                print(f"  Parent1: input={parent1.input_size}, hidden={parent1.hidden_size}, layers={parent1.num_layers}")
                print(f"  Parent2: input={parent2.input_size}, hidden={parent2.hidden_size}, layers={parent2.num_layers}")
                print(f"  Error: {e}")
                
                # Robust fallback: create copies with slight mutation
                try:
                    child1 = V7P3RGPU_LSTM(
                        parent1.input_size, parent1.hidden_size, parent1.num_layers,
                        parent1.output_size, device=parent1.device
                    )
                    child1.load_state_dict(parent1.state_dict())
                    child1.mutate(mutation_rate=0.1)
                    
                    child2 = V7P3RGPU_LSTM(
                        parent2.input_size, parent2.hidden_size, parent2.num_layers,
                        parent2.output_size, device=parent2.device
                    )
                    child2.load_state_dict(parent2.state_dict())
                    child2.mutate(mutation_rate=0.1)
                    
                    offspring.extend([child1, child2])
                except Exception as e2:
                    print(f"Fallback creation failed: {e2}, using original parents")
                    offspring.extend([parent1, parent2])
        
        return offspring[:len(parents)]  # Maintain population size
    
    def compact_population_memory(self):
        """Ensure all models in population have optimal memory layout"""
        for model in self.population:
            if hasattr(model, 'lstm'):
                model.lstm.flatten_parameters()
    
    def train_generation(self) -> Dict[str, Any]:
        """Train one generation"""
        generation_start = time.time()
        
        # Validate population architecture consistency
        self.validate_population_architecture()
        
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
            model_path = f"models/best_gpu_model_gen_{self.generation}.pth"
            os.makedirs("models", exist_ok=True)
            
            best_model.save_model(model_path)
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
        
        # Compact memory layout for optimal performance
        self.compact_population_memory()
        
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
                    self.save_training_progress(f"reports/training_progress_gen_{self.generation-1}.json")
                
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
        
        # Ensure directory exists
        dir_path = os.path.dirname(filepath)
        if dir_path:  # Only create if there's a directory path
            os.makedirs(dir_path, exist_ok=True)
        
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
        'population_size': 4,
        'games_per_evaluation': 1,
        'max_parallel_evaluations': 2,
        'tournament_size': 2,
        'mutation_rate': 0.15,
        'max_moves_per_game': 20,
        'model': {
            'input_size': 816,
            'hidden_size': 32,
            'num_layers': 1,
            'output_size': 16
        }
    }
    
    print("Initializing GPU Genetic Trainer...")
    trainer = V7P3RGPUGeneticTrainer(config)
    
    # Quick training test
    print("Running 1-generation test...")
    report = trainer.train(1)
    
    print(f"Test completed. Best fitness: {report['training_summary']['best_fitness_achieved']:.2f}")

if __name__ == "__main__":
    main()
