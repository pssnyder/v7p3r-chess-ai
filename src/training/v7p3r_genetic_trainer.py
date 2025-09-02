# v7p3r_genetic_trainer.py
"""
V7P3R Chess AI 2.0 - Genetic Algorithm Training System
Trains the RNN using genetic algorithms with 128 parallel games and bounty-based fitness.
"""

import chess
import chess.engine
import numpy as np
import random
import time
import json
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import pickle
from pathlib import Path

from v7p3r_rnn_model import V7P3RNeuralNetwork, ChessFeatureExtractor
from v7p3r_bounty_system import BountyEvaluator, ExtendedBountyEvaluator, BountyScore


@dataclass
class GeneticConfig:
    """Configuration for genetic algorithm training"""
    population_size: int = 128
    generations: int = 1000
    mutation_rate: float = 0.15
    mutation_strength: float = 0.02
    crossover_rate: float = 0.7
    elite_percentage: float = 0.1
    tournament_size: int = 5
    games_per_individual: int = 3
    max_moves_per_game: int = 200
    parallel_workers: int = 8
    save_frequency: int = 10
    extended_bounty: bool = True


@dataclass
class IndividualStats:
    """Statistics for an individual AI"""
    genome_id: int
    generation: int
    total_bounty: float = 0.0
    games_played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    average_game_length: float = 0.0
    tactical_score: float = 0.0
    positional_score: float = 0.0
    
    def fitness(self) -> float:
        """Calculate fitness score"""
        if self.games_played == 0:
            return 0.0
        
        # Primary fitness is bounty per game
        base_fitness = self.total_bounty / self.games_played
        
        # Bonus for winning
        win_rate = self.wins / self.games_played
        base_fitness += win_rate * 50.0
        
        # Small penalty for very long games (encourage decisive play)
        if self.average_game_length > 100:
            base_fitness -= (self.average_game_length - 100) * 0.1
        
        return base_fitness


class V7P3RGeneticAI:
    """AI individual for genetic algorithm training"""
    
    def __init__(self, network: V7P3RNeuralNetwork, genome_id: int, generation: int):
        self.network = network
        self.genome_id = genome_id
        self.generation = generation
        self.feature_extractor = ChessFeatureExtractor()
        self.bounty_evaluator = ExtendedBountyEvaluator()
        self.stats = IndividualStats(genome_id, generation)
        self.move_history: List[chess.Move] = []
    
    def get_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Get best move using RNN + bounty evaluation"""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        best_move = None
        best_score = float('-inf')
        
        # Extract features for current position
        features = self.feature_extractor.extract_features(board, self.move_history)
        
        # Evaluate each legal move
        for move in legal_moves:
            # Get RNN evaluation
            temp_board = board.copy()
            temp_board.push(move)
            temp_features = self.feature_extractor.extract_features(temp_board, self.move_history + [move])
            rnn_score = self.network.forward(temp_features)
            
            # Get bounty evaluation
            bounty_score = self.bounty_evaluator.evaluate_move(board, move)
            
            # Combine scores (70% RNN, 30% bounty)
            combined_score = 0.7 * rnn_score + 0.3 * bounty_score.total()
            
            if combined_score > best_score:
                best_score = combined_score
                best_move = move
        
        if best_move:
            self.move_history.append(best_move)
            # Keep history manageable
            if len(self.move_history) > 64:
                self.move_history.pop(0)
        
        return best_move
    
    def reset_for_new_game(self):
        """Reset AI state for a new game"""
        self.network.reset_memory()
        self.move_history = []


def play_game(ai1: V7P3RGeneticAI, ai2: V7P3RGeneticAI, max_moves: int = 200) -> Tuple[str, List[BountyScore], List[BountyScore], int]:
    """Play a single game between two AIs"""
    board = chess.Board()
    ai1.reset_for_new_game()
    ai2.reset_for_new_game()
    
    ai1_bounty_scores = []
    ai2_bounty_scores = []
    moves_played = 0
    
    while not board.is_game_over() and moves_played < max_moves:
        current_ai = ai1 if board.turn == chess.WHITE else ai2
        
        move = current_ai.get_move(board)
        if not move:
            break
        
        # Calculate bounty for this move
        if board.turn == chess.WHITE:
            bounty = ai1.bounty_evaluator.evaluate_move(board, move)
            ai1_bounty_scores.append(bounty)
        else:
            bounty = ai2.bounty_evaluator.evaluate_move(board, move)
            ai2_bounty_scores.append(bounty)
        
        board.push(move)
        moves_played += 1
    
    # Determine result
    if board.is_checkmate():
        result = "1-0" if board.turn == chess.BLACK else "0-1"
    elif board.is_stalemate() or board.is_insufficient_material() or moves_played >= max_moves:
        result = "1/2-1/2"
    else:
        result = "1/2-1/2"
    
    return result, ai1_bounty_scores, ai2_bounty_scores, moves_played


def evaluate_individual(args: Tuple[bytes, int, int, GeneticConfig]) -> IndividualStats:
    """Evaluate a single individual by playing games"""
    network_data, genome_id, generation, config = args
    
    # Reconstruct network from serialized data
    network = pickle.loads(network_data)
    ai = V7P3RGeneticAI(network, genome_id, generation)
    
    # Create a simple opponent (random player for now)
    opponent_network = V7P3RNeuralNetwork()
    opponent = V7P3RGeneticAI(opponent_network, -1, generation)
    
    total_bounty = 0.0
    total_moves = 0
    wins = 0
    draws = 0
    losses = 0
    
    # Play multiple games
    for game_num in range(config.games_per_individual):
        # Alternate colors
        if game_num % 2 == 0:
            result, ai_bounty, opp_bounty, moves = play_game(ai, opponent, config.max_moves_per_game)
            ai_total_bounty = sum(score.total() for score in ai_bounty)
        else:
            result, opp_bounty, ai_bounty, moves = play_game(opponent, ai, config.max_moves_per_game)
            ai_total_bounty = sum(score.total() for score in ai_bounty)
        
        total_bounty += ai_total_bounty
        total_moves += moves
        
        # Update win/loss record
        if (game_num % 2 == 0 and result == "1-0") or (game_num % 2 == 1 and result == "0-1"):
            wins += 1
        elif result == "1/2-1/2":
            draws += 1
        else:
            losses += 1
    
    # Update stats
    ai.stats.total_bounty = total_bounty
    ai.stats.games_played = config.games_per_individual
    ai.stats.wins = wins
    ai.stats.draws = draws
    ai.stats.losses = losses
    ai.stats.average_game_length = total_moves / config.games_per_individual if config.games_per_individual > 0 else 0
    
    return ai.stats


class GeneticTrainer:
    """Main genetic algorithm trainer for V7P3R 2.0"""
    
    def __init__(self, config: GeneticConfig):
        self.config = config
        self.population: List[V7P3RNeuralNetwork] = []
        self.generation = 0
        self.best_fitness_history: List[float] = []
        self.average_fitness_history: List[float] = []
        
        # Create directories
        self.models_dir = Path("models/genetic")
        self.logs_dir = Path("logs/genetic") 
        self.reports_dir = Path("reports/genetic")
        
        for dir_path in [self.models_dir, self.logs_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def initialize_population(self):
        """Initialize the population with random networks"""
        print(f"Initializing population of {self.config.population_size} individuals...")
        
        self.population = []
        for i in range(self.config.population_size):
            network = V7P3RNeuralNetwork()
            self.population.append(network)
        
        print("Population initialized!")
    
    def evaluate_population(self) -> List[IndividualStats]:
        """Evaluate entire population using parallel processing"""
        print(f"Evaluating generation {self.generation}...")
        
        # Prepare arguments for parallel evaluation
        eval_args = []
        for i, network in enumerate(self.population):
            network_data = pickle.dumps(network)
            eval_args.append((network_data, i, self.generation, self.config))
        
        # Evaluate in parallel
        stats = []
        with ProcessPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            futures = [executor.submit(evaluate_individual, args) for args in eval_args]
            
            for i, future in enumerate(as_completed(futures)):
                try:
                    individual_stats = future.result()
                    stats.append(individual_stats)
                    
                    if i % 10 == 0:
                        print(f"Evaluated {i+1}/{len(futures)} individuals...")
                        
                except Exception as e:
                    print(f"Error evaluating individual: {e}")
                    # Create dummy stats for failed individuals
                    dummy_stats = IndividualStats(i, self.generation)
                    stats.append(dummy_stats)
        
        # Sort by fitness
        stats.sort(key=lambda x: x.fitness(), reverse=True)
        
        return stats
    
    def selection_tournament(self, stats: List[IndividualStats]) -> List[int]:
        """Tournament selection for breeding"""
        selected_indices = []
        
        for _ in range(self.config.population_size):
            # Select random individuals for tournament
            tournament = random.sample(range(len(stats)), min(self.config.tournament_size, len(stats)))
            
            # Find best in tournament
            best_idx = min(tournament, key=lambda i: -stats[i].fitness())
            selected_indices.append(best_idx)
        
        return selected_indices
    
    def create_next_generation(self, stats: List[IndividualStats]) -> List[V7P3RNeuralNetwork]:
        """Create next generation through selection, crossover, and mutation"""
        next_generation = []
        
        # Elitism - keep best individuals
        elite_count = int(self.config.population_size * self.config.elite_percentage)
        for i in range(elite_count):
            genome_idx = stats[i].genome_id
            if genome_idx < len(self.population):
                next_generation.append(self.population[genome_idx])
        
        # Fill rest with offspring
        selected_indices = self.selection_tournament(stats)
        
        while len(next_generation) < self.config.population_size:
            # Select parents
            parent1_idx = selected_indices[random.randint(0, len(selected_indices) - 1)]
            parent2_idx = selected_indices[random.randint(0, len(selected_indices) - 1)]
            
            if parent1_idx >= len(self.population) or parent2_idx >= len(self.population):
                continue
                
            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]
            
            # Crossover
            if random.random() < self.config.crossover_rate:
                child = parent1.crossover(parent2)
            else:
                child = parent1 if random.random() < 0.5 else parent2
            
            # Mutation
            child = child.mutate(self.config.mutation_rate, self.config.mutation_strength)
            
            next_generation.append(child)
        
        return next_generation[:self.config.population_size]
    
    def train(self):
        """Main training loop"""
        print("Starting V7P3R 2.0 Genetic Training!")
        print(f"Config: {asdict(self.config)}")
        
        # Initialize population
        self.initialize_population()
        
        for generation in range(self.config.generations):
            self.generation = generation
            start_time = time.time()
            
            print(f"\n=== Generation {generation + 1}/{self.config.generations} ===")
            
            # Evaluate population
            stats = self.evaluate_population()
            
            # Track fitness statistics
            fitnesses = [stat.fitness() for stat in stats]
            best_fitness = max(fitnesses)
            avg_fitness = sum(fitnesses) / len(fitnesses)
            
            self.best_fitness_history.append(best_fitness)
            self.average_fitness_history.append(avg_fitness)
            
            # Print statistics
            elapsed = time.time() - start_time
            print(f"Best fitness: {best_fitness:.2f}")
            print(f"Average fitness: {avg_fitness:.2f}")
            print(f"Generation time: {elapsed:.1f}s")
            
            # Save best individual
            if generation % self.config.save_frequency == 0:
                self.save_generation(generation, stats)
            
            # Create next generation
            if generation < self.config.generations - 1:
                self.population = self.create_next_generation(stats)
                print(f"Created next generation with {len(self.population)} individuals")
        
        # Save final results
        self.save_final_results(stats)
        print("Training completed!")
    
    def save_generation(self, generation: int, stats: List[IndividualStats]):
        """Save generation results"""
        # Save best network
        best_stats = stats[0]
        if best_stats.genome_id < len(self.population):
            best_network = self.population[best_stats.genome_id]
            model_path = self.models_dir / f"best_gen_{generation:04d}.json"
            best_network.save_model(str(model_path))
        
        # Save statistics
        stats_data = {
            'generation': generation,
            'best_fitness': best_stats.fitness(),
            'average_fitness': sum(stat.fitness() for stat in stats) / len(stats),
            'population_stats': [asdict(stat) for stat in stats[:10]]  # Top 10
        }
        
        stats_path = self.logs_dir / f"gen_{generation:04d}_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats_data, f, indent=2)
    
    def save_final_results(self, final_stats: List[IndividualStats]):
        """Save final training results"""
        # Save best network
        best_network = self.population[final_stats[0].genome_id]
        best_network.save_model(str(self.models_dir / "v7p3r_2.0_final.json"))
        
        # Save training summary
        summary = {
            'total_generations': self.generation + 1,
            'final_best_fitness': final_stats[0].fitness(),
            'best_fitness_history': self.best_fitness_history,
            'average_fitness_history': self.average_fitness_history,
            'config': asdict(self.config),
            'final_population_stats': [asdict(stat) for stat in final_stats]
        }
        
        with open(self.reports_dir / "training_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Final best fitness: {final_stats[0].fitness():.2f}")
        print(f"Best network saved to: {self.models_dir / 'v7p3r_2.0_final.json'}")


def main():
    """Main training entry point"""
    # Default genetic algorithm configuration
    config = GeneticConfig(
        population_size=128,
        generations=100,  # Start with fewer generations for testing
        mutation_rate=0.15,
        mutation_strength=0.02,
        crossover_rate=0.7,
        elite_percentage=0.1,
        tournament_size=5,
        games_per_individual=2,  # Reduced for faster training
        max_moves_per_game=150,
        parallel_workers=min(8, mp.cpu_count()),
        save_frequency=10,
        extended_bounty=True
    )
    
    trainer = GeneticTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
