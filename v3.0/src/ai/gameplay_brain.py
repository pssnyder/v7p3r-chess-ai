"""
V7P3R AI v3.0 - "Gameplay Brain" Genetic Algorithm
==================================================

The tactical validation system that takes move candidates from the Thinking Brain
and simulates short game continuations to select the optimal move for the current
position. Pure objective evaluation - no human chess knowledge.

Architecture:
- Real-time genetic algorithm
- Population: Top-N moves from Thinking Brain  
- Fitness: Tactical simulation results
- Selection: Best move for actual play
"""

import chess
import chess.engine
import numpy as np
import random
import time
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Import our systems
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from core.chess_state import ChessStateExtractor
from core.neural_features import NeuralFeatureConverter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MoveCandidate:
    """Individual move candidate in the genetic algorithm population"""
    move: chess.Move
    fitness: float = 0.0
    simulation_depth: int = 0
    tactical_score: float = 0.0
    material_delta: float = 0.0
    king_safety_delta: float = 0.0
    mobility_delta: float = 0.0
    
    def __hash__(self):
        return hash(str(self.move))
    
    def __eq__(self, other):
        return isinstance(other, MoveCandidate) and self.move == other.move


class GameplayBrain:
    """
    Genetic Algorithm for real-time tactical move validation
    
    Takes move candidates from the Thinking Brain and simulates
    short game continuations to find the tactically superior move.
    """
    
    def __init__(
        self,
        population_size: int = 20,          # GA population size
        simulation_depth: int = 4,          # Plies to simulate ahead
        generations: int = 10,              # GA generations per move
        mutation_rate: float = 0.3,         # Chance to mutate candidates
        crossover_rate: float = 0.7,        # Chance to crossover candidates
        time_limit: float = 2.0,            # Max time per move (seconds)
        parallel_simulations: int = 4       # Parallel simulation threads
    ):
        self.population_size = population_size
        self.simulation_depth = simulation_depth
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.time_limit = time_limit
        self.parallel_simulations = parallel_simulations
        
        # Systems for evaluation
        self.state_extractor = ChessStateExtractor()
        self.feature_converter = NeuralFeatureConverter()
        
        # Performance tracking
        self.move_times = []
        self.simulation_counts = []
        
        logger.info(f"GameplayBrain initialized - Population: {population_size}, Depth: {simulation_depth}")
    
    def select_best_move(
        self,
        board: chess.Board,
        thinking_brain_candidates: List[chess.Move],
        thinking_brain_probabilities: Optional[List[float]] = None
    ) -> Tuple[chess.Move, Dict[str, Any]]:
        """
        Use genetic algorithm to select the best move from candidates
        
        Args:
            board: Current chess position
            thinking_brain_candidates: Move candidates from Thinking Brain
            thinking_brain_probabilities: Candidate probabilities (optional)
            
        Returns:
            best_move: Selected move to play
            analysis: Detailed analysis data
        """
        start_time = time.time()
        
        # Initialize population from Thinking Brain candidates
        population = self._initialize_population(
            board, thinking_brain_candidates, thinking_brain_probabilities
        )
        
        # Track best candidate across generations
        best_candidate = None
        generation_stats = []
        
        # Evolve population through generations
        for generation in range(self.generations):
            # Check time limit
            if time.time() - start_time > self.time_limit:
                logger.debug(f"Time limit reached at generation {generation}")
                break
            
            # Evaluate fitness for all candidates
            population = self._evaluate_population_fitness(board, population)
            
            # Track statistics
            fitness_scores = [candidate.fitness for candidate in population]
            generation_stats.append({
                'generation': generation,
                'best_fitness': max(fitness_scores),
                'avg_fitness': np.mean(fitness_scores),
                'population_size': len(population)
            })
            
            # Update best candidate
            current_best = max(population, key=lambda x: x.fitness)
            if best_candidate is None or current_best.fitness > best_candidate.fitness:
                best_candidate = current_best
            
            # Early stopping if we have a dominant solution
            if current_best.fitness > 0.9:  # Very strong tactical advantage
                logger.debug(f"Early stopping at generation {generation} - dominant solution found")
                break
            
            # Evolve population for next generation
            if generation < self.generations - 1:  # Don't evolve on last generation
                population = self._evolve_population(board, population)
        
        # Calculate final metrics
        total_time = time.time() - start_time
        self.move_times.append(total_time)
        
        # Prepare analysis data
        analysis = {
            'best_candidate': best_candidate,
            'total_time': total_time,
            'generations_completed': len(generation_stats),
            'generation_stats': generation_stats,
            'final_population_size': len(population),
            'simulations_performed': sum(self.simulation_counts[-10:]) if self.simulation_counts else 0
        }
        
        selected_move = best_candidate.move if best_candidate else thinking_brain_candidates[0]
        
        logger.info(f"Selected move: {selected_move} (fitness: {best_candidate.fitness:.3f}, time: {total_time:.2f}s)")
        
        return selected_move, analysis
    
    def _initialize_population(
        self,
        board: chess.Board,
        candidates: List[chess.Move],
        probabilities: Optional[List[float]] = None
    ) -> List[MoveCandidate]:
        """Initialize GA population from Thinking Brain candidates"""
        population = []
        
        # Add all Thinking Brain candidates
        for i, move in enumerate(candidates):
            if len(population) >= self.population_size:
                break
            
            candidate = MoveCandidate(
                move=move,
                fitness=probabilities[i] if probabilities else 0.5  # Use thinking brain probability as initial fitness
            )
            population.append(candidate)
        
        # Fill remaining population with random legal moves if needed
        legal_moves = list(board.legal_moves)
        while len(population) < self.population_size and legal_moves:
            random_move = random.choice(legal_moves)
            
            # Avoid duplicates
            if not any(candidate.move == random_move for candidate in population):
                candidate = MoveCandidate(move=random_move, fitness=0.1)  # Lower initial fitness
                population.append(candidate)
            
            # Remove to avoid infinite loop
            legal_moves.remove(random_move)
        
        logger.debug(f"Initialized population with {len(population)} candidates")
        return population
    
    def _evaluate_population_fitness(
        self,
        board: chess.Board,
        population: List[MoveCandidate]
    ) -> List[MoveCandidate]:
        """Evaluate fitness for all candidates in population"""
        
        # Use parallel simulation for speed
        if self.parallel_simulations > 1:
            return self._evaluate_population_parallel(board, population)
        else:
            return self._evaluate_population_sequential(board, population)
    
    def _evaluate_population_sequential(
        self,
        board: chess.Board,
        population: List[MoveCandidate]
    ) -> List[MoveCandidate]:
        """Sequential fitness evaluation"""
        simulation_count = 0
        
        for candidate in population:
            candidate.fitness = self._evaluate_move_fitness(board, candidate.move)
            simulation_count += 1
        
        self.simulation_counts.append(simulation_count)
        return population
    
    def _evaluate_population_parallel(
        self,
        board: chess.Board,
        population: List[MoveCandidate]
    ) -> List[MoveCandidate]:
        """Sequential fitness evaluation for stability (temporarily disabled parallel)"""
        simulation_count = 0
        
        # Use sequential evaluation for stability
        for candidate in population:
            try:
                candidate.fitness = self._evaluate_move_fitness(board.copy(), candidate.move)
                simulation_count += 1
            except Exception as e:
                logger.warning(f"Simulation failed for {candidate.move}: {e}")
                candidate.fitness = 0.0
        
        self.simulation_counts.append(simulation_count)
        return population
    
    def _evaluate_move_fitness(self, board: chess.Board, move: chess.Move) -> float:
        """
        Evaluate fitness of a single move through tactical simulation
        
        Fitness based on objective game state improvements after simulation
        """
        try:
            # Get baseline state
            baseline_state = self.state_extractor.extract_state(board)
            baseline_features = self._extract_evaluation_features(baseline_state)
            
            # Make the move
            board_copy = board.copy()
            board_copy.push(move)
            
            # Check for immediate tactical outcomes
            immediate_score = self._evaluate_immediate_tactics(board_copy)
            
            # Simulate ahead to evaluate position
            simulated_score = self._simulate_position(board_copy, self.simulation_depth)
            
            # Combine scores
            total_fitness = immediate_score * 0.6 + simulated_score * 0.4
            
            return max(0.0, min(1.0, total_fitness))  # Clamp to [0,1]
            
        except Exception as e:
            logger.warning(f"Move evaluation failed for {move}: {e}")
            return 0.0
    
    def _evaluate_immediate_tactics(self, board: chess.Board) -> float:
        """Evaluate immediate tactical consequences of a move"""
        score = 0.5  # Neutral baseline
        
        # Check for game-ending moves
        if board.is_checkmate():
            return 1.0 if board.turn != board.fen().split()[1] else 0.0
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.5  # Draw
        
        # Extract current state for analysis
        current_state = self.state_extractor.extract_state(board)
        
        # Material advantage
        material_balance = current_state.board_features.materialBalance
        material_diff = material_balance.get('difference', 0)
        
        # Adjust for side to move
        if board.turn == chess.BLACK:
            material_diff = -material_diff
        
        # Normalize material advantage
        material_score = (material_diff + 10) / 20.0  # Range [-10,10] -> [0,1]
        score += (material_score - 0.5) * 0.3
        
        # King safety
        if current_state.board_features.isKingInCheck:
            if board.turn == chess.WHITE:  # Black king in check
                score += 0.1
            else:  # White king in check
                score -= 0.1
        
        # Mobility advantage
        mobility = current_state.board_features.mobility
        white_mobility = mobility.get('totalWhite', 0)
        black_mobility = mobility.get('totalBlack', 0)
        
        if white_mobility + black_mobility > 0:
            mobility_ratio = white_mobility / (white_mobility + black_mobility)
            if board.turn == chess.BLACK:
                mobility_ratio = 1.0 - mobility_ratio
            score += (mobility_ratio - 0.5) * 0.2
        
        return max(0.0, min(1.0, score))
    
    def _simulate_position(self, board: chess.Board, depth: int) -> float:
        """Simulate position forward to evaluate longer-term consequences"""
        if depth <= 0 or board.is_game_over():
            return self._evaluate_immediate_tactics(board)
        
        # Get all legal moves
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return 0.5  # Stalemate
        
        # Sample moves for simulation (to keep computation reasonable)
        max_simulation_moves = min(5, len(legal_moves))
        sample_moves = random.sample(legal_moves, max_simulation_moves)
        
        scores = []
        for move in sample_moves:
            try:
                board_copy = board.copy()
                board_copy.push(move)
                future_score = self._simulate_position(board_copy, depth - 1)
                scores.append(future_score)
            except:
                continue
        
        if not scores:
            return 0.5
        
        # Return average score from simulations
        return np.mean(scores)
    
    def _extract_evaluation_features(self, chess_state) -> Dict[str, float]:
        """Extract key features for position evaluation"""
        features = {}
        
        # Material balance
        material = chess_state.board_features.materialBalance
        features['material'] = material.get('difference', 0) / 10.0  # Normalize
        
        # King safety
        features['king_safety'] = 1.0 if not chess_state.board_features.isKingInCheck else 0.0
        
        # Mobility
        mobility = chess_state.board_features.mobility
        total_mobility = mobility.get('totalWhite', 0) + mobility.get('totalBlack', 0)
        features['mobility'] = total_mobility / 50.0  # Normalize
        
        # Game phase
        features['game_phase'] = chess_state.board_features.gamePhase
        
        return features
    
    def _evolve_population(
        self,
        board: chess.Board,
        population: List[MoveCandidate]
    ) -> List[MoveCandidate]:
        """Evolve population through selection, crossover, and mutation"""
        
        # Sort population by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Selection: Keep top performers
        elite_size = max(2, len(population) // 4)
        next_generation = population[:elite_size].copy()
        
        # Generate offspring through crossover and mutation
        legal_moves = list(board.legal_moves)
        
        while len(next_generation) < self.population_size and legal_moves:
            if random.random() < self.crossover_rate and len(population) >= 2:
                # Crossover: Combine features of two parents
                parent_pool_size = max(2, len(population) // 2)
                if parent_pool_size >= 2:
                    parent1, parent2 = random.sample(population[:parent_pool_size], 2)
                    offspring = self._crossover(parent1, parent2, legal_moves)
                    if offspring:
                        next_generation.append(offspring)
            
            if len(next_generation) < self.population_size and random.random() < self.mutation_rate:
                # Mutation: Random legal move
                random_move = random.choice(legal_moves)
                if not any(candidate.move == random_move for candidate in next_generation):
                    mutant = MoveCandidate(move=random_move, fitness=0.0)
                    next_generation.append(mutant)
        
        return next_generation[:self.population_size]
    
    def _crossover(
        self,
        parent1: MoveCandidate,
        parent2: MoveCandidate,
        legal_moves: List[chess.Move]
    ) -> Optional[MoveCandidate]:
        """Create offspring through crossover of two parent candidates"""
        
        # Simple crossover: Pick a legal move that's similar to parents
        # In chess, "similarity" can be same from square, same to square, etc.
        
        candidate_moves = []
        
        # Moves with same from/to squares as parents
        for move in legal_moves:
            if (move.from_square == parent1.move.from_square or 
                move.from_square == parent2.move.from_square or
                move.to_square == parent1.move.to_square or
                move.to_square == parent2.move.to_square):
                candidate_moves.append(move)
        
        if candidate_moves:
            selected_move = random.choice(candidate_moves)
            # Inherit average fitness
            avg_fitness = (parent1.fitness + parent2.fitness) / 2.0
            return MoveCandidate(move=selected_move, fitness=avg_fitness)
        
        return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the Gameplay Brain"""
        if not self.move_times:
            return {}
        
        return {
            'avg_move_time': np.mean(self.move_times),
            'max_move_time': np.max(self.move_times),
            'min_move_time': np.min(self.move_times),
            'total_moves_analyzed': len(self.move_times),
            'avg_simulations_per_move': np.mean(self.simulation_counts) if self.simulation_counts else 0
        }


def test_gameplay_brain():
    """Test the Gameplay Brain genetic algorithm"""
    logger.info("Testing Gameplay Brain...")
    
    # Create test position
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))  # Simple opening move
    
    # Create Gameplay Brain
    gameplay_brain = GameplayBrain(
        population_size=10,  # Small for testing
        simulation_depth=2,   # Shallow for speed
        generations=5,        # Few generations
        time_limit=1.0       # Quick test
    )
    
    # Test move candidates (simulate Thinking Brain output)
    test_candidates = [
        chess.Move.from_uci("d2d4"),
        chess.Move.from_uci("g1f3"),
        chess.Move.from_uci("b1c3"),
        chess.Move.from_uci("f1c4")
    ]
    
    # Select best move
    start_time = time.time()
    best_move, analysis = gameplay_brain.select_best_move(board, test_candidates)
    test_time = time.time() - start_time
    
    logger.info(f"Selected move: {best_move}")
    logger.info(f"Test completed in {test_time:.2f}s")
    logger.info(f"Generations completed: {analysis['generations_completed']}")
    logger.info(f"Final fitness: {analysis['best_candidate'].fitness:.3f}")
    
    # Performance stats
    stats = gameplay_brain.get_performance_stats()
    logger.info(f"Performance: {stats}")
    
    logger.info("✅ Gameplay Brain test completed!")
    return best_move, analysis


if __name__ == "__main__":
    test_gameplay_brain()
