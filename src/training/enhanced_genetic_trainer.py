#!/usr/bin/env python3
"""
Enhanced Genetic Algorithm with Multi-Opponent Training
Implements your video game approach to chess AI training
"""

import torch
import torch.nn as nn
import numpy as np
import random
import chess
import time
from typing import List, Dict, Optional, Tuple
import sys
import os
import json
from pathlib import Path

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.v7p3r_gpu_model import V7P3RGPU_LSTM, GPUChessFeatureExtractor
from evaluation.v7p3r_bounty_system import ExtendedBountyEvaluator
from training.multi_opponent_system import OpponentManager
from core.chess_core import BoardEvaluator

class GameLikeEvaluator:
    """Game-like evaluation system treating chess pieces like video game characters"""
    
    def __init__(self):
        self.piece_roles = {
            chess.PAWN: "Soldier",
            chess.KNIGHT: "Cavalry", 
            chess.BISHOP: "Archer",
            chess.ROOK: "Tank",
            chess.QUEEN: "Hero",
            chess.KING: "Commander"
        }
        
    def evaluate_piece_performance(self, board: chess.Board, color: chess.Color) -> Dict[str, float]:
        """Evaluate how well each piece type is performing their 'role'"""
        scores = {}
        
        # Knight mobility evaluation (your knight example)
        scores['knight_mobility'] = self._evaluate_knight_mobility(board, color)
        
        # Bishop diagonal control
        scores['bishop_vision'] = self._evaluate_bishop_vision(board, color)
        
        # Rook file/rank control
        scores['rook_power'] = self._evaluate_rook_power(board, color)
        
        # Pawn formation and advancement
        scores['pawn_army'] = self._evaluate_pawn_army(board, color)
        
        # Queen activity and safety
        scores['queen_dominance'] = self._evaluate_queen_dominance(board, color)
        
        # King safety and activity
        scores['king_leadership'] = self._evaluate_king_leadership(board, color)
        
        return scores
        
    def _evaluate_knight_mobility(self, board: chess.Board, color: chess.Color) -> float:
        """Knights should control maximum squares (your example)"""
        score = 0.0
        knights = board.pieces(chess.KNIGHT, color)
        
        for knight_square in knights:
            # Maximum knight attacks = 8, but edge/corner positions limit this
            possible_attacks = len(list(board.attacks(knight_square)))
            max_possible = 8
            
            # Penalty for limiting knight's power
            mobility_ratio = possible_attacks / max_possible
            score += mobility_ratio * 10  # Base mobility bonus
            
            # Extra bonus for central knights (more "battlefield control")
            center_distance = min(
                abs(chess.square_file(knight_square) - 3.5),
                abs(chess.square_rank(knight_square) - 3.5)
            )
            centralization_bonus = max(0, 3 - center_distance) * 2
            score += centralization_bonus
            
            # Penalty for edge/corner placement (limiting potential)
            if knight_square in [chess.A1, chess.A8, chess.H1, chess.H8]:
                score -= 5  # Corner penalty
            elif chess.square_file(knight_square) in [0, 7] or chess.square_rank(knight_square) in [0, 7]:
                score -= 2  # Edge penalty
                
        return score
        
    def _evaluate_bishop_vision(self, board: chess.Board, color: chess.Color) -> float:
        """Bishops should have clear diagonal sight lines"""
        score = 0.0
        bishops = board.pieces(chess.BISHOP, color)
        
        for bishop_square in bishops:
            # Count unobstructed diagonal squares
            attacks = len(list(board.attacks(bishop_square)))
            score += attacks * 0.5
            
            # Bonus for long diagonals
            diagonal_length = min(
                chess.square_file(bishop_square),
                chess.square_rank(bishop_square),
                7 - chess.square_file(bishop_square),
                7 - chess.square_rank(bishop_square)
            )
            score += diagonal_length
            
        # Penalty for having only one bishop (poor color coverage)
        if len(bishops) == 1:
            score -= 3
            
        return score
        
    def _evaluate_rook_power(self, board: chess.Board, color: chess.Color) -> float:
        """Rooks should control files and ranks"""
        score = 0.0
        rooks = board.pieces(chess.ROOK, color)
        
        for rook_square in rooks:
            # File and rank control
            file_attacks = 0
            rank_attacks = 0
            
            for square in chess.SQUARES:
                if chess.square_file(square) == chess.square_file(rook_square):
                    if board.is_attacked_by(color, square):
                        file_attacks += 1
                if chess.square_rank(square) == chess.square_rank(rook_square):
                    if board.is_attacked_by(color, square):
                        rank_attacks += 1
                        
            score += (file_attacks + rank_attacks) * 0.3
            
            # Bonus for 7th rank (opponent's territory)
            if color == chess.WHITE and chess.square_rank(rook_square) == 6:
                score += 5
            elif color == chess.BLACK and chess.square_rank(rook_square) == 1:
                score += 5
                
        # Bonus for rook coordination (seeing each other)
        if len(rooks) >= 2:
            rook_list = list(rooks)
            for i, rook1 in enumerate(rook_list):
                for rook2 in rook_list[i+1:]:
                    if (chess.square_file(rook1) == chess.square_file(rook2) or 
                        chess.square_rank(rook1) == chess.square_rank(rook2)):
                        # Check if path is clear
                        if not self._pieces_between(board, rook1, rook2):
                            score += 3
                            
        return score
        
    def _evaluate_pawn_army(self, board: chess.Board, color: chess.Color) -> float:
        """Pawns should advance and support each other"""
        score = 0.0
        pawns = board.pieces(chess.PAWN, color)
        
        for pawn_square in pawns:
            # Advancement bonus
            if color == chess.WHITE:
                advancement = chess.square_rank(pawn_square) - 1
            else:
                advancement = 6 - chess.square_rank(pawn_square)
            score += advancement * 0.5
            
            # Pawn chain support
            pawn_file = chess.square_file(pawn_square)
            if color == chess.WHITE:
                support_squares = [
                    chess.square(pawn_file-1, chess.square_rank(pawn_square)-1),
                    chess.square(pawn_file+1, chess.square_rank(pawn_square)-1)
                ]
            else:
                support_squares = [
                    chess.square(pawn_file-1, chess.square_rank(pawn_square)+1),
                    chess.square(pawn_file+1, chess.square_rank(pawn_square)+1)
                ]
                
            for support_sq in support_squares:
                if (0 <= chess.square_file(support_sq) <= 7 and 
                    0 <= chess.square_rank(support_sq) <= 7):
                    piece = board.piece_at(support_sq)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        score += 1  # Pawn chain bonus
                        
        return score
        
    def _evaluate_queen_dominance(self, board: chess.Board, color: chess.Color) -> float:
        """Queen should be active but safe"""
        score = 0.0
        queens = board.pieces(chess.QUEEN, color)
        
        for queen_square in queens:
            # Activity bonus
            attacks = len(list(board.attacks(queen_square)))
            score += attacks * 0.8
            
            # Safety penalty if attacked by lower value pieces
            if board.is_attacked_by(not color, queen_square):
                attackers = board.attackers(not color, queen_square)
                for attacker_sq in attackers:
                    attacker = board.piece_at(attacker_sq)
                    if attacker and attacker.piece_type != chess.QUEEN:
                        score -= 8  # Danger penalty
                        
        return score
        
    def _evaluate_king_leadership(self, board: chess.Board, color: chess.Color) -> float:
        """King should be safe in opening/middlegame, active in endgame"""
        score = 0.0
        king_square = board.king(color)
        
        if king_square is None:
            return -1000  # No king = game over
            
        # Count total pieces to determine game phase
        total_pieces = len(board.piece_map())
        
        if total_pieces > 20:  # Opening/Middlegame - prioritize safety
            # Penalty for exposed king
            if board.is_attacked_by(not color, king_square):
                score -= 10
                
            # Bonus for castling rights
            if color == chess.WHITE:
                if board.has_kingside_castling_rights(color):
                    score += 2
                if board.has_queenside_castling_rights(color):
                    score += 2
            else:
                if board.has_kingside_castling_rights(color):
                    score += 2
                if board.has_queenside_castling_rights(color):
                    score += 2
                    
        else:  # Endgame - king should be active
            # Bonus for king centralization
            center_distance = min(
                abs(chess.square_file(king_square) - 3.5),
                abs(chess.square_rank(king_square) - 3.5)
            )
            score += max(0, 4 - center_distance)
            
        return score
        
    def _pieces_between(self, board: chess.Board, square1: int, square2: int) -> bool:
        """Check if there are pieces between two squares"""
        # Get squares between the two positions
        file1, rank1 = chess.square_file(square1), chess.square_rank(square1)
        file2, rank2 = chess.square_file(square2), chess.square_rank(square2)
        
        # Check if on same file or rank
        if file1 == file2:  # Same file
            min_rank, max_rank = min(rank1, rank2), max(rank1, rank2)
            for rank in range(min_rank + 1, max_rank):
                if board.piece_at(chess.square(file1, rank)):
                    return True
        elif rank1 == rank2:  # Same rank
            min_file, max_file = min(file1, file2), max(file1, file2)
            for file in range(min_file + 1, max_file):
                if board.piece_at(chess.square(file, rank1)):
                    return True
                    
        return False

class EnhancedGeneticTrainer:
    """Enhanced genetic algorithm with game-like evaluation and multi-opponent training"""
    
    def __init__(self, 
                 population_size: int = 32,
                 mutation_rate: float = 0.2,
                 crossover_rate: float = 0.8,
                 elite_percentage: float = 0.1,
                 device: str = "cuda",
                 config_path: str = "config.json"):
        
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_count = max(1, int(population_size * elite_percentage))
        self.device = device
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.feature_extractor = GPUChessFeatureExtractor()
        self.bounty_evaluator = ExtendedBountyEvaluator()
        self.game_evaluator = GameLikeEvaluator()
        self.opponent_manager = OpponentManager()
        
        # Training state
        self.generation = 0
        self.population = []
        self.fitness_history = []
        self.best_model = None
        self.best_fitness = float('-inf')
        
        print(f"Enhanced Genetic Trainer initialized:")
        print(f"  Population size: {population_size}")
        print(f"  Mutation rate: {mutation_rate}")
        print(f"  Crossover rate: {crossover_rate}")
        print(f"  Elite preservation: {self.elite_count} individuals")
        print(f"  Available opponents: {len(self.opponent_manager.opponents)}")
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except:
            # Default configuration
            return {
                "games_per_individual": 4,
                "max_moves_per_game": 100,
                "bounty_weight": 0.3,
                "neural_weight": 0.7,
                "game_evaluation_weight": 0.2
            }
            
    def evaluate_game_like(self, model, board: chess.Board, color: chess.Color) -> float:
        """Evaluate position using game-like piece performance metrics"""
        piece_scores = self.game_evaluator.evaluate_piece_performance(board, color)
        
        total_score = 0.0
        weights = {
            'knight_mobility': 1.5,    # Knights are tactical pieces
            'bishop_vision': 1.2,      # Long-range control
            'rook_power': 1.8,         # File/rank dominance
            'pawn_army': 1.0,          # Foundation army
            'queen_dominance': 2.0,    # Most powerful piece
            'king_leadership': 1.3     # Command center
        }
        
        for metric, score in piece_scores.items():
            weight = weights.get(metric, 1.0)
            total_score += score * weight
            
        return total_score
        
    def play_training_game(self, model, opponent, max_moves: int = 100) -> Tuple[float, str]:
        """Play a training game with diverse evaluation"""
        board = chess.Board()
        model_color = random.choice([chess.WHITE, chess.BLACK])
        
        total_bounty = 0.0
        game_like_score = 0.0
        moves_played = 0
        
        # Add randomness to starting position occasionally
        if random.random() < 0.2:  # 20% chance of random opening
            for _ in range(random.randint(1, 4)):
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    board.push(random.choice(legal_moves))
        
        while not board.is_game_over() and moves_played < max_moves:
            current_color = board.turn
            
            if current_color == model_color:
                # AI's turn
                move = self._get_model_move(model, board)
                if move:
                    # Evaluate position before move
                    pre_move_bounty = self.bounty_evaluator.evaluate_position(board)
                    pre_move_game_score = self.evaluate_game_like(model, board, model_color)
                    
                    board.push(move)
                    
                    # Evaluate position after move
                    post_move_bounty = self.bounty_evaluator.evaluate_position(board)
                    post_move_game_score = self.evaluate_game_like(model, board, model_color)
                    
                    # Calculate improvements
                    bounty_gain = post_move_bounty - pre_move_bounty
                    game_gain = post_move_game_score - pre_move_game_score
                    
                    # Add randomness to evaluation (critical for non-deterministic training)
                    random_factor = random.uniform(0.8, 1.2)
                    total_bounty += bounty_gain * random_factor
                    game_like_score += game_gain * random_factor
                else:
                    break
            else:
                # Opponent's turn
                move = opponent.get_move(board)
                if move:
                    board.push(move)
                else:
                    break
                    
            moves_played += 1
            
        # Game outcome bonuses
        result = board.result()
        outcome_bonus = 0.0
        
        if result == "1-0" and model_color == chess.WHITE:
            outcome_bonus = 1000
        elif result == "0-1" and model_color == chess.BLACK:
            outcome_bonus = 1000
        elif result == "1/2-1/2":
            outcome_bonus = 200
        else:
            outcome_bonus = -200  # Loss penalty
            
        # Combine all evaluation components
        bounty_component = total_bounty * self.config.get("bounty_weight", 0.3)
        game_component = game_like_score * self.config.get("game_evaluation_weight", 0.2)
        
        final_score = bounty_component + game_component + outcome_bonus
        
        # Add exploration bonus for longer games (encourages complex play)
        exploration_bonus = min(50, moves_played * 0.5)
        final_score += exploration_bonus
        
        return final_score, result
        
    def _get_model_move(self, model, board: chess.Board) -> Optional[chess.Move]:
        """Get move from model with added randomness"""
        try:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return None
                
            # Extract features
            features = self.feature_extractor.extract_features(board)
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # Get model prediction
            with torch.no_grad():
                model.eval()
                output = model(features_tensor)
                move_scores = output.squeeze().cpu().numpy()
                
            # Add randomness to move selection (critical for diversity)
            temperature = random.uniform(0.1, 0.5)  # Random temperature
            softmax_scores = np.exp(move_scores / temperature)
            softmax_scores = softmax_scores / np.sum(softmax_scores)
            
            # Select move probabilistically
            if random.random() < 0.1:  # 10% completely random exploration
                return random.choice(legal_moves)
            else:
                # Weighted random selection
                move_index = np.random.choice(len(softmax_scores), p=softmax_scores)
                if move_index < len(legal_moves):
                    return legal_moves[move_index]
                else:
                    return random.choice(legal_moves)
                    
        except Exception as e:
            print(f"Error in model move generation: {e}")
            return random.choice(legal_moves) if legal_moves else None
            
    def evaluate_individual(self, model) -> float:
        """Evaluate an individual using diverse opponents"""
        total_fitness = 0.0
        games_per_individual = self.config.get("games_per_individual", 4)
        
        # Get diverse opponents based on training rotation
        rotation = self.opponent_manager.get_training_rotation()
        
        for strength, percentage in rotation:
            num_games = max(1, int(games_per_individual * percentage))
            
            for _ in range(num_games):
                opponent = self.opponent_manager.get_opponent_by_strength(strength)
                fitness, result = self.play_training_game(model, opponent)
                total_fitness += fitness
                
        return total_fitness / games_per_individual
        
    def mutate(self, model) -> None:
        """Mutate model parameters"""
        with torch.no_grad():
            for param in model.parameters():
                if random.random() < self.mutation_rate:
                    # Gaussian mutation with random strength
                    mutation_strength = random.uniform(0.01, 0.1)
                    noise = torch.randn_like(param) * mutation_strength
                    param.add_(noise)
                    
    def crossover(self, parent1, parent2):
        """Create offspring through crossover"""
        child = V7P3RGPU_LSTM().to(self.device)
        
        with torch.no_grad():
            for child_param, p1_param, p2_param in zip(
                child.parameters(), parent1.parameters(), parent2.parameters()
            ):
                if random.random() < self.crossover_rate:
                    # Uniform crossover
                    mask = torch.rand_like(child_param) < 0.5
                    child_param.data = torch.where(mask, p1_param.data, p2_param.data)
                else:
                    # Copy from parent 1
                    child_param.data.copy_(p1_param.data)
                    
        return child
        
    def evolve_generation(self):
        """Evolve one generation"""
        print(f"\n=== Generation {self.generation} ===")
        
        # Evaluate all individuals
        fitness_scores = []
        for i, individual in enumerate(self.population):
            fitness = self.evaluate_individual(individual)
            fitness_scores.append((fitness, individual))
            print(f"Individual {i+1}: {fitness:.2f}")
            
        # Sort by fitness
        fitness_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Track best model
        best_fitness_this_gen = fitness_scores[0][0]
        if best_fitness_this_gen > self.best_fitness:
            self.best_fitness = best_fitness_this_gen
            self.best_model = fitness_scores[0][1]
            
            # Save best model
            model_path = f"models/best_enhanced_model_gen_{self.generation}.pth"
            torch.save(self.best_model.state_dict(), model_path)
            print(f"🏆 New best model saved: {model_path} (fitness: {best_fitness_this_gen:.2f})")
            
        # Create next generation
        new_population = []
        
        # Elite preservation
        for i in range(self.elite_count):
            new_population.append(fitness_scores[i][1])
            
        # Fill remaining population
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection(fitness_scores)
            parent2 = self._tournament_selection(fitness_scores)
            
            # Create offspring
            if random.random() < self.crossover_rate:
                child = self.crossover(parent1, parent2)
            else:
                child = parent1  # Copy parent
                
            # Mutate
            self.mutate(child)
            new_population.append(child)
            
        self.population = new_population
        self.fitness_history.append([score for score, _ in fitness_scores])
        self.generation += 1
        
    def _tournament_selection(self, fitness_scores, tournament_size: int = 3):
        """Select individual through tournament selection"""
        tournament = random.sample(fitness_scores, min(tournament_size, len(fitness_scores)))
        return max(tournament, key=lambda x: x[0])[1]
        
    def train(self, generations: int = 100):
        """Train for specified number of generations"""
        print(f"Starting enhanced genetic training for {generations} generations...")
        
        # Initialize population if empty
        if not self.population:
            self.population = [V7P3RGPU_LSTM().to(self.device) for _ in range(self.population_size)]
            
        for gen in range(generations):
            self.evolve_generation()
            
            # Print generation summary
            avg_fitness = np.mean(self.fitness_history[-1])
            max_fitness = np.max(self.fitness_history[-1])
            print(f"Generation {self.generation-1} Summary:")
            print(f"  Average fitness: {avg_fitness:.2f}")
            print(f"  Best fitness: {max_fitness:.2f}")
            print(f"  All-time best: {self.best_fitness:.2f}")
            
        self.opponent_manager.cleanup()
        print(f"\n🎯 Training complete! Best fitness achieved: {self.best_fitness:.2f}")

def main():
    """Test the enhanced genetic trainer"""
    trainer = EnhancedGeneticTrainer(
        population_size=8,  # Small for testing
        mutation_rate=0.25,
        crossover_rate=0.8,
        elite_percentage=0.2
    )
    
    trainer.train(generations=2)  # Short test run

if __name__ == "__main__":
    main()
