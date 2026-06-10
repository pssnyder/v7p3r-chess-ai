"""
Generational Training Architecture - AlphaZero Style

Trains new model generations against previous best:
- New generation trains via self-play
- Evaluated against previous best in 6-game match (3 White, 3 Black)
- Accepted only if win rate > 50% (with tiebreaker logic)
- Provides MEANINGFUL win/loss metrics (unlike pure self-play)
"""

import chess
import chess.pgn
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
from datetime import datetime
import json
import copy

from network import V7ValueNetwork, V7Trainer, create_v7_network
from selfplay_trainer import SelfPlayGame, ExperienceBuffer, GameResult
from stockfish_oracle import StockfishOracle
from personality_rewards import PersonalityRewardCalculator
from comprehensive_features import ComprehensiveFeatureExtractor
from phase_manager import PhaseAwareTrainingTarget, DynamicWeightCalculator
from opening_book import OpeningBookManager
from tablebase_oracle import TablebaseOracle


class GenerationResult:
    """Results from evaluating a new generation against previous best."""
    
    def __init__(
        self,
        generation_number: int,
        wins_as_white: int,
        wins_as_black: int,
        draws: int,
        total_games: int
    ):
        self.generation_number = generation_number
        self.wins_as_white = wins_as_white
        self.wins_as_black = wins_as_black
        self.draws = draws
        self.total_games = total_games
        
        # Calculate metrics
        self.total_wins = wins_as_white + wins_as_black
        self.win_rate = self.total_wins / total_games if total_games > 0 else 0.0
        self.accepted = self.win_rate > 0.5 or (self.win_rate == 0.5 and self.total_wins > self.draws)
        
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'generation_number': self.generation_number,
            'wins_as_white': self.wins_as_white,
            'wins_as_black': self.wins_as_black,
            'draws': self.draws,
            'total_games': self.total_games,
            'total_wins': self.total_wins,
            'win_rate': self.win_rate,
            'accepted': self.accepted
        }


class GenerationalTrainer:
    """
    Manages generational training:
    1. New model trains via self-play (learning diversity)
    2. New model evaluated vs best model (measuring improvement)
    3. New model accepted only if it beats the previous generation
    
    This provides MEANINGFUL win/loss metrics unlike pure self-play.
    """
    
    def __init__(
        self,
        profile_path: str,
        stockfish_path: str,
        output_dir: str = "../../training/v7_generational",
        opening_book_pgn: Optional[str] = None,
        tablebase_path: Optional[str] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load profile
        import json
        from personality_tuner import PlaystyleProfile
        with open(profile_path, 'r') as f:
            profile_data = json.load(f)
        self.profile = PlaystyleProfile.from_dict(profile_data)
        
        # Initialize components
        self.extractor = ComprehensiveFeatureExtractor()
        self.calculator = PersonalityRewardCalculator(self.profile.weights)
        self.oracle = StockfishOracle(stockfish_path)
        
        # Phase manager with NEW v7.1 weights
        self.phase_manager = PhaseAwareTrainingTarget(DynamicWeightCalculator(
            opening_sf_weight=0.9,      # 90% in opening
            middlegame_sf_weight=0.2,   # 20% in middlegame (up from 10%)
            endgame_sf_weight=1.0,      # 100% in endgame (up from 50%)
            tablebase_sf_weight=1.0     # 100% when tablebase available
        ))
        
        # Opening book and tablebases
        self.opening_book = OpeningBookManager(opening_book_pgn)
        print(f"[OK] Opening book loaded: {len(self.opening_book)} lines")
        
        self.tablebase_oracle = TablebaseOracle(tablebase_path)
        if self.tablebase_oracle.enabled:
            print(f"[OK] Tablebases loaded: {self.tablebase_oracle.max_pieces}-piece")
        else:
            print("[INFO] Tablebases disabled")
        
        # Model management
        self.best_model = None
        self.best_trainer = None
        self.new_model = None
        self.new_trainer = None
        
        # Generation tracking
        self.current_generation = 0
        self.generation_history = []
        
        # Experience buffer for training
        self.buffer = ExperienceBuffer(max_size=100000)
        
    def initialize_first_generation(self):
        """Create the initial model (generation 0)."""
        print("\n" + "=" * 60)
        print("INITIALIZING GENERATION 0 (Baseline)")
        print("=" * 60)
        
        self.best_model, self.best_trainer = create_v7_network()
        self.current_generation = 0
        
        # Save initial model
        model_path = self.output_dir / "gen_0000_initial.pt"
        torch.save(self.best_model.state_dict(), model_path)
        print(f"[OK] Generation 0 saved: {model_path}")
        
    def train_new_generation(
        self,
        selfplay_games: int = 100,
        batch_size: int = 256,
        train_every_n_games: int = 10
    ):
        """
        Train a new generation via self-play.
        
        Args:
            selfplay_games: Number of self-play games for training
            batch_size: Batch size for neural network training
            train_every_n_games: Train network after this many games
        """
        print("\n" + "=" * 60)
        print(f"TRAINING GENERATION {self.current_generation + 1}")
        print("=" * 60)
        print(f"Self-play games: {selfplay_games}")
        print(f"Training every: {train_every_n_games} games")
        print()
        
        # Clone best model to create new generation
        self.new_model, self.new_trainer = create_v7_network()
        if self.best_model is not None:
            self.new_model.load_state_dict(copy.deepcopy(self.best_model.state_dict()))
            print(f"[OK] Cloned generation {self.current_generation} as starting point")
        
        # Clear buffer for new generation
        self.buffer = ExperienceBuffer(max_size=100000)
        
        # Self-play training loop
        game_player = SelfPlayGame(
            network=self.new_model,
            oracle=self.oracle,
            calculator=self.calculator,
            extractor=self.extractor,
            phase_manager=self.phase_manager,
            max_moves=400,  # Extended for training to see if engine can convert
            temperature=0.3,
            opening_book=self.opening_book,
            tablebase_oracle=self.tablebase_oracle
        )
        
        self.oracle.start()
        
        for game_num in range(1, selfplay_games + 1):
            experiences, result = game_player.play_game(game_num)
            
            # Add full game to buffer
            self.buffer.add_game(experiences, result)
            
            # Show game result with diagnostics
            tb_info = f", TB:{result.tablebase_positions}" if result.tablebase_positions > 0 else ""
            if result.num_moves >= 380:
                piece_info = f", pieces:{result.final_piece_count}, halfmove:{result.final_halfmove_clock}"
            else:
                piece_info = ""
            print(f"  Game {game_num}/{selfplay_games}: {result.result} "
                  f"({result.num_moves} moves, darkness={result.avg_forest_darkness:.3f}{tb_info}{piece_info})")
            
            # Train periodically
            if game_num % train_every_n_games == 0:
                avg_loss = self._train_on_buffer(batch_size)
                print(f"  → Training update: loss={avg_loss:.4f}, buffer={len(self.buffer.experiences)}")
        
        self.oracle.stop()
        
        # Save trained model
        gen_num = self.current_generation + 1
        model_path = self.output_dir / f"gen_{gen_num:04d}_trained.pt"
        torch.save(self.new_model.state_dict(), model_path)
        print(f"\n[OK] Generation {gen_num} training complete")
        print(f"[OK] Model saved: {model_path}")
        
    def evaluate_generation(self, num_games: int = 6) -> GenerationResult:
        """
        Evaluate new generation vs previous best in tournament.
        
        Plays 6-game match: 3 as White, 3 as Black (alternating colors for fairness).
        
        Args:
            num_games: Total games (must be even for color balance)
        
        Returns:
            GenerationResult with win/loss/draw statistics
        """
        if num_games % 2 != 0:
            raise ValueError("num_games must be even for color balance")
        
        print("\n" + "=" * 60)
        print(f"EVALUATING GENERATION {self.current_generation + 1}")
        print("=" * 60)
        print(f"Match format: {num_games} games ({num_games//2} as White, {num_games//2} as Black)")
        print()
        
        wins_as_white = 0
        wins_as_black = 0
        draws = 0
        
        self.oracle.start()
        
        for game_num in range(1, num_games + 1):
            # Alternate colors
            new_plays_white = (game_num % 2 == 1)
            
            if new_plays_white:
                result = self._play_evaluation_game(
                    white_model=self.new_model,
                    black_model=self.best_model,
                    game_num=game_num
                )
                
                if result == "1-0":
                    wins_as_white += 1
                    outcome_str = "WIN (as White)"
                elif result == "0-1":
                    outcome_str = "LOSS (as White)"
                else:
                    draws += 1
                    outcome_str = "DRAW (as White)"
            else:
                result = self._play_evaluation_game(
                    white_model=self.best_model,
                    black_model=self.new_model,
                    game_num=game_num
                )
                
                if result == "0-1":
                    wins_as_black += 1
                    outcome_str = "WIN (as Black)"
                elif result == "1-0":
                    outcome_str = "LOSS (as Black)"
                else:
                    draws += 1
                    outcome_str = "DRAW (as Black)"
            
            print(f"  Game {game_num}/{num_games}: {outcome_str}")
        
        self.oracle.stop()
        
        # Create result
        gen_result = GenerationResult(
            generation_number=self.current_generation + 1,
            wins_as_white=wins_as_white,
            wins_as_black=wins_as_black,
            draws=draws,
            total_games=num_games
        )
        
        print()
        print("=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Wins as White: {wins_as_white}/{num_games//2}")
        print(f"Wins as Black: {wins_as_black}/{num_games//2}")
        print(f"Total Wins:    {gen_result.total_wins}/{num_games}")
        print(f"Draws:         {draws}/{num_games}")
        print(f"Win Rate:      {gen_result.win_rate:.1%}")
        print()
        
        if gen_result.accepted:
            print("✅ GENERATION ACCEPTED - New model is better!")
        else:
            print("❌ GENERATION REJECTED - Old model still stronger")
        
        print("=" * 60)
        
        return gen_result
    
    def _play_evaluation_game(
        self,
        white_model: V7ValueNetwork,
        black_model: V7ValueNetwork,
        game_num: int
    ) -> str:
        """
        Play a single evaluation game between two models.
        
        Returns:
            Result string: "1-0", "0-1", or "1/2-1/2"
        """
        board = chess.Board()
        
        # Apply opening book
        opening_line, opening_moves = self.opening_book.apply_random_opening(board)
        move_number = len(opening_moves)
        
        # Create game players for each model
        white_player = SelfPlayGame(
            network=white_model,
            oracle=self.oracle,
            calculator=self.calculator,
            extractor=self.extractor,
            phase_manager=self.phase_manager,
            max_moves=400,  # Extended for training
            temperature=0.1,  # Lower temperature for evaluation (more deterministic)
            opening_book=None,  # Already applied
            tablebase_oracle=self.tablebase_oracle
        )
        
        black_player = SelfPlayGame(
            network=black_model,
            oracle=self.oracle,
            calculator=self.calculator,
            extractor=self.extractor,
            phase_manager=self.phase_manager,
            max_moves=400,  # Extended for training
            temperature=0.1,
            opening_book=None,
            tablebase_oracle=self.tablebase_oracle
        )
        
        # Play game
        while not board.is_game_over() and move_number < 400:
            move_number += 1
            
            # Select player based on turn
            current_player = white_player if board.turn == chess.WHITE else black_player
            
            # Get move
            move = current_player.select_move(board)
            if move is None:
                break
            
            board.push(move)
        
        # Return result
        return board.result()
    
    def accept_generation(self, gen_result: GenerationResult):
        """Accept new generation as the new best."""
        self.best_model = self.new_model
        self.best_trainer = self.new_trainer
        self.current_generation += 1
        self.generation_history.append(gen_result.to_dict())
        
        # Save as best
        best_path = self.output_dir / "best_model.pt"
        torch.save(self.best_model.state_dict(), best_path)
        print(f"[OK] Generation {self.current_generation} accepted as new best")
        
        # Save history
        self._save_history()
    
    def reject_generation(self, gen_result: GenerationResult):
        """Reject new generation, keep old best."""
        self.generation_history.append(gen_result.to_dict())
        print(f"[WARN] Generation {self.current_generation + 1} rejected")
        
        # Save history (but don't update best model)
        self._save_history()
    
    def _train_on_buffer(self, batch_size: int) -> float:
        """Train new model on experiences in buffer."""
        experiences = self.buffer.get_all_experiences()
        
        features = np.array([e.features for e in experiences])
        targets = np.array([e.training_target for e in experiences])
        
        # Split for validation
        split_idx = int(len(features) * 0.8)
        train_features = features[:split_idx]
        train_targets = targets[:split_idx]
        val_features = features[split_idx:]
        val_targets = targets[split_idx:]
        
        # Train
        history = self.new_trainer.fit(
            train_features,
            train_targets,
            val_features=val_features,
            val_targets=val_targets,
            batch_size=batch_size,
            epochs=1,
            verbose=0
        )
        
        return np.mean(history['train_loss'][-5:]) if history['train_loss'] else 0.0
    
    def _save_history(self):
        """Save generation history to JSON."""
        history_path = self.output_dir / "generation_history.json"
        with open(history_path, 'w') as f:
            json.dump({
                'current_generation': self.current_generation,
                'generations': self.generation_history
            }, f, indent=2)
        print(f"[OK] History saved: {history_path}")
    
    def run_full_cycle(
        self,
        selfplay_games: int = 100,
        evaluation_games: int = 6,
        max_generations: int = 10
    ):
        """
        Run complete generational training cycle.
        
        Args:
            selfplay_games: Games per generation for training
            evaluation_games: Games for evaluation match
            max_generations: Maximum generations to train
        """
        print("=" * 80)
        print("V7P3R GENERATIONAL TRAINING - AlphaZero Style")
        print("=" * 80)
        print(f"Self-play games per generation: {selfplay_games}")
        print(f"Evaluation match: {evaluation_games} games")
        print(f"Maximum generations: {max_generations}")
        print("=" * 80)
        
        # Initialize if needed
        if self.best_model is None:
            self.initialize_first_generation()
        
        # Training loop
        for gen in range(max_generations):
            # Train new generation
            self.train_new_generation(
                selfplay_games=selfplay_games,
                batch_size=256,
                train_every_n_games=10
            )
            
            # Evaluate vs best
            result = self.evaluate_generation(num_games=evaluation_games)
            
            # Accept or reject
            if result.accepted:
                self.accept_generation(result)
            else:
                self.reject_generation(result)
            
            print()
        
        print("=" * 80)
        print("GENERATIONAL TRAINING COMPLETE")
        print("=" * 80)
        print(f"Final generation: {self.current_generation}")
        print(f"Generations trained: {len(self.generation_history)}")
        print(f"Accepted: {sum(1 for g in self.generation_history if g['accepted'])}")
        print(f"Rejected: {sum(1 for g in self.generation_history if not g['accepted'])}")
        print("=" * 80)
