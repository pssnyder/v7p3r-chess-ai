"""
V7P3R AI v3.0 - Self-Play Training System
=========================================

Pure autonomous self-play training system that teaches the Thinking Brain
through game experience. No human chess knowledge - only objective reward
signals from game outcomes and position improvements.

Architecture:
- Self-play game loops using both Thinking Brain + Gameplay Brain
- Objective bounty system based on game status changes
- Experience collection and replay training
- Performance tracking every 1000 games
"""

import chess
import chess.pgn
import torch
import numpy as np
import random
import time
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging
from tqdm import tqdm

# Import our AI systems
import sys
sys.path.append(str(Path(__file__).parent.parent))
from ai.thinking_brain import ThinkingBrain, PositionMemory, ThinkingBrainTrainer
from ai.gameplay_brain import GameplayBrain
from core.chess_state import ChessStateExtractor
from core.neural_features import NeuralFeatureConverter
from monitoring.integration import IntegratedTrainingMonitor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GameExperience:
    """Single game experience for training"""
    positions: List[torch.Tensor]  # Position features throughout game
    moves: List[chess.Move]        # Moves played throughout game
    rewards: List[float]           # Bounty rewards at each position
    final_outcome: float           # Final game result (1.0=win, 0.5=draw, 0.0=loss)
    game_length: int               # Number of moves in game
    thinking_times: List[float]    # Time spent on each move
    gameplay_scores: List[float]   # Gameplay brain fitness scores


@dataclass
class TrainingProgress:
    """Training progress tracking"""
    games_played: int = 0
    total_training_time: float = 0.0
    wins: int = 0
    draws: int = 0 
    losses: int = 0
    avg_game_length: float = 0.0
    avg_move_time: float = 0.0
    latest_model_loss: float = 0.0
    performance_milestones: Optional[List[Dict]] = None
    
    def __post_init__(self):
        if self.performance_milestones is None:
            self.performance_milestones = []


class BountySystem:
    """
    Objective reward system based on game status improvements
    
    Calculates turn-by-turn rewards based on measurable chess metrics
    without any human chess knowledge or heuristics.
    """
    
    def __init__(self):
        self.state_extractor = ChessStateExtractor()
        
        # Reward weights (AI will potentially learn to adjust these)
        self.weights = {
            'king_safety': 1.0,
            'piece_activity': 1.0, 
            'material_balance': 2.0,
            'pawn_structure': 0.5,
            'game_phase_adaptation': 0.3
        }
    
    def calculate_position_reward(
        self, 
        previous_state, 
        current_state, 
        side_to_move: bool  # True if AI's turn
    ) -> float:
        """
        Calculate reward for a single position based on objective improvements
        
        Args:
            previous_state: Previous position state
            current_state: Current position state  
            side_to_move: Whether it's the AI's turn to move
            
        Returns:
            reward: Objective reward value
        """
        if previous_state is None:
            return 0.0  # No previous state to compare
        
        reward = 0.0
        
        # King Safety Delta
        prev_check = previous_state.board_features.isKingInCheck
        curr_check = current_state.board_features.isKingInCheck
        
        if side_to_move:
            # AI's perspective: avoid check, put opponent in check
            if prev_check and not curr_check:  # Escaped check
                reward += 0.2 * self.weights['king_safety']
            elif not prev_check and curr_check:  # Put opponent in check
                reward += 0.1 * self.weights['king_safety']
        else:
            # Opponent's perspective: reverse rewards
            if prev_check and not curr_check:  # Opponent escaped check
                reward -= 0.2 * self.weights['king_safety']
            elif not prev_check and curr_check:  # Opponent put AI in check
                reward -= 0.1 * self.weights['king_safety']
        
        # Piece Activity Delta
        prev_mobility = previous_state.board_features.mobility
        curr_mobility = current_state.board_features.mobility
        
        prev_total = prev_mobility.get('totalWhite', 0) + prev_mobility.get('totalBlack', 0)
        curr_total = curr_mobility.get('totalWhite', 0) + curr_mobility.get('totalBlack', 0)
        
        if prev_total > 0:
            mobility_change = (curr_total - prev_total) / prev_total
            reward += mobility_change * 0.1 * self.weights['piece_activity']
        
        # Material Balance Delta
        prev_material = previous_state.board_features.materialBalance.get('difference', 0)
        curr_material = current_state.board_features.materialBalance.get('difference', 0)
        
        material_change = curr_material - prev_material
        if not side_to_move:  # Adjust for black's perspective
            material_change = -material_change
        
        reward += material_change * 0.05 * self.weights['material_balance']
        
        # Pawn Structure Improvements (simplified)
        prev_pawn_metrics = self._calculate_pawn_score(previous_state.board_features.pawnStructure)
        curr_pawn_metrics = self._calculate_pawn_score(current_state.board_features.pawnStructure)
        
        pawn_improvement = curr_pawn_metrics - prev_pawn_metrics
        reward += pawn_improvement * 0.02 * self.weights['pawn_structure']
        
        # Game Phase Adaptation
        game_phase = current_state.board_features.gamePhase
        phase_bonus = self._calculate_phase_bonus(current_state, game_phase)
        reward += phase_bonus * self.weights['game_phase_adaptation']
        
        return reward
    
    def _calculate_pawn_score(self, pawn_structure: Dict) -> float:
        """Calculate simple pawn structure score"""
        score = 0.0
        
        # Passed pawns are good
        passed_pawns = pawn_structure.get('passedPawns', {'white': 0, 'black': 0})
        score += (passed_pawns.get('white', 0) - passed_pawns.get('black', 0)) * 0.1
        
        # Doubled pawns are bad  
        doubled_pawns = pawn_structure.get('doubledPawns', {'white': 0, 'black': 0})
        score -= (doubled_pawns.get('white', 0) - doubled_pawns.get('black', 0)) * 0.05
        
        return score
    
    def _calculate_phase_bonus(self, state, game_phase: float) -> float:
        """Calculate game phase adaptation bonus"""
        # Encourage different priorities based on game phase
        if game_phase < 0.3:  # Opening
            return 0.01  # Small bonus for development
        elif game_phase > 0.7:  # Endgame
            return 0.02  # Bonus for king activity
        else:  # Middlegame
            return 0.015  # Balanced bonus
    
    def calculate_final_game_reward(self, game_outcome: str, side: str) -> float:
        """Calculate final reward based on game outcome"""
        if game_outcome == '1-0':  # White wins
            return 1.0 if side == 'white' else 0.0
        elif game_outcome == '0-1':  # Black wins
            return 1.0 if side == 'black' else 0.0
        else:  # Draw
            return 0.5


class SelfPlayTrainer:
    """
    Main self-play training system that orchestrates the complete learning loop
    """
    
    def __init__(
        self,
        thinking_brain: ThinkingBrain,
        gameplay_brain: GameplayBrain,
        save_directory: str = "models",
        games_per_checkpoint: int = 1000,
        max_game_length: int = 200,
        enable_progress_tracking: bool = True,
        enable_visual_monitoring: bool = False,
        monitoring_config: Optional[Dict[str, Any]] = None
    ):
        self.thinking_brain = thinking_brain
        self.gameplay_brain = gameplay_brain
        self.save_directory = Path(save_directory)
        self.save_directory.mkdir(exist_ok=True)
        
        self.games_per_checkpoint = games_per_checkpoint
        self.max_game_length = max_game_length
        self.enable_progress_tracking = enable_progress_tracking
        
        # Training systems
        self.trainer = ThinkingBrainTrainer(thinking_brain)
        self.bounty_system = BountySystem()
        self.feature_converter = NeuralFeatureConverter()
        
        # Progress tracking
        self.progress = TrainingProgress()
        self.game_experiences = []
        self.current_checkpoint = 0
        
        # Performance data for monitoring
        self.recent_games = []  # Store recent game data for analysis
        self.position_heatmap_data = {}  # For Phase 5 monitoring
        
        # Visual monitoring system (Phase 5)
        self.enable_visual_monitoring = enable_visual_monitoring
        self.monitor = None
        if enable_visual_monitoring:
            # Initialize monitoring with configuration
            if monitoring_config is None:
                monitoring_config = {
                    'enable_visual': True,
                    'save_data': True,
                    'output_dir': str(self.save_directory / "monitoring")
                }
            
            self.monitor = IntegratedTrainingMonitor(
                enable_visual=monitoring_config.get('enable_visual', True),
                save_data=monitoring_config.get('save_data', True),
                output_dir=Path(monitoring_config.get('output_dir', str(self.save_directory / "monitoring")))
            )
            logger.info("Visual monitoring system initialized")
        
        logger.info(f"SelfPlayTrainer initialized - Checkpoints every {games_per_checkpoint} games")
    
    def train(self, target_games: int, resume_from_checkpoint: bool = True):
        """
        Main training loop
        
        Args:
            target_games: Total number of games to train
            resume_from_checkpoint: Whether to resume from saved checkpoint
        """
        logger.info(f"Starting self-play training for {target_games} games")
        
        # Resume from checkpoint if requested
        if resume_from_checkpoint:
            self._load_checkpoint()
        
        # Training loop
        games_to_play = target_games - self.progress.games_played
        
        # Start monitoring if enabled
        if self.monitor:
            self.monitor.start_monitoring(f"training_session_{int(time.time())}")
        
        with tqdm(total=games_to_play, desc="Self-Play Training") as pbar:
            while self.progress.games_played < target_games:
                # Play a single game
                game_experience = self._play_game()
                
                # Store experience for training
                self.game_experiences.append(game_experience)
                
                # Update progress
                self._update_progress(game_experience)
                pbar.update(1)
                pbar.set_postfix({
                    'Win Rate': f"{self.progress.wins / max(self.progress.games_played, 1):.3f}",
                    'Avg Length': f"{self.progress.avg_game_length:.1f}",
                    'Latest Loss': f"{self.progress.latest_model_loss:.4f}"
                })
                
                # Train on accumulated experiences
                if len(self.game_experiences) >= 10:  # Train every 10 games
                    self._train_on_experiences()
                
                # Checkpoint every N games
                if self.progress.games_played % self.games_per_checkpoint == 0:
                    self._save_checkpoint()
                    self._evaluate_performance()
        
        # Stop monitoring
        if self.monitor:
            self.monitor.stop_monitoring()
        
        logger.info(f"Training completed! Total games: {self.progress.games_played}")
        self._save_final_model()
    
    def _play_game(self) -> GameExperience:
        """Play a single self-play game"""
        board = chess.Board()
        
        # Start game monitoring
        game_id = f"game_{self.progress.games_played + 1}_{int(time.time())}"
        if self.monitor:
            self.monitor.update_position(board)
        
        # Game tracking
        positions = []
        moves = []
        rewards = []
        thinking_times = []
        gameplay_scores = []
        
        # Enhanced AI memory systems with dynamic management
        memory_config = {
            'max_memory_positions': 15,     # Reduced from default 20
            'memory_decay_factor': 0.9,     # More aggressive decay
            'status_trend_window': 4,       # Shorter trend window
            'pruning_threshold': 0.4,       # More aggressive pruning
            'critical_memory_boost': 1.3    # Moderate boost for critical moments
        }
        
        white_memory = PositionMemory(self.thinking_brain, memory_config)
        black_memory = PositionMemory(self.thinking_brain, memory_config)
        white_memory.start_new_game()
        black_memory.start_new_game()
        
        # Previous state for bounty calculation
        previous_state = None
        
        # Game loop
        move_count = 0
        while not board.is_game_over() and move_count < self.max_game_length:
            start_time = time.time()
            
            # Extract current position state
            current_state = self.bounty_system.state_extractor.extract_state(board)
            position_features = self.feature_converter.convert_to_features(
                current_state, device=str(self.thinking_brain.device)
            )
            
            # Calculate bounty reward
            is_ai_turn = True  # For self-play, AI plays both sides
            reward = self.bounty_system.calculate_position_reward(
                previous_state, current_state, is_ai_turn
            )
            
            # Select memory system for current player
            current_memory = white_memory if board.turn == chess.WHITE else black_memory
            
            # Get move candidates from Thinking Brain
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            
            candidates, probabilities = current_memory.process_position(
                position_features, legal_moves, top_k=min(10, len(legal_moves))
            )
            
            thinking_time = time.time() - start_time
            
            # Update monitoring with Thinking Brain activity
            if self.monitor:
                # Convert probabilities to list format for monitoring
                prob_list = probabilities.cpu().numpy().tolist() if hasattr(probabilities, 'cpu') else list(probabilities)
                current_player = "white" if board.turn else "black"
                self.monitor.update_thinking_brain_activity(
                    candidates, 
                    prob_list,
                    thinking_time,
                    current_player
                )
            
            # Select best move using Gameplay Brain
            gameplay_start = time.time()
            selected_move, analysis = self.gameplay_brain.select_best_move(
                board, candidates, probabilities.cpu().numpy().tolist()
            )
            gameplay_time = time.time() - gameplay_start
            
            # Update monitoring with Gameplay Brain activity
            if self.monitor:
                # Extract fitness scores from analysis
                fitness_scores = []
                for candidate_info in analysis.get('candidates', []):
                    fitness_scores.append(candidate_info.fitness)
                
                self.monitor.update_gameplay_brain_activity(
                    candidates, fitness_scores, gameplay_time
                )
                
                # Record the actual move made
                self.monitor.record_move_made(
                    selected_move, 
                    "gameplay_brain", 
                    analysis['best_candidate'].fitness
                )
            
            # Record move and update memory
            current_memory.record_move(selected_move)
            
            # Store training data
            positions.append(position_features)
            moves.append(selected_move)
            rewards.append(reward)
            thinking_times.append(thinking_time + gameplay_time)
            gameplay_scores.append(analysis['best_candidate'].fitness)
            
            # Store position data for heatmap visualization (Phase 5)
            self._record_position_for_heatmap(board, candidates, probabilities)
            
            # Make the move
            board.push(selected_move)
            previous_state = current_state
            move_count += 1
            
            # Update position in monitor
            if self.monitor:
                self.monitor.update_position(board)
        
        # Calculate final game outcome
        result = board.result()
        if result == '1-0':
            final_outcome = 1.0  # White wins
            result_str = "White wins"
        elif result == '0-1':
            final_outcome = 0.0  # Black wins  
            result_str = "Black wins"
        else:
            final_outcome = 0.5  # Draw
            result_str = "Draw"
        
        # Finalize memory based on game outcome (enhanced memory management)
        # White's perspective: 1.0 if white wins, 0.0 if black wins, 0.5 if draw
        white_outcome = final_outcome
        # Black's perspective: opposite of white's outcome
        black_outcome = 1.0 - final_outcome if final_outcome != 0.5 else 0.5
        
        white_memory.finalize_game_memory(white_outcome)
        black_memory.finalize_game_memory(black_outcome)
        
        # Record game completion in monitor
        if self.monitor:
            reason = "Checkmate" if board.is_checkmate() else \
                    "Stalemate" if board.is_stalemate() else \
                    "Insufficient material" if board.is_insufficient_material() else \
                    "Fifty moves" if board.is_fifty_moves() else \
                    "Repetition" if board.is_repetition() else \
                    "Max length reached" if move_count >= self.max_game_length else \
                    "Game over"
            
            self.monitor.record_game_end(result_str, reason, move_count)
        
        # Create game experience
        experience = GameExperience(
            positions=positions,
            moves=moves,
            rewards=rewards,
            final_outcome=final_outcome,
            game_length=len(moves),
            thinking_times=thinking_times,
            gameplay_scores=gameplay_scores
        )
        
        return experience
    
    def _record_position_for_heatmap(
        self, 
        board: chess.Board, 
        candidates: List[chess.Move], 
        probabilities: torch.Tensor
    ):
        """Record position data for visual heatmap monitoring (Phase 5)"""
        if not self.enable_progress_tracking:
            return
        
        # Convert board to FEN for position tracking
        fen = board.fen().split()[0]  # Just the piece positions
        
        # Initialize position data if new
        if fen not in self.position_heatmap_data:
            self.position_heatmap_data[fen] = {
                'square_attention': np.zeros(64),  # Attention on each square
                'move_frequencies': {},  # How often each move is considered
                'total_visits': 0
            }
        
        pos_data = self.position_heatmap_data[fen]
        pos_data['total_visits'] += 1
        
        # Record move candidate data
        for i, move in enumerate(candidates):
            prob = probabilities[i].item() if i < len(probabilities) else 0.0
            
            # Track square attention
            from_square = move.from_square
            to_square = move.to_square
            
            pos_data['square_attention'][from_square] += prob
            pos_data['square_attention'][to_square] += prob
            
            # Track move frequencies
            move_str = str(move)
            if move_str not in pos_data['move_frequencies']:
                pos_data['move_frequencies'][move_str] = 0
            pos_data['move_frequencies'][move_str] += prob
        
        # Keep only recent position data to prevent memory bloat
        if len(self.position_heatmap_data) > 1000:
            # Remove oldest entries
            oldest_keys = sorted(
                self.position_heatmap_data.keys(), 
                key=lambda k: self.position_heatmap_data[k]['total_visits']
            )[:100]
            for key in oldest_keys:
                del self.position_heatmap_data[key]
    
    def _train_on_experiences(self):
        """Train the Thinking Brain on accumulated game experiences"""
        if not self.game_experiences:
            return
        
        total_loss = 0.0
        games_trained = 0
        
        for experience in self.game_experiences:
            # Train on this game
            loss = self.trainer.train_on_game(
                experience.positions,
                experience.moves, 
                experience.final_outcome
            )
            total_loss += loss
            games_trained += 1
        
        # Update progress
        if games_trained > 0:
            self.progress.latest_model_loss = total_loss / games_trained
        
        # Clear experiences to free memory
        self.game_experiences.clear()
        
        logger.debug(f"Trained on {games_trained} games, avg loss: {self.progress.latest_model_loss:.4f}")
    
    def _update_progress(self, experience: GameExperience):
        """Update training progress statistics"""
        self.progress.games_played += 1
        
        # Update game outcomes
        if experience.final_outcome == 1.0:
            self.progress.wins += 1
        elif experience.final_outcome == 0.0:
            self.progress.losses += 1
        else:
            self.progress.draws += 1
        
        # Update averages
        total_games = self.progress.games_played
        self.progress.avg_game_length = (
            (self.progress.avg_game_length * (total_games - 1) + experience.game_length) / total_games
        )
        
        if experience.thinking_times:
            avg_time_this_game = np.mean(experience.thinking_times)
            self.progress.avg_move_time = float(
                (self.progress.avg_move_time * (total_games - 1) + avg_time_this_game) / total_games
            )
        
        # Store recent game for analysis
        self.recent_games.append({
            'game_number': self.progress.games_played,
            'outcome': experience.final_outcome,
            'length': experience.game_length,
            'avg_thinking_time': np.mean(experience.thinking_times) if experience.thinking_times else 0.0,
            'avg_gameplay_score': np.mean(experience.gameplay_scores) if experience.gameplay_scores else 0.0
        })
        
        # Keep only last 100 games
        if len(self.recent_games) > 100:
            self.recent_games.pop(0)
    
    def _save_checkpoint(self):
        """Save training checkpoint"""
        checkpoint_dir = self.save_directory / f"checkpoint_{self.progress.games_played}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = checkpoint_dir / "thinking_brain.pth"
        self.thinking_brain.save_model(str(model_path))
        
        # Save progress
        progress_path = checkpoint_dir / "progress.json"
        with open(progress_path, 'w') as f:
            json.dump(asdict(self.progress), f, indent=2)
        
        # Save heatmap data for monitoring
        heatmap_path = checkpoint_dir / "heatmap_data.json"
        with open(heatmap_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_data = {}
            for fen, data in self.position_heatmap_data.items():
                serializable_data[fen] = {
                    'square_attention': data['square_attention'].tolist(),
                    'move_frequencies': data['move_frequencies'],
                    'total_visits': data['total_visits']
                }
            json.dump(serializable_data, f)
        
        logger.info(f"Checkpoint saved at game {self.progress.games_played}")
    
    def _load_checkpoint(self):
        """Load the most recent checkpoint"""
        # Find most recent checkpoint
        checkpoints = [d for d in self.save_directory.iterdir() if d.is_dir() and d.name.startswith('checkpoint_')]
        if not checkpoints:
            logger.info("No checkpoints found, starting fresh")
            return
        
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split('_')[1]))
        
        # Load model
        model_path = latest_checkpoint / "thinking_brain.pth"
        if model_path.exists():
            self.thinking_brain = ThinkingBrain.load_model(str(model_path))
            self.trainer = ThinkingBrainTrainer(self.thinking_brain)
        
        # Load progress
        progress_path = latest_checkpoint / "progress.json"
        if progress_path.exists():
            with open(progress_path, 'r') as f:
                progress_data = json.load(f)
                self.progress = TrainingProgress(**progress_data)
        
        logger.info(f"Resumed from checkpoint at game {self.progress.games_played}")
    
    def _evaluate_performance(self):
        """Evaluate and record performance milestone"""
        if self.progress.games_played % self.games_per_checkpoint != 0:
            return
        
        # Calculate performance metrics
        total_games = self.progress.games_played
        win_rate = self.progress.wins / total_games if total_games > 0 else 0.0
        draw_rate = self.progress.draws / total_games if total_games > 0 else 0.0
        
        milestone = {
            'games_played': total_games,
            'win_rate': win_rate,
            'draw_rate': draw_rate,
            'avg_game_length': self.progress.avg_game_length,
            'avg_move_time': self.progress.avg_move_time,
            'model_loss': self.progress.latest_model_loss,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.progress.performance_milestones is None:
            self.progress.performance_milestones = []
        self.progress.performance_milestones.append(milestone)
        
        logger.info(f"Performance at {total_games} games: Win rate: {win_rate:.3f}, "
                   f"Draw rate: {draw_rate:.3f}, Avg length: {self.progress.avg_game_length:.1f}")
    
    def _save_final_model(self):
        """Save final trained model"""
        final_dir = self.save_directory / "final_model"
        final_dir.mkdir(exist_ok=True)
        
        model_path = final_dir / "thinking_brain_final.pth"
        self.thinking_brain.save_model(str(model_path))
        
        # Save final statistics
        stats_path = final_dir / "training_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(asdict(self.progress), f, indent=2)
        
        logger.info(f"Final model saved to {final_dir}")
    
    def get_heatmap_data(self) -> Dict:
        """Get current heatmap data for visualization (Phase 5)"""
        return self.position_heatmap_data.copy()


def test_self_play_training():
    """Test the self-play training system"""
    logger.info("Testing Self-Play Training System...")
    
    # Create AI systems
    thinking_brain = ThinkingBrain()
    gameplay_brain = GameplayBrain(
        population_size=5,  # Small for testing
        simulation_depth=2,
        generations=3,
        time_limit=0.5
    )
    
    # Create trainer
    trainer = SelfPlayTrainer(
        thinking_brain=thinking_brain,
        gameplay_brain=gameplay_brain,
        save_directory="test_models",
        games_per_checkpoint=5,  # Small for testing
        enable_progress_tracking=True
    )
    
    # Run short training
    trainer.train(target_games=10, resume_from_checkpoint=False)
    
    # Check results
    logger.info(f"Training completed!")
    logger.info(f"Games played: {trainer.progress.games_played}")
    logger.info(f"Win rate: {trainer.progress.wins / trainer.progress.games_played:.3f}")
    logger.info(f"Average game length: {trainer.progress.avg_game_length:.1f}")
    
    logger.info("✅ Self-Play Training test completed!")


if __name__ == "__main__":
    test_self_play_training()
