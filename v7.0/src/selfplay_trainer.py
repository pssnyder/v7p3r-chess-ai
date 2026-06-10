"""
V7P3R v7.0 - Self-Play Training System

Plays games, queries Stockfish oracle, applies personality rewards,
trains neural network. Monitors performance and learning progress.
"""

import chess
import chess.pgn
import numpy as np
import torch
import json
import time
import psutil
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

from comprehensive_features import ComprehensiveFeatureExtractor
from network import V7ValueNetwork, V7Trainer, create_v7_network
from stockfish_oracle import StockfishOracle
from personality_rewards import PersonalityRewardCalculator, PersonalityWeights
from personality_tuner import PlaystyleProfile
from phase_manager import DynamicWeightCalculator, PhaseAwareTrainingTarget, GamePhase
from opening_book import OpeningBookManager
from tablebase_oracle import TablebaseOracle


@dataclass
class GameExperience:
    """Single position experience from self-play."""
    fen: str
    features: np.ndarray  # 55-dim features (51 positional + 4 temporal)
    move_played: str  # UCI format
    stockfish_eval: float  # Normalized [-1, 1]
    personality_reward: float
    game_outcome: float  # 1.0 = win, 0.0 = draw, -1.0 = loss (from current player perspective)
    move_number: int
    
    # Feature tracking for analysis
    forest_darkness: float
    material_balance: float
    complexity_reward: float
    sacrifice_reward: float
    
    # NEW: Phase-aware training
    game_phase: str  # opening, middlegame, endgame, tablebase
    training_target: float  # Actual target value (phase-weighted)
    stockfish_weight: float  # Weight used for Stockfish
    personality_weight: float  # Weight used for personality
    tablebase_eval: Optional[float] = None  # Tablebase eval if available
    
    # NEW v7.2: Time management
    inference_time_ms: float = 0.0  # How long this move took to compute


@dataclass
class GameResult:
    """Complete game result with statistics."""
    game_number: int
    result: str  # "1-0", "0-1", "1/2-1/2"
    num_moves: int
    duration_seconds: float
    avg_stockfish_eval: float
    avg_personality_reward: float
    avg_forest_darkness: float
    total_sacrifices: int
    final_material: int
    termination: str  # "checkmate", "draw", "resignation"
    pgn: str
    
    # NEW: Phase-aware stats
    opening_name: Optional[str] = None  # Which opening was used
    tablebase_positions: int = 0  # Number of TB-consulted positions
    phase_distribution: Optional[Dict[str, int]] = None  # Count per phase
    final_piece_count: int = 0  # Pieces remaining at game end
    final_halfmove_clock: int = 0  # Halfmove clock at game end (50-move rule)


@dataclass
class TrainingMetrics:
    """Performance metrics for training session."""
    game_number: int
    timestamp: str
    
    # Speed metrics
    game_duration_seconds: float
    positions_per_second: float
    stockfish_time_ms: float
    
    # Resource usage
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    
    # Learning metrics
    avg_loss: float
    network_output_range: Tuple[float, float]
    
    # Game quality metrics
    avg_forest_darkness: float
    avg_personality_reward: float
    win_rate: float
    draw_rate: float


class ExperienceBuffer:
    """Stores and manages self-play experiences."""
    
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.experiences: List[GameExperience] = []
        self.game_results: List[GameResult] = []
    
    def add_game(self, experiences: List[GameExperience], result: GameResult):
        """Add experiences from a complete game."""
        self.experiences.extend(experiences)
        self.game_results.append(result)
        
        # Trim if over max size (remove oldest)
        if len(self.experiences) > self.max_size:
            self.experiences = self.experiences[-self.max_size:]
    
    def sample_batch(self, batch_size: int) -> List[GameExperience]:
        """Sample random batch for training."""
        if len(self.experiences) < batch_size:
            return self.experiences
        
        indices = np.random.choice(len(self.experiences), batch_size, replace=False)
        return [self.experiences[i] for i in indices]
    
    def get_all_experiences(self) -> List[GameExperience]:
        """Get all experiences."""
        return self.experiences
    
    def get_statistics(self) -> Dict:
        """Get buffer statistics."""
        if not self.experiences:
            return {}
        
        return {
            'total_experiences': len(self.experiences),
            'total_games': len(self.game_results),
            'avg_forest_darkness': float(np.mean([e.forest_darkness for e in self.experiences])),
            'avg_personality_reward': float(np.mean([e.personality_reward for e in self.experiences])),
            'total_sacrifices': sum(r.total_sacrifices for r in self.game_results),
            'win_rate': float(sum(1 for r in self.game_results if r.result in ['1-0', '0-1']) / len(self.game_results)),
            'draw_rate': float(sum(1 for r in self.game_results if r.result == '1/2-1/2') / len(self.game_results)),
        }


class SelfPlayGame:
    """Plays a single self-play game."""
    
    def __init__(
        self,
        network: V7ValueNetwork,
        oracle: StockfishOracle,
        calculator: PersonalityRewardCalculator,
        extractor: ComprehensiveFeatureExtractor,
        phase_manager: PhaseAwareTrainingTarget,
        max_moves: int = 200,
        temperature: float = 0.3,
        opening_book: Optional[OpeningBookManager] = None,
        tablebase_oracle: Optional[TablebaseOracle] = None
    ):
        self.network = network
        self.oracle = oracle
        self.calculator = calculator
        self.extractor = extractor
        self.phase_manager = phase_manager
        self.max_moves = max_moves
        self.temperature = temperature
        self.opening_book = opening_book
        self.tablebase_oracle = tablebase_oracle
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)
    
    def select_move(self, board: chess.Board) -> chess.Move:
        """
        Select move using tablebase (if available) or network evaluation + exploration.
        
        Priority:
        1. Tablebase move (if position is in tablebase)
        2. Network evaluation with softmax exploration
        """
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            return None
        
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        # CRITICAL FIX: Check tablebase first!
        # This prevents 190-move games where engine can't convert won tablebase positions
        if self.tablebase_oracle and self.tablebase_oracle.is_available(board):
            tablebase_move = self.tablebase_oracle.get_best_move(board)
            if tablebase_move is not None:
                # DEBUG: Log tablebase usage (remove this in production)
                if not hasattr(self, '_tb_move_count'):
                    self._tb_move_count = 0
                self._tb_move_count += 1
                return tablebase_move
        
        # Set network to eval mode for inference
        self.network.eval()
        
        # Evaluate each move
        move_values = []
        for move in legal_moves:
            board_copy = board.copy()
            board_copy.push(move)
            
            # Extract features
            features = self.extractor.extract_all_features(board_copy)
            
            # Network evaluation
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                value = self.network(features_tensor).item()
            
            # Negate if black to move (network trained from white perspective)
            if board.turn == chess.BLACK:
                value = -value
            
            move_values.append(value)
        
        # Softmax selection with temperature
        move_values = np.array(move_values)
        if self.temperature > 0:
            exp_values = np.exp(move_values / self.temperature)
            probabilities = exp_values / exp_values.sum()
            selected_idx = np.random.choice(len(legal_moves), p=probabilities)
        else:
            selected_idx = np.argmax(move_values)
        
        return legal_moves[selected_idx]
    
    def play_game(self, game_number: int) -> Tuple[List[GameExperience], GameResult]:
        """
        Play a complete self-play game.
        
        Returns:
            (experiences, result)
        """
        board = chess.Board()
        experiences = []
        
        start_time = time.time()
        stockfish_times = []
        
        move_number = 0
        previous_features = None
        opening_name = None
        forced_opening_moves = []
        result_str = None  # Will be set if early resignation happens
        previous_inference_ms = 0.0  # Track move selection timing
        
        # NEW: Apply opening book if available (fast-forward to interesting position)
        if self.opening_book:
            opening_line, opening_moves = self.opening_book.apply_random_opening(board)
            opening_name = opening_line.name
            forced_opening_moves = opening_moves
            move_number = len(opening_moves)
            
            # Still extract features from forced opening moves for analysis
            # but don't train the network on these (they're forced, not learned)
            # We DO want to track them for proper phase detection though
        
        while not board.is_game_over() and move_number < self.max_moves:
            move_number += 1
            
            # Select and make move (TIME THIS for temporal features)
            inference_start = time.time()
            move = self.select_move(board)
            inference_ms = (time.time() - inference_start) * 1000
            
            if move is None:
                break
            
            board.push(move)
            
            # Extract features after move (NOW WITH TEMPORAL DATA)
            features = self.extractor.extract_all_features(
                board, 
                move_number=move_number,
                previous_inference_ms=previous_inference_ms
            )
            features_dict = self.extractor.extract_all_features_dict(board)
            
            # Store inference time for next iteration
            previous_inference_ms = inference_ms
            
            # Query Stockfish oracle
            sf_start = time.time()
            sf_result = self.oracle.evaluate(board)
            stockfish_times.append((time.time() - sf_start) * 1000)
            
            # EARLY RESIGNATION: If position is hopeless (-5.0 pawns for 20 moves), resign
            # This prevents endless shuffling in lost positions
            if not hasattr(self, '_resign_counter'):
                self._resign_counter = 0
                self._last_eval = 0.0
            
            if sf_result.normalized_score < -0.5:  # Losing by 5 pawns or more
                self._resign_counter += 1
                if self._resign_counter >= 20:  # Consistently losing for 20 moves
                    # Force resignation
                    result_str = "0-1" if board.turn == chess.WHITE else "1-0"
                    break
            else:
                self._resign_counter = 0  # Reset if position improves
            
            # DRAW DETECTION: Enforce 100-move rule (stricter than FIDE's 75 for training)
            # This prevents endless shuffling in drawn positions
            if board.halfmove_clock >= 100:
                result_str = "1/2-1/2"
                break
            
            # Calculate personality reward
            reward_result = self.calculator.calculate_total_reward(
                features_dict,
                previous_features,
                stockfish_eval=sf_result.normalized_score
            )
            
            # NEW: Check if tablebase evaluation available
            tablebase_eval = None
            if self.tablebase_oracle and self.tablebase_oracle.is_available(board):
                tablebase_eval = self.tablebase_oracle.get_normalized_eval(board)
            
            # NEW: Calculate phase-aware training target
            # This uses dynamic weights based on game phase
            training_target, weight_info = self.phase_manager.calculate_target(
                board=board,
                move_number=move_number,
                stockfish_eval=sf_result.normalized_score,
                personality_reward=reward_result['personality_total'],
                game_outcome=0.0,  # Will be backfilled after game ends
                tablebase_eval=tablebase_eval
            )
            
            # Store experience (outcome will be backfilled later)
            experience = GameExperience(
                fen=board.fen(),
                features=features,
                move_played=move.uci(),
                stockfish_eval=sf_result.normalized_score,
                personality_reward=reward_result['personality_total'],
                game_outcome=0.0,  # Will be set after game ends
                move_number=move_number,
                forest_darkness=features_dict['forest_darkness_score'],
                material_balance=features_dict['material_balance'],
                complexity_reward=reward_result['complexity_reward'],
                sacrifice_reward=reward_result['sacrifice_reward'],
                # NEW: Phase-aware fields
                game_phase=weight_info['phase'],
                training_target=training_target,
                stockfish_weight=weight_info['stockfish'],
                personality_weight=weight_info['personality'],
                tablebase_eval=tablebase_eval,
                # NEW v7.2: Time management
                inference_time_ms=inference_ms
            )
            experiences.append(experience)
            
            previous_features = features_dict
        
        # Determine game outcome
        result_str = board.result()
        
        if result_str == "1-0":
            outcome_white = 1.0
        elif result_str == "0-1":
            outcome_white = -1.0
        else:
            outcome_white = 0.0
        
        # Backfill outcomes and recalculate training targets
        # (from perspective of player who made the move)
        for i, exp in enumerate(experiences):
            # Reconstruct board to move i
            temp_board = chess.Board()
            
            # Apply opening moves if they were used
            if forced_opening_moves:
                for opening_move in forced_opening_moves:
                    temp_board.push(opening_move)
            
            # Apply game moves up to this experience
            for j in range(i + 1):
                temp_board.push(chess.Move.from_uci(experiences[j].move_played))
            
            # Outcome from perspective of player who just moved
            if temp_board.turn == chess.WHITE:  # Black just moved
                exp.game_outcome = -outcome_white
            else:  # White just moved
                exp.game_outcome = outcome_white
            
            # NEW: Recalculate training target with final outcome
            training_target, _ = self.phase_manager.calculate_target(
                board=temp_board,
                move_number=exp.move_number,
                stockfish_eval=exp.stockfish_eval,
                personality_reward=exp.personality_reward,
                game_outcome=exp.game_outcome,  # Now has real value
                tablebase_eval=exp.tablebase_eval
            )
            exp.training_target = training_target
        
        # Game statistics
        duration = time.time() - start_time
        
        termination = "checkmate" if board.is_checkmate() else \
                     "draw" if board.is_stalemate() or board.is_insufficient_material() else \
                     "fifty_moves" if board.is_fifty_moves() else \
                     "repetition" if board.is_repetition() else \
                     "max_moves"
        
        # Create PGN
        game = chess.pgn.Game()
        game.headers["Event"] = f"V7P3R SelfPlay Game {game_number}"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["White"] = "V7P3R-DarkForest"
        game.headers["Black"] = "V7P3R-DarkForest"
        game.headers["Result"] = result_str
        
        node = game
        temp_board = chess.Board()
        
        # Apply opening moves first if they were used
        if forced_opening_moves:
            for opening_move in forced_opening_moves:
                node = node.add_variation(opening_move)
                temp_board.push(opening_move)
        
        # Then apply the game moves
        for exp in experiences:
            move = chess.Move.from_uci(exp.move_played)
            node = node.add_variation(move)
            temp_board.push(move)
        
        # NEW: Calculate phase distribution for analysis
        phase_distribution = {}
        for exp in experiences:
            phase = exp.game_phase
            phase_distribution[phase] = phase_distribution.get(phase, 0) + 1
        
        # NEW: Count tablebase consultations
        tablebase_positions = sum(1 for exp in experiences if exp.tablebase_eval is not None)
        
        # Count final pieces (to understand why games hit max moves)
        final_piece_count = len(board.piece_map())
        final_halfmove_clock = board.halfmove_clock
        
        result = GameResult(
            game_number=game_number,
            result=result_str,
            num_moves=len(experiences),
            duration_seconds=duration,
            avg_stockfish_eval=float(np.mean([e.stockfish_eval for e in experiences])),
            avg_personality_reward=float(np.mean([e.personality_reward for e in experiences])),
            avg_forest_darkness=float(np.mean([e.forest_darkness for e in experiences])),
            total_sacrifices=sum(1 for e in experiences if e.sacrifice_reward > 0.01),
            final_material=int(experiences[-1].material_balance) if experiences else 0,
            termination=termination,
            pgn=str(game),
            # NEW: Phase-aware fields
            opening_name=opening_name,
            tablebase_positions=tablebase_positions,
            phase_distribution=phase_distribution,
            final_piece_count=final_piece_count,
            final_halfmove_clock=final_halfmove_clock
        )
        
        return experiences, result


class SelfPlayTrainer:
    """Main self-play training orchestrator."""
    
    def __init__(
        self,
        profile_path: str,
        stockfish_path: str,
        output_dir: str = "../../training/v7_selfplay",
        max_buffer_size: int = 100000,
        opening_book_pgn: Optional[str] = None,
        tablebase_path: Optional[str] = None,
        use_opening_book: bool = True,
        use_tablebases: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load personality profile
        with open(profile_path, 'r') as f:
            profile_data = json.load(f)
        self.profile = PlaystyleProfile.from_dict(profile_data)
        
        # Initialize components
        self.extractor = ComprehensiveFeatureExtractor()
        self.calculator = PersonalityRewardCalculator(self.profile.weights)
        self.oracle = StockfishOracle(stockfish_path)
        
        # NEW: Phase-aware weighting system (v7.1 REVISED)
        self.phase_manager = PhaseAwareTrainingTarget(DynamicWeightCalculator(
            opening_sf_weight=0.9,      # 90% SF in opening
            middlegame_sf_weight=0.2,   # 20% SF in middlegame (controlled chaos, up from 10%)
            endgame_sf_weight=1.0,      # 100% SF in endgame (perfect technique, up from 50%)
            tablebase_sf_weight=1.0     # 100% when tablebase available
        ))
        
        # NEW: Opening book (fast-forward to interesting positions)
        self.use_opening_book = use_opening_book
        if use_opening_book:
            self.opening_book = OpeningBookManager(opening_book_pgn)
            print(f"[OK] Opening book loaded: {len(self.opening_book)} lines")
        else:
            self.opening_book = None
            print("[INFO] Opening book disabled")
        
        # NEW: Tablebase oracle (perfect endgame play)
        self.use_tablebases = use_tablebases
        if use_tablebases:
            self.tablebase_oracle = TablebaseOracle(tablebase_path)
            if self.tablebase_oracle.enabled:
                print(f"[OK] Tablebases loaded: {self.tablebase_oracle.max_pieces}-piece")
            else:
                print("[WARN] Tablebases requested but not available")
                self.use_tablebases = False
        else:
            self.tablebase_oracle = None
            print("[INFO] Tablebases disabled")
        
        # Create network (create_v7_network returns tuple)
        self.network, self.trainer = create_v7_network()
        
        # Experience buffer
        self.buffer = ExperienceBuffer(max_size=max_buffer_size)
        
        # Metrics tracking
        self.metrics: List[TrainingMetrics] = []
        
        # Performance monitoring
        self.process = psutil.Process()
    
    def train_from_selfplay(
        self,
        num_games: int,
        batch_size: int = 256,
        epochs_per_game: int = 1,
        train_every_n_games: int = 10
    ):
        """
        Main training loop: play games, collect data, train network.
        
        Args:
            num_games: Number of self-play games
            batch_size: Training batch size
            epochs_per_game: Training epochs per game batch
            train_every_n_games: Train network every N games
        """
        print("="*60)
        print("V7P3R SELF-PLAY TRAINING")
        print("="*60)
        print(f"Profile: {self.profile.name}")
        print(f"Games: {num_games}")
        print(f"Batch Size: {batch_size}")
        print(f"Train Every: {train_every_n_games} games")
        print(f"Output: {self.output_dir}")
        print("="*60)
        
        self.oracle.start()
        
        try:
            session_start = time.time()
            
            for game_num in range(1, num_games + 1):
                # Play game
                game_start = time.time()
                
                game_player = SelfPlayGame(
                    self.network,
                    self.oracle,
                    self.calculator,
                    self.extractor,
                    self.phase_manager,  # NEW: Phase-aware weighting
                    temperature=0.3,     # Exploration
                    opening_book=self.opening_book if self.use_opening_book else None,
                    tablebase_oracle=self.tablebase_oracle if self.use_tablebases else None
                )
                
                experiences, result = game_player.play_game(game_num)
                self.buffer.add_game(experiences, result)
                
                game_duration = time.time() - game_start
                
                # Print game summary
                print(f"\n{'='*60}")
                print(f"GAME {game_num}/{num_games} - {result.result}")
                print(f"{'='*60}")
                print(f"Moves: {result.num_moves}")
                print(f"Duration: {game_duration:.1f}s ({result.num_moves/game_duration:.1f} moves/sec)")
                print(f"Termination: {result.termination}")
                print(f"Avg Forest Darkness: {result.avg_forest_darkness:.3f}")
                print(f"Avg Personality Reward: {result.avg_personality_reward:+.3f}")
                print(f"Avg Stockfish Eval: {result.avg_stockfish_eval:+.3f}")
                print(f"Sacrifices Made: {result.total_sacrifices}")
                
                # Save PGN
                pgn_file = self.output_dir / f"game_{game_num:04d}.pgn"
                with open(pgn_file, 'w') as f:
                    f.write(result.pgn)
                
                # Train network periodically
                if game_num % train_every_n_games == 0 and len(self.buffer.experiences) >= batch_size:
                    print(f"\n{'─'*60}")
                    print(f"TRAINING NETWORK (Games {game_num-train_every_n_games+1}-{game_num})")
                    print(f"{'─'*60}")
                    
                    train_start = time.time()
                    avg_loss = self._train_on_buffer(batch_size, epochs_per_game)
                    train_duration = time.time() - train_start
                    
                    print(f"Training Duration: {train_duration:.1f}s")
                    print(f"Avg Loss: {avg_loss:.4f}")
                    
                    # Collect metrics
                    metrics = self._collect_metrics(
                        game_num,
                        game_duration,
                        len(experiences),
                        avg_loss
                    )
                    self.metrics.append(metrics)
                    
                    # Print performance stats
                    print(f"CPU: {metrics.cpu_percent:.1f}%")
                    print(f"Memory: {metrics.memory_mb:.0f} MB ({metrics.memory_percent:.1f}%)")
                    print(f"Positions/sec: {metrics.positions_per_second:.1f}")
                    
                    # Save checkpoint
                    self._save_checkpoint(game_num)
            
            # Final training on all data
            print(f"\n{'='*60}")
            print("FINAL TRAINING PASS")
            print(f"{'='*60}")
            final_loss = self._train_on_buffer(batch_size, epochs=5)
            print(f"Final Loss: {final_loss:.4f}")
            
            # Save final model
            self._save_checkpoint(num_games, final=True)
            
            # Generate summary report
            self._generate_report(num_games, time.time() - session_start)
        
        finally:
            self.oracle.stop()
    
    def _train_on_buffer(self, batch_size: int, epochs: int = 1) -> float:
        """Train network on experiences in buffer."""
        experiences = self.buffer.get_all_experiences()
        
        # Prepare training data
        features = np.array([e.features for e in experiences])
        
        # NEW: Use phase-aware training targets (already calculated per experience)
        # Each experience now has pre-calculated training_target with dynamic weights
        targets = np.array([e.training_target for e in experiences])
        
        # Split data for validation (80/20)
        split_idx = int(len(features) * 0.8)
        train_features = features[:split_idx]
        train_targets = targets[:split_idx]
        val_features = features[split_idx:]
        val_targets = targets[split_idx:]
        
        # Train
        history = self.trainer.fit(
            train_features,
            train_targets,
            val_features=val_features,
            val_targets=val_targets,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0
        )
        
        return np.mean(history['train_loss'][-5:]) if history['train_loss'] else 0.0
    
    def _collect_metrics(
        self,
        game_number: int,
        game_duration: float,
        num_positions: int,
        avg_loss: float
    ) -> TrainingMetrics:
        """Collect performance metrics."""
        # System metrics
        cpu_percent = self.process.cpu_percent()
        mem_info = self.process.memory_info()
        memory_mb = mem_info.rss / 1024 / 1024
        memory_percent = self.process.memory_percent()
        
        # Buffer stats
        stats = self.buffer.get_statistics()
        
        # Network output range
        with torch.no_grad():
            sample_features = np.array([e.features for e in self.buffer.experiences[:100]])
            sample_tensor = torch.FloatTensor(sample_features).to(self.network.input_layer.weight.device)
            outputs = self.network(sample_tensor).cpu().numpy()
            output_range = (float(outputs.min()), float(outputs.max()))
        
        return TrainingMetrics(
            game_number=game_number,
            timestamp=datetime.now().isoformat(),
            game_duration_seconds=game_duration,
            positions_per_second=num_positions / game_duration if game_duration > 0 else 0,
            stockfish_time_ms=0.0,  # Would need to track separately
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            memory_percent=memory_percent,
            avg_loss=avg_loss,
            network_output_range=output_range,
            avg_forest_darkness=stats.get('avg_forest_darkness', 0.0),
            avg_personality_reward=stats.get('avg_personality_reward', 0.0),
            win_rate=stats.get('win_rate', 0.0),
            draw_rate=stats.get('draw_rate', 0.0)
        )
    
    def _save_checkpoint(self, game_number: int, final: bool = False):
        """Save model checkpoint and metadata."""
        suffix = "final" if final else f"game_{game_number:04d}"
        
        # Save model (simplified - just save state dict)
        model_path = self.output_dir / f"model_{suffix}.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'game_number': game_number,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
        }, str(model_path))
        
        # Save buffer stats
        stats_path = self.output_dir / f"stats_{suffix}.json"
        with open(stats_path, 'w') as f:
            json.dump(self.buffer.get_statistics(), f, indent=2)
        
        print(f"[SAVED] Checkpoint saved: {model_path}")
    
    def _generate_report(self, num_games: int, total_duration: float):
        """Generate comprehensive training report."""
        stats = self.buffer.get_statistics()
        
        report = {
            'session': {
                'profile': self.profile.name,
                'num_games': num_games,
                'total_duration_seconds': total_duration,
                'total_positions': stats['total_experiences'],
                'avg_game_duration': total_duration / num_games,
                'positions_per_second': stats['total_experiences'] / total_duration
            },
            'game_results': {
                'win_rate': stats['win_rate'],
                'draw_rate': stats['draw_rate'],
                'loss_rate': 1 - stats['win_rate'] - stats['draw_rate']
            },
            'personality_emergence': {
                'avg_forest_darkness': stats['avg_forest_darkness'],
                'avg_personality_reward': stats['avg_personality_reward'],
                'total_sacrifices': stats['total_sacrifices'],
                'sacrifices_per_game': stats['total_sacrifices'] / num_games
            },
            'performance': {
                'avg_cpu_percent': np.mean([m.cpu_percent for m in self.metrics]),
                'avg_memory_mb': np.mean([m.memory_mb for m in self.metrics]),
                'avg_positions_per_second': np.mean([m.positions_per_second for m in self.metrics])
            },
            'training': {
                'final_loss': self.metrics[-1].avg_loss if self.metrics else 0.0,
                'network_output_range': self.metrics[-1].network_output_range if self.metrics else (0, 0)
            }
        }
        
        # Save report
        report_path = self.output_dir / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print("TRAINING SESSION SUMMARY")
        print(f"{'='*60}")
        print(f"Profile: {report['session']['profile']}")
        print(f"Games Played: {report['session']['num_games']}")
        print(f"Total Duration: {report['session']['total_duration_seconds']/60:.1f} min")
        print(f"Total Positions: {report['session']['total_positions']}")
        print(f"\n[GAME RESULTS]")
        print(f"  Win Rate: {report['game_results']['win_rate']*100:.1f}%")
        print(f"  Draw Rate: {report['game_results']['draw_rate']*100:.1f}%")
        print(f"  Loss Rate: {report['game_results']['loss_rate']*100:.1f}%")
        print(f"\n[PERSONALITY EMERGENCE]")
        print(f"  Avg Forest Darkness: {report['personality_emergence']['avg_forest_darkness']:.3f}")
        print(f"  Avg Personality Reward: {report['personality_emergence']['avg_personality_reward']:+.3f}")
        print(f"  Total Sacrifices: {report['personality_emergence']['total_sacrifices']}")
        print(f"  Sacrifices/Game: {report['personality_emergence']['sacrifices_per_game']:.1f}")
        print(f"\n[PERFORMANCE]")
        print(f"  Avg Speed: {report['performance']['avg_positions_per_second']:.1f} pos/sec")
        print(f"  Avg CPU: {report['performance']['avg_cpu_percent']:.1f}%")
        print(f"  Avg Memory: {report['performance']['avg_memory_mb']:.0f} MB")
        print(f"\n[TRAINING]")
        print(f"  Final Loss: {report['training']['final_loss']:.4f}")
        print(f"  Network Range: {report['training']['network_output_range']}")
        print(f"\n[OK] Report saved: {report_path}")
        print(f"{'='*60}")


if __name__ == "__main__":
    # Configuration
    PROFILE_PATH = "../profiles/dark_forest_assassin.json"
    STOCKFISH_PATH = r"E:\Programming Stuff\Chess Engines\Tournament Engines\Stockfish\stockfish-windows-x86-64-avx2.exe"
    
    # Create trainer
    trainer = SelfPlayTrainer(
        profile_path=PROFILE_PATH,
        stockfish_path=STOCKFISH_PATH,
        output_dir="../training/v7_selfplay"
    )
    
    # Run training
    trainer.train_from_selfplay(
        num_games=100,
        batch_size=256,
        epochs_per_game=1,
        train_every_n_games=10
    )
