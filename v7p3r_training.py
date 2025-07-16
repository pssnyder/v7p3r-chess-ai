# v7p3r_training.py
"""Training Module for V7P3R Chess AI
Handles training of the V7P3R Chess AI models using reinforcement learning."""

import chess
import pickle
import os
import random
import numpy as np
import time
from collections import deque
from chess_core import ChessConfig, GameState, BoardEvaluator, RewardCalculator, FeatureExtractor
from v7p3r_ai import V7P3RAI, QNetwork
from stockfish_handler import StockfishHandler
from personal_style_analyzer import PersonalStyleAnalyzer


class ExperienceReplay:
    """Experience replay buffer for reinforcement learning"""
    
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of experiences from memory"""
        return random.sample(self.memory, min(len(self.memory), batch_size))
    
    def __len__(self):
        return len(self.memory)


class V7P3RTrainer:
    """Trainer for V7P3R Chess AI using reinforcement learning"""
    
    def __init__(self, config=None):
        self.config = config if config else ChessConfig()
        self.training_config = self.config.get_training_config()
        self.v7p3r_config = self.config.get_v7p3r_config()
        
        # Training parameters
        self.learning_rate = self.training_config.get("learning_rate", 0.001)
        self.discount_factor = self.training_config.get("discount_factor", 0.99)
        self.exploration_rate = self.training_config.get("exploration_rate", 0.1)
        self.exploration_decay = self.training_config.get("exploration_decay", 0.995)
        self.min_exploration_rate = self.training_config.get("min_exploration_rate", 0.01)
        self.batch_size = self.training_config.get("batch_size", 64)
        self.episodes = self.training_config.get("episodes", 1000)
        self.target_update_frequency = self.training_config.get("target_update_frequency", 10)
        self.save_frequency = self.training_config.get("save_frequency", 100)
        self.validation_frequency = self.training_config.get("validation_frequency", 50)
        
        # Initialize components
        self.feature_extractor = FeatureExtractor()
        self.reward_calculator = RewardCalculator(self.config)
        self.replay_buffer = ExperienceReplay(self.training_config.get("memory_size", 10000))
        
        # Initialize personal style analyzer
        self.personal_style = PersonalStyleAnalyzer(self.config)
        
        # Initialize model
        self.model_path = self.v7p3r_config.get("model_path", "models/v7p3r_model.pkl")
        self._init_model()
        
        # Target network for stability
        self.target_model = self._clone_model(self.model)
        
        # Initialize opponents
        self.stockfish = StockfishHandler(self.config)
    
    def _init_model(self):
        """Initialize or load Q-network model"""
        # Check if model directory exists
        model_dir = os.path.dirname(self.model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Try to load existing model
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"Model loaded from {self.model_path}")
                return
            except Exception as e:
                print(f"Error loading model: {e}")
        
        # Create new model if loading fails
        # Feature size is 8x8x12 (board) + 14 (additional features)
        feature_size = 8 * 8 * 12 + 14
        self.model = QNetwork(feature_size, hidden_size=256)
        print("New model created")
    
    def _clone_model(self, model):
        """Create a copy of the model for target network"""
        new_model = QNetwork(model.input_size, model.hidden_size)
        new_model.weights1 = model.weights1.copy()
        new_model.bias1 = model.bias1.copy()
        new_model.weights2 = model.weights2.copy()
        new_model.bias2 = model.bias2.copy()
        return new_model
    
    def _update_target_model(self):
        """Update target model with current model weights"""
        self.target_model = self._clone_model(self.model)
    
    def _save_model(self):
        """Save model to disk"""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Model saved to {self.model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def train(self):
        """Main training loop using reinforcement learning"""
        print("Starting V7P3R Chess AI training...")
        
        # Statistics
        episode_rewards = []
        win_count = 0
        draw_count = 0
        loss_count = 0
        
        for episode in range(1, self.episodes + 1):
            print(f"Episode {episode}/{self.episodes}")
            
            # Reset game state
            game_state = GameState(self.config)
            total_reward = 0
            done = False
            
            # Decide who plays white (V7P3R plays white every other game)
            v7p3r_plays_white = (episode % 2 == 1)
            
            # Initialize state
            board = game_state.get_board_state()
            state_features = self.feature_extractor.extract_features(board)
            
            # Game loop
            move_count = 0
            while not done:
                # Determine whose turn it is
                is_v7p3r_turn = (board.turn == chess.WHITE and v7p3r_plays_white) or \
                               (board.turn == chess.BLACK and not v7p3r_plays_white)
                
                if is_v7p3r_turn:
                    # V7P3R's turn
                    
                    # Exploration vs exploitation
                    if random.random() < self.exploration_rate:
                        # Random move
                        legal_moves = list(board.legal_moves)
                        action = random.choice(legal_moves) if legal_moves else None
                    else:
                        # Best move according to model
                        legal_moves = list(board.legal_moves)
                        best_action = None
                        best_q_value = float('-inf')
                        
                        for move in legal_moves:
                            # Make move on a copy of the board
                            board_copy = board.copy()
                            board_copy.push(move)
                            
                            # Get features of next state
                            next_features = self.feature_extractor.extract_features(board_copy)
                            
                            # Get Q-value from model
                            q_value = self.model.predict(next_features)[0]
                            
                            if q_value > best_q_value:
                                best_q_value = q_value
                                best_action = move
                        
                        action = best_action
                    
                    # Make the move
                    if action:
                        # Create a temporary state for the transition
                        temp_state = GameState(self.config)
                        temp_state.board = board.copy()
                        
                        # Make the move on the real board
                        game_state.make_move(action)
                        
                        # Get new board state
                        new_board = game_state.get_board_state()
                        next_state_features = self.feature_extractor.extract_features(new_board)
                        
                        # Check if game is over
                        done = game_state.is_game_over()
                        
                        # Create a temporary next state for reward calculation
                        temp_next_state = GameState(self.config)
                        temp_next_state.board = new_board.copy()
                        
                        # Calculate reward
                        reward = self.reward_calculator.calculate_reward(temp_state, action, temp_next_state)
                        
                        # Add bonus reward for moves that match personal style
                        game_phase = self.personal_style.detect_game_phase(board)
                        personal_move = self.personal_style.get_personal_move(board, phase=game_phase)
                        
                        # Apply style influence bonus based on game phase
                        style_weights = self.v7p3r_config.get("personal_style_weights", {})
                        
                        if personal_move and personal_move == action:
                            # Personal style bonus
                            if game_phase == "opening":
                                bonus = style_weights.get("opening_style_influence", 0.8) * 2.0
                            elif game_phase == "middlegame":
                                bonus = style_weights.get("middlegame_style_influence", 0.6) * 2.0
                            else:  # endgame
                                bonus = style_weights.get("endgame_style_influence", 0.7) * 2.0
                                
                            print(f"Applied personal style bonus: +{bonus:.2f} for {action.uci()} in {game_phase}")
                            reward += bonus
                            
                        # Check if the move leads to a pattern similar to a winning/checkmate sequence
                        checkmate_move = self.personal_style.check_for_checkmate_pattern(board)
                        if checkmate_move and checkmate_move == action:
                            checkmate_bonus = style_weights.get("checkmate_patterns_influence", 1.0) * 4.0
                            print(f"Applied checkmate pattern bonus: +{checkmate_bonus:.2f} for {action.uci()}")
                            reward += checkmate_bonus
                        
                        # Check for winning sequence patterns
                        winning_move = self.personal_style.check_for_winning_sequence(board)
                        if winning_move and winning_move == action:
                            winning_bonus = style_weights.get("winning_patterns_influence", 0.9) * 3.0
                            print(f"Applied winning sequence bonus: +{winning_bonus:.2f} for {action.uci()}")
                            reward += winning_bonus
                            
                        total_reward += reward
                        
                        # Store experience
                        self.replay_buffer.add(
                            state_features,
                            action,
                            reward,
                            next_state_features,
                            done
                        )
                        
                        # Update state
                        state_features = next_state_features
                        
                        # Train on batch if enough samples
                        if len(self.replay_buffer) >= self.batch_size:
                            self._train_on_batch()
                    else:
                        # No legal moves
                        done = True
                else:
                    # Opponent's turn (Stockfish)
                    action = self.stockfish.get_move(board)
                    if action:
                        game_state.make_move(action)
                        
                        # Get new board state
                        new_board = game_state.get_board_state()
                        next_state_features = self.feature_extractor.extract_features(new_board)
                        
                        # Check if game is over
                        done = game_state.is_game_over()
                        
                        # Update state
                        state_features = next_state_features
                    else:
                        # No legal moves
                        done = True
                
                # Update board reference
                board = game_state.get_board_state()
                
                # Safety check for max moves
                move_count += 1
                if move_count > 200:  # Prevent infinite games
                    done = True
                    print("Game terminated due to move limit")
            
            # End of episode
            
            # Update statistics
            episode_rewards.append(total_reward)
            result = game_state.get_result()
            
            if result == "1-0" and v7p3r_plays_white:
                win_count += 1
                print("V7P3R won as White")
            elif result == "0-1" and not v7p3r_plays_white:
                win_count += 1
                print("V7P3R won as Black")
            elif result == "1/2-1/2":
                draw_count += 1
                print("Game ended in draw")
            else:
                loss_count += 1
                print("V7P3R lost")
            
            # Decay exploration rate
            self.exploration_rate = max(
                self.min_exploration_rate,
                self.exploration_rate * self.exploration_decay
            )
            
            # Update target network periodically
            if episode % self.target_update_frequency == 0:
                self._update_target_model()
                print("Target network updated")
            
            # Save model periodically
            if episode % self.save_frequency == 0:
                self._save_model()
            
            # Print progress
            print(f"Episode {episode}: Reward={total_reward:.2f}, Exploration={self.exploration_rate:.4f}")
            print(f"Stats - Wins: {win_count}, Draws: {draw_count}, Losses: {loss_count}")
            print(f"Win rate: {win_count / episode:.2%}")
        
        # Final save
        self._save_model()
        
        print("Training complete!")
        print(f"Final stats - Wins: {win_count}, Draws: {draw_count}, Losses: {loss_count}")
        print(f"Final win rate: {win_count / self.episodes:.2%}")
    
    def _train_on_batch(self):
        """Train model on a batch of experiences"""
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        
        for state_features, action, reward, next_state_features, done in batch:
            # Get current Q-value prediction
            current_q = self.model.predict(state_features)[0]
            
            # Calculate target Q-value
            if done:
                # Terminal state
                target_q = reward
            else:
                # Non-terminal state: Q(s,a) = r + γ * max(Q(s',a'))
                next_q = self.target_model.predict(next_state_features)[0]
                target_q = reward + self.discount_factor * next_q
            
            # Create target for training
            target = np.array([[target_q]])
            
            # Update model
            self.model.backward(np.array(state_features).reshape(1, -1), target, self.learning_rate)


def create_directories():
    """Create necessary directories for training"""
    directories = [
        "models",
        "data"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")


if __name__ == "__main__":
    # Create directories
    create_directories()
    
    # Initialize config
    config = ChessConfig()
    
    # Create and run trainer
    trainer = V7P3RTrainer(config)
    trainer.train()