# v7p3r_ai.py
"""Main AI handler for V7P3R Chess AI
Coordinates between different components of the chess AI to facilitate gameplay using the specified model.
Handles move selection when asked for moves by play_game.py.
"""

import chess
import random
import pickle
import os
import numpy as np
import time
from chess_core import ChessConfig, BoardEvaluator, FeatureExtractor
from personal_style_analyzer import PersonalStyleAnalyzer


class V7P3RAI:
    """Main AI class for V7P3R Chess AI"""
    
    def __init__(self, config=None):
        self.config = config if config else ChessConfig()
        self.v7p3r_config = self.config.get_v7p3r_config()
        self.model = None
        self.evaluator = BoardEvaluator(self.config)
        self.feature_extractor = FeatureExtractor()
        self.search_depth = self.v7p3r_config.get("search_depth", 3)
        self.model_path = self.v7p3r_config.get("model_path", "models/v7p3r_model.pkl")
        self.use_personal_history = self.v7p3r_config.get("use_personal_history", True)
        
        # Load personal style analyzer if enabled
        self.personal_style = None
        if self.use_personal_history:
            self.personal_style = PersonalStyleAnalyzer(self.config)
        
        # Load trained model if available
        self._load_model()
    
    def _load_model(self):
        """Load trained model from disk"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"Model loaded from {self.model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None
        else:
            print(f"No model found at {self.model_path}, using static evaluation")
            self.model = None
    
    def get_move(self, board):
        """Get the best move in the given position"""
        if board.is_game_over():
            return None
        
        # Check for personal style moves if enabled
        if self.personal_style:
            # Detect game phase
            game_phase = self.personal_style.detect_game_phase(board)
            
            # Check for checkmate patterns first (highest priority)
            checkmate_move = self.personal_style.check_for_checkmate_pattern(board)
            if checkmate_move:
                print("Using checkmate pattern move")
                return checkmate_move
            
            # Try to get a move from personal history based on game phase
            personal_move = self.personal_style.get_personal_move(board, phase=game_phase)
            if personal_move:
                print(f"Using personal {game_phase} style move")
                return personal_move
        
        # Use model-based evaluation if model is loaded
        if self.model:
            return self._get_model_move(board)
        else:
            # Fallback to static evaluation
            return self._get_static_evaluation_move(board)
    
    def _get_model_move(self, board):
        """Use the trained model to select the best move"""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        # For exploration during training, sometimes choose a random move
        training_config = self.config.get_training_config()
        exploration_rate = training_config.get("exploration_rate", 0.1)
        
        if random.random() < exploration_rate:
            return random.choice(legal_moves)
        
        # Get features for current state
        features = self.feature_extractor.extract_features(board)
        
        # Evaluate each move using the model
        best_move = None
        best_score = float('-inf')
        
        for move in legal_moves:
            # Make the move on a copy of the board
            board_copy = board.copy()
            board_copy.push(move)
            
            # Extract features for the resulting position
            next_features = self.feature_extractor.extract_features(board_copy)
            
            # Get model prediction (Q-value)
            if self.model is not None:
                score = self.model.predict([next_features])[0]
            else:
                # Fallback to static evaluation if model is None
                score = self.evaluator.evaluate(board_copy, chess.WHITE)
            
            # Track best move
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def _get_static_evaluation_move(self, board):
        """Use static evaluation and minimax to select the best move"""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        # Use minimax with alpha-beta pruning
        best_move, _ = self._minimax(board, self.search_depth, float('-inf'), float('inf'), True)
        
        return best_move
    
    def _minimax(self, board, depth, alpha, beta, maximizing_player):
        """Minimax algorithm with alpha-beta pruning"""
        if depth == 0 or board.is_game_over():
            return None, self.evaluator.evaluate(board, chess.WHITE)
        
        legal_moves = list(board.legal_moves)
        
        if maximizing_player:
            best_score = float('-inf')
            best_move = None
            
            for move in legal_moves:
                board.push(move)
                _, score = self._minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                
                if score > best_score:
                    best_score = score
                    best_move = move
                
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            
            return best_move, best_score
        else:
            best_score = float('inf')
            best_move = None
            
            for move in legal_moves:
                board.push(move)
                _, score = self._minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                
                if score < best_score:
                    best_score = score
                    best_move = move
                
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
            
            return best_move, best_score


# Simple neural network model for Q-learning
class QNetwork:
    """Simple neural network for Q-learning"""
    
    def __init__(self, input_size, hidden_size=256):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 1  # Q-value
        
        # Initialize weights with consistent shapes
        self.weights1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.bias1 = np.zeros((1, self.hidden_size))
        self.weights2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.bias2 = np.zeros((1, self.output_size))
        
        # Initialize dummy activation values to avoid None errors
        self.z1 = np.zeros((1, self.hidden_size))
        self.a1 = np.zeros((1, self.hidden_size))
        self.z2 = np.zeros((1, self.output_size))
        self.a2 = np.zeros((1, self.output_size))
    
    def forward(self, X):
        """Forward pass through the network"""
        # Ensure X is a 2D array
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # First layer
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self._relu(self.z1)
        
        # Output layer
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.z2  # Linear activation for Q-value
        
        return self.a2
    
    def backward(self, X, y, learning_rate=0.001):
        """Backward pass for weight updates"""
        # Ensure X and y are 2D arrays
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        
        # Run forward pass if needed
        if self.a1 is None or self.a2 is None:
            self.forward(X)
        
        # Output error
        output_error = y - self.a2
        
        # Output layer gradients
        d_weights2 = np.dot(self.a1.T, output_error)
        d_bias2 = np.sum(output_error, axis=0, keepdims=True)
        
        # Hidden layer error
        hidden_error = np.dot(output_error, self.weights2.T)
        hidden_error = hidden_error * self._relu_derivative(self.z1)
        
        # Hidden layer gradients
        d_weights1 = np.dot(X.T, hidden_error)
        d_bias1 = np.sum(hidden_error, axis=0, keepdims=True)
        
        # Fix common matrix shape issues
        if d_weights2.ndim == 3 and d_weights2.shape[2] == 1:
            d_weights2 = d_weights2.reshape(d_weights2.shape[0], d_weights2.shape[1])
        if d_weights1.ndim == 3 and d_weights1.shape[2] == 1:
            d_weights1 = d_weights1.reshape(d_weights1.shape[0], d_weights1.shape[1])
        if d_bias2.ndim == 3 and d_bias2.shape[2] == 1:
            d_bias2 = d_bias2.reshape(d_bias2.shape[0], d_bias2.shape[1])
        if d_bias1.ndim == 3 and d_bias1.shape[2] == 1:
            d_bias1 = d_bias1.reshape(d_bias1.shape[0], d_bias1.shape[1])
            
        # Ensure output weight shapes match
        if d_weights2.shape != self.weights2.shape:
            try:
                # Try reshaping to match expected dimensions
                d_weights2 = d_weights2.reshape(self.weights2.shape)
            except ValueError:
                # If direct reshape fails, create a compatible gradient matrix
                temp = np.zeros_like(self.weights2)
                if d_weights2.size == self.weights2.size:
                    # If sizes match but shapes don't, flatten and reshape
                    flat_weights = d_weights2.flatten()
                    temp = flat_weights.reshape(self.weights2.shape)
                else:
                    # If sizes don't match, use what we can
                    temp[:d_weights2.shape[0], :d_weights2.shape[1]] = d_weights2
                d_weights2 = temp
                
        # Ensure bias shapes match
        if d_bias2.shape != self.bias2.shape:
            try:
                d_bias2 = d_bias2.reshape(self.bias2.shape)
            except ValueError:
                temp = np.zeros_like(self.bias2)
                if d_bias2.size == self.bias2.size:
                    temp = d_bias2.flatten().reshape(self.bias2.shape)
                else:
                    temp[:d_bias2.shape[0], :d_bias2.shape[1]] = d_bias2
                d_bias2 = temp
                
        # Ensure weights1 shapes match
        if d_weights1.shape != self.weights1.shape:
            try:
                d_weights1 = d_weights1.reshape(self.weights1.shape)
            except ValueError:
                temp = np.zeros_like(self.weights1)
                if d_weights1.size == self.weights1.size:
                    temp = d_weights1.flatten().reshape(self.weights1.shape)
                else:
                    temp[:d_weights1.shape[0], :d_weights1.shape[1]] = d_weights1
                d_weights1 = temp
                
        # Ensure bias1 shapes match
        if d_bias1.shape != self.bias1.shape:
            try:
                d_bias1 = d_bias1.reshape(self.bias1.shape)
            except ValueError:
                temp = np.zeros_like(self.bias1)
                if d_bias1.size == self.bias1.size:
                    temp = d_bias1.flatten().reshape(self.bias1.shape)
                else:
                    temp[:d_bias1.shape[0], :d_bias1.shape[1]] = d_bias1
                d_bias1 = temp
        
        # Update weights
        self.weights2 += learning_rate * d_weights2
        self.bias2 += learning_rate * d_bias2
        self.weights1 += learning_rate * d_weights1
        self.bias1 += learning_rate * d_bias1
    
    def predict(self, X):
        """Predict Q-value for given input"""
        return self.forward(np.array(X))
    
    def _relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        """Derivative of ReLU function"""
        return np.where(x > 0, 1, 0)