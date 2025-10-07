"""
V7P3R AI v3.0 - Neural Feature Conversion System
===============================================

Converts ChessState objects into fixed-size feature vectors for the
Thinking Brain GRU neural network. Handles all the data preprocessing
and normalization for optimal neural network training.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
import chess
from dataclasses import asdict

# Import our ChessState system
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from core.chess_state import ChessState, ChessStateExtractor


class NeuralFeatureConverter:
    """
    Converts ChessState objects to fixed-size neural network input vectors
    
    Enhanced version that includes puzzle performance metadata for richer training signals.
    The converter normalizes all features and integrates historical performance data.
    """
    
    # Feature dimensions
    BOARD_FEATURE_SIZE = 50      # Global board features
    PIECE_FEATURE_SIZE = 20      # Per-piece features  
    NUM_PIECES = 32              # Fixed number of pieces
    PUZZLE_PERFORMANCE_SIZE = 15 # Puzzle performance metadata features
    
    TOTAL_FEATURE_SIZE = BOARD_FEATURE_SIZE + (PIECE_FEATURE_SIZE * NUM_PIECES) + PUZZLE_PERFORMANCE_SIZE  # ~705
    
    def __init__(self, enhanced_puzzle_db=None):
        """Initialize the feature converter with optional puzzle database"""
        self.feature_stats = {}  # For normalization statistics
        self.feature_means = None
        self.feature_stds = None
        self.enhanced_puzzle_db = enhanced_puzzle_db  # For puzzle performance features
        
    def convert_to_features(self, chess_state: ChessState, device: Optional[str] = None) -> torch.Tensor:
        """
        Convert ChessState to neural network input tensor
        
        Args:
            chess_state: Complete chess position state
            device: Target device for tensor (cuda/cpu)
            
        Returns:
            feature_tensor: Normalized feature vector [1, TOTAL_FEATURE_SIZE]
        """
        # Extract board features
        board_features = self._extract_board_features(chess_state.board_features)
        
        # Extract piece features  
        piece_features = self._extract_piece_features(chess_state.pieces)
        
        # Combine all features
        combined_features = np.concatenate([board_features, piece_features])
        
        # Normalize features
        normalized_features = self._normalize_features(combined_features)
        
        # Convert to tensor
        feature_tensor = torch.tensor(
            normalized_features, 
            dtype=torch.float32
        ).unsqueeze(0)  # Add batch dimension
        
        # Move to specified device
        if device:
            feature_tensor = feature_tensor.to(device)
        
        return feature_tensor
    
    def convert_to_features_enhanced(self, chess_state: ChessState, device: Optional[str] = None, 
                                   puzzle_id: Optional[str] = None, stockfish_grading: Optional[Dict] = None) -> torch.Tensor:
        """
        Enhanced feature conversion including puzzle performance metadata
        
        Args:
            chess_state: Complete chess position state
            device: Target device for tensor (cuda/cpu)
            puzzle_id: Optional puzzle ID for performance feature lookup
            stockfish_grading: Optional Stockfish move quality data
            
        Returns:
            feature_tensor: Enhanced feature vector [1, TOTAL_FEATURE_SIZE]
        """
        # Extract basic features
        board_features = self._extract_board_features(chess_state.board_features)
        piece_features = self._extract_piece_features(chess_state.pieces)
        
        # Extract puzzle performance features (new enhancement)
        puzzle_features = self._extract_puzzle_performance_features(puzzle_id, stockfish_grading)
        
        # Combine all features
        combined_features = np.concatenate([board_features, piece_features, puzzle_features])
        
        # Normalize features
        normalized_features = self._normalize_features(combined_features)
        
        # Convert to tensor
        feature_tensor = torch.tensor(
            normalized_features, 
            dtype=torch.float32
        ).unsqueeze(0)  # Add batch dimension
        
        # Move to specified device
        if device:
            feature_tensor = feature_tensor.to(device)
        
        return feature_tensor
    
    def _extract_board_features(self, board_features) -> np.ndarray:
        """Extract and normalize global board features"""
        features = []
        
        # Basic game state (5 features)
        features.append(float(board_features.sideToMove))
        features.extend([float(x) for x in board_features.castlingRights])  # 4 features
        
        # En passant (1 feature, normalized)
        ep_square = board_features.enPassantSquare
        features.append(ep_square / 63.0 if ep_square is not None else 0.0)
        
        # Move clocks (2 features, normalized)
        features.append(min(board_features.halfmoveClock / 50.0, 1.0))  # Cap at 50
        features.append(min(board_features.fullmoveNumber / 200.0, 1.0))  # Cap at 200
        
        # Check status (1 feature)
        features.append(float(board_features.isKingInCheck))
        
        # Threat counts (2 features, normalized)
        threat_counts = board_features.threatCounts
        features.append(threat_counts.get('whiteAttacked', 0) / 64.0)
        features.append(threat_counts.get('blackAttacked', 0) / 64.0)
        
        # Material balance (3 features, normalized)
        material = board_features.materialBalance
        features.append(material.get('whiteTotal', 0) / 39.0)  # Max material ~39
        features.append(material.get('blackTotal', 0) / 39.0)
        features.append((material.get('difference', 0) + 39) / 78.0)  # Range [-39,39] -> [0,1]
        
        # Piece counts (12 features - 6 piece types × 2 colors)
        piece_counts = board_features.pieceCounts
        for color in ['white', 'black']:
            color_counts = piece_counts.get(color, {})
            features.append(color_counts.get('pawn', 0) / 8.0)
            features.append(color_counts.get('knight', 0) / 2.0)
            features.append(color_counts.get('bishop', 0) / 2.0) 
            features.append(color_counts.get('rook', 0) / 2.0)
            features.append(color_counts.get('queen', 0) / 1.0)
            features.append(color_counts.get('king', 0) / 1.0)
        
        # Pawn structure (12 features)
        pawn_structure = board_features.pawnStructure
        for metric in ['isolatedPawns', 'doubledPawns', 'passedPawns', 'connectedPawns', 'pawnIslands', 'backwardPawns']:
            metric_data = pawn_structure.get(metric, {'white': 0, 'black': 0})
            features.append(metric_data.get('white', 0) / 8.0)  # Normalize by max pawns
            features.append(metric_data.get('black', 0) / 8.0)
        
        # King safety (4 features)
        king_safety = board_features.kingSafety
        features.append(king_safety.get('pawnShieldStrength', 0) / 8.0)
        features.append(king_safety.get('escapeSquares', 0) / 8.0)
        features.append(king_safety.get('exposedFiles', 0) / 8.0)
        features.append(king_safety.get('exposedDiagonals', 0) / 8.0)
        
        # Game phase and dynamics (3 features)
        features.append(board_features.gamePhase)  # Already 0-1
        features.append(board_features.symmetrical)  # Already 0-1
        
        # Mobility (2 features)
        mobility = board_features.mobility
        features.append(mobility.get('totalWhite', 0) / 50.0)  # Normalize by reasonable max
        features.append(mobility.get('totalBlack', 0) / 50.0)
        
        # Tactical patterns (6 features)
        interactions = board_features.uniqueInteractions
        features.append(float(interactions.get('isSkewer', False)))
        features.append(float(interactions.get('isFork', False)))
        features.append(float(interactions.get('isPin', False)))
        features.append(float(interactions.get('isDiscoveredAttack', False)))
        features.append(float(interactions.get('isBattery', False)))
        features.append(float(interactions.get('threatensKingThroughPawn', False)))
        
        # Pad to BOARD_FEATURE_SIZE if needed
        while len(features) < self.BOARD_FEATURE_SIZE:
            features.append(0.0)
        
        return np.array(features[:self.BOARD_FEATURE_SIZE], dtype=np.float32)
    
    def _extract_piece_features(self, pieces: List) -> np.ndarray:
        """Extract features for all 32 pieces"""
        all_piece_features = []
        
        for piece in pieces[:self.NUM_PIECES]:  # Ensure exactly 32 pieces
            piece_features = self._extract_single_piece_features(piece)
            all_piece_features.extend(piece_features)
        
        # Pad with inactive pieces if needed
        while len(all_piece_features) < self.PIECE_FEATURE_SIZE * self.NUM_PIECES:
            inactive_features = [0.0] * self.PIECE_FEATURE_SIZE
            all_piece_features.extend(inactive_features)
        
        return np.array(all_piece_features[:self.PIECE_FEATURE_SIZE * self.NUM_PIECES], dtype=np.float32)
    
    def _extract_single_piece_features(self, piece) -> List[float]:
        """Extract features for a single piece"""
        features = []
        
        # Basic piece info (4 features)
        features.append(piece.type / 6.0)  # Piece type normalized
        features.append(float(piece.color))  # Color (0 or 1)
        features.append(piece.currentSquare / 63.0 if piece.currentSquare >= 0 else 0.0)  # Position
        features.append(piece.baseValue / 9.0)  # Value normalized by queen value
        
        # Mobility features (3 features)
        mobility = piece.mobility if hasattr(piece, 'mobility') and piece.mobility else {}
        features.append(mobility.get('legalMoves', 0) / 27.0)  # Queen max mobility
        features.append(mobility.get('attackingMoves', 0) / 27.0)
        features.append(mobility.get('maxUnobstructed', 0) / 27.0)
        
        # Relationship features (7 features)
        relationships = piece.relationships if hasattr(piece, 'relationships') and piece.relationships else {}
        features.append(min(relationships.get('attackedByCount', 0) / 8.0, 1.0))
        features.append(min(relationships.get('defendedByCount', 0) / 8.0, 1.0))
        features.append(float(relationships.get('isPinned', False)))
        features.append(float(relationships.get('attackingHigherValue', False)))
        features.append(float(relationships.get('attackingLowerValue', False)))
        features.append(float(relationships.get('attackingEqualValue', False)))
        features.append(relationships.get('kingDistance', 0) / 7.0)  # Max distance on board
        
        # Positional features (4 features)
        positional = piece.positional if hasattr(piece, 'positional') and piece.positional else {}
        features.append(positional.get('centrality', 0) / 3.5)  # Max distance from center
        features.append(positional.get('edgeProximity', 0) / 3.0)  # Max edge distance
        features.append(float(positional.get('backRankStatus', False)))
        features.append(float(positional.get('seventhRankStatus', False)))
        
        # Vector analysis features (2 features - simplified)
        vector_analysis = piece.vectorAnalysis if hasattr(piece, 'vectorAnalysis') and piece.vectorAnalysis else {}
        features.append(min(sum(vector_analysis.values()) / 20.0, 1.0))  # Total vector activity
        features.append(0.0)  # Reserved for future vector features
        
        # Ensure exactly PIECE_FEATURE_SIZE features
        while len(features) < self.PIECE_FEATURE_SIZE:
            features.append(0.0)
        
        return features
    
    def _extract_puzzle_performance_features(self, puzzle_id: Optional[str], stockfish_grading: Optional[Dict]) -> np.ndarray:
        """
        Extract puzzle performance metadata features for enhanced learning
        
        This creates a "secondary memory" system where the AI can recall its past
        performance on similar positions and use Stockfish quality grading.
        
        Args:
            puzzle_id: ID of current puzzle (for historical lookup)
            stockfish_grading: Current Stockfish evaluation data
            
        Returns:
            puzzle_features: Array of 15 performance-related features
        """
        features = []
        
        # Get historical performance data if available
        if puzzle_id and self.enhanced_puzzle_db:
            perf_data = self.enhanced_puzzle_db.get_puzzle_performance_features(puzzle_id)
        else:
            perf_data = self._get_default_performance_features()
        
        # Historical AI performance features (8 features)
        features.append(min(perf_data['encounter_count'] / 10.0, 1.0))  # Normalize encounter count
        features.append(perf_data['solve_rate'])                       # Solve rate (0-1)
        features.append(perf_data['best_score'] / 5.0)                 # Best score normalized
        features.append(perf_data['average_score'] / 5.0)              # Average score normalized
        features.append(min(perf_data['consecutive_fails'] / 5.0, 1.0)) # Consecutive fails
        features.append(float(perf_data['regression_detected']))        # Regression flag
        features.append(min(perf_data['average_solve_time'] / 30.0, 1.0)) # Solve time normalized
        features.append(perf_data['recent_performance_trend'])          # Trend (-1 to 1)
        
        # Difficulty and context features (3 features)
        features.append(perf_data['difficulty_tier_encoded'])           # Difficulty level
        features.append(perf_data['rating_normalized'])                 # Rating normalized
        features.append(float(perf_data['last_encounter_success']))     # Last success flag
        
        # Stockfish grading features (4 features)
        if stockfish_grading:
            features.append(stockfish_grading.get('score', 0) / 1000.0)  # Stockfish score normalized
            features.append((stockfish_grading.get('rank', 6) - 1) / 5.0) # Rank normalized (1-5 -> 0-0.8)
            features.append(stockfish_grading.get('depth', 15) / 20.0)    # Search depth normalized
            features.append(1.0)  # Stockfish data available flag
        else:
            features.extend([0.0, 0.5, 0.75, 0.0])  # Default values when no Stockfish data
        
        return np.array(features, dtype=np.float32)
    
    def _get_default_performance_features(self) -> Dict:
        """Get default performance features for positions without history"""
        return {
            'encounter_count': 0,
            'solve_rate': 0.0,
            'best_score': 0,
            'average_score': 0.0,
            'consecutive_fails': 0,
            'regression_detected': 0,
            'average_solve_time': 0.0,
            'recent_performance_trend': 0.0,
            'difficulty_tier_encoded': 0.5,  # Medium difficulty default
            'rating_normalized': 0.5,        # Average rating default
            'last_encounter_success': 0,
        }[:self.PIECE_FEATURE_SIZE]
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features for neural network training
        
        For now, features are already normalized during extraction.
        This method can be enhanced with learned statistics later.
        """
        # Apply basic clipping to ensure all features are in reasonable ranges
        normalized = np.clip(features, -3.0, 3.0)
        
        # Optional: Apply learned normalization statistics
        if self.feature_means is not None and self.feature_stds is not None:
            normalized = (normalized - self.feature_means) / (self.feature_stds + 1e-8)
        
        return normalized
    
    def compute_normalization_stats(self, feature_samples: List[np.ndarray]):
        """
        Compute normalization statistics from training data
        
        Args:
            feature_samples: List of feature vectors from training positions
        """
        if not feature_samples:
            return
        
        # Stack all samples
        all_features = np.vstack(feature_samples)
        
        # Compute statistics
        self.feature_means = np.mean(all_features, axis=0)
        self.feature_stds = np.std(all_features, axis=0)
        
        print(f"Computed normalization stats from {len(feature_samples)} samples")
        print(f"Feature means range: [{self.feature_means.min():.3f}, {self.feature_means.max():.3f}]")
        print(f"Feature stds range: [{self.feature_stds.min():.3f}, {self.feature_stds.max():.3f}]")
    
    def save_normalization_stats(self, filepath: str):
        """Save normalization statistics"""
        if self.feature_means is not None and self.feature_stds is not None:
            np.savez(filepath, means=self.feature_means, stds=self.feature_stds)
            print(f"Normalization stats saved to {filepath}")
    
    def load_normalization_stats(self, filepath: str):
        """Load normalization statistics"""
        try:
            data = np.load(filepath)
            self.feature_means = data['means']
            self.feature_stds = data['stds']
            print(f"Normalization stats loaded from {filepath}")
        except Exception as e:
            print(f"Failed to load normalization stats: {e}")


def test_feature_conversion():
    """Test the feature conversion system"""
    print("Testing Neural Feature Conversion...")
    
    # Create test chess position
    board = chess.Board()  # Starting position
    extractor = ChessStateExtractor()
    chess_state = extractor.extract_state(board)
    
    # Convert to features
    converter = NeuralFeatureConverter()
    feature_vector = converter.convert_to_features(chess_state)
    
    print(f"Feature vector shape: {feature_vector.shape}")
    print(f"Expected shape: [1, {converter.TOTAL_FEATURE_SIZE}]")
    print(f"Feature range: [{feature_vector.min():.3f}, {feature_vector.max():.3f}]")
    
    # Test with different positions
    board.push(chess.Move.from_uci("e2e4"))
    chess_state_2 = extractor.extract_state(board)
    feature_vector_2 = converter.convert_to_features(chess_state_2)
    
    # Check that different positions produce different features
    difference = torch.abs(feature_vector - feature_vector_2).sum()
    print(f"Feature difference after e2e4: {difference:.3f}")
    
    print("✅ Feature conversion test completed!")
    return feature_vector


if __name__ == "__main__":
    test_feature_conversion()
