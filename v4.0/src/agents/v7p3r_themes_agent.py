"""
V7P3R Themes Agent (Stage 1)
Pattern Recognition & Move Ordering Agent

Trained on 4M puzzle library for comprehensive tactical pattern recognition.
Categorizes positions by themes and ranks moves by tactical promise.
"""

import torch
import torch.nn as nn
import chess
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class ThemeScores:
    """Position theme classification results"""
    themes: Dict[str, float]  # Theme name -> probability
    dominant_theme: str
    confidence: float


@dataclass
class MoveRanking:
    """Move ranking results"""
    ranked_moves: List[chess.Move]
    scores: Dict[chess.Move, float]
    time_budget_used: str  # 'fast', 'normal', 'deep', 'ultra_deep'
    inference_time_ms: float


class ThemeClassifier(nn.Module):
    """
    Multi-label classifier for tactical themes
    Input: 690-dimensional position features
    Output: Probability distribution over 50 tactical themes
    """
    def __init__(self, input_size: int = 690, num_themes: int = 50):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 384),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(384, 256),
            nn.ReLU(),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_themes),
            nn.Sigmoid()  # Multi-label classification
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        theme_probs = self.classifier(features)
        return theme_probs


class MoveRankingNetwork(nn.Module):
    """
    Neural network for ranking legal moves by tactical promise
    Input: Position features + move encoding
    Output: Move quality score
    """
    def __init__(self, input_size: int = 690, move_encoding_size: int = 64):
        super().__init__()
        
        combined_size = input_size + move_encoding_size
        
        self.ranker = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ranker(x)


class V7P3RThemesAgent:
    """
    Stage 1 Agent: Pattern Recognition & Move Ordering
    
    Responsibilities:
    - Categorize positions by tactical themes (pins, forks, etc.)
    - Rank legal moves by tactical promise
    - Provide adaptive move candidate selection based on time budget
    """
    
    THEME_NAMES = [
        "pin", "fork", "skewer", "discovered_attack", "double_attack",
        "deflection", "decoy", "interference", "overloading", "zugzwang",
        "mate_in_1", "mate_in_2", "mate_in_3", "back_rank_mate", "smothered_mate",
        "sacrifice", "exchange", "endgame", "opening_trap", "middlegame_tactics",
        "pawn_breakthrough", "passed_pawn", "isolated_pawn", "doubled_pawns",
        "weak_squares", "outpost", "bishop_pair", "knight_outpost",
        "rook_on_7th", "open_file", "semi_open_file", "battery",
        "king_safety", "king_attack", "castling_rights", "development",
        "center_control", "space_advantage", "piece_activity",
        "initiative", "tempo", "compensation", "positional_sacrifice",
        "prophylaxis", "quiet_move", "intermediate_move", "in_between",
        "clearance", "blocking", "x_ray", "windmill"
    ]
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        """
        Initialize Themes Agent
        
        Args:
            model_path: Path to trained model checkpoint (None = untrained)
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.theme_classifier = ThemeClassifier(
            input_size=690,
            num_themes=len(self.THEME_NAMES)
        ).to(self.device)
        
        self.move_ranker = MoveRankingNetwork(
            input_size=690,
            move_encoding_size=64
        ).to(self.device)
        
        # Load trained weights if provided
        if model_path:
            self.load_model(model_path)
        
        # Time budget thresholds (seconds)
        self.time_thresholds = {
            'fast': 0.5,
            'normal': 2.0,
            'deep': 5.0,
            'ultra_deep': float('inf')
        }
        
        # Candidate counts per time budget
        self.candidate_counts = {
            'fast': 5,
            'normal': 10,
            'deep': 50,
            'ultra_deep': 100
        }
        
        logger.info(f"V7P3R Themes Agent initialized on {self.device}")
    
    def load_model(self, model_path: str):
        """Load trained model from checkpoint"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.theme_classifier.load_state_dict(checkpoint['theme_classifier'])
            self.move_ranker.load_state_dict(checkpoint['move_ranker'])
            self.theme_classifier.eval()
            self.move_ranker.eval()
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def categorize_position(self, board: chess.Board) -> ThemeScores:
        """
        Categorize position by tactical themes
        
        Args:
            board: Chess board position
            
        Returns:
            ThemeScores with theme probabilities
        """
        # TODO: Implement feature extraction (see chess_state_extractor.py)
        # For now, using placeholder
        features = self._extract_features(board)
        
        with torch.no_grad():
            features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
            theme_probs = self.theme_classifier(features_tensor.unsqueeze(0))[0]
        
        # Convert to dictionary
        theme_dict = {
            name: prob.item() 
            for name, prob in zip(self.THEME_NAMES, theme_probs)
        }
        
        # Find dominant theme
        dominant_theme = max(theme_dict.keys(), key=lambda k: theme_dict[k])
        confidence = theme_dict[dominant_theme]
        
        return ThemeScores(
            themes=theme_dict,
            dominant_theme=dominant_theme,
            confidence=confidence
        )
    
    def rank_moves(
        self, 
        board: chess.Board, 
        time_budget: float
    ) -> MoveRanking:
        """
        Rank legal moves by tactical promise
        
        Args:
            board: Chess board position
            time_budget: Time available for this move (seconds)
            
        Returns:
            MoveRanking with ordered moves
        """
        start_time = time.time()
        
        # Determine time budget category
        budget_category = self._get_budget_category(time_budget)
        top_k = self.candidate_counts[budget_category]
        
        # Get legal moves
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            return MoveRanking([], {}, budget_category, 0.0)
        
        # Extract position features
        position_features = self._extract_features(board)
        
        # Score each move
        move_scores = {}
        with torch.no_grad():
            for move in legal_moves:
                move_features = self._encode_move(board, move)
                combined_features = position_features + move_features
                
                features_tensor = torch.tensor(
                    combined_features, 
                    dtype=torch.float32
                ).to(self.device)
                
                score = self.move_ranker(features_tensor.unsqueeze(0))[0].item()
                move_scores[move] = score
        
        # Sort moves by score
        ranked_moves = sorted(
            legal_moves, 
            key=lambda m: move_scores[m], 
            reverse=True
        )[:top_k]
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return MoveRanking(
            ranked_moves=ranked_moves,
            scores=move_scores,
            time_budget_used=budget_category,
            inference_time_ms=inference_time
        )
    
    def _get_budget_category(self, time_budget: float) -> str:
        """Determine time budget category"""
        if time_budget < self.time_thresholds['fast']:
            return 'fast'
        elif time_budget < self.time_thresholds['normal']:
            return 'normal'
        elif time_budget < self.time_thresholds['deep']:
            return 'deep'
        else:
            return 'ultra_deep'
    
    def _extract_features(self, board: chess.Board) -> List[float]:
        """
        Extract 690-dimensional features from position
        TODO: Integrate with ChessStateExtractor from v3.0
        """
        # Placeholder implementation
        # In production, use v3.0's ChessState feature extractor
        return [0.0] * 690
    
    def _encode_move(self, board: chess.Board, move: chess.Move) -> List[float]:
        """
        Encode move as 64-dimensional feature vector
        TODO: Implement proper move encoding
        """
        # Placeholder implementation
        # Encode: from_square, to_square, piece_type, capture, promotion, check, etc.
        return [0.0] * 64


if __name__ == "__main__":
    # Quick test
    agent = V7P3RThemesAgent()
    board = chess.Board()
    
    themes = agent.categorize_position(board)
    print(f"Dominant theme: {themes.dominant_theme} (confidence: {themes.confidence:.2f})")
    
    ranking = agent.rank_moves(board, time_budget=2.0)
    print(f"Top 5 moves ({ranking.time_budget_used}): {ranking.ranked_moves[:5]}")
    print(f"Inference time: {ranking.inference_time_ms:.2f}ms")
