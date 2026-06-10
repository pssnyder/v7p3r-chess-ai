"""
Phase Manager - Dynamic Weight Adjustment Throughout the Game

Implements the "Chess as Story" philosophy:
- Opening: Learn good principles (high Stockfish weight)
- Middlegame: Creative chaos (high personality weight)
- Endgame: Mathematical perfection (tablebase oracle)

The weighting function uses a sinusoidal curve to smoothly transition
between phases, mimicking the natural ebb and flow of a chess game.
"""

import chess
import numpy as np
from typing import Tuple, Dict


class GamePhase:
    """Detects and manages game phases."""
    
    OPENING = "opening"
    EARLY_MIDDLEGAME = "early_middlegame"
    DEEP_MIDDLEGAME = "deep_middlegame"
    LATE_MIDDLEGAME = "late_middlegame"
    ENDGAME = "endgame"
    TABLEBASE = "tablebase"
    
    @staticmethod
    def detect_phase(board: chess.Board, move_number: int) -> str:
        """
        Detect current game phase based on move count and material.
        
        Phases:
        - Opening: Moves 1-10
        - Early Middlegame: Moves 11-20
        - Deep Middlegame: Moves 21-40 (peak chaos)
        - Late Middlegame: Moves 41-60
        - Endgame: <14 pieces or move 61+
        - Tablebase: ≤7 pieces
        """
        piece_count = len(board.piece_map())
        
        # Tablebase territory (perfect knowledge available)
        if piece_count <= 7:
            return GamePhase.TABLEBASE
        
        # Endgame (reduced material)
        if piece_count <= 13 or move_number > 60:
            return GamePhase.ENDGAME
        
        # Move-based phases
        if move_number <= 10:
            return GamePhase.OPENING
        elif move_number <= 20:
            return GamePhase.EARLY_MIDDLEGAME
        elif move_number <= 40:
            return GamePhase.DEEP_MIDDLEGAME
        elif move_number <= 60:
            return GamePhase.LATE_MIDDLEGAME
        else:
            return GamePhase.ENDGAME
    
    @staticmethod
    def get_material_stage(board: chess.Board) -> str:
        """Get material-based stage (for tablebase detection)."""
        piece_count = len(board.piece_map())
        
        if piece_count <= 5:
            return "5-man"
        elif piece_count <= 6:
            return "6-man"
        elif piece_count <= 7:
            return "7-man"
        else:
            return "complex"


class DynamicWeightCalculator:
    """
    Calculates dynamic weights for Stockfish vs Personality throughout the game.
    
    Philosophy:
    - Opening: Trust established theory (90% Stockfish)
    - Middlegame: Embrace creative chaos (10% Stockfish)
    - Endgame: Return to precision (50-100% Stockfish/Tablebase)
    
    Uses a modified sinusoidal function to create smooth transitions.
    """
    
    def __init__(
        self,
        opening_sf_weight: float = 0.9,
        middlegame_sf_weight: float = 0.2,
        endgame_sf_weight: float = 1.0,
        tablebase_sf_weight: float = 1.0
    ):
        """
        Initialize weight calculator with phase-specific weights.
        
        NEW CURVE (v7.1):
        - Opening (1-10): 90% SF - Learn fundamentals
        - Early Transition (11-20): 90% → 10% SF - Entering chaos
        - Middlegame (21-40): 20% SF - Creative exploration (up from 10%)
        - Late Transition (41-60): 20% → 80% SF - Returning to precision
        - Endgame (61+): 100% SF - Mathematical perfection (up from 50%)
        
        Args:
            opening_sf_weight: Stockfish weight in opening (default 0.9)
            middlegame_sf_weight: Stockfish weight in middlegame (default 0.2)
            endgame_sf_weight: Stockfish weight in endgame (default 1.0)
            tablebase_sf_weight: Stockfish/tablebase weight when TB active (default 1.0)
        """
        self.opening_sf = opening_sf_weight
        self.middlegame_sf = middlegame_sf_weight
        self.endgame_sf = endgame_sf_weight
        self.tablebase_sf = tablebase_sf_weight
    
    def calculate_weights(
        self,
        board: chess.Board,
        move_number: int
    ) -> Dict[str, float]:
        """
        Calculate dynamic weights for current position.
        
        Returns:
            Dict with keys: 'stockfish', 'personality', 'outcome', 'phase'
        """
        phase = GamePhase.detect_phase(board, move_number)
        
        # Tablebase territory - perfect knowledge
        if phase == GamePhase.TABLEBASE:
            return {
                'stockfish': self.tablebase_sf,
                'personality': 0.0,
                'outcome': 0.0,
                'phase': phase
            }
        
        # Calculate Stockfish weight based on phase
        stockfish_weight = self._calculate_stockfish_weight(phase, move_number)
        
        # Personality weight is inverse of Stockfish (zero-sum for main components)
        personality_weight = 1.0 - stockfish_weight
        
        # Outcome gets small constant weight (not part of SF vs Personality trade-off)
        # This is the actual game result signal
        outcome_weight = 0.1
        
        # Normalize so SF + Personality = 0.9, Outcome = 0.1
        total_main = stockfish_weight + personality_weight
        stockfish_weight = stockfish_weight / total_main * 0.9
        personality_weight = personality_weight / total_main * 0.9
        
        return {
            'stockfish': stockfish_weight,
            'personality': personality_weight,
            'outcome': outcome_weight,
            'phase': phase
        }
    
    def _calculate_stockfish_weight(self, phase: str, move_number: int) -> float:
        """
        Calculate Stockfish weight using smooth transitions.
        
        NEW CURVE (v7.1):
        - Opening (1-10): 90% constant - Strong fundamentals
        - Early MG (11-20): 90% → 10% - Steep transition to chaos
        - Deep MG (21-40): 20% constant - Creative middlegame (controlled chaos)
        - Late MG (41-60): 20% → 80% - Steep return to precision
        - Endgame (61+): 100% - Perfect technique
        """
        if phase == GamePhase.OPENING:
            # Stay at high Stockfish weight
            # Move 1-10: 90%
            return self.opening_sf
        
        elif phase == GamePhase.EARLY_MIDDLEGAME:
            # STEEP transition from opening to chaos
            # Move 11: 90%, Move 20: 10%
            progress = (move_number - 10) / 10.0
            return 0.9 - (0.9 - 0.1) * progress
        
        elif phase == GamePhase.DEEP_MIDDLEGAME:
            # CONTROLLED chaos - stable at 20% SF
            # Move 21-40: 20% constant
            # This allows personality to emerge while maintaining some structure
            return self.middlegame_sf
        
        elif phase == GamePhase.LATE_MIDDLEGAME:
            # STEEP return to precision as endgame approaches
            # Move 41: 20%, Move 60: 80%
            progress = (move_number - 40) / 20.0
            return 0.2 + (0.8 - 0.2) * progress
        
        elif phase == GamePhase.ENDGAME:
            # Perfect technique required
            return self.endgame_sf
        
        else:  # TABLEBASE
            return self.tablebase_sf
    
    def get_phase_description(self, phase: str) -> str:
        """Get human-readable phase description."""
        descriptions = {
            GamePhase.OPENING: "Opening - Learning principles",
            GamePhase.EARLY_MIDDLEGAME: "Early Middlegame - Building complexity",
            GamePhase.DEEP_MIDDLEGAME: "Deep Middlegame - CHAOS MODE",
            GamePhase.LATE_MIDDLEGAME: "Late Middlegame - Refining tactics",
            GamePhase.ENDGAME: "Endgame - Precision required",
            GamePhase.TABLEBASE: "Tablebase Territory - Perfect play"
        }
        return descriptions.get(phase, "Unknown")
    
    def visualize_weight_curve(self, max_moves: int = 80) -> str:
        """
        Generate ASCII visualization of weight curve over game.
        
        Returns string representation for debugging/documentation.
        """
        lines = []
        lines.append("Stockfish Weight Throughout Game (v7.1 REVISED):")
        lines.append("1.0 |██████████                          ████████████████")
        lines.append("0.9 |          ██                      ██                ")
        lines.append("0.8 |            ██                  ██                  ")
        lines.append("0.7 |              ██              ██                    ")
        lines.append("0.6 |                ██          ██                      ")
        lines.append("0.5 |                  ██      ██                        ")
        lines.append("0.4 |                    ██  ██                          ")
        lines.append("0.3 |                      ██                            ")
        lines.append("0.2 |                      ████████████████              ")
        lines.append("0.1 |                      ██                            ")
        lines.append("0.0 |__________________________________________________ ")
        lines.append("    0  10  20  30  40  50  60  70  80  (moves)")
        lines.append("")
        lines.append("Phase Breakdown (v7.1):")
        lines.append("  Moves  1-10: Opening (90% SF) - Fundamentals")
        lines.append("  Moves 11-20: Early MG (90% → 10% SF) - Enter chaos")
        lines.append("  Moves 21-40: Deep MG (20% SF) - CONTROLLED CHAOS")
        lines.append("  Moves 41-60: Late MG (20% → 80% SF) - Return to precision")
        lines.append("  Moves 61+:   Endgame (100% SF) - Perfect technique")
        lines.append("  ≤7 pieces:   Tablebase (100% perfect)")
        
        return "\n".join(lines)


class PhaseAwareTrainingTarget:
    """
    Calculates training targets with phase-aware dynamic weighting.
    
    This is the core of the "Chess as Story" training philosophy.
    """
    
    def __init__(self, weight_calculator: DynamicWeightCalculator = None):
        """Initialize with optional custom weight calculator."""
        self.weight_calc = weight_calculator or DynamicWeightCalculator()
    
    def calculate_target(
        self,
        stockfish_eval: float,
        personality_reward: float,
        game_outcome: float,
        board: chess.Board,
        move_number: int,
        tablebase_eval: float = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate training target with dynamic weighting.
        
        Args:
            stockfish_eval: Normalized Stockfish evaluation [-1, 1]
            personality_reward: Personality-based reward [0, 1]
            game_outcome: Final game result [-1, 0, 1]
            board: Current board position
            move_number: Current move number
            tablebase_eval: Optional tablebase evaluation (if available)
        
        Returns:
            Tuple of (target_value, weight_breakdown_dict)
        """
        # Get dynamic weights for current phase
        weights = self.weight_calc.calculate_weights(board, move_number)
        
        # If in tablebase territory and TB eval available, use it
        if weights['phase'] == GamePhase.TABLEBASE and tablebase_eval is not None:
            target = tablebase_eval  # Perfect knowledge
            weights['used_tablebase'] = True
        else:
            # Dynamic weighted combination
            target = (
                weights['stockfish'] * stockfish_eval +
                weights['personality'] * personality_reward +
                weights['outcome'] * game_outcome
            )
            weights['used_tablebase'] = False
        
        # Add component values for debugging
        weights['stockfish_component'] = weights['stockfish'] * stockfish_eval
        weights['personality_component'] = weights['personality'] * personality_reward
        weights['outcome_component'] = weights['outcome'] * game_outcome
        weights['target'] = target
        
        return target, weights


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Phase-Aware Dynamic Weighting System")
    print("=" * 60)
    print()
    
    calc = DynamicWeightCalculator()
    
    # Visualize the curve
    print(calc.visualize_weight_curve())
    print()
    
    # Test at different moves
    print("Sample Weight Distributions:")
    print("-" * 60)
    
    test_board = chess.Board()
    test_moves = [1, 5, 10, 15, 25, 35, 50, 65]
    
    for move in test_moves:
        weights = calc.calculate_weights(test_board, move)
        phase_desc = calc.get_phase_description(weights['phase'])
        
        print(f"Move {move:2d}: SF={weights['stockfish']:.2f}, "
              f"Pers={weights['personality']:.2f}, "
              f"Out={weights['outcome']:.2f}")
        print(f"         Phase: {phase_desc}")
        print()
    
    print("=" * 60)
    print("The 'Chess as Story' training system is ready!")
    print("=" * 60)
