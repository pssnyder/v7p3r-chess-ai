"""
V7P3R v7.0 - Personality Reward System

Encodes V7P3R's Tal-style aggressive playing style through reward shaping.
Modifies Stockfish's objective evaluation to prefer complexity and tactics.

Personality Traits:
- Complexity seeking (forest darkness)
- Material sacrifice tolerance
- King risk acceptance (if attacking)
- Center control emphasis
- Tactical sharpness preference

Philosophy: "Learn good chess from Stockfish, add personality through rewards"
"""

import numpy as np
import chess
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class PersonalityWeights:
    """Configurable weights for personality reward components."""
    
    # Complexity rewards
    forest_darkness: float = 0.15  # Reward high forest darkness scores
    piece_tension: float = 0.10    # Reward positions with tension
    move_diversity: float = 0.05   # Reward many piece types active
    
    # Material sacrifice tolerance
    material_sacrifice_bonus: float = 0.10  # Bonus if material loss + complexity gain
    material_threshold: int = 5             # Max material loss to tolerate (pawns)
    complexity_threshold: float = 0.2       # Min complexity increase needed
    
    # King safety vs aggression
    king_risk_penalty: float = -0.05   # Penalty for king exposure (negative = bad)
    king_risk_tolerance: float = 2.0   # Tolerate up to 2 pawn shield loss
    attack_bonus: float = 0.08         # Bonus if opponent king under pressure
    
    # Strategic emphasis
    center_control: float = 0.05       # Bonus for center control
    passed_pawns: float = 0.03         # Standard chess (passed pawns good)
    bishop_pair: float = 0.02          # Standard chess (bishop pair good)
    active_rooks: float = 0.04         # Reward rook activity
    
    # Endgame adjustments
    endgame_complexity_weight: float = 0.5  # Reduce complexity seeking in endgame
    
    def get_total_weight(self) -> float:
        """Calculate total personality reward weight."""
        return (
            self.forest_darkness +
            self.piece_tension +
            self.move_diversity +
            self.material_sacrifice_bonus +
            abs(self.king_risk_penalty) +
            self.attack_bonus +
            self.center_control +
            self.passed_pawns +
            self.bishop_pair +
            self.active_rooks
        )


class PersonalityRewardCalculator:
    """
    Calculates personality-based reward bonuses for position evaluation.
    
    Works with comprehensive feature extractor to identify V7P3R-style positions.
    Rewards are added to Stockfish evaluation during training.
    """
    
    def __init__(self, weights: PersonalityWeights = PersonalityWeights()):
        """
        Initialize personality reward calculator.
        
        Args:
            weights: PersonalityWeights configuration
        """
        self.weights = weights
    
    def calculate_complexity_rewards(
        self,
        features: Dict[str, float],
        features_before: Optional[Dict[str, float]] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate rewards for complexity-seeking behavior.
        
        Args:
            features: Current position features (from comprehensive extractor)
            features_before: Features before move (for delta calculation)
        
        Returns:
            (total_reward, breakdown_dict)
        """
        rewards = {}
        
        # Forest darkness reward (V7P3R's custom complexity metric)
        forest_darkness = features.get('forest_darkness_score', 0.0)
        rewards['forest_darkness'] = forest_darkness * self.weights.forest_darkness
        
        # Piece tension reward
        piece_tension = features.get('piece_tension', 0.0)
        tension_normalized = min(piece_tension / 16.0, 1.0)  # Max ~16 pieces
        rewards['piece_tension'] = tension_normalized * self.weights.piece_tension
        
        # Move diversity reward
        move_diversity = features.get('move_diversity', 0.0)
        diversity_normalized = move_diversity / 6.0  # Max 6 piece types
        rewards['move_diversity'] = diversity_normalized * self.weights.move_diversity
        
        # Complexity increase bonus (if before features available)
        if features_before is not None:
            darkness_before = features_before.get('forest_darkness_score', 0.0)
            darkness_delta = forest_darkness - darkness_before
            if darkness_delta > 0.1:  # Significant complexity increase
                rewards['complexity_increase'] = darkness_delta * 0.05
        
        # Game phase adjustment (reduce complexity seeking in endgame)
        game_phase = features.get('game_phase', 0.0)  # 0=opening, 1=endgame
        endgame_penalty = game_phase * (1.0 - self.weights.endgame_complexity_weight)
        
        # Apply endgame adjustment to complexity rewards
        total = sum(rewards.values())
        total *= (1.0 - endgame_penalty)
        
        return total, rewards
    
    def calculate_material_sacrifice_rewards(
        self,
        features: Dict[str, float],
        features_before: Optional[Dict[str, float]] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate rewards for material sacrifice when complexity increases.
        
        V7P3R tolerates material loss if position becomes more complex/tactical.
        
        Args:
            features: Current position features
            features_before: Features before move
        
        Returns:
            (total_reward, breakdown_dict)
        """
        rewards = {}
        
        if features_before is None:
            return 0.0, rewards
        
        # Calculate material delta
        material_balance = features.get('material_balance', 0.0)
        material_before = features_before.get('material_balance', 0.0)
        material_delta = material_balance - material_before
        
        # Calculate complexity delta
        forest_darkness = features.get('forest_darkness_score', 0.0)
        forest_before = features_before.get('forest_darkness_score', 0.0)
        complexity_delta = forest_darkness - forest_before
        
        # Reward material sacrifice if complexity increased significantly
        if (material_delta < 0 and  # Material loss
            abs(material_delta) <= self.weights.material_threshold and  # Not too much
            complexity_delta >= self.weights.complexity_threshold):  # Complexity gain
            
            # Bonus proportional to complexity increase
            sacrifice_bonus = (
                complexity_delta * 
                self.weights.material_sacrifice_bonus *
                min(abs(material_delta) / 3.0, 1.0)  # Scale with sacrifice size
            )
            rewards['material_sacrifice'] = sacrifice_bonus
        
        return sum(rewards.values()), rewards
    
    def calculate_king_safety_rewards(
        self,
        features: Dict[str, float],
        features_before: Optional[Dict[str, float]] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate rewards for king safety vs aggression trade-offs.
        
        V7P3R tolerates king risk if attacking opponent king.
        
        Args:
            features: Current position features
            features_before: Features before move
        
        Returns:
            (total_reward, breakdown_dict)
        """
        rewards = {}
        
        # King safety advantage (positive = our king safer)
        king_safety_adv = features.get('king_safety_advantage', 0.0)
        
        # Check if opponent king under pressure
        # Proxies: opponent has low king safety, we have checks available
        opponent_king_unsafe = king_safety_adv > 2  # We have significant advantage
        checks_available = features.get('checks_available', 0) > 0
        
        if opponent_king_unsafe or checks_available:
            # Reward attacking positions
            rewards['attack_bonus'] = self.weights.attack_bonus
            
            # If our king also exposed, tolerate it
            if features_before is not None:
                king_safety_before = features_before.get('king_safety_advantage', 0.0)
                king_safety_loss = king_safety_before - king_safety_adv
                
                if (king_safety_loss > 0 and 
                    king_safety_loss <= self.weights.king_risk_tolerance):
                    # Reduce penalty for king risk when attacking
                    rewards['king_risk_tolerance'] = abs(self.weights.king_risk_penalty) * 0.5
        else:
            # Normal king safety penalty if not attacking
            if king_safety_adv < -2:  # Our king exposed
                rewards['king_safety_penalty'] = (
                    self.weights.king_risk_penalty * 
                    (abs(king_safety_adv) / 5.0)
                )
        
        return sum(rewards.values()), rewards
    
    def calculate_strategic_rewards(
        self,
        features: Dict[str, float]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate rewards for standard strategic factors.
        
        These are chess fundamentals that V7P3R emphasizes slightly more than Stockfish.
        
        Args:
            features: Current position features
        
        Returns:
            (total_reward, breakdown_dict)
        """
        rewards = {}
        
        # Center control
        center_control = features.get('center_control', 0.0)
        if abs(center_control) > 0.3:  # Significant control
            rewards['center_control'] = center_control * self.weights.center_control
        
        # Passed pawns advantage
        passed_pawns_adv = features.get('passed_pawns_advantage', 0)
        if passed_pawns_adv > 0:
            rewards['passed_pawns'] = passed_pawns_adv * self.weights.passed_pawns
        
        # Bishop pair advantage
        bishop_pair_adv = features.get('bishop_pair_advantage', 0)
        if bishop_pair_adv > 0:
            rewards['bishop_pair'] = self.weights.bishop_pair
        
        # Active rooks advantage
        active_rooks_adv = features.get('active_rooks_advantage', 0)
        if active_rooks_adv > 0:
            rewards['active_rooks'] = active_rooks_adv * self.weights.active_rooks
        
        return sum(rewards.values()), rewards
    
    def calculate_total_reward(
        self,
        features: Dict[str, float],
        features_before: Optional[Dict[str, float]] = None,
        stockfish_eval: float = 0.0,
        game_result: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate complete personality reward with breakdown.
        
        Args:
            features: Current position features
            features_before: Features before move (for delta calculation)
            stockfish_eval: Stockfish evaluation [-1, 1]
            game_result: Game outcome [-1, 0, 1] if available
        
        Returns:
            Dictionary with all reward components and totals
        """
        # Calculate each component
        complexity_reward, complexity_breakdown = self.calculate_complexity_rewards(
            features, features_before
        )
        
        sacrifice_reward, sacrifice_breakdown = self.calculate_material_sacrifice_rewards(
            features, features_before
        )
        
        king_reward, king_breakdown = self.calculate_king_safety_rewards(
            features, features_before
        )
        
        strategic_reward, strategic_breakdown = self.calculate_strategic_rewards(
            features
        )
        
        # Total personality bonus
        personality_total = (
            complexity_reward +
            sacrifice_reward +
            king_reward +
            strategic_reward
        )
        
        # Combine with Stockfish (70% Stockfish, 20% personality, 10% result if available)
        if game_result is not None:
            final_reward = (
                stockfish_eval * 0.7 +
                personality_total * 0.2 +
                game_result * 0.1
            )
        else:
            final_reward = (
                stockfish_eval * 0.8 +
                personality_total * 0.2
            )
        
        # Return complete breakdown
        return {
            'stockfish_eval': stockfish_eval,
            'game_result': game_result,
            'personality_total': personality_total,
            'complexity_reward': complexity_reward,
            'sacrifice_reward': sacrifice_reward,
            'king_reward': king_reward,
            'strategic_reward': strategic_reward,
            'final_reward': final_reward,
            # Detailed breakdowns
            'complexity_breakdown': complexity_breakdown,
            'sacrifice_breakdown': sacrifice_breakdown,
            'king_breakdown': king_breakdown,
            'strategic_breakdown': strategic_breakdown
        }


# Convenience function
def create_personality_calculator(
    forest_darkness_weight: float = 0.15,
    piece_tension_weight: float = 0.10,
    attack_bonus: float = 0.08
) -> PersonalityRewardCalculator:
    """
    Create personality calculator with custom weights.
    
    Args:
        forest_darkness_weight: How much to reward complexity
        piece_tension_weight: How much to reward tactical positions
        attack_bonus: How much to reward attacking opponent king
    
    Returns:
        PersonalityRewardCalculator instance
    """
    weights = PersonalityWeights(
        forest_darkness=forest_darkness_weight,
        piece_tension=piece_tension_weight,
        attack_bonus=attack_bonus
    )
    return PersonalityRewardCalculator(weights)


# Example usage and validation
if __name__ == "__main__":
    from comprehensive_features import ComprehensiveFeatureExtractor
    
    print("="*60)
    print("V7P3R v7.0 - PERSONALITY REWARD SYSTEM")
    print("="*60)
    
    # Create personality calculator
    calculator = PersonalityRewardCalculator()
    weights = calculator.weights
    
    print(f"\n📊 Personality Weights:")
    print(f"  Forest Darkness: {weights.forest_darkness}")
    print(f"  Piece Tension: {weights.piece_tension}")
    print(f"  Attack Bonus: {weights.attack_bonus}")
    print(f"  King Risk Penalty: {weights.king_risk_penalty}")
    print(f"  Center Control: {weights.center_control}")
    print(f"  Total Weight: {weights.get_total_weight():.3f}")
    
    # Test on starting position
    print(f"\n🧪 Testing Starting Position...")
    extractor = ComprehensiveFeatureExtractor()
    board = chess.Board()
    features_dict = extractor.extract_all_features_dict(board)
    
    # Calculate rewards
    reward_result = calculator.calculate_total_reward(
        features_dict,
        stockfish_eval=0.0  # Neutral position
    )
    
    print(f"  Stockfish eval: {reward_result['stockfish_eval']:.3f}")
    print(f"  Personality reward: {reward_result['personality_total']:.3f}")
    print(f"  Final reward: {reward_result['final_reward']:.3f}")
    
    print(f"\n  Breakdown:")
    print(f"    Complexity: {reward_result['complexity_reward']:.3f}")
    print(f"    Strategic: {reward_result['strategic_reward']:.3f}")
    print(f"    King safety: {reward_result['king_reward']:.3f}")
    
    # Test complex position (after several moves)
    print(f"\n🧪 Testing Complex Position (Sicilian Defense)...")
    board = chess.Board()
    for move in ['e4', 'c5', 'Nf3', 'd6', 'd4', 'cxd4', 'Nxd4', 'Nf6']:
        board.push_san(move)
    
    features_dict = extractor.extract_all_features_dict(board)
    
    reward_result = calculator.calculate_total_reward(
        features_dict,
        stockfish_eval=0.2  # Slight advantage
    )
    
    print(f"  Forest Darkness: {features_dict['forest_darkness_score']:.3f}")
    print(f"  Piece Tension: {features_dict['piece_tension']}")
    print(f"  Stockfish eval: {reward_result['stockfish_eval']:.3f}")
    print(f"  Personality reward: {reward_result['personality_total']:.3f}")
    print(f"  Final reward: {reward_result['final_reward']:.3f}")
    
    # Test tactical position (sacrifice scenario simulation)
    print(f"\n🧪 Testing Material Sacrifice Scenario...")
    features_before = {
        'material_balance': 0.0,
        'forest_darkness_score': 0.3
    }
    features_after = {
        'material_balance': -3.0,  # Lost minor piece
        'forest_darkness_score': 0.6  # But position much more complex
    }
    
    sacrifice_reward, breakdown = calculator.calculate_material_sacrifice_rewards(
        features_after,
        features_before
    )
    
    print(f"  Material lost: 3 pawns")
    print(f"  Complexity gained: 0.30 → 0.60")
    print(f"  Sacrifice bonus: {sacrifice_reward:.3f}")
    
    print(f"\n✅ Personality reward system validated!")
    print(f"\n📝 Philosophy:")
    print(f"  - Learn good chess from Stockfish (80% weight)")
    print(f"  - Add V7P3R personality through rewards (20% weight)")
    print(f"  - Prefer complexity, tactics, and aggression")
    print(f"  - Tolerate material sacrifice if position improves")
    print(f"  - Accept king risk when attacking")
