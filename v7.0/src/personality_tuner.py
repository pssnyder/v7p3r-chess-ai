"""
V7P3R v7.0 - Personality Tuning Framework

Interactive system for customizing engine playing style:
1. Analyze positions you like → extract feature patterns
2. Adjust reward weights to prefer those patterns
3. Test personality changes on position suites
4. Save/load personality profiles

Workflow:
- Show me positions you like → I'll tell you what features they have
- Describe playstyle → I'll suggest reward weights
- Iteratively tune until engine plays how you want
"""

import chess
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from comprehensive_features import ComprehensiveFeatureExtractor
from personality_rewards import PersonalityWeights, PersonalityRewardCalculator


@dataclass
class PlaystyleProfile:
    """Complete personality profile with metadata."""
    name: str
    description: str
    weights: PersonalityWeights
    example_positions: List[str] = None  # FENs that exemplify this style
    author: str = "V7P3R"
    version: str = "v7.0"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'weights': asdict(self.weights),
            'example_positions': self.example_positions or [],
            'author': self.author,
            'version': self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PlaystyleProfile':
        """Load from dictionary."""
        weights = PersonalityWeights(**data['weights'])
        return cls(
            name=data['name'],
            description=data['description'],
            weights=weights,
            example_positions=data.get('example_positions', []),
            author=data.get('author', 'V7P3R'),
            version=data.get('version', 'v7.0')
        )


class PositionAnalyzer:
    """Analyze positions to understand their characteristics."""
    
    def __init__(self):
        self.extractor = ComprehensiveFeatureExtractor()
    
    def analyze_position(self, fen: str) -> Dict:
        """
        Analyze a single position and report key features.
        
        Args:
            fen: Position to analyze
        
        Returns:
            Dictionary with feature analysis
        """
        board = chess.Board(fen)
        features = self.extractor.extract_all_features_dict(board)
        
        # Categorize features
        analysis = {
            'fen': fen,
            'complexity': {
                'forest_darkness': features['forest_darkness_score'],
                'piece_tension': features['piece_tension'],
                'legal_moves': features['legal_moves_count'],
                'captures_available': features['captures_available'],
                'checks_available': features['checks_available']
            },
            'material': {
                'material_balance': features['material_balance'],
                'piece_counts': {
                    'white_total': sum(features[f'white_{p}_count'] 
                                     for p in ['pawns', 'knights', 'bishops', 'rooks', 'queens']),
                    'black_total': sum(features[f'black_{p}_count'] 
                                     for p in ['pawns', 'knights', 'bishops', 'rooks', 'queens'])
                }
            },
            'structure': {
                'passed_pawns_advantage': features['passed_pawns_advantage'],
                'doubled_pawns_disadvantage': features['doubled_pawns_disadvantage'],
                'isolated_pawns_disadvantage': features['isolated_pawns_disadvantage']
            },
            'activity': {
                'mobility_advantage': features['mobility_advantage'],
                'active_rooks_advantage': features['active_rooks_advantage'],
                'development_advantage': features['development_advantage'],
                'center_control': features['center_control']
            },
            'king_safety': {
                'king_safety_advantage': features['king_safety_advantage']
            },
            'game_phase': features['game_phase'],
            'all_features': features
        }
        
        return analysis
    
    def analyze_position_set(self, fens: List[str]) -> Dict:
        """
        Analyze multiple positions to find common patterns.
        
        Args:
            fens: List of positions to analyze
        
        Returns:
            Statistical summary of features
        """
        analyses = [self.analyze_position(fen) for fen in fens]
        
        # Aggregate statistics
        summary = {
            'count': len(fens),
            'complexity_avg': {
                'forest_darkness': np.mean([a['complexity']['forest_darkness'] for a in analyses]),
                'piece_tension': np.mean([a['complexity']['piece_tension'] for a in analyses]),
                'legal_moves': np.mean([a['complexity']['legal_moves'] for a in analyses])
            },
            'material_avg': {
                'balance': np.mean([a['material']['material_balance'] for a in analyses])
            },
            'structure_avg': {
                'passed_pawns': np.mean([a['structure']['passed_pawns_advantage'] for a in analyses])
            },
            'activity_avg': {
                'mobility': np.mean([a['activity']['mobility_advantage'] for a in analyses]),
                'center_control': np.mean([a['activity']['center_control'] for a in analyses])
            },
            'king_safety_avg': {
                'advantage': np.mean([a['king_safety']['king_safety_advantage'] for a in analyses])
            }
        }
        
        return summary
    
    def suggest_weights_from_positions(self, fens: List[str]) -> PersonalityWeights:
        """
        Analyze positions you like and suggest reward weights.
        
        Args:
            fens: Positions that represent desired playing style
        
        Returns:
            Suggested PersonalityWeights
        """
        summary = self.analyze_position_set(fens)
        
        # Create weights based on patterns
        weights = PersonalityWeights()
        
        # High complexity → increase complexity rewards
        avg_darkness = summary['complexity_avg']['forest_darkness']
        if avg_darkness > 0.5:
            weights.forest_darkness = 0.20  # Aggressive
        elif avg_darkness > 0.3:
            weights.forest_darkness = 0.15  # Default
        else:
            weights.forest_darkness = 0.10  # Positional
        
        # High tension → increase tension rewards
        avg_tension = summary['complexity_avg']['piece_tension']
        if avg_tension > 6:
            weights.piece_tension = 0.15
        elif avg_tension > 3:
            weights.piece_tension = 0.10  # Default
        else:
            weights.piece_tension = 0.05
        
        # Material imbalance tolerance
        avg_material = abs(summary['material_avg']['balance'])
        if avg_material > 2:
            weights.material_sacrifice_bonus = 0.15
            weights.material_threshold = 7
        
        # King safety preferences
        avg_king_safety = summary['king_safety_avg']['advantage']
        if avg_king_safety < -1:  # Positions where your king is exposed
            weights.king_risk_penalty = -0.03  # More tolerant
            weights.attack_bonus = 0.12  # Must be attacking
        
        # Center control emphasis
        avg_center = summary['activity_avg']['center_control']
        if avg_center > 0.3:
            weights.center_control = 0.08
        
        return weights


class PersonalityTuner:
    """Interactive personality tuning system."""
    
    def __init__(self):
        self.analyzer = PositionAnalyzer()
        self.profiles: Dict[str, PlaystyleProfile] = {}
        self._load_builtin_profiles()
    
    def _load_builtin_profiles(self):
        """Load built-in personality profiles."""
        
        # Default: Tal-style aggressive
        tal_weights = PersonalityWeights(
            forest_darkness=0.15,
            piece_tension=0.10,
            attack_bonus=0.08,
            king_risk_penalty=-0.05,
            material_sacrifice_bonus=0.10,
            material_threshold=5
        )
        self.profiles['tal'] = PlaystyleProfile(
            name="Tal",
            description="Aggressive, tactical, sacrificial style. High complexity seeking.",
            weights=tal_weights
        )
        
        # Positional: Karpov-style
        karpov_weights = PersonalityWeights(
            forest_darkness=0.08,
            piece_tension=0.05,
            attack_bonus=0.05,
            king_risk_penalty=-0.10,
            center_control=0.08,
            passed_pawns=0.05,
            bishop_pair=0.04,
            material_sacrifice_bonus=0.03,
            material_threshold=2
        )
        self.profiles['karpov'] = PlaystyleProfile(
            name="Karpov",
            description="Positional, strategic, risk-averse. Emphasizes structure and safety.",
            weights=karpov_weights
        )
        
        # Balanced: Modern engine style
        balanced_weights = PersonalityWeights(
            forest_darkness=0.12,
            piece_tension=0.08,
            attack_bonus=0.06,
            king_risk_penalty=-0.07,
            center_control=0.06,
            passed_pawns=0.04,
            material_sacrifice_bonus=0.08,
            material_threshold=4
        )
        self.profiles['balanced'] = PlaystyleProfile(
            name="Balanced",
            description="Flexible style adapting to position requirements.",
            weights=balanced_weights
        )
    
    def create_profile_from_positions(
        self,
        name: str,
        description: str,
        example_fens: List[str]
    ) -> PlaystyleProfile:
        """
        Create personality profile from example positions.
        
        Args:
            name: Profile name
            description: What this style represents
            example_fens: Positions that exemplify this style
        
        Returns:
            New PlaystyleProfile
        """
        weights = self.analyzer.suggest_weights_from_positions(example_fens)
        
        profile = PlaystyleProfile(
            name=name,
            description=description,
            weights=weights,
            example_positions=example_fens
        )
        
        self.profiles[name.lower()] = profile
        return profile
    
    def compare_profiles(
        self,
        profile1: str,
        profile2: str
    ) -> Dict:
        """
        Compare two personality profiles.
        
        Args:
            profile1: First profile name
            profile2: Second profile name
        
        Returns:
            Comparison dictionary
        """
        p1 = self.profiles.get(profile1.lower())
        p2 = self.profiles.get(profile2.lower())
        
        if not p1 or not p2:
            raise ValueError(f"Profile not found: {profile1} or {profile2}")
        
        w1 = asdict(p1.weights)
        w2 = asdict(p2.weights)
        
        comparison = {
            'profiles': [p1.name, p2.name],
            'differences': {}
        }
        
        for key in w1.keys():
            diff = w1[key] - w2[key]
            if abs(diff) > 0.01:  # Significant difference
                comparison['differences'][key] = {
                    profile1: w1[key],
                    profile2: w2[key],
                    'delta': diff
                }
        
        return comparison
    
    def save_profile(self, profile_name: str, filepath: str):
        """Save personality profile to JSON."""
        profile = self.profiles.get(profile_name.lower())
        if not profile:
            raise ValueError(f"Profile not found: {profile_name}")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(profile.to_dict(), f, indent=2)
    
    def load_profile(self, filepath: str) -> PlaystyleProfile:
        """Load personality profile from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        profile = PlaystyleProfile.from_dict(data)
        self.profiles[profile.name.lower()] = profile
        return profile
    
    def get_profile(self, name: str) -> Optional[PlaystyleProfile]:
        """Get profile by name."""
        return self.profiles.get(name.lower())
    
    def list_profiles(self) -> List[str]:
        """List all available profiles."""
        return list(self.profiles.keys())


# Interactive questionnaire for personality design
def personality_questionnaire() -> Dict[str, any]:
    """
    Questions to ask user about desired playing style.
    
    Returns:
        Dictionary with user preferences
    """
    questions = {
        'aggression': {
            'question': "How aggressive should the engine be?",
            'options': [
                "Ultra-aggressive (Tal, Shirov) - sacrifice freely for attack",
                "Aggressive (Kasparov, Kramnik) - calculated risks",
                "Balanced (Modern engines) - flexible based on position",
                "Positional (Karpov, Petrosian) - minimize risk",
                "Ultra-solid (Steinitz) - maximum safety"
            ],
            'weight_mapping': {
                0: {'forest_darkness': 0.20, 'material_sacrifice_bonus': 0.15, 'material_threshold': 7},
                1: {'forest_darkness': 0.15, 'material_sacrifice_bonus': 0.10, 'material_threshold': 5},
                2: {'forest_darkness': 0.12, 'material_sacrifice_bonus': 0.08, 'material_threshold': 4},
                3: {'forest_darkness': 0.08, 'material_sacrifice_bonus': 0.05, 'material_threshold': 3},
                4: {'forest_darkness': 0.05, 'material_sacrifice_bonus': 0.02, 'material_threshold': 2}
            }
        },
        'king_safety': {
            'question': "How important is king safety?",
            'options': [
                "Very important - never expose king",
                "Important - only expose when attacking",
                "Balanced - evaluate case-by-case",
                "Flexible - tolerate exposure for compensation",
                "Unimportant - king can take care of itself"
            ],
            'weight_mapping': {
                0: {'king_risk_penalty': -0.15, 'attack_bonus': 0.04},
                1: {'king_risk_penalty': -0.10, 'attack_bonus': 0.06},
                2: {'king_risk_penalty': -0.07, 'attack_bonus': 0.08},
                3: {'king_risk_penalty': -0.05, 'attack_bonus': 0.10},
                4: {'king_risk_penalty': -0.02, 'attack_bonus': 0.12}
            }
        },
        'complexity': {
            'question': "What type of positions do you prefer?",
            'options': [
                "Ultra-sharp - maximum tactics and complications",
                "Sharp - lots of tactics but some strategy",
                "Balanced - mix of tactics and strategy",
                "Strategic - prefer long-term plans",
                "Simple - clean, clear positions"
            ],
            'weight_mapping': {
                0: {'piece_tension': 0.15, 'move_diversity': 0.08},
                1: {'piece_tension': 0.12, 'move_diversity': 0.06},
                2: {'piece_tension': 0.10, 'move_diversity': 0.05},
                3: {'piece_tension': 0.07, 'move_diversity': 0.03},
                4: {'piece_tension': 0.05, 'move_diversity': 0.02}
            }
        },
        'opening_style': {
            'question': "What opening style do you prefer?",
            'options': [
                "Hyper-modern - control center from distance",
                "Classical - occupy center with pawns",
                "Flexible - adapt to opponent",
                "Gambit-oriented - sacrifice for initiative",
                "Solid - safe and sound"
            ],
            'weight_mapping': {
                0: {'center_control': 0.04, 'development_advantage': 0.08},
                1: {'center_control': 0.08, 'development_advantage': 0.05},
                2: {'center_control': 0.06, 'development_advantage': 0.05},
                3: {'center_control': 0.05, 'material_sacrifice_bonus': 0.12},
                4: {'center_control': 0.06, 'king_risk_penalty': -0.12}
            }
        },
        'endgame_style': {
            'question': "How should the engine handle endgames?",
            'options': [
                "Technical - convert advantages methodically",
                "Practical - find easiest win",
                "Ambitious - create complications even when winning",
                "Drawish - accept draws in equal positions",
                "Fighting - never accept draws"
            ],
            'weight_mapping': {
                0: {'endgame_complexity_weight': 0.3, 'passed_pawns': 0.06},
                1: {'endgame_complexity_weight': 0.5, 'passed_pawns': 0.04},
                2: {'endgame_complexity_weight': 0.7, 'forest_darkness': 0.15},
                3: {'endgame_complexity_weight': 0.4, 'forest_darkness': 0.08},
                4: {'endgame_complexity_weight': 0.8, 'forest_darkness': 0.18}
            }
        }
    }
    
    return questions


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("V7P3R v7.0 - PERSONALITY TUNING FRAMEWORK")
    print("="*60)
    
    tuner = PersonalityTuner()
    
    print(f"\n📋 Built-in Profiles:")
    for name in tuner.list_profiles():
        profile = tuner.get_profile(name)
        print(f"  - {profile.name}: {profile.description}")
    
    # Compare profiles
    print(f"\n🔍 Comparing Tal vs Karpov:")
    comparison = tuner.compare_profiles('tal', 'karpov')
    print(f"  Significant differences:")
    for feature, data in comparison['differences'].items():
        print(f"    {feature}:")
        print(f"      Tal: {data['tal']:.3f}")
        print(f"      Karpov: {data['karpov']:.3f}")
        print(f"      Delta: {data['delta']:+.3f}")
    
    # Analyze example position (Tal's famous game)
    print(f"\n🧪 Analyzing Tal-style Position:")
    tal_position = "r1bqk2r/pp3ppp/2n1pn2/2ppP3/3P4/2P2N2/PP3PPP/RNBQKB1R w KQkq - 0 1"
    analysis = tuner.analyzer.analyze_position(tal_position)
    print(f"  Complexity: {analysis['complexity']['forest_darkness']:.3f}")
    print(f"  Piece Tension: {analysis['complexity']['piece_tension']}")
    print(f"  Center Control: {analysis['activity']['center_control']:.3f}")
    
    # Show questionnaire structure
    print(f"\n❓ Personality Questionnaire Categories:")
    questions = personality_questionnaire()
    for category, data in questions.items():
        print(f"  - {category}: {data['question']}")
        print(f"    Options: {len(data['options'])}")
    
    print(f"\n✅ Personality tuning framework ready!")
    print(f"\n📝 Usage Workflow:")
    print(f"  1. Answer questionnaire → get suggested weights")
    print(f"  2. OR provide example positions → auto-detect patterns")
    print(f"  3. Test profile on position suite")
    print(f"  4. Iteratively refine weights")
    print(f"  5. Save custom profile for training")
