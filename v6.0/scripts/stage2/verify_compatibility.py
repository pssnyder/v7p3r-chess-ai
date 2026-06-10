#!/usr/bin/env python3
"""
Stage 1 → Stage 2 Feature Compatibility Checker
V7P3R AI v6.1 - Data Pipeline Validation

Verifies that Stage 1 outputs properly connect to Stage 2 inputs.
Ensures feature dimensions, types, and ranges are compatible.

Author: Pat Snyder
Created: 2026-05-31
"""

import sys
from pathlib import Path
import json
import numpy as np
from typing import Dict, List, Tuple
import chess

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.stage1.feature_extractor import extract_fast_features
from src.stage1.position_evaluator import PositionEvaluator


class FeatureCompatibilityChecker:
    """
    Validates Stage 1 → Stage 2 feature pipeline.
    
    Checks:
    1. Stage 1 output dimensions (19 features)
    2. Stage 2 input dimensions (~40 features)
    3. Feature value ranges
    4. Data type consistency
    5. Missing value handling
    """
    
    def __init__(self):
        self.stage1_feature_dim = 19
        self.stage2_feature_dim_expected = 40  # Approximate
        
        # Expected Stage 1 features (from extract_fast_features)
        self.stage1_feature_names = [
            'white_pawns', 'white_knights', 'white_bishops', 'white_rooks', 
            'white_queens', 'white_kings',
            'black_pawns', 'black_knights', 'black_bishops', 'black_rooks',
            'black_queens', 'black_kings',
            'material_balance',
            'white_can_castle_kingside', 'white_can_castle_queenside',
            'black_can_castle_kingside', 'black_can_castle_queenside',
            'is_in_check',
            'white_mobility', 'black_mobility',
        ]
        
        # Expected Stage 2 features (from STAGE2_DESIGN_ARCHITECTURE.md)
        self.stage2_feature_groups = {
            'time_based': [
                'time_remaining_white', 'time_remaining_black',
                'increment_white', 'increment_black',
                'processing_tick_count_1ply', 'processing_tick_count_2ply',
                'cache_hit_rate'
            ],
            'complexity': [
                'legal_moves_count', 'capture_moves_count', 'check_moves_count',
                'forced_moves_count', 'branching_factor_1ply', 'branching_factor_2ply',
                'tactical_density', 'forest_darkness_score'
            ],
            'tactical_priority': [
                'pieces_under_attack', 'pieces_undefended',
                'material_delta_after_move', 'material_delta_2ply',
                'king_safety_delta', 'pawn_structure_disruption'
            ],
            'stage1_integration': [
                'stage1_prob_good', 'stage1_confidence',
                'stage1_material_balance', 'stage1_mobility_white', 'stage1_mobility_black',
                'stage1_castling_rights', 'stage1_in_check'
            ],
        }
    
    def check_stage1_output(self, model_path: Path) -> Tuple[bool, List[str]]:
        """
        Verify Stage 1 model outputs correct dimensions.
        
        Returns:
            (success, issues)
        """
        issues = []
        
        print("Checking Stage 1 Output...")
        print("-" * 60)
        
        # Load model
        try:
            model = PositionEvaluator.load(model_path, device='cpu')
            model.eval()
            print(f"✓ Loaded Stage 1 model from {model_path}")
        except Exception as e:
            issues.append(f"Failed to load Stage 1 model: {e}")
            return False, issues
        
        # Test on sample position
        test_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        
        try:
            features = extract_fast_features(test_fen)
            print(f"✓ Extracted features: {features.shape}")
            
            # Check dimension
            if features.shape[0] != self.stage1_feature_dim:
                issues.append(
                    f"Stage 1 feature dimension mismatch: "
                    f"expected {self.stage1_feature_dim}, got {features.shape[0]}"
                )
            else:
                print(f"✓ Feature dimension correct: {self.stage1_feature_dim}")
            
            # Check feature ranges
            for i, (name, value) in enumerate(zip(self.stage1_feature_names, features)):
                if np.isnan(value) or np.isinf(value):
                    issues.append(f"Feature '{name}' has invalid value: {value}")
                
                # Check reasonable ranges
                if 'pawns' in name and (value < 0 or value > 8):
                    issues.append(f"Feature '{name}' out of range: {value}")
                if 'mobility' in name and (value < 0 or value > 100):
                    issues.append(f"Feature '{name}' out of range: {value}")
            
            if not issues:
                print(f"✓ Feature values in valid ranges")
            
            # Test model prediction
            prob = model.predict_probability(features)
            print(f"✓ Model prediction: {prob:.4f}")
            
            if prob < 0.0 or prob > 1.0:
                issues.append(f"Model output out of range [0,1]: {prob}")
            else:
                print(f"✓ Model output in valid range [0, 1]")
            
        except Exception as e:
            issues.append(f"Failed to extract/predict features: {e}")
            return False, issues
        
        success = len(issues) == 0
        return success, issues
    
    def check_stage2_input_format(self, sample_position_file: Path) -> Tuple[bool, List[str]]:
        """
        Verify Stage 2 input format from self-play data.
        
        Args:
            sample_position_file: JSONL file with self-play position data
            
        Returns:
            (success, issues)
        """
        issues = []
        
        print("\nChecking Stage 2 Input Format...")
        print("-" * 60)
        
        if not sample_position_file.exists():
            issues.append(f"Sample position file not found: {sample_position_file}")
            return False, issues
        
        # Read first position
        try:
            with open(sample_position_file, 'r') as f:
                first_line = f.readline()
                position_data = json.loads(first_line)
            
            print(f"✓ Loaded sample position from {sample_position_file.name}")
            
        except Exception as e:
            issues.append(f"Failed to load sample position: {e}")
            return False, issues
        
        # Check required top-level fields
        required_fields = [
            'game_id', 'position_id', 'fen', 'move_number', 'side_to_move',
            'stage1_features', 'complexity_metrics', 'time_state', 'labels'
        ]
        
        for field in required_fields:
            if field not in position_data:
                issues.append(f"Missing required field: {field}")
            else:
                print(f"✓ Found field: {field}")
        
        # Check Stage 1 features
        if 'stage1_features' in position_data:
            stage1_features = position_data['stage1_features']
            if len(stage1_features) != self.stage1_feature_dim:
                issues.append(
                    f"Stage 1 features dimension mismatch: "
                    f"expected {self.stage1_feature_dim}, got {len(stage1_features)}"
                )
            else:
                print(f"✓ Stage 1 features dimension: {len(stage1_features)}")
        
        # Check complexity metrics
        if 'complexity_metrics' in position_data:
            complexity = position_data['complexity_metrics']
            required_complexity = [
                'legal_moves_count', 'capture_moves_count', 'check_moves_count',
                'pieces_under_attack', 'pieces_undefended', 'branching_factor_1ply',
                'forest_darkness_score', 'tactical_density'
            ]
            for metric in required_complexity:
                if metric not in complexity:
                    issues.append(f"Missing complexity metric: {metric}")
        
        # Check time state
        if 'time_state' in position_data:
            time_state = position_data['time_state']
            required_time = [
                'time_white', 'time_black', 'increment', 'time_budget', 'time_remaining'
            ]
            for metric in required_time:
                if metric not in time_state:
                    issues.append(f"Missing time state: {metric}")
        
        # Check labels
        if 'labels' in position_data:
            labels = position_data['labels']
            required_labels = [
                'complexity_score', 'time_allocation', 'processing_tick_count'
            ]
            for label in required_labels:
                if label not in labels:
                    issues.append(f"Missing label: {label}")
                else:
                    value = labels[label]
                    # Validate ranges
                    if label == 'complexity_score' and (value < 0 or value > 10):
                        issues.append(f"Label {label} out of range [0,10]: {value}")
                    if label == 'time_allocation' and (value < 0 or value > 1):
                        issues.append(f"Label {label} out of range [0,1]: {value}")
        
        success = len(issues) == 0
        return success, issues
    
    def check_full_pipeline(
        self, 
        model_path: Path, 
        sample_data_dir: Path
    ) -> Tuple[bool, List[str]]:
        """
        Check entire Stage 1 → Stage 2 pipeline.
        
        Args:
            model_path: Path to Stage 1 model
            sample_data_dir: Directory with self-play data
            
        Returns:
            (success, issues)
        """
        all_issues = []
        
        print("=" * 60)
        print("Stage 1 → Stage 2 Feature Compatibility Check")
        print("=" * 60)
        
        # Check Stage 1 output
        stage1_success, stage1_issues = self.check_stage1_output(model_path)
        all_issues.extend(stage1_issues)
        
        # Find sample position file
        sample_files = list(sample_data_dir.glob("*_positions.jsonl"))
        if not sample_files:
            all_issues.append(f"No position files found in {sample_data_dir}")
            return False, all_issues
        
        # Check Stage 2 input format
        stage2_success, stage2_issues = self.check_stage2_input_format(sample_files[0])
        all_issues.extend(stage2_issues)
        
        # Summary
        print("\n" + "=" * 60)
        print("COMPATIBILITY CHECK SUMMARY")
        print("=" * 60)
        
        if not all_issues:
            print("✓ ALL CHECKS PASSED")
            print("\nStage 1 → Stage 2 pipeline is compatible!")
            print("\nFeature mapping:")
            print("  Stage 1 (19 features) → Stage 1 Integration (7 features in Stage 2)")
            print("  Complexity analysis → Complexity Features (8 features)")
            print("  Time tracking → Time-Based Features (7 features)")
            print("  Tactical analysis → Tactical Priority Features (6+ features)")
            print(f"\nTotal Stage 2 features: ~{sum(len(v) for v in self.stage2_feature_groups.values())} features")
            return True, []
        else:
            print(f"✗ FOUND {len(all_issues)} ISSUES:")
            for i, issue in enumerate(all_issues, 1):
                print(f"  {i}. {issue}")
            return False, all_issues
    
    def generate_feature_map(self) -> Dict:
        """
        Generate feature mapping documentation.
        
        Returns:
            Dictionary describing Stage 1 → Stage 2 feature mapping
        """
        return {
            'stage1_output': {
                'dimension': self.stage1_feature_dim,
                'features': self.stage1_feature_names,
                'type': 'numpy.ndarray',
                'range': 'varies by feature (see feature_extractor.py)',
            },
            'stage2_input': {
                'dimension': sum(len(v) for v in self.stage2_feature_groups.values()),
                'feature_groups': self.stage2_feature_groups,
                'type': 'torch.Tensor',
                'sources': {
                    'stage1_integration': 'Direct from Stage 1 output (7 features)',
                    'complexity': 'Calculated from board analysis (8 features)',
                    'time_based': 'Tracked during game play (7 features)',
                    'tactical_priority': 'Move-specific analysis (6+ features)',
                },
            },
            'mapping': {
                'stage1_material_balance': 'stage1_features[12]',
                'stage1_mobility_white': 'stage1_features[18]',
                'stage1_mobility_black': 'stage1_features[19]',
                'stage1_castling_rights': 'stage1_features[13:17]',
                'stage1_in_check': 'stage1_features[17]',
            },
        }


# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify Stage 1 → Stage 2 feature compatibility")
    parser.add_argument(
        '--model',
        type=Path,
        default=Path('models/position_evaluator_best.pth'),
        help='Path to Stage 1 model'
    )
    parser.add_argument(
        '--data',
        type=Path,
        default=Path('data/stage2/selfplay_batch_284'),
        help='Directory with self-play data'
    )
    parser.add_argument(
        '--save-map',
        action='store_true',
        help='Save feature mapping to JSON'
    )
    
    args = parser.parse_args()
    
    # Run compatibility check
    checker = FeatureCompatibilityChecker()
    success, issues = checker.check_full_pipeline(args.model, args.data)
    
    # Save feature map if requested
    if args.save_map:
        feature_map = checker.generate_feature_map()
        output_file = Path('docs/FEATURE_MAPPING.json')
        with open(output_file, 'w') as f:
            json.dump(feature_map, f, indent=2)
        print(f"\n✓ Feature mapping saved to {output_file}")
    
    # Exit code
    sys.exit(0 if success else 1)
