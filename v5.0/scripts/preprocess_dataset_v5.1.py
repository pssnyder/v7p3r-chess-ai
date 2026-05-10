"""
V7P3R AI v5.1 - Dataset Preprocessing Script (Expanded Features + Temporal)

Converts JSONL training data into preprocessed NumPy arrays for training.
Updated to handle v5.1 expansion (92+ features) + Phase 1C temporal features (F200-F220).

Total Features:
- F000-F114: Current position features (106 post one-hot)
- F200-F220: Temporal persistence features (21 raw + 135 one-hot = 156)
- TOTAL: 262 features after all encoding

Pipeline:
1. Load train/val/test splits from JSONL
2. Extract features and targets (262 features)
3. Fit transformers on training data (StandardScaler, OneHotEncoder)
4. Transform all splits
5. Save preprocessed arrays and transformers

Usage:
    python scripts/preprocess_dataset_v5.1.py
    
Output:
    - data/preprocessed_v5.1/X_train.npy, y_train_policy.npy, y_train_value.npy
    - data/preprocessed_v5.1/X_val.npy, y_val_policy.npy, y_val_value.npy
    - data/preprocessed_v5.1/X_test.npy, y_test_policy.npy, y_test_value.npy
    - data/preprocessed_v5.1/transformers.pkl (scaler + encoders)
    - data/preprocessed_v5.1/preprocessing_stats.json
"""

import json
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from datetime import datetime


def load_jsonl(filepath):
    """Load JSONL file into list of records"""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    return records


def extract_features(records):
    """
    Extract features from dataset records (v5.1 + temporal features)
    
    Returns dict with:
        - numerical: (N, ~67) array - All continuous numerical features (current + temporal)
        - boolean: (N, ~42) array - All binary boolean features (current + temporal)
        - game_phase: (N, 1) array - Categorical game phase
        - material_cat: (N, 1) array - Categorical material advantage
        - move_types: (N, 3) array - Categorical move types (best, second, v7p3r)
    """
    numerical = []
    boolean = []
    game_phase = []
    material_cat = []
    move_types = []
    
    for record in records:
        f = record['features']
        
        # ========== NUMERICAL FEATURES (~50) ==========
        numerical_row = [
            # Core position features
            f['F003_material_balance_cp'],
            f['F005_total_piece_count'],
            
            # Pawn structure counts
            f['F021_white_passed_pawn_count'],
            f['F021_black_passed_pawn_count'],
            f['F024_white_backward_pawn_count'],
            f['F024_black_backward_pawn_count'],
            f['F025_white_pawn_chain_length'],
            f['F025_black_pawn_chain_length'],
            f['F026_white_advanced_pawn_count'],
            f['F026_black_advanced_pawn_count'],
            f['F027_white_pawn_island_count'],
            f['F027_black_pawn_island_count'],
            
            # Piece activity
            f['F030_white_piece_mobility'],
            f['F030_black_piece_mobility'],
            f['F031_white_pieces_on_strong_squares'],
            f['F031_black_pieces_on_strong_squares'],
            
            # Tactical features
            f['F041_white_pieces_under_attack'],
            f['F041_black_pieces_under_attack'],
            f['F043_white_pieces_en_prise_value'],
            f['F043_black_pieces_en_prise_value'],
            f['F048_white_trapped_piece_count'],
            f['F048_black_trapped_piece_count'],
            
            # Rook placement
            f['F060_white_rooks_on_open_files'],
            f['F060_black_rooks_on_open_files'],
            f['F061_white_rooks_on_semi_open_files'],
            f['F061_black_rooks_on_semi_open_files'],
            f['F064_white_rook_activity_score'],
            f['F064_black_rook_activity_score'],
            
            # Knight features
            f['F070_white_knight_outposts'],
            f['F070_black_knight_outposts'],
            f['F071_white_knight_mobility_avg'],
            f['F071_black_knight_mobility_avg'],
            
            # Center control
            f['F080_white_center_pawn_count'],
            f['F080_black_center_pawn_count'],
            f['F081_white_center_control_score'],
            f['F081_black_center_control_score'],
            f['F082_white_space_advantage'],
            f['F082_black_space_advantage'],
            
            # Development
            f['F090_white_pieces_developed'],
            f['F090_black_pieces_developed'],
            
            # Multi-move context (CRITICAL FOR FIXING BINARY CLASSIFICATION)
            # Note: Puzzle positions won't have these, use defaults
            f.get('F100_best_move_eval_cp', 0),
            f.get('F101_second_move_eval_cp', 0),
            f.get('F102_third_move_eval_cp', 0),
            f.get('F103_fourth_move_eval_cp', 0),
            f.get('F104_fifth_move_eval_cp', 0),
            f.get('F105_eval_gap_best_to_second', 0),
            f.get('F106_eval_gap_second_to_third', 0),
            f.get('F107_v7p3r_move_eval_cp', 0),
            f.get('F108_v7p3r_eval_loss', 0),
            f.get('F109_move_diversity_score', 0),
            f.get('F110_position_sharpness', 0),
            f.get('F114_alternative_move_quality', 0)
        ]
        
        # ========== BOOLEAN FEATURES (~40) ==========
        boolean_row = [
            # King safety
            f['F010_white_king_castled'],
            f['F010_black_king_castled'],
            f['F011_white_king_has_pawn_shield'],
            f['F011_black_king_has_pawn_shield'],
            f['F012_white_king_under_attack'],
            f['F012_black_king_under_attack'],
            
            # Pawn structure
            f['F020_white_has_passed_pawns'],
            f['F020_black_has_passed_pawns'],
            f['F022_white_has_doubled_pawns'],
            f['F022_black_has_doubled_pawns'],
            f['F023_white_has_isolated_pawns'],
            f['F023_black_has_isolated_pawns'],
            
            # Piece pairs
            f['F032_white_has_bishop_pair'],
            f['F032_black_has_bishop_pair'],
            
            # Tactical threats
            f['F040_white_has_hanging_pieces'],
            f['F040_black_has_hanging_pieces'],
            f['F044_white_has_fork_threat'],
            f['F044_black_has_fork_threat'],
            f['F045_white_has_pin'],
            f['F045_black_has_pin'],
            f['F046_white_has_skewer'],
            f['F046_black_has_skewer'],
            f['F047_white_has_discovered_attack'],
            f['F047_black_has_discovered_attack'],
            f['F049_white_back_rank_threat'],
            f['F049_black_back_rank_threat'],
            
            # Move context (puzzle positions may not have these)
            f.get('F050_is_capture', 0),
            f.get('F051_is_check', 0),
            f.get('F052_is_promotion', 0),
            f.get('F053_is_castling', 0),
            
            # Rook placement
            f['F062_white_rook_on_7th_rank'],
            f['F062_black_rook_on_7th_rank'],
            f['F063_white_connected_rooks'],
            f['F063_black_connected_rooks'],
            
            # ========== TEMPORAL FEATURES (F200-F220) ==========
            # Historical tactical state (booleans/counts)
            f.get('white_hanging_pieces_historical', -1),
            f.get('black_hanging_pieces_historical', -1),
            f.get('white_pins_historical', -1),
            f.get('black_pins_historical', -1),
            f.get('white_king_under_attack_historical', -1),
            f.get('black_king_under_attack_historical', -1),
            f.get('white_trapped_pieces_historical', -1),
            f.get('black_trapped_pieces_historical', -1),
            
            # Forcing sequence flag
            f.get('is_forcing_sequence', 0),
            
            # Has history mask
            f.get('has_history', 0)
        ]
        
        # ========== TEMPORAL NUMERICAL FEATURES ==========
        # Add to numerical features (continuous values)
        numerical_temporal = [
            # Historical evaluation/material
            f.get('white_en_prise_value_historical', -999.0),
            f.get('black_en_prise_value_historical', -999.0),
            f.get('position_eval_historical', -999.0),
            f.get('material_balance_historical', -999.0),
            f.get('king_safety_white_historical', -1.0),
            f.get('king_safety_black_historical', -1.0),
            f.get('center_control_historical', -999.0),
            
            # Move sequence context
            f.get('move_sequence_index', 0)
        ]
        
        numerical_row.extend(numerical_temporal)
        
        # ========== CATEGORICAL FEATURES ==========
        # Game phase (opening/middlegame/endgame)
        game_phase.append(f['F002_game_phase'])
        
        # Material advantage category (balanced/white_advantage/black_advantage/etc.)
        material_cat.append(f['F004_material_advantage_category'])
        
        # Move types (quiet/capture/check/promotion) - 3 features
        # Note: Puzzle positions won't have these, use 'unknown' default
        move_types.append([
            f.get('F111_best_move_type', 'unknown'),
            f.get('F112_second_move_type', 'unknown'),
            f.get('F113_v7p3r_move_type', 'unknown')
        ])
        
        # ========== TEMPORAL CATEGORICAL FEATURES ==========
        # Move encoding (from_square, to_square, piece_type)
        # These will be one-hot encoded (64 + 64 + 7 = 135 features)
        move_encoding = [
            f.get('last_move_from_square', -1),  # 0-63 or -1 (no move)
            f.get('last_move_to_square', -1),     # 0-63 or -1 (no move)
            f.get('last_move_piece_type', 0)      # 0-6 (0=none, 1-6=pieces)
        ]
        
        numerical.append(numerical_row)
        boolean.append(boolean_row)
        move_encoding_list.append(move_encoding) if 'move_encoding_list' in locals() else None
    
    # Initialize move encoding list if not done yet
    if 'move_encoding_list' not in locals():
        move_encoding_list = []
        for record in records:
            f = record['features']
            move_encoding_list.append([
                f.get('last_move_from_square', -1),
                f.get('last_move_to_square', -1),
                f.get('last_move_piece_type', 0)
            ])
    
    return {
        'numerical': np.array(numerical, dtype=np.float32),
        'boolean': np.array(boolean, dtype=np.float32),
        'game_phase': np.array(game_phase).reshape(-1, 1),
        'material_cat': np.array(material_cat).reshape(-1, 1),
        'move_types': np.array(move_types),
        'move_encoding': np.array(move_encoding_list)  # NEW: Temporal move encoding
    }


def extract_targets(records):
    """
    Extract target variables for dual-head model
    
    Returns dict with:
        - policy: (N,) array of move quality grades 0-5
        - value: (N,) array of normalized position evaluations [-1, 1]
    """
    policy_targets = []
    value_targets = []
    
    for record in records:
        # Game position - has stockfish analysis
        if 'stockfish_analysis' in record and 'move_quality_grade' in record['stockfish_analysis']:
            stockfish = record['stockfish_analysis']
            
            # Policy: move quality grade (0-5)
            policy_targets.append(stockfish['move_quality_grade'])
            
            # Value: position evaluation (clipped and normalized)
            eval_cp = stockfish.get('best_move_eval_cp', 0)
            
            # Handle None (mate positions) - set to max advantage
            if eval_cp is None:
                eval_cp = 10000  # Treat as maximum advantage
            
            eval_cp = np.clip(eval_cp, -10000, 10000)
            eval_normalized = eval_cp / 10000.0  # Range: [-1, 1]
            value_targets.append(eval_normalized)
        
        # Puzzle position - map puzzle rating to synthetic grade
        elif 'puzzle_rating' in record:
            rating = record['puzzle_rating']
            # Map puzzle difficulty to grades (harder puzzles = better moves required)
            if rating < 1200:
                grade = 2  # Easy puzzles
            elif rating < 1600:
                grade = 3  # Medium puzzles
            elif rating < 2000:
                grade = 4  # Hard puzzles
            else:
                grade = 5  # Very hard puzzles
            
            policy_targets.append(grade)
            
            # Puzzles are typically winning positions (advantage to side to move)
            # Use moderate positive value
            value_targets.append(0.3)  # +3 pawns normalized
        
        # Fallback: unknown source, use neutral values
        else:
            policy_targets.append(3)  # Neutral grade
            value_targets.append(0.0)  # Equal position
    
    return {
        'policy': np.array(policy_targets, dtype=np.int64),
        'value': np.array(value_targets, dtype=np.float32)
    }


def preprocess_features(features, scaler=None, phase_enc=None, material_enc=None, move_type_enc=None, move_enc=None):
    """
    Apply preprocessing transformations (v5.1 + temporal features)
    
    Args:
        features: Dict with numerical, boolean, game_phase, material_cat, move_types, move_encoding
        scaler: StandardScaler (fit on training data if provided)
        phase_enc: OneHotEncoder for game phase
        material_enc: OneHotEncoder for material category
        move_type_enc: OneHotEncoder for move types
        move_enc: OneHotEncoder for temporal move encoding (NEW)
    
    Returns:
        X: (N, 262) preprocessed feature array
        transformers: Dict with fitted transformers (if fitting)
    """
    fitting = scaler is None
    
    # Fit or transform numerical features (includes temporal numerical)
    if fitting:
        scaler = StandardScaler()
        numerical_scaled = scaler.fit_transform(features['numerical'])
    else:
        numerical_scaled = scaler.transform(features['numerical'])
    
    # Fit or transform categorical features
    if fitting:
        phase_enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        phase_encoded = phase_enc.fit_transform(features['game_phase'])
        
        material_enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        material_encoded = material_enc.fit_transform(features['material_cat'])
        
        # Flatten move_types (N, 3) to (N*3, 1) for fitting, then reshape back
        move_types_flat = features['move_types'].reshape(-1, 1)
        move_type_enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        move_type_encoded_flat = move_type_enc.fit_transform(move_types_flat)
        
        # Reshape to (N, 3 * num_categories)
        n_samples = features['move_types'].shape[0]
        n_categories = move_type_encoded_flat.shape[1]
        move_type_encoded = move_type_encoded_flat.reshape(n_samples, 3 * n_categories)
        
        # NEW: Temporal move encoding (from_square, to_square, piece_type)
        # Flatten (N, 3) to (N*3, 1) for fitting
        move_encoding_flat = features['move_encoding'].reshape(-1, 1)
        move_enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        move_encoded_flat = move_enc.fit_transform(move_encoding_flat)
        
        # Reshape to (N, 3 * num_values)
        # Expected: 65 (0-63 + -1) for squares, 7 for piece types (0-6)
        # Total: 65 + 65 + 7 = 137 one-hot features
        n_move_values = move_encoded_flat.shape[1]
        move_encoded = move_encoded_flat.reshape(n_samples, 3 * n_move_values)
        
    else:
        phase_encoded = phase_enc.transform(features['game_phase'])
        material_encoded = material_enc.transform(features['material_cat'])
        
        move_types_flat = features['move_types'].reshape(-1, 1)
        move_type_encoded_flat = move_type_enc.transform(move_types_flat)
        n_samples = features['move_types'].shape[0]
        n_categories = move_type_encoded_flat.shape[1]
        move_type_encoded = move_type_encoded_flat.reshape(n_samples, 3 * n_categories)
        
        move_encoding_flat = features['move_encoding'].reshape(-1, 1)
        move_encoded_flat = move_enc.transform(move_encoding_flat)
        n_move_values = move_encoded_flat.shape[1]
        move_encoded = move_encoded_flat.reshape(n_samples, 3 * n_move_values)
    
    # Concatenate all features
    X = np.concatenate([
        numerical_scaled,           # ~60 features (current + temporal numerical)
        features['boolean'],        # ~50 features (current + temporal boolean)
        phase_encoded,              # 3 features (opening/middlegame/endgame)
        material_encoded,           # 5 features (balanced/white_advantage/etc.)
        move_type_encoded,          # ~12 features (3 move types * 4 categories)
        move_encoded                # ~137 features (temporal move encoding)
    ], axis=1)
    
    if fitting:
        return X, {
            'scaler': scaler,
            'phase_encoder': phase_enc,
            'material_encoder': material_enc,
            'move_type_encoder': move_type_enc,
            'move_encoder': move_enc  # NEW: Temporal move encoder
        }
    else:
        return X


def main():
    print("=" * 80)
    print("V7P3R AI v5.1 - Dataset Preprocessing (Expanded Features)")
    print("=" * 80)
    
    # Paths (allow override via environment variable)
    import os
    split_dir_str = os.environ.get('V7P3R_SPLIT_DIR', 'data/analysis/splits_v5.1')
    split_dir = Path(split_dir_str)
    output_dir = Path('data/preprocessed_v5.1')
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nUsing splits from: {split_dir}")
    if not split_dir.exists():
        print(f"ERROR: Split directory not found: {split_dir}")
        print("Run split_dataset.py first or set V7P3R_SPLIT_DIR environment variable")
        return
    
    # Load splits
    print("\nLoading dataset splits...")
    train_records = load_jsonl(split_dir / 'train.jsonl')
    val_records = load_jsonl(split_dir / 'validation.jsonl')
    test_records = load_jsonl(split_dir / 'test.jsonl')
    
    print(f"  Train: {len(train_records):,} positions")
    print(f"  Val:   {len(val_records):,} positions")
    print(f"  Test:  {len(test_records):,} positions")
    
    # Extract features
    print("\nExtracting features...")
    train_features = extract_features(train_records)
    val_features = extract_features(val_records)
    test_features = extract_features(test_records)
    
    print(f"  Numerical: {train_features['numerical'].shape}")
    print(f"  Boolean: {train_features['boolean'].shape}")
    print(f"  Categorical: game_phase, material_cat, move_types (3)")
    
    # Extract targets
    print("\nExtracting targets...")
    y_train = extract_targets(train_records)
    y_val = extract_targets(val_records)
    y_test = extract_targets(test_records)
    
    print(f"  Policy grades: {y_train['policy'].shape}")
    print(f"  Value evals: {y_train['value'].shape}")
    
    # Check grade distribution
    unique, counts = np.unique(y_train['policy'], return_counts=True)
    print(f"\nTraining set grade distribution:")
    for grade, count in zip(unique, counts):
        pct = 100 * count / len(y_train['policy'])
        print(f"    Grade {grade}: {count:6,} ({pct:5.2f}%)")
    
    # Fit transformers on training data
    print("\n⚙️ Fitting transformers on training data...")
    X_train, transformers = preprocess_features(train_features)
    
    print(f"  StandardScaler fitted on {len(train_records):,} positions")
    print(f"  Game phase categories: {len(transformers['phase_encoder'].categories_[0])}")
    print(f"  Material categories: {len(transformers['material_encoder'].categories_[0])}")
    print(f"  Move type categories: {len(transformers['move_type_encoder'].categories_[0])}")
    print(f"  Move encoding categories: {len(transformers['move_encoder'].categories_[0])}")  # NEW
    
    # Transform validation and test sets
    print("\n🔄 Transforming validation and test sets...")
    X_val = preprocess_features(
        val_features,
        scaler=transformers['scaler'],
        phase_enc=transformers['phase_encoder'],
        material_enc=transformers['material_encoder'],
        move_type_enc=transformers['move_type_encoder'],
        move_enc=transformers['move_encoder']  # NEW
    )
    
    X_test = preprocess_features(
        test_features,
        scaler=transformers['scaler'],
        phase_enc=transformers['phase_encoder'],
        material_enc=transformers['material_encoder'],
        move_type_enc=transformers['move_type_encoder'],
        move_enc=transformers['move_encoder']  # NEW
    )
    
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  X_test: {X_test.shape}")
    
    # Save preprocessed arrays
    print("\nSaving preprocessed arrays...")
    
    np.save(output_dir / 'X_train.npy', X_train)
    np.save(output_dir / 'y_train_policy.npy', y_train['policy'])
    np.save(output_dir / 'y_train_value.npy', y_train['value'])
    
    np.save(output_dir / 'X_val.npy', X_val)
    np.save(output_dir / 'y_val_policy.npy', y_val['policy'])
    np.save(output_dir / 'y_val_value.npy', y_val['value'])
    
    np.save(output_dir / 'X_test.npy', X_test)
    np.save(output_dir / 'y_test_policy.npy', y_test['policy'])
    np.save(output_dir / 'y_test_value.npy', y_test['value'])
    
    print(f"  ✓ Saved train/val/test arrays to {output_dir}/")
    
    # Save transformers
    print("\nSaving transformers...")
    with open(output_dir / 'transformers.pkl', 'wb') as f:
        pickle.dump(transformers, f)
    print(f"  ✓ Saved transformers to {output_dir / 'transformers.pkl'}")
    
    # Generate preprocessing stats
    print("\nGenerating preprocessing statistics...")
    stats = {
        'timestamp': datetime.now().isoformat(),
        'version': 'v5.1',
        'total_features': X_train.shape[1],
        'feature_breakdown': {
            'numerical': train_features['numerical'].shape[1],
            'boolean': train_features['boolean'].shape[1],
            'game_phase_onehot': len(transformers['phase_encoder'].categories_[0]),
            'material_cat_onehot': len(transformers['material_encoder'].categories_[0]),
            'move_type_onehot': 3 * len(transformers['move_type_encoder'].categories_[0])
        },
        'dataset_sizes': {
            'train': len(train_records),
            'validation': len(val_records),
            'test': len(test_records),
            'total': len(train_records) + len(val_records) + len(test_records)
        },
        'grade_distribution': {
            f'grade_{int(grade)}': int(count) 
            for grade, count in zip(unique, counts)
        },
        'numerical_feature_ranges': {
            'mean': X_train[:, :train_features['numerical'].shape[1]].mean(axis=0).tolist(),
            'std': X_train[:, :train_features['numerical'].shape[1]].std(axis=0).tolist()
        }
    }
    
    with open(output_dir / 'preprocessing_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f"  ✓ Saved statistics to {output_dir / 'preprocessing_stats.json'}")
    
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"\nFinal feature count: {X_train.shape[1]} features")
    print(f"  - Numerical: {train_features['numerical'].shape[1]}")
    print(f"  - Boolean: {train_features['boolean'].shape[1]}")
    print(f"  - Game phase (one-hot): {len(transformers['phase_encoder'].categories_[0])}")
    print(f"  - Material category (one-hot): {len(transformers['material_encoder'].categories_[0])}")
    print(f"  - Move types (one-hot): {3 * len(transformers['move_type_encoder'].categories_[0])}")
    print("\nReady for training with updated model architecture!")


if __name__ == '__main__':
    main()
