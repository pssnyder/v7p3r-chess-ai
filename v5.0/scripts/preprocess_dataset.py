"""
V7P3R AI v5.0 - Dataset Preprocessing Script

Converts JSONL training data into preprocessed NumPy arrays for training.

Pipeline:
1. Load train/val/test splits from JSONL
2. Extract features and targets
3. Fit transformers on training data (StandardScaler, OneHotEncoder)
4. Transform all splits
5. Save preprocessed arrays and transformers

Usage:
    python scripts/preprocess_dataset.py
    
Output:
    - data/preprocessed/X_train.npy, y_train_policy.npy, y_train_value.npy
    - data/preprocessed/X_val.npy, y_val_policy.npy, y_val_value.npy
    - data/preprocessed/X_test.npy, y_test_policy.npy, y_test_value.npy
    - data/preprocessed/transformers.pkl (scaler + encoders)
    - data/preprocessed/preprocessing_stats.json
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
    Extract features from dataset records
    
    Returns dict with:
        - numerical: (N, 6) array
        - boolean: (N, 12) array
        - game_phase: (N, 1) array
        - material_cat: (N, 1) array
    """
    numerical = []
    boolean = []
    game_phase = []
    material_cat = []
    
    for record in records:
        f = record['features']
        
        # Numerical features (6)
        numerical.append([
            f['F003_material_balance_cp'],
            f['F005_total_piece_count'],
            f['F030_white_piece_mobility'],
            f['F030_black_piece_mobility'],
            f['F031_white_pieces_on_strong_squares'],
            f['F031_black_pieces_on_strong_squares']
        ])
        
        # Boolean features (12)
        boolean.append([
            f['F010_white_king_castled'],
            f['F010_black_king_castled'],
            f['F011_white_king_has_pawn_shield'],
            f['F011_black_king_has_pawn_shield'],
            f['F012_white_king_under_attack'],
            f['F012_black_king_under_attack'],
            f['F032_white_has_bishop_pair'],
            f['F032_black_has_bishop_pair'],
            f['F050_is_capture'],
            f['F051_is_check'],
            f['F052_is_promotion'],
            f['F053_is_castling']
        ])
        
        # Categorical features
        game_phase.append(f['F002_game_phase'])
        material_cat.append(f['F004_material_advantage_category'])
    
    return {
        'numerical': np.array(numerical, dtype=np.float32),
        'boolean': np.array(boolean, dtype=np.float32),
        'game_phase': np.array(game_phase).reshape(-1, 1),
        'material_cat': np.array(material_cat).reshape(-1, 1)
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
        stockfish = record['stockfish_analysis']
        
        # Policy: move quality grade (0-5)
        policy_targets.append(stockfish['move_quality_grade'])
        
        # Value: position evaluation (clipped and normalized)
        eval_cp = stockfish['best_move_eval_cp']
        
        # Handle None (mate positions) - set to max advantage
        if eval_cp is None:
            eval_cp = 10000  # Treat as maximum advantage
        
        eval_cp = np.clip(eval_cp, -10000, 10000)
        eval_normalized = eval_cp / 10000.0  # Range: [-1, 1]
        value_targets.append(eval_normalized)
    
    return {
        'policy': np.array(policy_targets, dtype=np.int64),
        'value': np.array(value_targets, dtype=np.float32)
    }


def preprocess_features(features, scaler=None, phase_enc=None, material_enc=None):
    """
    Apply preprocessing transformations
    
    Args:
        features: Dict with numerical, boolean, game_phase, material_cat
        scaler: StandardScaler (fit on training data if provided)
        phase_enc: OneHotEncoder for game phase
        material_enc: OneHotEncoder for material category
    
    Returns:
        X: (N, 26) preprocessed feature array
        transformers: Dict with fitted transformers (if fitting)
    """
    fitting = scaler is None
    
    # Fit or transform numerical features
    if fitting:
        scaler = StandardScaler()
        numerical_scaled = scaler.fit_transform(features['numerical'])
    else:
        numerical_scaled = scaler.transform(features['numerical'])
    
    # Fit or transform categorical features
    if fitting:
        phase_enc = OneHotEncoder(sparse_output=False)
        phase_encoded = phase_enc.fit_transform(features['game_phase'])
        
        material_enc = OneHotEncoder(sparse_output=False)
        material_encoded = material_enc.fit_transform(features['material_cat'])
    else:
        phase_encoded = phase_enc.transform(features['game_phase'])
        material_encoded = material_enc.transform(features['material_cat'])
    
    # Concatenate all features
    X = np.concatenate([
        numerical_scaled,           # 6 features
        features['boolean'],        # 12 features
        phase_encoded,              # 3 features
        material_encoded            # 5 features
    ], axis=1)  # Total: 26 features
    
    if fitting:
        return X, {
            'scaler': scaler,
            'phase_encoder': phase_enc,
            'material_encoder': material_enc
        }
    else:
        return X


def main():
    print("=" * 80)
    print("V7P3R AI v5.0 - Dataset Preprocessing")
    print("=" * 80)
    
    # Paths
    split_dir = Path('data/analysis/splits')
    output_dir = Path('data/preprocessed')
    output_dir.mkdir(exist_ok=True)
    
    # Load splits
    print("\n📂 Loading dataset splits...")
    train_records = load_jsonl(split_dir / 'train.jsonl')
    val_records = load_jsonl(split_dir / 'validation.jsonl')
    test_records = load_jsonl(split_dir / 'test.jsonl')
    
    print(f"  Train: {len(train_records):,} positions")
    print(f"  Val:   {len(val_records):,} positions")
    print(f"  Test:  {len(test_records):,} positions")
    
    # Extract features
    print("\n🔧 Extracting features...")
    train_features = extract_features(train_records)
    val_features = extract_features(val_records)
    test_features = extract_features(test_records)
    
    print(f"  Numerical: {train_features['numerical'].shape}")
    print(f"  Boolean: {train_features['boolean'].shape}")
    print(f"  Categorical: 2 fields")
    
    # Extract targets
    print("\n🎯 Extracting targets...")
    y_train = extract_targets(train_records)
    y_val = extract_targets(val_records)
    y_test = extract_targets(test_records)
    
    print(f"  Policy grades: {y_train['policy'].shape}")
    print(f"  Value evals: {y_train['value'].shape}")
    
    # Fit transformers on training data
    print("\n⚙️ Fitting transformers on training data...")
    X_train, transformers = preprocess_features(train_features)
    
    print(f"  StandardScaler fitted on {len(train_records):,} positions")
    print(f"  Game phase categories: {len(transformers['phase_encoder'].categories_[0])}")
    print(f"  Material categories: {len(transformers['material_encoder'].categories_[0])}")
    
    # Transform validation and test sets
    print("\n🔄 Transforming validation and test sets...")
    X_val = preprocess_features(
        val_features,
        scaler=transformers['scaler'],
        phase_enc=transformers['phase_encoder'],
        material_enc=transformers['material_encoder']
    )
    
    X_test = preprocess_features(
        test_features,
        scaler=transformers['scaler'],
        phase_enc=transformers['phase_encoder'],
        material_enc=transformers['material_encoder']
    )
    
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  X_test: {X_test.shape}")
    
    # Save preprocessed arrays
    print("\n💾 Saving preprocessed arrays...")
    
    np.save(output_dir / 'X_train.npy', X_train)
    np.save(output_dir / 'y_train_policy.npy', y_train['policy'])
    np.save(output_dir / 'y_train_value.npy', y_train['value'])
    
    np.save(output_dir / 'X_val.npy', X_val)
    np.save(output_dir / 'y_val_policy.npy', y_val['policy'])
    np.save(output_dir / 'y_val_value.npy', y_val['value'])
    
    np.save(output_dir / 'X_test.npy', X_test)
    np.save(output_dir / 'y_test_policy.npy', y_test['policy'])
    np.save(output_dir / 'y_test_value.npy', y_test['value'])
    
    print(f"  ✅ Arrays saved to {output_dir}/")
    
    # Save transformers
    print("\n💾 Saving transformers...")
    with open(output_dir / 'transformers.pkl', 'wb') as f:
        pickle.dump(transformers, f)
    
    print(f"  ✅ Transformers saved to {output_dir / 'transformers.pkl'}")
    
    # Generate statistics
    print("\n📊 Generating preprocessing statistics...")
    
    stats = {
        'timestamp': datetime.now().isoformat(),
        'splits': {
            'train': len(train_records),
            'val': len(val_records),
            'test': len(test_records),
            'total': len(train_records) + len(val_records) + len(test_records)
        },
        'feature_dimensions': {
            'numerical': 6,
            'boolean': 12,
            'game_phase': 3,
            'material_category': 5,
            'total': 26
        },
        'numerical_stats': {
            'mean': transformers['scaler'].mean_.tolist(),
            'std': transformers['scaler'].scale_.tolist()
        },
        'categorical_stats': {
            'game_phase_categories': transformers['phase_encoder'].categories_[0].tolist(),
            'material_categories': transformers['material_encoder'].categories_[0].tolist()
        },
        'target_stats': {
            'policy_grade_distribution': {
                str(grade): int(np.sum(y_train['policy'] == grade))
                for grade in range(6)
            },
            'value_range': [float(y_train['value'].min()), float(y_train['value'].max())],
            'value_mean': float(y_train['value'].mean()),
            'value_std': float(y_train['value'].std())
        }
    }
    
    with open(output_dir / 'preprocessing_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"  ✅ Statistics saved to {output_dir / 'preprocessing_stats.json'}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("✅ Preprocessing Complete!")
    print("=" * 80)
    print(f"\nPreprocessed data saved to: {output_dir}/")
    print(f"  - X_train.npy ({X_train.nbytes / (1024**2):.2f} MB)")
    print(f"  - X_val.npy ({X_val.nbytes / (1024**2):.2f} MB)")
    print(f"  - X_test.npy ({X_test.nbytes / (1024**2):.2f} MB)")
    print(f"  - Target arrays (policy + value)")
    print(f"  - transformers.pkl")
    print(f"  - preprocessing_stats.json")
    
    print(f"\nReady for training!")
    print(f"  python src/train.py")
    print("=" * 80)


if __name__ == '__main__':
    main()
