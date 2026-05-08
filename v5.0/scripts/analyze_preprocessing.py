"""
Analyze dataset for preprocessing requirements:
- Null/missing values in feature fields
- Categorical fields needing one-hot encoding
- Numerical fields needing normalization
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

def analyze_preprocessing_needs(dataset_path):
    """Analyze dataset for preprocessing requirements"""
    
    print(f"Analyzing: {dataset_path}")
    print("=" * 80)
    
    # Counters
    null_features = Counter()
    feature_types = defaultdict(set)
    feature_ranges = defaultdict(lambda: {'min': float('inf'), 'max': float('-inf')})
    total_positions = 0
    
    # Sample records for categorical analysis
    categorical_samples = defaultdict(set)
    
    # Analyze all records
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            total_positions += 1
            
            features = record.get('features', {})
            
            for field, value in features.items():
                # Check for nulls
                if value is None:
                    null_features[field] += 1
                else:
                    # Track data type
                    feature_types[field].add(type(value).__name__)
                    
                    # Track numerical ranges
                    if isinstance(value, (int, float)):
                        feature_ranges[field]['min'] = min(feature_ranges[field]['min'], value)
                        feature_ranges[field]['max'] = max(feature_ranges[field]['max'], value)
                    
                    # Sample categorical values (limit to 20 unique values)
                    elif isinstance(value, (str, bool)):
                        if len(categorical_samples[field]) < 20:
                            categorical_samples[field].add(str(value))
    
    print(f"\n📊 DATASET SUMMARY")
    print(f"Total positions: {total_positions:,}")
    print(f"Total feature fields: {len(features.keys())}")
    
    # NULL ANALYSIS
    print(f"\n❌ NULL/MISSING VALUES")
    if null_features:
        print(f"Features with null values: {len(null_features)}")
        for field, count in sorted(null_features.items(), key=lambda x: x[1], reverse=True)[:20]:
            pct = (count / total_positions) * 100
            print(f"  {field}: {count:,} ({pct:.2f}%)")
    else:
        print("✅ No null values found!")
    
    # FEATURE TYPE ANALYSIS
    print(f"\n🔤 FEATURE DATA TYPES")
    
    numerical_features = []
    categorical_features = []
    boolean_features = []
    mixed_features = []
    
    for field, types in sorted(feature_types.items()):
        if len(types) > 1:
            mixed_features.append((field, types))
        elif 'bool' in types:
            boolean_features.append(field)
        elif 'int' in types or 'float' in types:
            numerical_features.append(field)
        elif 'str' in types:
            categorical_features.append(field)
    
    print(f"\nNumerical features: {len(numerical_features)}")
    for field in numerical_features[:10]:
        r = feature_ranges[field]
        print(f"  {field}: [{r['min']}, {r['max']}]")
    if len(numerical_features) > 10:
        print(f"  ... and {len(numerical_features) - 10} more")
    
    print(f"\nBoolean features: {len(boolean_features)}")
    for field in boolean_features[:10]:
        print(f"  {field}")
    if len(boolean_features) > 10:
        print(f"  ... and {len(boolean_features) - 10} more")
    
    print(f"\nCategorical (string) features: {len(categorical_features)}")
    for field in categorical_features:
        samples = categorical_samples[field]
        print(f"  {field}: {len(samples)} unique values")
        if len(samples) <= 5:
            print(f"    Values: {sorted(samples)}")
    
    if mixed_features:
        print(f"\n⚠️ Mixed-type features: {len(mixed_features)}")
        for field, types in mixed_features:
            print(f"  {field}: {types}")
    
    # PREPROCESSING RECOMMENDATIONS
    print(f"\n💡 PREPROCESSING RECOMMENDATIONS")
    
    print("\n1. NULL HANDLING:")
    if not null_features:
        print("   ✅ No null values - no imputation needed")
    else:
        print(f"   ⚠️ {len(null_features)} features have nulls")
        print("   Options:")
        print("   - Impute with 0 (if meaningful)")
        print("   - Impute with mean/median (for numerical)")
        print("   - Add 'missing' indicator feature")
        print("   - Drop positions with nulls (if <1% affected)")
    
    print("\n2. ONE-HOT ENCODING:")
    if categorical_features:
        print(f"   🔥 {len(categorical_features)} categorical features need encoding:")
        for field in categorical_features:
            n_unique = len(categorical_samples[field])
            print(f"   - {field} → {n_unique} binary features")
    else:
        print("   ✅ No categorical features - no one-hot encoding needed")
    
    print("\n3. NORMALIZATION:")
    print(f"   📏 {len(numerical_features)} numerical features should be normalized:")
    print("   Recommended: StandardScaler (mean=0, std=1)")
    large_range_features = [f for f in numerical_features 
                            if feature_ranges[f]['max'] - feature_ranges[f]['min'] > 100]
    if large_range_features:
        print(f"   ⚠️ {len(large_range_features)} features have large ranges (>100):")
        for field in large_range_features[:5]:
            r = feature_ranges[field]
            print(f"   - {field}: [{r['min']}, {r['max']}]")
    
    print("\n4. BOOLEAN FEATURES:")
    print(f"   ✅ {len(boolean_features)} boolean features are already 0/1")
    print("   No additional encoding needed")
    
    # FEATURE DIMENSIONALITY
    print(f"\n📐 FEATURE DIMENSIONALITY ESTIMATE")
    
    numerical_dims = len(numerical_features)
    boolean_dims = len(boolean_features)
    categorical_dims = sum(len(categorical_samples[f]) for f in categorical_features)
    
    total_dims = numerical_dims + boolean_dims + categorical_dims
    
    print(f"   Numerical: {numerical_dims}")
    print(f"   Boolean: {boolean_dims}")
    print(f"   Categorical (one-hot): {categorical_dims}")
    print(f"   " + "=" * 40)
    print(f"   TOTAL INPUT DIMENSIONS: {total_dims}")
    
    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    
    return {
        'total_positions': total_positions,
        'null_features': dict(null_features),
        'numerical_features': numerical_features,
        'boolean_features': boolean_features,
        'categorical_features': categorical_features,
        'feature_ranges': dict(feature_ranges),
        'categorical_samples': {k: list(v) for k, v in categorical_samples.items()},
        'total_input_dims': total_dims
    }

if __name__ == '__main__':
    dataset_path = Path('data/final/v7p3r_ai_v5_training_dataset_complete.jsonl')
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        sys.exit(1)
    
    results = analyze_preprocessing_needs(dataset_path)
    
    # Save results
    output_path = Path('data/analysis/preprocessing_analysis.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
