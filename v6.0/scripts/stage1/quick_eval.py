"""
Quick sampled evaluation - check if model is learning or just predicting majority class.
"""
import torch
import numpy as np
from pathlib import Path
import sys
import json
from collections import Counter

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from train_policy import GraphAugmentedPolicyNetwork


def discover_features(good_path, bad_path, sample_size=1000):
    """Discover all numeric features from a sample."""
    feature_names = set()
    
    # Sample from both files
    with open(good_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            pos = json.loads(line.strip())
            # Features are nested inside 'features' field
            if 'features' in pos and isinstance(pos['features'], dict):
                for key, val in pos['features'].items():
                    # Include all numeric types (int, float, bool)
                    if isinstance(val, (int, float, bool)) and not isinstance(val, str):
                        feature_names.add(key)
    
    with open(bad_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            pos = json.loads(line.strip())
            # Features are nested inside 'features' field
            if 'features' in pos and isinstance(pos['features'], dict):
                for key, val in pos['features'].items():
                    # Include all numeric types (int, float, bool)
                    if isinstance(val, (int, float, bool)) and not isinstance(val, str):
                        feature_names.add(key)
    
    return sorted(list(feature_names))


def load_sample_positions(good_path, bad_path, feature_names, sample_size=10000):
    """Load a sample of positions for quick evaluation."""
    print(f"\n📊 Loading sample ({sample_size:,} from each class)...")
    
    good_features = []
    good_labels = []
    bad_features = []
    bad_labels = []
    
    # Load good positions
    with open(good_path, 'r') as f:
        count = 0
        for line in f:
            if count >= sample_size:
                break
            
            pos = json.loads(line.strip())
            features_dict = pos.get('features', {})
            
            # Extract features
            vector = []
            for fname in feature_names:
                val = features_dict.get(fname, 0.0)
                try:
                    vector.append(float(val))
                except:
                    vector.append(0.0)
            
            good_features.append(vector)
            good_labels.append(0)  # Good = 0
            count += 1
    
    # Load bad positions (load ALL - there's only 69k)
    with open(bad_path, 'r') as f:
        count = 0
        for line in f:
            if count >= sample_size:
                break
            
            pos = json.loads(line.strip())
            features_dict = pos.get('features', {})
            
            # Extract features
            vector = []
            for fname in feature_names:
                val = features_dict.get(fname, 0.0)
                try:
                    vector.append(float(val))
                except:
                    vector.append(0.0)
            
            bad_features.append(vector)
            bad_labels.append(1)  # Bad = 1
            count += 1
    
    print(f"   Loaded {len(good_features):,} good + {len(bad_features):,} bad positions")
    
    # Combine and convert to numpy
    all_features = np.array(good_features + bad_features, dtype=np.float32)
    all_labels = np.array(good_labels + bad_labels, dtype=np.int32)
    
    # Normalize features (simple standardization)
    mean = all_features.mean(axis=0)
    std = all_features.std(axis=0) + 1e-8
    all_features = (all_features - mean) / std
    
    return all_features, all_labels


def evaluate(model, features, labels, device):
    """Evaluate model on features."""
    model.eval()
    
    features_tensor = torch.FloatTensor(features).to(device)
    labels_tensor = torch.FloatTensor(labels).to(device)
    
    with torch.no_grad():
        logits = model(features_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        predictions = (probs > 0.5).astype(int)
    
    labels_np = labels_tensor.cpu().numpy().astype(int)
    
    # Calculate metrics
    tp = np.sum((predictions == 1) & (labels_np == 1))  # True positives (bad detected as bad)
    fp = np.sum((predictions == 1) & (labels_np == 0))  # False positives (good detected as bad)
    tn = np.sum((predictions == 0) & (labels_np == 0))  # True negatives (good detected as good)
    fn = np.sum((predictions == 0) & (labels_np == 1))  # False negatives (bad detected as good)
    
    accuracy = (tp + tn) / len(labels_np)
    
    # Bad class metrics (class 1)
    bad_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    bad_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    bad_f1 = 2 * (bad_precision * bad_recall) / (bad_precision + bad_recall) if (bad_precision + bad_recall) > 0 else 0
    
    # Good class metrics (class 0)
    good_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
    good_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    good_f1 = 2 * (good_precision * good_recall) / (good_precision + good_recall) if (good_precision + good_recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': {'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)},
        'good': {'precision': good_precision, 'recall': good_recall, 'f1': good_f1, 'support': int(tn + fp)},
        'bad': {'precision': bad_precision, 'recall': bad_recall, 'f1': bad_f1, 'support': int(tp + fn)},
        'predictions': Counter(predictions.tolist())
    }


def main():
    """Quick evaluation."""
    base_path = Path(__file__).parent.parent.parent
    
    print("=" * 70)
    print("QUICK MODEL EVALUATION (SAMPLED)")
    print("=" * 70)
    
    # Load sample data
    good_path = base_path / "data" / "stage1" / "good_positions.jsonl"
    bad_path = base_path / "data" / "stage1" / "bad_positions.jsonl"
    
    # Discover features first
    print("\n🔍 Discovering feature schema...")
    feature_names = discover_features(str(good_path), str(bad_path))
    print(f"   Found {len(feature_names)} numeric features")
    
    features, labels = load_sample_positions(str(good_path), str(bad_path), feature_names, sample_size=10000)
    
    print(f"\n📊 Sample composition:")
    print(f"   Good (0): {np.sum(labels == 0):,}")
    print(f"   Bad (1): {np.sum(labels == 1):,}")
    print(f"   Class ratio: {np.sum(labels == 0) / np.sum(labels == 1):.1f}:1")
    
    # Load model
    print(f"\n🔄 Loading best model...")
    model_path = base_path / "models" / "stage1" / "stage1_policy_best.pt"
    checkpoint = torch.load(model_path, map_location='cpu')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    config = {
        'use_graph_attention': False
    }
    
    feature_dim = features.shape[1]
    print(f"   Feature dimension: {feature_dim}")
    
    model = GraphAugmentedPolicyNetwork(
        input_dim=feature_dim,
        config=config
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"   Loaded from epoch {checkpoint['epoch']+1}")
    
    # Evaluate
    print(f"\n🎯 Evaluating...")
    metrics = evaluate(model, features, labels, device)
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\n📈 Overall:")
    print(f"   Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    
    print(f"\n📊 Predictions Distribution:")
    print(f"   Predicted Good: {metrics['predictions'].get(0, 0):,}")
    print(f"   Predicted Bad: {metrics['predictions'].get(1, 0):,}")
    
    cm = metrics['confusion_matrix']
    print(f"\n📊 Confusion Matrix:")
    print(f"                Predicted Good   Predicted Bad")
    print(f"   Actual Good    {cm['tn']:>10,}     {cm['fp']:>10,}")
    print(f"   Actual Bad     {cm['fn']:>10,}     {cm['tp']:>10,}")
    
    print(f"\n🟢 Good Moves (Class 0):")
    print(f"   Precision: {metrics['good']['precision']:.4f} ({metrics['good']['precision']*100:.2f}%)")
    print(f"   Recall: {metrics['good']['recall']:.4f} ({metrics['good']['recall']*100:.2f}%)")
    print(f"   F1-Score: {metrics['good']['f1']:.4f} ({metrics['good']['f1']*100:.2f}%)")
    print(f"   Support: {metrics['good']['support']:,}")
    
    print(f"\n🔴 Bad Moves (Class 1):")
    print(f"   Precision: {metrics['bad']['precision']:.4f} ({metrics['bad']['precision']*100:.2f}%)")
    print(f"   Recall: {metrics['bad']['recall']:.4f} ({metrics['bad']['recall']*100:.2f}%)")
    print(f"   F1-Score: {metrics['bad']['f1']:.4f} ({metrics['bad']['f1']*100:.2f}%)")
    print(f"   Support: {metrics['bad']['support']:,}")
    
    # Target check
    print("\n" + "=" * 70)
    print("TARGET METRICS CHECK")
    print("=" * 70)
    
    checks = [
        ("Accuracy ≥ 95%", metrics['accuracy'] >= 0.95, metrics['accuracy'] * 100),
        ("Bad Recall ≥ 60%", metrics['bad']['recall'] >= 0.60, metrics['bad']['recall'] * 100),
        ("Bad F1 ≥ 55%", metrics['bad']['f1'] >= 0.55, metrics['bad']['f1'] * 100),
    ]
    
    for check_name, passed, value in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {check_name}: {value:.2f}%")
    
    all_passed = all(check[1] for check in checks)
    
    # Diagnosis
    print("\n" + "=" * 70)
    if metrics['predictions'].get(1, 0) == 0:
        print("❌ MODEL IS PREDICTING MAJORITY CLASS ONLY!")
        print("   The model is not learning - it predicts everything as 'good'")
        print("   This is a common issue with imbalanced datasets")
        print("\n💡 Recommended fixes:")
        print("   1. Increase class weight for bad moves (currently good=0.006, bad=1.0)")
        print("   2. Try focal loss instead of BCE")
        print("   3. Undersample good moves or oversample bad moves")
        print("   4. Lower learning rate for more careful optimization")
    elif all_passed:
        print("🎉 ALL TARGET METRICS ACHIEVED!")
        print("   Model successfully learns to detect bad moves")
        print("   Ready for Stage 2 (self-play training)")
    else:
        print("⚠️  Some metrics not met but model IS learning")
        print("   Model detects some bad moves but needs improvement")
        print("\n💡 Consider:")
        print("   - Adjust class weights")
        print("   - Train longer (increase patience)")
        print("   - Increase model capacity")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
