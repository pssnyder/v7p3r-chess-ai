"""
Evaluate trained model on test set with detailed metrics.
"""
import torch
import numpy as np
from pathlib import Path
import sys
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from train_policy import DataLoader, GraphAugmentedPolicyNetwork


def evaluate_model(model, test_features, test_labels, device):
    """
    Evaluate model on test set.
    
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    
    # Convert to tensors
    features_tensor = torch.FloatTensor(test_features).to(device)
    labels_tensor = torch.FloatTensor(test_labels).to(device)
    
    # Get predictions
    with torch.no_grad():
        logits = model(features_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        predictions = (probs > 0.5).astype(int)
    
    labels_np = labels_tensor.cpu().numpy().astype(int)
    
    # Calculate metrics
    cm = confusion_matrix(labels_np, predictions)
    
    # True negatives, false positives, false negatives, true positives
    tn, fp, fn, tp = cm.ravel()
    
    # Per-class metrics
    good_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
    good_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    good_f1 = 2 * (good_precision * good_recall) / (good_precision + good_recall) if (good_precision + good_recall) > 0 else 0
    
    bad_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    bad_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    bad_f1 = 2 * (bad_precision * bad_recall) / (bad_precision + bad_recall) if (bad_precision + bad_recall) > 0 else 0
    
    # Overall metrics
    accuracy = (tn + tp) / (tn + fp + fn + tp)
    
    try:
        auc = roc_auc_score(labels_np, probs)
    except:
        auc = None
    
    metrics = {
        'accuracy': accuracy,
        'auc': auc,
        'confusion_matrix': cm.tolist(),
        'good_class': {
            'precision': good_precision,
            'recall': good_recall,
            'f1': good_f1,
            'support': int(tn + fp)
        },
        'bad_class': {
            'precision': bad_precision,
            'recall': bad_recall,
            'f1': bad_f1,
            'support': int(tp + fn)
        }
    }
    
    return metrics, predictions, probs


def main():
    """Main evaluation pipeline."""
    base_path = Path(__file__).parent.parent.parent
    
    print("=" * 70)
    print("MODEL EVALUATION ON TEST SET")
    print("=" * 70)
    
    # Configuration
    config = {
        'batch_size': 2048,
        'use_graph_attention': False,
        'hidden_layers': [512, 256, 128],
        'dropout_rate': 0.3,
        'output_dir': 'models/stage1'
    }
    
    # Load data
    print("\n📊 Loading data...")
    good_path = base_path / "data" / "stage1" / "good_positions.jsonl"
    bad_path = base_path / "data" / "stage1" / "bad_positions.jsonl"
    graph_path = base_path / "data" / "stage1" / "transposition_graph.pkl"
    
    loader = DataLoader(str(good_path), str(bad_path), str(graph_path), config)
    data = loader.load(max_samples=None)
    
    print(f"   Test set size: {len(data['test_labels']):,} positions")
    print(f"   Good: {np.sum(data['test_labels'] == 0):,}")
    print(f"   Bad: {np.sum(data['test_labels'] == 1):,}")
    
    # Load best model
    print("\n🔄 Loading best model...")
    model_path = base_path / "models" / "stage1" / "stage1_policy_best.pt"
    checkpoint = torch.load(model_path, map_location='cpu')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    # Create model
    model = GraphAugmentedPolicyNetwork(
        input_dim=data['feature_dim'],
        hidden_layers=config['hidden_layers'],
        dropout_rate=config['dropout_rate'],
        use_graph_attention=config['use_graph_attention']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"   Loaded from epoch {checkpoint['epoch']+1}")
    
    # Evaluate
    print("\n🎯 Evaluating on test set...")
    metrics, predictions, probs = evaluate_model(
        model, 
        data['test_features'], 
        data['test_labels'], 
        device
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\n📈 Overall Metrics:")
    print(f"   Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    if metrics['auc']:
        print(f"   AUC-ROC: {metrics['auc']:.4f}")
    
    print(f"\n🟢 Good Moves (Class 0):")
    print(f"   Precision: {metrics['good_class']['precision']:.4f}")
    print(f"   Recall: {metrics['good_class']['recall']:.4f}")
    print(f"   F1-Score: {metrics['good_class']['f1']:.4f}")
    print(f"   Support: {metrics['good_class']['support']:,}")
    
    print(f"\n🔴 Bad Moves (Class 1):")
    print(f"   Precision: {metrics['bad_class']['precision']:.4f}")
    print(f"   Recall: {metrics['bad_class']['recall']:.4f}")
    print(f"   F1-Score: {metrics['bad_class']['f1']:.4f}")
    print(f"   Support: {metrics['bad_class']['support']:,}")
    
    print(f"\n📊 Confusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    print(f"                Predicted Good   Predicted Bad")
    print(f"   Actual Good    {cm[0][0]:>10,}     {cm[0][1]:>10,}")
    print(f"   Actual Bad     {cm[1][0]:>10,}     {cm[1][1]:>10,}")
    
    # Target metrics check
    print("\n" + "=" * 70)
    print("TARGET METRICS CHECK")
    print("=" * 70)
    
    checks = []
    checks.append(("Accuracy ≥ 95%", metrics['accuracy'] >= 0.95, metrics['accuracy'] * 100))
    checks.append(("Bad Recall ≥ 60%", metrics['bad_class']['recall'] >= 0.60, metrics['bad_class']['recall'] * 100))
    checks.append(("Bad F1 ≥ 55%", metrics['bad_class']['f1'] >= 0.55, metrics['bad_class']['f1'] * 100))
    
    for check_name, passed, value in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {check_name}: {value:.2f}%")
    
    all_passed = all(check[1] for check in checks)
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL TARGET METRICS ACHIEVED!")
        print("   Model is ready for Stage 2 (self-play training)")
    else:
        print("⚠️  Some target metrics not met")
        print("   Consider hyperparameter tuning or model adjustments")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
