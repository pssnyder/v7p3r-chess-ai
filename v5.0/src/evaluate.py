"""
V7P3R AI v5.0 - Model Evaluation Script

Evaluates trained model on test set and generates detailed metrics.

Usage:
    python src/evaluate.py --checkpoint checkpoints/best_model.pth
    python src/evaluate.py --checkpoint checkpoints/best_model.pth --save-predictions
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
import sys

sys.path.append(str(Path(__file__).parent))

from model import V7P3R_AI_v5
from dataset import V7P3RDataset, create_dataloaders


def evaluate_model(model, test_loader, device):
    """
    Comprehensive model evaluation
    
    Returns:
        metrics: Dict with evaluation metrics
        predictions: Dict with predictions and targets
    """
    model.eval()
    
    all_policy_preds = []
    all_policy_targets = []
    all_value_preds = []
    all_value_targets = []
    
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.HuberLoss(delta=0.5)
    
    total_loss = 0
    total_policy_loss = 0
    total_value_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            policy_targets = batch['policy_target'].to(device)
            value_targets = batch['value_target'].to(device)
            
            policy_logits, value_preds = model(features)
            
            # Losses
            policy_loss = policy_criterion(policy_logits, policy_targets)
            value_loss = value_criterion(value_preds, value_targets)
            loss = policy_loss + 0.1 * value_loss
            
            # Store predictions
            policy_pred = policy_logits.argmax(1)
            all_policy_preds.extend(policy_pred.cpu().numpy())
            all_policy_targets.extend(policy_targets.cpu().numpy())
            all_value_preds.extend(value_preds.cpu().numpy().flatten())
            all_value_targets.extend(value_targets.cpu().numpy().flatten())
            
            # Accumulate losses
            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            total_policy_loss += policy_loss.item() * batch_size
            total_value_loss += value_loss.item() * batch_size
            total_samples += batch_size
    
    # Convert to numpy
    all_policy_preds = np.array(all_policy_preds)
    all_policy_targets = np.array(all_policy_targets)
    all_value_preds = np.array(all_value_preds)
    all_value_targets = np.array(all_value_targets)
    
    # Calculate metrics
    metrics = calculate_metrics(
        all_policy_preds, all_policy_targets,
        all_value_preds, all_value_targets,
        total_loss / total_samples,
        total_policy_loss / total_samples,
        total_value_loss / total_samples
    )
    
    predictions = {
        'policy_preds': all_policy_preds,
        'policy_targets': all_policy_targets,
        'value_preds': all_value_preds,
        'value_targets': all_value_targets
    }
    
    return metrics, predictions


def calculate_metrics(policy_preds, policy_targets, value_preds, value_targets,
                     total_loss, policy_loss, value_loss):
    """Calculate comprehensive evaluation metrics"""
    
    # Policy metrics
    policy_accuracy = (policy_preds == policy_targets).mean()
    
    # Top-2 accuracy (within 1 grade)
    top2_correct = np.abs(policy_preds - policy_targets) <= 1
    top2_accuracy = top2_correct.mean()
    
    # Top-3 accuracy (within 2 grades)
    top3_correct = np.abs(policy_preds - policy_targets) <= 2
    top3_accuracy = top3_correct.mean()
    
    # Value metrics
    value_mae = np.abs(value_preds - value_targets).mean()
    value_mse = ((value_preds - value_targets) ** 2).mean()
    value_rmse = np.sqrt(value_mse)
    
    # Correlation
    value_corr = np.corrcoef(value_preds, value_targets)[0, 1]
    
    # Per-grade metrics
    per_grade_metrics = {}
    for grade in range(6):
        mask = policy_targets == grade
        if mask.sum() > 0:
            grade_acc = (policy_preds[mask] == grade).mean()
            per_grade_metrics[f'grade_{grade}_accuracy'] = float(grade_acc)
            per_grade_metrics[f'grade_{grade}_count'] = int(mask.sum())
    
    return {
        'loss': float(total_loss),
        'policy_loss': float(policy_loss),
        'value_loss': float(value_loss),
        'policy_accuracy': float(policy_accuracy),
        'policy_top2_accuracy': float(top2_accuracy),
        'policy_top3_accuracy': float(top3_accuracy),
        'value_mae': float(value_mae),
        'value_mse': float(value_mse),
        'value_rmse': float(value_rmse),
        'value_correlation': float(value_corr),
        'per_grade': per_grade_metrics
    }


def generate_report(metrics, predictions, output_dir):
    """Generate evaluation report with visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Confusion matrix
    cm = confusion_matrix(predictions['policy_targets'], predictions['policy_preds'])
    
    # Classification report
    clf_report = classification_report(
        predictions['policy_targets'],
        predictions['policy_preds'],
        target_names=[f'Grade {i}' for i in range(6)],
        output_dict=True
    )
    
    # Create markdown report
    report_md = f"""# V7P3R AI v5.0 - Evaluation Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Overall Performance

| Metric | Value |
|--------|-------|
| **Total Loss** | {metrics['loss']:.4f} |
| **Policy Loss** | {metrics['policy_loss']:.4f} |
| **Value Loss** | {metrics['value_loss']:.4f} |

---

## Policy Head Metrics (Move Quality)

| Metric | Value | Target |
|--------|-------|--------|
| **Exact Match Accuracy** | {metrics['policy_accuracy']*100:.2f}% | >50% |
| **Top-2 Accuracy** (±1 grade) | {metrics['policy_top2_accuracy']*100:.2f}% | >75% |
| **Top-3 Accuracy** (±2 grades) | {metrics['policy_top3_accuracy']*100:.2f}% | >85% |

### Per-Grade Performance

| Grade | Accuracy | Sample Count |
|-------|----------|--------------|
"""
    
    for grade in range(6):
        if f'grade_{grade}_accuracy' in metrics['per_grade']:
            acc = metrics['per_grade'][f'grade_{grade}_accuracy'] * 100
            count = metrics['per_grade'][f'grade_{grade}_count']
            report_md += f"| {grade} | {acc:.2f}% | {count:,} |\n"
    
    report_md += f"""
### Confusion Matrix

```
     Predicted Grade
     0      1      2      3      4      5
"""
    
    for i, row in enumerate(cm):
        row_str = f"{i}  "
        for val in row:
            row_str += f"{val:6d} "
        report_md += row_str + "\n"
    
    report_md += f"""```

---

## Value Head Metrics (Position Evaluation)

| Metric | Value | Target |
|--------|-------|--------|
| **MAE** (Mean Absolute Error) | {metrics['value_mae']:.4f} | <0.15 |
| **RMSE** (Root Mean Squared Error) | {metrics['value_rmse']:.4f} | <0.20 |
| **Correlation** | {metrics['value_correlation']:.4f} | >0.80 |

**Note**: Value predictions are in [-1, 1] range (multiply by 10000 for centipawns)

---

## Interpretation

### Policy Head
- Model correctly predicts exact move quality **{metrics['policy_accuracy']*100:.1f}%** of the time
- Within ±1 grade: **{metrics['policy_top2_accuracy']*100:.1f}%** (practical accuracy)
- Within ±2 grades: **{metrics['policy_top3_accuracy']*100:.1f}%** (near-miss tolerance)

### Value Head
- Average evaluation error: **{metrics['value_mae']*10000:.0f} centipawns**
- Position evaluation correlation: **{metrics['value_correlation']:.3f}** (vs Stockfish)

---

## Baseline Comparison

| Metric | Baseline (Random) | Model | Improvement |
|--------|-------------------|-------|-------------|
| Policy Accuracy | 16.7% | {metrics['policy_accuracy']*100:.1f}% | {metrics['policy_accuracy']*100/16.7:.1f}x |
| Value MAE | ~0.30 | {metrics['value_mae']:.3f} | {0.30/metrics['value_mae']:.1f}x better |

---

## Training Targets Status

"""
    
    # Check targets
    target_checks = []
    target_checks.append(("Policy Accuracy >50%", metrics['policy_accuracy'] > 0.50, metrics['policy_accuracy']))
    target_checks.append(("Top-2 Accuracy >75%", metrics['policy_top2_accuracy'] > 0.75, metrics['policy_top2_accuracy']))
    target_checks.append(("Value MAE <0.15", metrics['value_mae'] < 0.15, metrics['value_mae']))
    
    for target_name, achieved, value in target_checks:
        status = "✅" if achieved else "❌"
        report_md += f"{status} {target_name}\n"
    
    report_md += "\n---\n\n*End of Report*\n"
    
    # Save report
    report_path = output_dir / 'evaluation_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_md)
    
    print(f"📄 Evaluation report saved to: {report_path}")
    
    # Save metrics JSON
    metrics_path = output_dir / 'evaluation_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"💾 Metrics saved to: {metrics_path}")
    
    # Save confusion matrix
    cm_path = output_dir / 'confusion_matrix.npy'
    np.save(cm_path, cm)
    
    return report_md


def main():
    parser = argparse.ArgumentParser(description='Evaluate V7P3R AI v5.0')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data/preprocessed',
                        help='Directory with preprocessed data')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save predictions to file')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("V7P3R AI v5.0 - Model Evaluation")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Create model
    model_config = checkpoint['config']['model']
    model = V7P3R_AI_v5(**model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    
    # Load test data
    data_dir = Path(args.data_dir)
    X_test = np.load(data_dir / 'X_test.npy')
    y_test_policy = np.load(data_dir / 'y_test_policy.npy')
    y_test_value = np.load(data_dir / 'y_test_value.npy')
    
    print(f"Test set: {X_test.shape[0]:,} positions")
    
    # Create test dataloader
    test_dataset = V7P3RDataset(X_test, y_test_policy, y_test_value)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4
    )
    
    # Evaluate
    print("\n🔍 Evaluating model...")
    metrics, predictions = evaluate_model(model, test_loader, device)
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 Evaluation Results")
    print("=" * 80)
    print(f"\nPolicy Head:")
    print(f"  Accuracy: {metrics['policy_accuracy']*100:.2f}%")
    print(f"  Top-2 Accuracy: {metrics['policy_top2_accuracy']*100:.2f}%")
    print(f"  Top-3 Accuracy: {metrics['policy_top3_accuracy']*100:.2f}%")
    
    print(f"\nValue Head:")
    print(f"  MAE: {metrics['value_mae']:.4f} ({metrics['value_mae']*10000:.0f} cp)")
    print(f"  RMSE: {metrics['value_rmse']:.4f}")
    print(f"  Correlation: {metrics['value_correlation']:.4f}")
    
    # Generate report
    print("\n📝 Generating evaluation report...")
    generate_report(metrics, predictions, args.output_dir)
    
    # Save predictions if requested
    if args.save_predictions:
        pred_path = Path(args.output_dir) / 'predictions.npz'
        np.savez_compressed(pred_path, **predictions)
        print(f"💾 Predictions saved to: {pred_path}")
    
    print("\n" + "=" * 80)
    print("✅ Evaluation Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
