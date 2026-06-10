"""
V7P3R AI - Revised Evaluation Metrics
Focus on GOOD MOVE PREDICTION (grades 0-2) instead of balanced accuracy

Key Insight:
- Grades 3-5 are "what not to do" - they're training signal but not the goal
- Goal: Maximize % of predictions in grades 0-2 (good moves)
- Secondary: Among good predictions, prefer grade 0-1 over grade 2
"""

import torch
import numpy as np
from typing import Dict, Tuple
import json


class GoodMoveFocusedMetrics:
    """
    Evaluation metrics focused on good move prediction
    
    Primary Metrics:
    1. Good Move Rate: % predictions in grades 0-2
    2. Excellent Move Rate: % predictions in grades 0-1
    3. Bad Move Avoidance: % predictions NOT in grades 4-5
    
    Secondary Metrics:
    4. Grade 0 precision: When predicting grade 0, how often correct?
    5. Grade 1-2 recall: Of actual grades 1-2, how many found?
    """
    
    @staticmethod
    def calculate_good_move_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict:
        """
        Calculate good-move-focused metrics
        
        Args:
            predictions: Model predictions (grade 0-5)
            targets: True grades (grade 0-5)
        
        Returns:
            Dictionary of metrics
        """
        total = len(predictions)
        
        # Primary: Good Move Rate (predictions in grades 0-2)
        good_predictions = np.sum(predictions <= 2)
        good_move_rate = (good_predictions / total) * 100
        
        # Excellent Move Rate (predictions in grades 0-1)
        excellent_predictions = np.sum(predictions <= 1)
        excellent_move_rate = (excellent_predictions / total) * 100
        
        # Bad Move Avoidance (predictions NOT in grades 4-5)
        non_bad_predictions = np.sum(predictions <= 3)
        bad_avoidance_rate = (non_bad_predictions / total) * 100
        
        # Grade 0 Precision (when predicting 0, how often correct?)
        grade0_pred_mask = (predictions == 0)
        if grade0_pred_mask.sum() > 0:
            grade0_precision = np.sum((predictions == 0) & (targets == 0)) / grade0_pred_mask.sum() * 100
        else:
            grade0_precision = 0.0
        
        # Grade 0-1 Recall (of actual 0-1, how many found?)
        grade01_true_mask = (targets <= 1)
        if grade01_true_mask.sum() > 0:
            grade01_recall = np.sum((predictions <= 1) & (targets <= 1)) / grade01_true_mask.sum() * 100
        else:
            grade01_recall = 0.0
        
        # Grade 0-2 Recall (of actual good moves, how many found?)
        grade02_true_mask = (targets <= 2)
        if grade02_true_mask.sum() > 0:
            grade02_recall = np.sum((predictions <= 2) & (targets <= 2)) / grade02_true_mask.sum() * 100
        else:
            grade02_recall = 0.0
        
        # Grade-wise prediction distribution
        prediction_distribution = {}
        for grade in range(6):
            count = np.sum(predictions == grade)
            prediction_distribution[f"pred_grade_{grade}"] = int(count)
            prediction_distribution[f"pred_grade_{grade}_pct"] = (count / total) * 100
        
        # Target distribution (for comparison)
        target_distribution = {}
        for grade in range(6):
            count = np.sum(targets == grade)
            target_distribution[f"true_grade_{grade}"] = int(count)
            target_distribution[f"true_grade_{grade}_pct"] = (count / total) * 100
        
        # Confusion within good moves (grades 0-2)
        good_mask = (targets <= 2)
        if good_mask.sum() > 0:
            good_correct = np.sum((predictions <= 2) & (targets <= 2))
            good_move_accuracy = (good_correct / good_mask.sum()) * 100
        else:
            good_move_accuracy = 0.0
        
        # Expected value calculation (grade 0 = best, grade 5 = worst)
        # Lower average = better performance
        avg_predicted_grade = np.mean(predictions)
        avg_true_grade = np.mean(targets)
        grade_bias = avg_predicted_grade - avg_true_grade
        
        return {
            # PRIMARY METRICS (what we optimize for)
            "good_move_rate": good_move_rate,  # % predictions in 0-2
            "excellent_move_rate": excellent_move_rate,  # % predictions in 0-1
            "bad_avoidance_rate": bad_avoidance_rate,  # % predictions NOT in 4-5
            
            # QUALITY METRICS
            "grade0_precision": grade0_precision,
            "grade01_recall": grade01_recall,
            "grade02_recall": grade02_recall,
            "good_move_accuracy": good_move_accuracy,  # Accuracy within grades 0-2
            
            # DISTRIBUTION METRICS
            "avg_predicted_grade": avg_predicted_grade,
            "avg_true_grade": avg_true_grade,
            "grade_bias": grade_bias,  # Negative = optimistic, Positive = pessimistic
            
            # RAW COUNTS
            "total_predictions": int(total),
            "good_predictions": int(good_predictions),
            "excellent_predictions": int(excellent_predictions),
            
            # DISTRIBUTIONS
            "prediction_distribution": prediction_distribution,
            "target_distribution": target_distribution
        }
    
    @staticmethod
    def print_focused_report(metrics: Dict):
        """Print human-readable report focused on good move metrics"""
        print("\n" + "="*80)
        print("V7P3R AI - GOOD MOVE FOCUSED EVALUATION")
        print("="*80)
        
        print("\n🎯 PRIMARY METRICS (Optimization Targets)")
        print("-"*80)
        print(f"Good Move Rate (Grades 0-2):     {metrics['good_move_rate']:.2f}%")
        print(f"Excellent Move Rate (Grades 0-1): {metrics['excellent_move_rate']:.2f}%")
        print(f"Bad Move Avoidance:               {metrics['bad_avoidance_rate']:.2f}%")
        
        print("\n✨ QUALITY METRICS")
        print("-"*80)
        print(f"Grade 0 Precision:    {metrics['grade0_precision']:.2f}%")
        print(f"Grade 0-1 Recall:     {metrics['grade01_recall']:.2f}%")
        print(f"Grade 0-2 Recall:     {metrics['grade02_recall']:.2f}%")
        print(f"Good Move Accuracy:   {metrics['good_move_accuracy']:.2f}%")
        
        print("\n📊 GRADE DISTRIBUTION")
        print("-"*80)
        print("Grade | Predicted | True")
        print("------|-----------|------")
        for grade in range(6):
            pred_pct = metrics['prediction_distribution'][f'pred_grade_{grade}_pct']
            true_pct = metrics['target_distribution'][f'true_grade_{grade}_pct']
            pred_bar = '█' * int(pred_pct / 2)
            true_bar = '░' * int(true_pct / 2)
            print(f"  {grade}   | {pred_pct:5.1f}% {pred_bar:20s} | {true_pct:5.1f}% {true_bar}")
        
        print(f"\nAverage Predicted Grade: {metrics['avg_predicted_grade']:.2f}")
        print(f"Average True Grade:      {metrics['avg_true_grade']:.2f}")
        
        if metrics['grade_bias'] < 0:
            print(f"Grade Bias: {metrics['grade_bias']:.2f} (OPTIMISTIC - predicts better than reality)")
        else:
            print(f"Grade Bias: {metrics['grade_bias']:.2f} (PESSIMISTIC - predicts worse than reality)")
        
        print("\n" + "="*80)
        
        # Success criteria
        print("\n✅ SUCCESS CRITERIA")
        print("-"*80)
        
        criteria = [
            ("Good Move Rate >70%", metrics['good_move_rate'] > 70),
            ("Excellent Move Rate >40%", metrics['excellent_move_rate'] > 40),
            ("Bad Avoidance >85%", metrics['bad_avoidance_rate'] > 85),
            ("Grade 0 Precision >60%", metrics['grade0_precision'] > 60),
            ("Good Move Accuracy >80%", metrics['good_move_accuracy'] > 80),
        ]
        
        passed = sum(1 for _, status in criteria if status)
        
        for criterion, status in criteria:
            symbol = "✅" if status else "❌"
            print(f"{symbol} {criterion}")
        
        print(f"\nPassed: {passed}/{len(criteria)} criteria")
        
        if passed == len(criteria):
            print("\n🎉 MODEL READY FOR DEPLOYMENT!")
        elif passed >= 3:
            print("\n⚠️  MODEL SHOWS PROMISE - Continue training")
        else:
            print("\n🔴 MODEL NEEDS MORE TRAINING")
        
        print("="*80 + "\n")


# Example usage
if __name__ == "__main__":
    # Simulated predictions and targets
    np.random.seed(42)
    
    # Simulate a model that prefers good moves
    predictions = np.random.choice([0, 1, 2, 3, 4, 5], size=10000, p=[0.25, 0.20, 0.25, 0.15, 0.10, 0.05])
    targets = np.random.choice([0, 1, 2, 3, 4, 5], size=10000, p=[0.15, 0.10, 0.20, 0.20, 0.20, 0.15])
    
    metrics = GoodMoveFocusedMetrics.calculate_good_move_metrics(predictions, targets)
    GoodMoveFocusedMetrics.print_focused_report(metrics)
    
    # Save to file
    with open('good_move_metrics_example.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("Metrics saved to: good_move_metrics_example.json")
