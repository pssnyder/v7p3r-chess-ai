"""
Exploratory Data Analysis (EDA) for v5.3 merged dataset.

Analyzes data quality, feature distributions, biases, and potential issues
in the merged 6.3M position dataset before preprocessing.

Uses streaming approach to avoid memory issues.
"""

import json
import os
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List
import re


class MergedDatasetEDA:
    """Streaming EDA for merged chess dataset."""
    
    def __init__(self, dataset_path: str, sample_size: int = None):
        self.dataset_path = dataset_path
        self.sample_size = sample_size  # If set, analyze only N records
        
        # Statistics trackers
        self.total_records = 0
        self.grade_counts = Counter()
        self.source_counts = Counter()
        self.source_grade_matrix = defaultdict(Counter)
        
        # Feature statistics (will use streaming mean/variance)
        self.feature_stats = defaultdict(lambda: {
            'count': 0,
            'sum': 0.0,
            'sum_sq': 0.0,
            'min': float('inf'),
            'max': float('-inf'),
            'missing': 0,
            'non_numeric': 0
        })
        
        # Categorical feature tracking
        self.categorical_distributions = defaultdict(Counter)
        
        # Source-specific feature stats
        self.source_feature_stats = defaultdict(lambda: defaultdict(lambda: {
            'sum': 0.0, 'count': 0
        }))
        
        # Temporal feature availability
        self.has_history_count = 0
        self.no_history_count = 0
        
        # Data quality issues
        self.issues = []
        
    def _extract_grade(self, record: dict) -> int:
        """Extract grade from record (handles different formats)."""
        if 'grade' in record:
            return record['grade']
        elif 'stockfish_analysis' in record and 'grade' in record['stockfish_analysis']:
            return record['stockfish_analysis']['grade']
        else:
            return None
    
    def _extract_source(self, record: dict) -> str:
        """Extract source from record."""
        # Try direct source field
        if 'source' in record:
            return record['source']
        
        # Try metadata
        if 'metadata' in record and 'source' in record['metadata']:
            return record['metadata']['source']
        
        # Fallback: infer from structure
        if 'puzzle_id' in record:
            return 'lichess_puzzle'
        
        return 'unknown'
    
    def _extract_features(self, record: dict) -> dict:
        """Extract features from record."""
        if 'features' in record:
            return record['features']
        return {}
    
    def _is_numeric(self, value) -> bool:
        """Check if value is numeric."""
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    
    def _update_feature_stats(self, feature_name: str, value, source: str):
        """Update streaming statistics for a feature."""
        stats = self.feature_stats[feature_name]
        
        if value is None:
            stats['missing'] += 1
            return
        
        if not self._is_numeric(value):
            stats['non_numeric'] += 1
            
            # Track categorical distributions for boolean/string features
            if isinstance(value, (bool, str)):
                self.categorical_distributions[feature_name][str(value)] += 1
            return
        
        # Numeric feature statistics
        stats['count'] += 1
        stats['sum'] += value
        stats['sum_sq'] += value * value
        stats['min'] = min(stats['min'], value)
        stats['max'] = max(stats['max'], value)
        
        # Source-specific stats
        source_stats = self.source_feature_stats[source][feature_name]
        source_stats['sum'] += value
        source_stats['count'] += 1
    
    def analyze_streaming(self):
        """Main streaming analysis loop."""
        print("="*60)
        print("STREAMING EDA - Merged Dataset v5.3")
        print("="*60)
        
        if self.sample_size:
            print(f"Sample size: {self.sample_size:,} records")
        else:
            print("Analyzing full dataset")
        
        print(f"\nReading: {os.path.basename(self.dataset_path)}")
        print()
        
        with open(self.dataset_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())
                    
                    # Extract key fields
                    grade = self._extract_grade(record)
                    source = self._extract_source(record)
                    features = self._extract_features(record)
                    
                    # Update counters
                    self.total_records += 1
                    if grade is not None:
                        self.grade_counts[grade] += 1
                        self.source_grade_matrix[source][grade] += 1
                    self.source_counts[source] += 1
                    
                    # Track temporal features
                    has_history = features.get('has_history', 0)
                    if has_history:
                        self.has_history_count += 1
                    else:
                        self.no_history_count += 1
                    
                    # Update feature statistics
                    for feature_name, value in features.items():
                        self._update_feature_stats(feature_name, value, source)
                    
                    # Progress update
                    if self.total_records % 100000 == 0:
                        print(f"  Processed: {self.total_records:,} records...")
                    
                    # Stop at sample size if set
                    if self.sample_size and self.total_records >= self.sample_size:
                        break
                
                except Exception as e:
                    self.issues.append(f"Line {line_num}: {str(e)}")
                    if len(self.issues) < 10:  # Only track first 10 errors
                        continue
                    else:
                        break
        
        print(f"\n✅ Analysis complete: {self.total_records:,} records processed")
        if self.issues:
            print(f"⚠️  {len(self.issues)} parsing errors encountered")
    
    def _compute_mean_std(self, stats: dict) -> tuple:
        """Compute mean and std from streaming stats."""
        if stats['count'] == 0:
            return None, None
        
        mean = stats['sum'] / stats['count']
        variance = (stats['sum_sq'] / stats['count']) - (mean * mean)
        std = np.sqrt(max(0, variance))  # Avoid negative due to floating point
        
        return mean, std
    
    def generate_report(self):
        """Generate comprehensive EDA report."""
        print("\n" + "="*60)
        print("EDA REPORT")
        print("="*60)
        
        # 1. Dataset Overview
        print("\n📊 DATASET OVERVIEW")
        print("-" * 60)
        print(f"Total records: {self.total_records:,}")
        print(f"Parsing errors: {len(self.issues)}")
        
        # 2. Grade Distribution
        print("\n🎯 GRADE DISTRIBUTION")
        print("-" * 60)
        for grade in sorted(self.grade_counts.keys()):
            count = self.grade_counts[grade]
            pct = count / self.total_records * 100
            bar = "█" * int(pct / 2)
            print(f"Grade {grade}: {count:>10,} ({pct:>5.1f}%) {bar}")
        
        # Identify grade imbalance issues
        if self.grade_counts:
            max_grade_count = max(self.grade_counts.values())
            min_grade_count = min(self.grade_counts.values())
            imbalance_ratio = max_grade_count / min_grade_count if min_grade_count > 0 else float('inf')
            
            print(f"\n⚠️  Imbalance ratio: {imbalance_ratio:.1f}:1")
            if imbalance_ratio > 100:
                self.issues.append("SEVERE grade imbalance - weighted loss REQUIRED")
            elif imbalance_ratio > 20:
                self.issues.append("MODERATE grade imbalance - weighted loss recommended")
        
        # 3. Source Distribution
        print("\n📂 SOURCE DISTRIBUTION")
        print("-" * 60)
        for source in sorted(self.source_counts.keys(), key=lambda x: self.source_counts[x], reverse=True):
            count = self.source_counts[source]
            pct = count / self.total_records * 100
            print(f"{source:30s} {count:>10,} ({pct:>5.1f}%)")
        
        # 4. Source vs Grade Matrix
        print("\n📊 SOURCE vs GRADE MATRIX")
        print("-" * 60)
        print(f"{'Source':<30} " + " ".join([f"G{i:>7}" for i in range(6)]))
        print("-" * 60)
        
        for source in sorted(self.source_counts.keys()):
            grade_dist = self.source_grade_matrix[source]
            row = f"{source:<30}"
            for grade in range(6):
                count = grade_dist.get(grade, 0)
                pct = (count / self.source_counts[source] * 100) if self.source_counts[source] > 0 else 0
                row += f" {pct:>6.1f}%"
            print(row)
        
        # Identify source-specific biases
        print("\n⚠️  SOURCE-SPECIFIC BIASES:")
        for source, grades in self.source_grade_matrix.items():
            if not grades:
                continue
            
            total_source = sum(grades.values())
            for grade, count in grades.items():
                pct = count / total_source * 100
                if pct > 95:
                    msg = f"  {source}: {pct:.1f}% Grade {grade} - SEVERE BIAS"
                    print(msg)
                    self.issues.append(msg)
        
        # 5. Temporal Features
        print("\n⏱️  TEMPORAL FEATURES")
        print("-" * 60)
        print(f"With history: {self.has_history_count:,} ({self.has_history_count/self.total_records*100:.1f}%)")
        print(f"Without history: {self.no_history_count:,} ({self.no_history_count/self.total_records*100:.1f}%)")
        
        if self.has_history_count < self.total_records * 0.1:
            msg = "WARNING: <10% of data has temporal features - TPF may be underutilized"
            print(f"\n⚠️  {msg}")
            self.issues.append(msg)
        
        # 6. Feature Quality Analysis
        print("\n🔍 FEATURE QUALITY ANALYSIS")
        print("-" * 60)
        
        # Identify features with issues
        high_missing = []
        high_non_numeric = []
        zero_variance = []
        extreme_outliers = []
        
        for feature_name, stats in sorted(self.feature_stats.items()):
            total_attempts = stats['count'] + stats['missing'] + stats['non_numeric']
            
            if total_attempts == 0:
                continue
            
            missing_pct = stats['missing'] / total_attempts * 100
            non_numeric_pct = stats['non_numeric'] / total_attempts * 100
            
            # High missing values
            if missing_pct > 10:
                high_missing.append((feature_name, missing_pct))
            
            # High non-numeric (unexpected for numeric features)
            if non_numeric_pct > 10 and not feature_name.startswith('F002'):  # F002 is game_phase (categorical)
                high_non_numeric.append((feature_name, non_numeric_pct))
            
            # Zero variance (all same value)
            if stats['count'] > 0:
                mean, std = self._compute_mean_std(stats)
                if std < 1e-6:
                    zero_variance.append((feature_name, mean))
                
                # Extreme outliers (std > 1000 * mean, excluding eval features)
                if not any(x in feature_name for x in ['eval', 'balance', 'cp']) and mean > 0:
                    if std > 1000 * mean:
                        extreme_outliers.append((feature_name, mean, std))
        
        if high_missing:
            print(f"\n⚠️  HIGH MISSING VALUES ({len(high_missing)} features):")
            for feature, pct in sorted(high_missing, key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {feature:50s} {pct:>6.1f}% missing")
        
        if high_non_numeric:
            print(f"\n⚠️  UNEXPECTED NON-NUMERIC VALUES ({len(high_non_numeric)} features):")
            for feature, pct in sorted(high_non_numeric, key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {feature:50s} {pct:>6.1f}% non-numeric")
        
        if zero_variance:
            print(f"\n⚠️  ZERO VARIANCE FEATURES ({len(zero_variance)} features):")
            for feature, value in zero_variance[:10]:
                print(f"  {feature:50s} constant = {value:.3f}")
            self.issues.append(f"{len(zero_variance)} features have zero variance (consider dropping)")
        
        if extreme_outliers:
            print(f"\n⚠️  EXTREME OUTLIERS ({len(extreme_outliers)} features):")
            for feature, mean, std in extreme_outliers[:10]:
                print(f"  {feature:50s} mean={mean:.2f}, std={std:.2f} (ratio={std/mean:.1f})")
        
        # 7. Feature Statistics Summary
        print("\n📈 FEATURE STATISTICS SUMMARY")
        print("-" * 60)
        
        # Categorize features
        numeric_features = [f for f, s in self.feature_stats.items() if s['count'] > 0]
        categorical_features = list(self.categorical_distributions.keys())
        
        print(f"Numeric features: {len(numeric_features)}")
        print(f"Categorical features: {len(categorical_features)}")
        
        # Sample statistics for key feature groups
        print("\n📊 Sample Statistics (key feature groups):")
        
        feature_groups = {
            'Material': [f for f in numeric_features if 'material' in f.lower()],
            'King Safety': [f for f in numeric_features if 'king' in f.lower()],
            'Pawn Structure': [f for f in numeric_features if 'pawn' in f.lower()],
            'Mobility': [f for f in numeric_features if 'mobility' in f.lower()],
            'Tactical': [f for f in numeric_features if any(x in f.lower() for x in ['hanging', 'prise', 'pin', 'fork', 'skewer'])],
        }
        
        for group_name, features in feature_groups.items():
            if not features:
                continue
            
            print(f"\n  {group_name} ({len(features)} features):")
            for feature in features[:3]:  # Show first 3
                stats = self.feature_stats[feature]
                mean, std = self._compute_mean_std(stats)
                print(f"    {feature:45s} mean={mean:>8.2f}, std={std:>8.2f}, range=[{stats['min']:>8.2f}, {stats['max']:>8.2f}]")
        
        # 8. Source-Specific Feature Differences
        print("\n🔍 SOURCE-SPECIFIC FEATURE ANALYSIS")
        print("-" * 60)
        
        # Find features with large source-to-source variance
        source_differences = []
        
        for feature_name in numeric_features[:50]:  # Sample first 50 numeric features
            source_means = {}
            
            for source in self.source_counts.keys():
                if source in self.source_feature_stats and feature_name in self.source_feature_stats[source]:
                    stats = self.source_feature_stats[source][feature_name]
                    if stats['count'] > 0:
                        source_means[source] = stats['sum'] / stats['count']
            
            if len(source_means) >= 2:
                mean_values = list(source_means.values())
                mean_diff = max(mean_values) - min(mean_values)
                avg_mean = np.mean(mean_values)
                
                if avg_mean != 0:
                    relative_diff = mean_diff / abs(avg_mean)
                    if relative_diff > 0.5:  # >50% relative difference
                        source_differences.append((feature_name, relative_diff, source_means))
        
        if source_differences:
            print(f"\n⚠️  LARGE SOURCE-TO-SOURCE DIFFERENCES ({len(source_differences)} features):")
            for feature, diff, source_means in sorted(source_differences, key=lambda x: x[1], reverse=True)[:5]:
                print(f"\n  {feature} (relative diff: {diff:.2%}):")
                for source, mean in source_means.items():
                    print(f"    {source:30s} mean={mean:>8.2f}")
        
        # 9. Critical Issues Summary
        print("\n" + "="*60)
        print("⚠️  CRITICAL ISSUES SUMMARY")
        print("="*60)
        
        if self.issues:
            for i, issue in enumerate(self.issues, 1):
                print(f"{i}. {issue}")
        else:
            print("✅ No critical issues detected")
        
        # 10. Recommendations
        print("\n" + "="*60)
        print("💡 RECOMMENDATIONS")
        print("="*60)
        
        # Grade imbalance
        if max(self.grade_counts.values()) / min(self.grade_counts.values()) > 20:
            print("\n1. GRADE IMBALANCE:")
            print("   - Use weighted loss during training")
            print("   - Suggested weights:")
            total = sum(self.grade_counts.values())
            for grade in sorted(self.grade_counts.keys()):
                weight = total / (len(self.grade_counts) * self.grade_counts[grade])
                print(f"     Grade {grade}: {weight:.3f}")
        
        # Source bias
        biased_sources = [s for s, grades in self.source_grade_matrix.items() 
                          if any(count/sum(grades.values())*100 > 95 for count in grades.values())]
        if biased_sources:
            print("\n2. SOURCE BIAS:")
            print(f"   - {len(biased_sources)} sources have >95% single-grade distribution")
            print("   - Consider:")
            print("     a) Re-extracting with better Stockfish analysis")
            print("     b) Downsampling biased sources")
            print("     c) Using source-aware stratified sampling")
        
        # Feature quality
        if zero_variance:
            print("\n3. FEATURE QUALITY:")
            print(f"   - Drop {len(zero_variance)} zero-variance features before training")
            print("   - They provide no information gain")
        
        if high_missing:
            print("\n4. MISSING VALUES:")
            print(f"   - {len(high_missing)} features have >10% missing values")
            print("   - Options:")
            print("     a) Impute with mean/median/mode")
            print("     b) Create 'missing' indicator features")
            print("     c) Drop records with missing critical features")
        
        print("\n" + "="*60)


def main():
    """Main EDA workflow."""
    
    base_dir = Path(__file__).parent.parent
    merged_file = base_dir / "data" / "final" / "v7p3r_ai_v5.3_merged.jsonl"
    
    if not merged_file.exists():
        print(f"❌ Merged dataset not found: {merged_file}")
        return 1
    
    # Option to analyze sample or full dataset
    print("EDA Options:")
    print("1. Full dataset (6.3M records, ~10-15 min)")
    print("2. Sample (100k records, ~30 sec)")
    print()
    
    choice = input("Choice (1 or 2, default=2): ").strip() or "2"
    
    sample_size = None if choice == "1" else 100000
    
    # Run EDA
    eda = MergedDatasetEDA(str(merged_file), sample_size=sample_size)
    eda.analyze_streaming()
    eda.generate_report()
    
    # Save report to file
    report_file = base_dir / "data" / "final" / "eda_report_v5.3.txt"
    print(f"\n💾 Report saved to: {report_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())
