"""
V7P3R AI v6.0 - Filtered Data Quality Analysis

Analyzes the filtered binary classification dataset to validate:
- Feature distributions
- Class balance
- Data quality
- Source diversity
- Readiness for training
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

class FilteredDataAnalyzer:
    def __init__(self, good_path: str, bad_path: str):
        self.good_path = Path(good_path)
        self.bad_path = Path(bad_path)
        
        # Statistics
        self.stats = {
            'good': {
                'count': 0,
                'sources': {},
                'grades': {},
            },
            'bad': {
                'count': 0,
                'sources': {},
                'grades': {},
            },
            'feature_stats': {},  # feature_name -> {'good': {...}, 'bad': {...}}
        }
        
    def analyze(self, sample_size: int = 100000):
        """Run comprehensive analysis."""
        print("=" * 60)
        print("FILTERED DATA QUALITY ANALYSIS - V7P3R AI v6.0")
        print("=" * 60)
        
        # Analyze good positions
        print("\n📊 Analyzing GOOD positions...")
        self._analyze_file(self.good_path, 'good', sample_size)
        
        # Analyze bad positions
        print("\n📊 Analyzing BAD positions...")
        self._analyze_file(self.bad_path, 'bad', sample_size)
        
        # Generate report
        self.print_report()
        
    def _analyze_file(self, filepath: Path, label: str, sample_size: int):
        """Analyze a single dataset file."""
        count = 0
        sources = Counter()
        grades = Counter()
        feature_sample = []
        
        # Sample positions for feature analysis
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                
                # Count records
                count += 1
                
                # Track source
                source = record.get('source', 'unknown')
                sources[source] += 1
                
                # Track original grade
                grade = record.get('grade', -1)
                grades[grade] += 1
                
                # Sample features (first N records)
                if len(feature_sample) < sample_size:
                    features = record.get('features', {})
                    feature_sample.append(features)
                
                # Progress
                if count % 500000 == 0:
                    print(f"  Processed: {count:,} records...")
        
        print(f"✅ Analyzed {count:,} {label.upper()} positions")
        
        # Store statistics
        self.stats[label]['count'] = count
        self.stats[label]['sources'] = dict(sources)
        self.stats[label]['grades'] = dict(grades)
        
        # Analyze features
        if feature_sample:
            self._analyze_features(feature_sample, label)
    
    def _analyze_features(self, feature_sample: list, label: str):
        """Analyze feature distributions."""
        print(f"  Analyzing features from {len(feature_sample):,} samples...")
        
        # Get all feature names
        all_features = set()
        for features in feature_sample:
            all_features.update(features.keys())
        
        # Calculate statistics for each feature
        for feature_name in all_features:
            values = []
            for features in feature_sample:
                val = features.get(feature_name, 0)
                
                # Convert boolean to int
                if isinstance(val, bool):
                    val = int(val)
                
                # Only include numeric values
                if isinstance(val, (int, float)):
                    values.append(val)
            
            if values:
                # Initialize feature dict if needed
                if feature_name not in self.stats['feature_stats']:
                    self.stats['feature_stats'][feature_name] = {}
                
                # Store stats for this label
                self.stats['feature_stats'][feature_name][label] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'nonzero_pct': sum(1 for v in values if v != 0) / len(values) * 100,
                }
    
    def print_report(self):
        """Print comprehensive analysis report."""
        print("\n" + "=" * 60)
        print("DATA QUALITY REPORT")
        print("=" * 60)
        
        # Class balance
        print("\n📊 CLASS BALANCE")
        print("-" * 60)
        good_count = self.stats['good']['count']
        bad_count = self.stats['bad']['count']
        total = good_count + bad_count
        imbalance = good_count / bad_count if bad_count > 0 else 0
        
        print(f"Good positions: {good_count:,} ({good_count/total*100:.2f}%)")
        print(f"Bad positions:  {bad_count:,} ({bad_count/total*100:.2f}%)")
        print(f"Total:          {total:,}")
        print(f"Imbalance ratio: {imbalance:.1f}:1 (good:bad)")
        
        # Source distribution
        print("\n📂 SOURCE DISTRIBUTION")
        print("-" * 60)
        
        print("\nGood positions by source:")
        for source, count in sorted(self.stats['good']['sources'].items()):
            pct = count / good_count * 100
            print(f"  {source:20s}: {count:,} ({pct:.2f}%)")
        
        print("\nBad positions by source:")
        for source, count in sorted(self.stats['bad']['sources'].items()):
            pct = count / bad_count * 100
            print(f"  {source:20s}: {count:,} ({pct:.2f}%)")
        
        # Grade distribution
        print("\n🎯 ORIGINAL GRADE DISTRIBUTION")
        print("-" * 60)
        
        print("\nGood positions by original grade:")
        for grade, count in sorted(self.stats['good']['grades'].items()):
            pct = count / good_count * 100
            print(f"  Grade {grade}: {count:,} ({pct:.2f}%)")
        
        print("\nBad positions by original grade:")
        for grade, count in sorted(self.stats['bad']['grades'].items()):
            pct = count / bad_count * 100
            print(f"  Grade {grade}: {count:,} ({pct:.2f}%)")
        
        # Feature analysis
        print("\n📈 FEATURE QUALITY ANALYSIS")
        print("-" * 60)
        
        # Find features with high variance between classes
        discriminative_features = []
        
        for feature_name, stats in self.stats['feature_stats'].items():
            if 'good' in stats and 'bad' in stats:
                # Debug: check types
                if not isinstance(stats['good'], dict):
                    print(f"ERROR: Feature '{feature_name}' has stats['good'] = {type(stats['good'])}: {stats['good']}")
                    continue
                if not isinstance(stats['bad'], dict):
                    print(f"ERROR: Feature '{feature_name}' has stats['bad'] = {type(stats['bad'])}: {stats['bad']}")
                    continue
                    
                good_mean = stats['good']['mean']
                bad_mean = stats['bad']['mean']
                
                # Calculate absolute difference
                diff = abs(good_mean - bad_mean)
                
                # Only consider features with meaningful difference
                if diff > 0.01:  # Threshold for significance
                    discriminative_features.append((feature_name, diff, good_mean, bad_mean))
        
        # Sort by discriminative power
        discriminative_features.sort(key=lambda x: x[1], reverse=True)
        
        print("\nTop 20 Most Discriminative Features:")
        print(f"{'Feature':<50s} {'Diff':>8s} {'Good':>8s} {'Bad':>8s}")
        print("-" * 76)
        
        for feature_name, diff, good_mean, bad_mean in discriminative_features[:20]:
            # Truncate long feature names
            display_name = feature_name if len(feature_name) <= 47 else feature_name[:44] + "..."
            print(f"{display_name:<50s} {diff:>8.3f} {good_mean:>8.3f} {bad_mean:>8.3f}")
        
        # Zero-variance features
        zero_variance_features = []
        
        for feature_name, stats in self.stats['feature_stats'].items():
            if 'good' in stats:
                if stats['good']['std'] == 0 and stats['good']['nonzero_pct'] == 0:
                    zero_variance_features.append(feature_name)
        
        if zero_variance_features:
            print(f"\n⚠️  Zero-Variance Features (should be dropped): {len(zero_variance_features)}")
            for i, feature_name in enumerate(zero_variance_features[:10]):
                print(f"  {i+1}. {feature_name}")
            if len(zero_variance_features) > 10:
                print(f"  ... and {len(zero_variance_features) - 10} more")
        
        # Sparse features
        sparse_features = []
        
        for feature_name, stats in self.stats['feature_stats'].items():
            if 'good' in stats:
                if stats['good']['nonzero_pct'] < 1.0:  # Less than 1% non-zero
                    sparse_features.append((feature_name, stats['good']['nonzero_pct']))
        
        sparse_features.sort(key=lambda x: x[1])
        
        if sparse_features:
            print(f"\n⚠️  Very Sparse Features (<1% non-zero): {len(sparse_features)}")
            for i, (feature_name, pct) in enumerate(sparse_features[:10]):
                print(f"  {i+1}. {feature_name:<45s} ({pct:.3f}% non-zero)")
            if len(sparse_features) > 10:
                print(f"  ... and {len(sparse_features) - 10} more")
        
        # Training readiness
        print("\n✅ TRAINING READINESS")
        print("-" * 60)
        
        total_features = len(self.stats['feature_stats'])
        usable_features = total_features - len(zero_variance_features)
        
        print(f"Total features:           {total_features}")
        print(f"Zero-variance (drop):     {len(zero_variance_features)}")
        print(f"Usable features:          {usable_features}")
        print(f"Highly discriminative:    {len([f for f in discriminative_features if f[1] > 0.1])}")
        print(f"Class imbalance ratio:    {imbalance:.1f}:1")
        print(f"Recommended class weights: good={1/imbalance:.4f}, bad=1.0")
        
        # Quality checks
        print("\n🔍 QUALITY CHECKS")
        print("-" * 60)
        
        checks = []
        
        # Check 1: Sufficient data
        if good_count > 1000000 and bad_count > 10000:
            checks.append(("✅", "Sufficient data for training"))
        else:
            checks.append(("⚠️ ", f"Limited data: {good_count:,} good, {bad_count:,} bad"))
        
        # Check 2: Source diversity
        good_sources = len(self.stats['good']['sources'])
        if good_sources >= 2:
            checks.append(("✅", f"Good source diversity ({good_sources} sources)"))
        else:
            checks.append(("⚠️ ", f"Limited source diversity ({good_sources} source)"))
        
        # Check 3: Feature quality
        if usable_features >= 300:
            checks.append(("✅", f"Rich feature set ({usable_features} features)"))
        else:
            checks.append(("⚠️ ", f"Limited features ({usable_features} features)"))
        
        # Check 4: Discriminative power
        highly_discriminative = len([f for f in discriminative_features if f[1] > 0.1])
        if highly_discriminative >= 20:
            checks.append(("✅", f"Strong discriminative features ({highly_discriminative} features)"))
        else:
            checks.append(("⚠️ ", f"Weak discriminative power ({highly_discriminative} features)"))
        
        # Check 5: Class balance
        if 20 <= imbalance <= 200:
            checks.append(("✅", f"Manageable class imbalance ({imbalance:.1f}:1)"))
        else:
            checks.append(("⚠️ ", f"Extreme class imbalance ({imbalance:.1f}:1)"))
        
        for status, message in checks:
            print(f"{status} {message}")
        
        # Overall assessment
        passed = sum(1 for status, _ in checks if status == "✅")
        
        print("\n" + "=" * 60)
        if passed >= 4:
            print("✅ DATASET READY FOR TRAINING")
            print(f"   {passed}/5 quality checks passed")
        else:
            print("⚠️  DATASET NEEDS ATTENTION")
            print(f"   Only {passed}/5 quality checks passed")
        print("=" * 60)


def main():
    # Paths
    base_path = Path(__file__).parent.parent.parent
    good_path = base_path / "data" / "stage1" / "good_positions.jsonl"
    bad_path = base_path / "data" / "stage1" / "bad_positions.jsonl"
    
    # Validate paths
    if not good_path.exists():
        print(f"❌ Error: Good positions file not found: {good_path}")
        return 1
    
    if not bad_path.exists():
        print(f"❌ Error: Bad positions file not found: {bad_path}")
        return 1
    
    # Run analysis
    analyzer = FilteredDataAnalyzer(str(good_path), str(bad_path))
    analyzer.analyze(sample_size=100000)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
