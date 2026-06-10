"""
Merge all v5.3 datasets into single training file.

Data Sources:
1. Lichess puzzles: 5.6M positions, all grade 0 (puzzle solutions)
2. C0BR4 games: 492k positions, mixed grades (game analysis)
3. V7P3R games: 231k positions, mixed grades (game analysis)
4. Multi-engine puzzles: 4k positions, mixed grades (engine character)

Output: Merged JSONL with ~6.35M positions
Strategy: Include all data, analyze distribution, optional rebalancing
"""

import json
import os
from pathlib import Path
from collections import Counter
from typing import Dict, List
import random


class DatasetMerger:
    """Merge multiple JSONL datasets with statistics and optional balancing."""
    
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.sources = []
        self.total_positions = 0
        self.grade_distribution = Counter()
        
    def add_source(self, path: str, name: str, expected_count: int = None):
        """Register a data source for merging."""
        if not os.path.exists(path):
            print(f"⚠️  Source not found: {name} ({path})")
            return False
        
        self.sources.append({
            'path': path,
            'name': name,
            'expected': expected_count
        })
        print(f"✓ Registered: {name}")
        return True
    
    def _read_jsonl(self, path: str) -> tuple:
        """Read JSONL file and return (records, grade_counts)."""
        records = []
        grade_counts = Counter()
        
        print(f"  Reading {path}...")
        with open(path, 'r') as f:
            for line in f:
                record = json.loads(line.strip())
                records.append(record)
                
                # Handle different grade locations
                if 'grade' in record:
                    grade = record['grade']
                elif 'stockfish_analysis' in record and 'grade' in record['stockfish_analysis']:
                    grade = record['stockfish_analysis']['grade']
                else:
                    grade = 5  # Default to worst grade if not found
                
                grade_counts[grade] += 1
        
        return records, grade_counts
    
    def merge_all(self, shuffle: bool = True):
        """Merge all registered sources into output file."""
        print("\n" + "="*60)
        print("MERGING DATASETS")
        print("="*60)
        
        all_records = []
        source_stats = []
        
        # Load all sources
        for source in self.sources:
            print(f"\n📂 {source['name']}")
            records, grade_counts = self._read_jsonl(source['path'])
            
            count = len(records)
            all_records.extend(records)
            self.grade_distribution.update(grade_counts)
            
            # Track stats
            source_stats.append({
                'name': source['name'],
                'count': count,
                'expected': source['expected'],
                'grades': dict(grade_counts)
            })
            
            print(f"  Positions: {count:,}")
            if source['expected']:
                diff = count - source['expected']
                status = "✓" if abs(diff) < 100 else "⚠️"
                print(f"  Expected: {source['expected']:,} {status}")
            
            # Show grade distribution
            print(f"  Grades: ", end="")
            for grade in sorted(grade_counts.keys()):
                pct = grade_counts[grade] / count * 100
                print(f"G{grade}={pct:.1f}% ", end="")
            print()
        
        self.total_positions = len(all_records)
        
        # Shuffle for better training distribution
        if shuffle:
            print(f"\n🔀 Shuffling {self.total_positions:,} positions...")
            random.shuffle(all_records)
        
        # Write merged dataset
        print(f"\n💾 Writing to {self.output_path}...")
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        with open(self.output_path, 'w') as f:
            for record in all_records:
                f.write(json.dumps(record) + '\n')
        
        # Print summary
        self._print_summary(source_stats)
        
        return source_stats
    
    def _print_summary(self, source_stats: List[Dict]):
        """Print detailed merge summary."""
        print("\n" + "="*60)
        print("MERGE SUMMARY")
        print("="*60)
        
        # Source breakdown
        print("\n📊 Source Breakdown:")
        for stat in source_stats:
            pct = stat['count'] / self.total_positions * 100
            print(f"  {stat['name']:30s} {stat['count']:>10,} ({pct:>5.1f}%)")
        
        print(f"  {'TOTAL':30s} {self.total_positions:>10,}")
        
        # Grade distribution
        print("\n🎯 Grade Distribution:")
        for grade in sorted(self.grade_distribution.keys()):
            count = self.grade_distribution[grade]
            pct = count / self.total_positions * 100
            bar = "█" * int(pct / 2)
            print(f"  Grade {grade}: {count:>10,} ({pct:>5.1f}%) {bar}")
        
        # Data quality metrics
        print("\n📈 Data Quality:")
        
        # Samples per parameter (v5.1 architecture = 239k params)
        samples_per_param = self.total_positions / 239000
        print(f"  Samples/param: {samples_per_param:.1f} (v5.1 architecture)")
        
        if samples_per_param >= 10:
            quality = "✓ EXCELLENT (industry standard)"
        elif samples_per_param >= 5:
            quality = "✓ GOOD"
        elif samples_per_param >= 1:
            quality = "⚠️  ADEQUATE"
        else:
            quality = "❌ INSUFFICIENT"
        print(f"  Quality: {quality}")
        
        # Grade balance analysis
        grade_0_pct = self.grade_distribution[0] / self.total_positions * 100
        if grade_0_pct > 80:
            print(f"\n⚠️  WARNING: Grade 0 dominance ({grade_0_pct:.1f}%)")
            print("  Consider oversampling rare grades (1,3,4) during training")
            print("  Or use weighted loss to balance learning")
        
        print(f"\n✅ Output: {self.output_path}")
        file_size_mb = os.path.getsize(self.output_path) / (1024**2)
        print(f"   Size: {file_size_mb:,.1f} MB")
    
    def analyze_balance(self):
        """Analyze grade balance and suggest resampling strategy."""
        print("\n" + "="*60)
        print("BALANCE ANALYSIS")
        print("="*60)
        
        total = self.total_positions
        
        # Calculate imbalance ratio
        max_count = max(self.grade_distribution.values())
        min_count = min(self.grade_distribution.values())
        imbalance = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"\nImbalance ratio: {imbalance:.1f}:1")
        
        if imbalance > 100:
            print("  Status: ❌ SEVERE imbalance")
            print("  Action: REQUIRED - Use weighted loss or oversample")
        elif imbalance > 20:
            print("  Status: ⚠️  MODERATE imbalance")
            print("  Action: RECOMMENDED - Use class weights")
        else:
            print("  Status: ✓ ACCEPTABLE imbalance")
            print("  Action: Optional - May improve with balancing")
        
        # Suggest class weights
        print("\n💡 Suggested class weights:")
        weights = {}
        for grade in sorted(self.grade_distribution.keys()):
            count = self.grade_distribution[grade]
            # Inverse frequency weighting
            weight = total / (len(self.grade_distribution) * count)
            weights[grade] = weight
            print(f"  Grade {grade}: {weight:.3f} (count={count:,})")
        
        # Suggest oversampling targets
        print("\n💡 Oversampling strategy:")
        target_per_grade = max_count // 2  # Target half of max
        for grade in sorted(self.grade_distribution.keys()):
            count = self.grade_distribution[grade]
            if count < target_per_grade:
                oversample_factor = target_per_grade / count
                print(f"  Grade {grade}: Oversample {oversample_factor:.1f}x (from {count:,} to {target_per_grade:,})")
        
        return weights


def main():
    """Main merge workflow."""
    
    print("="*60)
    print("V7P3R AI v5.3 - Dataset Merge")
    print("="*60)
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    output_file = base_dir / "data" / "final" / "v7p3r_ai_v5.3_merged.jsonl"
    
    # Initialize merger
    merger = DatasetMerger(str(output_file))
    
    # Register all data sources
    print("\n📋 Registering data sources...")
    
    # 1. Lichess puzzles (5.6M positions, all grade 0)
    lichess_path = base_dir / "data" / "puzzles" / "lichess_puzzles_v5.3_full.jsonl"
    merger.add_source(
        str(lichess_path),
        "Lichess Puzzles",
        expected_count=5_622_293
    )
    
    # 2. C0BR4 games (492k positions, mixed grades)
    cobra_path = base_dir / "data" / "games" / "c0br4_games_v5.3.jsonl"
    merger.add_source(
        str(cobra_path),
        "C0BR4 Games",
        expected_count=492_654
    )
    
    # 3. V7P3R games (194k positions, mixed grades)
    v7p3r_path = base_dir / "data" / "games" / "v7p3r_games_v5.3.jsonl"
    merger.add_source(
        str(v7p3r_path),
        "V7P3R Games",
        expected_count=194_517
    )
    
    # 4. Multi-engine puzzles (4k positions, mixed grades)
    multi_engine_files = [
        base_dir / "data" / "multi_engine_puzzles" / f"v{version}_puzzle_results_final.jsonl"
        for version in ["18.5", "18.3", "18.0", "17.1.1"]
    ]
    
    # Combine multi-engine files into one registration
    for me_file in multi_engine_files:
        if me_file.exists():
            version = me_file.stem.split('_')[0]
            merger.add_source(
                str(me_file),
                f"Multi-Engine {version}",
                expected_count=1000
            )
    
    # Execute merge
    print("\n" + "="*60)
    input("Press Enter to start merge (or Ctrl+C to cancel)...")
    
    stats = merger.merge_all(shuffle=True)
    
    # Analyze balance
    merger.analyze_balance()
    
    print("\n" + "="*60)
    print("✅ MERGE COMPLETE")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. Review grade distribution above")
    print(f"2. Run: python scripts/preprocess_dataset_v5.1.py")
    print(f"   (Update input path to: {output_file})")
    print(f"3. Train v5.3 with suggested class weights")
    print()


if __name__ == "__main__":
    random.seed(42)  # Reproducible shuffle
    main()
