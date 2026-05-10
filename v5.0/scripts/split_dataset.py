"""
V7P3R AI v5.1 - Dataset Splitting Script

Splits the full dataset into train/validation/test sets with stratification by grade.

Usage:
    python scripts/split_dataset.py \\
        --input data/final/v7p3r_ai_v5.1_expanded_features.jsonl \\
        --output-dir data/analysis/splits_v5.1 \\
        --train-ratio 0.8 \\
        --val-ratio 0.1 \\
        --test-ratio 0.1

Output:
    - {output_dir}/train.jsonl
    - {output_dir}/validation.jsonl
    - {output_dir}/test.jsonl
    - {output_dir}/split_stats.json
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
from datetime import datetime


def load_jsonl(filepath):
    """Load JSONL file into list of records"""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    return records


def save_jsonl(records, filepath):
    """Save records to JSONL file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def get_record_grade(record):
    """
    Extract move quality grade from a record.
    
    For game positions: use stockfish_analysis.move_quality_grade
    For puzzle positions: use synthetic grade based on puzzle rating
    """
    # Game position - has stockfish analysis
    if 'stockfish_analysis' in record and 'move_quality_grade' in record['stockfish_analysis']:
        return record['stockfish_analysis']['move_quality_grade']
    
    # Puzzle position - map puzzle rating to synthetic grade
    if 'puzzle_rating' in record:
        rating = record['puzzle_rating']
        # Map puzzle difficulty to grades (harder puzzles = better moves required)
        if rating < 1200:
            return 2  # Easy puzzles
        elif rating < 1600:
            return 3  # Medium puzzles
        elif rating < 2000:
            return 4  # Hard puzzles
        else:
            return 5  # Very hard puzzles
    
    # Fallback: unknown source, use neutral grade
    return 3


def stratified_split(records, train_ratio, val_ratio, test_ratio, random_seed=42):
    """
    Split records into train/val/test with stratification by move quality grade
    
    Args:
        records: List of dataset records
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        random_seed: Random seed for reproducibility
    
    Returns:
        train_records, val_records, test_records
    """
    np.random.seed(random_seed)
    
    # Group records by grade
    grade_groups = {}
    for record in records:
        grade = get_record_grade(record)
        if grade not in grade_groups:
            grade_groups[grade] = []
        grade_groups[grade].append(record)
    
    train_records = []
    val_records = []
    test_records = []
    
    # Split each grade group proportionally
    for grade, group in grade_groups.items():
        # Shuffle group
        np.random.shuffle(group)
        
        # Calculate split indices
        n = len(group)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        # Split
        train_records.extend(group[:train_end])
        val_records.extend(group[train_end:val_end])
        test_records.extend(group[val_end:])
    
    # Shuffle final sets (grades are now mixed)
    np.random.shuffle(train_records)
    np.random.shuffle(val_records)
    np.random.shuffle(test_records)
    
    return train_records, val_records, test_records


def get_grade_distribution(records):
    """Get distribution of grades in records"""
    grades = [get_record_grade(r) for r in records]
    return dict(Counter(grades))


def main():
    parser = argparse.ArgumentParser(description='Split V7P3R AI dataset into train/val/test')
    parser.add_argument('--input', required=True, help='Input JSONL file')
    parser.add_argument('--output-dir', required=True, help='Output directory for splits')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {args.train_ratio + args.val_ratio + args.test_ratio}")
    
    print("=" * 80)
    print("V7P3R AI v5.1 - Dataset Splitting")
    print("=" * 80)
    
    # Load dataset
    print(f"\n📂 Loading dataset from {args.input}...")
    records = load_jsonl(args.input)
    print(f"  Total records: {len(records):,}")
    
    # Show grade distribution
    total_dist = get_grade_distribution(records)
    print(f"\n📊 Grade distribution in full dataset:")
    for grade in sorted(total_dist.keys()):
        count = total_dist[grade]
        pct = 100 * count / len(records)
        print(f"    Grade {grade}: {count:7,} ({pct:5.2f}%)")
    
    # Split dataset
    print(f"\n✂️ Splitting with stratification...")
    print(f"  Train: {args.train_ratio*100:.1f}%")
    print(f"  Val:   {args.val_ratio*100:.1f}%")
    print(f"  Test:  {args.test_ratio*100:.1f}%")
    
    train, val, test = stratified_split(
        records,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )
    
    print(f"\n✓ Split complete:")
    print(f"  Train: {len(train):,} positions")
    print(f"  Val:   {len(val):,} positions")
    print(f"  Test:  {len(test):,} positions")
    
    # Verify stratification
    print(f"\n📊 Verifying stratification...")
    train_dist = get_grade_distribution(train)
    val_dist = get_grade_distribution(val)
    test_dist = get_grade_distribution(test)
    
    print(f"\nGrade distribution across splits:")
    print(f"  {'Grade':<8} {'Train':<12} {'Val':<12} {'Test':<12}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
    
    for grade in sorted(total_dist.keys()):
        train_pct = 100 * train_dist.get(grade, 0) / len(train)
        val_pct = 100 * val_dist.get(grade, 0) / len(val)
        test_pct = 100 * test_dist.get(grade, 0) / len(test)
        print(f"  {grade:<8} {train_pct:>6.2f}%      {val_pct:>6.2f}%      {test_pct:>6.2f}%")
    
    # Save splits
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving splits to {output_dir}/...")
    save_jsonl(train, output_dir / 'train.jsonl')
    save_jsonl(val, output_dir / 'validation.jsonl')
    save_jsonl(test, output_dir / 'test.jsonl')
    
    print(f"  ✓ train.jsonl ({len(train):,} records)")
    print(f"  ✓ validation.jsonl ({len(val):,} records)")
    print(f"  ✓ test.jsonl ({len(test):,} records)")
    
    # Save split statistics
    stats = {
        'timestamp': datetime.now().isoformat(),
        'input_file': str(args.input),
        'total_records': len(records),
        'splits': {
            'train': {
                'count': len(train),
                'ratio': args.train_ratio,
                'grade_distribution': train_dist
            },
            'validation': {
                'count': len(val),
                'ratio': args.val_ratio,
                'grade_distribution': val_dist
            },
            'test': {
                'count': len(test),
                'ratio': args.test_ratio,
                'grade_distribution': test_dist
            }
        },
        'random_seed': args.seed,
        'full_grade_distribution': total_dist
    }
    
    with open(output_dir / 'split_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f"  ✓ split_stats.json")
    
    print("\n" + "=" * 80)
    print("✅ DATASET SPLITTING COMPLETE")
    print("=" * 80)
    print(f"\nNext step: Run preprocessing on {output_dir}/")


if __name__ == '__main__':
    main()
