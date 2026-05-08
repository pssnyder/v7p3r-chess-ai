"""
Analyze the complete V7P3R AI v5.0 training dataset and generate statistics.

This script:
1. Loads the complete training dataset
2. Generates comprehensive statistics
3. Analyzes grade distribution
4. Examines game phase distribution
5. Studies feature correlations
6. Creates train/validation/test splits
7. Exports analysis report

Usage:
    python scripts/analyze_dataset.py --input data/final/v7p3r_ai_v5_training_dataset_complete.jsonl --output data/analysis
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetAnalyzer:
    """Analyze the V7P3R training dataset and generate statistics."""
    
    def __init__(self, input_file: str, output_dir: str):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics containers
        self.total_positions = 0
        self.grade_distribution = Counter()
        self.phase_distribution = Counter()
        self.source_distribution = Counter()
        self.version_distribution = Counter()
        self.move_type_distribution = {
            'capture': 0,
            'check': 0,
            'castling': 0,
            'en_passant': 0,
            'promotion': 0,
            'quiet': 0
        }
        self.eval_statistics = {
            'best_move_evals': [],
            'eval_drops': [],
            'mate_positions': 0
        }
        self.feature_stats = defaultdict(lambda: {'count': 0, 'sum': 0, 'values': []})
        
        # Grade-specific statistics
        self.grade_by_phase = defaultdict(Counter)
        self.grade_by_source = defaultdict(Counter)
        
    def analyze(self):
        """Run complete dataset analysis."""
        logger.info(f"Loading dataset from {self.input_file}")
        
        with open(self.input_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line_num % 10000 == 0:
                    logger.info(f"Processed {line_num:,} positions...")
                
                try:
                    record = json.loads(line)
                    self._analyze_record(record)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error at line {line_num}: {e}")
                    continue
        
        logger.info(f"Analysis complete! Processed {self.total_positions:,} positions")
        
    def _analyze_record(self, record: Dict[str, Any]):
        """Analyze a single training record."""
        self.total_positions += 1
        
        # Metadata analysis
        metadata = record.get('metadata', {})
        self.source_distribution[metadata.get('source', 'unknown')] += 1
        self.version_distribution[metadata.get('v7p3r_version', 'unknown')] += 1
        
        # Position analysis
        position = record.get('position', {})
        phase = position.get('game_phase', 'unknown')
        self.phase_distribution[phase] += 1
        
        # Engine decision analysis
        decision = record.get('engine_decision', {})
        if decision.get('is_capture'):
            self.move_type_distribution['capture'] += 1
        if decision.get('is_check'):
            self.move_type_distribution['check'] += 1
        if decision.get('is_castling'):
            self.move_type_distribution['castling'] += 1
        if decision.get('is_en_passant'):
            self.move_type_distribution['en_passant'] += 1
        if decision.get('promotion'):
            self.move_type_distribution['promotion'] += 1
        if not any([decision.get('is_capture'), decision.get('is_check'), 
                   decision.get('is_castling'), decision.get('is_en_passant')]):
            self.move_type_distribution['quiet'] += 1
        
        # Stockfish analysis
        sf_analysis = record.get('stockfish_analysis', {})
        grade = sf_analysis.get('move_quality_grade')
        if grade is not None:
            self.grade_distribution[grade] += 1
            self.grade_by_phase[phase][grade] += 1
            self.grade_by_source[metadata.get('source', 'unknown')][grade] += 1
        
        # Evaluation statistics
        best_eval = sf_analysis.get('best_move_eval_cp')
        if best_eval is not None:
            self.eval_statistics['best_move_evals'].append(best_eval)
        
        if sf_analysis.get('best_move_eval_mate') is not None:
            self.eval_statistics['mate_positions'] += 1
        
        eval_drop = sf_analysis.get('eval_drop_cp')
        if eval_drop is not None:
            self.eval_statistics['eval_drops'].append(eval_drop)
        
        # Feature analysis (sample numeric features)
        features = record.get('features', {})
        for key, value in features.items():
            if isinstance(value, (int, float)):
                self.feature_stats[key]['count'] += 1
                self.feature_stats[key]['sum'] += value
                if len(self.feature_stats[key]['values']) < 1000:  # Sample first 1000
                    self.feature_stats[key]['values'].append(value)
    
    def generate_report(self):
        """Generate comprehensive analysis report."""
        logger.info("Generating analysis report...")
        
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'dataset_file': str(self.input_file),
            'total_positions': self.total_positions,
            
            'source_distribution': dict(self.source_distribution),
            'version_distribution': dict(self.version_distribution),
            'phase_distribution': dict(self.phase_distribution),
            
            'grade_distribution': {
                'overall': dict(self.grade_distribution),
                'by_phase': {k: dict(v) for k, v in self.grade_by_phase.items()},
                'by_source': {k: dict(v) for k, v in self.grade_by_source.items()}
            },
            
            'move_type_distribution': self.move_type_distribution,
            
            'evaluation_statistics': {
                'mate_positions': self.eval_statistics['mate_positions'],
                'best_move_eval_mean': self._safe_mean(self.eval_statistics['best_move_evals']),
                'best_move_eval_median': self._safe_median(self.eval_statistics['best_move_evals']),
                'eval_drop_mean': self._safe_mean(self.eval_statistics['eval_drops']),
                'eval_drop_median': self._safe_median(self.eval_statistics['eval_drops']),
                'eval_drop_max': max(self.eval_statistics['eval_drops']) if self.eval_statistics['eval_drops'] else 0
            },
            
            'feature_statistics': {
                key: {
                    'count': stats['count'],
                    'mean': stats['sum'] / stats['count'] if stats['count'] > 0 else 0,
                    'sample_min': min(stats['values']) if stats['values'] else None,
                    'sample_max': max(stats['values']) if stats['values'] else None
                }
                for key, stats in self.feature_stats.items()
            }
        }
        
        # Save JSON report
        report_file = self.output_dir / 'dataset_analysis.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved JSON report to {report_file}")
        
        # Generate human-readable markdown report
        self._generate_markdown_report(report)
        
        return report
    
    def _generate_markdown_report(self, report: Dict[str, Any]):
        """Generate human-readable markdown analysis report."""
        md_file = self.output_dir / 'dataset_analysis.md'
        
        with open(md_file, 'w') as f:
            f.write("# V7P3R AI v5.0 - Dataset Analysis Report\n\n")
            f.write(f"**Analysis Date**: {report['analysis_timestamp']}\n")
            f.write(f"**Dataset**: {report['dataset_file']}\n")
            f.write(f"**Total Positions**: {report['total_positions']:,}\n\n")
            
            f.write("---\n\n## Source Distribution\n\n")
            f.write("| Source | Positions | Percentage |\n")
            f.write("|--------|-----------|------------|\n")
            for source, count in sorted(report['source_distribution'].items(), key=lambda x: -x[1]):
                pct = (count / report['total_positions']) * 100
                f.write(f"| {source} | {count:,} | {pct:.2f}% |\n")
            
            f.write("\n## Version Distribution\n\n")
            f.write("| Version | Positions | Percentage |\n")
            f.write("|---------|-----------|------------|\n")
            for version, count in sorted(report['version_distribution'].items()):
                pct = (count / report['total_positions']) * 100
                f.write(f"| {version} | {count:,} | {pct:.2f}% |\n")
            
            f.write("\n## Game Phase Distribution\n\n")
            f.write("| Phase | Positions | Percentage |\n")
            f.write("|-------|-----------|------------|\n")
            phase_order = ['opening', 'middlegame', 'endgame', 'unknown']
            for phase in phase_order:
                count = report['phase_distribution'].get(phase, 0)
                if count > 0:
                    pct = (count / report['total_positions']) * 100
                    f.write(f"| {phase} | {count:,} | {pct:.2f}% |\n")
            
            f.write("\n## Move Quality Grade Distribution\n\n")
            f.write("| Grade | Description | Positions | Percentage |\n")
            f.write("|-------|-------------|-----------|------------|\n")
            grade_desc = {
                5: "Best move",
                4: "2nd best",
                3: "3rd best",
                2: "4th best",
                1: "5th best",
                0: "Not in top-5"
            }
            for grade in sorted(report['grade_distribution']['overall'].keys(), reverse=True):
                count = report['grade_distribution']['overall'][grade]
                pct = (count / report['total_positions']) * 100
                desc = grade_desc.get(grade, f"Grade {grade}")
                f.write(f"| {grade} | {desc} | {count:,} | {pct:.2f}% |\n")
            
            f.write("\n## Move Type Distribution\n\n")
            f.write("| Move Type | Positions | Percentage |\n")
            f.write("|-----------|-----------|------------|\n")
            for move_type, count in sorted(report['move_type_distribution'].items(), key=lambda x: -x[1]):
                pct = (count / report['total_positions']) * 100
                f.write(f"| {move_type} | {count:,} | {pct:.2f}% |\n")
            
            f.write("\n## Evaluation Statistics\n\n")
            eval_stats = report['evaluation_statistics']
            f.write(f"- **Mate Positions**: {eval_stats['mate_positions']:,}\n")
            f.write(f"- **Best Move Eval (mean)**: {eval_stats['best_move_eval_mean']:.2f} cp\n")
            f.write(f"- **Best Move Eval (median)**: {eval_stats['best_move_eval_median']:.2f} cp\n")
            f.write(f"- **Eval Drop (mean)**: {eval_stats['eval_drop_mean']:.2f} cp\n")
            f.write(f"- **Eval Drop (median)**: {eval_stats['eval_drop_median']:.2f} cp\n")
            f.write(f"- **Max Eval Drop**: {eval_stats['eval_drop_max']:.2f} cp\n")
            
            f.write("\n## Feature Statistics (Top 10 by Average Value)\n\n")
            f.write("| Feature | Count | Mean | Sample Min | Sample Max |\n")
            f.write("|---------|-------|------|------------|------------|\n")
            
            # Sort features by mean value (numeric features only)
            sorted_features = sorted(
                [(k, v) for k, v in report['feature_statistics'].items() if v['sample_min'] is not None],
                key=lambda x: abs(x[1]['mean']),
                reverse=True
            )[:10]
            
            for feature, stats in sorted_features:
                f.write(f"| {feature} | {stats['count']:,} | {stats['mean']:.2f} | ")
                f.write(f"{stats['sample_min']} | {stats['sample_max']} |\n")
            
            f.write("\n---\n\n")
            f.write("*Report generated by `scripts/analyze_dataset.py`*\n")
        
        logger.info(f"Saved markdown report to {md_file}")
    
    def create_splits(self, train_ratio: float = 0.8, val_ratio: float = 0.1):
        """Create stratified train/validation/test splits."""
        logger.info(f"Creating dataset splits ({train_ratio:.0%}/{val_ratio:.0%}/{1-train_ratio-val_ratio:.0%})")
        
        # Load all positions and group by grade
        positions_by_grade = defaultdict(list)
        
        with open(self.input_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line_num % 10000 == 0:
                    logger.info(f"Loading positions for split: {line_num:,}")
                
                record = json.loads(line)
                grade = record.get('stockfish_analysis', {}).get('move_quality_grade', -1)
                positions_by_grade[grade].append(line)
        
        # Create splits directory
        splits_dir = self.output_dir / 'splits'
        splits_dir.mkdir(exist_ok=True)
        
        # Stratified split for each grade
        import random
        random.seed(42)  # For reproducibility
        
        train_file = open(splits_dir / 'train.jsonl', 'w')
        val_file = open(splits_dir / 'validation.jsonl', 'w')
        test_file = open(splits_dir / 'test.jsonl', 'w')
        
        train_count, val_count, test_count = 0, 0, 0
        
        for grade, positions in positions_by_grade.items():
            random.shuffle(positions)
            
            n_total = len(positions)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            train_positions = positions[:n_train]
            val_positions = positions[n_train:n_train+n_val]
            test_positions = positions[n_train+n_val:]
            
            for pos in train_positions:
                train_file.write(pos)
                train_count += 1
            
            for pos in val_positions:
                val_file.write(pos)
                val_count += 1
            
            for pos in test_positions:
                test_file.write(pos)
                test_count += 1
        
        train_file.close()
        val_file.close()
        test_file.close()
        
        logger.info(f"Created splits:")
        logger.info(f"  Train: {train_count:,} positions ({train_count/self.total_positions*100:.1f}%)")
        logger.info(f"  Validation: {val_count:,} positions ({val_count/self.total_positions*100:.1f}%)")
        logger.info(f"  Test: {test_count:,} positions ({test_count/self.total_positions*100:.1f}%)")
        
        # Save split metadata
        split_info = {
            'creation_timestamp': datetime.now().isoformat(),
            'total_positions': self.total_positions,
            'train': {'count': train_count, 'ratio': train_ratio, 'file': 'splits/train.jsonl'},
            'validation': {'count': val_count, 'ratio': val_ratio, 'file': 'splits/validation.jsonl'},
            'test': {'count': test_count, 'ratio': 1-train_ratio-val_ratio, 'file': 'splits/test.jsonl'},
            'stratification': 'by_move_quality_grade',
            'random_seed': 42
        }
        
        with open(splits_dir / 'split_info.json', 'w') as f:
            json.dump(split_info, f, indent=2)
        
        logger.info(f"Split metadata saved to {splits_dir / 'split_info.json'}")
    
    @staticmethod
    def _safe_mean(values: List[float]) -> float:
        """Calculate mean, handling empty lists."""
        return sum(values) / len(values) if values else 0.0
    
    @staticmethod
    def _safe_median(values: List[float]) -> float:
        """Calculate median, handling empty lists."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        n = len(sorted_values)
        if n % 2 == 0:
            return (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
        else:
            return sorted_values[n//2]


def main():
    parser = argparse.ArgumentParser(description='Analyze V7P3R AI training dataset')
    parser.add_argument('--input', required=True, help='Input JSONL dataset file')
    parser.add_argument('--output', default='data/analysis', help='Output directory for analysis')
    parser.add_argument('--create-splits', action='store_true', help='Create train/val/test splits')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Training set ratio (default: 0.8)')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation set ratio (default: 0.1)')
    
    args = parser.parse_args()
    
    analyzer = DatasetAnalyzer(args.input, args.output)
    analyzer.analyze()
    analyzer.generate_report()
    
    if args.create_splits:
        analyzer.create_splits(args.train_ratio, args.val_ratio)
    
    logger.info("✅ Analysis complete!")


if __name__ == '__main__':
    main()
