"""
Stage 1 Data Filtering - V7P3R AI v6.0

Filters the v5.3 merged dataset for binary classification:
- Good moves: Grade 0 (best) + Grade 1 (if within eval variance)
- Bad moves: Grades 2-5 (for negative reinforcement)

Filters out C0BR4 data (failed Stockfish analysis).
Keeps only Lichess puzzles + V7P3R games.
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict


class Stage1DataFilter:
    """Filter v5.3 dataset for Stage 1 training."""
    
    def __init__(self, 
                 input_path: str,
                 output_dir: str,
                 eval_variance_threshold: int = 50):
        """
        Args:
            input_path: Path to v5.3 merged dataset
            output_dir: Output directory for filtered datasets
            eval_variance_threshold: Max eval diff (cp) for Grade 1 inclusion
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.eval_threshold = eval_variance_threshold
        
        # Output paths
        self.good_output = self.output_dir / "good_positions.jsonl"
        self.bad_output = self.output_dir / "bad_positions.jsonl"
        
        # Statistics
        self.stats = defaultdict(int)
        self.source_stats = defaultdict(Counter)
        self.grade_stats = Counter()
    
    def _extract_eval_from_top_moves(self, top_moves: list) -> tuple:
        """
        Extract best and second-best evaluations.
        
        Returns:
            (best_eval, second_eval) or (None, None) if not available
        """
        if not top_moves or len(top_moves) < 1:
            return None, None
        
        best_eval = top_moves[0].get('eval')
        second_eval = top_moves[1].get('eval') if len(top_moves) >= 2 else None
        
        return best_eval, second_eval
    
    def _is_good_move(self, record: dict) -> bool:
        """
        Determine if move is "good" based on grade and eval variance.
        
        Good = Grade 0 (always) or Grade 1 (if within eval threshold of best)
        """
        try:
            # Extract grade
            if 'stockfish_analysis' in record and 'grade' in record['stockfish_analysis']:
                grade = record['stockfish_analysis']['grade']
            else:
                grade = record.get('grade', 5)
            
            # Grade 0 always good
            if grade == 0:
                return True
            
            # Grade 1: Check eval variance
            if grade == 1:
                sf_analysis = record.get('stockfish_analysis', {})
                top_moves = sf_analysis.get('top_moves', [])
                
                best_eval, second_eval = self._extract_eval_from_top_moves(top_moves)
                
                # If we can't get evals, exclude (conservative)
                if best_eval is None or second_eval is None:
                    self.stats['grade1_excluded_no_eval'] += 1
                    return False
                
                # Check if within threshold
                eval_diff = abs(best_eval - second_eval)
                
                if eval_diff <= self.eval_threshold:
                    self.stats['grade1_included_eval_ok'] += 1
                    return True
                else:
                    self.stats['grade1_excluded_eval_diff'] += 1
                    return False
            
            # Grades 2-5 are not "good"
            return False
        
        except Exception as e:
            self.stats['errors'] += 1
            return False
    
    def _is_bad_move(self, record: dict) -> bool:
        """
        Determine if move is "bad" (for negative reinforcement).
        
        Bad = Grades 2-5
        """
        try:
            if 'stockfish_analysis' in record and 'grade' in record['stockfish_analysis']:
                grade = record['stockfish_analysis']['grade']
            else:
                grade = record.get('grade', 5)
            
            return grade >= 2
        
        except Exception as e:
            return False
    
    def _extract_source(self, record: dict) -> str:
        """Extract source identifier."""
        return record.get('source', 'unknown')
    
    def filter_dataset(self):
        """Main filtering loop."""
        print("="*60)
        print("STAGE 1 DATA FILTERING - V7P3R AI v6.0")
        print("="*60)
        print(f"\nInput: {self.input_path}")
        print(f"Eval variance threshold: {self.eval_threshold}cp")
        print()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Open output files
        with open(self.good_output, 'w') as good_file, \
             open(self.bad_output, 'w') as bad_file, \
             open(self.input_path, 'r') as input_file:
            
            for line_num, line in enumerate(input_file, 1):
                try:
                    record = json.loads(line.strip())
                    
                    # Extract source and grade
                    source = self._extract_source(record)
                    
                    # Extract grade for stats
                    if 'stockfish_analysis' in record:
                        grade = record['stockfish_analysis'].get('grade', 5)
                    else:
                        grade = record.get('grade', 5)
                    
                    self.grade_stats[grade] += 1
                    self.source_stats[source][grade] += 1
                    
                    # Skip C0BR4 (failed Stockfish analysis)
                    if source == 'c0br4_game':
                        self.stats['c0br4_excluded'] += 1
                        continue
                    
                    # Classify as good or bad
                    is_good = self._is_good_move(record)
                    is_bad = self._is_bad_move(record)
                    
                    # Write to appropriate file
                    if is_good:
                        good_file.write(json.dumps(record) + '\n')
                        self.stats['good_positions'] += 1
                    elif is_bad:
                        bad_file.write(json.dumps(record) + '\n')
                        self.stats['bad_positions'] += 1
                    
                    # Progress update
                    if line_num % 100000 == 0:
                        print(f"  Processed: {line_num:,} records...")
                
                except Exception as e:
                    self.stats['errors'] += 1
                    if self.stats['errors'] < 10:
                        print(f"  Error on line {line_num}: {e}")
        
        print(f"\n✅ Filtering complete: {line_num:,} records processed")
        self.print_report()
    
    def print_report(self):
        """Print filtering statistics report."""
        print("\n" + "="*60)
        print("FILTERING REPORT")
        print("="*60)
        
        # Summary
        print("\n📊 SUMMARY")
        print("-" * 60)
        print(f"Good positions (G0 + filtered G1): {self.stats['good_positions']:,}")
        print(f"Bad positions (G2-G5):              {self.stats['bad_positions']:,}")
        print(f"C0BR4 excluded (failed analysis):   {self.stats['c0br4_excluded']:,}")
        print(f"Parsing errors:                      {self.stats['errors']:,}")
        
        # Imbalance ratio
        if self.stats['bad_positions'] > 0:
            ratio = self.stats['good_positions'] / self.stats['bad_positions']
            print(f"\nImbalance ratio: {ratio:.1f}:1 (good:bad)")
        
        # Grade 1 filtering details
        print("\n📈 GRADE 1 FILTERING")
        print("-" * 60)
        print(f"Included (eval variance ≤{self.eval_threshold}cp): {self.stats.get('grade1_included_eval_ok', 0):,}")
        print(f"Excluded (eval variance >{self.eval_threshold}cp):  {self.stats.get('grade1_excluded_eval_diff', 0):,}")
        print(f"Excluded (no eval data):             {self.stats.get('grade1_excluded_no_eval', 0):,}")
        
        # Grade distribution
        print("\n🎯 ORIGINAL GRADE DISTRIBUTION")
        print("-" * 60)
        for grade in sorted(self.grade_stats.keys()):
            count = self.grade_stats[grade]
            print(f"Grade {grade}: {count:>10,}")
        
        # Source breakdown
        print("\n📂 SOURCE BREAKDOWN")
        print("-" * 60)
        for source in sorted(self.source_stats.keys()):
            print(f"\n{source}:")
            for grade in sorted(self.source_stats[source].keys()):
                count = self.source_stats[source][grade]
                print(f"  Grade {grade}: {count:>10,}")
        
        # Output files
        print("\n💾 OUTPUT FILES")
        print("-" * 60)
        print(f"Good positions: {self.good_output}")
        print(f"  Size: {self.good_output.stat().st_size / 1e9:.2f} GB")
        print(f"Bad positions:  {self.bad_output}")
        print(f"  Size: {self.bad_output.stat().st_size / 1e6:.2f} MB")
        
        print("\n" + "="*60)


def main():
    """Main execution."""
    
    # Paths
    base_dir = Path(__file__).parent.parent.parent
    input_file = base_dir / ".." / "v5.0" / "data" / "final" / "v7p3r_ai_v5.3_merged.jsonl"
    output_dir = base_dir / "data" / "stage1"
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        return 1
    
    # Run filtering
    filter = Stage1DataFilter(
        input_path=str(input_file),
        output_dir=str(output_dir),
        eval_variance_threshold=50  # 50cp threshold for Grade 1
    )
    
    filter.filter_dataset()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
