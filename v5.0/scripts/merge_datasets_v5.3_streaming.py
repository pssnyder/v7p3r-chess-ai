"""
Memory-efficient streaming merge for v5.3 datasets.

Uses two-pass approach:
1. First pass: Count records and analyze grade distribution
2. Second pass: Stream merge with chunk-based shuffling

Designed to handle 6M+ positions without loading everything into RAM.
"""

import json
import os
import random
from pathlib import Path
from collections import Counter
from typing import Dict, List
import tempfile
import shutil


class StreamingDatasetMerger:
    """Memory-efficient dataset merger using streaming and chunked shuffling."""
    
    def __init__(self, output_path: str, chunk_size: int = 50000):
        self.output_path = output_path
        self.chunk_size = chunk_size  # Records per shuffle chunk
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
    
    def _extract_grade(self, record: dict) -> int:
        """Extract grade from record (handles different formats)."""
        if 'grade' in record:
            return record['grade']
        elif 'stockfish_analysis' in record and 'grade' in record['stockfish_analysis']:
            return record['stockfish_analysis']['grade']
        else:
            return 5  # Default to worst grade
    
    def _count_and_analyze(self, path: str) -> tuple:
        """First pass: count records and analyze grades without loading all into memory."""
        count = 0
        grade_counts = Counter()
        
        with open(path, 'r') as f:
            for line in f:
                count += 1
                record = json.loads(line.strip())
                grade = self._extract_grade(record)
                grade_counts[grade] += 1
                
                # Progress update every 100k records
                if count % 100000 == 0:
                    print(f"    Analyzed: {count:,} records...")
        
        return count, grade_counts
    
    def analyze_all_sources(self):
        """First pass: analyze all sources to get statistics."""
        print("\n" + "="*60)
        print("ANALYZING SOURCES (Pass 1)")
        print("="*60)
        
        source_stats = []
        
        for source in self.sources:
            print(f"\n📂 {source['name']}")
            print(f"  Counting records in {os.path.basename(source['path'])}...")
            
            count, grade_counts = self._count_and_analyze(source['path'])
            
            self.total_positions += count
            self.grade_distribution.update(grade_counts)
            
            # Track stats
            source_stats.append({
                'name': source['name'],
                'count': count,
                'expected': source['expected'],
                'grades': dict(grade_counts)
            })
            
            print(f"  ✓ Records: {count:,}")
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
        
        print(f"\n📊 Total positions across all sources: {self.total_positions:,}")
        
        return source_stats
    
    def _stream_merge_chunk(self, chunk: list, output_handle):
        """Shuffle a chunk and write to output."""
        random.shuffle(chunk)
        for record in chunk:
            output_handle.write(json.dumps(record) + '\n')
    
    def merge_all_streaming(self):
        """Second pass: stream merge with chunked shuffling."""
        print("\n" + "="*60)
        print("MERGING DATASETS (Pass 2 - Streaming)")
        print("="*60)
        print(f"Chunk size: {self.chunk_size:,} records")
        print(f"Output: {self.output_path}")
        
        # Create output directory
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        chunk = []
        positions_written = 0
        
        with open(self.output_path, 'w') as output_handle:
            # Stream through all sources
            for source in self.sources:
                print(f"\n📂 Processing: {source['name']}")
                
                with open(source['path'], 'r') as f:
                    for line in f:
                        record = json.loads(line.strip())
                        chunk.append(record)
                        
                        # When chunk is full, shuffle and write
                        if len(chunk) >= self.chunk_size:
                            self._stream_merge_chunk(chunk, output_handle)
                            positions_written += len(chunk)
                            chunk = []
                            
                            if positions_written % 500000 == 0:
                                print(f"  ✓ Written: {positions_written:,} positions...")
            
            # Write remaining chunk
            if chunk:
                self._stream_merge_chunk(chunk, output_handle)
                positions_written += len(chunk)
        
        print(f"\n✅ Total written: {positions_written:,} positions")
        
        return positions_written
    
    def print_summary(self, source_stats: List[Dict], positions_written: int):
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
            print("  Consider using weighted loss during training")
            print("  Suggested weights:")
            
            # Calculate inverse frequency weights
            total = self.total_positions
            for grade in sorted(self.grade_distribution.keys()):
                count = self.grade_distribution[grade]
                weight = total / (len(self.grade_distribution) * count)
                print(f"    Grade {grade}: {weight:.3f}")
        
        print(f"\n✅ Output: {self.output_path}")
        file_size_mb = os.path.getsize(self.output_path) / (1024**2)
        print(f"   Size: {file_size_mb:,.1f} MB")
        print(f"   Positions: {positions_written:,}")


def main():
    """Main streaming merge workflow."""
    
    print("="*60)
    print("V7P3R AI v5.3 - Streaming Dataset Merge")
    print("="*60)
    print("Memory-efficient approach using chunked processing")
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    output_file = base_dir / "data" / "final" / "v7p3r_ai_v5.3_merged.jsonl"
    
    # Initialize merger with 50k chunk size (smaller chunks = less RAM)
    merger = StreamingDatasetMerger(str(output_file), chunk_size=50000)
    
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
    
    for me_file in multi_engine_files:
        if me_file.exists():
            version = me_file.stem.split('_')[0]
            merger.add_source(
                str(me_file),
                f"Multi-Engine {version}",
                expected_count=1000
            )
    
    # First pass: analyze all sources
    source_stats = merger.analyze_all_sources()
    
    # Ask user to proceed
    print("\n" + "="*60)
    print("Analysis complete. Ready to merge.")
    print(f"Total positions: {merger.total_positions:,}")
    print(f"Estimated output size: ~{merger.total_positions * 3.5 / 1024:.1f} GB")
    print("="*60)
    input("\nPress Enter to start merge (or Ctrl+C to cancel)...")
    
    # Second pass: streaming merge
    positions_written = merger.merge_all_streaming()
    
    # Print summary
    merger.print_summary(source_stats, positions_written)
    
    print("\n" + "="*60)
    print("✅ MERGE COMPLETE")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. Review grade distribution above")
    print(f"2. Run batched preprocessing (to be created)")
    print(f"3. Train v5.3 with suggested class weights")
    print()


if __name__ == "__main__":
    random.seed(42)  # Reproducible shuffle
    main()
