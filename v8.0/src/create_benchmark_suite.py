#!/usr/bin/env python3
"""
Chess Engine Benchmark Suite Creator

Samples 100 puzzles from Lichess database across 5 difficulty tiers for standardized
engine testing. Creates a reusable benchmark suite that can test any UCI engine.

Tiers:
- Tier 1 (400-800):    Beginner (hanging pieces, mate-in-1/2)
- Tier 2 (800-1200):   Weak (basic tactics, simple endgames)
- Tier 3 (1200-1600):  Intermediate (complex tactics, technique)
- Tier 4 (1600-2000):  Advanced (positional play, deep calculation)
- Tier 5 (2000-2400):  Expert (master-level tactics)

Output: benchmark_suite.json with 20 puzzles per tier (100 total)
"""

import sys
import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

# Add engine-tester databases to path
ENGINE_TESTER_PATH = Path(r"E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-tester")
sys.path.insert(0, str(ENGINE_TESTER_PATH / "databases"))

from database import PuzzleDatabase, Puzzle


class BenchmarkSuiteCreator:
    """Creates standardized benchmark suite from Lichess puzzle database"""
    
    def __init__(self, db_path: str):
        self.db = PuzzleDatabase(db_path)
        self.tiers = [
            {"name": "tier1_beginner", "min": 400, "max": 800, "count": 20},
            {"name": "tier2_weak", "min": 800, "max": 1200, "count": 20},
            {"name": "tier3_intermediate", "min": 1200, "max": 1600, "count": 20},
            {"name": "tier4_advanced", "min": 1600, "max": 2000, "count": 20},
            {"name": "tier5_expert", "min": 2000, "max": 2400, "count": 20},
        ]
    
    def sample_puzzles_for_tier(self, tier_config: Dict) -> List[Dict]:
        """Sample puzzles from database for a specific tier"""
        name = tier_config['name']
        min_rating = tier_config['min']
        max_rating = tier_config['max']
        count = tier_config['count']
        
        print(f"\n📊 Sampling {count} puzzles for {name} (rating {min_rating}-{max_rating})...")
        
        # Query more than we need for diversity selection
        sample_size = count * 5
        puzzles = self.db.query_puzzles(
            min_rating=min_rating,
            max_rating=max_rating,
            quantity=sample_size
        )
        
        if len(puzzles) < count:
            print(f"⚠️  Warning: Only found {len(puzzles)} puzzles in this tier (wanted {count})")
            selected = puzzles
        else:
            # Randomly select from the sample to ensure diversity
            selected = random.sample(puzzles, count)
        
        # Convert to serializable format
        puzzle_data = []
        for puzzle in selected:
            puzzle_data.append({
                'id': puzzle.id,
                'fen': puzzle.fen,
                'moves': puzzle.moves,
                'rating': puzzle.rating,
                'themes': puzzle.themes,
                'popularity': puzzle.popularity if hasattr(puzzle, 'popularity') else 0
            })
        
        print(f"✅ Selected {len(puzzle_data)} puzzles for {name}")
        return puzzle_data
    
    def create_suite(self, output_path: str = None) -> Dict:
        """Create complete benchmark suite with all tiers"""
        if output_path is None:
            output_path = str(Path(__file__).parent.parent / "benchmarks" / "benchmark_suite.json")
        
        print("🎯 Creating Chess Engine Benchmark Suite")
        print("=" * 60)
        
        suite = {
            'metadata': {
                'version': '1.0',
                'created': str(Path(__file__).parent),
                'total_puzzles': 100,
                'tiers': 5,
                'puzzles_per_tier': 20,
                'source': 'Lichess Puzzle Database',
                'description': 'Standardized benchmark for UCI chess engine ELO estimation'
            },
            'tiers': {}
        }
        
        # Sample puzzles for each tier
        for tier_config in self.tiers:
            tier_name = tier_config['name']
            puzzles = self.sample_puzzles_for_tier(tier_config)
            
            suite['tiers'][tier_name] = {
                'rating_range': [tier_config['min'], tier_config['max']],
                'expected_count': tier_config['count'],
                'actual_count': len(puzzles),
                'puzzles': puzzles
            }
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save suite to JSON
        with open(output_file, 'w') as f:
            json.dump(suite, f, indent=2)
        
        print("\n" + "=" * 60)
        print(f"✅ Benchmark suite created: {output_file}")
        print(f"📊 Total puzzles: {sum(t['actual_count'] for t in suite['tiers'].values())}")
        print("\nTier Summary:")
        for tier_name, tier_data in suite['tiers'].items():
            count = tier_data['actual_count']
            rating_range = tier_data['rating_range']
            print(f"  {tier_name}: {count} puzzles ({rating_range[0]}-{rating_range[1]} rating)")
        
        return suite
    
    def validate_suite(self, suite_path: str):
        """Validate that a benchmark suite is properly formatted"""
        with open(suite_path, 'r') as f:
            suite = json.load(f)
        
        print("\n🔍 Validating benchmark suite...")
        
        total_puzzles = 0
        issues = []
        
        for tier_name, tier_data in suite['tiers'].items():
            tier_puzzles = tier_data['puzzles']
            total_puzzles += len(tier_puzzles)
            
            # Check puzzle structure
            for i, puzzle in enumerate(tier_puzzles):
                required_fields = ['id', 'fen', 'moves', 'rating']
                missing = [f for f in required_fields if f not in puzzle]
                if missing:
                    issues.append(f"{tier_name} puzzle {i}: Missing fields {missing}")
        
        print(f"Total puzzles: {total_puzzles}")
        print(f"Expected: {suite['metadata']['total_puzzles']}")
        
        if issues:
            print(f"\n⚠️  Found {len(issues)} issues:")
            for issue in issues[:10]:  # Show first 10
                print(f"  - {issue}")
        else:
            print("✅ Suite is valid!")
        
        return len(issues) == 0


def main():
    """Create benchmark suite from Lichess database"""
    
    # Database path
    db_path = ENGINE_TESTER_PATH / "databases" / "puzzles.db"
    
    if not db_path.exists():
        print(f"❌ Error: Puzzle database not found at {db_path}")
        print("Please ensure the database exists before creating benchmark suite.")
        return 1
    
    # Create suite
    creator = BenchmarkSuiteCreator(str(db_path))
    
    # Set random seed for reproducibility (can be changed to regenerate different suite)
    random.seed(42)
    
    suite = creator.create_suite()
    
    # Validate the created suite
    suite_path = Path(__file__).parent.parent / "benchmarks" / "benchmark_suite.json"
    creator.validate_suite(str(suite_path))
    
    print("\n🎯 Benchmark suite ready for engine testing!")
    print(f"Next step: Run benchmark_single_engine.py to test an engine")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
