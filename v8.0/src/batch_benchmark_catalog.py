#!/usr/bin/env python3
"""
Batch Benchmark All Engines in Catalog

Tests all engines listed in opponents_catalog.csv using the benchmark suite.
Updates the CSV with actual benchmark ELO estimates and tier performance data.

Usage:
    python batch_benchmark_catalog.py

Output: Updated opponents_catalog.csv with benchmark results
Runtime: ~10 minutes per engine (~10 hours for 60 engines)
"""

import sys
import csv
import json
import time
from pathlib import Path
from typing import List, Dict
from benchmark_single_engine import EngineBenchmark


class BatchBenchmark:
    """Batch benchmark all engines in catalog"""
    
    def __init__(self, catalog_path: str = None):
        if catalog_path is None:
            catalog_path = Path(__file__).parent.parent / "docs" / "opponents_catalog.csv"
        
        self.catalog_path = Path(catalog_path)
        if not self.catalog_path.exists():
            raise FileNotFoundError(f"Catalog not found: {catalog_path}")
        
        self.engines_base_path = Path(r"E:\Programming Stuff\Chess Engines\Tournament Engines")
        self.results = []
    
    def load_catalog(self) -> List[Dict]:
        """Load engines from CSV catalog"""
        engines = []
        with open(self.catalog_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                engines.append(row)
        return engines
    
    def save_catalog(self, engines: List[Dict]):
        """Save updated catalog back to CSV"""
        if not engines:
            return
        
        fieldnames = list(engines[0].keys())
        
        with open(self.catalog_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(engines)
        
        print(f"✅ Updated catalog saved: {self.catalog_path}")
    
    def get_engine_full_path(self, engine_data: Dict) -> Path:
        """Construct full path to engine executable"""
        path = engine_data['Path']
        
        # Handle different path formats
        if path.endswith('/'):
            # Directory - look for .bat or .exe
            dir_path = self.engines_base_path / path.rstrip('/')
            
            # Try .bat first (for V7P3R, C0BR4, etc.)
            engine_name = Path(path.rstrip('/')).name
            bat_file = dir_path / f"{engine_name}.bat"
            if bat_file.exists():
                return bat_file
            
            # Try .exe
            exe_file = dir_path / f"{engine_name}.exe"
            if exe_file.exists():
                return exe_file
            
            # Try main.py or similar
            py_file = dir_path / "main.py"
            if py_file.exists():
                return py_file
            
            return dir_path  # Return directory if nothing found
        else:
            # Direct file path
            return self.engines_base_path / path
    
    def should_test_engine(self, engine_data: Dict) -> bool:
        """Determine if engine should be tested"""
        status = engine_data.get('Status', 'UNTESTED').upper()
        
        # Skip if already tested or known broken
        if 'BENCHMARK_ELO' in engine_data.get('Notes', ''):
            return False  # Already benchmarked
        
        if status == 'BROKEN':
            return False
        
        # Test UNTESTED and FUNCTIONAL engines
        return status in ['UNTESTED', 'FUNCTIONAL']
    
    def test_engine(self, engine_data: Dict) -> Dict:
        """Test a single engine and return results"""
        engine_name = f"{engine_data['Engine']} {engine_data['Version']}"
        print(f"\n{'='*60}")
        print(f"🎯 Testing: {engine_name}")
        print(f"{'='*60}")
        
        engine_path = self.get_engine_full_path(engine_data)
        
        # Check if engine exists
        if not engine_path.exists():
            print(f"⏭️  SKIPPED - Engine not found: {engine_path}")
            return {
                'engine_data': engine_data,
                'status': 'NOT_FOUND',
                'benchmark_elo': 0,
                'tier_scores': [],
                'skipped': True
            }
        
        try:
            # Run benchmark
            print(f"🔧 Initializing engine: {engine_path.name}")
            benchmark = EngineBenchmark(str(engine_path), time_per_puzzle=5.0)
            
            print(f"📊 Running benchmark...")
            report = benchmark.run_benchmark()
            benchmark.print_summary(report)
            benchmark.save_report(report)
            
            # Extract tier scores for CSV
            tier_scores = []
            for tier_perf in report.tier_performance:
                tier_scores.append({
                    'tier': tier_perf.tier_name,
                    'solved': tier_perf.solved,
                    'total': tier_perf.total_puzzles,
                    'accuracy': tier_perf.accuracy
                })
            
            # Determine status based on performance
            if report.estimated_elo < 300:
                status = 'BROKEN'  # Can't solve basic puzzles
                print(f"⚠️  Status: BROKEN (ELO {report.estimated_elo} too low)")
            else:
                status = 'FUNCTIONAL'
                print(f"✅ Status: FUNCTIONAL (ELO {report.estimated_elo})")
            
            return {
                'engine_data': engine_data,
                'status': status,
                'benchmark_elo': report.estimated_elo,
                'confidence_range': report.confidence_range,
                'tier_scores': tier_scores,
                'overall_accuracy': report.overall_accuracy,
                'skipped': False
            }
            
        except FileNotFoundError as e:
            print(f"⏭️  SKIPPED - File error: {e}")
            return {
                'engine_data': engine_data,
                'status': 'FILE_ERROR',
                'benchmark_elo': 0,
                'tier_scores': [],
                'error': str(e),
                'skipped': True
            }
        
        except TimeoutError as e:
            print(f"⏭️  SKIPPED - Engine timeout (not responding to UCI)")
            return {
                'engine_data': engine_data,
                'status': 'TIMEOUT',
                'benchmark_elo': 0,
                'tier_scores': [],
                'error': "Engine did not respond to UCI commands",
                'skipped': True
            }
        
        except Exception as e:
            error_msg = str(e)
            print(f"⏭️  SKIPPED - Error: {error_msg}")
            
            # Log detailed error for later review
            if "illegal move" in error_msg.lower() or "invalid" in error_msg.lower():
                print(f"   (Engine may be incompatible with benchmark)")
            
            return {
                'engine_data': engine_data,
                'status': 'ERROR',
                'benchmark_elo': 0,
                'tier_scores': [],
                'error': error_msg,
                'skipped': True
            }
    
    def run_batch(self, max_engines: int = None, start_from: int = 0):
        """Run batch benchmark on all engines"""
        engines = self.load_catalog()
        
        print(f"📊 Batch Benchmark - Opponents Catalog")
        print(f"Total engines in catalog: {len(engines)}")
        
        # Filter engines to test
        to_test = [e for e in engines if self.should_test_engine(e)]
        print(f"Engines to test: {len(to_test)}")
        
        if max_engines:
            to_test = to_test[start_from:start_from + max_engines]
            print(f"Testing subset: {len(to_test)} engines (starting from #{start_from})")
        
        # Estimate time
        estimated_minutes = len(to_test) * 10
        print(f"Estimated time: {estimated_minutes} minutes ({estimated_minutes/60:.1f} hours)")
        
        input("\nPress Enter to start batch benchmark...")
        
        start_time = time.time()
        results = []
        
        # Test each engine
        for i, engine_data in enumerate(to_test, 1):
            print(f"\n{'#'*60}")
            print(f"Progress: {i}/{len(to_test)}")
            print(f"{'#'*60}")
            
            result = self.test_engine(engine_data)
            results.append(result)
            
            # Update catalog entry
            engine_data['Status'] = result['status']
            if result['benchmark_elo'] > 0:
                engine_data['Reference_ELO'] = str(result['benchmark_elo'])
                
                # Add tier scores to notes
                tier_summary = ", ".join([
                    f"{t['tier']}: {t['solved']}/{t['total']}"
                    for t in result.get('tier_scores', [])
                ])
                engine_data['Notes'] = f"Benchmark ELO {result['benchmark_elo']}, {tier_summary}"
            
            # Save progress after each engine
            self.save_catalog(engines)
            
            # Estimate remaining time
            elapsed = time.time() - start_time
            avg_time_per_engine = elapsed / i
            remaining = (len(to_test) - i) * avg_time_per_engine
            print(f"\n⏱️  Elapsed: {elapsed/60:.1f}m, Estimated remaining: {remaining/60:.1f}m")
        
        # Final summary
        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"✅ BATCH BENCHMARK COMPLETE")
        print(f"{'='*60}")
        print(f"Total engines tested: {len(results)}")
        print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
        
        # Count skipped vs. tested
        skipped_results = [r for r in results if r.get('skipped', False)]
        tested_results = [r for r in results if not r.get('skipped', False)]
        
        print(f"\nResults Summary:")
        print(f"  ✅ Successfully tested: {len(tested_results)}")
        print(f"  ⏭️  Skipped/Failed: {len(skipped_results)}")
        
        status_counts = {}
        for result in results:
            status = result['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print(f"\nStatus Breakdown:")
        for status, count in sorted(status_counts.items()):
            if status == 'FUNCTIONAL':
                symbol = "✅"
            elif status in ['NOT_FOUND', 'FILE_ERROR', 'TIMEOUT', 'ERROR']:
                symbol = "⏭️ "
            else:
                symbol = "⚠️ "
            print(f"  {symbol} {status}: {count} engines")
        
        # ELO distribution (only for tested engines)
        elos = [r['benchmark_elo'] for r in tested_results if r['benchmark_elo'] > 0]
        if elos:
            print(f"\nELO Distribution (tested engines only):")
            print(f"  Min: {min(elos)}")
            print(f"  Max: {max(elos)}")
            print(f"  Average: {sum(elos)/len(elos):.0f}")
            print(f"  Count: {len(elos)}")
        
        # Show skipped engines with reasons
        if skipped_results:
            print(f"\n⏭️  Skipped Engines ({len(skipped_results)}):")
            for result in skipped_results:
                engine_name = f"{result['engine_data']['Engine']} {result['engine_data']['Version']}"
                status = result['status']
                error = result.get('error', '')
                if error:
                    print(f"   • {engine_name}: {status} ({error[:60]}...)")
                else:
                    print(f"   • {engine_name}: {status}")
        
        print(f"\n✅ Updated catalog saved: {self.catalog_path}")
        
        return results


def main():
    """Main entry point"""
    
    # Parse command line args
    max_engines = None
    start_from = 0
    
    if len(sys.argv) > 1:
        max_engines = int(sys.argv[1])
    if len(sys.argv) > 2:
        start_from = int(sys.argv[2])
    
    try:
        batch = BatchBenchmark()
        results = batch.run_batch(max_engines=max_engines, start_from=start_from)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Batch benchmark interrupted by user")
        print("Progress has been saved to catalog")
        return 1
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
