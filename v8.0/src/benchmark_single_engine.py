#!/usr/bin/env python3
"""
Single Engine Benchmark Tester

Tests a single UCI chess engine against the 100-puzzle benchmark suite.
Estimates ELO based on performance across 5 difficulty tiers.

Usage:
    python benchmark_single_engine.py "path/to/engine.exe"
    python benchmark_single_engine.py "path/to/engine.bat"

Output: JSON report with ELO estimate, tier scores, and diagnostic data
Runtime: ~10 minutes (5 seconds per puzzle)
"""

import sys
import os
import json
import subprocess
import time
import chess
import chess.engine
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class PuzzleResult:
    """Result of testing engine on a single puzzle"""
    puzzle_id: str
    tier: str
    rating: int
    engine_move: Optional[str]
    expected_moves: List[str]
    score: int  # 0-5 points
    time_ms: int
    success: bool
    error: Optional[str] = None


@dataclass
class TierPerformance:
    """Aggregated performance for a single tier"""
    tier_name: str
    rating_range: Tuple[int, int]
    total_puzzles: int
    solved: int
    total_score: int
    max_score: int
    accuracy: float
    avg_time_ms: float


@dataclass
class BenchmarkReport:
    """Complete benchmark report for an engine"""
    engine_name: str
    engine_path: str
    estimated_elo: int
    confidence_range: Tuple[int, int]
    total_score: int
    max_score: int
    overall_accuracy: float
    tier_performance: List[TierPerformance]
    puzzle_results: List[PuzzleResult]
    runtime_seconds: float
    timestamp: str


def run_with_timeout(func, timeout_seconds: int = 30):
    """Run a function with timeout using threading"""
    result = [None]
    exception = [None]
    
    def wrapper():
        try:
            result[0] = func()
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=wrapper, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)
    
    if thread.is_alive():
        raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
    
    if exception[0]:
        raise exception[0]
    
    return result[0]


class EngineBenchmark:
    """Benchmarks a single UCI engine against puzzle suite"""
    
    def __init__(self, engine_path: str, suite_path: str = None, time_per_puzzle: float = 5.0):
        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(f"Engine not found: {engine_path}")
        
        # Default suite path
        if suite_path is None:
            suite_path = Path(__file__).parent.parent / "benchmarks" / "benchmark_suite.json"
        
        self.suite_path = Path(suite_path)
        if not self.suite_path.exists():
            raise FileNotFoundError(f"Benchmark suite not found: {suite_path}")
        
        self.time_per_puzzle = time_per_puzzle
        self.engine_name = None
        
    def load_suite(self) -> Dict:
        """Load benchmark suite from JSON"""
        with open(self.suite_path, 'r') as f:
            return json.load(f)
    
    def get_engine_info(self) -> str:
        """Get engine name via UCI protocol (with 10 second timeout)"""
        def get_info():
            try:
                # Determine command based on file type
                if str(self.engine_path).lower().endswith('.bat'):
                    cmd = ['cmd.exe', '/c', str(self.engine_path)]
                else:
                    cmd = [str(self.engine_path)]
                
                process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                process.stdin.write("uci\n")
                process.stdin.flush()
                
                name = self.engine_path.stem
                for _ in range(20):
                    line = process.stdout.readline().strip()
                    if line.startswith("id name"):
                        name = line.split("id name", 1)[1].strip()
                    if line == "uciok":
                        break
                
                process.stdin.write("quit\n")
                process.stdin.flush()
                
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                
                return name
                
            except Exception as e:
                raise e
        
        try:
            return run_with_timeout(get_info, timeout_seconds=10)
        except TimeoutError:
            raise TimeoutError(f"Engine initialization timed out - engine may be hanging or unresponsive")
        except Exception as e:
            print(f"⚠️  Could not get engine info: {e}")
            return self.engine_path.stem
    
    def test_puzzle(self, puzzle: Dict, tier_name: str) -> PuzzleResult:
        """Test engine on a single puzzle"""
        puzzle_id = puzzle['id']
        fen = puzzle['fen']
        solution_moves = puzzle['moves'].split()
        
        # Lichess puzzle format: moves[0] = opponent setup, moves[1] = player solution
        if len(solution_moves) < 2:
            return PuzzleResult(
                puzzle_id=puzzle_id,
                tier=tier_name,
                rating=puzzle['rating'],
                engine_move=None,
                expected_moves=[],
                score=0,
                time_ms=0,
                success=False,
                error="Invalid puzzle format (need at least 2 moves)"
            )
        
        opponent_setup_move = solution_moves[0]  # Opponent's move to create puzzle position
        expected_solution = solution_moves[1]    # Player's solution move
        
        try:
            # Set up board from FEN
            board = chess.Board(fen)
            
            # Apply opponent's setup move to create the actual puzzle position
            try:
                setup_move = chess.Move.from_uci(opponent_setup_move)
            except:
                # Try SAN notation if UCI fails
                setup_move = board.parse_san(opponent_setup_move)
            
            board.push(setup_move)
            
            # Now get FEN of the position where engine should find the solution
            challenge_fen = board.fen()
            
            # Launch engine
            if str(self.engine_path).lower().endswith('.bat'):
                cmd = ['cmd.exe', '/c', str(self.engine_path)]
            else:
                cmd = [str(self.engine_path)]
            
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Initialize UCI
            process.stdin.write("uci\n")
            process.stdin.flush()
            
            # Wait for uciok
            for _ in range(20):
                line = process.stdout.readline().strip()
                if line == "uciok":
                    break
            
            # Set position to the challenge position (after opponent's move)
            process.stdin.write(f"position fen {challenge_fen}\n")
            process.stdin.write(f"go movetime {int(self.time_per_puzzle * 1000)}\n")
            process.stdin.flush()
            
            # Read bestmove
            start_time = time.time()
            engine_move = None
            
            for _ in range(100):
                line = process.stdout.readline().strip()
                if line.startswith("bestmove"):
                    parts = line.split()
                    if len(parts) >= 2:
                        engine_move = parts[1]
                    break
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            # Cleanup
            try:
                process.stdin.write("quit\n")
                process.stdin.flush()
                process.wait(timeout=2)
            except:
                process.kill()  # Force kill if unresponsive
            
            # Score the move (5 points if matches exact solution)
            score = 5 if engine_move == expected_solution else 0
            
            return PuzzleResult(
                puzzle_id=puzzle_id,
                tier=tier_name,
                rating=puzzle['rating'],
                engine_move=engine_move,
                expected_moves=[expected_solution],
                score=score,
                time_ms=elapsed_ms,
                success=engine_move is not None,
                error=None
            )
            
        except Exception as e:
            return PuzzleResult(
                puzzle_id=puzzle_id,
                tier=tier_name,
                rating=puzzle['rating'],
                engine_move=None,
                expected_moves=[expected_solution] if 'expected_solution' in locals() else [],
                score=0,
                time_ms=0,
                success=False,
                error=str(e)
            )
    
    def calculate_tier_performance(self, results: List[PuzzleResult], tier_name: str, rating_range: Tuple[int, int]) -> TierPerformance:
        """Calculate aggregated performance for a tier"""
        tier_results = [r for r in results if r.tier == tier_name]
        
        if not tier_results:
            return TierPerformance(
                tier_name=tier_name,
                rating_range=rating_range,
                total_puzzles=0,
                solved=0,
                total_score=0,
                max_score=0,
                accuracy=0.0,
                avg_time_ms=0.0
            )
        
        solved = sum(1 for r in tier_results if r.score > 0)
        total_score = sum(r.score for r in tier_results)
        max_score = len(tier_results) * 5
        accuracy = total_score / max_score if max_score > 0 else 0.0
        avg_time = sum(r.time_ms for r in tier_results) / len(tier_results)
        
        return TierPerformance(
            tier_name=tier_name,
            rating_range=rating_range,
            total_puzzles=len(tier_results),
            solved=solved,
            total_score=total_score,
            max_score=max_score,
            accuracy=accuracy,
            avg_time_ms=avg_time
        )
    
    def estimate_elo(self, tier_performances: List[TierPerformance]) -> Tuple[int, Tuple[int, int]]:
        """Estimate ELO based on tier performance"""
        # Find highest tier where engine scores >40%
        ceiling_tier = None
        for tier_perf in reversed(tier_performances):
            if tier_perf.accuracy >= 0.4:
                ceiling_tier = tier_perf
                break
        
        if ceiling_tier is None:
            # Failed even Tier 1
            return 200, (100, 400)
        
        # Estimate ELO as interpolation within ceiling tier
        tier_min, tier_max = ceiling_tier.rating_range
        tier_mid = (tier_min + tier_max) // 2
        
        # Adjust based on accuracy
        accuracy = ceiling_tier.accuracy
        if accuracy >= 0.8:
            elo = tier_max - 100  # High end of tier
        elif accuracy >= 0.6:
            elo = tier_mid  # Middle of tier
        else:
            elo = tier_min + 100  # Low end of tier
        
        # Confidence range
        confidence = (elo - 100, elo + 100)
        
        return elo, confidence
    
    def run_benchmark(self) -> BenchmarkReport:
        """Run complete benchmark suite"""
        start_time = time.time()
        
        print(f"🎯 Chess Engine Benchmark Test")
        print("=" * 60)
        print(f"Engine: {self.engine_path}")
        print(f"Time per puzzle: {self.time_per_puzzle}s")
        print("=" * 60)
        
        # Get engine info
        self.engine_name = self.get_engine_info()
        print(f"Engine name: {self.engine_name}")
        
        # Load suite
        suite = self.load_suite()
        print(f"Loaded benchmark suite: {suite['metadata']['total_puzzles']} puzzles")
        
        # Test each tier
        all_results = []
        should_continue = True
        
        for tier_name, tier_data in suite['tiers'].items():
            if not should_continue:
                print(f"\n⏭️  Skipping {tier_name} (engine failed previous tier)")
                continue
            
            puzzles = tier_data['puzzles']
            rating_range = tuple(tier_data['rating_range'])
            
            print(f"\n📊 Testing {tier_name} ({rating_range[0]}-{rating_range[1]}): {len(puzzles)} puzzles")
            
            tier_results = []
            for i, puzzle in enumerate(puzzles, 1):
                result = self.test_puzzle(puzzle, tier_name)
                all_results.append(result)
                tier_results.append(result)
                
                status = "✅" if result.success and result.score > 0 else "❌"
                print(f"  [{i}/{len(puzzles)}] {status} Puzzle {result.puzzle_id}: {result.score}/5 pts ({result.time_ms}ms)")
            
            # Check if we should continue to next tier
            tier_score = sum(r.score for r in tier_results)
            tier_max = len(tier_results) * 5
            tier_accuracy = tier_score / tier_max if tier_max > 0 else 0.0
            
            print(f"  Tier score: {tier_score}/{tier_max} ({tier_accuracy*100:.1f}%)")
            
            # Early termination: If can't solve >20% of Tier 1, stop
            if tier_name == 'tier1_beginner' and tier_accuracy < 0.2:
                print(f"\n⚠️  Engine failed Tier 1 ({tier_accuracy*100:.1f}% < 20%) - skipping remaining tiers")
                should_continue = False
        
        # Calculate tier performances
        tier_performances = []
        for tier_name, tier_data in suite['tiers'].items():
            rating_range = tuple(tier_data['rating_range'])
            perf = self.calculate_tier_performance(all_results, tier_name, rating_range)
            tier_performances.append(perf)
        
        # Estimate ELO
        estimated_elo, confidence = self.estimate_elo(tier_performances)
        
        # Calculate overall stats
        total_score = sum(r.score for r in all_results)
        max_score = len(all_results) * 5
        overall_accuracy = total_score / max_score if max_score > 0 else 0.0
        
        runtime = time.time() - start_time
        
        report = BenchmarkReport(
            engine_name=self.engine_name,
            engine_path=str(self.engine_path),
            estimated_elo=estimated_elo,
            confidence_range=confidence,
            total_score=total_score,
            max_score=max_score,
            overall_accuracy=overall_accuracy,
            tier_performance=tier_performances,
            puzzle_results=all_results,
            runtime_seconds=runtime,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        return report
    
    def save_report(self, report: BenchmarkReport, output_path: str = None):
        """Save benchmark report to JSON"""
        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            safe_name = report.engine_name.replace(' ', '_').replace('.', '_')
            output_path = Path(__file__).parent.parent / "benchmarks" / f"report_{safe_name}_{timestamp}.json"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSON-serializable format
        report_dict = {
            'engine_name': report.engine_name,
            'engine_path': report.engine_path,
            'estimated_elo': report.estimated_elo,
            'confidence_range': list(report.confidence_range),
            'total_score': report.total_score,
            'max_score': report.max_score,
            'overall_accuracy': report.overall_accuracy,
            'tier_performance': [asdict(tp) for tp in report.tier_performance],
            'puzzle_results': [asdict(pr) for pr in report.puzzle_results],
            'runtime_seconds': report.runtime_seconds,
            'timestamp': report.timestamp
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        print(f"\n📊 Report saved: {output_path}")
    
    def print_summary(self, report: BenchmarkReport):
        """Print human-readable summary"""
        print("\n" + "=" * 60)
        print("📊 BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Engine: {report.engine_name}")
        print(f"Estimated ELO: {report.estimated_elo} (±{(report.confidence_range[1]-report.estimated_elo)})")
        print(f"Overall Score: {report.total_score}/{report.max_score} ({report.overall_accuracy*100:.1f}%)")
        print(f"Runtime: {report.runtime_seconds:.1f}s")
        print("\nTier Performance:")
        
        for tier_perf in report.tier_performance:
            tier_name = tier_perf.tier_name.replace('_', ' ').title()
            rating_range = f"{tier_perf.rating_range[0]}-{tier_perf.rating_range[1]}"
            solved = tier_perf.solved
            total = tier_perf.total_puzzles
            accuracy = tier_perf.accuracy * 100
            score = tier_perf.total_score
            max_score = tier_perf.max_score
            
            status = "✅" if accuracy >= 60 else "⚠️" if accuracy >= 40 else "❌"
            print(f"  {status} {tier_name:20} ({rating_range}): {solved}/{total} solved, {score}/{max_score} pts ({accuracy:.1f}%)")
        
        print("=" * 60)


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python benchmark_single_engine.py <engine_path>")
        print("\nExample:")
        print('  python benchmark_single_engine.py "E:/Tournament Engines/V7P3R/V7P3R_v17.1/V7P3R_v17.1.bat"')
        return 1
    
    engine_path = sys.argv[1]
    
    try:
        benchmark = EngineBenchmark(engine_path)
        report = benchmark.run_benchmark()
        benchmark.print_summary(report)
        benchmark.save_report(report)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
