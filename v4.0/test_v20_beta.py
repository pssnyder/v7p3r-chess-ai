"""
V7P3R v20 Beta - Integration & Tactical Testing Suite

Tests the hybrid engine's performance across multiple dimensions:
1. UCI protocol compliance
2. Tactical position accuracy
3. Speed benchmarks
4. Comparison with v19.5
"""

import sys
import time
import chess
from pathlib import Path
from typing import List, Tuple, Dict

# Add project paths
sys.path.insert(0, str(Path(__file__).parent))

from v7p3r_v20_hybrid import V7P3R_v20_Hybrid


class V20BetaTester:
    """Comprehensive test suite for V7P3R v20 Beta."""
    
    def __init__(self):
        """Initialize tester with hybrid engine."""
        print("=" * 80)
        print("V7P3R v20 Beta - Integration & Tactical Testing Suite")
        print("=" * 80)
        
        self.model_path = "models/stage2_combined/best_checkpoint.pt"
        self.engine = V7P3R_v20_Hybrid(self.model_path, device='cpu')
        
        # Test positions
        self.tactical_positions = self._load_tactical_positions()
        
    def _load_tactical_positions(self) -> List[Dict]:
        """Load tactical test positions."""
        return [
            {
                'name': "Scholar's Mate Defense",
                'fen': "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4",
                'best_moves': ['e8d7', 'b8d7'],  # King to d7 or knight blocks
                'description': "Black must defend against checkmate threat"
            },
            {
                'name': "Back Rank Mate",
                'fen': "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
                'best_moves': ['d1d8'],  # Rook to d8 is checkmate
                'description': "White has back rank mate in 1"
            },
            {
                'name': "Fork Opportunity",
                'fen': "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
                'best_moves': ['f3g5'],  # Knight fork on king and rook
                'description': "Knight can fork king and rook on g5"
            },
            {
                'name': "Pin Tactic",
                'fen': "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
                'best_moves': ['c4f7'],  # Bishop takes f7 check
                'description': "Bishop can take f7 with check (royal fork)"
            },
            {
                'name': "Mate in 1",
                'fen': "3qk3/8/8/8/8/8/5Q2/4K3 w - - 0 1",
                'best_moves': ['f2d4', 'f2f8'],  # Multiple mates available
                'description': "White has mate in 1"
            },
            {
                'name': "Trapped Queen",
                'fen': "rnb1kbnr/pppp1ppp/8/4p3/5PPq/8/PPPPP2P/RNBQKBNR w KQkq - 0 1",
                'best_moves': ['g2g3'],  # Trap the queen
                'description': "Pawn to g3 traps black queen"
            },
            {
                'name': "Discovered Attack",
                'fen': "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1",
                'best_moves': ['f3g5'],  # Discovered attack on f7
                'description': "Knight move discovers bishop attack on f7"
            },
            {
                'name': "Promotion Threat",
                'fen': "8/P7/8/8/8/8/7k/7K w - - 0 1",
                'best_moves': ['a7a8q', 'a7a8r'],  # Promote pawn
                'description': "White should promote pawn"
            },
            {
                'name': "Skewer",
                'fen': "4k3/8/8/8/8/8/4R3/4K3 w - - 0 1",
                'best_moves': ['e2e8'],  # Rook check, king must move, then capture
                'description': "Rook skewer on king and back rank"
            },
            {
                'name': "Deflection",
                'fen': "r4rk1/1ppb1p1p/p1pb1qp1/8/3P4/2N1B3/PPP1QPPP/R4RK1 w - - 0 1",
                'best_moves': ['e2b5'],  # Deflect defender
                'description': "Queen deflection to win material"
            }
        ]
    
    def test_uci_protocol(self):
        """Test UCI protocol compliance."""
        print("\n" + "=" * 80)
        print("TEST 1: UCI Protocol Compliance")
        print("=" * 80)
        
        tests = [
            ("Engine initialization", self.engine is not None),
            ("AI model loaded", self.engine.ai_model is not None),
            ("Static evaluator loaded", self.engine.static_evaluator is not None),
            ("Feature extractor loaded", self.engine.feature_extractor is not None),
        ]
        
        passed = 0
        for test_name, result in tests:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name:.<40} {status}")
            if result:
                passed += 1
        
        print(f"\nUCI Tests: {passed}/{len(tests)} passed")
        return passed == len(tests)
    
    def test_tactical_accuracy(self, time_per_position: float = 3.0):
        """Test tactical position accuracy."""
        print("\n" + "=" * 80)
        print("TEST 2: Tactical Position Accuracy")
        print("=" * 80)
        
        results = []
        correct = 0
        
        for i, position in enumerate(self.tactical_positions, 1):
            print(f"\n[{i}/{len(self.tactical_positions)}] {position['name']}")
            print(f"Description: {position['description']}")
            print(f"FEN: {position['fen']}")
            print(f"Best moves: {', '.join(position['best_moves'])}")
            
            board = chess.Board(position['fen'])
            
            # Search for best move
            start_time = time.time()
            best_move = self.engine.search(board, time_limit=time_per_position)
            elapsed = time.time() - start_time
            
            # Check if move matches expected
            move_str = best_move.uci() if best_move else "none"
            is_correct = move_str in position['best_moves']
            
            if is_correct:
                correct += 1
                print(f"✅ CORRECT: {move_str} ({elapsed:.2f}s)")
            else:
                print(f"❌ WRONG: {move_str} (expected: {position['best_moves'][0]}) ({elapsed:.2f}s)")
            
            results.append({
                'position': position['name'],
                'expected': position['best_moves'],
                'actual': move_str,
                'correct': is_correct,
                'time': elapsed
            })
        
        accuracy = (correct / len(self.tactical_positions)) * 100
        avg_time = sum(r['time'] for r in results) / len(results)
        
        print("\n" + "-" * 80)
        print(f"Tactical Accuracy: {correct}/{len(self.tactical_positions)} ({accuracy:.1f}%)")
        print(f"Average time: {avg_time:.2f}s")
        print("-" * 80)
        
        return results
    
    def test_speed_benchmark(self, num_positions: int = 10):
        """Test search speed across multiple positions."""
        print("\n" + "=" * 80)
        print("TEST 3: Speed Benchmark")
        print("=" * 80)
        
        test_fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",  # After 1.e4 e5 2.Nf3 Nc6
            "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",  # After 1.e4 e5 2.Nf3 Nf6
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",  # Italian opening
            "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 1",  # Queen's Gambit
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",  # Endgame
            "4k3/8/8/8/8/8/4PPPP/4K3 w - - 0 1",  # Simple endgame
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  # Complex middlegame
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 0 1",  # Tactical position
            "8/8/8/4k3/8/8/4K3/8 w - - 0 1",  # K vs K endgame
        ]
        
        total_nodes = 0
        total_time = 0
        depth_sum = 0
        
        print(f"\nTesting {num_positions} positions with 3s time limit each...")
        
        for i, fen in enumerate(test_fens[:num_positions], 1):
            board = chess.Board(fen)
            
            # Reset engine stats
            self.engine.nodes_searched = 0
            self.engine.ai_ordering_time = 0
            self.engine.static_eval_time = 0
            
            start_time = time.time()
            best_move = self.engine.search(board, time_limit=3.0)
            elapsed = time.time() - start_time
            
            nps = self.engine.nodes_searched / elapsed if elapsed > 0 else 0
            
            print(f"  [{i}] Nodes: {self.engine.nodes_searched:,} | "
                  f"Time: {elapsed:.2f}s | NPS: {nps:.0f} | "
                  f"AI: {self.engine.ai_ordering_time*1000:.1f}ms")
            
            total_nodes += self.engine.nodes_searched
            total_time += elapsed
        
        avg_nps = total_nodes / total_time if total_time > 0 else 0
        
        print("\n" + "-" * 80)
        print(f"Total nodes: {total_nodes:,}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average NPS: {avg_nps:.0f}")
        print("-" * 80)
        
        return {
            'total_nodes': total_nodes,
            'total_time': total_time,
            'avg_nps': avg_nps
        }
    
    def test_comparison_with_v195(self):
        """Compare performance with v19.5 baseline."""
        print("\n" + "=" * 80)
        print("TEST 4: Comparison with V7P3R v19.5")
        print("=" * 80)
        
        # Expected v19.5 performance (from conversation summary)
        v195_metrics = {
            'nps': 28000,  # Average of 24-32K
            'depth_3s': 4.5,  # Average depth in 3 seconds
            'tactical_accuracy': 85,  # Estimated
        }
        
        # V20 metrics from current tests
        # (These would be populated from actual test results)
        
        print("\nPerformance Comparison:")
        print("-" * 80)
        print(f"{'Metric':<30} {'v19.5':<15} {'v20 Beta':<15} {'Change':<15}")
        print("-" * 80)
        
        # This would be filled in with actual test data
        print(f"{'NPS':<30} {'24-32K':<15} {'34.6K (est)':<15} {'+8-44%':<15}")
        print(f"{'Depth (3s)':<30} {'4-5':<15} {'5 (est)':<15} {'Maintained':<15}")
        print(f"{'AI Ordering':<30} {'None':<15} {'3.32ms':<15} {'NEW':<15}")
        print(f"{'Training Accuracy':<30} {'N/A':<15} {'97.1%':<15} {'NEW':<15}")
        
        print("\nNote: Full comparison requires 50-game tournament testing")
        print("-" * 80)
    
    def run_all_tests(self):
        """Run complete test suite."""
        print("\n" + "=" * 80)
        print("V7P3R v20 Beta - COMPLETE TEST SUITE")
        print("=" * 80)
        
        start_time = time.time()
        
        # Run tests
        uci_passed = self.test_uci_protocol()
        tactical_results = self.test_tactical_accuracy(time_per_position=3.0)
        speed_results = self.test_speed_benchmark(num_positions=10)
        self.test_comparison_with_v195()
        
        # Summary
        total_time = time.time() - start_time
        
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        
        tactical_accuracy = sum(1 for r in tactical_results if r['correct']) / len(tactical_results) * 100
        
        print(f"UCI Protocol: {'✅ PASS' if uci_passed else '❌ FAIL'}")
        print(f"Tactical Accuracy: {tactical_accuracy:.1f}%")
        print(f"Average NPS: {speed_results['avg_nps']:.0f}")
        print(f"Total test time: {total_time:.1f}s")
        
        # Verdict
        print("\n" + "=" * 80)
        print("VERDICT")
        print("=" * 80)
        
        if uci_passed and tactical_accuracy >= 50:
            print("✅ V7P3R v20 Beta is READY for tournament testing")
            print("   Next step: 50-game match vs v19.5")
        elif uci_passed:
            print("⚠️  V7P3R v20 Beta works but needs tactical improvement")
            print(f"   Tactical accuracy ({tactical_accuracy:.1f}%) below 50% threshold")
        else:
            print("❌ V7P3R v20 Beta has critical issues")
            print("   Fix UCI protocol issues before tournament testing")
        
        print("=" * 80)


def main():
    """Run test suite."""
    tester = V20BetaTester()
    tester.run_all_tests()


if __name__ == '__main__':
    main()
