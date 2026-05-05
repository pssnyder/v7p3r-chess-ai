#!/usr/bin/env python3
"""
Quick test for v20.0.2 A/B test variants
Verifies both Material and Positional variants initialize and can evaluate positions
"""

import chess
import time

def test_variant(variant_name, variant_module):
    """Test a single variant"""
    print(f"\n{'='*70}")
    print(f"Testing {variant_name}")
    print(f"{'='*70}\n")
    
    # Import variant
    if variant_name == "Material":
        from v7p3r_v20_material_hybrid import V7P3R_v20_Hybrid
    else:
        from v7p3r_v20_positional_hybrid import V7P3R_v20_Hybrid
    
    # Initialize engine
    model_path = "models/stage2_combined/best_checkpoint.pt"
    engine = V7P3R_v20_Hybrid(model_path=model_path, device='cpu')
    
    # Test positions
    test_positions = [
        ("Starting position", chess.Board()),
        ("After 1.e4", chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")),
        ("Sicilian Defense", chess.Board("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2")),
        ("Center control", chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2")),
    ]
    
    print(f"\n📊 Evaluation Scores for {variant_name}:\n")
    
    for name, board in test_positions:
        start = time.time()
        score = engine.evaluate_position(board)
        eval_time = (time.time() - start) * 1000  # ms
        
        print(f"  {name:25s} Score: {score:+6d} cp  ({eval_time:.2f}ms)")
    
    # Test search
    print(f"\n🔍 Search Test ({variant_name}):\n")
    board = chess.Board()
    start = time.time()
    best_move = engine.search(board, depth=5, time_limit=3.0)
    search_time = time.time() - start
    
    print(f"  Best move: {best_move.uci() if best_move else 'None'}")
    print(f"  Search time: {search_time:.2f}s")
    print(f"  Nodes searched: {engine.nodes_searched:,}")
    print(f"  NPS: {int(engine.nodes_searched / search_time):,}")
    print(f"  TT hits: {engine.tt_hits:,}")
    print(f"  Killer hits: {engine.killer_hits:,}")
    
    return engine

def main():
    """Run all tests"""
    print("="*70)
    print("V7P3R v20.0.2 A/B Test Variant Verification")
    print("Testing Material and Positional variants")
    print("="*70)
    
    # Test Material variant
    material_engine = test_variant("Material", "v7p3r_v20_material_hybrid")
    
    # Test Positional variant
    positional_engine = test_variant("Positional", "v7p3r_v20_positional_hybrid")
    
    # Compare evaluations
    print(f"\n{'='*70}")
    print("📊 Evaluation Comparison")
    print(f"{'='*70}\n")
    
    test_boards = [
        ("Starting position", chess.Board()),
        ("Material imbalance", chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1")),  # White missing knight
        ("Center pawns", chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")),
    ]
    
    for name, board in test_boards:
        mat_score = material_engine.evaluate_position(board)
        pos_score = positional_engine.evaluate_position(board)
        diff = mat_score - pos_score
        
        print(f"{name:25s}")
        print(f"  Material variant:   {mat_score:+6d} cp")
        print(f"  Positional variant: {pos_score:+6d} cp")
        print(f"  Difference:         {diff:+6d} cp")
        print()
    
    print("="*70)
    print("✅ Both variants initialized successfully!")
    print("="*70)
    print("\nNext steps:")
    print("1. Add engines to Arena/Cutechess:")
    print("   - V7P3R v20.0.2-Material Beta → V7P3R_v20_Material.bat")
    print("   - V7P3R v20.0.2-Positional Beta → V7P3R_v20_Positional.bat")
    print("2. Run tournament: v20.0.2 vs v20.0.2-Material vs v20.0.2-Positional")
    print("3. Compare tactical accuracy, NPS, win rates")
    print("="*70)

if __name__ == "__main__":
    main()
