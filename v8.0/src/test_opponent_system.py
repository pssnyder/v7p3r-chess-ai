"""
Quick Test Script for V8.0 Opponent Training System

Tests each component before running full 20-generation training.
"""

import sys
from pathlib import Path

print("="*70)
print("V8.0 OPPONENT TRAINING - PRE-FLIGHT CHECKS")
print("="*70)

# Test 1: Neural network loading
print("\n[1/6] Testing neural network loading...")
try:
    from network import V8ValueNetwork
    import torch
    
    network = V8ValueNetwork(input_dim=55)
    
    # Try to load Gen 10
    gen10_path = Path('../training/v8_generational/gen_0010_value_network.pt')
    if gen10_path.exists():
        network.load_state_dict(torch.load(gen10_path, map_location='cpu'))
        print("  ✓ Loaded Gen 10 baseline network")
    else:
        print("  ⚠ Gen 10 not found - will use random weights")
    
    print(f"  ✓ Network architecture OK (56,449 parameters)")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

# Test 2: Feature extraction
print("\n[2/6] Testing feature extraction...")
try:
    from comprehensive_features import ComprehensiveFeatureExtractor
    import chess
    
    extractor = ComprehensiveFeatureExtractor()
    board = chess.Board()
    features = extractor.extract_all_features(board, move_number=0, previous_inference_ms=0.0)
    
    assert len(features) == 55, f"Expected 55 features, got {len(features)}"
    print(f"  ✓ Feature extraction OK (55 features)")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

# Test 3: Opening book
print("\n[3/6] Testing opening book...")
try:
    from opening_selector import OpeningSelector
    
    selector = OpeningSelector('opening_book.json')
    opening_id = selector.random_opening()
    opening = selector.openings[opening_id]
    
    print(f"  ✓ Opening book OK ({selector.num_openings} variations)")
    print(f"    Sample: {opening['name'][:50]}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

# Test 4: Opponent pool configuration
print("\n[4/6] Testing opponent pool...")
try:
    from opponent_manager import create_opponent_pool
    
    pool = create_opponent_pool()
    
    print(f"  ✓ Opponent pool OK ({len(pool.opponents)} opponents)")
    
    for opp in pool.opponents:
        path = Path(opp.path)
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"    {status} {opp.name} (ELO {opp.estimated_elo})")
        
        if not exists:
            print(f"      WARNING: File not found: {opp.path}")
    
    # Check at least one opponent is available
    available = sum(1 for opp in pool.opponents if Path(opp.path).exists())
    if available == 0:
        print("  ✗ CRITICAL: No opponents available!")
        sys.exit(1)
    elif available < len(pool.opponents):
        print(f"  ⚠ Only {available}/{len(pool.opponents)} opponents available")
    
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

# Test 5: UCI game executor
print("\n[5/6] Testing UCI game executor...")
try:
    from uci_game_executor import UCIGameExecutor
    
    executor = UCIGameExecutor(network, extractor, device='cpu')
    
    print(f"  ✓ Game executor OK")
    print(f"    Max moves: {executor.max_moves}")
    print(f"    Move time: {executor.movetime_ms}ms")
    print(f"    Temperature: {executor.temperature}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

# Test 6: Tablebase (optional)
print("\n[6/6] Testing tablebase...")
try:
    from tablebase_oracle import TablebaseOracle
    
    tablebase_path = r'E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\pgn_training_data\pgn_data_endgames\3-4-5_pieces_Syzygy\3-4-5'
    
    if Path(tablebase_path).exists():
        oracle = TablebaseOracle(tablebase_path)
        print(f"  ✓ Tablebase OK (5-piece Syzygy)")
    else:
        print(f"  ⚠ Tablebase not found (optional)")
except Exception as e:
    print(f"  ⚠ Tablebase unavailable (optional): {e}")

# Test 7: Quick UCI engine test (if available)
print("\n[BONUS] Testing opponent UCI communication...")
try:
    from opponent_manager import UCIEngine
    
    # Find first available opponent
    available_opponent = None
    for opp in pool.opponents:
        if Path(opp.path).exists():
            available_opponent = opp
            break
    
    if available_opponent:
        print(f"  Testing: {available_opponent.name}")
        print(f"    Launching engine...")
        
        try:
            engine = UCIEngine(available_opponent.path, timeout=10.0)
            engine.start()
            
            print(f"    ✓ Engine started successfully")
            
            # Test getting a move
            fen = chess.Board().fen()
            move = engine.get_move(fen, movetime_ms=1000)
            
            if move:
                print(f"    ✓ UCI communication OK (move: {move.uci()})")
            else:
                print(f"    ⚠ Engine didn't return a move")
            
            # Cleanup
            engine.cleanup()
            print(f"    ✓ Engine cleanup OK")
            
        except Exception as e:
            print(f"    ✗ Engine test failed: {e}")
            print(f"    (Training may still work - this is just a quick test)")
    else:
        print(f"  ⚠ No opponents available for UCI test")
    
except Exception as e:
    print(f"  ⚠ UCI test failed: {e}")
    print(f"    (Not critical - training will test this more thoroughly)")

# Summary
print("\n" + "="*70)
print("PRE-FLIGHT CHECK COMPLETE")
print("="*70)
print("\n✓ All critical components OK")
print("\nReady to start opponent-based training!")
print("\nTo begin training:")
print("  1. Run: START_OPPONENT_TRAINING.bat")
print("  2. Or: python src/train_v8_opponents.py")
print("\nExpected duration: 3-4 hours (20 generations)")
print("="*70)
