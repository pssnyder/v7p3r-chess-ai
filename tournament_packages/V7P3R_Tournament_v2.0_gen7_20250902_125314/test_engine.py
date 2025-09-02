
import sys
import time

try:
    from v7p3r_tournament_ai import V7P3RTournamentAI
    print("✅ Tournament AI import successful")
    
    ai = V7P3RTournamentAI()
    print("✅ Tournament AI initialization successful")
    
    # Test model loading
    ai.load_tournament_model("v7p3r_tournament_model.pth")
    print("✅ Model loading successful")
    
    # Test position setting
    ai.set_position("position startpos moves e2e4")
    print("✅ Position setting successful")
    
    # Test move generation
    move = ai.get_best_move("go movetime 1000")
    print(f"✅ Move generation successful: {move}")
    
    print("\n🎉 All tests passed! Tournament engine is ready.")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    sys.exit(1)
