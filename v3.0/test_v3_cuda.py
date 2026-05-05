"""
V7P3R CUDA Test for V3 Architecture
====================================

Quick test to verify RTX 4070 Ti can run V3 Thinking Brain
"""

import torch
import sys
from pathlib import Path

# Add v3.0/src to path
v3_src = Path(__file__).parent / "v3.0" / "src"
sys.path.insert(0, str(v3_src))

def test_cuda_setup():
    """Test basic CUDA functionality"""
    print("🔍 CUDA VERIFICATION TEST")
    print("=" * 50)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Device capability: {torch.cuda.get_device_capability(0)}")
        print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print()

def test_v3_thinking_brain():
    """Test V3 Thinking Brain on GPU"""
    print("🧠 V3 THINKING BRAIN GPU TEST")
    print("=" * 50)
    
    try:
        from ai.thinking_brain import ThinkingBrain
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Create Thinking Brain
        thinking_brain = ThinkingBrain(
            input_size=690,
            hidden_size=256,
            num_layers=8,
            output_size=4096,
            device=device
        )
        
        print(f"✅ Thinking Brain created successfully")
        print(f"   Model parameters: {sum(p.numel() for p in thinking_brain.model.parameters()):,}")
        print(f"   Model device: {next(thinking_brain.model.parameters()).device}")
        
        # Test forward pass with dummy data
        batch_size = 32
        dummy_input = torch.randn(batch_size, 690).to(device)
        
        print(f"\n🔄 Testing forward pass with batch size {batch_size}...")
        
        with torch.no_grad():
            start_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
            end_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
            
            if device == "cuda":
                start_time.record()
            
            output = thinking_brain.model(dummy_input)
            
            if device == "cuda":
                end_time.record()
                torch.cuda.synchronize()
                elapsed_time = start_time.elapsed_time(end_time)
                print(f"⚡ GPU inference time: {elapsed_time:.2f} ms")
                print(f"   Throughput: {batch_size / (elapsed_time / 1000):.0f} positions/second")
            
            print(f"✅ Output shape: {output.shape}")
            print(f"   Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        
        # Test memory usage
        if device == "cuda":
            memory_allocated = torch.cuda.memory_allocated(0) / 1e6
            memory_cached = torch.cuda.memory_reserved(0) / 1e6
            print(f"\n💾 GPU Memory usage:")
            print(f"   Allocated: {memory_allocated:.1f} MB")
            print(f"   Cached: {memory_cached:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Thinking Brain: {e}")
        return False

def test_chess_state_processing():
    """Test ChessState feature extraction"""
    print("\n♟️  CHESSSTATE FEATURE EXTRACTION TEST")
    print("=" * 50)
    
    try:
        import chess
        from core.chess_state import ChessState
        
        # Create test position
        board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
        chess_state = ChessState(board)
        
        # Extract features
        features = chess_state.extract_all_features()
        
        print(f"✅ ChessState created from position")
        print(f"   FEN: {board.fen()}")
        print(f"   Feature vector size: {len(features)}")
        print(f"   Feature range: [{min(features):.3f}, {max(features):.3f}]")
        
        # Convert to tensor and test GPU transfer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        feature_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        
        print(f"   Tensor device: {feature_tensor.device}")
        print(f"   Tensor shape: {feature_tensor.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing ChessState: {e}")
        return False

def main():
    """Run all CUDA tests"""
    print("🚀 V7P3R V3.0 CUDA COMPATIBILITY TEST")
    print("Testing RTX 4070 Ti with V3 Two-Brain Architecture")
    print("=" * 70)
    print()
    
    # Test basic CUDA
    test_cuda_setup()
    
    # Test V3 components
    brain_success = test_v3_thinking_brain()
    chess_success = test_chess_state_processing()
    
    print("\n" + "=" * 70)
    print("🏁 TEST SUMMARY")
    print("=" * 70)
    
    if brain_success and chess_success:
        print("✅ ALL TESTS PASSED!")
        print("🎉 Your RTX 4070 Ti is ready for V3 hybrid training!")
        print()
        print("Next steps:")
        print("1. Run: python v3_hybrid_main.py --hpts 1 --batch-size 30")
        print("2. Monitor GPU usage with: nvidia-smi")
        print("3. Watch training progress with enhanced analytics")
    else:
        print("❌ Some tests failed. Check error messages above.")
    
    print("=" * 70)

if __name__ == "__main__":
    main()