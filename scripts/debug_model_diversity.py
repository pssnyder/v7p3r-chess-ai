#!/usr/bin/env python3
"""
Debug script to test if different models produce different outputs
"""

import os
import sys
import torch
import chess

# Add src directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '..', 'src')
sys.path.insert(0, src_dir)
sys.path.insert(0, os.path.join(src_dir, 'core'))
sys.path.insert(0, os.path.join(src_dir, 'models'))
sys.path.insert(0, os.path.join(src_dir, 'evaluation'))
sys.path.insert(0, os.path.join(src_dir, 'training'))

from v7p3r_gpu_model import V7P3RGPU_LSTM, GPUChessFeatureExtractor

def test_model_diversity():
    """Test if different models produce different outputs"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load existing best model
    try:
        best_model = V7P3RGPU_LSTM.load_model("models/best_gpu_model_gen_7.pth", device)
        print("✅ Loaded best_gpu_model_gen_7.pth")
    except Exception as e:
        print(f"❌ Failed to load best model: {e}")
        return
    
    # Create a few random models with same architecture
    config = {
        'input_size': 816,
        'hidden_size': 128, 
        'num_layers': 2,
        'output_size': 64
    }
    
    random_model1 = V7P3RGPU_LSTM(**config, device=device)
    random_model2 = V7P3RGPU_LSTM(**config, device=device)
    random_model1.to(device)
    random_model2.to(device)
    
    # Create feature extractor
    feature_extractor = GPUChessFeatureExtractor()
    feature_extractor.to(device)
    
    # Test on a simple chess position
    board = chess.Board()
    features = feature_extractor.extract_features(board)
    features_tensor = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    
    print(f"Input features shape: {features_tensor.shape}")
    
    # Test outputs from different models
    with torch.no_grad():
        best_model.eval()
        random_model1.eval()
        random_model2.eval()
        
        output_best, _ = best_model(features_tensor)
        output_random1, _ = random_model1(features_tensor)
        output_random2, _ = random_model2(features_tensor)
        
        print(f"Best model output shape: {output_best.shape}")
        print(f"Random model 1 output shape: {output_random1.shape}")
        print(f"Random model 2 output shape: {output_random2.shape}")
        
        # Convert to scalars using mean (like in training)
        best_scalar = output_best.squeeze().mean().cpu().item()
        random1_scalar = output_random1.squeeze().mean().cpu().item()
        random2_scalar = output_random2.squeeze().mean().cpu().item()
        
        print(f"\nScalar outputs:")
        print(f"Best model: {best_scalar:.6f}")
        print(f"Random model 1: {random1_scalar:.6f}")
        print(f"Random model 2: {random2_scalar:.6f}")
        
        # Check if outputs are different
        if abs(best_scalar - random1_scalar) > 1e-6:
            print("✅ Models produce different outputs - diversity exists")
        else:
            print("❌ Models produce identical outputs - no diversity!")
            
        if abs(random1_scalar - random2_scalar) > 1e-6:
            print("✅ Random models are different from each other")
        else:
            print("❌ Random models are identical - potential bug!")
        
        # Test on different positions
        print(f"\nTesting on different positions:")
        boards = [chess.Board(), chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")]
        
        for i, test_board in enumerate(boards):
            test_features = feature_extractor.extract_features(test_board)
            test_tensor = torch.tensor(test_features, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            
            test_output, _ = best_model(test_tensor)
            test_scalar = test_output.squeeze().mean().cpu().item()
            print(f"Position {i}: {test_scalar:.6f}")

if __name__ == "__main__":
    test_model_diversity()
