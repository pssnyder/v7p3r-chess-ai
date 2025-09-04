"""
V7P3R AI v3.0 - End-to-End Pipeline Test
========================================

Comprehensive test of the complete v3.0 autonomous training pipeline.
Tests all components working together and validates the system is ready
for full training deployment.
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_phase_1_system_initialization():
    """Phase 1: Test all system components can initialize"""
    print("🔧 PHASE 1: System Initialization Test")
    print("=" * 50)
    
    try:
        # Test AI components
        from ai.thinking_brain import ThinkingBrain
        from ai.gameplay_brain import GameplayBrain
        from training.self_play import SelfPlayTrainer
        
        # Initialize Thinking Brain
        thinking_brain = ThinkingBrain(
            input_size=690,
            hidden_size=128,  # Smaller for faster testing
            num_layers=4,
            output_size=2048,
            dropout=0.1,
            device="cuda" if __import__("torch").cuda.is_available() else "cpu"
        )
        print(f"✅ Thinking Brain initialized on {thinking_brain.device}")
        print(f"   Parameters: {sum(p.numel() for p in thinking_brain.parameters()):,}")
        
        # Initialize Gameplay Brain
        gameplay_brain = GameplayBrain(
            population_size=8,    # Smaller for faster testing
            simulation_depth=2,
            generations=4,
            mutation_rate=0.3,
            crossover_rate=0.7,
            time_limit=0.8,      # Faster for testing
            parallel_simulations=2
        )
        print("✅ Gameplay Brain initialized")
        print(f"   Population: {gameplay_brain.population_size}, Depth: {gameplay_brain.simulation_depth}")
        
        # Initialize Training System
        trainer = SelfPlayTrainer(
            thinking_brain=thinking_brain,
            gameplay_brain=gameplay_brain,
            save_directory="test_models",
            games_per_checkpoint=5,
            max_game_length=50,    # Shorter games for testing
            enable_progress_tracking=True,
            enable_visual_monitoring=False,  # Headless for this test
            monitoring_config={'enable_visual': False, 'save_data': True}
        )
        print("✅ Self-Play Trainer initialized")
        
        return thinking_brain, gameplay_brain, trainer
        
    except Exception as e:
        print(f"❌ Phase 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def test_phase_2_component_validation(thinking_brain, gameplay_brain, trainer):
    """Phase 2: Test individual components work correctly"""
    print("\n🔍 PHASE 2: Component Validation Test")
    print("=" * 50)
    
    try:
        import chess
        from core.chess_state import ChessStateExtractor
        from core.neural_features import NeuralFeatureConverter
        
        # Test ChessState extraction
        extractor = ChessStateExtractor()
        board = chess.Board()
        state = extractor.extract_state(board)
        print(f"✅ ChessState extraction: {len([p for p in state.pieces if p.type > 0])} pieces")
        
        # Test feature conversion
        converter = NeuralFeatureConverter()
        features = converter.convert_to_features(state, device=str(thinking_brain.device))
        print(f"✅ Feature conversion: {features.shape} tensor on {features.device}")
        
        # Test Thinking Brain
        legal_moves = list(board.legal_moves)
        candidates, probs, hidden = thinking_brain.generate_move_candidates(
            features, legal_moves[:5], top_k=3
        )
        print(f"✅ Thinking Brain: Generated {len(candidates)} move candidates")
        print(f"   Probabilities: {[f'{p:.3f}' for p in probs.cpu().numpy()]}")
        
        # Test Gameplay Brain
        start_time = time.time()
        best_move, analysis = gameplay_brain.select_best_move(
            board, candidates[:3]
        )
        selection_time = time.time() - start_time
        print(f"✅ Gameplay Brain: Selected {best_move} in {selection_time:.2f}s")
        print(f"   Fitness: {analysis['best_candidate'].fitness:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Phase 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase_3_short_training(trainer):
    """Phase 3: Run a very short training session"""
    print("\n🎮 PHASE 3: Short Training Session Test")
    print("=" * 50)
    
    try:
        print("Starting 3-game training session...")
        start_time = time.time()
        
        # Run very short training
        trainer.train(target_games=3, resume_from_checkpoint=False)
        
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.1f} seconds")
        
        # Check progress
        progress = trainer.progress
        print(f"   Games played: {progress.games_played}")
        print(f"   Win rate: {progress.wins / max(progress.games_played, 1):.3f}")
        print(f"   Average game length: {progress.avg_game_length:.1f}")
        print(f"   Model loss: {progress.latest_model_loss:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Phase 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase_4_visual_monitoring():
    """Phase 4: Test visual monitoring system (headless)"""
    print("\n👁️  PHASE 4: Visual Monitoring System Test")
    print("=" * 50)
    
    try:
        from monitoring.integration import IntegratedTrainingMonitor
        import chess
        import random
        
        # Test headless monitoring
        monitor = IntegratedTrainingMonitor(
            enable_visual=False,  # Headless for stability
            save_data=True,
            output_dir=Path("test_monitoring")
        )
        
        monitor.start_monitoring("e2e_test")
        
        # Simulate game activity
        board = chess.Board()
        monitor.update_position(board)
        
        for i in range(10):
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
                
            # Simulate AI activity
            candidates = legal_moves[:3]
            probabilities = [random.random() for _ in candidates]
            fitness_scores = [random.random() for _ in candidates]
            
            monitor.update_thinking_brain_activity(candidates, probabilities, 0.5)
            monitor.update_gameplay_brain_activity(candidates, fitness_scores, 0.2)
            
            # Make a move
            move = random.choice(legal_moves)
            monitor.record_move_made(move, "test_brain", random.random())
            board.push(move)
            monitor.update_position(board)
        
        monitor.record_game_end("Test complete", "E2E test", 10)
        
        # Get stats
        stats = monitor.get_session_stats()
        print(f"✅ Monitoring system: {stats['total_events']} events recorded")
        print(f"   Positions: {stats['total_positions']}")
        print(f"   Thinking updates: {stats['total_thinking_updates']}")
        print(f"   Moves made: {stats['total_moves_made']}")
        
        monitor.stop_monitoring()
        return True
        
    except Exception as e:
        print(f"❌ Phase 4 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase_5_model_deployment():
    """Phase 5: Test trained model can be used for gameplay"""
    print("\n🚀 PHASE 5: Model Deployment Test")
    print("=" * 50)
    
    try:
        # Load the model from Phase 3
        from ai.thinking_brain import ThinkingBrain
        from ai.gameplay_brain import GameplayBrain
        import chess
        import torch
        
        # Create fresh instances
        thinking_brain = ThinkingBrain(
            input_size=690,
            hidden_size=128,
            num_layers=4,
            output_size=2048,
            dropout=0.1,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        gameplay_brain = GameplayBrain(
            population_size=8,
            simulation_depth=2,
            generations=4,
            time_limit=0.8
        )
        
        # Try to load trained weights if they exist
        model_path = Path("test_models/v7p3r_model.pkl")
        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location=thinking_brain.device)
                thinking_brain.load_state_dict(checkpoint['model_state_dict'])
                print("✅ Loaded trained model weights")
            except:
                print("⚠️  Using fresh model (trained weights not compatible)")
        else:
            print("⚠️  Using fresh model (no trained weights found)")
        
        # Test playing a few moves
        board = chess.Board()
        print(f"✅ Starting position: {board.fen()}")
        
        for move_num in range(5):
            if board.is_game_over():
                break
                
            # Get move from AI
            from core.chess_state import ChessStateExtractor
            from core.neural_features import NeuralFeatureConverter
            
            extractor = ChessStateExtractor()
            converter = NeuralFeatureConverter()
            
            state = extractor.extract_state(board)
            features = converter.convert_to_features(state, device=str(thinking_brain.device))
            
            legal_moves = list(board.legal_moves)
            candidates, probs, _ = thinking_brain.generate_move_candidates(
                features, legal_moves[:5], top_k=3
            )
            
            best_move, analysis = gameplay_brain.select_best_move(board, candidates)
            
            print(f"   Move {move_num + 1}: {best_move} (fitness: {analysis['best_candidate'].fitness:.3f})")
            board.push(best_move)
        
        print(f"✅ Final position: {board.fen()}")
        print(f"   Game result: {board.result()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Phase 5 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_comprehensive_e2e_test():
    """Run the complete end-to-end test suite"""
    print("🎯 V7P3R AI v3.0 - Comprehensive End-to-End Test Suite")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = []
    start_time = time.time()
    
    # Phase 1: System Initialization
    thinking_brain, gameplay_brain, trainer = test_phase_1_system_initialization()
    results.append(("System Initialization", thinking_brain is not None))
    
    if thinking_brain is None:
        print("\n❌ Cannot continue - system initialization failed")
        return False
    
    # Phase 2: Component Validation
    validation_result = test_phase_2_component_validation(thinking_brain, gameplay_brain, trainer)
    results.append(("Component Validation", validation_result))
    
    if not validation_result:
        print("\n❌ Cannot continue - component validation failed")
        return False
    
    # Phase 3: Short Training
    training_result = test_phase_3_short_training(trainer)
    results.append(("Short Training", training_result))
    
    # Phase 4: Visual Monitoring
    monitoring_result = test_phase_4_visual_monitoring()
    results.append(("Visual Monitoring", monitoring_result))
    
    # Phase 5: Model Deployment
    deployment_result = test_phase_5_model_deployment()
    results.append(("Model Deployment", deployment_result))
    
    # Summary
    total_time = time.time() - start_time
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print("\n" + "=" * 60)
    print("🏁 END-TO-END TEST SUMMARY")
    print("=" * 60)
    
    for phase, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {phase}")
    
    print(f"\nOverall Result: {passed}/{total} phases passed")
    print(f"Total Time: {total_time:.1f} seconds")
    
    if passed == total:
        print("\n🎉 V7P3R AI v3.0 is READY for full training deployment!")
        print("\nNext steps:")
        print("• Run: python main_trainer.py --visual (for visual training)")
        print("• Run: python main_trainer.py --no-visual (for headless training)")
        print("• Monitor training progress in logs and reports/")
        return True
    else:
        print(f"\n⚠️  {total - passed} phase(s) failed - system needs attention")
        print("Please review the error messages above before full deployment")
        return False


if __name__ == "__main__":
    try:
        success = run_comprehensive_e2e_test()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
