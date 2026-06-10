"""
V7P3R AI v6.0 - Subset Training Test

Quick validation test (30-60 minutes):
- Load 10k positions (instead of 5.7M)
- Train for 5 epochs (instead of 100)
- Verify pipeline works end-to-end

If this succeeds, proceed to full training.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from train_policy import DataLoader, GraphAugmentedPolicyNetwork, Trainer

def main():
    print("=" * 70)
    print("SUBSET TRAINING TEST - V7P3R AI v6.0")
    print("=" * 70)
    print("\n🎯 Goal: Validate training pipeline on small dataset")
    print("⏱️  Expected time: 30-60 minutes")
    print("📊 Dataset: 10k positions (vs 5.7M full)")
    print("🔄 Epochs: 5 (vs 100 full)")
    print()
    
    # Configuration
    config = {
        'epochs': 5,
        'batch_size': 256,  # Smaller batch for CPU
        'learning_rate': 0.001,
        'patience': 10,
        'use_graph_attention': False,  # Start simple
        'output_dir': 'models/stage1_test',
    }
    
    print("📋 Configuration:")
    for key, val in config.items():
        print(f"   {key}: {val}")
    
    # Paths
    base_path = Path(__file__).parent.parent.parent
    good_path = base_path / "data" / "stage1" / "good_positions.jsonl"
    bad_path = base_path / "data" / "stage1" / "bad_positions.jsonl"
    graph_path = base_path / "data" / "stage1" / "transposition_graph.pkl"
    
    # Verify files exist
    if not good_path.exists():
        print(f"\n❌ ERROR: {good_path} not found!")
        return 1
    if not bad_path.exists():
        print(f"\n❌ ERROR: {bad_path} not found!")
        return 1
    if not graph_path.exists():
        print(f"\n❌ ERROR: {graph_path} not found!")
        return 1
    
    # Create output directory
    output_dir = base_path / config['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data (SUBSET)
    print("\n" + "=" * 70)
    print("LOADING SUBSET")
    print("=" * 70)
    
    loader = DataLoader(str(good_path), str(bad_path), str(graph_path), config)
    
    # Load 5k good + all bad (~69k)
    print("\n⚠️  Loading subset: 5,000 good positions + all bad positions")
    data = loader.load(max_samples=5000)
    
    print(f"\n✅ Data loaded:")
    print(f"   Train: {data['train'][0].shape[0]:,} positions")
    print(f"   Val:   {data['val'][0].shape[0]:,} positions")
    print(f"   Test:  {data['test'][0].shape[0]:,} positions")
    print(f"   Features: {data['feature_dim']}")
    
    # Train
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    trainer = Trainer(config)
    history = trainer.train(data, loader)
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]
    final_val_acc = history['val_acc'][-1]
    
    print(f"\n📊 Final Metrics (Epoch {len(history['train_loss'])}):")
    print(f"   Train Loss:     {final_train_loss:.4f}")
    print(f"   Val Loss:       {final_val_loss:.4f}")
    print(f"   Val Accuracy:   {final_val_acc:.4f}")
    
    # Evaluation
    print("\n📈 Training Progress:")
    print(f"   Epochs completed: {len(history['train_loss'])}")
    print(f"   Best val loss:    {min(history['val_loss']):.4f}")
    
    # Success criteria
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
    
    success = True
    
    # Check 1: No NaN losses
    if any(x != x for x in history['train_loss']):  # NaN check
        print("❌ FAIL: NaN losses detected")
        success = False
    else:
        print("✅ PASS: No NaN losses")
    
    # Check 2: Loss decreased
    if history['train_loss'][-1] < history['train_loss'][0]:
        print("✅ PASS: Training loss decreased")
    else:
        print("❌ FAIL: Training loss did not decrease")
        success = False
    
    # Check 3: Reasonable accuracy (>85% on easy subset)
    if final_val_acc > 0.85:
        print(f"✅ PASS: Val accuracy {final_val_acc:.2%} > 85%")
    else:
        print(f"⚠️  WARNING: Val accuracy {final_val_acc:.2%} < 85% (may be OK for subset)")
    
    # Check 4: Not severely overfitting
    overfit_gap = abs(final_train_loss - final_val_loss)
    if overfit_gap < 0.1:
        print(f"✅ PASS: No severe overfitting (gap = {overfit_gap:.4f})")
    else:
        print(f"⚠️  WARNING: Large train/val gap ({overfit_gap:.4f})")
    
    # Final verdict
    print("\n" + "=" * 70)
    if success:
        print("✅ SUCCESS - Pipeline validated!")
        print("\n🚀 Next steps:")
        print("   1. Review training curves in TensorBoard")
        print("   2. If satisfied, proceed to full training:")
        print("      python scripts/stage1/train_policy.py")
        print("   3. Monitor progress with: tensorboard --logdir models/stage1/logs")
    else:
        print("❌ FAILURE - Issues detected")
        print("\n🔧 Troubleshooting:")
        print("   1. Check data quality analysis output")
        print("   2. Review TESTING_GUIDE.md for common issues")
        print("   3. Try adjusting hyperparameters (LR, batch size)")
    print("=" * 70)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
