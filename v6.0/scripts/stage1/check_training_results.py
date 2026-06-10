"""
Check training results from saved checkpoint.
"""
import torch
from pathlib import Path


def main():
    """Load checkpoint and print training summary."""
    base_path = Path(__file__).parent.parent.parent
    
    # Load final model (has full history)
    final_model_path = base_path / "models" / "stage1" / "stage1_policy_final.pt"
    best_model_path = base_path / "models" / "stage1" / "stage1_policy_best.pt"
    
    print("=" * 60)
    print("TRAINING RESULTS SUMMARY")
    print("=" * 60)
    
    # Check final model
    if final_model_path.exists():
        print(f"\n✅ Final model: {final_model_path}")
        final_checkpoint = torch.load(final_model_path, map_location='cpu')
        
        if 'history' in final_checkpoint:
            history = final_checkpoint['history']
            
            num_epochs = len(history['train_loss'])
            print(f"\n📊 Training History ({num_epochs} epochs completed)")
            print("-" * 60)
            print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12}")
            print("-" * 60)
            
            for i in range(num_epochs):
                print(f"{i+1:<8} "
                      f"{history['train_loss'][i]:<12.4f} "
                      f"{history['train_acc'][i]:<12.4f} "
                      f"{history['val_loss'][i]:<12.4f} "
                      f"{history['val_acc'][i]:<12.4f}")
            
            # Summary stats
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            print(f"Total epochs: {num_epochs}")
            print(f"Best val_loss: {min(history['val_loss']):.4f} (epoch {history['val_loss'].index(min(history['val_loss']))+1})")
            print(f"Best val_acc: {max(history['val_acc']):.4f} (epoch {history['val_acc'].index(max(history['val_acc']))+1})")
            print(f"Final train_loss: {history['train_loss'][-1]:.4f}")
            print(f"Final train_acc: {history['train_acc'][-1]:.4f}")
            print(f"Final val_loss: {history['val_loss'][-1]:.4f}")
            print(f"Final val_acc: {history['val_acc'][-1]:.4f}")
            
            # Feature dimension
            if 'feature_dim' in final_checkpoint:
                print(f"Feature dimension: {final_checkpoint['feature_dim']}")
            
            # Check for overfitting
            gap = history['train_loss'][-1] - history['val_loss'][-1]
            if abs(gap) > 0.05:
                print(f"\n⚠️  Possible overfitting: train/val gap = {gap:.4f}")
            else:
                print(f"\n✅ No overfitting detected (train/val gap = {gap:.4f})")
        else:
            print("⚠️  No history found in final checkpoint")
    else:
        print(f"❌ Final model not found: {final_model_path}")
    
    # Check best model
    print("\n" + "=" * 60)
    if best_model_path.exists():
        print(f"✅ Best model: {best_model_path}")
        best_checkpoint = torch.load(best_model_path, map_location='cpu')
        
        if 'epoch' in best_checkpoint:
            print(f"   Best epoch: {best_checkpoint['epoch']+1}")
        if 'val_loss' in best_checkpoint:
            print(f"   Best val_loss: {best_checkpoint['val_loss']:.4f}")
    else:
        print(f"❌ Best model not found: {best_model_path}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
