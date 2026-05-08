"""
V7P3R AI v5.0 - Training Metrics Snapshot Tool

Automatically captures training metrics and saves them for historical trend analysis.

Usage:
    python scripts/snapshot_metrics.py --checkpoint checkpoints/best_model.pth --session-name "Full Training Session 1"
"""

import argparse
import json
import torch
from pathlib import Path
from datetime import datetime
import shutil


def load_training_data(checkpoint_path):
    """Load checkpoint and training history"""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_dir = checkpoint_path.parent
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load training history
    history_path = checkpoint_dir / 'training_history.json'
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
    else:
        history = None
    
    return checkpoint, history


def create_snapshot(checkpoint, history, session_name, session_id):
    """Create a metrics snapshot"""
    
    if history is None:
        print("⚠️  Warning: No training history found, using checkpoint data only")
        metrics = {
            'final_epoch': checkpoint['epoch'],
            'best_val_loss': checkpoint.get('best_val_loss', None)
        }
    else:
        metrics_data = history['metrics']
        final_epoch = len(metrics_data['train_loss'])
        
        # Calculate estimated time (12.5s per epoch based on training logs)
        # Try to get from history, otherwise estimate
        total_time = history.get('total_time', final_epoch * 12.5)
        time_per_epoch = total_time / final_epoch if final_epoch > 0 else 12.5
        
        metrics = {
            'final_epoch': final_epoch,
            'train_loss': metrics_data['train_loss'][-1],
            'val_loss': metrics_data['val_loss'][-1],
            'policy_acc': metrics_data['val_policy_acc'][-1],
            'policy_top2_acc': metrics_data.get('val_policy_top2_acc', [None])[-1],
            'value_mae': metrics_data['val_value_mae'][-1],
            'value_corr': metrics_data.get('val_value_corr', [None])[-1],
            'best_val_loss': min(metrics_data['val_loss']),
            'best_epoch': metrics_data['val_loss'].index(min(metrics_data['val_loss'])) + 1,
            'total_time': total_time,
            'time_per_epoch': time_per_epoch,
        }
    
    snapshot = {
        'session_id': session_id,
        'session_name': session_name,
        'timestamp': datetime.now().isoformat(),
        'date_formatted': datetime.now().strftime('%B %d, %Y %H:%M:%S'),
        'config': checkpoint['config'],
        'metrics': metrics,
        'model_info': {
            'parameters': sum(p.numel() for p in checkpoint['model_state_dict'].values()),
            'architecture': checkpoint['config']['model'],
        }
    }
    
    return snapshot


def save_snapshot(snapshot, output_dir):
    """Save snapshot to JSON file"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save individual snapshot
    session_id = snapshot['session_id']
    snapshot_file = output_dir / f'session_{session_id}_snapshot.json'
    
    with open(snapshot_file, 'w') as f:
        json.dump(snapshot, f, indent=2)
    
    print(f"📸 Snapshot saved: {snapshot_file}")
    
    # Update master timeline
    update_timeline(snapshot, output_dir)
    
    return snapshot_file


def update_timeline(snapshot, output_dir):
    """Update the master timeline with new snapshot"""
    timeline_file = output_dir / 'training_timeline.json'
    
    if timeline_file.exists():
        with open(timeline_file, 'r') as f:
            timeline = json.load(f)
    else:
        timeline = {
            'sessions': [],
            'last_updated': None
        }
    
    # Add or update session
    session_id = snapshot['session_id']
    existing_idx = None
    for idx, session in enumerate(timeline['sessions']):
        if session['session_id'] == session_id:
            existing_idx = idx
            break
    
    if existing_idx is not None:
        timeline['sessions'][existing_idx] = snapshot
        print(f"📝 Updated existing session {session_id} in timeline")
    else:
        timeline['sessions'].append(snapshot)
        print(f"➕ Added new session {session_id} to timeline")
    
    timeline['last_updated'] = datetime.now().isoformat()
    
    with open(timeline_file, 'w') as f:
        json.dump(timeline, f, indent=2)
    
    print(f"📊 Timeline updated: {timeline_file}")


def generate_comparison_report(output_dir):
    """Generate a comparison report from timeline"""
    timeline_file = output_dir / 'training_timeline.json'
    
    if not timeline_file.exists():
        return
    
    with open(timeline_file, 'r') as f:
        timeline = json.load(f)
    
    if len(timeline['sessions']) < 2:
        print("\n📝 Not enough sessions for comparison (need at least 2)")
        return
    
    sessions = sorted(timeline['sessions'], key=lambda x: x['session_id'])
    
    report = []
    report.append("=" * 80)
    report.append("V7P3R AI v5.0 - Training Progression Report")
    report.append("=" * 80)
    report.append("")
    
    # Policy Accuracy Trend
    report.append("📈 Policy Accuracy Progression:")
    for session in sessions:
        acc = session['metrics']['policy_acc'] * 100
        epochs = session['metrics']['final_epoch']
        report.append(f"  Session {session['session_id']} ({epochs:3d} epochs): {acc:6.2f}%")
    
    report.append("")
    
    # Value MAE Trend
    report.append("📉 Value MAE Progression:")
    for session in sessions:
        mae = session['metrics']['value_mae']
        epochs = session['metrics']['final_epoch']
        report.append(f"  Session {session['session_id']} ({epochs:3d} epochs): {mae:.4f}")
    
    report.append("")
    
    # Training Efficiency
    report.append("⚡ Training Efficiency:")
    for session in sessions:
        time_per_epoch = session['metrics']['time_per_epoch']
        epochs = session['metrics']['final_epoch']
        total_time = session['metrics']['total_time']
        
        if total_time < 60:
            time_str = f"{total_time:.1f}s"
        elif total_time < 3600:
            time_str = f"{total_time/60:.1f}m"
        else:
            time_str = f"{total_time/3600:.2f}h"
        
        report.append(f"  Session {session['session_id']} ({epochs:3d} epochs): {time_per_epoch:.1f}s/epoch (total: {time_str})")
    
    report.append("")
    report.append("=" * 80)
    
    report_text = "\n".join(report)
    print("\n" + report_text)
    
    # Save report
    report_file = output_dir / 'progression_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\n📄 Report saved: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='Snapshot training metrics for historical tracking')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--session-name', type=str, required=True,
                        help='Name for this training session')
    parser.add_argument('--session-id', type=int, default=None,
                        help='Session ID (auto-increments if not provided)')
    parser.add_argument('--output-dir', type=str, default='metrics_snapshots',
                        help='Directory to save snapshots')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("V7P3R AI v5.0 - Metrics Snapshot Tool")
    print("=" * 80)
    
    # Load training data
    print(f"\n📂 Loading checkpoint: {args.checkpoint}")
    checkpoint, history = load_training_data(args.checkpoint)
    
    # Determine session ID
    output_dir = Path(args.output_dir)
    if args.session_id is None:
        timeline_file = output_dir / 'training_timeline.json'
        if timeline_file.exists():
            with open(timeline_file, 'r') as f:
                timeline = json.load(f)
            session_id = max([s['session_id'] for s in timeline['sessions']]) + 1
        else:
            session_id = 1
    else:
        session_id = args.session_id
    
    print(f"🔢 Session ID: {session_id}")
    print(f"📝 Session Name: {args.session_name}")
    
    # Create snapshot
    print("\n📸 Creating metrics snapshot...")
    snapshot = create_snapshot(checkpoint, history, args.session_name, session_id)
    
    # Save snapshot
    print(f"\n💾 Saving snapshot to: {args.output_dir}")
    save_snapshot(snapshot, args.output_dir)
    
    # Generate comparison report
    print("\n📊 Generating comparison report...")
    generate_comparison_report(output_dir)
    
    # Display summary
    print("\n" + "=" * 80)
    print("✅ Snapshot Complete!")
    print("=" * 80)
    print(f"\nSession {session_id} Metrics:")
    print(f"  Final Epoch:      {snapshot['metrics']['final_epoch']}")
    print(f"  Policy Accuracy:  {snapshot['metrics']['policy_acc']*100:.2f}%")
    print(f"  Value MAE:        {snapshot['metrics']['value_mae']:.4f}")
    print(f"  Best Val Loss:    {snapshot['metrics']['best_val_loss']:.4f} (epoch {snapshot['metrics']['best_epoch']})")
    print(f"  Total Time:       {snapshot['metrics']['total_time']:.1f}s")
    
    print(f"\n📁 Files created:")
    print(f"  - {output_dir}/session_{session_id}_snapshot.json")
    print(f"  - {output_dir}/training_timeline.json")
    print(f"  - {output_dir}/progression_report.txt")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
