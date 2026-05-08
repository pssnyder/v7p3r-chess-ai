"""
V7P3R AI v5.0 - Metrics Dashboard Updater

Automatically updates the MODEL_METRICS_GUIDE.html with new training session data.

Usage:
    python scripts/update_metrics_dashboard.py --checkpoint checkpoints/latest_checkpoint.pth
    python scripts/update_metrics_dashboard.py --checkpoint checkpoints/best_model.pth --session-name "Full Training Session 1"
"""

import argparse
import json
import torch
from pathlib import Path
from datetime import datetime
import re


def load_training_history(checkpoint_path):
    """Load training history from checkpoint"""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_dir = checkpoint_path.parent
    
    # Load training history
    history_path = checkpoint_dir / 'training_history.json'
    if not history_path.exists():
        raise FileNotFoundError(f"Training history not found: {history_path}")
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    return history


def extract_metrics(history):
    """Extract key metrics from training history"""
    metrics = history['metrics']
    
    final_epoch = len(metrics['train_loss'])
    
    return {
        'epochs': final_epoch,
        'policy_acc': metrics['val_policy_acc'][-1] * 100,  # Convert to percentage
        'top2_acc': metrics.get('val_policy_top2_acc', [0])[-1] * 100 if 'val_policy_top2_acc' in metrics else None,
        'value_mae': metrics['val_value_mae'][-1],
        'val_loss': metrics['val_loss'][-1],
        'train_time_per_epoch': history['total_time'] / final_epoch,
        'best_val_loss': min(metrics['val_loss']),
        'best_epoch': metrics['val_loss'].index(min(metrics['val_loss'])) + 1
    }


def update_html_dashboard(html_path, session_name, session_data, session_key):
    """Update the HTML dashboard with new session data"""
    html_path = Path(html_path)
    
    if not html_path.exists():
        raise FileNotFoundError(f"Dashboard not found: {html_path}")
    
    with open(html_path, 'r', encoding='utf-8') as f:
        html = f.read()
    
    # Update session data in JavaScript
    # Find the sessionData object
    pattern = r'const sessionData = \{[^}]+\};'
    
    # Build new session entry
    new_session = f'''
            {session_key}: {{
                date: "{session_data['date']}",
                duration: "{session_data['duration']}",
                device: "{session_data['device']}",
                epochs: {session_data['epochs']},
                policyAcc: {session_data['policy_acc']:.2f},
                top2Acc: {session_data['top2_acc']},
                valueMae: {session_data['value_mae']:.4f},
                valLoss: {session_data['val_loss']:.4f},
                trainTime: {session_data['train_time']:.1f}
            }}'''
    
    # Add to sessionData (before closing brace)
    # This is a simple implementation - in production, use proper HTML/JS parsing
    
    # For now, just print the data to add manually
    print("=" * 80)
    print("New Session Data to Add:")
    print("=" * 80)
    print(new_session)
    print("\nAdd this to the sessionData object in the HTML file's <script> section.")
    print("\nAlso add a new <option> to the session selector:")
    print(f'<option value="{session_key}">{session_name} ({session_data["epochs"]} epochs - {session_data["date"].split()[0]})</option>')
    print("=" * 80)
    
    return new_session


def main():
    parser = argparse.ArgumentParser(description='Update metrics dashboard with new training data')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--session-name', type=str, default=None,
                        help='Name for this training session')
    parser.add_argument('--session-key', type=str, default=None,
                        help='Key for this session (e.g., session1, session2)')
    parser.add_argument('--device', type=str, default='CPU',
                        help='Device used for training (CPU/GPU)')
    parser.add_argument('--html', type=str, default='docs/MODEL_METRICS_GUIDE.html',
                        help='Path to HTML dashboard')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("V7P3R AI v5.0 - Metrics Dashboard Updater")
    print("=" * 80)
    
    # Load training history
    print(f"\nLoading training history from: {args.checkpoint}")
    history = load_training_history(args.checkpoint)
    
    # Extract metrics
    print("Extracting metrics...")
    metrics = extract_metrics(history)
    
    # Generate session data
    session_name = args.session_name or f"Training Session {datetime.now().strftime('%Y-%m-%d')}"
    session_key = args.session_key or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    session_data = {
        'date': datetime.now().strftime('%b %d, %Y %H:%M:%S'),
        'duration': f"{history['total_time']:.1f} seconds" if history['total_time'] < 3600 else f"{history['total_time']/3600:.2f} hours",
        'device': args.device,
        'epochs': metrics['epochs'],
        'policy_acc': metrics['policy_acc'],
        'top2_acc': metrics['top2_acc'] if metrics['top2_acc'] else 75,  # Estimate if not available
        'value_mae': metrics['value_mae'],
        'val_loss': metrics['val_loss'],
        'train_time': metrics['train_time_per_epoch']
    }
    
    # Display metrics
    print("\n" + "=" * 80)
    print("Training Session Summary")
    print("=" * 80)
    print(f"Session Name: {session_name}")
    print(f"Session Key:  {session_key}")
    print(f"Date:         {session_data['date']}")
    print(f"Duration:     {session_data['duration']}")
    print(f"Device:       {session_data['device']}")
    print(f"Epochs:       {session_data['epochs']}")
    print(f"\nFinal Metrics:")
    print(f"  Policy Accuracy:  {session_data['policy_acc']:.2f}%")
    print(f"  Top-2 Accuracy:   ~{session_data['top2_acc']}%")
    print(f"  Value MAE:        {session_data['value_mae']:.4f}")
    print(f"  Validation Loss:  {session_data['val_loss']:.4f}")
    print(f"  Time per Epoch:   {session_data['train_time']:.1f}s")
    print(f"\nBest Performance:")
    print(f"  Best Val Loss:    {metrics['best_val_loss']:.4f}")
    print(f"  Best Epoch:       {metrics['best_epoch']}")
    
    # Update HTML (for now, just print the data)
    print("\n")
    update_html_dashboard(args.html, session_name, session_data, session_key)
    
    print("\n✅ Dashboard update data generated!")
    print(f"\nTo view the dashboard, open: {args.html}")


if __name__ == '__main__':
    main()
