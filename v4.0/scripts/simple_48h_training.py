#!/usr/bin/env python3
"""
Simplified 48-hour training script for Docker container.

Uses existing extracted game data and puzzle database.
Single continuous training session with checkpointing.
"""

import subprocess
import sys
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import signal
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/simple_training.log')
    ]
)
logger = logging.getLogger(__name__)

class SimpleTrainingRunner:
    def __init__(self, duration_hours: int = 48, auto_recover: bool = True):
        self.duration_hours = duration_hours
        self.auto_recover = auto_recover
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(hours=duration_hours)
        self.shutdown_requested = False
        self.state_file = Path('checkpoints/training_state.json')
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        
        # Create directories
        for dir_name in ['logs', 'checkpoints', 'models/continuous', 'tensorboard_logs']:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
    
    def load_state(self):
        """Load training state."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                logger.info(f"Resuming from checkpoint: {state.get('last_checkpoint', 'N/A')}")
                return state
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
        
        return {
            'start_timestamp': self.start_time.isoformat(),
            'last_checkpoint': None,
            'epochs_completed': 0,
            'training_runs': 0
        }
    
    def save_state(self, state):
        """Save training state."""
        try:
            state['last_update'] = datetime.now().isoformat()
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def run_training_session(self, epochs: int, checkpoint_path: str = None) -> bool:
        """Run a training session with automatic recovery."""
        max_retries = 3
        retry_delay = 60  # seconds
        
        for attempt in range(1, max_retries + 1):
            if self.shutdown_requested:
                logger.info("Shutdown requested, aborting training session")
                return False
            
            try:
                logger.info(f"Starting training session (attempt {attempt}/{max_retries})")
                
                # Build command
                cmd = [
                    'python', 'scripts/train_move_ordering.py',
                    '--data-path', 'data/preprocessed_puzzles/checkpoint_100000_100000.json',
                    '--num-epochs', str(epochs),
                    '--checkpoint-dir', 'checkpoints',
                    '--batch-size', '64',
                    '--learning-rate', '0.001',
                    '--num-workers', '4',
                    '--no-amp'  # Disable autocast to avoid binary_cross_entropy error
                ]
                
                if checkpoint_path and Path(checkpoint_path).exists():
                    # Note: train_move_ordering.py may use different flag for resume
                    logger.info(f"Resuming from: {checkpoint_path}")
                
                logger.info(f"Running: {' '.join(cmd)}")
                
                # Run training
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                
                if result.returncode == 0:
                    logger.info("✅ Training session completed successfully")
                    return True
                else:
                    logger.error(f"Training failed with code {result.returncode}")
                    logger.error(f"Output: {result.stdout[-1000:]}")  # Last 1000 chars
                    
                    if attempt < max_retries:
                        logger.info(f"Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    
            except Exception as e:
                logger.error(f"Training session error: {e}")
                if attempt < max_retries:
                    time.sleep(retry_delay)
                    retry_delay *= 2
        
        logger.error("❌ Training session failed after all retries")
        return False
    
    def run(self):
        """Run continuous training for specified duration."""
        logger.info("=" * 80)
        logger.info("🚀 V7P3R AI Simple 48-Hour Training")
        logger.info("=" * 80)
        logger.info(f"Start time: {self.start_time}")
        logger.info(f"Target duration: {self.duration_hours} hours")
        logger.info(f"Auto-recover: {self.auto_recover}")
        logger.info("=" * 80)
        
        state = self.load_state()
        total_epochs_target = 200  # Total epochs over 48 hours
        epochs_per_session = 25   # ~6 hours per session
        
        while datetime.now() < self.end_time and not self.shutdown_requested:
            remaining_time = (self.end_time - datetime.now()).total_seconds() / 3600
            logger.info(f"⏱️  Remaining time: {remaining_time:.1f} hours")
            
            # Find latest checkpoint
            checkpoint_dir = Path('checkpoints')
            checkpoints = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
            latest_checkpoint = str(checkpoints[-1]) if checkpoints else None
            
            # Run training session
            success = self.run_training_session(
                epochs=epochs_per_session,
                checkpoint_path=latest_checkpoint
            )
            
            if success:
                state['training_runs'] += 1
                state['epochs_completed'] += epochs_per_session
                state['last_checkpoint'] = latest_checkpoint
                self.save_state(state)
                
                logger.info(f"Progress: {state['epochs_completed']}/{total_epochs_target} epochs")
                
                # Check if we've reached target
                if state['epochs_completed'] >= total_epochs_target:
                    logger.info("✅ Training target reached!")
                    break
            else:
                if not self.auto_recover:
                    logger.error("Auto-recovery disabled, stopping")
                    break
                logger.warning("Training failed, but auto-recovery enabled. Waiting 5min...")
                time.sleep(300)  # Wait 5 minutes before retry
        
        # Summary
        logger.info("=" * 80)
        logger.info("📋 Training Summary")
        logger.info("=" * 80)
        elapsed = datetime.now() - self.start_time
        logger.info(f"Total time: {elapsed}")
        logger.info(f"Training runs: {state.get('training_runs', 0)}")
        logger.info(f"Epochs completed: {state.get('epochs_completed', 0)}/{total_epochs_target}")
        logger.info("=" * 80)
        
        if state.get('epochs_completed', 0) >= total_epochs_target:
            logger.info("✅ Training completed successfully")
            return 0
        else:
            logger.warning("⚠️  Training incomplete")
            return 1

def main():
    parser = argparse.ArgumentParser(description='Simple 48-hour training for Docker')
    parser.add_argument('--duration-hours', type=int, default=48,
                        help='Training duration in hours (default: 48)')
    parser.add_argument('--auto-recover', action='store_true', default=True,
                        help='Enable automatic recovery from failures')
    
    args = parser.parse_args()
    
    runner = SimpleTrainingRunner(
        duration_hours=args.duration_hours,
        auto_recover=args.auto_recover
    )
    
    return runner.run()

if __name__ == '__main__':
    sys.exit(main())
