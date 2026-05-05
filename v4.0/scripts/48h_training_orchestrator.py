#!/usr/bin/env python3
"""
48-Hour Training Orchestrator for V7P3R AI
==========================================

Manages multi-phase training with auto-recovery, checkpointing, and monitoring.
Designed for unattended weekend training runs.

Phases:
- Phase 1 (12h): Endgame expansion + 500K themed puzzles → v5.0
- Phase 2 (12h): + 100K opening positions → v5.1
- Phase 3 (12h): + 100K master games → v5.2
- Phase 4 (12h): + 50K positional patterns → v5.3

Features:
- Auto-resume from last checkpoint
- Health monitoring
- TensorBoard integration
- Error recovery with exponential backoff
- Graceful shutdown on SIGTERM
"""

import os
import sys
import json
import time
import signal
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/orchestrator.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """Orchestrates multi-phase 48-hour training run."""
    
    def __init__(self, auto_recover: bool = True, enable_tensorboard: bool = True):
        self.auto_recover = auto_recover
        self.enable_tensorboard = enable_tensorboard
        self.state_file = Path('checkpoints/orchestrator_state.json')
        self.start_time = datetime.now()
        self.target_duration = timedelta(hours=48)
        self.shutdown_requested = False
        
        # Phase configurations
        self.phases = [
            {
                'name': 'phase1_endgame_puzzles',
                'version': 'v5.0',
                'duration_hours': 12,
                'data_sources': ['endgame_pgns', 'themed_puzzles_500k'],
                'epochs': 50,
                'description': 'Endgame expansion + themed puzzles'
            },
            {
                'name': 'phase2_opening_theory',
                'version': 'v5.1',
                'duration_hours': 12,
                'data_sources': ['phase1_checkpoint', 'opening_positions_100k'],
                'epochs': 50,
                'description': 'Opening theory integration'
            },
            {
                'name': 'phase3_master_games',
                'version': 'v5.2',
                'duration_hours': 12,
                'data_sources': ['phase2_checkpoint', 'master_games_100k'],
                'epochs': 50,
                'description': 'Master game patterns'
            },
            {
                'name': 'phase4_positional',
                'version': 'v5.3',
                'duration_hours': 12,
                'data_sources': ['phase3_checkpoint', 'positional_patterns_50k'],
                'epochs': 50,
                'description': 'Positional refinement'
            }
        ]
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        
        # Create directories
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
    
    def load_state(self) -> Dict:
        """Load orchestrator state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                logger.info(f"Loaded state: Phase {state.get('current_phase', 0)}")
                return state
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
        
        return {
            'current_phase': 0,
            'completed_phases': [],
            'failed_attempts': {},
            'start_timestamp': datetime.now().isoformat(),
            'last_checkpoint': None
        }
    
    def save_state(self, state: Dict):
        """Save orchestrator state to disk."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            logger.debug("State saved successfully")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def check_data_availability(self, data_sources: List[str]) -> bool:
        """Verify required data sources are available."""
        for source in data_sources:
            # Check if data file exists
            data_path = Path(f'data/{source}.json')
            if not data_path.exists() and 'checkpoint' not in source:
                logger.warning(f"Data source not found: {source}")
                return False
        return True
    
    def run_data_extraction(self, phase: Dict) -> bool:
        """Run data extraction for a phase."""
        logger.info(f"Running data extraction for {phase['name']}...")
        
        # Determine which extraction scripts to run
        scripts = []
        if 'endgame' in phase['name']:
            scripts.append('scripts/extract_endgame_positions.py')
            scripts.append('scripts/extract_themed_puzzles.py')
        elif 'opening' in phase['name']:
            scripts.append('scripts/extract_opening_positions.py')
        elif 'master' in phase['name']:
            scripts.append('scripts/extract_master_games.py')
        elif 'positional' in phase['name']:
            scripts.append('scripts/extract_positional_patterns.py')
        
        # Run each extraction script
        for script in scripts:
            if not Path(script).exists():
                logger.warning(f"Extraction script not found: {script}")
                continue
            
            try:
                logger.info(f"Running {script}...")
                result = subprocess.run(
                    ['python', script, '--auto-mode'],
                    capture_output=True,
                    text=True,
                    timeout=3600  # 1 hour timeout per extraction
                )
                
                if result.returncode != 0:
                    logger.error(f"Extraction failed: {result.stderr}")
                    return False
                    
                logger.info(f"Extraction completed: {script}")
                
            except subprocess.TimeoutExpired:
                logger.error(f"Extraction timed out: {script}")
                return False
            except Exception as e:
                logger.error(f"Extraction error: {e}")
                return False
        
        return True
    
    def run_training_phase(self, phase: Dict, state: Dict) -> bool:
        """Run a single training phase with auto-recovery."""
        phase_name = phase['name']
        logger.info(f"🚀 Starting {phase_name}: {phase['description']}")
        
        # Check data availability
        if not self.check_data_availability(phase['data_sources']):
            logger.info("Required data not available, running extraction...")
            if not self.run_data_extraction(phase):
                logger.error(f"Data extraction failed for {phase_name}")
                return False
        
        # Prepare training command
        checkpoint_dir = Path(f"models/{phase_name}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine data path (merged dataset for this phase)
        data_path = Path(f"data/merged_{phase_name}.json")
        if not data_path.exists():
            # Use phase-specific data or previous checkpoint
            data_path = Path(f"data/training_data_{phase['version']}.json")
        
        training_cmd = [
            'python', 'scripts/train_move_ordering.py',
            '--data-path', str(data_path),
            '--checkpoint-dir', str(checkpoint_dir),
            '--num-epochs', str(phase['epochs']),
            '--batch-size', '128',
            '--learning-rate', '5e-4',
            '--gradient-accumulation', '2',
            '--num-workers', '8'
        ]
        
        # Add previous checkpoint if not first phase
        prev_phase_idx = self.phases.index(phase) - 1
        if prev_phase_idx >= 0:
            prev_checkpoint = Path(f"models/{self.phases[prev_phase_idx]['name']}/best_checkpoint.pt")
            if prev_checkpoint.exists():
                training_cmd.extend(['--resume-from', str(prev_checkpoint)])
        
        # GPU/CPU selection
        if os.environ.get('FORCE_CPU') != 'true':
            training_cmd.append('--use-amp')
        
        # Run training with retry logic
        max_retries = 3
        retry_count = state['failed_attempts'].get(phase_name, 0)
        
        while retry_count < max_retries:
            if self.shutdown_requested:
                logger.info("Shutdown requested, stopping phase execution")
                return False
            
            try:
                logger.info(f"Running training (attempt {retry_count + 1}/{max_retries})...")
                logger.info(f"Command: {' '.join(training_cmd)}")
                
                # Start training process
                process = subprocess.Popen(
                    training_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                # Monitor training output
                phase_start = datetime.now()
                phase_deadline = phase_start + timedelta(hours=phase['duration_hours'])
                
                for line in process.stdout:
                    print(line, end='')  # Echo to console
                    
                    # Check deadline
                    if datetime.now() > phase_deadline:
                        logger.warning(f"Phase time limit reached ({phase['duration_hours']}h)")
                        process.terminate()
                        process.wait(timeout=30)
                        break
                    
                    # Check shutdown request
                    if self.shutdown_requested:
                        logger.info("Shutdown requested, terminating training...")
                        process.terminate()
                        process.wait(timeout=30)
                        return False
                
                # Check exit code
                process.wait()
                if process.returncode == 0:
                    logger.info(f"✅ {phase_name} completed successfully")
                    
                    # Mark phase as completed
                    state['completed_phases'].append(phase_name)
                    state['failed_attempts'][phase_name] = 0
                    state['last_checkpoint'] = str(checkpoint_dir / 'best_checkpoint.pt')
                    self.save_state(state)
                    
                    return True
                else:
                    logger.error(f"Training exited with code {process.returncode}")
                    retry_count += 1
                    
            except Exception as e:
                logger.error(f"Training error: {e}", exc_info=True)
                retry_count += 1
            
            # Exponential backoff before retry
            if retry_count < max_retries:
                wait_time = 2 ** retry_count * 60  # 2min, 4min, 8min
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            
            # Update state
            state['failed_attempts'][phase_name] = retry_count
            self.save_state(state)
        
        logger.error(f"❌ {phase_name} failed after {max_retries} attempts")
        return False
    
    def start_tensorboard(self):
        """Start TensorBoard server in background."""
        if not self.enable_tensorboard:
            return
        
        try:
            tensorboard_cmd = [
                'tensorboard',
                '--logdir', 'tensorboard_logs',
                '--host', '0.0.0.0',
                '--port', '6006',
                '--reload_interval', '30'
            ]
            
            subprocess.Popen(
                tensorboard_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            logger.info("📊 TensorBoard started on http://0.0.0.0:6006")
            
        except Exception as e:
            logger.warning(f"Failed to start TensorBoard: {e}")
    
    def run(self):
        """Run the 48-hour training orchestration."""
        logger.info("=" * 80)
        logger.info("🚀 V7P3R AI 48-Hour Training Orchestrator")
        logger.info("=" * 80)
        logger.info(f"Start time: {self.start_time}")
        logger.info(f"Target duration: {self.target_duration}")
        logger.info(f"Auto-recover: {self.auto_recover}")
        logger.info(f"TensorBoard: {self.enable_tensorboard}")
        logger.info("=" * 80)
        
        # Start TensorBoard
        self.start_tensorboard()
        
        # Load state
        state = self.load_state() if self.auto_recover else {
            'current_phase': 0,
            'completed_phases': [],
            'failed_attempts': {},
            'start_timestamp': datetime.now().isoformat(),
            'last_checkpoint': None
        }
        
        # Run phases sequentially
        for phase_idx, phase in enumerate(self.phases):
            if self.shutdown_requested:
                logger.info("Shutdown requested, exiting orchestrator")
                break
            
            # Skip completed phases
            if phase['name'] in state['completed_phases']:
                logger.info(f"⏭️  Skipping completed phase: {phase['name']}")
                continue
            
            # Update current phase
            state['current_phase'] = phase_idx
            self.save_state(state)
            
            # Run phase
            success = self.run_training_phase(phase, state)
            
            if not success:
                logger.error(f"Phase {phase['name']} failed, stopping orchestration")
                break
            
            # Check overall time limit
            elapsed = datetime.now() - self.start_time
            if elapsed > self.target_duration:
                logger.warning(f"Overall time limit reached ({elapsed})")
                break
        
        # Final summary
        logger.info("=" * 80)
        logger.info("📋 Training Summary")
        logger.info("=" * 80)
        logger.info(f"Completed phases: {len(state['completed_phases'])}/{len(self.phases)}")
        logger.info(f"Total time: {datetime.now() - self.start_time}")
        
        for phase in self.phases:
            status = "✅ DONE" if phase['name'] in state['completed_phases'] else "❌ INCOMPLETE"
            logger.info(f"  {phase['name']} ({phase['version']}): {status}")
        
        logger.info("=" * 80)
        
        if len(state['completed_phases']) == len(self.phases):
            logger.info("🎉 All phases completed successfully!")
            return 0
        else:
            logger.warning("⚠️  Some phases incomplete")
            return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='48-Hour V7P3R AI Training Orchestrator'
    )
    parser.add_argument(
        '--auto-recover',
        action='store_true',
        help='Auto-resume from last checkpoint'
    )
    parser.add_argument(
        '--enable-tensorboard',
        action='store_true',
        help='Start TensorBoard server'
    )
    parser.add_argument(
        '--duration-hours',
        type=int,
        default=48,
        help='Total training duration in hours (default: 48)'
    )
    
    args = parser.parse_args()
    
    orchestrator = TrainingOrchestrator(
        auto_recover=args.auto_recover,
        enable_tensorboard=args.enable_tensorboard
    )
    
    orchestrator.target_duration = timedelta(hours=args.duration_hours)
    
    return orchestrator.run()


if __name__ == '__main__':
    sys.exit(main())
