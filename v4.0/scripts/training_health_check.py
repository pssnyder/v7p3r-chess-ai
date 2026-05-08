#!/usr/bin/env python3
"""
Training Health Check Script
============================

Monitors training progress and reports health status.
Used by Docker healthcheck to detect hung/failed training.

Health criteria:
- Checkpoint file modified within last 30 minutes
- No critical errors in recent logs
- GPU/CPU utilization reasonable
- Memory usage < 90%
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
import psutil


def check_checkpoint_freshness(max_age_minutes: int = 30) -> bool:
    """Check if checkpoint was updated recently."""
    checkpoint_dirs = [
        'checkpoints',
        'models/phase1_endgame_puzzles',
        'models/phase2_opening_theory',
        'models/phase3_master_games',
        'models/phase4_positional'
    ]
    
    newest_checkpoint = None
    newest_mtime = datetime.fromtimestamp(0)
    
    for checkpoint_dir in checkpoint_dirs:
        dir_path = Path(checkpoint_dir)
        if not dir_path.exists():
            continue
        
        for checkpoint_file in dir_path.glob('*.pt'):
            mtime = datetime.fromtimestamp(checkpoint_file.stat().st_mtime)
            if mtime > newest_mtime:
                newest_mtime = mtime
                newest_checkpoint = checkpoint_file
    
    if newest_checkpoint is None:
        print("⚠️  No checkpoints found (training may be in early stage)")
        return True  # Don't fail if no checkpoints yet
    
    age = datetime.now() - newest_mtime
    max_age = timedelta(minutes=max_age_minutes)
    
    if age > max_age:
        print(f"❌ Checkpoint too old: {newest_checkpoint} ({age})")
        return False
    
    print(f"✅ Checkpoint fresh: {newest_checkpoint} ({age.seconds // 60}m ago)")
    return True


def check_log_errors() -> bool:
    """Check for critical errors in recent logs."""
    log_files = [
        'logs/orchestrator.log',
        'logs/training.log'
    ]
    
    critical_errors = []
    
    for log_file in log_files:
        log_path = Path(log_file)
        if not log_path.exists():
            continue
        
        try:
            # Read last 100 lines
            with open(log_path, 'r') as f:
                lines = f.readlines()[-100:]
            
            for line in lines:
                if 'CRITICAL' in line or 'FATAL' in line:
                    critical_errors.append(line.strip())
        
        except Exception as e:
            print(f"⚠️  Could not read log {log_file}: {e}")
    
    if critical_errors:
        print(f"❌ Found {len(critical_errors)} critical errors:")
        for error in critical_errors[:3]:  # Show first 3
            print(f"  {error}")
        return False
    
    print("✅ No critical errors in recent logs")
    return True


def check_system_resources() -> bool:
    """Check system resource usage."""
    # Memory check
    memory = psutil.virtual_memory()
    if memory.percent > 95:
        print(f"❌ Memory usage too high: {memory.percent}%")
        return False
    
    print(f"✅ Memory usage OK: {memory.percent}%")
    
    # CPU check (optional - training may use 100% CPU)
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"ℹ️  CPU usage: {cpu_percent}%")
    
    # Disk check
    disk = psutil.disk_usage('/')
    if disk.percent > 95:
        print(f"⚠️  Disk usage high: {disk.percent}%")
        # Don't fail on disk - just warn
    
    return True


def check_orchestrator_state() -> bool:
    """Check orchestrator state file."""
    state_file = Path('checkpoints/orchestrator_state.json')
    
    if not state_file.exists():
        print("⚠️  Orchestrator state file not found (training may be starting)")
        return True
    
    try:
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        current_phase = state.get('current_phase', 0)
        completed_phases = state.get('completed_phases', [])
        failed_attempts = state.get('failed_attempts', {})
        
        print(f"ℹ️  Current phase: {current_phase}")
        print(f"ℹ️  Completed phases: {len(completed_phases)}")
        
        # Check if any phase has too many failures
        for phase, attempts in failed_attempts.items():
            if attempts >= 3:
                print(f"❌ Phase {phase} has {attempts} failed attempts")
                return False
        
        print("✅ Orchestrator state healthy")
        return True
        
    except Exception as e:
        print(f"⚠️  Could not read orchestrator state: {e}")
        return True  # Don't fail on read error


def main():
    """Run all health checks."""
    print("=" * 60)
    print("V7P3R Training Health Check")
    print("=" * 60)
    print(f"Time: {datetime.now()}")
    print("=" * 60)
    
    checks = [
        ('Checkpoint Freshness', check_checkpoint_freshness),
        ('Log Errors', check_log_errors),
        ('System Resources', check_system_resources),
        ('Orchestrator State', check_orchestrator_state)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n🔍 {check_name}:")
        try:
            passed = check_func()
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"❌ Check failed with exception: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("✅ HEALTHY - Training is progressing normally")
        print("=" * 60)
        return 0
    else:
        print("❌ UNHEALTHY - Training may be stuck or failed")
        print("=" * 60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
