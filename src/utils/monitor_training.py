#!/usr/bin/env python
# monitor_training.py
"""
Training Progress Monitor
This script checks on the status of training and runs validation when complete.
It's designed to be run in a separate terminal while training is happening.
"""

import os
import sys
import time
import subprocess
import json
from datetime import datetime

def get_training_status():
    """Check if training is still running"""
    try:
        # Check for processes running train_v7p3r.py
        result = subprocess.run('powershell -Command "Get-Process -Name python | Where-Object { $_.CommandLine -like \'*train_v7p3r.py*\' }"', 
                               shell=True, capture_output=True, text=True)
        return "train_v7p3r.py" in result.stdout
    except Exception as e:
        print(f"Error checking training status: {e}")
        return False

def check_model_timestamps():
    """Check model file timestamps to see if training is active"""
    model_path = "models/v7p3r_model.pkl"
    if not os.path.exists(model_path):
        return False
    
    # Get model file timestamp
    model_timestamp = os.path.getmtime(model_path)
    current_time = time.time()
    
    # If model was updated in the last 5 minutes, consider training active
    return (current_time - model_timestamp) < 300  # 5 minutes in seconds

def run_extended_validation():
    """Run extended validation when training is complete"""
    print("\n=== Training Complete - Running Extended Validation ===\n")
    
    try:
        subprocess.run("python extended_validation.py", shell=True)
        print("\n=== Extended Validation Complete ===\n")
    except Exception as e:
        print(f"Error running extended validation: {e}")

def main():
    """Main function"""
    print("=== V7P3R Training Monitor ===")
    print("Monitoring training progress...\n")
    
    training_active = True
    last_check_time = time.time()
    consecutive_inactive_checks = 0
    check_interval = 30  # seconds
    
    try:
        while training_active:
            current_time = time.time()
            
            # Check status every check_interval seconds
            if current_time - last_check_time >= check_interval:
                process_running = get_training_status()
                model_updated = check_model_timestamps()
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                if process_running or model_updated:
                    consecutive_inactive_checks = 0
                    print(f"[{timestamp}] Training active...")
                else:
                    consecutive_inactive_checks += 1
                    print(f"[{timestamp}] No active training detected ({consecutive_inactive_checks}/3)...")
                
                # If we've seen no activity for 3 consecutive checks, consider training complete
                if consecutive_inactive_checks >= 3:
                    training_active = False
                
                last_check_time = current_time
            
            # Sleep to prevent high CPU usage
            time.sleep(1)
        
        # Once training is complete, run validation
        run_extended_validation()
        
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    except Exception as e:
        print(f"Error monitoring training: {e}")

if __name__ == "__main__":
    main()
