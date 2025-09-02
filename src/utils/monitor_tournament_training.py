#!/usr/bin/env python3
"""
Monitor Tournament Training Progress
Real-time monitoring of the extended training session
"""

import os
import time
import glob
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

def find_latest_training_report():
    """Find the most recent training report"""
    reports = glob.glob("reports/training_summary_*.md")
    if not reports:
        return None
    return max(reports, key=os.path.getctime)

def parse_training_report(report_path):
    """Parse training report for key metrics"""
    if not os.path.exists(report_path):
        return None
    
    with open(report_path, 'r') as f:
        content = f.read()
    
    # Extract key information
    info = {}
    lines = content.split('\n')
    
    for line in lines:
        if "Final Best Fitness:" in line:
            try:
                info['best_fitness'] = float(line.split(':')[1].strip())
            except:
                pass
        elif "Total Generations:" in line:
            try:
                info['generations'] = int(line.split(':')[1].strip())
            except:
                pass
        elif "Training Duration:" in line:
            info['duration'] = line.split(':')[1].strip()
        elif "GPU Memory Usage:" in line:
            info['gpu_memory'] = line.split(':')[1].strip()
    
    return info

def get_current_generation():
    """Get the current generation from model files"""
    model_files = glob.glob("models/best_gpu_model_gen_*.pth")
    if not model_files:
        return 0
    
    def extract_gen(filepath):
        try:
            base = os.path.basename(filepath)
            return int(base.split('gen_')[1].split('.')[0])
        except:
            return 0
    
    return max(extract_gen(f) for f in model_files)

def estimate_completion_time(start_time, current_gen, target_gen):
    """Estimate completion time based on current progress"""
    if current_gen <= 7:  # Haven't made much progress yet
        return "Calculating..."
    
    elapsed = datetime.now() - start_time
    progress = (current_gen - 7) / (target_gen - 7)  # Subtract 7 since we started from gen 7
    
    if progress > 0:
        total_estimated = elapsed / progress
        remaining = total_estimated - elapsed
        completion_time = datetime.now() + remaining
        return completion_time.strftime("%H:%M:%S")
    
    return "Calculating..."

def display_training_status():
    """Display current training status"""
    print("=" * 80)
    print("V7P3R TOURNAMENT TRAINING MONITOR")
    print("=" * 80)
    
    # Check if training is running
    current_gen = get_current_generation()
    start_gen = 7  # We started from generation 7
    target_gen = 225 + start_gen  # Target is 225 additional generations
    
    print(f"📊 Training Progress:")
    print(f"   Current Generation: {current_gen}")
    print(f"   Starting Generation: {start_gen}")
    print(f"   Target Generation: {target_gen}")
    print(f"   Progress: {current_gen - start_gen}/{target_gen - start_gen} ({((current_gen - start_gen) / (target_gen - start_gen) * 100):.1f}%)")
    
    # Estimate completion time (assuming training started around 08:53)
    start_time = datetime.now().replace(hour=8, minute=53, second=0, microsecond=0)
    if datetime.now() < start_time:
        start_time = start_time - timedelta(days=1)
    
    completion_est = estimate_completion_time(start_time, current_gen, target_gen)
    print(f"   Estimated Completion: {completion_est}")
    
    # Check recent models
    print(f"\n📁 Recent Models:")
    model_files = glob.glob("models/best_gpu_model_gen_*.pth")
    model_files.sort(key=lambda x: os.path.getctime(x), reverse=True)
    
    for i, model in enumerate(model_files[:5]):
        gen_num = os.path.basename(model).split('gen_')[1].split('.')[0]
        mod_time = datetime.fromtimestamp(os.path.getctime(model))
        size_mb = os.path.getsize(model) / (1024 * 1024)
        print(f"   Gen {gen_num}: {mod_time.strftime('%H:%M:%S')} ({size_mb:.1f} MB)")
    
    # Check latest training report
    latest_report = find_latest_training_report()
    if latest_report:
        info = parse_training_report(latest_report)
        if info:
            print(f"\n📈 Latest Training Metrics:")
            if 'best_fitness' in info:
                print(f"   Best Fitness: {info['best_fitness']:.3f}")
            if 'duration' in info:
                print(f"   Duration: {info['duration']}")
            if 'gpu_memory' in info:
                print(f"   GPU Memory: {info['gpu_memory']}")
    
    # Tournament readiness status
    print(f"\n🏆 Tournament Readiness:")
    if current_gen >= target_gen * 0.8:  # 80% complete
        print("   ✅ Model should be tournament-ready")
        print("   🚀 Consider packaging the engine soon")
    elif current_gen >= target_gen * 0.5:  # 50% complete
        print("   🟡 Training progressing well")
        print("   ⏳ Continue monitoring for completion")
    else:
        print("   🔄 Training in early stages")
        print("   ⏰ Allow more time for convergence")

def main():
    """Main monitoring loop"""
    try:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            display_training_status()
            
            print(f"\n" + "=" * 80)
            print("Options:")
            print("  [ENTER] - Refresh status")
            print("  'q' - Quit monitor")
            print("  'p' - Package current model (if ready)")
            print("=" * 80)
            
            # Wait for input with timeout
            print("Refreshing in 30 seconds... (press ENTER to refresh now)")
            
            import select
            import sys
            
            # Simple input handling for Windows
            if os.name == 'nt':
                import msvcrt
                start_time = time.time()
                while time.time() - start_time < 30:
                    if msvcrt.kbhit():
                        key = msvcrt.getch().decode('utf-8', errors='ignore')
                        if key == '\r':  # Enter
                            break
                        elif key.lower() == 'q':
                            return
                        elif key.lower() == 'p':
                            print("\\nPackaging current model...")
                            os.system("python package_tournament_engine.py")
                            input("Press ENTER to continue monitoring...")
                            break
                    time.sleep(0.1)
            else:
                # Unix-like system
                time.sleep(30)
    
    except KeyboardInterrupt:
        print("\\nMonitoring stopped by user.")

if __name__ == "__main__":
    main()
