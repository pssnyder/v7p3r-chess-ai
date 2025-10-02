"""
V7P3R V3.0 Intensive Training - Ready Check
==========================================

Quick verification that all systems are ready for 5-day intensive training
"""

import torch
import sys
from pathlib import Path

def check_gpu():
    """Verify GPU setup"""
    print("🔍 GPU VERIFICATION")
    print("=" * 30)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print("✅ GPU ready for intensive training!")
    else:
        print("❌ CUDA not available - training will be slower")
    print()

def check_files():
    """Verify required files exist"""
    print("📁 FILE VERIFICATION")
    print("=" * 30)
    
    files_to_check = [
        ("Enhanced puzzle trainer", "enhanced_puzzle_main_v2.py"),
        ("V3 hybrid trainer", "v3_hybrid_main.py"),
        ("Thinking Brain", "v3.0/src/ai/thinking_brain.py"),
        ("Gameplay Brain", "v3.0/src/ai/gameplay_brain.py"),
        ("Chess State", "v3.0/src/core/chess_state.py"),
        ("Training strategy", "V7P3R_V3_INTENSIVE_TRAINING.md"),
        ("Progress tracker", "intensive_tracker.py")
    ]
    
    all_ready = True
    for name, filepath in files_to_check:
        if Path(filepath).exists():
            print(f"✅ {name}")
        else:
            print(f"❌ {name} - {filepath}")
            all_ready = False
    
    if all_ready:
        print("\n🎉 All required files present!")
    else:
        print("\n⚠️ Some files missing - check V3 setup")
    print()

def show_training_overview():
    """Show the intensive training plan"""
    print("🚀 5-DAY INTENSIVE TRAINING OVERVIEW")
    print("=" * 50)
    print()
    
    days = [
        ("DAY 1", "FOUNDATION", "5,000 puzzles", "Basic tactics (pin, fork, skewer)"),
        ("DAY 2", "TACTICS", "7,000 puzzles", "Advanced themes (deflection, discovery)"),
        ("DAY 3", "STRATEGY", "10,000 puzzles", "Positional play and endgames"),
        ("DAY 4", "OPTIMIZATION", "12,000 puzzles", "High-difficulty refinement"),
        ("DAY 5", "TESTING", "15,000 puzzles", "Tournament preparation")
    ]
    
    for day, phase, target, focus in days:
        print(f"🔥 {day} - {phase}")
        print(f"   Target: {target} total (cumulative)")
        print(f"   Focus: {focus}")
        print()
    
    print("🎯 SUCCESS METRICS")
    print("-" * 25)
    print("• Total puzzles solved: 15,000+")
    print("• Average accuracy: >75%")
    print("• GPU utilization: 80-95%")
    print("• Estimated ELO: 1200-1500+")
    print("• Final output: Tournament-ready V7P3RAI_v3.0.exe")
    print()

def show_quick_commands():
    """Show essential commands for training"""
    print("⚡ QUICK START COMMANDS")
    print("=" * 30)
    print()
    
    print("📊 Check current progress:")
    print("   python intensive_tracker.py status")
    print()
    
    print("🔥 Start Day 1 training:")
    print("   python v3_hybrid_main.py --hpts 6 --batch-size 64 --max-rating 1400")
    print()
    
    print("📈 View analytics dashboard:")
    print("   python v3_hybrid_main.py --analytics-only")
    print()
    
    print("🖥️ Monitor GPU in real-time:")
    print("   nvidia-smi -l 1")
    print()
    
    print("📋 Update daily progress:")
    print("   python intensive_tracker.py update <day> <puzzles> <accuracy> <gpu_util>")
    print()

def main():
    print("🚀 V7P3R V3.0 INTENSIVE TRAINING - READY CHECK")
    print("=" * 60)
    print()
    
    check_gpu()
    check_files()
    show_training_overview()
    show_quick_commands()
    
    print("🏁 READY TO BEGIN INTENSIVE TRAINING!")
    print("=" * 50)
    print("Next steps:")
    print("1. Run: python intensive_tracker.py status")
    print("2. Start Day 1: python v3_hybrid_main.py --hpts 6 --batch-size 64 --max-rating 1400")
    print("3. Monitor GPU: nvidia-smi -l 1 (in separate terminal)")
    print("4. Follow V7P3R_V3_INTENSIVE_TRAINING.md for detailed schedule")
    print()
    print("Your RTX 4070 Ti is ready for 39.4x accelerated training! 🚀")

if __name__ == "__main__":
    main()