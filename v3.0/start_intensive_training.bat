@echo off
REM V7P3R V3.0 Intensive Training Setup - Windows Version
REM Quick start script for 5-day GPU-accelerated training

echo 🚀 V7P3R V3.0 INTENSIVE TRAINING SETUP
echo ======================================
echo.

REM Check CUDA availability
echo 🔍 Checking GPU setup...
"C:/Users/patss/AppData/Local/Programs/Python/Python313/python.exe" -c "import torch; print(f'✅ PyTorch {torch.__version__}'); print(f'✅ CUDA available: {torch.cuda.is_available()}'); print(f'✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
echo.

REM Display training schedule
echo 📅 5-DAY INTENSIVE TRAINING SCHEDULE
echo ====================================
echo.
echo 🔥 DAY 1 - FOUNDATION (12-16 hours)
echo    Morning:   python v3_hybrid_main.py --hpts 6 --batch-size 64 --max-rating 1400
echo    Afternoon: python v3_hybrid_main.py --hpts 6 --difficulty-progression
echo    Evening:   python v3_hybrid_main.py --analytics-only
echo.
echo ⚡ DAY 2 - TACTICS (12-16 hours)
echo    Morning:   python v3_hybrid_main.py --hpts 6 --target-themes deflection,xRayAttack
echo    Afternoon: python v3_hybrid_main.py --hpts 6 --brain-coordination
echo    Evening:   python v3_hybrid_main.py --hpts 4 --max-rating 1700
echo.
echo 🧠 DAY 3 - STRATEGY (12-16 hours)
echo    Morning:   python v3_hybrid_main.py --hpts 6 --target-themes advantage,crushing
echo    Afternoon: python v3_hybrid_main.py --hpts 6 --target-themes endgame
echo    Evening:   python v3_hybrid_main.py --hpts 4 --max-rating 1800
echo.
echo ⚙️ DAY 4 - OPTIMIZATION (8-12 hours)
echo    Morning:   python v3_hybrid_main.py --hpts 4 --min-rating 1600
echo    Afternoon: Analytics review and model optimization
echo    Evening:   Target weak themes based on analytics
echo.
echo 🏆 DAY 5 - TESTING (8-12 hours)
echo    Morning:   python v3_hybrid_main.py --num-puzzles 1000
echo    Afternoon: Tournament simulation and time management
echo    Evening:   Export V7P3RAI_v3.0.exe for deployment
echo.

REM GPU monitoring setup
echo 📊 GPU MONITORING COMMANDS
echo ==========================
echo Real-time GPU stats:    nvidia-smi -l 1
echo Training progress:      python v3_hybrid_main.py --monitor-session ^<id^>
echo Analytics dashboard:    python v3_hybrid_main.py --analytics-only
echo.

REM Success metrics
echo 🎯 TARGET METRICS
echo ================
echo Total puzzles:     15,000+ in 5 days
echo GPU utilization:   80-95%% during training
echo Memory usage:      8-10 GB of 12.9 GB VRAM
echo Tactical accuracy: ^>75%% average
echo Estimated ELO:     1200-1500+
echo Final output:      V7P3RAI_v3.0.exe tournament-ready
echo.

echo 🚀 Ready to begin intensive training!
echo Start with: python v3_hybrid_main.py --analytics-only
echo Then follow Day 1 schedule from V7P3R_V3_INTENSIVE_TRAINING.md
echo.
pause