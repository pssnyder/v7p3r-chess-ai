@echo off
REM Stage 2: OPTIMIZED Corrective Training Launch Script
REM 
REM Production-grade configuration for potential V7P3R primary engine deployment
REM 
REM Key Enhancements:
REM - Warmup + cosine annealing LR schedule for better convergence
REM - Gradient accumulation (effective batch 128 = 32 x 4)
REM - Exponential Moving Average (EMA) for stable predictions
REM - Blunder-focused loss weighting (5x penalty on historical mistakes)
REM - Margin-based ranking loss for clearer move separation
REM - Label smoothing for better generalization
REM - Multi-metric early stopping (blunder avoidance + val loss)
REM - Comprehensive validation suite (top-1/3/5/10, avg rank, blunder avoidance)

echo ========================================
echo V7P3RAI Stage 2: OPTIMIZED Training
echo Production-Grade Configuration
echo ========================================
echo.
echo Target: Potential V7P3R Primary Engine
echo Expected: 90%% blunder avoidance, 85%%+ top-5 accuracy
echo.

cd /d "e:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v4.0"

python scripts/train_corrective_optimized.py ^
    --data-path data/stage2_training/corrective_dataset.json ^
    --stage1-model models/stage1_themes/best_checkpoint.pt ^
    --batch-size 32 ^
    --num-epochs 100 ^
    --learning-rate 1e-4 ^
    --correction-weight 3.0 ^
    --ranking-weight 1.0 ^
    --blunder-weight 5.0 ^
    --gradient-accumulation-steps 4 ^
    --early-stopping-patience 20 ^
    --val-split 0.1 ^
    --use-ema ^
    --label-smoothing 0.1 ^
    --warmup-ratio 0.1 ^
    --device cpu

echo.
echo ========================================
echo Training Complete!
echo ========================================
echo.
echo Check models/stage2_corrective_optimized/ for results:
echo - best_model.pt: Best blunder avoidance + val loss
echo - latest_model.pt: Most recent checkpoint
echo - ema_epoch_*.pt: EMA checkpoints every 5 epochs
echo.
echo Next Steps:
echo 1. Review training metrics in history
echo 2. Test against V7P3R baseline (50 games)
echo 3. Validate blunder avoidance on held-out positions
echo 4. If performance warrants, integrate as primary engine
echo.

pause
