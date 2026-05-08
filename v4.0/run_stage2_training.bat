@echo off
REM Stage 2: Corrective Training Launch Script
REM 
REM Fine-tunes Stage 1 model with historical failure correction
REM Uses dual-learning pattern: avoid V7P3R's mistakes + exploit opponent patterns

echo ========================================
echo V7P3RAI Stage 2: Corrective Training
echo ========================================
echo.

cd /d "e:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v4.0"

python scripts/train_corrective.py ^
    --data-path data/stage2_training/corrective_dataset.json ^
    --stage1-model models/stage1_themes/best_checkpoint.pt ^
    --batch-size 32 ^
    --num-epochs 50 ^
    --learning-rate 5e-5 ^
    --correction-weight 2.0 ^
    --ranking-weight 1.0 ^
    --early-stopping-patience 15 ^
    --val-split 0.1 ^
    --device cpu

echo.
echo ========================================
echo Training Complete!
echo ========================================
echo.
echo Check models/stage2_corrective/ for trained models
echo - best_model.pt: Best performing model
echo - latest_model.pt: Most recent checkpoint
echo.

pause
