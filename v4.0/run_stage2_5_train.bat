@echo off
REM Stage 2.5: Combined Training (Puzzles + Games)

echo ========================================
echo V7P3RAI Stage 2.5: Combined Training
echo Continue from Stage 1 (86.6%% top-5)
echo ========================================
echo.
echo Dataset:
echo   - 100K puzzles (Stage 1)
echo   - Game positions (Stage 2.5)
echo.
echo Goal:
echo   - Maintain 85%%+ top-5 on puzzles
echo   - Learn game patterns from opponents
echo   - Outperform V7P3R v18.4
echo.

cd /d "e:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v4.0"

python scripts/train_combined_dataset.py ^
    --stage1-checkpoint models/stage1_themes/best_checkpoint.pt ^
    --puzzle-data data/preprocessed_puzzles/enriched_puzzles_compact_20260420_003909.json ^
    --game-data data/stage2_games/historical_positions_1000games.json ^
    --batch-size 32 ^
    --num-epochs 50 ^
    --learning-rate 5e-5 ^
    --patience 15 ^
    --device cpu

echo.
echo ========================================
echo Training complete!
echo.
echo Best model: models/stage2_combined/best_checkpoint.pt
echo.
echo Next step: Test vs V7P3R baseline
echo ========================================
pause
