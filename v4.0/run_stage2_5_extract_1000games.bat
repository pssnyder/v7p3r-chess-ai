@echo off
REM Stage 2.5: Extract 1000 Games (Reasonable Dataset)

echo ========================================
echo V7P3RAI Stage 2.5: Extract 1000 Games
echo Reasonable Dataset for Training
echo ========================================
echo.
echo This will extract positions from:
echo   - First 1000 V7P3R games
echo   - Estimated positions: ~70,000
echo   - Stockfish analysis: 0.5s per position
echo   - Top-5 moves per position
echo   - Estimated time: ~10 hours
echo.
echo Press Ctrl+C to cancel or
pause

cd /d "e:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v4.0"

python scripts/extract_game_positions.py ^
    --pgn-file "E:/Programming Stuff/Chess Engines/Chess Engine Playground/engine-metrics/raw_data/game_records/Lichess V7P3R Bot/lichess_v7p3r_bot_2026-04-09.pgn" ^
    --output data/stage2_games/historical_positions_1000games.json ^
    --analysis-time 0.5 ^
    --num-top-moves 5 ^
    --max-games 1000

echo.
echo ========================================
echo Extraction complete!
echo.
echo Next step: Train combined model
echo   run_stage2_5_train.bat
echo ========================================
pause
