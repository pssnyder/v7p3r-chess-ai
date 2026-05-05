@echo off
REM Stage 2.5: Full Extraction - All V7P3R Historical Games

echo ========================================
echo V7P3RAI Stage 2.5: Full Extraction
echo Extract ALL V7P3R Historical Games
echo ========================================
echo.
echo This will extract positions from:
echo   - All games in lichess_v7p3r_bot_2026-04-09.pgn
echo   - Estimated games: 5,107+
echo   - Estimated positions: ~350,000+
echo   - Stockfish analysis: 0.5s per position
echo   - Top-5 moves per position
echo   - Estimated time: 48+ hours
echo.
echo Press Ctrl+C to cancel or
pause

cd /d "e:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v4.0"

python scripts/extract_game_positions.py ^
    --pgn-file "E:/Programming Stuff/Chess Engines/Chess Engine Playground/engine-metrics/raw_data/game_records/Lichess V7P3R Bot/lichess_v7p3r_bot_2026-04-09.pgn" ^
    --output data/stage2_games/historical_positions_full.json ^
    --analysis-time 0.5 ^
    --num-top-moves 5

echo.
echo ========================================
echo Full extraction complete!
echo.
echo Next step: Train combined model
echo   run_stage2_5_train.bat
echo ========================================
pause
