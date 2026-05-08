@echo off
REM Stage 2.5: Quick Test - Extract 10 games to validate pipeline

echo ========================================
echo V7P3RAI Stage 2.5: Quick Test
echo Extract 10 Games for Validation
echo ========================================
echo.

cd /d "e:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v4.0"

python scripts/extract_game_positions.py ^
    --pgn-file "E:/Programming Stuff/Chess Engines/Chess Engine Playground/engine-metrics/raw_data/game_records/Lichess V7P3R Bot/lichess_v7p3r_bot_2026-04-09.pgn" ^
    --output data/stage2_games/test_positions_10games.json ^
    --analysis-time 0.3 ^
    --num-top-moves 5 ^
    --max-games 10

echo.
echo ========================================
echo Test extraction complete!
echo.
echo Expected:
echo   ~700 positions from 10 games
echo   File size: ~2-3 MB
echo   Top-5 moves per position (Stockfish best)
echo.
echo If successful, run 1000-game extraction:
echo   run_stage2_5_extract_1000games.bat
echo ========================================
pause
