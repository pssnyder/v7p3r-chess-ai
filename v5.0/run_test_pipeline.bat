@echo off
REM V7P3R AI v5.0 - Training Pipeline TEST RUN
REM ==========================================
REM Quick test with 100 games, minimal features, depth 15
REM Expected time: 5-10 minutes
REM Expected output: ~2,000 positions

echo ================================================
echo V7P3R AI v5.0 - TRAINING PIPELINE TEST RUN
echo ================================================
echo.
echo Configuration:
echo   - Source: Lichess V7P3R Bot PGNs
echo   - Max Games: 100
echo   - Feature Set: minimal
echo   - Stockfish Depth: 15
echo   - Expected Time: 5-10 minutes
echo.
echo ================================================
echo.

cd /d "E:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v5.0"

python scripts/run_training_pipeline.py ^
  --pgn-dir "E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\game_records\Lichess V7P3R Bot" ^
  --output-dir "data/test_run" ^
  --max-games 100 ^
  --feature-set minimal ^
  --stockfish-path "stockfish" ^
  --stockfish-depth 15 ^
  --stockfish-time-limit 5.0

echo.
echo ================================================
echo TEST RUN COMPLETE!
echo ================================================
echo.
echo Output location: data/test_run/stage3_graded/
echo.

pause
