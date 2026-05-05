@echo off
REM Full 4M Puzzle Preprocessing Pipeline
REM Estimated time: 8-12 hours on RTX 4070 Ti

echo ========================================
echo V7P3RAI v4.0 - Full Puzzle Preprocessing
echo ========================================
echo.
echo This will preprocess 4,000,000 puzzles
echo Estimated time: 8-12 hours
echo Output: ~2-3 GB enriched dataset
echo.
pause

cd "e:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v4.0"

python scripts/preprocess_puzzles_with_stockfish.py ^
  --puzzle-db "E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-tester\databases\puzzles.db" ^
  --stockfish "E:\Programming Stuff\Chess Engines\Tournament Engines\Stockfish\stockfish-windows-x86-64-avx2.exe" ^
  --max-puzzles 4000000 ^
  --rating-min 600 ^
  --rating-max 2500 ^
  --stockfish-time 1.0 ^
  --top-n 10 ^
  --batch-size 1000 ^
  --output-dir data/preprocessed_puzzles

echo.
echo ========================================
echo Preprocessing Complete!
echo ========================================
echo.
echo Next step: Train model with:
echo python scripts/train_move_ordering.py --data-path data/preprocessed_puzzles/enriched_puzzles_compact_*.json
echo.
pause
