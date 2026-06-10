"""
Debug version of bad position miner to see what's happening
"""

import chess
import chess.pgn
import re
from pathlib import Path

def parse_eval_from_comment(comment: str) -> float:
    """Extract evaluation from PGN comment."""
    if not comment:
        return None
    
    # Look for mate scores
    mate_match = re.search(r'[Mm]([+-]?\d+)', comment)
    if mate_match:
        mate_in = int(mate_match.group(1))
        return 100.0 if mate_in > 0 else -100.0
    
    # Look for centipawn eval like "+0.46/5" or "-2.35/4"
    eval_match = re.search(r'([+-]?\d+\.?\d*)/\d+', comment)
    if eval_match:
        try:
            return float(eval_match.group(1))
        except:
            return None
    
    return None

# Test on one game
pgn_path = Path("E:/Programming Stuff/Chess Engines/Chess Engine Playground/engine-metrics/raw_data/game_records/Engine Battle 202512")
pgn_files = list(pgn_path.glob("*.pgn"))

if pgn_files:
    test_file = pgn_files[0]
    print(f"Testing: {test_file.name}\n")
    
    with open(test_file, 'r', encoding='utf-8', errors='ignore') as f:
        game = chess.pgn.read_game(f)
        
        if game:
            print(f"Game: {game.headers.get('White')} vs {game.headers.get('Black')}\n")
            
            board = game.board()
            prev_eval = None
            move_num = 0
            
            for node in game.mainline():
                move = node.move
                comment = node.comment
                current_eval = parse_eval_from_comment(comment)
                
                if move_num < 20:  # First 20 moves
                    print(f"Move {move_num+1:2d}: {board.san(move):8s} eval={current_eval:+6.2f} " if current_eval is not None else f"Move {move_num+1:2d}: {board.san(move):8s} eval=None    ", end="")
                    
                    if current_eval is not None and prev_eval is not None:
                        if board.turn == chess.BLACK:  # White just moved
                            eval_change = current_eval - prev_eval
                            if eval_change < -0.5:
                                print(f" ⚠️  WHITE BLUNDER! Drop: {abs(eval_change):.2f} pawns")
                            else:
                                print(f" (change: {eval_change:+.2f})")
                        else:  # Black just moved
                            eval_change = current_eval - prev_eval
                            if eval_change > 0.5:
                                print(f" ⚠️  BLACK BLUNDER! Drop: {abs(eval_change):.2f} pawns")
                            else:
                                print(f" (change: {eval_change:+.2f})")
                    else:
                        print()
                
                board.push(move)
                if current_eval is not None:
                    prev_eval = current_eval
                move_num += 1
            
            print(f"\n...({move_num} total moves)")
else:
    print("No PGN files found!")
