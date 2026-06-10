"""
Massive Bad Position Miner - Extract 1M bad positions from entire game library

This script:
1. Processes ALL PGN files from your game library (V7P3R, human, bot games)
2. Detects bad moves (eval drops >= 100cp for mistakes, >= 200cp for blunders)
3. Uses FAST feature extraction (19 features) instead of slow 690-dim features
4. Outputs to bad_positions.jsonl with append mode (resumable)
5. Shows progress and ETA

Target: 1,000,000 bad positions
Current: ~73k existing bad positions
Need: ~927k more

Blunder detection:
- Small mistake: 50-99cp drop (weight 0.5)
- Mistake: 100-199cp drop (weight 1.0)
- Blunder: 200-299cp drop (weight 2.0)
- Major blunder: 300+ cp drop (weight 3.0)
"""

import chess
import chess.pgn
import json
import re
from pathlib import Path
import sys
from collections import Counter
from datetime import datetime
from tqdm import tqdm

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def extract_fast_features(fen):
    """Extract fast 19-dim features from FEN (same as train_balanced.py)"""
    try:
        board = chess.Board(fen)
        
        features = []
        
        # Piece counts (12 features)
        for color in [chess.WHITE, chess.BLACK]:
            features.append(len(board.pieces(chess.PAWN, color)))
            features.append(len(board.pieces(chess.KNIGHT, color)))
            features.append(len(board.pieces(chess.BISHOP, color)))
            features.append(len(board.pieces(chess.ROOK, color)))
            features.append(len(board.pieces(chess.QUEEN, color)))
            features.append(len(board.pieces(chess.KING, color)))
        
        # Material balance (1 feature)
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                       chess.ROOK: 5, chess.QUEEN: 9}
        white_material = sum(len(board.pieces(pt, chess.WHITE)) * val 
                            for pt, val in piece_values.items())
        black_material = sum(len(board.pieces(pt, chess.BLACK)) * val 
                            for pt, val in piece_values.items())
        features.append(white_material - black_material)
        
        # Positional features (4 features)
        features.append(1 if board.turn == chess.WHITE else 0)
        features.append(board.has_kingside_castling_rights(chess.WHITE))
        features.append(board.has_queenside_castling_rights(chess.WHITE))
        features.append(board.is_check())
        
        # Mobility (2 features)
        features.append(board.legal_moves.count())
        board.turn = not board.turn
        features.append(board.legal_moves.count())
        
        return features
        
    except Exception as e:
        return None


def parse_eval_from_comment(comment: str) -> float:
    """
    Extract evaluation from PGN comment.
    
    Example formats:
    - "{(e2-e3 e7-e6) +0.46/5 7}" -> 0.46
    - "{+4.42/5 7}" -> 4.42
    - "{M5/5}" -> 100.0 (mate in 5)
    
    Returns None if no eval found.
    """
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


def calculate_weight(eval_drop_cp: float) -> float:
    """
    Calculate position weight based on severity of blunder.
    
    Args:
        eval_drop_cp: Eval drop in centipawns
        
    Returns:
        Weight (0.5 to 3.0)
    """
    if eval_drop_cp >= 300:
        return 3.0  # Major blunder
    elif eval_drop_cp >= 200:
        return 2.0  # Blunder
    elif eval_drop_cp >= 100:
        return 1.0  # Mistake
    elif eval_drop_cp >= 50:
        return 0.5  # Small mistake
    else:
        return 0.0  # Not a mistake


def mine_pgn_file(pgn_path: Path, min_eval_drop_cp: float = 50.0) -> list:
    """
    Mine bad positions from a single PGN file.
    
    Args:
        pgn_path: Path to PGN file
        min_eval_drop_cp: Minimum eval drop in centipawns (default 50)
    
    Returns:
        List of bad position dictionaries
    """
    bad_positions = []
    games_processed = 0
    
    try:
        with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as pgn_file:
            while True:
                try:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break
                    
                    games_processed += 1
                    
                    # Parse game and track evals
                    board = game.board()
                    prev_eval = None
                    move_number = 0
                    
                    for node in game.mainline():
                        move = node.move
                        comment = node.comment
                        
                        # Get eval after this move
                        current_eval = parse_eval_from_comment(comment)
                        
                        # Push the move to advance the board
                        board.push(move)
                        move_number += 1
                        
                        if current_eval is not None and prev_eval is not None and move_number > 5:  # Skip opening
                            # Evals are ALWAYS from White's perspective (positive = good for White)
                            # board.turn shows who plays NEXT (after the move was made)
                            # So if board.turn == BLACK, White just moved
                            
                            eval_drop_cp = 0
                            is_blunder = False
                            
                            if board.turn == chess.BLACK:  # White just moved
                                # White wants eval to increase (or stay same)
                                eval_change = current_eval - prev_eval
                                if eval_change < -0.5:  # Eval dropped (bad for White)
                                    eval_drop_cp = abs(eval_change) * 100
                                    is_blunder = True
                            else:  # Black just moved (board.turn == WHITE)
                                # Black wants eval to decrease (from White's perspective)
                                eval_change = current_eval - prev_eval
                                if eval_change > 0.5:  # Eval increased (bad for Black)
                                    eval_drop_cp = abs(eval_change) * 100
                                    is_blunder = True
                            
                            # If this was a blunder/mistake, record it
                            if is_blunder and eval_drop_cp >= min_eval_drop_cp:
                                # Position BEFORE the bad move
                                board.pop()  # Undo the bad move
                                fen = board.fen()
                                
                                # Calculate fast features
                                features = extract_fast_features(fen)
                                
                                if features is not None:
                                    # Calculate weight
                                    weight = calculate_weight(eval_drop_cp)
                                    
                                    # Create position record
                                    position_data = {
                                        'fen': fen,
                                        'features': features,  # Fast 19-dim features (not FEN string!)
                                        'label': 0,  # BAD position
                                        'weight': weight,
                                        'source': 'bad_move',
                                        'eval_drop_cp': eval_drop_cp,
                                        'move_uci': move.uci(),
                                        'game_info': {
                                            'white': game.headers.get('White', 'Unknown'),
                                            'black': game.headers.get('Black', 'Unknown'),
                                            'result': game.headers.get('Result', '*'),
                                            'move_number': move_number,
                                            'pgn_file': pgn_path.name
                                        }
                                    }
                                    
                                    bad_positions.append(position_data)
                                
                                # Redo the move to continue
                                board.push(move)
                        
                        # Update for next iteration
                        if current_eval is not None:
                            prev_eval = current_eval
                        
                except Exception as e:
                    # Skip corrupted games
                    continue
    
    except Exception as e:
        print(f"   ERROR reading {pgn_path.name}: {e}")
        return []
    
    return bad_positions


def mine_directory(directory: Path, output_path: Path, target_positions: int = 1000000):
    """
    Mine bad positions from all PGN files in directory and subdirectories.
    
    Args:
        directory: Root directory to search for PGN files
        output_path: Output JSONL file path
        target_positions: Target number of positions to mine
    """
    print(f"\n{'='*80}")
    print(f"MASSIVE BAD POSITION MINER")
    print(f"{'='*80}\n")
    print(f"📁 Source: {directory}")
    print(f"💾 Output: {output_path}")
    print(f"🎯 Target: {target_positions:,} bad positions\n")
    
    # Count existing positions if file exists
    existing_count = 0
    if output_path.exists():
        print("🔍 Counting existing positions...")
        with open(output_path, 'r') as f:
            existing_count = sum(1 for _ in f)
        print(f"   Found {existing_count:,} existing positions\n")
    
    needed = target_positions - existing_count
    if needed <= 0:
        print(f"✅ Target already reached! ({existing_count:,} >= {target_positions:,})")
        return
    
    print(f"📊 Need to mine {needed:,} more positions\n")
    
    # Find all PGN files
    print("🔍 Scanning for PGN files...")
    pgn_files = list(directory.glob("**/*.pgn"))
    print(f"   Found {len(pgn_files)} PGN files\n")
    
    if len(pgn_files) == 0:
        print("❌ No PGN files found!")
        return
    
    # Process files with progress bar
    total_mined = 0
    total_games = 0
    
    with open(output_path, 'a', encoding='utf-8') as outfile:
        with tqdm(total=needed, desc="Mining bad positions", unit="pos") as pbar:
            for pgn_path in pgn_files:
                if total_mined >= needed:
                    break
                
                # Mine this file
                bad_positions = mine_pgn_file(pgn_path, min_eval_drop_cp=50.0)
                
                # Write to output
                for pos in bad_positions:
                    json.dump(pos, outfile)
                    outfile.write('\n')
                    total_mined += 1
                    pbar.update(1)
                    
                    if total_mined >= needed:
                        break
    
    print(f"\n{'='*80}")
    print(f"✅ Mining complete!")
    print(f"   Mined: {total_mined:,} new bad positions")
    print(f"   Total: {existing_count + total_mined:,} bad positions")
    print(f"   Output: {output_path}")
    print(f"{'='*80}\n")


def main():
    """Main mining function"""
    
    # Configuration
    GAME_LIBRARY = Path("E:/Programming Stuff/Chess Engines/Chess Engine Playground/engine-metrics/raw_data/game_records")
    OUTPUT_FILE = project_root / "data" / "stage1" / "bad_positions_massive.jsonl"
    TARGET_POSITIONS = 1_000_000
    
    print(f"\n🚀 Starting massive bad position mining...")
    print(f"   Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Mine from game library
    mine_directory(GAME_LIBRARY, OUTPUT_FILE, TARGET_POSITIONS)
    
    print(f"\n✅ Done!")
    print(f"   End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n💡 Next steps:")
    print(f"   1. Merge with existing bad_positions.jsonl:")
    print(f"      cat bad_positions.jsonl bad_positions_massive.jsonl > bad_positions_merged.jsonl")
    print(f"   2. Train with balanced dataset (5.7M good + 1M bad)")
    print(f"      python scripts/stage1/train_balanced.py")


if __name__ == '__main__':
    main()
