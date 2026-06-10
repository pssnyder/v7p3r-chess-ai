"""
Mine bad positions from V7P3R engine battle PGN files.

This script:
1. Parses PGN files from engine tournaments
2. Identifies moves where eval dropped significantly (mistakes/blunders)
3. Calculates features for those positions
4. Outputs to v7p3r_bad_positions.jsonl

Blunder thresholds:
- Mistake: Eval drop > 100cp
- Blunder: Eval drop > 200cp
"""
import chess
import chess.pgn
import json
import re
from pathlib import Path
import sys
from collections import Counter

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.calculate_features import FeatureCalculator, FeatureConfig


def parse_eval_from_comment(comment: str) -> float:
    """
    Extract evaluation from PGN comment.
    
    Example: "{(e2-e3 e7-e6) +0.46/5 7}" -> 0.46
    Returns None if no eval found.
    """
    if not comment:
        return None
    
    # Look for patterns like "+0.46/5" or "-2.35/4"
    # Also handle mate scores like "+289.96/5" (mate in X)
    match = re.search(r'([+-]?\d+\.?\d*)/\d+', comment)
    if match:
        try:
            return float(match.group(1))
        except:
            return None
    
    return None


def is_v7p3r_game(white: str, black: str) -> bool:
    """Check if both players are V7P3R engines."""
    white_lower = white.lower()
    black_lower = black.lower()
    
    return ('v7p3r' in white_lower or 'v7p3r' in white_lower) and \
           ('v7p3r' in black_lower or 'v7p3r' in black_lower)


def mine_pgn_file(pgn_path: Path, feature_calc: FeatureCalculator, 
                  min_eval_drop: float = 100.0) -> list:
    """
    Mine bad positions from a single PGN file.
    
    Args:
        pgn_path: Path to PGN file
        feature_calc: Feature calculator instance
        min_eval_drop: Minimum eval drop to consider (centipawns)
    
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
                    
                    # Check if V7P3R vs V7P3R game
                    white = game.headers.get('White', '')
                    black = game.headers.get('Black', '')
                    
                    if not is_v7p3r_game(white, black):
                        continue
                    
                    games_processed += 1
                    
                    # Parse game and track evals
                    board = game.board()
                    prev_eval = 0.0
                    move_number = 0
                    
                    for node in game.mainline():
                        move = node.move
                        comment = node.comment
                        
                        # Get eval after this move
                        current_eval = parse_eval_from_comment(comment)
                        
                        if current_eval is not None and move_number > 0:
                            # Check eval drop from perspective of player who just moved
                            # If White moved, we want to see if position got worse for White
                            # If Black moved, flip the eval sign
                            
                            if board.turn == chess.BLACK:  # White just moved
                                eval_drop = prev_eval - current_eval
                            else:  # Black just moved (need to flip signs)
                                eval_drop = (-prev_eval) - (-current_eval)
                            
                            # If eval dropped significantly, this was a bad move
                            if eval_drop >= min_eval_drop / 100.0:  # Convert cp to pawns
                                # Position BEFORE the bad move
                                board.pop()  # Undo the bad move
                                
                                # Calculate features for this position
                                try:
                                    features = feature_calc.calculate_features_from_fen(board.fen())
                                    
                                    # Add stockfish evaluation info
                                    features['F027_stockfish_eval_current'] = prev_eval
                                    features['F028_stockfish_eval_best'] = prev_eval  # We don't have best move eval
                                    features['F029_stockfish_eval_diff'] = eval_drop
                                    
                                    # Create position record
                                    position_data = {
                                        'fen': board.fen(),
                                        'move_uci': move.uci(),
                                        'source': 'v7p3r_game',
                                        'features': features,
                                        'stockfish_analysis': {
                                            'eval_before': prev_eval,
                                            'eval_after': current_eval,
                                            'eval_drop': eval_drop,
                                            'grade': 3 if eval_drop < 2.0 else 4 if eval_drop < 3.0 else 5
                                        },
                                        'game_info': {
                                            'white': white,
                                            'black': black,
                                            'move_number': move_number,
                                            'pgn_file': pgn_path.name
                                        }
                                    }
                                    
                                    bad_positions.append(position_data)
                                
                                except Exception as e:
                                    print(f"   Error calculating features: {e}")
                                
                                # Redo the move to continue
                                board.push(move)
                        
                        # Update for next iteration
                        prev_eval = current_eval if current_eval is not None else prev_eval
                        board.push(move)
                        move_number += 1
                
                except Exception as e:
                    # Skip corrupted games within PGN file
                    pass
    
    except Exception as e:
        print(f"   Error reading PGN file {pgn_path.name}: {e}")
    
    return bad_positions, games_processed


def main():
    """Main mining pipeline."""
    print("=" * 70)
    print("MINING BAD POSITIONS FROM V7P3R ENGINE BATTLES")
    print("=" * 70)
    
    # Configuration
    base_path = Path(__file__).parent.parent.parent
    game_records_path = Path("E:/Programming Stuff/Chess Engines/Chess Engine Playground/engine-metrics/raw_data/game_records")
    output_path = base_path / "data" / "stage1" / "v7p3r_bad_positions.jsonl"
    
    min_eval_drop_cp = 100  # Centipawns (1.0 pawns)
    max_positions = 500000  # Stop after collecting this many
    
    print(f"\nGame records: {game_records_path}")
    print(f"Output: {output_path}")
    print(f"Min eval drop: {min_eval_drop_cp}cp")
    print(f"Max positions: {max_positions:,}")
    
    # Find all PGN files
    pgn_files = list(game_records_path.rglob("*.pgn"))
    print(f"\nFound {len(pgn_files)} PGN files")
    
    # Initialize feature calculator
    print("\nInitializing feature calculator...")
    feature_config = FeatureConfig(
        core_position=True,
        king_safety=True,
        pawn_structure=True,
        enhanced_pawn_structure=True,
        piece_activity=True,
        tactical=True,
        rook_placement=True,
        knight_outposts=True,
        center_control=True
    )
    feature_calc = FeatureCalculator(feature_config)
    
    # Mine positions
    all_bad_positions = []
    total_games = 0
    files_processed = 0
    
    print(f"\nMining bad positions...")
    print(f"{'File':<50} {'Games':<8} {'Bad Pos':<10} {'Total':<10}")
    print("-" * 70)
    
    for pgn_file in pgn_files:
        if len(all_bad_positions) >= max_positions:
            print(f"\nReached max positions ({max_positions:,}), stopping...")
            break
        
        bad_positions, games_count = mine_pgn_file(
            pgn_file, 
            feature_calc, 
            min_eval_drop=min_eval_drop_cp
        )
        
        if games_count > 0:
            all_bad_positions.extend(bad_positions)
            total_games += games_count
            files_processed += 1
            
            print(f"{pgn_file.name[:50]:<50} {games_count:<8} {len(bad_positions):<10} {len(all_bad_positions):<10}")
    
    # Summary stats
    print("\n" + "=" * 70)
    print("MINING SUMMARY")
    print("=" * 70)
    print(f"PGN files processed: {files_processed:,}")
    print(f"V7P3R games found: {total_games:,}")
    print(f"Bad positions extracted: {len(all_bad_positions):,}")
    
    if len(all_bad_positions) > 0:
        # Grade distribution
        grades = [pos['stockfish_analysis']['grade'] for pos in all_bad_positions]
        grade_counts = Counter(grades)
        print(f"\nGrade distribution:")
        for grade in sorted(grade_counts.keys()):
            count = grade_counts[grade]
            pct = count / len(all_bad_positions) * 100
            print(f"   Grade {grade}: {count:,} ({pct:.1f}%)")
        
        # Eval drop stats
        eval_drops = [pos['stockfish_analysis']['eval_drop'] for pos in all_bad_positions]
        print(f"\nEval drop stats:")
        print(f"   Min: {min(eval_drops):.2f} pawns")
        print(f"   Max: {max(eval_drops):.2f} pawns")
        print(f"   Avg: {sum(eval_drops) / len(eval_drops):.2f} pawns")
        
        # Save to file
        print(f"Saving to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for pos in all_bad_positions:
                f.write(json.dumps(pos) + '\n')
        
        print(f"Saved {len(all_bad_positions):,} bad positions")
        
        # Next steps
        print("\n" + "=" * 70)
        print("NEXT STEPS")
        print("=" * 70)
        print("1. Merge with existing bad_positions.jsonl:")
        print(f"   - Current: 69,240 bad positions")
        print(f"   - V7P3R games: {len(all_bad_positions):,}")
        print(f"   - Total: {69240 + len(all_bad_positions):,}")
        print("\n2. Retrain model with larger bad dataset")
        print("3. Expect better bad move detection")
    else:
        print("\nNo bad positions found - check PGN format or thresholds")
    
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
