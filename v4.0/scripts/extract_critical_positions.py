"""
Stage 2: Extract Critical Positions from V7P3R Losing Games

Analyzes each losing game with Stockfish to identify positions where V7P3R
made critical blunders. These become the foundation for corrective training.

Usage:
    python scripts/extract_critical_positions.py --games-file data/stage2_games/v7p3r_losses_correct.json --max-games 1693
"""

import os
import json
import chess
import chess.engine
import chess.pgn
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse
from io import StringIO


@dataclass
class CriticalPosition:
    """A position where V7P3R made a critical mistake."""
    game_id: str
    move_number: int  # Full move number (e.g., 15 for move 15)
    ply: int  # Half-move number (0-indexed)
    fen: str
    v7p3r_move: str  # What V7P3R played (UCI format)
    v7p3r_move_san: str  # SAN notation
    eval_before: int  # Centipawn eval before V7P3R's move (from V7P3R's perspective)
    eval_after: int  # Centipawn eval after V7P3R's move
    eval_drop: int  # How much eval dropped (positive = mistake)
    best_move: str  # Stockfish's best move (UCI)
    best_move_san: str  # Best move in SAN
    best_eval: int  # Eval if best move was played
    move_classification: str  # "blunder", "mistake", "inaccuracy", "good" (Stockfish-style)
    context: str  # "final_blunder", "tactical_failure", "positional_mistake"
    opponent: str
    game_date: str
    
    def to_dict(self):
        return asdict(self)


class CriticalPositionExtractor:
    """Extract critical positions from losing games using Stockfish analysis."""
    
    def __init__(self, stockfish_path: str, analysis_time: float = 0.3):
        self.stockfish_path = Path(stockfish_path)
        self.analysis_time = analysis_time
        self.engine: Optional[chess.engine.SimpleEngine] = None
        self.critical_positions: List[CriticalPosition] = []
        
        # Thresholds for identifying critical positions
        self.FINAL_MOVES_COUNT = 5  # Analyze last N moves before checkmate
        self.BLUNDER_THRESHOLD = 150  # Centipawns (1.5 pawns)
        self.TACTICAL_THRESHOLD = 300  # Centipawns (3 pawns) for tactical failures
        
        # Stockfish-style move classification thresholds
        self.BLUNDER_CP = 300  # Stockfish blunder: 3+ pawns
        self.MISTAKE_CP = 100   # Stockfish mistake: 1-3 pawns
        self.INACCURACY_CP = 50 # Stockfish inaccuracy: 0.5-1 pawn
        
    def initialize_stockfish(self):
        """Start Stockfish engine (reuse for efficiency)."""
        print(f"   Starting Stockfish engine...")
        self.engine = chess.engine.SimpleEngine.popen_uci(str(self.stockfish_path))
        print(f"   ✅ Stockfish ready")
    
    def cleanup(self):
        """Close Stockfish engine."""
        if self.engine:
            self.engine.quit()
            print(f"   ✅ Stockfish engine closed")
    
    def analyze_position(self, board: chess.Board) -> Tuple[int, str, str]:
        """
        Analyze position with Stockfish.
        
        Returns:
            (eval_cp, best_move_uci, best_move_san)
        """
        if not self.engine:
            raise RuntimeError("Stockfish engine not initialized")
        
        # Get analysis
        info = self.engine.analyse(
            board,
            chess.engine.Limit(time=self.analysis_time)
        )
        
        # Extract evaluation (from White's perspective)
        score = info.get("score")
        if score is None:
            return 0, "0000", "??"
        
        score_white = score.white()
        if score_white.is_mate():
            # Convert mate to large centipawn value
            mate_moves = score_white.mate()
            eval_cp = 10000 * (1 if mate_moves > 0 else -1)
        else:
            eval_cp = score_white.score() if score_white.score() is not None else 0
        
        # Get best move from PV
        pv = info.get("pv")
        if not pv or len(pv) == 0:
            return eval_cp, "0000", "??"
        
        best_move = pv[0]
        best_move_san = board.san(best_move)
        
        return eval_cp, best_move.uci(), best_move_san
    
    def extract_from_game(self, game_data: Dict) -> List[CriticalPosition]:
        """
        Extract critical positions from a single game.
        
        Returns list of CriticalPosition objects.
        """
        critical_positions = []
        
        # Reconstruct game
        moves = game_data["moves"]
        v7p3r_color = game_data["v7p3r_color"]
        game_id = game_data["game_id"]
        opponent = game_data["opponent"]
        game_date = game_data["date"]
        
        board = chess.Board()
        
        # Track positions to analyze
        # Strategy: Analyze all V7P3R moves, then filter for critical ones
        v7p3r_is_white = (v7p3r_color == "white")
        
        previous_eval = 0  # Start at equality
        
        for ply, move_san in enumerate(moves):
            move_number = (ply // 2) + 1
            is_v7p3r_move = (ply % 2 == 0) if v7p3r_is_white else (ply % 2 == 1)
            
            # Skip opponent moves
            if not is_v7p3r_move:
                try:
                    move = board.parse_san(move_san)
                    board.push(move)
                except Exception as e:
                    print(f"      ⚠️  Error parsing opponent move {move_san} in {game_id}: {e}")
                    return critical_positions
                continue
            
            # This is a V7P3R move - analyze position BEFORE move
            fen_before = board.fen()
            
            # Get eval before move
            try:
                eval_before_raw, best_move_uci, best_move_san = self.analyze_position(board)
            except Exception as e:
                print(f"      ⚠️  Stockfish error at ply {ply} in {game_id}: {e}")
                # Skip this position, continue with game
                try:
                    move = board.parse_san(move_san)
                    board.push(move)
                except:
                    return critical_positions
                continue
            
            # Normalize eval from V7P3R's perspective
            if v7p3r_is_white:
                eval_before = eval_before_raw
            else:
                eval_before = -eval_before_raw  # Flip for black
            
            # Parse and make V7P3R's actual move
            try:
                v7p3r_move = board.parse_san(move_san)
                v7p3r_move_uci = v7p3r_move.uci()
                board.push(v7p3r_move)
            except Exception as e:
                print(f"      ⚠️  Error parsing V7P3R move {move_san} in {game_id}: {e}")
                return critical_positions
            
            # Get eval after move
            try:
                eval_after_raw, _, _ = self.analyze_position(board)
            except Exception as e:
                print(f"      ⚠️  Stockfish error after ply {ply} in {game_id}: {e}")
                continue
            
            # Normalize eval from V7P3R's perspective (after move, it's opponent's turn)
            if v7p3r_is_white:
                eval_after = -eval_after_raw  # Opponent's eval, flip back
            else:
                eval_after = eval_after_raw  # Opponent's eval, already correct sign
            
            # Calculate eval drop (positive = V7P3R's position got worse)
            eval_drop = eval_before - eval_after
            
            # Get best move eval (what would have happened with best move)
            board.pop()  # Undo V7P3R's move
            try:
                best_move_obj = chess.Move.from_uci(best_move_uci)
                board.push(best_move_obj)
                best_eval_raw, _, _ = self.analyze_position(board)
                board.pop()  # Undo best move
                board.push(v7p3r_move)  # Re-apply actual move
                
                if v7p3r_is_white:
                    best_eval = -best_eval_raw
                else:
                    best_eval = best_eval_raw
            except:
                best_eval = eval_before  # Fallback
            
            # Classify move using Stockfish-style thresholds
            if eval_drop >= self.BLUNDER_CP:
                move_classification = "blunder"
            elif eval_drop >= self.MISTAKE_CP:
                move_classification = "mistake"
            elif eval_drop >= self.INACCURACY_CP:
                move_classification = "inaccuracy"
            else:
                move_classification = "good"
            
            # Determine if this is a critical position
            is_critical = False
            context = ""
            
            # Check if it's in final moves (high priority)
            moves_from_end = len(moves) - ply
            if moves_from_end <= self.FINAL_MOVES_COUNT:
                is_critical = True
                context = "final_blunder"
            
            # Check if it's a tactical failure (large eval drop)
            elif eval_drop >= self.TACTICAL_THRESHOLD:
                is_critical = True
                context = "tactical_failure"
            
            # Check if it's a significant blunder or mistake (Stockfish classification)
            elif move_classification in ["blunder", "mistake"]:
                is_critical = True
                context = "positional_mistake"
            
            # Also capture inaccuracies in opening/early middlegame (move 1-20)
            elif move_classification == "inaccuracy" and move_number <= 20:
                is_critical = True
                context = "opening_inaccuracy"
            
            # Add to critical positions if identified
            if is_critical:
                critical_pos = CriticalPosition(
                    game_id=game_id,
                    move_number=move_number,
                    ply=ply,
                    fen=fen_before,
                    v7p3r_move=v7p3r_move_uci,
                    v7p3r_move_san=move_san,
                    eval_before=eval_before,
                    eval_after=eval_after,
                    eval_drop=eval_drop,
                    best_move=best_move_uci,
                    best_move_san=best_move_san,
                    best_eval=best_eval,
                    move_classification=move_classification,
                    context=context,
                    opponent=opponent,
                    game_date=game_date
                )
                critical_positions.append(critical_pos)
        
        return critical_positions
    
    def process_all_games(self, games_data: List[Dict], max_games: Optional[int] = None) -> List[CriticalPosition]:
        """Process all games and extract critical positions."""
        if max_games:
            games_data = games_data[:max_games]
        
        print(f"📊 Processing {len(games_data)} games...")
        
        for i, game_data in enumerate(games_data):
            game_id = game_data["game_id"]
            print(f"   [{i+1}/{len(games_data)}] Analyzing {game_id}...", end="")
            
            try:
                positions = self.extract_from_game(game_data)
                self.critical_positions.extend(positions)
                print(f" {len(positions)} critical positions found")
            except Exception as e:
                print(f" ERROR: {e}")
        
        return self.critical_positions
    
    def save_to_json(self, output_path: str):
        """Save critical positions to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "total_positions": len(self.critical_positions),
                "analysis_time_per_position": self.analysis_time,
                "blunder_threshold_cp": self.BLUNDER_THRESHOLD,
                "tactical_threshold_cp": self.TACTICAL_THRESHOLD
            },
            "positions": [pos.to_dict() for pos in self.critical_positions]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Saved {len(self.critical_positions)} critical positions to: {output_path}")
        
        # Print statistics
        self._print_statistics()
    
    def _print_statistics(self):
        """Print statistics about extracted positions."""
        if not self.critical_positions:
            return
        
        print(f"\n📈 Critical Position Statistics:")
        
        # Context distribution
        contexts = {}
        for pos in self.critical_positions:
            contexts[pos.context] = contexts.get(pos.context, 0) + 1
        
        print(f"   Total: {len(self.critical_positions)}")
        print(f"\n   By Context:")
        for context, count in sorted(contexts.items(), key=lambda x: -x[1]):
            print(f"      {context}: {count} ({count/len(self.critical_positions)*100:.1f}%)")
        
        # Move classification distribution (Stockfish-style)
        classifications = {}
        for pos in self.critical_positions:
            classifications[pos.move_classification] = classifications.get(pos.move_classification, 0) + 1
        
        print(f"\n   By Move Quality (Stockfish Classification):")
        for classification, count in sorted(classifications.items(), key=lambda x: -x[1]):
            print(f"      {classification}: {count} ({count/len(self.critical_positions)*100:.1f}%)")
        
        # Eval drop distribution
        avg_eval_drop = sum(pos.eval_drop for pos in self.critical_positions) / len(self.critical_positions)
        max_eval_drop = max(pos.eval_drop for pos in self.critical_positions)
        blunders = sum(1 for pos in self.critical_positions if pos.move_classification == "blunder")
        mistakes = sum(1 for pos in self.critical_positions if pos.move_classification == "mistake")
        inaccuracies = sum(1 for pos in self.critical_positions if pos.move_classification == "inaccuracy")
        
        print(f"\n   Eval Drop (Centipawns):")
        print(f"      Average: {avg_eval_drop:.0f}cp")
        print(f"      Maximum: {max_eval_drop:.0f}cp")
        print(f"      Blunders (300+ cp): {blunders}")
        print(f"      Mistakes (100-300 cp): {mistakes}")
        print(f"      Inaccuracies (50-100 cp): {inaccuracies}")
        
        # Top blunders
        print(f"\n   Worst Blunders (Top 5):")
        sorted_positions = sorted(self.critical_positions, key=lambda x: -x.eval_drop)
        for pos in sorted_positions[:5]:
            print(f"      Game {pos.game_id}, Move {pos.move_number}: {pos.v7p3r_move_san} ({pos.move_classification}, dropped {pos.eval_drop}cp)")


def main():
    parser = argparse.ArgumentParser(description="Extract critical positions from V7P3R losing games")
    parser.add_argument(
        "--games-file",
        type=str,
        default="data/stage2_games/v7p3r_losses_correct.json",
        help="JSON file with losing games"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/stage2_positions/critical_positions.json",
        help="Output JSON file for critical positions"
    )
    parser.add_argument(
        "--stockfish-path",
        type=str,
        default=r"E:\Programming Stuff\Chess Engines\Tournament Engines\Stockfish\stockfish-windows-x86-64-avx2.exe",
        help="Path to Stockfish executable"
    )
    parser.add_argument(
        "--analysis-time",
        type=float,
        default=0.3,
        help="Stockfish analysis time per position (seconds)"
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Maximum number of games to process (for testing)"
    )
    
    args = parser.parse_args()
    
    print("🚀 V7P3R Critical Position Extractor (Stage 2)")
    print("=" * 60)
    print(f"📂 Games file: {args.games_file}")
    print(f"💾 Output: {args.output}")
    print(f"⚙️  Stockfish: {args.stockfish_path}")
    print(f"⏱️  Analysis time: {args.analysis_time}s per position")
    print("=" * 60)
    
    # Load games
    print(f"\n📂 Loading games...")
    with open(args.games_file, 'r', encoding='utf-8') as f:
        games_data = json.load(f)
    
    games = games_data["games"]
    print(f"   Loaded {len(games)} losing games")
    
    if args.max_games:
        print(f"   Limiting to first {args.max_games} games for testing")
    
    # Initialize extractor
    extractor = CriticalPositionExtractor(
        stockfish_path=args.stockfish_path,
        analysis_time=args.analysis_time
    )
    
    try:
        extractor.initialize_stockfish()
        positions = extractor.process_all_games(games, max_games=args.max_games)
        
        if positions:
            extractor.save_to_json(args.output)
            print(f"\n✅ Extraction complete! {len(positions)} critical positions extracted.")
        else:
            print(f"\n⚠️  No critical positions found!")
    
    finally:
        extractor.cleanup()


if __name__ == "__main__":
    main()
