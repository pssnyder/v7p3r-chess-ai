#!/usr/bin/env python3
"""
V7P3R v20.0.2-Material Beta UCI Interface
A/B Test Variant: AI Ordering + MaterialOpponent Simple Eval + v18.3 Search

For Arena/Cutechess tournament testing
"""

import sys
import chess
import io
import contextlib
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from v7p3r_v20_material_hybrid import V7P3R_v20_Hybrid


class V7P3R_Material_UCI:
    """UCI protocol handler for Material variant."""
    
    def __init__(self):
        """Initialize UCI handler."""
        self.engine = None
        self.board = chess.Board()
        self.model_path = "models/stage2_combined/best_checkpoint.pt"
        self.device = 'cpu'
    
    def uci(self):
        """Respond to UCI command."""
        print("id name V7P3R v20.0.2-Material Beta (AI + Simple Eval)")
        print("id author Pat Snyder")
        print("uciok")
    
    def isready(self):
        """Initialize engine if not already done."""
        if self.engine is None:
            try:
                # Suppress model loading output for UCI
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    self.engine = V7P3R_v20_Hybrid(
                        model_path=self.model_path,
                        device=self.device
                    )
                
                print("readyok", flush=True)
            except Exception as e:
                print(f"info string Error loading engine: {e}", flush=True)
                print("readyok", flush=True)
        else:
            print("readyok", flush=True)
    
    def position(self, args: list):
        """Set up position from FEN or moves."""
        if not args:
            return
        
        if args[0] == "startpos":
            self.board = chess.Board()
            args = args[1:]
        elif args[0] == "fen":
            # Find where moves start
            if "moves" in args:
                moves_idx = args.index("moves")
                fen = " ".join(args[1:moves_idx])
                args = args[moves_idx:]
            else:
                fen = " ".join(args[1:])
                args = []
            self.board = chess.Board(fen)
        
        # Apply moves
        if args and args[0] == "moves":
            for move_str in args[1:]:
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in self.board.legal_moves:
                        self.board.push(move)
                except:
                    print(f"info string Invalid move: {move_str}", flush=True)
    
    def go(self, args: list):
        """Search for best move."""
        if self.engine is None:
            print("info string Engine not initialized", flush=True)
            return
        
        # Parse time controls
        wtime = None
        btime = None
        winc = 0
        binc = 0
        movetime = None
        depth = None
        
        i = 0
        while i < len(args):
            if args[i] == "wtime" and i + 1 < len(args):
                wtime = int(args[i + 1])
                i += 2
            elif args[i] == "btime" and i + 1 < len(args):
                btime = int(args[i + 1])
                i += 2
            elif args[i] == "winc" and i + 1 < len(args):
                winc = int(args[i + 1])
                i += 2
            elif args[i] == "binc" and i + 1 < len(args):
                binc = int(args[i + 1])
                i += 2
            elif args[i] == "movetime" and i + 1 < len(args):
                movetime = int(args[i + 1])
                i += 2
            elif args[i] == "depth" and i + 1 < len(args):
                depth = int(args[i + 1])
                i += 2
            else:
                i += 1
        
        # Determine time limit
        if movetime:
            time_limit = movetime / 1000.0
        else:
            my_time = wtime if self.board.turn == chess.WHITE else btime
            my_inc = winc if self.board.turn == chess.WHITE else binc
            if my_time:
                time_limit = (my_time / 1000.0) / 30 + (my_inc / 1000.0) * 0.8
            else:
                time_limit = 5.0
        
        # Search for best move
        best_move = self.engine.search(self.board, depth=depth or 8, time_limit=time_limit)
        
        if best_move:
            print(f"bestmove {best_move.uci()}", flush=True)
        else:
            # Fallback: pick any legal move
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                print(f"bestmove {legal_moves[0].uci()}", flush=True)
    
    def run(self):
        """Main UCI loop."""
        while True:
            try:
                command = input().strip()
                if not command:
                    continue
                
                parts = command.split()
                cmd = parts[0]
                args = parts[1:]
                
                if cmd == "uci":
                    self.uci()
                elif cmd == "isready":
                    self.isready()
                elif cmd == "ucinewgame":
                    self.board = chess.Board()
                    if self.engine:
                        self.engine.transposition_table.clear()
                        self.engine.killer_moves = self.engine.killer_moves.__class__()
                        self.engine.history_heuristic = self.engine.history_heuristic.__class__()
                elif cmd == "position":
                    self.position(args)
                elif cmd == "go":
                    self.go(args)
                elif cmd == "quit":
                    break
                    
            except EOFError:
                break
            except Exception as e:
                print(f"info string ERROR: {str(e)}", flush=True)


def main():
    """UCI protocol handler for Material variant"""
    uci = V7P3R_Material_UCI()
    uci.run()

if __name__ == "__main__":
    main()
