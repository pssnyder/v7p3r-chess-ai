#!/usr/bin/env python3
"""
UCI Interface for V7P3R v20 Beta - Hybrid AI/Static Engine

Provides Universal Chess Interface protocol support for tournament play.
Compatible with Arena, Cutechess, and other UCI chess GUIs.
"""

import sys
import chess
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from v7p3r_v20_hybrid import V7P3R_v20_Hybrid


class V7P3R_v20_UCI:
    """UCI protocol handler for V7P3R v20 Beta."""
    
    def __init__(self):
        """Initialize UCI handler."""
        self.engine = None
        self.board = chess.Board()
        self.model_path = "models/stage2_combined/best_checkpoint.pt"
        self.device = 'cpu'
        
    def uci(self):
        """Respond to UCI command."""
        print("id name V7P3R v20.0.2 Beta (Hybrid AI + v18.3 Search)")
        print("id author Pat Snyder")
        print("uciok")
    
    def isready(self):
        """Initialize engine if not already done."""
        if self.engine is None:
            try:
                # Suppress model loading output for UCI
                import io
                import contextlib
                
                # Redirect stdout during engine initialization
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
        
        # Parse time control
        time_limit = 5.0  # Default
        depth = None
        
        i = 0
        while i < len(args):
            if args[i] == "wtime" and self.board.turn == chess.WHITE:
                # Time in milliseconds
                time_ms = int(args[i + 1])
                time_limit = min(time_ms / 1000.0 * 0.05, 10.0)  # Use 5% of remaining time
                i += 2
            elif args[i] == "btime" and self.board.turn == chess.BLACK:
                time_ms = int(args[i + 1])
                time_limit = min(time_ms / 1000.0 * 0.05, 10.0)
                i += 2
            elif args[i] == "movetime":
                time_limit = int(args[i + 1]) / 1000.0
                i += 2
            elif args[i] == "depth":
                depth = int(args[i + 1])
                i += 2
            else:
                i += 1
        
        # Search
        try:
            best_move = self.engine.search(self.board, time_limit=time_limit, depth=depth)
            if best_move:
                print(f"bestmove {best_move.uci()}", flush=True)
            else:
                # No legal moves
                legal_moves = list(self.board.legal_moves)
                if legal_moves:
                    print(f"bestmove {legal_moves[0].uci()}", flush=True)
                else:
                    print("bestmove 0000", flush=True)
        except Exception as e:
            print(f"info string Search error: {e}", flush=True)
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                print(f"bestmove {legal_moves[0].uci()}", flush=True)
            else:
                print("bestmove 0000", flush=True)
    
    def quit(self):
        """Exit UCI mode."""
        sys.exit(0)
    
    def run(self):
        """Main UCI loop."""
        while True:
            try:
                line = input().strip()
                if not line:
                    continue
                
                parts = line.split()
                command = parts[0]
                args = parts[1:] if len(parts) > 1 else []
                
                if command == "uci":
                    self.uci()
                elif command == "isready":
                    self.isready()
                elif command == "ucinewgame":
                    self.board = chess.Board()
                elif command == "position":
                    self.position(args)
                elif command == "go":
                    self.go(args)
                elif command == "quit":
                    self.quit()
                elif command == "stop":
                    pass  # Already handled in search
                else:
                    # Ignore unknown commands
                    pass
                    
            except EOFError:
                break
            except Exception as e:
                print(f"info string Error: {e}", flush=True)


def main():
    """Start UCI mode."""
    uci = V7P3R_v20_UCI()
    uci.run()


if __name__ == '__main__':
    main()
