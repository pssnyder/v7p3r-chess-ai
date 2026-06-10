"""
Tablebase Oracle - Perfect Endgame Knowledge

Provides exact evaluations for positions with ≤7 pieces using
Syzygy endgame tablebases.

When active, the network learns what "checkmate in N moves" actually
looks like, dramatically improving endgame conversion.
"""

import chess
import chess.syzygy
from typing import Optional, Tuple
import os


class TablebaseOracle:
    """
    Interface to Syzygy endgame tablebases for perfect play.
    
    Tablebases provide:
    - Exact win/loss/draw evaluation
    - Distance to mate (DTZ - distance to zeroing move)
    - Best moves in endgame positions
    """
    
    def __init__(self, tablebase_path: Optional[str] = None):
        """
        Initialize tablebase oracle.
        
        Args:
            tablebase_path: Path to Syzygy tablebase files
                           If None, tablebases are disabled
        """
        self.enabled = False
        self.tablebase = None
        self.max_pieces = 0
        
        if tablebase_path and os.path.exists(tablebase_path):
            try:
                self.tablebase = chess.syzygy.open_tablebase(tablebase_path)
                # Detect available tablebases
                self.max_pieces = self._detect_max_pieces(tablebase_path)
                self.enabled = True
                print(f"[OK] Tablebase oracle initialized: {tablebase_path}")
                print(f"     Available: {self.max_pieces}-piece tablebases")
            except Exception as e:
                print(f"[WARN] Failed to load tablebases: {e}")
                self.enabled = False
        else:
            if tablebase_path:
                print(f"[WARN] Tablebase path not found: {tablebase_path}")
            print("[INFO] Tablebase oracle disabled (will use Stockfish for endgames)")
    
    def _detect_max_pieces(self, path: str) -> int:
        """Detect maximum piece count available in tablebase directory."""
        # Check for common tablebase files
        # 3-piece: KPK, etc.
        # 4-piece: KQKP, etc.
        # 5-piece: KQPKP, etc.
        # 6-piece: KQQKQP, etc.
        # 7-piece: (very large, uncommon)
        
        files = os.listdir(path)
        max_pieces = 0
        
        for filename in files:
            if filename.endswith('.rtbw') or filename.endswith('.rtbz'):
                # Count letters in filename (represents pieces)
                # e.g., "KPK.rtbw" = 3 pieces
                base = filename.split('.')[0]
                piece_count = len([c for c in base if c.isupper()])
                max_pieces = max(max_pieces, piece_count)
        
        return max_pieces if max_pieces >= 3 else 5  # Default to 5 if detection fails
    
    def is_available(self, board: chess.Board) -> bool:
        """
        Check if tablebase is available for this position.
        
        Returns True if:
        - Tablebases are enabled
        - Position has ≤ max_pieces pieces
        - Position is legal
        """
        if not self.enabled or not self.tablebase:
            return False
        
        piece_count = len(board.piece_map())
        return piece_count <= self.max_pieces and not board.is_game_over()
    
    def probe_wdl(self, board: chess.Board) -> Optional[int]:
        """
        Probe Win-Draw-Loss (WDL) evaluation.
        
        Returns:
            2: Win for side to move
            1: Cursed win (win but 50-move rule can be claimed)
            0: Draw
           -1: Blessed loss (loss but can claim 50-move rule draw)
           -2: Loss for side to move
           None: Position not in tablebase
        """
        if not self.is_available(board):
            return None
        
        try:
            return self.tablebase.probe_wdl(board)
        except (KeyError, chess.syzygy.MissingTableError):
            return None
    
    def probe_dtz(self, board: chess.Board) -> Optional[int]:
        """
        Probe Distance-To-Zero (DTZ) evaluation.
        
        Returns:
            >0: Winning, mate in N half-moves
            0: Draw or stalemate
            <0: Losing, mated in N half-moves
            None: Position not in tablebase
        """
        if not self.is_available(board):
            return None
        
        try:
            return self.tablebase.probe_dtz(board)
        except (KeyError, chess.syzygy.MissingTableError):
            return None
    
    def get_normalized_eval(self, board: chess.Board) -> Optional[float]:
        """
        Get tablebase evaluation normalized to [-1, 1].
        
        Perfect for training targets!
        
        Returns:
            +1.0: Forced win
            -1.0: Forced loss
            0.0: Draw
            None: Not in tablebase
        """
        wdl = self.probe_wdl(board)
        
        if wdl is None:
            return None
        
        # Normalize WDL to training range
        if wdl >= 1:  # Win or cursed win
            return 1.0
        elif wdl <= -1:  # Loss or blessed loss
            return -1.0
        else:  # Draw
            return 0.0
    
    def get_best_move(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Get the best move according to tablebases.
        
        Returns:
            Best move or None if not in tablebase
        """
        if not self.is_available(board):
            return None
        
        try:
            # Find all winning moves, or if none, drawing moves
            best_wdl = -3  # Worse than any real WDL
            best_move = None
            
            for move in board.legal_moves:
                board.push(move)
                wdl = self.probe_wdl(board)
                board.pop()
                
                if wdl is not None:
                    # Flip sign (opponent's loss is our win)
                    move_wdl = -wdl
                    
                    if move_wdl > best_wdl:
                        best_wdl = move_wdl
                        best_move = move
            
            return best_move
        
        except Exception as e:
            print(f"[WARN] Tablebase probe error: {e}")
            return None
    
    def get_mate_distance(self, board: chess.Board) -> Optional[int]:
        """
        Get distance to mate (in plies/half-moves).
        
        Returns:
            >0: Win in N moves
            0: Draw
            <0: Loss in N moves
            None: Not in tablebase
        """
        dtz = self.probe_dtz(board)
        if dtz is not None:
            return dtz
        return None
    
    def analyze_position(self, board: chess.Board) -> dict:
        """
        Full tablebase analysis of position.
        
        Returns dict with all available information.
        """
        if not self.is_available(board):
            return {
                'available': False,
                'reason': 'Tablebases not loaded or too many pieces'
            }
        
        wdl = self.probe_wdl(board)
        dtz = self.probe_dtz(board)
        normalized_eval = self.get_normalized_eval(board)
        best_move = self.get_best_move(board)
        
        result = {
            'available': True,
            'piece_count': len(board.piece_map()),
            'wdl': wdl,
            'dtz': dtz,
            'normalized_eval': normalized_eval,
            'best_move': best_move.uci() if best_move else None,
        }
        
        # Human-readable interpretation
        if wdl is not None:
            if wdl >= 1:
                result['outcome'] = 'Win'
            elif wdl <= -1:
                result['outcome'] = 'Loss'
            else:
                result['outcome'] = 'Draw'
        
        if dtz is not None and dtz != 0:
            result['mate_in'] = abs(dtz)
            result['side_to_mate'] = 'White' if dtz > 0 else 'Black'
        
        return result
    
    def __repr__(self):
        if self.enabled:
            return f"TablebaseOracle(enabled={self.enabled}, max_pieces={self.max_pieces})"
        else:
            return "TablebaseOracle(disabled)"


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Tablebase Oracle - Perfect Endgame Knowledge")
    print("=" * 60)
    print()
    
    # Try to initialize with common tablebase locations
    possible_paths = [
        "E:/Chess/Tablebases/syzygy",
        "C:/Chess/Tablebases",
        "./tablebases",
        "../tablebases"
    ]
    
    oracle = None
    for path in possible_paths:
        if os.path.exists(path):
            oracle = TablebaseOracle(path)
            if oracle.enabled:
                break
    
    if not oracle or not oracle.enabled:
        print("[INFO] Creating disabled oracle for demonstration")
        oracle = TablebaseOracle(None)
    
    print()
    print(f"Oracle status: {oracle}")
    print()
    
    # Test with a simple endgame position
    print("Testing with KPK endgame:")
    print("-" * 60)
    
    # KPK: White pawn on e7, can promote
    board = chess.Board("4k3/4P3/4K3/8/8/8/8/8 w - - 0 1")
    print(board)
    print()
    
    if oracle.enabled:
        analysis = oracle.analyze_position(board)
        
        print("Tablebase Analysis:")
        for key, value in analysis.items():
            print(f"  {key}: {value}")
        print()
        
        if analysis.get('available'):
            eval_val = analysis.get('normalized_eval')
            print(f"Training target value: {eval_val}")
            print("This perfect knowledge will teach the network what")
            print("'forced win' actually looks like!")
    else:
        print("Tablebases not available - download Syzygy 3-4-5 piece")
        print("tablebases from: https://syzygy-tables.info/")
        print()
        print("Download instructions:")
        print("1. Download 3-4-5 piece sets (~1 GB)")
        print("2. Extract to a folder (e.g., E:/Chess/Tablebases/syzygy)")
        print("3. Update tablebase_path in trainer configuration")
    
    print()
    print("=" * 60)
    print("Tablebase oracle ready for perfect endgame training!")
    print("=" * 60)
