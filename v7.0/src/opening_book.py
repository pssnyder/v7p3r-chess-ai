"""
Opening Book System - Forces Interesting Starting Positions

Instead of playing through the opening every time, we "fast-forward"
to interesting positions by forcing predetermined opening moves.

This accelerates training by:
- Skipping repetitive opening theory
- Starting games in complex middlegame positions
- Providing diverse tactical scenarios
- Still maintaining move history for features
"""

import chess
import chess.pgn
import random
from typing import List, Tuple, Optional
from io import StringIO


class OpeningLine:
    """Represents a single opening line."""
    
    def __init__(self, name: str, moves: List[str], eco: str = ""):
        """
        Args:
            name: Opening name (e.g., "Sicilian Dragon")
            moves: List of moves in SAN notation
            eco: ECO code (optional)
        """
        self.name = name
        self.moves = moves
        self.eco = eco
    
    def apply_to_board(self, board: chess.Board) -> List[chess.Move]:
        """
        Apply this opening to a board.
        
        Returns list of moves that were made.
        """
        applied_moves = []
        for san_move in self.moves:
            move = board.parse_san(san_move)
            board.push(move)
            applied_moves.append(move)
        return applied_moves
    
    def __repr__(self):
        return f"OpeningLine({self.name}, {len(self.moves)} moves)"


class OpeningBookManager:
    """
    Manages a collection of opening lines for training.
    
    Loads from PGN file and randomly selects openings to start games.
    """
    
    def __init__(self, pgn_path: Optional[str] = None):
        """
        Initialize opening book.
        
        Args:
            pgn_path: Path to PGN file with opening lines (optional)
        """
        self.openings: List[OpeningLine] = []
        
        if pgn_path:
            self.load_from_pgn(pgn_path)
        else:
            self._load_default_openings()
    
    def _load_default_openings(self):
        """Load a default set of aggressive, tactical openings."""
        
        # Sicilian variations (sharp, tactical)
        self.openings.append(OpeningLine(
            "Sicilian Dragon",
            ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "g6"],
            "B70"
        ))
        
        self.openings.append(OpeningLine(
            "Sicilian Najdorf",
            ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6"],
            "B90"
        ))
        
        # King's Indian (complex middlegames)
        self.openings.append(OpeningLine(
            "King's Indian Defense",
            ["d4", "Nf6", "c4", "g6", "Nc3", "Bg7", "e4", "d6", "Nf3", "O-O"],
            "E60"
        ))
        
        # Queen's Gambit Accepted (tactical)
        self.openings.append(OpeningLine(
            "Queen's Gambit Accepted",
            ["d4", "d5", "c4", "dxc4", "Nf3", "Nf6", "e3", "e6", "Bxc4", "c5"],
            "D20"
        ))
        
        # Ruy Lopez (classical)
        self.openings.append(OpeningLine(
            "Ruy Lopez Marshall",
            ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"],
            "C80"
        ))
        
        # French Defense (strategic complexity)
        self.openings.append(OpeningLine(
            "French Winawer",
            ["e4", "e6", "d4", "d5", "Nc3", "Bb4", "e5", "c5", "a3", "Bxc3+"],
            "C15"
        ))
        
        # Dutch Defense (aggressive for Black)
        self.openings.append(OpeningLine(
            "Dutch Leningrad",
            ["d4", "f5", "g3", "Nf6", "Bg2", "g6", "Nf3", "Bg7", "O-O", "O-O"],
            "A80"
        ))
        
        # Benoni Defense (sharp positions)
        self.openings.append(OpeningLine(
            "Modern Benoni",
            ["d4", "Nf6", "c4", "c5", "d5", "e6", "Nc3", "exd5", "cxd5", "d6"],
            "A60"
        ))
        
        # Grünfeld (dynamic play)
        self.openings.append(OpeningLine(
            "Grünfeld Defense",
            ["d4", "Nf6", "c4", "g6", "Nc3", "d5", "cxd5", "Nxd5", "e4", "Nxc3"],
            "D80"
        ))
        
        # Scotch Game (open positions)
        self.openings.append(OpeningLine(
            "Scotch Game",
            ["e4", "e5", "Nf3", "Nc6", "d4", "exd4", "Nxd4", "Nf6", "Nxc6", "bxc6"],
            "C45"
        ))
        
        # Caro-Kann (solid but complex)
        self.openings.append(OpeningLine(
            "Caro-Kann Advance",
            ["e4", "c6", "d4", "d5", "e5", "Bf5", "Nf3", "e6", "Be2", "c5"],
            "B12"
        ))
        
        # Alekhine Defense (sharp)
        self.openings.append(OpeningLine(
            "Alekhine Defense",
            ["e4", "Nf6", "e5", "Nd5", "d4", "d6", "Nf3", "Bg4", "Be2", "e6"],
            "B02"
        ))
        
        print(f"[INFO] Loaded {len(self.openings)} default opening lines")
    
    def load_from_pgn(self, pgn_path: str, max_moves: int = 10):
        """
        Load openings from a PGN file.
        
        Args:
            pgn_path: Path to PGN file
            max_moves: Maximum moves to extract per game (default 10)
        """
        try:
            with open(pgn_path, 'r', encoding='utf-8') as pgn_file:
                while True:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break
                    
                    # Extract opening moves
                    board = game.board()
                    moves = []
                    node = game
                    
                    for i in range(max_moves):
                        if node.variations:
                            node = node.variations[0]
                            moves.append(board.san(node.move))
                            board.push(node.move)
                        else:
                            break
                    
                    if len(moves) >= 4:  # Require at least 4 moves
                        name = game.headers.get("Opening", f"Line {len(self.openings) + 1}")
                        eco = game.headers.get("ECO", "")
                        self.openings.append(OpeningLine(name, moves, eco))
            
            print(f"[INFO] Loaded {len(self.openings)} openings from {pgn_path}")
        
        except FileNotFoundError:
            print(f"[WARN] PGN file not found: {pgn_path}, using defaults")
            self._load_default_openings()
        except Exception as e:
            print(f"[ERROR] Failed to load PGN: {e}, using defaults")
            self._load_default_openings()
    
    def get_random_opening(self) -> OpeningLine:
        """Get a random opening from the book."""
        if not self.openings:
            self._load_default_openings()
        return random.choice(self.openings)
    
    def apply_random_opening(self, board: chess.Board) -> Tuple[OpeningLine, List[chess.Move]]:
        """
        Apply a random opening to a board.
        
        Returns:
            Tuple of (opening_line, applied_moves)
        """
        opening = self.get_random_opening()
        moves = opening.apply_to_board(board)
        return opening, moves
    
    def get_opening_by_name(self, name: str) -> Optional[OpeningLine]:
        """Get specific opening by name (case-insensitive partial match)."""
        name_lower = name.lower()
        for opening in self.openings:
            if name_lower in opening.name.lower():
                return opening
        return None
    
    def list_openings(self) -> List[str]:
        """Get list of all opening names."""
        return [opening.name for opening in self.openings]
    
    def __len__(self):
        return len(self.openings)
    
    def __repr__(self):
        return f"OpeningBookManager({len(self.openings)} openings)"


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Opening Book System - Fast-Forward to Interesting Positions")
    print("=" * 60)
    print()
    
    # Create book with defaults
    book = OpeningBookManager()
    
    print(f"Available openings: {len(book)}")
    print()
    
    print("Opening repertoire:")
    for i, opening_name in enumerate(book.list_openings(), 1):
        print(f"  {i}. {opening_name}")
    print()
    
    # Test random opening application
    print("Testing random opening application:")
    print("-" * 60)
    
    board = chess.Board()
    opening, moves = book.apply_random_opening(board)
    
    print(f"Selected: {opening.name} ({opening.eco})")
    print(f"Moves: {len(moves)}")
    print(f"Position after opening: {board.fen()}")
    print()
    print("Move sequence:")
    for i, move in enumerate(moves, 1):
        if i % 2 == 1:
            print(f"{(i+1)//2}. {move}", end=" ")
        else:
            print(move)
    print()
    print()
    
    print("Board state:")
    print(board)
    print()
    
    print("=" * 60)
    print("Opening book ready! Training will start from interesting positions.")
    print("=" * 60)
