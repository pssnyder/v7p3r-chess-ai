"""
Zobrist Hashing for Position Identification and Transposition Detection.

Provides fast position hashing for building transposition tables and
detecting duplicate positions across different games/puzzles.
"""

import chess
import random


class ZobristHasher:
    """
    Zobrist hashing implementation for chess positions.
    
    Each piece on each square gets a random 64-bit number.
    Position hash = XOR of all piece-square combinations.
    """
    
    def __init__(self, seed=42):
        """Initialize Zobrist random numbers with fixed seed for consistency."""
        random.seed(seed)
        
        # Generate random numbers for each piece type on each square
        self.piece_square_table = {}
        
        piece_types = [
            chess.PAWN, chess.KNIGHT, chess.BISHOP,
            chess.ROOK, chess.QUEEN, chess.KING
        ]
        colors = [chess.WHITE, chess.BLACK]
        
        for square in chess.SQUARES:
            for color in colors:
                for piece_type in piece_types:
                    piece = chess.Piece(piece_type, color)
                    key = (piece, square)
                    self.piece_square_table[key] = random.getrandbits(64)
        
        # Additional factors
        self.side_to_move_hash = random.getrandbits(64)
        self.castling_rights_hash = {
            'K': random.getrandbits(64),
            'Q': random.getrandbits(64),
            'k': random.getrandbits(64),
            'q': random.getrandbits(64)
        }
        self.en_passant_file_hash = {
            file: random.getrandbits(64) for file in range(8)
        }
    
    def hash_position(self, board: chess.Board) -> int:
        """
        Compute Zobrist hash for a chess position.
        
        Args:
            board: python-chess Board object
        
        Returns:
            64-bit integer hash
        """
        h = 0
        
        # XOR piece positions
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                h ^= self.piece_square_table[(piece, square)]
        
        # XOR side to move
        if board.turn == chess.BLACK:
            h ^= self.side_to_move_hash
        
        # XOR castling rights
        castling_fen = board.castling_xfen()
        for right in castling_fen:
            if right != '-':
                h ^= self.castling_rights_hash.get(right, 0)
        
        # XOR en passant square
        if board.ep_square is not None:
            file = chess.square_file(board.ep_square)
            h ^= self.en_passant_file_hash[file]
        
        return h
    
    def hash_fen(self, fen: str) -> int:
        """Hash a FEN string directly."""
        board = chess.Board(fen)
        return self.hash_position(board)


# Global hasher instance (singleton)
_global_hasher = None


def get_hasher() -> ZobristHasher:
    """Get global Zobrist hasher instance."""
    global _global_hasher
    if _global_hasher is None:
        _global_hasher = ZobristHasher(seed=42)
    return _global_hasher


def hash_position(board: chess.Board) -> int:
    """Convenience function to hash a position."""
    return get_hasher().hash_position(board)


def hash_fen(fen: str) -> int:
    """Convenience function to hash a FEN string."""
    return get_hasher().hash_fen(fen)


if __name__ == "__main__":
    # Test Zobrist hashing
    board = chess.Board()
    
    print("Testing Zobrist Hashing:")
    print(f"Starting position hash: {hash_position(board)}")
    
    # Make a move
    board.push_san("e4")
    hash1 = hash_position(board)
    print(f"After e4: {hash1}")
    
    # Undo and redo - should get same hash
    board.pop()
    board.push_san("e4")
    hash2 = hash_position(board)
    print(f"After undo/redo e4: {hash2}")
    print(f"Hashes match: {hash1 == hash2}")
    
    # Different position should have different hash
    board2 = chess.Board()
    board2.push_san("d4")
    hash3 = hash_position(board2)
    print(f"After d4: {hash3}")
    print(f"Different positions have different hashes: {hash1 != hash3}")
