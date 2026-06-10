"""
Tal Games Loader - Extract tactical positions from Mikhail Tal's master games.

Loads positions from Mikhail Tal's games (especially his wins with sacrificial attacks)
to teach the AI aggressive, chaotic, Tal-inspired tactical play.

Philosophy: "Deep dark forest where 2+2=5" - complicate the opponent's perspective
while maintaining a deterministic path through the chaos.
"""

import chess
import chess.pgn
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
import io
import sys

# Handle both relative and absolute imports
try:
    from .base_loader import DataSourceLoader
except ImportError:
    # Add parent directory to path for standalone testing
    sys.path.insert(0, str(Path(__file__).parent))
    from base_loader import DataSourceLoader


class TalGamesLoader(DataSourceLoader):
    """Load positions from Mikhail Tal's master games."""
    
    # Extract positions from these phases of the game
    MIN_MOVE = 10  # Skip opening book (first 9 moves)
    MAX_MOVE = 40  # Stop before pure endgames
    
    # Filter for tactical games
    MIN_GAME_LENGTH = 20  # Skip very short games
    MAX_GAME_LENGTH = 80  # Skip long positional grinds
    
    def __init__(
        self,
        pgn_path: str,
        filter_tal_wins: bool = True,
        extract_sacrifices: bool = True,
        seed: int = 42,
        shuffle: bool = True
    ):
        """
        Initialize Tal games loader.
        
        Args:
            pgn_path: Path to mikhail_tal_master_games.pgn
            filter_tal_wins: Only load games Tal won (default: True)
            extract_sacrifices: Prioritize positions with material sacrifices (default: True)
            seed: Random seed for reproducibility
            shuffle: Whether to shuffle positions
        """
        super().__init__(seed=seed, shuffle=shuffle)
        
        self.pgn_path = Path(pgn_path)
        self.filter_tal_wins = filter_tal_wins
        self.extract_sacrifices = extract_sacrifices
        
        # Initialize index tracker
        self._index = 0
        
        if not self.pgn_path.exists():
            raise FileNotFoundError(f"Tal games PGN not found: {pgn_path}")
        
        # Load all positions into memory (Tal games are small dataset)
        self.positions = []
        self._load_all_positions()
        
        # Shuffle if requested
        if self.shuffle:
            self.random.shuffle(self.positions)
        
        print(f"Loaded {len(self.positions)} positions from {self._games_loaded} Tal games")
        
    def _load_all_positions(self):
        """Extract all tactical positions from Tal's games."""
        self._games_loaded = 0
        self._positions_extracted = 0
        
        with open(self.pgn_path) as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                
                try:
                    positions = self._extract_positions_from_game(game)
                    if positions:
                        self.positions.extend(positions)
                        self._games_loaded += 1
                        self._positions_extracted += len(positions)
                except Exception as e:
                    # Skip corrupted games
                    print(f"Warning: Skipped Tal game due to error: {e}")
                    continue
    
    def _extract_positions_from_game(self, game: chess.pgn.Game) -> List[Dict[str, Any]]:
        """
        Extract tactical positions from a single Tal game.
        
        Strategy:
        - If Tal won: positions from Tal's side = GOOD (label=1, grade=1)
        - If Tal won: positions from opponent's side = BAD (label=0, grade=5)
        - If Tal lost/drew: skip game (don't want to learn losing patterns)
        - Prioritize middlegame positions (moves 10-40)
        - Extract positions with material imbalances (sacrifices)
        
        Args:
            game: chess.pgn.Game object
            
        Returns:
            List of position dictionaries with FEN, label, grade, source
        """
        positions = []
        
        # Get game metadata
        white = game.headers.get("White", "")
        black = game.headers.get("Black", "")
        result = game.headers.get("Result", "*")
        
        # Determine if Tal won
        tal_is_white = "Tal" in white
        tal_is_black = "Tal" in black
        
        if not (tal_is_white or tal_is_black):
            return []  # Not a Tal game
        
        # Filter for Tal wins if enabled
        if self.filter_tal_wins:
            if tal_is_white and result != "1-0":
                return []  # Tal didn't win as White
            if tal_is_black and result != "0-1":
                return []  # Tal didn't win as Black
        
        # Check game length (filter out very short or very long games)
        board = game.board()
        move_count = sum(1 for _ in game.mainline_moves())
        
        if move_count < self.MIN_GAME_LENGTH or move_count > self.MAX_GAME_LENGTH:
            return []  # Not a tactical game (too short/long)
        
        # Extract positions from middlegame
        node = game
        move_number = 0
        
        for move in game.mainline_moves():
            board.push(move)
            move_number += 1
            
            # Only extract from middlegame
            if move_number < self.MIN_MOVE or move_number > self.MAX_MOVE:
                node = node.next()
                continue
            
            # Check for material imbalance (potential sacrifice)
            is_sacrifice_position = False
            if self.extract_sacrifices:
                material_balance = self._calculate_material_balance(board)
                # Material imbalance > 3 (sacrifice of bishop/knight or more)
                if abs(material_balance) >= 300:
                    is_sacrifice_position = True
            
            # Determine whose turn it is
            is_tal_turn = (tal_is_white and board.turn == chess.WHITE) or \
                         (tal_is_black and board.turn == chess.BLACK)
            
            # Label positions
            if is_tal_turn:
                # Tal to move in a position he's winning from → GOOD position
                label = 1  # GOOD
                grade = 1  # Excellent (Tal's winning position)
            else:
                # Opponent to move in a losing position → BAD for opponent
                label = 0  # BAD
                grade = 5  # Poor (opponent in losing position)
            
            # Create position record
            position = {
                'fen': board.fen(),
                'label': label,
                'grade': grade,
                'source': 'tal_games',
                'game_info': {
                    'white': white,
                    'black': black,
                    'result': result,
                    'move_number': move_number,
                    'is_sacrifice': is_sacrifice_position,
                    'tal_color': 'white' if tal_is_white else 'black'
                }
            }
            
            # If this is a sacrifice position, add weight multiplier
            if is_sacrifice_position:
                position['weight'] = 1.5  # Emphasize sacrificial positions
            
            positions.append(position)
            node = node.next()
        
        return positions
    
    def _calculate_material_balance(self, board: chess.Board) -> int:
        """
        Calculate material balance in centipawns (White perspective).
        
        Args:
            board: Current chess board
            
        Returns:
            Material balance in centipawns (positive = White ahead)
        """
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900
        }
        
        balance = 0
        for piece_type in piece_values:
            white_count = len(board.pieces(piece_type, chess.WHITE))
            black_count = len(board.pieces(piece_type, chess.BLACK))
            balance += (white_count - black_count) * piece_values[piece_type]
        
        return balance
    
    def load_batch(self, size: int) -> List[Dict[str, Any]]:
        """
        Load a batch of positions from Tal games.
        
        Args:
            size: Number of positions to load
            
        Returns:
            List of position dictionaries with FEN, label, grade, source
        """
        if self._index >= len(self.positions):
            return []
        
        batch = self.positions[self._index:self._index + size]
        self._index += len(batch)
        
        return batch
    
    def reset(self):
        """Reset to beginning of dataset."""
        self._index = 0
        if self.shuffle:
            self.random.shuffle(self.positions)
    
    def get_name(self) -> str:
        """Get loader name for logging."""
        return "TalGamesLoader"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded data."""
        good_count = sum(1 for p in self.positions if p['label'] == 1)
        bad_count = len(self.positions) - good_count
        sacrifice_count = sum(1 for p in self.positions if p.get('weight', 1.0) > 1.0)
        
        return {
            'total_positions': len(self.positions),
            'games_loaded': self._games_loaded,
            'good_positions': good_count,
            'bad_positions': bad_count,
            'sacrifice_positions': sacrifice_count,
            'balance_ratio': f"{good_count}:{bad_count}",
            'filter_wins_only': self.filter_tal_wins,
            'extract_sacrifices': self.extract_sacrifices
        }


if __name__ == "__main__":
    # Test the loader
    import sys
    
    tal_pgn = Path("E:/Programming Stuff/Chess Engines/Chess Engine Playground/engine-metrics/raw_data/pgn_training_data/pgn_data_general/mikhail_tal_master_games.pgn")
    
    if not tal_pgn.exists():
        print(f"Error: Tal games file not found at {tal_pgn}")
        sys.exit(1)
    
    print("Testing TalGamesLoader...")
    print("-" * 60)
    
    loader = TalGamesLoader(
        pgn_path=str(tal_pgn),
        filter_tal_wins=True,
        extract_sacrifices=True,
        seed=42,
        shuffle=False
    )
    
    stats = loader.get_stats()
    print(f"\nLoader Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\nSample Positions (first 5):")
    print("-" * 60)
    
    sample_batch = loader.load_batch(5)
    for i, pos in enumerate(sample_batch, 1):
        print(f"\nPosition {i}:")
        print(f"  FEN: {pos['fen']}")
        print(f"  Label: {pos['label']} ({'GOOD' if pos['label'] == 1 else 'BAD'})")
        print(f"  Grade: {pos['grade']}")
        print(f"  Move: {pos['game_info']['move_number']}")
        print(f"  Sacrifice: {pos['game_info']['is_sacrifice']}")
        print(f"  Weight: {pos.get('weight', 1.0)}")
        print(f"  Game: {pos['game_info']['white']} vs {pos['game_info']['black']}")
        print(f"  Result: {pos['game_info']['result']}")
    
    print("\n" + "=" * 60)
    print("✅ TalGamesLoader test complete!")
