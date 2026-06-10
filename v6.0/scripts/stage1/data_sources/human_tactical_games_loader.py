"""
Human Tactical Games Loader - Extract tactical patterns from v7p3r human games.

Loads positions from YOUR Chess.com games, focusing on:
- Bxf7+ king hunt sacrifices (signature move - 5.0x weight)
- Quick tactical wins (under 25 moves)
- Aggressive attacking sequences
- Piece sacrifices for initiative

Philosophy: Teach the AI YOUR actual tactical style, not aspirational theory.
"""

import chess
import chess.pgn
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
import re
import sys

# Handle both relative and absolute imports
try:
    from .base_loader import DataSourceLoader
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from base_loader import DataSourceLoader


class HumanTacticalGamesLoader(DataSourceLoader):
    """Load tactical positions from v7p3r human games."""
    
    # Tactical game filters
    MAX_MOVES_FOR_QUICK_WIN = 25  # Quick tactical wins
    MIN_MOVE_EXTRACT = 8   # Skip opening book
    MAX_MOVE_EXTRACT = 40  # Stop before pure endgame
    
    # Signature patterns
    BXFF_PATTERN = r'Bxf[27]\+'  # Bishop takes f7/f2 check (signature sacrifice)
    PIECE_SACRIFICE_THRESHOLD = 300  # Material sacrifice ≥ knight/bishop
    
    def __init__(
        self,
        pgn_path: str,
        filter_wins_only: bool = True,
        prioritize_bxf7: bool = True,
        prioritize_quick_wins: bool = True,
        seed: int = 42,
        shuffle: bool = True
    ):
        """
        Initialize human tactical games loader.
        
        Args:
            pgn_path: Path to v7p3r_20250530.pgn (Chess.com games)
            filter_wins_only: Only extract from games you won
            prioritize_bxf7: Weight Bxf7+ patterns at 5.0x (signature move)
            prioritize_quick_wins: Weight wins under 25 moves at 2.0x
            seed: Random seed for reproducibility
            shuffle: Whether to shuffle positions
        """
        super().__init__(seed=seed, shuffle=shuffle)
        
        self.pgn_path = Path(pgn_path)
        self.filter_wins_only = filter_wins_only
        self.prioritize_bxf7 = prioritize_bxf7
        self.prioritize_quick_wins = prioritize_quick_wins
        
        # Initialize index
        self._index = 0
        
        if not self.pgn_path.exists():
            raise FileNotFoundError(f"Human games PGN not found: {pgn_path}")
        
        # Load all tactical positions
        self.positions = []
        self._load_all_positions()
        
        # Shuffle if requested
        if self.shuffle:
            self.random.shuffle(self.positions)
        
        print(f"Loaded {len(self.positions)} positions from {self._games_loaded} human games")
        print(f"  Bxf7+ patterns found: {self._bxf7_games}")
        print(f"  Quick tactical wins: {self._quick_wins}")
    
    def _load_all_positions(self):
        """Extract all tactical positions from human games."""
        self._games_loaded = 0
        self._positions_extracted = 0
        self._bxf7_games = 0
        self._quick_wins = 0
        
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
                    print(f"Warning: Skipped game due to error: {e}")
                    continue
    
    def _extract_positions_from_game(self, game: chess.pgn.Game) -> List[Dict[str, Any]]:
        """
        Extract tactical positions from a single human game.
        
        Strategy:
        - Filter for v7p3r wins (YOUR wins only)
        - Detect Bxf7+ king hunt patterns (weight 5.0x)
        - Extract from tactical phase (moves 8-40)
        - Label your attacking positions as GOOD (1)
        - Prioritize quick wins (under 25 moves, weight 2.0x)
        
        Args:
            game: chess.pgn.Game object
            
        Returns:
            List of position dictionaries with FEN, label, grade, source, weight
        """
        positions = []
        
        # Get game metadata
        white = game.headers.get("White", "")
        black = game.headers.get("Black", "")
        result = game.headers.get("Result", "*")
        
        # Determine if v7p3r (human) played and won
        v7p3r_is_white = "v7p3r" in white.lower()
        v7p3r_is_black = "v7p3r" in black.lower()
        
        if not (v7p3r_is_white or v7p3r_is_black):
            return []  # Not a v7p3r game
        
        # Filter for wins only if enabled
        if self.filter_wins_only:
            if v7p3r_is_white and result != "1-0":
                return []  # Didn't win as White
            if v7p3r_is_black and result != "0-1":
                return []  # Didn't win as Black
        
        # Detect signature tactical patterns
        game_moves_san = []
        has_bxf7_sacrifice = False
        is_quick_win = False
        
        # Count total moves
        board = game.board()
        move_count = sum(1 for _ in game.mainline_moves())
        
        # Check if quick tactical win
        if move_count <= self.MAX_MOVES_FOR_QUICK_WIN:
            is_quick_win = True
            self._quick_wins += 1
        
        # Build move list and detect Bxf7+
        node = game
        for move in game.mainline_moves():
            san = board.san(move)
            game_moves_san.append(san)
            
            # Check for Bxf7+ or Bxf2+ (signature sacrifice)
            if re.match(self.BXFF_PATTERN, san):
                has_bxf7_sacrifice = True
                self._bxf7_games += 1
            
            board.push(move)
        
        # Extract positions from tactical phase
        board = game.board()
        move_number = 0
        bxf7_move_number = None
        
        for i, move in enumerate(game.mainline_moves()):
            san = game_moves_san[i]
            board.push(move)
            move_number += 1
            
            # Track where Bxf7+ occurred
            if re.match(self.BXFF_PATTERN, san):
                bxf7_move_number = move_number
            
            # Only extract from tactical middlegame
            if move_number < self.MIN_MOVE_EXTRACT or move_number > self.MAX_MOVE_EXTRACT:
                continue
            
            # Determine whose turn it is
            is_v7p3r_turn = (v7p3r_is_white and board.turn == chess.WHITE) or \
                           (v7p3r_is_black and board.turn == chess.BLACK)
            
            # Label positions
            if is_v7p3r_turn:
                # Your turn in a winning game → GOOD position
                label = 1
                grade = 1  # Excellent (your winning position)
            else:
                # Opponent's turn in a losing position → BAD for them
                label = 0
                grade = 5  # Poor (opponent in bad position)
            
            # Calculate base weight
            weight = 1.0
            
            # Apply Bxf7+ weight multiplier (5.0x for positions around the sacrifice)
            if has_bxf7_sacrifice and bxf7_move_number is not None:
                # Weight positions within 5 moves of Bxf7+ sacrifice
                distance_from_sacrifice = abs(move_number - bxf7_move_number)
                if distance_from_sacrifice <= 5:
                    weight = 5.0  # Signature tactical pattern
            
            # Apply quick win multiplier (2.0x for fast tactical wins)
            elif is_quick_win:
                weight = 2.0
            
            # Create position record
            position = {
                'fen': board.fen(),
                'label': label,
                'grade': grade,
                'source': 'human_tactical_games',
                'weight': weight,
                'game_info': {
                    'white': white,
                    'black': black,
                    'result': result,
                    'move_number': move_number,
                    'move_san': san,
                    'has_bxf7': has_bxf7_sacrifice,
                    'is_quick_win': is_quick_win,
                    'v7p3r_color': 'white' if v7p3r_is_white else 'black',
                    'total_moves': move_count
                }
            }
            
            positions.append(position)
        
        return positions
    
    def load_batch(self, size: int) -> List[Dict[str, Any]]:
        """
        Load a batch of positions from human tactical games.
        
        Args:
            size: Number of positions to load
            
        Returns:
            List of position dictionaries with FEN, label, grade, source, weight
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
        return "HumanTacticalGamesLoader"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded data."""
        good_count = sum(1 for p in self.positions if p['label'] == 1)
        bad_count = len(self.positions) - good_count
        
        # Count weighted positions
        bxf7_positions = sum(1 for p in self.positions if p.get('weight', 1.0) >= 5.0)
        quick_win_positions = sum(1 for p in self.positions if p.get('weight', 1.0) == 2.0)
        
        # Average weight
        avg_weight = sum(p.get('weight', 1.0) for p in self.positions) / len(self.positions) if self.positions else 0
        
        return {
            'total_positions': len(self.positions),
            'games_loaded': self._games_loaded,
            'good_positions': good_count,
            'bad_positions': bad_count,
            'balance_ratio': f"{good_count}:{bad_count}",
            'bxf7_games': self._bxf7_games,
            'bxf7_weighted_positions': bxf7_positions,
            'quick_wins': self._quick_wins,
            'quick_win_positions': quick_win_positions,
            'average_weight': f"{avg_weight:.2f}",
            'filter_wins_only': self.filter_wins_only
        }


if __name__ == "__main__":
    # Test the loader
    import sys
    
    human_pgn = Path("E:/Programming Stuff/Chess Engines/Chess Engine Playground/engine-metrics/raw_data/game_records/v7p3r Human/v7p3r_20250530.pgn")
    
    if not human_pgn.exists():
        print(f"Error: Human games file not found at {human_pgn}")
        sys.exit(1)
    
    print("Testing HumanTacticalGamesLoader...")
    print("=" * 60)
    
    loader = HumanTacticalGamesLoader(
        pgn_path=str(human_pgn),
        filter_wins_only=True,
        prioritize_bxf7=True,
        prioritize_quick_wins=True,
        seed=42,
        shuffle=False
    )
    
    stats = loader.get_stats()
    print(f"\nLoader Statistics:")
    print("-" * 60)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n🎯 Signature Tactical Patterns:")
    print("-" * 60)
    
    # Find and display Bxf7+ positions
    bxf7_positions = [p for p in loader.positions if p.get('weight', 1.0) >= 5.0]
    
    if bxf7_positions:
        print(f"\n⭐ Found {len(bxf7_positions)} Bxf7+ KING HUNT positions (weight 5.0x)!")
        print("\nSample Bxf7+ Positions (first 3):")
        print("-" * 60)
        
        for i, pos in enumerate(bxf7_positions[:3], 1):
            print(f"\nBxf7+ Position {i}:")
            print(f"  FEN: {pos['fen']}")
            print(f"  Label: {pos['label']} ({'GOOD - YOUR ATTACK' if pos['label'] == 1 else 'BAD - OPPONENT'})")
            print(f"  Grade: {pos['grade']}")
            print(f"  Weight: {pos['weight']}x (SIGNATURE TACTICAL PATTERN)")
            print(f"  Move: {pos['game_info']['move_number']} - {pos['game_info']['move_san']}")
            print(f"  Game: {pos['game_info']['white']} vs {pos['game_info']['black']}")
            print(f"  Result: {pos['game_info']['result']} (Total moves: {pos['game_info']['total_moves']})")
    else:
        print("  No Bxf7+ patterns found in wins (checking all games...)")
    
    print(f"\n📊 General Tactical Positions (first 5):")
    print("-" * 60)
    
    sample_batch = loader.load_batch(5)
    for i, pos in enumerate(sample_batch, 1):
        print(f"\nPosition {i}:")
        print(f"  FEN: {pos['fen']}")
        print(f"  Label: {pos['label']} ({'GOOD' if pos['label'] == 1 else 'BAD'})")
        print(f"  Weight: {pos.get('weight', 1.0)}x")
        print(f"  Move: {pos['game_info']['move_number']}")
        print(f"  Quick win: {pos['game_info']['is_quick_win']}")
        print(f"  Game: {pos['game_info']['white']} vs {pos['game_info']['black']}")
    
    print("\n" + "=" * 60)
    print("✅ HumanTacticalGamesLoader test complete!")
    print("\n🎯 Key Insights:")
    print(f"  • Your signature Bxf7+ king hunts will train the AI at 5.0x weight")
    print(f"  • Quick tactical wins (≤{loader.MAX_MOVES_FOR_QUICK_WIN} moves) weighted at 2.0x")
    print(f"  • Teaching the AI YOUR actual aggressive style, not theory")
