"""
V7P3R v8.0 - Opening Selector

Loads opening book and enables model to choose opening variations as meta-actions.
Each opening is executed as a macro (10-20 moves instantly), then model plays from there.
"""

import json
import logging
import chess
import random
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class OpeningSelector:
    """Manages opening book and executes opening variations"""
    
    def __init__(self, book_path: str = 'opening_book.json'):
        """
        Args:
            book_path: Path to opening_book.json file
        """
        self.book_path = Path(book_path)
        self.openings = []
        self.num_openings = 0
        
        self.load_book()
    
    def load_book(self):
        """Load opening book from JSON"""
        if not self.book_path.exists():
            raise FileNotFoundError(f"Opening book not found: {self.book_path}")
        
        with open(self.book_path, 'r', encoding='utf-8') as f:
            book_data = json.load(f)
        
        self.openings = book_data['openings']
        self.num_openings = len(self.openings)
        
        logging.info(f"Loaded {self.num_openings} openings from {self.book_path.name}")
    
    def get_opening(self, opening_id: int) -> Dict:
        """
        Get opening data by ID
        
        Args:
            opening_id: Integer 0 to num_openings-1
        
        Returns:
            Opening dictionary with 'name', 'moves', 'ply_count'
        """
        if not 0 <= opening_id < self.num_openings:
            raise ValueError(f"Invalid opening ID: {opening_id}. Must be 0-{self.num_openings-1}")
        
        return self.openings[opening_id]
    
    def execute_opening(self, board: chess.Board, opening_id: int) -> Tuple[chess.Board, int, str]:
        """
        Execute full opening variation on board
        
        Args:
            board: Chess board (should be starting position)
            opening_id: Which opening to play
        
        Returns:
            (updated_board, ply_count, opening_name)
        """
        opening = self.get_opening(opening_id)
        
        for move_uci in opening['moves']:
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    logging.warning(f"Illegal move in opening {opening['name']}: {move_uci}")
                    break
            except Exception as e:
                logging.warning(f"Error executing move {move_uci} in {opening['name']}: {e}")
                break
        
        return board, opening['ply_count'], opening['name']
    
    def random_opening(self) -> int:
        """Select random opening ID for exploration"""
        return random.randint(0, self.num_openings - 1)


class OpeningSelectorNetwork(nn.Module):
    """
    Neural network head for selecting opening variations
    
    Learns which openings lead to favorable positions through experience.
    """
    
    def __init__(self, num_openings: int = 100, hidden_dim: int = 128):
        """
        Args:
            num_openings: Number of opening variations in book
            hidden_dim: Size of hidden layer
        """
        super().__init__()
        
        self.num_openings = num_openings
        
        # Simple network: starting position features → opening choice probabilities
        # Input: 1 (just a bias - starting position is always the same)
        # Could expand to accept opponent style features later
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_openings)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, 1)
        
        Returns:
            Opening probabilities (batch_size, num_openings)
        """
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)
    
    def select_opening(self, temperature: float = 0.3, epsilon: float = 0.2) -> int:
        """
        Select opening variation with exploration
        
        Args:
            temperature: Lower = more deterministic (0.1-0.5 typical)
            epsilon: Probability of random exploration (0.1-0.3 typical)
        
        Returns:
            opening_id: Integer 0 to num_openings-1
        """
        # Epsilon-greedy exploration
        if random.random() < epsilon:
            return random.randint(0, self.num_openings - 1)
        
        # Use network to choose
        with torch.no_grad():
            x = torch.tensor([[1.0]])  # Dummy input (starting position)
            probs = self.forward(x).squeeze()
            
            # Apply temperature
            if temperature > 0:
                probs = probs ** (1 / temperature)
                probs = probs / probs.sum()
                
                # Sample from distribution
                opening_id = torch.multinomial(probs, 1).item()
            else:
                # Greedy: choose highest probability
                opening_id = probs.argmax().item()
        
        return opening_id
    
    def update_from_result(self, opening_id: int, reward: float, lr: float = 0.01):
        """
        Update opening selection preferences based on game result
        
        Args:
            opening_id: Which opening was used
            reward: Game outcome (+1 win, 0 draw, -1 loss)
            lr: Learning rate
        """
        # Simple REINFORCE-style update
        x = torch.tensor([[1.0]])
        probs = self.forward(x).squeeze()
        
        # Increase probability of winning openings, decrease losing ones
        target = probs.clone().detach()
        target[opening_id] += lr * reward
        target = target / target.sum()  # Renormalize
        
        # Update weights (could use proper optimizer, but this is simple)
        loss = nn.functional.kl_div(probs.log(), target, reduction='batchmean')
        loss.backward()


class OpeningDiversityTracker:
    """Track which openings are being used and their win rates"""
    
    def __init__(self, num_openings: int):
        self.num_openings = num_openings
        self.usage_counts = [0] * num_openings
        self.win_counts = [0] * num_openings
        self.draw_counts = [0] * num_openings
        self.loss_counts = [0] * num_openings
    
    def record_game(self, opening_id: int, result: str):
        """
        Record game result for an opening
        
        Args:
            opening_id: Which opening was used
            result: "1-0", "0-1", or "1/2-1/2"
        """
        self.usage_counts[opening_id] += 1
        
        if result == "1-0":
            self.win_counts[opening_id] += 1
        elif result == "0-1":
            self.loss_counts[opening_id] += 1
        else:
            self.draw_counts[opening_id] += 1
    
    def get_win_rate(self, opening_id: int) -> float:
        """Calculate win rate for opening"""
        total = self.usage_counts[opening_id]
        if total == 0:
            return 0.0
        return self.win_counts[opening_id] / total
    
    def get_most_used(self, top_k: int = 10) -> List[Tuple[int, int]]:
        """Get most frequently used openings"""
        indexed_counts = [(i, count) for i, count in enumerate(self.usage_counts)]
        return sorted(indexed_counts, key=lambda x: -x[1])[:top_k]
    
    def get_highest_win_rate(self, min_games: int = 5, top_k: int = 10) -> List[Tuple[int, float]]:
        """Get openings with highest win rates (minimum games required)"""
        win_rates = []
        for i in range(self.num_openings):
            if self.usage_counts[i] >= min_games:
                win_rate = self.get_win_rate(i)
                win_rates.append((i, win_rate))
        
        return sorted(win_rates, key=lambda x: -x[1])[:top_k]
    
    def print_summary(self, opening_selector: OpeningSelector, top_k: int = 10):
        """Print usage statistics"""
        print("\n" + "="*60)
        print("OPENING USAGE SUMMARY")
        print("="*60)
        
        print("\nMost Used Openings:")
        for i, (opening_id, count) in enumerate(self.get_most_used(top_k), 1):
            opening = opening_selector.get_opening(opening_id)
            win_rate = self.get_win_rate(opening_id)
            print(f"{i:2d}. {opening['name']:40s} {count:3d} games ({win_rate:.1%} wins)")
        
        print("\nHighest Win Rates (min 5 games):")
        for i, (opening_id, win_rate) in enumerate(self.get_highest_win_rate(5, top_k), 1):
            opening = opening_selector.get_opening(opening_id)
            count = self.usage_counts[opening_id]
            print(f"{i:2d}. {opening['name']:40s} {win_rate:.1%} ({count} games)")
        
        print("="*60)


def test_opening_selector():
    """Test opening selector functionality"""
    print("Testing Opening Selector...")
    
    # Load opening book
    selector = OpeningSelector()
    
    # Test random opening selection
    opening_id = selector.random_opening()
    print(f"\nRandom opening ID: {opening_id}")
    
    opening = selector.get_opening(opening_id)
    print(f"Opening: {opening['name']}")
    print(f"Moves: {len(opening['moves'])} half-moves")
    print(f"First 6 moves: {' '.join(opening['moves'][:6])}")
    
    # Test opening execution
    board = chess.Board()
    board, ply_count, name = selector.execute_opening(board, opening_id)
    
    print(f"\nAfter executing opening:")
    print(f"  Name: {name}")
    print(f"  Ply count: {ply_count}")
    print(f"  Position: {board.fen()}")
    
    # Test diversity tracker
    tracker = OpeningDiversityTracker(selector.num_openings)
    
    # Simulate some games
    for _ in range(50):
        op_id = selector.random_opening()
        result = random.choice(["1-0", "1/2-1/2", "0-1"])
        tracker.record_game(op_id, result)
    
    tracker.print_summary(selector, top_k=5)
    
    print("\n✓ Opening selector tests passed!")


if __name__ == '__main__':
    test_opening_selector()
