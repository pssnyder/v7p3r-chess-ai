"""
Supervised Learning from Grandmaster Games - "Matrix Plug-in" Knowledge Injection

Trains V7 neural network using positions from grandmaster games,
extracting features from the WINNER'S perspective only.

This provides fast baseline training (~10k games in minutes vs days of self-play)
while maintaining single-model architecture.
"""

import chess
import chess.pgn
from pathlib import Path
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
from tqdm import tqdm
import json
from dataclasses import dataclass
import logging

# V7 components
from comprehensive_features import ComprehensiveFeatureExtractor
from network import V7ValueNetwork, V7Trainer, create_v7_network
from personality_tuner import PlaystyleProfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GamePosition:
    """A position from a grandmaster game."""
    fen: str
    features: torch.Tensor
    target_eval: float  # High value for winning side
    move_number: int
    game_result: str
    

class SupervisedGMTrainer:
    """Train neural network from grandmaster games."""
    
    def __init__(
        self,
        profile_path: str,
        output_dir: str = "../training/supervised_gm",
        device: str = "cpu"
    ):
        """
        Initialize supervised trainer.
        
        Args:
            profile_path: Path to Dark Forest Assassin profile
            output_dir: Where to save models and training data
            device: "cpu" or "cuda"
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        # Load profile
        logger.info(f"Loading profile: {profile_path}")
        with open(profile_path, 'r') as f:
            profile_data = json.load(f)
        self.profile = PlaystyleProfile.from_dict(profile_data)
        
        # Initialize components
        self.extractor = ComprehensiveFeatureExtractor()
        
        # Create network and trainer
        self.network, self.trainer = create_v7_network(device=device)
        
        # Training data accumulator
        self.positions: List[GamePosition] = []
        
        # Statistics
        self.games_processed = 0
        self.positions_extracted = 0
        
    def parse_pgn_file(self, pgn_path: Path) -> List[chess.pgn.Game]:
        """
        Parse PGN file into game objects.
        
        Args:
            pgn_path: Path to cleaned PGN file
            
        Returns:
            List of chess.pgn.Game objects
        """
        games = []
        
        with open(pgn_path, 'r', encoding='utf-8') as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                games.append(game)
        
        return games
    
    def extract_positions_from_game(
        self,
        game: chess.pgn.Game,
        winner_only: bool = True,
        target_weight: float = 0.85
    ) -> List[GamePosition]:
        """
        Extract positions from a single game.
        
        Args:
            game: chess.pgn.Game object
            winner_only: Only extract from winner's perspective
            target_weight: Target evaluation for winning positions (+0.85 = strong advantage)
            
        Returns:
            List of GamePosition objects
        """
        positions = []
        
        # Get game result
        result = game.headers.get("Result", "*")
        
        if result == "*" or result == "1/2-1/2":
            # Skip incomplete or drawn games
            return positions
        
        # Determine winner
        white_won = (result == "1-0")
        
        # Play through game
        board = game.board()
        move_num = 0
        
        for node in game.mainline():
            move_num += 1
            board.push(node.move)
            
            # Skip very early positions (opening book territory)
            if move_num < 6:
                continue
            
            # Skip positions with <8 pieces (likely in tablebases)
            if len(board.piece_map()) < 8:
                continue
            
            # Determine if this is a winning position
            is_winning_perspective = False
            if winner_only:
                # White's turn and White won, OR Black's turn and Black won
                if (board.turn == chess.WHITE and white_won) or \
                   (board.turn == chess.BLACK and not white_won):
                    is_winning_perspective = True
            else:
                # Always extract, but adjust target based on result
                is_winning_perspective = True
            
            if not is_winning_perspective:
                continue
            
            # Extract features (with temporal data from move number)
            try:
                features = self.extractor.extract_all_features(
                    board, 
                    move_number=move_num,
                    previous_inference_ms=0.0  # Not available in PGN
                )
                
                # Adjust target based on game phase
                # Early game: moderate confidence (learning fundamentals)
                # Middlegame: high confidence (learning tactics)
                # Endgame: very high confidence (learning technique)
                if move_num < 15:
                    target = target_weight * 0.7  # ~0.6
                elif move_num < 35:
                    target = target_weight * 1.0  # ~0.85
                else:
                    target = target_weight * 1.1  # ~0.93 (capped at 1.0)
                
                target = min(target, 1.0)  # Cap at +1.0
                
                # If extracting from losing side, negate
                if not winner_only:
                    if (board.turn == chess.WHITE and not white_won) or \
                       (board.turn == chess.BLACK and white_won):
                        target = -target
                
                position = GamePosition(
                    fen=board.fen(),
                    features=features,
                    target_eval=target,
                    move_number=move_num,
                    game_result=result
                )
                
                positions.append(position)
                
            except Exception as e:
                logger.warning(f"Failed to extract position at move {move_num}: {e}")
                continue
        
        return positions
    
    def load_games_from_directory(
        self,
        pgn_dir: Path,
        pattern: str = "*_clean.pgn",
        max_games: Optional[int] = None,
        winner_only: bool = True
    ) -> int:
        """
        Load positions from all PGN files in directory.
        
        Args:
            pgn_dir: Directory containing cleaned PGN files
            pattern: File pattern to match
            max_games: Maximum games to process (None = all)
            winner_only: Only use winner's positions
            
        Returns:
            Number of positions extracted
        """
        pgn_dir = Path(pgn_dir)
        pgn_files = list(pgn_dir.glob(pattern))
        
        if not pgn_files:
            logger.warning(f"No PGN files found matching '{pattern}' in {pgn_dir}")
            return 0
        
        logger.info(f"Found {len(pgn_files)} PGN files")
        
        positions_before = len(self.positions)
        games_processed = 0
        
        for pgn_file in tqdm(pgn_files, desc="Processing PGN files"):
            try:
                games = self.parse_pgn_file(pgn_file)
                
                for game in games:
                    if max_games and games_processed >= max_games:
                        break
                    
                    positions = self.extract_positions_from_game(
                        game,
                        winner_only=winner_only
                    )
                    
                    self.positions.extend(positions)
                    games_processed += 1
                
                if max_games and games_processed >= max_games:
                    break
                    
            except Exception as e:
                logger.error(f"Failed to process {pgn_file.name}: {e}")
        
        positions_added = len(self.positions) - positions_before
        self.games_processed += games_processed
        self.positions_extracted += positions_added
        
        logger.info(f"✓ Extracted {positions_added} positions from {games_processed} games")
        
        return positions_added
    
    def train_on_positions(
        self,
        epochs: int = 5,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        save_interval: int = 1
    ) -> List[float]:
        """
        Train network on extracted positions.
        
        Args:
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            save_interval: Save model every N epochs
            
        Returns:
            List of epoch losses
        """
        if not self.positions:
            raise ValueError("No positions loaded - call load_games_from_directory() first")
        
        logger.info(f"Training on {len(self.positions)} positions")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
        
        # Prepare data - convert numpy arrays to tensors
        features = torch.stack([torch.from_numpy(p.features).float() for p in self.positions])
        targets = torch.tensor([p.target_eval for p in self.positions], dtype=torch.float32)
        
        # Move to device
        features = features.to(self.device)
        targets = targets.to(self.device).unsqueeze(1)
        
        # Optimizer
        optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        epoch_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Shuffle data
            indices = torch.randperm(len(features))
            features_shuffled = features[indices]
            targets_shuffled = targets[indices]
            
            # Mini-batch training
            for i in range(0, len(features), batch_size):
                batch_features = features_shuffled[i:i+batch_size]
                batch_targets = targets_shuffled[i:i+batch_size]
                
                # Forward pass
                optimizer.zero_grad()
                predictions = self.network(batch_features)
                loss = criterion(predictions, batch_targets)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # Average loss
            avg_loss = epoch_loss / num_batches
            epoch_losses.append(avg_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                checkpoint_path = self.output_dir / f"supervised_epoch_{epoch+1}.pt"
                torch.save(self.network.state_dict(), checkpoint_path)
                logger.info(f"✓ Saved checkpoint: {checkpoint_path.name}")
        
        # Save final model
        final_path = self.output_dir / "supervised_final.pt"
        torch.save(self.network.state_dict(), final_path)
        logger.info(f"✓ Saved final model: {final_path.name}")
        
        return epoch_losses
    
    def save_training_stats(self):
        """Save training statistics to JSON."""
        stats = {
            'games_processed': self.games_processed,
            'positions_extracted': self.positions_extracted,
            'output_dir': str(self.output_dir),
            'network_parameters': sum(p.numel() for p in self.network.parameters()),
        }
        
        stats_path = self.output_dir / "training_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"✓ Saved stats: {stats_path.name}")


def main():
    """Example: Train from cleaned GM games."""
    print("=" * 80)
    print("V7P3R SUPERVISED LEARNING - \"Matrix Plug-in\" Knowledge Injection")
    print("=" * 80)
    print()
    print("This will train the neural network from grandmaster games,")
    print("extracting features from WINNING positions only.")
    print()
    print("Expected: ~10,000 positions from ~500-1000 games")
    print("Training time: 5-15 minutes (vs 4-6 hours for self-play)")
    print()
    print("=" * 80)
    print()
    
    # Configuration
    PROFILE_PATH = "../profiles/dark_forest_assassin.json"
    OUTPUT_DIR = "../training/supervised_gm"
    
    # PGN directories (assuming cleaned versions exist)
    PGN_DIRS = [
        Path(r"E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\pgn_training_data\pgn_data_important_games\cleaned"),
        Path(r"E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\pgn_training_data\pgn_data_tactics\cleaned"),
    ]
    
    # Initialize trainer
    trainer = SupervisedGMTrainer(
        profile_path=PROFILE_PATH,
        output_dir=OUTPUT_DIR
    )
    
    # Load games
    print("Loading positions from cleaned PGN files...")
    print("-" * 80)
    
    for pgn_dir in PGN_DIRS:
        if pgn_dir.exists():
            trainer.load_games_from_directory(
                pgn_dir,
                pattern="*_clean.pgn",
                max_games=None,  # Load all
                winner_only=True
            )
    
    print()
    print("=" * 80)
    print(f"Total positions extracted: {trainer.positions_extracted}")
    print(f"Total games processed: {trainer.games_processed}")
    print("=" * 80)
    print()
    
    if trainer.positions_extracted == 0:
        print("ERROR: No positions extracted!")
        print()
        print("Make sure you've run pgn_preprocessor.py first to create cleaned files.")
        return
    
    input("Press ENTER to start training (or Ctrl+C to cancel)...")
    print()
    
    # Train
    print("=" * 80)
    print("TRAINING")
    print("=" * 80)
    
    losses = trainer.train_on_positions(
        epochs=10,
        batch_size=256,
        learning_rate=0.001,
        save_interval=2
    )
    
    # Save stats
    trainer.save_training_stats()
    
    print()
    print("=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Model saved to: {trainer.output_dir}")
    print()
    print("You can now use this as Generation 0 for generational training!")
    print("=" * 80)


if __name__ == "__main__":
    main()
