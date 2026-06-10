"""
Multi-Source Data Loader - orchestrates mixing data from multiple sources.

TAL-INSPIRED MIXING (v6.1 - Tactical Focus):
- 20% Tal Games (GM chaos mastery - NEW)
- 15% Human Games (YOUR Bxf7+ king hunts - NEW)
- 15% Tactics (pattern recognition - INCREASED)
- 15% Openings (aggressive repertoire - INCREASED)
- 20% Lichess DB (general knowledge - DECREASED)
- 10% V7P3R Engine (baseline - MAINTAINED)
- 5% Endgames (conversion - MAINTAINED)

Philosophy: 50% tactical focus (Tal + Human + Tactics) to train aggressive chess.
Bxf7+ king hunt patterns weighted 5.0x (signature tactical fingerprint).
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import random

from .base_loader import DataSourceLoader
from .lichess_loader import LichessDBLoader
from .opening_loader import OpeningPGNLoader
from .v7p3r_loader import V7P3RGameLoader
from .tactics_loader import TacticsLoader
from .endgame_loader import EndgameLoader
from .tal_games_loader import TalGamesLoader
from .human_tactical_games_loader import HumanTacticalGamesLoader


class MultiSourceDataLoader:
    """Orchestrate loading from multiple data sources with configurable mixing."""
    
    DEFAULT_MIX = {
        'lichess': 0.70,
        'v7p3r': 0.10,
        'openings': 0.10,
        'tactics': 0.05,
        'endgames': 0.05
    }
    
    TAL_INSPIRED_MIX = {
        'tal_games': 0.20,      # GM tactical mastery
        'human_games': 0.15,    # YOUR Bxf7+ signature
        'tactics': 0.15,        # Pattern recognition
        'openings': 0.15,       # Aggressive repertoire
        'lichess': 0.20,        # General knowledge
        'v7p3r': 0.10,          # Engine baseline
        'endgames': 0.05        # Conversion skills
    }
    
    def __init__(
        self,
        lichess_db_path: str,
        v7p3r_bad_positions: str,
        opening_pgn_dir: str,
        tactics_csv_path: str,
        endgame_pgn_dir: str,
        tal_games_pgn: Optional[str] = None,
        human_games_pgn: Optional[str] = None,
        v7p3r_pgn_dir: Optional[str] = None,
        mix_ratios: Optional[Dict[str, float]] = None,
        use_tal_mix: bool = False,
        seed: int = 42,
        shuffle: bool = True,
        preferred_opening_weight: float = 1.5
    ):
        """
        Initialize multi-source data loader.
        
        Args:
            lichess_db_path: Path to lichess_db_eval.jsonl
            v7p3r_bad_positions: Path to v7p3r_bad_positions.jsonl
            opening_pgn_dir: Directory with opening PGN files
            tactics_csv_path: Path to tactics CSV file(s)
            endgame_pgn_dir: Directory with endgame PGN files
            tal_games_pgn: Path to mikhail_tal_master_games.pgn (NEW)
            human_games_pgn: Path to v7p3r_20250530.pgn (YOUR games, NEW)
            v7p3r_pgn_dir: Optional directory with V7P3R vs V7P3R games
            mix_ratios: Custom mixing ratios (if None, uses DEFAULT_MIX or TAL_INSPIRED_MIX)
            use_tal_mix: Use TAL_INSPIRED_MIX ratios (20/15/15/15/20/10/5, default: False)
            seed: Random seed for reproducibility
            shuffle: Whether to shuffle final batches
            preferred_opening_weight: Weight multiplier for preferred openings
        """
        self.seed = seed
        self.shuffle = shuffle
        self.random = random.Random(seed)
        self.preferred_opening_weight = preferred_opening_weight
        
        # Choose mix ratios
        if mix_ratios:
            self.mix_ratios = mix_ratios
        elif use_tal_mix:
            self.mix_ratios = self.TAL_INSPIRED_MIX.copy()
            print("\n🎯 Using TAL-INSPIRED mixing (50% tactical focus)...")
        else:
            self.mix_ratios = self.DEFAULT_MIX.copy()
        
        # Validate mix ratios sum to 1.0 (with floating point tolerance)
        total = sum(self.mix_ratios.values())
        if abs(total - 1.0) > 0.02:  # Allow 2% tolerance for floating point precision
            raise ValueError(f"Mix ratios must sum to 1.0, got {total}")
        
        # Normalize to exactly 1.0 to avoid precision issues
        if total > 0:
            self.mix_ratios = {k: v/total for k, v in self.mix_ratios.items()}
        
        # Initialize all loaders
        self.loaders = {}
        
        # Tal Games Loader (NEW - GM tactical mastery)
        if tal_games_pgn:
            try:
                self.loaders['tal_games'] = TalGamesLoader(
                    pgn_path=tal_games_pgn,
                    filter_tal_wins=True,
                    extract_sacrifices=True,
                    seed=seed,
                    shuffle=shuffle
                )
                print("✓ Loaded Tal games (GM chaos mastery)")
            except FileNotFoundError:
                print(f"Warning: Tal games not found at {tal_games_pgn}, skipping")
                if 'tal_games' in self.mix_ratios:
                    self.mix_ratios['tal_games'] = 0
        elif 'tal_games' in self.mix_ratios:
            print("Warning: Tal games path not provided, skipping")
            self.mix_ratios['tal_games'] = 0
        
        # Human Tactical Games Loader (NEW - YOUR Bxf7+ signature)
        if human_games_pgn:
            try:
                self.loaders['human_games'] = HumanTacticalGamesLoader(
                    pgn_path=human_games_pgn,
                    filter_wins_only=True,
                    prioritize_bxf7=True,
                    prioritize_quick_wins=True,
                    seed=seed,
                    shuffle=shuffle
                )
                print("✓ Loaded human games (YOUR tactical style)")
            except FileNotFoundError:
                print(f"Warning: Human games not found at {human_games_pgn}, skipping")
                if 'human_games' in self.mix_ratios:
                    self.mix_ratios['human_games'] = 0
        elif 'human_games' in self.mix_ratios:
            print("Warning: Human games path not provided, skipping")
            self.mix_ratios['human_games'] = 0
        
        try:
            self.loaders['lichess'] = LichessDBLoader(
                db_path=lichess_db_path,
                seed=seed,
                shuffle=shuffle
            )
            print("✓ Loaded Lichess DB (general knowledge)")
        except FileNotFoundError:
            print(f"Warning: Lichess DB not found at {lichess_db_path}, skipping")
            self.mix_ratios['lichess'] = 0
        
        try:
            self.loaders['v7p3r'] = V7P3RGameLoader(
                bad_positions_jsonl=v7p3r_bad_positions,
                pgn_dir=v7p3r_pgn_dir,
                seed=seed,
                shuffle=shuffle,
                include_good_moves=True
            )
            print("✓ Loaded V7P3R engine games (baseline)")
        except FileNotFoundError:
            print(f"Warning: V7P3R positions not found, skipping")
            self.mix_ratios['v7p3r'] = 0
        
        try:
            self.loaders['openings'] = OpeningPGNLoader(
                pgn_dir=opening_pgn_dir,
                seed=seed,
                shuffle=shuffle,
                preferred_only=True
            )
            print("✓ Loaded opening repertoire (aggressive openings)")
        except (FileNotFoundError, ValueError):
            print(f"Warning: Opening PGNs not found at {opening_pgn_dir}, skipping")
            self.mix_ratios['openings'] = 0
        
        try:
            self.loaders['tactics'] = TacticsLoader(
                csv_path=tactics_csv_path,
                seed=seed,
                shuffle=shuffle
            )
            print("✓ Loaded tactics puzzles (pattern recognition)")
        except (FileNotFoundError, ValueError):
            print(f"Warning: Tactics data not found at {tactics_csv_path}, skipping")
            self.mix_ratios['tactics'] = 0
        
        try:
            self.loaders['endgames'] = EndgameLoader(
                pgn_dir=endgame_pgn_dir,
                seed=seed,
                shuffle=shuffle
            )
            print("✓ Loaded endgame databases (conversion skills)")
        except (FileNotFoundError, ValueError):
            print(f"Warning: Endgame PGNs not found at {endgame_pgn_dir}, skipping")
            self.mix_ratios['endgames'] = 0
        
        # Renormalize mix ratios after removing missing sources
        active_ratios = {k: v for k, v in self.mix_ratios.items() if v > 0}
        total = sum(active_ratios.values())
        if total > 0:
            self.mix_ratios = {k: v/total for k, v in active_ratios.items()}
        else:
            raise ValueError("No valid data sources found!")
        
        self._total_loaded = 0
        
    def _backfill_missing_evals(self, positions: List[Dict[str, Any]]) -> int:
        """Emergency Stockfish backfill for positions missing eval_cp.
        
        Returns:
            Number of positions backfilled
        """
        missing_eval = [p for p in positions if 'eval_cp' not in p or p['eval_cp'] is None]
        
        if not missing_eval:
            return 0
        
        print(f"  Backfilling {len(missing_eval)} positions with missing evals...")
        
        # Import here to avoid circular dependency
        from scripts.stage1.stockfish_validator import StockfishValidator
        
        # Create validator if not already available
        if not hasattr(self, '_stockfish_validator'):
            self._stockfish_validator = StockfishValidator(
                stockfish_path="stockfish/stockfish.exe",
                db_path="data/stage1/stockfish_cache.db"
            )
        
        # Batch validate
        eval_results = self._stockfish_validator.validate_batch(missing_eval)
        
        # Update positions with eval results
        for pos, result in zip(missing_eval, eval_results):
            pos['eval_cp'] = result['eval_cp']
            pos['grade'] = result['grade']
        
        return len(missing_eval)
    
    def load_batch(self, size: int, target_balance: Optional[Dict[int, float]] = None) -> List[Dict[str, Any]]:
        """
        Load a mixed batch from all data sources.
        
        Args:
            size: Total number of positions to load
            target_balance: Optional target balance for labels (e.g., {0: 0.5, 1: 0.5})
            
        Returns:
            List of position dictionaries from mixed sources
        """
        batch = []
        
        # Calculate how many positions to load from each source
        source_sizes = {}
        for source, ratio in self.mix_ratios.items():
            if ratio > 0 and source in self.loaders:
                source_sizes[source] = int(size * ratio)
        
        # Adjust for rounding errors
        total_allocated = sum(source_sizes.values())
        if total_allocated < size:
            # Add remaining to largest source
            largest = max(source_sizes.items(), key=lambda x: x[1])[0]
            source_sizes[largest] += (size - total_allocated)
        
        # Load from each source
        for source, count in source_sizes.items():
            if count > 0:
                try:
                    positions = self.loaders[source].load_batch(count)
                    
                    # Apply preferred opening weight
                    if source == 'openings':
                        # Duplicate some positions to give them more weight
                        extra_count = int(len(positions) * (self.preferred_opening_weight - 1.0))
                        if extra_count > 0:
                            duplicates = self.random.sample(positions, min(extra_count, len(positions)))
                            positions.extend(duplicates)
                    
                    batch.extend(positions)
                    
                except Exception as e:
                    print(f"Warning: Failed to load from {source}: {e}")
                    continue
        
        # CRITICAL: Backfill missing evals before training
        backfilled_count = self._backfill_missing_evals(batch)
        if backfilled_count > 0:
            print(f"  ✓ Backfilled {backfilled_count} missing evaluations")
        
        # Balance labels if target specified
        if target_balance:
            batch = self._balance_labels(batch, target_balance)
        
        # Final shuffle
        if self.shuffle:
            self.random.shuffle(batch)
        
        self._total_loaded += len(batch)
        return batch
        
    def _balance_labels(
        self,
        positions: List[Dict[str, Any]],
        target_balance: Dict[int, float]
    ) -> List[Dict[str, Any]]:
        """
        Balance position labels to match target distribution.
        
        Args:
            positions: List of position dictionaries
            target_balance: Target distribution (e.g., {0: 0.5, 1: 0.5})
            
        Returns:
            Balanced list of positions
        """
        # Separate by label
        by_label = {}
        for pos in positions:
            label = pos['label']
            if label not in by_label:
                by_label[label] = []
            by_label[label].append(pos)
        
        # Calculate target counts
        total = len(positions)
        target_counts = {label: int(total * ratio) for label, ratio in target_balance.items()}
        
        # Sample to meet targets
        balanced = []
        for label, target_count in target_counts.items():
            if label in by_label:
                available = by_label[label]
                if len(available) >= target_count:
                    # Sample down
                    balanced.extend(self.random.sample(available, target_count))
                else:
                    # Use all available and oversample
                    balanced.extend(available)
                    needed = target_count - len(available)
                    balanced.extend(self.random.choices(available, k=needed))
        
        return balanced
        
    def reset_all(self):
        """Reset all loaders to beginning."""
        for loader in self.loaders.values():
            loader.reset()
        self._total_loaded = 0
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about all loaders.
        
        Returns:
            Dictionary with statistics for each source
        """
        stats = {
            'total_loaded': self._total_loaded,
            'mix_ratios': self.mix_ratios,
            'sources': {}
        }
        
        for name, loader in self.loaders.items():
            stats['sources'][name] = loader.get_stats()
        
        return stats
        
    def print_summary(self):
        """Print summary of data sources and mixing."""
        print("\n" + "="*60)
        print("Multi-Source Data Loader Summary")
        print("="*60)
        print(f"Total positions loaded: {self._total_loaded:,}")
        print(f"\nMixing Ratios:")
        for source, ratio in sorted(self.mix_ratios.items(), key=lambda x: -x[1]):
            if ratio > 0:
                print(f"  {source:12s}: {ratio:5.1%}")
        
        print(f"\nSource Statistics:")
        for name, loader in self.loaders.items():
            stats = loader.get_stats()
            print(f"  {stats['name']:20s}: {stats['total_loaded']:,} positions loaded")
        
        print("="*60 + "\n")
