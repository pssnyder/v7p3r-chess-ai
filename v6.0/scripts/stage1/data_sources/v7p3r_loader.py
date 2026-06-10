"""
V7P3R Game Loader - loads positions from V7P3R engine battle records.

Data sources:
- v7p3r_bad_positions.jsonl (mined bad positions with eval drops >= 100cp)
- V7P3R vs V7P3R PGN games (extracting both good and bad positions)

This provides real-world positions from engine self-play, valuable for learning
patterns specific to V7P3R's playing style.
"""

import json
import chess
import chess.pgn
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from .base_loader import DataSourceLoader
from scripts.utils.calculate_features import FeatureCalculator, FeatureConfig


class V7P3RGameLoader(DataSourceLoader):
    """Load positions from V7P3R engine battle records and original training dataset."""
    
    def __init__(
        self,
        bad_positions_jsonl: Optional[str] = None,
        good_positions_jsonl: Optional[str] = None,
        pgn_dir: Optional[str] = None,
        seed: int = 42,
        shuffle: bool = True,
        feature_config: Optional[FeatureConfig] = None,
        include_good_moves: bool = True
    ):
        """
        Initialize V7P3R game loader.
        
        Args:
            bad_positions_jsonl: Path to bad positions JSONL (v7p3r_bad_positions.jsonl or bad_positions.jsonl)
            good_positions_jsonl: Path to good positions JSONL (good_positions.jsonl)
            pgn_dir: Optional directory with V7P3R vs V7P3R PGN games
            seed: Random seed for reproducibility
            shuffle: Whether to shuffle positions
            feature_config: Configuration for feature calculation
            include_good_moves: Also extract good moves from PGN games
        """
        super().__init__(seed=seed, shuffle=shuffle)
        self.bad_positions_path = Path(bad_positions_jsonl) if bad_positions_jsonl else None
        self.good_positions_path = Path(good_positions_jsonl) if good_positions_jsonl else None
        self.pgn_dir = Path(pgn_dir) if pgn_dir else None
        self.include_good_moves = include_good_moves
        self.feature_calculator = FeatureCalculator(config=feature_config or FeatureConfig())
        
        # At least one data source must be provided
        if not self.bad_positions_path and not self.good_positions_path and not self.pgn_dir:
            raise ValueError("At least one data source (bad_positions, good_positions, or pgn_dir) must be provided")
        
        # Load positions (streaming for large files)
        self._bad_positions = []
        self._good_positions_file = None  # File handle for streaming large good positions
        self._good_positions_stream_exhausted = False
        
        if self.bad_positions_path and self.bad_positions_path.exists():
            self._load_bad_positions()
        
        if self.good_positions_path and self.good_positions_path.exists():
            self._open_good_positions_stream()
        
        # Discover PGN files if directory provided
        self._pgn_files = []
        if self.pgn_dir and self.pgn_dir.exists():
            self._pgn_files = list(self.pgn_dir.glob("**/*.pgn"))
        
        self._current_file_idx = 0
        self._position_buffer = []
    
    def __del__(self):
        """Close file handles on cleanup."""
        if hasattr(self, '_good_positions_file') and self._good_positions_file:
            try:
                self._good_positions_file.close()
            except:
                pass
        
    def _load_bad_positions(self):
        """Load bad positions from JSONL file into memory."""
        total_loaded = 0
        total_skipped = 0
        
        with open(self.bad_positions_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    
                    # CRITICAL: Validate features exist and are complete
                    if 'features' not in record:
                        total_skipped += 1
                        continue
                    
                    if len(record['features']) < 76:
                        total_skipped += 1
                        continue
                    
                    # Ensure record has source field
                    if 'source' not in record:
                        record['source'] = 'v7p3r_dataset'
                    
                    # Record already has features and label
                    self._bad_positions.append(record)
                    total_loaded += 1
                except (json.JSONDecodeError, TypeError):
                    total_skipped += 1
                    continue
        
        if total_skipped > 0:
            print(f"Warning: Skipped {total_skipped} positions with missing/incomplete features")
        
        if self.shuffle:
            self.random.shuffle(self._bad_positions)
    
    def _open_good_positions_stream(self):
        """Open good positions file for streaming (too large to load into memory)."""
        try:
            self._good_positions_file = open(self.good_positions_path, 'r', encoding='utf-8')
            self._good_positions_stream_exhausted = False
        except Exception as e:
            print(f"Warning: Failed to open good positions file: {e}")
            self._good_positions_file = None
    
    def _read_good_positions(self, count: int) -> List[Dict[str, Any]]:
        """Stream read 'count' good positions from file."""
        if not self._good_positions_file or self._good_positions_stream_exhausted:
            return []
        
        positions = []
        attempted = 0
        max_attempts = count * 2  # Allow skipping up to 50% bad records
        
        while len(positions) < count and attempted < max_attempts:
            line = self._good_positions_file.readline()
            if not line:
                # End of file, reset to beginning for continuous streaming
                self._good_positions_file.seek(0)
                self._good_positions_stream_exhausted = True
                break
            
            attempted += 1
            
            try:
                record = json.loads(line.strip())
                
                # CRITICAL: Validate features exist and are complete
                if 'features' not in record:
                    continue
                
                if len(record['features']) < 76:
                    continue
                
                # Ensure record has source and label fields
                if 'source' not in record:
                    record['source'] = 'v7p3r_dataset'
                if 'label' not in record:
                    record['label'] = 1  # Good position
                
                positions.append(record)
            except (json.JSONDecodeError, TypeError):
                continue
        
        return positions
            
    def _parse_eval_from_comment(self, comment: str) -> Optional[float]:
        """Extract evaluation from PGN comment like '{+4.42/5 7}'."""
        try:
            if '{' in comment:
                eval_part = comment.split('{')[1].split('}')[0].strip()
                eval_str = eval_part.split('/')[0].strip()
                
                # Handle mate scores
                if 'M' in eval_str or '#' in eval_str:
                    return 1000.0 if '+' in eval_str else -1000.0
                
                return float(eval_str)
        except:
            pass
        return None
        
    def _extract_positions_from_pgn(self, game: chess.pgn.Game) -> List[Dict[str, Any]]:
        """
        Extract both good and bad positions from V7P3R game.
        
        Args:
            game: PGN game object
            
        Returns:
            List of position records
        """
        # Only process V7P3R vs V7P3R games
        white = game.headers.get('White', '').lower()
        black = game.headers.get('Black', '').lower()
        if 'v7p3r' not in white or 'v7p3r' not in black:
            return []
        
        positions = []
        board = game.board()
        prev_eval = None
        
        try:
            for node in game.mainline():
                try:
                    move = node.move
                    comment = node.comment
                    
                    # Extract evaluation from comment
                    curr_eval = self._parse_eval_from_comment(comment)
                    
                    if curr_eval is not None and prev_eval is not None:
                        # Calculate eval drop
                        eval_drop = abs(prev_eval - curr_eval)
                        
                        # Determine if good or bad move
                        if eval_drop >= 1.0:  # >= 100cp drop
                            label = 0  # Bad move
                            if eval_drop >= 8.0:
                                grade = 5
                            elif eval_drop >= 4.0:
                                grade = 4
                            elif eval_drop >= 2.0:
                                grade = 3
                            else:
                                grade = 2
                        elif self.include_good_moves and eval_drop < 0.3:  # < 30cp change
                            label = 1  # Good move
                            grade = 1
                        else:
                            # Skip mediocre moves
                            prev_eval = curr_eval
                            board.push(move)
                            continue
                        
                        try:
                            features = self.feature_calculator.calculate_features_from_fen(board.fen())
                            
                            position = {
                                'fen': board.fen(),
                                'move_uci': move.uci(),
                                'label': label,
                                'source': 'v7p3r_game',
                                'features': features,
                                'eval_cp': int(curr_eval * 100),
                                'grade': grade,
                                'eval_drop': round(eval_drop * 100, 2),
                                'game_white': game.headers.get('White', ''),
                                'game_black': game.headers.get('Black', '')
                            }
                            
                            positions.append(position)
                            
                        except Exception:
                            pass
                    
                    prev_eval = curr_eval
                    board.push(move)
                    
                except Exception:
                    # Skip moves that can't be processed (corrupted PGN)
                    continue
                    
        except Exception:
            # If entire game fails, return what we have so far
            pass
        
        return positions
        
    def load_batch(self, size: int) -> List[Dict[str, Any]]:
        """
        Load a batch of positions from V7P3R games.
        
        Args:
            size: Number of positions to load
            
        Returns:
            List of position dictionaries
        """
        positions = []
        
        # Strategy: Split batch between good and bad positions
        # If we have both good and bad, aim for 50:50 balance
        has_good = self._good_positions_file is not None
        has_bad = len(self._bad_positions) > 0
        
        if has_good and has_bad:
            # Load 50% good, 50% bad
            good_count = size // 2
            bad_count = size - good_count
        elif has_bad:
            # Only bad positions available
            good_count = 0
            bad_count = size
        elif has_good:
            # Only good positions available
            good_count = size
            bad_count = 0
        else:
            # No positions, try PGN extraction
            good_count = 0
            bad_count = 0
        
        # Sample from bad positions
        if bad_count > 0 and len(self._bad_positions) > 0:
            sample_size = min(bad_count, len(self._bad_positions))
            sample_indices = self.random.sample(range(len(self._bad_positions)), sample_size)
            for idx in sample_indices:
                pos = self._bad_positions[idx].copy()
                pos['label'] = 0  # Ensure bad label
                positions.append(pos)
        
        # Stream from good positions
        if good_count > 0:
            good_positions = self._read_good_positions(good_count)
            positions.extend(good_positions)
        
        # If still need more, extract from PGN games
        if len(positions) < size and self.include_good_moves and self._pgn_files:
            remaining = size - len(positions)
            
            while len(positions) < size and self._current_file_idx < len(self._pgn_files):
                file_path = self._pgn_files[self._current_file_idx]
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        while len(positions) < size:
                            game = chess.pgn.read_game(f)
                            if game is None:
                                break
                            
                            game_positions = self._extract_positions_from_pgn(game)
                            positions.extend(game_positions[:size - len(positions)])
                            
                except Exception:
                    pass
                
                self._current_file_idx += 1
            
            # Reset if exhausted
            if self._current_file_idx >= len(self._pgn_files):
                self._current_file_idx = 0
                if self.shuffle:
                    self.random.shuffle(self._pgn_files)
        
        if self.shuffle:
            self.random.shuffle(positions)
        
        self._total_loaded += len(positions)
        return positions[:size]
        
    def reset(self):
        """Reset loader to beginning."""
        self._current_file_idx = 0
        self._total_loaded = 0
        if self.shuffle:
            self.random.shuffle(self._bad_positions)
            if self._pgn_files:
                self.random.shuffle(self._pgn_files)
        
    def get_name(self) -> str:
        """Get human-readable name of this data source."""
        return "V7P3R Engine Battles"
