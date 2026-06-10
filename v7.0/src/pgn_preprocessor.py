"""
PGN Preprocessor - Clean Grandmaster Game Files

Removes commentary, variations, and annotations from PGN files
to create clean, parseable game data for supervised learning.

Handles:
- Curly brace comments: {This is commentary}
- Parenthetical variations: (15. f3 exf3)
- NAG codes: $2, $6, etc.
- Arrow annotations: [%cal ...]
- Move annotations: !, !!, ?, ??, !?, ?!
"""

import re
from pathlib import Path
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PGNPreprocessor:
    """Clean PGN files for supervised learning."""
    
    def __init__(self):
        # Regex patterns for cleaning
        self.patterns = {
            'curly_comments': r'\{[^}]*\}',           # {commentary}
            'variations': r'\([^)]*\)',               # (variation moves)
            'nag_codes': r'\$\d+',                    # $2, $6, etc.
            'annotations': r'[!?]{1,2}',              # !, !!, ?, ??, !?, ?!
            'arrows': r'\[%[^\]]*\]',                 # [%cal ...]
            'extra_spaces': r'\s+',                   # Multiple spaces
        }
        
    def clean_game_text(self, game_text: str) -> str:
        """
        Clean a single game's move text.
        
        Args:
            game_text: Raw PGN game text with commentary
            
        Returns:
            Clean move text with only headers and moves
        """
        # Remove curly brace comments (may be multi-line)
        cleaned = re.sub(self.patterns['curly_comments'], '', game_text)
        
        # Remove variations (PROPERLY HANDLE NESTED VARIATIONS)
        # Keep removing innermost variations until none remain
        max_iterations = 100  # Prevent infinite loops
        iteration = 0
        while '(' in cleaned and iteration < max_iterations:
            iteration += 1
            # Find innermost variation (no nested parens inside)
            match = re.search(r'\([^()]*\)', cleaned)
            if match:
                # Remove the innermost variation
                cleaned = cleaned[:match.start()] + cleaned[match.end():]
            else:
                # No complete pairs found but parens exist - remove orphans
                logger.warning(f"Found orphaned parentheses after {iteration} iterations, cleaning up")
                cleaned = cleaned.replace('(', '').replace(')', '')
                break
        
        # Remove NAG codes
        cleaned = re.sub(self.patterns['nag_codes'], '', cleaned)
        
        # Remove arrow annotations
        cleaned = re.sub(self.patterns['arrows'], '', cleaned)
        
        # Remove move annotations (but be careful with "1-0", "0-1", etc.)
        # Only remove ! and ? when they're not part of a result
        cleaned = re.sub(r'(?<!\d)[!?]{1,2}(?!\d)', '', cleaned)
        
        # Clean up extra whitespace
        cleaned = re.sub(self.patterns['extra_spaces'], ' ', cleaned)
        
        # Clean up spaces before line breaks
        cleaned = re.sub(r' +\n', '\n', cleaned)
        
        return cleaned.strip()
    
    def split_games(self, pgn_content: str) -> List[str]:
        """
        Split PGN file into individual games.
        
        Args:
            pgn_content: Full PGN file content
            
        Returns:
            List of individual game strings
        """
        # Games are separated by blank lines after result
        # Split on double newlines, keeping headers with games
        games = []
        current_game = []
        
        for line in pgn_content.split('\n'):
            line = line.strip()
            
            if line.startswith('[Event'):
                # Start of new game
                if current_game:
                    games.append('\n'.join(current_game))
                current_game = [line]
            elif line:
                current_game.append(line)
            elif current_game and not line:
                # Blank line might signal end of game
                if any(r in ' '.join(current_game) for r in ['1-0', '0-1', '1/2-1/2', '*']):
                    games.append('\n'.join(current_game))
                    current_game = []
        
        # Add last game if exists
        if current_game:
            games.append('\n'.join(current_game))
        
        return games
    
    def process_file(
        self,
        input_path: Path,
        output_path: Path = None,
        overwrite: bool = False
    ) -> Tuple[int, int]:
        """
        Process a PGN file, cleaning all games.
        
        Args:
            input_path: Source PGN file
            output_path: Destination file (default: input_path with _clean suffix)
            overwrite: Whether to overwrite existing output file
            
        Returns:
            Tuple of (games_processed, games_written)
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Default output path
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_clean.pgn"
        else:
            output_path = Path(output_path)
        
        # Check overwrite
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output file exists: {output_path}\n"
                "Set overwrite=True to replace it."
            )
        
        logger.info(f"Processing: {input_path.name}")
        
        # Read input
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into games
        games = self.split_games(content)
        logger.info(f"Found {len(games)} games")
        
        # Clean each game
        cleaned_games = []
        for i, game in enumerate(games, 1):
            try:
                cleaned = self.clean_game_text(game)
                if cleaned:
                    cleaned_games.append(cleaned)
                    
                if i % 100 == 0:
                    logger.info(f"Cleaned {i}/{len(games)} games...")
            except Exception as e:
                logger.warning(f"Failed to clean game {i}: {e}")
        
        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for game in cleaned_games:
                f.write(game)
                f.write('\n\n')  # Separate games
        
        logger.info(f"✓ Wrote {len(cleaned_games)} clean games to: {output_path.name}")
        
        return len(games), len(cleaned_games)
    
    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path = None,
        pattern: str = "*.pgn",
        overwrite: bool = False
    ) -> dict:
        """
        Process all PGN files in a directory.
        
        Args:
            input_dir: Source directory
            output_dir: Destination directory (default: input_dir/cleaned)
            pattern: File pattern to match
            overwrite: Whether to overwrite existing files
            
        Returns:
            Dictionary with processing statistics
        """
        input_dir = Path(input_dir)
        
        if output_dir is None:
            output_dir = input_dir / "cleaned"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find PGN files
        pgn_files = list(input_dir.glob(pattern))
        
        if not pgn_files:
            logger.warning(f"No PGN files found matching '{pattern}' in {input_dir}")
            return {}
        
        logger.info(f"Found {len(pgn_files)} PGN files to process")
        
        # Process each file
        stats = {}
        total_games = 0
        total_cleaned = 0
        
        for pgn_file in pgn_files:
            output_file = output_dir / f"{pgn_file.stem}_clean.pgn"
            
            try:
                games, cleaned = self.process_file(
                    pgn_file,
                    output_file,
                    overwrite=overwrite
                )
                
                stats[pgn_file.name] = {
                    'original': games,
                    'cleaned': cleaned,
                    'output': output_file.name
                }
                
                total_games += games
                total_cleaned += cleaned
                
            except Exception as e:
                logger.error(f"Failed to process {pgn_file.name}: {e}")
                stats[pgn_file.name] = {'error': str(e)}
        
        # Summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total files processed: {len([s for s in stats.values() if 'error' not in s])}")
        logger.info(f"Total games found: {total_games}")
        logger.info(f"Total games cleaned: {total_cleaned}")
        logger.info(f"Output directory: {output_dir}")
        logger.info("=" * 80)
        
        return stats


def main():
    """Example usage: Clean tactical puzzles directory."""
    import sys
    
    # Default: Clean the tactics directory
    tactics_dir = Path(r"E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\pgn_training_data\pgn_data_tactics")
    important_games_dir = Path(r"E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\pgn_training_data\pgn_data_important_games")
    general_dir = Path(r"E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\pgn_training_data\pgn_data_general")
    
    preprocessor = PGNPreprocessor()
    
    print("=" * 80)
    print("PGN PREPROCESSOR - Clean Grandmaster Games")
    print("=" * 80)
    print()
    print("This will create cleaned versions of all PGN files")
    print("with commentary, variations, and annotations removed.")
    print()
    print("Available directories:")
    print("  1. Tactics (1001 combinations, positional play, etc.)")
    print("  2. Important Games (100 golden games, Tarrasch, Lasker)")
    print("  3. General (Art of Chess Analysis, Lasker's Manual)")
    print("  4. All directories")
    print()
    
    choice = input("Select option (1-4, or 'q' to quit): ").strip()
    
    if choice == 'q':
        return
    
    dirs_to_process = []
    if choice == '1':
        dirs_to_process.append(tactics_dir)
    elif choice == '2':
        dirs_to_process.append(important_games_dir)
    elif choice == '3':
        dirs_to_process.append(general_dir)
    elif choice == '4':
        dirs_to_process.extend([tactics_dir, important_games_dir, general_dir])
    else:
        print("Invalid choice")
        return
    
    print()
    overwrite = input("Overwrite existing cleaned files? (y/n): ").strip().lower() == 'y'
    print()
    
    # Process selected directories
    all_stats = {}
    for directory in dirs_to_process:
        print(f"\nProcessing: {directory.name}")
        print("-" * 80)
        
        stats = preprocessor.process_directory(
            directory,
            pattern="*.pgn",
            overwrite=overwrite
        )
        
        all_stats[directory.name] = stats
    
    # Final summary
    print()
    print("=" * 80)
    print("ALL DIRECTORIES PROCESSED")
    print("=" * 80)
    for dir_name, stats in all_stats.items():
        successful = len([s for s in stats.values() if 'error' not in s])
        total_games = sum(s['cleaned'] for s in stats.values() if 'cleaned' in s)
        print(f"{dir_name}: {successful} files, {total_games} games")
    print("=" * 80)


if __name__ == "__main__":
    main()
