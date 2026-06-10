"""
V7P3R v8.0 - Opening Book Builder

Extracts opening variations from PGN files and creates a learnable opening book.
Each opening becomes a single choice the AI can make at the start of a game.

Usage:
    python build_opening_book.py
    
Output:
    opening_book.json - 50-100 opening variations with move sequences
"""

import chess.pgn
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class OpeningBookBuilder:
    """Extract opening variations from PGN files"""
    
    def __init__(self, pgn_directory: str, max_ply: int = 20):
        """
        Args:
            pgn_directory: Path to directory containing opening PGN files
            max_ply: Maximum number of half-moves to extract (default 20 = 10 full moves)
        """
        self.pgn_directory = Path(pgn_directory)
        self.max_ply = max_ply
        self.openings = []
        
    def extract_opening_from_game(self, game, source_filename: str = "") -> Tuple[str, List[str], int]:
        """
        Extract opening name and move sequence from a single game
        
        Returns:
            (opening_name, move_list, ply_count)
        """
        # Get opening name from PGN headers
        opening_name = game.headers.get("Opening", "")
        eco = game.headers.get("ECO", "")
        
        # If no opening name from headers, use filename
        if not opening_name or opening_name == "?" or opening_name.strip() == "":
            # Convert filename to readable name
            # "Caro-KannClassic" -> "Caro-Kann Classic"
            # "SicilianDragon" -> "Sicilian Dragon"
            name_from_file = source_filename.replace('-', ' ')
            # Add spaces before capitals (SicilianDragon -> Sicilian Dragon)
            import re
            name_from_file = re.sub(r'([a-z])([A-Z])', r'\1 \2', name_from_file)
            opening_name = name_from_file
        
        # If has ECO code, prepend it
        if eco and eco != "?" and eco.strip():
            opening_name = f"{eco}: {opening_name}"
        
        # Extract moves
        board = game.board()
        moves = []
        
        for i, move in enumerate(game.mainline_moves()):
            if i >= self.max_ply:
                break
            
            # Convert to UCI notation (e2e4, g1f3, etc.)
            moves.append(move.uci())
            board.push(move)
        
        return opening_name, moves, len(moves)
    
    def process_pgn_file(self, pgn_path: Path) -> List[Dict]:
        """
        Extract all opening variations from a PGN file
        
        Returns:
            List of opening dictionaries
        """
        openings = []
        
        try:
            with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as pgn_file:
                game_count = 0
                
                while True:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break
                    
                    game_count += 1
                    
                    # Extract opening (pass filename for fallback naming)
                    name, moves, ply_count = self.extract_opening_from_game(game, pgn_path.stem)
                    
                    if moves:  # Only add if has moves
                        openings.append({
                            'name': name,
                            'moves': moves,
                            'ply_count': ply_count,
                            'source_file': pgn_path.stem,
                            'game_number': game_count
                        })
                    
                    # Take first 2 variations per file for diversity
                    if game_count >= 2:
                        break
        
        except Exception as e:
            logging.error(f"Error processing {pgn_path.name}: {e}")
        
        return openings
    
    def deduplicate_openings(self, openings: List[Dict]) -> List[Dict]:
        """
        Remove duplicate opening variations (same move sequence)
        
        Returns:
            Deduplicated list of openings
        """
        seen_sequences = {}
        unique_openings = []
        
        for opening in openings:
            # Create signature from move sequence
            move_sig = '-'.join(opening['moves'])
            
            if move_sig not in seen_sequences:
                seen_sequences[move_sig] = opening
                unique_openings.append(opening)
            else:
                # If duplicate, prefer the one with more descriptive name
                existing = seen_sequences[move_sig]
                if len(opening['name']) > len(existing['name']):
                    # Replace with better named version
                    idx = unique_openings.index(existing)
                    unique_openings[idx] = opening
                    seen_sequences[move_sig] = opening
        
        return unique_openings
    
    def categorize_openings(self, openings: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Categorize openings by type for better organization
        
        Returns:
            Dictionary mapping category -> openings
        """
        categories = defaultdict(list)
        
        for opening in openings:
            name = opening['name'].lower()
            first_move = opening['moves'][0] if opening['moves'] else ''
            
            # Categorize by first move and opening type
            if 'sicilian' in name:
                categories['Sicilian Defense'].append(opening)
            elif 'caro' in name or 'caro-kann' in name:
                categories['Caro-Kann Defense'].append(opening)
            elif 'french' in name:
                categories['French Defense'].append(opening)
            elif 'ruy lopez' in name or 'spanish' in name:
                categories['Ruy Lopez'].append(opening)
            elif 'italian' in name:
                categories['Italian Game'].append(opening)
            elif 'nimzo' in name:
                categories['Nimzo-Indian'].append(opening)
            elif 'queen' in name and 'gambit' in name:
                categories['Queen\'s Gambit'].append(opening)
            elif 'king' in name and 'indian' in name:
                categories['King\'s Indian'].append(opening)
            elif 'grunfeld' in name or 'grünfeld' in name:
                categories['Grünfeld Defense'].append(opening)
            elif 'dutch' in name:
                categories['Dutch Defense'].append(opening)
            elif 'pirc' in name:
                categories['Pirc Defense'].append(opening)
            elif 'modern' in name:
                categories['Modern Defense'].append(opening)
            elif 'alekhine' in name:
                categories['Alekhine Defense'].append(opening)
            elif 'scand' in name:
                categories['Scandinavian Defense'].append(opening)
            elif first_move == 'e2e4':
                categories['e4 Openings'].append(opening)
            elif first_move == 'd2d4':
                categories['d4 Openings'].append(opening)
            elif first_move == 'c2c4':
                categories['English Opening'].append(opening)
            elif first_move == 'g1f3':
                categories['Réti Opening'].append(opening)
            else:
                categories['Other Openings'].append(opening)
        
        return dict(categories)
    
    def build_book(self, output_path: str = 'opening_book.json', target_count: int = 100):
        """
        Build complete opening book from all PGN files
        
        Args:
            output_path: Where to save the JSON file
            target_count: Target number of unique openings (will select best)
        """
        logging.info(f"Scanning {self.pgn_directory} for opening PGN files...")
        
        all_openings = []
        pgn_files = list(self.pgn_directory.glob('*.pgn'))
        
        logging.info(f"Found {len(pgn_files)} PGN files")
        
        # Process each PGN file
        for i, pgn_path in enumerate(pgn_files, 1):
            openings = self.process_pgn_file(pgn_path)
            all_openings.extend(openings)
            
            if i % 20 == 0:
                logging.info(f"  Processed {i}/{len(pgn_files)} files, {len(all_openings)} openings extracted")
        
        logging.info(f"Total openings extracted: {len(all_openings)}")
        
        # Deduplicate
        unique_openings = self.deduplicate_openings(all_openings)
        logging.info(f"Unique openings after deduplication: {len(unique_openings)}")
        
        # Categorize
        categorized = self.categorize_openings(unique_openings)
        
        # Select best openings (diverse set)
        selected_openings = []
        
        # Take from each category to ensure diversity
        for category, openings in sorted(categorized.items()):
            # Take up to 15 from each category for good coverage
            sample_size = min(15, len(openings))
            selected_openings.extend(openings[:sample_size])
        
        # If we have more than target, trim to target
        if len(selected_openings) > target_count:
            selected_openings = selected_openings[:target_count]
        
        logging.info(f"Selected {len(selected_openings)} openings from {len(categorized)} categories")
        
        # Assign IDs
        for i, opening in enumerate(selected_openings):
            opening['id'] = i
            
            # Add tags based on characteristics
            tags = []
            if len(opening['moves']) >= 16:
                tags.append('deep_theory')
            if 'gambit' in opening['name'].lower():
                tags.append('gambit')
            if 'defense' in opening['name'].lower():
                tags.append('defensive')
            if 'attack' in opening['name'].lower() or 'dragon' in opening['name'].lower():
                tags.append('aggressive')
            
            opening['tags'] = tags
        
        # Create final book structure
        book = {
            'version': '8.0',
            'created': '2026-06-06',
            'num_openings': len(selected_openings),
            'max_ply': self.max_ply,
            'openings': selected_openings,
            'categories': {cat: len(ops) for cat, ops in categorized.items()}
        }
        
        # Save to JSON
        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(book, f, indent=2, ensure_ascii=False)
        
        logging.info(f"\n✓ Opening book saved to {output_path}")
        logging.info(f"  Total openings: {len(selected_openings)}")
        logging.info(f"  Categories: {len(categorized)}")
        
        # Print summary
        print("\n" + "="*60)
        print("OPENING BOOK SUMMARY")
        print("="*60)
        for category, openings in sorted(categorized.items(), key=lambda x: -len(x[1]))[:10]:
            print(f"{category:30s}: {len(openings):3d} variations")
        print("="*60)
        
        return book


def main():
    """Build opening book from PGN files"""
    
    # Path to opening PGN directory
    pgn_dir = Path(r"E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\pgn_training_data\pgn_data_openings")
    
    if not pgn_dir.exists():
        logging.error(f"Directory not found: {pgn_dir}")
        logging.info("Please update the path to your opening PGN directory")
        return
    
    # Build book
    builder = OpeningBookBuilder(
        pgn_directory=str(pgn_dir),
        max_ply=20  # 10 full moves
    )
    
    output_path = Path(__file__).parent / 'opening_book.json'
    book = builder.build_book(output_path=str(output_path), target_count=100)
    
    # Print first 5 openings as sample
    print("\nSample Openings:")
    print("-" * 60)
    for opening in book['openings'][:5]:
        print(f"ID {opening['id']}: {opening['name']}")
        print(f"  Moves: {' '.join(opening['moves'][:6])}...")
        print(f"  Ply: {opening['ply_count']}, Tags: {', '.join(opening['tags'])}")
        print()


if __name__ == '__main__':
    main()
