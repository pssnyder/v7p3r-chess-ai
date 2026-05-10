"""
Convert Lichess puzzle CSV to JSONL format for v5.1 pipeline.
"""

import csv
import json
import argparse
from pathlib import Path


def convert_csv_to_jsonl(input_csv: Path, output_jsonl: Path, limit: int = None):
    """Convert puzzle CSV to JSONL."""
    count = 0
    
    with open(input_csv, 'r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        
        with open(output_jsonl, 'w', encoding='utf-8') as jsonl_file:
            for row in reader:
                if limit and count >= limit:
                    break
                
                # Convert to expected format
                puzzle = {
                    'PuzzleId': row['PuzzleId'],
                    'FEN': row['FEN'],
                    'Moves': row['Moves'],
                    'Rating': int(row['Rating']) if row['Rating'] else 0,
                    'RatingDeviation': int(row['RatingDeviation']) if row['RatingDeviation'] else 0,
                    'Popularity': int(row['Popularity']) if row['Popularity'] else 0,
                    'NbPlays': int(row['NbPlays']) if row['NbPlays'] else 0,
                    'Themes': row['Themes'],
                    'GameUrl': row.get('GameUrl', ''),
                    'OpeningTags': row.get('OpeningTags', '')
                }
                
                jsonl_file.write(json.dumps(puzzle) + '\n')
                count += 1
    
    print(f"Converted {count} puzzles from CSV to JSONL")
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Lichess puzzle CSV to JSONL')
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--output', required=True, help='Output JSONL file')
    parser.add_argument('--limit', type=int, help='Limit number of puzzles (optional)')
    
    args = parser.parse_args()
    
    convert_csv_to_jsonl(
        Path(args.input),
        Path(args.output),
        args.limit
    )
