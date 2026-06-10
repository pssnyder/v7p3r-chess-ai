"""
Calculate Historical Game Counts per V7P3R Version

Purpose: Determine how many games were required per version during manual engine tuning.
This becomes the baseline for "Can AI learn faster than human tuning?"

Process:
1. Scan all Engine Battle directories for V7P3R games
2. Extract version from filename/content
3. Count games per major version (v12, v13, v14, etc.)
4. Calculate median and standard deviation
5. Report target game count for Stage 2 self-play training

Author: V7P3R AI Development Team
Date: 2026-05-31
"""

import os
import re
from pathlib import Path
from collections import defaultdict
import statistics
import json
import chess.pgn

# Global counters (modified by count_games_in_pgn)
version_counts = defaultdict(int)
version_files = defaultdict(list)

# Game record directories
GAME_DIRS = [
    r"E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\game_records\Engine Battle 202507",
    r"E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\game_records\Engine Battle 202508",
    r"E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\game_records\Engine Battle 202509",
    r"E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\game_records\Engine Battle 202510",
    r"E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\game_records\Engine Battle 202511",
    r"E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\game_records\Engine Battle 202512",
    r"E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\game_records\Lichess V7P3R Bot",
]


def extract_version_from_filename(filename: str) -> str | None:
    """
    Extract V7P3R version from PGN filename.
    
    Examples:
    - "V7P3R_v12.3_vs_Stockfish.pgn" -> "v12"
    - "v14_1_game_001.pgn" -> "v14"
    - "V7P3R_17.8.pgn" -> "v17"
    """
    # Pattern 1: V7P3R_v12.3 or v12.3 or v12_3
    match = re.search(r'v(\d+)[._]\d+', filename, re.IGNORECASE)
    if match:
        return f"v{match.group(1)}"
    
    # Pattern 2: V7P3R_v12 or v12
    match = re.search(r'v(\d+)(?:[^0-9]|$)', filename, re.IGNORECASE)
    if match:
        return f"v{match.group(1)}"
    
    # Pattern 3: V7P3R_17.8 (no 'v' prefix)
    match = re.search(r'V7P3R[_\s]+(\d+)[._]\d+', filename, re.IGNORECASE)
    if match:
        return f"v{match.group(1)}"
    
    return None


def extract_version_from_pgn_headers(pgn_path: Path) -> str | None:
    """
    Extract version from PGN file headers (White/Black player names).
    
    Looks for V7P3R version in player names like "V7P3R v14.1" or "v7p3r_bot (v17.8)"
    """
    try:
        with open(pgn_path, 'r', encoding='utf-8') as f:
            game = chess.pgn.read_game(f)
            
            if game is None:
                return None
            
            white = game.headers.get('White', '')
            black = game.headers.get('Black', '')
            
            # Check both players
            for player in [white, black]:
                if 'v7p3r' in player.lower():
                    # Pattern: v14.1 or v14_1
                    match = re.search(r'v(\d+)[._]\d+', player, re.IGNORECASE)
                    if match:
                        return f"v{match.group(1)}"
                    
                    # Pattern: v14
                    match = re.search(r'v(\d+)(?:[^0-9]|$)', player, re.IGNORECASE)
                    if match:
                        return f"v{match.group(1)}"
    
    except Exception as e:
        # Corrupted PGN, skip
        return None
    
    return None


def scan_game_directories() -> dict:
    """
    Scan all game directories and count GAMES per V7P3R version.
    
    Returns:
        Dict mapping version to game count
        Example: {"v12": 450, "v13": 670, "v14": 1030, ...}
    """
    version_counts = defaultdict(int)
    version_files = defaultdict(list)
    total_games_scanned = 0
    
    for game_dir in GAME_DIRS:
        game_dir_path = Path(game_dir)
        
        if not game_dir_path.exists():
            print(f"Warning: Directory not found: {game_dir}")
            continue
        
        print(f"\nScanning: {game_dir_path.name}")
        
        # Find all PGN files
        pgn_files = list(game_dir_path.rglob("*.pgn"))
        print(f"  Found {len(pgn_files)} PGN files")
        
        for pgn_file in pgn_files:
            # Count individual games in this PGN file
            games_in_file = count_games_in_pgn(pgn_file)
            total_games_scanned += games_in_file
            
            if games_in_file > 0:
                print(f"    {pgn_file.name}: {games_in_file} games", end='\r')
    
    print()  # New line after progress
    print(f"\n  Total games scanned: {total_games_scanned}")
    
    return version_counts, version_files


def count_games_in_pgn(pgn_path: Path) -> int:
    """
    Count number of games in a PGN file and attribute them to versions.
    
    Args:
        pgn_path: Path to PGN file
        
    Returns:
        Number of games counted
    """
    global version_counts, version_files
    
    game_count = 0
    
    try:
        with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as f:
            while True:
                game = chess.pgn.read_game(f)
                
                if game is None:
                    break  # End of file
                
                game_count += 1
                
                # Extract version from this game
                white = game.headers.get('White', '')
                black = game.headers.get('Black', '')
                
                version = None
                
                # Check both players
                for player in [white, black]:
                    if 'v7p3r' in player.lower():
                        # Pattern: v14.1 or v14_1
                        match = re.search(r'v(\d+)[._]\d+', player, re.IGNORECASE)
                        if match:
                            version = f"v{match.group(1)}"
                            break
                        
                        # Pattern: v14
                        match = re.search(r'v(\d+)(?:[^0-9]|$)', player, re.IGNORECASE)
                        if match:
                            version = f"v{match.group(1)}"
                            break
                
                # Fallback to filename if no version in headers
                if version is None:
                    version = extract_version_from_filename(pgn_path.name)
                
                if version:
                    version_counts[version] += 1
                    if pgn_path.name not in version_files[version]:
                        version_files[version].append(pgn_path.name)
    
    except Exception as e:
        print(f"\n  Error reading {pgn_path.name}: {e}")
        return 0
    
    return game_count


def calculate_statistics(version_counts: dict) -> dict:
    """
    Calculate statistics on game counts per version.
    
    Returns:
        Statistics dict with mean, median, std, outliers
    """
    if not version_counts:
        return {
            'error': 'No version data found'
        }
    
    counts = list(version_counts.values())
    
    # Calculate stats
    mean_games = statistics.mean(counts)
    median_games = statistics.median(counts)
    
    if len(counts) > 1:
        std_games = statistics.stdev(counts)
    else:
        std_games = 0.0
    
    # Identify outliers (> 1 std deviation from mean)
    outliers = []
    counts_without_outliers = []
    
    for version, count in version_counts.items():
        if abs(count - mean_games) > std_games:
            outliers.append((version, count))
        else:
            counts_without_outliers.append(count)
    
    # Recalculate without outliers
    if counts_without_outliers:
        mean_without_outliers = statistics.mean(counts_without_outliers)
        median_without_outliers = statistics.median(counts_without_outliers)
    else:
        mean_without_outliers = mean_games
        median_without_outliers = median_games
    
    return {
        'total_versions': len(version_counts),
        'total_games': sum(counts),
        'mean_games_per_version': mean_games,
        'median_games_per_version': median_games,
        'std_games_per_version': std_games,
        'min_games': min(counts),
        'max_games': max(counts),
        'outliers': outliers,
        'mean_without_outliers': mean_without_outliers,
        'median_without_outliers': median_without_outliers,
        'recommended_target': int(median_without_outliers),
    }


def print_report(version_counts: dict, version_files: dict, stats: dict):
    """Print detailed report of findings."""
    
    print("\n" + "="*80)
    print("V7P3R HISTORICAL GAME COUNT ANALYSIS")
    print("="*80)
    
    # Version breakdown
    print("\nGames per Version:")
    print("-" * 40)
    
    # Sort by version number
    versions_sorted = sorted(
        version_counts.items(),
        key=lambda x: int(x[0].replace('v', ''))
    )
    
    for version, count in versions_sorted:
        outlier_marker = " (OUTLIER)" if any(v == version for v, _ in stats['outliers']) else ""
        print(f"  {version}: {count:4d} games{outlier_marker}")
    
    # Statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"Total versions analyzed: {stats['total_versions']}")
    print(f"Total games found:       {stats['total_games']}")
    print(f"\nMean games/version:      {stats['mean_games_per_version']:.1f}")
    print(f"Median games/version:    {stats['median_games_per_version']:.1f}")
    print(f"Std deviation:           {stats['std_games_per_version']:.1f}")
    print(f"Range:                   {stats['min_games']} - {stats['max_games']}")
    
    if stats['outliers']:
        print(f"\nOutliers (>{stats['std_games_per_version']:.1f} from mean):")
        for version, count in stats['outliers']:
            print(f"  {version}: {count} games")
        
        print(f"\nAdjusted statistics (without outliers):")
        print(f"  Mean:   {stats['mean_without_outliers']:.1f}")
        print(f"  Median: {stats['median_without_outliers']:.1f}")
    
    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION FOR STAGE 2 SELF-PLAY TRAINING")
    print("="*80)
    print(f"\nTarget game count: {stats['recommended_target']} games")
    print(f"\nRationale:")
    print(f"  - This is the median game count (excluding outliers)")
    print(f"  - Represents typical learning cycle for manual tuning")
    print(f"  - Control metric: 'Can AI learn faster than human?'")
    
    print("\n" + "="*80)


def save_results(version_counts: dict, version_files: dict, stats: dict):
    """Save results to JSON for later reference."""
    
    output_path = Path("analytics/historical_game_counts.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        'analysis_date': '2026-05-31',
        'version_counts': version_counts,
        'statistics': stats,
        'recommended_selfplay_target': stats['recommended_target'],
        'version_files_sample': {
            version: files[:5] for version, files in version_files.items()
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def main():
    """Main execution."""
    global version_counts, version_files
    
    print("Calculating historical V7P3R game counts...")
    print("This will help determine Stage 2 self-play training target.\n")
    print("NOTE: Counting individual GAMES within PGN files (not just files)\n")
    
    # Initialize globals
    version_counts = defaultdict(int)
    version_files = defaultdict(list)
    
    # Scan directories
    scan_game_directories()
    
    if not version_counts:
        print("\nError: No V7P3R games found in specified directories.")
        print("Please verify directories exist and contain V7P3R PGN files.")
        return
    
    # Calculate statistics
    stats = calculate_statistics(version_counts)
    
    # Print report
    print_report(version_counts, version_files, stats)
    
    # Save results
    save_results(version_counts, version_files, stats)


if __name__ == "__main__":
    main()
