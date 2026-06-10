"""
V7P3R v7.0 - Training Progress Monitor

Real-time analysis of self-play training progress.
Tracks personality emergence, game quality, and system performance.
"""

import json
import chess.pgn
from pathlib import Path
from typing import Dict, List


def analyze_training_progress(training_dir: str = "../../training/v7_selfplay"):
    """Analyze current training progress from saved games and stats."""
    
    training_path = Path(training_dir)
    
    # Find all PGN files
    pgn_files = sorted(training_path.glob("game_*.pgn"))
    
    if not pgn_files:
        print("No games found yet...")
        return
    
    print("="*60)
    print(f"V7P3R TRAINING PROGRESS - {len(pgn_files)} GAMES ANALYZED")
    print("="*60)
    
    # Analyze each game
    game_stats = []
    
    for pgn_file in pgn_files:
        with open(pgn_file) as f:
            game = chess.pgn.read_game(f)
            
            if game is None:
                continue
            
            # Extract metadata from headers
            result = game.headers.get("Result", "*")
            
            # Count moves
            board = game.board()
            moves = list(game.mainline_moves())
            num_moves = len(moves)
            
            game_stats.append({
                'number': len(game_stats) + 1,
                'result': result,
                'moves': num_moves
            })
    
    # Calculate metrics
    total_games = len(game_stats)
    wins_white = sum(1 for g in game_stats if g['result'] == '1-0')
    wins_black = sum(1 for g in game_stats if g['result'] == '0-1')
    draws = sum(1 for g in game_stats if g['result'] == '1/2-1/2')
    ongoing = sum(1 for g in game_stats if g['result'] == '*')
    
    avg_moves = sum(g['moves'] for g in game_stats) / total_games
    games_200_moves = sum(1 for g in game_stats if g['moves'] >= 200)
    games_decisive = wins_white + wins_black
    
    print(f"\n📊 GAME RESULTS")
    print(f"  Total Games: {total_games}")
    print(f"  White Wins: {wins_white} ({wins_white/total_games*100:.1f}%)")
    print(f"  Black Wins: {wins_black} ({wins_black/total_games*100:.1f}%)")
    print(f"  Draws: {draws} ({draws/total_games*100:.1f}%)")
    print(f"  Ongoing: {ongoing} ({ongoing/total_games*100:.1f}%)")
    print(f"  Decisive: {games_decisive} ({games_decisive/total_games*100:.1f}%)")
    
    print(f"\n♟️  GAME QUALITY")
    print(f"  Avg Moves/Game: {avg_moves:.1f}")
    print(f"  Games at 200 moves: {games_200_moves} ({games_200_moves/total_games*100:.1f}%)")
    print(f"  Games concluded: {total_games - games_200_moves} ({(total_games-games_200_moves)/total_games*100:.1f}%)")
    
    # Check for stats files
    stats_files = sorted(training_path.glob("stats_*.json"))
    
    if stats_files:
        latest_stats = stats_files[-1]
        with open(latest_stats) as f:
            stats = json.load(f)
        
        print(f"\n⚔️  PERSONALITY EMERGENCE")
        print(f"  Avg Forest Darkness: {stats.get('avg_forest_darkness', 0):.3f}")
        print(f"  Avg Personality Reward: {stats.get('avg_personality_reward', 0):+.3f}")
        print(f"  Total Sacrifices: {stats.get('total_sacrifices', 0)}")
        print(f"  Sacrifices/Game: {stats.get('total_sacrifices', 0)/stats.get('total_games', 1):.2f}")
    
    # Game-by-game breakdown
    print(f"\n📋 GAME BREAKDOWN (Last 10)")
    print(f"{'Game':<6} {'Result':<12} {'Moves':<6} {'Status':<20}")
    print("-"*60)
    
    for game in game_stats[-10:]:
        status = "Max moves" if game['moves'] >= 200 else "Concluded"
        print(f"{game['number']:<6} {game['result']:<12} {game['moves']:<6} {status:<20}")
    
    # Training checkpoints
    model_files = sorted(training_path.glob("model_*.pt"))
    
    if model_files:
        print(f"\n💾 CHECKPOINTS")
        for model in model_files:
            size_mb = model.stat().st_size / 1024 / 1024
            print(f"  {model.name} ({size_mb:.2f} MB)")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    analyze_training_progress()
