# simulate_games.py
"""
Game Simulation Tool for V7P3R Chess AI
This simulates games between V7P3R and Stockfish and records the results.
It can be used to quickly test the AI's performance against different ELO levels.
"""

import os
import sys
import time
import json
import argparse
import chess
import chess.pgn
from datetime import datetime
from pathlib import Path

# Import V7P3R components
from v7p3r_ai import V7P3RAI
from stockfish_handler import StockfishHandler
from chess_core import ChessConfig, GameState


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="V7P3R Chess AI Game Simulator")
    parser.add_argument("--games", type=int, default=1, help="Number of games to simulate")
    parser.add_argument("--elo", type=int, default=400, help="Stockfish ELO rating")
    parser.add_argument("--v7p3r-color", type=str, default="both", 
                        choices=["white", "black", "both"], help="V7P3R plays as white, black, or both")
    parser.add_argument("--time-per-move", type=float, default=1.0, help="Seconds per move (0 for instant)")
    parser.add_argument("--save-dir", type=str, default="simulations", help="Directory to save PGN files")
    parser.add_argument("--verbose", action="store_true", help="Print each move")
    
    return parser.parse_args()


def setup_directories(save_dir):
    """Ensure save directory exists"""
    os.makedirs(save_dir, exist_ok=True)


def play_game(v7p3r, stockfish, v7p3r_is_white, config, time_per_move=0, verbose=False):
    """Play a game between V7P3R and Stockfish"""
    game_state = GameState(config)
    
    # Set player names in headers
    if v7p3r_is_white:
        game_state.game.headers["White"] = "V7P3R"
        game_state.game.headers["Black"] = "Stockfish"
    else:
        game_state.game.headers["White"] = "Stockfish"
        game_state.game.headers["Black"] = "V7P3R"
    
    # Set event and date
    game_state.game.headers["Event"] = "V7P3R vs Stockfish Simulation"
    game_state.game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game_state.game.headers["Round"] = "1"
    
    # Set Stockfish ELO
    stockfish_elo = config.get_stockfish_config().get("elo_rating", 400)
    game_state.game.headers["WhiteElo"] = str(0 if not v7p3r_is_white else stockfish_elo)
    game_state.game.headers["BlackElo"] = str(0 if v7p3r_is_white else stockfish_elo)
    
    # Play the game
    move_count = 0
    while not game_state.is_game_over():
        # Get current board state
        board = game_state.board
        
        # Determine which player's turn it is
        is_v7p3r_turn = (board.turn == chess.WHITE and v7p3r_is_white) or \
                         (board.turn == chess.BLACK and not v7p3r_is_white)
        
        # Get move
        if is_v7p3r_turn:
            move = v7p3r.get_move(board)
        else:
            move = stockfish.get_move(board)
        
        if move is None:
            # No legal moves or error
            print("No legal moves available or error occurred")
            break
        
        # Print move if verbose
        if verbose:
            player = "V7P3R" if is_v7p3r_turn else "Stockfish"
            move_san = board.san(move)
            print(f"Move {move_count//2 + 1}{' (black)' if board.turn == chess.BLACK else ''}: {player} plays {move_san}")
        
        # Make the move
        game_state.make_move(move)
        move_count += 1
        
        # Optional delay to visualize the game
        if time_per_move > 0:
            time.sleep(time_per_move)
    
    # Set result
    result = game_state.get_result()
    if result is not None:
        game_state.game.headers["Result"] = result
    else:
        game_state.game.headers["Result"] = "*"  # Unknown result
    
    # Determine winner
    if result == "1-0":
        winner = "V7P3R" if v7p3r_is_white else "Stockfish"
    elif result == "0-1":
        winner = "V7P3R" if not v7p3r_is_white else "Stockfish"
    else:
        winner = "Draw"
    
    # Get termination reason
    if game_state.board.is_checkmate():
        termination = "Checkmate"
    elif game_state.board.is_stalemate():
        termination = "Stalemate"
    elif game_state.board.is_insufficient_material():
        termination = "Insufficient Material"
    elif game_state.board.can_claim_threefold_repetition():
        termination = "Threefold Repetition"
    elif game_state.board.can_claim_fifty_moves():
        termination = "Fifty-Move Rule"
    else:
        termination = "Unknown"
    
    game_state.game.headers["Termination"] = termination
    
    return game_state.game, result, winner, termination


def simulate_games(args):
    """Simulate games between V7P3R and Stockfish"""
    config = ChessConfig()
    
    # Update config with args
    stockfish_config = config.get_stockfish_config()
    stockfish_config["elo_rating"] = args.elo
    
    # Initialize V7P3R and Stockfish
    v7p3r = V7P3RAI(config)
    stockfish = StockfishHandler(config)
    
    # Prepare results
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "games_played": 0,
        "v7p3r_wins": 0,
        "stockfish_wins": 0,
        "draws": 0,
        "stockfish_elo": args.elo,
        "v7p3r_config": config.get_v7p3r_config(),
        "game_results": []
    }
    
    # Generate timestamp for this simulation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pgn_file = os.path.join(args.save_dir, f"simulation_{timestamp}.pgn")
    
    # Determine colors
    colors = []
    if args.v7p3r_color == "white":
        colors = [True] * args.games
    elif args.v7p3r_color == "black":
        colors = [False] * args.games
    else:  # both
        colors = [True, False] * (args.games // 2)
        if args.games % 2 == 1:
            colors.append(True)  # Extra game as white if odd number
    
    print(f"Simulating {args.games} games - V7P3R vs Stockfish ELO {args.elo}")
    
    # Play games
    with open(pgn_file, "w") as pgn_output:
        for game_idx, v7p3r_is_white in enumerate(colors):
            color = "White" if v7p3r_is_white else "Black"
            print(f"\nGame {game_idx+1}/{len(colors)} - V7P3R playing as {color}")
            
            # Play game
            game, result, winner, termination = play_game(
                v7p3r, stockfish, v7p3r_is_white, config, args.time_per_move, args.verbose
            )
            
            # Update results
            results["games_played"] += 1
            if winner == "V7P3R":
                results["v7p3r_wins"] += 1
            elif winner == "Stockfish":
                results["stockfish_wins"] += 1
            else:
                results["draws"] += 1
            
            # Store individual game result
            game_result = {
                "game_num": game_idx + 1,
                "v7p3r_color": color.lower(),
                "result": result,
                "winner": winner,
                "termination": termination,
                "move_count": len(list(game.mainline_moves()))
            }
            results["game_results"].append(game_result)
            
            # Print result
            print(f"Result: {result} - {winner} wins by {termination}")
            
            # Write to PGN file
            print(game, file=pgn_output)
            print("", file=pgn_output)  # Empty line between games
    
    # Calculate win rate
    if results["games_played"] > 0:
        results["v7p3r_win_rate"] = results["v7p3r_wins"] / results["games_played"]
    else:
        results["v7p3r_win_rate"] = 0
    
    # Save results
    results_file = os.path.join(args.save_dir, f"simulation_results_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n=== Simulation Complete ===")
    print(f"Games played: {results['games_played']}")
    print(f"V7P3R wins: {results['v7p3r_wins']}")
    print(f"Stockfish wins: {results['stockfish_wins']}")
    print(f"Draws: {results['draws']}")
    print(f"V7P3R win rate: {results['v7p3r_win_rate']:.2%}")
    print(f"\nPGN file: {pgn_file}")
    print(f"Results file: {results_file}")


def main():
    """Main function"""
    args = parse_arguments()
    setup_directories(args.save_dir)
    simulate_games(args)


if __name__ == "__main__":
    main()
