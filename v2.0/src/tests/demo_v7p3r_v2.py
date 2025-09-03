#!/usr/bin/env python3
"""
V7P3R Chess AI 2.0 - Demo and Testing Script
Test the new AI capabilities and bounty system.
"""

import chess
import chess.pgn
import time
from pathlib import Path

from v7p3r_ai_v2 import V7P3RAI_v2
from v7p3r_bounty_system import ExtendedBountyEvaluator


def test_bounty_system():
    """Test the bounty evaluation system"""
    print("=" * 50)
    print("Testing V7P3R 2.0 Bounty System")
    print("=" * 50)
    
    evaluator = ExtendedBountyEvaluator()
    
    # Test position 1: Opening move
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    
    score = evaluator.evaluate_move(board, move)
    print(f"Move: {move.uci()} (e2-e4)")
    print(f"Center control: {score.center_control}")
    print(f"Piece value: {score.piece_value}")
    print(f"Total bounty: {score.total()}")
    print()
    
    # Test position 2: Tactical position
    board.set_fen("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 4")
    legal_moves = list(board.legal_moves)
    
    print("Tactical position evaluation:")
    move_scores = []
    for move in legal_moves[:5]:  # Top 5 moves
        score = evaluator.evaluate_move(board, move)
        move_scores.append((move, score.total()))
        print(f"{move.uci()}: {score.total():.2f} gold")
    
    move_scores.sort(key=lambda x: x[1], reverse=True)
    print(f"Best move by bounty: {move_scores[0][0].uci()} ({move_scores[0][1]:.2f} gold)")
    print()


def test_rnn_model():
    """Test the RNN model"""
    print("=" * 50)
    print("Testing V7P3R 2.0 RNN Model")
    print("=" * 50)
    
    ai = V7P3RAI_v2()
    
    # Test starting position
    board = chess.Board()
    evaluation = ai.get_position_evaluation(board)
    
    print("Starting position evaluation:")
    for key, value in evaluation.items():
        print(f"{key}: {value}")
    print()
    
    # Get AI's best move
    move = ai.get_move(board)
    if move:
        analysis = ai.analyze_move(board, move)
        print(f"AI's chosen move: {move.uci()}")
        print(f"RNN score: {analysis['rnn_score']:.3f}")
        print(f"Bounty score: {analysis['bounty_breakdown']['total']:.3f}")
        print(f"Combined score: {analysis['combined_score']:.3f}")
    print()


def play_demo_game():
    """Play a demo game AI vs AI"""
    print("=" * 50)
    print("V7P3R 2.0 Demo Game")
    print("=" * 50)
    
    ai1 = V7P3RAI_v2(config_path="config_v2.json")
    ai2 = V7P3RAI_v2(config_path="config_v2.json")
    
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "V7P3R 2.0 Demo"
    game.headers["White"] = "V7P3R AI 2.0 (White)"
    game.headers["Black"] = "V7P3R AI 2.0 (Black)"
    
    node = game
    move_count = 0
    max_moves = 50  # Limit for demo
    
    print("Game starting...")
    print(f"Initial position: {board.fen()}")
    print()
    
    while not board.is_game_over() and move_count < max_moves:
        current_ai = ai1 if board.turn == chess.WHITE else ai2
        color_name = "White" if board.turn == chess.WHITE else "Black"
        
        start_time = time.time()
        move = current_ai.get_move(board)
        think_time = time.time() - start_time
        
        if not move:
            print(f"No legal moves for {color_name}")
            break
        
        # Analyze the move
        analysis = current_ai.analyze_move(board, move)
        
        print(f"Move {move_count + 1}: {color_name} plays {move.uci()}")
        print(f"  Think time: {think_time:.2f}s")
        print(f"  RNN: {analysis['rnn_score']:.2f}, Bounty: {analysis['bounty_breakdown']['total']:.2f}")
        print(f"  Game phase: {analysis['game_phase']}")
        
        # Make the move
        board.push(move)
        node = node.add_variation(move)
        move_count += 1
        
        # Show board occasionally
        if move_count % 10 == 0:
            print(f"\nPosition after move {move_count}:")
            print(board)
            print()
    
    # Game result
    if board.is_game_over():
        result = board.result()
        print(f"\nGame over: {result}")
        if board.is_checkmate():
            winner = "White" if result == "1-0" else "Black"
            print(f"{winner} wins by checkmate!")
        elif board.is_stalemate():
            print("Game drawn by stalemate")
        else:
            print("Game drawn")
    else:
        print(f"\nDemo ended after {move_count} moves")
    
    # Save game
    pgn_path = Path("demo_game_v2.pgn")
    with open(pgn_path, "w") as f:
        print(game, file=f)
    
    print(f"Game saved to: {pgn_path}")


def benchmark_performance():
    """Benchmark AI performance"""
    print("=" * 50)
    print("V7P3R 2.0 Performance Benchmark")
    print("=" * 50)
    
    ai = V7P3RAI_v2()
    
    # Test positions
    positions = [
        ("Starting", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Middlegame", "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/3P1N2/PPPN1PPP/R1BQK2R w KQ - 0 8"),
        ("Endgame", "8/8/8/8/8/8/8/R6K w - - 0 1")
    ]
    
    total_time = 0
    total_moves = 0
    
    for name, fen in positions:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        
        print(f"\n{name} position:")
        print(f"FEN: {fen}")
        print(f"Legal moves: {len(legal_moves)}")
        
        start_time = time.time()
        move = ai.get_move(board)
        elapsed = time.time() - start_time
        
        total_time += elapsed
        total_moves += 1
        
        if move:
            analysis = ai.analyze_move(board, move)
            print(f"Best move: {move.uci()} ({elapsed:.3f}s)")
            print(f"Evaluation: {analysis['combined_score']:.3f}")
        else:
            print("No move found")
    
    print(f"\nBenchmark Results:")
    print(f"Total time: {total_time:.3f}s")
    print(f"Average time per move: {total_time/total_moves:.3f}s")
    print(f"Moves per second: {total_moves/total_time:.1f}")


def main():
    """Main demo function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="V7P3R Chess AI 2.0 Demo")
    parser.add_argument('--bounty', action='store_true', help='Test bounty system')
    parser.add_argument('--rnn', action='store_true', help='Test RNN model')
    parser.add_argument('--game', action='store_true', help='Play demo game')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    
    args = parser.parse_args()
    
    if args.all or not any([args.bounty, args.rnn, args.game, args.benchmark]):
        # Run all tests if no specific test selected
        test_bounty_system()
        test_rnn_model()
        benchmark_performance()
        play_demo_game()
    else:
        if args.bounty:
            test_bounty_system()
        if args.rnn:
            test_rnn_model()
        if args.benchmark:
            benchmark_performance()
        if args.game:
            play_demo_game()


if __name__ == "__main__":
    main()
