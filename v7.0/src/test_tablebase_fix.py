"""
Quick test to verify tablebase move selection is working in selfplay.

This should show that when a tablebase position is reached,
the engine uses tablebase moves instead of the weak neural network.
"""

import chess
import torch
from pathlib import Path

from selfplay_trainer import SelfPlayGame
from comprehensive_features import ComprehensiveFeatureExtractor
from stockfish_oracle import StockfishOracle
from personality_tuner import PlaystyleProfile
from personality_rewards import PersonalityRewardCalculator
from phase_manager import PhaseAwareTrainingTarget, DynamicWeightCalculator
from tablebase_oracle import TablebaseOracle
from network import create_v7_network

print("=" * 60)
print("TABLEBASE INTEGRATION TEST")
print("=" * 60)

# Load the supervised model
SUPERVISED_MODEL = "../training/supervised_gm/supervised_final.pt"
PROFILE_PATH = "../profiles/dark_forest_assassin.json"
STOCKFISH_PATH = "../../stockfish.exe"
TABLEBASE_PATH = r"E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\pgn_training_data\pgn_data_endgames\3-4-5_pieces_Syzygy\3-4-5"

print(f"\nLoading supervised model: {SUPERVISED_MODEL}")
network, trainer = create_v7_network()
network.load_state_dict(torch.load(SUPERVISED_MODEL))
print("✓ Model loaded")

print(f"\nLoading profile: {PROFILE_PATH}")
import json
with open(PROFILE_PATH, 'r') as f:
    profile_data = json.load(f)
profile = PlaystyleProfile.from_dict(profile_data)
print("✓ Profile loaded")

print("\nInitializing components...")
extractor = ComprehensiveFeatureExtractor()
calculator = PersonalityRewardCalculator(profile.weights)
oracle = StockfishOracle(STOCKFISH_PATH)
phase_manager = PhaseAwareTrainingTarget(DynamicWeightCalculator(
    opening_sf_weight=0.9,
    middlegame_sf_weight=0.2,
    endgame_sf_weight=1.0,
    tablebase_sf_weight=1.0
))
tablebase_oracle = TablebaseOracle(TABLEBASE_PATH)

print(f"✓ Tablebase enabled: {tablebase_oracle.enabled}")
print(f"✓ Max pieces: {tablebase_oracle.max_pieces}")

# Create SelfPlayGame instance
game_player = SelfPlayGame(
    network=network,
    oracle=oracle,
    calculator=calculator,
    extractor=extractor,
    phase_manager=phase_manager,
    tablebase_oracle=tablebase_oracle,
    temperature=0.5
)

print("\n" + "=" * 60)
print("TEST 1: Simple K+Q vs K endgame (tablebase position)")
print("=" * 60)

# Set up K+Q vs K position (White to move, forced mate)
board = chess.Board("4k3/8/8/8/8/8/4Q3/4K3 w - - 0 1")
print(f"Position: {board.fen()}")
print(f"Pieces on board: {len(board.piece_map())} (should be in tablebase)")

# Check if tablebase is available
if tablebase_oracle.is_available(board):
    print("✓ Position IS in tablebase")
    
    # Get tablebase move
    tb_move = tablebase_oracle.get_best_move(board)
    print(f"✓ Tablebase best move: {tb_move}")
    
    # Get engine's selected move
    engine_move = game_player.select_move(board)
    print(f"✓ Engine selected move: {engine_move}")
    
    if tb_move == engine_move:
        print("\n✅ SUCCESS: Engine is using tablebase moves!")
    else:
        print(f"\n❌ FAILURE: Engine chose {engine_move}, tablebase says {tb_move}")
else:
    print("❌ Position NOT in tablebase (tablebase not working)")

print("\n" + "=" * 60)
print("TEST 2: Complex middlegame (NOT in tablebase)")
print("=" * 60)

# Standard opening position (many pieces)
board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
print(f"Position: {board.fen()}")
print(f"Pieces on board: {len(board.piece_map())} (too many for tablebase)")

# Check if tablebase is available
if not tablebase_oracle.is_available(board):
    print("✓ Position NOT in tablebase (as expected)")
    
    # Get engine's selected move
    engine_move = game_player.select_move(board)
    print(f"✓ Engine selected move using neural network: {engine_move}")
    print("\n✅ SUCCESS: Engine falls back to neural network when tablebase unavailable")
else:
    print("❌ Unexpected: This position should NOT be in tablebase")

print("\n" + "=" * 60)
print("TEST 3: K+R vs K endgame")
print("=" * 60)

board = chess.Board("4k3/8/8/8/8/8/4R3/4K3 w - - 0 1")
print(f"Position: {board.fen()}")

if tablebase_oracle.is_available(board):
    print("✓ Position IS in tablebase")
    tb_move = tablebase_oracle.get_best_move(board)
    engine_move = game_player.select_move(board)
    print(f"✓ Tablebase move: {tb_move}")
    print(f"✓ Engine move: {engine_move}")
    
    if tb_move == engine_move:
        print("\n✅ SUCCESS: Engine using tablebase for K+R vs K")
    else:
        print(f"\n❌ FAILURE: Mismatch in K+R vs K endgame")
else:
    print("❌ K+R vs K should be in tablebase")

oracle.stop()

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("If all tests passed, the tablebase integration is working correctly.")
print("Games should now finish instead of hitting 190 move limit!")
print("=" * 60)
