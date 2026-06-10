"""
V7P3R v7.0 - Position Testing Tool

Test personality profiles on chess positions to see move preferences.
Shows which moves different personalities would prefer and why.
"""

import chess
import json
from pathlib import Path
from comprehensive_features import ComprehensiveFeatureExtractor
from personality_rewards import PersonalityRewardCalculator, PersonalityWeights
from personality_tuner import PlaystyleProfile


class PositionTester:
    """Test personality profiles on specific positions."""
    
    def __init__(self):
        self.extractor = ComprehensiveFeatureExtractor()
    
    def load_profile(self, profile_path: str) -> PlaystyleProfile:
        """Load personality profile from JSON."""
        with open(profile_path, 'r') as f:
            data = json.load(f)
        return PlaystyleProfile.from_dict(data)
    
    def evaluate_move(
        self,
        board: chess.Board,
        move: chess.Move,
        calculator: PersonalityRewardCalculator,
        stockfish_eval: float = 0.0
    ) -> dict:
        """
        Evaluate a move using personality rewards.
        
        Args:
            board: Current position
            move: Move to evaluate
            calculator: Personality calculator
            stockfish_eval: Stockfish evaluation after move
        
        Returns:
            Evaluation breakdown
        """
        # Features before move
        features_before = self.extractor.extract_all_features_dict(board)
        
        # Make move
        board_copy = board.copy()
        board_copy.push(move)
        
        # Features after move
        features_after = self.extractor.extract_all_features_dict(board_copy)
        
        # Calculate personality reward
        reward_result = calculator.calculate_total_reward(
            features_after,
            features_before,
            stockfish_eval=stockfish_eval
        )
        
        return {
            'move': board.san(move),
            'move_uci': move.uci(),
            'features_before': features_before,
            'features_after': features_after,
            'stockfish_eval': stockfish_eval,
            'personality_reward': reward_result['personality_total'],
            'final_reward': reward_result['final_reward'],
            'breakdown': reward_result
        }
    
    def compare_moves(
        self,
        fen: str,
        moves: list,  # List of UCI moves or SAN moves
        profile_name: str,
        stockfish_evals: dict = None  # {move_uci: eval_cp}
    ) -> dict:
        """
        Compare how a personality ranks different moves.
        
        Args:
            fen: Position FEN
            moves: Candidate moves to compare
            profile_name: Personality profile to use
            stockfish_evals: Optional Stockfish evaluations per move
        
        Returns:
            Comparison results
        """
        board = chess.Board(fen)
        
        # Load profile
        profile = self.load_profile(f"../../profiles/{profile_name}.json")
        calculator = PersonalityRewardCalculator(profile.weights)
        
        # Evaluate each move
        evaluations = []
        for move_str in moves:
            try:
                # Parse move
                if len(move_str) in [4, 5]:  # UCI format
                    move = chess.Move.from_uci(move_str)
                else:  # SAN format
                    move = board.parse_san(move_str)
                
                # Get Stockfish eval if provided
                stockfish_eval = 0.0
                if stockfish_evals and move.uci() in stockfish_evals:
                    stockfish_eval = stockfish_evals[move.uci()] / 100.0  # cp to eval
                
                # Evaluate move
                evaluation = self.evaluate_move(board, move, calculator, stockfish_eval)
                evaluations.append(evaluation)
                
            except Exception as e:
                print(f"Error evaluating move {move_str}: {e}")
        
        # Sort by final reward
        evaluations.sort(key=lambda x: x['final_reward'], reverse=True)
        
        return {
            'fen': fen,
            'profile': profile_name,
            'evaluations': evaluations
        }
    
    def print_move_comparison(self, results: dict):
        """Pretty-print move comparison results."""
        print(f"\n{'='*60}")
        print(f"POSITION: {results['fen']}")
        print(f"PROFILE: {results['profile']}")
        print(f"{'='*60}")
        
        for i, eval_data in enumerate(results['evaluations'], 1):
            print(f"\n{i}. {eval_data['move']:8s} (UCI: {eval_data['move_uci']})")
            print(f"   Stockfish: {eval_data['stockfish_eval']:+.2f}")
            print(f"   Personality: {eval_data['personality_reward']:+.3f}")
            print(f"   Final Reward: {eval_data['final_reward']:+.3f}")
            
            # Show key feature changes
            features_before = eval_data['features_before']
            features_after = eval_data['features_after']
            
            darkness_delta = features_after['forest_darkness_score'] - features_before['forest_darkness_score']
            material_delta = features_after['material_balance'] - features_before['material_balance']
            
            print(f"   Forest Darkness: {features_before['forest_darkness_score']:.2f} → {features_after['forest_darkness_score']:.2f} ({darkness_delta:+.2f})")
            print(f"   Material: {features_before['material_balance']:+.0f} → {features_after['material_balance']:+.0f} ({material_delta:+.0f})")
            
            # Show breakdown
            breakdown = eval_data['breakdown']
            if breakdown['complexity_reward'] > 0.02:
                print(f"   Complexity Bonus: +{breakdown['complexity_reward']:.3f}")
            if breakdown['sacrifice_reward'] > 0.01:
                print(f"   Sacrifice Bonus: +{breakdown['sacrifice_reward']:.3f}")
            if breakdown['king_reward'] > 0.01:
                print(f"   Attack Bonus: +{breakdown['king_reward']:.3f}")


# Example test positions
TEST_POSITIONS = {
    'starting': "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    
    'sicilian_sharp': "r1bqkb1r/pp3ppp/2n1pn2/2ppP3/3P4/2P2N2/PP3PPP/RNBQKB1R w KQkq - 0 1",
    
    'tal_sacrifice': "r2qk2r/ppp2ppp/2n1bn2/3p4/3P4/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 1",
    
    'endgame_technical': "8/4kp2/8/4P3/4K3/8/8/8 w - - 0 1",
}


if __name__ == "__main__":
    print("="*60)
    print("V7P3R POSITION TESTING TOOL")
    print("="*60)
    
    tester = PositionTester()
    
    # Test 1: Starting position - compare e4 vs d4 vs Nf3
    print("\n\nTEST 1: STARTING POSITION")
    print("Comparing: e4 (classical) vs d4 (classical) vs Nf3 (flexible)")
    
    results = tester.compare_moves(
        fen=TEST_POSITIONS['starting'],
        moves=['e2e4', 'd2d4', 'g1f3'],
        profile_name='dark_forest_assassin',
        stockfish_evals={
            'e2e4': 30,  # Stockfish slightly prefers e4
            'd2d4': 35,
            'g1f3': 25
        }
    )
    
    tester.print_move_comparison(results)
    
    # Test 2: Sharp Sicilian - sacrifice vs safe
    print("\n\n" + "="*60)
    print("TEST 2: SHARP SICILIAN POSITION")
    print("Comparing: Nd5! (sacrifice) vs Be2 (safe) vs O-O (develop)")
    
    results = tester.compare_moves(
        fen=TEST_POSITIONS['sicilian_sharp'],
        moves=['c3d5', 'f1e2', 'e1g1'],  # Nd5, Be2, O-O
        profile_name='dark_forest_assassin',
        stockfish_evals={
            'c3d5': 50,   # Sacrifice with compensation
            'f1e2': 40,   # Safe development
            'e1g1': 35    # Castle
        }
    )
    
    tester.print_move_comparison(results)
    
    # Test 3: Compare Dark Forest vs Tal
    print("\n\n" + "="*60)
    print("TEST 3: PERSONALITY COMPARISON ON SACRIFICE POSITION")
    print("Position: Tal-style attacking position")
    print("Comparing: Dark Forest Assassin vs Tal profile")
    
    # Dark Forest
    results_df = tester.compare_moves(
        fen=TEST_POSITIONS['tal_sacrifice'],
        moves=['c3d5', 'f1e2', 'e1g1'],  # Nd5 sacrifice, Be2 safe, O-O
        profile_name='dark_forest_assassin',
        stockfish_evals={
            'c3d5': 60,
            'f1e2': 50,
            'e1g1': 45
        }
    )
    
    print("\n\n🔥 DARK FOREST ASSASSIN:")
    tester.print_move_comparison(results_df)
    
    # Tal
    # First create Tal profile if it doesn't exist
    from personality_tuner import PersonalityTuner
    tuner = PersonalityTuner()
    tal_profile = tuner.get_profile('tal')
    with open('../../profiles/tal.json', 'w') as f:
        json.dump(tal_profile.to_dict(), f, indent=2)
    
    results_tal = tester.compare_moves(
        fen=TEST_POSITIONS['tal_sacrifice'],
        moves=['c3d5', 'f1e2', 'e1g1'],
        profile_name='tal',
        stockfish_evals={
            'c3d5': 60,
            'f1e2': 50,
            'e1g1': 45
        }
    )
    
    print("\n\n⚡ TAL:")
    tester.print_move_comparison(results_tal)
    
    print("\n\n" + "="*60)
    print("📊 COMPARISON SUMMARY")
    print("="*60)
    print("\nDark Forest Assassin should prefer sacrifices MORE than Tal")
    print("due to higher weights:")
    print("  - Forest Darkness: 0.20 vs 0.15 (+33%)")
    print("  - Piece Tension: 0.15 vs 0.10 (+50%)")
    print("  - Material Sacrifice: 0.15 vs 0.10 (+50%)")
    print("  - Attack Bonus: 0.12 vs 0.08 (+50%)")
    
    print("\n✅ Position testing framework ready!")
    print("\n📝 Usage:")
    print("  from position_tester import PositionTester")
    print("  tester = PositionTester()")
    print("  results = tester.compare_moves(fen, moves, 'dark_forest_assassin')")
    print("  tester.print_move_comparison(results)")
