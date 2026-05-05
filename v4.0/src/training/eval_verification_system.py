#!/usr/bin/env python3
"""
V7P3R Evaluation Verification & Flagging System

Active learning system that:
1. Uses V7P3R's evaluations as primary training signal
2. Verifies against Lichess Stockfish database (95GB)
3. Flags positions where V7P3R disagrees significantly
4. Creates corrective training dataset for V7P3R improvement

Training Philosophy:
- Imitate V7P3R's personality (not Stockfish's)
- Use Stockfish only as sanity check
- Flag eval bugs → Fix V7P3R → Retrain on corrections
- Preserve V7P3R's unique playing style

Author: Pat Snyder
Created: 2026-05-03 (Eval Verification System v1.0)
"""

import chess
import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from tqdm import tqdm

from data.lichess_eval_indexer import LichessEvalIndexer, LichessEvaluation
from evaluation.v7p3r_ai_evaluator import V7P3RAIEvaluator, EvaluationFeatures
from training.v7p3r_reward_system import V7P3RRewardCalculator, TrainingReward


class EvalAgreementLevel(Enum):
    """Classification of V7P3R vs Lichess agreement"""
    PERFECT_MATCH = "perfect_match"        # Within 10cp
    GOOD_MATCH = "good_match"              # Within 50cp
    ACCEPTABLE = "acceptable"              # Within 100cp
    SIGNIFICANT_DIFFERENCE = "significant" # 100-200cp difference
    MAJOR_DISAGREEMENT = "major"           # >200cp difference
    MOVE_MISMATCH = "move_mismatch"        # Different best moves
    MATE_DISAGREEMENT = "mate_disagree"    # One sees mate, other doesn't


@dataclass
class VerificationResult:
    """
    Result of verifying V7P3R evaluation against Lichess database.
    """
    # Position
    fen: str
    
    # V7P3R evaluation
    v7p3r_score: int          # Centipawns
    v7p3r_best_move: str      # UCI move
    v7p3r_features: np.ndarray  # 58-dim feature vector
    
    # Lichess evaluation (Stockfish ground truth)
    lichess_score: Optional[int]  # Centipawns (None if not in DB)
    lichess_mate_in: Optional[int]
    lichess_best_move: Optional[str]  # UCI move
    lichess_depth: int
    
    # Agreement analysis
    agreement_level: EvalAgreementLevel
    eval_difference: int      # abs(v7p3r - lichess) in centipawns
    moves_match: bool         # Do best moves agree?
    
    # Flagging
    flagged: bool             # Should this position be reviewed?
    flag_reason: str          # Why flagged?
    
    # Training decision
    use_for_training: bool    # Safe to use V7P3R eval for training?
    confidence: float         # Confidence in V7P3R eval [0, 1]


class EvalVerificationSystem:
    """
    Verification system for V7P3R evaluations during training.
    
    Workflow:
    1. Get position from training data
    2. V7P3R evaluates position → score + features
    3. Look up position in Lichess DB → Stockfish score
    4. Compare evaluations:
       - If close match (< threshold) → Use V7P3R eval confidently
       - If significant difference → FLAG for review
    5. Flagged positions are saved for later V7P3R improvement
    """
    
    # Thresholds for agreement levels (centipawns)
    PERFECT_MATCH_THRESHOLD = 10
    GOOD_MATCH_THRESHOLD = 50
    ACCEPTABLE_THRESHOLD = 100
    SIGNIFICANT_THRESHOLD = 200
    
    # Confidence levels for training
    HIGH_CONFIDENCE = 1.0     # Perfect match
    MEDIUM_CONFIDENCE = 0.7   # Good match
    LOW_CONFIDENCE = 0.3      # Acceptable
    NO_CONFIDENCE = 0.0       # Flag for review
    
    def __init__(self,
                 v7p3r_reward_calculator: V7P3RRewardCalculator,
                 lichess_indexer: LichessEvalIndexer,
                 flag_output_dir: str = "flags/eval_discrepancies"):
        """
        Initialize verification system.
        
        Args:
            v7p3r_reward_calculator: V7P3R reward calculator
            lichess_indexer: Lichess evaluation database indexer
            flag_output_dir: Directory to save flagged positions
        """
        self.v7p3r_calc = v7p3r_reward_calculator
        self.lichess_db = lichess_indexer
        self.flag_dir = flag_output_dir
        
        os.makedirs(flag_output_dir, exist_ok=True)
        
        # Statistics
        self.stats = {
            'total_verified': 0,
            'perfect_matches': 0,
            'good_matches': 0,
            'acceptable': 0,
            'significant_differences': 0,
            'major_disagreements': 0,
            'flagged': 0,
            'not_in_db': 0
        }
    
    def verify_position(self, board: chess.Board) -> VerificationResult:
        """
        Verify V7P3R's evaluation of a position against Lichess DB.
        
        Args:
            board: Chess position to verify
            
        Returns:
            VerificationResult with agreement analysis and flagging decision
        """
        fen = board.fen()
        
        # Get V7P3R evaluation
        v7p3r_rewards = self.v7p3r_calc.calculate_move_rewards(board)
        
        if not v7p3r_rewards:
            # No legal moves (checkmate/stalemate)
            return self._create_terminal_result(board)
        
        # Best move according to V7P3R
        best_reward = v7p3r_rewards[0]  # Sorted by quality
        v7p3r_score = best_reward.v7p3r_score
        v7p3r_move = best_reward.move_uci
        v7p3r_features = best_reward.features
        
        # Look up in Lichess database
        lichess_eval = self.lichess_db.lookup(fen)
        
        if lichess_eval is None:
            # Position not in database
            self.stats['not_in_db'] += 1
            
            return VerificationResult(
                fen=fen,
                v7p3r_score=v7p3r_score,
                v7p3r_best_move=v7p3r_move,
                v7p3r_features=v7p3r_features,
                lichess_score=None,
                lichess_mate_in=None,
                lichess_best_move=None,
                lichess_depth=0,
                agreement_level=EvalAgreementLevel.ACCEPTABLE,
                eval_difference=0,
                moves_match=False,
                flagged=False,
                flag_reason="",
                use_for_training=True,  # Trust V7P3R when no ground truth
                confidence=self.MEDIUM_CONFIDENCE
            )
        
        # Convert Lichess evaluation to centipawns
        if lichess_eval.mate_in is not None:
            # Mate score - convert to large centipawn value
            lichess_score = 10000 if lichess_eval.mate_in > 0 else -10000
        else:
            lichess_score = lichess_eval.cp_score
        
        # Get Lichess best move
        lichess_move = lichess_eval.pv[0] if lichess_eval.pv else None
        
        # Calculate agreement
        eval_diff = abs(v7p3r_score - lichess_score)
        moves_match = (v7p3r_move == lichess_move) if lichess_move else False
        
        # Classify agreement level
        agreement, confidence, flagged, flag_reason = self._classify_agreement(
            eval_diff=eval_diff,
            moves_match=moves_match,
            v7p3r_score=v7p3r_score,
            lichess_score=lichess_score,
            lichess_mate=lichess_eval.mate_in
        )
        
        # Update statistics
        self.stats['total_verified'] += 1
        if agreement == EvalAgreementLevel.PERFECT_MATCH:
            self.stats['perfect_matches'] += 1
        elif agreement == EvalAgreementLevel.GOOD_MATCH:
            self.stats['good_matches'] += 1
        elif agreement == EvalAgreementLevel.ACCEPTABLE:
            self.stats['acceptable'] += 1
        elif agreement == EvalAgreementLevel.SIGNIFICANT_DIFFERENCE:
            self.stats['significant_differences'] += 1
        elif agreement == EvalAgreementLevel.MAJOR_DISAGREEMENT:
            self.stats['major_disagreements'] += 1
        
        if flagged:
            self.stats['flagged'] += 1
        
        return VerificationResult(
            fen=fen,
            v7p3r_score=v7p3r_score,
            v7p3r_best_move=v7p3r_move,
            v7p3r_features=v7p3r_features,
            lichess_score=lichess_score,
            lichess_mate_in=lichess_eval.mate_in,
            lichess_best_move=lichess_move,
            lichess_depth=lichess_eval.depth,
            agreement_level=agreement,
            eval_difference=eval_diff,
            moves_match=moves_match,
            flagged=flagged,
            flag_reason=flag_reason,
            use_for_training=not flagged,  # Don't train on flagged positions
            confidence=confidence
        )
    
    def _classify_agreement(self,
                          eval_diff: int,
                          moves_match: bool,
                          v7p3r_score: int,
                          lichess_score: int,
                          lichess_mate: Optional[int]) -> Tuple[EvalAgreementLevel, float, bool, str]:
        """
        Classify agreement level and decide if position should be flagged.
        
        Returns:
            (agreement_level, confidence, flagged, flag_reason)
        """
        # Check for mate disagreements
        v7p3r_sees_mate = abs(v7p3r_score) > 9000
        lichess_sees_mate = lichess_mate is not None
        
        if v7p3r_sees_mate != lichess_sees_mate:
            return (
                EvalAgreementLevel.MATE_DISAGREEMENT,
                self.NO_CONFIDENCE,
                True,
                f"V7P3R sees {'mate' if v7p3r_sees_mate else 'no mate'}, "
                f"Stockfish sees {'mate' if lichess_sees_mate else 'no mate'}"
            )
        
        # Check move agreement
        if not moves_match and eval_diff > self.GOOD_MATCH_THRESHOLD:
            return (
                EvalAgreementLevel.MOVE_MISMATCH,
                self.LOW_CONFIDENCE,
                True,
                f"Different best moves with {eval_diff}cp difference"
            )
        
        # Classify by eval difference
        if eval_diff <= self.PERFECT_MATCH_THRESHOLD:
            return (
                EvalAgreementLevel.PERFECT_MATCH,
                self.HIGH_CONFIDENCE,
                False,
                ""
            )
        elif eval_diff <= self.GOOD_MATCH_THRESHOLD:
            return (
                EvalAgreementLevel.GOOD_MATCH,
                self.MEDIUM_CONFIDENCE,
                False,
                ""
            )
        elif eval_diff <= self.ACCEPTABLE_THRESHOLD:
            return (
                EvalAgreementLevel.ACCEPTABLE,
                self.LOW_CONFIDENCE,
                False,
                ""
            )
        elif eval_diff <= self.SIGNIFICANT_THRESHOLD:
            return (
                EvalAgreementLevel.SIGNIFICANT_DIFFERENCE,
                self.NO_CONFIDENCE,
                True,
                f"Eval difference: {eval_diff}cp (V7P3R: {v7p3r_score}, Stockfish: {lichess_score})"
            )
        else:
            return (
                EvalAgreementLevel.MAJOR_DISAGREEMENT,
                self.NO_CONFIDENCE,
                True,
                f"Major eval difference: {eval_diff}cp (V7P3R: {v7p3r_score}, Stockfish: {lichess_score})"
            )
    
    def _create_terminal_result(self, board: chess.Board) -> VerificationResult:
        """Handle checkmate/stalemate positions"""
        fen = board.fen()
        
        if board.is_checkmate():
            score = -10000  # Mated
        else:
            score = 0  # Stalemate/draw
        
        return VerificationResult(
            fen=fen,
            v7p3r_score=score,
            v7p3r_best_move="",
            v7p3r_features=np.zeros(58, dtype=np.float32),
            lichess_score=score,
            lichess_mate_in=1 if board.is_checkmate() else None,
            lichess_best_move=None,
            lichess_depth=0,
            agreement_level=EvalAgreementLevel.PERFECT_MATCH,
            eval_difference=0,
            moves_match=True,
            flagged=False,
            flag_reason="",
            use_for_training=False,  # Terminal position
            confidence=1.0
        )
    
    def save_flagged_positions(self, results: List[VerificationResult], batch_name: str = "batch"):
        """
        Save flagged positions to file for later review and V7P3R improvement.
        
        Args:
            results: List of verification results
            batch_name: Name for this batch of flagged positions
        """
        flagged = [r for r in results if r.flagged]
        
        if not flagged:
            return
        
        output_path = os.path.join(self.flag_dir, f"flagged_{batch_name}.jsonl")
        
        with open(output_path, 'w') as f:
            for result in flagged:
                # Convert to serializable format
                data = {
                    'fen': result.fen,
                    'v7p3r_score': result.v7p3r_score,
                    'v7p3r_best_move': result.v7p3r_best_move,
                    'lichess_score': result.lichess_score,
                    'lichess_mate_in': result.lichess_mate_in,
                    'lichess_best_move': result.lichess_best_move,
                    'eval_difference': result.eval_difference,
                    'agreement_level': result.agreement_level.value,
                    'flag_reason': result.flag_reason,
                    'features': result.v7p3r_features.tolist()
                }
                
                f.write(json.dumps(data) + '\n')
        
        print(f"✓ Saved {len(flagged)} flagged positions to {output_path}")
    
    def get_statistics(self) -> Dict:
        """Get verification statistics"""
        total = self.stats['total_verified']
        
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'perfect_match_rate': self.stats['perfect_matches'] / total,
            'good_match_rate': self.stats['good_matches'] / total,
            'flag_rate': self.stats['flagged'] / total,
            'db_coverage': 1.0 - (self.stats['not_in_db'] / total)
        }
    
    def print_statistics(self):
        """Print verification statistics"""
        stats = self.get_statistics()
        
        print("\n" + "="*80)
        print("Evaluation Verification Statistics")
        print("="*80)
        print(f"Total positions verified: {stats['total_verified']:,}")
        print(f"\nAgreement Levels:")
        print(f"  Perfect matches (≤10cp):  {stats['perfect_matches']:6,} ({stats.get('perfect_match_rate', 0):.1%})")
        print(f"  Good matches (≤50cp):     {stats['good_matches']:6,} ({stats.get('good_match_rate', 0):.1%})")
        print(f"  Acceptable (≤100cp):      {stats['acceptable']:6,}")
        print(f"  Significant diff (>100cp): {stats['significant_differences']:6,}")
        print(f"  Major disagreements:      {stats['major_disagreements']:6,}")
        print(f"\nFlagged for Review:        {stats['flagged']:6,} ({stats.get('flag_rate', 0):.1%})")
        print(f"Not in Lichess DB:         {stats['not_in_db']:6,}")
        print(f"Database coverage:         {stats.get('db_coverage', 0):.1%}")


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    from evaluation.v7p3r_ai_evaluator import V7P3RAIEvaluator
    from training.v7p3r_reward_system import V7P3RRewardCalculator
    from data.lichess_eval_indexer import LichessEvalIndexer
    
    # Initialize components
    print("Initializing verification system...")
    
    # V7P3R evaluator
    evaluator = V7P3RAIEvaluator()
    
    # V7P3R engine path (v18.3 - highest achiever)
    v7p3r_path = r"e:\Programming Stuff\Chess Engines\V7P3R Chess Engine\v7p3r-chess-engine\lichess\engines\V7P3R_v18.3_20251229\v7p3r_uci.py"
    
    reward_calc = V7P3RRewardCalculator(
        v7p3r_engine_path=v7p3r_path,
        feature_evaluator=evaluator,
        search_depth=3
    )
    
    # Lichess database indexer
    lichess_db_path = r"E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\pgn_training_data\json_data_lichess_evaluations_db\lichess_db_eval.jsonl\lichess_db_eval.jsonl"
    
    indexer = LichessEvalIndexer(
        jsonl_path=lichess_db_path,
        rebuild_index=False
    )
    
    # Verification system
    verifier = EvalVerificationSystem(
        v7p3r_reward_calculator=reward_calc,
        lichess_indexer=indexer,
        flag_output_dir="flags/eval_discrepancies"
    )
    
    # Test positions
    test_positions = [
        chess.Board(),  # Starting position
        chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"),  # Sicilian
    ]
    
    print("\n" + "="*80)
    print("Verifying Test Positions")
    print("="*80)
    
    results = []
    for board in test_positions:
        print(f"\nVerifying: {board.fen()}")
        result = verifier.verify_position(board)
        
        print(f"  V7P3R:    {result.v7p3r_score:+5d} cp → {result.v7p3r_best_move}")
        print(f"  Stockfish: {result.lichess_score:+5d} cp → {result.lichess_best_move}")
        print(f"  Difference: {result.eval_difference} cp")
        print(f"  Agreement: {result.agreement_level.value}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Flagged: {'YES - ' + result.flag_reason if result.flagged else 'NO'}")
        
        results.append(result)
    
    # Save flagged positions
    verifier.save_flagged_positions(results, batch_name="test_batch")
    
    # Print statistics
    verifier.print_statistics()
