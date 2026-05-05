"""
Stage 2: Create Corrective Training Dataset with Dual Learning Pattern

Transforms critical positions into paired training examples:
1. Negative Example: V7P3R's position → penalize bad move, reward best move
2. Positive Example: Inverted position → reward opponent's winning pattern

Author: V7P3RAI Development Team
Date: 2026-04-24
"""

import json
import sys
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import chess
import chess.engine
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from core.chess_state_extractor import ChessStateExtractor


@dataclass
class CorrectiveExample:
    """A single training example for corrective learning."""
    example_id: str  # Unique ID
    example_type: str  # "negative" or "positive"
    source_game_id: str
    source_ply: int
    fen: str  # Position FEN
    position_features: List[float]  # 690-dim features from ChessStateExtractor
    
    # Move data (top-N moves with Stockfish analysis)
    moves: List[Tuple[str, str, str]]  # [(uci, san, promotion)] up to 10 moves
    move_scores: List[int]  # Centipawn evaluations (normalized to position's perspective)
    move_weights: List[float]  # Training weights (0.0-1.0)
    
    # Context metadata
    move_classification: str  # "blunder", "mistake", "inaccuracy"
    context: str  # "final_blunder", "tactical_failure", etc.
    original_eval_drop: int  # Original mistake severity
    opponent: str
    game_date: str


class CorrectiveDatasetGenerator:
    """Generates dual-learning corrective dataset from critical positions."""
    
    def __init__(self, stockfish_path: str, analysis_time: float = 0.5):
        self.stockfish_path = stockfish_path
        self.analysis_time = analysis_time
        self.engine = None
        self.extractor = ChessStateExtractor()
        self.examples = []
        
    def initialize_stockfish(self):
        """Initialize Stockfish engine (reuse for all analyses)."""
        print(f"   Starting Stockfish engine...")
        self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        print(f"   ✅ Stockfish ready")
        
    def cleanup(self):
        """Clean up Stockfish engine."""
        if self.engine:
            try:
                self.engine.quit()
                print(f"   ✅ Stockfish engine closed")
            except Exception:
                # Engine already dead, ignore
                pass
    
    def get_stockfish_top_moves(self, board: chess.Board, num_moves: int = 10) -> Tuple[List[Tuple[str, str]], List[int]]:
        """
        Get top N moves from Stockfish with evaluations.
        
        Returns:
            moves: List of (uci, san) move tuples
            scores: List of centipawn evaluations (from White's perspective)
        """
        if not self.engine:
            raise RuntimeError("Stockfish engine not initialized")
        
        # Analyze position with multipv for top moves
        info = self.engine.analyse(
            board, 
            chess.engine.Limit(time=self.analysis_time),
            multipv=num_moves
        )
        
        moves = []
        scores = []
        
        # Extract move and score from each PV
        for pv_info in info:
            pv = pv_info.get("pv")
            if not pv or len(pv) == 0:
                continue
            
            move = pv[0]
            move_uci = move.uci()
            move_san = board.san(move)
            
            # Get score from White's perspective
            score = pv_info.get("score")
            if score is None:
                continue
            
            score_white = score.white()
            if score_white.is_mate():
                mate_moves = score_white.mate()
                eval_cp = 10000 * (1 if mate_moves > 0 else -1)
            else:
                eval_cp = score_white.score() if score_white.score() is not None else 0
            
            moves.append((move_uci, move_san))
            scores.append(eval_cp)
        
        return moves, scores
    
    def normalize_eval_to_side(self, eval_cp: int, board: chess.Board) -> int:
        """Convert eval from White's perspective to current side's perspective."""
        if board.turn == chess.BLACK:
            return -eval_cp
        return eval_cp
    
    def calculate_move_weights(self, bad_move_uci: str, best_move_uci: str, 
                               top_moves: List[Tuple[str, str]], top_scores: List[int],
                               classification: str) -> List[float]:
        """
        Calculate training weights for moves based on classification.
        
        Bad move gets very low weight (0.0-0.2 based on severity)
        Best move gets 1.0
        Other moves get exponential decay
        """
        weights = []
        
        # Weight for bad move based on classification
        bad_move_weight = {
            "blunder": 0.0,      # Never play blunders
            "mistake": 0.1,      # Strongly avoid mistakes
            "inaccuracy": 0.2,   # Slightly penalize inaccuracies
            "good": 0.3          # Forced bad moves in lost positions
        }.get(classification, 0.1)
        
        for i, (move_uci, move_san) in enumerate(top_moves):
            if move_uci == bad_move_uci:
                # This is the bad move V7P3R played
                weights.append(bad_move_weight)
            elif move_uci == best_move_uci:
                # This is Stockfish's best move
                weights.append(1.0)
            else:
                # Other moves: exponential decay
                # Top moves after best get high weights: 0.8, 0.6, 0.4, 0.2
                decay_weight = max(0.2, 1.0 - (i * 0.2))
                weights.append(decay_weight)
        
        return weights
    
    def create_negative_example(self, critical_pos: Dict, example_id: str) -> CorrectiveExample:
        """
        Create negative example from V7P3R's perspective.
        
        Position: V7P3R's actual position before the mistake
        Goal: Penalize bad move, reward best move
        """
        # Set up board
        board = chess.Board(critical_pos['fen'])
        
        # Get Stockfish top moves
        top_moves, top_scores = self.get_stockfish_top_moves(board)
        
        # Ensure bad move and best move are in the list
        bad_move_uci = critical_pos['v7p3r_move']
        best_move_uci = critical_pos['best_move']
        
        # Check if bad move is in top moves, if not add it at the end
        move_ucis = [m[0] for m in top_moves]
        if bad_move_uci not in move_ucis:
            top_moves.append((bad_move_uci, critical_pos['v7p3r_move_san']))
            # Estimate bad move score (worse than all top moves)
            worst_score = min(top_scores) if top_scores else 0
            top_scores.append(worst_score - abs(critical_pos['eval_drop']))
        
        # Normalize scores to current side's perspective
        normalized_scores = [self.normalize_eval_to_side(score, board) for score in top_scores]
        
        # Calculate weights
        weights = self.calculate_move_weights(
            bad_move_uci, best_move_uci, top_moves, normalized_scores,
            critical_pos['move_classification']
        )
        
        # Extract position features
        features = self.extractor.extract(board).tolist()
        
        # Convert moves to (uci, san, promotion) format
        moves_encoded = []
        for uci, san in top_moves:
            move = chess.Move.from_uci(uci)
            promo = 0
            if move.promotion:
                promo = {chess.QUEEN: 1, chess.ROOK: 2, chess.BISHOP: 3, chess.KNIGHT: 4}.get(move.promotion, 0)
            moves_encoded.append((uci, san, str(promo)))
        
        return CorrectiveExample(
            example_id=example_id,
            example_type="negative",
            source_game_id=critical_pos['game_id'],
            source_ply=critical_pos['ply'],
            fen=critical_pos['fen'],
            position_features=features,
            moves=moves_encoded,
            move_scores=normalized_scores,
            move_weights=weights,
            move_classification=critical_pos['move_classification'],
            context=critical_pos['context'],
            original_eval_drop=critical_pos['eval_drop'],
            opponent=critical_pos['opponent'],
            game_date=critical_pos['game_date']
        )
    
    def create_positive_example(self, critical_pos: Dict, example_id: str) -> CorrectiveExample:
        """
        Create positive example from opponent's perspective.
        
        Position: After V7P3R's mistake (opponent's turn to capitalize)
        Goal: Learn the opponent's punishing response pattern
        """
        # Set up board and make V7P3R's bad move to get opponent's position
        board = chess.Board(critical_pos['fen'])
        v7p3r_move = chess.Move.from_uci(critical_pos['v7p3r_move'])
        board.push(v7p3r_move)
        
        # Now it's opponent's turn - get their best response
        top_moves, top_scores = self.get_stockfish_top_moves(board)
        
        # Calculate weights (standard exponential decay, all positive)
        # This teaches "when opponent makes mistake, here's how to punish them"
        weights = []
        for i in range(len(top_moves)):
            if i == 0:
                weights.append(1.0)  # Best punishing move
            else:
                weights.append(max(0.2, 1.0 - (i * 0.2)))
        
        # Normalize scores to current side's perspective (opponent's)
        normalized_scores = [self.normalize_eval_to_side(score, board) for score in top_scores]
        
        # Extract position features
        features = self.extractor.extract(board).tolist()
        
        # Convert moves to (uci, san, promotion) format
        moves_encoded = []
        for uci, san in top_moves:
            move = chess.Move.from_uci(uci)
            promo = 0
            if move.promotion:
                promo = {chess.QUEEN: 1, chess.ROOK: 2, chess.BISHOP: 3, chess.KNIGHT: 4}.get(move.promotion, 0)
            moves_encoded.append((uci, san, str(promo)))
        
        return CorrectiveExample(
            example_id=example_id,
            example_type="positive",
            source_game_id=critical_pos['game_id'],
            source_ply=critical_pos['ply'] + 1,  # One ply after (opponent's turn)
            fen=board.fen(),  # Position after V7P3R's mistake
            position_features=features,
            moves=moves_encoded,
            move_scores=normalized_scores,
            move_weights=weights,
            move_classification=critical_pos['move_classification'],
            context=f"punish_{critical_pos['context']}",
            original_eval_drop=critical_pos['eval_drop'],
            opponent=critical_pos['opponent'],
            game_date=critical_pos['game_date']
        )
    
    def process_critical_positions(self, critical_positions_file: str, max_positions: int = None):
        """
        Process critical positions and generate dual examples.
        
        Args:
            critical_positions_file: Path to critical_positions.json
            max_positions: Max positions to process (None = all)
        """
        print(f"📂 Loading critical positions...")
        with open(critical_positions_file, 'r') as f:
            data = json.load(f)
        
        critical_positions = data['positions']
        total = len(critical_positions)
        
        if max_positions:
            critical_positions = critical_positions[:max_positions]
            print(f"   Loaded {len(critical_positions)} of {total} positions (limited)")
        else:
            print(f"   Loaded {total} critical positions")
        
        print(f"\n📊 Generating dual training examples...")
        print(f"   Target: {len(critical_positions) * 2} examples (2x positions)")
        
        for i, critical_pos in enumerate(critical_positions, 1):
            if i % 100 == 0 or i == 1:
                print(f"   [{i}/{len(critical_positions)}] Processing {critical_pos['game_id']}...")
            
            # Create negative example (avoid V7P3R's mistake)
            neg_id = f"{critical_pos['game_id']}_{critical_pos['ply']}_neg"
            neg_example = self.create_negative_example(critical_pos, neg_id)
            self.examples.append(neg_example)
            
            # Create positive example (learn opponent's pattern)
            pos_id = f"{critical_pos['game_id']}_{critical_pos['ply']}_pos"
            pos_example = self.create_positive_example(critical_pos, pos_id)
            self.examples.append(pos_example)
    
    def save_dataset(self, output_file: str):
        """Save corrective dataset to JSON."""
        # Convert examples to dicts
        examples_data = [asdict(ex) for ex in self.examples]
        
        # Create metadata
        metadata = {
            "dataset_type": "corrective_training",
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "num_examples": len(self.examples),
            "num_negative": sum(1 for ex in self.examples if ex.example_type == "negative"),
            "num_positive": sum(1 for ex in self.examples if ex.example_type == "positive"),
            "stockfish_analysis_time": self.analysis_time,
            "feature_dimensions": 690
        }
        
        # Save
        dataset = {
            "metadata": metadata,
            "examples": examples_data
        }
        
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"\n💾 Saved dataset to: {output_file}")
        print(f"   Size: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")
    
    def print_statistics(self):
        """Print dataset statistics."""
        if not self.examples:
            return
        
        print(f"\n📈 Dataset Statistics:")
        print(f"   Total Examples: {len(self.examples)}")
        
        # By type
        neg_count = sum(1 for ex in self.examples if ex.example_type == "negative")
        pos_count = sum(1 for ex in self.examples if ex.example_type == "positive")
        print(f"   Negative (Avoid): {neg_count} ({neg_count/len(self.examples)*100:.1f}%)")
        print(f"   Positive (Exploit): {pos_count} ({pos_count/len(self.examples)*100:.1f}%)")
        
        # By classification
        classifications = {}
        for ex in self.examples:
            classifications[ex.move_classification] = classifications.get(ex.move_classification, 0) + 1
        
        print(f"\n   By Move Classification:")
        for classification, count in sorted(classifications.items(), key=lambda x: -x[1]):
            print(f"      {classification}: {count} ({count/len(self.examples)*100:.1f}%)")
        
        # By context
        contexts = {}
        for ex in self.examples:
            contexts[ex.context] = contexts.get(ex.context, 0) + 1
        
        print(f"\n   By Context:")
        for context, count in sorted(contexts.items(), key=lambda x: -x[1])[:10]:
            print(f"      {context}: {count} ({count/len(self.examples)*100:.1f}%)")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate corrective training dataset with dual learning')
    parser.add_argument('--critical-positions', type=str, 
                       default='data/stage2_positions/critical_positions.json',
                       help='Path to critical positions JSON')
    parser.add_argument('--output', type=str,
                       default='data/stage2_training/corrective_dataset.json',
                       help='Output dataset path')
    parser.add_argument('--stockfish-path', type=str,
                       default=r'E:\Programming Stuff\Chess Engines\Tournament Engines\Stockfish\stockfish-windows-x86-64-avx2.exe',
                       help='Path to Stockfish executable')
    parser.add_argument('--analysis-time', type=float, default=0.5,
                       help='Stockfish analysis time per position (seconds)')
    parser.add_argument('--max-positions', type=int, default=None,
                       help='Max positions to process (for testing)')
    
    args = parser.parse_args()
    
    print("🚀 V7P3R Corrective Dataset Generator (Stage 2)")
    print("=" * 60)
    print(f"📂 Critical positions: {args.critical_positions}")
    print(f"💾 Output: {args.output}")
    print(f"⚙️  Stockfish: {args.stockfish_path}")
    print(f"⏱️  Analysis time: {args.analysis_time}s per position")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    generator = CorrectiveDatasetGenerator(args.stockfish_path, args.analysis_time)
    
    try:
        # Initialize Stockfish
        generator.initialize_stockfish()
        
        # Process positions
        generator.process_critical_positions(args.critical_positions, args.max_positions)
        
        # Save dataset
        generator.save_dataset(args.output)
        
        # Print statistics
        generator.print_statistics()
        
        print(f"\n✅ Dataset generation complete!")
        
    finally:
        # Cleanup
        generator.cleanup()


if __name__ == '__main__':
    main()
