"""
Extract Stage 1 Training Data from Stage 2 Self-Play Results
Converts self-play position data into GOOD/BAD labels for Stage 1 incremental training.

Labeling Strategy:
- GOOD positions: eval_cp > 50 (clear advantage)
- BAD positions: eval_cp < -50 (clear disadvantage)
- NEUTRAL positions: -50 <= eval_cp <= 50 (skip or balance)

This creates balanced training data from self-play results to feed back into Stage 1.
"""
import json
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

class Stage1LabelExtractor:
    def __init__(self, selfplay_dir: str, output_dir: str):
        self.selfplay_dir = Path(selfplay_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Labeling thresholds
        self.good_threshold = 50  # cp
        self.bad_threshold = -50  # cp
        
    def load_selfplay_positions(self) -> List[Dict]:
        """Load all positions from self-play JSONL files."""
        positions = []
        position_files = list(self.selfplay_dir.glob("selfplay_*_positions.jsonl"))
        
        print(f"Loading positions from {len(position_files)} games...")
        for file_path in position_files:
            with open(file_path, 'r') as f:
                for line in f:
                    position = json.loads(line)
                    positions.append(position)
        
        print(f"✅ Loaded {len(positions)} positions")
        return positions
    
    def extract_labels(self, positions: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Extract GOOD, BAD, and NEUTRAL positions based on eval_cp and game outcome.
        
        Label strategy:
        - Use game_result to determine position quality from each player's perspective
        - Winning side's positions = GOOD, losing side's positions = BAD
        - Draws = NEUTRAL (though we have zero draws in current data)
        
        This creates training signal from actual game outcomes, not just eval scores.
        
        Returns:
            (good_positions, bad_positions, neutral_positions)
        """
        good_positions = []
        bad_positions = []
        neutral_positions = []
        
        for pos in positions:
            game_result = pos.get('game_result', '1/2-1/2')
            side_to_move = pos.get('side_to_move', 'white')
            
            # Determine if this position was on winning or losing side
            if game_result == '1-0':  # White won
                if side_to_move == 'white':
                    label = 'GOOD'
                else:
                    label = 'BAD'
            elif game_result == '0-1':  # Black won
                if side_to_move == 'black':
                    label = 'GOOD'
                else:
                    label = 'BAD'
            else:  # Draw
                label = 'NEUTRAL'
            
            # Create position record
            pos_record = {
                'fen': pos['fen'],
                'eval_cp': pos.get('eval_cp', 0),
                'label': label,
                'side_to_move': side_to_move,
                'game_result': game_result,
                'move_number': pos['move_number'],
                'game_id': pos['game_id'],
                'complexity_score': pos.get('labels', {}).get('complexity_score', 0)
            }
            
            if label == 'GOOD':
                good_positions.append(pos_record)
            elif label == 'BAD':
                bad_positions.append(pos_record)
            else:
                neutral_positions.append(pos_record)
        
        return good_positions, bad_positions, neutral_positions
    
    def analyze_distribution(self, good_pos: List[Dict], bad_pos: List[Dict], 
                           neutral_pos: List[Dict]):
        """Analyze and report data distribution."""
        print("\n" + "="*60)
        print("STAGE 1 LABEL EXTRACTION ANALYSIS")
        print("="*60)
        
        total = len(good_pos) + len(bad_pos) + len(neutral_pos)
        
        print(f"\n📊 Label Distribution:")
        print(f"  GOOD (eval > +50cp):     {len(good_pos):5d} ({len(good_pos)/total*100:5.1f}%)")
        print(f"  BAD (eval < -50cp):      {len(bad_pos):5d} ({len(bad_pos)/total*100:5.1f}%)")
        print(f"  NEUTRAL (-50 to +50cp):  {len(neutral_pos):5d} ({len(neutral_pos)/total*100:5.1f}%)")
        print(f"  TOTAL:                   {total:5d}")
        
        # Analyze side-to-move distribution for GOOD positions
        if good_pos:
            good_colors = Counter(p['side_to_move'] for p in good_pos)
            print(f"\n🎯 GOOD Positions by Side to Move:")
            for color, count in sorted(good_colors.items()):
                print(f"  {color:10s}: {count:4d} ({count/len(good_pos)*100:5.1f}%)")
        
        # Analyze side-to-move distribution for BAD positions
        if bad_pos:
            bad_colors = Counter(p['side_to_move'] for p in bad_pos)
            print(f"\n⚠️  BAD Positions by Side to Move:")
            for color, count in sorted(bad_colors.items()):
                print(f"  {color:10s}: {count:4d} ({count/len(bad_pos)*100:5.1f}%)")
        
        # Check balance
        balance_ratio = len(good_pos) / len(bad_pos) if bad_pos else float('inf')
        print(f"\n⚖️  Balance Analysis:")
        print(f"  GOOD/BAD Ratio: {balance_ratio:.2f}")
        
        if 0.8 <= balance_ratio <= 1.2:
            print(f"  ✅ WELL BALANCED (ratio between 0.8 and 1.2)")
        elif 0.5 <= balance_ratio <= 2.0:
            print(f"  ⚠️  SLIGHTLY IMBALANCED (ratio between 0.5 and 2.0)")
            print(f"  → Consider balancing before adding to training set")
        else:
            print(f"  ❌ SEVERELY IMBALANCED (ratio outside 0.5-2.0 range)")
            print(f"  → MUST balance before training (undersample majority class)")
    
    def save_labeled_data(self, good_pos: List[Dict], bad_pos: List[Dict]):
        """Save GOOD and BAD positions in Stage 1 training format."""
        
        # Save GOOD positions
        good_file = self.output_dir / "selfplay_good_positions.jsonl"
        with open(good_file, 'w') as f:
            for pos in good_pos:
                f.write(json.dumps(pos) + '\n')
        print(f"\n💾 Saved {len(good_pos)} GOOD positions to {good_file}")
        
        # Save BAD positions
        bad_file = self.output_dir / "selfplay_bad_positions.jsonl"
        with open(bad_file, 'w') as f:
            for pos in bad_pos:
                f.write(json.dumps(pos) + '\n')
        print(f"💾 Saved {len(bad_pos)} BAD positions to {bad_file}")
        
        # Save metadata for tracking
        metadata = {
            'extraction_date': '2026-06-01',
            'source_dir': str(self.selfplay_dir),
            'total_positions': len(good_pos) + len(bad_pos),
            'good_count': len(good_pos),
            'bad_count': len(bad_pos),
            'good_threshold_cp': self.good_threshold,
            'bad_threshold_cp': self.bad_threshold,
            'balance_ratio': len(good_pos) / len(bad_pos) if bad_pos else 0
        }
        
        metadata_file = self.output_dir / "extraction_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"📋 Saved metadata to {metadata_file}")
    
    def balance_dataset(self, good_pos: List[Dict], bad_pos: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Balance GOOD and BAD positions by undersampling the majority class.
        
        Returns balanced (good_positions, bad_positions).
        """
        if len(good_pos) > len(bad_pos):
            # Undersample GOOD positions
            import random
            random.seed(42)
            good_balanced = random.sample(good_pos, len(bad_pos))
            bad_balanced = bad_pos
            print(f"\n⚖️  Balanced by undersampling GOOD: {len(good_pos)} → {len(good_balanced)}")
        elif len(bad_pos) > len(good_pos):
            # Undersample BAD positions
            import random
            random.seed(42)
            bad_balanced = random.sample(bad_pos, len(good_pos))
            good_balanced = good_pos
            print(f"⚖️  Balanced by undersampling BAD: {len(bad_pos)} → {len(bad_balanced)}")
        else:
            # Already balanced
            good_balanced = good_pos
            bad_balanced = bad_pos
            print(f"\n✅ Already balanced: {len(good_pos)} GOOD = {len(bad_pos)} BAD")
        
        return good_balanced, bad_balanced
    
    def run_extraction(self, auto_balance: bool = True):
        """Execute full extraction pipeline."""
        print("="*60)
        print("EXTRACTING STAGE 1 LABELS FROM SELF-PLAY DATA")
        print("="*60)
        
        # Load positions
        positions = self.load_selfplay_positions()
        
        # Extract labels
        good_pos, bad_pos, neutral_pos = self.extract_labels(positions)
        
        # Analyze distribution
        self.analyze_distribution(good_pos, bad_pos, neutral_pos)
        
        # Balance if requested
        if auto_balance:
            good_pos, bad_pos = self.balance_dataset(good_pos, bad_pos)
        
        # Save labeled data
        self.save_labeled_data(good_pos, bad_pos)
        
        print("\n" + "="*60)
        print("✅ EXTRACTION COMPLETE")
        print("="*60)
        print(f"\n📝 Next Steps:")
        print(f"  1. Review extraction_metadata.json for balance metrics")
        print(f"  2. Merge with existing Stage 1 dataset (1.648M positions)")
        print(f"  3. Run incremental training (5-10 epochs)")
        print(f"  4. Validate model performance on held-out test set")

if __name__ == "__main__":
    selfplay_dir = "data/stage2/selfplay_batch_284"
    output_dir = "data/stage1/selfplay_extracted"
    
    extractor = Stage1LabelExtractor(selfplay_dir, output_dir)
    extractor.run_extraction(auto_balance=True)
