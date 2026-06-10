"""
Dataset Merger: Combines Original + Self-Play Extracted Positions
Merges the original 1.648M training dataset with new self-play positions.
Maintains balance, tracks metadata, and prepares for incremental training.
"""
import json
from pathlib import Path
from collections import Counter
import random

class DatasetMerger:
    def __init__(self, 
                 original_data_dir: str,
                 selfplay_data_dir: str,
                 output_dir: str):
        self.original_data_dir = Path(original_data_dir)
        self.selfplay_data_dir = Path(selfplay_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        random.seed(42)
    
    def load_original_good_positions(self, max_count=824000):
        """Load original GOOD positions from training data."""
        good_path = self.original_data_dir / "good_positions.jsonl"
        
        print(f"📦 Loading original GOOD positions from {good_path}")
        positions = []
        
        if not good_path.exists():
            print(f"⚠️  File not found: {good_path}")
            return positions
        
        with open(good_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_count:
                    break
                try:
                    record = json.loads(line.strip())
                    positions.append(record)
                    
                    if (i + 1) % 100000 == 0:
                        print(f"  Loaded {i + 1} positions...")
                except:
                    continue
        
        print(f"✅ Loaded {len(positions)} original GOOD positions")
        return positions
    
    def load_original_bad_positions(self, max_count=824000):
        """Load original BAD positions from training data."""
        bad_path = self.original_data_dir / "bad_positions_massive.jsonl"
        
        print(f"\n📦 Loading original BAD positions from {bad_path}")
        positions = []
        
        if not bad_path.exists():
            print(f"⚠️  File not found: {bad_path}")
            return positions
        
        with open(bad_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_count:
                    break
                try:
                    record = json.loads(line.strip())
                    positions.append(record)
                    
                    if (i + 1) % 100000 == 0:
                        print(f"  Loaded {i + 1} positions...")
                except:
                    continue
        
        print(f"✅ Loaded {len(positions)} original BAD positions")
        return positions
    
    def load_selfplay_positions(self):
        """Load self-play extracted GOOD and BAD positions."""
        good_path = self.selfplay_data_dir / "selfplay_good_positions.jsonl"
        bad_path = self.selfplay_data_dir / "selfplay_bad_positions.jsonl"
        
        print(f"\n📦 Loading self-play GOOD positions from {good_path}")
        good_positions = []
        with open(good_path, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    good_positions.append(record)
                except:
                    continue
        print(f"✅ Loaded {len(good_positions)} self-play GOOD positions")
        
        print(f"\n📦 Loading self-play BAD positions from {bad_path}")
        bad_positions = []
        with open(bad_path, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    bad_positions.append(record)
                except:
                    continue
        print(f"✅ Loaded {len(bad_positions)} self-play BAD positions")
        
        return good_positions, bad_positions
    
    def merge_and_save(self, 
                       original_good, 
                       original_bad,
                       selfplay_good, 
                       selfplay_bad):
        """Merge datasets and save to output directory."""
        
        print("\n" + "="*60)
        print("MERGING DATASETS")
        print("="*60)
        
        # Combine positions
        all_good = original_good + selfplay_good
        all_bad = original_bad + selfplay_bad
        
        print(f"\n📊 Combined Dataset:")
        print(f"  GOOD positions: {len(all_good):,}")
        print(f"    - Original:  {len(original_good):,}")
        print(f"    - Self-play: {len(selfplay_good):,}")
        print(f"  BAD positions: {len(all_bad):,}")
        print(f"    - Original:  {len(original_bad):,}")
        print(f"    - Self-play: {len(selfplay_bad):,}")
        print(f"  TOTAL: {len(all_good) + len(all_bad):,}")
        
        # Check balance
        balance_ratio = len(all_good) / len(all_bad) if all_bad else 0
        print(f"\n⚖️  Balance Ratio: {balance_ratio:.4f}")
        
        if 0.95 <= balance_ratio <= 1.05:
            print(f"  ✅ EXCELLENT BALANCE")
        elif 0.9 <= balance_ratio <= 1.1:
            print(f"  ✅ GOOD BALANCE")
        else:
            print(f"  ⚠️  IMBALANCED - Consider balancing")
        
        # Shuffle datasets for better training
        print(f"\n🔀 Shuffling datasets...")
        random.shuffle(all_good)
        random.shuffle(all_bad)
        
        # Save merged datasets
        good_output = self.output_dir / "merged_good_positions.jsonl"
        bad_output = self.output_dir / "merged_bad_positions.jsonl"
        
        print(f"\n💾 Saving merged datasets...")
        with open(good_output, 'w') as f:
            for pos in all_good:
                f.write(json.dumps(pos) + '\n')
        print(f"  ✅ Saved {len(all_good):,} GOOD positions to {good_output}")
        
        with open(bad_output, 'w') as f:
            for pos in all_bad:
                f.write(json.dumps(pos) + '\n')
        print(f"  ✅ Saved {len(all_bad):,} BAD positions to {bad_output}")
        
        # Save metadata
        metadata = {
            'merge_date': '2026-06-02',
            'dataset_version': '1.1',
            'total_positions': len(all_good) + len(all_bad),
            'good_count': len(all_good),
            'bad_count': len(all_bad),
            'balance_ratio': balance_ratio,
            'sources': {
                'original_good': len(original_good),
                'original_bad': len(original_bad),
                'selfplay_good': len(selfplay_good),
                'selfplay_bad': len(selfplay_bad)
            }
        }
        
        metadata_path = self.output_dir / "merge_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  📋 Saved metadata to {metadata_path}")
        
        return len(all_good), len(all_bad)
    
    def run_merge(self):
        """Execute full merge pipeline."""
        print("="*60)
        print("DATASET MERGE PIPELINE")
        print("="*60)
        
        # Load original datasets
        original_good = self.load_original_good_positions()
        original_bad = self.load_original_bad_positions()
        
        # Load self-play datasets
        selfplay_good, selfplay_bad = self.load_selfplay_positions()
        
        # Merge and save
        good_count, bad_count = self.merge_and_save(
            original_good, original_bad,
            selfplay_good, selfplay_bad
        )
        
        print("\n" + "="*60)
        print("✅ MERGE COMPLETE")
        print("="*60)
        print(f"\n📝 Next Steps:")
        print(f"  1. Run incremental training on merged dataset")
        print(f"  2. Train for 5-10 epochs (starting from epoch 18 weights)")
        print(f"  3. Validate performance vs original F1=0.8776")
        print(f"  4. Compare with held-out test set")

if __name__ == "__main__":
    original_dir = "data/stage1"
    selfplay_dir = "data/stage1/selfplay_extracted"
    output_dir = "data/stage1/merged"
    
    merger = DatasetMerger(original_dir, selfplay_dir, output_dir)
    merger.run_merge()
