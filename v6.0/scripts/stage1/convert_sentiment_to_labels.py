"""
Sentiment to Labels Conversion Pipeline
Converts heuristic sentiment scores to GOOD/BAD position labels for Stage 1 v2.0 training.

Strategy:
- Apply zero-crossing threshold (sentiment > 0 = GOOD, < 0 = BAD)
- Exclude neutral positions (sentiment == 0)
- Undersample GOOD positions to match BAD count (balance dataset)
- Export in JSONL format compatible with train_balanced.py
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import Counter

# Configuration
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "stage1" / "heuristic_labeled"

HUMAN_SENTIMENT_PATH = DATA_DIR / "human_move_sentiment.csv"
BOT_SENTIMENT_PATH = DATA_DIR / "bot_move_sentiment.csv"

THRESHOLD = 0.0  # Zero-crossing threshold

def load_sentiment_data():
    """Load and combine sentiment datasets."""
    print("📊 Loading sentiment data...")
    
    df_human = pd.read_csv(HUMAN_SENTIMENT_PATH)
    df_bot = pd.read_csv(BOT_SENTIMENT_PATH)
    
    print(f"  ✅ Human moves: {len(df_human):,}")
    print(f"  ✅ Bot moves: {len(df_bot):,}")
    
    df_combined = pd.concat([df_human, df_bot], ignore_index=True)
    print(f"  ✅ Total moves: {len(df_combined):,}")
    
    return df_combined

def apply_threshold_and_extract(df):
    """Apply threshold and extract GOOD/BAD positions."""
    print(f"\n🎯 Applying threshold: sentiment > {THRESHOLD} = GOOD, < {THRESHOLD} = BAD")
    
    sentiment_col = 'weighted_sentiment_delta'
    
    # Apply threshold
    df_good = df[df[sentiment_col] > THRESHOLD].copy()
    df_bad = df[df[sentiment_col] < THRESHOLD].copy()
    df_neutral = df[df[sentiment_col] == THRESHOLD].copy()
    
    print(f"\n📊 Initial Classification:")
    print(f"  GOOD positions:    {len(df_good):,} ({len(df_good)/len(df)*100:.2f}%)")
    print(f"  BAD positions:     {len(df_bad):,} ({len(df_bad)/len(df)*100:.2f}%)")
    print(f"  Neutral (excluded): {len(df_neutral):,} ({len(df_neutral)/len(df)*100:.2f}%)")
    
    return df_good, df_bad

def analyze_color_balance(df, label):
    """Analyze color distribution in dataset."""
    if 'move_player' not in df.columns:
        return
    
    white_count = (df['move_player'] == 'White').sum()
    black_count = (df['move_player'] == 'Black').sum()
    total = len(df)
    
    print(f"\n🎨 Color Balance ({label}):")
    print(f"  White: {white_count:,} ({white_count/total*100:.1f}%)")
    print(f"  Black: {black_count:,} ({black_count/total*100:.1f}%)")
    
    if abs(white_count - black_count) / total > 0.1:
        print(f"  ⚠️  Warning: Color imbalance detected")
    else:
        print(f"  ✅ Color balance acceptable")

def balance_dataset(df_good, df_bad):
    """Balance dataset by undersampling majority class."""
    print(f"\n⚖️  Balancing Dataset:")
    print(f"  Before: GOOD={len(df_good):,}, BAD={len(df_bad):,}")
    
    # Determine which class to undersample
    min_count = min(len(df_good), len(df_bad))
    
    # Randomly sample to match minority class size
    if len(df_good) > min_count:
        df_good_balanced = df_good.sample(n=min_count, random_state=42)
        print(f"  Undersampled GOOD: {len(df_good):,} → {min_count:,}")
    else:
        df_good_balanced = df_good
    
    if len(df_bad) > min_count:
        df_bad_balanced = df_bad.sample(n=min_count, random_state=42)
        print(f"  Undersampled BAD: {len(df_bad):,} → {min_count:,}")
    else:
        df_bad_balanced = df_bad
    
    print(f"  After:  GOOD={len(df_good_balanced):,}, BAD={len(df_bad_balanced):,}")
    print(f"  Balance Ratio: {len(df_good_balanced) / len(df_bad_balanced):.4f}")
    
    return df_good_balanced, df_bad_balanced

def extract_positions(df, label):
    """Extract FEN positions and create training format."""
    print(f"\n📦 Extracting {label} positions...")
    
    # Use fen_after (the position resulting from the move)
    positions = []
    
    for idx, row in df.iterrows():
        fen = row['fen_after']
        sentiment = row['weighted_sentiment_delta']
        
        positions.append({
            'fen': fen,
            'label': label.lower(),
            'sentiment_score': float(sentiment),  # Keep original score for reference
            'move_player': row.get('move_player', 'unknown'),
            'game_id': row.get('game_id', -1),
            'ply': row.get('ply', -1),
            'move_san': row.get('move_san', 'unknown')
        })
    
    print(f"  ✅ Extracted {len(positions):,} {label} positions")
    
    return positions

def save_positions(positions, output_path):
    """Save positions to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for pos in positions:
            f.write(json.dumps(pos) + '\n')
    
    print(f"  💾 Saved to: {output_path}")

def create_metadata(df_good, df_bad, good_positions, bad_positions):
    """Create metadata file documenting the conversion."""
    metadata = {
        'conversion_date': pd.Timestamp.now().isoformat(),
        'threshold': THRESHOLD,
        'threshold_type': 'zero-crossing',
        'source_files': [
            str(HUMAN_SENTIMENT_PATH),
            str(BOT_SENTIMENT_PATH)
        ],
        'original_counts': {
            'good': len(df_good),
            'bad': len(df_bad),
            'total': len(df_good) + len(df_bad)
        },
        'balanced_counts': {
            'good': len(good_positions),
            'bad': len(bad_positions),
            'total': len(good_positions) + len(bad_positions)
        },
        'color_distribution_good': {
            'white': int((pd.DataFrame(good_positions)['move_player'] == 'White').sum()),
            'black': int((pd.DataFrame(good_positions)['move_player'] == 'Black').sum())
        },
        'color_distribution_bad': {
            'white': int((pd.DataFrame(bad_positions)['move_player'] == 'White').sum()),
            'black': int((pd.DataFrame(bad_positions)['move_player'] == 'Black').sum())
        },
        'sentiment_stats': {
            'good_mean': float(df_good['weighted_sentiment_delta'].mean()),
            'good_std': float(df_good['weighted_sentiment_delta'].std()),
            'bad_mean': float(df_bad['weighted_sentiment_delta'].mean()),
            'bad_std': float(df_bad['weighted_sentiment_delta'].std())
        }
    }
    
    return metadata

def main():
    """Main conversion pipeline."""
    print("="*60)
    print("SENTIMENT TO LABELS CONVERSION PIPELINE")
    print("="*60)
    
    # Load data
    df_combined = load_sentiment_data()
    
    # Apply threshold
    df_good, df_bad = apply_threshold_and_extract(df_combined)
    
    # Analyze color balance before balancing
    analyze_color_balance(df_good, "GOOD (before balance)")
    analyze_color_balance(df_bad, "BAD (before balance)")
    
    # Balance dataset
    df_good_balanced, df_bad_balanced = balance_dataset(df_good, df_bad)
    
    # Analyze color balance after balancing
    analyze_color_balance(df_good_balanced, "GOOD (after balance)")
    analyze_color_balance(df_bad_balanced, "BAD (after balance)")
    
    # Extract positions
    good_positions = extract_positions(df_good_balanced, "GOOD")
    bad_positions = extract_positions(df_bad_balanced, "BAD")
    
    # Save to JSONL
    print(f"\n💾 Saving datasets...")
    save_positions(good_positions, OUTPUT_DIR / "heuristic_good_positions.jsonl")
    save_positions(bad_positions, OUTPUT_DIR / "heuristic_bad_positions.jsonl")
    
    # Create and save metadata
    metadata = create_metadata(df_good, df_bad, good_positions, bad_positions)
    metadata_path = OUTPUT_DIR / "conversion_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  📋 Saved metadata to: {metadata_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("CONVERSION COMPLETE")
    print('='*60)
    print(f"\n📊 Final Dataset Summary:")
    print(f"  GOOD positions: {len(good_positions):,}")
    print(f"  BAD positions:  {len(bad_positions):,}")
    print(f"  Total:          {len(good_positions) + len(bad_positions):,}")
    print(f"  Balance ratio:  {len(good_positions) / len(bad_positions):.4f}")
    
    print(f"\n📈 Comparison to Original Stage 1 Dataset:")
    original_size = 1_648_000
    new_size = len(good_positions) + len(bad_positions)
    print(f"  Original: {original_size:,} positions")
    print(f"  New:      {new_size:,} positions")
    print(f"  Reduction: {(1 - new_size/original_size)*100:.1f}%")
    
    print(f"\n📝 Next Steps:")
    print(f"  1. Review conversion_metadata.json for statistics")
    print(f"  2. Update train_balanced.py to use heuristic_labeled/ directory")
    print(f"  3. Train Stage 1 v2.0 from scratch (NOT incremental)")
    print(f"  4. Compare v2.0 performance to v1.1 (F1=0.8957)")
    print(f"  5. Test behavioral differences in self-play")
    
    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()
