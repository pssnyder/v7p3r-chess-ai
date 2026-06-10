"""
Sentiment Distribution Analysis Script
Analyzes heuristic-based move sentiment scores from historical games
to determine optimal threshold for GOOD/BAD position classification.

Purpose: Validate threshold assumptions before converting to training labels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
HUMAN_SENTIMENT_PATH = DATA_DIR / "human_move_sentiment.csv"
BOT_SENTIMENT_PATH = DATA_DIR / "bot_move_sentiment.csv"

def load_sentiment_data():
    """Load sentiment CSV files."""
    print("📊 Loading sentiment data...")
    
    if not HUMAN_SENTIMENT_PATH.exists():
        raise FileNotFoundError(f"Human sentiment file not found: {HUMAN_SENTIMENT_PATH}")
    if not BOT_SENTIMENT_PATH.exists():
        raise FileNotFoundError(f"Bot sentiment file not found: {BOT_SENTIMENT_PATH}")
    
    df_human = pd.read_csv(HUMAN_SENTIMENT_PATH)
    df_bot = pd.read_csv(BOT_SENTIMENT_PATH)
    
    print(f"  ✅ Human moves: {len(df_human):,}")
    print(f"  ✅ Bot moves: {len(df_bot):,}")
    
    # Combine datasets
    df_combined = pd.concat([df_human, df_bot], ignore_index=True)
    print(f"  ✅ Total moves: {len(df_combined):,}")
    
    return df_human, df_bot, df_combined

def analyze_distribution(df, label="Combined"):
    """Analyze sentiment score distribution."""
    print(f"\n{'='*60}")
    print(f"DISTRIBUTION ANALYSIS: {label}")
    print('='*60)
    
    sentiment_col = 'weighted_sentiment_delta'
    
    # Basic statistics
    print("\n📈 Weighted Sentiment Delta Statistics:")
    print(f"  Mean:     {df[sentiment_col].mean():.4f}")
    print(f"  Median:   {df[sentiment_col].median():.4f}")
    print(f"  Std Dev:  {df[sentiment_col].std():.4f}")
    print(f"  Min:      {df[sentiment_col].min():.4f}")
    print(f"  Max:      {df[sentiment_col].max():.4f}")
    
    # Percentiles
    print("\n📊 Percentile Distribution:")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = df[sentiment_col].quantile(p/100)
        print(f"  {p:2d}th percentile: {val:8.4f}")
    
    # Zero-crossing analysis
    positive = (df[sentiment_col] > 0).sum()
    negative = (df[sentiment_col] < 0).sum()
    zero = (df[sentiment_col] == 0).sum()
    
    print("\n🎯 Zero-Crossing Threshold Analysis:")
    print(f"  Positive (GOOD): {positive:8,} ({positive/len(df)*100:5.2f}%)")
    print(f"  Negative (BAD):  {negative:8,} ({negative/len(df)*100:5.2f}%)")
    print(f"  Zero (Neutral):  {zero:8,} ({zero/len(df)*100:5.2f}%)")
    
    # Alternative thresholds
    print("\n🔍 Alternative Threshold Options:")
    thresholds = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    for thresh in thresholds:
        above = (df[sentiment_col] > thresh).sum()
        below = (df[sentiment_col] < -thresh).sum()
        excluded = len(df) - above - below
        print(f"  Threshold ±{thresh:4.1f}: GOOD={above:7,} BAD={below:7,} Excluded={excluded:7,}")
    
    # Color balance analysis
    print("\n🎨 Color Balance (for zero-crossing threshold):")
    df_positive = df[df[sentiment_col] > 0]
    df_negative = df[df[sentiment_col] < 0]
    
    if 'move_player' in df.columns:
        positive_white = (df_positive['move_player'] == 'White').sum()
        positive_black = (df_positive['move_player'] == 'Black').sum()
        negative_white = (df_negative['move_player'] == 'White').sum()
        negative_black = (df_negative['move_player'] == 'Black').sum()
        
        print(f"  GOOD positions: White={positive_white:,} ({positive_white/len(df_positive)*100:.1f}%), Black={positive_black:,} ({positive_black/len(df_positive)*100:.1f}%)")
        print(f"  BAD positions:  White={negative_white:,} ({negative_white/len(df_negative)*100:.1f}%), Black={negative_black:,} ({negative_black/len(df_negative)*100:.1f}%)")
    
    return {
        'mean': df[sentiment_col].mean(),
        'median': df[sentiment_col].median(),
        'std': df[sentiment_col].std(),
        'positive_count': positive,
        'negative_count': negative,
        'zero_count': zero
    }

def visualize_distribution(df_human, df_bot, df_combined):
    """Create visualization of sentiment distributions."""
    print("\n📊 Creating visualizations...")
    
    sentiment_col = 'weighted_sentiment_delta'
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Combined distribution histogram
    ax1 = axes[0, 0]
    ax1.hist(df_combined[sentiment_col], bins=100, alpha=0.7, edgecolor='black')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero threshold')
    ax1.axvline(df_combined[sentiment_col].mean(), color='green', linestyle='--', linewidth=2, label=f'Mean ({df_combined[sentiment_col].mean():.2f})')
    ax1.axvline(df_combined[sentiment_col].median(), color='blue', linestyle='--', linewidth=2, label=f'Median ({df_combined[sentiment_col].median():.2f})')
    ax1.set_xlabel('Weighted Sentiment Delta')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Combined Sentiment Distribution (All Moves)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Human vs Bot comparison
    ax2 = axes[0, 1]
    ax2.hist(df_human[sentiment_col], bins=50, alpha=0.5, label='Human', edgecolor='black')
    ax2.hist(df_bot[sentiment_col], bins=50, alpha=0.5, label='Bot', edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero threshold')
    ax2.set_xlabel('Weighted Sentiment Delta')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Human vs Bot Sentiment Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot comparison
    ax3 = axes[1, 0]
    data_to_plot = [df_human[sentiment_col], df_bot[sentiment_col]]
    ax3.boxplot(data_to_plot, labels=['Human', 'Bot'], vert=True)
    ax3.axhline(0, color='red', linestyle='--', linewidth=2, label='Zero threshold')
    ax3.set_ylabel('Weighted Sentiment Delta')
    ax3.set_title('Sentiment Distribution Comparison (Box Plot)')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend()
    
    # 4. Threshold impact visualization
    ax4 = axes[1, 1]
    thresholds = np.linspace(-3, 3, 61)
    good_counts = []
    bad_counts = []
    
    for thresh in thresholds:
        good_counts.append((df_combined[sentiment_col] > thresh).sum())
        bad_counts.append((df_combined[sentiment_col] < -thresh).sum())
    
    ax4.plot(thresholds, good_counts, label='GOOD positions', linewidth=2)
    ax4.plot(thresholds, bad_counts, label='BAD positions', linewidth=2)
    ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero threshold')
    ax4.set_xlabel('Threshold')
    ax4.set_ylabel('Position Count')
    ax4.set_title('Threshold Impact on Dataset Size')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(__file__).parent.parent.parent / "data" / "raw" / "sentiment_distribution_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved visualization to: {output_path}")
    
    plt.show()

def recommend_threshold(stats_combined):
    """Provide threshold recommendation based on analysis."""
    print("\n" + "="*60)
    print("THRESHOLD RECOMMENDATION")
    print("="*60)
    
    positive_pct = stats_combined['positive_count'] / (stats_combined['positive_count'] + stats_combined['negative_count'] + stats_combined['zero_count']) * 100
    negative_pct = stats_combined['negative_count'] / (stats_combined['positive_count'] + stats_combined['negative_count'] + stats_combined['zero_count']) * 100
    
    print(f"\n✅ RECOMMENDED: Zero-crossing threshold (sentiment > 0 = GOOD, < 0 = BAD)")
    print(f"\nRationale:")
    print(f"  • Natural interpretation: positive delta = position improved")
    print(f"  • Distribution: {positive_pct:.1f}% GOOD, {negative_pct:.1f}% BAD")
    print(f"  • Balance: {'GOOD' if positive_pct > negative_pct else 'BAD'} positions are {abs(positive_pct - negative_pct):.1f}% more frequent")
    print(f"  • Neutral exclusion: {stats_combined['zero_count']:,} positions (keeps signal clean)")
    
    if abs(positive_pct - negative_pct) > 20:
        print(f"\n⚠️  WARNING: Significant class imbalance ({positive_pct:.1f}% vs {negative_pct:.1f}%)")
        print(f"  → Will need to undersample majority class during conversion")
    else:
        print(f"\n✅ Class balance is acceptable (within 20% difference)")
    
    print(f"\nNext Steps:")
    print(f"  1. Review visualization for outliers or unexpected patterns")
    print(f"  2. Confirm zero-crossing threshold is appropriate")
    print(f"  3. Run conversion script to create training dataset")
    print(f"  4. Train Stage 1 v2.0 with new labels")

def main():
    """Main execution."""
    print("="*60)
    print("SENTIMENT DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Load data
    df_human, df_bot, df_combined = load_sentiment_data()
    
    # Analyze distributions
    print("\n" + "="*60)
    print("INDIVIDUAL DATASET ANALYSIS")
    print("="*60)
    
    stats_human = analyze_distribution(df_human, "Human Games")
    stats_bot = analyze_distribution(df_bot, "Bot Games")
    stats_combined = analyze_distribution(df_combined, "Combined Dataset")
    
    # Visualize
    visualize_distribution(df_human, df_bot, df_combined)
    
    # Recommendation
    recommend_threshold(stats_combined)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
