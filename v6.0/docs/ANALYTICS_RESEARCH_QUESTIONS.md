# V7P3R AI - Analytics Research Questions
## Understanding Good vs Bad Moves in Chess

**Created**: 2026-05-31  
**Status**: 📊 **RESEARCH PLANNING**  
**Purpose**: Answer fundamental questions about chess move quality and feature importance  

---

## Research Questions

### Primary Question 1: Good vs Bad Move Ratio

**Question**: "For each chess position, what is the average ratio of 'good' vs 'bad' moves?"

**Sub-Questions**:
1. Does the ratio vary by game phase (opening, middlegame, endgame)?
2. Does the ratio correlate with position complexity?
3. Do stronger players have more "good" moves available, or do they just avoid "bad" moves?
4. Is there a minimum/maximum ratio across all legal positions?

**Methodology**:

```python
def analyze_good_bad_ratio(positions_dataset: list) -> dict:
    """
    Analyze ratio of good to bad moves across positions.
    
    Process:
    1. For each position, evaluate all legal moves with Stage 1
    2. Count good moves (prob_good >= 0.5) vs bad moves (prob_good < 0.5)
    3. Calculate ratio and distribution statistics
    
    Args:
        positions_dataset: List of FEN positions with metadata
        
    Returns:
        Statistics dict with ratios, distributions, correlations
    """
    ratios = []
    
    for position in positions_dataset:
        board = chess.Board(position['fen'])
        legal_moves = list(board.legal_moves)
        
        # Evaluate all moves with Stage 1
        evaluations = []
        for move in legal_moves:
            board.push(move)
            prob_good = stage1_model.predict(board.fen())
            evaluations.append({
                'move': move.uci(),
                'prob_good': prob_good,
                'is_good': prob_good >= 0.5,
            })
            board.pop()
        
        # Calculate ratio
        num_good = sum(1 for e in evaluations if e['is_good'])
        num_bad = len(evaluations) - num_good
        
        if num_bad > 0:
            ratio = num_good / num_bad
        else:
            ratio = float('inf')  # All moves good
        
        ratios.append({
            'fen': position['fen'],
            'legal_moves': len(legal_moves),
            'good_moves': num_good,
            'bad_moves': num_bad,
            'ratio': ratio,
            'game_phase': position['game_phase'],
            'complexity': position['complexity'],
        })
    
    return calculate_statistics(ratios)


def calculate_statistics(ratios: list) -> dict:
    """Calculate summary statistics for good/bad ratios."""
    import numpy as np
    
    # Overall statistics
    ratio_values = [r['ratio'] for r in ratios if r['ratio'] != float('inf')]
    
    stats = {
        'overall': {
            'mean_ratio': np.mean(ratio_values),
            'median_ratio': np.median(ratio_values),
            'std_ratio': np.std(ratio_values),
            'min_ratio': np.min(ratio_values),
            'max_ratio': np.max(ratio_values),
        },
        
        # By game phase
        'by_phase': {},
        
        # By complexity
        'by_complexity': {},
        
        # Distribution bins
        'distribution': {
            'all_good': sum(1 for r in ratios if r['bad_moves'] == 0),
            'mostly_good': sum(1 for r in ratios if 0 < r['ratio'] < 1),
            'balanced': sum(1 for r in ratios if 1 <= r['ratio'] < 2),
            'mostly_bad': sum(1 for r in ratios if r['ratio'] >= 2),
        }
    }
    
    # By game phase
    for phase in ['opening', 'middlegame', 'endgame']:
        phase_ratios = [r['ratio'] for r in ratios if r['game_phase'] == phase and r['ratio'] != float('inf')]
        if phase_ratios:
            stats['by_phase'][phase] = {
                'mean_ratio': np.mean(phase_ratios),
                'median_ratio': np.median(phase_ratios),
                'sample_count': len(phase_ratios),
            }
    
    # By complexity (binned)
    complexity_bins = [(0, 3), (3, 6), (6, 10)]
    for low, high in complexity_bins:
        bin_ratios = [
            r['ratio'] for r in ratios 
            if low <= r['complexity'] < high and r['ratio'] != float('inf')
        ]
        if bin_ratios:
            stats['by_complexity'][f'{low}-{high}'] = {
                'mean_ratio': np.mean(bin_ratios),
                'median_ratio': np.median(bin_ratios),
                'sample_count': len(bin_ratios),
            }
    
    return stats
```

**Expected Findings** (hypotheses to test):
- **Opening**: High good/bad ratio (~3:1) - many reasonable moves
- **Middlegame**: Lower ratio (~1:1) - more critical decisions
- **Endgame**: High ratio (~2:1) - fewer pieces, clearer objectives
- **Complexity correlation**: Higher complexity → lower ratio (more ways to go wrong)

**Visualization**:

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_good_bad_ratios(stats: dict):
    """Create visualizations for good/bad ratio analysis."""
    
    # 1. Distribution histogram
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Overall distribution
    axes[0, 0].hist(ratio_values, bins=50, edgecolor='black')
    axes[0, 0].set_xlabel('Good/Bad Ratio')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Overall Good/Bad Ratio Distribution')
    axes[0, 0].axvline(stats['overall']['mean_ratio'], color='red', linestyle='--', label='Mean')
    axes[0, 0].legend()
    
    # By game phase
    phase_means = [stats['by_phase'][p]['mean_ratio'] for p in ['opening', 'middlegame', 'endgame']]
    axes[0, 1].bar(['Opening', 'Middlegame', 'Endgame'], phase_means, color=['green', 'blue', 'orange'])
    axes[0, 1].set_ylabel('Mean Good/Bad Ratio')
    axes[0, 1].set_title('Ratio by Game Phase')
    
    # By complexity
    complexity_labels = list(stats['by_complexity'].keys())
    complexity_means = [stats['by_complexity'][label]['mean_ratio'] for label in complexity_labels]
    axes[1, 0].bar(complexity_labels, complexity_means, color='purple')
    axes[1, 0].set_xlabel('Complexity Range')
    axes[1, 0].set_ylabel('Mean Good/Bad Ratio')
    axes[1, 0].set_title('Ratio by Position Complexity')
    
    # Distribution pie chart
    distribution_labels = list(stats['distribution'].keys())
    distribution_values = list(stats['distribution'].values())
    axes[1, 1].pie(distribution_values, labels=distribution_labels, autopct='%1.1f%%')
    axes[1, 1].set_title('Position Distribution by Ratio Type')
    
    plt.tight_layout()
    plt.savefig('good_bad_ratio_analysis.png', dpi=300)
    plt.show()
```

---

### Primary Question 2: Feature Importance

**Question**: "What is the distribution across most impactful features? What is most deterministic related to good vs bad?"

**Sub-Questions**:
1. Which of the 19 Stage 1 features correlate most strongly with good moves?
2. Are there features that are ALWAYS bad when activated (deterministic)?
3. Do feature importances change by game phase?
4. Are there feature interactions (e.g., feature A + feature B together)?

**Methodology**:

```python
def analyze_feature_importance(positions_dataset: list, model: PositionEvaluator) -> dict:
    """
    Analyze which features are most predictive of good vs bad moves.
    
    Methods:
    1. Permutation importance (shuffle feature, measure performance drop)
    2. SHAP values (Shapley Additive Explanations)
    3. Correlation analysis (feature value vs prob_good)
    4. Ablation study (remove feature, retrain, measure loss)
    
    Args:
        positions_dataset: List of labeled positions
        model: Trained Stage 1 model
        
    Returns:
        Feature importance rankings and statistics
    """
    from sklearn.inspection import permutation_importance
    import shap
    
    # Prepare data
    X = []
    y = []
    
    for position in positions_dataset:
        features = extract_fast_features(position['fen'])
        X.append(features)
        y.append(position['label'])  # 1=good, 0=bad
    
    X = np.array(X)
    y = np.array(y)
    
    # Method 1: Permutation Importance
    perm_importance = permutation_importance(
        model, X, y, 
        n_repeats=10, 
        random_state=42
    )
    
    # Method 2: SHAP Values
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    
    # Method 3: Correlation Analysis
    correlations = []
    for i in range(X.shape[1]):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        correlations.append(corr)
    
    # Combine results
    feature_names = [
        'white_pawns', 'white_knights', 'white_bishops', 'white_rooks', 'white_queens', 'white_kings',
        'black_pawns', 'black_knights', 'black_bishops', 'black_rooks', 'black_queens', 'black_kings',
        'material_balance',
        'side_to_move', 'white_kingside_castle', 'white_queenside_castle', 'in_check',
        'legal_moves_current', 'legal_moves_opponent'
    ]
    
    results = []
    for i, name in enumerate(feature_names):
        results.append({
            'feature': name,
            'permutation_importance': perm_importance.importances_mean[i],
            'shap_importance': np.abs(shap_values.values[:, i]).mean(),
            'correlation': correlations[i],
        })
    
    # Rank by permutation importance (most reliable method)
    results.sort(key=lambda x: x['permutation_importance'], reverse=True)
    
    return results


def analyze_deterministic_features(positions_dataset: list) -> dict:
    """
    Find features that are deterministically good or bad.
    
    Example:
    - If 'in_check' = 1 AND no escape moves → ALWAYS bad
    - If 'material_balance' < -500 (5 pawns down) → ALWAYS bad (unless checkmate threat)
    """
    deterministic_rules = []
    
    # Rule 1: In check with no escape
    rule1_positions = [
        p for p in positions_dataset
        if p['features']['in_check'] == 1 and p['features']['legal_moves_current'] == 0
    ]
    rule1_accuracy = sum(1 for p in rule1_positions if p['label'] == 0) / len(rule1_positions)
    
    deterministic_rules.append({
        'rule': 'in_check AND legal_moves_current == 0',
        'prediction': 'BAD',
        'accuracy': rule1_accuracy,
        'support': len(rule1_positions),
    })
    
    # Rule 2: Material balance < -500
    rule2_positions = [
        p for p in positions_dataset
        if p['features']['material_balance'] < -500
    ]
    rule2_accuracy = sum(1 for p in rule2_positions if p['label'] == 0) / len(rule2_positions)
    
    deterministic_rules.append({
        'rule': 'material_balance < -500',
        'prediction': 'BAD',
        'accuracy': rule2_accuracy,
        'support': len(rule2_positions),
    })
    
    # Rule 3: King exposed (no nearby pawns)
    # (Implement king safety calculation)
    
    # Rule 4: Hanging queen (queen undefended and attacked)
    # (Implement piece attack/defense calculation)
    
    return deterministic_rules
```

**Visualization**:

```python
def visualize_feature_importance(importance_results: list):
    """Create feature importance visualizations."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Extract data
    features = [r['feature'] for r in importance_results]
    perm_imp = [r['permutation_importance'] for r in importance_results]
    shap_imp = [r['shap_importance'] for r in importance_results]
    corr = [r['correlation'] for r in importance_results]
    
    # Plot 1: Permutation Importance
    axes[0].barh(features, perm_imp, color='skyblue')
    axes[0].set_xlabel('Permutation Importance')
    axes[0].set_title('Feature Importance (Permutation)')
    axes[0].invert_yaxis()
    
    # Plot 2: SHAP Importance
    axes[1].barh(features, shap_imp, color='lightcoral')
    axes[1].set_xlabel('SHAP Importance (|mean|)')
    axes[1].set_title('Feature Importance (SHAP)')
    axes[1].invert_yaxis()
    
    # Plot 3: Correlation with Label
    colors = ['green' if c > 0 else 'red' for c in corr]
    axes[2].barh(features, corr, color=colors)
    axes[2].set_xlabel('Correlation with Good Move')
    axes[2].set_title('Feature Correlation')
    axes[2].axvline(0, color='black', linestyle='--', linewidth=0.8)
    axes[2].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('feature_importance_analysis.png', dpi=300)
    plt.show()
```

---

### Primary Question 3: Feature Heatmap

**Question**: "Is there a hot spot on the heatmap for worse or better moves?"

**Sub-Questions**:
1. In feature space, where do "very good" moves cluster?
2. Where do "blunders" cluster?
3. Are good/bad moves separable in 2D/3D feature space?
4. Which feature pairs create the clearest decision boundaries?

**Methodology**:

```python
def create_feature_space_heatmap(positions_dataset: list) -> None:
    """
    Visualize positions in feature space with good/bad labels.
    
    Process:
    1. Extract all 19 features for each position
    2. Reduce to 2D using PCA or t-SNE
    3. Color by label (good=green, bad=red)
    4. Identify clusters/hotspots
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    # Prepare data
    X = []
    y = []
    
    for position in positions_dataset:
        features = extract_fast_features(position['fen'])
        X.append(features)
        y.append(position['label'])
    
    X = np.array(X)
    y = np.array(y)
    
    # Dimensionality reduction
    # Method 1: PCA (linear)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Method 2: t-SNE (non-linear)
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X[:5000])  # Sample for speed
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # PCA plot
    scatter1 = axes[0].scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=y, cmap='RdYlGn', alpha=0.5, s=10
    )
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].set_title('Feature Space (PCA)')
    plt.colorbar(scatter1, ax=axes[0], label='Good (1) vs Bad (0)')
    
    # t-SNE plot
    scatter2 = axes[1].scatter(
        X_tsne[:, 0], X_tsne[:, 1],
        c=y[:5000], cmap='RdYlGn', alpha=0.5, s=10
    )
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    axes[1].set_title('Feature Space (t-SNE)')
    plt.colorbar(scatter2, ax=axes[1], label='Good (1) vs Bad (0)')
    
    plt.tight_layout()
    plt.savefig('feature_space_heatmap.png', dpi=300)
    plt.show()


def analyze_feature_pair_heatmaps(positions_dataset: list, top_features: list) -> None:
    """
    Create 2D heatmaps for top feature pairs.
    
    Args:
        positions_dataset: Dataset with features and labels
        top_features: List of most important features (from importance analysis)
    """
    # Select top 5 features
    top_5 = top_features[:5]
    
    # Create pairwise heatmaps
    fig, axes = plt.subplots(len(top_5), len(top_5), figsize=(20, 20))
    
    for i, feat1 in enumerate(top_5):
        for j, feat2 in enumerate(top_5):
            if i == j:
                # Diagonal: histogram
                axes[i, j].hist(
                    [p['features'][feat1] for p in positions_dataset],
                    bins=30, edgecolor='black'
                )
                axes[i, j].set_title(feat1)
            else:
                # Off-diagonal: scatter with color by label
                x = [p['features'][feat1] for p in positions_dataset]
                y = [p['features'][feat2] for p in positions_dataset]
                c = [p['label'] for p in positions_dataset]
                
                axes[i, j].scatter(x, y, c=c, cmap='RdYlGn', alpha=0.3, s=1)
                axes[i, j].set_xlabel(feat1)
                axes[i, j].set_ylabel(feat2)
    
    plt.tight_layout()
    plt.savefig('feature_pair_heatmaps.png', dpi=300)
    plt.show()


def identify_decision_boundary_hotspots(model: PositionEvaluator, feature_ranges: dict) -> None:
    """
    Create decision boundary heatmap for top 2 features.
    
    Shows regions where model predicts good vs bad.
    """
    # Select top 2 features
    feat1_name = 'material_balance'
    feat2_name = 'legal_moves_current'
    
    # Create grid
    feat1_range = np.linspace(feature_ranges[feat1_name][0], feature_ranges[feat1_name][1], 100)
    feat2_range = np.linspace(feature_ranges[feat2_name][0], feature_ranges[feat2_name][1], 100)
    
    X1, X2 = np.meshgrid(feat1_range, feat2_range)
    
    # Predict for each grid point
    predictions = []
    for i in range(100):
        for j in range(100):
            # Create feature vector (set other features to mean)
            features = np.zeros(19)
            features[12] = X1[i, j]  # material_balance index
            features[17] = X2[i, j]  # legal_moves_current index
            # Set other features to mean values
            # (In reality, would use actual mean from dataset)
            
            prob_good = model.predict([features])[0]
            predictions.append(prob_good)
    
    predictions = np.array(predictions).reshape(100, 100)
    
    # Plot decision boundary
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X1, X2, predictions, levels=20, cmap='RdYlGn', alpha=0.8)
    plt.colorbar(contour, label='P(good)')
    plt.xlabel(feat1_name)
    plt.ylabel(feat2_name)
    plt.title('Decision Boundary Heatmap (Material Balance vs Legal Moves)')
    
    # Add decision boundary line (prob = 0.5)
    plt.contour(X1, X2, predictions, levels=[0.5], colors='black', linewidths=2)
    
    plt.savefig('decision_boundary_heatmap.png', dpi=300)
    plt.show()
```

**Expected Findings**:
- **Blunder Hotspot**: Low material balance + low legal moves → red cluster (bad)
- **Good Move Hotspot**: Moderate material balance + high legal moves → green cluster (good)
- **Decision Boundary**: Clear separation in (material_balance, mobility) space

---

## Experimental Design

### Dataset Selection

**Positions to Analyze**:
1. **Training Set Sample**: 100,000 positions from Stage 1 training (balanced 50/50)
2. **Test Set**: 20,000 held-out positions (never seen during training)
3. **Expert Games**: 10,000 positions from GM games (Tal, Kasparov, Carlsen)
4. **Engine Self-Play**: 10,000 positions from V7P3R self-play games

**Total**: ~140,000 positions for comprehensive analysis

### Analysis Pipeline

```python
# Step 1: Load datasets
training_data = load_jsonl('data/stage1/good_positions.jsonl', max_lines=50000)
training_data += load_jsonl('data/stage1/bad_positions_massive.jsonl', max_lines=50000)

test_data = load_jsonl('data/stage1/test_set.jsonl')
gm_games = load_pgn_positions('data/gm_games/*.pgn', filter='critical_positions')
selfplay_data = load_jsonl('data/selfplay/positions.jsonl')

# Step 2: Analyze good/bad ratios
ratio_stats = analyze_good_bad_ratio(training_data + test_data)
print(json.dumps(ratio_stats, indent=2))
visualize_good_bad_ratios(ratio_stats)

# Step 3: Feature importance
importance_results = analyze_feature_importance(training_data, stage1_model)
print("\nTop 10 Most Important Features:")
for i, result in enumerate(importance_results[:10]):
    print(f"{i+1}. {result['feature']}: {result['permutation_importance']:.4f}")

visualize_feature_importance(importance_results)

# Step 4: Deterministic rules
deterministic_rules = analyze_deterministic_features(training_data)
print("\nDeterministic Rules:")
for rule in deterministic_rules:
    print(f"Rule: {rule['rule']} → {rule['prediction']}")
    print(f"  Accuracy: {rule['accuracy']:.2%}, Support: {rule['support']} positions")

# Step 5: Feature space heatmaps
create_feature_space_heatmap(training_data)
analyze_feature_pair_heatmaps(training_data, [r['feature'] for r in importance_results])
identify_decision_boundary_hotspots(stage1_model, feature_ranges)

# Step 6: Save results
results = {
    'ratio_statistics': ratio_stats,
    'feature_importance': importance_results,
    'deterministic_rules': deterministic_rules,
    'dataset_info': {
        'training_positions': len(training_data),
        'test_positions': len(test_data),
        'gm_positions': len(gm_games),
        'selfplay_positions': len(selfplay_data),
    }
}

with open('analytics/research_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nAnalysis complete! Results saved to analytics/research_results.json")
```

---

## Expected Deliverables

### 1. Research Report (Markdown)
**File**: `analytics/RESEARCH_REPORT.md`

Contents:
- Executive summary of findings
- Good/bad ratio statistics (overall, by phase, by complexity)
- Top 10 most important features
- Deterministic rules discovered
- Feature space clustering analysis
- Recommendations for model improvement

### 2. Visualizations (PNG)
**Files**: `analytics/figures/*.png`

- `good_bad_ratio_analysis.png`: Ratio distributions and breakdowns
- `feature_importance_analysis.png`: Three-way importance comparison
- `feature_space_heatmap.png`: PCA and t-SNE visualizations
- `feature_pair_heatmaps.png`: Pairwise feature relationships
- `decision_boundary_heatmap.png`: Model decision regions

### 3. Data Export (JSON)
**File**: `analytics/research_results.json`

Complete numerical results for programmatic access

### 4. Interactive Dashboard (Optional)
**Tool**: Plotly or Streamlit

Interactive exploration of:
- Filter positions by phase/complexity
- Hover over points to see FEN and evaluation
- Adjust feature pair selections for heatmaps
- Compare Stage 1 vs Stockfish evaluations

---

## Implementation Priority

**Phase 1: Basic Analysis** (1-2 days)
1. Good/bad ratio calculation
2. Feature importance (permutation only)
3. Generate report

**Phase 2: Advanced Visualization** (2-3 days)
4. Feature space heatmaps (PCA, t-SNE)
5. Pairwise feature plots
6. Decision boundary visualization

**Phase 3: Deep Dive** (3-5 days)
7. SHAP analysis (if needed)
8. Deterministic rule mining
9. Interactive dashboard (optional)

---

## Follow-Up Research Questions

Once initial analysis is complete, these questions may arise:

1. **Temporal Dynamics**: Do good/bad ratios change as game progresses (move 1 vs move 40)?
2. **Player Strength**: How do ratios differ at different ELO levels?
3. **Opening Theory**: Are some openings inherently more forgiving (higher good/bad ratio)?
4. **Tactical Motifs**: Which tactical patterns (pins, forks) correlate most with bad moves?
5. **Model Calibration**: Is Stage 1's probability well-calibrated (does 80% confidence = 80% accuracy)?

---

## Conclusion

These analytics research questions will provide deep insights into:
1. **How many good moves exist** in typical chess positions
2. **Which features matter most** for distinguishing good from bad moves
3. **Where in feature space** blunders and excellent moves cluster

The findings will inform:
- Stage 1 model improvements (feature engineering)
- Stage 2 training (which features to prioritize)
- Engine personality tuning (acceptable risk levels)

**Ready to implement after Stage 1 model is fully validated!** 📊
