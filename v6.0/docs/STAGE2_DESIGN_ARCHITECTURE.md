# V7P3R AI v6.1 - Stage 2 Design Architecture
## Complexity & Time Management Model

**Created**: 2026-05-31  
**Status**: � **IMPLEMENTATION READY** (Architectural decisions confirmed)  
**Dependencies**: Stage 1 Position Evaluator (✅ COMPLETE)  
**Training Target**: 284 self-play games (median historical benchmark)  
**Architecture**: Option C - Combined complexity/time model + separate move priority ranker  

---

## Executive Summary

Stage 2 is the **tactical time management and move prioritization layer** that transforms Stage 1's "good moves" into a final move selection with timing intelligence and complexity awareness.

**Core Philosophy**: "Progress over perfection" - drive the game into complex, Tal-inspired positions where V7P3R thrives while traditional depth-seeking engines struggle.

**"Viper Strike" Philosophy**: Complexity and time scaling inversion - when complexity hits threshold but checkmate/forced win detected, time weight drops to 0 and engine strikes instantly without thinking. Pure instinct trigger mechanism.

---

## Mission Statement

Stage 2 answers three critical questions:

1. **Complexity**: "How dark and dense has the forest become?" (Tal's concept of tactical complexity)
2. **Time**: "How much time do I need to navigate this complexity safely?"
3. **Priority**: "Which of my 'good moves' are available for play, and WHY?"

**NOT**: "Is this move safe?" (traditional engines)  
**YES**: "Does this move follow V7P3R's personality?" (aggressive, sacrificial, complex)

---

## Stage 2 Model Architecture

### Model Type
**Multi-Output Regression Neural Network**

### Inputs
**Source**: Stage 1 evaluated position + game state metadata

**Feature Categories** (estimated 30-40 total features):

#### 1. Time-Based Features (6-8 features)
- `time_remaining_white`: Seconds left on White's clock
- `time_remaining_black`: Seconds left on Black's clock
- `increment_bonus_white`: Increment per move (e.g., +2s, +4s, +10s)
- `increment_bonus_black`: Increment per move
- `last_move_time_actual`: Actual time spent on previous move
- `processing_tick_count_1ply`: Node/legal move count for 1-ply (proxy for time)
- `processing_tick_count_2ply`: Node/legal move count for 2-ply (proxy for complexity)
- `cache_hit_rate`: Percentage of pre-calculated positions used

**Note on Time Prediction**: Instead of predicting actual time, we use **processing tick counts** (node counts, legal move counts) as a rapid profiling proxy. This avoids complex time modeling while still capturing computational complexity.

**Calculation**: Total time available for current move
```python
if side_to_move == WHITE:
    time_budget = time_remaining_white + (increment_bonus_white * 0.8)
    time_pressure = 1.0 if time_remaining_white < 60 else 0.5 if time_remaining_white < 300 else 0.0
```

#### 2. Complexity Features (8-10 features)
- `legal_moves_count`: Number of legal moves in position (0-280 max per chess theory)
- `legal_moves_after_candidate`: Legal moves after playing candidate move
- `capture_moves_count`: Number of captures available
- `check_moves_count`: Number of checking moves available
- `forced_moves_count`: Number of "only reasonable" moves (e.g., escape check)
- `branching_factor_1ply`: Average legal moves in 1-ply continuation
- `branching_factor_2ply`: Average legal moves in 2-ply continuation
- `tactical_density`: Count of hanging pieces + pins + forks + skewers
- `forest_darkness_score`: **TAL METRIC** - Composite complexity score (higher = deeper forest)

**Forest Darkness Calculation**:
```python
forest_darkness = (
    0.3 * (legal_moves_count / 280.0) +  # Normalized move count
    0.2 * (tactical_density / 10.0) +    # Normalized tactical patterns
    0.2 * (capture_moves_count / legal_moves_count) +  # Capture ratio
    0.15 * (check_moves_count > 0) +     # Checks available
    0.15 * (pieces_under_attack / total_pieces)  # Attack pressure
)
# Range: 0.0 (simple) to 1.0+ (extremely complex)
```

#### 3. Tactical Priority Features (12-15 features)
- `pieces_under_multi_attack`: Count of pieces attacked by 2+ opponent pieces
- `pieces_undefended`: Count of pieces with no defenders
- `material_delta_after_move`: Centipawn change after candidate move (+/- cp)
- `material_delta_2ply`: Centipawn change after 2-ply continuation
- `material_delta_from_start`: Total cp change from position start
- `warning_flags_count`: Count of "negative" Stage 1 features activated
- `critical_piece_threats`: Queen/Rook/King under attack (binary flags)
- `pawn_structure_disruption`: Isolated/doubled/backward pawn creation
- `king_safety_delta`: Change in king safety score
- `tactical_motif_detected`: Pin/fork/skewer/discovered attack detected (one-hot encoded)
- `sacrifice_compensation`: Material sacrificed but compensation available (bool)
- `gambit_recovery_potential`: Lost material recoverable within 2 moves (bool)

#### 4. Stage 1 Integration Features (5-8 features)
- `stage1_prob_good`: Stage 1 output probability (0.0-1.0)
- `stage1_confidence`: Distance from decision boundary (|prob - 0.5|)
- `stage1_top_features`: Top 3 most activated Stage 1 features (one-hot or indices)
- `stage1_warning_features`: Top 3 most negative Stage 1 features
- `position_evaluator_score`: Stage 1's raw position score

### Outputs

**Multi-Output Prediction** (3-5 outputs):

#### 1. `complexity_score` (float, 0.0-10.0)
**Interpretation**: How complex is this position to evaluate correctly?
- 0.0-2.0: Simple (few pieces, forced lines, obvious moves)
- 2.0-5.0: Moderate (standard middlegame, clear plans)
- 5.0-8.0: Complex (tactical, multiple candidate moves, unclear evaluation)
- 8.0-10.0: Extreme (Tal-level chaos, deep sacrifices, "forest darkness")

**Usage**: Determine how much time to allocate

#### 2. `time_allocation` (float, 0.1-1.0)
**Interpretation**: Fraction of available time budget to spend on this move
- 0.1: Instant move (obvious, pre-calculated, or simple)
- 0.3: Quick move (standard opening/endgame)
- 0.5: Normal move (typical middlegame)
- 0.7: Deep thought (critical tactical decision)
- 1.0: Maximum time (complex sacrifice, unclear position)

**Calculation**:
```python
actual_time_seconds = time_budget * time_allocation
# Example: 120s budget * 0.5 allocation = 60s spent on move
```

#### 3. `move_priority_distribution` (array, length = num_good_moves)
**Interpretation**: Priority scores for each "good move" from Stage 1
- Higher score = higher priority (more aligned with V7P3R style)
- Not probabilities (don't sum to 1.0)
- Range: 0.0 (avoid) to 10.0 (strongly prefer)

**Usage**: Rank Stage 1's good moves by V7P3R personality fit

#### 4. `confidence_level` (float, 0.0-1.0)
**Interpretation**: How confident is the model in its recommendations?
- 0.0-0.3: Low confidence (position outside training distribution)
- 0.3-0.7: Moderate confidence (typical game position)
- 0.7-1.0: High confidence (position similar to training data)

**Usage**: If confidence < 0.5, fall back to traditional static engine evaluation

#### 5. `recommended_features` (array of indices)
**Interpretation**: Which Stage 1 features should be "explained" to justify move?
- Used for debugging and user-facing "why" explanations
- Helps understand model's reasoning

---

## V7P3R Personality Thresholds

### Material Sacrifice Tolerance

**Threshold**: Up to **550 centipawns** (5 pawns) material loss is acceptable IF:
1. **Recovery Possible**: Material recoverable within 2 moves via tactics OR
2. **Compensation Exists**: Positional advantages justify sacrifice:
   - King attack (mating threats)
   - Piece activity (rooks on 7th rank, active bishops)
   - Pawn structure damage to opponent
   - Time advantage (opponent forced into defensive moves)
3. **NOT Checkmate**: Sacrifice does NOT lead to forced checkmate for opponent

**Emergency Time Management Integration**:
- **Severe Time Pressure** (<10s remaining): Skip Stage 2 entirely, use cache only, instant move threshold = 0.0
- **Moderate Time Pressure** (<30s remaining): Reduce Stage 2 depth, increase cache dependency
- **Time Reserve**: Always keep 10% of clock as emergency reserve
- **Increment Farming**: In blitz (1+2s), capable of farming increment for timeout victories

**Implementation**:
```python
def is_sacrifice_acceptable(material_delta: int, position_features: dict) -> bool:
    """
    Determine if material sacrifice follows V7P3R personality.
    
    Args:
        material_delta: Centipawn loss (negative = sacrifice)
        position_features: Dict of tactical/positional features
        
    Returns:
        True if sacrifice acceptable, False if blunder
    """
    if material_delta >= 0:
        return True  # Gaining material always OK
    
    if abs(material_delta) > 550:
        return False  # Exceeds 5-pawn sacrifice limit
    
    # Check for forced checkmate against us
    if position_features['forced_checkmate_against'] > 0:
        return False  # Never sacrifice into checkmate
    
    # Check for recovery potential
    recovery = (
        position_features['gambit_recovery_potential'] or
        position_features['material_delta_2ply'] >= material_delta * 0.5
    )
    
    # Check for compensation
    compensation = (
        position_features['king_attack_score'] > 5.0 or
        position_features['piece_activity_score'] > 7.0 or
        position_features['pawn_structure_advantage'] > 2.0
    )
    
    return recovery or compensation
```

### Tal-Style Position Preferences

**Rewarded Features** (high priority in training):
1. **Sacrificial Attacks**: Material sacrifices with king attack compensation
2. **Complex Tactics**: Positions with 3+ tactical motifs (pins, forks, skewers)
3. **Initiative**: Forcing moves (checks, threats, captures)
4. **Piece Activity**: Active pieces even if material down
5. **Mating Attacks**: Direct king assault patterns

**Penalized Features** (low priority):
1. **Passive Defense**: Retreating moves without counterplay
2. **Simplification**: Trading pieces when behind in material
3. **Drawish Positions**: Symmetric pawn structures, equal material, low complexity
4. **Safe Play**: Moves that don't create threats or complications

### Special "Magic" Tactics

**Rewarded Chess Tricks**:
1. **Misdirection**: Threatening one piece while real target is elsewhere
   - Example: Queen threatens bishop, but real tactic is knight fork
   
2. **Vanishing Sacrifices**: Sacrificing piece that "reappears" later via promotion or tactical recovery
   - Example: Sacrifice rook for pawns, but passed pawn promotes to queen
   
3. **Traps**: Setting up positions where opponent has "obvious" move that loses
   - Example: Hanging piece bait that leads to back-rank mate
   
4. **Desperado Moves**: Capturing/sacrificing doomed piece for maximum damage
   - Example: Hanging knight captures queen before being taken

**Feature Encoding**: `tactical_trick_type` (one-hot encoded or multi-label)

---

## Training Methodology

### Training Data Generation

**Method**: **Self-Play Monte Carlo** (preferred)

#### Option 1: Monte Carlo Self-Play (PREFERRED)
**Process**:
1. Start from opening position or random middlegame position
2. V7P3R AI plays against itself using Stage 1 evaluator
3. Record at each move:
   - Position state (FEN)
   - Time remaining (both sides)
   - Legal moves available
   - Stage 1 evaluations for each legal move
   - Move actually played
   - Time spent on move
   - Material delta before/after move
   - Tactical features in position
4. Label data:
   - `complexity_score`: Calculated from actual branching factor + tactical density
   - `time_allocation`: Actual time spent / time budget
   - `move_priority`: Rank moves by game outcome (winning games = higher priority for moves played)

**Self-Play Configuration** (CONFIRMED):
- **Games**: **284 games minimum** (median historical learning benchmark from manual V7P3R tuning)
- **Target**: Match or exceed human learning efficiency ("Can AI learn faster than human?")
- **Time Control Distribution**:
  - 60% at 5+4 blitz (primary training)
  - 20% at 1+2 bullet (time pressure training)
  - 20% at 15+10 rapid (deep calculation training)
- **Opening Book**: Varied openings to ensure diverse positions
- **Stopping Condition**: 
  - Checkmate/stalemate (natural end)
  - Resignation when >800cp down for 5 consecutive moves (avoid pointless play)
  - Maximum 150 moves (prevent endless games)

**Historical Context**: Human (manual) V7P3R tuning required median of 284 games per version to achieve performance improvements. Stage 2 training target matches this to establish quantitative learning comparison.

#### Option 2: Play Against Old V7P3R Engines
**Process**:
1. V7P3R AI (Stage 1) plays against old static engines (v17.8, v14.1, etc.)
2. Record same data as Monte Carlo
3. Label based on engine strength differential

**Advantage**: Tests against known-strength opponents  
**Disadvantage**: Less diverse position types (static engines avoid complex positions)

### Draw and Repetition Prevention

**Threefold Repetition**:
- Detect using `board.is_repetition(2)` (O(1) hash lookup)
- **Rejection Threshold**: Reject threefold when eval >50cp
- Prefer losing tactically over accepting draws when ahead

**50-Move Rule**:
- Track half-move clock
- Force pawn move or capture before 50-move draw

**Philosophy**: "Blaze of glory over boring draw" - avoid simplification, seek complications

### Labeling Strategy

**Complexity Score** (ground truth):
```python
def calculate_complexity_ground_truth(position: dict) -> float:
    """Calculate actual complexity from game tree analysis."""
    complexity = 0.0
    
    # Legal moves (normalized)
    complexity += (position['legal_moves_count'] / 280.0) * 2.0
    
    # Tactical density
    complexity += position['tactical_patterns_count'] * 0.5
    
    # Branching factor (average of next 2 plies)
    complexity += (position['branching_factor_2ply'] / 40.0) * 3.0
    
    # Material imbalance (more imbalanced = more complex)
    mat_imbalance = abs(position['material_balance']) / 100.0
    complexity += mat_imbalance * 1.0
    
    # King safety differential (unsafe king = complex)
    king_safety_diff = abs(position['king_safety_white'] - position['king_safety_black'])
    complexity += (king_safety_diff / 10.0) * 2.0
    
    return min(complexity, 10.0)  # Cap at 10.0
```

**Time Allocation** (ground truth):
```python
def calculate_time_allocation_ground_truth(actual_time: float, budget: float) -> float:
    """Calculate actual time fraction spent."""
    return min(actual_time / budget, 1.0)  # Cap at 1.0 (100%)
```

**Move Priority** (ground truth):
```python
def calculate_move_priority_ground_truth(move: dict, game_result: str) -> float:
    """
    Calculate priority based on game outcome and move characteristics.
    
    Higher priority if:
    - Move was played in winning game
    - Move creates complications (Tal-style)
    - Move sacrifices material with compensation
    """
    priority = 5.0  # Baseline (neutral)
    
    # Game result bonus/penalty
    if game_result == 'WIN':
        priority += 2.0
    elif game_result == 'DRAW':
        priority += 0.0
    elif game_result == 'LOSS':
        priority -= 2.0
    
    # Tal-style characteristics bonus
    if move['is_sacrifice'] and move['has_compensation']:
        priority += 2.0
    if move['tactical_motifs_count'] >= 2:
        priority += 1.0
    if move['creates_threats']:
        priority += 1.0
    
    # Passive play penalty
    if move['is_retreat'] and not move['creates_counterplay']:
        priority -= 2.0
    
    return max(0.0, min(priority, 10.0))  # Clamp to [0, 10]
```

### Model Architecture

**Network Type**: Multi-Output Regression

```python
class ComplexityTimeManager(nn.Module):
    def __init__(self, input_dim=40, hidden_dims=[256, 128, 64]):
        super().__init__()
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Output heads
        self.complexity_head = nn.Sequential(
            nn.Linear(hidden_dims[2], 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Outputs 0-1, scale to 0-10 later
        )
        
        self.time_allocation_head = nn.Sequential(
            nn.Linear(hidden_dims[2], 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Outputs 0-1 (fraction)
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dims[2], 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Outputs 0-1 (confidence)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        
        complexity = self.complexity_head(features) * 10.0  # Scale to 0-10
        time_alloc = self.time_allocation_head(features)
        confidence = self.confidence_head(features)
        
        return {
            'complexity_score': complexity,
            'time_allocation': time_alloc,
            'confidence_level': confidence,
        }
```

**Note**: `move_priority_distribution` requires separate architecture (per-move scoring) - see below.

### Loss Functions

**Multi-Task Loss**:
```python
def stage2_loss(predictions, targets, weights):
    """
    Combined loss for Stage 2 multi-output training.
    
    Args:
        predictions: Dict with 'complexity_score', 'time_allocation', 'confidence_level'
        targets: Dict with ground truth values
        weights: Task weights (alpha, beta, gamma)
    """
    # Complexity loss (MSE)
    complexity_loss = F.mse_loss(predictions['complexity_score'], targets['complexity_score'])
    
    # Time allocation loss (MSE)
    time_loss = F.mse_loss(predictions['time_allocation'], targets['time_allocation'])
    
    # Confidence loss (BCE if binary, MSE if continuous)
    confidence_loss = F.mse_loss(predictions['confidence_level'], targets['confidence_level'])
    
    # Weighted combination
    total_loss = (
        weights['alpha'] * complexity_loss +
        weights['beta'] * time_loss +
        weights['gamma'] * confidence_loss
    )
    
    return total_loss, {
        'complexity_loss': complexity_loss.item(),
        'time_loss': time_loss.item(),
        'confidence_loss': confidence_loss.item(),
    }
```

**Task Weights** (initial):
- `alpha` (complexity): 0.4
- `beta` (time): 0.4
- `gamma` (confidence): 0.2

---

## Move Priority Scoring (Separate Model)

**Purpose**: Rank Stage 1's "good moves" by V7P3R personality fit

**Architecture Decision**: **Individual Scoring** (CONFIRMED - Option A)
- Each move scored independently (not pairwise or listwise ranking)
- Simpler training, faster inference
- Output: Priority score 0-10 per move

```python
class MovePriorityRanker(nn.Module):
    def __init__(self, input_dim=50):
        """
        Score individual moves for priority ranking.
        
        Input: Position features + move features + Stage 1 features
        Output: Priority score (0-10)
        
        DESIGN: Individual scoring (not pairwise ranking)
        - Faster inference (no O(n²) comparisons)
        - Simpler training (direct regression)
        - Scales better with many candidate moves
        """
        super().__init__()
        
        self.ranker = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 1),
            nn.Sigmoid()  # 0-1, scale to 0-10
        )
    
    def forward(self, move_features):
        """
        Score single move independently.
        
        Args:
            move_features: [batch_size, input_dim] tensor
            
        Returns:
            priority: [batch_size, 1] tensor (0-10 scale)
        """
        priority = self.ranker(move_features) * 10.0
        return priority
```

**Training Data**: Per-move features with priority labels from game outcomes

**Inference**: Score all Stage 1 "good moves" in parallel, select highest priority

---

## Static Checkmate Calculator Integration

**Philosophy**: Always include traditional rapid checkmate detection

**Source**: V7P3R v18.6.3 checkmate detection (validated efficient implementation)

**Implementation**: Minimax with alpha-beta pruning + **adaptive depth**

**Adaptive Depth Strategy** (CONFIRMED):
- **Base depth**: 5 (finds mate-in-3, ~100-200ms)
- **Time-based adjustment**: If Stage 2 allocates extra time, increase to depth 7 (mate-in-4)
- **Emergency depth**: If <10s remaining, reduce to depth 3 (mate-in-2, ~50ms)
- **Viper Strike**: If checkmate found, time_allocation drops to 0.0 (instant move)

```python
def static_checkmate_search(position: Board, max_depth: int = 5, time_available: float = None) -> Optional[Move]:
    """
    Traditional minimax search for forced checkmate with adaptive depth.
    
    Args:
        position: Current board position
        max_depth: Maximum search depth (default 5 = mate-in-3)
        time_available: Seconds available (optional, enables adaptive depth)
        
    Returns:
        Checkmating move if found within max_depth, else None
        
    Adaptive Depth:
        - time_available > 10s: depth 7 (mate-in-4)
        - time_available 3-10s: depth 5 (mate-in-3) [DEFAULT]
        - time_available < 3s: depth 3 (mate-in-2)
    """
    # Adaptive depth based on time
    if time_available is not None:
        if time_available > 10.0:
            max_depth = 7  # Deep search if time permits
        elif time_available < 3.0:
            max_depth = 3  # Shallow search if time pressure
    """
    Traditional minimax search for forced checkmate.
    
    Returns:
        Checkmating move if found within max_depth, else None
    """
    def minimax(board, depth, alpha, beta, maximizing):
        if depth == 0 or board.is_game_over():
            if board.is_checkmate():
                return 99999 if maximizing else -99999
            return 0
        
        if maximizing:
            max_eval = -99999
            for move in board.legal_moves:
                board.push(move)
                eval = minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = 99999
            for move in board.legal_moves:
                board.push(move)
                eval = minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
    
    # Check each legal move for forced mate
    for move in position.legal_moves:
        position.push(move)
        eval = minimax(position, max_depth - 1, -99999, 99999, False)
        position.pop()
        
        if eval == 99999:  # Found forced mate
            return move
    
    return None  # No mate found
```

**Integration**: Run checkmate search in parallel with Stage 2 evaluation

**Priority**: If checkmate found, ALWAYS play it (override Stage 2)

**Viper Strike Trigger**: When checkmate detected, `time_allocation` = 0.0 (instant strike)

### Endgame Tablebases (Lightweight Integration)

**Philosophy**: Include **common V7P3R endgames only** (not comprehensive tablebases)

**Approach**:
1. Analyze self-play game distribution to identify most frequent endgames
2. Build custom lightweight tables for top 10-20 endgame patterns
3. Package with model (no external dependencies)
4. **Omit rare endgame chunks** to save space (e.g., skip K+B+N vs K)

**Common Endgames to Include** (based on V7P3R aggressive play style):
- K+Q vs K (trivial but must know)
- K+R vs K
- K+R+R vs K
- K+Q+Q vs K (rare but appears in promotions)
- K+R+B vs K (tactical endgame)
- K+R+N vs K (tactical endgame)
- K+P vs K (pawn races)

**Omit Complex/Rare**:
- K+B+N vs K (too rare for V7P3R's style)
- Triple minor pieces (statistical noise)

**Implementation**: SQLite database or Python dict for instant lookup (FEN → result)

---

## Training Configuration

### Hyperparameters (Initial)

```python
STAGE2_CONFIG = {
    'epochs': 30,
    'batch_size': 256,
    'learning_rate': 0.0005,
    'hidden_dims': [256, 128, 64],
    'dropout': 0.3,
    'train_val_split': 0.8,
    'random_seed': 42,
    
    # Loss weights
    'alpha': 0.4,  # Complexity weight
    'beta': 0.4,   # Time allocation weight
    'gamma': 0.2,  # Confidence weight
    
    # Self-play config (CONFIRMED)
    'num_selfplay_games': 284,  # Median historical learning benchmark
    'time_control_distribution': {
        '5+4': 0.60,  # 60% blitz (primary training)
        '1+2': 0.20,  # 20% bullet (time pressure)
        '15+10': 0.20,  # 20% rapid (deep calculation)
    },
    'max_moves_per_game': 150,
    'resignation_threshold_cp': 800,  # Resign if down 800cp for 5 moves
    'resignation_move_count': 5,
}
```

### Target Metrics

**Complexity Prediction**:
- **MSE**: ≤1.0 (on 0-10 scale)
- **MAE**: ≤0.7

**Time Allocation Prediction**:
- **MSE**: ≤0.05 (on 0-1 scale)
- **MAE**: ≤0.15

**Confidence Prediction**:
- **MSE**: ≤0.1
- **Accuracy**: ≥80% (if thresholded at 0.5)

**Move Priority Ranking**:
- **NDCG** (Normalized Discounted Cumulative Gain): ≥0.75
- **Kendall's Tau**: ≥0.6 (rank correlation with ground truth)

---

## Data Pipeline

### Self-Play Game Recording Format

**JSONL Format** (one line per position):
```json
{
  "game_id": "selfplay_00001",
  "position_id": "selfplay_00001_move_12",
  "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 5",
  "move_number": 12,
  "side_to_move": "white",
  
  "time_state": {
    "white_remaining": 180.5,
    "black_remaining": 195.2,
    "white_increment": 4.0,
    "black_increment": 4.0,
    "last_move_time": 8.3
  },
  
  "position_features": {
    "legal_moves_count": 34,
    "capture_moves_count": 3,
    "check_moves_count": 1,
    "tactical_density": 2.5,
    "forest_darkness_score": 4.2,
    "pieces_under_attack": 1,
    "pieces_undefended": 0,
    "material_balance": 0
  },
  
  "stage1_evaluations": [
    {"move": "Nf3-g5", "prob_good": 0.87, "features": [...]},
    {"move": "Bc4-b5", "prob_good": 0.82, "features": [...]},
    {"move": "d2-d4", "prob_good": 0.76, "features": [...]}
  ],
  
  "move_played": "Nf3-g5",
  "time_spent": 12.4,
  "material_delta_after": -30,
  "material_delta_2ply": 50,
  
  "game_result": "1-0",
  "game_outcome_cp": 250,
  
  "labels": {
    "complexity_score": 6.8,
    "time_allocation": 0.45,
    "move_priority": 8.5,
    "confidence_level": 0.75
  }
}
```

### Data Preprocessing

**Feature Extraction**:
1. Load position from FEN
2. Calculate all 40 Stage 2 input features
3. Normalize time features (seconds → fraction of game time)
4. Normalize complexity features (0-1 range)
5. Encode tactical motifs (one-hot)

**Batching**: Group by similar complexity (avoid mixing simple/complex in same batch)

---

## Next Steps

### Phase 1: Data Collection (1-2 weeks)
1. Implement self-play infrastructure
2. Run 10,000 self-play games using Stage 1 evaluator
3. Record all position data with labels
4. Validate data quality (check distributions)

### Phase 2: Feature Engineering (1 week)
1. Implement all 40 Stage 2 features
2. Validate feature calculations on sample positions
3. Analyze feature importance (correlation with labels)
4. Consider feature selection/PCA

### Phase 3: Model Training (1-2 weeks)
1. Implement ComplexityTimeManager network
2. Implement MovePriorityRanker network
3. Train both models on self-play data
4. Hyperparameter tuning (grid search or Optuna)
5. Validate on held-out test set

### Phase 4: Integration Testing (1 week)
1. Integrate Stage 2 with Stage 1
2. Test on sample positions
3. Measure inference speed
4. Optimize for real-time gameplay

---

## Confirmed Architectural Decisions

**All design questions resolved** - ready for implementation:

✅ **Move Priority Model**: Option A - Individual scoring (faster, simpler)  
✅ **Self-Play Termination**: Resignation at >800cp down for 5 moves OR 150 move limit  
✅ **Complexity Calibration**: Absolute 0-10 scale (game-phase agnostic)  
✅ **Checkmate Search Depth**: Adaptive depth 3-7 based on time available  
✅ **Feature Count**: ~40 total features (balanced)  
✅ **Training Target**: 284 games (median historical benchmark)  
✅ **Time Proxy Metric**: Processing tick counts (not actual time prediction)  
✅ **Architecture**: Combined complexity/time model + separate move priority ranker  
✅ **Material Sacrifice**: 550cp threshold with compensation/recovery  
✅ **Emergency Time**: <10s = skip Stage 2, cache only  
✅ **Tablebases**: Lightweight custom tables for common endgames only  
✅ **Draw Prevention**: Reject threefold when eval >50cp  

### Pre-Calculation Queue & Ponder Integration

**UCI Ponder Legality**: Must verify Lichess.org allows ponder mode (calculate during opponent thinking)

**Queue Design**:
- **Storage**: SQLite database (Zobrist/FEN hash → pre-calculated Stage 1/Stage 2 results)
- **Population Depth**: 2-ply recommended (~25-150 positions per opponent move)
- **Cache Size**: Max 100k positions (~500MB memory)
- **Eviction**: LRU (Least Recently Used)
- **Target Hit Rate**: 30%+ cache hits per game

**Legal Considerations**:
- Ponder time = opponent thinking time (UCI standard)
- Pre-calculation allowed if engine declares ponder support
- **ACTION REQUIRED**: Verify Lichess competition rules permit ponder mode
- If ponder disallowed, use queue only during own time (reduced effectiveness)

---

## Conclusion

Stage 2 transforms V7P3R AI from a position evaluator into a **time-intelligent tactical decision-maker** that thrives in complex, Tal-inspired positions where traditional engines struggle.

**"Viper Strike" Philosophy**: When checkmate detected, complexity and time calculations drop to zero - pure instinct takes over for instant killing blow.

By learning to manage complexity and allocate time wisely, the AI can **drive the game into its preferred territory** while avoiding time pressure and maintaining its aggressive, sacrificial personality.

**Can AI learn faster than human?** Training target: 284 games (median historical manual tuning benchmark).

**Status**: 🚀 **READY FOR IMPLEMENTATION** - All architectural decisions confirmed. Next: Build static modules, then self-play infrastructure.
