# V7P3R v8.0 - Pure Learned Architecture (Simplified Genius Design)

## Core Philosophy: Let the Model Discover Chess

> "We give it darkness score and all custom features, but let the model decide what good vs bad is. We've been trying so hard to teach the model that, but I want it to just find out on its own."

---

## Architectural Comparison

### CURRENT v7.2 (Hand-Coded Complexity)
```python
# We extract features (good!)
features = extract_all_features(board)  # 55 dimensions

# WE decide what's good/bad (complexity we're removing!)
if features['forest_darkness'] > 0.6:
    complexity_reward = 0.3  # ← WE SAY "complexity = good"
if material_sacrifice:
    sacrifice_reward = 0.5   # ← WE SAY "sacrifices = good"

# Combine with Stockfish (training wheels)
target = stockfish_eval * 0.2 + personality_reward * 0.8

# Complex validation logic
if not is_safe(move):
    reject_move()
if not maintains_style(move):
    penalize_move()
```

**Problems**:
- ✗ We're guessing which features matter
- ✗ Personality rewards might be wrong
- ✗ Complex validation logic adds overhead
- ✗ Model never learns true cause-and-effect

---

### PROPOSED v8.0 (Pure Learned Simplicity)
```python
# Extract features (same 55 dims - unchanged)
features = extract_all_features(board)  # Darkness, mobility, king safety, etc.

# NO HAND-CODED REWARDS
# Just raw features → neural network → move choice

# ONLY ground truth: Did I win?
reward = +1.0 if game_won else -1.0

# Reward shaper learns automatically:
# "When I saw high darkness + low material + active pieces, I won 70% of games"
# "When I saw low darkness + high material + passive pieces, I won 30% of games"
# → Conclusion: Darkness + activity > material (in certain positions)
```

**Benefits**:
- ✅ Model discovers what's good/bad through experience
- ✅ No guessing about reward weights
- ✅ Simple, elegant training loop
- ✅ Can discover patterns we never thought of

---

## What We KEEP (Proven Infrastructure)

### 1. Feature Extraction (55 Dimensions)
```python
# ComprehensiveFeatureExtractor - UNCHANGED
features = {
    # Stage 1: Fast features
    'piece_counts': [...],           # Keep
    'material_balance': 0.5,         # Keep
    'legal_moves_count': 24,         # Keep
    
    # Heuristics
    'forest_darkness': 0.72,         # Keep - let MODEL learn if it's good
    'bishop_pair': 1,                # Keep - let MODEL learn value
    'passed_pawns': 2,               # Keep - let MODEL learn importance
    
    # Complexity
    'tactical_density': 0.45,        # Keep - MODEL decides if tactics are good
    'move_diversity': 0.68,          # Keep - MODEL discovers correlation
    
    # Temporal
    'move_number_normalized': 0.35,  # Keep - MODEL learns time urgency
    'urgency_score': 0.42            # Keep - MODEL finds its meaning
}
```

**Key Insight**: We keep ALL features, but remove the "is this good?" judgments. Raw data in, model learns correlations.

### 2. Tablebase Integration
```python
# UNCHANGED - Keep exact same logic
if tablebase_oracle.is_available(board):
    best_move = tablebase_oracle.get_best_move(board)
    if best_move:
        # Declare winner immediately
        return winner, experience_buffer
```

**Why Keep**: Endgames are solved mathematically. No need for model to learn K+Q vs K mates - just use perfect play.

### 3. Opening Book (NEW APPROACH)
See "Opening Book as Learnable Feature" section below.

---

## What We REMOVE (Hand-Coded Complexity)

### ❌ 1. Personality Rewards
```python
# DELETE THIS ENTIRE SECTION
class PlaystyleProfile:
    def __init__(self):
        self.complexity_preference = 0.7    # ← DELETE
        self.sacrifice_tolerance = 0.5      # ← DELETE
        self.defensive_weight = 0.3         # ← DELETE

def calculate_personality_reward(position, profile):
    # DELETE ALL OF THIS
    reward = 0.0
    if darkness > 0.6:
        reward += profile.complexity_preference * 0.3
    if material_loss > 2:
        reward += profile.sacrifice_tolerance * 0.5
    return reward
```

**Replacement**: Model learns these patterns automatically through win/loss.

### ❌ 2. Phase-Aware Stockfish Weighting
```python
# DELETE THIS
class PhaseAwareTrainingTarget:
    def calculate_target(self, stockfish_eval, game_phase):
        if game_phase == 'opening':
            return stockfish_eval * 0.9  # ← DELETE
        elif game_phase == 'middlegame':
            return stockfish_eval * 0.2  # ← DELETE
        elif game_phase == 'endgame':
            return stockfish_eval * 1.0  # ← DELETE
```

**Replacement**: Model discovers which phases matter through experience.

### ❌ 3. Complex Move Validation
```python
# DELETE (or simplify to just legal move check)
def is_move_safe(board, move):
    # Complex safety checks - DELETE
    if exposes_king(move):
        return False
    if loses_material_unnecessarily(move):
        return False
    return True
```

**Replacement**: Model learns "moves that expose king tend to lead to losses."

### ❌ 4. Hand-Coded Evaluation Targets
```python
# DELETE THIS
def winner_only_target(game_phase, move_number):
    if game_phase == 'opening':
        return 0.6   # ← We're guessing!
    elif game_phase == 'middlegame':
        return 0.85  # ← We're guessing!
    elif game_phase == 'endgame':
        return 0.93  # ← We're guessing!
```

**Replacement**: Model learns what "winning positions" feel like through actual wins.

---

## Opening Book as Learnable Feature (THE GENIUS PART)

### Current Approach (Simplistic)
```python
# Just look up and execute
if board.ply() < 12:
    move = opening_book.get_move(board.fen())
    board.push(move)
```

**Problem**: Model doesn't learn anything about openings. It's just a lookup table.

---

### NEW Approach (Opening Choice as Meta-Action)

#### Step 1: Encode Openings as Features
```python
OPENING_BOOK = {
    0: "Sicilian Dragon",           # e4 c5, Nf3 d6, d4 cxd4, Nxd4 Nf6, Nc3 g6
    1: "King's Indian Defense",     # d4 Nf6, c4 g6, Nc3 Bg7, e4 d6
    2: "Ruy Lopez Berlin",          # e4 e5, Nf3 Nc6, Bb5 Nf6
    3: "French Defense",            # e4 e6, d4 d5, Nc3 Nf6
    4: "Queen's Gambit Declined",   # d4 d5, c4 e6, Nc3 Nf6
    5: "Caro-Kann",                 # e4 c6, d4 d5, Nc3 dxe4
    # ... 20-50 variations total
}
```

#### Step 2: Model Chooses Opening (Single Decision)
```python
# At start of game, model makes ONE meta-choice
opening_id = model.select_opening()  # Returns integer 0-49

print(f"Model chose: {OPENING_BOOK[opening_id]}")

# Execute ENTIRE opening variation as macro
opening_moves = OPENING_SEQUENCES[opening_id]
for move_uci in opening_moves:
    board.push_uci(move_uci)
    
# Now model learns from move 10-15 onward
```

#### Step 3: Opening Becomes Learnable Feature
```python
class GameExperience:
    def __init__(self):
        self.opening_choice = None  # ← New field
        self.features = []
        self.reward = 0.0
        
# Store opening choice with game result
experience.opening_choice = opening_id
experience.reward = +1.0 if won else -1.0

# Reward shaper learns:
# "When I chose Sicilian Dragon (id=0), I won 65% of games vs Gen 3"
# "When I chose French Defense (id=3), I won 40% of games vs Gen 3"
# → Next generation: More likely to choose Sicilian Dragon
```

#### Step 4: Opening Diversity Through Exploration
```python
# During training, use epsilon-greedy
if random.random() < epsilon:
    opening_id = random.randint(0, NUM_OPENINGS - 1)  # Explore
else:
    opening_id = model.select_best_opening()          # Exploit

# Epsilon decays: Gen 1 = 0.5 (50% random), Gen 10 = 0.1 (10% random)
```

**Why This Works**:
- ✅ Model learns which openings lead to favorable positions
- ✅ Discovers opening preferences through experience (might prefer sharp tactics)
- ✅ Can adapt to opponent's weaknesses (if opponent weak vs Sicilian, play more Sicilian)
- ✅ Entire opening variation executed instantly (no learning opening theory move-by-move)

---

## Reward Shaper Architecture

### Network Design
```python
class PureRewardShaper(nn.Module):
    """
    Learns which raw features correlate with winning.
    NO hand-coded judgments - pure correlation discovery.
    """
    
    def __init__(self, feature_dim=55, num_reward_types=10):
        super().__init__()
        
        # Deeper network for complex pattern recognition
        self.fc1 = nn.Linear(feature_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.weight_head = nn.Linear(64, num_reward_types)
        
    def forward(self, features):
        """
        Input: 55-dim raw features (darkness, mobility, king safety, etc.)
        Output: 10 learned weights (0-1 each, sum to 1.0)
        
        Learns patterns like:
        - "High darkness + low material → Won 70% of games → Weight darkness high"
        - "High material + low mobility → Won 30% of games → Weight material low"
        """
        x = torch.relu(self.fc1(features))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        weights = torch.softmax(self.weight_head(x), dim=-1)
        
        return weights
```

### Feature Groupings for Shaper
```python
FEATURE_GROUPS = {
    'material': [0, 1, 2, 3, 4, 5],           # Piece counts, material balance
    'mobility': [19, 20, 21],                 # Legal moves, piece activity
    'king_safety': [22, 23, 24],              # King pawn shield, king tropism
    'pawn_structure': [25, 26, 27, 28],       # Passed pawns, doubled, isolated
    'complexity': [43, 44, 45, 46],           # Darkness, tactical density, diversity
    'development': [29, 30],                  # Development score, castling
    'center_control': [31, 32],               # Center pawns, center pieces
    'piece_coordination': [33, 34, 35],       # Rook/knight/bishop placement
    'endgame_patterns': [47, 48, 49],         # Pawn promotion potential, opposition
    'temporal_urgency': [51, 52, 53, 54]      # Move number, halfmove clock, urgency
}

# Shaper learns: "In this position, weight complexity group 80%, material 20%"
```

---

## Simplified Training Loop

### Before (v7.2 - Complex)
```python
def train_generation():
    # 1. Play game with personality rewards
    for move_num in range(max_moves):
        features = extract_features(board)
        
        # Complex reward calculation
        stockfish_eval = oracle.evaluate(board)
        personality_reward = calculate_personality(features, profile)
        phase_weight = get_phase_weight(game_phase)
        
        target = (
            stockfish_eval * (1 - phase_weight) +
            personality_reward * phase_weight
        )
        
        # Complex move validation
        legal_moves = filter_safe_moves(
            filter_style_moves(
                board.legal_moves
            )
        )
        
        move = select_from_filtered(legal_moves)
        board.push(move)
    
    # 2. Train on complex targets
    train(experiences, targets)
```

### After (v8.0 - Simple & Pure)
```python
def train_generation():
    # 1. Choose opening (meta-action)
    opening_id = model.select_opening()
    execute_opening_macro(opening_id)
    
    # 2. Play from middle game onward
    for move_num in range(opening_length, max_moves):
        # Check tablebase (keep this)
        if tablebase.is_available(board):
            winner = tablebase.get_winner(board)
            break
        
        # Extract raw features (no judgments)
        features = extract_all_features(board)  # 55 dims
        
        # Model chooses move (NO filtering, NO validation)
        move = model.select_move(features)
        board.push(move)
    
    # 3. Game ends - ONLY ground truth matters
    reward = +1.0 if won else -1.0
    
    # 4. Train reward shaper
    shaper.learn(
        features_sequence=game_features,
        opening_choice=opening_id,
        final_reward=reward
    )
    
    # 5. Train policy with shaped rewards
    policy.train(experiences, shaped_rewards)
```

**Lines of Code**:
- v7.2: ~800 lines (personality, phase-aware, validation)
- v8.0: ~300 lines (pure learning, no complexity)

---

## Training Workflow

### Phase 1: Supervised Bootstrap (Unchanged)
```
GM Games (winner positions only)
  ↓
Extract 55-dim features
  ↓
Train both Policy AND Shaper
  ↓
Shaper learns: "In GM games, these feature patterns correlated with winning"
```

### Phase 2: Pure Self-Play Evolution
```
For each generation (1-10):
    For each game (1-100):
        1. Model chooses opening variation (1 of 50)
        2. Execute opening macro (10-15 moves instantly)
        3. Play from middlegame using raw features
        4. If tablebase position reached → Declare winner
        5. Otherwise play until checkmate/draw/max-moves
        6. Store: features, opening_choice, final_reward
    
    Train reward shaper:
        - Learn which feature patterns → wins
        - Learn which opening choices → wins
    
    Train policy:
        - Use shaper's learned rewards as training signal
    
    Evaluate vs previous generation:
        - If win rate > 55% → Accept
        - Else → Reject, rollback
```

---

## Opening Book Implementation

### Opening Book Format (JSON)
```json
{
    "openings": [
        {
            "id": 0,
            "name": "Sicilian Dragon",
            "moves": ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "g7g6"],
            "ply_count": 10,
            "initial_position": "starting",
            "tags": ["sharp", "tactical", "complex"]
        },
        {
            "id": 1,
            "name": "Ruy Lopez Berlin",
            "moves": ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "g8f6"],
            "ply_count": 6,
            "initial_position": "starting",
            "tags": ["solid", "defensive", "endgame-oriented"]
        }
        // ... 48 more openings
    ]
}
```

### Opening Encoder
```python
class OpeningEncoder:
    """Converts opening variations into model-usable format"""
    
    def __init__(self, opening_book_path):
        with open(opening_book_path) as f:
            self.openings = json.load(f)['openings']
        
        self.num_openings = len(self.openings)
        
    def execute_opening(self, board, opening_id):
        """Execute full opening variation"""
        opening = self.openings[opening_id]
        
        for move_uci in opening['moves']:
            board.push_uci(move_uci)
        
        return board, opening['ply_count']
    
    def get_opening_feature(self, opening_id):
        """
        Convert opening ID to feature vector
        Could be one-hot encoding or learned embedding
        """
        # Simple one-hot (for 50 openings)
        feature = np.zeros(self.num_openings)
        feature[opening_id] = 1.0
        return feature
```

### Model Opening Selection
```python
class V8ValueNetwork(nn.Module):
    def __init__(self, position_features=55, num_openings=50):
        super().__init__()
        
        # Separate head for opening selection
        self.opening_selector = nn.Sequential(
            nn.Linear(position_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_openings),
            nn.Softmax(dim=-1)
        )
        
        # Main value head (unchanged)
        self.value_head = nn.Sequential(...)
        
    def select_opening(self, initial_features, temperature=0.3):
        """
        Choose opening variation at game start
        
        Args:
            initial_features: 55-dim features of starting position
            temperature: Exploration parameter (lower = more deterministic)
        
        Returns:
            opening_id: Integer 0-49
        """
        opening_probs = self.opening_selector(initial_features)
        
        # Sample with temperature
        if temperature > 0:
            opening_probs = opening_probs ** (1 / temperature)
            opening_probs = opening_probs / opening_probs.sum()
            opening_id = torch.multinomial(opening_probs, 1).item()
        else:
            opening_id = opening_probs.argmax().item()
        
        return opening_id
```

---

## What Model Learns Automatically

### Without Hand-Coding, Model Discovers:

#### 1. Material Value
```
Observation over 1000 games:
- When material_balance > 2.0 → Won 75% of games
- When material_balance < -2.0 → Won 15% of games

Shaper learns: "Weight material feature highly"
```

#### 2. Complexity Correlation
```
Observation:
- High darkness + active pieces → Won 68% vs passive opponents
- High darkness + passive pieces → Won 32%

Shaper learns: "Darkness only good if paired with activity"
```

#### 3. King Safety Phases
```
Observation:
- Opening: King_safety < 0.3 → Won 40% (lost to attacks)
- Middlegame: King_safety < 0.3 → Won 25% (got mated)
- Endgame: King_safety < 0.3 → Won 60% (active king is good!)

Shaper learns: "King safety weight decreases in endgame"
```

#### 4. Opening Effectiveness
```
Observation:
- Sicilian Dragon (id=0) → Win rate 62% vs Gen 5
- French Defense (id=3) → Win rate 48% vs Gen 5

Shaper learns: "Choose Sicilian Dragon more often vs this opponent style"
```

**The Beauty**: We don't tell it any of this. It discovers these patterns through experience, just like a human player learning chess!

---

## Manual Override Mechanism

> "If it starts to ignore certain characteristics we want it to focus on, we can manually weight certain features"

### Feature Importance Monitoring
```python
class FeatureMonitor:
    """Track which features model is using"""
    
    def analyze_feature_usage(self, generation_num):
        """After each generation, check feature weights"""
        
        avg_weights = calculate_average_shaper_weights()
        
        print(f"Generation {generation_num} Feature Usage:")
        for feature_name, weight in avg_weights.items():
            print(f"  {feature_name}: {weight:.3f}")
        
        # Flag underused features
        if avg_weights['forest_darkness'] < 0.05:
            print("⚠️  WARNING: Model ignoring complexity features")
        
        if avg_weights['king_safety'] < 0.05:
            print("⚠️  WARNING: Model ignoring king safety")
```

### Manual Weight Injection
```python
class ManualWeightOverride:
    """Force model to consider specific features"""
    
    def __init__(self):
        self.overrides = {}
    
    def set_minimum_weight(self, feature_name, min_weight):
        """
        Ensure feature gets at least this weight
        
        Example:
            override.set_minimum_weight('forest_darkness', 0.15)
            # Model must weight complexity at least 15%
        """
        self.overrides[feature_name] = min_weight
    
    def apply_overrides(self, shaper_weights):
        """Apply manual constraints to learned weights"""
        
        for feature, min_weight in self.overrides.items():
            if shaper_weights[feature] < min_weight:
                # Boost this weight, normalize others
                deficit = min_weight - shaper_weights[feature]
                shaper_weights[feature] = min_weight
                
                # Subtract deficit from other weights proportionally
                other_features = [f for f in shaper_weights if f != feature]
                for other in other_features:
                    shaper_weights[other] *= (1 - deficit)
        
        return shaper_weights
```

**Usage**:
```python
# If model ignores complexity
if gen_5_complexity_weight < 0.1:
    override.set_minimum_weight('forest_darkness', 0.2)
    print("Forcing model to consider complexity at least 20%")
```

---

## File Modifications Required

### Files to MODIFY (Strip Complexity)

#### 1. `supervised_gm_trainer.py`
```python
# REMOVE
- PlaystyleProfile loading
- calculate_personality_reward()
- phase_aware_target()

# KEEP
- Feature extraction
- Neural network training
- Winner-only position extraction

# CHANGE
def extract_positions_from_game(pgn_game):
    # OLD: Calculate complex targets
    target = stockfish * 0.2 + personality * 0.8
    
    # NEW: Simple binary target
    target = 1.0 if winner else 0.0  # Just "winning position" or not
```

#### 2. `selfplay_trainer.py`
```python
# REMOVE
- Complex move validation
- Personality reward calculation
- Phase-aware evaluation

# KEEP
- Feature extraction
- Tablebase integration
- Game loop

# ADD
- Opening selection logic
- Opening macro execution
```

#### 3. `generational_trainer.py`
```python
# REMOVE
- PhaseAwareTrainingTarget
- Stockfish blending logic

# KEEP
- Generation acceptance/rejection
- Tournament evaluation

# ADD
- Opening book loading
- Opening diversity tracking
```

#### 4. `network.py`
```python
# ADD
class V8ValueNetwork(nn.Module):
    def __init__(self, position_features=55, num_openings=50):
        # Add opening selector head
        self.opening_selector = nn.Sequential(...)
        
        # Keep existing value head
        self.value_head = nn.Sequential(...)

class PureRewardShaper(nn.Module):
    # New network for learned reward weights
```

### Files to DELETE
- `personality_tuner.py` (no longer needed)
- `phase_manager.py` (model learns phases automatically)
- `stockfish_oracle.py` (optional - only keep if want validation, not for training)

### Files to CREATE
- `opening_encoder.py` (opening book loading and execution)
- `reward_shaper.py` (learned reward weighting network)
- `feature_monitor.py` (track feature usage, manual overrides)

---

## Expected Training Improvements

### Quantitative Predictions

| Metric | v7.2 (Complex) | v8.0 (Pure) | Improvement |
|--------|---------------|-------------|-------------|
| **Training Speed** | 10-15 hours | 6-10 hours | 35% faster |
| **Code Complexity** | 800 lines | 300 lines | 62% simpler |
| **Acceptance Rate** | 40-60% | 60-80% | Better convergence |
| **Opening Diversity** | Fixed book | Learns preferences | Adaptive |
| **Endgame Performance** | Tablebase-assisted | Same (keep TB) | Unchanged |
| **Novelty Discovery** | Limited | High | Finds new patterns |

### Qualitative Benefits
- ✅ **Self-Discovery**: Model finds patterns we didn't code
- ✅ **Adaptability**: Learns which openings work vs different opponents
- ✅ **Simplicity**: Easier to debug, maintain, extend
- ✅ **Robustness**: No risk of wrong reward assumptions
- ✅ **Transferability**: Same architecture works for other games (your fighting game AI!)

---

## Success Criteria

**v8.0 is successful if:**

1. **Training Completes**: 10 generations finish without manual intervention
2. **Learns Opening Preferences**: Opening diversity decreases as model finds effective variations
3. **Discovers Feature Importance**: Shaper weights make intuitive sense (e.g., high material weight in endgames)
4. **Beats v7.2**: >55% win rate in 100-game tournament
5. **Code Simplicity**: <400 lines total (vs v7.2's ~800)
6. **No Manual Tuning**: Zero hand-coded reward weights required

---

## Implementation Timeline

### Week 1: Strip Complexity
- [ ] Remove personality rewards from supervised trainer
- [ ] Remove phase-aware logic from generational trainer
- [ ] Simplify training targets to binary win/loss
- [ ] Test that training still runs (even if worse performance)

### Week 2: Add Opening System
- [ ] Create opening book JSON (50 variations)
- [ ] Implement `OpeningEncoder` class
- [ ] Add opening selector head to `V8ValueNetwork`
- [ ] Test opening macro execution

### Week 3: Implement Reward Shaper
- [ ] Create `PureRewardShaper` network
- [ ] Modify training loop to use learned rewards
- [ ] Add feature usage monitoring
- [ ] Implement manual override system

### Week 4: Full Training Run
- [ ] Train v8.0 for 10 generations
- [ ] Compare vs v7.2 baseline
- [ ] Analyze learned feature weights
- [ ] Document discoveries

---

## Next Steps

1. **Let v7.2 finish** - Establishes baseline performance
2. **Create opening book JSON** - Research best 50 variations to include
3. **Prototype reward shaper** - Test on 2 generations first
4. **Full v8.0 training** - Complete 10-generation run
5. **Transfer to fighting game** - Use same architecture!

---

*Document created: 2026-06-06*  
*Author: AI Research Team + User Vision*  
*Status: DESIGN PROPOSAL - Ready for implementation after v7.2 completion*
