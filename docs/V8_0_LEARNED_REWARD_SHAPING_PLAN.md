# V7P3R v8.0 - Learned Reward Shaping Plan

## Vision: Self-Tuning Rewards for Chess Mastery

Instead of hand-tuning reward weights (complexity, sacrifices, king safety), let the AI **learn which rewards matter most** in different positions through meta-reinforcement learning.

---

## Current State (v7.2 - Fixed Rewards)

### Hard-Coded Reward Structure
```python
# Humans decide these weights
stockfish_weight = 0.20  # Fixed
personality_weight = 0.80  # Fixed

# Personality rewards (hardcoded)
complexity_reward = 0.3 if forest_darkness > 0.6 else 0.0
sacrifice_reward = 0.5 if material_loss > 2 else 0.0
```

### Problems
- ✗ We guess which features matter (development? king safety? center control?)
- ✗ Fixed weights don't adapt to position type (opening vs endgame)
- ✗ Manual tuning required after every architecture change
- ✗ Can't discover novel evaluation concepts on its own

---

## Proposed State (v8.0 - Meta-Learned Rewards)

### Architecture: Dual-Network System

```
┌─────────────────────────────────────────────────────────┐
│                   POLICY NETWORK                        │
│            (Learns what moves to play)                  │
│   Input: 55-dim features → Output: Move probabilities   │
└─────────────┬───────────────────────────────────────────┘
              │ Uses shaped rewards
              ↓
┌─────────────────────────────────────────────────────────┐
│                   REWARD SHAPER                         │
│         (Learns which rewards matter when)              │
│   Input: 55-dim features → Output: Reward weights       │
└─────────────┬───────────────────────────────────────────┘
              │ Proposes dynamic weights
              ↓
         [Self-Play Loop]
              ↓
    [Win/Loss Ground Truth] ← Ultimate supervisor
```

### Reward Shaper Network

```python
class DynamicRewardShaper(nn.Module):
    """
    Meta-learner that discovers which evaluation features matter most.
    
    Learns position-dependent reward weights:
    - Opening: Prioritize development, center control
    - Middlegame: Prioritize tactics, king safety
    - Endgame: Prioritize pawn promotion, tablebase conversion
    """
    
    def __init__(self, feature_dim=55, num_reward_types=8):
        super().__init__()
        
        # Architecture: 55 → 128 → 64 → 8 weights
        self.fc1 = nn.Linear(feature_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.weight_head = nn.Linear(64, num_reward_types)
        
    def forward(self, position_features):
        """
        Output: 8 learned weights (0-1 each, sum to 1.0)
        
        Returns:
            material_weight: How much to care about material balance
            king_safety_weight: How much to care about king protection
            development_weight: How much to care about piece activity
            center_control_weight: How much to care about center dominance
            pawn_structure_weight: How much to care about pawn weaknesses
            mobility_weight: How much to care about legal moves
            tactical_weight: How much to care about pins/forks
            endgame_weight: How much to care about pawn promotion
        """
        x = torch.relu(self.fc1(position_features))
        x = torch.relu(self.fc2(x))
        weights = torch.softmax(self.weight_head(x), dim=-1)
        
        return weights
```

---

## Training Workflow

### Phase 1: Supervised Pre-Training (Bootstrap)
```
GM Games (winner-only positions)
  ↓
Train both Policy AND Shaper on known-good positions
  ↓
Shaper learns: "In GM games, they valued development early, king safety mid-game"
```

### Phase 2: Meta-Generational Training
```
For each generation (1-10):
    1. Shaper proposes reward weights for current position
    2. Policy trains using those shaped rewards
    3. Play 100 self-play games
    4. Measure: Win rate vs previous generation
    5. Update Shaper: 
       - If win rate > 55% → Reinforce shaper's weight choices
       - If win rate < 45% → Penalize shaper's weight choices
    6. Save both Policy and Shaper if accepted
```

### Phase 3: Adversarial Shaping (Advanced)
```
Two competing shapers:
  - Shaper A: Tries to help Policy win
  - Shaper B: Tries to make Policy fail (adversarial)
  
Policy learns robust strategy that works despite adversarial shaping
```

---

## Reward Types (What Shaper Learns to Weight)

### Sparse Rewards (Game Outcomes)
- **Win**: +1.0 (checkmate)
- **Draw**: 0.0 (stalemate, repetition, 50-move rule)
- **Loss**: -1.0 (checkmated)

### Dense Rewards (Position Features - LEARNED WEIGHTS)
```python
# Extract from ComprehensiveFeatureExtractor
dense_rewards = {
    'material': features['material_balance'],           # Weight learned
    'king_safety': features['king_pawn_shield'],        # Weight learned
    'development': features['development_score'],       # Weight learned
    'center_control': features['center_control'],       # Weight learned
    'pawn_structure': -features['doubled_pawns'],       # Weight learned
    'mobility': features['legal_moves_count'],          # Weight learned
    'tactical': features['piece_tension'],              # Weight learned
    'endgame': features['passed_pawns']                 # Weight learned
}

# Shaper dynamically weights these
shaped_reward = sum(
    dense_rewards[key] * shaper_weights[key]
    for key in dense_rewards
)
```

---

## Preventing Reward Hacking

### Problem: Shaper Exploitation
The shaper might learn to give high rewards for irrelevant features, causing the policy to pursue wrong goals.

### Solution 1: Potential-Based Shaping (Mathematically Safe)
```python
def potential(state):
    """Value estimate of being in this state"""
    return value_network(state)

# Shaped reward (proven to preserve optimal policy)
shaped_reward = (
    actual_reward +                      # Win/loss (ground truth)
    gamma * potential(next_state) -      # Expected future value
    potential(current_state)             # Current value
)
```

This formulation **guarantees** shaped rewards don't change the optimal policy. The policy still converges to winning chess, but learns faster.

### Solution 2: Shaper Validation
```python
# Every 10 generations
validation_games = play_tournament(
    policy_with_shaped_rewards,
    policy_with_stockfish_only,
    games=50
)

if shaped_policy_wins < 45%:
    # Shaper is hurting performance
    rollback_shaper()
    apply_penalty()
```

---

## Implementation Timeline

### v8.0-alpha (Month 1): Proof of Concept
- [ ] Implement `DynamicRewardShaper` network
- [ ] Modify `GameExperience` to store shaper weights
- [ ] Test on 2 generations only
- [ ] Validate weights make sense (opening = development, endgame = promotion)

### v8.0-beta (Month 2): Full Training
- [ ] Train 10 generations with learned shaping
- [ ] Compare vs v7.2 baseline (fixed rewards)
- [ ] Measure: Acceptance rate, win rate, ELO improvement

### v8.0-stable (Month 3): Production Deployment
- [ ] Tournament validation vs v7.2
- [ ] Deploy to Lichess if stronger
- [ ] Document learned reward patterns

---

## Expected Benefits

### Quantitative
- **Faster convergence**: 10-30% fewer games to reach competence
- **Higher acceptance rate**: 60-80% generations accepted (vs v7.2's 40%)
- **Better generalization**: Adapts to opponent styles

### Qualitative
- **Discovers novel evaluation concepts**: May find patterns humans missed
- **Self-tuning**: No manual reward engineering after architecture changes
- **Interpretable**: Can visualize which rewards mattered in famous games

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Reward hacking | Policy optimizes wrong objective | Use potential-based shaping |
| Slower training | Meta-learning overhead | Start with supervised bootstrap |
| Overfitting to self | Weak vs external opponents | Test vs Stockfish regularly |
| Unstable learning | Shaper changes too fast | Use target networks, slow updates |

---

## Research References

### Learned Reward Shaping
1. **Meta-Gradient RL**: [arXiv:1805.09801](https://arxiv.org/abs/1805.09801)
2. **Adversarial Reward Shaping**: [arXiv:2103.09159](https://arxiv.org/abs/2103.09159)
3. **Potential-Based Shaping**: [Ng et al., 1999](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)

### Chess RL
4. **AlphaZero**: [arXiv:1712.01815](https://arxiv.org/abs/1712.01815) - Self-play but fixed rewards
5. **Giraffe**: [arXiv:1509.01549](https://arxiv.org/abs/1509.01549) - Feature-based evaluation

---

## Next Steps

1. **Read Paper**: "Adversarial Reward Shaping for Reinforcement Learning" (arXiv:2103.09159)
2. **Implement Shaper**: Add `DynamicRewardShaper` to `network.py`
3. **Modify Training Loop**: Update `selfplay_trainer.py` to use shaped rewards
4. **Test on 2 Generations**: Validate concept before full 10-gen run
5. **Visualize Weights**: Plot how shaper weights change across game phases

---

## Code Skeleton

```python
# In network.py
class DynamicRewardShaper(nn.Module):
    def __init__(self, feature_dim=55, num_rewards=8):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.weight_head = nn.Linear(64, num_rewards)
    
    def forward(self, features):
        x = torch.relu(self.fc1(features))
        x = torch.relu(self.fc2(x))
        weights = torch.softmax(self.weight_head(x), dim=-1)
        return weights

# In selfplay_trainer.py
def calculate_shaped_reward(self, features, dense_rewards):
    """Use learned reward shaper instead of fixed weights"""
    shaper_weights = self.reward_shaper(features)
    
    shaped_reward = sum(
        dense_rewards[i] * shaper_weights[i]
        for i in range(len(dense_rewards))
    )
    
    return shaped_reward, shaper_weights

# In generational_trainer.py
def train_generation_with_shaping(self):
    """Train policy AND shaper together"""
    
    # Train policy with shaped rewards
    policy_loss = train_policy(shaped_rewards)
    
    # Train shaper based on generation outcome
    if generation_accepted:
        shaper_loss = -log_prob(shaper_weights)  # Reinforce
    else:
        shaper_loss = log_prob(shaper_weights)   # Penalize
    
    optimizer.step([policy_loss, shaper_loss])
```

---

## Success Metrics

**v8.0 is successful if:**
- ✅ Acceptance rate > 60% (vs v7.2's ~40%)
- ✅ Shaper learns intuitive patterns (development in opening, king safety in middlegame)
- ✅ Beats v7.2 in 100-game tournament (>55% win rate)
- ✅ Training completes in <20 hours (not slower than v7.2)

---

*Document created: 2026-06-06*  
*Author: AI Research Team*  
*Status: PROPOSAL - Awaiting v7.2 completion for baseline data*
