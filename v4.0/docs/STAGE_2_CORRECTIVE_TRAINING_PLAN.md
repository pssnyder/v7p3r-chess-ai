# Stage 2: Historical Self-Correction Training Plan

## Vision
Learn from V7P3R's actual game failures to create a feedback loop that corrects biases and strengthens weaknesses discovered in real play. The "trap learning" approach teaches both avoidance (when we're the victim) and exploitation (when opponent falls into similar patterns).

## Data Sources
- **51 PGN files** from Lichess V7P3R Bot (Nov 2025 - Apr 2026)
- Located: `E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\game_records\Lichess V7P3R Bot\`
- Focus: Games lost by **checkmate** or **resignation** only

## Corrective Training Philosophy

### Dual Learning Pattern
For every losing position, create TWO training examples:

#### 1. Negative Example (Avoidance)
- **Position**: V7P3R's perspective before critical blunder
- **Bad Move**: What V7P3R actually played
- **Good Moves**: Stockfish top-10 alternatives
- **Loss Function**: High penalty for repeating historical mistakes

#### 2. Positive Example (Exploitation)  
- **Position**: Same board state, **inverted colors** (opponent's perspective)
- **Good Move**: What opponent played to punish V7P3R
- **Training Goal**: Learn to recognize and exploit when opponent is in similar trap
- **Analogy**: "I got fooled once, now I can fool others"

### Position Selection Criteria

#### Critical Positions (High Priority)
1. **Final Blunders**: Last 3-5 moves before checkmate
2. **Resignation Triggers**: Position where V7P3R resigned (material ≥10 points down)
3. **Tactical Failures**: Positions with sharp material swings (≥3 points lost in 1-2 moves)

#### Context Positions (Medium Priority)  
4. **Opening Mistakes**: First 15 moves if game ended quickly (<30 total moves)
5. **Middlegame Weaknesses**: Moves 16-40 where evaluation dropped ≥2 pawns

#### Exclusions
- Time forfeit games (no chess lesson)
- Draws (neutral outcomes)
- Wins (save for Stage 4 reinforcement learning)

## Data Pipeline Architecture

### Step 1: Game Parsing & Filtering
**Script**: `scripts/parse_historical_games.py`

```python
def filter_losing_games(pgn_files: List[str]) -> List[Game]:
    """
    Extract only games where V7P3R lost by checkmate or resignation.
    
    Returns:
        List of Game objects with metadata:
        - game_id, date, opponent
        - result (0-1 or 1-0 depending on color)
        - termination (Normal=checkmate, resignation)
        - time_control, V7P3R_color
    """
```

**Output**: `data/stage2_games/v7p3r_losses.json`
- Expected: 100-300 losing games (depends on bot's win rate)

### Step 2: Critical Position Extraction
**Script**: `scripts/extract_critical_positions.py`

```python
def extract_critical_positions(game: Game) -> List[CriticalPosition]:
    """
    For each losing game, identify positions to learn from.
    
    Returns:
        List of CriticalPosition objects:
        - fen: Board state
        - move_number: Ply in game
        - v7p3r_move: What V7P3R actually played
        - evaluation_before: Stockfish eval before move
        - evaluation_after: Stockfish eval after move
        - eval_drop: Material/positional loss
        - context: "final_blunder", "resignation_trigger", "tactical_failure"
    """
```

**Output**: `data/stage2_positions/critical_positions.json`
- Expected: 500-2000 positions (5-10 per losing game)

### Step 3: Dual Dataset Creation
**Script**: `scripts/create_corrective_dataset.py`

```python
def create_dual_training_examples(position: CriticalPosition) -> Tuple[NegativeExample, PositiveExample]:
    """
    Create paired training examples from V7P3R's losing position.
    
    Negative Example (Avoidance):
    - position_features: ChessStateExtractor(fen, v7p3r_perspective)
    - bad_move: v7p3r_move (encoded)
    - good_moves: Stockfish top-10 (with scores)
    - correction_weight: High (1.5-2.0x) for critical blunders
    
    Positive Example (Exploitation):
    - position_features: ChessStateExtractor(fen_inverted, opponent_perspective)
    - good_move: opponent_winning_move (encoded)
    - alternative_moves: Stockfish top-10 from opponent's view
    - exploitation_weight: Medium (1.0-1.5x)
    """
```

**Output**: `data/stage2_training/corrective_dataset.json`
- Expected: 1000-4000 training examples (2x positions)

### Step 4: Stockfish Analysis Enrichment
**Script**: `scripts/enrich_with_stockfish_corrections.py`

```python
def enrich_position(fen: str, v7p3r_move: str) -> EnrichedPosition:
    """
    Run Stockfish analysis on critical positions.
    
    Analysis (0.5s per position):
    - Top-10 moves with evaluations (multipv=10)
    - Eval before V7P3R's move
    - Eval after V7P3R's move
    - Best continuation (3-5 moves deep)
    
    Returns:
        EnrichedPosition with:
        - position_features (690-dim)
        - bad_move_encoded, good_moves_encoded
        - move_evaluations, correction_importance
    """
```

## Training Architecture

### Model Extensions

#### Option A: Fine-tune Stage 1 Model (Recommended)
Continue training `best_model.pt` from Stage 1 with corrective loss:

```python
total_loss = ranking_loss + theme_loss + correction_loss

correction_loss = (
    negative_weight * loss(v7p3r_move, target=0.0) +  # Penalize bad moves
    positive_weight * loss(good_moves, target=1.0)     # Reward alternatives
)
```

**Pros**: Preserves 86.6% puzzle accuracy, adds failure knowledge
**Cons**: Risk of catastrophic forgetting if not balanced

#### Option B: Separate Corrector Network
Train dedicated `v7p3r-corrector` agent that vetos moves:

```python
class MoveCorrector(nn.Module):
    def __init__(self, base_model):
        self.position_encoder = base_model.position_encoder  # Frozen
        self.correction_head = CorrectionHead()  # New trainable
        
    def forward(self, position, candidate_move):
        return danger_score  # 0.0-1.0 (1.0 = historical failure pattern)
```

**Pros**: No interference with Stage 1 weights, explicit veto mechanism  
**Cons**: Adds inference latency

### Training Configuration

```yaml
# config/stage2_corrective_training.json
{
  "model_approach": "fine_tune",  # or "separate_corrector"
  "dataset_path": "data/stage2_training/corrective_dataset.json",
  "base_model": "models/stage1_themes/best_model.pt",
  "training": {
    "batch_size": 32,
    "num_epochs": 50,
    "learning_rate": 0.00005,  # Lower than Stage 1 (fine-tuning)
    "correction_weight": 2.0,   # High penalty for historical failures
    "exploitation_weight": 1.5,
    "early_stopping_patience": 10
  },
  "validation": {
    "test_on_historical_failures": true,
    "holdout_games": 20,  # Reserve recent games for testing
    "target_correction_rate": 0.85  # 85% avoid historical blunders
  }
}
```

## Validation & Testing

### Test 1: Historical Failure Replay
- **Setup**: Load 20 held-out losing games
- **Test**: Present critical positions to corrected model
- **Success**: Model chooses Stockfish top-5 move ≥85% (vs V7P3R's blunder)

### Test 2: Trap Recognition (Inverted)
- **Setup**: Present opponent-perspective positions from losing games
- **Test**: Model chooses winning exploitation move
- **Success**: ≥75% accuracy recognizing advantage patterns

### Test 3: Regression Prevention
- **Setup**: Re-run Stage 1 puzzle validation (10K puzzles)
- **Test**: Ensure puzzle accuracy doesn't drop
- **Success**: Maintain ≥84% top-5 accuracy (allow 2-3% drop for correction gain)

### Test 4: Live Game Validation (Gold Standard)
- **Setup**: Play 50 games with corrected model vs baseline V7P3R v18.4
- **Test**: Measure win rate and blunder reduction
- **Success**: 
  - Win rate ≥55% vs baseline
  - Blunders/game <5.0 (baseline: 5.8)
  - No new failure patterns emerge

## Expected Performance Gains

### Conservative Estimates
- **Historical Blunder Avoidance**: 70-80% (was 0% before training)
- **Trap Exploitation**: 60-70% (new capability)
- **Overall ELO Gain**: +50-100 points (avoiding critical blunders)
- **Training Time**: 6-12 hours (depends on dataset size)

### Optimistic Estimates
- **Historical Blunder Avoidance**: 85-90%
- **Trap Exploitation**: 75-85%
- **Overall ELO Gain**: +100-150 points
- **Reduced Resignation Rate**: 30-40% fewer hopeless positions

## Implementation Timeline

### Phase 1: Data Pipeline (Current Priority)
- [ ] Parse 51 PGN files → filter losing games
- [ ] Extract critical positions (500-2000)
- [ ] Create dual training examples (1000-4000)
- [ ] Enrich with Stockfish corrections
- **Est. Time**: 3-6 hours (mostly Stockfish analysis)

### Phase 2: Model Training
- [ ] Choose approach (fine-tune vs separate corrector)
- [ ] Implement correction loss function
- [ ] Train for 50 epochs
- [ ] Run validation tests
- **Est. Time**: 6-12 hours training + 2 hours validation

### Phase 3: Integration & Testing
- [ ] Integrate corrected model into V7P3R engine
- [ ] Run 50-game tournament vs baseline
- [ ] Analyze failure patterns
- [ ] Document improvements
- **Est. Time**: 1-2 days (mostly game play)

## Next Steps

1. **Create `scripts/parse_historical_games.py`**: Filter losing games from 51 PGNs
2. **Create `scripts/extract_critical_positions.py`**: Identify blunder positions
3. **Create `scripts/create_corrective_dataset.py`**: Build dual training examples
4. **Create `scripts/enrich_with_stockfish_corrections.py`**: Stockfish analysis
5. **Create `scripts/train_corrector.py`**: Fine-tune Stage 1 model with correction loss

## Success Criteria

Stage 2 is **COMPLETE** when:
- ✅ Corrective dataset created (≥1000 dual examples)
- ✅ Model trained (correction loss converged)
- ✅ Historical failure test ≥85% avoidance
- ✅ Stage 1 regression test ≥84% maintained
- ✅ Live game validation +50 ELO minimum

---

**Key Insight**: This stage transforms V7P3R from a "puzzle solver" to a "failure learner" - the most valuable knowledge comes from understanding what went wrong in real games, not just what's right in puzzles.
