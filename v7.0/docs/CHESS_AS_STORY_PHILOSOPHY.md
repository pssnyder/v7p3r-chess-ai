# "Chess as Story" Training System - v7.0

## Overview

**"Chess as Story"** is a revolutionary training philosophy that treats each game as a narrative arc with distinct phases, each requiring different decision-making priorities. Instead of static evaluation weights throughout the game, the system dynamically adjusts between Stockfish accuracy and personality-driven chaos based on the game's current "chapter."

### Core Philosophy

> "I want to get the engine finding 'decisive' play, not necessarily 'good' play"

Traditional engines optimize for Stockfish-like "perfect" play throughout the entire game. This system recognizes that **different game phases demand different approaches**:

- **Opening**: Learn established theory (high Stockfish weight)
- **Middlegame Chaos**: Maximize complexity and personality (low Stockfish weight)  
- **Endgame Precision**: Convert advantages with accuracy (medium-high Stockfish weight)
- **Tablebase Territory**: Use perfect mathematical knowledge (100% tablebase weight)

## The Mathematical Journey

### Phase Detection

The system automatically detects 6 distinct game phases based on move count and material:

1. **OPENING** (moves 1-10): Establishing position
2. **EARLY_MIDDLEGAME** (moves 11-20): Building complexity
3. **DEEP_MIDDLEGAME** (moves 21-40): Peak chaos
4. **LATE_MIDDLEGAME** (moves 41-60): Refining tactics
5. **ENDGAME** (moves 61+ or ≤13 pieces): Conversion phase
6. **TABLEBASE** (≤7 pieces): Perfect play territory

### Dynamic Weighting System

The training target at each position is calculated as:

```
target = SF_weight * stockfish_eval + Pers_weight * personality_reward + 0.1 * outcome
```

But unlike traditional systems where `SF_weight` is fixed (e.g., 0.7), **it varies smoothly throughout the game**:

#### Stockfish Weight Progression

```
Move 1-10:    0.90 → 0.80  (Linear decrease, learn fundamentals)
Move 11-20:   0.80 → 0.40  (Accelerating decrease, entering chaos)
Move 21-40:   0.30 → 0.10  (Sinusoidal dip to minimum - PEAK CHAOS)
Move 41-60:   0.20 → 0.50  (Recovery, refining position)
Move 61+:     0.50         (Constant, endgame accuracy)
Tablebase:    1.00         (Perfect mathematical knowledge)
```

#### Personality Weight Progression

```
Personality_weight = 1.0 - Stockfish_weight - 0.1

Move 1-10:    0.00 → 0.10  (Minimal personality in opening)
Move 21-40:   0.60 → 0.80  (MAXIMUM PERSONALITY in chaos)
Move 61+:     0.40         (Balanced endgame approach)
Tablebase:    0.00         (No personality needed - math is perfect)
```

### Visual Representation

```
Stockfish Weight (SF) vs Move Number:

1.0 |     Opening
    |   ╱
0.9 | ╱
    |╱             
0.8 |              Early MG
    |             ╱
0.6 |            ╱
    |           ╱
0.4 |          ╱    Deep MG (CHAOS)
    |         ╱    ╱‾‾‾‾‾╲
0.2 |        ╱    ╱       ╲    Late MG
    |       ╱    ╱         ╲  ╱
0.1 |______/____/___________╲╱__________  Endgame
    |    10   20    30   40   50   60   70+
    +-----------------------------------------
                Move Number

Tablebase: 1.0 (whenever ≤7 pieces, regardless of move count)
```

## Key Innovations

### 1. Opening Book Forcing

**Problem**: Self-play games waste training time on repetitive opening moves.

**Solution**: Pre-queue 8-10 moves from a curated opening database to "fast-forward" to interesting middlegame positions.

**Implementation**:
- 12 aggressive, tactical opening lines (Sicilian, King's Indian, Benoni, etc.)
- Randomly selected each game for diversity
- Moves are forced but still tracked for phase detection
- Network doesn't learn these moves (they're predetermined)

**Benefits**:
- Skip repetitive opening theory
- Start training in complex positions
- Accelerate overall training time
- Ensure diverse game types

### 2. Tablebase Integration

**Problem**: Baseline training showed 78% of games reaching max moves without resolution - endgame conversion failure.

**Solution**: Use Syzygy endgame tablebases (≤7 pieces) for perfect evaluation.

**Implementation**:
- Check tablebase availability for each position
- If available, use tablebase eval with 100% weight
- Overrides all other components (Stockfish, personality, outcome)
- Network learns what "forced mate in N" actually looks like

**Benefits**:
- Perfect endgame knowledge
- Dramatically reduce max-move games (expected 78% → 20%)
- Network learns decisive winning technique
- No ambiguity in tablebase territory

### 3. Sinusoidal Weight Transitions

**Problem**: Linear weight transitions create abrupt phase changes.

**Solution**: Use sinusoidal interpolation for smooth, natural transitions.

**Math**:
```python
# Deep middlegame chaos (moves 21-40)
progress = (move_number - 20) / 20.0  # Normalize to [0, 1]
chaos_factor = sin(progress * π)      # Sine wave
stockfish_weight = 0.3 - 0.2 * chaos_factor  # Dips to 0.1 at peak
```

**Benefits**:
- No sudden jumps in evaluation priorities
- Smooth transition between game phases
- Mathematically elegant
- Mirrors natural game flow

## Training Workflow Comparison

### Baseline System (v7_selfplay)

```
Game Start
    ↓
Play move 1 (e4)
    ↓
Stockfish eval: +0.2
Personality reward: +0.05
Target: 0.7 * 0.2 + 0.2 * 0.05 + 0.1 * outcome = 0.15
    ↓
[... 200 moves ...]
    ↓
Max moves reached, draw
```

**Issues**:
- Constant 70% Stockfish weight throughout entire game
- Network never learns aggressive middlegame play
- 78% max-move games (no decisive conversion)
- Training time wasted on repetitive openings

### "Chess as Story" System (v7_story_training)

```
Game Start
    ↓
Opening Book: Apply Sicilian Dragon (10 moves)
    ↓
Position after opening: Rich middlegame
    ↓
Move 25 (deep chaos):
    Stockfish eval: +0.5
    Personality reward: +0.8 (complex sacrifice)
    SF_weight: 0.15 (minimal Stockfish influence)
    Pers_weight: 0.75 (MAXIMUM personality)
    Target: 0.15 * 0.5 + 0.75 * 0.8 + 0.1 * outcome = 0.675
    → Network learns to value complexity!
    ↓
[... aggressive middlegame ...]
    ↓
Move 65 (endgame, 6 pieces left):
    Tablebase eval: +1.0 (forced mate in 12)
    SF_weight: 1.0 (tablebase override)
    Target: 1.0 * 1.0 = 1.0
    → Network learns perfect endgame technique!
    ↓
Checkmate! (not max moves)
```

**Improvements**:
- Dynamic weights match game requirements
- Personality maximized in middlegame
- Perfect tablebase knowledge in endgames
- Opening forcing saves training time
- Expected 20% max-move rate (vs 78%)

## Implementation Architecture

### File Structure

```
v7.0/src/
├── phase_manager.py          # Dynamic weighting system
├── opening_book.py           # Opening forcing
├── tablebase_oracle.py       # Perfect endgame knowledge
├── selfplay_trainer.py       # Updated trainer (MODIFIED)
└── train_story_mode.py       # Main training script
```

### Data Flow

```
1. Game Initialization
   ├── Load opening book
   ├── Initialize tablebase oracle
   └── Create phase manager

2. Game Start
   ├── Select random opening
   └── Force 8-10 opening moves

3. For Each Move (starting from move 11):
   ├── Network selects move
   ├── Push move to board
   ├── Detect game phase (move count + material)
   ├── Extract features (51-dimensional)
   ├── Query Stockfish evaluation
   ├── Calculate personality reward
   ├── Check tablebase availability
   │   ├── If available: Use tablebase eval (weight 1.0)
   │   └── Else: Use Stockfish eval (dynamic weight)
   ├── Calculate phase-aware training target
   └── Store experience with metadata

4. Game End
   ├── Backfill game outcomes
   ├── Recalculate all training targets with final outcome
   ├── Save PGN with opening name
   └── Record phase distribution statistics

5. Training (every 10 games)
   ├── Sample experience buffer
   ├── Train network on phase-aware targets
   └── Save checkpoint with phase statistics
```

### Experience Data Structure

Each position now stores:

```python
{
    'fen': str,
    'features': np.ndarray,  # 51-dim
    'stockfish_eval': float,
    'personality_reward': float,
    'game_outcome': float,
    'move_number': int,
    
    # NEW: Phase-aware fields
    'game_phase': str,           # "DEEP_MIDDLEGAME", etc.
    'training_target': float,    # Pre-calculated with dynamic weights
    'stockfish_weight': float,   # Weight used (e.g., 0.15)
    'personality_weight': float, # Weight used (e.g., 0.75)
    'tablebase_eval': float      # If position was in TB
}
```

### Game Result Enhancements

```python
{
    'game_number': int,
    'result': str,
    'num_moves': int,
    
    # NEW: Story-aware statistics
    'opening_name': str,                    # "Sicilian Dragon"
    'tablebase_positions': int,             # How many TB lookups
    'phase_distribution': {
        'OPENING': 10,
        'EARLY_MIDDLEGAME': 10,
        'DEEP_MIDDLEGAME': 15,
        'LATE_MIDDLEGAME': 12,
        'ENDGAME': 8,
        'TABLEBASE': 5
    }
}
```

## Expected Results

### Performance Metrics

**Baseline (v7_selfplay)**:
- Win Rate: 21% (checkmate)
- Draw Rate: 1%
- Max Moves: 78% (PROBLEM!)
- Forest Darkness: 0.335
- Avg Personality Reward: 0.34
- Total Positions: 17,917
- Avg Stockfish Weight: 0.7 (constant)

**Expected (v7_story_training)**:
- Win Rate: 50-60% (checkmate + resignation)
- Draw Rate: 5-10%
- Max Moves: 15-25% (MAJOR IMPROVEMENT)
- Forest Darkness: 0.40-0.45 (more aggressive)
- Avg Personality Reward: 0.50+ (higher variance by phase)
- Total Positions: ~12,000 (fewer due to opening forcing)
- Avg Stockfish Weight: ~0.35 (heavily weighted toward chaos)

### Quality Improvements

1. **Decisive Middlegame Play**
   - Higher tactical complexity
   - More sacrifices and attacks
   - Less "safe" play, more forcing moves
   - Stockfish might think it's bad, but it's **exciting**

2. **Perfect Endgame Conversion**
   - Every tablebase position resolved correctly
   - Network learns "forced mate in N" patterns
   - No more drawing winning endgames
   - 78% max-move rate drops dramatically

3. **Training Efficiency**
   - Opening forcing saves ~500 positions per 100 games
   - Faster convergence (diverse starting positions)
   - Less overfitting on opening theory
   - More training on critical phases

## Usage

### Basic Training Run

```bash
cd v7.0/src
python train_story_mode.py
```

### Custom Configuration

```python
trainer = SelfPlayTrainer(
    profile_path="../profiles/dark_forest_assassin.json",
    stockfish_path="path/to/stockfish.exe",
    output_dir="../training/my_experiment",
    
    # Story-specific parameters
    opening_book_pgn=None,              # Use default aggressive openings
    tablebase_path="path/to/syzygy",    # Optional but recommended
    use_opening_book=True,              # Enable fast-forward
    use_tablebases=True                 # Enable perfect endgames
)

trainer.train_from_selfplay(
    num_games=100,
    batch_size=256,
    train_every_n_games=10
)
```

### Analyzing Results

```bash
# Review training report
cat ../training/v7_story_training/training_report.json

# Check phase statistics per game
cat ../training/v7_story_training/stats_game_0100.json

# Compare to baseline
python analyze_story_vs_baseline.py
```

## Troubleshooting

### Tablebases Not Loading

**Symptom**: `[WARN] Tablebases requested but not available`

**Solution**:
1. Download Syzygy 3-4-5 piece tablebases (~1GB) from https://syzygy-tables.info/
2. Extract to a directory (e.g., `E:/Chess/Tablebases/syzygy`)
3. Update `TABLEBASE_PATH` in `train_story_mode.py`
4. Verify files exist: should see `*.rtbw` and `*.rtbz` files

### Opening Book Not Working

**Symptom**: Games start from initial position despite opening book enabled

**Solution**:
- Check console output for `[OK] Opening book loaded: 12 lines`
- If you provided a PGN path, verify it exists
- Default openings should always work (no file needed)

### High Max-Move Rate

**Symptom**: Still seeing 70%+ max-move games

**Causes**:
1. Tablebases not actually enabled (check logs)
2. Games not reaching tablebase territory (too complex)
3. Network hasn't learned endgame patterns yet (train longer)

**Solutions**:
- Verify tablebase oracle is active
- Increase endgame Stockfish weight if needed
- Train for more games (100 may not be enough)

## Future Enhancements

### Advanced Phase Detection

- Detect tactical positions (pins, forks, skewers)
- Adjust weights for materially unbalanced positions
- Recognize fortress positions (draw despite material advantage)

### Multi-Personality Training

- Train multiple personalities simultaneously
- Let personalities "vote" on move selection
- Ensemble approach for robustness

### Adaptive Weight Learning

- Let network learn its own phase weights
- Meta-learning optimization
- Self-adjusting personality balance

### Extended Tablebase Support

- 7-piece tablebases (~140GB, very slow)
- Cloud-based tablebase API (Lichess API)
- Tablebase-guided search (not just evaluation)

## Philosophy Summary

> "Stockfish eval starts near 90% impact, for the first 4-5 moves, then scales back to 50% into the middlegame, then once 8-10 moves in we start pushing fully into the personality weighted components and tactics, 10% stockfish, absolute chaos, then we progressively bring back in the stockfish weights and as soon as we hit a piece count that would be tablebase territory, we go 100% stockfish eval and let math solve what has already been solved."

This isn't just a training system - it's a **narrative framework** for chess decision-making. Each game tells a story:

1. **Prologue (Opening)**: Learn the established script
2. **Rising Action (Early Middlegame)**: Building tension
3. **Climax (Deep Middlegame)**: Maximum chaos and creativity
4. **Falling Action (Late Middlegame)**: Refining the position
5. **Resolution (Endgame)**: Converting with precision
6. **Epilogue (Tablebase)**: Mathematical certainty

The result: An engine that finds **decisive, interesting, personality-driven play** while maintaining the ability to convert advantages and finish games cleanly.

## Credits

Created by user request: "i.e. we need to analyze what the story of our game needs to be and tune the heuristics to weight differently throughout the game"

Implementation: December 2024

Philosophy: Chess is not just calculation - it's a story with distinct chapters, each demanding different approaches.
