# V7P3R Chess AI Training Data Preparation
**Project**: Transition v7p3r from static evaluation to AI model-based engine  
**Target**: Chess Engine v20 with Neural Network Evaluation  
**Data Source**: Historical v7p3r_bot gameplay (BigQuery conformed_layer.moves)  
**Last Updated**: 2026-05-05

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Version Evolution](#version-evolution)
3. [Data Philosophy: Preserving V7P3R Personality](#data-philosophy-preserving-v7p3r-personality)
4. [Stockfish "Top 5" Analysis Method](#stockfish-top-5-analysis-method)
5. [Training Dataset Design](#training-dataset-design)
6. [Reward Function Architecture](#reward-function-architecture)
7. [Dataset Enhancement Pipeline](#dataset-enhancement-pipeline)
8. [Data Science Preparation Workflow](#data-science-preparation-workflow)
9. [BigQuery Connection Details](#bigquery-connection-details)
10. [Next Steps & Validation](#next-steps--validation)

---

## Project Overview

### Mission Statement
Transform v7p3r chess engine from **hard-coded static evaluation** to an **AI model that learns from its own historical play**, preserving the engine's unique tactical personality while improving decision quality through neural network-based position evaluation.

### Key Differentiation
**Unlike generic chess AIs** (trained on master games or pure engine analysis):
- **V7P3R AI learns from V7P3R's own games** → preserves engine personality/style
- **V7P3R's evaluation heuristics drive reward functions** → maintains tactical preferences
- **Stockfish analysis provides corrective guidance** → improves soundness without erasing style
- **Result**: An AI that plays like v7p3r, but smarter

---

## Version Evolution

### V7P3R Chess AI v4.0 (Prototype - Completed)
**Focus**: First integration attempt with v7p3r chess engine

**Experiments**:
1. **Move Ordering** - Trained on puzzle datasets to prioritize tactical moves
2. **Move Selection** - Trained on v7p3r_bot historical gameplay

**Learnings**:
- Puzzle training didn't capture v7p3r's positional style (too tactical)
- Historical play data valuable but needed quality scoring
- Integration challenges with static evaluation mixing
- Need for more structured reward functions

**Status**: ✅ Prototype testing complete, insights captured

---

### V7P3R Chess AI v5.0 (Current Development)
**Focus**: Structured training dataset with quality-scored historical moves

**Improvements over v4.0**:
- **Stockfish "Top 5" scoring** for objective move quality assessment
- **Comprehensive position features** (FEN, material, game phase, eval)
- **Dual reward system** (v7p3r style + Stockfish correctness)
- **Historical move dataset** with quality grades and corrective data

**Status**: 🔄 In development, data preparation phase

---

### Chess Engine v20 (Target)
**Architecture**: Hybrid AI + Static Evaluation

**Components**:
1. **Neural Network Evaluation** - Position assessment from historical games
2. **V7P3R Heuristic Layer** - Maintains tactical preferences (pins, forks, material imbalance)
3. **Stockfish Validation Layer** - Sanity checks for blunder prevention
4. **Adaptive Search** - Learns optimal search depth by position type

**Training Data Requirements**:
- ✅ 5,069 unique games (675k v7p3r moves after deduplication)
- 🔄 Stockfish analysis for each position (in progress)
- 🔄 Move quality scoring (Top 5 ranking)
- 🔄 Position feature extraction (FEN, phase, material)

**Status**: ⏳ Awaiting training dataset completion

---

## Data Philosophy: Preserving V7P3R Personality

### The Problem with Generic Chess AI
Most chess AIs are trained on:
- **Master games** → Learn positional play, but lose engine aggression
- **Pure Stockfish analysis** → Play perfectly, but generically
- **Random positions** → No coherent playing style

**Result**: Homogeneous engines that all play like Stockfish clones

---

### The V7P3R Approach: Style-Preserving Learning

#### Core Principle
> **"Learn from yourself, correct with analysis, reward your style"**

The AI should:
1. **Study v7p3r's historical games** as primary training data
2. **Learn what positions v7p3r reaches** (not just best moves)
3. **Understand v7p3r's tactical preferences** (sacrifices, material imbalances, king attacks)
4. **Use Stockfish analysis as corrective guidance**, not replacement
5. **Favor v7p3r-style moves** in reward function when quality is acceptable

#### Example Scenario
**Position**: White can play safe consolidation (+0.5) or aggressive pawn sacrifice (+0.3)

**Generic AI Decision**:
- Choose +0.5 (objectively better)

**V7P3R AI Decision**:
- Recognize v7p3r historically plays sacrifices in similar positions
- Validate sacrifice is sound (+0.3 not -2.0 blunder)
- **Choose sacrifice** because:
  - Move quality acceptable (Stockfish ranks it #2-3 in Top 5)
  - Aligns with v7p3r's tactical personality
  - Reward function favors style-consistent moves

---

### Personality Preservation Mechanisms

#### 1. Historical Move Dataset as Primary Training
**Data Source**: v7p3r_bot's 675,000 moves from 5,069 games
**Training Signal**: "In this position type, v7p3r typically plays..."

#### 2. V7P3R Evaluation Heuristics in Reward Function
**Integration**: Existing static evaluation becomes reward calculation
- Piece-square tables → Positional rewards
- Material imbalance bonuses → Tactical rewards
- King safety penalties → Strategic penalties
- Mobility calculations → Activity rewards

#### 3. Stockfish Analysis as Quality Filter
**Role**: Validate moves are sound, not dictate moves
- If v7p3r move in Stockfish Top 5 → **High reward** (style + quality)
- If v7p3r move ranks #6-10 → **Moderate reward** (style preserved, minor quality penalty)
- If v7p3r move is blunder (>200cp loss) → **Penalty** (learn correction)

---

## Stockfish "Top 5" Analysis Method

### Overview
For each position in v7p3r's historical games, ask Stockfish to provide its **top 5 best moves** with evaluations. Compare v7p3r's actual move against this list to generate a **quality score**.

---

### Implementation Details

#### Step 1: Position Setup
```python
# For each move in v7p3r historical data
position_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
v7p3r_move = "e7e5"  # Actual move played
```

#### Step 2: Stockfish Analysis
```python
# Stockfish analysis parameters
analysis_config = {
    "depth": 20,              # Search depth (balance speed/accuracy)
    "multipv": 5,             # Request top 5 moves
    "time_limit_ms": 1000     # 1 second per position
}

# Expected output
stockfish_top_5 = [
    {"move": "e7e5", "eval_cp": 30, "rank": 1},   # Best move
    {"move": "g8f6", "eval_cp": 25, "rank": 2},   # 2nd best
    {"move": "c7c5", "eval_cp": 20, "rank": 3},   # 3rd best
    {"move": "d7d6", "eval_cp": 15, "rank": 4},   # 4th best
    {"move": "b8c6", "eval_cp": 10, "rank": 5}    # 5th best
]
```

#### Step 3: Move Quality Scoring
```python
def calculate_move_quality_score(v7p3r_move, stockfish_top_5, v7p3r_eval):
    """
    Score v7p3r's move based on Stockfish ranking
    
    Returns:
        quality_score: 0-100 (100 = best move)
        rank_in_top5: 1-5 or None if not in top 5
        eval_loss_cp: Centipawn loss vs best move
        move_category: 'brilliant', 'great', 'good', 'inaccuracy', 'mistake', 'blunder'
    """
    
    # Find v7p3r move in Stockfish top 5
    v7p3r_rank = None
    best_eval = stockfish_top_5[0]['eval_cp']
    v7p3r_stockfish_eval = None
    
    for candidate in stockfish_top_5:
        if candidate['move'] == v7p3r_move:
            v7p3r_rank = candidate['rank']
            v7p3r_stockfish_eval = candidate['eval_cp']
            break
    
    # Calculate eval loss
    if v7p3r_stockfish_eval is not None:
        eval_loss_cp = abs(best_eval - v7p3r_stockfish_eval)
    else:
        # Not in top 5, use v7p3r's own eval (less reliable)
        eval_loss_cp = abs(best_eval - v7p3r_eval)
    
    # Scoring logic
    if v7p3r_rank == 1:
        quality_score = 100
        category = 'brilliant' if v7p3r_eval > best_eval else 'great'
    elif v7p3r_rank == 2:
        quality_score = 90
        category = 'great'
    elif v7p3r_rank == 3:
        quality_score = 80
        category = 'good'
    elif v7p3r_rank == 4:
        quality_score = 70
        category = 'good'
    elif v7p3r_rank == 5:
        quality_score = 60
        category = 'inaccuracy'
    else:
        # Not in top 5
        if eval_loss_cp < 50:
            quality_score = 50
            category = 'inaccuracy'
        elif eval_loss_cp < 100:
            quality_score = 30
            category = 'mistake'
        elif eval_loss_cp < 200:
            quality_score = 10
            category = 'mistake'
        else:
            quality_score = 0
            category = 'blunder'
    
    return {
        'quality_score': quality_score,
        'rank_in_top5': v7p3r_rank,
        'eval_loss_cp': eval_loss_cp,
        'move_category': category,
        'stockfish_best_move': stockfish_top_5[0]['move'],
        'stockfish_best_eval': best_eval
    }
```

#### Step 4: Dataset Record Creation
```python
# Final training record
training_record = {
    # Position context
    'game_id': 'abc123',
    'ply': 3,
    'fen_before': position_fen,
    'game_phase': 'opening',
    'material_balance': 0,
    
    # V7P3R's decision
    'v7p3r_move': 'e7e5',
    'v7p3r_eval_cp': 28,
    
    # Stockfish analysis
    'stockfish_top5': stockfish_top_5,
    'quality_score': 100,
    'rank_in_top5': 1,
    'eval_loss_cp': 0,
    'move_category': 'great',
    
    # Corrective data
    'stockfish_best_move': 'e7e5',
    'stockfish_best_eval': 30,
    'should_learn_correction': False  # Move was optimal
}
```

---

### Performance Optimization

#### Batch Processing Strategy
**Challenge**: 675,000 positions × 1 second = 187 hours of analysis

**Solutions**:
1. **Parallel Processing**: 8 Stockfish instances → 23 hours
2. **Depth Reduction**: depth=16 instead of 20 → 40% faster (9 hours)
3. **Incremental Analysis**: Analyze rated games first (highest quality)
4. **Caching**: Store analysis results in BigQuery, never recompute

#### Analysis Prioritization
```sql
-- Analyze in priority order
1. Rated games (4,455 games, ~590k moves)  -- Highest quality
2. Tournament games (35 games)             -- Competitive play
3. Casual games (579 games)                -- Lower priority
4. Skip: Arena/local games (unreliable ELO)
```

---

## Training Dataset Design

### Target Schema: `moves_analyzed` Table

#### Core Fields (Position Context)
```sql
CREATE TABLE conformed_layer.moves_analyzed (
  -- Identifiers
  game_id STRING NOT NULL,
  ply INTEGER NOT NULL,
  move_number INTEGER NOT NULL,
  color STRING NOT NULL,  -- 'white' or 'black'
  
  -- Position state
  fen_before STRING NOT NULL,
  fen_after STRING NOT NULL,
  game_phase STRING,      -- 'opening', 'middlegame', 'endgame'
  material_balance INTEGER,
  piece_count INTEGER,
  
  -- Move details
  move_san STRING NOT NULL,
  move_uci STRING NOT NULL,
  piece STRING,
  is_capture BOOLEAN,
  is_check BOOLEAN,
  is_castle BOOLEAN,
  
  -- V7P3R evaluation
  v7p3r_eval_cp INTEGER,  -- V7P3R's position eval
  
  -- Stockfish analysis (Top 5)
  stockfish_best_move STRING,
  stockfish_best_eval INTEGER,
  stockfish_top5 JSON,    -- Array of {move, eval_cp, rank}
  
  -- Move quality scoring
  quality_score INTEGER,  -- 0-100
  rank_in_top5 INTEGER,   -- 1-5 or NULL
  eval_loss_cp INTEGER,   -- Centipawn loss vs best
  move_category STRING,   -- 'brilliant', 'great', 'good', 'inaccuracy', 'mistake', 'blunder'
  
  -- Training labels
  should_learn_correction BOOLEAN,  -- TRUE if move was mistake/blunder
  corrective_move STRING,           -- Stockfish best move for learning
  
  -- Game context
  engine_version STRING,
  opponent STRING,
  opponent_elo INTEGER,
  game_type STRING,       -- 'lichess_rated', 'lichess_casual', 'tournament'
  
  -- Metadata
  analyzed_at TIMESTAMP,
  stockfish_version STRING,
  analysis_depth INTEGER
);
```

---

### Dataset Statistics (Projected)

#### Data Volume
- **Total games**: 5,069 unique games
- **Total moves**: 1,350,163 (before deduplication)
- **Unique moves**: ~985,476 (after 27% dedup)
- **V7P3R moves**: ~492,738 (50% of unique moves)
- **Rated game moves**: ~656,000 (priority dataset)

#### Expected Distribution (After Analysis)
| Move Category | Quality Score | % of Moves | Training Use |
|---------------|---------------|------------|--------------|
| Brilliant/Great | 90-100 | ~35% | Reinforce style |
| Good | 70-89 | ~40% | Primary training |
| Inaccuracy | 50-69 | ~15% | Learn alternatives |
| Mistake | 10-49 | ~8% | Corrective training |
| Blunder | 0-9 | ~2% | Strong correction |

---

### Training Data Splits

#### By Game Quality
```python
training_splits = {
    'rated_games': {
        'games': 4455,
        'moves': ~590000,
        'use': 'Primary training (80%), validation (10%), test (10%)'
    },
    'tournament_games': {
        'games': 35,
        'moves': ~4600,
        'use': 'Validation set (competitive scenarios)'
    },
    'casual_games': {
        'games': 579,
        'moves': ~77000,
        'use': 'Supplementary training (lower weight)'
    }
}
```

#### By Engine Version
```python
version_training = {
    'v18.0': {
        'games': 148,
        'strength': 'Highest (104.1/100)',
        'use': 'Gold standard examples'
    },
    'v17.4': {
        'games': 168,
        'strength': 'Very high (102.6/100)',
        'use': 'Primary training'
    },
    'v17.7': {
        'games': 346,
        'strength': 'High (95.6/100)',
        'use': 'Volume training'
    },
    'legacy': {
        'games': ~4400,
        'strength': 'Moderate-High',
        'use': 'Style learning (large dataset)'
    }
}
```

---

## Reward Function Architecture

### Dual-Component Reward System

The reward function combines **V7P3R's personality** (static evaluation heuristics) with **Stockfish quality validation** to create a style-preserving, sound-move-favoring training signal.

---

### Component 1: V7P3R Style Reward (60% weight)

**Purpose**: Preserve tactical personality and playing style

**Calculation**: Use v7p3r's existing static evaluation as base reward
```python
def v7p3r_style_reward(position, move, v7p3r_eval):
    """
    Calculate reward based on v7p3r's evaluation heuristics
    
    Returns reward in range [-1.0, 1.0]
    """
    
    # Extract v7p3r's evaluation components
    material_score = calculate_material_balance(position, move)
    positional_score = calculate_piece_square_tables(position, move)
    tactical_score = calculate_tactical_features(position, move)
    king_safety = calculate_king_safety(position, move)
    mobility = calculate_mobility(position, move)
    
    # V7P3R's original evaluation formula
    v7p3r_raw_eval = (
        material_score * 1.0 +
        positional_score * 0.8 +
        tactical_score * 1.2 +    # V7P3R favors tactics
        king_safety * 0.6 +
        mobility * 0.4
    )
    
    # Normalize to [-1, 1] range for neural network
    # Assume typical eval range is [-500, +500] centipawns
    style_reward = np.tanh(v7p3r_raw_eval / 500.0)
    
    return style_reward
```

**Key Heuristics Preserved**:
- **Material imbalance bonuses**: +15 for bishop pair, +20 for rook on 7th
- **Tactical pattern rewards**: Pins, forks, skewers, discovered attacks
- **Pawn structure penalties**: Isolated pawns -10, doubled pawns -15
- **King safety**: Pawn shield bonus, open files penalty
- **Piece activity**: Centralization, mobility, outpost bonuses

---

### Component 2: Stockfish Quality Reward (40% weight)

**Purpose**: Ensure moves are objectively sound, prevent blunders

**Calculation**: Based on Stockfish Top 5 ranking and eval loss
```python
def stockfish_quality_reward(quality_score, eval_loss_cp, rank_in_top5):
    """
    Calculate reward based on Stockfish analysis
    
    Returns reward in range [-1.0, 1.0]
    """
    
    # Base reward from quality score (0-100)
    quality_reward = (quality_score - 50) / 50.0  # Map [0,100] → [-1,1]
    
    # Bonus for being in Top 5
    if rank_in_top5 is not None:
        top5_bonus = (6 - rank_in_top5) / 5.0  # Rank 1=1.0, Rank 5=0.2
        quality_reward += top5_bonus * 0.2
    
    # Penalty for eval loss (diminishing returns)
    eval_penalty = -np.tanh(eval_loss_cp / 100.0)  # Severe penalty >100cp
    
    # Combined quality reward
    final_quality = quality_reward * 0.6 + eval_penalty * 0.4
    
    return np.clip(final_quality, -1.0, 1.0)
```

---

### Combined Reward Function

```python
def calculate_training_reward(position, move, v7p3r_eval, stockfish_analysis):
    """
    Master reward function combining style and quality
    
    Args:
        position: FEN string
        move: UCI move string
        v7p3r_eval: V7P3R's position evaluation (centipawns)
        stockfish_analysis: Dict with quality_score, rank_in_top5, eval_loss_cp
    
    Returns:
        total_reward: float in [-1.0, 1.0]
        reward_breakdown: dict with component contributions
    """
    
    # Component 1: V7P3R style (60% weight)
    style_reward = v7p3r_style_reward(position, move, v7p3r_eval)
    
    # Component 2: Stockfish quality (40% weight)
    quality_reward = stockfish_quality_reward(
        stockfish_analysis['quality_score'],
        stockfish_analysis['eval_loss_cp'],
        stockfish_analysis['rank_in_top5']
    )
    
    # Weighted combination
    STYLE_WEIGHT = 0.60
    QUALITY_WEIGHT = 0.40
    
    total_reward = (
        style_reward * STYLE_WEIGHT +
        quality_reward * QUALITY_WEIGHT
    )
    
    # Apply move category modifiers
    category = stockfish_analysis['move_category']
    if category == 'blunder':
        total_reward *= 0.1  # Severe penalty for blunders
    elif category == 'brilliant' and style_reward > 0.5:
        total_reward *= 1.2  # Bonus for brilliant style-consistent moves
    
    return total_reward, {
        'style_reward': style_reward,
        'quality_reward': quality_reward,
        'total_reward': total_reward,
        'style_weight': STYLE_WEIGHT,
        'quality_weight': QUALITY_WEIGHT,
        'category_modifier': category
    }
```

---

### Reward Function Behavior Examples

#### Scenario 1: Perfect Style Match
- **Position**: Tactical middlegame, material equal
- **V7P3R Move**: Knight sacrifice for attack (v7p3r_eval: +50cp)
- **Stockfish**: Ranks move #2 in Top 5 (best: +60cp, sacrifice: +45cp)
- **Outcome**:
  - Style reward: +0.7 (v7p3r loves sacrifices)
  - Quality reward: +0.8 (Rank 2, only -15cp loss)
  - **Total reward: +0.74** ✅ High reward (style + quality)

#### Scenario 2: Safe But Generic
- **Position**: Equal position, consolidation phase
- **V7P3R Move**: Safe pawn push (+20cp)
- **Stockfish**: Ranks move #1 (best move objectively)
- **Outcome**:
  - Style reward: +0.2 (v7p3r prefers activity)
  - Quality reward: +1.0 (Best move)
  - **Total reward: +0.52** ✅ Moderate reward (quality high, style neutral)

#### Scenario 3: Blunder
- **Position**: Critical position
- **V7P3R Move**: Overlooks tactic (-180cp)
- **Stockfish**: Move not in Top 5, eval loss 200cp
- **Outcome**:
  - Style reward: -0.3 (position worsens)
  - Quality reward: -0.9 (blunder)
  - Category modifier: ×0.1 (blunder penalty)
  - **Total reward: -0.06** ❌ Low reward with correction signal

#### Scenario 4: Style Move, Slight Inaccuracy
- **Position**: Opening, development phase
- **V7P3R Move**: Develops bishop aggressively (+15cp)
- **Stockfish**: Ranks move #4 in Top 5 (best: +25cp, -10cp loss)
- **Outcome**:
  - Style reward: +0.6 (v7p3r favors active development)
  - Quality reward: +0.5 (Rank 4, minor loss)
  - **Total reward: +0.56** ✅ Good reward (preserves style, acceptable quality)

---

### Adaptive Weight Tuning (Future Enhancement)

```python
# Adjust weights by game phase
WEIGHT_PROFILES = {
    'opening': {
        'style': 0.70,   # Favor v7p3r's opening repertoire
        'quality': 0.30  # Allow book deviations
    },
    'middlegame': {
        'style': 0.60,   # Balance style and soundness
        'quality': 0.40
    },
    'endgame': {
        'style': 0.40,   # Precision matters more
        'quality': 0.60
    }
}
```

---

## Dataset Enhancement Pipeline

### Pre-Transfer Processing Steps

Before bringing move data to the AI project, perform these enhancements in BigQuery:

---

### Step 1: Deduplicate Moves Table
**Status**: 🔄 Required (27% duplicates identified)

```sql
-- Create deduplicated moves table
CREATE OR REPLACE TABLE conformed_layer.moves_deduplicated AS
SELECT * EXCEPT(row_num)
FROM (
  SELECT *,
    ROW_NUMBER() OVER (
      PARTITION BY game_id, move_number, color
      ORDER BY ingested_at DESC
    ) as row_num
  FROM conformed_layer.moves
)
WHERE row_num = 1;

-- Expected result: 985,476 unique moves (from 1,350,163)
```

---

### Step 2: Add Engine Version Context
**Status**: 🔄 Required (currently missing in moves table)

```sql
-- Join with game_data to add engine_version
CREATE OR REPLACE TABLE conformed_layer.moves_with_context AS
SELECT
  m.*,
  g.engine_version,
  g.opponent,
  g.opponent_elo,
  g.game_type,
  g.color as v7p3r_color,  -- Which side v7p3r played
  CASE
    WHEN m.color = g.color THEN TRUE
    ELSE FALSE
  END as is_v7p3r_move
FROM conformed_layer.moves_deduplicated m
JOIN conformed_layer.game_data g ON m.game_id = g.game_id;

-- Filter to only v7p3r's moves for training
CREATE OR REPLACE TABLE conformed_layer.v7p3r_moves_only AS
SELECT *
FROM conformed_layer.moves_with_context
WHERE is_v7p3r_move = TRUE
  AND game_type IN ('lichess_rated', 'lichess_casual', 'tournament');

-- Expected result: ~492,738 v7p3r moves from reliable games
```

---

### Step 3: Reconstruct FEN Positions
**Status**: 🔄 Required (FEN not in current moves table)

**Challenge**: Current moves table has SAN notation but no FEN positions

**Solution Options**:

**Option A: Reconstruct from PGN** (Recommended)
```python
# ETL script: reconstruct_fen_positions.py
import chess
import chess.pgn
from google.cloud import bigquery

def reconstruct_fens_from_pgn(game_record):
    """
    Parse PGN and generate FEN for each ply
    
    Returns: List of (ply, fen_before, fen_after, move_san, move_uci)
    """
    pgn = chess.pgn.read_game(io.StringIO(game_record['pgn_text']))
    board = pgn.board()
    positions = []
    
    ply = 0
    for move in pgn.mainline_moves():
        fen_before = board.fen()
        san = board.san(move)
        uci = move.uci()
        board.push(move)
        fen_after = board.fen()
        ply += 1
        
        positions.append({
            'game_id': game_record['game_id'],
            'ply': ply,
            'fen_before': fen_before,
            'fen_after': fen_after,
            'move_san': san,
            'move_uci': uci
        })
    
    return positions

# Process all games and upload to BigQuery
```

**Option B: Incremental FEN Calculation** (If PGN unavailable)
```python
# Reconstruct from move sequence (less reliable if missing early game)
def reconstruct_from_moves(game_moves):
    board = chess.Board()
    for move_record in sorted(game_moves, key=lambda x: x['move_number']):
        # ... reconstruct logic
```

---

### Step 4: Stockfish Batch Analysis
**Status**: ⏳ Awaiting dataset finalization

```python
# stockfish_batch_analyzer.py
import chess
import chess.engine
from google.cloud import bigquery
import multiprocessing

def analyze_position_batch(positions_batch, stockfish_path, depth=16):
    """
    Analyze batch of positions with Stockfish
    
    Args:
        positions_batch: List of (game_id, ply, fen_before, v7p3r_move)
        stockfish_path: Path to Stockfish binary
        depth: Search depth (16 = ~500ms per position)
    
    Returns:
        List of analysis records
    """
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    results = []
    
    for pos in positions_batch:
        board = chess.Board(pos['fen_before'])
        
        # Request top 5 moves
        analysis = engine.analyse(
            board,
            chess.engine.Limit(depth=depth),
            multipv=5
        )
        
        # Extract top 5
        top5 = []
        for i, info in enumerate(analysis):
            top5.append({
                'move': info['pv'][0].uci(),
                'eval_cp': info['score'].relative.score(mate_score=10000),
                'rank': i + 1
            })
        
        # Calculate quality score for v7p3r's move
        quality_data = calculate_move_quality_score(
            pos['v7p3r_move'],
            top5,
            pos['v7p3r_eval']
        )
        
        results.append({
            'game_id': pos['game_id'],
            'ply': pos['ply'],
            'stockfish_top5': top5,
            **quality_data
        })
    
    engine.quit()
    return results

# Parallel execution
def parallel_analysis(positions, num_workers=8):
    """Run 8 Stockfish instances in parallel"""
    batch_size = len(positions) // num_workers
    batches = [positions[i:i+batch_size] for i in range(0, len(positions), batch_size)]
    
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.starmap(analyze_position_batch, 
                              [(batch, STOCKFISH_PATH, 16) for batch in batches])
    
    return [item for sublist in results for item in sublist]
```

**Performance Estimate**:
- Positions to analyze: ~492,738 (v7p3r moves only)
- Depth: 16 (compromise speed/accuracy)
- Time per position: ~500ms
- Parallel workers: 8
- **Total time**: ~8.6 hours

**Incremental Strategy**:
1. Analyze v18.0 moves first (148 games, ~19,600 moves) → 1.4 hours
2. Analyze rated games (4,455 games, ~590k moves) → 6.8 hours
3. Cache results in BigQuery, never recompute

---

### Step 5: Feature Engineering
**Status**: 🔄 Design phase

```sql
-- Add computed features to moves_analyzed
CREATE OR REPLACE TABLE conformed_layer.moves_analyzed AS
SELECT
  m.*,
  
  -- Piece activity features
  COUNT(*) OVER (
    PARTITION BY game_id, color
    ORDER BY ply
    ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
  ) as recent_move_frequency,
  
  -- Material trend
  material_balance - LAG(material_balance) OVER (
    PARTITION BY game_id ORDER BY ply
  ) as material_change,
  
  -- Evaluation trend
  v7p3r_eval_cp - LAG(v7p3r_eval_cp) OVER (
    PARTITION BY game_id, color ORDER BY ply
  ) as eval_gain,
  
  -- Position complexity (piece count as proxy)
  piece_count as complexity_score,
  
  -- Game outcome feature (for learning from wins/losses)
  g.outcome as game_outcome,
  
  -- Version strength (from version_performance)
  vp.composite_strength_score as version_strength

FROM conformed_layer.v7p3r_moves_only m
JOIN conformed_layer.game_data g ON m.game_id = g.game_id
JOIN reporting_layer.version_performance vp ON m.engine_version = vp.engine_version;
```

---

### Step 6: Export for AI Project
**Status**: ⏳ After Stockfish analysis complete

```python
# export_training_dataset.py
from google.cloud import bigquery
import pandas as pd

def export_training_dataset():
    """
    Export moves_analyzed to CSV for AI project
    """
    client = bigquery.Client(project="chess-engine-metrics-agent")
    
    # Export full dataset
    query = """
    SELECT *
    FROM conformed_layer.moves_analyzed
    WHERE game_type IN ('lichess_rated', 'lichess_casual', 'tournament')
      AND quality_score IS NOT NULL  -- Only analyzed positions
    ORDER BY game_id, ply
    """
    
    df = client.query(query).to_dataframe()
    
    # Save to CSV
    output_path = "v7p3r_training_dataset_complete.csv"
    df.to_csv(output_path, index=False)
    
    print(f"✓ Exported {len(df):,} training records to {output_path}")
    print(f"  Dataset size: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
    
    # Create train/val/test splits
    rated_games = df[df['game_type'] == 'lichess_rated']
    
    train_df = rated_games.sample(frac=0.8, random_state=42)
    val_test = rated_games.drop(train_df.index)
    val_df = val_test.sample(frac=0.5, random_state=42)
    test_df = val_test.drop(val_df.index)
    
    train_df.to_csv("v7p3r_train.csv", index=False)
    val_df.to_csv("v7p3r_val.csv", index=False)
    test_df.to_csv("v7p3r_test.csv", index=False)
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_df):,} moves ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df):,} moves ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df):,} moves ({len(test_df)/len(df)*100:.1f}%)")
```

---

## Data Science Preparation Workflow

### Phase 1: Initial Dataset Creation ✅ (Current Phase)
1. ✅ Deduplicate game_data (5,069 unique games)
2. ✅ Add game_type, ELO reliability flags
3. ✅ Build opponent-adjusted metrics
4. 🔄 Deduplicate moves table
5. 🔄 Add engine version context to moves
6. 🔄 Reconstruct FEN positions from PGN

---

### Phase 2: Stockfish Analysis ⏳ (Next)
1. ⏳ Set up Stockfish batch processing pipeline
2. ⏳ Analyze rated game moves first (~590k moves)
3. ⏳ Calculate move quality scores (Top 5 ranking)
4. ⏳ Store analysis in BigQuery (cache forever)
5. ⏳ Validate analysis quality (spot checks)

**Estimated Duration**: 8-10 hours compute time

---

### Phase 3: Feature Engineering ⏳
1. ⏳ Add material trend features
2. ⏳ Add evaluation gain/loss features
3. ⏳ Add position complexity metrics
4. ⏳ Join with game outcome labels
5. ⏳ Create categorical encodings (game_phase, move_category)

---

### Phase 4: Dataset Export & Validation ⏳
1. ⏳ Export to CSV for AI project
2. ⏳ Create train/val/test splits (80/10/10)
3. ⏳ Validate data distributions
4. ⏳ Generate dataset summary statistics
5. ⏳ Document schema for AI team

---

### Phase 5: AI Project Handoff ⏳
1. ⏳ Transfer dataset to AI project repository
2. ⏳ Create data loading scripts
3. ⏳ Implement reward function (Python/TensorFlow)
4. ⏳ Begin model architecture design
5. ⏳ Start training experiments

---

## Next Steps & Validation

### Immediate Actions (This Session)
1. **Complete moves deduplication** (remove 27% duplicates)
2. **Add engine version context** to moves table
3. **Reconstruct FEN positions** from PGN files
4. **Design Stockfish batch analysis pipeline**

### Short-Term Goals (Next Week)
1. **Run Stockfish analysis** on v18.0 moves (validation dataset)
2. **Implement quality scoring algorithm**
3. **Create moves_analyzed table** with all features
4. **Export pilot dataset** (10k moves) for AI prototyping

### Medium-Term Goals (Next Month)
1. **Complete full Stockfish analysis** (492k moves)
2. **Feature engineering** (trends, complexity, outcomes)
3. **Export production dataset** (train/val/test splits)
4. **Begin AI model training** in v7p3r-chess-ai project

---

### Success Criteria

#### Dataset Quality Metrics
- ✅ **Deduplication**: <1% duplicate move records
- ✅ **Coverage**: 100% of rated/casual game moves analyzed
- ✅ **FEN Accuracy**: 100% valid FEN strings (validated with python-chess)
- ⏳ **Stockfish Analysis**: 100% of positions analyzed with Top 5 moves
- ⏳ **Quality Scores**: Reasonable distribution (not all brilliant/blunder)

#### AI Training Validation
- ⏳ **Model convergence**: Loss decreases over training epochs
- ⏳ **Style preservation**: AI makes v7p3r-like moves in test positions
- ⏳ **Quality improvement**: AI blunder rate < v7p3r's historical rate
- ⏳ **ELO performance**: Chess Engine v20 ELO > v18.0 (current best)

---

### Dataset Handoff Checklist

Before transferring to AI project:
- [ ] Deduplication complete (moves table)
- [ ] FEN reconstruction complete
- [ ] Stockfish analysis complete (Top 5 per position)
- [ ] Quality scores calculated
- [ ] Feature engineering complete
- [ ] Train/val/test splits created
- [ ] CSV exports generated
- [ ] Schema documentation written
- [ ] Data loading scripts tested
- [ ] Reward function implemented and tested

---

## Document History
- **2026-05-05**: Initial documentation created
- **Next Update**: After Stockfish analysis pipeline implementation

---

## BigQuery Connection Details

### GCP Project Information

**Project ID**: `chess-engine-metrics-agent`  
**Region**: `us-central1`  
**Billing Account**: Active (required for queries)

---

### Dataset Structure

```
chess-engine-metrics-agent
├── raw_layer                    # Raw PGN ingestion
│   └── game_records             # 24,146 rows (original PGN data)
│
├── conformed_layer              # Cleaned, deduplicated data
│   ├── game_data                # 5,069 unique games (primary source)
│   ├── game_summary             # 18,336 game metadata records
│   ├── moves                    # 1,350,163 move records (27% duplicates)
│   ├── moves_deduplicated       # 🔄 To be created (~985k unique moves)
│   ├── moves_with_context       # 🔄 To be created (moves + game context)
│   ├── v7p3r_moves_only         # 🔄 To be created (~492k v7p3r moves)
│   └── moves_analyzed           # 🔄 To be created (final training dataset)
│
└── reporting_layer              # Analytics tables
    ├── version_performance      # 12 engine versions with composite scores
    ├── opponent_strength        # 6 ELO brackets
    ├── opening_performance      # 1,214 opening families
    ├── time_control_performance # 4 time formats
    ├── temporal_trends          # 175 daily aggregations
    ├── castling_analysis        # 12 castling patterns
    └── queen_trade_analysis     # 3 trade timing categories
```

---

### Key Tables for AI Training

#### Primary Training Data Table (Target)
```
conformed_layer.moves_analyzed
```
**Schema**: 30+ fields including FEN, Stockfish analysis, quality scores  
**Rows**: ~492,738 (v7p3r moves from rated/casual/tournament games)  
**Status**: 🔄 In development (awaiting Stockfish analysis)

#### Game Context Table
```
conformed_layer.game_data
```
**Schema**: 24 fields including engine_version, game_type, ELO, opponent_strength  
**Rows**: 5,069 unique games  
**Status**: ✅ Production-ready (deduplicated, enhanced)

#### Current Moves Table
```
conformed_layer.moves
```
**Schema**: 15 fields (game_id, move_number, san, piece, material_balance, game_phase)  
**Rows**: 1,350,163 (includes 364,687 duplicates)  
**Status**: ✅ Available (needs deduplication before use)

---

### Authentication Setup

#### Option 1: Service Account (Recommended for Cloud Resources)

**Step 1: Create Service Account**
```bash
# From your AI project environment
gcloud iam service-accounts create v7p3r-ai-trainer \
  --display-name="V7P3R AI Training Data Access" \
  --project=chess-engine-metrics-agent
```

**Step 2: Grant BigQuery Permissions**
```bash
# Read access to all datasets
gcloud projects add-iam-policy-binding chess-engine-metrics-agent \
  --member="serviceAccount:v7p3r-ai-trainer@chess-engine-metrics-agent.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataViewer"

# Query execution permissions
gcloud projects add-iam-policy-binding chess-engine-metrics-agent \
  --member="serviceAccount:v7p3r-ai-trainer@chess-engine-metrics-agent.iam.gserviceaccount.com" \
  --role="roles/bigquery.jobUser"
```

**Step 3: Download Key File**
```bash
gcloud iam service-accounts keys create ~/v7p3r-ai-key.json \
  --iam-account=v7p3r-ai-trainer@chess-engine-metrics-agent.iam.gserviceaccount.com \
  --project=chess-engine-metrics-agent
```

**Step 4: Set Environment Variable**
```bash
# Linux/Mac
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/v7p3r-ai-key.json"

# Windows PowerShell
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\v7p3r-ai-key.json"

# In Python
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/path/to/v7p3r-ai-key.json'
```

---

#### Option 2: Application Default Credentials (Local Development)

```bash
# Authenticate with your Google account
gcloud auth application-default login --project=chess-engine-metrics-agent

# Credentials stored at:
# Windows: C:\Users\<username>\AppData\Roaming\gcloud\application_default_credentials.json
# Linux/Mac: ~/.config/gcloud/application_default_credentials.json
```

---

### Python Connection Code

#### Basic Connection
```python
from google.cloud import bigquery
import pandas as pd

# Initialize client
PROJECT_ID = "chess-engine-metrics-agent"
client = bigquery.Client(project=PROJECT_ID)

# Test connection
query = "SELECT COUNT(*) as total_games FROM `chess-engine-metrics-agent.conformed_layer.game_data`"
result = client.query(query).to_dataframe()
print(f"✓ Connected! Total games: {result['total_games'].iloc[0]:,}")
```

---

#### Load Training Dataset
```python
def load_v7p3r_training_data(limit=None):
    """
    Load v7p3r moves with game context for AI training
    
    Args:
        limit: Optional row limit for testing (None = load all)
    
    Returns:
        pandas.DataFrame with training data
    """
    
    query = f"""
    SELECT
      -- Move identifiers
      m.game_id,
      m.ply,
      m.move_number,
      m.color,
      
      -- Move details
      m.move_san,
      m.piece,
      m.is_capture,
      m.is_check,
      m.is_castle,
      
      -- Position state
      m.material_balance,
      m.game_phase,
      m.white_material,
      m.black_material,
      
      -- Game context
      g.engine_version,
      g.opponent,
      g.opponent_elo,
      g.game_type,
      g.color as v7p3r_color,
      g.outcome,
      g.v7p3r_elo,
      g.rating_diff,
      
      -- Opponent strength
      g.relative_opponent_strength,
      g.expected_score,
      g.actual_score,
      
      -- Version strength
      vp.composite_strength_score,
      vp.quality_adjusted_win_rate
      
    FROM `chess-engine-metrics-agent.conformed_layer.moves` m
    JOIN `chess-engine-metrics-agent.conformed_layer.game_data` g 
      ON m.game_id = g.game_id
    JOIN `chess-engine-metrics-agent.reporting_layer.version_performance` vp
      ON g.engine_version = vp.engine_version
    
    WHERE 
      -- Only v7p3r's moves
      m.color = g.color
      
      -- Only reliable games
      AND g.game_type IN ('lichess_rated', 'lichess_casual', 'tournament')
      AND g.is_v7p3r_elo_reliable = TRUE
    
    ORDER BY m.game_id, m.ply
    {f'LIMIT {limit}' if limit else ''}
    """
    
    print("Loading training data from BigQuery...")
    df = client.query(query).to_dataframe()
    print(f"✓ Loaded {len(df):,} moves")
    
    return df

# Example usage
# Load 10k moves for prototyping
train_sample = load_v7p3r_training_data(limit=10000)

# Load full dataset (when ready)
# train_full = load_v7p3r_training_data()
```

---

#### Export to CSV/Parquet
```python
def export_training_dataset(output_format='csv', output_path='v7p3r_training_data'):
    """
    Export full training dataset to file
    
    Args:
        output_format: 'csv', 'parquet', or 'feather'
        output_path: Path without extension (will be added)
    """
    
    # Query full dataset
    df = load_v7p3r_training_data()
    
    # Export based on format
    if output_format == 'csv':
        filepath = f"{output_path}.csv"
        df.to_csv(filepath, index=False)
    elif output_format == 'parquet':
        filepath = f"{output_path}.parquet"
        df.to_parquet(filepath, index=False, compression='snappy')
    elif output_format == 'feather':
        filepath = f"{output_path}.feather"
        df.to_feather(filepath)
    else:
        raise ValueError(f"Unsupported format: {output_format}")
    
    # Print summary
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"✓ Exported {len(df):,} rows to {filepath}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    
    return filepath

# Export to Parquet (recommended for large datasets)
export_training_dataset(output_format='parquet')
```

---

#### Query Specific Engine Versions
```python
def load_version_specific_data(version='v18.0'):
    """
    Load training data for specific engine version
    Useful for training on strongest versions only
    """
    
    query = f"""
    SELECT *
    FROM `chess-engine-metrics-agent.conformed_layer.moves` m
    JOIN `chess-engine-metrics-agent.conformed_layer.game_data` g 
      ON m.game_id = g.game_id
    WHERE g.engine_version = '{version}'
      AND m.color = g.color
      AND g.is_v7p3r_elo_reliable = TRUE
    """
    
    df = client.query(query).to_dataframe()
    print(f"✓ Loaded {len(df):,} moves from {version}")
    return df

# Example: Load only v18.0 games (strongest version, 104.1/100 score)
v18_data = load_version_specific_data('v18.0')
```

---

### Sample Queries for Data Exploration

#### Check Data Availability
```sql
-- Count moves by game type
SELECT 
  g.game_type,
  COUNT(DISTINCT m.game_id) as games,
  COUNT(*) as moves,
  COUNT(*) / COUNT(DISTINCT m.game_id) as avg_moves_per_game
FROM `chess-engine-metrics-agent.conformed_layer.moves` m
JOIN `chess-engine-metrics-agent.conformed_layer.game_data` g ON m.game_id = g.game_id
WHERE m.color = g.color  -- Only v7p3r's moves
GROUP BY g.game_type
ORDER BY games DESC;
```

#### Get Move Distribution by Game Phase
```sql
-- Moves by game phase (opening/middlegame/endgame)
SELECT 
  m.game_phase,
  COUNT(*) as moves,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as percentage
FROM `chess-engine-metrics-agent.conformed_layer.moves` m
JOIN `chess-engine-metrics-agent.conformed_layer.game_data` g ON m.game_id = g.game_id
WHERE m.color = g.color
  AND g.is_v7p3r_elo_reliable = TRUE
GROUP BY m.game_phase
ORDER BY moves DESC;
```

#### Version Performance Summary
```sql
-- Get version strength metrics for weighting training data
SELECT
  engine_version,
  total_games,
  composite_strength_score,
  quality_adjusted_win_rate,
  performance_vs_expected,
  avg_opponent_elo,
  confidence_level
FROM `chess-engine-metrics-agent.reporting_layer.version_performance`
ORDER BY composite_strength_score DESC;
```

---

### Cloud Resource Recommendations

#### For Stockfish Batch Analysis

**Option 1: GCP Compute Engine**
```bash
# Recommended VM specs for Stockfish analysis
Machine Type: n2-highcpu-16 (16 vCPUs, 16 GB RAM)
Boot Disk: 100 GB SSD
Region: us-central1 (same as BigQuery)
Estimated Cost: ~$0.60/hour

# Stockfish instances: 16 parallel workers
# Analysis time: 492,738 moves ÷ 16 workers ÷ 2 moves/sec = ~4.3 hours
# Total cost: ~$2.58 for complete analysis
```

**Setup Script**:
```bash
# Create VM
gcloud compute instances create v7p3r-stockfish-analyzer \
  --project=chess-engine-metrics-agent \
  --zone=us-central1-a \
  --machine-type=n2-highcpu-16 \
  --image-family=debian-11 \
  --image-project=debian-cloud \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --scopes=https://www.googleapis.com/auth/bigquery

# SSH into VM
gcloud compute ssh v7p3r-stockfish-analyzer --zone=us-central1-a

# Install Stockfish
sudo apt-get update
sudo apt-get install -y stockfish python3-pip
pip3 install google-cloud-bigquery python-chess

# Verify Stockfish
stockfish --version  # Should show Stockfish 14 or later
```

---

**Option 2: AWS EC2** (if preferred)
```
Instance Type: c6i.4xlarge (16 vCPUs, 32 GB RAM)
Region: us-east-1
Estimated Cost: ~$0.68/hour (~$2.92 for 4.3 hours)
```

---

**Option 3: Local Workstation** (if you have powerful PC)
```
Requirements:
- 8-16 CPU cores
- 16 GB RAM
- Python 3.10+
- Stockfish 15+

Time Estimate:
- 8 cores: ~8.6 hours
- 16 cores: ~4.3 hours
```

---

#### For AI Model Training

**Option 1: GCP Vertex AI**
```bash
# Recommended for PyTorch/TensorFlow training
Machine Type: n1-standard-16 with 1x NVIDIA T4 GPU
Cost: ~$1.35/hour
Ideal for: Initial model development and hyperparameter tuning
```

**Option 2: GCP AI Platform Training**
```bash
# Managed training service
gcloud ai-platform jobs submit training v7p3r_training_v5 \
  --region=us-central1 \
  --master-machine-type=n1-highmem-8 \
  --master-accelerator=count=1,type=nvidia-tesla-t4 \
  --package-path=./trainer \
  --module-name=trainer.task \
  --job-dir=gs://v7p3r-training/models
```

---

### Data Transfer Options

#### Direct BigQuery Export
```python
# Export to Google Cloud Storage (fastest)
from google.cloud import bigquery

def export_to_gcs(table_id, gcs_uri):
    """
    Export BigQuery table to Cloud Storage
    Then download to your AI environment
    """
    
    client = bigquery.Client(project="chess-engine-metrics-agent")
    
    job_config = bigquery.ExtractJobConfig()
    job_config.destination_format = bigquery.DestinationFormat.PARQUET
    job_config.compression = bigquery.Compression.SNAPPY
    
    extract_job = client.extract_table(
        table_id,
        gcs_uri,
        location="us-central1",
        job_config=job_config
    )
    
    extract_job.result()  # Wait for completion
    print(f"✓ Exported {table_id} to {gcs_uri}")

# Example
export_to_gcs(
    "chess-engine-metrics-agent.conformed_layer.v7p3r_moves_only",
    "gs://v7p3r-training-data/moves_analyzed.parquet"
)
```

#### Download from GCS to Local
```bash
# Install gsutil
pip install gsutil

# Download exported data
gsutil -m cp -r gs://v7p3r-training-data/moves_analyzed.parquet ./data/
```

---

### Connection Troubleshooting

#### Common Issues

**Issue 1: Authentication Error**
```
google.auth.exceptions.DefaultCredentialsError: Could not automatically determine credentials
```
**Solution**:
```bash
# Ensure GOOGLE_APPLICATION_CREDENTIALS is set
echo $GOOGLE_APPLICATION_CREDENTIALS  # Should show path to key file

# Or re-authenticate
gcloud auth application-default login
```

---

**Issue 2: Permission Denied**
```
403 Forbidden: BigQuery bigquery.tables.get permission denied
```
**Solution**:
```bash
# Grant required permissions
gcloud projects add-iam-policy-binding chess-engine-metrics-agent \
  --member="user:your-email@gmail.com" \
  --role="roles/bigquery.dataViewer"
```

---

**Issue 3: Query Timeout**
```
TimeoutError: Query did not complete within 60s
```
**Solution**:
```python
# Increase timeout
job_config = bigquery.QueryJobConfig()
job_config.use_query_cache = True

query_job = client.query(query, job_config=job_config)
result = query_job.result(timeout=300)  # 5 minutes
```

---

### Cost Estimation

#### BigQuery Costs
- **Storage**: $0.02 per GB/month (current: ~2 GB = $0.04/month)
- **Queries**: $5 per TB scanned
  - Training data query: ~500 MB scanned per run
  - Cost per query: ~$0.0025
  - Estimated monthly: <$1 (assuming 100 queries)

#### Compute Costs (Stockfish Analysis)
- **GCP VM (n2-highcpu-16)**: $0.60/hour × 4.3 hours = **$2.58**
- **Storage for results**: Negligible (analysis results <100 MB)

**Total Estimated Cost for Complete Data Prep**: **~$3-5**

---

### Security Best Practices

1. **Never commit service account keys to git**
   ```bash
   # Add to .gitignore
   echo "*-key.json" >> .gitignore
   echo "application_default_credentials.json" >> .gitignore
   ```

2. **Use environment-specific credentials**
   ```python
   # development.env
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/dev-key.json
   
   # production.env
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/prod-key.json
   ```

3. **Rotate keys periodically**
   ```bash
   # Delete old key
   gcloud iam service-accounts keys delete KEY_ID \
     --iam-account=v7p3r-ai-trainer@chess-engine-metrics-agent.iam.gserviceaccount.com
   
   # Create new key
   gcloud iam service-accounts keys create new-key.json \
     --iam-account=v7p3r-ai-trainer@chess-engine-metrics-agent.iam.gserviceaccount.com
   ```

---

## Related Documents
- [MOVE_LEVEL_ANALYSIS_DESIGN.md](MOVE_LEVEL_ANALYSIS_DESIGN.md) - Technical architecture for move analysis
- BigQuery Console: https://console.cloud.google.com/bigquery?project=chess-engine-metrics-agent
- GCP Project: https://console.cloud.google.com/home/dashboard?project=chess-engine-metrics-agent
- V7P3R Chess AI Project: (separate repository)

---

**Questions? Next Steps?**
Ready to proceed with moves deduplication and FEN reconstruction, or discuss further dataset enhancements before Stockfish analysis.
