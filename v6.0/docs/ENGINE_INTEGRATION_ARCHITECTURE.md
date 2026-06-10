# V7P3R v20 Engine Integration Architecture
## Combining Stage 1 + Stage 2 + Pre-calculation System

**Created**: 2026-05-31  
**Status**: 📋 **DESIGN PHASE**  
**Target Version**: V7P3R v20.0 (AI-powered engine)  

---

## Executive Summary

The V7P3R v20 engine is a **hybrid AI/traditional chess engine** that combines:
1. **Stage 1 AI**: Position evaluator (GOOD vs BAD binary classifier)
2. **Stage 2 AI**: Complexity & time manager (tactical prioritization)
3. **Pre-calculation Queue**: Transposition-like position caching for instant recall
4. **Static Checkmate Calculator**: Traditional minimax for forced mates
5. **UCI Interface**: Standard chess protocol for tournament play

**Philosophy**: "Play a completely different game while opponent is still playing old fashioned depth-seeking chess."

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     UCI INTERFACE (Input)                       │
│  - position fen ...                                              │
│  - go wtime 180000 btime 195000 winc 4000 binc 4000             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   PRE-CALCULATION CACHE CHECK                    │
│  "Have we already evaluated this position during queue time?"   │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Cache Hit? → Load pre-calculated Stage 1 evaluations   │   │
│  │  Cache Miss? → Proceed to Stage 1 calculation            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   PARALLEL PROCESSING (3 tasks)                  │
│                                                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│  │  STAGE 1 AI      │  │  STAGE 2 AI      │  │  CHECKMATE   │ │
│  │  (Position Eval) │  │  (Time Manager)  │  │  SEARCH      │ │
│  │                  │  │                  │  │  (Static)    │ │
│  │  Input: FEN      │  │  Input: Stage 1  │  │  Input: FEN  │ │
│  │  Output: Good    │  │  + Time state    │  │  Output:     │ │
│  │  moves + probs   │  │  Output:         │  │  Mate move   │ │
│  │                  │  │  - Complexity    │  │  (if exists) │ │
│  │                  │  │  - Time alloc    │  │              │ │
│  │                  │  │  - Move priority │  │              │ │
│  └──────────────────┘  └──────────────────┘  └──────────────┘ │
│         ↓                      ↓                      ↓         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      DECISION INTEGRATION                        │
│                                                                   │
│  Priority 1: Checkmate move found? → PLAY IT (override AI)      │
│  Priority 2: Stage 2 confidence high? → Use AI recommendation   │
│  Priority 3: Stage 2 confidence low? → Fall back to static eval │
│                                                                   │
│  Time Management: Enforce Stage 2 time allocation               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   MOVE SELECTION & EXECUTION                     │
│                                                                   │
│  Selected Move: e2e4 (example)                                  │
│  Time Spent: 8.3 seconds                                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  PRE-CALCULATION QUEUE (Housekeeping)            │
│  "Opponent is thinking - use this time to pre-calculate"        │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  1. Generate opponent's likely responses (top 5 moves)   │   │
│  │  2. For each response, generate OUR next moves           │   │
│  │  3. Run Stage 1 on all candidate positions                │   │
│  │  4. Cache results in Pre-calculation Queue                │   │
│  │  5. If opponent plays into cache → instant recall         │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     UCI INTERFACE (Output)                       │
│  - info depth 1 score cp 50 pv e2e4                             │
│  - bestmove e2e4                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Integration Details

### 1. Pre-Calculation Cache System

**Purpose**: Store pre-calculated Stage 1 evaluations for positions likely to occur

**Data Structure**: SQLite database with position hash lookup

```python
# Schema
CREATE TABLE precalc_cache (
    position_hash TEXT PRIMARY KEY,  -- Zobrist hash or FEN hash
    fen TEXT NOT NULL,
    last_accessed INTEGER,           -- Unix timestamp
    
    -- Stage 1 outputs (JSON)
    good_moves JSON,                 -- List of moves with prob_good
    bad_moves JSON,                  -- List of moves with prob_bad
    
    -- Metadata
    legal_moves_count INTEGER,
    created_at INTEGER,
    access_count INTEGER DEFAULT 1
);

CREATE INDEX idx_last_accessed ON precalc_cache(last_accessed);
```

**Cache Population Strategy**:

```python
def populate_cache_during_opponent_thinking(
    current_position: Board,
    opponent_color: Color,
    max_depth: int = 2,
    top_n_moves: int = 5
):
    """
    Pre-calculate positions during opponent's thinking time.
    
    Process:
    1. Generate opponent's top 5 likely moves
    2. For each opponent move, generate our top 5 responses
    3. Run Stage 1 on all resulting positions (5 * 5 = 25 max)
    4. Cache results for instant recall
    
    Args:
        current_position: Current board state
        opponent_color: Color of opponent (WHITE or BLACK)
        max_depth: How deep to pre-calculate (1 or 2 ply)
        top_n_moves: How many moves to consider per ply
    """
    cache_entries = []
    
    # Get opponent's legal moves
    opponent_moves = list(current_position.legal_moves)
    
    # Heuristic ranking (captures, checks, central moves)
    opponent_moves_ranked = rank_moves_heuristic(
        current_position, 
        opponent_moves,
        top_n=top_n_moves
    )
    
    for opp_move in opponent_moves_ranked:
        # Play opponent's move
        current_position.push(opp_move)
        
        # Generate our responses
        our_moves = list(current_position.legal_moves)
        
        # Evaluate position with Stage 1
        stage1_results = evaluate_all_moves_stage1(current_position, our_moves)
        
        # Cache this position
        cache_entry = {
            'position_hash': hash_position(current_position),
            'fen': current_position.fen(),
            'good_moves': [m for m in stage1_results if m['prob_good'] >= 0.5],
            'bad_moves': [m for m in stage1_results if m['prob_good'] < 0.5],
            'legal_moves_count': len(our_moves),
            'created_at': int(time.time()),
        }
        cache_entries.append(cache_entry)
        
        # Optionally go 1 more ply deeper
        if max_depth >= 2:
            our_moves_ranked = rank_moves_heuristic(
                current_position, 
                our_moves,
                top_n=top_n_moves
            )
            
            for our_move in our_moves_ranked:
                current_position.push(our_move)
                
                # Opponent's next responses
                opp_moves_2 = list(current_position.legal_moves)
                stage1_results_2 = evaluate_all_moves_stage1(current_position, opp_moves_2)
                
                cache_entry_2 = {
                    'position_hash': hash_position(current_position),
                    'fen': current_position.fen(),
                    'good_moves': [m for m in stage1_results_2 if m['prob_good'] >= 0.5],
                    'bad_moves': [m for m in stage1_results_2 if m['prob_good'] < 0.5],
                    'legal_moves_count': len(opp_moves_2),
                    'created_at': int(time.time()),
                }
                cache_entries.append(cache_entry_2)
                
                current_position.pop()
        
        current_position.pop()
    
    # Batch insert to database
    save_to_cache(cache_entries)
    
    return len(cache_entries)
```

**Cache Hit Strategy**:

```python
def check_cache(position: Board) -> Optional[dict]:
    """
    Check if position is in pre-calculation cache.
    
    Returns:
        Cache entry with Stage 1 results if found, else None
    """
    position_hash = hash_position(position)
    
    # Query database
    cache_entry = db.query(
        "SELECT * FROM precalc_cache WHERE position_hash = ?",
        (position_hash,)
    )
    
    if cache_entry:
        # Update access count and timestamp
        db.execute(
            "UPDATE precalc_cache SET access_count = access_count + 1, "
            "last_accessed = ? WHERE position_hash = ?",
            (int(time.time()), position_hash)
        )
        
        return {
            'fen': cache_entry['fen'],
            'good_moves': json.loads(cache_entry['good_moves']),
            'bad_moves': json.loads(cache_entry['bad_moves']),
            'cache_hit': True,
        }
    
    return None
```

**Cache Eviction**: LRU (Least Recently Used)
- Keep max 100,000 positions
- Evict oldest accessed positions when full

---

### 2. Stage 1 + Stage 2 Integration

**Data Flow**:

```python
def make_move_decision(position: Board, time_control: dict) -> Move:
    """
    Complete decision-making pipeline.
    
    Args:
        position: Current chess position
        time_control: Dict with wtime, btime, winc, binc
        
    Returns:
        Best move to play
    """
    # Step 1: Check pre-calculation cache
    cache_result = check_cache(position)
    
    if cache_result:
        print("info string Cache hit! Using pre-calculated Stage 1 results")
        good_moves = cache_result['good_moves']
        skip_stage1 = True
    else:
        print("info string Cache miss - running Stage 1 evaluation")
        skip_stage1 = False
    
    # Step 2: Parallel execution of 3 tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Task 1: Stage 1 (if not cached)
        if skip_stage1:
            future_stage1 = None
        else:
            future_stage1 = executor.submit(run_stage1, position)
        
        # Task 2: Static checkmate search
        future_checkmate = executor.submit(static_checkmate_search, position, max_depth=5)
        
        # Task 3: Stage 2 (runs after Stage 1 if needed)
        # Will wait for Stage 1 to complete before starting
        
        # Wait for results
        checkmate_move = future_checkmate.result()
        
        if checkmate_move:
            print(f"info string CHECKMATE FOUND! Playing {checkmate_move}")
            return checkmate_move
        
        if skip_stage1:
            stage1_good_moves = good_moves
        else:
            stage1_results = future_stage1.result()
            stage1_good_moves = [m for m in stage1_results if m['prob_good'] >= 0.5]
        
        # Now run Stage 2 with Stage 1 results
        future_stage2 = executor.submit(
            run_stage2,
            position,
            stage1_good_moves,
            time_control
        )
        
        stage2_results = future_stage2.result()
    
    # Step 3: Decision integration
    final_move = integrate_decisions(
        position,
        stage1_good_moves,
        stage2_results,
        time_control
    )
    
    return final_move


def integrate_decisions(
    position: Board,
    stage1_good_moves: list,
    stage2_results: dict,
    time_control: dict
) -> Move:
    """
    Combine Stage 1 + Stage 2 outputs into final move selection.
    
    Args:
        position: Current position
        stage1_good_moves: List of moves with prob_good >= 0.5
        stage2_results: Dict with complexity, time_allocation, move_priority
        time_control: Time control info
        
    Returns:
        Best move
    """
    # Extract Stage 2 outputs
    complexity = stage2_results['complexity_score']
    time_allocation = stage2_results['time_allocation']
    confidence = stage2_results['confidence_level']
    move_priorities = stage2_results['move_priority_distribution']
    
    # Check confidence level
    if confidence < 0.5:
        print(f"info string Low confidence ({confidence:.2f}) - falling back to static eval")
        # Fall back to traditional engine evaluation
        return fallback_static_engine(position)
    
    # Rank moves by Stage 2 priority
    ranked_moves = sorted(
        zip(stage1_good_moves, move_priorities),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Select top move
    best_move, best_priority = ranked_moves[0]
    
    # Calculate time to spend
    side_to_move = position.turn
    time_remaining = time_control['wtime'] if side_to_move else time_control['btime']
    increment = time_control['winc'] if side_to_move else time_control['binc']
    
    time_budget = (time_remaining / 1000.0) + (increment / 1000.0 * 0.8)
    actual_time = time_budget * time_allocation
    
    print(f"info string Complexity: {complexity:.1f}/10, Time: {actual_time:.1f}s, Confidence: {confidence:.2%}")
    print(f"info string Selected move: {best_move['move']} (priority: {best_priority:.1f}/10)")
    
    return chess.Move.from_uci(best_move['move'])
```

---

### 3. Housekeeping Task Scheduler

**Purpose**: Use idle time during opponent's thinking for pre-calculation

```python
class HousekeepingScheduler:
    def __init__(self):
        self.tasks = []
        self.current_task = None
        self.stop_flag = False
    
    def start_opponent_thinking(self, position: Board, opponent_color: Color):
        """
        Start housekeeping tasks when opponent starts thinking.
        
        Tasks:
        1. Pre-calculate next 2 plies of positions
        2. Evict old cache entries (LRU)
        3. Log metrics for training data collection
        """
        self.stop_flag = False
        
        # Task 1: Pre-calculation (high priority)
        self.tasks.append({
            'type': 'precalculation',
            'position': position.copy(),
            'opponent_color': opponent_color,
            'priority': 1,
        })
        
        # Task 2: Cache maintenance (low priority)
        self.tasks.append({
            'type': 'cache_eviction',
            'priority': 3,
        })
        
        # Task 3: Training data logging (low priority)
        self.tasks.append({
            'type': 'log_metrics',
            'position': position.copy(),
            'priority': 2,
        })
        
        # Execute tasks in priority order
        self._execute_tasks()
    
    def stop_opponent_thinking(self):
        """Stop housekeeping when opponent makes move."""
        self.stop_flag = True
    
    def _execute_tasks(self):
        """Execute tasks until opponent moves or all tasks complete."""
        # Sort by priority
        self.tasks.sort(key=lambda x: x['priority'])
        
        for task in self.tasks:
            if self.stop_flag:
                print("info string Housekeeping interrupted (opponent moved)")
                break
            
            if task['type'] == 'precalculation':
                print("info string Housekeeping: Pre-calculating positions")
                num_cached = populate_cache_during_opponent_thinking(
                    task['position'],
                    task['opponent_color'],
                    max_depth=2,
                    top_n_moves=5
                )
                print(f"info string Cached {num_cached} positions")
            
            elif task['type'] == 'cache_eviction':
                print("info string Housekeeping: Evicting old cache entries")
                evict_lru_cache(max_entries=100000)
            
            elif task['type'] == 'log_metrics':
                print("info string Housekeeping: Logging training data")
                log_position_for_training(task['position'])
        
        self.tasks.clear()
```

**UCI Integration**:

```python
# In UCI loop
if command.startswith("position"):
    # Parse position
    board = parse_position(command)
    
    # Determine whose move it is
    if board.turn == chess.WHITE:
        opponent_color = chess.BLACK
    else:
        opponent_color = chess.WHITE
    
    # Start housekeeping (opponent just moved, they're thinking now)
    # Actually no - wait for "go" command
    
elif command.startswith("go"):
    # Stop housekeeping (our turn to move)
    scheduler.stop_opponent_thinking()
    
    # Parse time control
    time_control = parse_go_command(command)
    
    # Make move
    best_move = make_move_decision(board, time_control)
    
    print(f"bestmove {best_move.uci()}")
    
    # Start housekeeping for next position
    # (Opponent is now thinking)
    board.push(best_move)
    scheduler.start_opponent_thinking(board, board.turn)
```

---

### 4. Fallback to Static Engine

**When to Fall Back**:
1. Stage 2 confidence < 0.5 (position outside training distribution)
2. Time pressure (< 10 seconds remaining)
3. Stage 1 or Stage 2 inference error/timeout

**Static Engine**: Use V7P3R v17.8 or v18.3 logic

```python
def fallback_static_engine(position: Board, time_limit: float = 5.0) -> Move:
    """
    Fall back to traditional static engine evaluation.
    
    Uses V7P3R v18.3 logic:
    - Quiescence search
    - Alpha-beta pruning
    - Transposition table
    - Iterative deepening
    
    Args:
        position: Current position
        time_limit: Max time to spend (seconds)
        
    Returns:
        Best move from static evaluation
    """
    print("info string Fallback: Using static engine")
    
    # Import V7P3R static engine
    from v7p3r_engine import V7P3REngine
    
    engine = V7P3REngine()
    best_move = engine.search(position, time_limit=time_limit)
    
    return best_move
```

---

### 5. Time Management Philosophy

**Core Principle**: "Always avoid timeouts, prefer to lose in a blaze of glory than draw"

**Time Allocation Strategy**:

```python
def calculate_time_budget(time_remaining: float, increment: float, moves_to_go: int = 40) -> float:
    """
    Calculate time budget for current move.
    
    Conservative estimate to avoid timeouts:
    - Assume 40 moves remaining
    - Use 80% of increment (save 20% as buffer)
    - Keep 10% of remaining time as reserve
    
    Args:
        time_remaining: Milliseconds on clock
        increment: Increment per move (milliseconds)
        moves_to_go: Estimated moves until end (default 40)
        
    Returns:
        Time budget in seconds
    """
    time_remaining_sec = time_remaining / 1000.0
    increment_sec = increment / 1000.0
    
    # Reserve 10% of remaining time
    reserve = time_remaining_sec * 0.1
    available = time_remaining_sec - reserve
    
    # Divide by moves to go
    base_time = available / moves_to_go
    
    # Add 80% of increment
    bonus_time = increment_sec * 0.8
    
    total_budget = base_time + bonus_time
    
    # Enforce minimum (1 second) and maximum (30% of remaining)
    min_time = 1.0
    max_time = time_remaining_sec * 0.3
    
    return max(min_time, min(total_budget, max_time))
```

**Time Pressure Handling**:

```python
def handle_time_pressure(time_remaining: float, complexity: float) -> dict:
    """
    Adjust behavior when low on time.
    
    Args:
        time_remaining: Seconds on clock
        complexity: Stage 2 complexity score (0-10)
        
    Returns:
        Adjusted parameters
    """
    if time_remaining < 10:
        # Severe time pressure
        return {
            'max_depth': 1,              # No deep search
            'skip_stage2': True,         # Skip time manager
            'use_cache_only': True,      # Only use cached positions
            'instant_move_threshold': 0.0,  # Play first good move
        }
    
    elif time_remaining < 30:
        # Moderate time pressure
        return {
            'max_depth': 2,
            'skip_stage2': False,
            'use_cache_only': False,
            'instant_move_threshold': 0.9,  # Play if very confident
        }
    
    else:
        # Normal time
        return {
            'max_depth': 5,
            'skip_stage2': False,
            'use_cache_only': False,
            'instant_move_threshold': 0.95,  # Only instant move if extremely confident
        }
```

---

### 6. Self-Improvement Data Collection

**Purpose**: Collect metrics during engine testing for Stage 1 & 2 retraining

**Data to Collect**:

```python
# After each game
game_data = {
    'game_id': 'v20_test_00001',
    'date': '2026-05-31',
    'result': '1-0',  # W-L-D
    'elo_opponent': 2000,
    'time_control': '5+4',
    
    'positions': [
        {
            'move_number': 1,
            'fen': '...',
            'stage1_output': {'move': 'e2e4', 'prob_good': 0.87},
            'stage2_output': {'complexity': 3.2, 'time_alloc': 0.3},
            'move_played': 'e2e4',
            'time_spent': 2.1,
            'eval_after': 15,  # Centipawns
            'outcome_contribution': 0.02,  # How much this move contributed to win
        },
        # ... all moves
    ],
    
    'metrics': {
        'avg_complexity': 5.2,
        'avg_time_per_move': 8.3,
        'cache_hit_rate': 0.35,
        'stage2_confidence_avg': 0.72,
        'blunders': 1,
        'mistakes': 3,
        'inaccuracies': 8,
    }
}

# Save to JSONL
with open('training_data/stage1_stage2_selfplay.jsonl', 'a') as f:
    f.write(json.dumps(game_data) + '\n')
```

**Feedback Loop**:

1. After 100+ test games, analyze data
2. Extract positions where:
   - Stage 1 was wrong (predicted good, but was blunder)
   - Stage 2 was wrong (allocated too much/little time)
   - Confidence was wrong (high confidence but bad outcome)
3. Add these positions to Stage 1/Stage 2 training data
4. Retrain models
5. Deploy updated models
6. Repeat

---

### 7. Engine Philosophy in Action

**"Progress Over Perfection"**:

```python
def select_tal_style_move(good_moves: list, stage2_priorities: list) -> Move:
    """
    Prefer aggressive, complex moves over safe moves.
    
    Characteristics of Tal-style moves:
    - Sacrifices with compensation
    - Forcing moves (checks, captures, threats)
    - Complex positions (high forest_darkness)
    - Piece activity over material
    """
    # Rank moves by Tal-style characteristics
    tal_scores = []
    
    for move, priority in zip(good_moves, stage2_priorities):
        tal_score = priority  # Start with Stage 2 priority
        
        # Bonus for sacrifices
        if move['material_delta'] < -100 and move['has_compensation']:
            tal_score += 2.0
        
        # Bonus for checks
        if move['is_check']:
            tal_score += 1.0
        
        # Bonus for complexity
        if move['complexity'] >= 6.0:
            tal_score += 1.5
        
        # Bonus for forcing moves
        if move['is_forcing']:
            tal_score += 1.0
        
        # Penalty for passive moves
        if move['is_retreat'] and not move['creates_counterplay']:
            tal_score -= 2.0
        
        tal_scores.append(tal_score)
    
    # Select highest Tal score
    best_idx = tal_scores.index(max(tal_scores))
    return good_moves[best_idx]['move']
```

**"Blaze of Glory" Philosophy**:

```python
def avoid_draws(position: Board, good_moves: list, game_phase: str) -> Move:
    """
    Prefer losing with tactics over accepting draws.
    
    If position is drawish (eval near 0.0, symmetric, simple):
    - Avoid move repetition
    - Avoid piece trades when material equal
    - Prefer complications even if slightly worse
    """
    eval_cp = evaluate_position(position)
    
    if abs(eval_cp) < 50 and game_phase == 'middlegame':
        # Position is drawish
        print("info string Position is drawish - seeking complications")
        
        # Filter out drawish moves
        good_moves_filtered = [
            m for m in good_moves
            if not m['simplifies'] and not m['repeats_position']
        ]
        
        if good_moves_filtered:
            # Prefer most complex move
            return max(good_moves_filtered, key=lambda m: m['complexity'])
        else:
            # No complex moves available - play normally
            return good_moves[0]
    
    else:
        # Not drawish - play normally
        return good_moves[0]
```

---

## Performance Targets

### Inference Speed
- **Stage 1**: <50ms per position (19-dim features, 200k params)
- **Stage 2**: <30ms per position (40-dim features, multi-output)
- **Static Checkmate**: <100ms (depth 5)
- **Total Decision Time**: <200ms (excluding allocated thinking time)

### Cache Performance
- **Hit Rate**: ≥30% (in typical games)
- **Eviction Rate**: <10% per game
- **Storage**: ≤500MB (100k positions)

### Time Management
- **Timeout Rate**: <1% (in 1000+ games)
- **Time Pressure**: <5% of moves in severe time pressure (<10s)
- **Average Time/Move**: 8-12 seconds (in 5+4 blitz)

---

## UCI Command Reference

**Custom Commands** (for testing/debugging):

```
# Check cache status
cache_stats
> info string Cache: 1234 positions, 567 hits (45.9%), 234MB

# Force cache clear
cache_clear
> info string Cache cleared

# Enable/disable Stage 2
setoption name UseStage2 value true

# Set complexity threshold for fallback
setoption name ComplexityThreshold value 8.0

# Enable housekeeping
setoption name EnableHousekeeping value true

# Show decision reasoning
setoption name ShowReasoning value true
> info string Stage 1: e2e4 (87% good), d2d4 (76% good)
> info string Stage 2: Complexity 4.2, Time 0.35, Priority [e2e4: 8.5, d2d4: 7.2]
> info string Decision: e2e4 (highest priority)
```

---

## File Structure

```
v7p3r_v20_engine/
├── src/
│   ├── engine.py                  # Main UCI loop
│   ├── stage1_inference.py        # Stage 1 model loading + inference
│   ├── stage2_inference.py        # Stage 2 model loading + inference
│   ├── precalc_cache.py           # Pre-calculation cache system
│   ├── housekeeping.py            # Background task scheduler
│   ├── static_checkmate.py        # Traditional minimax search
│   ├── decision_integration.py    # Combine Stage 1 + 2 outputs
│   ├── time_manager.py            # Time budget calculations
│   └── uci_protocol.py            # UCI command parsing
│
├── models/
│   ├── position_evaluator_best.pth    # Stage 1 model
│   ├── complexity_manager_best.pth    # Stage 2 model
│   └── move_priority_ranker_best.pth  # Stage 2 move ranker
│
├── config/
│   ├── engine_config.json         # Engine parameters
│   └── personality_thresholds.json  # V7P3R personality settings
│
├── data/
│   └── precalc_cache.db           # SQLite cache database
│
└── logs/
    ├── games/                     # Game records for retraining
    └── metrics/                   # Performance metrics
```

---

## Next Steps

1. **Implement Stage 1 Inference** (1-2 days)
   - Load model from checkpoint
   - Fast batch inference for all legal moves
   
2. **Implement Pre-calculation Cache** (2-3 days)
   - SQLite schema
   - Cache population during opponent thinking
   - LRU eviction
   
3. **Implement Static Checkmate Search** (1 day)
   - Traditional minimax with alpha-beta
   - Mate detection at depth 3-5
   
4. **Build UCI Interface** (2-3 days)
   - Parse UCI commands
   - Time control management
   - Info output formatting
   
5. **Integration Testing** (3-5 days)
   - Test Stage 1 only (baseline)
   - Test Stage 1 + cache
   - Test Stage 1 + checkmate search
   - Full integration after Stage 2 complete

---

## Open Questions for User

1. **Cache Depth**: How deep should pre-calculation go?
   - 1-ply (opponent moves only): Fast, ~5-30 positions
   - 2-ply (opponent + our response): Medium, ~25-150 positions
   - 3-ply (full exchange): Slow, ~125-900 positions

2. **Fallback Threshold**: At what Stage 2 confidence should we fall back to static?
   - Conservative (0.7): Use static more often
   - Moderate (0.5): Balanced
   - Aggressive (0.3): Trust AI more

3. **Housekeeping Priority**: Should housekeeping interrupt if opponent moves quickly?
   - Yes (responsive): Stop immediately
   - No (complete task): Finish current pre-calculation

4. **Training Data Format**: Should we log ALL games or only:
   - Games with mistakes (for correction)
   - Games with interesting tactics (for reinforcement)
   - All games (maximum data)

---

## Conclusion

The V7P3R v20 engine integrates AI intelligence with traditional chess algorithms to create a **tactically aggressive, time-aware chess player** that thrives in complex positions.

By combining Stage 1's position evaluation, Stage 2's time management, pre-calculation caching, and static mate detection, the engine can play chess at a completely different level—focusing on progress, pressure, and personality over pure depth and perfection.

**Ready for implementation once Stage 2 training is complete!** 🎯
