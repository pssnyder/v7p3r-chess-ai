# V7P3R Chess AI v5.0
**Next-Generation Chess AI: From Static Evaluation to Neural Network Intelligence**

---

## 🎯 Vision

Transform V7P3R from a static evaluation engine into an AI-powered chess engine that combines:
- **V7P3R's unique personality** - Aggressive, tactical, creative play style
- **Neural network accuracy** - Pattern recognition and position understanding
- **Static engine speed** - Efficient search with learned evaluation

**Target: V7P3R v20** - An AI-enhanced chess engine with V7P3R's soul and neural network precision.

---

## 📊 Architecture: Single Network, Dual Heads

### **Why Not Separate Models?**
V4.0's dual-model approach had synchronization issues because models were trained separately. V5.0 uses the **AlphaZero/Leela Chess Zero architecture**:

```
┌─────────────────────────────────────────┐
│     Input Features (Board + Heuristics)  │
└─────────────────┬───────────────────────┘
                  │
          ┌───────▼────────┐
          │  Shared Layers  │  ← Learn position understanding
          │  (Conv + FC)    │
          └───────┬────────┘
                  │
          ┌───────┴────────┐
          │                │
      ┌───▼───┐        ┌───▼────┐
      │Policy │        │ Value  │
      │ Head  │        │  Head  │
      └───┬───┘        └───┬────┘
          │                │
    Move Probs      Position Eval
   (ordering)        (-∞ to +∞ cp)
```

**Benefits:**
- ✅ Trained together (no synchronization issues)
- ✅ Shared feature learning
- ✅ Policy head = efficient move ordering
- ✅ Value head = accurate evaluation
- ✅ V7P3R personality via reward shaping

---

## 🚀 6-Phase Execution Plan

### **Phase 1: V18.3.1 Profiling Engine** (Weeks 1-2)
**Goal:** Understand V7P3R's decision-making and identify active evaluation functions

#### 1.1 Codebase Refactor (3 Files)
Consolidate V18.3's 10 files into streamlined architecture:

**File Structure:**
```
V7P3R_v18.3.1/
├── v7p3r_engine.py         # UCI, search, workflow (from v7p3r.py + v7p3r_uci.py)
├── v7p3r_evaluators.py     # ALL evaluation functions consolidated
└── v7p3r_profiler.py       # Profiling, logging, BigQuery integration
```

**Consolidation Map:**
- `v7p3r_engine.py` ← `v7p3r.py`, `v7p3r_uci.py` (search + UCI)
- `v7p3r_evaluators.py` ← `v7p3r_fast_evaluator.py`, `v7p3r_modular_eval.py`, `v7p3r_bitboard_evaluator.py`, `v7p3r_eval_modules.py`, `v7p3r_position_context.py`, `v7p3r_move_safety.py`, `v7p3r_eval_selector.py` (all evaluation logic)
- `v7p3r_profiler.py` ← NEW (profiling infrastructure)

#### 1.2 Profiling Data Schema
```json
{
  "position_id": "UUID",
  "timestamp": "2026-05-05T10:30:00Z",
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "move_played": "e2e4",
  
  "stockfish": {
    "top5_moves": ["e2e4", "d2d4", "g1f3", "c2c4", "g2g3"],
    "top5_evals": [15, 12, 10, 8, 5],
    "best_move": "e2e4",
    "eval": 15,
    "depth": 20,
    "time": 0.5
  },
  
  "v7p3r": {
    "eval": 18,
    "depth_reached": 8,
    "nodes": 12450,
    "time": 0.045,
    "move_rank": 1
  },
  
  "performance": {
    "eval_time_ms": 45,
    "reserved_time_ms": 150,
    "move_time_ms": 82,
    "tempo_gain_ms": 68,
    "function_calls": 247,
    "nodes_per_second": 276666,
    "cutoff_count": 89
  },
  
  "heuristics": {
    "material": 0,
    "pst": 30,
    "king_safety_basic": 15,
    "king_safety_complex": 0,
    "mobility": 25,
    "hanging_pieces": 0,
    "captures": 0,
    "checks": 0,
    "bishop_pair": 0,
    "pawn_structure": 5,
    "piece_activity": 12
  },
  
  "move_ordering": {
    "tt_move": "e2e4",
    "killer_moves": [],
    "capture_moves": [],
    "check_moves": [],
    "tactical_moves": [],
    "quiet_moves": ["e2e4", "d2d4", "g1f3"]
  },
  
  "position_context": {
    "material_balance": 0,
    "piece_count": 32,
    "game_phase": "opening",
    "tactical_flags": []
  },
  
  "metadata": {
    "game_id": "lichess_v7p3r_bot_2026-05-05_001",
    "move_number": 1,
    "source": "live_play",
    "confidence": 1.0
  }
}
```

#### 1.3 Active Function Identification
Profiling will reveal which of the 58+ evaluation functions are:
- ✅ **ACTIVE** - Called during search, impact decisions
- ⚠️ **PLACEHOLDER** - Defined but never used
- 🔧 **CONDITIONAL** - Only active in specific game phases

**Expected Active Set (~20-30 functions):**
- Material counting
- PST evaluation
- King safety (basic + complex)
- Mobility/activity
- Hanging pieces
- Tactical patterns
- Move safety

### **Phase 2: Data Collection Campaign** (Weeks 2-4)
**Goal:** Build comprehensive training dataset from multiple sources

#### 2.1 Data Sources

**Source 1: V18.3.1 Profiling** (Target: 5,000 positions)
- Live games via lichess-bot
- Arena tournament games
- Self-play matches
- BigQuery streaming ingestion

**Source 2: Historical V7P3R Games** (Target: 10,000 positions)
- 4 existing PGN files (lichess_v7p3r_bot_*.pgn)
- Extract key positions (every 3-5 moves)
- Stockfish analysis for each position
- V7P3R personality labeling

**Source 3: Tactical Puzzle Database** (Target: 100,000 positions)
- Lichess puzzle database (already preprocessed)
- Filtered by rating (800-2000)
- Themes: mate, fork, pin, skewer, discovery
- Stockfish solutions included

**Source 4: Endgame Tablebase** (Target: 5,000 positions)
- Critical endgame positions
- Theoretical wins/draws
- V7P3R performance vs perfect play

#### 2.2 Dataset Structure
```python
training_record = {
    # Input Features
    "board_representation": {
        "bitboard_planes": np.array(15, 8, 8),  # 15 channels
        "castling_rights": [1, 0, 1, 0],         # KQkq
        "en_passant": 0,                          # File or -1
        "side_to_move": 1                         # 1=white, -1=black
    },
    
    "v7p3r_heuristics": {
        "material": 0,
        "pst": 30,
        "king_safety": 15,
        "mobility": 25,
        # ... only ACTIVE functions (20-30 total)
    },
    
    "game_phase": {
        "opening": 0,
        "middlegame": 1,
        "endgame": 0
    },
    
    # Output Labels
    "policy_target": {
        "move_probabilities": np.array(4096),  # Softmax over legal moves
        "top5_moves": ["e2e4", "d2d4", "g1f3", "c2c4", "g2g3"]
    },
    
    "value_target": {
        "stockfish_eval": 15,     # Centipawns
        "game_outcome": 1.0,       # 1.0=win, 0.5=draw, 0.0=loss
        "mate_distance": None      # Moves to mate if applicable
    },
    
    # Training Metadata
    "confidence": 1.0,  # 1.0 if V7P3R matched SF, 0.8 otherwise
    "source": "historical_game",
    "v7p3r_personality": True  # Move preserves V7P3R style
}
```

### **Phase 3: Feature Engineering** (Week 5)
**Goal:** Design input features with precision

#### 3.1 Board Representation (960 features)
```python
# 15 planes × 8 × 8 = 960 features
board_planes = [
    # Piece planes (12)
    white_pawns,    white_knights,  white_bishops,
    white_rooks,    white_queens,   white_king,
    black_pawns,    black_knights,  black_bishops,
    black_rooks,    black_queens,   black_king,
    
    # Game state (3)
    en_passant_targets,
    castling_availability,
    side_to_move_plane  # All 1s or all 0s
]
```

#### 3.2 V7P3R Heuristics (20-30 features)
**Only ACTIVE functions from Phase 1 profiling:**
```python
heuristics_vector = [
    material_balance,        # -900 to +900
    pst_score,              # -200 to +200
    king_safety_basic,      # 0 to 100
    king_safety_complex,    # 0 to 200 (middlegame only)
    mobility_score,         # 0 to 100
    hanging_pieces_penalty, # 0 to -500
    bishop_pair_bonus,      # 0 or 50
    pawn_structure,         # -100 to +50
    piece_activity,         # 0 to 50
    # ... continue for all ACTIVE functions
]
```

#### 3.3 Auxiliary Features (10 features)
```python
auxiliary = [
    game_phase_opening,    # 1 or 0
    game_phase_middlegame, # 1 or 0
    game_phase_endgame,    # 1 or 0
    move_number,           # Normalized 0-1
    material_total,        # Pieces remaining
    castling_done_white,   # Boolean
    castling_done_black,   # Boolean
    tactical_position,     # Boolean (has pins/forks)
    forcing_position,      # Boolean (checks/captures)
    repetition_count       # 0, 1, or 2
]
```

**Total Input Size: 960 + 30 + 10 = 1000 features**

### **Phase 4: Neural Network Architecture** (Week 6)
**Goal:** Design single network with dual output heads

#### 4.1 Model Architecture
```python
class V7P3R_V5_NeuralEngine(nn.Module):
    def __init__(self):
        super().__init__()
        
        # ==========================================
        # CONVOLUTIONAL TOWER (Board Understanding)
        # ==========================================
        # Input: 15 × 8 × 8
        self.conv1 = nn.Conv2d(15, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks (Leela Chess Zero style)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(64, 64) for _ in range(10)
        ])
        
        # Output: 64 × 8 × 8 = 4096 features
        
        # ==========================================
        # HEURISTIC PROCESSOR (V7P3R DNA)
        # ==========================================
        self.heuristic_fc1 = nn.Linear(30, 128)
        self.heuristic_bn1 = nn.BatchNorm1d(128)
        self.heuristic_fc2 = nn.Linear(128, 256)
        self.heuristic_bn2 = nn.BatchNorm1d(256)
        
        # ==========================================
        # FEATURE FUSION
        # ==========================================
        # Combine board features + heuristics + auxiliary
        self.fusion_fc = nn.Linear(4096 + 256 + 10, 1024)
        self.fusion_bn = nn.BatchNorm1d(1024)
        self.fusion_dropout = nn.Dropout(0.3)
        
        # ==========================================
        # POLICY HEAD (Move Ordering)
        # ==========================================
        self.policy_fc1 = nn.Linear(1024, 512)
        self.policy_bn1 = nn.BatchNorm1d(512)
        self.policy_fc2 = nn.Linear(512, 4096)  # All possible moves
        
        # ==========================================
        # VALUE HEAD (Position Evaluation)
        # ==========================================
        self.value_fc1 = nn.Linear(1024, 256)
        self.value_bn1 = nn.BatchNorm1d(256)
        self.value_fc2 = nn.Linear(256, 128)
        self.value_bn2 = nn.BatchNorm1d(128)
        self.value_fc3 = nn.Linear(128, 1)  # Single eval score
        
    def forward(self, board_planes, heuristics, auxiliary):
        # Convolutional tower
        x = F.relu(self.bn1(self.conv1(board_planes)))
        for block in self.res_blocks:
            x = block(x)
        x = x.view(x.size(0), -1)  # Flatten to 4096
        
        # Heuristic processing
        h = F.relu(self.heuristic_bn1(self.heuristic_fc1(heuristics)))
        h = F.relu(self.heuristic_bn2(self.heuristic_fc2(h)))
        
        # Fusion
        combined = torch.cat([x, h, auxiliary], dim=1)
        fused = F.relu(self.fusion_bn(self.fusion_fc(combined)))
        fused = self.fusion_dropout(fused)
        
        # Policy head
        policy = F.relu(self.policy_bn1(self.policy_fc1(fused)))
        policy_logits = self.policy_fc2(policy)
        
        # Value head
        value = F.relu(self.value_bn1(self.value_fc1(fused)))
        value = F.relu(self.value_bn2(self.value_fc2(value)))
        value_score = torch.tanh(self.value_fc3(value)) * 2000  # ±2000cp range
        
        return policy_logits, value_score


class ResidualBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels_out)
        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels_out)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)
```

**Model Parameters:** ~2.5M (comparable to Leela Chess Zero's smaller networks)

#### 4.2 Training Configuration
```python
training_config = {
    "optimizer": "AdamW",
    "learning_rate": 0.001,
    "lr_scheduler": "CosineAnnealingWarmRestarts",
    "batch_size": 256,
    "epochs": 50,
    "weight_decay": 0.0001,
    "gradient_clipping": 1.0,
    
    "loss_weights": {
        "policy_weight": 0.7,
        "value_weight": 0.3
    },
    
    "augmentation": {
        "horizontal_flip": True,  # Mirror position
        "color_flip": True        # Swap colors
    }
}
```

### **Phase 5: Reward Function Design** (Week 7)
**Goal:** Define training objectives from V18.3.1 insights

#### 5.1 Composite Reward Function
```python
def calculate_reward(position, move, outcome, metadata):
    """
    Multi-component reward preserving V7P3R personality
    while learning from Stockfish corrections
    """
    reward = 0.0
    
    # ==========================================
    # Component 1: Move Quality (Stockfish)
    # ==========================================
    if move in metadata['stockfish_top5']:
        rank = metadata['stockfish_top5'].index(move) + 1
        # Rank 1: +5.0, Rank 2: +4.0, ..., Rank 5: +1.0
        reward += (6 - rank)
    else:
        reward -= 2.0  # Penalty for move not in top 5
    
    # ==========================================
    # Component 2: Evaluation Accuracy
    # ==========================================
    v7p3r_eval = metadata['v7p3r_eval']
    stockfish_eval = metadata['stockfish_eval']
    eval_error = abs(v7p3r_eval - stockfish_eval)
    
    # Penalize large evaluation errors
    if eval_error < 20:
        reward += 1.0
    elif eval_error < 50:
        reward += 0.5
    elif eval_error > 200:
        reward -= 1.0
    
    # ==========================================
    # Component 3: V7P3R Personality Preservation
    # ==========================================
    if metadata['v7p3r_personality']:
        # V7P3R played this move AND it's in Stockfish top-5
        reward += 1.5  # Strong bonus for personality match
    
    # ==========================================
    # Component 4: Tactical Correctness
    # ==========================================
    if metadata['tactical_position']:
        if metadata['tactical_success']:
            reward += 2.0  # Solved tactic correctly
        else:
            reward -= 3.0  # Failed tactic (critical error)
    
    # ==========================================
    # Component 5: Game Outcome (Terminal State)
    # ==========================================
    if outcome is not None:
        if outcome == 1.0:      # Win
            reward += 10.0
        elif outcome == 0.5:    # Draw
            if metadata['position_advantage'] > 100:
                reward -= 5.0   # Should have won
            else:
                reward += 2.0   # Fair draw
        else:                   # Loss
            reward -= 10.0
    
    # ==========================================
    # Component 6: Efficiency Bonus
    # ==========================================
    if metadata['nodes_searched'] < metadata['average_nodes']:
        reward += 0.5  # Reward efficient search
    
    # ==========================================
    # Component 7: Time Management
    # ==========================================
    if metadata['tempo_gain'] > 0:
        reward += 0.3  # Reward thinking faster than allocated time
    
    return reward
```

#### 5.2 Reward Shaping for V7P3R Personality
```python
personality_traits = {
    "aggression": {
        "description": "Prefers forcing moves (checks, captures, threats)",
        "reward_bonus": 0.5,
        "applies_when": lambda move: move.is_capture or gives_check(move)
    },
    
    "tactical_vision": {
        "description": "Recognizes tactical patterns (forks, pins, skewers)",
        "reward_bonus": 1.0,
        "applies_when": lambda pos: has_tactical_pattern(pos)
    },
    
    "king_safety_priority": {
        "description": "Values king safety highly",
        "reward_bonus": 0.8,
        "applies_when": lambda pos: improves_king_safety(pos)
    },
    
    "activity_over_material": {
        "description": "Sometimes trades material for activity",
        "reward_bonus": 0.3,
        "applies_when": lambda move: gains_activity_sacrifices_material(move)
    }
}
```

### **Phase 6: Training Pipeline** (Weeks 8-10)
**Goal:** Train unified model with comprehensive validation

#### 6.1 Training Loop
```python
def train_v5_model(model, train_loader, val_loader, config):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    for epoch in range(config['epochs']):
        model.train()
        epoch_policy_loss = 0
        epoch_value_loss = 0
        
        for batch in train_loader:
            # Unpack batch
            board_planes = batch['board_planes'].to(device)
            heuristics = batch['heuristics'].to(device)
            auxiliary = batch['auxiliary'].to(device)
            policy_target = batch['policy_target'].to(device)
            value_target = batch['value_target'].to(device)
            
            # Forward pass
            policy_logits, value_pred = model(
                board_planes, heuristics, auxiliary
            )
            
            # Calculate losses
            policy_loss = policy_criterion(policy_logits, policy_target)
            value_loss = value_criterion(value_pred, value_target)
            
            # Combined loss (weighted)
            total_loss = (
                config['loss_weights']['policy_weight'] * policy_loss +
                config['loss_weights']['value_weight'] * value_loss
            )
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['gradient_clipping']
            )
            optimizer.step()
            
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
        
        # Validation
        val_metrics = validate_model(model, val_loader)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Checkpointing
        if val_metrics['top5_accuracy'] > best_top5:
            save_checkpoint(model, epoch, val_metrics)
        
        print(f"Epoch {epoch}: Policy Loss = {epoch_policy_loss:.4f}, "
              f"Value Loss = {epoch_value_loss:.4f}, "
              f"Val Top-5 Acc = {val_metrics['top5_accuracy']:.2%}")
```

#### 6.2 Validation Metrics
```python
validation_metrics = {
    # Policy Head Metrics
    "top1_accuracy": 0.0,      # Best move accuracy
    "top3_accuracy": 0.0,
    "top5_accuracy": 0.0,      # Primary KPI
    "mean_reciprocal_rank": 0.0,
    
    # Value Head Metrics
    "eval_mae": 0.0,           # Mean Absolute Error (cp)
    "eval_rmse": 0.0,          # Root Mean Squared Error
    "eval_r2": 0.0,            # R² score
    
    # Combined Metrics
    "blunder_rate": 0.0,       # % moves with >300cp loss
    "personality_match": 0.0,  # % matching V7P3R style
    
    # Efficiency Metrics
    "inference_time_ms": 0.0,
    "nodes_per_second": 0.0
}
```

---

## 📁 Project Structure

```
v5.0/
├── README.md                          # This file
│
├── docs/
│   ├── V7P3R_v18_3_1_Refactor_Workshop.ipynb  # Refactor planning
│   ├── Data_Schema.md                          # Dataset specifications
│   ├── Model_Architecture.md                   # Neural network design
│   ├── Reward_Functions.md                     # RL reward engineering
│   └── Training_Guide.md                       # Training procedures
│
├── static_engines/
│   ├── V7P3R_v18.3/                   # Original engine (10 files)
│   │   ├── src/
│   │   │   ├── v7p3r.py
│   │   │   ├── v7p3r_uci.py
│   │   │   ├── v7p3r_fast_evaluator.py
│   │   │   ├── v7p3r_modular_eval.py
│   │   │   ├── v7p3r_bitboard_evaluator.py
│   │   │   ├── v7p3r_eval_modules.py
│   │   │   ├── v7p3r_position_context.py
│   │   │   ├── v7p3r_move_safety.py
│   │   │   ├── v7p3r_eval_selector.py
│   │   │   └── v7p3r_openings_v161.py
│   │   └── V7P3R_v18.3.bat
│   │
│   └── V7P3R_v18.3.1/                 # Refactored profiling engine (3 files)
│       ├── src/
│       │   ├── v7p3r_engine.py        # UCI + search + workflow
│       │   ├── v7p3r_evaluators.py    # All evaluation functions
│       │   └── v7p3r_profiler.py      # Profiling + BigQuery
│       └── V7P3R_v18.3.1.bat
│
├── data/
│   ├── profiling/                     # V18.3.1 profiling data
│   │   ├── live_games/
│   │   ├── arena_tournaments/
│   │   └── self_play/
│   │
│   ├── historical/                    # V7P3R game history
│   │   ├── pgn_files/
│   │   ├── extracted_positions/
│   │   └── stockfish_analysis/
│   │
│   ├── puzzles/                       # Tactical training
│   │   ├── lichess_puzzles_100k.json
│   │   └── themed_puzzles/
│   │
│   ├── endgames/                      # Tablebase positions
│   │
│   └── training/                      # Final training datasets
│       ├── train.h5
│       ├── val.h5
│       └── test.h5
│
├── src/
│   ├── models/
│   │   ├── v5_neural_engine.py        # Main neural network
│   │   ├── residual_blocks.py
│   │   └── feature_extractors.py
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   ├── dataset.py
│   │   ├── augmentation.py
│   │   └── rewards.py
│   │
│   ├── evaluation/
│   │   ├── validators.py
│   │   ├── metrics.py
│   │   └── personality_scorer.py
│   │
│   └── utils/
│       ├── feature_engineering.py
│       ├── data_preprocessing.py
│       └── bigquery_connector.py
│
├── scripts/
│   ├── refactor_v18_3_to_18_3_1.py   # Automated refactoring
│   ├── collect_profiling_data.py     # Live profiling
│   ├── process_historical_games.py   # PGN → training data
│   ├── build_training_dataset.py     # Combine all sources
│   └── train_v5_model.py             # Main training script
│
├── checkpoints/
│   └── v5_model_epoch_*.pth
│
├── configs/
│   ├── training_config.yaml
│   ├── model_architecture.yaml
│   └── bigquery_config.yaml
│
└── requirements.txt
```

---

## 🎯 Success Criteria

### Phase 1 (V18.3.1)
- [ ] Refactor complete (3 files functional)
- [ ] Profiling data streaming to BigQuery
- [ ] Active evaluation functions identified
- [ ] 5,000+ profiled positions collected

### Phase 2 (Data Collection)
- [ ] 5,000 profiling positions
- [ ] 10,000 historical game positions
- [ ] 100,000 puzzle positions
- [ ] 5,000 endgame positions
- [ ] Dataset schema validated

### Phase 3 (Feature Engineering)
- [ ] Board representation tested (960 features)
- [ ] V7P3R heuristics extracted (20-30 features)
- [ ] Auxiliary features defined (10 features)
- [ ] Feature normalization validated

### Phase 4 (Model Architecture)
- [ ] Neural network implemented
- [ ] Forward pass validated
- [ ] Parameter count: ~2.5M
- [ ] Inference time: <10ms (CPU)

### Phase 5 (Reward Functions)
- [ ] Composite reward function tested
- [ ] Personality traits encoded
- [ ] Reward shaping validated
- [ ] Edge cases handled

### Phase 6 (Training)
- [ ] Training pipeline functional
- [ ] Validation metrics tracking
- [ ] Top-5 accuracy >85%
- [ ] Eval MAE <30cp
- [ ] Personality match >70%

---

## 🚀 Getting Started

### Prerequisites
```bash
# Python 3.10+
pip install torch torchvision
pip install chess python-chess
pip install google-cloud-bigquery
pip install numpy pandas scikit-learn
pip install tqdm tensorboard
```

### Quick Start
```bash
# 1. Refactor V18.3 → V18.3.1
cd v5.0
python scripts/refactor_v18_3_to_18_3_1.py

# 2. Start profiling collection
python scripts/collect_profiling_data.py --games 1000

# 3. Process historical data
python scripts/process_historical_games.py --pgn-dir data/historical/pgn_files

# 4. Build training dataset
python scripts/build_training_dataset.py

# 5. Train model
python scripts/train_v5_model.py --config configs/training_config.yaml
```

---

## 📊 Monitoring & Metrics

### TensorBoard
```bash
tensorboard --logdir=runs/v5_training
```

### BigQuery Profiling Dashboard
- Real-time profiling data ingestion
- Evaluation function usage heatmaps
- Move quality distribution
- Performance bottleneck analysis

---

## 🔬 Research Questions

1. **Feature Importance:** Which V7P3R heuristics contribute most to model accuracy?
2. **Personality Transfer:** Can we preserve V7P3R's playing style while improving accuracy?
3. **Data Efficiency:** How many positions needed for 85%+ top-5 accuracy?
4. **Generalization:** Does model transfer to positions outside training distribution?
5. **Speed Trade-off:** Inference time vs evaluation accuracy curve?

---

## 📝 Next Steps

**Immediate (Week 1):**
1. Complete V18.3.1 refactor
2. Set up BigQuery profiling pipeline
3. Begin live profiling data collection

**User Tasks:**
- Historical game dataset preparation
- BigQuery connection details
- Profiling campaign parameters

**Agent Tasks:**
- V18.3.1 refactor automation
- Profiling infrastructure
- Active function identification

---

## 📚 References

- **AlphaZero:** Silver et al., "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" (2017)
- **Leela Chess Zero:** https://lczero.org/
- **V7P3R Chess Engine:** v18.3.0 (PST optimization, +56 ELO)

---

**Status:** Phase 1 In Progress - V18.3.1 Refactor  
**Target Completion:** June 2026  
**Next Milestone:** V7P3R v20 AI-Enhanced Engine
