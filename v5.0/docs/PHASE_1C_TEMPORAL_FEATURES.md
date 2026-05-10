# Phase 1C: Temporal Persistence Features (TPF)
**Giving the AI "Memory" and "Momentum Detection"**

---

## 🎯 Core Concept

### The Problem
Current v5.1 features are **Markovian** - the AI sees only the current position state:
- "There's a hanging piece" → but **when did it become hanging?**
- "King safety = 0.3" → but is it **improving or declining?**
- "Material advantage +1.0" → but is this a **temporary imbalance or endgame advantage?**

### The Solution: Temporal Differential
Add `_historical` versions of key features to create a **2-ply window**:

```
ΔFeature = Feature_current - Feature_historical
```

This allows the model to learn **position momentum** and **tactical inflection points**.

---

## 📊 Feature Groups

### **F200-F209: Historical Tactical State** (10 features)

Track when tactical themes **emerged** vs **persisted**:

| Feature ID | Name | Current Feature | Semantic Meaning |
|------------|------|-----------------|------------------|
| F200 | `white_hanging_pieces_historical` | F040 `white_has_hanging_pieces` | "New threat vs sustained oversight" |
| F201 | `black_hanging_pieces_historical` | F040 `black_has_hanging_pieces` | Same for Black |
| F202 | `white_en_prise_value_historical` | F043 `white_pieces_en_prise_value` | "Material tension momentum" |
| F203 | `black_en_prise_value_historical` | F043 `black_pieces_en_prise_value` | Same for Black |
| F204 | `white_pins_historical` | F045 `white_has_pin` | "Pin just created vs pre-existing" |
| F205 | `black_pins_historical` | F045 `black_has_pin` | Same for Black |
| F206 | `white_king_under_attack_historical` | F011 `white_king_under_attack` | "Attack intensifying vs stabilized" |
| F207 | `black_king_under_attack_historical` | F011 `black_king_under_attack` | Same for Black |
| F208 | `white_trapped_pieces_historical` | F048 `white_trapped_piece_count` | "Trap just sprung vs ignored" |
| F209 | `black_trapped_pieces_historical` | F048 `black_trapped_piece_count` | Same for Black |

**Why These Features?**
- **Hanging pieces**: Distinguish "blunder just made" from "sacrifice 3 moves ago"
- **Pins**: New pin = tactical opportunity; old pin = positional constraint
- **King attacks**: Detect **attack momentum** (ramping up vs holding steady)

---

### **F210-F214: Historical Position Evaluation** (5 features)

Track **evaluation momentum** and **initiative**:

| Feature ID | Name | Semantic Meaning |
|------------|------|------------------|
| F210 | `position_eval_historical` | Previous position evaluation (centipawns) |
| F211 | `material_balance_historical` | Previous material count (was material just traded?) |
| F212 | `king_safety_white_historical` | Previous White king safety score |
| F213 | `king_safety_black_historical` | Previous Black king safety score |
| F214 | `center_control_historical` | Previous center control score (is control shifting?) |

**Derivatives the AI Can Learn:**
```python
# Example: Detect "sacrificial attack" pattern
if material_balance_current < material_balance_historical:  # Lost material
    if king_safety_opponent_current < king_safety_opponent_historical:  # But opponent king worse
        → "Tactical sacrifice for attack" (likely grade 4-5 move)
```

---

### **F215-F219: Move Sequence Context** (5 features)

Encode **where pieces came from** and **multi-move sequences**:

| Feature ID | Name | Type | Description |
|------------|------|------|-------------|
| F215 | `last_move_from_square` | One-Hot (64) | Where the last piece moved FROM |
| F216 | `last_move_to_square` | One-Hot (64) | Where the last piece moved TO |
| F217 | `last_move_piece_type` | One-Hot (6) | What piece moved (P/N/B/R/Q/K) |
| F218 | `move_sequence_index` | Integer (0-10) | Position in multi-move sequence (puzzles) |
| F219 | `is_forcing_sequence` | Boolean | Is this part of a forced tactical line? |

**Why Path Encoding?**
- Helps AI detect **transpositions** by piece flow patterns
- Example: Knight from g1→f3→e5 vs Knight from b1→d2→e4 (different paths to central squares)
- **Puzzle sequences**: AI learns "move 1 sets up move 2" relationships

---

### **F220: Context Availability Mask** (1 feature)

| Feature ID | Name | Type | Description |
|------------|------|------|-------------|
| F220 | `has_history` | Binary | 1 = historical features valid, 0 = no history available |

**Purpose**: Tell the model when to **ignore** historical features

**Usage:**
- **Puzzles with sequences**: `has_history = 1` (populate all F200-F219)
- **Single-position games**: `has_history = 0` (set F200-F219 to -1.0 null sentinel)
- **Self-play/Real games**: `has_history = 1` (engine maintains state)

---

## 🔧 Implementation Strategy

### Data Structure Updates

#### Current JSONL Structure (v5.0):
```json
{
  "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
  "best_move": "d2d4",
  "v7p3r_move": "d2d3",
  "grade": 3,
  "stockfish_eval": 0.35,
  "features": { ... }
}
```

#### Enhanced Structure (v5.1 with TPF):
```json
{
  "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
  "best_move": "d2d4",
  "v7p3r_move": "d2d3",
  "grade": 3,
  "stockfish_eval": 0.35,
  
  // NEW: Temporal context
  "previous_fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/8/PPPPPPPP/RNBQK2R b KQkq - 3 3",
  "last_move_uci": "f3",
  "move_sequence_index": 2,
  "sequence_id": "puzzle_12345_line_1",
  "has_history": 1,
  
  "features": {
    // Current features (F000-F114)
    "white_hanging_pieces": 0,
    "white_en_prise_value": 0,
    
    // NEW: Historical features (F200-F220)
    "white_hanging_pieces_historical": 0,
    "white_en_prise_value_historical": 0,
    "position_eval_historical": 0.25,  // Position was +0.25, now +0.35 → improving
    "last_move_from_square": 6,  // f1 (one-hot index)
    "last_move_to_square": 21,  // f3 (one-hot index)
    "last_move_piece_type": 1,  // Knight (one-hot index)
    "move_sequence_index": 2,
    "has_history": 1
  }
}
```

---

### Script Updates

#### 1. **`scripts/calculate_features.py`** - Core Feature Calculator

**NEW: Temporal Feature Calculator**

```python
class TemporalFeatureCalculator:
    """
    Calculate historical features by maintaining state between positions
    """
    
    def __init__(self):
        self.position_cache = {}  # FEN → features mapping
        self.feature_calculator = ChessFeatureCalculator()
    
    def calculate_temporal_features(self, current_fen, previous_fen=None, 
                                   last_move_uci=None, sequence_index=0):
        """
        Calculate features with temporal context
        
        Args:
            current_fen: Current position FEN
            previous_fen: Previous position FEN (if available)
            last_move_uci: Last move in UCI format (e.g., "e2e4")
            sequence_index: Position in multi-move sequence
            
        Returns:
            dict: Features with F000-F114 (current) + F200-F220 (temporal)
        """
        # Calculate current features
        current_features = self.feature_calculator.calculate_all_features(current_fen)
        
        # Initialize temporal features
        temporal_features = {}
        
        if previous_fen is not None:
            # Has history - calculate previous state
            temporal_features['has_history'] = 1
            
            # Get cached previous features or calculate
            if previous_fen in self.position_cache:
                prev_features = self.position_cache[previous_fen]
            else:
                prev_features = self.feature_calculator.calculate_all_features(previous_fen)
                self.position_cache[previous_fen] = prev_features
            
            # F200-F209: Historical tactical state
            temporal_features['white_hanging_pieces_historical'] = prev_features.get('white_has_hanging_pieces', 0)
            temporal_features['black_hanging_pieces_historical'] = prev_features.get('black_has_hanging_pieces', 0)
            temporal_features['white_en_prise_value_historical'] = prev_features.get('white_pieces_en_prise_value', 0)
            temporal_features['black_en_prise_value_historical'] = prev_features.get('black_pieces_en_prise_value', 0)
            temporal_features['white_pins_historical'] = prev_features.get('white_has_pin', 0)
            temporal_features['black_pins_historical'] = prev_features.get('black_has_pin', 0)
            temporal_features['white_king_under_attack_historical'] = prev_features.get('white_king_under_attack', 0)
            temporal_features['black_king_under_attack_historical'] = prev_features.get('black_king_under_attack', 0)
            temporal_features['white_trapped_pieces_historical'] = prev_features.get('white_trapped_piece_count', 0)
            temporal_features['black_trapped_pieces_historical'] = prev_features.get('black_trapped_piece_count', 0)
            
            # F210-F214: Historical evaluation
            temporal_features['position_eval_historical'] = prev_features.get('stockfish_eval', 0.0)
            temporal_features['material_balance_historical'] = prev_features.get('material_balance_cp', 0)
            temporal_features['king_safety_white_historical'] = prev_features.get('white_king_safety_score', 0.5)
            temporal_features['king_safety_black_historical'] = prev_features.get('black_king_safety_score', 0.5)
            temporal_features['center_control_historical'] = prev_features.get('white_center_control_score', 0)
            
            # F215-F219: Move sequence encoding
            if last_move_uci:
                from_square = chess.SQUARE_NAMES.index(last_move_uci[:2])
                to_square = chess.SQUARE_NAMES.index(last_move_uci[2:4])
                
                board = chess.Board(previous_fen)
                piece = board.piece_at(from_square)
                piece_type = piece.piece_type if piece else 0
                
                temporal_features['last_move_from_square'] = from_square
                temporal_features['last_move_to_square'] = to_square
                temporal_features['last_move_piece_type'] = piece_type
            else:
                temporal_features['last_move_from_square'] = -1
                temporal_features['last_move_to_square'] = -1
                temporal_features['last_move_piece_type'] = 0
            
            temporal_features['move_sequence_index'] = sequence_index
            temporal_features['is_forcing_sequence'] = 0  # TODO: Detect forced lines
            
        else:
            # No history - set sentinel values
            temporal_features['has_history'] = 0
            
            # F200-F209: Set to -1.0 (null sentinel)
            for i in range(200, 210):
                temporal_features[f'F{i:03d}'] = -1.0
            
            # F210-F214: Set to -1.0
            temporal_features['position_eval_historical'] = -999.0  # Obvious null
            temporal_features['material_balance_historical'] = -999
            temporal_features['king_safety_white_historical'] = -1.0
            temporal_features['king_safety_black_historical'] = -1.0
            temporal_features['center_control_historical'] = -1.0
            
            # F215-F219: Set to -1 (no move)
            temporal_features['last_move_from_square'] = -1
            temporal_features['last_move_to_square'] = -1
            temporal_features['last_move_piece_type'] = 0
            temporal_features['move_sequence_index'] = 0
            temporal_features['is_forcing_sequence'] = 0
        
        # Merge current + temporal features
        all_features = {**current_features, **temporal_features}
        
        # Cache current position for next iteration
        self.position_cache[current_fen] = current_features
        
        return all_features
```

#### 2. **Puzzle Processing** - Extract Multi-Move Sequences

**NEW: `scripts/extract_puzzle_sequences.py`**

```python
"""
Extract multi-move sequences from puzzle database
Populates historical features with correct move sequences
"""

import json
import chess
import chess.pgn

def process_puzzle_with_sequence(puzzle_data):
    """
    Process a puzzle and generate training positions with temporal context
    
    Args:
        puzzle_data: {
            "fen": starting FEN,
            "moves": "e2e4 e7e5 g1f3" (solution PV),
            "rating": puzzle rating
        }
    
    Returns:
        list: Training positions with historical context
    """
    board = chess.Board(puzzle_data['fen'])
    moves = puzzle_data['moves'].split()
    
    temporal_calculator = TemporalFeatureCalculator()
    training_positions = []
    
    previous_fen = None
    previous_move = None
    
    for i, move_uci in enumerate(moves):
        current_fen = board.fen()
        
        # Calculate temporal features
        features = temporal_calculator.calculate_temporal_features(
            current_fen=current_fen,
            previous_fen=previous_fen,
            last_move_uci=previous_move,
            sequence_index=i
        )
        
        # Make the move
        move = chess.Move.from_uci(move_uci)
        board.push(move)
        
        # Store position data
        position = {
            'fen': current_fen,
            'previous_fen': previous_fen,
            'move': move_uci,
            'sequence_index': i,
            'sequence_id': f"puzzle_{puzzle_data['id']}_line_1",
            'has_history': 1 if previous_fen else 0,
            'features': features
        }
        
        training_positions.append(position)
        
        # Update for next iteration
        previous_fen = current_fen
        previous_move = move_uci
    
    return training_positions
```

---

## 🎯 Expected Impact

### Model Learning Capabilities

**Before TPF (v5.1 without temporal):**
- "This position has a hanging piece" → **static snapshot**
- Can't distinguish "new blunder" from "multi-move sacrifice"

**After TPF (v5.1 with temporal):**
- "Hanging piece count: 1 → 1" → **persistent state** (sacrifice pattern)
- "Hanging piece count: 0 → 1" → **new tactical oversight** (likely blunder)
- "King safety: 0.7 → 0.4 → 0.2" → **ramping attack** (tactical sequence)

### Performance Targets

| Metric | v5.1 (no TPF) | v5.1 + TPF | Improvement |
|--------|---------------|------------|-------------|
| **Policy Accuracy** | 54-57% | **56-60%** | +2-3% |
| **Puzzle Accuracy** | 55-60% | **65-70%** | +10% (multi-move context!) |
| **Temporal Consistency** | N/A | **+15%** | New metric: maintains plan across moves |
| **Transposition Detection** | 0% | **40-50%** | Via piece path encoding |

---

## 📋 Implementation Checklist

### Phase 1: Data Preparation
- [ ] Update JSONL schema to include `previous_fen`, `last_move_uci`, `sequence_id`
- [ ] Create `extract_puzzle_sequences.py` for puzzle multi-move extraction
- [ ] Process puzzle database (~20,876 positions → ~60,000 sequence positions)
- [ ] Tag game positions with `has_history=0`

### Phase 2: Feature Calculator
- [ ] Implement `TemporalFeatureCalculator` class
- [ ] Add F200-F220 feature calculation
- [ ] Add position caching for efficiency
- [ ] Test on sample puzzle sequence

### Phase 3: Preprocessing
- [ ] Update `preprocess_dataset_v5.1.py` to handle F200-F220
- [ ] Add one-hot encoding for `last_move_from_square` (64 values)
- [ ] Add one-hot encoding for `last_move_to_square` (64 values)
- [ ] Add one-hot encoding for `last_move_piece_type` (6 values)
- [ ] Handle `-1` sentinel values in normalization

### Phase 4: Model Architecture
- [ ] Update input_dim: 106 → **106 + 21 + 64 + 64 + 6 = 261 features**
- [ ] Add attention masking layer conditioned on `has_history` bit
- [ ] Test forward pass with mixed history/no-history batches

### Phase 5: Training
- [ ] Train on mixed dataset (puzzles + games)
- [ ] Monitor puzzle accuracy separately (should be higher with sequences)
- [ ] Validate temporal consistency metric

### Phase 6: Self-Play Integration
- [ ] Implement state persistence in V7P3R engine
- [ ] Maintain feature history during game
- [ ] Test AI suggestions with temporal context

---

## 🔬 Validation Strategy

### Test 1: Temporal Awareness
**Input**: Puzzle sequence where piece becomes hanging on move 2
**Expected**: AI predicts different grades for:
- Move 1: Piece not hanging → grade based on other factors
- Move 2: Piece just became hanging → **tactical oversight detected** (grade 0-1)

### Test 2: Momentum Detection
**Input**: Multi-move attack sequence (king safety declining)
**Expected**: AI learns "ramping attack" pattern, predicts higher grades for attack continuation moves

### Test 3: Graceful Degradation
**Input**: Game position with `has_history=0`
**Expected**: Model ignores F200-F220, performs similar to v5.1 without TPF

---

## 📊 Feature Summary

**Total New Features: 21 + 136 one-hot = 157 additional inputs**

- F200-F209: Historical tactical (10)
- F210-F214: Historical evaluation (5)
- F215: Last move from square (one-hot 64)
- F216: Last move to square (one-hot 64)
- F217: Last move piece type (one-hot 6)
- F218: Sequence index (1)
- F219: Forcing sequence flag (1)
- F220: Has history mask (1)

**Total Input Dimension: 106 (v5.1) + 157 (TPF) = 263 features**

---

## 🚀 Next Steps

1. **Immediate**: Create `extract_puzzle_sequences.py` to test concept on 100 puzzles
2. **Short-term**: Implement `TemporalFeatureCalculator` in `calculate_features.py`
3. **Medium-term**: Retrain model with 263 input features
4. **Long-term**: Integrate with V7P3R self-play for real-time temporal context

This gives your AI **true temporal learning** - it's no longer Markovian!
