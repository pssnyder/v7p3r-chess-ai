# V7 Tablebase Integration Plan

## Problem Statement

**Observation from first 10 training games:**
- High forest darkness achieved (avg 0.371, peak 0.445) ✓
- Personality emergence successful (1.7 sacrifices/game) ✓
- **But 9/10 games hit 200-move limit** ❌

**Root Cause:**
The network optimizes for:
1. Stockfish evaluation (70%) - positional quality
2. Personality rewards (20%) - complexity/chaos
3. Game outcome (10%) - win/loss/draw

**Missing:** Explicit reward for **reaching winning tablebase positions**

Current training signal:
- "This position is complex" ✓
- "This position is objectively good" ✓  
- "This game was eventually won" ✓
- **"This move leads to forced mate in N" ❌**

## Proposed Solution: Syzygy Tablebase Integration

### Architecture

```
Position → Check Piece Count
         ↓
    ≤7 pieces? → YES → Query Syzygy Tablebase
         ↓                    ↓
         NO              DTZ (Distance to Zero)
         ↓                    ↓
    Neural Network      Convert to reward:
    Evaluation          - Win: +1.0 (perfect)
                        - Loss: -1.0 (perfect)
                        - Draw: 0.0 (perfect)
                        - Bonus: Shorter DTZ = higher reward
```

### Training Signal Modification

**Current (V7.0):**
```python
target = 0.7*stockfish_eval + 0.2*personality + 0.1*outcome
```

**Enhanced (V7.1 with Tablebases):**
```python
if is_tablebase_position(board):
    # Perfect endgame knowledge
    tb_result = probe_tablebase(board)
    if tb_result.is_win():
        target = 1.0 - (tb_result.dtz / 200)  # Faster wins = better
    elif tb_result.is_loss():
        target = -1.0 + (tb_result.dtz / 200)  # Delay losses
    else:  # Draw
        target = 0.0
    
    # Add personality bonus ONLY if winning
    if tb_result.is_win():
        target += 0.1 * personality_reward  # Technical precision bonus
else:
    # Standard non-tablebase training
    target = 0.7*stockfish_eval + 0.2*personality + 0.1*outcome
```

### Implementation Plan

#### Phase 1: Syzygy Integration (Immediate)

**Dependencies:**
```bash
pip install python-chess  # Already installed (has tablebase support)
```

**Download Tablebases:**
- 3-4-5 piece: ~1 GB (essential)
- 6 piece: ~150 GB (optional, high impact)
- 7 piece: ~17 TB (skip for now)

**Storage location:**
```
E:\Chess\Tablebases\
├── 3-4-5\  (download first)
└── 6\      (download later if space permits)
```

**Code changes:**
1. Create `src/v7/tablebase_oracle.py`
2. Modify `selfplay_trainer.py` to check tablebases before Stockfish
3. Add tablebase reward to training target calculation

#### Phase 2: Move Selection Enhancement

**Current move selection:**
```python
# Evaluate each legal move with network
for move in legal_moves:
    board.push(move)
    value = network.predict(board)
    board.pop()
```

**Enhanced with tablebase awareness:**
```python
for move in legal_moves:
    board.push(move)
    
    # Check tablebase first
    if is_tablebase_position(board):
        tb_result = probe_tablebase(board)
        if tb_result.is_win():
            value = 1.0 - (tb_result.dtz / 200)  # Prioritize tablebase wins
        elif tb_result.is_loss():
            value = -1.0  # Avoid tablebase losses
        else:
            value = 0.0  # Tablebase draw
    else:
        value = network.predict(board)
    
    board.pop()
```

**Impact:** Games will naturally converge to tablebase wins instead of wandering for 200 moves.

#### Phase 3: Personality-Aware Endgame Play

Your "Dark Forest Assassin" profile specifies:
- **Endgame**: "Technical precision + fighting chess"
- **Never accept draws prematurely**

**Tablebase integration respects this:**
```python
if tb_result.is_draw():
    # Only accept tablebase draws if material is truly insufficient
    if material_advantage <= 0:
        value = 0.0  # Accept draw
    else:
        # Reject tablebase draw, keep fighting
        value = network.predict(board)  # Use neural network instead
```

**Result:** Engine will:
- Accept draws in dead-drawn positions (K+B vs K)
- **Fight on** in unclear positions (even if tablebase says draw)
- Convert winning endgames with perfect technique

## Expected Improvements

### Quantitative

**Before Tablebases (Current):**
- Games at 200 moves: 90% (9/10)
- Natural conclusions: 10% (1/10)
- Avg game length: 195 moves

**After Tablebases (Projected):**
- Games at 200 moves: <30%
- Natural conclusions: >70%
- Avg game length: 120-140 moves
- **Decisive games should checkmate in endgame**

### Qualitative

**Middlegame:** No change
- High forest darkness ✓
- Free material sacrifices ✓
- Ultra-sharp chaos ✓

**Endgame:** Major improvement
- Knows when position is won (tablebase proof)
- Finds fastest conversion path (DTZ minimization)
- Never "wanders" in won positions
- Technical precision (your profile requirement) ✓

## Implementation Code Sketch

### `src/v7/tablebase_oracle.py`

```python
"""
Syzygy Tablebase Integration for V7P3R

Provides perfect endgame knowledge for positions with ≤7 pieces.
"""

import chess
import chess.syzygy
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class TablebaseResult:
    """Result from tablebase probe."""
    wdl: int  # Win/Draw/Loss: 2=win, 1=cursed win, 0=draw, -1=blessed loss, -2=loss
    dtz: int  # Distance to zeroing (50-move rule)
    is_tablebase_position: bool = True
    
    def is_win(self) -> bool:
        """Check if position is winning."""
        return self.wdl > 0
    
    def is_loss(self) -> bool:
        """Check if position is losing."""
        return self.wdl < 0
    
    def is_draw(self) -> bool:
        """Check if position is drawn."""
        return self.wdl == 0
    
    def get_reward(self, perspective_white: bool = True) -> float:
        """
        Convert tablebase result to training reward.
        
        Returns value in [-1, 1] range with DTZ bonus.
        """
        if self.is_draw():
            return 0.0
        
        # Win/loss base value
        if self.is_win():
            base_value = 1.0
        else:  # Loss
            base_value = -1.0
        
        # DTZ bonus: faster wins/slower losses are better
        # Max DTZ is typically ~200 moves
        dtz_bonus = (200 - abs(self.dtz)) / 200 * 0.2
        
        if perspective_white:
            return base_value + (dtz_bonus if self.is_win() else -dtz_bonus)
        else:
            return -(base_value + (dtz_bonus if self.is_win() else -dtz_bonus))


class TablebaseOracle:
    """Manages Syzygy tablebase queries."""
    
    def __init__(self, tablebase_path: str):
        """
        Initialize tablebase oracle.
        
        Args:
            tablebase_path: Path to Syzygy tablebase files
        """
        self.tablebase_path = Path(tablebase_path)
        
        if not self.tablebase_path.exists():
            raise FileNotFoundError(f"Tablebase path not found: {tablebase_path}")
        
        # Load tablebases
        self.tables = chess.syzygy.open_tablebase(str(self.tablebase_path))
        
        # Get maximum piece count
        self.max_pieces = self._detect_max_pieces()
    
    def _detect_max_pieces(self) -> int:
        """Detect maximum number of pieces in available tablebases."""
        # Probe a simple position with increasing piece counts
        # Return the maximum supported
        for pieces in range(7, 2, -1):
            try:
                # Test position with N pieces
                test_fen = self._create_test_position(pieces)
                board = chess.Board(test_fen)
                self.tables.probe_wdl(board)
                return pieces
            except KeyError:
                continue
        return 3  # Minimum
    
    def _create_test_position(self, num_pieces: int) -> str:
        """Create a test position with N pieces for tablebase detection."""
        if num_pieces == 3:
            return "8/8/8/8/8/4k3/8/4K3 w - - 0 1"  # K vs K (illegal but for testing)
        elif num_pieces == 4:
            return "8/8/8/8/8/4k3/8/3QK3 w - - 0 1"  # KQ vs K
        elif num_pieces == 5:
            return "8/8/8/8/8/4k3/8/3QKR2 w - - 0 1"  # KQR vs K
        # Add more test positions for 6, 7 pieces
        return "8/8/8/8/8/4k3/8/4K3 w - - 0 1"
    
    def is_tablebase_position(self, board: chess.Board) -> bool:
        """Check if position can be looked up in tablebases."""
        # Count pieces (excluding kings which are always present)
        piece_count = len(board.piece_map())
        return piece_count <= self.max_pieces
    
    def probe(self, board: chess.Board) -> Optional[TablebaseResult]:
        """
        Probe tablebase for position evaluation.
        
        Args:
            board: Chess position to probe
        
        Returns:
            TablebaseResult if found, None if not in tablebase
        """
        if not self.is_tablebase_position(board):
            return None
        
        try:
            # Probe WDL (Win/Draw/Loss)
            wdl = self.tables.probe_wdl(board)
            
            # Probe DTZ (Distance to Zeroing)
            dtz = self.tables.probe_dtz(board)
            
            return TablebaseResult(wdl=wdl, dtz=dtz)
        
        except KeyError:
            # Position not in tablebase
            return None
    
    def get_best_move(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Get best move according to tablebase.
        
        Returns move that maintains/achieves win, or None if not in tablebase.
        """
        if not self.is_tablebase_position(board):
            return None
        
        current_result = self.probe(board)
        if current_result is None:
            return None
        
        # Find move that maintains best result
        best_move = None
        best_value = -float('inf')
        
        for move in board.legal_moves:
            board.push(move)
            
            if self.is_tablebase_position(board):
                result = self.probe(board)
                if result:
                    # Negate because we're checking opponent's perspective
                    value = -result.get_reward(board.turn)
                    
                    if value > best_value:
                        best_value = value
                        best_move = move
            
            board.pop()
        
        return best_move


# Example usage
if __name__ == "__main__":
    # Assuming tablebases are downloaded
    TB_PATH = r"E:\Chess\Tablebases\3-4-5"
    
    oracle = TablebaseOracle(TB_PATH)
    
    # Test position: KQ vs K (White wins)
    board = chess.Board("8/8/8/8/8/4k3/8/3QK3 w - - 0 1")
    
    result = oracle.probe(board)
    if result:
        print(f"WDL: {result.wdl} (Win={result.is_win()})")
        print(f"DTZ: {result.dtz} (moves to conversion)")
        print(f"Reward: {result.get_reward(perspective_white=True):.3f}")
        
        best_move = oracle.get_best_move(board)
        print(f"Best move: {best_move}")
```

## Integration with Current Training

### Modify `selfplay_trainer.py`

**Add tablebase oracle to `SelfPlayTrainer.__init__`:**
```python
def __init__(self, profile_path, stockfish_path, tablebase_path=None, ...):
    # ... existing code ...
    
    # Initialize tablebase oracle
    self.tablebase_oracle = None
    if tablebase_path and Path(tablebase_path).exists():
        try:
            from tablebase_oracle import TablebaseOracle
            self.tablebase_oracle = TablebaseOracle(tablebase_path)
            print(f"[OK] Tablebase oracle loaded: {tablebase_path}")
            print(f"  Max pieces: {self.tablebase_oracle.max_pieces}")
        except Exception as e:
            print(f"[WARN] Tablebase loading failed: {e}")
```

**Modify move selection in `SelfPlayGame.select_move`:**
```python
def select_move(self, board: chess.Board) -> chess.Move:
    """Select move with tablebase awareness."""
    legal_moves = list(board.legal_moves)
    
    # Check tablebase first
    if self.tablebase_oracle and self.tablebase_oracle.is_tablebase_position(board):
        tb_move = self.tablebase_oracle.get_best_move(board)
        if tb_move:
            return tb_move  # Use perfect tablebase move
    
    # Fall back to network evaluation (existing code)
    # ... existing network-based move selection ...
```

**Modify training target in `play_game`:**
```python
# Query tablebase first
tb_result = None
if self.tablebase_oracle and self.tablebase_oracle.is_tablebase_position(board_copy):
    tb_result = self.tablebase_oracle.probe(board_copy)

if tb_result:
    # Use perfect tablebase knowledge
    stockfish_eval = tb_result.get_reward(board_copy.turn == chess.WHITE)
else:
    # Query Stockfish oracle (existing code)
    sf_result = self.oracle.evaluate(board_copy)
    stockfish_eval = sf_result.normalized_score
```

## Download Instructions

### Syzygy 3-4-5 Piece Tablebases (~1 GB)

**Download from:**
http://tablebase.sesse.net/syzygy/3-4-5/

**Required files:**
```
KBvK.rtbz       KNvK.rtbz       KPvK.rtbz       KQvK.rtbz       KRvK.rtbz
KBBvK.rtbz      KBNvK.rtbz      KBPvK.rtbz      ... (all 3-4-5 piece)
```

**Install:**
```powershell
# Create directory
New-Item -ItemType Directory -Path "E:\Chess\Tablebases\3-4-5" -Force

# Download (use aria2 or wget for batch download)
# Then extract to E:\Chess\Tablebases\3-4-5\
```

### Optional: 6-Piece Tablebases (~150 GB)

**Only if you have storage:**
- Dramatically improves 6-piece endgame play
- Not required for initial testing

## Testing Plan

### Phase 1: Tablebase Oracle Test
```python
python src/v7/tablebase_oracle.py
```
Expected: Correct WDL and DTZ for test positions

### Phase 2: Integration Test
Run 10 games with tablebase integration:
```python
trainer = SelfPlayTrainer(
    profile_path="profiles/dark_forest_assassin.json",
    stockfish_path="...",
    tablebase_path="E:\\Chess\\Tablebases\\3-4-5"
)
trainer.train_from_selfplay(num_games=10)
```

**Expected results:**
- More games conclude naturally (<200 moves)
- Games entering 3-5 piece endgames should checkmate efficiently
- No more "wandering" in won endgames

### Phase 3: Full Training
After validation, run 100-game training with tablebases.

**Expected improvement:**
- Natural conclusion rate: 10% → 70%+
- Avg game length: 195 → 130 moves
- Decisive games end in checkmate (not max-moves)

## Technical Precision Alignment

This perfectly aligns with your "Dark Forest Assassin" profile:

**Middlegame (Unchanged):**
- Ultra-aggressive ✓
- High complexity ✓
- Free sacrifices ✓

**Endgame (Enhanced):**
- **Technical precision** via tablebase perfect play ✓
- **Fighting chess** - only accepts tablebase draws if truly dead ✓
- **Practical time management** - fast tablebase lookups ✓

Your engine will play like:
- **Early/Middle game**: Chaotic, aggressive, sacrificial (Dark Forest)
- **Late game**: Precise, technical, converting (Assassin)

## Next Steps

1. **Complete current 100-game training** (baseline without tablebases)
2. **Download Syzygy 3-4-5 tablebases** (~1 GB)
3. **Implement `tablebase_oracle.py`**
4. **Test on 10-game sample**
5. **Run 100-game training with tablebases** (compare results)

## Success Metrics

| Metric | Before TB | After TB | Target |
|--------|-----------|----------|--------|
| Natural conclusions | 10% | ??? | >70% |
| Avg game length | 195 | ??? | <140 |
| Endgame checkmate rate | Low | ??? | >90% |
| Forest darkness (middlegame) | 0.37 | 0.37 | Same |
| Sacrifices/game | 1.7 | 1.7 | Same |

**Goal:** Preserve aggressive middlegame personality, add endgame precision.

---

**Status:** Design complete, ready for implementation after baseline training completes.
