# V7P3R Chess AI v10.0 aka VvXAI Conceptual Framework and Development Plan
A High-Performance Chess Engine with Integrated Psychological & Temporal Awareness, Built on a Legacy of 19 Generations and millions of positions.

## Conceptual Overview

### Mission Statement

**To evolve chess intelligence from sterile mathematical calculation to pragmatic strategic mastery. By unifying high-cohesion structural geometry with real-time spatiotemporal awareness, VvXAI models the psychological and resource-driven boundaries of competitive play—proving that the optimal move is defined not just by the topology of the board, but by the pressure of the situation.**

---

### Problem Statement

Data such as move time, clock, depth, and phase are metadata invaluable for filtering, cleaning, and curating a training dataset, its traditional thought that feeding them directly into a neural network during training is detrimental to an engine's playing strength. Classical chess evaluative thinking tells us a chess model needs to learn how to evaluate a position based strictly on the board's structural reality, regardless of how much time is left on the clock or what depth complexity of the outcome was based on. With today's computer vs. computer chess landscape being quite strict and stagnant as most engines fight to achieve positional perfection, this results in what I feel is substantial over-computing, unrealistic decision making, and non-human chess. 

>The best chess engines are so good at finding the "best move" in a position that they often miss the forest for the trees; blind to the psychological and temporal context of the game, which is where human players excel. 

### Hypothesis

By offering the model a more holistic view of the game state, including time pressure and psychological tension, I believe it can learn to make more human-like decisions that are not just about the position on the board, but about the context in which that position exists. In chess, I'm not playing the position—I'm playing the person across from me. An important part of my journey in artificial chess intelligence is to understand the chess that is being played, to define the heuristics behind each position, and to understand the motives behind each move. I feel there is a feature gap in knowledge between humans and the computational machines we have created to solve a mathematical problem and the way to close that gap is by modeling our own more nuanced psychological and temporal vision. I've always thought the path to higher intelligence in chess was in, "thinking like a computer", but I now believe the path to higher intelligence in chess is in, "*feeling* like a human".  

>I believe that much of chess is not won by playing the "best move in that position" but the "best move in that situation". 

### Solution Definition

The solution I am proposing is a sentiment based model. Instead of interpreting the position alone, it also has features I identified for various sentiment heuristics, such as the current theme, tactics, piece pairing, pawn structure, king safety, etc. Similar to heuristics of my static eval engine but re-written so they are not subjective to my definition of good vs bad, but objective of the current perspective. 

```
example: "question presented: are there two bishops on the board? here is a bit to represent whether that fact is true or false" as opposed to static engines which state "two bishops on the board are good so give that a +0.02 eval unless x, y, z then give it a -0.03". 
```

I want the engine to have "highlighted" connections already, so it not only can find on its own how the bishops relate to the king and other pieces, but also has a specific value to tell it that, which can be clustered with other "sentiment" values and thus have their clustered weights adjusted as needed for more or less personality.

>Intelligence through **sentiment-based feature mapping** and **temporal psychological metadata**.

I also have very specific objective calculations that find pre-selected patterns for identification or intrinsically combine other more distant metrics and highlight them. 

```
Example: I have a dark forest score and positional complexity score that are calculated based on patterns matching key Tal positions, sacrifice counts, SEE or MVV-LVA calculations, etc., that generate data about the further "psychological" tension of the position. 
```

I believe that time remaining has a huge impact on playing style. Grandmasters adjust their approach in the final minutes or when their opponent is under time pressure, purposefully playing sharp, complex lines to convert losing positions into wins. 

```
Example 1: Bullet players exploit premoved openings online by predicting the setup and playing an alternative that turns the opponent's premoved sequence into a losing position—a sophisticated psychological tactic where the best move depends on understanding opponent behavior, not just board position. 

Example 2: In endgames under time pressure, speed trumps computation. Grandmasters employ piece shuffles and three-fold repetition avoidance strategies to manage the opponent's time—flagging their clock in non-increment games or gaining time in increment formats. These temporal tactics are invisible in static position evaluation but critical to practical play. 

```

>Pushing the boundary where **traditional brute-force evaluation** clashes with **human cognitive psychology**.

**Summation:** There is an underlying, non-computational, psychology to chess that I want my engine to learn. 

---

## Problem Validation

### Part 1: The Sentiment Features (The Objective Reality Trap)

Presenting the network with objective, unweighted facts (e.g., a bit for `has_bishop_pair`) and letting the model cluster those values with other metrics is **100% correct and aligns with modern ML philosophy.** Traditional engines failed at this because their human-crafted evaluation (HCE) applied static weights ($+0.02$). By passing these facts into a neural network instead, I let the model discover non-linear combinations that humans may never explicitly codify.

#### The Blind Spot to Guard Against

Pre-calculated complex positional formulas (like a "dark forest score" patterned after Tal's games) passed as flat input features, might inadvertently introduce **human confirmation bias** back into a neural network.

```
The Risk: If the formula for a "Tal-like position" looks for specific sacrifice counts or piece clusters, the network can only learn based on that math of what a Tal game looks like.
The Fix: Don't give the network the pre-calculated score. Give it the raw component pieces of the tension (the specific material imbalances, the line-of-sight counts between heavy pieces and the King, the exact count of unresolved tactical trades/SEE metrics). Let the neural network’s hidden layers discover its *own* definition of "psychological tension." It might find a pattern that human grandmasters feel instinctively but have never successfully codified into a formula.
```
---

### Part 2: The Clock & Time Context (The Core Thesis)

This is where the hypothesis gets fascinating. I am arguing that chess is not a closed mathematical proof played in a vacuum; it is a **finite-resource game played against an opponent with a fluctuating cognitive load.** Grandmasters change their style based on the clock. They will intentionally play suboptimal, messy, high-complexity moves ("swindles") if their opponent is in time trouble, knowing the opponent lacks the clock cycles to calculate the refutation.

>Can a model learn to calculate this cognitive load? **Yes, it actually can—but it fundamentally changes what my model is predicting.**

#### Unified Architecture: Single Network with Dual Inputs and Three Output Heads

Rather than separate networks, I use a **single neural network architecture** that accepts both HalfKA and sentiment features as parallel inputs, fuses them in shared hidden layers, and outputs three prediction heads simultaneously. This allows gradient flow between evaluation and policy learning, enabling the model to discover how sentiment features predict stronger moves under pressure.

```
Board State (FEN)
      │
      ├──────────────────────┬─────────────────┬──────────────┐
      │                      │                 │              │
      ▼                      ▼                 ▼              ▼
HalfKA Features    Sentiment Features    Clock Data    Move History
(45K sparse)       (8-12 dense dims)     (2-3 dims)    (context)
      │                      │                 │              │
      ▼                      ▼                 ▼              ▼
┌────────────────┐ ┌──────────────────┐ ┌───────────┐ ┌──────────┐
│ Accumulators   │ │ Dense Encoder    │ │TimeEmbed  │ │ History  │
│ (1024-2048)    │ │ (512-1024)       │ │ (256)     │ │ (256)    │
│ ClippedReLU    │ │ ReLU             │ │ ReLU      │ │ ReLU     │
│ [0,1] bounded  │ │ LayerNorm        │ │           │ │          │
└────────┬───────┘ └────────┬─────────┘ └─────┬─────┘ └────┬─────┘
         │                  │                  │            │
         └──────────────────┼──────────────────┼────────────┘
                            │
                ┌───────────▼───────────┐
                │ Shared Hidden Layers  │  ◄─ BOTH inputs interact here
                │ (2048-4096 neurons)   │     through residual blocks
                │ Residual blocks +     │     and LayerNorm
                │ LayerNorm             │
                └───────────┬───────────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
    ┌──────────┐      ┌──────────┐     ┌─────────┐
    │Evaluation│      │Character │     │  WDL    │
    │Head      │      │Head      │     │ Head    │
    │(MSE Loss)│      │(CE Loss) │     │(CE Loss)│
    │  70%     │      │   20%    │     │   10%   │
    └────┬─────┘      └────┬─────┘     └────┬────┘
         │                 │                │
         ▼                 ▼                ▼
    Eval Score       Move Probs      W/D/L Probs
    (cp)             (4000 moves)    (confidence)
         │                 │                │
         └─────────────────┼────────────────┘
                           │
            ┌──────────────▼──────────────┐
            │  Time-Aware Move Selector   │  ◄─ Inference Logic
            │                            │
            │ sentiment_weight =          │
            │   f(clock_remaining,       │
            │     opp_time,              │
            │     position_complexity)   │
            │                            │
            │ final_move_score =         │
            │   (1 - w) * eval_score +   │
            │   w * sentiment_score      │
            │                            │
            │ Sentiment overrides eval   │
            │ when time pressure > 0.75  │
            └──────────────┬─────────────┘
                           │
                           ▼
                  Best Move Selected
```

#### Why This Architecture Works

The key insight is that **sentiment features don't replace evaluation—they provide context for when to trust intuition over calculation.** During training, all three heads learn simultaneously from the same position:

- **Evaluation Head** learns: "What is the objective value of this position?"
- **Character Head** learns: "What move would V7P3R play here (based on 19 generations of personality)?"
- **WDL Head** learns: "What are the endgame probabilities from Syzygy tablebases?"

The shared hidden layers learn to represent board states in a way that benefits all three predictions. Gradients from all three losses flow backward through the same network, allowing the model to discover that:
- High forest_darkness correlates with positions where sentiment (move preference) beats pure evaluation
- Time pressure amplifies the value of piece_tension and move_diversity
- King safety and pawn_structure matter differently under time scramble vs. contemplation

#### At Inference Time: Dynamic Blending

```python
def select_move(board, halfdka_features, sentiment_features, clock_time):
    """
    Single network output three signals. Blend them based on time pressure.
    """
    # Forward pass (single network)
    eval_score = model.eval_head(halfdka_features)      # Objective evaluation
    move_probs = model.policy_head(sentiment_features)  # Human-like moves
    wdl_probs = model.wdl_head(shared_hidden)          # Endgame truth
    
    # Time-aware blending: sentiment overrides eval under pressure
    time_ratio = clock_time / initial_time
    position_complexity = sentiment_features['forest_darkness']
    
    if time_ratio < 0.1:          # <10% time left
        sentiment_weight = 0.7    # Trust intuition, not calculation
    elif time_ratio < 0.25:       # <25% time left
        sentiment_weight = 0.5    # Equal balance
    elif position_complexity > 0.6: # Complex position
        sentiment_weight = 0.4    # Slightly favor sentiment
    else:
        sentiment_weight = 0.2    # Favor precise evaluation
    
    # Blend signals
    blended_probs = (1 - sentiment_weight) * eval_probs + \
                    sentiment_weight * policy_probs
    
    # Select move with highest blended score
    return select_best_move(blended_probs)
```

This means **under time pressure, sentiment features can suppress lower-ranked moves from the eval head and elevate complex but practical moves preferred by V7P3R's personality.**

---

### The Verdict on Approach

1. Building a "sentiment-driven, psychological" engine is a completely open frontier in chess engine design that standard chess software avoids because they are chasing absolute mathematical maximum ELO.
2. The goal is to build an engine that exhibits genuine human-like **personality, aggression, and contextual pragmatism**, completely justifying exploring these features.
3. To bridge the gap in the data pipeline, I structure the neural network as a **single unified architecture with dual input streams**:
    - **HalfKA Input Stream:** ~45K sparse indices representing pure board topology (pieces, king positions, king buckets)
    - **Sentiment Input Stream:** 8-12 dense dimensions capturing psychological context (forest_darkness, piece_tension, king_safety, pawn_structure, center_control, game_phase, urgency, move_history)
    - **Shared Hidden Layers:** Both inputs merge in 2048-4096 neuron hidden layers where they interact through residual blocks
    - **Three Output Heads:** Evaluation head (MSE loss, 70%), Character head (CrossEntropy loss, 20%), WDL head (CrossEntropy loss, 10%)
    - **At Inference:** Time-aware blending logic dynamically weights how much sentiment overrides evaluation based on clock pressure and position complexity

This architecture ensures that **sentiment doesn't corrupt evaluation**—both signals remain independent in their respective heads, but the shared hidden layers learn to use sentiment as context for when the policy head's move preference should dominate the evaluation head's cp score under time pressure.

>**Key Principle:** The network learns that sentiment-preferred moves are often winning moves when the opponent is under time pressure, but evaluation-preferred moves are more reliable when both players have ample thinking time.

---

## Technical Concept: Sentiment-Driven & Spatiotemporal Chess Modeling

>This formal concept bridges V7P3R's 19-generation legacy engine with machine learning principles, directly reconciling data structure with human chess psychology.

### 1. The Unified Network Architecture

The VvXAI model is a **single neural network with three input streams and three output heads**, designed to learn both objective evaluation and contextual move selection:

#### Input Streams

```
Stream 1 (HalfKA):
  ├─ King bucket assignment (8x8=64 buckets)
  ├─ Piece positions (one-hot per bucket)
  ├─ Perspective flip (white vs black to move)
  └─ Result: ~45K sparse indices per position

Stream 2 (Sentiment Features):
  ├─ forest_darkness (0.0-1.0)
  ├─ piece_tension (0-16 pieces)
  ├─ king_safety_score (computed from evaluator)
  ├─ pawn_structure_score (passed, doubled, isolated count)
  ├─ center_control (-1.0 to +1.0)
  ├─ game_phase (0.0 opening, 0.5 midgame, 1.0 endgame)
  ├─ move_urgency (0.1-1.0)
  └─ Result: 8-12 dense features

Stream 3 (Temporal Context):
  ├─ clock_time_remaining (seconds)
  ├─ opponent_time_remaining (seconds)
  ├─ time_control_type (bullet/blitz/rapid)
  ├─ move_number (normalized)
  └─ Result: 4 dense features (optional at training time, mandatory at inference)
```

#### Processing and Fusion

```
HalfKA Stream        Sentiment Stream      Temporal Stream
  (sparse)             (dense 8-12)          (dense 4)
      │                    │                      │
      ▼                    ▼                      ▼
Accumulator         Dense Encoder           TimeEmbed
(ClippedReLU)       (512-1024 neurons)     (256 neurons)
      │                    │                      │
      └────────────────────┼──────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  Merge      │  (Concatenate all streams)
                    │  Layers     │  (2048 neurons)
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │   Shared Hidden Layers            │
         │   (3-4 residual blocks)           │
         │   (4096-8192 neurons total)       │
         │   LayerNorm after each block      │
         └─────────────────┼─────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
   Eval Head          Policy Head           WDL Head
   (64 neurons)       (256 neurons)        (32 neurons)
        │                  │                  │
        ▼                  ▼                  ▼
   Eval Out          Policy Probs         WDL Probs
  (cp score)        (4000 moves)         (W/D/L)
```

#### Output Heads and Loss Functions

Three independent heads trained simultaneously with combined loss:

```python
total_loss = 0.70 * eval_loss + 0.20 * policy_loss + 0.10 * wdl_loss

where:
  eval_loss = MSE(eval_out, eval_label)           # Objective position value
  policy_loss = CrossEntropy(policy_out, move_label)  # Human-like moves
  wdl_loss = CrossEntropy(wdl_out, wdl_label)     # Endgame ground truth
```

This **three-term loss allows gradient flow from all three objectives back through the shared hidden layers**, enabling the network to learn a representation that serves all three purposes simultaneously.

### 2. Why Three Heads Instead of Two

Evaluation networks trained only on game outcomes (W/L/D) suffer from **label ambiguity**: a brilliant move played in a position where the opponent blundered still receives the winning label, obscuring the true quality of the move.

By training **three separate predictions on the same board state with different ground-truth labels**, the network learns to separate:

- **What the position is actually worth** (Eval head, trained on Stockfish evals)
- **What move V7P3R would play** (Policy head, trained on V7P3R's historical move choices)
- **Who wins with best play** (WDL head, trained on Syzygy tablebase endgame truth)

The shared hidden layers learn a representation that balances all three objectives. Under high-dimensional pressure, the network discovers that:
- Sentiment features are excellent predictors of policy head outputs
- High forest_darkness positions correlate with policy moves diverging from eval moves
- Time pressure amplifies this divergence

### 3. Resolution of the Binary Label Trap via Multi-Signal Training

Traditional evaluation networks trained on binary W/L labels struggle with contradictions: "Position X led to a loss, but objectively should be winning." 

The VvXAI model resolves this through **multi-task learning**:

$$\text{Total Loss} = w_{\text{eval}} \cdot L_{\text{eval}} + w_{\text{policy}} \cdot L_{\text{policy}} + w_{\text{wdl}} \cdot L_{\text{wdl}}$$

This means Position X can simultaneously:
- Have a high Eval head output (position is objectively +2.5)
- Have a specific move as the highest policy output (V7P3R prefers this move)
- Have W/D/L probabilities from Syzygy (50% win probability in endgame with best play)

The model learns that **evaluation, move preference, and endgame truth are three separate but related concepts**, not contradictions. Sentiment features help the network understand *when* to weight policy (move preference) more heavily than eval (objective calculation).

### 4. Spatiotemporal Context Integration: Clock Pressure as Signal

The clock data is treated as a **contextual filter** that modulates how much the network trusts sentiment features:

- **High Time (>50% remaining):** Sentiment weight = 0.2, Eval weight = 0.8
  - Prioritize precise calculation, trust HalfKA-derived evaluation
  - Sentiment is secondary context
  
- **Medium Time (25-50% remaining):** Sentiment weight = 0.5, Eval weight = 0.5
  - Balance intuition and calculation
  - Both input streams equally important
  
- **Low Time (<25% remaining):** Sentiment weight = 0.7-0.8, Eval weight = 0.2-0.3
  - Trust complexity-seeking intuition over deep calculation
  - Sentiment features become dominant signal
  - Play sharp, forcing lines that punish opponent's time scramble

This **dynamic blending happens at inference time**, not training time, so the evaluation head never becomes corrupted by clock pressure. The model learns that:
- "When clock is low, follow the policy head more than the eval head"
- "When position is complex, sentiment is more reliable"
- "When opponent has even less time, amplify complexity-seeking moves"

### 5. Retention and Continuance of Engine Personality

To maintain the behavioral profile established over 19 previous versions (~10,000 historical games), this architecture preserves V7P3R's personality through the **Policy head training signal**.

By training the Policy head exclusively on V7P3R's historical move choices (not on optimal play), the network learns to prefer:
- Positions with high forest_darkness
- Moves that create piece tension
- Sacrificial tactics that lead to chaotic play
- Aggressive king attacks despite own king safety concerns

The **Personality Weights** from v7.0-v8.0 are encoded not in hardcoded bonuses, but in the historical move preferences themselves. The network's policy head learns to reproduce V7P3R's Tal-like playing style naturally, without explicit reward engineering.
* **Spatiotemporal Architecture:** Integrating remaining time, time controls, and move velocity directly into the core neural network layers. VvXAI natively understands that time is a physical, volatile resource, shifting its policy toward high-complexity, high-cognitive-load lines when the opponent is pressed against the clock.
* **Objective Sentiment Clustering:** Stripping out hardcoded human valuation bias ($+0.02$). VvXAI ingests raw, unweighted structural facts (tactical tension, king safety boundaries, piece pairing indicators), allowing the deep hidden layers to organically discover non-linear combinations and capture the true psychological tension of the game.

---
# V7P3R vX Development Plan

## Executive Summary

This plan transforms raw chess data into a production neural network engine through **4 sprints**:
1. **Data → Binary** (parsing, encoding, compression of raw data into positional records)
2. **HalfKA Features** (sparse position encoding for the network)
3. **Training Loop** (dual evaluation and policy network architecture for learning evaluation and situational strategy from positions)
4. **Quantization** (making the model fast/small)

**Metaphor**: Building a chess pupil who learns from millions of games, then compressing that knowledge into a fast, efficient expert.

**🔴 CRITICAL**: This plan minimizes Copilot tool usage. Each sprint is **one file, one task, one goal**.

---

## Sprint 1: Data Serialization & Streaming (3-4 days)

**The What**: Convert 120GB of messy chess data (PGN, CSV, JSONL) into a compact, machine-readable binary format (88 bytes per position).

**The Why**: 
- **PGN files are text-heavy** (~1KB per position) — wasteful for machine learning
- **Streaming format** prevents loading entire dataset into RAM (enables training on massive datasets)
- **88-byte record = cache-optimal** (fits in CPU cache line, fast retrieval)
- **Binary encoding = interpretable features** (we know exactly where each piece is, no string parsing)

**The Big Picture**: Think of this as converting a library of 1.3M chess books into a standardized filing system where each position is a 88-byte index card. Instead of reading English text, the neural network reads pure bit patterns.

**Goal**: Transform 120GB library into streaming binary format (27GB compressed).  
**Output**: One production module (binary_format_converter.py) ready for Phase 0 execution.

### The 88-Byte Record: Anatomy of Efficiency

```
Core Record (88 bytes total):
├── fen_hash (8 bytes)              ← Deduplication: unique ID for each position
├── evaluation (2 bytes)            ← Engine's judgment (-32000 to +32000 centipawns)
├── depth (1 byte)                  ← How deep the engine searched (0-128 plies)
├── material (2 bytes)              ← Total piece value on board (material count)
├── piece_count (1 byte)            ← Number of pieces remaining
├── phase (1 byte)                  ← Game stage (0=opening, 24=endgame)
├── wdl (1 byte)                    ← Win/Draw/Loss outcome label
├── time (4 bytes)                  ← Move time spent (milliseconds)
├── clock (2 bytes)                 ← Remaining time on clock (seconds)
├── castling_en_passant (1 byte)    ← Bitpacked: castling rights + en passant file
├── active_color_halfmove (1 byte)  ← Bitpacked: whose turn + halfmove clock
└── board_state (64 bytes)          ← The actual position: 1 byte per square (4 bits piece type)
```

**Why 88 bytes specifically?**
- CPU cache line = 64 bytes (we fit ~1.3 records per cache line)
- 88 bytes = sweet spot between data density and access speed
- Parquet columns store efficiently (columnar format compresses repetitive data types)
- No FEN string needed at training time (board_state contains all position info as bits)

**What we gain by NOT storing FEN string:**
- Remove ~40 bytes of text overhead per position
- Positions become "self-describing" (pure binary, no parsing)
- Training-time feature extraction is pure bitwise operations (faster than string parsing)
- Can reconstruct full FEN from board_state if needed (via decode function)

### Day 1.1: PGN → Binary Conversion

| File | Input | Output | Target Performance |
|------|-------|--------|---------------------|
| `src/binary_format_converter.py` | 120GB PGN files | pgns.bin (1.5GB) | >50 MB/sec |
| **Dependencies** | chess, struct, hashlib, tqdm | 88-byte records | 2-byte move encoding |

**Specific Module**: Create `binary_format_converter.py` with:
- Class `BinaryPositionRecord` (88-byte struct)
  - **What it does**: Represents a single chess position in binary form
  - **Why**: Type-safe encoding/decoding, reusable across ingestion functions
  - **Benefit**: Can validate that every record is exactly 88 bytes (no surprises)
  
- Method `pgn_to_binary(pgn_path, output_path)`
  - **What it does**: Parse chess games from PGN, extract each position, encode to 88 bytes
  - **Why**: PGN is human-readable but space-inefficient for ML
  - **How**: For each move in each game, pause and record the position + evaluation from comments
  - **Benefit**: Streaming — don't load all games into memory, process one at a time

- Method `jsonl_to_binary(jsonl_path, output_path)`
  - **What it does**: Convert JSONL evaluations (fen + eval pairs) into 88-byte records
  - **Why**: JSONL stores structured data but redundantly (repeated field names)
  - **Benefit**: Achieves higher compression ratio, faster I/O

**Validation**: Benchmark on small sample (1000 positions), expect >50 MB/sec
- **Why this speed target?**: At 50 MB/sec, processing 120GB takes ~40 minutes (acceptable for batch job)
- **How to verify**: Compare input file size / elapsed time = throughput

**No Copilot tool calls**: Read existing code comments, implement once, validate.

### Day 1.2-1.3: Filtering & Dataset Streaming

| File | Purpose | Output |
|------|---------|--------|
| `src/position_filters.py` | Filter quiet positions, balance evals, apply material distribution | filtered.bin (24GB) |
| `src/pytorch_dataset.py` | IterableDataset for streaming | Ready for Phase 1 |

**The Why Filtering Matters**:

**Problem**: Raw 1.3M positions include noise:
- Mid-blunder positions (piece hanging, eval wildly swings)
- Duplicates from opening books (same starting position 400 times)
- Unbalanced eval distribution (too many evals near 0, not enough interesting positions)
- Material outliers (positions with unrealistic piece counts from data corruption)

**Think of it like this**: Imagine training a student on chess by showing them 400 copies of the starting position, then 1 endgame. They'd memorize the opening but never learn endings. Filtering is the "curriculum" — show positions in balanced proportions.

**How Filtering Works**:

1. **Quiet Position Detection**
   - **What**: Only keep positions where no pieces are under immediate attack
   - **Why**: Noisy positions (where a piece just got captured or is hanging) have unstable evaluations
   - **Benefit**: Cleaner labels = better learning (evaluation is "ground truth" not "this move was bad")
   - **Effect**: Removes ~20-30% of positions → 950K positions remain

2. **Eval Balance**
   - **What**: Sample uniformly from eval buckets (0-50cp, 50-100cp, 100-200cp, etc.)
   - **Why**: Raw data has 70% near-equal positions, 25% slightly better, 5% winning positions
   - **Without balancing**: Model learns "most positions are equal" and plays conservatively
   - **With balancing**: Model learns the full spectrum (when positions are good/bad)
   - **Benefit**: Leads to more fighting, ambitious play

3. **Material Distribution Constraints**
   - **What**: Ensure training set includes all piece combinations (K+P vs K, K+R+N, etc.)
   - **Why**: Certain endgames are rare in games but crucial for engine strength
   - **Benefit**: Engine doesn't blunder in unusual positions

4. **Move Count Tracking**
   - **What**: Record how many moves into the game (phase progression)
   - **Why**: Opening positions differ strategically from move 40 middlegame
   - **Benefit**: Can later apply phase-specific training (opening knowledge ≠ endgame knowledge)

**Why a Separate Streaming Dataset?**

**PyTorch's IterableDataset**:
- **What**: Loads positions on-demand, one batch at a time (not all 950K at once)
- **Why**: Memory efficient — GPU might have 12GB VRAM, but dataset is 100GB
- **Metaphor**: Like a library that streams one page per request, not loading entire books
- **Benefit**: Can train on datasets larger than available RAM

**Validation**: Run on 1GB sample, verify no memory overflow.

**No Copilot tool calls**: Copy patterns from existing codebase, test locally.

---

## Sprint 2: Architecture & Feature Engineering (HalfKA) (3-5 days)

**The What**: Convert raw board positions (64 squares, 12 piece types) into **HalfKA sparse features** — the language the neural network speaks.

**The Why**: 
- **Raw board = 768 possible inputs** (64 squares × 12 piece types) — wasteful, most are zero
- **HalfKA = 45K sparse features** — only active features get computed, massive speedup
- **Incremental updates = game changer** — only 2-3 pieces move per turn, so recompute only those features
- **Stockfish uses this** — battle-tested by the world's best engine

**The Big Picture**: Imagine instead of memorizing "there's a white pawn on e2" (raw data), the network learns "when my king is on g1 and opponent's king is on g8, having a pawn on e4 is good" (feature context). HalfKA encodes this relationship.

### Day 2.1-2.2: Python HalfKA Feature Generator

| File | Purpose | Output |
|------|---------|--------|
| `src/halfdka_features.py` | (Piece, Square, King) → index mapping | Feature indices (sparse) |
| **Sparse Mapping** | 45,056 features per side (vs 55 original) | King buckets: 8-32 zones |

**HalfKA Explained: The "Half King-Piece-Square" Indexing**

Standard approach (naive):
```
Each piece on board = one feature
64 squares × 6 piece types × 2 colors = 768 features per side
Problem: Most positions have <16 pieces, wasting 750+ feature slots
```

HalfKA (smart approach):
```
Feature = (Piece Type, Piece Square, Friendly King Square)

Example: 
  "White pawn on e4, with white king on g1" = Feature #12847
  "White pawn on e4, with white king on e8" = Feature #15291 (different!)
  
Why? Because pawn value depends on king proximity (king-pawn endgames vs midgame tactics)

Math:
  - 6 piece types × 64 squares × 32 king zones = 12,288 features per piece per perspective
  - 4 pieces (P, N, B, R) × 12,288 = 49,152 (round to 45,056 after bucketing)
```

**King Buckets (32 zones)**:
- **What**: Divide the 64 squares into 32 regions (like a chess "zip code")
- **Why**: Don't need to distinguish g1 vs h1 (both "kingside" strategy), saves memory
- **Benefit**: Reduces feature space from 49K to 45K without losing strategic info

**Implementation in `halfdka_features.py`**:

```python
def get_halfdka_index(piece_type, piece_square, king_square) -> int:
    """
    Compute single HalfKA feature index.
    
    Args:
        piece_type: 0=Pawn, 1=Knight, 2=Bishop, 3=Rook, 4=Queen, 5=King
        piece_square: 0-63 (a1 to h8)
        king_square: 0-63 (king position)
    
    Returns: Integer 0-45055 (unique feature index)
    
    Why this mapping?
    - (King on g1, Pawn on e4) should map to different index than (King on e8, Pawn on e4)
    - Same pawn, different strategic value based on king proximity
    """
    king_bucket = get_king_bucket(king_square)  # 32 zones
    piece_index = piece_type * (64 * 32) + piece_square * 32 + king_bucket
    return piece_index % 45056

def get_active_features(board) -> List[int]:
    """
    Return list of active HalfKA indices for current position.
    
    Example output: [102, 5432, 8843, 12847, ...]  (only ~30-32 features)
    
    Why sparse?
    - Max 32 pieces on board
    - So max 32 active features
    - Instead of computing all 45K, only touch 32
    - ~1000x faster than dense computation
    """
    features = []
    our_king_square = board.king(board.turn)
    
    for piece_type in range(6):
        for piece_square in board.pieces(piece_type, board.turn):
            idx = get_halfdka_index(piece_type, piece_square, our_king_square)
            features.append(idx)
    
    return features
```

**Validation**: Test on 1000 positions, verify ~30-32 features active per position
- **Why this validation?**: Confirms sparsity assumption — if I get 5000 active features, something's broken
- **How to test**: 
  ```python
  positions = load_1000_positions()
  for pos in positions:
      features = get_active_features(pos)
      assert 20 < len(features) < 35  # Most have 24-32 pieces
  ```

**Decision Gate**: If performance acceptable (<100ms per batch of 1000), proceed. Otherwise, implement C++ in Phase 0.5.
- **Why C++?**: If Python is too slow, C++ is 10-100x faster (compiled, no interpreter overhead)
- **When to decide**: After benchmarking, not before

### Day 2.3-2.4: Perspective Accumulator Design

Create `src/accumulator_architecture.py`:

**The What**: Dual accumulators (one for White's perspective, one for Black's) that transform sparse features into a 1024-2048 dimensional "position summary."

**The Why**: 
- **Sparse features are not positions yet** — [102, 5432, 8843] is just a list of piece locations
- **Accumulators compress this** — turn 45K sparse indices into 1024 dense numbers
- **Dual perspective** — White evaluates positions differently than Black (natural chess asymmetry)
- **ClippedReLU bounds** — prevents unbounded activations (makes INT8 quantization later possible)

**The Big Picture**: Think of accumulators as "feature compressors." They take "pawn on e4, king on g1, bishop on f1..." and compress it into a 1024-number summary, like a lossy photo compression — loses pixel details but keeps the strategic essence.

**Architecture**:
- Dual accumulators (white/black perspective)
  - **What**: Two separate 1024-2048 neuron networks (one for each side's evaluation)
  - **Why**: White's king on g1 means different things to White than to Black
  - **Benefit**: Captures asymmetric chess knowledge (White wins on kingside ≠ Black wins on kingside)
  
- 1024-2048 neuron size (tunable)
  - **Why 1024?** Stockfish uses 1024, proven in practice
  - **Why tunable?** Can trade off speed (smaller) vs expressiveness (larger)
  - **Memory impact**: 1024 neurons × 2 perspectives × 4 bytes float = 8KB per batch sample
  
- ClippedReLU activation (0, 1 bounded)
  - **What**: ReLU activation but clamped to [0, 1]
  - **Why**: Unbounded activations cause overflow in INT8 quantization
  - **Formula**: `max(0, min(1, x))`
  - **Benefit**: Naturally bounded = safer quantization later

**Test**: Verify perspective symmetry preserves evaluation
- **What this means**: Position A from White's view should have opposite evaluation to same position from Black's view
- **How to test**:
  ```python
  eval_white = accumulator_white(position_features)
  eval_black = accumulator_black(flip_perspective(position_features))
  assert eval_white ≈ -eval_black  # Should be opposite
  ```
- **Why it matters**: If symmetry breaks, model is learning nonsense

### Network Architecture Explained

```
45K sparse inputs (HalfKA per perspective)
    ↓ [Why sparse? Only 2-3 pieces move per turn, recompute only those]
Perspective Accumulators (1024-2048 neurons each)
    ↓ [Why dual? White king on g1 ≠ Black king on g1 strategically]
ClippedReLU (0, 1 bounded for INT8 later)
    ↓ [Why clamped? Prevents numerical overflow during quantization]
Hidden layers (128 → 32 neurons)
    ↓ [Why compress further? Reduce overfitting, speed up final layer]
    ├─ Strength Head (MSE, 70%)
    │  └─ Single output: evaluation score (centipawns)
    │     Why MSE loss? Continuous target (0, +500, -1200, etc.)
    │
    ├─ Character Head (CE, 20%)
    │  └─ Multi-class output: which move the engine chose (move legality)
    │     Why CE loss? Categorical target (4000 possible moves)
    │     Why "character"? Captures personality (aggressive vs defensive play)
    │
    └─ WDL Head (CE, 10%)
       └─ 3-class output: Win/Draw/Loss outcome (from Syzygy endgames)
          Why CE loss? 3-class classification
          Why WDL? Teaches endgame truth (king vs king = draw, not "equal eval")

Total loss = 0.7 × strength_loss + 0.2 × character_loss + 0.1 × wdl_loss
Weighted toward strength (evals) because that's the main signal.
```

**Why three heads instead of one?**

Metaphor: Training a chess student with three teachers:
1. **Strength teacher** (70%): "That position is worth +300 centipawns for you"
2. **Character teacher** (20%): "A strong player would move the rook here, not the bishop"
3. **WDL teacher** (10%): "This is a drawn position (perfect endgame knowledge)"

Each teacher pushes the student in slightly different directions. Combined, they create a well-rounded player.

---

## Sprint 3: Training Loop & Personality Preservation (2 weeks)

**The What**: Train the neural network on 950K filtered chess positions using three simultaneous loss signals.

**The Why**:
- **Single loss (just eval) = weak engine**: Learns to evaluate but doesn't learn *how to move*
- **Three-head training = balanced skill**: Evaluation + move selection + endgame truth
- **Personality preservation = character**: Adding "move preference" loss keeps the engine's style

**The Big Picture**: This is the learning phase. I'm showing the network millions of examples: "When the position looks like THIS (HalfKA features), the correct evaluation is THIS, the best move is THIS, and the outcome was THIS." The network adjusts its weights to minimize prediction errors across all three objectives simultaneously.

### Day 3.1-3.2: Multi-Signal Loss Function

Create `src/training_loss.py`:

**Loss Functions Explained**:

Each head in the network optimizes a different objective:

```python
class MultiSignalLoss(nn.Module):
    def __init__(self, strength_weight=0.7, character_weight=0.2, wdl_weight=0.1):
        self.strength_loss = nn.MSELoss()         # Continuous: eval is -32000 to +32000
        self.character_loss = nn.CrossEntropyLoss()  # Categorical: which of 4000 moves?
        self.wdl_loss = nn.CrossEntropyLoss()    # Categorical: Win/Draw/Loss?
    
    def forward(self, predictions, evals, moves, wdls):
        # predictions = [batch, 3 heads] from network
        # evals = ground truth evaluation (centipawns)
        # moves = ground truth move played
        # wdls = ground truth outcome (0=draw, 1=win, -1=loss)
        
        loss_strength = self.strength_loss(predictions[:, 0], evals)
        # MSE = (predicted_eval - actual_eval)^2
        # Example: if we predicted +200 but eval was +500, loss = (200-500)^2 = 90,000
        
        loss_character = self.character_loss(predictions[:, 1:], moves)
        # CrossEntropy = -log(prob of correct move)
        # Example: if engine played e2-e4 (move #1205), we penalize if output doesn't favor move #1205
        
        loss_wdl = self.wdl_loss(predictions[:, -3:], wdls)
        # 3-class cross entropy (output has 3 neurons: win probability, draw, loss)
        # Example: if outcome was a draw, penalize the network for predicting "win"
        
        return (0.7 * loss_strength + 0.2 * loss_character + 0.1 * loss_wdl)
```

**Why these weights (0.7, 0.2, 0.1)?**

| Signal | Weight | Why |
|--------|--------|-----|
| **Strength (eval)** | 70% | This is the core: "Is this position good/bad?" Evaluation is the primary objective |
| **Character (moves)** | 20% | Important but secondary: "Which move to play?" Moves are derivative of evaluation |
| **WDL (outcome)** | 10% | Ground truth but rare: Syzygy only applies ≤7 pieces (~5% of training data) |

**Real-world analogy**:
- Strength = "Am I thinking deeply about the position?"
- Character = "Do my move choices match strong players?"
- WDL = "In known endgames, do I understand the truth?"

**Validation**: Verify all three losses are decreasing independently
- **What this means**: 
  ```
  Epoch 1: strength_loss=1500, character_loss=8.2, wdl_loss=0.95
  Epoch 2: strength_loss=1200, character_loss=7.8, wdl_loss=0.92
  Epoch 3: strength_loss=950,  character_loss=7.4, wdl_loss=0.89
  All decreasing ✓ Good training
  ```
- **If one stops decreasing**, something's broken (maybe that signal's data is corrupted)

### Day 3.3-3.4: Training Loop

Create `src/train.py`:

**What it does**:
- **DataLoader from Phase 1 IterableDataset**: Streams positions from Parquet on-the-fly (no full load into RAM)
- **Gradient accumulation (16-32 steps)**: Update weights every 16-32 batches instead of every batch
  - Why? If batch size is 256, gradient accumulation of 32 = effective batch of 8192 (more stable gradients)
  - Benefit: More stable training, less noisy weight updates

- **Learning rate scheduler (cosine annealing)**: Learning rate starts high, slowly decreases to near-zero
  - Why? Early epochs need aggressive learning (big weight changes), late epochs need fine-tuning (tiny changes)
  - Schedule: Learning rate = base_lr × cos(π × current_step / total_steps)
  - Benefit: Avoids getting stuck in local minima (high lr early), prevents overshooting (low lr late)

- **Checkpoint saving every N steps**: Save model weights every 1000 steps
  - Why? If training crashes on step 50,000, don't restart from step 0
  - Benefit: Resume training from last checkpoint

- **Stop condition**: ELO measurement every epoch (via tournament)
  - Why? Training loss alone doesn't tell me if the engine is *good* (only if it matches data)
  - How? Run model against baseline engine, measure win rate, convert to ELO gain
  - Example: If new model beats baseline 55%, that's roughly +50 ELO

**Training Loop Pseudocode**:
```python
for epoch in range(num_epochs):
    for batch_idx, (features, evals, moves, wdls) in enumerate(dataloader):
        
        # Forward pass
        predictions = model(features)
        loss = loss_fn(predictions, evals, moves, wdls)
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation: only update weights every 32 batches
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    
    # Checkpoint
    save_checkpoint(model, optimizer, epoch)
    
    # Measure ELO
    elo_gain = measure_tournament(model, baseline_model, num_games=50)
    print(f"Epoch {epoch}: Loss={loss:.4f}, ELO={elo_gain:+.1f}")
    
    if elo_gain < 0:
        print("WARNING: Model getting worse, might want to stop")
```

**Why elo measurement?**
- Training loss can decrease while ELO decreases (overfitting to training data, loses to new positions)
- ELO is the real metric (how strong is the engine in practice?)
- If ELO stops improving for 3 epochs, stop training (diminishing returns)

### Day 3.5-3.6: Monitoring & Validation

Use existing `monitoring_performance_tracker.py`:

**Metrics to track**:
- **GPU memory usage**: If it spikes to 99%, training slows down (page faulting)
  - Goal: Keep <80% for stable performance
  
- **Training throughput (pos/sec)**: How many positions processed per second?
  - Goal: >10K pos/sec on modern GPU (faster = cheaper training)
  - Too slow? Indicates bottleneck in I/O or compute
  
- **Loss curves (separate plots for 3 signals)**:
  - Plot strength_loss, character_loss, wdl_loss on same graph
  - Healthy training: all three decrease smoothly
  - Red flag: one plateaus while others decrease (data quality issue for that signal)

**Example metrics dashboard**:
```
Epoch 1:  Loss=2.1  | Strength=1.8  | Character=8.2  | WDL=0.95  | ELO=+15  | GPU=45%
Epoch 2:  Loss=1.8  | Strength=1.5  | Character=7.8  | WDL=0.92  | ELO=+42  | GPU=48%
Epoch 3:  Loss=1.6  | Strength=1.2  | Character=7.4  | WDL=0.89  | ELO=+68  | GPU=52%
Epoch 4:  Loss=1.5  | Strength=1.1  | Character=7.1  | WDL=0.88  | ELO=+88  | GPU=55%
Epoch 5:  Loss=1.4  | Strength=1.0  | Character=6.8  | WDL=0.87  | ELO=+102 | GPU=58%
```

All metrics moving in right direction = training is healthy.

**Do NOT use Copilot**: Run locally, analyze logs, iterate.

---

## Sprint 4: Syzygy Integration & Production Scaling (1 week)

**The What**: Two things: (1) Perfect endgame labels from Syzygy tablebases, (2) Compress the model from float32 to int8 for speed.

**The Why**:
- **Syzygy integration**: Positions with ≤7 pieces can be looked up in precomputed tables (100% accurate WDL ground truth)
- **INT8 quantization**: Model weights as 1-byte integers instead of 4-byte floats = 4x smaller, 2-4x faster

**The Big Picture**: I've trained a model using imperfect data (human evaluations, game outcomes). Now I'm adding two refinements: perfect endgame knowledge (from Syzygy), and making the model's brain more efficient (quantization) so it can think faster during actual play.

### Day 4.1-4.2: Syzygy Integration

Use existing `src/syzygy_integration.py`:

**What Syzygy is**:
- **Syzygy endgame tablebase**: Precomputed database of ALL positions with ≤7 pieces
  - 5 pieces = 300 billion positions, all solved (perfect play = draw, white wins, or black wins)
  - 6-7 pieces = trillions of positions, all solved
  
**Why integrate Syzygy?**:
```
Problem: Training data has human labels or engine evals (imperfect)
  Example: Position with K+Q vs K+R might be labeled "white wins" 
           but actually proven draw with perfect play
           
Solution: For every position in Ir training set with ≤7 pieces,
          look up the TRUE outcome from Syzygy
          
Result: Training sees "this is actually a draw, human eval was wrong"
```

**How it works**:
```python
def integrate_syzygy(training_data):
    for position in training_data:
        piece_count = position['piece_count']
        
        if piece_count <= 7:
            # Probe Syzygy
            outcome = syzygy_probe(position['fen'])
            # outcome = 'win', 'draw', or 'loss' (from side-to-move perspective)
            
            # Override training WDL label
            position['wdl'] = outcome  # Replace human label with truth
            position['wdl_source'] = 'syzygy'  # Track that this is ground truth
        else:
            # Keep original label
            position['wdl_source'] = 'game_outcome'
    
    return training_data
```

**Implementation**:
- Probe positions ≤7 pieces with Fathom API (fast C++ library wrapped in Python)
- Replace JSONL evals with WDL/DTZ ground truth
  - **DTZ**: Distance To Zeroing (moves until capture or pawn move, resets 50-move rule)
  - **WDL**: Win/Draw/Loss outcome with perfect play
  
- Track source (Syzygy vs eval) in metadata
  - Why? To distinguish "I know this is absolute truth" vs "I estimate this based on eval"

**Validation**: Check 5-piece endgame accuracy (should be 100%)
- Run model on 5-piece positions, compare predictions to Syzygy ground truth
- Expected: Accuracy >99% (only errors are model approximation)
- If <95%, something's wrong (maybe Syzygy integration failed)

### Day 4.3-4.4: INT8 Quantization

Create `src/quantize_model.py`:

**What is quantization?**
```
Float32: 4 bytes per weight
  Example: weight = 0.123456789 (stored as IEEE 754 float, 32 bits)

INT8: 1 byte per weight
  Example: weight = 0.123456789 → scale and round → 31 (stored as unsigned 0-255)
  Recovery: 31 / 255 ≈ 0.122 (close enough, <1% error)
```

**Why quantize?**
| Metric | Float32 | INT8 |
|--------|---------|------|
| Model size | 4MB | 1MB |
| GPU memory | 4MB | 1MB |
| Inference speed | 1x | 2-4x faster |
| Accuracy | 100% | >99% |
| Training | Yes | No (post-training only) |

**The tradeoff**: Smaller/faster but ~1% accuracy loss. For chess, 1% eval error is negligible.

**How quantization works**:

```python
def quantize_model(model, calibration_data):
    """Convert float32 weights to int8."""
    
    # Step 1: Find the scale factor (max weight range)
    max_weight = max(abs(w) for w in model.weights)
    # Example: if max weight is 0.6, scale factor = 255 / 0.6 ≈ 425
    
    # Step 2: Convert each weight
    quantized_weights = []
    for weight in model.weights:
        # Map from [-0.6, +0.6] to [0, 255]
        int8_weight = round(weight * 425)
        quantized_weights.append(int8_weight)  # Now stored as uint8
    
    # Step 3: Verify clipping ReLU bounds preserved
    for neuron in model.accumulators:
        # ClippedReLU outputs are [0, 1] → scale to [0, 255] → round
        # This is safe because outputs are already bounded
    
    return quantized_weights, scale_factor  # Store scale for inference
```

**Verify ClippedReLU preserves bounds**:
- **What this means**: Accumulator outputs MUST be in [0, 1] for quantization to work
- **Why? Because if outputs were [-5, +5], quantizing to [0, 255] would lose information
- **Test**:
  ```python
  for batch in calibration_data:
      output = accumulator(batch)
      assert 0 <= output <= 1  # Should be true for ClippedReLU
  ```

**Test inference on quantized model**:
```python
# Original model
eval_float32 = model_float32(position)

# Quantized model  
eval_int8 = model_int8(position)

# Compare
error = abs(eval_float32 - eval_int8)
# Expected: <50 centipawns error (out of ±32000 range = 0.15% error)
```

**Target**: <1% ELO loss vs FP32
- Why this metric? Quantization shouldn't make the engine weaker
- How to measure? Run tournament: quantized vs original, measure win rate
- Acceptance: If quantized model scores 49-51% vs original (statistically equal)

### Day 4.5-4.6: Production Deployment

Create `src/inference_wrapper.py` and `src/uci_integration.py`:

**Inference wrapper**:
```python
class QuantizedModelInference:
    def __init__(self, model_path):
        self.model = load_quantized_model(model_path)
        self.scale_factor = load_scale_factor(model_path)
        self.halfdka_encoder = HalfKAFeatures()
    
    def evaluate(self, fen):
        # FEN → HalfKA features → model output → evaluation
        
        # Step 1: Convert FEN to HalfKA sparse features
        board = chess.Board(fen)
        features = self.halfdka_encoder.get_active_features(board)
        
        # Step 2: Forward pass (only 30-32 features active, huge speedup)
        output = self.model(features)
        
        # Step 3: De-quantize (INT8 back to float)
        evaluation_cp = output / self.scale_factor
        
        # Step 4: Return in centipawns
        return int(evaluation_cp)
```

**Why wrapper?**
- Separates neural network internals from UCI communication
- Can swap implementations (C++ later) without changing interface
- Makes testing easier (mock the wrapper)

**UCI Integration**:
- Hook evaluation into existing V7P3R UCI engine
- Evaluation output used in:
  - Move ordering (best moves first in alpha-beta search)
  - Time management (spent more time in equal positions)
  - Personality selection (aggressive vs conservative play)

**Deployment checklist**:
- [ ] Quantized model <5MB (fast loading)
- [ ] Inference <10ms per position on CPU
- [ ] Accuracy >99% vs FP32 model
- [ ] ELO ≥95% of unquantized model
- [ ] No errors on edge cases (weird positions, 1 piece, 32 pieces)

**Do NOT use Copilot**: Manual testing required, custom UCI interface.

---

## Token-Efficiency Rules (Mandatory)

### ✅ What to Do
1. **Work file-by-file**: One module per day max
2. **Validate locally**: Run tests in terminal, don't ask Copilot
3. **Copy patterns**: Reuse code structure from existing codebase
4. **One tool call per task**: No exploratory searches
5. **Batch edits**: If multiple changes needed, use multi_replace

### ❌ What NOT to Do
1. **No semantic_search**: Search manually in existing files
2. **No file_search**: I know my codebase structure
3. **No runSubagent**: Do exploration myself
4. **No multiple read_file calls**: Read once, understand, implement
5. **No ask questions**: Make decisions, implement, validate

### 📊 Token Budget Per Sprint
- **Sprint 1**: 1 file creation + 2 edits = 3 tool calls max
- **Sprint 2**: 2 file creations + testing = 4 tool calls max
- **Sprint 3**: 2 file creations + monitoring = 4 tool calls max
- **Sprint 4**: 2 file creations + export = 4 tool calls max

**Total**: ~15 tool calls for entire 4-week plan (vs 50+ if inefficient)