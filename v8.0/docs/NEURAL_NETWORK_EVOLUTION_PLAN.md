# V7P3R Neural Network Architecture Evolution

**Status**: Planning Phase  
**Current Architecture**: 55 features → 256 → 128 → 64 → 1 output  
**Target Architecture**: 768+ features (sparse NNUE-style) with integer quantization  
**GPU Scale Path**: 1M → 10M → 100M → 500M → 1B parameters  
**Last Updated**: 2026-06-09

---

## 1. The Current Problem: ELI5 Data Science

Your network is a **funneling machine** designed to turn human-crafted chess statistics into a winning probability.

```
[55 Board Clues] ➔ [256 Combos] ➔ [128 Combos] ➔ [64 Combos] ➔ [Value: -1 to 1]
```

### The Architecture Components

**The Clues (55 Inputs)**
- You are feeding the network 55 pre-calculated metrics
- Likely includes: material balance, pawn counts, king safety, piece activity, etc.
- These are **human-engineered features** - the network never sees raw board state

**The Blender (Hidden Layers)**
- Layers (256 → 128 → 64) mix these 55 clues to find hidden patterns
- Example learned: "having a queen is good (+9), but having a queen AND an exposed king is terrible"
- Network learns feature interactions, not raw chess concepts
- Each layer learns progressively more complex relationships

**The Output (Value/Tanh)**
- Final `node_tanh` layer squeezes everything into a single score
- Output range: -1 (Black winning) to +1 (White winning)
- Tanh activation smooth but computationally expensive

**The Tensor Bottleneck (GatherND/Mean)**
- Complex slicing, grouping, and averaging of feature vectors before multiplication
- Memory-bandwidth killer for chess evaluation
- Creates cache-busting pattern during single-position inference

---

## 2. Why This Approach Has Critical Flaws for Chess

### Problem 1: The 55-Input Bottleneck
- Spoon-feeding only 55 human-calculated metrics makes the AI blind to raw tactics
- Chess engines need to see all piece placements and relationships
- Missing: piece interactions, square control patterns, king-relative positions
- Result: Network cannot discover tactical motifs (pins, forks, skewers) from raw data

### Problem 2: BatchNorm During Inference
- Batch Normalization designed for batch training (all data together)
- Chess engines evaluate positions **one by one** during search tree
- BatchNorm during single-position inference causes:
  - Significant inference slowdown (extra operations per position)
  - Logical inconsistencies (statistics computed on batch vs single position)
  - Must be completely fused into weights before production deployment
- Solution: Remove or fuse BatchNorm completely before deployment

### Problem 3: Tensor Slicing Speed Killer
- `GatherND`, `transpose`, and `cat` operations create massive memory-bandwidth bottleneck
- Chess evaluation must be lightning-fast (thousands of positions/second during search)
- Complex tensor slicing destroys CPU cache locality
- Cache misses = 100-1000x slower memory access
- Result: Engine searches 10-100x slower than it should be

### Problem 4: Floating-Point Slowness
- Current network relies on standard floating-point operations (f32)
- Thousands of float matrix multiplications per second needed
- Modern CPUs have limited floating-point units compared to integer units
- Float operations cannot use SIMD (AVX2/AVX-512) as efficiently as integers
- Result: Engines 3-5x slower than integer-based equivalents

---

## 3. The True Architectural Solution: NNUE Paradigm

NNUE = **Efficiently Updatable Neural Network** (used by Stockfish, Leela, modern engines)

### Core Principles

#### Principle 1: Massive, Sparse Feature Inputs
Instead of 55 hand-crafted metrics, use **raw piece-square features**:

```
Standard Sparse NNUE:    768 inputs
├─ White pieces: 6 types × 64 squares = 384
└─ Black pieces: 6 types × 64 squares = 384

King-Relative NNUE:     40,960 inputs  
├─ For each of 32 king positions
├─ 64 squares relative to king position
└─ 20 piece/occupancy features per square
```

**Advantage**: Network learns to evaluate piece relationships from raw data, not filtered metrics.

#### Principle 2: Incremental Updates (The Magic Trick)
When a piece moves, **99% of the board stays the same**:

```
Standard Evaluation (SLOW):
  [New board] → [All 768 inputs recalculated] → [Forward pass] ➔ SLOW

Accumulator-Based NNUE (FAST):
  [Old accumulator state] → [Remove piece from old square] → [Add piece to new square] → [Forward pass] ➔ FAST
```

**How it works**:
1. Maintain a running "accumulator" - the sum of weights for all active pieces
2. When a piece moves: 
   - Subtract the weights corresponding to that piece at its old position
   - Add the weights corresponding to that piece at its new position
3. Forward pass uses the updated accumulator (not recalculating all 768 features)

**Speed gain**: 100-1000x faster per position evaluation!

#### Principle 3: Integer Quantization (Int8/Int16)
Ditch floating-point numbers:

```
Float32 Network:
  [Position] → [F32 forward pass] → [Result] (slow, large memory)

Int8 Quantized Network:
  [Position] → [Int8 accumulator] → [Int8 linear layer] → [Int16 output] → [Fast!]
```

**Why integers are faster**:
- CPUs have more integer units than float units
- SIMD instructions (AVX2/AVX-512) execute 4-16x faster on integers
- Int8 weights = 4x less memory (lower cache misses)
- No floating-point arithmetic overhead

**Quantization process**:
1. Train network normally with floats
2. Collect activation distributions (min/max values at each layer)
3. Map floats to integers using scale factors
4. Convert all weights and activations to Int8/Int16
5. Fine-tune with quantization-aware training (optional)

#### Principle 4: Simple Activation Functions
Replace slow `tanh` with fast clipped operations:

```
Tanh (SLOW):        exp(-2x) complex computation
  Output: [-1, 1]

ReLU (FAST):        max(0, x)
  Output: [0, ∞]

Clipped ReLU (FAST + BOUNDED):    max(0, min(1, x))
  Output: [0, 1]

Squared Clipped ReLU (EVEN BETTER): (max(0, min(1, x)))²
  Output: [0, 1], smoother gradient
```

**Why clipped ReLU is better for chess**:
- Single integer comparison (no exponentiation)
- Bounded output (no numerical instability)
- Better for SIMD vectorization
- Faster inference with integer arithmetic

---

## 4. Phase 1: Feature Extraction Expansion

### Current State (55 Features)
Likely includes:
- Material count (pawns, knights, bishops, rooks, queens)
- Pawn structure (passed pawns, isolated pawns, doubled pawns)
- King safety (attacked squares around king, escape squares)
- Piece activity (piece mobility, piece control)
- Basic positional factors

### Target: 768+ Features (Piece-Square Basis)

#### Feature Set Design

**Base Feature: Piece-Square Encoding**
```python
# 768 features = 64 squares × 12 piece types (6 white + 6 black)
# Or: 64 squares × (2 colors × 6 pieces)

Features:
├─ Square 0: [White Pawn, White Knight, White Bishop, ..., Black Queen, Empty]
├─ Square 1: [White Pawn, White Knight, White Bishop, ..., Black Queen, Empty]
├─ ...
└─ Square 63: [White Pawn, White Knight, White Bishop, ..., Black Queen, Empty]

Total: 64 × 12 = 768 binary features (one piece type per square)
```

**Extended Feature Set (Phase 1B)**
Beyond pure piece-square, add:
- **Piece interactions** (pins, attacks, defenses):
  - For each piece: list of squares it attacks
  - For each square: list of pieces attacking it
  - Encoding: 64 squares × 64 attack patterns = 4,096 features
  
- **Pawn structure** (critical for chess):
  - Pawn on each square: 8 features per square × 64 = 512 features
  - For each square: pawn proximity, pawn chains, passed pawns
  
- **King safety** (essential for endgame):
  - King position: 64 features (one for each square)
  - Escape squares: 64 features
  - Attacker positions relative to king: 64 × 8 directions = 512 features
  
- **Phase/Stage** (opening vs middlegame vs endgame):
  - Material count (encode as integer)
  - Piece development score
  - Pawn advancement score

**Recommended Phase 1 Target: 2,000-4,000 features**
- 768 piece-square base
- 1,000 piece interactions
- 512 pawn features
- 512 king safety
- 256 game phase features
- Remaining: custom chess-specific patterns

### Implementation: Bitboard Feature Extraction

**Why Bitboards?**
- Native binary representation (each feature = 1 bit)
- Blazing-fast operations (single CPU instruction)
- Perfect for integer quantization
- Minimal memory footprint

**Example: Piece-Square Features**
```python
# Current (likely): 55 floating-point features (~220 bytes)
# Better: 768 binary features (~96 bytes with bitwise packing)
# Memory reduction: 56% less bandwidth

# Instead of:
features = [
    material_white - material_black,  # float32
    pawn_count_white,                  # float32
    king_safety_score,                 # float32
    # ... 52 more floats
]

# Use:
piece_squares = [
    0b111...111,  # Square 0: which pieces present? (12 bits)
    0b100...001,  # Square 1: which pieces present? (12 bits)
    # ... 62 more squares (64 × 12 bits = 768 bits total)
]
```

**Operations (all bitwise = fast)**
```python
# Count white pieces: popcount(white_pieces & rank_2)
# Find attacked squares: knight_attacks[square] & enemy_knights
# Check if square controlled: (white_attacks | white_pieces) & square_mask
```

---

## 5. Performance Monitoring & Profiling

### Goals
1. **Prevent System Overload**: Monitor GPU/CPU to avoid cooking hardware
2. **Understand Compute Requirements**: Quantify memory, power, speed for each model size
3. **Optimize Bottlenecks**: Identify which layers/operations are slowest
4. **Plan Infrastructure**: Know what GPU/TPU needed for future models

### Monitoring Framework

#### Metrics to Collect

**GPU Metrics**
```
GPU Utilization:      0-100% (target: 80-95%)
GPU Memory:           Current / Max (GB)
GPU Temperature:      Current / Max (°C)
GPU Power Draw:       Current / TDP limit (Watts)
GPU Memory Bandwidth: Current vs peak (GB/s)
```

**CPU Metrics**
```
CPU Utilization:      Per-core utilization (%)
CPU Temperature:      Current / Max (°C)
Cache Efficiency:     L1/L2/L3 hit rates (%)
Context Switches:     Per second (lower = better)
```

**Training Metrics**
```
Batch Processing Time:    ms per batch
Forward Pass Time:        ms per forward pass
Backward Pass Time:       ms per backward pass
Loss/Accuracy:           Current training metrics
Learning Rate:           Current LR for adjustment
Gradient Norm:           Monitor for exploding gradients
```

**Memory Metrics**
```
System RAM:          Current / Total (GB)
GPU VRAM:            Current / Total (GB)
Model Weights:       Size of model (MB)
Activation Cache:    Size of intermediate outputs (MB)
Total Memory Peak:   Maximum during training (GB)
```

**Inference Metrics**
```
Positions/Second:    Throughput
Latency/Position:    ms per position
Batch Size:          Optimal batch size for throughput
Cache Hit Rate:      CPU cache efficiency
```

#### Monitoring Implementation

```python
# monitoring/performance_tracker.py

class PerformanceMonitor:
    """Real-time performance tracking for chess NN training"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = time.time()
    
    def track_gpu(self):
        """GPU utilization, memory, temperature"""
        try:
            # Using nvidia-ml-py library
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, 0)
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Watts
            
            return {
                'gpu_util': util.gpu,
                'gpu_mem_used': mem.used / 1e9,  # GB
                'gpu_mem_total': mem.total / 1e9,
                'gpu_temp': temp,
                'gpu_power': power
            }
        except Exception as e:
            return {}
    
    def track_cpu(self):
        """CPU utilization, temperature, memory"""
        return {
            'cpu_util': psutil.cpu_percent(interval=0.1),
            'cpu_temp': psutil.sensors_temperatures()['coretemp'][0].current,
            'ram_used': psutil.virtual_memory().used / 1e9,
            'ram_total': psutil.virtual_memory().total / 1e9,
        }
    
    def track_training(self, loss, accuracy, batch_time):
        """Training metrics"""
        return {
            'loss': loss,
            'accuracy': accuracy,
            'batch_time_ms': batch_time * 1000,
            'positions_per_sec': 64 / (batch_time + 1e-6)  # Assuming 64-pos batch
        }
    
    def log(self, **kwargs):
        """Log metrics"""
        timestamp = time.time() - self.start_time
        for key, val in kwargs.items():
            self.metrics[key].append((timestamp, val))
    
    def report(self):
        """Generate performance report"""
        report = {
            'duration_minutes': (time.time() - self.start_time) / 60,
            'avg_gpu_util': np.mean([v for _, v in self.metrics.get('gpu_util', [])]),
            'max_gpu_temp': np.max([v for _, v in self.metrics.get('gpu_temp', [])]),
            'peak_gpu_mem': np.max([v for _, v in self.metrics.get('gpu_mem_used', [])]),
            'avg_positions_per_sec': np.mean([v for _, v in self.metrics.get('positions_per_sec', [])]),
            'throughput_gb_per_sec': np.mean([v for _, v in self.metrics.get('throughput_gb_s', [])]),
        }
        return report
```

---

## 6. Phased Scaling Strategy

### Phase 1: Baseline (Current Architecture + Enhanced Features)
```
Current: 55 features → 256 → 128 → 64 → 1
Target:  2000 features → 256 → 128 → 64 → 1

Parameters: ~600K
Timeline: 2 weeks
Goals:
  ✓ Expand feature extraction beyond 55 metrics
  ✓ Add comprehensive monitoring
  ✓ Validate performance improvements
  ✓ Establish baseline metrics for comparison
```

### Phase 2: 1M Parameters (First Scale)
```
Architecture: 2000 → 512 → 256 → 128 → 64 → 1
Parameters: ~1.3M

Timeline: 3 weeks
Changes:
  ✓ Increase first hidden layer (256 → 512)
  ✓ Add feature interaction layers
  ✓ Implement integer quantization experiments
  ✓ Optimize tensor operations (remove GatherND)

Target Performance:
  - Positions/sec: 10K+ (from current ~5K?)
  - GPU memory: <2GB
  - Training time: <24 hours for 1M positions
```

### Phase 3: 10M Parameters
```
Architecture: 3000 → 1024 → 512 → 256 → 128 → 64 → 1
Parameters: ~10M

Timeline: 4 weeks
Changes:
  ✓ Expand to NNUE accumulator architecture
  ✓ Implement piece-square features (768+)
  ✓ Full integer quantization (Int8)
  ✓ Simple activation functions (Clipped ReLU)
  ✓ Incremental update capability

Target Performance:
  - Positions/sec: 50K+ (with accumulator)
  - GPU memory: <4GB
  - Inference latency: <1ms per position
```

### Phase 4: 100M Parameters
```
Architecture: 4000 → 2048 → 1024 → 512 → 256 → 128 → 1
Parameters: ~100M

Timeline: 6 weeks
Hardware: May require GPU upgrade (RTX 4080/5090 or A100)
Changes:
  ✓ Deep NNUE with multiple refinement layers
  ✓ Dual heads (eval + winrate)
  ✓ Advanced feature engineering (king-relative positions)
  ✓ Mixed precision training (float16 + Int8)

Target Performance:
  - Positions/sec: 100K+
  - GPU memory: <8GB
  - Search depth: 25-30 plies with time management
```

### Phase 5: 500M - 1B Parameters
```
Parameters: 500M - 1B

Timeline: 8-12 weeks
Hardware: Enterprise GPU (H100) or distributed training
Architecture:
  ✓ Massive NNUE with full king-relative features (40K+ inputs)
  ✓ Auxiliary tasks (winrate prediction, move ordering)
  ✓ Multi-head attention mechanisms (optional)
  ✓ Knowledge distillation from Stockfish/Leela

Expected Result:
  - 2500+ ELO engine
  - Positions/sec: 200K+
  - Competitive with Stockfish 16
```

---

## 7. Implementation Roadmap

### Immediate Actions (Week 1)

- [ ] Create `neural_network/feature_extractor.py`
  - [ ] Implement 768-feature piece-square encoding
  - [ ] Implement piece interaction features
  - [ ] Implement king safety features
  - [ ] Add bitwise operations for speed
  
- [ ] Create `monitoring/performance_tracker.py`
  - [ ] GPU monitoring (NVIDIA, AMD, Apple Metal)
  - [ ] CPU monitoring (temperature, utilization)
  - [ ] Training metrics collection
  - [ ] Real-time alerting for overheating

- [ ] Create `training/model_v2.py`
  - [ ] Update architecture with 2000 input features
  - [ ] Remove GatherND operations
  - [ ] Fuse BatchNorm if present
  - [ ] Add simple activation functions

### Short-term (Weeks 2-4)

- [ ] Implement accumulator-based updates (foundation for NNUE)
- [ ] Add integer quantization pipeline
- [ ] Create benchmark suite for inference speed
- [ ] Document feature engineering decisions
- [ ] Run Phase 1 training with monitoring

### Medium-term (Weeks 5-8)

- [ ] Full NNUE refactor (Phase 3)
- [ ] King-relative feature implementation
- [ ] Multi-threaded inference optimization
- [ ] SIMD optimizations (AVX2/AVX-512)
- [ ] Distributed training setup (if needed)

### Long-term (Weeks 9+)

- [ ] Scale to 100M+ parameters
- [ ] Multi-GPU training if hardware allows
- [ ] Knowledge distillation from strong baseline engines
- [ ] Tournament validation against real opponents

---

## 8. Expected Performance Gains

### Current Limitations
- 55 features = chess-blind
- BatchNorm slows inference
- Tensor slicing destroys cache
- Float operations slow

**Current Speed**: ~5K positions/second (estimated)  
**Current Strength**: ~1550 ELO (Lichess)

### Phase 1 (2000 features)
- **Speed**: 5K → 8K positions/sec (+60%)
- **ELO**: 1550 → 1650 (+100)
- **Reason**: Better feature representation, less overfitting

### Phase 2 (1M parameters)
- **Speed**: 8K → 15K positions/sec (+87%)
- **ELO**: 1650 → 1750 (+100)
- **Reason**: Larger model capacity, integer ops

### Phase 3 (10M parameters + NNUE)
- **Speed**: 15K → 100K positions/sec (+567%)
- **ELO**: 1750 → 1850 (+100)
- **Reason**: Incremental updates, SIMD optimization

### Phase 4 (100M parameters)
- **Speed**: 100K → 150K positions/sec (+50%)
- **ELO**: 1850 → 1950 (+100)
- **Reason**: Deeper evaluation, better pattern recognition

### Phase 5 (500M-1B parameters)
- **Speed**: 150K → 300K+ positions/sec (+100%)
- **ELO**: 1950 → 2100+ (+150+)
- **Reason**: Master-level evaluation, deep understanding

---

## 9. Key Technical Decisions

### Decision 1: NNUE vs Transformer
**Choice**: NNUE (not Transformer)
**Reasoning**:
- NNUE proven for chess (Stockfish, etc.)
- Transformers slower for single-position inference
- NNUE more interpretable (piece-square basis)
- Integer quantization easier with NNUE

### Decision 2: Accumulator Architecture
**Choice**: Incremental updates from move to move
**Implementation**:
- Maintain running accumulator (layer 1 activations)
- On move: subtract old piece, add new piece
- 100-1000x speedup vs recalculating all features

### Decision 3: Quantization Strategy
**Choice**: Int8 quantization with scale factors
**Why**:
- 4x smaller model (faster loading, less memory)
- 3-5x faster inference (SIMD on integers)
- Sufficient precision for chess evaluation

### Decision 4: Activation Functions
**Choice**: Clipped ReLU → Squared Clipped ReLU
**Why**:
- Single integer comparison (no exponentiation)
- Bounded output (no numerical issues)
- SIMD-friendly
- Better gradient flow than ReLU

---

## 10. Success Metrics

### Training Metrics
- [ ] Loss convergence: Target <0.01 MAE
- [ ] Validation accuracy: Target >90% win prediction
- [ ] Feature quality: Piece-square features learn chess concepts
- [ ] Gradient health: No exploding/vanishing gradients

### Performance Metrics
- [ ] Inference speed: 100K+ positions/sec by Phase 3
- [ ] Memory efficiency: <4GB GPU memory for 10M params
- [ ] Cache efficiency: >80% CPU L3 cache hit rate
- [ ] Power efficiency: <200W sustained GPU power

### Chess Strength Metrics
- [ ] ELO progression: 1550 → 2100+ across phases
- [ ] Tactical solving: >95% accuracy on Lichess puzzles
- [ ] Search efficiency: Depth increase without slowdown
- [ ] Time management: Effective use of remaining time

---

## References & Further Reading

- **NNUE Architecture**: Stockfish GitHub (NNUE architecture documentation)
- **Integer Quantization**: TensorFlow Lite Quantization Guide
- **AVX2/SIMD**: Intel Intrinsics Guide
- **Accumulator Pattern**: Stockfish incrementally_add_piece()
- **Clipped ReLU**: arxiv.org/abs/1803.04579 "An Empirical Study of Modern Convolutional Networks"

---

**Document Status**: Architecture Planning Complete  
**Next Step**: Begin Phase 1 implementation  
**Estimated Total Timeline**: 3-4 months to reach Phase 4 (100M parameters)
