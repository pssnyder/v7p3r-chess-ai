# V7P3R Chess AI 2.0 - RNN-Based Chess Engine

## Overview

V7P3R Chess AI 2.0 represents a complete architectural overhaul of the original chess engine, featuring:

- **Recurrent Neural Network (RNN)** with LSTM cells for move prediction
- **Genetic Algorithm Training** with 128 parallel games per generation
- **Bounty-based Fitness System** with comprehensive chess heuristics
- **Hybrid Evaluation** combining RNN output with tactical bounty scores

## Architecture

### RNN Model (`v7p3r_rnn_model.py`)
- **LSTM Network**: 2-layer LSTM with 512 hidden units
- **Input Features**: 773-dimensional feature vector (board state + metadata)
- **Memory System**: 64-move sequence history for pattern recognition
- **Output**: Move evaluation scores

### Bounty System (`v7p3r_bounty_system.py`)
Implements your specified bounty heuristics:

1. **Center Control**: +3 gold for center squares, +2 for center ring, +1 for central edge, -1 for outer edges
2. **Piece Values**: King +100, Queen +9, Rook +5, Bishop +4, Knight +3, Pawn +1 gold per piece
3. **Attack/Defense**: +1 equal value, +2 higher value, -5 lower undefended attacks
4. **King Safety**: +5 near king attacks, +10 checks, +1000 checkmate
5. **Tactical Patterns**: +5 each for pins, skewers, forks, deflections
6. **Mate Finding**: +100 mate in 1, +500 mate in 2, +1000 mate in 3+
7. **Piece Coordination**: +5 for rook files/ranks, +5 for bishop long diagonals
8. **Castling**: +25 for castling, -10 for losing rights without castling

Additional heuristics:
- Pawn structure bonuses/penalties
- Piece activity and mobility
- Space control and tempo

### Genetic Algorithm Training (`v7p3r_genetic_trainer.py`)
- **Population**: 128 individuals per generation
- **Selection**: Tournament selection with elitism
- **Mutation**: Gaussian noise with configurable rate/strength
- **Crossover**: Uniform crossover between parent genomes
- **Fitness**: Bounty accumulation + win rate bonuses
- **Parallel Evaluation**: Multi-process game simulation

## Installation

1. **Install Dependencies**:
```bash
pip install chess numpy python-chess
```

2. **Install Required Packages**:
```bash
pip install -r requirements.txt
```

## Usage

### Training the AI

Start genetic algorithm training:
```bash
python train_v7p3r_v2.py
```

Quick training mode for testing:
```bash
python train_v7p3r_v2.py --quick
```

Custom parameters:
```bash
python train_v7p3r_v2.py --population 64 --generations 500 --workers 4
```

### Running the AI

UCI mode (for chess GUIs):
```bash
python v7p3r_ai_v2.py --uci
```

Position analysis:
```bash
python v7p3r_ai_v2.py --analyze "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
```

### Demo and Testing

Run all demos:
```bash
python demo_v7p3r_v2.py
```

Test specific components:
```bash
python demo_v7p3r_v2.py --bounty    # Test bounty system
python demo_v7p3r_v2.py --rnn       # Test RNN model
python demo_v7p3r_v2.py --game      # Play demo game
python demo_v7p3r_v2.py --benchmark # Performance test
```

## Configuration

The AI is configured via `config_v2.json`:

### Key Parameters

**RNN Configuration**:
- `hidden_size`: LSTM hidden units (512)
- `num_layers`: LSTM layers (2)
- `sequence_length`: Move history length (64)

**Genetic Algorithm**:
- `population_size`: Individuals per generation (128)
- `mutation_rate`: Probability of parameter mutation (0.15)
- `games_per_individual`: Fitness evaluation games (3)

**Bounty System**:
- `bounty_weight`: Influence of bounty vs RNN (0.3)
- Individual bounty values for all heuristics

**AI Behavior**:
- `search_depth`: Minimax depth fallback (3)
- `time_per_move`: Maximum thinking time (5.0s)

## Genetic Training Process

1. **Population Initialization**: 128 random RNN networks
2. **Fitness Evaluation**: Each network plays games and accumulates bounty
3. **Selection**: Tournament selection favors high-fitness individuals
4. **Reproduction**: Crossover and mutation create offspring
5. **Evolution**: Process repeats for many generations

### Fitness Function
```
fitness = (total_bounty / games_played) + (win_rate * 50) - (avg_game_length - 100) * 0.1
```

## Model Architecture Details

### Feature Extraction (773 dimensions)
- **Board State**: 8×8×12 piece placement (768 features)
- **Game State**: Turn, castling rights, en passant (14 features)
- **Positional**: Center control, development (32 features)
- **Tactical**: Pins, forks, attacks (16 features)
- **History**: Recent move patterns (8 features)

### RNN Network
```
Input (773) → LSTM Layer 1 (512) → LSTM Layer 2 (512) → Output (1)
```

### Bounty Evaluator
- Real-time move evaluation using chess principles
- Combines tactical and positional assessments
- Extensible heuristic system

## Performance Recommendations

### Parameter Tuning

**For stronger tactical play**:
- Increase `bounty_weight` to 0.4-0.5
- Focus on tactical bounty heuristics

**For positional understanding**:
- Increase RNN training generations
- Use longer sequence history

**For faster training**:
- Reduce population size to 64
- Decrease games per individual to 2
- Use quick mode for testing

### Hardware Requirements

**Minimum**:
- 8 GB RAM
- 4 CPU cores
- Training time: ~24 hours for 100 generations

**Recommended**:
- 16 GB RAM
- 8+ CPU cores
- Training time: ~12 hours for 100 generations

## File Structure

```
v7p3r-chess-ai/
├── v7p3r_rnn_model.py        # RNN architecture
├── v7p3r_bounty_system.py    # Bounty evaluation
├── v7p3r_genetic_trainer.py  # Genetic algorithm
├── v7p3r_ai_v2.py           # Main AI implementation
├── train_v7p3r_v2.py        # Training script
├── demo_v7p3r_v2.py         # Demo and testing
├── config_v2.json           # Configuration
├── models/genetic/          # Trained models
├── logs/genetic/            # Training logs
└── reports/genetic/         # Training reports
```

## Extending the AI

### Adding New Bounty Heuristics

1. Extend `ExtendedBountyEvaluator` class
2. Add evaluation method for new heuristic
3. Update configuration with bounty values
4. Test with demo script

### Modifying RNN Architecture

1. Adjust `V7P3RNeuralNetwork` parameters
2. Update feature extraction if needed
3. Retrain with genetic algorithm
4. Validate performance improvements

### Custom Training Scenarios

1. Modify `play_game()` function for specific opponents
2. Add position-specific fitness bonuses
3. Implement opening book integration
4. Add endgame tablebase support

## Advanced Features

### Game Phase Adaptation
- Opening: 40% RNN, 60% bounty (favor principles)
- Middlegame: 70% RNN, 30% bounty (balanced)
- Endgame: 80% RNN, 20% bounty (favor calculation)

### UCI Options
- `BountyWeight`: Adjust bounty vs RNN balance
- `SearchDepth`: Fallback minimax depth
- `Debug`: Enable detailed move analysis

### Training Checkpoints
- Automatic model saving every 10 generations
- Best fitness tracking and visualization
- Resume training from saved states

## Troubleshooting

**Training Issues**:
- Reduce population size if memory limited
- Check parallel worker count vs CPU cores
- Monitor fitness convergence

**Performance Issues**:
- Adjust time per move limit
- Reduce search depth for faster play
- Check model loading path

**UCI Compatibility**:
- Ensure proper engine installation
- Test with supported chess GUIs
- Check UCI protocol compliance

## Contributing

1. Test new bounty heuristics with demo script
2. Validate genetic algorithm improvements
3. Add comprehensive unit tests
4. Document configuration changes

## License

This project is part of the V7P3R Chess AI development series.

---

**V7P3R Chess AI 2.0** - Where recurrent neural networks meet chess mastery through evolutionary algorithms and bounty-driven learning.
