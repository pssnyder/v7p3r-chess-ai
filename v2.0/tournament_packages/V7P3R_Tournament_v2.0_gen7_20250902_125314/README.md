# V7P3R Chess AI v2.0 Tournament Engine

## Tournament Package Information
- **Generation**: 7
- **Package Date**: 2025-09-02 12:53:14
- **Engine Type**: GPU-accelerated neural network with genetic training
- **Protocol**: UCI compatible

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Test the engine:
```bash
python v7p3r_tournament_engine.py
```

## Arena Integration

1. In Arena, go to Engines -> Install New Engine
2. Browse to `v7p3r_tournament_engine.py`
3. Set engine name to "V7P3R v2.0 Gen7"
4. Configure time controls as needed

## Engine Features

- Neural network trained with genetic algorithms
- Advanced bounty-based position evaluation
- GPU acceleration (CUDA)
- Time management for tournament play
- UCI protocol compliant

## Technical Specifications

- Model: 7-generation GPU-trained LSTM
- Input features: 816-dimensional chess position encoding
- Evaluation: Hybrid neural network + tactical bounties
- Time management: Adaptive based on remaining time
- Hardware: Optimized for CUDA-capable GPUs

## Tournament Results

Record your tournament results here:

| Tournament | Date | Score | Rating | Notes |
|------------|------|-------|--------|-------|
|            |      |       |        |       |

