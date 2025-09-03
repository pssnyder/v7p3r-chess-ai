# V7P3R Chess AI

A reinforcement learning chess AI that learns from human play style and improves over time.

## Overview

V7P3R Chess AI is a chess engine that uses reinforcement learning to develop its playing style. The AI evaluates positions, considers various features of the chess board, and uses rewards and penalties to learn which moves lead to better outcomes. What makes this AI unique is that it also learns from the historical games of the human player it's named after, incorporating human-like playing patterns and style.

## Features

- Reinforcement learning model for move selection
- Personalized play style based on V7P3R's historical games
- Focus on checkmate patterns from winning games
- Adaptable style influence from opening to endgame
- Static evaluation as a fallback for untrained models
- Visual game watcher for real-time game viewing
- Training against Stockfish at various ELO levels
- Model validation to track performance
- Game statistics and history

## Requirements

- Python 3.8+
- Stockfish chess engine
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/v7p3r_chess_ai.git
   cd v7p3r_chess_ai
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Download Stockfish chess engine and place the executable in the project directory or update the config file with its path.

## Usage

### Play a Game

To play a game between V7P3R and Stockfish:

```
python play_game.py
```

This will start a game with the settings specified in `config.json`.

### Train the AI

To run the complete training workflow (analyze, train, validate, and report):

```
python train_v7p3r.py
```

Options:
- `--analyze-only`: Only analyze personal games, don't train
- `--train-only`: Only train model, don't analyze
- `--validate-only`: Only validate model, don't train
- `--episodes N`: Set custom number of training episodes
- `--no-reports`: Skip generating reports
- `--pgn-path PATH`: Specify custom path to PGN file

To train just the V7P3R model without the workflow:

```
python v7p3r_training.py
```

The model will be saved to the path specified in the configuration.

### Validate the Model

To validate the performance of the V7P3R model:

```
python v7p3r_validation.py
```

This will run test games against Stockfish at various ELO levels.

### Analyze Personal Game History

To analyze your personal game history and generate insights:

```
python analyze_personal_games.py
```

This tool will:
- Analyze your games from `data/v7p3r_games.pgn`
- Generate statistics on your playing style
- Create visualizations of game patterns
- Output detailed analysis to `data/analysis/` folder

The analysis provides insights into:
- Win/loss ratios
- Opening preferences
- Piece activity patterns
- Checkmate patterns
- Game length distribution
- Move preferences

### Simulate Games

To simulate and record games between V7P3R and Stockfish:

```
python simulate_games.py
```

Options:
- `--games N`: Number of games to simulate (default: 1)
- `--elo N`: Stockfish ELO rating (default: 400)
- `--v7p3r-color [white|black|both]`: Which color V7P3R plays as (default: both)
- `--time-per-move SECONDS`: Seconds per move, 0 for instant (default: 1.0)
- `--verbose`: Print each move
- `--save-dir DIR`: Directory to save PGN files (default: simulations)

This tool is useful for quickly testing the AI's performance and generating games for analysis.

### Extended Training and Validation

For more intensive training and comprehensive validation:

```
# Extended training with more episodes
python train_v7p3r.py --train-only --episodes 500

# Extended validation across multiple ELO levels
python extended_validation.py
```

The extended validation script will:
- Test V7P3R against multiple Stockfish ELO levels
- Play multiple games at each level
- Generate detailed performance reports
- Save game PGNs for later analysis

Options for extended validation:
- `--elo-levels`: Comma-separated ELO levels to test against (default: 400,800,1200,1600)
- `--games-per-level`: Number of games per ELO level (default: 5)
- `--time-per-move`: Seconds per move (default: 0.5)
- `--output-dir`: Directory for validation results (default: reports/extended_validation)

You can also monitor training progress in a separate terminal:

```
python monitor_training.py
```

This script will:
- Check if training is still active
- Run extended validation automatically when training completes
- Provide periodic status updates

To visualize training progress and results:

```
python visualize_training.py
```

This creates visualizations showing:
- Win rate over time
- Performance against different ELO levels
- Training progress metrics

### Watch a Game

To watch an ongoing game visually:

```
python active_game_watcher.py
```

This will display the current state of the game in progress.

## Personalization

What makes V7P3R unique is its ability to learn from and incorporate the play style of a specific human player:

- **Personal Game History**: The AI analyzes games from `data/v7p3r_games.pgn` to learn opening preferences, tactical patterns, and successful checkmate sequences.
  
- **Style Influence**: Configurable weights determine how much influence the personal style has on different game phases (opening, middlegame, endgame).
  
- **Checkmate Focus**: The AI pays special attention to successful checkmate patterns from winning games, allowing it to recognize and replicate winning tactics.

- **Humanistic Play**: The system can be tuned to play more like a human by occasionally selecting moves that mirror the player's style, even when not optimal according to the engine evaluation.

## Configuration

All settings are managed in the `config.json` file:

- `game_config`: Settings for game play (player selection, game count)
- `v7p3r_config`: V7P3R AI settings (model path, search depth, personal style weights)
- `stockfish_config`: Stockfish settings (ELO rating, path)
- `training_config`: Training parameters (learning rate, exploration rate)
  - `rewards`: Reward values for different actions
  - `penalties`: Penalty values for different actions
  - `feature_weights`: Weights for different board features

## Project Structure

- `active_game_watcher.py`: Visual display of the current game
- `analyze_personal_games.py`: Tool for analyzing personal game history
- `chess_core.py`: Core chess functionality and board evaluation
- `personal_style_analyzer.py`: Analyzes human games for play patterns
- `play_game.py`: Main game orchestration
- `simulate_games.py`: Simulation tool for testing and recording games
- `stockfish_handler.py`: Interface with Stockfish engine
- `train_v7p3r.py`: Complete training workflow (analyze, train, validate)
- `v7p3r_ai.py`: V7P3R AI implementation
- `v7p3r_training.py`: Reinforcement learning training
- `v7p3r_validation.py`: Model validation and performance testing
- `config.json`: Configuration settings
- `requirements.txt`: Python dependencies
- `docs/`: Documentation files
- `images/`: Chess piece images
- `data/`: Game data including personal game history

## License

MIT License

## Acknowledgements

- Python-chess library
- Stockfish chess engine