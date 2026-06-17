# Chess AI Project
*by Pat Snyder*
*Created: 2026-06-11*

This project implements an  simple ML model to solve chess puzzles with the goal of providing relatively decent performance at most critical chess positions.

### Overview

#### Phase 1: Architectural Coverage :green_circle: *In Progress*
- Layer 1 (The Lens Stack): Implements a Mixed Convolutional (Inception-style) Layer using `6 parallel, rule-agnostic geometric filter shapes (2 x 2, 3 x 3, 4 x 4, 2 x 6, 1 x 5, 5 x 1) with 16 variations each, yielding 96 total feature channels.`
- Layer 2 (The Synthesizer): Compresses the `96 raw perspectives down to 32 combined channels`, forcing the network to synthesize micro-patterns into high-level structural concepts.

#### Phase 2: Analytics & Visualizing the Machine :blue_circle: *To Do*
- Weight Auditing: Extracting raw floating-point matrices post-training to isolate and identify "dead" vs. highly weighted channels.
- Feature Map Snaps: Exporting 8 x 8 grayscale visual snapshots of Layer 1 and Layer 2 to explicitly show what the model "sees" and "prioritizes" mid-decision.
- The Metaphor Layer: Translating mathematical matrices into strategic narratives to bridge the gap between computer bytes and human understanding.

#### Phase 3: Feedback & Tuning :purple_circle: *To Do*
- Iterative Refinement: Using insights from the analytics phase to adjust architecture, training data, and hyperparameters in a transparent, feedback-driven loop.
- Consider regularization techniques such as nn.Dropout2d to encourage the model to learn more robust, generalizable features rather than overfitting to specific patterns in the training data.
- Implement model training continuation and checkpointing to allow for longer training runs and the ability to resume training from previous models.

#### Phase 4: Dataset Refinement & Expansion :red_circle: *To Do*
- Generate a larger and more diverse dataset of chess puzzles, so that the model is exposed to a wider variety of patterns and relationships, which can help it learn more generalizable features.
- Increase the number of blanks first to reinforce spatial learning, so that the model learns to calculate more distant relationships and patterns rather than relying on local cues.
- Introduce puzzles with multiple solutions to encourage temporal reasoning, so that the model is able to balance out overly confident weights with the understanding that there may be multiple valid paths to a solution.

#### Phase X: Parking Lot :yellow_circle: *Ideation*
- Refine the models convolutional layer using feature selection and regression analysis techniques to pre-analyze our dataset to identify which new features to test and which could potentially be retired.
- Of the 96 channels, identify which are most impactful and determine whether to prune the rest to encourage more efficient learning or expand the layer by replacing and adding new filter shapes to encourage more diverse perspectives.
- Generate more difficult puzzles by eliminating biased patterns introduced during generation.

### Project Updates:
- **2026-06-16**: Initial project setup with data generation and one-hot encoding scripts.
- **2026-06-17**: Basic CNN architecture defined. Training loop implemented with live loss plotting and early stopping.
- **2026-06-18**: Initial training runs completed with preliminary results.