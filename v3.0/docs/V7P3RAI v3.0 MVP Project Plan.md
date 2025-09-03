# **V7P3RAI v3.0 MVP Project Plan: The "Derks Battlegrounds" Chess AI**

## **1\. Product Vision**

To build a chess engine that learns and plays like an interactive video game, treating the board as a sequence of frames and pieces as weapons. The AI will leverage a metadata-driven, bounty-based system to train a Recurrent Neural Network (RNN) that autonomously discovers strategic patterns. This model will generate a diverse set of move candidates, which are then validated by a genetic algorithm for in-game tactical advantage. The final product will be a deployable engine that can interface with chess GUIs via a UCI protocol.

## **2\. MVP Goals**

* **Create the "Thinking Brain" MVP:** Develop an RNN-based model capable of processing a chess position as a single "frame" and generating a set of top-N move candidates based on autonomously learned strategies. The learning will be guided by an objective, calculated "game status" metric.  
* **Create the "Gameplay Brain" MVP:** Implement a genetic algorithm that takes the move candidates from the "Thinking Brain" and simulates short game continuations to select the single best move for the current turn.  
* **Establish a Learning Pipeline:** Build a functional, end-to-end MVP that can train using simulated self-play and against other opponents.

## **3\. Key Features & User Stories**

### **Feature: Thinking Brain (Autonomous RL Model)**

* **Idea:** Develop an RNN-based model to learn and generate a diverse set of move candidates by interpreting objective game metadata.  
* **Goal:** The model will interpret a board state (frame) and, using its internal memory, output a probability distribution over a limited set of legal moves. It will learn which patterns in the objective data correlate with a higher "game status" value.  
* **Benefit:** Allows the AI to discover its own strategic concepts and patterns, transcending the limitations of human-authored heuristics. This creates a truly autonomous and potentially superhuman chess AI.  
* **Tasks:**  
  * **User Story:** As a developer, I need to define the input "state" of a single chess "frame," providing objective metadata about each piece and the overall board (e.g., piece activity, king status, etc.).  
  * **User Story:** As a developer, I need to design a "bounty system" (reward function) that provides a small, numerical value for each turn based on a calculated "game status."  
  * **User Story:** As a developer, I need to train the RNN model using past PGN game data, correlating the metadata at each "frame" with the final game outcome to reinforce or deter certain actions.  
* **Priority:** HIGH (This is the core of the new approach)

### **Feature: Gameplay Brain (Genetic Algorithm)**

* **Idea:** Implement a genetic algorithm to validate move candidates in real-time.  
* **Goal:** The GA will take the top-N move candidates from the "Thinking Brain" and simulate their tactical outcomes over several turns to find the most advantageous move in the current game.  
* **Benefit:** Provides a more dynamic and adaptive decision-making process, allowing the engine to react to unique in-game situations not seen in training data.  
* **Tasks:**  
  * **User Story:** As a developer, I need to create a GA that can take a list of moves as its initial "population."  
  * **User Story:** As a developer, I need to define a "fitness function" for the GA that evaluates the quality of a simulated game branch based on the tactical outcomes of the simulation (e.g., material advantage, checkmating sequence).  
  * **User Story:** As a developer, I need to integrate the GA with the RL model so that the GA only considers moves provided by the "Thinking Brain."  
* **Priority:** HIGH (Essential for validating the new approach's effectiveness)

### **Feature: Training Pipeline**

* **Idea:** Establish a workflow to train the "Thinking Brain" on PGN data and the defined bounty system.  
* **Goal:** Create a script or set of scripts that can read a PGN file, process each move and its resulting board state, apply the final game outcome as the bounty, and use that as the training data for the RNN.  
* **Benefit:** Ensures a repeatable and scalable process for improving the AI as I add more training data.  
* **Tasks:**  
  * **User Story:** As a developer, I need a data processing script to parse PGN files and convert game states into a format usable by the RNN, including the objective metadata.  
  * **User Story:** As a developer, I need a training loop that feeds the processed data to the RNN and updates its weights based on the bounty system rewards.  
  * **User Story:** As a developer, I need to persist the trained RNN model so that it can be loaded for real-time play.  
* **Priority:** HIGH (The foundation for the entire project)

## **4\. Technical Details & Data Contract**

### **Neural Network Architecture: GRU**

We will use a **Gated Recurrent Unit (GRU)** model for the "Thinking Brain." The GRU is a type of recurrent neural network (RNN) that is highly effective at processing sequential data, like the frames of a chess game. It maintains a memory of past events (moves) through its internal state, allowing it to learn long-term dependencies and strategic patterns. This is an ideal fit for your single-machine setup due to its computational efficiency.

**Hyperparameters (Initial Values):**

* **Layers:** 8 GRU layers  
* **Layer Size:** 256 neurons  
* **Learning Rate:** Start higher to allow for rapid initial learning, then tune down for fine-tuning.  
* **Reward Discount Factor (**γ**):** A high value, such as 0.99, to emphasize long-term strategic outcomes over immediate rewards.  
* **Optimizer:** Adam or Nadam, as they are robust and perform well with RNNs.

### **Data Model and Metadata: ChessState**

This data model represents a single "frame" of the game. It is a non-negotiable data contract between the data pipeline and the AI. Any changes to this schema will require retraining the model from scratch.

| Field Name | Data Type | Description |
| :---- | :---- | :---- |
| **Board\_Features** | Dictionary/Object | A collection of global, board-wide metrics. |
| Board\_Features.sideToMove | Integer (0, 1\) | The side to move (White=0, Black=1). |
| Board\_Features.castlingRights | Array of Booleans | \[White\_KS, White\_QS, Black\_KS, Black\_QS\] |
| Board\_Features.enPassantSquare | Integer (0-63) or null | The en passant target square, if available. |
| Board\_Features.halfmoveClock | Integer | The number of halfmoves since the last pawn move or capture. |
| Board\_Features.fullmoveNumber | Integer | The current game number. |
| Board\_Features.isKingInCheck | Boolean | Is the current player's king in check? |
| Board\_Features.threatCounts | Object | A breakdown of threats and control. |
| Board\_Features.threatCounts.whiteAttacked | Integer | Total squares attacked by white. |
| Board\_Features.threatCounts.blackAttacked | Integer | Total squares attacked by black. |
| Board\_Features.materialBalance | Object | The raw material values on the board. |
| Board\_Features.materialBalance.whiteTotal | Integer | Sum of material values for white. |
| Board\_Features.materialBalance.blackTotal | Integer | Sum of material values for black. |
| Board\_Features.materialBalance.difference | Integer | White total minus black total. |
| Board\_Features.pieceCounts | Object | Count of each piece type for each side. |
| Board\_Features.pawnStructure | Object | Metrics on pawn positions. |
| Board\_Features.pawnStructure.isolatedPawns | Object | Counts of isolated pawns. |
| Board\_Features.pawnStructure.doubledPawns | Object | Counts of doubled pawns. |
| Board\_Features.pawnStructure.passedPawns | Object | Counts of passed pawns. |
| Board\_Features.pawnStructure.connectedPawns | Object | Counts of connected pawns. |
| Board\_Features.pawnStructure.pawnIslands | Object | Counts of pawn islands. |
| Board\_Features.pawnStructure.backwardPawns | Object | Counts of backward pawns. |
| Board\_Features.kingSafety | Object | King safety metrics. |
| Board\_Features.kingSafety.pawnShieldStrength | Integer | Pawns protecting castled King. |
| Board\_Features.kingSafety.escapeSquares | Integer | Number of escape squares. |
| Board\_Features.kingSafety.exposedFiles | Integer | Number of open files on King's side. |
| Board\_Features.kingSafety.exposedDiagonals | Integer | Number of open diagonals on King's side. |
| Board\_Features.gamePhase | Float (0-1) | A value (0-Opening, 1-Endgame) based on remaining pieces. |
| Board\_Features.symmetrical | Float (0-1) | A measure of board symmetry. |
| Board\_Features.mobility | Object | Piece activity metrics. |
| Board\_Features.mobility.totalWhite | Integer | Sum of legal moves for all White pieces. |
| Board\_Features.mobility.totalBlack | Integer | Sum of legal moves for all Black pieces. |
| Board\_Features.uniqueInteractions | Object | Special interaction flags. |
| Board\_Features.uniqueInteractions.isSkewer | Boolean | Is a skewer threat present? |
| Board\_Features.uniqueInteractions.isFork | Boolean | Is a fork threat present? |
| Board\_Features.uniqueInteractions.isPin | Boolean | Is a pin threat present? |
| Board\_Features.uniqueInteractions.isDiscoveredAttack | Boolean | Is a discovered attack threat present? |
| Board\_Features.uniqueInteractions.isBattery | Boolean | Is a battery formation present? |
| Board\_Features.uniqueInteractions.threatensKingThroughPawn | Boolean | Is a sliding piece attacking King through a pawn? |
| **Pieces** | Array of Dictionaries | A list of features for each of the 32 pieces on the board. |
| Pieces\[i\].type | Integer (1-6) | The piece type. |
| Pieces\[i\].color | Integer (0, 1\) | The piece color. |
| Pieces\[i\].currentSquare | Integer (0-63) | The piece's current position. |
| Pieces\[i\].baseValue | Integer | The piece's standard material value. |
| Pieces\[i\].mobility | Object | All mobility metrics for the piece. |
| Pieces\[i\].mobility.legalMoves | Integer | Number of legal moves from this square. |
| Pieces\[i\].mobility.attackingMoves | Integer | Number of moves to capture a piece. |
| Pieces\[i\].mobility.maxUnobstructed | Integer | Max possible moves on an empty board. |
| Pieces\[i\].relationships | Object | Metrics on attackers/defenders. |
| Pieces\[i\].relationships.attackedByCount | Integer | Number of enemy pieces attacking this piece. |
| Pieces\[i\].relationships.defendedByCount | Integer | Number of friendly pieces defending this piece. |
| Pieces\[i\].relationships.isPinned | Boolean | Is the piece pinned? |
| Pieces\[i\].relationships.pinner | Integer or null | The piece that is pinning it. |
| Pieces\[i\].relationships.pinnedTo | Integer or null | The piece it is pinned to. |
| Pieces\[i\].relationships.attackingHigherValue | Boolean | Attacks a higher value piece. |
| Pieces\[i\].relationships.attackingLowerValue | Boolean | Attacks a lower value piece. |
| Pieces\[i\].relationships.attackingEqualValue | Boolean | Attacks an equal value piece. |
| Pieces\[i\].relationships.interposingSquares | Integer | Squares between this piece and enemy King. |
| Pieces\[i\].relationships.kingDistance | Integer | Distance from friendly King. |
| Pieces\[i\].relationships.enemyKingDistance | Integer | Distance from enemy King. |
| Pieces\[i\].positional | Object | Positional metrics. |
| Pieces\[i\].positional.centrality | Integer | Distance from the board center. |
| Pieces\[i\].positional.edgeProximity | Integer | Distance from the nearest edge. |
| Pieces\[i\].positional.backRankStatus | Boolean | Is it on the back rank? |
| Pieces\[i\].positional.seventhRankStatus | Boolean | Is it a pawn/rook on the 7th rank? |
| Pieces\[i\].vectorAnalysis | Object | Counts for move vectors. |

### **The New Bounty System (Reward Shaping)**

The AI's learning will be driven by a reward value calculated at each turn. Instead of subjective rules, this value will be derived from a handful of objective "game status" indicators. The AI's objective is to learn a strategy that maximizes this value over time.

**Game Status Indicators:**

* **King Safety:** A normalized score from 0 to 1 based on factors like check status, number of escape squares, and proximity of enemy pieces. The closer to 1, the safer the king.  
* **Piece Activity:** A normalized score from 0 to 1 based on the total number of legal moves available to a side's pieces, relative to the max possible. The closer to 1, the more active the pieces.  
* **Material Balance:** A score based on the material difference (whiteTotal \- blackTotal), normalized to a range (e.g., \-9 to \+9) to provide context. A positive score is better for the current player.  
* **Pawn Structure:** A score based on a composite of objective pawn metrics such as the number of passed pawns, pawn islands, and doubled pawns. A higher score indicates a stronger pawn structure.  
* **Game Phase:** A float from 0 to 1 that indicates the estimated game phase (0-Opening, 0.5-Middlegame, 1-Endgame), calculated based on remaining pieces and their types.

**Reward Function:**

Turn\_Reward \= (delta(KingSafety) \+ delta(PieceActivity) \+ delta(MaterialBalance) \+ ...) \+ final\_game\_outcome

* delta(...) refers to the change in the metric from the previous turn. For example, if a move increases the KingSafety score, that turn receives a positive reward from that metric.  
* final\_game\_outcome is a bonus/penalty (+1 for a win, 0 for a draw, \-1 for a loss) given only at the end of the game.

This approach provides a constant stream of objective feedback, which is far superior to a sparse, single reward at the end of a long game. The AI will learn, for instance, that consistently increasing its PieceActivity metric leads to higher cumulative rewards, which ultimately correlates with winning the game. You are not telling it to be active; you are simply providing the data that allows it to discover that a more active playstyle is a winning one.

## **5\. Risks & Considerations**

* **Computational Cost:** Training a recurrent neural network on a massive number of chess games will be very resource-intensive. This approach, while more powerful, will require significantly more data and compute than a heuristic-based model to learn effectively.  
* **Reward Sparsity:** While we've mitigated this with reward shaping, the primary long-term reward is still the final game outcome.  
* **Coupling:** The "Thinking Brain" (RNN) and "Gameplay Brain" (GA) have a high degree of coupling. Any changes to the output format of the RNN will directly impact the GA's input, and vice versa.  
* **Tuning the Genetic Algorithm:** The performance of the GA is highly dependent on parameters like population size, mutation rate, and generations. These will require extensive tuning.  
* **Lack of Intermediate Feedback:** Without intermediate heuristic guidance, the AI may take a long time to converge to a good policy. Initial results may be unpredictable.

## **6\. Success Metrics**

* **RL Model:** The model should consistently output a diverse set of move candidates that correlate with a higher "game status" metric.  
* **GA:** The GA should quickly converge to a single best move that demonstrates a tactical advantage over the other candidates.  
* **Overall Engine Performance:** The MVP engine should be able to win a significant percentage of games against a simple, traditional chess engine, demonstrating that its autonomously learned strategy is effective.