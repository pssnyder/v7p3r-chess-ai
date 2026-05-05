# V7P3RAI v4.0 Master Plan: Agentic Chess AI Enhancement Layer

**Project Vision**: Transform V7P3R Chess Engine into an AI-augmented agentic chess system by integrating specialized neural network agents trained on 4,000,000 puzzles, historical game data, and reinforcement learning from engine evaluations.

**Architecture**: Multi-agent system with specialized sub-agents for different game phases and tasks, seamlessly integrated into V7P3R Chess Engine's decision-making pipeline.

**Timeline**: 4-stage implementation over 3-6 months
**Target**: +400-600 ELO improvement (1400 → 1800-2000+ actual ELO)

---

## 🎯 Project Overview

### Current State
- **V7P3R Chess Engine v18.4.0**: ~1600 Lichess ELO (~1400 actual), 50% vs Stockfish 1%
- **V7P3RAI v3.0**: 129K puzzles trained, two-brain architecture (GRU + GA)
- **Available Resources**: 4,000,000 local puzzle library, engine-tester infrastructure, thousands of historical V7P3R games

### V4.0 Vision: Agentic Chess Intelligence
Build a modular AI agent system where each agent specializes in a specific chess task:
- **v7p3r-themes** (Stage 1): Position categorization & move ordering
- **v7p3r-corrector** (Stage 2): Historical move validation & correction
- **v7p3r-opening** (Stage 3): Opening book mastery
- **v7p3r-endgame** (Stage 3): Endgame & checkmate patterns (up to mate-in-5, 6-piece tablebases)
- **v7p3r-tactics** (Stage 4): Middlegame augmentation & evaluation replacement

### Integration Philosophy
- **Non-Invasive**: Agents enhance existing engine without replacing core search
- **Modular**: Each agent can be enabled/disabled independently
- **Graceful Degradation**: Engine falls back to traditional methods if agent fails
- **Data-Driven**: Continuous improvement through game analysis and retraining

---

## 📋 Stage 1: Pattern Recognition & Move Ordering 2.0

**Goal**: Train V7P3RAI on 4,000,000 puzzle library for comprehensive pattern recognition and integrate as intelligent move ordering system.

### 1.1 Full Puzzle Library Training

**Dataset**: 4,000,000 local chess puzzles
**Focus**: Pattern recognition across all tactical themes
**Output**: v7p3r-themes agent (position categorization + move ranking)

**Training Objectives**:
- Categorize positions by tactical themes (pins, forks, skewers, discovered attacks, etc.)
- Learn pattern frequency and importance
- Recognize complex multi-move combinations
- Associate position features with best move candidates

**Architecture**:
```python
class V7P3RThemesAgent:
    """
    Specialized agent for position categorization and move ordering
    """
    def __init__(self):
        self.pattern_recognizer = GRU(layers=12, neurons=512)  # Larger than v3.0
        self.theme_classifier = MultiLabelClassifier(num_themes=50)
        self.move_ranker = MoveRankingNetwork(top_k=100)
        
    def categorize_position(self, board: chess.Board) -> Dict[str, float]:
        """Returns probability distribution over tactical themes"""
        features = extract_chess_state(board)
        theme_probs = self.theme_classifier.predict(features)
        return theme_probs  # e.g., {'pin': 0.85, 'fork': 0.42, ...}
    
    def rank_moves(self, board: chess.Board, time_budget: float) -> List[chess.Move]:
        """Returns moves ranked by tactical promise, filtered by time budget"""
        legal_moves = list(board.legal_moves)
        move_scores = self.move_ranker.score_batch(board, legal_moves)
        
        # Dynamic candidate selection based on time
        if time_budget < 0.5:
            top_k = 5   # Fast: top 5 moves
        elif time_budget < 2.0:
            top_k = 10  # Normal: top 10 moves
        elif time_budget < 5.0:
            top_k = 50  # Deep: top 50 moves
        else:
            top_k = 100 # Ultra-deep: top 100 moves
        
        ranked_moves = sorted(legal_moves, key=lambda m: move_scores[m], reverse=True)
        return ranked_moves[:top_k]
```

**Training Pipeline**:
1. **Data Processing** (Week 1)
   - Parse 4M puzzle database
   - Extract positions, themes, solutions
   - Augment with position transformations (mirroring, rotations)
   - Split: 3.2M training, 400K validation, 400K test

2. **Model Training** (Week 2-3)
   - Train theme classifier on position → theme labels
   - Train move ranker on position → best_move pairs
   - Multi-task learning: shared feature extractor
   - GPU acceleration: RTX 4070 Ti

3. **Validation** (Week 3)
   - Theme classification accuracy: target >90%
   - Top-5 move accuracy: target >85%
   - Top-10 move accuracy: target >95%
   - Inference speed: <5ms per position

**Success Metrics**:
- ✅ Theme classification: >90% accuracy
- ✅ Top-5 move inclusion: >85% (Stockfish validation)
- ✅ Top-10 move inclusion: >95%
- ✅ Inference speed: <5ms per position
- ✅ 4M puzzle coverage: 100%

### 1.2 Move Ordering Integration

**Integration Point**: `v7p3r.py::search()` method
**Strategy**: Replace existing move ordering logic with AI-weighted ranking

**Implementation**:
```python
# v7p3r_chess_engine/src/v7p3r_ai_move_ordering.py

class AIMoveSorter:
    def __init__(self, themes_agent: V7P3RThemesAgent):
        self.themes_agent = themes_agent
        self.fallback_sorter = TraditionalMoveSorter()  # MVV-LVA, history heuristic
        
    def sort_moves(self, board: chess.Board, legal_moves: List[chess.Move], 
                   time_remaining: float) -> List[chess.Move]:
        """
        Sort moves using AI ranking with graceful fallback
        """
        try:
            # Calculate time budget for this move
            time_budget = self.calculate_time_budget(time_remaining, board)
            
            # Get AI-ranked moves (filtered by time budget)
            ranked_moves = self.themes_agent.rank_moves(board, time_budget)
            
            # Theme-based logging for analysis
            themes = self.themes_agent.categorize_position(board)
            self.log_position_themes(board, themes)
            
            return ranked_moves
            
        except Exception as e:
            # Graceful fallback to traditional ordering
            logging.warning(f"AI move ordering failed: {e}, using fallback")
            return self.fallback_sorter.sort(board, legal_moves)
```

**Engine Integration** (in `v7p3r.py`):
```python
class V7P3REngine:
    def __init__(self):
        # ... existing initialization ...
        self.ai_move_sorter = AIMoveSorter(themes_agent)
        self.use_ai_ordering = True  # Feature flag
        
    def search(self, board: chess.Board, time_limit: float = 3.0, ...):
        """Main search function with AI move ordering"""
        legal_moves = list(board.legal_moves)
        
        # AI-powered move ordering (Stage 1 feature)
        if self.use_ai_ordering:
            ordered_moves = self.ai_move_sorter.sort_moves(
                board, legal_moves, self.time_manager.remaining_time()
            )
        else:
            ordered_moves = self.traditional_move_sort(legal_moves)
        
        # Proceed with alpha-beta search on ordered moves
        for move in ordered_moves:
            # ... existing search logic ...
```

**Validation Testing**:
1. **Move Ordering Efficiency**
   - Measure alpha-beta cutoffs before/after AI ordering
   - Target: >30% improvement in cutoff rate
   - Metric: Nodes searched per move (should decrease)

2. **Search Depth Improvement**
   - Better ordering → deeper effective search
   - Target: +0.5 to +1.0 ply depth improvement
   - Measure on 1000 test positions

3. **Speed Validation**
   - Total move time should stay <10ms average
   - AI overhead: <5ms
   - Search improvement should compensate for AI cost

**Success Metrics**:
- ✅ Alpha-beta cutoffs: +30% improvement
- ✅ Effective search depth: +0.5-1.0 plies
- ✅ Move time: <10ms average maintained
- ✅ Zero crashes or fallbacks in 1000-game test
- ✅ ELO improvement: +100-150 vs baseline

---

## 📋 Stage 2: Historical Game Analysis & Move Correction

**Goal**: Train V7P3RAI on V7P3R's historical games to learn engine-specific patterns and create self-correcting move validation system.

### 2.1 Historical Game Dataset Preparation

**Data Source**: Thousands of V7P3R historical games
**Processing Pipeline**: Extract positions, deduplicate, Stockfish validation

**Dataset Creation**:
```python
# v4.0/scripts/create_historical_dataset.py

class HistoricalGameAnalyzer:
    def __init__(self, pgn_directory: str, stockfish_path: str):
        self.pgn_dir = pgn_directory
        self.stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.position_frequency = defaultdict(int)
        
    def process_games(self):
        """
        Extract all positions from V7P3R games and analyze with Stockfish
        """
        positions = []
        
        # Phase 1: Extract all positions from PGNs
        for pgn_file in glob.glob(f"{self.pgn_dir}/**/*.pgn"):
            game_positions = self.extract_positions_from_pgn(pgn_file)
            positions.extend(game_positions)
        
        # Phase 2: Count position frequency
        for pos in positions:
            fen = pos.board.fen()
            self.position_frequency[fen] += 1
        
        # Phase 3: Filter for recurring positions (appeared 2+ times)
        recurring_positions = [
            pos for pos in positions 
            if self.position_frequency[pos.board.fen()] >= 2
        ]
        
        print(f"Total positions: {len(positions)}")
        print(f"Unique positions: {len(self.position_frequency)}")
        print(f"Recurring positions (2+ times): {len(recurring_positions)}")
        
        # Phase 4: Stockfish analysis of recurring positions
        dataset = self.analyze_with_stockfish(recurring_positions)
        
        return dataset
    
    def analyze_with_stockfish(self, positions: List[PositionRecord]) -> List[TrainingExample]:
        """
        Analyze each position with Stockfish and compare to V7P3R's actual move
        """
        training_examples = []
        
        for pos_record in tqdm(positions, desc="Stockfish analysis"):
            board = pos_record.board
            v7p3r_move = pos_record.played_move
            
            # Get Stockfish top 5 moves
            info = self.stockfish.analyse(board, chess.engine.Limit(depth=20), multipv=5)
            stockfish_top5 = [pv_info["pv"][0] for pv_info in info]
            
            # Check if V7P3R played top-5 move
            if v7p3r_move not in stockfish_top5:
                # V7P3R made suboptimal move - create correction training example
                best_move = stockfish_top5[0]
                example = TrainingExample(
                    position=board.fen(),
                    v7p3r_move=v7p3r_move,
                    best_move=best_move,
                    move_quality="poor",
                    weight=10.0  # High weight for correction
                )
                training_examples.append(example)
            elif v7p3r_move == stockfish_top5[0]:
                # V7P3R played best move - reinforce
                example = TrainingExample(
                    position=board.fen(),
                    v7p3r_move=v7p3r_move,
                    best_move=v7p3r_move,
                    move_quality="excellent",
                    weight=1.0  # Normal weight
                )
                training_examples.append(example)
        
        return training_examples
```

**Dataset Statistics**:
- Total V7P3R games: ~5,000-10,000 (estimated)
- Total positions: ~500,000-1,000,000
- Recurring positions (2+ times): ~50,000-100,000 (estimated)
- Poor moves requiring correction: ~5,000-10,000 (estimated 10% suboptimal rate)

### 2.2 V7P3R-Corrector Agent Training

**Architecture**:
```python
class V7P3RCorrectorAgent:
    """
    Specialized agent for historical move validation and correction
    Trained on V7P3R's actual games to recognize and correct suboptimal patterns
    """
    def __init__(self):
        self.position_evaluator = GRU(layers=10, neurons=384)
        self.move_validator = MoveValidationNetwork()
        self.correction_db = PositionCorrectionDatabase()
        
    def validate_move(self, board: chess.Board, candidate_move: chess.Move) -> MoveValidation:
        """
        Check if candidate move matches known poor patterns and suggest corrections
        """
        position_features = extract_chess_state(board)
        
        # Check against historical correction database
        if self.correction_db.has_correction(board.fen()):
            known_correction = self.correction_db.get_correction(board.fen())
            if candidate_move == known_correction.poor_move:
                return MoveValidation(
                    is_valid=False,
                    suggested_move=known_correction.best_move,
                    confidence=0.95,
                    reason="Historical correction: V7P3R previously played poorly here"
                )
        
        # Neural network validation
        move_quality = self.move_validator.evaluate(position_features, candidate_move)
        
        if move_quality < 0.3:  # Low quality threshold
            # Get better alternative
            better_move = self.suggest_better_move(board, candidate_move)
            return MoveValidation(
                is_valid=False,
                suggested_move=better_move,
                confidence=move_quality,
                reason="AI predicts suboptimal move based on historical patterns"
            )
        
        return MoveValidation(is_valid=True, confidence=move_quality)
    
    def suggest_better_move(self, board: chess.Board, current_candidate: chess.Move) -> chess.Move:
        """Find better alternative using learned patterns"""
        legal_moves = list(board.legal_moves)
        move_scores = {m: self.move_validator.evaluate(board, m) for m in legal_moves}
        best_move = max(move_scores.keys(), key=lambda m: move_scores[m])
        return best_move
```

**Training Process**:
1. **Data Preparation** (Week 1)
   - Extract all V7P3R games from engine-tester archives
   - Identify recurring positions
   - Stockfish validation of all recurring positions
   - Build correction database (poor_move → best_move mapping)

2. **Model Training** (Week 2)
   - Train position evaluator on V7P3R-specific patterns
   - Train move validator with weighted loss (10x weight on corrections)
   - Curriculum learning: Start with obvious mistakes, progress to subtle errors
   - Validation: Hold out 20% of games for testing

3. **Correction Database** (Week 1)
   - Index all positions requiring correction
   - Store: FEN → (poor_move, best_move, stockfish_score, frequency)
   - Fast lookup: O(1) hash table for real-time validation

**Success Metrics**:
- ✅ Historical move accuracy: >90% (V7P3R played top-5 move)
- ✅ Correction detection: >95% (identify all suboptimal moves)
- ✅ Better move suggestion: >85% (suggested move in Stockfish top-3)
- ✅ Lookup speed: <1ms per position
- ✅ Database coverage: All recurring positions indexed

### 2.3 Move Correction Integration

**Integration Point**: Post-evaluation validation before move execution

**Implementation**:
```python
# In v7p3r.py::search() after move selection

def get_move(self, board: chess.Board, time_limit: float = 3.0) -> chess.Move:
    """
    Main move selection function with AI correction layer
    """
    # Stage 1: AI-ordered search (from previous stage)
    best_move, best_score = self.search(board, time_limit)
    
    # Stage 2: Historical move validation & correction
    if self.use_ai_correction:
        validation = self.corrector_agent.validate_move(board, best_move)
        
        if not validation.is_valid and validation.confidence > 0.7:
            # AI found historical pattern suggesting better move
            self.log_correction(
                position=board.fen(),
                engine_move=best_move,
                corrected_move=validation.suggested_move,
                reason=validation.reason
            )
            
            # Override with corrected move
            best_move = validation.suggested_move
            
            # Log for future analysis
            self.correction_count += 1
    
    return best_move
```

**Validation Testing**:
1. **Correction Accuracy**
   - Replay historical games with correction enabled
   - Measure: How many poor moves would be corrected?
   - Target: >90% of historically poor moves corrected

2. **False Positive Rate**
   - Ensure AI doesn't "correct" good moves
   - Target: <5% false positive rate on good moves
   - Validation: Check against Stockfish top-5

3. **ELO Impact**
   - Play 500 games with/without correction enabled
   - Target: +50-100 ELO from correction layer alone
   - Measure: Win rate improvement in recurring positions

**Success Metrics**:
- ✅ Poor move correction rate: >90%
- ✅ False positive rate: <5%
- ✅ ELO improvement: +50-100
- ✅ Correction overhead: <2ms per move
- ✅ Self-improvement visible in recurring positions

---

## 📋 Stage 3: Opening & Endgame Specialization

**Goal**: Create specialized agents for opening book mastery and endgame perfection.

### 3.1 V7P3R-Opening Agent

**Objective**: Complete opening book coverage with AI-powered novelty detection

**Training Data**:
- Master-level opening database (1M+ positions)
- V7P3R's historical opening performance
- Stockfish analysis of opening variations

**Architecture**:
```python
class V7P3ROpeningAgent:
    """
    Specialized agent for opening theory and book moves
    """
    def __init__(self):
        self.opening_book = OpeningBookDatabase()  # Traditional book
        self.novelty_detector = GRU(layers=8, neurons=256)
        self.opening_evaluator = OpeningEvaluationNetwork()
        
    def get_opening_move(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Returns book move or AI-suggested opening novelty
        """
        # Phase 1: Check traditional opening book
        book_moves = self.opening_book.get_moves(board)
        if book_moves:
            # Evaluate book moves with AI
            best_book_move = self.select_best_book_move(board, book_moves)
            return best_book_move
        
        # Phase 2: Out of book - AI novelty suggestion
        if board.fullmove_number <= 15:  # Still in opening phase
            novelty = self.suggest_novelty(board)
            if novelty.quality_score > 0.7:
                return novelty.move
        
        return None  # Hand over to main engine
    
    def suggest_novelty(self, board: chess.Board) -> OpeningNovelty:
        """AI-suggested move when out of book"""
        position_features = extract_opening_features(board)
        move_evaluations = self.opening_evaluator.evaluate_moves(board)
        
        best_move = max(move_evaluations.keys(), key=lambda m: move_evaluations[m])
        quality = move_evaluations[best_move]
        
        return OpeningNovelty(move=best_move, quality_score=quality)
```

**Training Objectives**:
- Learn opening principles (development, center control, king safety)
- Recognize transpositions and move order flexibility
- Evaluate opening novelties
- Adapt to opponent's repertoire

**Success Metrics**:
- ✅ Opening book coverage: 100% up to move 10
- ✅ Novelty quality: >80% (Stockfish validation)
- ✅ Opening phase win rate: +10% improvement
- ✅ Inference: <1ms per move

### 3.2 V7P3R-Endgame Agent

**Objective**: Perfect endgame play up to mate-in-5 and 6-piece tablebase coverage

**Training Data**:
- Syzygy 6-piece tablebases
- Mate-in-1/2/3/4/5 puzzle databases
- Endgame theory positions (Rook+Pawn, Bishop+Knight, etc.)

**Architecture**:
```python
class V7P3REndgameAgent:
    """
    Specialized agent for endgame perfection
    Combines tablebase lookups with AI-learned endgame principles
    """
    def __init__(self):
        self.tablebase = SyzygyTablebase(path="./tablebases/")
        self.mate_detector = MatePatternRecognizer()
        self.endgame_evaluator = EndgameEvaluationNetwork()
        
    def get_endgame_move(self, board: chess.Board) -> Optional[EndgameMove]:
        """
        Returns perfect endgame move or AI-guided play
        """
        piece_count = len(board.piece_map())
        
        # Phase 1: Tablebase perfect play (6 pieces or fewer)
        if piece_count <= 6 and self.tablebase.is_loaded():
            tb_result = self.tablebase.probe_dtz(board)
            if tb_result is not None:
                return EndgameMove(
                    move=tb_result.move,
                    type="tablebase",
                    dtz=tb_result.dtz,
                    is_winning=tb_result.wdl > 0
                )
        
        # Phase 2: Fast mate detection (mate in 1-5)
        mate_move = self.mate_detector.find_mate(board, max_depth=5)
        if mate_move:
            return EndgameMove(
                move=mate_move.move,
                type="forced_mate",
                mate_in=mate_move.depth
            )
        
        # Phase 3: AI endgame principles (7+ pieces)
        if piece_count <= 12:  # Still endgame-like
            ai_move = self.endgame_evaluator.evaluate_position(board)
            return EndgameMove(
                move=ai_move.best_move,
                type="ai_endgame",
                evaluation=ai_move.score
            )
        
        return None  # Hand over to main engine
```

**Training Objectives**:
- Perfect tablebase integration (DTZ optimal play)
- Fast mate detection (mate-in-1 through mate-in-5)
- Endgame principles (opposition, triangulation, zugzwang)
- King activity in endgames

**Success Metrics**:
- ✅ Tablebase accuracy: 100% (perfect play in all 6-piece positions)
- ✅ Mate detection: 100% for mate-in-1/2/3, >95% for mate-in-4/5
- ✅ Mate detection speed: <50ms for mate-in-3, <500ms for mate-in-5
- ✅ Endgame conversion rate: +20% (winning endgames converted)
- ✅ Draw avoidance: -50% (fewer draws from winning positions)

### 3.3 Opening/Endgame Integration

**Integration Point**: Pre-search phase detection

```python
def get_move(self, board: chess.Board, time_limit: float = 3.0) -> chess.Move:
    """
    Main move selection with specialized agents
    """
    # Stage 3a: Opening agent (moves 1-15)
    if board.fullmove_number <= 15:
        opening_move = self.opening_agent.get_opening_move(board)
        if opening_move:
            return opening_move
    
    # Stage 3b: Endgame agent (≤12 pieces or mate patterns)
    piece_count = len(board.piece_map())
    if piece_count <= 12 or self.detect_mating_attack(board):
        endgame_move = self.endgame_agent.get_endgame_move(board)
        if endgame_move and endgame_move.type in ["tablebase", "forced_mate"]:
            return endgame_move.move  # Perfect play
    
    # Stage 1+2: Main engine with AI ordering & correction
    best_move = self.search_with_ai(board, time_limit)
    return best_move
```

**Success Metrics**:
- ✅ Opening phase: <1ms per move (instant book moves)
- ✅ Endgame phase: Perfect play in tablebase positions
- ✅ Mate detection: Zero missed mates (mate-in-3 or less)
- ✅ Overall ELO: +100-150 from opening/endgame specialization
- ✅ Game phase coverage: 100% (opening → middlegame → endgame)

---

## 📋 Stage 4: Middlegame Augmentation & Evaluation Replacement

**Goal**: Train V7P3R-Tactics agent using reinforcement learning from V7P3R's own evaluation functions, eventually replacing static evaluation with neural network.

### 4.1 V7P3R-Tactics Agent (Middlegame Specialist)

**Objective**: Master middlegame play using RL with V7P3R's evaluation as reward signal

**Architecture**:
```python
class V7P3RTacticsAgent:
    """
    Middlegame specialist trained via self-play with V7P3R evaluation as reward
    """
    def __init__(self):
        self.policy_network = ResidualNetwork(layers=20, filters=256)
        self.value_network = ValueHead(input_size=256)
        self.v7p3r_evaluator = V7P3RFastEvaluator()  # Reward function
        
    def train_via_self_play(self, num_games: int = 10000):
        """
        Monte Carlo Tree Search + V7P3R evaluation rewards
        """
        for game_num in range(num_games):
            board = chess.Board()
            game_history = []
            
            while not board.is_game_over():
                # MCTS with neural network prior
                move, mcts_policy = self.mcts_search(board)
                
                # Get V7P3R evaluation before move
                eval_before = self.v7p3r_evaluator.evaluate(board)
                
                # Make move
                board.push(move)
                
                # Get V7P3R evaluation after move
                eval_after = self.v7p3r_evaluator.evaluate(board)
                
                # Reward = improvement in V7P3R's evaluation
                reward = (eval_after - eval_before) * (1 if board.turn else -1)
                
                game_history.append(TrainingExample(
                    position=board.fen(),
                    policy=mcts_policy,
                    value=eval_after,
                    reward=reward
                ))
            
            # Backpropagate final game result
            final_reward = self.compute_game_outcome(board)
            self.update_network(game_history, final_reward)
```

**Training Pipeline**:
1. **Self-Play Generation** (Week 1-2)
   - Generate 10,000 games via MCTS + neural network
   - Use V7P3R's evaluation function as reward signal
   - Time controls: Varied (bullet, blitz, rapid)
   - Opponent: Mix of self-play + Stockfish 5%

2. **Neural Network Training** (Week 2-3)
   - Train policy network to mimic successful MCTS decisions
   - Train value network to predict V7P3R evaluations
   - Loss function: Policy loss + Value loss + L2 regularization
   - Architecture: ResNet-inspired (20 residual blocks)

3. **Evaluation Function Replacement** (Week 4)
   - Gradually replace V7P3RFastEvaluator with neural network
   - Hybrid mode: 50% NN eval, 50% traditional eval
   - Monitor for regression: ELO tracking, puzzle accuracy
   - Full replacement only if NN matches or exceeds traditional

**Success Metrics**:
- ✅ Self-play games: 10,000 minimum
- ✅ Policy accuracy: >70% (match MCTS best move)
- ✅ Value prediction: MAE <50cp vs V7P3R evaluation
- ✅ Inference speed: <2ms per position
- ✅ ELO neutral or positive: NN eval >= traditional eval

### 4.2 Evaluation Function Replacement Strategy

**Phase 1: Hybrid Evaluation** (Week 4)
```python
class HybridEvaluator:
    def __init__(self, nn_weight=0.5):
        self.nn_evaluator = V7P3RTacticsAgent()
        self.traditional_evaluator = V7P3RFastEvaluator()
        self.nn_weight = nn_weight  # Gradually increase from 0.5 → 1.0
        
    def evaluate(self, board: chess.Board) -> int:
        """Weighted combination of NN and traditional evaluation"""
        nn_score = self.nn_evaluator.value_network.predict(board)
        traditional_score = self.traditional_evaluator.evaluate(board)
        
        hybrid_score = (self.nn_weight * nn_score + 
                       (1 - self.nn_weight) * traditional_score)
        
        return int(hybrid_score)
```

**Phase 2: Full NN Evaluation** (Week 5-6)
- Once NN eval proven stable and accurate
- Set `nn_weight = 1.0` (100% neural network)
- Keep traditional eval as fallback (feature flag)
- Monitor production performance closely

**Phase 3: Continuous Improvement** (Ongoing)
- Collect V7P3R game data from production
- Retrain NN on successful games
- Update weights periodically (weekly/monthly)
- A/B testing: NN eval vs traditional eval

**Success Metrics**:
- ✅ Hybrid eval stability: 0% crashes in 1000 games
- ✅ ELO with NN eval: >= baseline (no regression)
- ✅ Puzzle accuracy: >= baseline
- ✅ Tactical strength: +10% in tactical test suite
- ✅ Production readiness: 1000-game validation passed

### 4.3 Reinforcement Learning Pipeline

**Training Infrastructure**:
- **Hardware**: RTX 4070 Ti GPU + 32GB RAM
- **Framework**: PyTorch with CUDA acceleration
- **Parallelization**: 8 parallel self-play workers
- **Storage**: SSD for fast model checkpoints

**RL Training Loop**:
```python
def train_tactics_agent():
    """
    Main reinforcement learning training loop
    """
    agent = V7P3RTacticsAgent()
    
    for iteration in range(1000):  # ~1000 iterations
        # 1. Self-play generation (100 games per iteration)
        game_data = agent.generate_self_play_games(num_games=100)
        
        # 2. Add to replay buffer
        agent.replay_buffer.add(game_data)
        
        # 3. Sample mini-batches and train
        for epoch in range(10):
            batch = agent.replay_buffer.sample(batch_size=256)
            loss = agent.train_on_batch(batch)
        
        # 4. Evaluate every 10 iterations
        if iteration % 10 == 0:
            elo = agent.evaluate_against_baseline(num_games=100)
            print(f"Iteration {iteration}: ELO = {elo}")
            
            # 5. Checkpoint if improvement
            if elo > agent.best_elo:
                agent.save_checkpoint(f"tactics_agent_iter{iteration}.pth")
                agent.best_elo = elo
```

**Success Metrics**:
- ✅ Training iterations: 1000 minimum
- ✅ Total games: 100,000 self-play games
- ✅ Model convergence: Loss plateaus, ELO stabilizes
- ✅ Final ELO: +200-300 vs baseline (target 1800-2000 actual ELO)
- ✅ Generalization: Performs well on unseen positions

---

## 🤖 Agentic Architecture Overview

### Multi-Agent System Design

```
V7P3R Chess Engine (Orchestrator)
│
├── V7P3R-Themes Agent (Stage 1)
│   ├── Pattern Recognition (4M puzzles)
│   ├── Position Categorization (50 themes)
│   └── Move Ordering (Top-K selection)
│
├── V7P3R-Corrector Agent (Stage 2)
│   ├── Historical Move Database
│   ├── Move Validation Network
│   └── Correction Suggestion System
│
├── V7P3R-Opening Agent (Stage 3)
│   ├── Opening Book Integration
│   ├── Novelty Detection
│   └── Transposition Recognition
│
├── V7P3R-Endgame Agent (Stage 3)
│   ├── Tablebase Integration (6 pieces)
│   ├── Mate Detection (mate-in-5)
│   └── Endgame Principles Network
│
└── V7P3R-Tactics Agent (Stage 4)
    ├── Policy Network (Move Selection)
    ├── Value Network (Evaluation)
    └── MCTS Integration
```

### Agent Coordination

**Decision Flow**:
```python
def get_move(self, board: chess.Board, time_limit: float) -> chess.Move:
    """
    Multi-agent decision pipeline
    """
    # Priority 1: Opening Agent (moves 1-15)
    if board.fullmove_number <= 15:
        move = self.opening_agent.get_opening_move(board)
        if move:
            return move
    
    # Priority 2: Endgame Agent (perfect play)
    if len(board.piece_map()) <= 6:
        move = self.endgame_agent.get_tablebase_move(board)
        if move:
            return move
    
    # Priority 3: Mate Detection (any phase)
    mate_move = self.endgame_agent.find_mate(board, max_depth=3)
    if mate_move:
        return mate_move
    
    # Priority 4: Main Search (Middlegame/Complex)
    # - Themes Agent: Move ordering
    ordered_moves = self.themes_agent.rank_moves(board, time_limit)
    
    # - Tactics Agent: Evaluation (if Stage 4 complete)
    if self.use_nn_eval:
        best_move = self.tactics_agent.search(board, ordered_moves, time_limit)
    else:
        best_move = self.traditional_search(board, ordered_moves, time_limit)
    
    # - Corrector Agent: Validation
    validation = self.corrector_agent.validate_move(board, best_move)
    if not validation.is_valid:
        best_move = validation.suggested_move
    
    return best_move
```

### Agent Communication Protocol

**Message Passing**:
```python
class AgentMessage:
    sender: str           # Agent name
    receiver: str         # Target agent or "orchestrator"
    message_type: str     # "move_request", "evaluation", "correction", etc.
    payload: Dict         # Agent-specific data
    timestamp: float
    priority: int         # 0 = highest, 10 = lowest
```

**Example Communication**:
```python
# Themes agent suggests moves
themes_msg = AgentMessage(
    sender="v7p3r-themes",
    receiver="orchestrator",
    message_type="move_ranking",
    payload={"ranked_moves": [...], "themes": {"pin": 0.85}},
    priority=1
)

# Corrector agent validates
corrector_msg = AgentMessage(
    sender="v7p3r-corrector",
    receiver="orchestrator",
    message_type="validation_result",
    payload={"is_valid": False, "suggested_move": "Nf3"},
    priority=0  # High priority - overrides other suggestions
)
```

---

## 📊 Overall Success Metrics

### Performance Targets

| Metric | Baseline (v18.4) | Stage 1 | Stage 2 | Stage 3 | Stage 4 (Final) |
|--------|-----------------|---------|---------|---------|-----------------|
| **Actual ELO** | 1400 | 1500-1550 | 1550-1600 | 1650-1700 | 1800-2000 |
| **Lichess ELO** | 1600 | 1700-1750 | 1750-1800 | 1850-1900 | 2000-2200 |
| **Puzzle Accuracy** | N/A | 85% | 90% | 92% | 95% |
| **Move Time (avg)** | 5-8ms | <10ms | <12ms | <15ms | <10ms (optimized) |
| **Opening Win %** | 50% | 55% | 60% | 70% | 75% |
| **Endgame Win %** | 50% | 50% | 55% | 80% | 85% |
| **Tactical Win %** | 55% | 65% | 70% | 75% | 80% |

### Quality Metrics

**Stability**:
- Zero crashes in 10,000-game test suite
- Graceful degradation: All agents have fallback modes
- Memory usage: <2GB RAM (keep within GCP limits)
- CPU/GPU efficiency: <50% average utilization

**Accuracy**:
- Theme classification: >90%
- Move ordering efficiency: >85% (correct move in top-10)
- Historical correction: >90% (catch suboptimal moves)
- Tablebase accuracy: 100% (perfect play)
- Mate detection: 100% for mate-in-3, >95% for mate-in-5

**Speed**:
- Theme classification: <5ms
- Move validation: <2ms
- Tablebase lookup: <1ms
- Opening book: <1ms
- Full agent pipeline: <20ms total overhead

### Validation Protocol

**Stage 1 Validation**:
1. 1000-puzzle test suite (all themes)
2. 500-game tournament vs v18.4.0 baseline
3. Alpha-beta cutoff efficiency measurement
4. Speed regression testing

**Stage 2 Validation**:
1. Replay 1000 historical games with correction
2. Measure correction accuracy vs Stockfish
3. 500-game tournament with/without correction
4. False positive rate analysis

**Stage 3 Validation**:
1. Opening repertoire coverage test
2. Tablebase position accuracy (10,000 positions)
3. Mate-in-3 puzzle suite (100% accuracy required)
4. 500-game tournament with specialized agents

**Stage 4 Validation**:
1. 10,000 self-play games (convergence test)
2. NN eval vs traditional eval comparison (5000 positions)
3. Puzzle accuracy with NN eval
4. Production 1000-game tournament
5. Lichess rating validation (50 rated games)

---

## 🗂️ Project Structure

### v7p3r-chess-ai v4.0 Directory Structure

```
v7p3r-chess-ai/
├── v4.0/
│   ├── README.md
│   ├── requirements.txt
│   ├── setup.py
│   │
│   ├── config/
│   │   ├── training_config.json
│   │   ├── agent_config.json
│   │   └── integration_config.json
│   │
│   ├── src/
│   │   ├── agents/
│   │   │   ├── v7p3r_themes_agent.py
│   │   │   ├── v7p3r_corrector_agent.py
│   │   │   ├── v7p3r_opening_agent.py
│   │   │   ├── v7p3r_endgame_agent.py
│   │   │   └── v7p3r_tactics_agent.py
│   │   │
│   │   ├── core/
│   │   │   ├── chess_state_extractor.py  # 690-feature extraction
│   │   │   ├── agent_orchestrator.py     # Multi-agent coordination
│   │   │   └── agent_communication.py    # Message passing
│   │   │
│   │   ├── engine_integration/
│   │   │   ├── ai_evaluator.py           # NN evaluation module
│   │   │   ├── ai_move_sorter.py         # Move ordering module
│   │   │   ├── ai_validator.py           # Move correction module
│   │   │   └── feature_bridge.py         # ChessState → Engine
│   │   │
│   │   ├── training/
│   │   │   ├── puzzle_trainer.py         # Stage 1: 4M puzzles
│   │   │   ├── historical_trainer.py     # Stage 2: Historical games
│   │   │   ├── opening_trainer.py        # Stage 3: Opening theory
│   │   │   ├── endgame_trainer.py        # Stage 3: Endgame + tablebases
│   │   │   └── rl_trainer.py             # Stage 4: Reinforcement learning
│   │   │
│   │   ├── models/
│   │   │   ├── v3_model_loader.py        # Load v3.0 GRU
│   │   │   ├── gru_networks.py           # GRU architectures
│   │   │   ├── residual_networks.py      # ResNet for tactics
│   │   │   └── model_utils.py            # Training utilities
│   │   │
│   │   └── utils/
│   │       ├── position_database.py      # Position indexing
│   │       ├── stockfish_analyzer.py     # Stockfish integration
│   │       ├── tablebase_interface.py    # Syzygy interface
│   │       └── metrics.py                # Performance tracking
│   │
│   ├── data/
│   │   ├── puzzles/
│   │   │   └── 4M_puzzle_library/        # Local puzzle database
│   │   ├── historical_games/
│   │   │   └── v7p3r_pgns/               # Historical V7P3R games
│   │   ├── opening_book/
│   │   │   └── master_games_db/          # Opening theory
│   │   └── tablebases/
│   │       └── syzygy_6piece/            # Endgame tablebases
│   │
│   ├── models/
│   │   ├── stage1_themes/                # Trained theme classifier
│   │   ├── stage2_corrector/             # Historical correction model
│   │   ├── stage3_opening/               # Opening specialist
│   │   ├── stage3_endgame/               # Endgame specialist
│   │   └── stage4_tactics/               # RL middlegame model
│   │
│   ├── scripts/
│   │   ├── stage1_train_themes.py
│   │   ├── stage2_analyze_historical.py
│   │   ├── stage3_prepare_openings.py
│   │   ├── stage3_prepare_endgames.py
│   │   ├── stage4_self_play.py
│   │   ├── validate_agents.py
│   │   └── deploy_to_engine.py
│   │
│   ├── tests/
│   │   ├── test_themes_agent.py
│   │   ├── test_corrector_agent.py
│   │   ├── test_opening_agent.py
│   │   ├── test_endgame_agent.py
│   │   ├── test_tactics_agent.py
│   │   └── test_integration.py
│   │
│   └── docs/
│       ├── V7P3RAI_V4.0_MASTER_PLAN.md   # This document
│       ├── STAGE1_IMPLEMENTATION.md
│       ├── STAGE2_IMPLEMENTATION.md
│       ├── STAGE3_IMPLEMENTATION.md
│       ├── STAGE4_IMPLEMENTATION.md
│       ├── AGENT_ARCHITECTURE.md
│       └── DEPLOYMENT_GUIDE.md
```

### v7p3r-chess-engine Integration

**New Files in Engine**:
```
v7p3r-chess-engine/src/
├── v7p3r_ai_move_ordering.py      # Stage 1: AI move sorter
├── v7p3r_ai_validator.py          # Stage 2: Move correction
├── v7p3r_ai_opening.py            # Stage 3: Opening integration
├── v7p3r_ai_endgame.py            # Stage 3: Endgame integration
├── v7p3r_ai_evaluator.py          # Stage 4: NN evaluation
└── v7p3r_agent_orchestrator.py    # Multi-agent coordinator
```

**Modified Files**:
```
v7p3r-chess-engine/src/
├── v7p3r.py                       # Main engine - agent integration
├── v7p3r_modular_eval.py          # Add AI evaluation module
└── v7p3r_uci.py                   # UCI options for agent control
```

---

## 📅 Development Timeline

### Stage 1: Pattern Recognition & Move Ordering (Weeks 1-4)
- **Week 1**: Data preparation (4M puzzles)
- **Week 2-3**: Model training (themes + move ranking)
- **Week 3**: Integration into v7p3r-chess-engine
- **Week 4**: Validation & optimization

### Stage 2: Historical Analysis & Correction (Weeks 5-7)
- **Week 5**: Historical game extraction & Stockfish analysis
- **Week 6**: Corrector agent training
- **Week 7**: Integration & validation

### Stage 3: Opening & Endgame Specialists (Weeks 8-11)
- **Week 8**: Opening book preparation & training
- **Week 9**: Tablebase integration & mate detection
- **Week 10**: Opening agent integration
- **Week 11**: Endgame agent integration & validation

### Stage 4: Middlegame Augmentation (Weeks 12-18)
- **Week 12-14**: Self-play data generation (10K games)
- **Week 15-16**: RL training (policy + value networks)
- **Week 17**: Hybrid evaluation testing
- **Week 18**: Full NN evaluation deployment & validation

### Post-Stage 4: Optimization & Production (Weeks 19-20)
- **Week 19**: Performance optimization, caching, GPU acceleration
- **Week 20**: Production deployment, monitoring, A/B testing

**Total Timeline**: 20 weeks (5 months)

---

## 🚀 Deployment Strategy

### Phase 1: Local Development & Testing
- Train and validate each agent locally
- Integration testing on development machine
- Benchmark against v18.4.0 baseline

### Phase 2: Staging Environment
- Deploy to separate GCP staging instance
- Run 1000-game validation tournament
- Measure stability and performance
- Collect metrics for analysis

### Phase 3: Production Deployment
- Deploy to production GCP instance (v7p3r-production-bot)
- Gradual rollout: 10% traffic → 50% → 100%
- Monitor performance metrics continuously
- Rollback plan ready (keep v18.4.0 as fallback)

### Phase 4: Continuous Improvement
- Collect game data from production
- Retrain agents weekly/monthly
- A/B testing of new models
- Incremental ELO improvements

---

## 🎯 Key Decisions & Assumptions

### Decisions
1. **Supplementary Architecture**: AI agents enhance existing engine, don't replace core search
2. **Multi-Agent Design**: Specialized agents for different tasks (opening, endgame, tactics, etc.)
3. **Graceful Degradation**: All agents have fallback to traditional methods
4. **Data-Driven Training**: Use 4M puzzles, historical games, and self-play
5. **Staged Rollout**: Implement and validate each stage before proceeding

### Assumptions
1. **Puzzle Database Access**: 4M puzzles are accessible and properly formatted
2. **Historical Game Availability**: V7P3R has sufficient historical game data (~5K+ games)
3. **Computational Resources**: RTX 4070 Ti GPU available for training
4. **Stockfish Integration**: Stockfish 15+ available for analysis and validation
5. **GCP Deployment**: Production environment can accommodate AI overhead (<100MB model size, <500MB RAM)

### Risks & Mitigation

**Risk 1: AI Overhead Too High**
- **Mitigation**: Aggressive optimization, model pruning, caching, GPU acceleration
- **Fallback**: Feature flags to disable AI agents if performance degrades

**Risk 2: Training Data Quality**
- **Mitigation**: Extensive data validation, Stockfish cross-checking, manual review
- **Fallback**: Use smaller validated datasets if full 4M shows quality issues

**Risk 3: Agent Coordination Complexity**
- **Mitigation**: Clear priority system, simple message passing, extensive testing
- **Fallback**: Disable multi-agent features, use single best agent only

**Risk 4: Production Instability**
- **Mitigation**: Staged rollout, extensive validation, monitoring, quick rollback
- **Fallback**: Keep v18.4.0 as fallback engine (one-command rollback)

---

## 📈 Expected Outcomes

### Quantitative Improvements
- **ELO Gain**: +400-600 actual ELO (1400 → 1800-2000)
- **Puzzle Accuracy**: 95%+ on tactical puzzles
- **Opening Win Rate**: 75%+ (from 50%)
- **Endgame Conversion**: 85%+ (from 50%)
- **Zero Missed Mates**: 100% accuracy on mate-in-3 or less

### Qualitative Improvements
- **Positional Understanding**: Better strategic play through pattern recognition
- **Consistency**: Fewer blunders via historical correction
- **Specialization**: Expert-level play in openings and endgames
- **Adaptability**: Self-improving through continuous learning

### Innovation
- **First Agentic Chess Engine**: Multi-agent coordination in chess AI
- **Hybrid AI-Traditional**: Best of both worlds (NN intuition + alpha-beta precision)
- **Self-Correcting System**: Engine learns from its own mistakes
- **Massive Pattern Library**: 4M puzzle patterns + historical game knowledge

---

## 🎓 Learning Objectives

### Technical Skills Developed
1. **Deep Learning**: GRU, ResNet, policy/value networks
2. **Reinforcement Learning**: MCTS, self-play, reward shaping
3. **Multi-Agent Systems**: Coordination, communication, priority handling
4. **Chess AI**: Pattern recognition, move ordering, evaluation functions
5. **Production ML**: Model deployment, monitoring, A/B testing

### Chess Knowledge Gained
1. **Tactical Patterns**: 4M puzzle library coverage
2. **Opening Theory**: Master-level opening book
3. **Endgame Mastery**: Tablebase-perfect play
4. **Historical Analysis**: Learning from past games
5. **Strategic Principles**: Positional evaluation, planning

---

## 📝 Documentation Requirements

### Per-Stage Documentation
Each stage requires:
1. **Implementation Guide**: Step-by-step setup and execution
2. **Training Logs**: Dataset stats, hyperparameters, convergence metrics
3. **Validation Report**: Test results, benchmarks, ELO measurements
4. **Integration Guide**: How to integrate agent into engine
5. **Troubleshooting**: Common issues and solutions

### Final Documentation
1. **Deployment Guide**: Production deployment procedures
2. **Maintenance Guide**: Model retraining, updating agents
3. **API Reference**: Agent interfaces and communication protocol
4. **Performance Tuning**: Optimization techniques and benchmarks
5. **Research Report**: Findings, insights, future work

---

## 🏁 Success Criteria

### Stage 1 Complete When:
- ✅ 4M puzzles trained successfully
- ✅ Theme classification >90% accuracy
- ✅ Move ordering integrated and validated
- ✅ +100-150 ELO improvement demonstrated

### Stage 2 Complete When:
- ✅ Historical games analyzed and indexed
- ✅ Correction database built and validated
- ✅ Move validation integrated
- ✅ +50-100 ELO improvement from corrections

### Stage 3 Complete When:
- ✅ Opening agent provides 100% book coverage (first 10 moves)
- ✅ Endgame agent achieves perfect tablebase play
- ✅ Mate-in-3 detection: 100% accuracy
- ✅ +100-150 ELO improvement in opening/endgame

### Stage 4 Complete When:
- ✅ 10K self-play games generated
- ✅ NN evaluation matches traditional eval performance
- ✅ Hybrid evaluation stable in production
- ✅ +200-300 ELO improvement demonstrated

### V4.0 Complete When:
- ✅ All 4 stages implemented and validated
- ✅ Multi-agent system stable in production
- ✅ 1800-2000+ actual ELO achieved
- ✅ 1000-game production validation passed
- ✅ Zero critical bugs or crashes
- ✅ Documentation complete and published

---

**Document Version**: 1.0  
**Last Updated**: April 18, 2026  
**Status**: Planning Phase  
**Next Action**: Begin Stage 1 - Data preparation for 4M puzzle training
