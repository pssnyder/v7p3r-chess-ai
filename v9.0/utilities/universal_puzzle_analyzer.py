#!/usr/bin/env python3
"""
Universal Chess Engine Puzzle Analyzer

Advanced puzzle testing system that works with any UCI-compatible chess engine:
1. Pulls positions from puzzle database
2. Analyzes complete puzzle sequences, not just first position
3. Plays through entire solution chains with the test engine
4. Compares each move against Stockfish's top 5 moves
5. Calculates weighted accuracy scores (later moves weighted higher)
6. Estimates engine's puzzle rating based on perfect/high-accuracy performance
7. Tracks performance degradation through sequence depth
8. Provides comprehensive theme-based analysis

Enhanced Features:
- Universal UCI engine support: V7P3R, C0BR4, SlowMate, or any UCI engine (.exe or .bat)
- Supports both executable engines (.exe) and Python source engines (.bat)
- Sequence analysis: Plays opponent moves and challenges engine on each position
- Weighted scoring: Later positions in sequences count for more
- Rating estimation: Analyzes puzzle ratings where engine excels
- Position depth analysis: Shows how performance changes with sequence depth
- Comprehensive reporting: Theme performance, accuracy distributions, insights
- Dynamic engine info: Automatically detects engine name and version via UCI

Scoring: 5pts (1st), 4pts (2nd), 3pts (3rd), 2pts (4th), 1pt (5th), 0pts (not in top 5)
Sequence Accuracy: Weighted exponentially (1, 1.5, 2.25, 3.375, etc.)
"""

import subprocess
import time
import json
import os
import sys
import signal
import gc
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Union
import chess
import chess.engine

# Add the databases directory to path for database access
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'databases'))
try:
    from database import PuzzleDatabase, Puzzle
except ImportError:
    print("Error: Could not import puzzle database. Make sure databases directory is available.")
    sys.exit(1)


class TimeControl:
    """Manages chess time controls for realistic engine testing"""
    
    def __init__(self, base_time_minutes: float = 30.0, increment_seconds: float = 2.0):
        self.base_time_ms = int(base_time_minutes * 60 * 1000)
        self.increment_ms = int(increment_seconds * 1000)
        self.white_time_remaining = self.base_time_ms
        self.black_time_remaining = self.base_time_ms
    
    def get_time_for_move(self, is_white: bool) -> int:
        """Get remaining time in milliseconds for the moving side"""
        return self.white_time_remaining if is_white else self.black_time_remaining
    
    def consume_time(self, is_white: bool, time_used_ms: int):
        """Update remaining time after a move, including increment"""
        if is_white:
            self.white_time_remaining = max(0, self.white_time_remaining - time_used_ms + self.increment_ms)
        else:
            self.black_time_remaining = max(0, self.black_time_remaining - time_used_ms + self.increment_ms)
    
    def is_time_trouble(self, is_white: bool) -> bool:
        """Check if side is in time trouble (less than 10% of base time)"""
        remaining = self.white_time_remaining if is_white else self.black_time_remaining
        return remaining < (self.base_time_ms * 0.1)
    
    def format_time(self, time_ms: int) -> str:
        """Format time in milliseconds to readable format"""
        total_seconds = time_ms / 1000
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        return f"{minutes}:{seconds:02d}"


class UniversalPuzzleAnalyzer:
    """Analyzes any UCI chess engine's performance against puzzle database using Stockfish comparison"""
    
    def __init__(self, 
                 engine_path: str,
                 stockfish_path: str = r"E:\Programming Stuff\Chess Engines\Tournament Engines\Stockfish\stockfish-windows-x86-64-avx2.exe",
                 puzzle_db_path: str = r"E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-tester\databases\puzzles.db",
                 time_control: Optional[TimeControl] = None):
        
        self.engine_path = engine_path
        self.stockfish_path = stockfish_path
        self.puzzle_db_path = puzzle_db_path
        self.results = []
        
        # Determine engine type and command
        self.engine_type = self._detect_engine_type(engine_path)
        self.engine_command = self._build_engine_command(engine_path)
        
        # Extended session controls
        self.session_start_time = None
        self.session_duration_hours = None
        self.progress_save_interval = 300  # Save progress every 5 minutes
        self.report_interval = 1800  # Generate reports every 30 minutes
        self.last_progress_save = 0
        self.last_report_time = 0
        self.checkpoint_file = None
        self.session_active = False
        self.interrupted = False
        
        # Threading for periodic tasks
        self.progress_thread = None
        self.stop_event = threading.Event()
        
        # Verify engines exist
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Test engine not found: {engine_path}")
        if not os.path.exists(stockfish_path):
            raise FileNotFoundError(f"Stockfish engine not found: {stockfish_path}")
        if not os.path.exists(puzzle_db_path):
            raise FileNotFoundError(f"Puzzle database not found: {puzzle_db_path}")
        
        # Get engine information via UCI
        self.engine_info = self.get_engine_info()
        self.engine_name = self.engine_info.get('name', os.path.basename(engine_path))
        print(f"Initialized Universal Puzzle Analyzer for: {self.engine_name}")
        
        # Set up signal handlers for graceful shutdown
        self.setup_signal_handlers()
        
        # Time control management
        self.default_time_control = time_control or TimeControl(30.0, 2.0)  # 30+2 default
    
    def _detect_engine_type(self, engine_path: str) -> str:
        """Detect whether engine is .exe or .bat file"""
        path_lower = engine_path.lower()
        if path_lower.endswith('.bat'):
            return 'bat'
        elif path_lower.endswith('.exe'):
            return 'exe'
        else:
            # Default to treating as executable
            return 'exe'
    
    def _build_engine_command(self, engine_path: str) -> List[str]:
        """Build the command to launch the engine based on file type"""
        if self.engine_type == 'bat':
            # For .bat files, use cmd.exe to execute them
            return ['cmd.exe', '/c', engine_path]
        else:
            # For .exe files, launch directly
            return [engine_path]
    
    def setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            print(f"\n🛑 Received signal {signum}, initiating graceful shutdown...")
            self.interrupted = True
            self.session_active = False
            if self.stop_event:
                self.stop_event.set()
        
        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        except Exception as e:
            print(f"Warning: Could not set up signal handlers: {e}")
    
    def create_checkpoint_filename(self) -> str:
        """Create checkpoint filename based on engine and timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        engine_safe_name = self.engine_name.replace(' ', '_').replace('.', '_')
        return f"checkpoint_{engine_safe_name}_{timestamp}.json"
    
    def save_progress_checkpoint(self, puzzles_processed: int, current_puzzle_id: Optional[str] = None):
        """Save current progress to checkpoint file"""
        if not self.checkpoint_file:
            self.checkpoint_file = self.create_checkpoint_filename()
        
        elapsed_time = time.time() - self.session_start_time if self.session_start_time else 0
        
        checkpoint_data = {
            'metadata': {
                'engine_path': self.engine_path,
                'engine_name': self.engine_name,
                'engine_info': self.engine_info,
                'session_start_time': self.session_start_time,
                'last_checkpoint_time': time.time(),
                'elapsed_hours': elapsed_time / 3600,
                'puzzles_processed': puzzles_processed,
                'current_puzzle_id': current_puzzle_id,
                'session_duration_hours': self.session_duration_hours
            },
            'results': self.results,
            'partial_report': self.generate_report(self.results) if self.results else {}
        }
        
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            print(f"📁 Progress saved to checkpoint: {self.checkpoint_file}")
        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_file: str) -> bool:
        """Load progress from checkpoint file"""
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            
            self.results = data.get('results', [])
            metadata = data.get('metadata', {})
            
            self.session_start_time = metadata.get('session_start_time')
            self.session_duration_hours = metadata.get('session_duration_hours')
            self.checkpoint_file = checkpoint_file
            
            elapsed = metadata.get('elapsed_hours', 0)
            puzzles_processed = metadata.get('puzzles_processed', 0)
            
            print(f"📂 Loaded checkpoint from {checkpoint_file}")
            print(f"   Previous session: {elapsed:.1f} hours, {puzzles_processed} puzzles")
            print(f"   Results loaded: {len(self.results)} puzzle analyses")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load checkpoint {checkpoint_file}: {e}")
            return False
    
    def should_continue_session(self) -> bool:
        """Check if session should continue based on time and interruption status"""
        if self.interrupted or not self.session_active:
            return False
        
        if self.session_duration_hours and self.session_start_time:
            elapsed_hours = (time.time() - self.session_start_time) / 3600
            if elapsed_hours >= self.session_duration_hours:
                print(f"⏰ Session duration reached: {elapsed_hours:.1f}/{self.session_duration_hours} hours")
                return False
        
        return True
    
    def start_progress_monitoring_thread(self):
        """Start background thread for periodic progress saving and reporting"""
        def monitor_progress():
            while not self.stop_event.is_set() and self.session_active:
                current_time = time.time()
                
                # Save progress checkpoint
                if current_time - self.last_progress_save >= self.progress_save_interval:
                    self.save_progress_checkpoint(len(self.results))
                    self.last_progress_save = current_time
                
                # Generate intermediate report
                if current_time - self.last_report_time >= self.report_interval:
                    if self.results:
                        print(f"\n📊 INTERMEDIATE REPORT ({len(self.results)} puzzles analyzed)")
                        print("-" * 50)
                        report = self.generate_report(self.results)
                        self.print_intermediate_report(report)
                        print("-" * 50)
                    self.last_report_time = current_time
                
                # Clean up memory periodically
                if len(self.results) % 100 == 0 and len(self.results) > 0:
                    gc.collect()
                
                self.stop_event.wait(30)  # Check every 30 seconds
        
        self.progress_thread = threading.Thread(target=monitor_progress, daemon=True)
        self.progress_thread.start()
    
    def print_intermediate_report(self, report: Dict):
        """Print condensed intermediate report during long sessions"""
        if not report:
            return
        
        elapsed_hours = (time.time() - self.session_start_time) / 3600 if self.session_start_time else 0
        remaining_hours = max(0, self.session_duration_hours - elapsed_hours) if self.session_duration_hours else float('inf')
        
        seq_metrics = report.get('sequence_metrics', {})
        puzzles_per_hour = len(self.results) / elapsed_hours if elapsed_hours > 0 else 0
        
        print(f"Engine: {report.get('engine_name', 'Unknown')}")
        print(f"Time: {elapsed_hours:.1f}h elapsed" + (f", {remaining_hours:.1f}h remaining" if remaining_hours != float('inf') else ""))
        print(f"Progress: {len(self.results)} puzzles ({puzzles_per_hour:.1f} puzzles/hour)")
        print(f"Performance: {seq_metrics.get('avg_weighted_accuracy', 0):.1f}% weighted accuracy")
        print(f"Perfect sequences: {seq_metrics.get('perfect_sequences', 0)}/{report['total_puzzles']} ({seq_metrics.get('perfect_sequence_rate', 0):.1f}%)")
        
        # Show top 3 themes
        theme_items = list(report.get('theme_performance', {}).items())
        if theme_items:
            theme_items.sort(key=lambda x: x[1]['avg_weighted_accuracy'], reverse=True)
            print("Top themes: " + ", ".join([f"{t[0]}({t[1]['avg_weighted_accuracy']:.0f}%)" for t in theme_items[:3]]))
    
    def unlimited_puzzle_generator(self, rating_min: int = 800, rating_max: int = 3000, 
                                   themes_filter: Optional[List[str]] = None, 
                                   batch_size: int = 1000):
        """
        Generator that yields puzzles continuously without running out
        Queries database in batches and cycles through different rating ranges
        """
        db = PuzzleDatabase(self.puzzle_db_path)
        
        # Define rating ranges to cycle through
        rating_ranges = [
            (800, 1200), (1200, 1600), (1600, 2000), (2000, 2400), (2400, 3000),
            (rating_min, rating_max)  # User specified range
        ]
        
        range_index = 0
        processed_ids = set()
        
        while self.should_continue_session():
            # Get current rating range
            current_min, current_max = rating_ranges[range_index % len(rating_ranges)]
            
            # Query puzzles in current range
            puzzles = db.query_puzzles(
                themes=themes_filter,
                min_rating=current_min,
                max_rating=current_max,
                quantity=batch_size
            )
            
            # Filter out already processed puzzles
            new_puzzles = [p for p in puzzles if p.id not in processed_ids]
            
            if not new_puzzles:
                # Move to next rating range if no new puzzles
                range_index += 1
                if range_index >= len(rating_ranges) * 2:  # Prevent infinite loop
                    print("🔄 Cycling through all rating ranges again...")
                    processed_ids.clear()  # Allow re-analysis of puzzles
                    range_index = 0
                continue
            
            # Yield puzzles from current batch
            for puzzle in new_puzzles:
                if not self.should_continue_session():
                    return
                
                processed_ids.add(puzzle.id)
                yield puzzle
            
            # Move to next rating range
            range_index += 1
    
    def get_engine_info(self) -> Dict[str, str]:
        """Get engine information via UCI protocol"""
        try:
            process = subprocess.Popen(
                self.engine_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,
                cwd=os.path.dirname(self.engine_path) if self.engine_type == 'bat' else None
            )
            
            engine_info = {}
            
            # Send UCI command
            if process.stdin:
                process.stdin.write("uci\n")
                process.stdin.flush()
            
            # Read UCI response
            start_time = time.time()
            while time.time() - start_time < 5:  # 5 second timeout
                if not process.stdout:
                    break
                    
                if process.poll() is not None:
                    break
                
                try:
                    line = process.stdout.readline()
                    if not line:
                        time.sleep(0.1)
                        continue
                    
                    line = line.strip()
                    
                    if line.startswith("id name"):
                        engine_info['name'] = line[8:].strip()
                    elif line.startswith("id author"):
                        engine_info['author'] = line[9:].strip()
                    elif line == "uciok":
                        break
                        
                except:
                    break
            
            # Ensure process is terminated
            try:
                process.terminate()
                process.wait(timeout=2)
            except:
                try:
                    process.kill()
                    process.wait(timeout=1)
                except:
                    pass
            
            return engine_info
            
        except Exception as e:
            print(f"Warning: Could not get engine info via UCI: {e}")
            return {'name': os.path.basename(self.engine_path)}
    
    def get_engine_move(self, fen: str, time_seconds: float = 10.0) -> Optional[str]:
        """Get the test engine's best move for a position with generous time"""
        try:
            process = subprocess.Popen(
                self.engine_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,
                cwd=os.path.dirname(self.engine_path) if self.engine_type == 'bat' else None
            )
            
            # UCI commands
            commands = [
                "uci",
                "isready",
                f"position fen {fen}",
                f"go movetime {int(time_seconds * 1000)}"  # Convert to milliseconds
            ]
            
            for cmd in commands:
                if process.stdin:
                    process.stdin.write(f"{cmd}\n")
                    process.stdin.flush()
                if cmd == "uci" or cmd == "isready":
                    time.sleep(0.2)  # Brief pause for initialization
            
            # Read output until bestmove
            best_move = None
            output_lines = []
            start_time = time.time()
            timeout = time_seconds + 3  # Add 3 second buffer
            
            while time.time() - start_time < timeout:
                if not process.stdout:
                    break
                
                # Use poll to check if process is still running
                if process.poll() is not None:
                    break
                    
                try:
                    line = process.stdout.readline()
                    if not line:
                        time.sleep(0.1)  # Brief pause if no output
                        continue
                        
                    line = line.strip()
                    output_lines.append(line)
                    
                    if line.startswith("bestmove"):
                        parts = line.split()
                        if len(parts) > 1:
                            best_move = parts[1]
                        break
                except:
                    break
            
            # Ensure process is terminated
            try:
                process.terminate()
                process.wait(timeout=2)
            except:
                try:
                    process.kill()
                    process.wait(timeout=1)
                except:
                    pass
            
            return best_move
            
        except Exception as e:
            print(f"Error getting {self.engine_name} move: {e}")
            return None
    
    def get_stockfish_top_moves(self, fen: str, num_moves: int = 5, time_seconds: float = 2.0) -> List[Tuple[str, int]]:
        """Get Stockfish's top N moves with scores (move, centipawn_score)"""
        try:
            with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
                board = chess.Board(fen)
                
                # Use analyse with multipv parameter instead of configure
                result = engine.analyse(
                    board, 
                    chess.engine.Limit(time=time_seconds),
                    multipv=num_moves
                )
                
                moves_with_scores = []
                for analysis in result:
                    if 'pv' in analysis and analysis['pv']:
                        move = analysis['pv'][0]
                        score = analysis.get('score', chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE))
                        
                        # Convert score to centipawns from white's perspective
                        if score.is_mate():
                            # Convert mate scores to large centipawn values
                            mate_in = score.white().mate()
                            if mate_in is not None:
                                cp_score = 10000 - abs(mate_in) * 100 if mate_in > 0 else -10000 + abs(mate_in) * 100
                            else:
                                cp_score = 0
                        else:
                            cp_score = score.white().score()
                        
                        moves_with_scores.append((str(move), cp_score))
                
                return moves_with_scores
                
        except Exception as e:
            print(f"Error getting Stockfish moves: {e}")
            return []
    
    def score_engine_move(self, engine_move: str, stockfish_moves: List[Tuple[str, int]]) -> Tuple[int, int]:
        """
        Score engine's move based on Stockfish ranking
        Returns: (score, rank) where rank is 1-5 or 0 if not in top 5
        """
        if not engine_move or not stockfish_moves:
            return 0, 0
        
        for rank, (sf_move, _) in enumerate(stockfish_moves, 1):
            if engine_move == sf_move:
                score = 6 - rank  # 5pts for 1st, 4pts for 2nd, ..., 1pt for 5th
                return score, rank
        
        return 0, 0  # Not in top 5
    
    def calculate_weighted_sequence_score(self, sequence_results: List[bool]) -> float:
        """
        Calculate weighted accuracy score for puzzle sequence
        Later moves in the sequence are weighted more heavily
        Returns: weighted accuracy percentage (0-100)
        """
        if not sequence_results:
            return 0.0
        
        total_weight = 0.0
        weighted_correct = 0.0
        
        for i, is_correct in enumerate(sequence_results):
            # Exponential weighting: later moves are more important
            # Weight increases exponentially: 1, 1.5, 2.25, 3.375, etc.
            weight = 1.5 ** i
            total_weight += weight
            
            if is_correct:
                weighted_correct += weight
        
        return (weighted_correct / total_weight) * 100 if total_weight > 0 else 0.0
    
    def parse_puzzle_sequence(self, puzzle: Puzzle) -> List[str]:
        """Parse puzzle moves into sequence of individual moves"""
        if not puzzle.moves:
            return []
        return puzzle.moves.split()
    
    def get_engine_move_with_time_control(self, fen: str, time_control: TimeControl, 
                                         suggested_time_seconds: float = 20.0) -> Tuple[Optional[str], float, Dict]:
        """
        Get engine move using proper time control with realistic time management
        Always accepts the move regardless of time taken
        
        Returns: (move, actual_time_used_seconds, time_analysis)
        """
        try:
            board = chess.Board(fen)
            is_white = board.turn
            
            process = subprocess.Popen(
                self.engine_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,
                cwd=os.path.dirname(self.engine_path) if self.engine_type == 'bat' else None
            )
            
            # Calculate time allocation
            remaining_time_ms = time_control.get_time_for_move(is_white)
            increment_ms = time_control.increment_ms
            
            # UCI commands with proper time control
            commands = [
                "uci",
                "isready",
                f"position fen {fen}",
                f"go wtime {time_control.white_time_remaining} btime {time_control.black_time_remaining} "
                f"winc {increment_ms} binc {increment_ms}"
            ]
            
            for cmd in commands:
                if process.stdin:
                    process.stdin.write(f"{cmd}\n")
                    process.stdin.flush()
                if cmd == "uci" or cmd == "isready":
                    time.sleep(0.2)
            
            # Track actual thinking time
            start_time = time.time()
            best_move = None
            output_lines = []
            
            # Give engine generous maximum time (suggested_time * 3) but don't enforce strictly
            max_wait_time = suggested_time_seconds * 3
            
            while time.time() - start_time < max_wait_time:
                if not process.stdout:
                    break
                
                if process.poll() is not None:
                    break
                    
                try:
                    line = process.stdout.readline()
                    if not line:
                        time.sleep(0.1)
                        continue
                        
                    line = line.strip()
                    output_lines.append(line)
                    
                    if line.startswith("bestmove"):
                        parts = line.split()
                        if len(parts) > 1:
                            best_move = parts[1]
                        break
                except:
                    break
            
            actual_time_used = time.time() - start_time
            actual_time_used_ms = int(actual_time_used * 1000)
            
            # Update time control
            time_control.consume_time(is_white, actual_time_used_ms)
            
            # Analyze time usage
            time_analysis = {
                'suggested_time_seconds': suggested_time_seconds,
                'actual_time_seconds': actual_time_used,
                'remaining_time_before_ms': remaining_time_ms,
                'remaining_time_after_ms': time_control.get_time_for_move(is_white),
                'increment_ms': increment_ms,
                'time_pressure': time_control.is_time_trouble(is_white),
                'time_efficiency': min(1.0, suggested_time_seconds / actual_time_used) if actual_time_used > 0 else 1.0,
                'exceeded_suggestion': actual_time_used > suggested_time_seconds,
                'time_management_score': self._calculate_time_management_score(suggested_time_seconds, actual_time_used)
            }
            
            # Ensure process is terminated
            try:
                process.terminate()
                process.wait(timeout=2)
            except:
                try:
                    process.kill()
                    process.wait(timeout=1)
                except:
                    pass
            
            return best_move, actual_time_used, time_analysis
            
        except Exception as e:
            print(f"Error getting {self.engine_name} move: {e}")
            return None, 0.0, {}
    
    def _calculate_time_management_score(self, suggested_time: float, actual_time: float) -> float:
        """
        Calculate time management score (0-1)
        1.0 = used exactly suggested time
        0.8+ = used reasonable time (within 50% of suggestion)
        0.5-0.8 = used too much or too little time
        0.0-0.5 = poor time management
        """
        if actual_time <= 0 or suggested_time <= 0:
            return 0.0
        
        ratio = actual_time / suggested_time
        
        if 0.8 <= ratio <= 1.2:  # Within 20% of suggestion
            return 1.0
        elif 0.5 <= ratio <= 1.5:  # Within 50% of suggestion
            return 0.8
        elif 0.2 <= ratio <= 2.0:  # Within reasonable bounds
            return 0.6
        else:
            return 0.3  # Poor time management
    
    def analyze_puzzle_sequence(self, puzzle: Puzzle, suggested_time_seconds: float = 20.0) -> Optional[Dict]:
        """
        Analyze complete puzzle sequence with proper time control management
        Always accepts engine moves and analyzes time usage separately
        """
        print(f"Analyzing puzzle {puzzle.id} (Rating: {puzzle.rating})")
        print(f"Themes: {puzzle.themes}")
        print(f"Original FEN: {puzzle.fen}")
        
        sequence = self.parse_puzzle_sequence(puzzle)
        if len(sequence) < 2:
            print(f"❌ Insufficient moves in sequence: {len(sequence)}")
            return None
        
        print(f"Solution sequence ({len(sequence)} moves): {' '.join(sequence)}")
        
        # Initialize time control for this puzzle
        puzzle_time_control = TimeControl(
            self.default_time_control.base_time_ms / 60000,  # Convert back to minutes
            self.default_time_control.increment_ms / 1000    # Convert back to seconds
        )
        
        # Initialize tracking variables
        board = chess.Board(puzzle.fen)
        sequence_results = []
        position_analyses = []
        engine_found_all = True
        total_time_used = 0.0
        time_analyses = []
        
        # Process each position in the sequence
        for move_index in range(0, len(sequence), 2):
            position_num = (move_index // 2) + 1
            
            # Check if we have both opponent move and expected response
            if move_index >= len(sequence):
                break
                
            opponent_move_text = sequence[move_index]
            expected_move_text = sequence[move_index + 1] if move_index + 1 < len(sequence) else None
            
            if not expected_move_text:
                print(f"Position {position_num}: No expected response for opponent move {opponent_move_text}")
                break
            
            print(f"\n--- Position {position_num} ---")
            current_fen = board.fen()
            turn_info = f"{'White' if board.turn else 'Black'} to move"
            print(f"Current position: {turn_info}")
            
            # Apply opponent's move (simulate time usage for opponent)
            try:
                try:
                    opponent_move = chess.Move.from_uci(opponent_move_text)
                    if opponent_move not in board.legal_moves:
                        raise ValueError("Move not legal")
                except:
                    opponent_move = board.parse_san(opponent_move_text)
                
                # Simulate opponent thinking time (half of suggested time)
                opponent_time_ms = int((suggested_time_seconds / 2) * 1000)
                puzzle_time_control.consume_time(board.turn, opponent_time_ms)
                
                board.push(opponent_move)
                challenge_fen = board.fen()
                print(f"After opponent plays {opponent_move_text}: {challenge_fen}")
                
            except Exception as e:
                print(f"❌ Cannot apply opponent move {opponent_move_text}: {e}")
                break
            
            # Parse expected move
            try:
                try:
                    expected_move = chess.Move.from_uci(expected_move_text)
                    if expected_move not in board.legal_moves:
                        raise ValueError("Move not legal")
                    expected_move_uci = str(expected_move)
                except:
                    expected_move = board.parse_san(expected_move_text)
                    expected_move_uci = str(expected_move)
                
                print(f"Expected response: {expected_move_uci}")
                
            except Exception as e:
                print(f"❌ Cannot parse expected move {expected_move_text}: {e}")
                break
            
            # Get engine's move with proper time control
            remaining_time = puzzle_time_control.get_time_for_move(board.turn)
            print(f"Challenging {self.engine_name} (Time remaining: {puzzle_time_control.format_time(remaining_time)}, suggested: {suggested_time_seconds}s)...")
            
            engine_move, actual_time, time_analysis = self.get_engine_move_with_time_control(
                challenge_fen, puzzle_time_control, suggested_time_seconds
            )
            
            total_time_used += actual_time
            time_analyses.append(time_analysis)
            
            # Always report move, never skip analysis
            if not engine_move:
                print(f"❌ {self.engine_name} failed to return any move (took {actual_time:.1f}s)")
                sequence_results.append(False)
                engine_found_all = False
                # Continue to next position anyway
                try:
                    board.push(expected_move)
                except:
                    break
                continue
            
            # Report engine's move with time analysis
            time_status = ""
            if time_analysis.get('exceeded_suggestion', False):
                time_status = f" [exceeded suggested time by {actual_time - suggested_time_seconds:.1f}s]"
            elif time_analysis.get('time_pressure', False):
                time_status = f" [time pressure]"
            
            print(f"{self.engine_name} chose: {engine_move} (took {actual_time:.1f}s{time_status})")
            
            # Get Stockfish analysis for this position
            stockfish_moves = self.get_stockfish_top_moves(challenge_fen, 5, 2.0)
            if stockfish_moves:
                print("Stockfish's top 5:")
                for i, (move, score) in enumerate(stockfish_moves, 1):
                    indicator = "🎯" if move == expected_move_uci else "  "
                    engine_indicator = "👑" if move == engine_move else "  "
                    print(f"  {i}. {move} (score: {score:+d}) {indicator}{engine_indicator}")
            
            # Score engine's move
            score, rank = self.score_engine_move(engine_move, stockfish_moves)
            found_solution = engine_move == expected_move_uci
            sequence_results.append(found_solution)
            
            if found_solution:
                print(f"✅ {self.engine_name} found the correct move! (Stockfish rank: #{rank if rank > 0 else 'not in top 5'})")
            else:
                print(f"❌ {self.engine_name} missed the correct move (chose rank #{rank if rank > 0 else 'not in top 5'})")
                engine_found_all = False
            
            # Store enhanced position analysis
            position_analysis = {
                'position_number': position_num,
                'challenge_fen': challenge_fen,
                'opponent_move': opponent_move_text,
                'expected_move': expected_move_uci,
                'engine_move': engine_move,
                'engine_found_solution': found_solution,
                'engine_stockfish_score': score,
                'engine_stockfish_rank': rank,
                'stockfish_top_moves': stockfish_moves,
                'time_analysis': time_analysis,
                'turn_info': f"{'White' if not board.turn else 'Black'} to move after opponent's {opponent_move_text}"
            }
            position_analyses.append(position_analysis)
            
            # Apply the expected move to continue sequence
            try:
                board.push(expected_move)
                # Simulate expected move time consumption
                expected_time_ms = int((actual_time if found_solution else suggested_time_seconds) * 1000)
                puzzle_time_control.consume_time(board.turn, expected_time_ms)
            except Exception as e:
                print(f"❌ Cannot continue sequence after {expected_move_text}: {e}")
                break
        
        # Calculate sequence metrics
        if not sequence_results:
            print("❌ No positions were successfully analyzed")
            return None
        
        sequence_accuracy = (sum(sequence_results) / len(sequence_results)) * 100
        weighted_accuracy = self.calculate_weighted_sequence_score(sequence_results)
        
        # Time management analysis
        avg_time_per_move = total_time_used / len(time_analyses) if time_analyses else 0
        time_efficiency_scores = [ta.get('time_management_score', 0) for ta in time_analyses if ta]
        avg_time_management = sum(time_efficiency_scores) / len(time_efficiency_scores) if time_efficiency_scores else 0
        total_time_exceeded = sum(1 for ta in time_analyses if ta.get('exceeded_suggestion', False))
        
        print(f"\n🎯 SEQUENCE SUMMARY:")
        print(f"Positions analyzed: {len(sequence_results)}")
        print(f"Correct solutions: {sum(sequence_results)}/{len(sequence_results)}")
        print(f"Linear accuracy: {sequence_accuracy:.1f}%")
        print(f"Weighted accuracy: {weighted_accuracy:.1f}%")
        print(f"Perfect sequence: {'Yes' if engine_found_all else 'No'}")
        print(f"Time management: {avg_time_management:.1f}/1.0 avg score")
        print(f"Average time per move: {avg_time_per_move:.1f}s")
        print(f"Exceeded suggestions: {total_time_exceeded}/{len(time_analyses)}")
        
        # Compile comprehensive result with time analysis
        result = {
            'puzzle_id': puzzle.id,
            'original_fen': puzzle.fen,
            'rating': puzzle.rating,
            'themes': puzzle.themes.split() if puzzle.themes else [],
            'solution_sequence': sequence,
            'positions_analyzed': len(sequence_results),
            'sequence_results': sequence_results,
            'sequence_accuracy_linear': sequence_accuracy,
            'sequence_accuracy_weighted': weighted_accuracy,
            'perfect_sequence': engine_found_all,
            'position_analyses': position_analyses,
            'suggested_time_seconds': suggested_time_seconds,
            'total_time_used': total_time_used,
            'avg_time_per_move': avg_time_per_move,
            'time_management_score': avg_time_management,
            'time_exceeded_count': total_time_exceeded,
            'time_analyses': time_analyses,
            'timestamp': datetime.now().isoformat()
        }
        
        print("-" * 60)
        return result
    
    def analyze_puzzle(self, puzzle: Puzzle, engine_time: float = 10.0) -> Optional[Dict]:
        """Analyze a puzzle using the enhanced sequence-based approach"""
        return self.analyze_puzzle_sequence(puzzle, engine_time)
    
    def extract_puzzle_ids_from_results(self, results_file: str) -> List[str]:
        """Extract puzzle IDs from a previous analysis results file"""
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            puzzle_ids = []
            
            # Try to extract from analysis_results
            if 'analysis_results' in data:
                puzzle_ids = [result['puzzle_id'] for result in data['analysis_results'] if 'puzzle_id' in result]
            # Fallback: try direct list of results
            elif isinstance(data, list):
                puzzle_ids = [result['puzzle_id'] for result in data if 'puzzle_id' in result]
            
            print(f"Extracted {len(puzzle_ids)} puzzle IDs from {results_file}")
            return puzzle_ids
            
        except Exception as e:
            print(f"Error extracting puzzle IDs from {results_file}: {e}")
            return []
    
    def run_analysis(self, 
                     num_puzzles: int = 100,
                     rating_min: int = 1200,
                     rating_max: int = 2000,
                     suggested_time: float = 20.0,  # Changed from engine_time
                     themes_filter: Optional[List[str]] = None,
                     force_puzzle_ids: Optional[List[str]] = None,
                     comparison_file: Optional[str] = None,
                     duration_hours: Optional[float] = None,
                     resume_checkpoint: Optional[str] = None,
                     time_control: Optional[TimeControl] = None) -> List[Dict]:
        """Run analysis with proper time control management"""
        
        # Set time control for this session
        if time_control:
            self.default_time_control = time_control
        
        # Handle checkpoint resume
        if resume_checkpoint:
            if not self.load_checkpoint(resume_checkpoint):
                print("Failed to load checkpoint, starting fresh session")
            else:
                print("Resumed from checkpoint, continuing analysis...")
        
        # Set up session parameters
        self.session_active = True  # Always activate session for any analysis
        
        if duration_hours:
            self.session_duration_hours = duration_hours
            self.session_start_time = time.time()
            print(f"🕐 Starting extended session: {duration_hours} hours")
            
            # Start progress monitoring for extended sessions
            self.start_progress_monitoring_thread()
        else:
            # For normal analysis, still set session as active but no time limits
            self.session_start_time = time.time()
        
        # Handle comparison file input
        if comparison_file and not force_puzzle_ids:
            force_puzzle_ids = self.extract_puzzle_ids_from_results(comparison_file)
            if not force_puzzle_ids:
                print(f"Warning: Could not extract puzzle IDs from {comparison_file}, proceeding with normal analysis")
        
        # Print session header
        if force_puzzle_ids:
            print(f"{self.engine_name} Universal Puzzle Analysis - {len(force_puzzle_ids)} forced puzzles")
            print(f"Engine: {self.engine_name}")
            print(f"Puzzle forcing mode: Using specific puzzle IDs")
            print(f"Engine thinking time: {suggested_time} seconds")
            if comparison_file:
                print(f"Comparison file: {comparison_file}")
        elif duration_hours:
            print(f"{self.engine_name} Extended Puzzle Analysis - {duration_hours} hour session")
            print(f"Engine: {self.engine_name}")
            print(f"Rating range: {rating_min}-{rating_max}")
            print(f"Engine thinking time: {suggested_time} seconds")
            print(f"Unlimited puzzle streaming: Enabled")
            if themes_filter:
                print(f"Theme filter: {themes_filter}")
        else:
            print(f"{self.engine_name} Universal Puzzle Analysis - {num_puzzles} puzzles")
            print(f"Engine: {self.engine_name}")
            print(f"Rating range: {rating_min}-{rating_max}")
            print(f"Engine thinking time: {suggested_time} seconds")
            if themes_filter:
                print(f"Theme filter: {themes_filter}")
        print("=" * 60)
        
        # Get puzzles based on mode
        if force_puzzle_ids:
            # Get specific puzzles by ID
            db = PuzzleDatabase(self.puzzle_db_path)
            puzzles = []
            for puzzle_id in force_puzzle_ids:
                puzzle = db.get_puzzle_by_id(puzzle_id)
                if puzzle:
                    puzzles.append(puzzle)
                else:
                    print(f"Warning: Puzzle ID {puzzle_id} not found in database")
            puzzle_source = iter(puzzles)
            total_expected = len(puzzles)
            
        elif duration_hours:
            # Use unlimited puzzle generator for extended sessions
            puzzle_source = self.unlimited_puzzle_generator(rating_min, rating_max, themes_filter)
            total_expected = "unlimited"
            
        else:
            # Normal puzzle query
            db = PuzzleDatabase(self.puzzle_db_path)
            puzzles = db.query_puzzles(
                themes=themes_filter,
                min_rating=rating_min,
                max_rating=rating_max,
                quantity=num_puzzles
            )
            puzzle_source = iter(puzzles)
            total_expected = len(puzzles)
        
        if not force_puzzle_ids and total_expected != "unlimited" and total_expected == 0:
            print("No puzzles found matching criteria!")
            return []
        
        if total_expected != "unlimited":
            print(f"Found {total_expected} puzzles to analyze")
        else:
            print("Using unlimited puzzle streaming for extended session")
        print("-" * 60)
        
        # Analyze puzzles
        results = []
        puzzles_analyzed = len(self.results)  # Account for resumed sessions
        
        try:
            for puzzle in puzzle_source:
                if not self.should_continue_session():
                    print("\n🛑 Session stopping...")
                    break
                
                puzzle_num = puzzles_analyzed + 1
                if total_expected != "unlimited":
                    print(f"Puzzle {puzzle_num}/{total_expected}")
                else:
                    elapsed_hours = (time.time() - self.session_start_time) / 3600 if self.session_start_time else 0
                    remaining_hours = max(0, self.session_duration_hours - elapsed_hours) if self.session_duration_hours else float('inf')
                    remaining_str = f", {remaining_hours:.1f}h remaining" if remaining_hours != float('inf') else ""
                    print(f"Puzzle #{puzzle_num} ({elapsed_hours:.1f}h elapsed{remaining_str})")
                
                result = self.analyze_puzzle(puzzle, suggested_time)
                if result:
                    results.append(result)
                    self.results.append(result)
                    puzzles_analyzed += 1
                
                # Save checkpoint periodically during long runs
                if duration_hours and puzzles_analyzed % 10 == 0:
                    self.save_progress_checkpoint(puzzles_analyzed, puzzle.id)
        
        except KeyboardInterrupt:
            print("\n🛑 Analysis interrupted by user")
            self.interrupted = True
        except Exception as e:
            print(f"\n❌ Analysis error: {e}")
            self.interrupted = True
        finally:
            # Clean up extended session
            if duration_hours:
                self.session_active = False
                if self.stop_event:
                    self.stop_event.set()
                if self.progress_thread and self.progress_thread.is_alive():
                    self.progress_thread.join(timeout=2)
                
                # Final checkpoint save
                if self.results:
                    self.save_progress_checkpoint(len(self.results))
                    print(f"\n💾 Final checkpoint saved with {len(self.results)} results")
        
        return results
    
    def generate_report(self, results: List[Dict]) -> Dict:
        """Generate enhanced analysis report with time management metrics"""
        if not results:
            return {}
        
        total_puzzles = len(results)
        
        # Legacy single-position metrics (for backward compatibility)
        total_positions = sum(r.get('positions_analyzed', 1) for r in results)
        
        # Sequence-based metrics
        linear_accuracies = [r.get('sequence_accuracy_linear', 0) for r in results if 'sequence_accuracy_linear' in r]
        weighted_accuracies = [r.get('sequence_accuracy_weighted', 0) for r in results if 'sequence_accuracy_weighted' in r]
        perfect_sequences = sum(1 for r in results if r.get('perfect_sequence', False))
        
        avg_linear_accuracy = sum(linear_accuracies) / len(linear_accuracies) if linear_accuracies else 0
        avg_weighted_accuracy = sum(weighted_accuracies) / len(weighted_accuracies) if weighted_accuracies else 0
        perfect_sequence_rate = (perfect_sequences / total_puzzles) * 100
        
        # Rating analysis for estimation
        perfect_puzzle_ratings = [r['rating'] for r in results if r.get('perfect_sequence', False)]
        high_accuracy_ratings = [r['rating'] for r in results if r.get('sequence_accuracy_weighted', 0) >= 80]
        
        # Calculate estimated rating range where engine performs well
        estimated_rating_range = {
            'perfect_sequences': {
                'count': len(perfect_puzzle_ratings),
                'min_rating': min(perfect_puzzle_ratings) if perfect_puzzle_ratings else 0,
                'max_rating': max(perfect_puzzle_ratings) if perfect_puzzle_ratings else 0,
                'avg_rating': sum(perfect_puzzle_ratings) / len(perfect_puzzle_ratings) if perfect_puzzle_ratings else 0
            },
            'high_accuracy': {
                'count': len(high_accuracy_ratings),
                'min_rating': min(high_accuracy_ratings) if high_accuracy_ratings else 0,
                'max_rating': max(high_accuracy_ratings) if high_accuracy_ratings else 0,
                'avg_rating': sum(high_accuracy_ratings) / len(high_accuracy_ratings) if high_accuracy_ratings else 0
            }
        }
        
        # Position-by-position performance analysis
        position_performance = {}
        for result in results:
            if 'position_analyses' in result:
                for pos_analysis in result['position_analyses']:
                    pos_num = pos_analysis['position_number']
                    if pos_num not in position_performance:
                        position_performance[pos_num] = {'total': 0, 'correct': 0, 'stockfish_scores': []}
                    
                    position_performance[pos_num]['total'] += 1
                    if pos_analysis['engine_found_solution']:
                        position_performance[pos_num]['correct'] += 1
                    position_performance[pos_num]['stockfish_scores'].append(pos_analysis['engine_stockfish_score'])
        
        # Calculate position accuracy rates
        for pos_num in position_performance:
            data = position_performance[pos_num]
            data['accuracy_rate'] = (data['correct'] / data['total']) * 100
            data['avg_stockfish_score'] = sum(data['stockfish_scores']) / len(data['stockfish_scores'])
        
        # Theme analysis with sequence metrics
        theme_performance = {}
        for result in results:
            for theme in result.get('themes', []):
                if theme not in theme_performance:
                    theme_performance[theme] = {
                        'total': 0, 
                        'perfect_sequences': 0,
                        'linear_accuracy_sum': 0,
                        'weighted_accuracy_sum': 0,
                        'ratings': []
                    }
                
                theme_data = theme_performance[theme]
                theme_data['total'] += 1
                theme_data['linear_accuracy_sum'] += result.get('sequence_accuracy_linear', 0)
                theme_data['weighted_accuracy_sum'] += result.get('sequence_accuracy_weighted', 0)
                theme_data['ratings'].append(result['rating'])
                
                if result.get('perfect_sequence', False):
                    theme_data['perfect_sequences'] += 1
        
        # Calculate theme averages
        for theme in theme_performance:
            data = theme_performance[theme]
            data['avg_linear_accuracy'] = data['linear_accuracy_sum'] / data['total']
            data['avg_weighted_accuracy'] = data['weighted_accuracy_sum'] / data['total']
            data['perfect_sequence_rate'] = (data['perfect_sequences'] / data['total']) * 100
            data['avg_rating'] = sum(data['ratings']) / len(data['ratings'])
        
        # Accuracy distribution
        accuracy_buckets = {'0-20%': 0, '20-40%': 0, '40-60%': 0, '60-80%': 0, '80-100%': 0}
        for accuracy in weighted_accuracies:
            if accuracy < 20:
                accuracy_buckets['0-20%'] += 1
            elif accuracy < 40:
                accuracy_buckets['20-40%'] += 1
            elif accuracy < 60:
                accuracy_buckets['40-60%'] += 1
            elif accuracy < 80:
                accuracy_buckets['60-80%'] += 1
            else:
                accuracy_buckets['80-100%'] += 1
        
        # Time management analysis
        time_management_scores = [r.get('time_management_score', 0) for r in results if 'time_management_score' in r]
        time_exceeded_total = sum(r.get('time_exceeded_count', 0) for r in results)
        total_moves = sum(r.get('positions_analyzed', 0) for r in results)
        avg_time_per_move = sum(r.get('avg_time_per_move', 0) for r in results) / len(results) if results else 0
        
        time_metrics = {
            'avg_time_management_score': sum(time_management_scores) / len(time_management_scores) if time_management_scores else 0,
            'time_exceeded_rate': (time_exceeded_total / total_moves) * 100 if total_moves > 0 else 0,
            'avg_time_per_move': avg_time_per_move,
            'total_time_exceeded': time_exceeded_total,
            'total_moves': total_moves
        }
        
        report = {
            'engine_name': self.engine_name,
            'engine_info': self.engine_info,
            'total_puzzles': total_puzzles,
            'total_positions_analyzed': total_positions,
            'sequence_metrics': {
                'avg_linear_accuracy': avg_linear_accuracy,
                'avg_weighted_accuracy': avg_weighted_accuracy,
                'perfect_sequences': perfect_sequences,
                'perfect_sequence_rate': perfect_sequence_rate
            },
            'estimated_rating_analysis': estimated_rating_range,
            'position_performance': position_performance,
            'theme_performance': theme_performance,
            'accuracy_distribution': accuracy_buckets,
            'time_management_metrics': time_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        return report
    
    def print_report(self, report: Dict):
        """Print enhanced analysis report with time management insights"""
        engine_name = report.get('engine_name', 'Unknown Engine')
        
        print("\n" + "=" * 80)
        print(f"{engine_name.upper()} ENHANCED PUZZLE ANALYSIS REPORT")
        print("=" * 80)
        
        print(f"Engine: {engine_name}")
        if 'engine_info' in report and 'author' in report['engine_info']:
            print(f"Author: {report['engine_info']['author']}")
        
        print(f"Puzzles Analyzed: {report['total_puzzles']}")
        print(f"Total Positions: {report['total_positions_analyzed']}")
        
        # Sequence Performance Metrics
        seq_metrics = report['sequence_metrics']
        print(f"\n🎯 SEQUENCE PERFORMANCE:")
        print(f"Average Linear Accuracy: {seq_metrics['avg_linear_accuracy']:.1f}%")
        print(f"Average Weighted Accuracy: {seq_metrics['avg_weighted_accuracy']:.1f}%")
        print(f"Perfect Sequences: {seq_metrics['perfect_sequences']}/{report['total_puzzles']} ({seq_metrics['perfect_sequence_rate']:.1f}%)")
        
        # Rating Analysis for Engine Estimation
        rating_analysis = report['estimated_rating_analysis']
        print(f"\n📊 ESTIMATED {engine_name.upper()} RATING ANALYSIS:")
        
        perfect_data = rating_analysis['perfect_sequences']
        if perfect_data['count'] > 0:
            print(f"Perfect Sequences ({perfect_data['count']} puzzles):")
            print(f"  Rating Range: {perfect_data['min_rating']}-{perfect_data['max_rating']}")
            print(f"  Average Rating: {perfect_data['avg_rating']:.0f}")
        
        high_acc_data = rating_analysis['high_accuracy']
        if high_acc_data['count'] > 0:
            print(f"High Accuracy ≥80% ({high_acc_data['count']} puzzles):")
            print(f"  Rating Range: {high_acc_data['min_rating']}-{high_acc_data['max_rating']}")
            print(f"  Average Rating: {high_acc_data['avg_rating']:.0f}")
        
        # Engine Estimated Rating Range
        if perfect_data['count'] > 0 and high_acc_data['count'] > 0:
            estimated_min = min(perfect_data['min_rating'], high_acc_data['min_rating'])
            estimated_max = max(perfect_data['max_rating'], high_acc_data['max_rating'])
            estimated_avg = (perfect_data['avg_rating'] + high_acc_data['avg_rating']) / 2
            print(f"\n🎲 ESTIMATED {engine_name.upper()} RATING: {estimated_min}-{estimated_max} (avg: {estimated_avg:.0f})")
        
        # Accuracy Distribution
        print(f"\nAccuracy Distribution:")
        for bucket, count in report['accuracy_distribution'].items():
            percentage = (count / report['total_puzzles']) * 100
            bar = "█" * int(percentage / 2)
            print(f"  {bucket:8s}: {count:3d} ({percentage:4.1f}%) {bar}")
        
        # Position-by-Position Performance
        if report['position_performance']:
            print(f"\n📍 POSITION PERFORMANCE (Sequence Depth Analysis):")
            for pos_num in sorted(report['position_performance'].keys()):
                data = report['position_performance'][pos_num]
                print(f"  Position {pos_num}: {data['correct']}/{data['total']} ({data['accuracy_rate']:.1f}%) - Avg SF Score: {data['avg_stockfish_score']:.1f}")
        
        # Theme Performance (Top 15)
        print(f"\n🎨 THEME PERFORMANCE (Top 15 by Weighted Accuracy):")
        theme_items = list(report['theme_performance'].items())
        theme_items.sort(key=lambda x: x[1]['avg_weighted_accuracy'], reverse=True)
        
        for theme, data in theme_items[:15]:
            perfect_rate = data['perfect_sequence_rate']
            weighted_acc = data['avg_weighted_accuracy']
            count = data['total']
            avg_rating = data['avg_rating']
            print(f"  {theme:20s}: {weighted_acc:4.1f}% weighted ({perfect_rate:4.1f}% perfect) [{count:2d} puzzles, avg {avg_rating:.0f}]")
        
        # Performance Insights
        print(f"\n💡 PERFORMANCE INSIGHTS:")
        
        # Calculate performance degradation through sequence
        if report['position_performance']:
            pos_data = report['position_performance']
            if 1 in pos_data and len(pos_data) > 1:
                first_pos_acc = pos_data[1]['accuracy_rate']
                later_pos_accs = [pos_data[i]['accuracy_rate'] for i in pos_data if i > 1]
                if later_pos_accs:
                    avg_later_acc = sum(later_pos_accs) / len(later_pos_accs)
                    degradation = first_pos_acc - avg_later_acc
                    print(f"  Sequence Degradation: {degradation:+.1f}% (first pos: {first_pos_acc:.1f}%, later avg: {avg_later_acc:.1f}%)")
        
        # Theme strengths and weaknesses
        if theme_items:
            strongest_theme = theme_items[0]
            weakest_theme = theme_items[-1]
            print(f"  Strongest Theme: {strongest_theme[0]} ({strongest_theme[1]['avg_weighted_accuracy']:.1f}%)")
            print(f"  Weakest Theme: {weakest_theme[0]} ({weakest_theme[1]['avg_weighted_accuracy']:.1f}%)")
        
        # Time Management Performance
        if 'time_management_metrics' in report:
            time_metrics = report['time_management_metrics']
            print(f"\n⏱️  TIME MANAGEMENT PERFORMANCE:")
            print(f"Average Time Management Score: {time_metrics['avg_time_management_score']:.2f}/1.00")
            print(f"Average Time Per Move: {time_metrics['avg_time_per_move']:.1f} seconds")
            print(f"Exceeded Suggestions: {time_metrics['total_time_exceeded']}/{time_metrics['total_moves']} ({time_metrics['time_exceeded_rate']:.1f}%)")
        
        print("=" * 80)
    
    def save_results(self, filename: Optional[str] = None):
        """Save results to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            engine_safe_name = self.engine_name.replace(' ', '_').replace('.', '_')
            filename = f"{engine_safe_name}_enhanced_sequence_analysis_{timestamp}.json"
        
        data = {
            'analysis_results': self.results,
            'report': self.generate_report(self.results),
            'metadata': {
                'engine_path': self.engine_path,
                'engine_name': self.engine_name,
                'engine_info': self.engine_info,
                'stockfish_path': self.stockfish_path,
                'puzzle_db_path': self.puzzle_db_path,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nResults saved to: {filename}")


def main():
    """Main execution function with time control options"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Universal Chess Engine Puzzle Analyzer with Realistic Time Controls - Supports .exe and .bat engines')
    parser.add_argument('--engine', required=True, help='Path to the UCI chess engine to test (.exe or .bat file)')
    parser.add_argument('--puzzles', type=int, default=100, help='Number of puzzles to analyze (default: 100, ignored if --duration used)')
    parser.add_argument('--time', type=float, default=20.0, help='Suggested time per position in seconds (default: 20.0)')
    parser.add_argument('--min-rating', type=int, default=1, help='Minimum puzzle rating (default: 1)')
    parser.add_argument('--max-rating', type=int, default=9999, help='Maximum puzzle rating (default: 9999)')
    parser.add_argument('--themes', nargs='*', help='Filter by puzzle themes (optional)')
    parser.add_argument('--comparison-file', type=str, help='JSON file from previous analysis to use same puzzle IDs for comparison')
    parser.add_argument('--force-puzzle-ids', nargs='*', help='Specific puzzle IDs to analyze (optional)')
    
    # Time control arguments
    parser.add_argument('--time-control', type=str, default='30+2', help='Time control format: "base_minutes+increment_seconds" (default: 30+2)')
    
    # Extended session arguments
    parser.add_argument('--duration', type=float, help='Session duration in hours for extended testing (enables unlimited puzzle streaming)')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint file')
    parser.add_argument('--progress-interval', type=int, default=300, help='Progress save interval in seconds (default: 300)')
    parser.add_argument('--report-interval', type=int, default=1800, help='Intermediate report interval in seconds (default: 1800)')
    
    args = parser.parse_args()
    
    # Parse time control
    try:
        if '+' in args.time_control:
            base_str, inc_str = args.time_control.split('+')
            base_minutes = float(base_str)
            increment_seconds = float(inc_str)
        else:
            base_minutes = float(args.time_control)
            increment_seconds = 0.0
        
        time_control = TimeControl(base_minutes, increment_seconds)
        print(f"Using time control: {base_minutes}+{increment_seconds}")
        
    except:
        print(f"Invalid time control format: {args.time_control}, using default 30+2")
        time_control = TimeControl(30.0, 2.0)
    
    try:
        analyzer = UniversalPuzzleAnalyzer(engine_path=args.engine, time_control=time_control)
        
        # Set custom intervals if provided
        if args.progress_interval:
            analyzer.progress_save_interval = args.progress_interval
        if args.report_interval:
            analyzer.report_interval = args.report_interval
        
        # Run analysis with suggested time instead of enforced time
        results = analyzer.run_analysis(
            num_puzzles=args.puzzles,
            rating_min=args.min_rating,
            rating_max=args.max_rating,
            suggested_time=args.time,
            themes_filter=args.themes,
            force_puzzle_ids=args.force_puzzle_ids,
            comparison_file=args.comparison_file,
            duration_hours=args.duration,
            resume_checkpoint=args.resume,
            time_control=time_control
        )
        
        if results or analyzer.results:
            # Use all results (including resumed ones)
            all_results = analyzer.results
            
            # Generate and print enhanced report
            report = analyzer.generate_report(all_results)
            analyzer.print_report(report)
            
            # Save results with timestamp
            analyzer.save_results()
            
            # Print session summary for extended runs
            if args.duration and analyzer.session_start_time:
                total_time = time.time() - analyzer.session_start_time
                puzzles_per_hour = len(all_results) / (total_time / 3600) if total_time > 0 else 0
                print(f"\n📈 EXTENDED SESSION SUMMARY:")
                print(f"Total runtime: {total_time / 3600:.1f} hours")
                print(f"Puzzles analyzed: {len(all_results)}")
                print(f"Analysis rate: {puzzles_per_hour:.1f} puzzles/hour")
                if analyzer.checkpoint_file:
                    print(f"Checkpoint file: {analyzer.checkpoint_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
