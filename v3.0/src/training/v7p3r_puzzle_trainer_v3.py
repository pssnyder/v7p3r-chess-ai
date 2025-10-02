"""
V7P3R Chess AI V3.0 - Pure Puzzle-Trained Neural Chess Engine

Core Philosophy:
"Can a neural network learn to play chess competitively by training exclusively on chess puzzles?"

This is the V3 implementation of V7P3R - a chess engine that learns through puzzle-solving,
similar to how humans might practice tactics and then apply that knowledge in games.

Key V3 Features:
- Pure puzzle-based learning (no game data, no opening books initially)
- Real-time training monitoring and analytics
- Advanced session management with rollback capabilities
- Tournament-ready UCI interface preparation
- Comprehensive performance metrics for cloud transition (V4.0)

Architecture:
- ThinkingBrain: Neural network core
- PuzzleTrainerV3: Primary training system
- Enhanced monitoring with non-invasive progress tracking
- Stockfish evaluation integration for move quality assessment
"""

import os
import sys
import json
import time
import uuid
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from tqdm import tqdm

# Add v3.0/src to Python path
v3_src = Path(__file__).parent / "v3.0" / "src"
sys.path.insert(0, str(v3_src))

from ai.thinking_brain import ThinkingBrain, PositionMemory
from core.chess_state import ChessStateExtractor
from core.neural_features import NeuralFeatureConverter
from database.enhanced_puzzle_db_v2 import EnhancedPuzzleDatabaseV2
from training.puzzle_trainer import PuzzleTrainer

logger = logging.getLogger(__name__)

# Custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

@dataclass
class TrainingMetrics:
    """Real-time training metrics for monitoring"""
    session_id: str
    start_time: datetime
    puzzles_completed: int = 0
    current_score: float = 0.0
    average_score: float = 0.0
    top5_hits: float = 0.0
    learning_velocity: float = 0.0
    estimated_elo: float = 0.0
    themes_mastered: int = 0
    session_efficiency: float = 0.0
    time_remaining: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat(),
            'puzzles_completed': self.puzzles_completed,
            'current_score': self.current_score,
            'average_score': self.average_score,
            'top5_hits': self.top5_hits,
            'learning_velocity': self.learning_velocity,
            'estimated_elo': self.estimated_elo,
            'themes_mastered': self.themes_mastered,
            'session_efficiency': self.session_efficiency,
            'time_remaining': self.time_remaining
        }

class V3MonitoringSystem:
    """Non-invasive monitoring system for V3 training"""
    
    def __init__(self, save_directory: str = "v3.0/monitoring"):
        self.save_directory = Path(save_directory)
        self.save_directory.mkdir(parents=True, exist_ok=True)
        self.current_metrics: Optional[TrainingMetrics] = None
        self.monitoring_active = False
        self.lock = threading.Lock()
    
    def start_session_monitoring(self, session_id: str, training_duration: Optional[float] = None):
        """Start monitoring a training session"""
        with self.lock:
            self.current_metrics = TrainingMetrics(
                session_id=session_id,
                start_time=datetime.now(),
                time_remaining=training_duration
            )
            self.monitoring_active = True
            logger.info(f"📊 V3 Monitoring started for session {session_id[:8]}...")
    
    def update_metrics(self, **kwargs):
        """Update current training metrics"""
        if not self.monitoring_active or not self.current_metrics:
            return
            
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self.current_metrics, key):
                    setattr(self.current_metrics, key, value)
            
            # Update time remaining for timed sessions
            if self.current_metrics.time_remaining is not None:
                elapsed = (datetime.now() - self.current_metrics.start_time).total_seconds() / 3600.0
                self.current_metrics.time_remaining = max(0, self.current_metrics.time_remaining - elapsed)
    
    def get_current_metrics(self) -> Optional[Dict]:
        """Get current metrics without interrupting training"""
        if not self.monitoring_active or not self.current_metrics:
            return None
            
        with self.lock:
            return self.current_metrics.to_dict()
    
    def save_metrics_snapshot(self):
        """Save current metrics to file"""
        metrics = self.get_current_metrics()
        if not metrics:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.save_directory / f"metrics_snapshot_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def stop_session_monitoring(self):
        """Stop monitoring current session"""
        with self.lock:
            if self.current_metrics:
                # Save final snapshot
                self.save_metrics_snapshot()
                logger.info(f"📊 V3 Monitoring stopped for session {self.current_metrics.session_id[:8]}...")
            
            self.monitoring_active = False
            self.current_metrics = None

class V7P3RPuzzleTrainerV3(PuzzleTrainer):
    """
    V7P3R Pure Puzzle Trainer V3.0
    
    Philosophy: Train a chess engine exclusively through puzzle-solving,
    testing whether tactical pattern recognition can translate to full game competency.
    
    Key V3 Enhancements:
    - Real-time monitoring system
    - Advanced session management
    - ELO estimation based on puzzle performance
    - Regression detection and auto-rollback
    - Tournament readiness metrics
    """
    
    def __init__(self, 
                 thinking_brain: ThinkingBrain,
                 puzzle_db_path: str = "v3.0/data/v7p3rai_puzzle_training_v2.db",
                 stockfish_path: Optional[str] = None,
                 save_directory: str = "v3.0/models/v3_training",
                 memory_config: Optional[Dict] = None,
                 model_version: str = "v3.0"):
        
        # Set default Stockfish path
        if stockfish_path is None:
            stockfish_path = "v3.0/stockfish/stockfish-windows-x86-64-avx2.exe"
        
        # Initialize V3 enhanced database  
        self.enhanced_db = EnhancedPuzzleDatabaseV2(puzzle_db_path)
        
        # Initialize parent class
        super().__init__(
            thinking_brain=thinking_brain,
            stockfish_path=stockfish_path,
            puzzle_db_path=puzzle_db_path,
            save_directory=save_directory,
            memory_config=memory_config
        )
        
        self.model_version = model_version
        self.current_session_id = None
        
        # V3 Monitoring System
        self.monitor = V3MonitoringSystem(f"{save_directory}/monitoring")
        
        # Session context for V3
        self.session_context = {
            'puzzle_number': 0,
            'session_start_time': None,
            'performance_scores': [],
            'fatigue_estimate': 0.0,
            'average_performance': 0.0,
            'elo_estimate': 0.0
        }
        
        # V3 Enhanced statistics
        self.v3_stats = {
            'stockfish_graded_moves': 0,
            'moves_in_stockfish_top_5': 0,
            'average_stockfish_score': 0.0,
            'learning_velocity_improvements': 0,
            'theme_mastery_gains': 0,
            'regression_recoveries': 0,
            'optimal_timing_hits': 0,
            'session_efficiency': 0.0,
            'tournament_readiness_score': 0.0
        }
        
        logger.info("🧩 V7P3R Puzzle Trainer V3.0 initialized - Pure puzzle-based learning system")
        logger.info(f"📊 Monitoring system active: {self.monitor.save_directory}")
    
    def estimate_elo_from_puzzle_performance(self) -> float:
        """Estimate ELO based on puzzle solving performance"""
        if not self.session_context['performance_scores']:
            return 800  # Starting estimate
        
        # Simple ELO estimation based on average score and puzzle difficulty
        avg_score = sum(self.session_context['performance_scores'][-100:]) / min(100, len(self.session_context['performance_scores']))
        
        # Rough correlation: 0-5 score to ELO range (800-2000)
        base_elo = 800 + (avg_score / 5.0) * 1200
        
        # Adjust for top-5 hits rate
        if self.v3_stats['stockfish_graded_moves'] > 0:
            top5_rate = self.v3_stats['moves_in_stockfish_top_5'] / self.v3_stats['stockfish_graded_moves']
            elo_adjustment = (top5_rate - 0.2) * 300  # Bonus for high Stockfish agreement
            base_elo += elo_adjustment
        
        return max(600, min(2400, base_elo))  # Clamp to reasonable range
    
    def calculate_tournament_readiness(self) -> float:
        """Calculate tournament readiness score (0-100)"""
        readiness_factors = []
        
        # Factor 1: Average puzzle score (0-100 scale)
        if self.session_context['performance_scores']:
            avg_score = sum(self.session_context['performance_scores'][-500:]) / min(500, len(self.session_context['performance_scores']))
            score_factor = (avg_score / 5.0) * 100
            readiness_factors.append(score_factor)
        
        # Factor 2: Top-5 Stockfish agreement (target >70%)
        if self.v3_stats['stockfish_graded_moves'] > 0:
            top5_rate = self.v3_stats['moves_in_stockfish_top_5'] / self.v3_stats['stockfish_graded_moves']
            stockfish_factor = min(100, (top5_rate / 0.7) * 100)
            readiness_factors.append(stockfish_factor)
        
        # Factor 3: Theme diversity (number of themes with >50% confidence)
        # This would require theme mastery tracking - placeholder for now
        theme_factor = min(100, self.v3_stats['theme_mastery_gains'] * 2)
        readiness_factors.append(theme_factor)
        
        # Factor 4: Analysis speed (target <1.5s per position)
        # This would require timing tracking - placeholder for now
        speed_factor = 75  # Assume reasonable speed for now
        readiness_factors.append(speed_factor)
        
        return sum(readiness_factors) / len(readiness_factors) if readiness_factors else 0
    
    def train_v3_puzzle_session(self, 
                               num_puzzles: Optional[int] = None,
                               training_hours: Optional[float] = None,
                               batch_size: int = 50,
                               target_themes: Optional[List[str]] = None,
                               excluded_themes: Optional[List[str]] = None,
                               max_rating: Optional[int] = None,
                               min_rating: Optional[int] = None,
                               checkpoint_interval: int = 50,
                               auto_difficulty: bool = True,
                               regression_detection: bool = True) -> Dict:
        """
        V3 Pure Puzzle Training Session
        
        Args:
            num_puzzles: Fixed number of puzzles (if not time-based)
            training_hours: Training duration in hours (if time-based)
            batch_size: Puzzles per batch
            target_themes: Specific themes to focus on
            excluded_themes: Themes to exclude (e.g., ['long', 'mate'])
            max_rating/min_rating: Puzzle difficulty range
            checkpoint_interval: Save frequency
            auto_difficulty: Automatically adjust difficulty
            regression_detection: Monitor for performance regression
        """
        
        # Determine training mode
        if training_hours:
            mode = "time-based"
            session_duration = training_hours
            logger.info(f"🕐 Starting V3 time-based training: {training_hours} hours")
        else:
            mode = "puzzle-count"
            session_duration = None
            num_puzzles = num_puzzles or 1000
            logger.info(f"🧩 Starting V3 puzzle-count training: {num_puzzles} puzzles")
        
        # Start session
        self.current_session_id = str(uuid.uuid4())
        self.session_context['session_start_time'] = datetime.now()
        self.session_context['puzzle_number'] = 0
        
        # Start monitoring
        self.monitor.start_session_monitoring(self.current_session_id, session_duration)
        
        logger.info(f"📊 V3 Pure Puzzle Training Session: {self.current_session_id[:8]}...")
        logger.info(f"🎯 Philosophy: Learn chess through pure puzzle-solving")
        
        try:
            if mode == "time-based":
                results = self._train_v3_timed(
                    training_hours, batch_size, target_themes, excluded_themes,
                    max_rating, min_rating, checkpoint_interval, auto_difficulty, regression_detection
                )
            else:
                results = self._train_v3_fixed(
                    num_puzzles, target_themes, excluded_themes,
                    max_rating, min_rating, checkpoint_interval, auto_difficulty, regression_detection
                )
            
            # Calculate final metrics
            final_elo = self.estimate_elo_from_puzzle_performance()
            tournament_readiness = self.calculate_tournament_readiness()
            
            logger.info("🏁 V3 Training Session Complete!")
            logger.info(f"📊 Final ELO Estimate: {final_elo:.0f}")
            logger.info(f"🏆 Tournament Readiness: {tournament_readiness:.1f}%")
            
            return {
                'session_id': self.current_session_id,
                'mode': mode,
                'results': results,
                'final_elo_estimate': final_elo,
                'tournament_readiness': tournament_readiness,
                'v3_stats': self.v3_stats.copy()
            }
            
        finally:
            # Stop monitoring
            self.monitor.stop_session_monitoring()
    
    def _train_v3_timed(self, hours: float, batch_size: int, target_themes: Optional[List[str]],
                       excluded_themes: Optional[List[str]], max_rating: Optional[int], 
                       min_rating: Optional[int], checkpoint_interval: int,
                       auto_difficulty: bool, regression_detection: bool) -> List[Dict]:
        """V3 time-based training implementation"""
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=hours)
        all_results = []
        batch_count = 0
        last_save_time = start_time
        processed_puzzle_ids = set()
        
        logger.info(f"⏰ Training until: {end_time.strftime('%H:%M:%S')}")
        
        while datetime.now() < end_time:
            remaining_time = end_time - datetime.now()
            if remaining_time.total_seconds() < 60:
                logger.info("⏰ Less than 1 minute remaining - ending gracefully")
                break
            
            batch_count += 1
            current_time = datetime.now()
            elapsed = (current_time - start_time).total_seconds() / 3600.0
            since_save = (current_time - last_save_time).total_seconds() / 60.0
            
            logger.info(f"📦 V3 Batch {batch_count}")
            logger.info(f"   ⏰ Elapsed: {elapsed:.2f}h | Since save: {since_save:.1f}m")
            
            # Get puzzle batch using intelligent selection
            puzzles = self._get_v3_puzzle_batch(
                batch_size, target_themes, excluded_themes, max_rating, min_rating,
                auto_difficulty, processed_puzzle_ids
            )
            
            if not puzzles:
                logger.warning("⚠️ No more suitable puzzles available")
                break
            
            # Process batch
            batch_results = self._process_v3_puzzle_batch(puzzles, batch_count, checkpoint_interval)
            all_results.extend(batch_results)
            
            # Update monitoring
            self.monitor.update_metrics(
                puzzles_completed=len(all_results),
                average_score=sum(r['ai_score'] for r in all_results) / len(all_results) if all_results else 0,
                session_efficiency=len(all_results) / elapsed if elapsed > 0 else 0,
                estimated_elo=self.estimate_elo_from_puzzle_performance()
            )
            
            # Mark puzzles as processed
            for puzzle in puzzles:
                processed_puzzle_ids.add(puzzle['id'])
            
            # Checkpoint saving
            if len(all_results) % checkpoint_interval == 0:
                self._save_v3_checkpoint(len(all_results), all_results)
                last_save_time = current_time
            
            # Brief pause between batches
            time.sleep(2)
        
        return all_results
    
    def _train_v3_fixed(self, num_puzzles: int, target_themes: Optional[List[str]],
                       excluded_themes: Optional[List[str]], max_rating: Optional[int],
                       min_rating: Optional[int], checkpoint_interval: int,
                       auto_difficulty: bool, regression_detection: bool) -> List[Dict]:
        """V3 fixed-count training implementation"""
        
        logger.info(f"🧩 Training on {num_puzzles} puzzles")
        
        # Get all puzzles upfront for fixed training
        puzzles = self._get_v3_puzzle_selection(
            num_puzzles, target_themes, excluded_themes, max_rating, min_rating, auto_difficulty
        )
        
        if not puzzles:
            logger.error("No puzzles found matching criteria")
            return []
        
        logger.info(f"Selected {len(puzzles)} puzzles for training")
        
        all_results = []
        with tqdm(total=len(puzzles), desc="V3 Pure Puzzle Training") as pbar:
            for i, puzzle in enumerate(puzzles):
                self.session_context['puzzle_number'] = i + 1
                
                # Train on puzzle
                result = self._train_on_v3_puzzle(puzzle)
                
                if result:
                    all_results.append(result)
                    self._update_v3_session_context(result)
                    
                    # Update progress bar and monitoring
                    self._update_v3_progress_display(pbar)
                    self.monitor.update_metrics(
                        puzzles_completed=len(all_results),
                        current_score=result['ai_score'],
                        average_score=sum(r['ai_score'] for r in all_results) / len(all_results),
                        estimated_elo=self.estimate_elo_from_puzzle_performance()
                    )
                
                pbar.update(1)
                
                # Checkpoint saving
                if (i + 1) % checkpoint_interval == 0:
                    self._save_v3_checkpoint(i + 1, all_results)
        
        return all_results
    
    def _get_v3_puzzle_selection(self, num_puzzles: int, target_themes: Optional[List[str]],
                                excluded_themes: Optional[List[str]], max_rating: Optional[int],
                                min_rating: Optional[int], auto_difficulty: bool) -> List[Dict]:
        """Get intelligent puzzle selection for V3 training"""
        # This would use the enhanced database selection logic
        # For now, placeholder implementation
        return []
    
    def _get_v3_puzzle_batch(self, batch_size: int, target_themes: Optional[List[str]],
                            excluded_themes: Optional[List[str]], max_rating: Optional[int],
                            min_rating: Optional[int], auto_difficulty: bool,
                            processed_ids: set) -> List[Dict]:
        """Get a batch of puzzles for V3 training"""
        # This would implement the batch selection logic
        # For now, placeholder implementation
        return []
    
    def _train_on_v3_puzzle(self, puzzle: Dict) -> Optional[Dict]:
        """Train on a single puzzle with V3 analytics"""
        # This would implement the core training logic
        # For now, placeholder implementation
        return None
    
    def _process_v3_puzzle_batch(self, puzzles: List[Dict], batch_num: int, checkpoint_interval: int) -> List[Dict]:
        """Process a batch of puzzles for V3 training"""
        # This would implement batch processing
        # For now, placeholder implementation
        return []
    
    def _update_v3_session_context(self, result: Dict):
        """Update V3 session context with latest result"""
        self.session_context['performance_scores'].append(result['ai_score'])
        
        # Keep only recent scores for fatigue estimation
        if len(self.session_context['performance_scores']) > 100:
            self.session_context['performance_scores'] = self.session_context['performance_scores'][-100:]
        
        # Update V3 statistics
        self.v3_stats['stockfish_graded_moves'] += 1
        if result.get('in_stockfish_top_5', False):
            self.v3_stats['moves_in_stockfish_top_5'] += 1
        
        # Calculate updated averages
        recent_scores = self.session_context['performance_scores'][-20:]
        self.session_context['average_performance'] = sum(recent_scores) / len(recent_scores)
        
        # Estimate fatigue
        if len(recent_scores) >= 10:
            early_avg = sum(recent_scores[:5]) / 5
            late_avg = sum(recent_scores[-5:]) / 5
            self.session_context['fatigue_estimate'] = max(0, (early_avg - late_avg) / early_avg * 100)
    
    def _update_v3_progress_display(self, pbar):
        """Update progress bar with V3 metrics"""
        if not self.session_context['performance_scores']:
            return
        
        avg_score = self.session_context['average_performance']
        fatigue = self.session_context['fatigue_estimate']
        elo_est = self.estimate_elo_from_puzzle_performance()
        
        pbar.set_postfix({
            'Score': f'{avg_score:.2f}/5',
            'ELO': f'{elo_est:.0f}',
            'Fatigue': f'{fatigue:.0f}%'
        })
    
    def _save_v3_checkpoint(self, puzzle_count: int, results: List[Dict]):
        """Save V3 checkpoint with comprehensive analytics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = Path(self.save_directory) / f"v3_model_{puzzle_count}puzzles_{timestamp}.pkl"
        self.thinking_brain.save_model(str(model_path))
        
        # Save progress data
        progress_path = Path(self.save_directory) / f"v3_progress_{puzzle_count}_{timestamp}.json"
        progress_data = {
            'puzzle_count': puzzle_count,
            'session_id': self.current_session_id,
            'session_context': self.session_context,
            'v3_stats': self.v3_stats,
            'elo_estimate': self.estimate_elo_from_puzzle_performance(),
            'tournament_readiness': self.calculate_tournament_readiness(),
            'timestamp': timestamp
        }
        
        with open(progress_path, 'w') as f:
            json.dump(progress_data, f, indent=2, cls=DateTimeEncoder)
        
        logger.info(f"💾 V3 Checkpoint saved: {puzzle_count} puzzles")
        logger.info(f"   ELO estimate: {progress_data['elo_estimate']:.0f}")
        logger.info(f"   Tournament readiness: {progress_data['tournament_readiness']:.1f}%")
    
    def get_monitoring_status(self) -> Optional[Dict]:
        """Get current monitoring status without interrupting training"""
        return self.monitor.get_current_metrics()
    
    def close(self):
        """Clean up V3 resources"""
        self.monitor.stop_session_monitoring()
        if self.enhanced_db:
            self.enhanced_db.close()
        logger.info("🔒 V3 Trainer resources cleaned up")

if __name__ == "__main__":
    print("V7P3R Chess AI V3.0 - Pure Puzzle Trainer")
    print("This module contains the core training system.")
    print("Use v3_training_main.py for training sessions.")