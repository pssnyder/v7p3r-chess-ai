"""
V7P3R AI v5.0 - Training Data Pipeline Orchestrator
====================================================
Orchestrates the complete training dataset creation pipeline.

Pipeline Stages:
1. Extract positions from V7P3R PGN games
2. Calculate heuristic features
3. Grade moves with Stockfish
4. Output final training dataset

Usage:
    # Full pipeline from PGN directory
    python scripts/run_training_pipeline.py --pgn-dir "path/to/pgns" --output-dir data/training/v1

    # Quick test run (100 games, minimal features)
    python scripts/run_training_pipeline.py --pgn-dir "path/to/pgns" --output-dir data/test --max-games 100 --feature-set minimal --stockfish-depth 15

    # Production run (all games, full features, depth 20)
    python scripts/run_training_pipeline.py --pgn-dir "path/to/pgns" --output-dir data/training/production --feature-set full --stockfish-depth 20
"""

import argparse
import logging
from pathlib import Path
import time
from datetime import datetime
import json

# Import our pipeline components
from extract_v7p3r_pgns import PGNPositionExtractor
from calculate_features import FeatureCalculator, FeatureConfig
from grade_with_stockfish import StockfishGrader


class TrainingPipeline:
    """Orchestrate the complete training dataset creation pipeline."""
    
    def __init__(self, output_dir: Path, stockfish_path: str = "stockfish"):
        """
        Initialize pipeline.
        
        Args:
            output_dir: Directory for all pipeline outputs
            stockfish_path: Path to Stockfish executable
        """
        self.output_dir = output_dir
        self.stockfish_path = stockfish_path
        
        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "stage1_raw").mkdir(exist_ok=True)
        (self.output_dir / "stage2_features").mkdir(exist_ok=True)
        (self.output_dir / "stage3_graded").mkdir(exist_ok=True)
        
        self.pipeline_start_time = None
        self.stats = {
            "stage1_positions": 0,
            "stage1_games": 0,
            "stage1_time": 0,
            "stage2_positions": 0,
            "stage2_time": 0,
            "stage3_positions": 0,
            "stage3_time": 0,
            "total_time": 0,
        }
        
        logging.info(f"Pipeline initialized - output: {output_dir}")
    
    def run_full_pipeline(
        self,
        pgn_dir: Path,
        max_games: int = None,
        feature_set: str = "standard",
        stockfish_depth: int = 20,
        stockfish_time_limit: float = 10.0,
    ) -> Path:
        """
        Run complete pipeline from PGN extraction to final training dataset.
        
        Args:
            pgn_dir: Directory containing V7P3R PGN files
            max_games: Maximum games to process (None = all)
            feature_set: Feature preset ("minimal", "standard", "full")
            stockfish_depth: Stockfish analysis depth
            stockfish_time_limit: Max time per position for Stockfish
        
        Returns:
            Path to final training dataset JSONL file
        """
        self.pipeline_start_time = time.time()
        
        logging.info("=" * 80)
        logging.info("TRAINING DATA PIPELINE - V7P3R AI v5.0")
        logging.info("=" * 80)
        logging.info(f"PGN Source: {pgn_dir}")
        logging.info(f"Max Games: {max_games if max_games else 'ALL'}")
        logging.info(f"Feature Set: {feature_set}")
        logging.info(f"Stockfish Depth: {stockfish_depth}")
        logging.info("=" * 80)
        
        # Stage 1: Extract positions from PGNs
        stage1_output = self._run_stage1_extraction(pgn_dir, max_games)
        
        # Stage 2: Calculate features
        stage2_output = self._run_stage2_features(stage1_output, feature_set)
        
        # Stage 3: Grade with Stockfish
        stage3_output = self._run_stage3_grading(stage2_output, stockfish_depth, stockfish_time_limit)
        
        # Pipeline complete
        self._finalize_pipeline(stage3_output)
        
        return stage3_output
    
    def _run_stage1_extraction(self, pgn_dir: Path, max_games: int) -> Path:
        """Stage 1: Extract positions from PGN files."""
        logging.info("\n" + "=" * 80)
        logging.info("STAGE 1: PGN POSITION EXTRACTION")
        logging.info("=" * 80)
        
        stage_start = time.time()
        output_file = self.output_dir / "stage1_raw" / "positions_raw.jsonl"
        
        extractor = PGNPositionExtractor(
            output_path=output_file,
            player_name="v7p3r_bot"
        )
        
        extractor.extract_from_directory(
            pgn_dir=pgn_dir,
            max_games=max_games,
            recursive=True
        )
        
        self.stats['stage1_positions'] = extractor.positions_extracted
        self.stats['stage1_games'] = extractor.games_processed
        self.stats['stage1_time'] = time.time() - stage_start
        
        logging.info(f"Stage 1 complete: {self.stats['stage1_positions']} positions from "
                    f"{self.stats['stage1_games']} games in {self.stats['stage1_time']:.1f}s")
        
        return output_file
    
    def _run_stage2_features(self, input_file: Path, feature_set: str) -> Path:
        """Stage 2: Calculate heuristic features."""
        logging.info("\n" + "=" * 80)
        logging.info("STAGE 2: FEATURE CALCULATION")
        logging.info("=" * 80)
        
        stage_start = time.time()
        output_file = self.output_dir / "stage2_features" / "positions_with_features.jsonl"
        
        config = FeatureConfig.from_preset(feature_set)
        calculator = FeatureCalculator(config)
        
        calculator.process_file(input_file, output_file)
        
        self.stats['stage2_positions'] = calculator.positions_processed
        self.stats['stage2_time'] = time.time() - stage_start
        
        logging.info(f"Stage 2 complete: {self.stats['stage2_positions']} positions with features "
                    f"in {self.stats['stage2_time']:.1f}s")
        
        return output_file
    
    def _run_stage3_grading(self, input_file: Path, depth: int, time_limit: float) -> Path:
        """Stage 3: Grade moves with Stockfish."""
        logging.info("\n" + "=" * 80)
        logging.info("STAGE 3: STOCKFISH MOVE GRADING")
        logging.info("=" * 80)
        
        stage_start = time.time()
        
        # Create timestamped output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / "stage3_graded" / f"training_dataset_{timestamp}.jsonl"
        
        grader = StockfishGrader(
            stockfish_path=self.stockfish_path,
            depth=depth,
            multipv=5,
            time_limit=time_limit
        )
        
        grader.process_file(input_file, output_file)
        
        self.stats['stage3_positions'] = grader.positions_graded
        self.stats['stage3_time'] = time.time() - stage_start
        
        logging.info(f"Stage 3 complete: {self.stats['stage3_positions']} positions graded "
                    f"in {self.stats['stage3_time']:.1f}s")
        
        return output_file
    
    def _finalize_pipeline(self, final_output: Path) -> None:
        """Finalize pipeline and save statistics."""
        self.stats['total_time'] = time.time() - self.pipeline_start_time
        
        # Save pipeline statistics
        stats_file = self.output_dir / "pipeline_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logging.info("\n" + "=" * 80)
        logging.info("PIPELINE COMPLETE!")
        logging.info("=" * 80)
        logging.info(f"Total positions: {self.stats['stage3_positions']}")
        logging.info(f"Total time: {self.stats['total_time']:.1f}s ({self.stats['total_time']/60:.1f} min)")
        logging.info(f"Average rate: {self.stats['stage3_positions']/self.stats['total_time']:.2f} pos/sec")
        logging.info(f"\nFinal dataset: {final_output}")
        logging.info(f"File size: {final_output.stat().st_size / (1024*1024):.2f} MB")
        logging.info(f"\nPipeline stats: {stats_file}")
        logging.info("=" * 80)


def main():
    """Main entry point for pipeline orchestration."""
    parser = argparse.ArgumentParser(
        description="Run complete V7P3R training data pipeline"
    )
    parser.add_argument(
        "--pgn-dir",
        type=Path,
        required=True,
        help="Directory containing V7P3R PGN files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for all pipeline outputs"
    )
    parser.add_argument(
        "--max-games",
        type=int,
        help="Maximum number of games to process (default: all)"
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        choices=["minimal", "standard", "full"],
        default="standard",
        help="Feature set preset (default: standard)"
    )
    parser.add_argument(
        "--stockfish-path",
        type=str,
        default="stockfish",
        help="Path to Stockfish executable (default: stockfish)"
    )
    parser.add_argument(
        "--stockfish-depth",
        type=int,
        default=20,
        help="Stockfish analysis depth (default: 20)"
    )
    parser.add_argument(
        "--stockfish-time-limit",
        type=float,
        default=10.0,
        help="Max time per position for Stockfish (default: 10.0s)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = args.output_dir / "pipeline.log"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Create pipeline
    pipeline = TrainingPipeline(
        output_dir=args.output_dir,
        stockfish_path=args.stockfish_path
    )
    
    # Run full pipeline
    final_dataset = pipeline.run_full_pipeline(
        pgn_dir=args.pgn_dir,
        max_games=args.max_games,
        feature_set=args.feature_set,
        stockfish_depth=args.stockfish_depth,
        stockfish_time_limit=args.stockfish_time_limit,
    )
    
    logging.info(f"\n✅ Training dataset ready: {final_dataset}")


if __name__ == "__main__":
    main()
