"""
Database Migration Script - V1 to V2 Enhanced Schema

This script migrates the existing puzzle database to the enhanced V2 schema
while preserving all existing data and adding new analytics capabilities.
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import shutil

logger = logging.getLogger(__name__)


class DatabaseMigrator:
    """Migrates puzzle database from V1 to enhanced V2 schema"""
    
    def __init__(self, source_db_path: str, target_db_path: Optional[str] = None):
        self.source_db_path = source_db_path
        self.target_db_path = target_db_path or source_db_path.replace('.db', '_v2.db')
        self.backup_path = source_db_path.replace('.db', '_backup.db')
        
    def migrate(self, create_backup: bool = True) -> bool:
        """Perform full migration from V1 to V2 schema"""
        try:
            logger.info(f"Starting database migration: {self.source_db_path} -> {self.target_db_path}")
            
            # Create backup if requested
            if create_backup:
                self._create_backup()
            
            # Check if source database exists
            if not Path(self.source_db_path).exists():
                logger.error(f"Source database not found: {self.source_db_path}")
                return False
            
            # Create V2 database with enhanced schema
            self._create_v2_database()
            
            # Migrate data
            self._migrate_puzzles_data()
            self._migrate_performance_history()
            self._migrate_stockfish_data()
            self._migrate_training_sessions()
            
            # Initialize new V2 features
            self._initialize_enhanced_features()
            
            # Validate migration
            migration_valid = self._validate_migration()
            
            if migration_valid:
                logger.info("✅ Database migration completed successfully!")
                logger.info(f"Enhanced database created: {self.target_db_path}")
                if create_backup:
                    logger.info(f"Backup created: {self.backup_path}")
                return True
            else:
                logger.error("❌ Migration validation failed")
                return False
                
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_backup(self):
        """Create backup of original database"""
        logger.info(f"Creating backup: {self.backup_path}")
        shutil.copy2(self.source_db_path, self.backup_path)
    
    def _create_v2_database(self):
        """Create new V2 database with enhanced schema"""
        logger.info("Creating V2 database with enhanced schema...")
        
        # Import and initialize V2 database
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent / "src"))
        from database.enhanced_puzzle_db_v2 import EnhancedPuzzleDatabaseV2
        
        # Create V2 database (this will create all tables and indices)
        v2_db = EnhancedPuzzleDatabaseV2(self.target_db_path)
        v2_db.close()
        
        logger.info("V2 schema created successfully")
    
    def _migrate_puzzles_data(self):
        """Migrate puzzles table data"""
        logger.info("Migrating puzzles data...")
        
        source_conn = sqlite3.connect(self.source_db_path)
        target_conn = sqlite3.connect(self.target_db_path)
        
        source_conn.row_factory = sqlite3.Row
        target_conn.row_factory = sqlite3.Row
        
        # Get all puzzles from source
        source_cursor = source_conn.cursor()
        source_cursor.execute("SELECT * FROM puzzles")
        
        target_cursor = target_conn.cursor()
        
        migrated_count = 0
        for row in source_cursor.fetchall():
            # Helper function to safely get row values
            def get_row_value(row, key, default=None):
                try:
                    return row[key] if row[key] is not None else default
                except (KeyError, IndexError):
                    return default
            
            # Map V1 fields to V2 fields
            puzzle_data = {
                'id': row['id'],
                'fen': row['fen'],
                'moves': row['moves'],
                'rating': get_row_value(row, 'rating', 0),
                'themes': get_row_value(row, 'themes', ''),
                'popularity': get_row_value(row, 'popularity', 0),
                'nb_plays': get_row_value(row, 'nb_plays', 0),
                'dataset_split': get_row_value(row, 'dataset_split', 'train'),
                'difficulty_tier': get_row_value(row, 'difficulty_tier', 'medium'),
                
                # Migrate existing AI performance data
                'ai_encounter_count': get_row_value(row, 'ai_encounter_count', 0),
                'ai_solved_count': get_row_value(row, 'ai_solved_count', 0),
                'ai_best_score': get_row_value(row, 'ai_best_score', 0),
                'ai_average_score': get_row_value(row, 'ai_average_score', 0.0),
                'ai_first_solved_date': get_row_value(row, 'ai_first_solved_date'),
                'ai_last_solved_date': get_row_value(row, 'ai_last_solved_date'),
                'ai_last_encounter_date': get_row_value(row, 'ai_last_encounter_date'),
                'ai_consecutive_fails': get_row_value(row, 'ai_consecutive_fails', 0),
                'ai_regression_detected': get_row_value(row, 'ai_regression_detected', 0),
                'ai_average_solve_time': get_row_value(row, 'ai_average_solve_time', 0.0),
                'ai_fastest_solve_time': get_row_value(row, 'ai_fastest_solve_time', 0.0),
                
                # Initialize new V2 fields with defaults
                'ai_best_stockfish_score': None,
                'ai_latest_stockfish_score': None,
                'ai_stockfish_score_trend': 0.0,
                'ai_learning_velocity': 0.0,
                'ai_mastery_level': 'novice',
                'ai_stability_score': 0.0,
                'ai_solve_time_trend': 0.0,
                'ai_session_context_avg': 0.0,
                'ai_time_between_encounters_avg': 0.0,
                'ai_optimal_revisit_interval': 24.0,  # Default 24 hours
                'difficulty_appropriateness': 0.5,
                'theme_complexity_score': 0.0,
                'prerequisite_themes_mastered': 0,
                'created_date': get_row_value(row, 'created_date', datetime.now().isoformat()),
                'updated_date': datetime.now().isoformat()
            }
            
            # Insert into V2 puzzles table
            target_cursor.execute("""
                INSERT INTO puzzles_v2 (
                    id, fen, moves, rating, themes, popularity, nb_plays,
                    dataset_split, difficulty_tier, ai_encounter_count, ai_solved_count,
                    ai_best_score, ai_average_score, ai_first_solved_date, ai_last_solved_date,
                    ai_last_encounter_date, ai_consecutive_fails, ai_regression_detected,
                    ai_average_solve_time, ai_fastest_solve_time, ai_best_stockfish_score,
                    ai_latest_stockfish_score, ai_stockfish_score_trend, ai_learning_velocity,
                    ai_mastery_level, ai_stability_score, ai_solve_time_trend,
                    ai_session_context_avg, ai_time_between_encounters_avg, ai_optimal_revisit_interval,
                    difficulty_appropriateness, theme_complexity_score, prerequisite_themes_mastered,
                    created_date, updated_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                puzzle_data['id'], puzzle_data['fen'], puzzle_data['moves'], 
                puzzle_data['rating'], puzzle_data['themes'], puzzle_data['popularity'],
                puzzle_data['nb_plays'], puzzle_data['dataset_split'], puzzle_data['difficulty_tier'],
                puzzle_data['ai_encounter_count'], puzzle_data['ai_solved_count'],
                puzzle_data['ai_best_score'], puzzle_data['ai_average_score'],
                puzzle_data['ai_first_solved_date'], puzzle_data['ai_last_solved_date'],
                puzzle_data['ai_last_encounter_date'], puzzle_data['ai_consecutive_fails'],
                puzzle_data['ai_regression_detected'], puzzle_data['ai_average_solve_time'],
                puzzle_data['ai_fastest_solve_time'], puzzle_data['ai_best_stockfish_score'],
                puzzle_data['ai_latest_stockfish_score'], puzzle_data['ai_stockfish_score_trend'],
                puzzle_data['ai_learning_velocity'], puzzle_data['ai_mastery_level'],
                puzzle_data['ai_stability_score'], puzzle_data['ai_solve_time_trend'],
                puzzle_data['ai_session_context_avg'], puzzle_data['ai_time_between_encounters_avg'],
                puzzle_data['ai_optimal_revisit_interval'], puzzle_data['difficulty_appropriateness'],
                puzzle_data['theme_complexity_score'], puzzle_data['prerequisite_themes_mastered'],
                puzzle_data['created_date'], puzzle_data['updated_date']
            ))
            
            migrated_count += 1
        
        target_conn.commit()
        source_conn.close()
        target_conn.close()
        
        logger.info(f"Migrated {migrated_count} puzzles to V2 schema")
    
    def _migrate_performance_history(self):
        """Migrate AI performance history data"""
        logger.info("Migrating performance history...")
        
        source_conn = sqlite3.connect(self.source_db_path)
        target_conn = sqlite3.connect(self.target_db_path)
        
        source_conn.row_factory = sqlite3.Row
        target_conn.row_factory = sqlite3.Row
        
        # Check if source table exists
        source_cursor = source_conn.cursor()
        source_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ai_performance_history'")
        if not source_cursor.fetchone():
            logger.info("No performance history table found in source database - skipping")
            source_conn.close()
            target_conn.close()
            return
        
        # Get all performance history from source
        source_cursor.execute("SELECT * FROM ai_performance_history")
        target_cursor = target_conn.cursor()
        
        migrated_count = 0
        for row in source_cursor.fetchall():
            # Helper function to safely get row values
            def get_row_value(row, key, default=None):
                try:
                    return row[key] if row[key] is not None else default
                except (KeyError, IndexError):
                    return default
            
            # Insert into V2 performance history with enhanced fields
            target_cursor.execute("""
                INSERT INTO ai_performance_history_v2 (
                    puzzle_id, encounter_date, ai_move, expected_move, ai_score, ai_rank,
                    found_solution, solve_time, stockfish_top_moves, ai_move_quality,
                    learning_context, model_version, session_id,
                    ai_move_stockfish_score, ai_move_stockfish_rank, ai_move_stockfish_evaluation,
                    move_improvement_from_last, session_performance_context, time_since_last_encounter,
                    session_puzzle_number, fatigue_indicator, similar_puzzles_recent_performance
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row['puzzle_id'], row['encounter_date'], row['ai_move'], row['expected_move'],
                row['ai_score'], row['ai_rank'], row['found_solution'], row['solve_time'],
                row['stockfish_top_moves'], get_row_value(row, 'ai_move_quality', 'unknown'),
                get_row_value(row, 'learning_context', 'training'), get_row_value(row, 'model_version', 'v3.0'),
                get_row_value(row, 'session_id', 'migrated'),
                # New V2 fields - initialize with defaults
                None, None, None, 0.0, 0.0, 0.0, 0, 0.0, 0.0
            ))
            
            migrated_count += 1
        
        target_conn.commit()
        source_conn.close()
        target_conn.close()
        
        logger.info(f"Migrated {migrated_count} performance history records")
    
    def _migrate_stockfish_data(self):
        """Migrate Stockfish grading data"""
        logger.info("Migrating Stockfish data...")
        
        source_conn = sqlite3.connect(self.source_db_path)
        target_conn = sqlite3.connect(self.target_db_path)
        
        source_conn.row_factory = sqlite3.Row
        target_conn.row_factory = sqlite3.Row
        
        # Check if source table exists
        source_cursor = source_conn.cursor()
        source_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stockfish_gradings'")
        if not source_cursor.fetchone():
            logger.info("No Stockfish gradings table found in source database - skipping")
            source_conn.close()
            target_conn.close()
            return
        
        # Get all Stockfish data from source
        source_cursor.execute("SELECT * FROM stockfish_gradings")
        target_cursor = target_conn.cursor()
        
        migrated_count = 0
        for row in source_cursor.fetchall():
            # Insert into V2 Stockfish evaluations with enhanced fields
            target_cursor.execute("""
                INSERT INTO stockfish_evaluations_v2 (
                    fen, move, stockfish_score, stockfish_rank, evaluation_depth,
                    evaluation_time, evaluation_date, engine_version,
                    stockfish_centipawn_eval, stockfish_mate_in, position_complexity,
                    tactical_themes, move_category, alternative_quality
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row['fen'], row['move'], row['stockfish_score'], row['stockfish_rank'],
                row.get('evaluation_depth', 15), row.get('evaluation_time', 0.0),
                row.get('evaluation_date', datetime.now().isoformat()), 'stockfish_15',
                # New V2 fields - initialize with defaults
                None, None, 0.0, None, 'unknown', 0.0
            ))
            
            migrated_count += 1
        
        target_conn.commit()
        source_conn.close()
        target_conn.close()
        
        logger.info(f"Migrated {migrated_count} Stockfish evaluation records")
    
    def _migrate_training_sessions(self):
        """Migrate training sessions data"""
        logger.info("Migrating training sessions...")
        
        source_conn = sqlite3.connect(self.source_db_path)
        target_conn = sqlite3.connect(self.target_db_path)
        
        source_conn.row_factory = sqlite3.Row
        target_conn.row_factory = sqlite3.Row
        
        # Check if source table exists
        source_cursor = source_conn.cursor()
        source_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='training_sessions'")
        if not source_cursor.fetchone():
            logger.info("No training sessions table found in source database - skipping")
            source_conn.close()
            target_conn.close()
            return
        
        # Get all training sessions from source
        source_cursor.execute("SELECT * FROM training_sessions")
        target_cursor = target_conn.cursor()
        
        migrated_count = 0
        for row in source_cursor.fetchall():
            # Insert into V2 training sessions with enhanced fields
            target_cursor.execute("""
                INSERT INTO training_sessions_v2 (
                    session_id, start_time, end_time, total_puzzles, puzzles_solved,
                    average_score, model_version, training_config, performance_summary, notes,
                    session_efficiency, learning_momentum, fatigue_detected, peak_performance_time,
                    session_quality_score, previous_session_gap, model_state_before,
                    model_state_after, environmental_factors
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row['session_id'], row['start_time'], row.get('end_time'),
                row.get('total_puzzles', 0), row.get('puzzles_solved', 0),
                row.get('average_score', 0.0), row.get('model_version', 'v3.0'),
                row.get('training_config'), row.get('performance_summary'), row.get('notes'),
                # New V2 fields - initialize with defaults
                0.0, 0.0, 0, None, 0.0, 0.0, None, None, None
            ))
            
            migrated_count += 1
        
        target_conn.commit()
        source_conn.close()
        target_conn.close()
        
        logger.info(f"Migrated {migrated_count} training session records")
    
    def _initialize_enhanced_features(self):
        """Initialize enhanced V2 features for migrated data"""
        logger.info("Initializing enhanced V2 features...")
        
        conn = sqlite3.connect(self.target_db_path)
        cursor = conn.cursor()
        
        # Initialize theme mastery data based on existing performance
        cursor.execute("""
            INSERT OR IGNORE INTO theme_mastery (theme, model_version, first_encounter_date)
            SELECT DISTINCT 
                TRIM(value) as theme,
                'v3.0' as model_version,
                MIN(ai_first_solved_date) as first_encounter_date
            FROM puzzles_v2, json_each('["' || REPLACE(themes, ' ', '","') || '"]')
            WHERE themes != '' AND ai_encounter_count > 0
            GROUP BY TRIM(value)
        """)
        
        # Update theme mastery statistics
        cursor.execute("""
            UPDATE theme_mastery SET
                puzzles_encountered = (
                    SELECT COUNT(*) FROM puzzles_v2 
                    WHERE themes LIKE '%' || theme_mastery.theme || '%' 
                    AND ai_encounter_count > 0
                ),
                puzzles_solved = (
                    SELECT COUNT(*) FROM puzzles_v2 
                    WHERE themes LIKE '%' || theme_mastery.theme || '%' 
                    AND ai_solved_count > 0
                ),
                average_score = (
                    SELECT AVG(ai_average_score) FROM puzzles_v2 
                    WHERE themes LIKE '%' || theme_mastery.theme || '%' 
                    AND ai_encounter_count > 0
                ),
                last_updated = ?
        """, (datetime.now().isoformat(),))
        
        # Calculate initial confidence scores and mastery levels
        cursor.execute("""
            UPDATE theme_mastery SET
                confidence_score = CASE 
                    WHEN puzzles_encountered > 0 THEN 
                        MIN(1.0, (CAST(puzzles_solved AS REAL) / puzzles_encountered) * (average_score / 5.0))
                    ELSE 0.0
                END,
                mastery_level = CASE 
                    WHEN puzzles_encountered < 3 THEN 'novice'
                    WHEN (CAST(puzzles_solved AS REAL) / puzzles_encountered) >= 0.8 AND average_score >= 4.0 THEN 'expert'
                    WHEN (CAST(puzzles_solved AS REAL) / puzzles_encountered) >= 0.6 AND average_score >= 3.5 THEN 'proficient'
                    WHEN (CAST(puzzles_solved AS REAL) / puzzles_encountered) >= 0.4 AND average_score >= 2.5 THEN 'competent'
                    WHEN puzzles_encountered >= 5 THEN 'learning'
                    ELSE 'novice'
                END
        """)
        
        conn.commit()
        conn.close()
        
        logger.info("Enhanced features initialized successfully")
    
    def _validate_migration(self) -> bool:
        """Validate the migration was successful"""
        logger.info("Validating migration...")
        
        source_conn = sqlite3.connect(self.source_db_path)
        target_conn = sqlite3.connect(self.target_db_path)
        
        # Check puzzle count
        source_cursor = source_conn.cursor()
        target_cursor = target_conn.cursor()
        
        source_cursor.execute("SELECT COUNT(*) FROM puzzles")
        source_count = source_cursor.fetchone()[0]
        
        target_cursor.execute("SELECT COUNT(*) FROM puzzles_v2")
        target_count = target_cursor.fetchone()[0]
        
        if source_count != target_count:
            logger.error(f"Puzzle count mismatch: source={source_count}, target={target_count}")
            return False
        
        # Check that V2 features are initialized
        target_cursor.execute("SELECT COUNT(*) FROM theme_mastery")
        theme_count = target_cursor.fetchone()[0]
        
        if theme_count == 0:
            logger.warning("No theme mastery data found - this may be normal for new databases")
        
        # Check V2 tables exist
        v2_tables = ['puzzles_v2', 'ai_performance_history_v2', 'stockfish_evaluations_v2', 
                     'theme_mastery', 'learning_efficiency', 'training_sessions_v2']
        
        for table in v2_tables:
            target_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
            if not target_cursor.fetchone():
                logger.error(f"V2 table missing: {table}")
                return False
        
        source_conn.close()
        target_conn.close()
        
        logger.info(f"✅ Migration validation passed: {source_count} puzzles migrated successfully")
        return True
    
    def get_migration_summary(self) -> Dict:
        """Get summary of migration results"""
        if not Path(self.target_db_path).exists():
            return {'status': 'not_migrated', 'error': 'Target database does not exist'}
        
        conn = sqlite3.connect(self.target_db_path)
        cursor = conn.cursor()
        
        # Get counts for summary
        cursor.execute("SELECT COUNT(*) FROM puzzles_v2")
        puzzle_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM ai_performance_history_v2")
        history_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM theme_mastery")
        theme_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM stockfish_evaluations_v2")
        stockfish_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'status': 'completed',
            'target_database': self.target_db_path,
            'backup_database': self.backup_path,
            'puzzle_count': puzzle_count,
            'history_records': history_count,
            'theme_mastery_records': theme_count,
            'stockfish_evaluations': stockfish_count,
            'migration_date': datetime.now().isoformat()
        }


def main():
    """Run database migration"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate puzzle database to enhanced V2 schema")
    parser.add_argument('source_db', help='Path to source database')
    parser.add_argument('--target-db', help='Path for target V2 database (optional)')
    parser.add_argument('--no-backup', action='store_true', help='Skip creating backup')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run migration
    migrator = DatabaseMigrator(args.source_db, args.target_db)
    success = migrator.migrate(create_backup=not args.no_backup)
    
    if success:
        summary = migrator.get_migration_summary()
        print("\n" + "="*60)
        print("MIGRATION COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Source: {args.source_db}")
        print(f"Target: {summary['target_database']}")
        print(f"Backup: {summary['backup_database']}")
        print(f"Puzzles migrated: {summary['puzzle_count']}")
        print(f"History records: {summary['history_records']}")
        print(f"Theme mastery records: {summary['theme_mastery_records']}")
        print(f"Stockfish evaluations: {summary['stockfish_evaluations']}")
        print("="*60)
    else:
        print("❌ Migration failed - check logs for details")
        exit(1)


if __name__ == "__main__":
    main()
