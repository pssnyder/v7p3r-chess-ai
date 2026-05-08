#!/usr/bin/env python3
"""
BigQuery Historical Data Extraction
Extracts v7p3r_bot moves from BigQuery and converts to unified training format

Data Source:
- conformed_layer.game_data (5,069 games)
- conformed_layer.moves (1,350,163 moves, needs dedup)

Output:
- v5.0/data/raw/pgn_extractions/bigquery_records_YYYYMMDD.jsonl

Author: Pat Snyder
Created: 2026-05-06
"""

import os
import sys
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from google.cloud import bigquery
from google.oauth2 import service_account
import chess

# BigQuery Configuration
PROJECT_ID = "chess-engine-metrics-agent"
GAME_DATA_TABLE = "conformed_layer.game_data"
MOVES_TABLE = "conformed_layer.moves"

# Output configuration
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "pgn_extractions")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class BigQueryExtractor:
    """Extract v7p3r historical moves from BigQuery in unified training format"""
    
    def __init__(self, credentials_path: Optional[str] = None):
        """
        Initialize BigQuery client
        
        Args:
            credentials_path: Path to service account JSON key (optional)
                            If None, uses Application Default Credentials
        """
        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/bigquery"]
            )
            self.client = bigquery.Client(
                credentials=credentials,
                project=PROJECT_ID
            )
        else:
            # Use Application Default Credentials (gcloud auth)
            self.client = bigquery.Client(project=PROJECT_ID)
        
        self.records_extracted = 0
        self.errors = []
    
    def extract_v7p3r_moves(
        self,
        limit: Optional[int] = None,
        game_types: List[str] = ["lichess_rated", "lichess_casual", "tournament"],
        min_elo: int = 1200
    ) -> List[Dict]:
        """
        Extract v7p3r moves with game context from BigQuery
        
        Args:
            limit: Maximum number of records to extract (None = all)
            game_types: Game types to include
            min_elo: Minimum opponent ELO to include
        
        Returns:
            List of training records in unified format
        """
        print(f"\n{'='*60}")
        print("BigQuery Historical Data Extraction")
        print(f"{'='*60}\n")
        
        # Build query
        query = self._build_extraction_query(limit, game_types, min_elo)
        
        print(f"Executing BigQuery query...")
        print(f"  Game types: {', '.join(game_types)}")
        print(f"  Min opponent ELO: {min_elo}")
        if limit:
            print(f"  Limit: {limit:,} records")
        print()
        
        # Execute query
        query_job = self.client.query(query)
        results = query_job.result()
        
        total_rows = results.total_rows
        print(f"✓ Query complete: {total_rows:,} rows returned")
        print(f"\nProcessing records to unified format...\n")
        
        # Convert to unified format
        training_records = []
        for i, row in enumerate(results, 1):
            try:
                record = self._convert_to_unified_format(row)
                training_records.append(record)
                
                if i % 10000 == 0:
                    print(f"  Processed {i:,}/{total_rows:,} records...")
            
            except Exception as e:
                self.errors.append({
                    "row_index": i,
                    "error": str(e),
                    "game_id": row.get("game_id", "unknown")
                })
                if len(self.errors) <= 5:  # Show first 5 errors
                    print(f"  ⚠️  Error processing row {i}: {e}")
        
        self.records_extracted = len(training_records)
        print(f"\n✓ Extraction complete: {self.records_extracted:,} records")
        if self.errors:
            print(f"  ⚠️  {len(self.errors)} errors encountered")
        
        return training_records
    
    def _build_extraction_query(
        self,
        limit: Optional[int],
        game_types: List[str],
        min_elo: int
    ) -> str:
        """Build SQL query to extract v7p3r moves with game context
        
        NOTE: BigQuery schema has:
        - moves table: game_id, move_number, color, san, piece, is_capture, is_check,
          is_castle, white_material, black_material, material_balance, game_phase
        - game_data table: game_id, date, engine_version, event, color (v7p3r's side),
          outcome, result, opponent, opponent_elo, game_type, etc.
        - NO FEN positions, UCI moves, or evaluations in database
        - Positions must be reconstructed from move sequence
        """
        
        game_type_filter = ", ".join([f"'{gt}'" for gt in game_types])
        
        query = f"""
        SELECT
            -- Identifiers
            m.game_id,
            m.move_number,
            m.color,
            
            -- Move details (SAN only, no UCI)
            m.san,
            m.piece,
            m.is_capture,
            m.is_check,
            m.is_castle,
            m.castle_side,
            
            -- Position state (calculated, no FEN)
            m.material_balance,
            m.game_phase,
            m.white_material,
            m.black_material,
            
            -- Game context
            g.game_id as game_game_id,
            g.color as v7p3r_color,
            g.result,
            g.outcome,
            g.time_control,
            g.date,
            g.event,
            g.engine_version,
            g.opponent,
            g.opponent_elo,
            g.game_type,
            g.termination,
            g.eco,
            g.opening,
            g.move_count,
            g.url
            
        FROM `{PROJECT_ID}.{MOVES_TABLE}` m
        JOIN `{PROJECT_ID}.{GAME_DATA_TABLE}` g ON m.game_id = g.game_id
        WHERE
            -- Only v7p3r moves (where move color matches v7p3r's side)
            m.color = g.color
            -- Game type filter
            AND g.game_type IN ({game_type_filter})
            -- Minimum opponent ELO
            AND g.opponent_elo >= {min_elo}
            -- Valid moves only
            AND m.san IS NOT NULL
        ORDER BY g.date DESC, m.game_id, m.move_number
        {f'LIMIT {limit}' if limit else ''}
        """
        
        return query
    
    def _convert_to_unified_format(self, row: bigquery.Row) -> Dict:
        """
        Convert BigQuery row to unified training dataset format
        
        Args:
            row: BigQuery row object
        
        Returns:
            Training record dict matching UNIFIED_TRAINING_DATASET.md schema
        """
        # Generate unique ID
        record_id = str(uuid.uuid4())
        
        # Build position block
        position = self._extract_position_data(row)
        
        # Build engine decision block (from v7p3r's historical play)
        engine_decision = self._extract_engine_decision(row, position)
        
        # Build metadata block
        metadata = {
            "source": "pgn",
            "source_details": {
                "game_id": row["game_id"],
                "v7p3r_color": row["v7p3r_color"],
                "opponent": row["opponent"],
                "opponent_elo": row.get("opponent_elo"),
                "result": row["result"],
                "time_control": row.get("time_control", "unknown"),
                "date": row["date"].isoformat() if row["date"] else None,
                "event": row.get("event"),
                "site": row.get("site"),
                "game_type": row.get("game_type", "unknown")
            },
            "collection_date": datetime.now().isoformat(),
            "v7p3r_version": row.get("engine_version", "unknown"),
            "stockfish_version": None  # Added during Stockfish analysis stage
        }
        
        # Assemble complete record (without Stockfish analysis - added later)
        record = {
            "record_id": record_id,
            "source": "pgn",
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata,
            "position": position,
            "engine_decision": engine_decision,
            "stockfish_analysis": None,  # Placeholder for Stage 2 analysis
            "features": None  # Placeholder for Stage 3 feature extraction
        }
        
        return record
    
    def _extract_position_data(self, row: bigquery.Row) -> Dict:
        """Extract position context from BigQuery row"""
        
        # Parse FEN to get tactical flags
        board = chess.Board(row["fen_before"])
        
        # Calculate material counts
        material_white = self._calculate_material(board, chess.WHITE)
        material_black = self._calculate_material(board, chess.BLACK)
        material_balance = material_white["total_value"] - material_black["total_value"]
        
        # Determine material imbalance category
        if material_balance > 300:
            imbalance = "winning"
        elif material_balance > 100:
            imbalance = "advantage"
        elif material_balance > -100:
            imbalance = "even"
        elif material_balance > -300:
            imbalance = "disadvantage"
        else:
            imbalance = "losing"
        
        # Determine game phase from BigQuery or recalculate
        game_phase = row.get("game_phase", "middlegame")
        if not game_phase:
            phase_score = self._calculate_phase_score(board)
            if phase_score >= 20:
                game_phase = "opening"
            elif phase_score >= 10:
                game_phase = "middlegame"
            elif phase_score >= 4:
                game_phase = "endgame"
            else:
                game_phase = "late_endgame"
        
        # Check for en passant square
        en_passant_square = None
        if board.ep_square is not None:
            en_passant_square = chess.square_name(board.ep_square)
        
        position = {
            "fen": row["fen_before"],
            "move_number": row["move_number"],
            "halfmove_clock": board.halfmove_clock,
            "fullmove_number": board.fullmove_number,
            
            "game_phase": game_phase,
            "phase_score": self._calculate_phase_score(board),
            
            "material": {
                "white": material_white,
                "black": material_black,
                "balance": material_balance,
                "imbalance": imbalance
            },
            
            "tactical_state": {
                "in_check": board.is_check(),
                "white_can_castle_kingside": board.has_kingside_castling_rights(chess.WHITE),
                "white_can_castle_queenside": board.has_queenside_castling_rights(chess.WHITE),
                "black_can_castle_kingside": board.has_kingside_castling_rights(chess.BLACK),
                "black_can_castle_queenside": board.has_queenside_castling_rights(chess.BLACK),
                "en_passant_square": en_passant_square,
                "num_legal_moves": board.legal_moves.count()
            },
            
            "characteristics": {
                "is_tactical": self._is_tactical_position(board),
                "is_quiet": not self._is_tactical_position(board),
                "is_endgame": material_white["total_value"] < 1300 and material_black["total_value"] < 1300,
                "is_drawish": self._is_drawish(board),
                "pawn_structure": self._classify_pawn_structure(board)
            }
        }
        
        return position
    
    def _extract_engine_decision(self, row: bigquery.Row, position: Dict) -> Dict:
        """Extract engine decision from BigQuery row"""
        
        board = chess.Board(row["fen_before"])
        move = chess.Move.from_uci(row["move_uci"])
        
        engine_decision = {
            "move_uci": row["move_uci"],
            "move_san": row["move_san"],
            
            "evaluation": {
                "total_cp": row.get("v7p3r_eval_cp"),  # May be None
                "material_cp": None,  # Not stored in BigQuery
                "pst_cp": None,       # Not stored in BigQuery
                "strategic_cp": None, # Not stored in BigQuery
                "is_endgame_mode": position["characteristics"]["is_endgame"],
                "perspective": row["color"]
            },
            
            "search": {
                "depth_reached": None,    # Not stored in BigQuery
                "nodes_searched": None,   # Not stored in BigQuery
                "time_ms": None,          # Not stored in BigQuery
                "nps": None,
                "selective_depth": None,
                "pv_line": [],
                "tt_hits": None,
                "tt_hit_rate": None,
                "cache_hits": None,
                "killer_hits": None,
                "null_move_cutoffs": None
            },
            
            "move_type": {
                "is_capture": row.get("is_capture", False),
                "is_check": row.get("is_check", False),
                "is_castling": row.get("is_castle", False),
                "is_promotion": board.piece_at(move.from_square).piece_type == chess.PAWN and chess.square_rank(move.to_square) in [0, 7],
                "is_en_passant": board.is_en_passant(move),
                "piece_moved": chess.piece_name(board.piece_at(move.from_square).piece_type) if board.piece_at(move.from_square) else None,
                "piece_captured": chess.piece_name(board.piece_at(move.to_square).piece_type) if board.piece_at(move.to_square) else None
            }
        }
        
        return engine_decision
    
    def _calculate_material(self, board: chess.Board, color: chess.Color) -> Dict:
        """Calculate material for one side"""
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900
        }
        
        material = {
            "pawns": len(board.pieces(chess.PAWN, color)),
            "knights": len(board.pieces(chess.KNIGHT, color)),
            "bishops": len(board.pieces(chess.BISHOP, color)),
            "rooks": len(board.pieces(chess.ROOK, color)),
            "queens": len(board.pieces(chess.QUEEN, color)),
            "total_value": 0
        }
        
        for piece_type, value in piece_values.items():
            material["total_value"] += len(board.pieces(piece_type, color)) * value
        
        return material
    
    def _calculate_phase_score(self, board: chess.Board) -> int:
        """Calculate game phase score (0-24)"""
        phase_score = 0
        
        # Pawns contribute 0
        # Minors contribute 1
        phase_score += len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.KNIGHT, chess.BLACK))
        phase_score += len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.BLACK))
        
        # Rooks contribute 2
        phase_score += 2 * (len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK)))
        
        # Queens contribute 4
        phase_score += 4 * (len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK)))
        
        return phase_score
    
    def _is_tactical_position(self, board: chess.Board) -> bool:
        """Check if position has tactical complications"""
        # Check for available captures or checks
        for move in board.legal_moves:
            if board.is_capture(move) or board.gives_check(move):
                return True
        return False
    
    def _is_drawish(self, board: chess.Board) -> bool:
        """Check if position is drawish (opposite color bishops, etc.)"""
        # Simplified check - just look for very low material
        white_material = sum(len(board.pieces(pt, chess.WHITE)) for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
        black_material = sum(len(board.pieces(pt, chess.BLACK)) for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
        
        return white_material <= 2 and black_material <= 2
    
    def _classify_pawn_structure(self, board: chess.Board) -> str:
        """Classify pawn structure (simplified)"""
        white_pawns = len(board.pieces(chess.PAWN, chess.WHITE))
        black_pawns = len(board.pieces(chess.PAWN, chess.BLACK))
        
        if white_pawns + black_pawns >= 12:
            return "closed"
        elif white_pawns + black_pawns >= 8:
            return "semi_open"
        else:
            return "open"
    
    def save_to_jsonl(self, records: List[Dict], filename: Optional[str] = None) -> str:
        """
        Save records to JSONL file
        
        Args:
            records: List of training records
            filename: Output filename (auto-generated if None)
        
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bigquery_records_{timestamp}.jsonl"
        
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        print(f"\nSaving {len(records):,} records to {output_path}...")
        
        with open(output_path, 'w') as f:
            for record in records:
                f.write(json.dumps(record) + '\n')
        
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✓ Saved: {output_path}")
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  Records: {len(records):,}")
        
        return output_path


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract v7p3r historical data from BigQuery")
    parser.add_argument("--credentials", type=str, help="Path to service account JSON key")
    parser.add_argument("--limit", type=int, help="Max records to extract (default: all)")
    parser.add_argument("--min-elo", type=int, default=1200, help="Minimum opponent ELO (default: 1200)")
    parser.add_argument("--game-types", nargs="+", default=["lichess_rated", "lichess_casual", "tournament"],
                       help="Game types to include (default: lichess_rated lichess_casual tournament)")
    parser.add_argument("--output", type=str, help="Output filename (default: auto-generated)")
    
    args = parser.parse_args()
    
    try:
        # Initialize extractor
        extractor = BigQueryExtractor(credentials_path=args.credentials)
        
        # Extract records
        records = extractor.extract_v7p3r_moves(
            limit=args.limit,
            game_types=args.game_types,
            min_elo=args.min_elo
        )
        
        # Save to file
        output_path = extractor.save_to_jsonl(records, filename=args.output)
        
        # Summary
        print(f"\n{'='*60}")
        print("Extraction Summary")
        print(f"{'='*60}")
        print(f"  Total records: {len(records):,}")
        print(f"  Errors: {len(extractor.errors)}")
        print(f"  Output: {output_path}")
        print(f"\n✓ BigQuery extraction complete!")
        print(f"\nNext steps:")
        print(f"  1. Run Stockfish analysis: python analyze_with_stockfish.py {output_path}")
        print(f"  2. Extract features: python extract_features.py")
        print(f"  3. Merge datasets: python merge_datasets.py")
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
