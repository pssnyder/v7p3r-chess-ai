"""
V7P3R AI v5.0 - Feature Calculator
===================================
Calculates heuristic features for extracted positions.

Purpose:
- Read JSONL positions (from PGN extractor or other sources)
- Calculate binary/categorical heuristic observations
- Add 'features' block to each record
- Output enhanced JSONL

Feature Categories (from V7P3R_FEATURE_SET_DEFINITION.md):
- Core Position (F001-F005): REQUIRED
- King Safety (F010-F013): Optional
- Pawn Structure (F020-F023): Optional (expensive)
- Piece Activity (F030-F033): Optional
- Tactical (F040-F042): Optional (expensive)
- Move Context (F050-F053): Required if move available
- Source-Specific (F060-F063): Depends on source

Usage:
    python scripts/calculate_features.py --input positions.jsonl --output positions_with_features.jsonl
    python scripts/calculate_features.py --input data.jsonl --output enhanced.jsonl --feature-set minimal
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

import chess


@dataclass
class FeatureConfig:
    """Configuration for which features to calculate."""
    # Core features (always calculated)
    core_position: bool = True
    
    # Optional feature groups
    king_safety: bool = True
    pawn_structure: bool = True
    enhanced_pawn_structure: bool = True  # NEW: F024-F029
    piece_activity: bool = True
    tactical: bool = True  # NOW INCLUDES F043-F049
    rook_placement: bool = True  # NEW: F060-F064
    knight_outposts: bool = True  # NEW: F070-F071
    center_control: bool = True  # NEW: F080-F082
    development: bool = True  # NEW: F090-F091
    move_context: bool = True
    multi_move_context: bool = True  # NEW: F100-F114 (CRITICAL)
    
    @classmethod
    def from_preset(cls, preset: str) -> 'FeatureConfig':
        """Create config from preset name."""
        if preset == "minimal":
            return cls(
                king_safety=False,
                pawn_structure=False,
                enhanced_pawn_structure=False,
                piece_activity=False,
                tactical=False,
                rook_placement=False,
                knight_outposts=False,
                center_control=False,
                development=False,
                move_context=True,
                multi_move_context=False,
            )
        elif preset == "standard":
            return cls(
                king_safety=True,
                pawn_structure=True,
                enhanced_pawn_structure=True,
                piece_activity=True,
                tactical=True,
                rook_placement=True,
                knight_outposts=True,
                center_control=True,
                development=True,
                move_context=True,
                multi_move_context=True,  # ENABLED for v5.1
            )
        elif preset == "full":
            return cls(
                king_safety=True,
                pawn_structure=True,
                enhanced_pawn_structure=True,
                piece_activity=True,
                tactical=True,
                rook_placement=True,
                knight_outposts=True,
                center_control=True,
                development=True,
                move_context=True,
                multi_move_context=True,
            )
        else:
            return cls()


class FeatureCalculator:
    """Calculate heuristic features for chess positions."""
    
    def __init__(self, config: FeatureConfig):
        """
        Initialize feature calculator.
        
        Args:
            config: Configuration specifying which features to calculate
        """
        self.config = config
        self.positions_processed = 0
        
        logging.info(f"Feature Calculator initialized with config: {config}")
    
    def process_file(self, input_path: Path, output_path: Path) -> None:
        """
        Process JSONL file and add features to each record.
        
        Args:
            input_path: Input JSONL file (positions without features)
            output_path: Output JSONL file (positions with features)
        """
        logging.info(f"Processing: {input_path}")
        logging.info(f"Output: {output_path}")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, start=1):
                try:
                    record = json.loads(line)
                    
                    # Calculate features
                    features = self.calculate_features(record)
                    
                    # Add features block to record
                    record['features'] = features
                    
                    # Write enhanced record
                    outfile.write(json.dumps(record) + '\n')
                    
                    self.positions_processed += 1
                    
                    if self.positions_processed % 1000 == 0:
                        logging.info(f"Processed {self.positions_processed} positions")
                
                except Exception as e:
                    logging.error(f"Error processing line {line_num}: {e}")
                    continue
        
        logging.info(f"Complete! Processed {self.positions_processed} positions")
    
    def calculate_features(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate all configured features for a position.
        
        Args:
            record: Position record from JSONL
        
        Returns:
            Features dictionary
        """
        # Parse position
        fen = record['position']['fen']
        board = chess.Board(fen)
        
        # Get move if available
        move_uci = record.get('engine_decision', {}).get('move_uci')
        move = chess.Move.from_uci(move_uci) if move_uci else None
        
        # Get stockfish analysis if available
        stockfish_analysis = record.get('stockfish_analysis', {})
        
        features = {}
        
        # CORE POSITION FEATURES (F001-F005) - Always calculated
        features.update(self._calc_core_position(board))
        
        # KING SAFETY FEATURES (F010-F013)
        if self.config.king_safety:
            features.update(self._calc_king_safety(board))
        
        # PAWN STRUCTURE FEATURES (F020-F023)
        if self.config.pawn_structure:
            features.update(self._calc_pawn_structure(board))
        
        # ENHANCED PAWN STRUCTURE FEATURES (F024-F029)
        if self.config.enhanced_pawn_structure:
            features.update(self._calc_enhanced_pawn_structure(board))
        
        # PIECE ACTIVITY FEATURES (F030-F033)
        if self.config.piece_activity:
            features.update(self._calc_piece_activity(board))
        
        # TACTICAL FEATURES (F040-F049) - NOW EXPANDED
        if self.config.tactical:
            features.update(self._calc_tactical(board))
        
        # MOVE CONTEXT FEATURES (F050-F053)
        if self.config.move_context and move:
            features.update(self._calc_move_context(board, move))
        
        # ROOK PLACEMENT FEATURES (F060-F064)
        if self.config.rook_placement:
            features.update(self._calc_rook_placement(board))
        
        # KNIGHT OUTPOST FEATURES (F070-F071)
        if self.config.knight_outposts:
            features.update(self._calc_knight_outposts(board))
        
        # CENTER CONTROL FEATURES (F080-F082)
        if self.config.center_control:
            features.update(self._calc_center_control(board))
        
        # DEVELOPMENT FEATURES (F090-F091)
        if self.config.development:
            features.update(self._calc_development(board))
        
        # MULTI-MOVE CONTEXT FEATURES (F100-F114) - CRITICAL FOR GRADE SPECTRUM
        if self.config.multi_move_context and stockfish_analysis:
            features.update(self._calc_multi_move_context(stockfish_analysis, move, board))
        
        return features
    
    def calculate_features_from_fen(self, fen: str, move_uci: str = None) -> Dict[str, Any]:
        """
        Calculate features from just a FEN string (for temporal calculator).
        
        Args:
            fen: Position FEN string
            move_uci: Ignored (kept for backwards compatibility)
            
        Returns:
            Features dictionary
        """
        # Create minimal record structure WITHOUT move
        # The FEN represents the position AFTER any move, so we don't analyze a move
        record = {
            'position': {'fen': fen},
            'engine_decision': {},  # No move to analyze
            'stockfish_analysis': {}
        }
        return self.calculate_features(record)
    
    # ========================================================================
    # CORE POSITION FEATURES (F001-F005) - REQUIRED
    # ========================================================================
    
    def _calc_core_position(self, board: chess.Board) -> Dict[str, Any]:
        """Calculate core position features (F001-F005)."""
        material_count = self._count_total_material(board)
        material_balance = self._calc_material_balance_cp(board)
        
        return {
            "F001_position_fen": board.fen(),
            "F002_game_phase": self._calc_game_phase(material_count),
            "F003_material_balance_cp": material_balance,
            "F004_material_advantage_category": self._categorize_material_advantage(material_balance),
            "F005_total_piece_count": self._count_total_pieces(board),
        }
    
    def _count_total_material(self, board: chess.Board) -> int:
        """Count total material (pawn units)."""
        piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
        }
        total = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                total += piece_values.get(piece.piece_type, 0)
        return total
    
    def _count_total_pieces(self, board: chess.Board) -> int:
        """Count total pieces on board."""
        return len(board.piece_map())
    
    def _calc_material_balance_cp(self, board: chess.Board) -> int:
        """Calculate material balance in centipawns."""
        piece_values = {
            chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300,
            chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0,
        }
        balance = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values.get(piece.piece_type, 0)
                balance += value if piece.color == chess.WHITE else -value
        return balance
    
    def _calc_game_phase(self, material_count: int) -> str:
        """Determine game phase from material count."""
        if material_count > 28:
            return "opening"
        elif material_count >= 14:
            return "middlegame"
        else:
            return "endgame"
    
    def _categorize_material_advantage(self, balance_cp: int) -> str:
        """Categorize material advantage."""
        if balance_cp > 300:
            return "white_winning"
        elif balance_cp > 100:
            return "white_advantage"
        elif balance_cp > -100:
            return "equal"
        elif balance_cp > -300:
            return "black_advantage"
        else:
            return "black_winning"
    
    # ========================================================================
    # KING SAFETY FEATURES (F010-F013)
    # ========================================================================
    
    def _calc_king_safety(self, board: chess.Board) -> Dict[str, Any]:
        """Calculate king safety features (F010-F013)."""
        white_king_sq = board.king(chess.WHITE)
        black_king_sq = board.king(chess.BLACK)
        
        return {
            "F010_white_king_castled": self._has_castled(board, chess.WHITE),
            "F010_black_king_castled": self._has_castled(board, chess.BLACK),
            "F011_white_king_has_pawn_shield": self._has_pawn_shield(board, white_king_sq, chess.WHITE),
            "F011_black_king_has_pawn_shield": self._has_pawn_shield(board, black_king_sq, chess.BLACK),
            "F012_white_king_under_attack": self._is_king_attacked(board, white_king_sq, chess.BLACK),
            "F012_black_king_under_attack": self._is_king_attacked(board, black_king_sq, chess.WHITE),
        }
    
    def _has_castled(self, board: chess.Board, color: chess.Color) -> bool:
        """Check if king has castled (heuristic: king on g/c file and no castling rights)."""
        king_sq = board.king(color)
        if king_sq is None:
            return False
        
        file = chess.square_file(king_sq)
        rank = chess.square_rank(king_sq)
        
        # King on back rank and on c or g file suggests castled
        expected_rank = 0 if color == chess.WHITE else 7
        if rank == expected_rank and (file == 2 or file == 6):
            return True
        return False
    
    def _has_pawn_shield(self, board: chess.Board, king_sq: Optional[int], color: chess.Color) -> bool:
        """Check if king has pawn shield in front."""
        if king_sq is None:
            return False
        
        file = chess.square_file(king_sq)
        rank = chess.square_rank(king_sq)
        
        # Check squares in front of king
        direction = 1 if color == chess.WHITE else -1
        shield_rank = rank + direction
        
        if not (0 <= shield_rank <= 7):
            return False
        
        # Check 3 files (left, center, right)
        shield_count = 0
        for f in [file - 1, file, file + 1]:
            if 0 <= f <= 7:
                sq = chess.square(f, shield_rank)
                piece = board.piece_at(sq)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    shield_count += 1
        
        return shield_count >= 2
    
    def _is_king_attacked(self, board: chess.Board, king_sq: Optional[int], attacker_color: chess.Color) -> bool:
        """Check if king is attacked by opponent pieces."""
        if king_sq is None:
            return False
        return board.is_attacked_by(attacker_color, king_sq)
    
    # ========================================================================
    # PAWN STRUCTURE FEATURES (F020-F023)
    # ========================================================================
    
    def _calc_pawn_structure(self, board: chess.Board) -> Dict[str, Any]:
        """Calculate pawn structure features (F020-F023)."""
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)
        
        return {
            "F020_white_has_passed_pawns": self._has_passed_pawns(board, white_pawns, chess.WHITE),
            "F020_black_has_passed_pawns": self._has_passed_pawns(board, black_pawns, chess.BLACK),
            "F021_white_passed_pawn_count": self._count_passed_pawns(board, white_pawns, chess.WHITE),
            "F021_black_passed_pawn_count": self._count_passed_pawns(board, black_pawns, chess.BLACK),
            "F022_white_has_doubled_pawns": self._has_doubled_pawns(white_pawns),
            "F022_black_has_doubled_pawns": self._has_doubled_pawns(black_pawns),
            "F023_white_has_isolated_pawns": self._has_isolated_pawns(white_pawns),
            "F023_black_has_isolated_pawns": self._has_isolated_pawns(black_pawns),
        }
    
    def _has_passed_pawns(self, board: chess.Board, pawns: chess.SquareSet, color: chess.Color) -> bool:
        """Check if any passed pawns exist."""
        return self._count_passed_pawns(board, pawns, color) > 0
    
    def _count_passed_pawns(self, board: chess.Board, pawns: chess.SquareSet, color: chess.Color) -> int:
        """Count passed pawns."""
        count = 0
        opponent_pawns = board.pieces(chess.PAWN, not color)
        
        for sq in pawns:
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            
            # Check if any opponent pawns block this pawn
            is_passed = True
            for opp_sq in opponent_pawns:
                opp_file = chess.square_file(opp_sq)
                opp_rank = chess.square_rank(opp_sq)
                
                # Check if opponent pawn is in front and on same or adjacent file
                if abs(file - opp_file) <= 1:
                    if color == chess.WHITE and opp_rank > rank:
                        is_passed = False
                        break
                    elif color == chess.BLACK and opp_rank < rank:
                        is_passed = False
                        break
            
            if is_passed:
                count += 1
        
        return count
    
    def _has_doubled_pawns(self, pawns: chess.SquareSet) -> bool:
        """Check if any doubled pawns exist."""
        files = [chess.square_file(sq) for sq in pawns]
        return len(files) != len(set(files))
    
    def _has_isolated_pawns(self, pawns: chess.SquareSet) -> bool:
        """Check if any isolated pawns exist."""
        pawn_files = {chess.square_file(sq) for sq in pawns}
        
        for sq in pawns:
            file = chess.square_file(sq)
            # Check if no friendly pawns on adjacent files
            if (file - 1) not in pawn_files and (file + 1) not in pawn_files:
                return True
        
        return False
    
    # ========================================================================
    # PIECE ACTIVITY FEATURES (F030-F033)
    # ========================================================================
    
    def _calc_piece_activity(self, board: chess.Board) -> Dict[str, Any]:
        """Calculate piece activity features (F030-F033)."""
        return {
            "F030_white_piece_mobility": self._calc_mobility(board, chess.WHITE),
            "F030_black_piece_mobility": self._calc_mobility(board, chess.BLACK),
            "F031_white_pieces_on_strong_squares": self._count_strong_squares(board, chess.WHITE),
            "F031_black_pieces_on_strong_squares": self._count_strong_squares(board, chess.BLACK),
            "F032_white_has_bishop_pair": self._has_bishop_pair(board, chess.WHITE),
            "F032_black_has_bishop_pair": self._has_bishop_pair(board, chess.BLACK),
        }
    
    def _calc_mobility(self, board: chess.Board, color: chess.Color) -> int:
        """Count pseudo-legal moves (mobility)."""
        original_turn = board.turn
        board.turn = color
        mobility = sum(1 for _ in board.legal_moves)
        board.turn = original_turn
        return mobility
    
    def _count_strong_squares(self, board: chess.Board, color: chess.Color) -> int:
        """Count pieces on central squares (d4, d5, e4, e5)."""
        central_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        count = 0
        for sq in central_squares:
            piece = board.piece_at(sq)
            if piece and piece.color == color and piece.piece_type != chess.PAWN:
                count += 1
        return count
    
    def _has_bishop_pair(self, board: chess.Board, color: chess.Color) -> bool:
        """Check if player has both bishops."""
        bishops = board.pieces(chess.BISHOP, color)
        return len(bishops) >= 2
    
    # ========================================================================
    # TACTICAL FEATURES (F040-F042)
    # ========================================================================
    
    def _calc_tactical(self, board: chess.Board) -> Dict[str, Any]:
        """Calculate tactical features (F040-F049) - EXPANDED."""
        return {
            # Original features
            "F040_white_has_hanging_pieces": self._has_hanging_pieces(board, chess.WHITE),
            "F040_black_has_hanging_pieces": self._has_hanging_pieces(board, chess.BLACK),
            "F041_white_pieces_under_attack": self._count_attacked_pieces(board, chess.WHITE),
            "F041_black_pieces_under_attack": self._count_attacked_pieces(board, chess.BLACK),
            
            # NEW: En prise value (F043)
            "F043_white_pieces_en_prise_value": self._calc_en_prise_value(board, chess.WHITE),
            "F043_black_pieces_en_prise_value": self._calc_en_prise_value(board, chess.BLACK),
            
            # NEW: Fork threats (F044)
            "F044_white_has_fork_threat": self._has_fork_threat(board, chess.WHITE),
            "F044_black_has_fork_threat": self._has_fork_threat(board, chess.BLACK),
            
            # NEW: Pins (F045)
            "F045_white_has_pin": self._has_pin(board, chess.WHITE),
            "F045_black_has_pin": self._has_pin(board, chess.BLACK),
            
            # NEW: Skewers (F046)
            "F046_white_has_skewer": self._has_skewer(board, chess.WHITE),
            "F046_black_has_skewer": self._has_skewer(board, chess.BLACK),
            
            # NEW: Discovered attacks (F047)
            "F047_white_has_discovered_attack": self._has_discovered_attack(board, chess.WHITE),
            "F047_black_has_discovered_attack": self._has_discovered_attack(board, chess.BLACK),
            
            # NEW: Trapped pieces (F048)
            "F048_white_trapped_piece_count": self._count_trapped_pieces(board, chess.WHITE),
            "F048_black_trapped_piece_count": self._count_trapped_pieces(board, chess.BLACK),
            
            # NEW: Back rank threats (F049)
            "F049_white_back_rank_threat": self._has_back_rank_threat(board, chess.WHITE),
            "F049_black_back_rank_threat": self._has_back_rank_threat(board, chess.BLACK),
        }
    
    def _calc_en_prise_value(self, board: chess.Board, color: chess.Color) -> int:
        """Calculate total value of hanging pieces in centipawns."""
        piece_values = {
            chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300,
            chess.ROOK: 500, chess.QUEEN: 900
        }
        
        total_value = 0
        opponent = not color
        
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == color and piece.piece_type != chess.KING:
                if board.is_attacked_by(opponent, sq):
                    if not board.is_attacked_by(color, sq):
                        total_value += piece_values.get(piece.piece_type, 0)
        
        return total_value
    
    def _has_fork_threat(self, board: chess.Board, color: chess.Color) -> bool:
        """Detect if color has a fork threat (one piece attacking 2+ valuable pieces)."""
        opponent = not color
        
        # Check knights (most common fork piece)
        for sq in board.pieces(chess.KNIGHT, color):
            attacks = board.attacks(sq)
            valuable_targets = 0
            for target_sq in attacks:
                target = board.piece_at(target_sq)
                if target and target.color == opponent and target.piece_type in [chess.ROOK, chess.QUEEN, chess.KING]:
                    valuable_targets += 1
            if valuable_targets >= 2:
                return True
        
        return False
    
    def _has_pin(self, board: chess.Board, color: chess.Color) -> bool:
        """Detect if color has pinned an opponent piece."""
        opponent = not color
        opponent_king_sq = board.king(opponent)
        
        if opponent_king_sq is None:
            return False
        
        # Check bishops and queens (diagonal pins)
        for piece_type in [chess.BISHOP, chess.QUEEN]:
            for sq in board.pieces(piece_type, color):
                if self._is_pinning(board, sq, opponent_king_sq, opponent):
                    return True
        
        # Check rooks and queens (straight pins)
        for piece_type in [chess.ROOK, chess.QUEEN]:
            for sq in board.pieces(piece_type, color):
                if self._is_pinning(board, sq, opponent_king_sq, opponent):
                    return True
        
        return False
    
    def _is_pinning(self, board: chess.Board, attacker_sq: int, king_sq: int, pinned_color: chess.Color) -> bool:
        """Check if attacker is pinning a piece to the king."""
        # Simple heuristic: check if there's an opponent piece between attacker and king
        attacks = board.attacks(attacker_sq)
        if king_sq not in attacks:
            return False
        
        # Check squares between attacker and king
        between = chess.SquareSet.between(attacker_sq, king_sq)
        pinned_pieces = 0
        for sq in between:
            piece = board.piece_at(sq)
            if piece and piece.color == pinned_color:
                pinned_pieces += 1
        
        return pinned_pieces == 1
    
    def _has_skewer(self, board: chess.Board, color: chess.Color) -> bool:
        """Detect if color has a skewer (attacking valuable piece with less valuable behind)."""
        # Simplified: similar to pin but valuable piece in front
        # For performance, use heuristic check
        opponent = not color
        
        for piece_type in [chess.BISHOP, chess.QUEEN, chess.ROOK]:
            for sq in board.pieces(piece_type, color):
                attacks = board.attacks(sq)
                for target_sq in attacks:
                    target = board.piece_at(target_sq)
                    if target and target.color == opponent and target.piece_type in [chess.QUEEN, chess.ROOK]:
                        # Check if another piece is behind
                        behind = chess.SquareSet.ray(sq, target_sq) - chess.SquareSet.between(sq, target_sq)
                        for behind_sq in behind:
                            behind_piece = board.piece_at(behind_sq)
                            if behind_piece and behind_piece.color == opponent:
                                return True
        
        return False
    
    def _has_discovered_attack(self, board: chess.Board, color: chess.Color) -> bool:
        """Detect potential discovered attack (piece blocking long-range piece)."""
        # Heuristic: check if moving a piece would expose an attack
        opponent_king_sq = board.king(not color)
        if opponent_king_sq is None:
            return False
        
        # Check bishops/queens on same diagonal as opponent king
        for piece_type in [chess.BISHOP, chess.QUEEN]:
            for sq in board.pieces(piece_type, color):
                if chess.SquareSet.ray(sq, opponent_king_sq):
                    between = chess.SquareSet.between(sq, opponent_king_sq)
                    if len(between) == 1:  # One piece blocking
                        blocking_sq = list(between)[0]
                        blocking_piece = board.piece_at(blocking_sq)
                        if blocking_piece and blocking_piece.color == color:
                            return True
        
        return False
    
    def _count_trapped_pieces(self, board: chess.Board, color: chess.Color) -> int:
        """Count pieces with very limited mobility (< 3 moves)."""
        count = 0
        
        for piece_type in [chess.KNIGHT, chess.BISHOP]:
            for sq in board.pieces(piece_type, color):
                # Count legal moves for this piece
                original_turn = board.turn
                board.turn = color
                moves = sum(1 for move in board.legal_moves if move.from_square == sq)
                board.turn = original_turn
                
                if moves < 3:
                    count += 1
        
        return count
    
    def _has_back_rank_threat(self, board: chess.Board, color: chess.Color) -> bool:
        """Detect back rank mate threat."""
        opponent = not color
        opponent_king_sq = board.king(opponent)
        
        if opponent_king_sq is None:
            return False
        
        # Check if king is on back rank
        rank = chess.square_rank(opponent_king_sq)
        back_rank = 0 if opponent == chess.WHITE else 7
        
        if rank != back_rank:
            return False
        
        # Check if rook or queen is attacking back rank
        for piece_type in [chess.ROOK, chess.QUEEN]:
            for sq in board.pieces(piece_type, color):
                if chess.square_rank(sq) == back_rank or board.is_attacked_by(color, opponent_king_sq):
                    # Check if king has escape squares
                    king_file = chess.square_file(opponent_king_sq)
                    escape_count = 0
                    for escape_file in [king_file - 1, king_file, king_file + 1]:
                        if 0 <= escape_file <= 7:
                            escape_sq = chess.square(escape_file, back_rank + (1 if opponent == chess.WHITE else -1))
                            if not board.is_attacked_by(color, escape_sq):
                                escape_count += 1
                    
                    if escape_count == 0:
                        return True
        
        return False
    
    def _has_hanging_pieces(self, board: chess.Board, color: chess.Color) -> bool:
        """Check if any undefended pieces exist."""
        return self._count_attacked_pieces(board, color) > 0
    
    def _count_attacked_pieces(self, board: chess.Board, color: chess.Color) -> int:
        """Count pieces under attack by opponent."""
        count = 0
        opponent = not color
        
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == color and piece.piece_type != chess.KING:
                if board.is_attacked_by(opponent, sq):
                    # Check if defended
                    if not board.is_attacked_by(color, sq):
                        count += 1
        
        return count
    
    # ========================================================================
    # ROOK PLACEMENT FEATURES (F060-F064)
    # ========================================================================
    
    def _calc_rook_placement(self, board: chess.Board) -> Dict[str, Any]:
        """Calculate rook placement features (F060-F064)."""
        return {
            "F060_white_rooks_on_open_files": self._count_rooks_on_open_files(board, chess.WHITE),
            "F060_black_rooks_on_open_files": self._count_rooks_on_open_files(board, chess.BLACK),
            "F061_white_rooks_on_semi_open_files": self._count_rooks_on_semi_open_files(board, chess.WHITE),
            "F061_black_rooks_on_semi_open_files": self._count_rooks_on_semi_open_files(board, chess.BLACK),
            "F062_white_rook_on_7th_rank": self._has_rook_on_7th(board, chess.WHITE),
            "F062_black_rook_on_7th_rank": self._has_rook_on_7th(board, chess.BLACK),
            "F063_white_connected_rooks": self._has_connected_rooks(board, chess.WHITE),
            "F063_black_connected_rooks": self._has_connected_rooks(board, chess.BLACK),
            "F064_white_rook_activity_score": self._calc_rook_activity(board, chess.WHITE),
            "F064_black_rook_activity_score": self._calc_rook_activity(board, chess.BLACK),
        }
    
    def _count_rooks_on_open_files(self, board: chess.Board, color: chess.Color) -> int:
        """Count rooks on files with no pawns."""
        count = 0
        for sq in board.pieces(chess.ROOK, color):
            file = chess.square_file(sq)
            # Check if any pawns on this file
            has_pawn = any(
                board.piece_at(chess.square(file, rank)) and 
                board.piece_at(chess.square(file, rank)).piece_type == chess.PAWN
                for rank in range(8)
            )
            if not has_pawn:
                count += 1
        return count
    
    def _count_rooks_on_semi_open_files(self, board: chess.Board, color: chess.Color) -> int:
        """Count rooks on files with no friendly pawns."""
        count = 0
        for sq in board.pieces(chess.ROOK, color):
            file = chess.square_file(sq)
            # Check if any friendly pawns on this file
            has_friendly_pawn = any(
                board.piece_at(chess.square(file, rank)) and 
                board.piece_at(chess.square(file, rank)).piece_type == chess.PAWN and
                board.piece_at(chess.square(file, rank)).color == color
                for rank in range(8)
            )
            if not has_friendly_pawn:
                count += 1
        return count
    
    def _has_rook_on_7th(self, board: chess.Board, color: chess.Color) -> bool:
        """Check if rook on 7th rank (2nd rank for Black)."""
        target_rank = 6 if color == chess.WHITE else 1
        for sq in board.pieces(chess.ROOK, color):
            if chess.square_rank(sq) == target_rank:
                return True
        return False
    
    def _has_connected_rooks(self, board: chess.Board, color: chess.Color) -> bool:
        """Check if rooks are connected (on same rank/file with no pieces between)."""
        rooks = list(board.pieces(chess.ROOK, color))
        if len(rooks) < 2:
            return False
        
        for i in range(len(rooks)):
            for j in range(i + 1, len(rooks)):
                sq1, sq2 = rooks[i], rooks[j]
                # Check if on same rank or file
                if chess.square_rank(sq1) == chess.square_rank(sq2) or chess.square_file(sq1) == chess.square_file(sq2):
                    # Check if clear path between
                    between = chess.SquareSet.between(sq1, sq2)
                    if all(board.piece_at(sq) is None for sq in between):
                        return True
        return False
    
    def _calc_rook_activity(self, board: chess.Board, color: chess.Color) -> float:
        """Calculate normalized rook activity score (0-1)."""
        rooks = list(board.pieces(chess.ROOK, color))
        if not rooks:
            return 0.0
        
        total_score = 0
        for sq in rooks:
            # Mobility
            original_turn = board.turn
            board.turn = color
            mobility = sum(1 for move in board.legal_moves if move.from_square == sq)
            board.turn = original_turn
            total_score += min(mobility / 14.0, 1.0)  # Max 14 squares for rook
        
        return total_score / len(rooks)
    
    # ========================================================================
    # ENHANCED PAWN STRUCTURE FEATURES (F024-F029)
    # ========================================================================
    
    def _calc_enhanced_pawn_structure(self, board: chess.Board) -> Dict[str, Any]:
        """Calculate enhanced pawn structure features (F024-F029)."""
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)
        
        return {
            "F024_white_backward_pawn_count": self._count_backward_pawns(board, white_pawns, chess.WHITE),
            "F024_black_backward_pawn_count": self._count_backward_pawns(board, black_pawns, chess.BLACK),
            "F025_white_pawn_chain_length": self._calc_pawn_chain_length(white_pawns),
            "F025_black_pawn_chain_length": self._calc_pawn_chain_length(black_pawns),
            "F026_white_advanced_pawn_count": self._count_advanced_pawns(white_pawns, chess.WHITE),
            "F026_black_advanced_pawn_count": self._count_advanced_pawns(black_pawns, chess.BLACK),
            "F027_white_pawn_island_count": self._count_pawn_islands(white_pawns),
            "F027_black_pawn_island_count": self._count_pawn_islands(black_pawns),
        }
    
    def _count_backward_pawns(self, board: chess.Board, pawns: chess.SquareSet, color: chess.Color) -> int:
        """Count backward pawns (behind neighboring pawns and cannot advance safely)."""
        count = 0
        pawn_files = {chess.square_file(sq) for sq in pawns}
        
        for sq in pawns:
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            
            # Check if pawns on adjacent files are ahead
            is_backward = False
            for adj_file in [file - 1, file + 1]:
                if adj_file in pawn_files:
                    for adj_sq in pawns:
                        if chess.square_file(adj_sq) == adj_file:
                            adj_rank = chess.square_rank(adj_sq)
                            if color == chess.WHITE and adj_rank > rank:
                                is_backward = True
                            elif color == chess.BLACK and adj_rank < rank:
                                is_backward = True
            
            if is_backward:
                count += 1
        
        return count
    
    def _calc_pawn_chain_length(self, pawns: chess.SquareSet) -> int:
        """Calculate length of longest pawn chain."""
        if not pawns:
            return 0
        
        max_chain = 1
        for sq in pawns:
            chain_length = 1
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            
            # Check diagonal connections
            for direction in [(1, 1), (1, -1)]:
                check_file = file + direction[0]
                check_rank = rank + direction[1]
                while 0 <= check_file <= 7 and 0 <= check_rank <= 7:
                    check_sq = chess.square(check_file, check_rank)
                    if check_sq in pawns:
                        chain_length += 1
                        check_file += direction[0]
                        check_rank += direction[1]
                    else:
                        break
            
            max_chain = max(max_chain, chain_length)
        
        return max_chain
    
    def _count_advanced_pawns(self, pawns: chess.SquareSet, color: chess.Color) -> int:
        """Count pawns past 4th rank."""
        count = 0
        threshold_rank = 4 if color == chess.WHITE else 3
        
        for sq in pawns:
            rank = chess.square_rank(sq)
            if color == chess.WHITE and rank >= threshold_rank:
                count += 1
            elif color == chess.BLACK and rank <= threshold_rank:
                count += 1
        
        return count
    
    def _count_pawn_islands(self, pawns: chess.SquareSet) -> int:
        """Count disconnected pawn groups."""
        if not pawns:
            return 0
        
        pawn_files = sorted(set(chess.square_file(sq) for sq in pawns))
        islands = 1
        
        for i in range(1, len(pawn_files)):
            if pawn_files[i] - pawn_files[i-1] > 1:
                islands += 1
        
        return islands
    
    # ========================================================================
    # KNIGHT OUTPOST FEATURES (F070-F071)
    # ========================================================================
    
    def _calc_knight_outposts(self, board: chess.Board) -> Dict[str, Any]:
        """Calculate knight outpost features (F070-F071)."""
        return {
            "F070_white_knight_outposts": self._count_knight_outposts(board, chess.WHITE),
            "F070_black_knight_outposts": self._count_knight_outposts(board, chess.BLACK),
            "F071_white_knight_mobility_avg": self._calc_avg_knight_mobility(board, chess.WHITE),
            "F071_black_knight_mobility_avg": self._calc_avg_knight_mobility(board, chess.BLACK),
        }
    
    def _count_knight_outposts(self, board: chess.Board, color: chess.Color) -> int:
        """Count knights on strong outpost squares (protected, in enemy territory)."""
        count = 0
        enemy_half_min = 4 if color == chess.WHITE else 0
        enemy_half_max = 7 if color == chess.WHITE else 3
        
        for sq in board.pieces(chess.KNIGHT, color):
            rank = chess.square_rank(sq)
            if enemy_half_min <= rank <= enemy_half_max:
                # Check if protected by own pawn
                if board.is_attacked_by(color, sq):
                    # Check if safe from enemy pawns
                    file = chess.square_file(sq)
                    safe_from_pawns = True
                    for enemy_pawn_sq in board.pieces(chess.PAWN, not color):
                        enemy_file = chess.square_file(enemy_pawn_sq)
                        if abs(file - enemy_file) <= 1:
                            safe_from_pawns = False
                            break
                    
                    if safe_from_pawns:
                        count += 1
        
        return count
    
    def _calc_avg_knight_mobility(self, board: chess.Board, color: chess.Color) -> float:
        """Calculate average mobility per knight."""
        knights = list(board.pieces(chess.KNIGHT, color))
        if not knights:
            return 0.0
        
        total_moves = 0
        original_turn = board.turn
        board.turn = color
        
        for sq in knights:
            total_moves += sum(1 for move in board.legal_moves if move.from_square == sq)
        
        board.turn = original_turn
        return total_moves / len(knights)
    
    # ========================================================================
    # CENTER CONTROL FEATURES (F080-F082)
    # ========================================================================
    
    def _calc_center_control(self, board: chess.Board) -> Dict[str, Any]:
        """Calculate center control features (F080-F082)."""
        return {
            "F080_white_center_pawn_count": self._count_center_pawns(board, chess.WHITE),
            "F080_black_center_pawn_count": self._count_center_pawns(board, chess.BLACK),
            "F081_white_center_control_score": self._calc_center_control_score(board, chess.WHITE),
            "F081_black_center_control_score": self._calc_center_control_score(board, chess.BLACK),
            "F082_white_space_advantage": self._calc_space_advantage(board, chess.WHITE),
            "F082_black_space_advantage": self._calc_space_advantage(board, chess.BLACK),
        }
    
    def _count_center_pawns(self, board: chess.Board, color: chess.Color) -> int:
        """Count pawns on central squares (d4, d5, e4, e5)."""
        central_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        count = 0
        for sq in central_squares:
            piece = board.piece_at(sq)
            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                count += 1
        return count
    
    def _calc_center_control_score(self, board: chess.Board, color: chess.Color) -> int:
        """Count attacks on central squares."""
        central_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        score = 0
        for sq in central_squares:
            if board.is_attacked_by(color, sq):
                score += 1
        return score
    
    def _calc_space_advantage(self, board: chess.Board, color: chess.Color) -> int:
        """Count squares controlled in opponent's half."""
        enemy_half_min = 4 if color == chess.WHITE else 0
        enemy_half_max = 7 if color == chess.WHITE else 3
        
        count = 0
        for rank in range(enemy_half_min, enemy_half_max + 1):
            for file in range(8):
                sq = chess.square(file, rank)
                if board.is_attacked_by(color, sq):
                    count += 1
        
        return count
    
    # ========================================================================
    # DEVELOPMENT FEATURES (F090-F091)
    # ========================================================================
    
    def _calc_development(self, board: chess.Board) -> Dict[str, Any]:
        """Calculate development features (F090-F091)."""
        return {
            "F090_white_pieces_developed": self._count_developed_pieces(board, chess.WHITE),
            "F090_black_pieces_developed": self._count_developed_pieces(board, chess.BLACK),
        }
    
    def _count_developed_pieces(self, board: chess.Board, color: chess.Color) -> int:
        """Count minor pieces off back rank."""
        back_rank = 0 if color == chess.WHITE else 7
        count = 0
        
        for piece_type in [chess.KNIGHT, chess.BISHOP]:
            for sq in board.pieces(piece_type, color):
                if chess.square_rank(sq) != back_rank:
                    count += 1
        
        return count
    
    # ========================================================================
    # MOVE CONTEXT FEATURES (F050-F053)
    # ========================================================================
    
    def _calc_move_context(self, board: chess.Board, move: chess.Move) -> Dict[str, Any]:
        """Calculate move context features (F050-F053)."""
        return {
            "F050_is_capture": board.is_capture(move),
            "F051_is_check": board.gives_check(move),
            "F052_is_promotion": move.promotion is not None,
            "F053_is_castling": board.is_castling(move),
        }
    
    # ========================================================================
    # MULTI-MOVE CONTEXT FEATURES (F100-F114) - CRITICAL FOR GRADE SPECTRUM
    # ========================================================================
    
    def _calc_multi_move_context(self, stockfish_analysis: Dict[str, Any], v7p3r_move: Optional[chess.Move], board: chess.Board) -> Dict[str, Any]:
        """
        Calculate multi-move context features from Stockfish top-5 analysis (F100-F114).
        
        This is CRITICAL for fixing binary classification - gives AI the full spectrum
        of move quality, not just "best" vs "worst".
        
        Args:
            stockfish_analysis: The stockfish_analysis block from JSONL record
            v7p3r_move: The move V7P3R actually played
            board: Board position before the move
        """
        # Extract top-5 move evaluations
        top_moves = stockfish_analysis.get('top_moves', [])
        
        # Helper to extract eval in centipawns
        def eval_to_cp(eval_dict):
            if eval_dict is None:
                return 0
            if 'mate' in eval_dict:
                mate_in = eval_dict['mate']
                return 10000 if mate_in > 0 else -10000
            return eval_dict.get('cp', 0)
        
        # Extract evaluations for top 5 moves (or fewer if not available)
        evals = [eval_to_cp(m.get('eval')) for m in top_moves[:5]]
        while len(evals) < 5:
            evals.append(0)  # Pad with zeros if < 5 moves
        
        # Calculate eval gaps
        eval_gap_1_2 = abs(evals[0] - evals[1]) if len(top_moves) >= 2 else 0
        eval_gap_2_3 = abs(evals[1] - evals[2]) if len(top_moves) >= 3 else 0
        
        # Find V7P3R's move in top-5 and get its eval
        v7p3r_eval = 0
        v7p3r_move_uci = v7p3r_move.uci() if v7p3r_move else None
        for move_data in top_moves:
            if move_data.get('move') == v7p3r_move_uci:
                v7p3r_eval = eval_to_cp(move_data.get('eval'))
                break
        
        # Calculate eval loss (how much worse than best)
        eval_loss = abs(evals[0] - v7p3r_eval) if v7p3r_move else 0
        
        # Move diversity score (standard deviation of top-5 evals)
        import statistics
        move_diversity = statistics.stdev(evals) if len([e for e in evals if e != 0]) > 1 else 0
        
        # Position sharpness (large gap between best and 2nd = tactical)
        position_sharpness = eval_gap_1_2
        
        # Categorize move types
        def categorize_move(move_uci):
            if not move_uci:
                return "quiet"
            try:
                move = chess.Move.from_uci(move_uci)
                if board.is_capture(move):
                    return "capture"
                elif board.gives_check(move):
                    return "check"
                elif move.promotion:
                    return "promotion"
                elif board.is_castling(move):
                    return "castling"
                else:
                    return "quiet"
            except:
                return "quiet"
        
        best_move_type = categorize_move(top_moves[0].get('move')) if top_moves else "quiet"
        second_move_type = categorize_move(top_moves[1].get('move')) if len(top_moves) > 1 else "quiet"
        v7p3r_move_type = categorize_move(v7p3r_move_uci) if v7p3r_move_uci else "quiet"
        
        # Alternative move quality (how many moves within 50cp of best)
        alternative_quality = sum(1 for e in evals if abs(e - evals[0]) <= 50)
        
        return {
            # Top-5 evaluations (F100-F104)
            "F100_best_move_eval_cp": evals[0],
            "F101_second_move_eval_cp": evals[1],
            "F102_third_move_eval_cp": evals[2],
            "F103_fourth_move_eval_cp": evals[3],
            "F104_fifth_move_eval_cp": evals[4],
            
            # Eval gaps (F105-F106)
            "F105_eval_gap_best_to_second": eval_gap_1_2,
            "F106_eval_gap_second_to_third": eval_gap_2_3,
            
            # V7P3R's move analysis (F107-F108)
            "F107_v7p3r_move_eval_cp": v7p3r_eval,
            "F108_v7p3r_eval_loss": eval_loss,
            
            # Position characteristics (F109-F110)
            "F109_move_diversity_score": move_diversity,
            "F110_position_sharpness": position_sharpness,
            
            # Move type categorization (F111-F113)
            "F111_best_move_type": best_move_type,
            "F112_second_move_type": second_move_type,
            "F113_v7p3r_move_type": v7p3r_move_type,
            
            # Alternative move quality (F114)
            "F114_alternative_move_quality": alternative_quality,
        }


def main():
    """Main entry point for feature calculation."""
    parser = argparse.ArgumentParser(
        description="Calculate heuristic features for extracted positions"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL file (positions without features)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file (positions with features)"
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        choices=["minimal", "standard", "full"],
        default="standard",
        help="Feature set preset (default: standard)"
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
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create feature config
    config = FeatureConfig.from_preset(args.feature_set)
    
    # Create calculator
    calculator = FeatureCalculator(config)
    
    # Process file
    calculator.process_file(args.input, args.output)


if __name__ == "__main__":
    main()
