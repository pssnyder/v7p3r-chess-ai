#!/usr/bin/env python3
"""
V7P3R Evaluators v18.3.1 - ALL Evaluation Functions Consolidated

CONSOLIDATION SUMMARY:
- v7p3r_fast_evaluator.py (440 lines)
- v7p3r_modular_eval.py (330 lines)
- v7p3r_bitboard_evaluator.py (1321 lines)
- v7p3r_eval_modules.py (553 lines)
- v7p3r_eval_selector.py (455 lines)
- v7p3r_position_context.py (409 lines)
- v7p3r_move_safety.py (168 lines)

TOTAL: ~3,676 lines of evaluation logic

PURPOSE:
- Single source of truth for all evaluation functions
- Enables systematic profiling of function usage
- Identifies active vs placeholder implementations

Author: Pat Snyder
Date: Auto-generated consolidation
"""

import chess
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# V7P3R_POSITION_CONTEXT
# ============================================================================

"""
V7P3R Position Context Calculator

Calculates position characteristics ONCE before search, persists through entire tree.

This module provides the foundation for modular evaluation by determining
what type of position we're in and what evaluation modules should be active.

Author: Pat Snyder
Created: 2025-12-23 (v18.2 Modular Evaluation System)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Set
import chess


class GamePhase(Enum):
    """Unified game phase classification (single source of truth)"""
    OPENING = "opening"                    # Move < 12, pieces ≥ 12
    MIDDLEGAME_COMPLEX = "middlegame_complex"  # Material 1300-2500cp, pieces 7-11
    MIDDLEGAME_SIMPLE = "middlegame_simple"    # Material 1300-2500cp, pieces 4-6
    ENDGAME_COMPLEX = "endgame_complex"        # Material < 1300cp, pieces 3-6
    ENDGAME_SIMPLE = "endgame_simple"          # Material < 800cp, pieces ≤ 2


class MaterialBalance(Enum):
    """Material imbalance classification"""
    EQUAL = "equal"                # |diff| < 100cp
    SLIGHT_ADVANTAGE = "slight"    # 100-300cp
    ADVANTAGE = "advantage"        # 300-500cp
    WINNING = "winning"            # 500-900cp
    CRUSHING = "crushing"          # > 900cp


class TacticalFlags(Enum):
    """Binary tactical indicators (hints for module selection)"""
    KING_EXPOSED = "king_exposed"              # King has ≤2 pawn shield
    PIECES_HANGING = "pieces_hanging"          # Undefended pieces exist (needs verification)
    CHECKS_AVAILABLE = "checks_available"      # Can give check (queen/rook near enemy king)
    PINS_PRESENT = "pins_present"              # Pin opportunities exist
    FORKS_PRESENT = "forks_present"            # Fork opportunities exist
    BACK_RANK_WEAK = "back_rank_weak"         # Back rank mate threat


@dataclass
class PositionContext:
    """
    Immutable position characteristics calculated once per search.
    
    This context is passed to ALL evaluation modules and persists
    through the entire search tree (not recalculated per node).
    
    Design Principle: Calculate expensive checks ONCE, use O(1) lookups everywhere else.
    """
    # Time management
    time_remaining: float        # Seconds left on clock
    time_per_move: float         # Allocated time for this move
    time_pressure: bool          # < 30 seconds remaining
    
    # Game phase (single source of truth)
    game_phase: GamePhase        # Authoritative phase classification
    move_number: int             # Full move count (1-based)
    
    # Material
    material_balance: MaterialBalance  # Who's winning materially
    material_diff_cp: int        # Centipawn difference (+ = we're winning)
    total_material: int          # Combined material on board
    
    # Piece inventory (for module activation)
    piece_types: Set[chess.PieceType]  # {PAWN, KNIGHT, BISHOP, ROOK, QUEEN}
    white_pieces: int            # Count of white pieces (excluding king)
    black_pieces: int            # Count of black pieces (excluding king)
    
    # Positional flags (quick checks for module relevance)
    queens_on_board: bool        # At least one queen present
    bishops_on_board: bool       # At least one bishop present
    opposite_bishops: bool       # Each side has 1+ bishops (bishop pair relevant)
    rooks_on_board: bool         # At least one rook present
    
    # Tactical indicators (hints, not full tactical analysis)
    tactical_flags: Set[TacticalFlags]  # Active tactical themes
    king_safety_critical: bool   # King exposure detected
    
    # Endgame specifics
    pawn_endgame: bool          # Only kings + pawns
    pure_piece_endgame: bool    # No pawns, only pieces
    theoretical_draw: bool       # Known drawn material (K vs K, K+B vs K, etc)
    
    # Search context
    depth_target: int           # Planned search depth based on time
    use_fast_profile: bool      # Force fast evaluation (time pressure)


class PositionContextCalculator:
    """
    Calculates position context once before search.
    
    Design Principles:
    - O(1) or O(64) complexity (single board scan, no move generation)
    - No recursive analysis
    - Cache-friendly (single object creation)
    - Fast enough to call every root search (~0.1ms target)
    """
    
    # Material values (standard)
    MATERIAL_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900
    }
    
    def calculate(self, board: chess.Board, time_remaining: float = 300.0, 
                  time_per_move: float = 5.0) -> PositionContext:
        """
        Main entry point: Calculate all position characteristics.
        
        Args:
            board: Current chess position
            time_remaining: Seconds left on clock
            time_per_move: Allocated time for this move
            
        Returns:
            PositionContext with all calculated fields
            
        Time Complexity: O(64) - single board scan
        Space Complexity: O(1) - fixed-size dataclass
        """
        # Material calculation (O(64))
        material_info = self._calculate_material(board)
        
        # Piece inventory (O(64))
        piece_info = self._calculate_piece_inventory(board)
        
        # Game phase (O(1) - uses material_info)
        game_phase = self._determine_game_phase(
            board, material_info, piece_info
        )
        
        # Tactical flags (O(64) - simple board scan, no move gen)
        tactical_flags = self._detect_tactical_flags(board, piece_info)
        
        # Time pressure detection (O(1))
        # CRITICAL: time_remaining is what we're allocating for THIS move (not total clock)
        # time_pressure = truly desperate (must move instantly)
        # use_fast_profile = less time available (skip expensive modules)
        time_pressure = time_remaining < 3.0  # Less than 3s for this move = emergency
        use_fast_profile = time_per_move < 2.0  # Less than 2s/move average = use fast profile
        
        # Depth target based on time (O(1))
        depth_target = self._calculate_depth_target(
            time_per_move, game_phase, time_pressure
        )
        
        return PositionContext(
            # Time
            time_remaining=time_remaining,
            time_per_move=time_per_move,
            time_pressure=time_pressure,
            
            # Phase
            game_phase=game_phase,
            move_number=board.fullmove_number,
            
            # Material
            material_balance=material_info['balance'],
            material_diff_cp=material_info['diff_cp'],
            total_material=material_info['total'],
            
            # Pieces
            piece_types=piece_info['types'],
            white_pieces=piece_info['white_count'],
            black_pieces=piece_info['black_count'],
            
            # Flags
            queens_on_board=chess.QUEEN in piece_info['types'],
            bishops_on_board=chess.BISHOP in piece_info['types'],
            opposite_bishops=piece_info['opposite_bishops'],
            rooks_on_board=chess.ROOK in piece_info['types'],
            
            # Tactical
            tactical_flags=tactical_flags,
            king_safety_critical=TacticalFlags.KING_EXPOSED in tactical_flags,
            
            # Endgame
            pawn_endgame=piece_info['pawn_endgame'],
            pure_piece_endgame=piece_info['pure_piece_endgame'],
            theoretical_draw=material_info['theoretical_draw'],
            
            # Search
            depth_target=depth_target,
            use_fast_profile=use_fast_profile
        )
    
    def _calculate_material(self, board: chess.Board) -> dict:
        """
        Calculate material counts and balance (O(64))
        
        Returns:
            dict with 'diff_cp', 'total', 'balance', 'theoretical_draw'
        """
        white_material = 0
        black_material = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                value = self.MATERIAL_VALUES[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value
        
        # Calculate from our perspective (positive = we're winning)
        diff_cp = white_material - black_material
        if not board.turn:  # Black to move
            diff_cp = -diff_cp
        
        # Determine balance category
        abs_diff = abs(diff_cp)
        if abs_diff < 100:
            balance = MaterialBalance.EQUAL
        elif abs_diff < 300:
            balance = MaterialBalance.SLIGHT_ADVANTAGE
        elif abs_diff < 500:
            balance = MaterialBalance.ADVANTAGE
        elif abs_diff < 900:
            balance = MaterialBalance.WINNING
        else:
            balance = MaterialBalance.CRUSHING
        
        # Theoretical draw detection
        total = white_material + black_material
        theoretical_draw = (
            total == 0 or  # K vs K
            total <= 330   # K+B vs K, K+N vs K, or K+B vs K+B (endgame tables)
        )
        
        return {
            'diff_cp': diff_cp,
            'total': total,
            'balance': balance,
            'theoretical_draw': theoretical_draw
        }
    
    def _calculate_piece_inventory(self, board: chess.Board) -> dict:
        """
        Count pieces and determine endgame types (O(64))
        
        Returns:
            dict with piece counts and flags
        """
        piece_types = set()
        white_count = 0
        black_count = 0
        white_bishops = 0
        black_bishops = 0
        has_pawns = False
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                piece_types.add(piece.piece_type)
                
                if piece.color == chess.WHITE:
                    white_count += 1
                    if piece.piece_type == chess.BISHOP:
                        white_bishops += 1
                else:
                    black_count += 1
                    if piece.piece_type == chess.BISHOP:
                        black_bishops += 1
                
                if piece.piece_type == chess.PAWN:
                    has_pawns = True
        
        return {
            'types': piece_types,
            'white_count': white_count,
            'black_count': black_count,
            'opposite_bishops': white_bishops > 0 and black_bishops > 0,
            'pawn_endgame': piece_types == {chess.PAWN},
            'pure_piece_endgame': len(piece_types) > 0 and not has_pawns
        }
    
    def _determine_game_phase(self, board: chess.Board, 
                              material_info: dict, piece_info: dict) -> GamePhase:
        """
        Unified game phase detection (single source of truth).
        
        Logic:
        1. Opening: move < 12 AND pieces ≥ 12
        2. Endgame: material < 1300cp OR (pieces ≤ 4 AND no queens)
        3. Middlegame: everything else
        4. Complex vs Simple: based on piece count
        
        This replaces all inconsistent thresholds across the codebase.
        """
        move_num = board.fullmove_number
        total_material = material_info['total']
        total_pieces = piece_info['white_count'] + piece_info['black_count']
        has_queens = chess.QUEEN in piece_info['types']
        
        # Opening
        if move_num < 12 and total_pieces >= 12:
            return GamePhase.OPENING
        
        # Endgame
        if total_material < 1300 or (total_pieces <= 4 and not has_queens):
            if total_pieces <= 2:
                return GamePhase.ENDGAME_SIMPLE
            else:
                return GamePhase.ENDGAME_COMPLEX
        
        # Middlegame
        if total_pieces <= 6:
            return GamePhase.MIDDLEGAME_SIMPLE
        else:
            return GamePhase.MIDDLEGAME_COMPLEX
    
    def _detect_tactical_flags(self, board: chess.Board, 
                               piece_info: dict) -> Set[TacticalFlags]:
        """
        Quick tactical flag detection (no move generation).
        
        Note: These are HINTS for evaluation selection, not full tactical analysis.
        Full tactical checks done by selected evaluation modules.
        """
        flags = set()
        
        # King exposure (simple pawn shield check)
        our_king = board.king(board.turn)
        if our_king is not None:
            pawn_shield_count = self._count_pawn_shield(board, our_king, board.turn)
            if pawn_shield_count <= 2:
                flags.add(TacticalFlags.KING_EXPOSED)
        
        # Checks available (heuristic: queen or rook near enemy king)
        if chess.QUEEN in piece_info['types'] or chess.ROOK in piece_info['types']:
            enemy_king = board.king(not board.turn)
            if enemy_king is not None:
                # Check if we have queen/rook (could potentially give check)
                flags.add(TacticalFlags.CHECKS_AVAILABLE)
        
        return flags
    
    def _count_pawn_shield(self, board: chess.Board, king_square: int, color: chess.Color) -> int:
        """
        Count pawns in front of king (pawn shield).
        
        Returns: Number of friendly pawns protecting king (0-3)
        """
        king_rank = chess.square_rank(king_square)
        king_file = chess.square_file(king_square)
        
        pawn_shield_count = 0
        
        if color == chess.WHITE and king_rank < 2:
            # Check squares in front of white king
            for file_offset in [-1, 0, 1]:
                check_file = king_file + file_offset
                if 0 <= check_file <= 7:
                    check_square = chess.square(check_file, king_rank + 1)
                    piece = board.piece_at(check_square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == chess.WHITE:
                        pawn_shield_count += 1
        
        elif color == chess.BLACK and king_rank > 5:
            # Check squares in front of black king
            for file_offset in [-1, 0, 1]:
                check_file = king_file + file_offset
                if 0 <= check_file <= 7:
                    check_square = chess.square(check_file, king_rank - 1)
                    piece = board.piece_at(check_square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == chess.BLACK:
                        pawn_shield_count += 1
        
        return pawn_shield_count
    
    def _calculate_depth_target(self, time_per_move: float, 
                                game_phase: GamePhase, 
                                time_pressure: bool) -> int:
        """
        Determine target search depth based on available time.
        
        Fast profiles can search deeper due to lower per-node cost.
        
        Args:
            time_per_move: Allocated time in seconds
            game_phase: Current game phase
            time_pressure: Whether in time pressure
            
        Returns:
            Target depth (4-8)
        """
        if time_pressure:
            return 4  # Emergency mode
        elif time_per_move < 5.0:
            return 5  # Blitz fast mode
        elif time_per_move < 15.0:
            return 6  # Blitz/rapid normal
        elif time_per_move < 60.0:
            return 7  # Rapid deep search
        else:
            return 8  # Long time control



# ============================================================================
# V7P3R_FAST_EVALUATOR
# ============================================================================

"""
V7P3R Fast Evaluator - V18.3 Optimized

Ultra-fast evaluation using PST + Material + Strategic bonuses

V18.3 Optimization: Direct square indexing for PST (30-40% faster)
- Pre-computed flipped tables for Black
- Single array lookup per piece (no rank flipping)
- Expected: 0.046ms -> 0.028ms per evaluation

Speed: ~0.001ms per position (40x faster than bitboard evaluator)
Architecture: 60% PST + 40% Material + Middlegame Bonuses
"""

import chess
from typing import Dict, Optional, List

# Material values (from MaterialOpponent - proven to prevent sacrifices)
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 325,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}

# Piece-Square Tables (from PositionalOpponent - proven 81% win rate)
# Values in centipawns, White perspective

PAWN_PST = [
    [  0,  0,  0,  0,  0,  0,  0,  0],  # 1st rank
    [ 50, 50, 50, 50, 50, 50, 50, 50],  # 2nd rank  
    [ 60, 60, 70, 80, 80, 70, 60, 60],  # 3rd rank
    [ 70, 70, 80, 90, 90, 80, 70, 70],  # 4th rank
    [100,100,110,120,120,110,100,100],  # 5th rank
    [200,200,220,250,250,220,200,200],  # 6th rank
    [400,400,450,500,500,450,400,400],  # 7th rank
    [900,900,900,900,900,900,900,900],  # 8th rank
]

KNIGHT_PST = [
    [200,220,240,250,250,240,220,200],
    [220,240,260,270,270,260,240,220],
    [240,260,300,320,320,300,260,240],
    [250,270,320,350,350,320,270,250],
    [250,270,320,350,350,320,270,250],
    [240,260,300,320,320,300,260,240],
    [220,240,260,270,270,260,240,220],
    [200,220,240,250,250,240,220,200],
]

BISHOP_PST = [
    [250,260,270,280,280,270,260,250],
    [260,300,290,290,290,290,300,260],
    [270,290,320,300,300,320,290,270],
    [280,290,300,350,350,300,290,280],
    [280,290,300,350,350,300,290,280],
    [270,290,320,300,300,320,290,270],
    [260,300,290,290,290,290,300,260],
    [250,260,270,280,280,270,260,250],
]

ROOK_PST = [
    [400,410,420,430,430,420,410,400],
    [450,450,450,450,450,450,450,450],
    [440,440,440,440,440,440,440,440],
    [440,440,440,440,440,440,440,440],
    [440,440,440,440,440,440,440,440],
    [450,450,450,450,450,450,450,450],
    [500,500,500,500,500,500,500,500],
    [480,480,480,480,480,480,480,480],
]

QUEEN_PST = [
    [700,710,720,730,730,720,710,700],
    [710,750,750,750,750,750,750,710],
    [720,750,800,800,800,800,750,720],
    [730,750,800,850,850,800,750,730],
    [730,750,800,850,850,800,750,730],
    [720,750,800,800,800,800,750,720],
    [710,750,750,750,750,750,750,710],
    [700,710,720,730,730,720,710,700],
]

KING_MIDDLEGAME_PST = [
    [ 50, 80,  0,  0,  0,  0, 80, 50],
    [  0,  0,  0,  0,  0,  0,  0,  0],
    [ -50,-50,-50,-50,-50,-50,-50,-50],
    [-100,-100,-100,-100,-100,-100,-100,-100],
    [-150,-150,-150,-150,-150,-150,-150,-150],
    [-200,-200,-200,-200,-200,-200,-200,-200],
    [-250,-250,-250,-250,-250,-250,-250,-250],
    [-300,-300,-300,-300,-300,-300,-300,-300],
]

KING_ENDGAME_PST = [
    [-50,-40,-30,-20,-20,-30,-40,-50],
    [-30,-20,-10,  0,  0,-10,-20,-30],
    [-30,-10, 20, 30, 30, 20,-10,-30],
    [-30,-10, 30, 40, 40, 30,-10,-30],
    [-30,-10, 30, 40, 40, 30,-10,-30],
    [-30,-10, 20, 30, 30, 20,-10,-30],
    [-30,-30,  0,  0,  0,  0,-30,-30],
    [-50,-30,-30,-30,-30,-30,-30,-50],
]


# =============================================================================
# V18.3 OPTIMIZATION: Pre-computed PST for direct square indexing
# =============================================================================
# Format: PST_DIRECT[piece_type][color][square] -> value
# This eliminates rank flipping and reduces PST lookups by 30-40%

def _flatten_pst(pst_2d: List[List[int]]) -> List[int]:
    """Convert 2D PST to 1D array indexed by square number"""
    flat = []
    for rank in range(8):
        for file in range(8):
            flat.append(pst_2d[rank][file])
    return flat

def _flip_pst(pst_2d: List[List[int]]) -> List[int]:
    """Flip PST for Black and convert to 1D"""
    flat = []
    for rank in range(7, -1, -1):  # Reverse ranks for Black
        for file in range(8):
            flat.append(pst_2d[rank][file])
    return flat

# Pre-compute flattened PSTs for White (normal) and Black (flipped)
PAWN_PST_WHITE = _flatten_pst(PAWN_PST)
PAWN_PST_BLACK = _flip_pst(PAWN_PST)

KNIGHT_PST_WHITE = _flatten_pst(KNIGHT_PST)
KNIGHT_PST_BLACK = _flip_pst(KNIGHT_PST)

BISHOP_PST_WHITE = _flatten_pst(BISHOP_PST)
BISHOP_PST_BLACK = _flip_pst(BISHOP_PST)

ROOK_PST_WHITE = _flatten_pst(ROOK_PST)
ROOK_PST_BLACK = _flip_pst(ROOK_PST)

QUEEN_PST_WHITE = _flatten_pst(QUEEN_PST)
QUEEN_PST_BLACK = _flip_pst(QUEEN_PST)

KING_MG_PST_WHITE = _flatten_pst(KING_MIDDLEGAME_PST)
KING_MG_PST_BLACK = _flip_pst(KING_MIDDLEGAME_PST)

KING_EG_PST_WHITE = _flatten_pst(KING_ENDGAME_PST)
KING_EG_PST_BLACK = _flip_pst(KING_ENDGAME_PST)

# Organize into lookup structure for O(1) access
PST_DIRECT = {
    chess.PAWN: {chess.WHITE: PAWN_PST_WHITE, chess.BLACK: PAWN_PST_BLACK},
    chess.KNIGHT: {chess.WHITE: KNIGHT_PST_WHITE, chess.BLACK: KNIGHT_PST_BLACK},
    chess.BISHOP: {chess.WHITE: BISHOP_PST_WHITE, chess.BLACK: BISHOP_PST_BLACK},
    chess.ROOK: {chess.WHITE: ROOK_PST_WHITE, chess.BLACK: ROOK_PST_BLACK},
    chess.QUEEN: {chess.WHITE: QUEEN_PST_WHITE, chess.BLACK: QUEEN_PST_BLACK},
}

# King has separate middlegame/endgame tables
KING_PST_DIRECT = {
    'middlegame': {chess.WHITE: KING_MG_PST_WHITE, chess.BLACK: KING_MG_PST_BLACK},
    'endgame': {chess.WHITE: KING_EG_PST_WHITE, chess.BLACK: KING_EG_PST_BLACK},
}


class V7P3RFastEvaluator:
    """
    Fast PST-based evaluator for maximum search depth
    Architecture: 60% PST + 40% Material + Middlegame Bonuses
    """
    
    def __init__(self):
        """Initialize fast evaluator"""
        self.piece_values = PIECE_VALUES
    
    def evaluate(self, board: chess.Board) -> int:
        """
        Main evaluation function - calls all component methods
        Returns: score in centipawns (positive = White advantage)
        """
        # V18.3: Now modular - can call components individually
        material_score = self.evaluate_material(board)
        pst_score = self.evaluate_pst(board)
        strategic_bonus = self.evaluate_strategic(board)
        
        # Combine scores: 60% PST + 40% Material + Strategic
        combined_score = int(pst_score * 0.6 + material_score * 0.4 + strategic_bonus)
        
        # Return from current player perspective
        return combined_score if board.turn == chess.WHITE else -combined_score
    
    def evaluate_material(self, board: chess.Board) -> int:
        """
        Calculate material balance only
        Returns: material score (White perspective)
        """
        material_score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                material_value = self.piece_values.get(piece.piece_type, 0)
                if piece.color == chess.WHITE:
                    material_score += material_value
                else:
                    material_score -= material_value
        return material_score
    
    def evaluate_pst(self, board: chess.Board) -> int:
        """
        Calculate piece-square table values only
        V18.3: Optimized with direct square indexing (30-40% faster)
        Returns: PST score (White perspective)
        """
        pst_score = 0
        is_endgame = self._is_endgame(board)
        
        # V18.3: Direct PST lookup - no rank flipping needed
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_type = piece.piece_type
                color = piece.color
                
                # King has separate middlegame/endgame tables
                if piece_type == chess.KING:
                    phase = 'endgame' if is_endgame else 'middlegame'
                    value = KING_PST_DIRECT[phase][color][square]
                else:
                    # Direct lookup: PST_DIRECT[piece_type][color][square]
                    value = PST_DIRECT[piece_type][color][square]
                
                # Add to score (White positive, Black negative)
                pst_score += value if color == chess.WHITE else -value
        
        return pst_score
    
    def evaluate_strategic(self, board: chess.Board) -> int:
        """
        Calculate strategic/positional bonuses
        Returns: bonus score (White perspective)
        """
        is_endgame = self._is_endgame(board)
        
        # Only apply middlegame bonuses in middlegame
        if not is_endgame and not self._is_opening(board):
            return self._calculate_middlegame_bonuses(board)
        
        return 0
    
    def _get_piece_square_value(self, piece: chess.Piece, square: chess.Square, is_endgame: bool = False) -> int:
        """Get PST value for piece at square"""
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        
        # Flip rank for black pieces (PST is from White perspective)
        if piece.color == chess.BLACK:
            rank = 7 - rank
        
        piece_type = piece.piece_type
        
        # Look up PST value
        if piece_type == chess.PAWN:
            value = PAWN_PST[rank][file]
        elif piece_type == chess.KNIGHT:
            value = KNIGHT_PST[rank][file]
        elif piece_type == chess.BISHOP:
            value = BISHOP_PST[rank][file]
        elif piece_type == chess.ROOK:
            value = ROOK_PST[rank][file]
        elif piece_type == chess.QUEEN:
            value = QUEEN_PST[rank][file]
        elif piece_type == chess.KING:
            value = KING_ENDGAME_PST[rank][file] if is_endgame else KING_MIDDLEGAME_PST[rank][file]
        else:
            value = 0
        
        # Negate for black pieces
        return value if piece.color == chess.WHITE else -value
    
    def _is_endgame(self, board: chess.Board) -> bool:
        """Detect endgame phase (no queens or low material)"""
        # No queens = endgame
        if not board.pieces(chess.QUEEN, chess.WHITE) and not board.pieces(chess.QUEEN, chess.BLACK):
            return True
        
        # Low material = endgame
        white_material = sum(len(board.pieces(pt, chess.WHITE)) * self.piece_values.get(pt, 0) 
                            for pt in [chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.QUEEN])
        black_material = sum(len(board.pieces(pt, chess.BLACK)) * self.piece_values.get(pt, 0)
                            for pt in [chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.QUEEN])
        
        return white_material < 800 and black_material < 800
    
    def _is_opening(self, board: chess.Board) -> bool:
        """Detect opening phase (< 10 moves, pieces not developed)"""
        return board.fullmove_number < 10
    
    def _calculate_middlegame_bonuses(self, board: chess.Board) -> int:
        """
        Calculate middlegame positional bonuses
        Returns: bonus in centipawns (White perspective)
        
        Bonuses:
        - Rooks on open files: +20cp
        - Rooks on semi-open files: +10cp
        - King pawn shield: +10cp per shield pawn
        - Passed pawns: +30cp
        - Doubled pawns: -20cp per extra pawn
        """
        bonus = 0
        
        # BONUS 1: Rooks on open/semi-open files
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.ROOK:
                file = chess.square_file(square)
                is_open = True
                is_semi_open = True
                
                # Check if file has pawns
                for rank in range(8):
                    sq = chess.square(file, rank)
                    p = board.piece_at(sq)
                    if p and p.piece_type == chess.PAWN:
                        if p.color == piece.color:
                            is_open = False
                            is_semi_open = False
                            break
                        else:
                            is_open = False
                
                if is_open:
                    bonus += 20 if piece.color == chess.WHITE else -20
                elif is_semi_open:
                    bonus += 10 if piece.color == chess.WHITE else -10
        
        # BONUS 2: King safety - pawn shield
        for color in [chess.WHITE, chess.BLACK]:
            king_square = board.king(color)
            if king_square is not None:
                king_file = chess.square_file(king_square)
                king_rank = chess.square_rank(king_square)
                
                # Check pawns in front of king (shield)
                shield_pawns = 0
                for file_offset in [-1, 0, 1]:
                    shield_file = king_file + file_offset
                    if 0 <= shield_file <= 7:
                        # Check 1-2 ranks ahead
                        for rank_offset in [1, 2]:
                            if color == chess.WHITE:
                                shield_rank = king_rank + rank_offset
                            else:
                                shield_rank = king_rank - rank_offset
                            
                            if 0 <= shield_rank <= 7:
                                sq = chess.square(shield_file, shield_rank)
                                p = board.piece_at(sq)
                                if p and p.piece_type == chess.PAWN and p.color == color:
                                    shield_pawns += 1
                                    break
                
                if color == chess.WHITE:
                    bonus += shield_pawns * 10
                else:
                    bonus -= shield_pawns * 10
        
        # BONUS 3: Pawn structure (passed pawns, doubled pawns)
        for file in range(8):
            white_pawns = []
            black_pawns = []
            
            for rank in range(8):
                sq = chess.square(file, rank)
                p = board.piece_at(sq)
                if p and p.piece_type == chess.PAWN:
                    if p.color == chess.WHITE:
                        white_pawns.append(rank)
                    else:
                        black_pawns.append(rank)
            
            # Doubled pawns penalty (-20cp per extra pawn)
            if len(white_pawns) > 1:
                bonus -= 20 * (len(white_pawns) - 1)
            if len(black_pawns) > 1:
                bonus += 20 * (len(black_pawns) - 1)
            
            # Passed pawns bonus (+30cp)
            for rank in white_pawns:
                if self._is_passed_pawn(board, chess.square(file, rank), chess.WHITE):
                    bonus += 30
            
            for rank in black_pawns:
                if self._is_passed_pawn(board, chess.square(file, rank), chess.BLACK):
                    bonus -= 30
        
        return bonus
    
    def _is_passed_pawn(self, board: chess.Board, square: chess.Square, color: chess.Color) -> bool:
        """Check if pawn at square is passed"""
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        
        # Check adjacent files and this file ahead
        for adj_file in [file - 1, file, file + 1]:
            if 0 <= adj_file <= 7:
                if color == chess.WHITE:
                    # Check ranks ahead for white
                    for r in range(rank + 1, 8):
                        sq = chess.square(adj_file, r)
                        p = board.piece_at(sq)
                        if p and p.piece_type == chess.PAWN and p.color == chess.BLACK:
                            return False
                else:
                    # Check ranks ahead for black
                    for r in range(0, rank):
                        sq = chess.square(adj_file, r)
                        p = board.piece_at(sq)
                        if p and p.piece_type == chess.PAWN and p.color == chess.WHITE:
                            return False
        
        return True


# For compatibility with existing code
class V7P3RScoringCalculationFast(V7P3RFastEvaluator):
    """Alias for compatibility with v14.1's naming convention"""
    
    def __init__(self, piece_values: Optional[Dict[int, int]] = None):
        super().__init__()
        if piece_values:
            self.piece_values = piece_values
    
    def calculate_current_board_score(self, board: chess.Board) -> int:
        """Compatibility method matching v14.1's interface"""
        return self.evaluate(board)



# ============================================================================
# V7P3R_MODULAR_EVAL
# ============================================================================

"""
V18.2: Modular Position Evaluation
Executes only the modules selected by the current profile.

Philosophy:
- DESPERATE mode: Skip 22 strategic modules, run only 10 tactical modules
- EMERGENCY mode: Minimal 5-module evaluation for time pressure
- FAST mode: Balanced 12-18 modules for speed
- TACTICAL mode: Full tactical suite (18-22 modules)
- ENDGAME mode: Endgame-specific subset (10-15 modules)
- COMPREHENSIVE mode: All relevant modules (20-28 modules)

Author: Pat Snyder
Created: 2025-12-27
"""

import chess
from typing import Dict, Set


class ModularEvaluator:
    """Executes only selected evaluation modules"""
    
    def __init__(self, fast_evaluator):
        """
        Initialize with reference to existing fast evaluator for module implementations.
        
        Args:
            fast_evaluator: The v7p3r_fast_evaluator.FastEvaluator instance
        """
        self.fast_eval = fast_evaluator
        
        # Piece values for material counting
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
    
    def evaluate_with_profile(self, board: chess.Board, profile: EvaluationProfile, 
                               context: PositionContext) -> float:
        """
        Evaluate position using selected profile's modules.
        
        **V18.3 ACTUAL MODULAR EXECUTION**:
        Execute only the evaluation components in the profile.
        
        KEY OPTIMIZATION:
        - DESPERATE/EMERGENCY/FAST: Material + PST only (skip strategic) → 2-3x faster
        - Full profiles: All components for complete evaluation
        
        Expected: DESPERATE depth 8-9 (vs baseline 6.0 with monolithic)
        
        Args:
            board: Chess position to evaluate
            profile: Selected evaluation profile with active modules
            context: Position context
        
        Returns:
            Evaluation score (centipawns, current player perspective)
        """
        # Get active modules for O(1) lookup
        active_modules = set(profile.active_modules)
        
        # Check if we need strategic evaluation (the expensive part)
        needs_strategic = any(module in active_modules for module in [
            'king_safety_basic', 'king_safety_complex', 'pawn_structure',
            'rook_open_files', 'bishop_pair', 'knight_outposts',
            'center_control', 'space_advantage', 'piece_mobility',
            'passed_pawns', 'pawn_chains', 'isolated_pawns',
            'doubled_pawns', 'backward_pawns'
        ])
        
        # FAST PATH: Material + PST only (DESPERATE, EMERGENCY, FAST modes)
        if not needs_strategic:
            material = self.fast_eval.evaluate_material(board)
            pst = self.fast_eval.evaluate_pst(board)
            # Combine with standard weights: 60% PST + 40% Material
            score = int(pst * 0.6 + material * 0.4)
            return score if board.turn == chess.WHITE else -score
        
        # FULL PATH: All components (TACTICAL, ENDGAME, COMPREHENSIVE modes)
        score = self.fast_eval.evaluate(board)
        return score
    
    # ==================== MATERIAL & PST ====================
    
    def _evaluate_material(self, board: chess.Board) -> float:
        """Material count - most critical module"""
        white_material = 0
        black_material = 0
        
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            white_material += len(board.pieces(piece_type, chess.WHITE)) * self.piece_values[piece_type]
            black_material += len(board.pieces(piece_type, chess.BLACK)) * self.piece_values[piece_type]
        
        diff = white_material - black_material
        return diff if board.turn == chess.WHITE else -diff
    
    def _evaluate_pst(self, board: chess.Board) -> float:
        """Piece-square table evaluation"""
        # Delegate to fast evaluator's PST logic
        white_score = self.fast_eval._evaluate_piece_placement(board, chess.WHITE)
        black_score = self.fast_eval._evaluate_piece_placement(board, chess.BLACK)
        diff = white_score - black_score
        return diff if board.turn == chess.WHITE else -diff
    
    # ==================== TACTICAL MODULES ====================
    
    def _evaluate_hanging_pieces(self, board: chess.Board) -> float:
        """Detect undefended pieces (critical for DESPERATE mode)"""
        score = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Check if piece is attacked and not defended
                attackers = len(board.attackers(not piece.color, square))
                defenders = len(board.attackers(piece.color, square))
                
                if attackers > 0 and defenders == 0:
                    # Hanging piece penalty
                    penalty = self.piece_values[piece.piece_type] * 0.5
                    if piece.color == board.turn:
                        score -= penalty
                    else:
                        score += penalty
        
        return score
    
    def _evaluate_captures(self, board: chess.Board) -> float:
        """Evaluate available captures"""
        score = 0
        for move in board.legal_moves:
            if board.is_capture(move):
                captured = board.piece_at(move.to_square)
                if captured:
                    score += self.piece_values[captured.piece_type] * 0.1
        return score
    
    def _evaluate_checks(self, board: chess.Board) -> float:
        """Bonus for checking moves"""
        score = 0
        for move in board.legal_moves:
            if board.gives_check(move):
                score += 15  # Small bonus for check availability
        return score
    
    def _evaluate_tactical_patterns(self, board: chess.Board) -> float:
        """Detect pins, forks, skewers (simplified)"""
        # Placeholder - full implementation would detect these patterns
        return 0
    
    def _evaluate_exchanges(self, board: chess.Board) -> float:
        """Static Exchange Evaluation (simplified)"""
        # Placeholder - full SEE implementation
        return 0
    
    def _evaluate_trapped_pieces(self, board: chess.Board) -> float:
        """Detect trapped pieces"""
        # Placeholder
        return 0
    
    def _evaluate_back_rank(self, board: chess.Board) -> float:
        """Back rank weakness detection"""
        # Placeholder
        return 0
    
    # ==================== STRATEGIC MODULES ====================
    
    def _evaluate_king_safety_basic(self, board: chess.Board) -> float:
        """Basic king safety (castling rights, pawn shield)"""
        score = 0
        
        # Castling rights bonus
        if board.has_kingside_castling_rights(board.turn):
            score += 15
        if board.has_queenside_castling_rights(board.turn):
            score += 10
        
        if board.has_kingside_castling_rights(not board.turn):
            score -= 15
        if board.has_queenside_castling_rights(not board.turn):
            score -= 10
        
        return score
    
    def _evaluate_king_safety_complex(self, board: chess.Board) -> float:
        """Advanced king safety (attacks near king)"""
        # Placeholder
        return 0
    
    def _evaluate_move_safety(self, board: chess.Board) -> float:
        """Basic move safety check"""
        # Simplified - just check if we're leaving pieces hanging
        return 0  # Fast evaluator handles this internally
    
    def _evaluate_pawn_structure(self, board: chess.Board) -> float:
        """Overall pawn structure quality"""
        # Delegate to fast evaluator
        white_score = self.fast_eval._evaluate_pawn_structure(board, chess.WHITE)
        black_score = self.fast_eval._evaluate_pawn_structure(board, chess.BLACK)
        diff = white_score - black_score
        return diff if board.turn == chess.WHITE else -diff
    
    def _evaluate_passed_pawns(self, board: chess.Board) -> float:
        """Passed pawn bonus"""
        # Placeholder
        return 0
    
    def _evaluate_pawn_chains(self, board: chess.Board) -> float:
        """Connected pawn bonus"""
        # Placeholder
        return 0
    
    def _evaluate_isolated_pawns(self, board: chess.Board) -> float:
        """Isolated pawn penalty"""
        # Placeholder
        return 0
    
    def _evaluate_backward_pawns(self, board: chess.Board) -> float:
        """Backward pawn penalty"""
        return 0
    
    def _evaluate_doubled_pawns(self, board: chess.Board) -> float:
        """Doubled pawn penalty"""
        # Placeholder
        return 0
    
    def _evaluate_bishop_pair(self, board: chess.Board) -> float:
        """Bishop pair bonus"""
        white_bishops = len(board.pieces(chess.BISHOP, chess.WHITE))
        black_bishops = len(board.pieces(chess.BISHOP, chess.BLACK))
        
        score = 0
        if white_bishops >= 2:
            score += 30
        if black_bishops >= 2:
            score -= 30
        
        return score if board.turn == chess.WHITE else -score
    
    def _evaluate_knight_outposts(self, board: chess.Board) -> float:
        """Knight outpost bonus"""
        # Placeholder
        return 0
    
    def _evaluate_rook_files(self, board: chess.Board) -> float:
        """Rook on open/semi-open file"""
        # Placeholder
        return 0
    
    def _evaluate_rook_seventh(self, board: chess.Board) -> float:
        """Rook on 7th rank bonus"""
        # Placeholder
        return 0
    
    def _evaluate_connected_rooks(self, board: chess.Board) -> float:
        """Connected rooks bonus"""
        # Placeholder
        return 0
    
    def _evaluate_queen_activity(self, board: chess.Board) -> float:
        """Queen mobility/centralization"""
        # Placeholder
        return 0
    
    def _evaluate_mobility(self, board: chess.Board) -> float:
        """Piece mobility"""
        our_moves = len(list(board.legal_moves))
        
        board.push(chess.Move.null())
        their_moves = len(list(board.legal_moves)) if board.legal_moves else 0
        board.pop()
        
        return (our_moves - their_moves) * 2
    
    def _evaluate_center_control(self, board: chess.Board) -> float:
        """Central square control"""
        # Placeholder
        return 0
    
    def _evaluate_space(self, board: chess.Board) -> float:
        """Space advantage"""
        # Placeholder
        return 0
    
    def _evaluate_development(self, board: chess.Board) -> float:
        """Piece development in opening"""
        # Placeholder
        return 0
    
    def _evaluate_tempo(self, board: chess.Board) -> float:
        """Time/tempo evaluation"""
        # Placeholder
        return 0
    
    # ==================== ENDGAME MODULES ====================
    
    def _evaluate_endgame_patterns(self, board: chess.Board, context: PositionContext) -> float:
        """Known endgame patterns (KPK, etc.)"""
        # Placeholder
        return 0
    
    def _evaluate_king_activity_endgame(self, board: chess.Board) -> float:
        """King centralization in endgame"""
        # Placeholder
        return 0
    
    def _evaluate_pawn_races(self, board: chess.Board) -> float:
        """Pawn race evaluation"""
        # Placeholder
        return 0
    
    def _evaluate_opposition(self, board: chess.Board) -> float:
        """King opposition in endgame"""
        return 0
    
    def _evaluate_zugzwang(self, board: chess.Board) -> float:
        """Zugzwang detection"""
        return 0
    
    def _evaluate_repetition(self, board: chess.Board) -> float:
        """Repetition penalty"""
        return 0



# ============================================================================
# V7P3R_BITBOARD_EVALUATOR
# ============================================================================

"""
V7P3R Bitboard-Based Evaluation System
Ultra-fast evaluation using bitboard operations for maximum performance

Optimized for tournament play with no nudge system overhead
"""

import chess
from typing import Dict, Tuple, List


class V7P3RBitboardEvaluator:
    """
    High-performance bitboard-based evaluation system
    Uses bitwise operations for 10x+ speed improvement
    Optimized for maximum performance without nudge system overhead
    """
    
    def __init__(self, piece_values: Dict[int, int], enable_nudges: bool = False):
        self.piece_values = piece_values
        
        # Pre-calculate bitboard masks for ultra-fast evaluation
        self._init_bitboard_constants()
        self._init_attack_tables()
    
    def _init_bitboard_constants(self):
        """Initialize constant bitboard masks"""
        
        # Rank masks
        self.RANK_1 = 0x00000000000000FF
        self.RANK_2 = 0x000000000000FF00
        self.RANK_3 = 0x0000000000FF0000
        self.RANK_4 = 0x00000000FF000000
        self.RANK_5 = 0x000000FF00000000
        self.RANK_6 = 0x0000FF0000000000
        self.RANK_7 = 0x00FF000000000000
        self.RANK_8 = 0xFF00000000000000
        
        # File masks
        self.FILE_A = 0x0101010101010101
        self.FILE_B = 0x0202020202020202
        self.FILE_C = 0x0404040404040404
        self.FILE_D = 0x0808080808080808
        self.FILE_E = 0x1010101010101010
        self.FILE_F = 0x2020202020202020
        self.FILE_G = 0x4040404040404040
        self.FILE_H = 0x8080808080808080
        
        # Center squares
        self.CENTER = 0x0000001818000000  # d4, d5, e4, e5
        self.EXTENDED_CENTER = 0x00003C3C3C3C0000  # c3-f3 to c6-f6
        
        # Edge squares for endgame king driving
        self.EDGES = (self.RANK_1 | self.RANK_8 | self.FILE_A | self.FILE_H)
        
        # King safety masks
        self.WHITE_KINGSIDE_CASTLE = 0x0000000000000060  # f1, g1
        self.WHITE_QUEENSIDE_CASTLE = 0x000000000000000E  # b1, c1, d1
        self.BLACK_KINGSIDE_CASTLE = 0x6000000000000000  # f8, g8
        self.BLACK_QUEENSIDE_CASTLE = 0x0E00000000000000  # b8, c8, d8
        
        # Pawn structure masks
        self.WHITE_PASSED_PAWN_MASKS = self._generate_passed_pawn_masks(True)
        self.BLACK_PASSED_PAWN_MASKS = self._generate_passed_pawn_masks(False)
        
        # Development squares
        self.KNIGHT_OUTPOSTS = 0x0000240000240000  # c4, c5, f4, f5
        self.BISHOP_DIAGONALS = 0x8040201008040201 | 0x0102040810204080
    
    def _init_attack_tables(self):
        """Initialize pre-calculated attack tables for super-fast lookups"""
        
        # Knight attack patterns from each square
        self.KNIGHT_ATTACKS = [0] * 64
        for sq in range(64):
            self.KNIGHT_ATTACKS[sq] = self._calc_knight_attacks(sq)
        
        # King attack patterns
        self.KING_ATTACKS = [0] * 64
        for sq in range(64):
            self.KING_ATTACKS[sq] = self._calc_king_attacks(sq)
        
        # Pawn attack patterns
        self.WHITE_PAWN_ATTACKS = [0] * 64
        self.BLACK_PAWN_ATTACKS = [0] * 64
        for sq in range(64):
            self.WHITE_PAWN_ATTACKS[sq] = self._calc_white_pawn_attacks(sq)
            self.BLACK_PAWN_ATTACKS[sq] = self._calc_black_pawn_attacks(sq)
    
    def _calc_knight_attacks(self, square: int) -> int:
        """Calculate knight attack bitboard for a square"""
        attacks = 0
        rank, file = divmod(square, 8)
        
        # All 8 possible knight moves
        knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), 
                       (1, -2), (1, 2), (2, -1), (2, 1)]
        
        for dr, df in knight_moves:
            new_rank, new_file = rank + dr, file + df
            if 0 <= new_rank < 8 and 0 <= new_file < 8:
                attacks |= (1 << (new_rank * 8 + new_file))
        
        return attacks
    
    def _calc_king_attacks(self, square: int) -> int:
        """Calculate king attack bitboard for a square"""
        attacks = 0
        rank, file = divmod(square, 8)
        
        # All 8 possible king moves
        for dr in [-1, 0, 1]:
            for df in [-1, 0, 1]:
                if dr == 0 and df == 0:
                    continue
                new_rank, new_file = rank + dr, file + df
                if 0 <= new_rank < 8 and 0 <= new_file < 8:
                    attacks |= (1 << (new_rank * 8 + new_file))
        
        return attacks
    
    def _calc_white_pawn_attacks(self, square: int) -> int:
        """Calculate white pawn attack bitboard"""
        attacks = 0
        rank, file = divmod(square, 8)
        
        if rank < 7:  # Can attack forward
            if file > 0:  # Can attack left
                attacks |= (1 << ((rank + 1) * 8 + file - 1))
            if file < 7:  # Can attack right
                attacks |= (1 << ((rank + 1) * 8 + file + 1))
        
        return attacks
    
    def _calc_black_pawn_attacks(self, square: int) -> int:
        """Calculate black pawn attack bitboard"""
        attacks = 0
        rank, file = divmod(square, 8)
        
        if rank > 0:  # Can attack forward
            if file > 0:  # Can attack left
                attacks |= (1 << ((rank - 1) * 8 + file - 1))
            if file < 7:  # Can attack right
                attacks |= (1 << ((rank - 1) * 8 + file + 1))
        
        return attacks
    
    def _generate_passed_pawn_masks(self, is_white: bool) -> list:
        """Generate passed pawn masks for fast passed pawn detection"""
        masks = [0] * 64
        
        for square in range(64):
            rank, file = divmod(square, 8)
            mask = 0
            
            # Add files to check (own file and adjacent files)
            for check_file in [file - 1, file, file + 1]:
                if 0 <= check_file < 8:
                    if is_white:
                        # For white, check ranks ahead
                        for check_rank in range(rank + 1, 8):
                            mask |= (1 << (check_rank * 8 + check_file))
                    else:
                        # For black, check ranks ahead (down)
                        for check_rank in range(0, rank):
                            mask |= (1 << (check_rank * 8 + check_file))
            
            masks[square] = mask
        
        return masks
    
    def evaluate_bitboard(self, board: chess.Board, color: chess.Color) -> float:
        """
        Ultra-fast bitboard evaluation using bitwise operations
        This should give us 20,000+ NPS
        """
        
        # Convert chess.Board to bitboards for fast processing
        white_pawns = int(board.pieces(chess.PAWN, chess.WHITE))
        black_pawns = int(board.pieces(chess.PAWN, chess.BLACK))
        white_knights = int(board.pieces(chess.KNIGHT, chess.WHITE))
        black_knights = int(board.pieces(chess.KNIGHT, chess.BLACK))
        white_bishops = int(board.pieces(chess.BISHOP, chess.WHITE))
        black_bishops = int(board.pieces(chess.BISHOP, chess.BLACK))
        white_rooks = int(board.pieces(chess.ROOK, chess.WHITE))
        black_rooks = int(board.pieces(chess.ROOK, chess.BLACK))
        white_queens = int(board.pieces(chess.QUEEN, chess.WHITE))
        black_queens = int(board.pieces(chess.QUEEN, chess.BLACK))
        white_king = int(board.pieces(chess.KING, chess.WHITE))
        black_king = int(board.pieces(chess.KING, chess.BLACK))
        
        white_pieces = white_pawns | white_knights | white_bishops | white_rooks | white_queens | white_king
        black_pieces = black_pawns | black_knights | black_bishops | black_rooks | black_queens | black_king
        all_pieces = white_pieces | black_pieces
        
        # V12.1: Calculate material count for game phase detection
        total_material = self._popcount(all_pieces & ~(white_pawns | black_pawns))
        
        score = 0.0
        
        # 1. MATERIAL (ultra-fast bit counting)
        score += self._popcount(white_pawns) * self.piece_values[chess.PAWN]
        score += self._popcount(white_knights) * self.piece_values[chess.KNIGHT]
        score += self._popcount(white_bishops) * self.piece_values[chess.BISHOP]
        score += self._popcount(white_rooks) * self.piece_values[chess.ROOK]
        score += self._popcount(white_queens) * self.piece_values[chess.QUEEN]
        
        score -= self._popcount(black_pawns) * self.piece_values[chess.PAWN]
        score -= self._popcount(black_knights) * self.piece_values[chess.KNIGHT]
        score -= self._popcount(black_bishops) * self.piece_values[chess.BISHOP]
        score -= self._popcount(black_rooks) * self.piece_values[chess.ROOK]
        score -= self._popcount(black_queens) * self.piece_values[chess.QUEEN]
        
        # 2. CENTER CONTROL (V12.1: Enhanced for opening aggression)
        white_center_pawns = white_pawns & self.CENTER
        black_center_pawns = black_pawns & self.CENTER
        score += self._popcount(white_center_pawns) * 10
        score -= self._popcount(black_center_pawns) * 10
        
        white_extended_center = white_pawns & self.EXTENDED_CENTER
        black_extended_center = black_pawns & self.EXTENDED_CENTER
        score += self._popcount(white_extended_center) * 5
        score -= self._popcount(black_extended_center) * 5
        
        # V12.1: Opening phase center control bonus for pieces (not just pawns)
        if total_material >= 20:  # Opening/early middlegame
            white_center_pieces = (white_knights | white_bishops) & self.CENTER
            black_center_pieces = (black_knights | black_bishops) & self.CENTER
            score += self._popcount(white_center_pieces) * 15  # Bonus for pieces on center
            score -= self._popcount(black_center_pieces) * 15
            
            white_extended_pieces = (white_knights | white_bishops) & self.EXTENDED_CENTER
            black_extended_pieces = (black_knights | black_bishops) & self.EXTENDED_CENTER
            score += self._popcount(white_extended_pieces) * 8  # Bonus for pieces near center
            score -= self._popcount(black_extended_pieces) * 8
        
        # 3. PIECE DEVELOPMENT (V12.1: Enhanced development evaluation)
        white_knight_outposts = white_knights & self.KNIGHT_OUTPOSTS
        black_knight_outposts = black_knights & self.KNIGHT_OUTPOSTS
        score += self._popcount(white_knight_outposts) * 15
        score -= self._popcount(black_knight_outposts) * 15
        
        # V12.1: Opening development penalty - punish undeveloped pieces
        if total_material >= 18:  # Opening phase
            # Count pieces still on starting squares
            white_undeveloped = 0
            black_undeveloped = 0
            
            # Knights on starting squares (b1, g1 for white; b8, g8 for black)
            if white_knights & (1 << 1):  # b1
                white_undeveloped += 1
            if white_knights & (1 << 6):  # g1
                white_undeveloped += 1
            if black_knights & (1 << 57):  # b8
                black_undeveloped += 1
            if black_knights & (1 << 62):  # g8
                black_undeveloped += 1
                
            # Bishops on starting squares (c1, f1 for white; c8, f8 for black)
            if white_bishops & (1 << 2):  # c1
                white_undeveloped += 1
            if white_bishops & (1 << 5):  # f1
                white_undeveloped += 1
            if black_bishops & (1 << 58):  # c8
                black_undeveloped += 1
            if black_bishops & (1 << 61):  # f8
                black_undeveloped += 1
            
            # Apply development penalties
            score -= white_undeveloped * 12  # Penalty for undeveloped White pieces
            score += black_undeveloped * 12  # Penalty for undeveloped Black pieces
        
        # 4. ENHANCED KING SAFETY & CASTLING EVALUATION (V12.4)
        score += self._evaluate_enhanced_castling(board, color)
        
        # 5. PAWN STRUCTURE (passed pawns - ultra-fast)
        score += self._count_passed_pawns(white_pawns, black_pawns, True) * 20
        score -= self._count_passed_pawns(black_pawns, white_pawns, False) * 20
        
        # 6. ENDGAME CONSIDERATIONS  
        if total_material <= 8:  # Endgame
            # Drive enemy king to edge (always from White's perspective)
            black_king_on_edge = black_king & self.EDGES
            white_king_on_edge = white_king & self.EDGES
            score += self._popcount(black_king_on_edge) * 10  # Good for White if Black king on edge
            score -= self._popcount(white_king_on_edge) * 10  # Bad for White if White king on edge
        
        # 7. V12.1: STRICTER DRAW PREVENTION
        # Encourage aggressive play and discourage repetitive/passive positions
        
        # Fifty-move rule awareness: stronger penalty as we approach limit
        if board.halfmove_clock > 30:
            draw_penalty = (board.halfmove_clock - 30) * 2.0  # Escalating penalty
            score -= draw_penalty if color == chess.WHITE else -draw_penalty
        
        # The repetition detection was calling board.fen() multiple times per evaluation,
        # causing massive performance degradation. Commenting out for tournament performance.
        # TODO: Implement fast repetition detection using zobrist hashing
        

        # Encourage piece activity: penalty for pieces on back ranks in middlegame
        if total_material >= 12:  # Middlegame
            white_back_rank_pieces = (white_knights | white_bishops | white_rooks | white_queens) & (self.RANK_1 | self.RANK_2)
            black_back_rank_pieces = (black_knights | black_bishops | black_rooks | black_queens) & (self.RANK_7 | self.RANK_8)
            
            activity_penalty = (self._popcount(white_back_rank_pieces) - self._popcount(black_back_rank_pieces)) * 3
            score -= activity_penalty if color == chess.WHITE else -activity_penalty

        return score if color == chess.WHITE else -score
    
    def _popcount(self, bitboard: int) -> int:
        """Ultra-fast population count (number of 1 bits)"""
        return bin(bitboard).count('1')
    
    def _count_passed_pawns(self, our_pawns: int, enemy_pawns: int, is_white: bool) -> int:
        """Count passed pawns using pre-calculated masks"""
        passed_count = 0
        pawns = our_pawns
        
        while pawns:
            # Get least significant bit (first pawn)
            pawn_square = (pawns & -pawns).bit_length() - 1
            
            # Check if it's passed using pre-calculated mask
            if is_white:
                mask = self.WHITE_PASSED_PAWN_MASKS[pawn_square]
            else:
                mask = self.BLACK_PASSED_PAWN_MASKS[pawn_square]
            
            if not (enemy_pawns & mask):
                passed_count += 1
            
            # Remove this pawn and continue
            pawns &= pawns - 1
        
        return passed_count
    
    def _evaluate_enhanced_castling(self, board: chess.Board, color: chess.Color) -> float:
        """
        Enhanced castling evaluation for V12.4
        Rewards actual castling, penalizes wasted castling rights
        ALWAYS returns score from White's perspective (positive = good for White)
        """
        score = 0.0
        
        # Determine if we're in opening phase
        opening_phase = len(board.move_stack) < 20
        
        # WHITE evaluation
        white_has_castled = self._has_castled(board, chess.WHITE)
        
        if white_has_castled:
            # Reward successful castling for White
            score += 50.0
            king_square = board.king(chess.WHITE)
            if king_square in [chess.G1, chess.C1]:
                score += 25.0  # Safety bonus for White
        else:
            # Check White castling availability
            can_castle_kingside = board.has_kingside_castling_rights(chess.WHITE)
            can_castle_queenside = board.has_queenside_castling_rights(chess.WHITE)
            
            if opening_phase:
                if can_castle_kingside:
                    score += 30.0  # Good for White to have castling rights
                if can_castle_queenside:
                    score += 20.0
                
                # Penalty for White moving king without castling
                king_square = board.king(chess.WHITE)
                if king_square != chess.E1 and not white_has_castled:
                    score -= 50.0  # Bad for White
            else:
                # Mild penalty for unused castling in middlegame
                if can_castle_kingside or can_castle_queenside:
                    score -= 10.0

        # BLACK evaluation (opposite perspective)
        black_has_castled = self._has_castled(board, chess.BLACK)
        
        if black_has_castled:
            # Penalize successful castling for Black (good for Black = bad for White)
            score -= 50.0
            king_square = board.king(chess.BLACK)
            if king_square in [chess.G8, chess.C8]:
                score -= 25.0  # Safety bonus for Black = penalty for White
        else:
            can_castle_kingside = board.has_kingside_castling_rights(chess.BLACK)
            can_castle_queenside = board.has_queenside_castling_rights(chess.BLACK)
            
            if opening_phase:
                if can_castle_kingside:
                    score -= 30.0  # Bad for White if Black has castling rights
                if can_castle_queenside:
                    score -= 20.0
                
                # CRITICAL FIX: Reward White when Black moves king without castling
                king_square = board.king(chess.BLACK)
                if king_square != chess.E8 and not black_has_castled:
                    score += 50.0  # GOOD for White when Black blunders king move!
            else:
                if can_castle_kingside or can_castle_queenside:
                    score += 10.0  # Good for White if Black wastes castling rights
        
        return score
    
    def _has_castled(self, board: chess.Board, color: chess.Color) -> bool:
        """Check if the specified color has already castled"""
        king_square = board.king(color)
        
        if color == chess.WHITE:
            if king_square == chess.G1:
                rook_on_f1 = board.piece_at(chess.F1)
                return (rook_on_f1 is not None and 
                       rook_on_f1.piece_type == chess.ROOK and 
                       rook_on_f1.color == chess.WHITE)
            elif king_square == chess.C1:
                rook_on_d1 = board.piece_at(chess.D1)
                return (rook_on_d1 is not None and 
                       rook_on_d1.piece_type == chess.ROOK and 
                       rook_on_d1.color == chess.WHITE)
        else:  # BLACK
            if king_square == chess.G8:
                rook_on_f8 = board.piece_at(chess.F8)
                return (rook_on_f8 is not None and 
                       rook_on_f8.piece_type == chess.ROOK and 
                       rook_on_f8.color == chess.BLACK)
            elif king_square == chess.C8:
                rook_on_d8 = board.piece_at(chess.D8)
                return (rook_on_d8 is not None and 
                       rook_on_d8.piece_type == chess.ROOK and 
                       rook_on_d8.color == chess.BLACK)
        
        # Check move history for explicit castling moves
        for move in board.move_stack:
            if board.is_castling(move):
                from_square = move.from_square
                if color == chess.WHITE and from_square == chess.E1:
                    return True
                elif color == chess.BLACK and from_square == chess.E8:
                    return True
        
        return False

    def detect_bitboard_tactics(self, board: chess.Board, move: chess.Move) -> float:
        """
        V12.6 CONSOLIDATED: Detect tactical patterns using bitboard operations
        Returns a bonus score for tactical moves (pins, forks, skewers, discovered attacks)
        """
        tactical_bonus = 0.0
        
        # Make the move to analyze the resulting position
        board.push(move)
        
        try:
            our_color = not board.turn  # We just moved, so it's opponent's turn
            
            # Legacy bitboard tactics for additional analysis
            moving_piece = board.piece_at(move.to_square)
            if moving_piece:
                fork_bonus = self._analyze_fork_bitboard(board, move.to_square, moving_piece, board.turn)
                tactical_bonus += fork_bonus
                
                # Analyze for pins and skewers using ray attacks
                if moving_piece.piece_type in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
                    pin_skewer_bonus = self._analyze_pins_skewers_bitboard(board, move.to_square, moving_piece, board.turn)
                    tactical_bonus += pin_skewer_bonus
            
        except Exception:
            # If analysis fails, return 0 bonus
            pass
        finally:
            board.pop()
        
        return tactical_bonus
    
    def _analyze_fork_bitboard(self, board: chess.Board, square: int, piece: chess.Piece, enemy_color: chess.Color) -> float:
        """Analyze fork patterns using bitboards"""
        if piece.piece_type == chess.KNIGHT:
            # Knight fork detection
            attacks = self.KNIGHT_ATTACKS[square]
            enemy_pieces = 0
            high_value_targets = 0
            
            for target_sq in range(64):
                if attacks & (1 << target_sq):
                    target_piece = board.piece_at(target_sq)
                    if target_piece and target_piece.color == enemy_color:
                        enemy_pieces += 1
                        if target_piece.piece_type in [chess.QUEEN, chess.ROOK, chess.KING]:
                            high_value_targets += 1
            
            # Knight forking 2+ pieces gets bonus, more for high-value targets
            if enemy_pieces >= 2:
                return 50.0 + (high_value_targets * 25.0)
        
        return 0.0
    
    def _analyze_pins_skewers_bitboard(self, board: chess.Board, square: int, piece: chess.Piece, enemy_color: chess.Color) -> float:
        """Analyze pin and skewer patterns using ray attacks"""
        # This is a simplified version - full implementation would need sliding piece attack generation
        # For now, just give a small bonus for pieces that could create pins/skewers
        
        if piece.piece_type in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
            # Look for aligned enemy pieces that could be pinned/skewered
            bonus = 0.0
            
            # Check if we're attacking towards the enemy king
            enemy_king_sq = None
            for sq in range(64):
                p = board.piece_at(sq)
                if p and p.piece_type == chess.KING and p.color == enemy_color:
                    enemy_king_sq = sq
                    break
            
            if enemy_king_sq is not None:
                # Simple heuristic: if we're on the same rank/file/diagonal as enemy king
                sq_rank, sq_file = divmod(square, 8)
                king_rank, king_file = divmod(enemy_king_sq, 8)
                
                if (sq_rank == king_rank or sq_file == king_file or 
                    abs(sq_rank - king_rank) == abs(sq_file - king_file)):
                    bonus += 15.0  # Potential pin/skewer bonus
            
            return bonus
        
        return 0.0

    def evaluate_pawn_structure(self, board: chess.Board, color: bool) -> float:
        """
        V12.6 CONSOLIDATED: Comprehensive pawn structure evaluation using bitboards
        Returns score from the perspective of the given color
        """
        total_score = 0.0
        
        # Get all pawns for this color as bitboard
        pawns = board.pieces(chess.PAWN, color)
        
        # Evaluate each pawn and overall structure
        total_score += self._evaluate_passed_pawns_bitboard(board, pawns, color)
        total_score += self._evaluate_isolated_pawns_bitboard(board, pawns, color)
        total_score += self._evaluate_doubled_pawns_bitboard(board, pawns, color)
        total_score += self._evaluate_backward_pawns_bitboard(board, pawns, color)
        total_score += self._evaluate_connected_pawns_bitboard(board, pawns, color)
        total_score += self._evaluate_pawn_chains_bitboard(board, pawns, color)
        total_score += self._evaluate_pawn_storms_bitboard(board, pawns, color)
        
        return total_score
    
    def _evaluate_passed_pawns_bitboard(self, board: chess.Board, pawns: chess.SquareSet, color: bool) -> float:
        """V18.1 ENHANCED: Passed pawns with exponential scaling by rank"""
        score = 0.0
        advanced_passed_bonus = 30
        
        # V18.1: Check if we're in endgame for king-pawn coordination bonus
        material_count = self._count_material_bitboard(board)
        is_endgame = material_count < 2000
        
        for pawn_square in pawns:
            if self._is_passed_pawn_bitboard(board, pawn_square, color):
                pawn_rank = chess.square_rank(pawn_square)
                
                # V18.1: EXPONENTIAL PASSED PAWN BONUS
                # Calculate advancement (how far pawn has advanced from starting rank)
                if color == chess.WHITE:
                    # White pawns start on rank 1, advance toward rank 7
                    advancement = pawn_rank - 1  # 0 (not moved) to 6 (7th rank)
                else:
                    # Black pawns start on rank 6, advance toward rank 0
                    advancement = 6 - pawn_rank  # 0 (not moved) to 6 (2nd rank)
                
                # Exponential bonus: 20 * 2^advancement
                # 2nd rank: 20cp, 3rd: 40cp, 4th: 80cp, 5th: 160cp, 6th: 320cp, 7th: 640cp
                passed_pawn_bonus = 20 * (2 ** advancement)
                score += passed_pawn_bonus
                
                # V18.1: Extra bonus if king supports the pawn (endgame only)
                if is_endgame:
                    king_square = board.king(color)
                    if king_square is not None:
                        king_dist = chess.square_distance(king_square, pawn_square)
                        if king_dist <= 2:
                            score += 30  # King-pawn coordination
                
                # Connected passed pawns get extra bonus
                if self._has_connected_passed_pawn_bitboard(board, pawn_square, color):
                    score += 20
        
        return score
    
    def _evaluate_isolated_pawns_bitboard(self, board: chess.Board, pawns: chess.SquareSet, color: bool) -> float:
        """Evaluate isolated pawns using bitboard operations"""
        score = 0.0
        isolated_pawn_penalty = 15
        
        for pawn_square in pawns:
            if self._is_isolated_pawn_bitboard(board, pawn_square, color):
                penalty = isolated_pawn_penalty
                
                # Isolated pawns on open files are worse
                if self._is_on_open_file_bitboard(board, pawn_square):
                    penalty += 10
                
                score -= penalty
        
        return score
    
    def _evaluate_doubled_pawns_bitboard(self, board: chess.Board, pawns: chess.SquareSet, color: bool) -> float:
        """Evaluate doubled pawns using bitboard operations"""
        score = 0.0
        doubled_pawn_penalty = 25
        file_counts = {}
        
        # Count pawns per file
        for pawn_square in pawns:
            file_idx = chess.square_file(pawn_square)
            file_counts[file_idx] = file_counts.get(file_idx, 0) + 1
        
        # Penalize multiple pawns on same file
        for file_idx, count in file_counts.items():
            if count > 1:
                penalty = doubled_pawn_penalty * (count - 1)
                score -= penalty
        
        return score
    
    def _evaluate_backward_pawns_bitboard(self, board: chess.Board, pawns: chess.SquareSet, color: bool) -> float:
        """Evaluate backward pawns using bitboard operations"""
        score = 0.0
        backward_pawn_penalty = 12
        
        for pawn_square in pawns:
            if self._is_backward_pawn_bitboard(board, pawn_square, color):
                penalty = backward_pawn_penalty
                
                # Backward pawns on semi-open files are worse
                if self._is_on_semi_open_file_bitboard(board, pawn_square, color):
                    penalty += 8
                
                score -= penalty
        
        return score
    
    def _evaluate_connected_pawns_bitboard(self, board: chess.Board, pawns: chess.SquareSet, color: bool) -> float:
        """Evaluate connected pawns using bitboard operations"""
        score = 0.0
        connected_pawn_bonus = 8
        
        for pawn_square in pawns:
            if self._has_pawn_support_bitboard(board, pawn_square, color):
                score += connected_pawn_bonus
        
        return score
    
    def _evaluate_pawn_chains_bitboard(self, board: chess.Board, pawns: chess.SquareSet, color: bool) -> float:
        """Evaluate pawn chains using bitboard operations"""
        score = 0.0
        pawn_chain_bonus = 5
        chain_lengths = self._find_pawn_chains_bitboard(board, pawns, color)
        
        for length in chain_lengths:
            if length >= 2:
                score += pawn_chain_bonus * length
        
        return score
    
    def _evaluate_pawn_storms_bitboard(self, board: chess.Board, pawns: chess.SquareSet, color: bool) -> float:
        """Evaluate pawn storms using bitboard operations"""
        score = 0.0
        pawn_storm_bonus = 10
        
        # Find enemy king
        enemy_king_square = board.king(not color)
        if enemy_king_square is None:
            return 0.0
        
        enemy_king_file = chess.square_file(enemy_king_square)
        
        for pawn_square in pawns:
            pawn_file = chess.square_file(pawn_square)
            pawn_rank = chess.square_rank(pawn_square)
            
            # Check if pawn is advancing toward enemy king
            if abs(pawn_file - enemy_king_file) <= 1:  # Adjacent or same file
                if color and pawn_rank >= 4:  # White pawn advanced
                    score += pawn_storm_bonus
                elif not color and pawn_rank <= 3:  # Black pawn advanced
                    score += pawn_storm_bonus
        
        return score
    
    # Helper methods for bitboard pawn analysis
    
    def _is_passed_pawn_bitboard(self, board: chess.Board, pawn_square: int, color: bool) -> bool:
        """Check if pawn is passed using bitboard operations"""
        pawn_file = chess.square_file(pawn_square)
        pawn_rank = chess.square_rank(pawn_square)
        
        # Get enemy pawns
        enemy_pawns = board.pieces(chess.PAWN, not color)
        
        # Check files that could block this pawn (same file + adjacent files)
        blocking_files = [pawn_file]
        if pawn_file > 0:
            blocking_files.append(pawn_file - 1)
        if pawn_file < 7:
            blocking_files.append(pawn_file + 1)
        
        # Check if any enemy pawns can block
        for file_idx in blocking_files:
            file_mask = self.FILE_A << file_idx
            enemy_pawns_on_file = enemy_pawns & file_mask
            
            if enemy_pawns_on_file:
                for enemy_square in enemy_pawns_on_file:
                    enemy_rank = chess.square_rank(enemy_square)
                    
                    # Check if enemy pawn is ahead of our pawn
                    if color and enemy_rank > pawn_rank:  # White pawn
                        return False
                    elif not color and enemy_rank < pawn_rank:  # Black pawn
                        return False
        
        return True
    
    def _is_isolated_pawn_bitboard(self, board: chess.Board, pawn_square: int, color: bool) -> bool:
        """Check if pawn is isolated using bitboard operations"""
        pawn_file = chess.square_file(pawn_square)
        our_pawns = board.pieces(chess.PAWN, color)
        
        # Check adjacent files for friendly pawns
        adjacent_files = []
        if pawn_file > 0:
            adjacent_files.append(pawn_file - 1)
        if pawn_file < 7:
            adjacent_files.append(pawn_file + 1)
        
        for file_idx in adjacent_files:
            file_mask = self.FILE_A << file_idx
            if our_pawns & file_mask:
                return False
        
        return True
    
    def _is_backward_pawn_bitboard(self, board: chess.Board, pawn_square: int, color: bool) -> bool:
        """Check if pawn is backward using bitboard operations"""
        # Simplified backward pawn detection
        pawn_file = chess.square_file(pawn_square)
        pawn_rank = chess.square_rank(pawn_square)
        our_pawns = board.pieces(chess.PAWN, color)
        
        # Check if adjacent pawns are ahead
        adjacent_files = [pawn_file - 1, pawn_file + 1]
        
        for file_idx in adjacent_files:
            if 0 <= file_idx <= 7:
                file_mask = self.FILE_A << file_idx
                adjacent_pawns = our_pawns & file_mask
                
                for adj_square in adjacent_pawns:
                    adj_rank = chess.square_rank(adj_square)
                    
                    # If adjacent pawn is ahead and we can't advance safely
                    if (color and adj_rank > pawn_rank) or (not color and adj_rank < pawn_rank):
                        return True
        
        return False
    
    def _has_pawn_support_bitboard(self, board: chess.Board, pawn_square: int, color: bool) -> bool:
        """Check if pawn has support using bitboard operations"""
        pawn_file = chess.square_file(pawn_square)
        pawn_rank = chess.square_rank(pawn_square)
        our_pawns = board.pieces(chess.PAWN, color)
        
        # Check diagonal squares behind for supporting pawns
        support_squares = []
        if color:  # White
            if pawn_rank > 0:
                if pawn_file > 0:
                    support_squares.append(chess.square(pawn_file - 1, pawn_rank - 1))
                if pawn_file < 7:
                    support_squares.append(chess.square(pawn_file + 1, pawn_rank - 1))
        else:  # Black
            if pawn_rank < 7:
                if pawn_file > 0:
                    support_squares.append(chess.square(pawn_file - 1, pawn_rank + 1))
                if pawn_file < 7:
                    support_squares.append(chess.square(pawn_file + 1, pawn_rank + 1))
        
        for support_square in support_squares:
            if support_square in our_pawns:
                return True
        
        return False
    
    def _find_pawn_chains_bitboard(self, board: chess.Board, pawns: chess.SquareSet, color: bool) -> List[int]:
        """Find pawn chains using bitboard operations"""
        # Simplified chain detection - count connected groups
        chains = []
        visited = set()
        
        for pawn_square in pawns:
            if pawn_square not in visited:
                chain_length = self._count_chain_length_bitboard(board, pawn_square, color, visited, pawns)
                if chain_length > 0:
                    chains.append(chain_length)
        
        return chains
    
    def _count_chain_length_bitboard(self, board: chess.Board, start_square: int, color: bool, visited: set, pawns: chess.SquareSet) -> int:
        """Count chain length recursively using bitboard operations"""
        if start_square in visited or start_square not in pawns:
            return 0
        
        visited.add(start_square)
        length = 1
        
        # Check connected pawns
        pawn_file = chess.square_file(start_square)
        pawn_rank = chess.square_rank(start_square)
        
        # Check diagonal connections
        connections = []
        if color:  # White
            if pawn_rank < 7:
                if pawn_file > 0:
                    connections.append(chess.square(pawn_file - 1, pawn_rank + 1))
                if pawn_file < 7:
                    connections.append(chess.square(pawn_file + 1, pawn_rank + 1))
        else:  # Black
            if pawn_rank > 0:
                if pawn_file > 0:
                    connections.append(chess.square(pawn_file - 1, pawn_rank - 1))
                if pawn_file < 7:
                    connections.append(chess.square(pawn_file + 1, pawn_rank - 1))
        
        for connected_square in connections:
            length += self._count_chain_length_bitboard(board, connected_square, color, visited, pawns)
        
        return length
    
    def _has_connected_passed_pawn_bitboard(self, board: chess.Board, pawn_square: int, color: bool) -> bool:
        """Check if passed pawn has connected passed pawn using bitboard operations"""
        pawn_file = chess.square_file(pawn_square)
        adjacent_files = [pawn_file - 1, pawn_file + 1]
        
        for file_idx in adjacent_files:
            if 0 <= file_idx <= 7:
                file_mask = self.FILE_A << file_idx
                our_pawns = board.pieces(chess.PAWN, color)
                adjacent_pawns = our_pawns & file_mask
                
                for adj_square in adjacent_pawns:
                    if self._is_passed_pawn_bitboard(board, adj_square, color):
                        return True
        
        return False
    
    def _is_on_open_file_bitboard(self, board: chess.Board, pawn_square: int) -> bool:
        """Check if pawn is on open file using bitboard operations"""
        pawn_file = chess.square_file(pawn_square)
        file_mask = self.FILE_A << pawn_file
        
        # Check if any other pawns exist on this file
        all_pawns = board.pieces(chess.PAWN, chess.WHITE) | board.pieces(chess.PAWN, chess.BLACK)
        other_pawns = all_pawns & file_mask
        
        return len(other_pawns) <= 1  # Only our pawn on the file
    
    def _is_on_semi_open_file_bitboard(self, board: chess.Board, pawn_square: int, color: bool) -> bool:
        """Check if pawn is on semi-open file using bitboard operations"""
        pawn_file = chess.square_file(pawn_square)
        file_mask = self.FILE_A << pawn_file
        
        # Check if enemy has no pawns on this file
        enemy_pawns = board.pieces(chess.PAWN, not color)
        enemy_pawns_on_file = enemy_pawns & file_mask
        
        return len(enemy_pawns_on_file) == 0

    def evaluate_king_safety(self, board: chess.Board, color: bool) -> float:
        """
        V18.1 ENHANCED: Comprehensive king safety evaluation with tuned penalties
        Returns score from the perspective of the given color
        """
        total_score = 0.0
        
        king_square = board.king(color)
        if king_square is None:
            return -1000.0  # King missing - critical error
        
        # Determine game phase for king safety vs activity balance
        material_count = self._count_material_bitboard(board)
        is_endgame = material_count < 2000  # Rough endgame threshold
        
        if is_endgame:
            # Endgame: King activity is important
            total_score += self._evaluate_king_activity_bitboard(board, king_square, color)
        else:
            # Opening/Middlegame: King safety is paramount
            total_score += self._evaluate_pawn_shelter_bitboard(board, king_square, color)
            total_score += self._evaluate_castling_rights_bitboard(board, color)
            total_score += self._evaluate_king_exposure_bitboard(board, king_square, color)
            total_score += self._evaluate_escape_squares_bitboard(board, king_square, color)
            total_score += self._evaluate_attack_zone_bitboard(board, king_square, color)
            total_score += self._evaluate_enemy_pawn_storms_bitboard(board, king_square, color)
            
            # V18.1: HIGH-VALUE ATTACKER PENALTY
            king_zone = self._get_king_zone_squares(king_square)
            high_value_attackers = 0
            total_attackers = 0
            
            for square in king_zone:
                attackers = board.attackers(not color, square)
                total_attackers += len(attackers)
                
                for attacker_sq in attackers:
                    piece = board.piece_at(attacker_sq)
                    if piece and piece.piece_type in [chess.QUEEN, chess.ROOK]:
                        high_value_attackers += 1
            
            # Escalating danger penalties
            if total_attackers > 3:
                total_score -= 50 * (total_attackers - 3)
            
            if high_value_attackers > 0:
                total_score -= 100 * high_value_attackers
            
            # V18.1: CENTER KING PENALTY (MIDDLEGAME)
            king_file = chess.square_file(king_square)
            
            # Penalty for being in center files (d, e = files 3, 4)
            if king_file in [3, 4]:
                total_score -= 30
            
            # Severe penalty if unmoved (no castling) and in center
            if not self._has_castled(board, color) and king_file in [3, 4]:
                total_score -= 80
        
        # V18.1: BISHOP PAIR BONUS (applies in all phases)
        total_score += self._evaluate_bishop_pair(board, color)
        
        return total_score
    
    def _count_material_bitboard(self, board: chess.Board) -> int:
        """Count total material on board using bitboard operations"""
        material = 0
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            white_pieces = board.pieces(piece_type, chess.WHITE)
            black_pieces = board.pieces(piece_type, chess.BLACK)
            piece_value = [100, 320, 330, 500, 900][piece_type - 1]
            material += (len(white_pieces) + len(black_pieces)) * piece_value
        return material
    
    def _evaluate_pawn_shelter_bitboard(self, board: chess.Board, king_square: int, color: bool) -> float:
        """Evaluate pawn shelter around the king using bitboards"""
        score = 0.0
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        friendly_pawns = board.pieces(chess.PAWN, color)
        shelter_pawns = 0
        pawn_shelter_bonus = [0, 5, 10, 15, 20]  # By number of shelter pawns
        
        # Check files around the king (king file and adjacent files)
        for file_offset in [-1, 0, 1]:
            check_file = king_file + file_offset
            if 0 <= check_file <= 7:
                file_mask = self.FILE_A << check_file
                pawns_on_file = friendly_pawns & file_mask
                
                # Look for pawn shelter in front of king
                shelter_found = False
                for pawn_square in pawns_on_file:
                    pawn_rank = chess.square_rank(pawn_square)
                    
                    # Check if pawn is providing shelter
                    if color and pawn_rank > king_rank:  # White king
                        if pawn_rank - king_rank <= 2:
                            shelter_pawns += 1
                            shelter_found = True
                        break
                    elif not color and pawn_rank < king_rank:  # Black king
                        if king_rank - pawn_rank <= 2:
                            shelter_pawns += 1
                            shelter_found = True
                        break
                
                # Penalty for missing pawn shelter
                if not shelter_found:
                    score -= 10
        
        # Bonus for pawn shelter
        if shelter_pawns < len(pawn_shelter_bonus):
            score += pawn_shelter_bonus[shelter_pawns]
        else:
            score += pawn_shelter_bonus[-1]
        
        return score
    
    def _evaluate_castling_rights_bitboard(self, board: chess.Board, color: bool) -> float:
        """Evaluate castling rights value using bitboard operations"""
        score = 0.0
        castling_rights_bonus = 25
        
        if color:  # White
            if board.has_kingside_castling_rights(chess.WHITE):
                score += castling_rights_bonus
            if board.has_queenside_castling_rights(chess.WHITE):
                score += castling_rights_bonus * 0.8  # Queenside slightly less valuable
        else:  # Black
            if board.has_kingside_castling_rights(chess.BLACK):
                score += castling_rights_bonus
            if board.has_queenside_castling_rights(chess.BLACK):
                score += castling_rights_bonus * 0.8
        
        return score
    
    def _evaluate_king_exposure_bitboard(self, board: chess.Board, king_square: int, color: bool) -> float:
        """Evaluate king exposure to enemy attacks using bitboards"""
        score = 0.0
        king_exposure_penalty = 30
        
        # Check if king is on an open file or rank
        if self._is_on_open_file_bitboard(board, king_square):
            score -= king_exposure_penalty
        
        if self._is_on_open_rank_bitboard(board, king_square):
            score -= king_exposure_penalty * 0.5  # Less dangerous than open file
        
        # Check for enemy pieces attacking king vicinity
        enemy_attacks = self._count_enemy_attacks_near_king_bitboard(board, king_square, color)
        score -= enemy_attacks * 5
        
        return score
    
    def _evaluate_escape_squares_bitboard(self, board: chess.Board, king_square: int, color: bool) -> float:
        """Evaluate available escape squares for the king using bitboards"""
        score = 0.0
        escape_squares = 0
        escape_square_bonus = 8
        
        # Check all adjacent squares
        for rank_offset in [-1, 0, 1]:
            for file_offset in [-1, 0, 1]:
                if rank_offset == 0 and file_offset == 0:
                    continue  # Skip king's current square
                
                target_square = king_square + rank_offset * 8 + file_offset
                
                if 0 <= target_square <= 63:
                    # Check if square is safe and accessible
                    if self._is_safe_escape_square_bitboard(board, target_square, color):
                        escape_squares += 1
        
        score += escape_squares * escape_square_bonus
        
        # Penalty for having very few escape squares
        if escape_squares <= 1:
            score -= 20
        
        return score
    
    def _evaluate_attack_zone_bitboard(self, board: chess.Board, king_square: int, color: bool) -> float:
        """Evaluate enemy control of squares around the king using bitboards"""
        score = 0.0
        attack_zone_penalty = 12
        
        # Define attack zone (3x3 squares around king)
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        enemy_controlled = 0
        for rank_offset in [-1, 0, 1]:
            for file_offset in [-1, 0, 1]:
                target_file = king_file + file_offset
                target_rank = king_rank + rank_offset
                
                if 0 <= target_file <= 7 and 0 <= target_rank <= 7:
                    target_square = target_rank * 8 + target_file
                    if self._is_square_attacked_by_enemy_bitboard(board, target_square, color):
                        enemy_controlled += 1
        
        score -= enemy_controlled * attack_zone_penalty
        return score
    
    def _evaluate_enemy_pawn_storms_bitboard(self, board: chess.Board, king_square: int, color: bool) -> float:
        """Evaluate enemy pawn storms using bitboards"""
        score = 0.0
        enemy_pawn_storm_penalty = 15
        
        king_file = chess.square_file(king_square)
        enemy_pawns = board.pieces(chess.PAWN, not color)
        
        # Check for advancing enemy pawns near king
        for pawn_square in enemy_pawns:
            pawn_file = chess.square_file(pawn_square)
            pawn_rank = chess.square_rank(pawn_square)
            
            # Check if pawn is advancing toward our king
            if abs(pawn_file - king_file) <= 1:  # Adjacent or same file
                if not color and pawn_rank >= 4:  # Enemy white pawn advanced
                    score -= enemy_pawn_storm_penalty
                elif color and pawn_rank <= 3:  # Enemy black pawn advanced
                    score -= enemy_pawn_storm_penalty
        
        return score
    
    def _evaluate_king_activity_bitboard(self, board: chess.Board, king_square: int, color: bool) -> float:
        """V18.1 ENHANCED: King activity in endgame with enhanced centralization bonus"""
        score = 0.0
        king_activity_bonus = 5
        
        # King centralization in endgame
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        # Distance from center
        center_distance = max(abs(king_file - 3.5), abs(king_rank - 3.5))
        centralization_bonus = [12, 8, 4, 2][min(int(center_distance), 3)]
        
        score += centralization_bonus
        
        # V18.1: ENHANCED CENTRALIZATION BONUS
        # Additional bonus for being close to ideal center squares (d4, e4, d5, e5)
        center_file_dist = min(abs(king_file - 3), abs(king_file - 4))
        center_rank_dist = min(abs(king_rank - 3), abs(king_rank - 4))
        total_center_dist = center_file_dist + center_rank_dist
        
        # Bonus for centralization (max 70cp for perfect center, e.g., Ke4)
        enhanced_centralization = (4 - total_center_dist) * 10
        score += enhanced_centralization
        
        # King mobility in endgame
        mobility = 0
        for rank_offset in [-1, 0, 1]:
            for file_offset in [-1, 0, 1]:
                if rank_offset == 0 and file_offset == 0:
                    continue
                
                target_square = king_square + rank_offset * 8 + file_offset
                if 0 <= target_square <= 63:
                    target_piece = board.piece_at(target_square)
                    if target_piece is None or target_piece.color != color:
                        mobility += 1
        
        score += mobility * king_activity_bonus
        
        return score
    
    def _is_on_open_rank_bitboard(self, board: chess.Board, square: int) -> bool:
        """Check if square is on open rank using bitboards"""
        rank = chess.square_rank(square)
        rank_mask = self.RANK_1 << (rank * 8)
        
        all_pieces = board.occupied
        pieces_on_rank = all_pieces & rank_mask
        
        return bin(pieces_on_rank).count('1') <= 2  # Only kings on the rank
    
    def _count_enemy_attacks_near_king_bitboard(self, board: chess.Board, king_square: int, color: bool) -> int:
        """Count enemy attacks near king using bitboards"""
        attacks = 0
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        # Check 3x3 area around king
        for rank_offset in [-1, 0, 1]:
            for file_offset in [-1, 0, 1]:
                target_file = king_file + file_offset
                target_rank = king_rank + rank_offset
                
                if 0 <= target_file <= 7 and 0 <= target_rank <= 7:
                    target_square = target_rank * 8 + target_file
                    if self._is_square_attacked_by_enemy_bitboard(board, target_square, color):
                        attacks += 1
        
        return attacks
    
    def _is_safe_escape_square_bitboard(self, board: chess.Board, square: int, color: bool) -> bool:
        """Check if square is safe escape square using bitboards"""
        # Check if square is occupied by our piece
        piece = board.piece_at(square)
        if piece and piece.color == color:
            return False
        
        # Check if square is attacked by enemy
        if self._is_square_attacked_by_enemy_bitboard(board, square, color):
            return False
        
        return True
    
    def _is_square_attacked_by_enemy_bitboard(self, board: chess.Board, square: int, our_color: bool) -> bool:
        """Check if square is attacked by enemy using bitboards"""
        # This is a simplified version - full implementation would check all enemy piece attacks
        enemy_pieces = board.occupied_co[not our_color]
        
        # Quick check for pawn attacks
        enemy_pawns = board.pieces(chess.PAWN, not our_color)
        for pawn_square in enemy_pawns:
            if self._pawn_attacks_square_bitboard(pawn_square, square, not our_color):
                return True
        
        # Check for knight attacks
        enemy_knights = board.pieces(chess.KNIGHT, not our_color)
        for knight_square in enemy_knights:
            knight_attacks = self.KNIGHT_ATTACKS[knight_square]
            if knight_attacks & (1 << square):
                return True
        
        return False
    
    def _pawn_attacks_square_bitboard(self, pawn_square: int, target_square: int, pawn_color: bool) -> bool:
        """Check if pawn attacks target square using bitboards"""
        pawn_file = chess.square_file(pawn_square)
        pawn_rank = chess.square_rank(pawn_square)
        target_file = chess.square_file(target_square)
        target_rank = chess.square_rank(target_square)
        
        # Check diagonal pawn attacks
        if abs(pawn_file - target_file) == 1:
            if pawn_color and target_rank == pawn_rank + 1:  # White pawn
                return True
            elif not pawn_color and target_rank == pawn_rank - 1:  # Black pawn
                return True
        
        return False


    def _get_king_zone_squares(self, king_square: int) -> list:
        """V18.1: Get 3x3 zone around king for attack detection"""
        king_zone = []
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        for rank_offset in [-1, 0, 1]:
            for file_offset in [-1, 0, 1]:
                target_file = king_file + file_offset
                target_rank = king_rank + rank_offset
                
                if 0 <= target_file <= 7 and 0 <= target_rank <= 7:
                    target_square = target_rank * 8 + target_file
                    king_zone.append(target_square)
        
        return king_zone
    
    def _has_castled(self, board: chess.Board, color: bool) -> bool:
        """V18.1: Check if king has moved from starting square"""
        king_square = board.king(color)
        if king_square is None:
            return False
        starting_square = chess.E1 if color == chess.WHITE else chess.E8
        return king_square != starting_square
    
    def _evaluate_bishop_pair(self, board: chess.Board, color: bool) -> float:
        """V18.1: Evaluate bishop pair bonus"""
        bishops = board.pieces(chess.BISHOP, color)
        
        if len(bishops) >= 2:
            # Check if bishops are on different colored squares
            light_square_bishop = False
            dark_square_bishop = False
            
            for bishop_square in bishops:
                square_color = (chess.square_file(bishop_square) + 
                              chess.square_rank(bishop_square)) % 2
                if square_color == 0:
                    dark_square_bishop = True
                else:
                    light_square_bishop = True
            
            # Only bonus if on different colored squares
            if light_square_bishop and dark_square_bishop:
                # Bonus stronger in open/endgame positions
                piece_count = len(board.piece_map())
                if piece_count < 20:  # Open/endgame
                    return 50.0
                else:  # Closed position
                    return 30.0
        
        return 0.0


class V7P3RScoringCalculationBitboard:
    """
    Drop-in replacement for the slow scoring calculator
    Uses bitboards for ultra-high performance
    """
    
    def __init__(self, piece_values: Dict[int, int], enable_nudges: bool = False):
        self.piece_values = piece_values
        self.bitboard_evaluator = V7P3RBitboardEvaluator(piece_values, enable_nudges=enable_nudges)
    
    def calculate_score_optimized(self, board: chess.Board, color: chess.Color, endgame_factor: float = 0.0) -> float:
        """
        Ultra-fast evaluation using bitboards
        Target: 20,000+ NPS
        """
        return self.bitboard_evaluator.evaluate_bitboard(board, color)
    
    def detect_bitboard_tactics(self, board: chess.Board, move: chess.Move) -> float:
        """
        V12.6 CONSOLIDATED: Detect tactical patterns using bitboard operations
        Delegate to the bitboard evaluator for consistency
        """
        return self.bitboard_evaluator.detect_bitboard_tactics(board, move)
    
    def evaluate_pawn_structure(self, board: chess.Board, color: bool) -> float:
        """
        V12.6 CONSOLIDATED: Evaluate pawn structure using bitboard operations
        Delegate to the bitboard evaluator for consistency
        """
        return self.bitboard_evaluator.evaluate_pawn_structure(board, color)
    
    def evaluate_king_safety(self, board: chess.Board, color: bool) -> float:
        """
        V12.6 CONSOLIDATED: Evaluate king safety using bitboard operations
        Delegate to the bitboard evaluator for consistency
        """
        return self.bitboard_evaluator.evaluate_king_safety(board, color)



# ============================================================================
# V7P3R_EVAL_MODULES
# ============================================================================

"""
V7P3R Evaluation Module Registry

Metadata-driven evaluation components with selective activation.

This module defines ALL evaluation components from v18.2, each with:
- Cost: NEGLIGIBLE, LOW, MEDIUM, HIGH (node evaluation overhead)
- Criticality: ESSENTIAL, IMPORTANT, SITUATIONAL, OPTIONAL
- Required pieces: What must be on board for module to be relevant
- Required phases: When module should be active
- Dependencies: Other modules that must run first

Author: Pat Snyder
Created: 2025-12-26 (v18.2 Modular Evaluation System - Day 2)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Set, List, Callable, Optional
import chess



class EvaluationCost(Enum):
    """Computational cost per node evaluation"""
    NEGLIGIBLE = "negligible"  # < 0.1ms, O(1) lookups
    LOW = "low"                # 0.1-0.5ms, simple board scans
    MEDIUM = "medium"          # 0.5-2ms, move generation or complex logic
    HIGH = "high"              # > 2ms, heavy analysis (SEE, mobility, etc.)


class EvaluationCriticality(Enum):
    """How important module is to engine strength"""
    ESSENTIAL = "essential"        # Always needed (material, basic tactics)
    IMPORTANT = "important"        # Needed for competitive play (king safety, PST)
    SITUATIONAL = "situational"    # Helpful in specific positions (bishop pair, passed pawns)
    OPTIONAL = "optional"          # Nice-to-have (knight outposts, rook on 7th)


@dataclass
class EvaluationModule:
    """
    Metadata for a single evaluation component.
    
    Each module is a self-contained evaluation that can be toggled on/off
    based on position context.
    """
    name: str                           # Unique identifier
    description: str                    # Human-readable purpose
    cost: EvaluationCost               # Computational overhead
    criticality: EvaluationCriticality # Strategic importance
    
    # Activation conditions (when module is RELEVANT)
    required_pieces: Set[chess.PieceType] = None  # Must have these pieces
    required_phases: Set[GamePhase] = None        # Active in these phases
    skip_when_desperate: bool = False             # Skip when down material
    skip_in_time_pressure: bool = False           # Skip when < 30s
    
    # Dependencies
    depends_on: List[str] = None  # Other modules that must run first
    
    def __post_init__(self):
        """Initialize optional fields"""
        if self.required_pieces is None:
            self.required_pieces = set()
        if self.required_phases is None:
            self.required_phases = set()
        if self.depends_on is None:
            self.depends_on = []


# =============================================================================
# MODULE REGISTRY
# =============================================================================

MODULE_REGISTRY: List[EvaluationModule] = [
    
    # -------------------------------------------------------------------------
    # ESSENTIAL MODULES (Always needed)
    # -------------------------------------------------------------------------
    
    EvaluationModule(
        name="material_counter",
        description="Basic material counting (P=100, N=320, B=330, R=500, Q=900)",
        cost=EvaluationCost.NEGLIGIBLE,
        criticality=EvaluationCriticality.ESSENTIAL,
        required_phases=set(GamePhase),  # All phases
    ),
    
    EvaluationModule(
        name="piece_square_tables",
        description="Positional bonuses for piece placement (PST)",
        cost=EvaluationCost.NEGLIGIBLE,
        criticality=EvaluationCriticality.ESSENTIAL,
        required_phases=set(GamePhase),
    ),
    
    # -------------------------------------------------------------------------
    # DESPERATE MODULES (Only when down material - TACTICAL RECOVERY)
    # -------------------------------------------------------------------------
    
    EvaluationModule(
        name="hanging_pieces",
        description="Detect undefended pieces (captures without recapture)",
        cost=EvaluationCost.MEDIUM,
        criticality=EvaluationCriticality.ESSENTIAL,
        skip_when_desperate=False,  # KEEP when desperate!
    ),
    
    EvaluationModule(
        name="capture_priority",
        description="Prioritize recaptures and material-winning captures",
        cost=EvaluationCost.LOW,
        criticality=EvaluationCriticality.ESSENTIAL,
        skip_when_desperate=False,
    ),
    
    EvaluationModule(
        name="check_threats",
        description="Evaluate check-giving moves and mate threats",
        cost=EvaluationCost.MEDIUM,
        criticality=EvaluationCriticality.IMPORTANT,
        skip_when_desperate=False,
    ),
    
    EvaluationModule(
        name="pins_forks_skewers",
        description="Tactical pattern detection (pins, forks, discovered attacks)",
        cost=EvaluationCost.MEDIUM,
        criticality=EvaluationCriticality.IMPORTANT,
        skip_when_desperate=False,
    ),
    
    # -------------------------------------------------------------------------
    # KING SAFETY (Critical in middlegame)
    # -------------------------------------------------------------------------
    
    EvaluationModule(
        name="king_safety_basic",
        description="Pawn shield and basic king exposure",
        cost=EvaluationCost.LOW,
        criticality=EvaluationCriticality.ESSENTIAL,
        required_phases={GamePhase.OPENING, GamePhase.MIDDLEGAME_COMPLEX, GamePhase.MIDDLEGAME_SIMPLE},
        skip_when_desperate=True,  # Skip if down material
    ),
    
    EvaluationModule(
        name="king_safety_complex",
        description="Attack patterns, tropism, storm detection",
        cost=EvaluationCost.HIGH,
        criticality=EvaluationCriticality.IMPORTANT,
        required_phases={GamePhase.MIDDLEGAME_COMPLEX},
        skip_when_desperate=True,
        skip_in_time_pressure=True,
    ),
    
    EvaluationModule(
        name="king_centralization",
        description="King activity bonus in endgame",
        cost=EvaluationCost.LOW,
        criticality=EvaluationCriticality.IMPORTANT,
        required_phases={GamePhase.ENDGAME_COMPLEX, GamePhase.ENDGAME_SIMPLE},
    ),
    
    # -------------------------------------------------------------------------
    # PAWN STRUCTURE
    # -------------------------------------------------------------------------
    
    EvaluationModule(
        name="passed_pawns",
        description="Passed pawn bonuses (distance to promotion, king proximity)",
        cost=EvaluationCost.MEDIUM,
        criticality=EvaluationCriticality.IMPORTANT,
        required_pieces={chess.PAWN},
        skip_when_desperate=True,
    ),
    
    EvaluationModule(
        name="doubled_pawns",
        description="Penalty for doubled/tripled pawns",
        cost=EvaluationCost.LOW,
        criticality=EvaluationCriticality.SITUATIONAL,
        required_pieces={chess.PAWN},
        skip_when_desperate=True,
        skip_in_time_pressure=True,
    ),
    
    EvaluationModule(
        name="isolated_pawns",
        description="Penalty for isolated pawns (no friendly pawns on adjacent files)",
        cost=EvaluationCost.LOW,
        criticality=EvaluationCriticality.SITUATIONAL,
        required_pieces={chess.PAWN},
        skip_when_desperate=True,
        skip_in_time_pressure=True,
    ),
    
    EvaluationModule(
        name="backward_pawns",
        description="Penalty for backward pawns (cannot advance safely)",
        cost=EvaluationCost.MEDIUM,
        criticality=EvaluationCriticality.OPTIONAL,
        required_pieces={chess.PAWN},
        skip_when_desperate=True,
        skip_in_time_pressure=True,
    ),
    
    EvaluationModule(
        name="pawn_chains",
        description="Bonus for connected pawn chains",
        cost=EvaluationCost.LOW,
        criticality=EvaluationCriticality.SITUATIONAL,
        required_pieces={chess.PAWN},
        skip_when_desperate=True,
        skip_in_time_pressure=True,
    ),
    
    # -------------------------------------------------------------------------
    # PIECE-SPECIFIC EVALUATIONS
    # -------------------------------------------------------------------------
    
    EvaluationModule(
        name="bishop_pair",
        description="Bonus for having both bishops (powerful in open positions)",
        cost=EvaluationCost.NEGLIGIBLE,
        criticality=EvaluationCriticality.SITUATIONAL,
        required_pieces={chess.BISHOP},
        skip_when_desperate=True,
    ),
    
    EvaluationModule(
        name="knight_outposts",
        description="Bonus for knights on strong outpost squares",
        cost=EvaluationCost.LOW,
        criticality=EvaluationCriticality.OPTIONAL,
        required_pieces={chess.KNIGHT},
        required_phases={GamePhase.MIDDLEGAME_COMPLEX, GamePhase.MIDDLEGAME_SIMPLE},
        skip_when_desperate=True,
        skip_in_time_pressure=True,
    ),
    
    EvaluationModule(
        name="rook_on_7th",
        description="Bonus for rook on 7th rank (attacking enemy pawns)",
        cost=EvaluationCost.LOW,
        criticality=EvaluationCriticality.SITUATIONAL,
        required_pieces={chess.ROOK},
        skip_when_desperate=True,
    ),
    
    EvaluationModule(
        name="rook_on_open_file",
        description="Bonus for rook on open/semi-open file",
        cost=EvaluationCost.LOW,
        criticality=EvaluationCriticality.SITUATIONAL,
        required_pieces={chess.ROOK},
        skip_when_desperate=True,
        skip_in_time_pressure=True,
    ),
    
    EvaluationModule(
        name="queen_mobility",
        description="Queen activity and mobility evaluation",
        cost=EvaluationCost.MEDIUM,
        criticality=EvaluationCriticality.IMPORTANT,
        required_pieces={chess.QUEEN},
        required_phases={GamePhase.MIDDLEGAME_COMPLEX, GamePhase.MIDDLEGAME_SIMPLE},
        skip_when_desperate=True,
    ),
    
    # -------------------------------------------------------------------------
    # MOBILITY & ACTIVITY
    # -------------------------------------------------------------------------
    
    EvaluationModule(
        name="piece_mobility",
        description="Count legal moves for all pieces (slow, accurate)",
        cost=EvaluationCost.HIGH,
        criticality=EvaluationCriticality.IMPORTANT,
        required_phases={GamePhase.MIDDLEGAME_COMPLEX},
        skip_when_desperate=True,
        skip_in_time_pressure=True,
    ),
    
    EvaluationModule(
        name="piece_activity",
        description="Simplified mobility (attacked squares, no move gen)",
        cost=EvaluationCost.MEDIUM,
        criticality=EvaluationCriticality.SITUATIONAL,
        skip_when_desperate=True,
    ),
    
    # -------------------------------------------------------------------------
    # POSITIONAL CONCEPTS
    # -------------------------------------------------------------------------
    
    EvaluationModule(
        name="center_control",
        description="Control of central squares (e4, d4, e5, d5)",
        cost=EvaluationCost.LOW,
        criticality=EvaluationCriticality.IMPORTANT,
        required_phases={GamePhase.OPENING, GamePhase.MIDDLEGAME_COMPLEX},
        skip_when_desperate=True,
    ),
    
    EvaluationModule(
        name="space_advantage",
        description="Territorial control (squares controlled in opponent's half)",
        cost=EvaluationCost.MEDIUM,
        criticality=EvaluationCriticality.OPTIONAL,
        required_phases={GamePhase.MIDDLEGAME_COMPLEX},
        skip_when_desperate=True,
        skip_in_time_pressure=True,
    ),
    
    EvaluationModule(
        name="development",
        description="Piece development bonus (pieces off back rank)",
        cost=EvaluationCost.LOW,
        criticality=EvaluationCriticality.IMPORTANT,
        required_phases={GamePhase.OPENING},
        skip_when_desperate=True,
    ),
    
    # -------------------------------------------------------------------------
    # ENDGAME SPECIALIZATIONS
    # -------------------------------------------------------------------------
    
    EvaluationModule(
        name="opposition",
        description="King opposition in pawn endgames",
        cost=EvaluationCost.LOW,
        criticality=EvaluationCriticality.IMPORTANT,
        required_phases={GamePhase.ENDGAME_SIMPLE},
        required_pieces={chess.PAWN},
    ),
    
    EvaluationModule(
        name="square_of_pawn",
        description="Can king catch passed pawn? (rule of square)",
        cost=EvaluationCost.LOW,
        criticality=EvaluationCriticality.IMPORTANT,
        required_phases={GamePhase.ENDGAME_SIMPLE, GamePhase.ENDGAME_COMPLEX},
        required_pieces={chess.PAWN},
    ),
    
    EvaluationModule(
        name="endgame_tables",
        description="Theoretical endgame knowledge (KQ vs K, KR vs K, etc.)",
        cost=EvaluationCost.LOW,
        criticality=EvaluationCriticality.ESSENTIAL,
        required_phases={GamePhase.ENDGAME_SIMPLE},
    ),
    
    # -------------------------------------------------------------------------
    # ADVANCED TACTICAL
    # -------------------------------------------------------------------------
    
    EvaluationModule(
        name="see_evaluation",
        description="Static Exchange Evaluation (capture sequences)",
        cost=EvaluationCost.HIGH,
        criticality=EvaluationCriticality.IMPORTANT,
        skip_when_desperate=False,  # Keep for tactical recovery
        skip_in_time_pressure=True,
    ),
    
    EvaluationModule(
        name="trapped_pieces",
        description="Detect pieces with no escape squares",
        cost=EvaluationCost.MEDIUM,
        criticality=EvaluationCriticality.SITUATIONAL,
        skip_when_desperate=False,  # Important for tactics
    ),
    
    EvaluationModule(
        name="back_rank_threats",
        description="Back rank mate detection and prevention",
        cost=EvaluationCost.LOW,
        criticality=EvaluationCriticality.IMPORTANT,
        required_pieces={chess.ROOK, chess.QUEEN},
        skip_when_desperate=False,
    ),
    
    # -------------------------------------------------------------------------
    # SAFETY & STABILITY
    # -------------------------------------------------------------------------
    
    EvaluationModule(
        name="move_safety_checker",
        description="Pre-move validation (hanging pieces, legality, repetition)",
        cost=EvaluationCost.MEDIUM,
        criticality=EvaluationCriticality.ESSENTIAL,
        skip_when_desperate=False,  # Always check safety
    ),
    
    EvaluationModule(
        name="repetition_detector",
        description="Avoid threefold repetition unless desperate",
        cost=EvaluationCost.LOW,
        criticality=EvaluationCriticality.ESSENTIAL,
    ),
]


# =============================================================================
# MODULE SELECTION HELPERS
# =============================================================================

def get_module(name: str) -> Optional[EvaluationModule]:
    """Get module by name from registry"""
    for module in MODULE_REGISTRY:
        if module.name == name:
            return module
    return None


def is_module_relevant(module: EvaluationModule, context: PositionContext) -> bool:
    """
    Determine if module should be active for this position.
    
    Args:
        module: Module to check
        context: Current position context
        
    Returns:
        True if module is relevant, False to skip
    """
    # Check piece requirements
    if module.required_pieces:
        if not module.required_pieces.intersection(context.piece_types):
            return False  # Required pieces not on board
    
    # Check phase requirements
    if module.required_phases:
        if context.game_phase not in module.required_phases:
            return False  # Not in required phase
    
    # DESPERATE MODE: Skip non-tactical modules when down material
    if module.skip_when_desperate:
        if context.material_diff_cp < -300:  # Down 3+ pawns
            return False  # Skip strategic evaluations, focus on tactics
    
    # TIME PRESSURE: Skip expensive modules
    if module.skip_in_time_pressure:
        if context.time_pressure or context.use_fast_profile:
            return False  # Too slow for time pressure
    
    return True


def get_active_modules(context: PositionContext) -> List[EvaluationModule]:
    """
    Get all relevant modules for this position context.
    
    Returns modules in dependency order (dependencies first).
    
    Args:
        context: Current position context
        
    Returns:
        List of active modules sorted by dependencies
    """
    active = [m for m in MODULE_REGISTRY if is_module_relevant(m, context)]
    
    # TODO: Sort by dependencies (topological sort)
    # For now, just return in registry order
    
    return active


def get_desperate_modules() -> List[EvaluationModule]:
    """
    Get minimal tactical module set for desperate positions.
    
    When down significant material, ONLY evaluate:
    - Material counting
    - Tactical opportunities (captures, checks, threats)
    - Safety checks
    
    Skip ALL strategic evaluations (pawn structure, mobility, etc.)
    """
    desperate_names = [
        "material_counter",
        "hanging_pieces",
        "capture_priority",
        "check_threats",
        "pins_forks_skewers",
        "see_evaluation",
        "trapped_pieces",
        "back_rank_threats",
        "move_safety_checker",
        "repetition_detector",
    ]
    
    return [get_module(name) for name in desperate_names if get_module(name)]


def get_emergency_modules() -> List[EvaluationModule]:
    """
    Get minimal module set for time pressure (< 30s or < 5s per move).
    
    Only essential evaluations, skip all expensive computations.
    """
    emergency_names = [
        "material_counter",
        "piece_square_tables",
        "king_safety_basic",
        "hanging_pieces",
        "move_safety_checker",
    ]
    
    return [get_module(name) for name in emergency_names if get_module(name)]


# =============================================================================
# STATISTICS & DEBUGGING
# =============================================================================

def print_module_summary():
    """Print registry statistics"""
    print(f"\n=== V7P3R Evaluation Module Registry ===")
    print(f"Total modules: {len(MODULE_REGISTRY)}")
    
    by_criticality = {}
    by_cost = {}
    
    for module in MODULE_REGISTRY:
        # Count by criticality
        crit = module.criticality.value
        by_criticality[crit] = by_criticality.get(crit, 0) + 1
        
        # Count by cost
        cost = module.cost.value
        by_cost[cost] = by_cost.get(cost, 0) + 1
    
    print(f"\nBy Criticality:")
    for crit, count in sorted(by_criticality.items()):
        print(f"  {crit}: {count}")
    
    print(f"\nBy Cost:")
    for cost, count in sorted(by_cost.items()):
        print(f"  {cost}: {count}")
    
    print(f"\nDesperate Profile: {len(get_desperate_modules())} modules")
    print(f"Emergency Profile: {len(get_emergency_modules())} modules")


if __name__ == "__main__":
    # Test module registry
    print_module_summary()



# ============================================================================
# V7P3R_EVAL_SELECTOR
# ============================================================================

"""
V7P3R Evaluation Profile Selector

Smart profile builder that selects evaluation modules based on position context.

This is the "brain" of the modular evaluation system - it decides which
evaluations to run based on time pressure, material balance, game phase,
and tactical considerations.

Author: Pat Snyder
Created: 2025-12-26 (v18.2 Modular Evaluation System - Day 3)
"""

from typing import List, Set
from dataclasses import dataclass
import chess



class EvaluationProfile:
    """Named evaluation profile with module list"""
    
    DESPERATE = "DESPERATE"         # Down material, tactics only
    EMERGENCY = "EMERGENCY"         # Time pressure, essentials only
    FAST = "FAST"                   # Fast time control, skip expensive
    TACTICAL = "TACTICAL"           # Tactical position, emphasize tactics
    ENDGAME = "ENDGAME"            # Endgame, focus on technique
    COMPREHENSIVE = "COMPREHENSIVE" # Full evaluation


@dataclass
class SelectedProfile:
    """
    Profile selection result with reasoning.
    
    This is what gets returned to the search engine.
    """
    name: str                          # Profile name (DESPERATE, EMERGENCY, etc.)
    modules: List[EvaluationModule]    # Active modules for this profile
    module_count: int                  # Number of active modules
    reason: str                        # Why this profile was chosen
    estimated_cost_ms: float           # Estimated evaluation time per node
    
    @property
    def active_modules(self) -> List[str]:
        """Return list of active module names (for compatibility with ModularEvaluator)"""
        return [m.name for m in self.modules]


class EvaluationProfileSelector:
    """
    Selects optimal evaluation profile based on position context.
    
    Priority order:
    1. DESPERATE: Down 300+ cp (tactics only, recover material)
    2. EMERGENCY: Time pressure < 30s (essentials only)
    3. FAST: Fast time control < 5s/move (skip expensive modules)
    4. TACTICAL: High tactical activity (emphasize tactics)
    5. ENDGAME: Endgame phase (technique focus)
    6. COMPREHENSIVE: Default full evaluation
    
    Each profile filters MODULE_REGISTRY based on context.
    """
    
    def select_profile(self, context: PositionContext) -> SelectedProfile:
        """
        Main entry point: Select evaluation profile for this position.
        
        Args:
            context: Position context from PositionContextCalculator
            
        Returns:
            SelectedProfile with modules and reasoning
        """
        # PRIORITY 1: DESPERATE (down material - tactical recovery)
        if context.material_diff_cp < -300:  # Down 3+ pawns
            return self._build_desperate_profile(context)
        
        # PRIORITY 2: EMERGENCY (critical time pressure)
        if context.time_pressure:  # < 3s for this move - must move NOW
            return self._build_emergency_profile(context)
        
        # PRIORITY 3: FAST (fast time control or low time)
        if context.use_fast_profile:  # < 2s per move average - skip expensive modules
            return self._build_fast_profile(context)
        
        # PRIORITY 4: TACTICAL (high tactical activity)
        if self._is_tactical_position(context):
            return self._build_tactical_profile(context)
        
        # PRIORITY 5: ENDGAME (endgame technique)
        if context.game_phase in {GamePhase.ENDGAME_SIMPLE, GamePhase.ENDGAME_COMPLEX}:
            return self._build_endgame_profile(context)
        
        # DEFAULT: COMPREHENSIVE (full evaluation)
        return self._build_comprehensive_profile(context)
    
    def _build_desperate_profile(self, context: PositionContext) -> SelectedProfile:
        """
        DESPERATE profile: Down significant material, need tactical recovery.
        
        Strategy:
        - ONLY tactical modules (captures, checks, threats)
        - Skip ALL strategic evaluations
        - Goal: Find forcing moves to regain material
        
        Modules: 10 (material, hanging, captures, checks, pins, SEE, traps, safety)
        """
        modules = get_desperate_modules()
        
        # Filter by context (some modules may not be relevant)
        modules = [m for m in modules if is_module_relevant(m, context)]
        
        material_deficit = abs(context.material_diff_cp)
        
        return SelectedProfile(
            name=EvaluationProfile.DESPERATE,
            modules=modules,
            module_count=len(modules),
            reason=f"Down {material_deficit}cp - tactical recovery mode",
            estimated_cost_ms=self._estimate_cost(modules)
        )
    
    def _build_emergency_profile(self, context: PositionContext) -> SelectedProfile:
        """
        EMERGENCY profile: Critical time pressure (<3s for this move).
        
        Strategy:
        - Absolute essentials only
        - No move generation, no expensive checks
        - Fast enough to avoid time forfeit
        
        Modules: 5 (material, PST, basic safety, hanging, move safety)
        """
        modules = get_emergency_modules()
        
        return SelectedProfile(
            name=EvaluationProfile.EMERGENCY,
            modules=modules,
            module_count=len(modules),
            reason=f"Time pressure ({context.time_remaining:.1f}s remaining)",
            estimated_cost_ms=self._estimate_cost(modules)
        )
    
    def _build_fast_profile(self, context: PositionContext) -> SelectedProfile:
        """
        FAST profile: Fast time control (<2s per move average).
        
        Strategy:
        - Include important evaluations
        - Skip HIGH cost modules (SEE, full mobility)
        - Target: 4.0+ depth in blitz
        
        Modules: 12-18 (essentials + important, skip expensive)
        """
        # Get all relevant modules
        all_modules = [m for m in MODULE_REGISTRY if is_module_relevant(m, context)]
        
        # Filter: Skip HIGH cost modules
        modules = [m for m in all_modules if m.cost != EvaluationCost.HIGH]
        
        return SelectedProfile(
            name=EvaluationProfile.FAST,
            modules=modules,
            module_count=len(modules),
            reason=f"Fast time control ({context.time_per_move:.1f}s/move)",
            estimated_cost_ms=self._estimate_cost(modules)
        )
    
    def _build_tactical_profile(self, context: PositionContext) -> SelectedProfile:
        """
        TACTICAL profile: High tactical activity, emphasize tactics.
        
        Strategy:
        - Include all tactical modules
        - Include strategic if relevant
        - Emphasize king safety, hanging pieces, threats
        
        Modules: 18-22 (tactical focus + strategic context)
        """
        # Get all relevant modules
        all_modules = [m for m in MODULE_REGISTRY if is_module_relevant(m, context)]
        
        # Ensure tactical modules are included
        tactical_names = [
            "hanging_pieces", "check_threats", "pins_forks_skewers",
            "see_evaluation", "trapped_pieces", "back_rank_threats",
            "king_safety_complex", "capture_priority"
        ]
        
        # Add tactical modules that aren't already included
        for name in tactical_names:
            module = get_module(name)
            if module and is_module_relevant(module, context):
                if module not in all_modules:
                    all_modules.append(module)
        
        tactical_flags = ", ".join([f.value for f in context.tactical_flags])
        
        return SelectedProfile(
            name=EvaluationProfile.TACTICAL,
            modules=all_modules,
            module_count=len(all_modules),
            reason=f"Tactical position ({tactical_flags or 'multiple threats'})",
            estimated_cost_ms=self._estimate_cost(all_modules)
        )
    
    def _build_endgame_profile(self, context: PositionContext) -> SelectedProfile:
        """
        ENDGAME profile: Endgame technique and precision.
        
        Strategy:
        - Endgame-specific modules (king centralization, opposition, square of pawn)
        - Skip opening/middlegame modules (development, center control)
        - Emphasize king activity and pawn races
        
        Modules: 10-15 (endgame technique focus)
        """
        # Get all relevant modules (phase filtering already applied)
        modules = [m for m in MODULE_REGISTRY if is_module_relevant(m, context)]
        
        # Ensure endgame modules are prioritized
        endgame_names = [
            "king_centralization", "opposition", "square_of_pawn", 
            "endgame_tables", "passed_pawns"
        ]
        
        for name in endgame_names:
            module = get_module(name)
            if module and is_module_relevant(module, context):
                if module not in modules:
                    modules.insert(0, module)  # Prioritize at front
        
        return SelectedProfile(
            name=EvaluationProfile.ENDGAME,
            modules=modules,
            module_count=len(modules),
            reason=f"{context.game_phase.value} - endgame technique",
            estimated_cost_ms=self._estimate_cost(modules)
        )
    
    def _build_comprehensive_profile(self, context: PositionContext) -> SelectedProfile:
        """
        COMPREHENSIVE profile: Full evaluation, no restrictions.
        
        Strategy:
        - Include all relevant modules
        - Use for long time controls and complex middlegames
        - Maximum accuracy, don't worry about speed
        
        Modules: 20-28 (filtered by relevance)
        """
        # Get all relevant modules
        modules = [m for m in MODULE_REGISTRY if is_module_relevant(m, context)]
        
        return SelectedProfile(
            name=EvaluationProfile.COMPREHENSIVE,
            modules=modules,
            module_count=len(modules),
            reason=f"Full evaluation ({context.game_phase.value})",
            estimated_cost_ms=self._estimate_cost(modules)
        )
    
    def _is_tactical_position(self, context: PositionContext) -> bool:
        """
        Detect if position has high tactical activity.
        
        Indicators:
        - King exposed (attack opportunities)
        - Multiple tactical flags active
        - Material imbalance (tactics to convert/recover)
        
        Returns:
            True if tactical profile should be used
        """
        # King safety issues = tactical
        if context.king_safety_critical:
            return True
        
        # Multiple tactical flags = tactical
        if len(context.tactical_flags) >= 2:
            return True
        
        # Material imbalance (but not desperate) = tactical conversion
        if context.material_balance in {MaterialBalance.ADVANTAGE, MaterialBalance.WINNING}:
            # We're winning, use tactics to convert
            return True
        
        if context.material_balance in {MaterialBalance.SLIGHT_ADVANTAGE}:
            # Small advantage, tactical opportunities
            if context.tactical_flags:
                return True
        
        return False
    
    def _estimate_cost(self, modules: List[EvaluationModule]) -> float:
        """
        Estimate total evaluation cost per node.
        
        Based on module cost metadata:
        - NEGLIGIBLE: 0.05ms
        - LOW: 0.2ms
        - MEDIUM: 1.0ms
        - HIGH: 3.0ms
        
        Args:
            modules: List of active modules
            
        Returns:
            Estimated milliseconds per node evaluation
        """
        
        cost_map = {
            EvaluationCost.NEGLIGIBLE: 0.05,
            EvaluationCost.LOW: 0.2,
            EvaluationCost.MEDIUM: 1.0,
            EvaluationCost.HIGH: 3.0
        }
        
        total = sum(cost_map.get(m.cost, 0.5) for m in modules)
        return round(total, 2)
    
    def get_dynamic_threefold_threshold(self, context: PositionContext) -> int:
        """
        Calculate dynamic threefold repetition threshold based on material balance.
        
        Philosophy:
        - Equal position: Never accept draw (0cp threshold)
        - Slight advantage: Very reluctant (10cp)
        - Advantage: Somewhat reluctant (15cp)
        - Winning: Only avoid if crushing (25cp)
        - Crushing: Avoid repetition unless forced (50cp)
        
        This replaces the fixed 100cp threshold that caused v18.2 draw issues.
        
        Args:
            context: Position context
            
        Returns:
            Threshold in centipawns (avoid repetition if eval > threshold)
        """
        if context.material_balance == MaterialBalance.EQUAL:
            return 0  # Never accept draw from equal position
        
        elif context.material_balance == MaterialBalance.SLIGHT_ADVANTAGE:
            return 10  # Very aggressive, avoid draws
        
        elif context.material_balance == MaterialBalance.ADVANTAGE:
            return 15  # Still aggressive
        
        elif context.material_balance == MaterialBalance.WINNING:
            return 25  # Only avoid if truly winning
        
        elif context.material_balance == MaterialBalance.CRUSHING:
            return 50  # Can afford to repeat if not completely crushing
        
        return 0  # Default: never accept draw


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def select_evaluation_profile(context: PositionContext) -> SelectedProfile:
    """
    Convenience function: Select profile for position context.
    
    Args:
        context: Position context from PositionContextCalculator
        
    Returns:
        SelectedProfile with active modules
    """
    selector = EvaluationProfileSelector()
    return selector.select_profile(context)


def get_threefold_threshold(context: PositionContext) -> int:
    """
    Convenience function: Get dynamic threefold threshold.
    
    Args:
        context: Position context
        
    Returns:
        Threshold in centipawns
    """
    selector = EvaluationProfileSelector()
    return selector.get_dynamic_threefold_threshold(context)


# =============================================================================
# DEBUGGING & ANALYSIS
# =============================================================================

def print_profile_details(profile: SelectedProfile):
    """Print detailed profile information"""
    print(f"\n=== Evaluation Profile: {profile.name} ===")
    print(f"Reason: {profile.reason}")
    print(f"Modules: {profile.module_count}")
    print(f"Estimated cost: {profile.estimated_cost_ms:.2f}ms per node")
    print(f"\nActive modules:")
    
    by_criticality = {}
    for module in profile.modules:
        crit = module.criticality.value
        if crit not in by_criticality:
            by_criticality[crit] = []
        by_criticality[crit].append(module.name)
    
    for crit in ["essential", "important", "situational", "optional"]:
        if crit in by_criticality:
            print(f"\n  {crit.upper()}:")
            for name in sorted(by_criticality[crit]):
                print(f"    - {name}")


if __name__ == "__main__":
    # Test profile selection on different positions
    import chess
    
    calculator = PositionContextCalculator()
    selector = EvaluationProfileSelector()
    
    test_positions = [
        ("Starting position", chess.Board(), 300.0, 10.0),
        ("Time pressure", chess.Board(), 15.0, 2.0),
        ("Down a queen", chess.Board("4k3/8/8/8/8/8/4q3/4K3 w - - 0 1"), 300.0, 10.0),
        ("Endgame", chess.Board("8/8/8/8/8/3r4/4P3/4K2R w - - 0 1"), 300.0, 10.0),
        ("King exposed", chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1"), 300.0, 10.0),
    ]
    
    for name, board, time_rem, time_per in test_positions:
        print(f"\n{'='*60}")
        print(f"Position: {name}")
        print(f"FEN: {board.fen()}")
        
        context = calculator.calculate(board, time_rem, time_per)
        profile = selector.select_profile(context)
        
        print_profile_details(profile)
        
        threshold = selector.get_dynamic_threefold_threshold(context)
        print(f"\nThreefold threshold: {threshold}cp")



# ============================================================================
# V7P3R_MOVE_SAFETY
# ============================================================================

"""
V7P3R Move Safety Checker - v18.0.0
Lightweight defensive tactical awareness to prevent hanging pieces

Focuses on:
- Detecting moves that leave pieces undefended
- Identifying opponent's forcing moves (captures, checks)
- Preventing middlegame material losses

Design principle: Speed-first - use minimal board copies and fast checks
"""

import chess
from typing import Optional, Tuple


class MoveSafetyChecker:
    """
    Ultra-lightweight move safety checker for defensive tactics
    Prevents hanging pieces without expensive deep search
    """
    
    def __init__(self, piece_values: dict):
        self.piece_values = piece_values
        
    def evaluate_move_safety(self, board: chess.Board, move: chess.Move) -> float:
        """
        Evaluate if a move creates tactical vulnerability
        Returns penalty (negative score) if move hangs material
        
        Speed: ~1000 checks per second (negligible impact on search)
        """
        penalty = 0.0
        
        # Make move temporarily
        board.push(move)
        
        try:
            # Check 1: Did we leave a piece hanging?
            hanging_penalty = self._check_hanging_pieces(board)
            penalty += hanging_penalty
            
            # Check 2: Did we expose our king to checks?
            if board.is_check():
                # Opponent can give check - mild penalty (checks aren't always bad)
                penalty -= 20.0
            
            # Check 3: Can opponent capture valuable material?
            capture_threat = self._check_immediate_captures(board)
            penalty += capture_threat
            
        finally:
            board.pop()
        
        return penalty
    
    def _check_hanging_pieces(self, board: chess.Board) -> float:
        """
        Check if our pieces are hanging (attacked and undefended)
        Returns negative penalty if material is hanging
        """
        penalty = 0.0
        our_color = not board.turn  # We just moved, so it's opponent's turn
        
        # Check each of our pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == our_color:
                # Skip pawns and king (too expensive to check everything)
                if piece.piece_type in [chess.QUEEN, chess.ROOK, chess.KNIGHT, chess.BISHOP]:
                    if self._is_piece_hanging(board, square, piece):
                        # Piece is hanging - apply penalty based on value
                        piece_value = self.piece_values.get(piece.piece_type, 0)
                        penalty -= piece_value * 0.35  # 35% of piece value as penalty (Phase 2: reduced from 50%)
        
        return penalty
    
    def _is_piece_hanging(self, board: chess.Board, square: int, piece: chess.Piece) -> bool:
        """
        Check if piece is hanging (attacked by opponent, not defended by us)
        Fast check using python-chess built-in methods
        """
        our_color = piece.color
        enemy_color = not our_color
        
        # Check if opponent attacks this square
        is_attacked = board.is_attacked_by(enemy_color, square)
        if not is_attacked:
            return False  # Not attacked, can't be hanging
        
        # Check if we defend this square
        is_defended = board.is_attacked_by(our_color, square)
        if is_defended:
            # Defended - do SEE (Static Exchange Evaluation) to check if it's safe
            # Simple heuristic: if attacker is lower value than defender, it's safe
            attackers = self._get_attackers(board, square, enemy_color)
            defenders = self._get_attackers(board, square, our_color)
            
            if attackers and defenders:
                # Get lowest value attacker vs lowest value defender
                min_attacker = min(attackers)
                min_defender = min(defenders)
                
                # If they can trade favorably, piece is hanging
                piece_value = self.piece_values.get(piece.piece_type, 0)
                if min_attacker < piece_value:
                    return True  # They can capture with lower value piece
            
            return False  # Defended adequately
        
        # Attacked and undefended = hanging
        return True
    
    def _get_attackers(self, board: chess.Board, square: int, color: bool) -> list:
        """Get list of piece values attacking a square"""
        attackers = []
        
        # Check all pieces of this color
        for attacker_square in chess.SQUARES:
            piece = board.piece_at(attacker_square)
            if piece and piece.color == color:
                if board.is_attacked_by(color, square):
                    # This piece attacks the square
                    attackers.append(self.piece_values.get(piece.piece_type, 0))
        
        return attackers
    
    def _check_immediate_captures(self, board: chess.Board) -> float:
        """
        Check if opponent can capture valuable material on their next move
        Returns penalty if high-value captures are available
        """
        penalty = 0.0
        our_color = not board.turn
        
        # Check opponent's capturing moves
        for move in board.legal_moves:
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                if captured_piece and captured_piece.color == our_color:
                    # Opponent can capture our piece
                    capture_value = self.piece_values.get(captured_piece.piece_type, 0)
                    
                    # Only penalize if it's a high-value piece (Q, R)
                    if captured_piece.piece_type in [chess.QUEEN, chess.ROOK]:
                        penalty -= capture_value * 0.10  # 10% penalty (Phase 2: reduced from 15%)
        
        return penalty
    
    def get_safe_moves(self, board: chess.Board, moves: list, threshold: float = -50.0) -> list:
        """
        Filter moves to only safe ones (penalty above threshold)
        Use this to avoid obviously bad moves
        
        threshold: minimum safety score (-50 = allow small penalties)
        """
        safe_moves = []
        
        for move in moves:
            safety_score = self.evaluate_move_safety(board, move)
            if safety_score >= threshold:
                safe_moves.append((move, safety_score))
        
        # Sort by safety (most safe first)
        safe_moves.sort(key=lambda x: x[1], reverse=True)
        
        return [move for move, score in safe_moves]


