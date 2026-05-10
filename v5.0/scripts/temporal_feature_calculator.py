"""
V7P3R AI v5.1 - Temporal Feature Calculator
============================================
Calculates historical/temporal features (F200-F220) for position sequences.

Implements Temporal Persistence Features (TPF) concept:
- Tracks feature changes over time (differential learning)
- Encodes move sequences and piece paths
- Maintains position history cache for efficiency

Usage:
    from temporal_feature_calculator import TemporalFeatureCalculator
    
    calc = TemporalFeatureCalculator(base_calculator)
    features = calc.calculate_temporal_features(
        current_fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        previous_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        last_move_uci="e2e4",
        sequence_index=1
    )
"""

import chess
from typing import Dict, Any, Optional
import logging


class TemporalFeatureCalculator:
    """
    Calculate temporal/historical features with position memory.
    
    Adds F200-F220 features to capture:
    - Historical tactical state (F200-F209)
    - Historical evaluation trends (F210-F214)
    - Move sequence encoding (F215-F219)
    - Context availability mask (F220)
    """
    
    def __init__(self, base_calculator):
        """
        Initialize temporal calculator.
        
        Args:
            base_calculator: FeatureCalculator instance for current features
        """
        self.base_calculator = base_calculator
        self.position_cache = {}  # FEN → features mapping for efficiency
        self.logger = logging.getLogger(__name__)
    
    def calculate_temporal_features(
        self,
        current_fen: str,
        previous_fen: Optional[str] = None,
        last_move_uci: Optional[str] = None,
        sequence_index: int = 0,
        stockfish_eval: float = 0.0
    ) -> Dict[str, Any]:
        """
        Calculate complete feature set with temporal context.
        
        Args:
            current_fen: Current position FEN string
            previous_fen: Previous position FEN (None if no history)
            last_move_uci: Last move in UCI format (e.g., "e2e4")
            sequence_index: Position in multi-move sequence (0 if standalone)
            stockfish_eval: Current position eval for historical tracking
            
        Returns:
            dict: Complete features (F000-F114 current + F200-F220 temporal)
        """
        # Calculate current position features (F000-F114)
        current_features = self.base_calculator.calculate_features_from_fen(current_fen, last_move_uci)
        current_features['stockfish_eval_current'] = stockfish_eval
        
        # Initialize temporal features dictionary
        temporal_features = {}
        
        if previous_fen is not None and last_move_uci:
            # Has history - calculate temporal features
            temporal_features = self._calculate_with_history(
                current_features=current_features,
                previous_fen=previous_fen,
                last_move_uci=last_move_uci,
                sequence_index=sequence_index
            )
        else:
            # No history - set null sentinels
            temporal_features = self._calculate_without_history()
        
        # Merge current + temporal features
        all_features = {**current_features, **temporal_features}
        
        # Cache current position for future iterations
        self.position_cache[current_fen] = current_features.copy()
        
        return all_features
    
    def _calculate_with_history(
        self,
        current_features: Dict[str, Any],
        previous_fen: str,
        last_move_uci: str,
        sequence_index: int
    ) -> Dict[str, Any]:
        """Calculate temporal features when history is available."""
        temporal = {}
        
        # Get or calculate previous position features
        if previous_fen in self.position_cache:
            prev_features = self.position_cache[previous_fen]
        else:
            prev_features = self.base_calculator.calculate_features_from_fen(previous_fen)
            self.position_cache[previous_fen] = prev_features
        
        # F200-F209: Historical Tactical State
        temporal['white_hanging_pieces_historical'] = int(prev_features.get('white_has_hanging_pieces', 0))
        temporal['black_hanging_pieces_historical'] = int(prev_features.get('black_has_hanging_pieces', 0))
        temporal['white_en_prise_value_historical'] = float(prev_features.get('white_en_prise_value', 0))
        temporal['black_en_prise_value_historical'] = float(prev_features.get('black_en_prise_value', 0))
        temporal['white_pins_historical'] = int(prev_features.get('white_has_pin', 0))
        temporal['black_pins_historical'] = int(prev_features.get('black_has_pin', 0))
        temporal['white_king_under_attack_historical'] = int(prev_features.get('white_king_under_attack', 0))
        temporal['black_king_under_attack_historical'] = int(prev_features.get('black_king_under_attack', 0))
        temporal['white_trapped_pieces_historical'] = int(prev_features.get('white_trapped_piece_count', 0))
        temporal['black_trapped_pieces_historical'] = int(prev_features.get('black_trapped_piece_count', 0))
        
        # F210-F214: Historical Position Evaluation
        temporal['position_eval_historical'] = float(prev_features.get('stockfish_eval_current', 0.0))
        temporal['material_balance_historical'] = float(prev_features.get('material_balance_cp', 0))
        temporal['king_safety_white_historical'] = float(prev_features.get('white_king_safety_score', 0.5))
        temporal['king_safety_black_historical'] = float(prev_features.get('black_king_safety_score', 0.5))
        temporal['center_control_historical'] = float(prev_features.get('white_center_control_score', 0))
        
        # F215-F219: Move Sequence Encoding
        move_encoding = self._encode_move(previous_fen, last_move_uci)
        temporal.update(move_encoding)
        temporal['move_sequence_index'] = sequence_index
        temporal['is_forcing_sequence'] = self._detect_forcing_sequence(current_features, prev_features)
        
        # F220: Has history flag
        temporal['has_history'] = 1
        
        return temporal
    
    def _calculate_without_history(self) -> Dict[str, Any]:
        """Calculate temporal features when no history is available (null sentinels)."""
        temporal = {}
        
        # F200-F209: Set to -1 (null sentinel for booleans/counts)
        temporal['white_hanging_pieces_historical'] = -1
        temporal['black_hanging_pieces_historical'] = -1
        temporal['white_en_prise_value_historical'] = -999.0  # Obvious null for continuous
        temporal['black_en_prise_value_historical'] = -999.0
        temporal['white_pins_historical'] = -1
        temporal['black_pins_historical'] = -1
        temporal['white_king_under_attack_historical'] = -1
        temporal['black_king_under_attack_historical'] = -1
        temporal['white_trapped_pieces_historical'] = -1
        temporal['black_trapped_pieces_historical'] = -1
        
        # F210-F214: Set to null sentinels
        temporal['position_eval_historical'] = -999.0
        temporal['material_balance_historical'] = -999.0
        temporal['king_safety_white_historical'] = -1.0
        temporal['king_safety_black_historical'] = -1.0
        temporal['center_control_historical'] = -999.0
        
        # F215-F219: No move encoding
        temporal['last_move_from_square'] = -1
        temporal['last_move_to_square'] = -1
        temporal['last_move_piece_type'] = 0  # 0 = no piece
        temporal['move_sequence_index'] = 0
        temporal['is_forcing_sequence'] = 0
        
        # F220: No history flag
        temporal['has_history'] = 0
        
        return temporal
    
    def _encode_move(self, previous_fen: str, move_uci: str) -> Dict[str, int]:
        """
        Encode move as from_square, to_square, piece_type.
        
        Returns:
            dict: {
                'last_move_from_square': 0-63,
                'last_move_to_square': 0-63,
                'last_move_piece_type': 1-6 (pawn, knight, bishop, rook, queen, king)
            }
        """
        try:
            board = chess.Board(previous_fen)
            move = chess.Move.from_uci(move_uci)
            
            from_square = move.from_square
            to_square = move.to_square
            piece = board.piece_at(from_square)
            piece_type = piece.piece_type if piece else 0
            
            return {
                'last_move_from_square': from_square,
                'last_move_to_square': to_square,
                'last_move_piece_type': piece_type
            }
        except Exception as e:
            self.logger.warning(f"Move encoding failed for {move_uci} from {previous_fen}: {e}")
            return {
                'last_move_from_square': -1,
                'last_move_to_square': -1,
                'last_move_piece_type': 0
            }
    
    def _detect_forcing_sequence(self, current_features: Dict, prev_features: Dict) -> int:
        """
        Detect if this is part of a forcing tactical sequence.
        
        Heuristic: Forcing if checks/captures and material/king safety changing.
        
        Returns:
            1 if forcing, 0 otherwise
        """
        # Check if current move is check or capture
        is_forcing_move = (
            current_features.get('is_check', False) or
            current_features.get('is_capture', False)
        )
        
        if not is_forcing_move:
            return 0
        
        # Check if king safety or material changed significantly
        material_change = abs(
            current_features.get('material_balance_cp', 0) -
            prev_features.get('material_balance_cp', 0)
        )
        
        king_safety_change = abs(
            current_features.get('white_king_safety_score', 0.5) -
            prev_features.get('white_king_safety_score', 0.5)
        )
        
        # Forcing if material traded (>100cp) or king safety deteriorated (>0.2)
        if material_change > 100 or king_safety_change > 0.2:
            return 1
        
        return 0
    
    def clear_cache(self):
        """Clear position cache to free memory."""
        self.position_cache.clear()
        self.logger.info("Position cache cleared")
    
    def get_cache_size(self) -> int:
        """Get number of cached positions."""
        return len(self.position_cache)
