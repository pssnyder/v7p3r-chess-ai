"""
V7P3R AI v3.0 - ChessState Data Extraction System
==================================================

This module implements comprehensive metadata extraction from chess positions,
providing rich objective data for the "Thinking Brain" to discover patterns.

No human chess knowledge or heuristics - pure mathematical feature extraction.
"""

import chess
import chess.engine
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json


@dataclass
class PieceFeatures:
    """Complete feature set for a single chess piece"""
    type: int  # 1=pawn, 2=knight, 3=bishop, 4=rook, 5=queen, 6=king
    color: int  # 0=white, 1=black
    currentSquare: int  # 0-63 board position
    baseValue: int  # Standard material value
    
    # Mobility metrics
    mobility: Dict[str, int]
    
    # Relationship metrics
    relationships: Dict[str, Any]
    
    # Positional metrics
    positional: Dict[str, Any]
    
    # Vector analysis
    vectorAnalysis: Dict[str, int]


@dataclass
class BoardFeatures:
    """Global board-wide metrics"""
    sideToMove: int
    castlingRights: List[bool]
    enPassantSquare: Optional[int]
    halfmoveClock: int
    fullmoveNumber: int
    isKingInCheck: bool
    
    # Threat analysis
    threatCounts: Dict[str, int]
    
    # Material analysis
    materialBalance: Dict[str, int]
    pieceCounts: Dict[str, Dict[str, int]]
    
    # Pawn structure
    pawnStructure: Dict[str, Dict[str, int]]
    
    # King safety
    kingSafety: Dict[str, int]
    
    # Game phase and dynamics
    gamePhase: float  # 0=opening, 1=endgame
    symmetrical: float  # Board symmetry measure
    mobility: Dict[str, int]
    
    # Special tactical patterns
    uniqueInteractions: Dict[str, bool]


@dataclass
class ChessState:
    """Complete state representation for AI training"""
    board_features: BoardFeatures
    pieces: List[PieceFeatures]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to flat numpy array for neural network input"""
        # This will be implemented to create a fixed-size input vector
        # for the GRU neural network
        return np.array([0])  # Placeholder until implementation


class ChessStateExtractor:
    """Extracts comprehensive metadata from chess positions"""
    
    # Piece values for material calculation
    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    
    def __init__(self):
        """Initialize the state extractor"""
        pass
    
    def extract_state(self, board: chess.Board) -> ChessState:
        """
        Extract complete ChessState from a python-chess Board
        
        Args:
            board: python-chess Board object
            
        Returns:
            ChessState: Complete feature representation
        """
        # Extract global board features
        board_features = self._extract_board_features(board)
        
        # Extract individual piece features
        pieces = self._extract_piece_features(board)
        
        return ChessState(
            board_features=board_features,
            pieces=pieces
        )
    
    def _extract_board_features(self, board: chess.Board) -> BoardFeatures:
        """Extract global board-wide features"""
        
        # Basic game state
        side_to_move = 0 if board.turn == chess.WHITE else 1
        castling_rights = [
            board.has_kingside_castling_rights(chess.WHITE),
            board.has_queenside_castling_rights(chess.WHITE),
            board.has_kingside_castling_rights(chess.BLACK),
            board.has_queenside_castling_rights(chess.BLACK)
        ]
        en_passant = board.ep_square if board.ep_square else None
        
        # Threat analysis
        threat_counts = self._calculate_threat_counts(board)
        
        # Material analysis
        material_balance = self._calculate_material_balance(board)
        piece_counts = self._calculate_piece_counts(board)
        
        # Pawn structure analysis
        pawn_structure = self._analyze_pawn_structure(board)
        
        # King safety analysis
        king_safety = self._analyze_king_safety(board)
        
        # Game phase calculation
        game_phase = self._calculate_game_phase(board)
        
        # Board symmetry
        symmetrical = self._calculate_symmetry(board)
        
        # Mobility analysis
        mobility = self._calculate_mobility(board)
        
        # Special tactical patterns
        unique_interactions = self._detect_tactical_patterns(board)
        
        return BoardFeatures(
            sideToMove=side_to_move,
            castlingRights=castling_rights,
            enPassantSquare=en_passant,
            halfmoveClock=board.halfmove_clock,
            fullmoveNumber=board.fullmove_number,
            isKingInCheck=board.is_check(),
            threatCounts=threat_counts,
            materialBalance=material_balance,
            pieceCounts=piece_counts,
            pawnStructure=pawn_structure,
            kingSafety=king_safety,
            gamePhase=game_phase,
            symmetrical=symmetrical,
            mobility=mobility,
            uniqueInteractions=unique_interactions
        )
    
    def _extract_piece_features(self, board: chess.Board) -> List[PieceFeatures]:
        """Extract features for all 32 pieces on the board"""
        pieces = []
        
        # Process all squares on the board
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_features = self._extract_single_piece_features(board, piece, square)
                pieces.append(piece_features)
        
        # Pad to ensure we always have 32 pieces (captured pieces as null/inactive)
        while len(pieces) < 32:
            pieces.append(self._create_inactive_piece())
        
        return pieces[:32]  # Ensure exactly 32 pieces
    
    def _extract_single_piece_features(self, board: chess.Board, piece: chess.Piece, square: int) -> PieceFeatures:
        """Extract comprehensive features for a single piece"""
        
        # Basic piece info
        piece_type = piece.piece_type
        color = 0 if piece.color == chess.WHITE else 1
        base_value = self.PIECE_VALUES[piece_type]
        
        # Mobility analysis
        mobility = self._calculate_piece_mobility(board, piece, square)
        
        # Relationship analysis
        relationships = self._analyze_piece_relationships(board, piece, square)
        
        # Positional analysis
        positional = self._analyze_piece_position(board, piece, square)
        
        # Vector analysis
        vector_analysis = self._analyze_piece_vectors(board, piece, square)
        
        return PieceFeatures(
            type=piece_type,
            color=color,
            currentSquare=square,
            baseValue=base_value,
            mobility=mobility,
            relationships=relationships,
            positional=positional,
            vectorAnalysis=vector_analysis
        )
    
    def _create_inactive_piece(self) -> PieceFeatures:
        """Create a null/inactive piece for padding"""
        return PieceFeatures(
            type=0,
            color=0,
            currentSquare=-1,
            baseValue=0,
            mobility={},
            relationships={},
            positional={},
            vectorAnalysis={}
        )
    
    # Placeholder methods for feature calculations
    # These will be implemented with comprehensive chess logic
    
    def _calculate_threat_counts(self, board: chess.Board) -> Dict[str, int]:
        """Calculate square attack counts for both sides"""
        white_attacked = len([sq for sq in chess.SQUARES if board.is_attacked_by(chess.WHITE, sq)])
        black_attacked = len([sq for sq in chess.SQUARES if board.is_attacked_by(chess.BLACK, sq)])
        
        return {
            "whiteAttacked": white_attacked,
            "blackAttacked": black_attacked
        }
    
    def _calculate_material_balance(self, board: chess.Board) -> Dict[str, int]:
        """Calculate material values for both sides"""
        white_total = 0
        black_total = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.PIECE_VALUES[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_total += value
                else:
                    black_total += value
        
        return {
            "whiteTotal": white_total,
            "blackTotal": black_total,
            "difference": white_total - black_total
        }
    
    def _calculate_piece_counts(self, board: chess.Board) -> Dict[str, Dict[str, int]]:
        """Count pieces by type for both sides"""
        piece_names = ["pawn", "knight", "bishop", "rook", "queen", "king"]
        counts = {
            "white": {name: 0 for name in piece_names},
            "black": {name: 0 for name in piece_names}
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                color_key = "white" if piece.color == chess.WHITE else "black"
                piece_name = chess.piece_name(piece.piece_type)
                counts[color_key][piece_name] += 1
        
        return counts
    
    def _analyze_pawn_structure(self, board: chess.Board) -> Dict[str, Dict[str, int]]:
        """Analyze pawn structure metrics"""
        # Placeholder - will implement comprehensive pawn analysis
        return {
            "isolatedPawns": {"white": 0, "black": 0},
            "doubledPawns": {"white": 0, "black": 0},
            "passedPawns": {"white": 0, "black": 0},
            "connectedPawns": {"white": 0, "black": 0},
            "pawnIslands": {"white": 0, "black": 0},
            "backwardPawns": {"white": 0, "black": 0}
        }
    
    def _analyze_king_safety(self, board: chess.Board) -> Dict[str, int]:
        """Analyze king safety metrics"""
        # Placeholder - will implement comprehensive king safety analysis
        return {
            "pawnShieldStrength": 0,
            "escapeSquares": 0,
            "exposedFiles": 0,
            "exposedDiagonals": 0
        }
    
    def _calculate_game_phase(self, board: chess.Board) -> float:
        """Calculate game phase (0=opening, 1=endgame) based on material"""
        total_material = sum(
            self.PIECE_VALUES[piece.piece_type] 
            for square in chess.SQUARES 
            for piece in [board.piece_at(square)] 
            if piece and piece.piece_type != chess.KING
        )
        
        # Starting material (without kings): 78 points
        starting_material = 78
        phase = 1.0 - (total_material / starting_material)
        return max(0.0, min(1.0, phase))
    
    def _calculate_symmetry(self, board: chess.Board) -> float:
        """Calculate board symmetry measure"""
        # Placeholder - will implement symmetry calculation
        return 0.5
    
    def _calculate_mobility(self, board: chess.Board) -> Dict[str, int]:
        """Calculate total mobility for both sides"""
        white_moves = len(list(board.legal_moves)) if board.turn == chess.WHITE else 0
        black_moves = len(list(board.legal_moves)) if board.turn == chess.BLACK else 0
        
        return {
            "totalWhite": white_moves,
            "totalBlack": black_moves
        }
    
    def _detect_tactical_patterns(self, board: chess.Board) -> Dict[str, bool]:
        """Detect special tactical patterns"""
        # Placeholder - will implement tactical pattern detection
        return {
            "isSkewer": False,
            "isFork": False,
            "isPin": False,
            "isDiscoveredAttack": False,
            "isBattery": False,
            "threatensKingThroughPawn": False
        }
    
    def _calculate_piece_mobility(self, board: chess.Board, piece: chess.Piece, square: int) -> Dict[str, int]:
        """Calculate mobility metrics for a single piece"""
        # Placeholder - will implement detailed piece mobility analysis
        return {
            "legalMoves": 0,
            "attackingMoves": 0,
            "maxUnobstructed": 0
        }
    
    def _analyze_piece_relationships(self, board: chess.Board, piece: chess.Piece, square: int) -> Dict[str, Any]:
        """Analyze piece relationships (attacks, defends, pins, etc.)"""
        # Placeholder - will implement comprehensive relationship analysis
        return {
            "attackedByCount": 0,
            "defendedByCount": 0,
            "isPinned": False,
            "pinner": None,
            "pinnedTo": None,
            "attackingHigherValue": False,
            "attackingLowerValue": False,
            "attackingEqualValue": False,
            "interposingSquares": 0,
            "kingDistance": 0,
            "enemyKingDistance": 0
        }
    
    def _analyze_piece_position(self, board: chess.Board, piece: chess.Piece, square: int) -> Dict[str, Any]:
        """Analyze positional metrics for a piece"""
        # Calculate centrality (distance from center)
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        center_distance = max(abs(file - 3.5), abs(rank - 3.5))
        
        # Calculate edge proximity
        edge_distance = min(file, 7 - file, rank, 7 - rank)
        
        return {
            "centrality": int(center_distance),
            "edgeProximity": edge_distance,
            "backRankStatus": rank == 0 or rank == 7,
            "seventhRankStatus": rank == 1 or rank == 6
        }
    
    def _analyze_piece_vectors(self, board: chess.Board, piece: chess.Piece, square: int) -> Dict[str, int]:
        """Analyze movement vectors for a piece"""
        # Placeholder - will implement vector analysis
        return {
            "forwardVectors": 0,
            "backwardVectors": 0,
            "lateralVectors": 0,
            "diagonalVectors": 0,
            "knightVectors": 0
        }


# Usage example
if __name__ == "__main__":
    # Test the ChessState extraction
    board = chess.Board()  # Starting position
    extractor = ChessStateExtractor()
    
    state = extractor.extract_state(board)
    
    # Print some basic info
    print(f"Side to move: {state.board_features.sideToMove}")
    print(f"Material balance: {state.board_features.materialBalance}")
    print(f"Game phase: {state.board_features.gamePhase}")
    print(f"Number of pieces: {len([p for p in state.pieces if p.type > 0])}")
    
    # Convert to JSON for inspection
    print("\nFull state as JSON:")
    print(json.dumps(state.to_dict(), indent=2))
