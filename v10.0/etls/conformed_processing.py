# Conformed Layer Processing

import os
import json

# Conformed layer schema:
CONFORMED_SCHEMA = {
    "fen_hash": "int64",
    "eval_cp": "int16",
    "depth": "int16",
    "time_ms": "uint32",
    "clock_s": "uint16",
    "wdl": "int8",
    "material": "uint16",
    "phase": "uint8",
    "piece_count": "int8",

    # New features extracted during conformed processing
    "quiet_move": "bool", # True if the move is a quiet move (Not in check, no immediate captures available, no immediate checks available, no pawn promotions, no major en prise hanging material)
    "en_pass_move": "bool", # True if the move is an en passant capture
    "move_count": "int8", # Move count in the game (fullmove number from FEN)
    "legal_move_count": "int8", # Number of legal moves available in the position
    
    # FEN string is currently included for feature extraction but will be removed
    "fen": "string"
}
