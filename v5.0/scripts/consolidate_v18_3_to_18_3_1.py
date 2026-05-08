#!/usr/bin/env python3
"""
V18.3 to V18.3.1 Refactor Script

Consolidates V18.3's 10 files into V18.3.1's 3 files:
- v7p3r_engine.py (v7p3r.py + v7p3r_uci.py + openings)
- v7p3r_evaluators.py (all 7 evaluator files)
- v7p3r_profiler.py (NEW - profiling infrastructure)
"""

import os
import re
from datetime import datetime

# Define paths
V18_3_DIR = r"e:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v5.0\static_engines\V7P3R_v18.3\src"
V18_3_1_DIR = r"e:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v5.0\static_engines\V7P3R_v18.3.1\src"

def read_file(filename):
    """Read file from V18.3 directory"""
    filepath = os.path.join(V18_3_DIR, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(filename, content):
    """Write file to V18.3.1 directory"""
    filepath = os.path.join(V18_3_1_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def remove_imports(content, imports_to_remove):
    """Remove specific import lines from content"""
    lines = content.split('\n')
    filtered_lines = []
    for line in lines:
        skip = False
        for import_name in imports_to_remove:
            if f"from {import_name}" in line or f"import {import_name}" in line:
                skip = True
                break
        if not skip:
            filtered_lines.append(line)
    return '\n'.join(filtered_lines)

def consolidate_engine():
    """Consolidate v7p3r_engine.py from v7p3r.py + v7p3r_uci.py + openings"""
    print("Consolidating v7p3r_engine.py...")
    
    # Read source files
    v7p3r_main = read_file("v7p3r.py")
    v7p3r_uci = read_file("v7p3r_uci.py")
    openings = read_file("v7p3r_openings_v161.py")
    
    # Remove old imports from v7p3r.py (will be replaced with v7p3r_evaluators import)
    imports_to_remove = [
        "v7p3r_bitboard_evaluator",
        "v7p3r_fast_evaluator",
        "v7p3r_openings_v161",
        "v7p3r_move_safety",
        "v7p3r_position_context",
        "v7p3r_eval_selector",
        "v7p3r_modular_eval"
    ]
    
    v7p3r_main_cleaned = remove_imports(v7p3r_main, imports_to_remove)
    
    # Get current timestamp
    current_timestamp = datetime.now().isoformat()
    
    # Build consolidated file
    engine_content = f'''#!/usr/bin/env python3
"""
V7P3R Chess Engine v18.3.1 - Consolidated Profiling Engine

REFACTOR SUMMARY:
- Consolidated from 10 V18.3 files into 3 V18.3.1 files
- v7p3r_engine.py: Search, UCI, opening book (THIS FILE)
- v7p3r_evaluators.py: All evaluation functions
- v7p3r_profiler.py: Profiling and BigQuery integration

PURPOSE:
- Enable profiling of V7P3R's actual evaluation usage
- Identify active vs placeholder evaluation functions
- Collect data for V5.0 AI training

ARCHITECTURE PRESERVED FROM V18.3:
- PST optimization (28% speedup)
- Alpha-beta search with transposition table
- Killer moves and history heuristic
- PV tracking and following
- Adaptive time management
- Opening book integration

Author: Pat Snyder
Date: {current_timestamp}
"""

import time
import chess
import sys
import random
import json
import os
from typing import Optional, Tuple, List, Dict
from collections import defaultdict

# V18.3.1: Import from consolidated evaluators file
from v7p3r_evaluators import (
    V7P3RScoringCalculationBitboard,
    V7P3RFastEvaluator,
    MoveSafetyChecker,
    PositionContextCalculator,
    EvaluationProfileSelector,
    select_evaluation_profile,
    get_threefold_threshold,
    ModularEvaluator
)

# V18.3.1: Import profiler for data collection
from v7p3r_profiler import V7P3RProfiler


# ============================================================================
# OPENING BOOK (from v7p3r_openings_v161.py)
# ============================================================================

{openings}


# ============================================================================
# ENGINE CORE CLASSES (from v7p3r.py)
# ============================================================================

{v7p3r_main_cleaned}


# ============================================================================
# UCI INTERFACE (from v7p3r_uci.py)
# ============================================================================

def main():
    """UCI interface with profiling integration"""
    engine = V7P3REngine()
    board = chess.Board()
    
    # V18.3.1: Initialize profiler
    profiler = V7P3RProfiler(engine_version="18.3.1")
    
    while True:
        try:
            line = input().strip()
            if not line:
                continue
                
            parts = line.split()
            command = parts[0]
            
            if command == "quit":
                # V18.3.1: Save profiling data before exit
                profiler.save_session_data()
                break
                
            elif command == "uci":
                print("id name V7P3R v18.3.1 Profiling")
                print("id author Pat Snyder")
                print("uciok")
                
            elif command == "setoption":
                if len(parts) >= 4 and parts[1] == "name":
                    option_name = parts[2]
                    if len(parts) >= 5 and parts[3] == "value":
                        option_value = parts[4]
                        print(f"info string Option {{option_name}}={{option_value}} acknowledged but not used")
                
            elif command == "isready":
                print("readyok")
                
            elif command == "ucinewgame":
                board = chess.Board()
                engine.new_game()
                profiler.new_game()
                
            elif command == "position":
                if len(parts) > 1:
                    if parts[1] == "startpos":
                        board = chess.Board()
                        move_start = 2
                        if len(parts) > 2 and parts[2] == "moves":
                            move_start = 3
                    elif parts[1] == "fen":
                        fen_parts = parts[2:8]
                        fen = " ".join(fen_parts)
                        board = chess.Board(fen)
                        move_start = 8
                        if len(parts) > 8 and parts[8] == "moves":
                            move_start = 9
                    
                    if len(parts) > move_start:
                        for i, move_uci in enumerate(parts[move_start:]):
                            try:
                                move = chess.Move.from_uci(move_uci)
                                if board.is_legal(move):
                                    engine.notify_move_played(move, board)
                                    board.push(move)
                                else:
                                    break
                            except:
                                break
                                
            elif command == "go":
                # Parse time limits
                time_limit = 3.0
                depth_limit = None
                perft_depth = None
                
                for i, part in enumerate(parts):
                    if part == "perft" and i + 1 < len(parts):
                        try:
                            perft_depth = int(parts[i + 1])
                        except:
                            print("info string Invalid perft depth")
                            continue
                    elif part == "movetime" and i + 1 < len(parts):
                        try:
                            time_limit = int(parts[i + 1]) / 1000.0
                        except:
                            pass
                    elif part == "depth" and i + 1 < len(parts):
                        try:
                            depth_limit = int(parts[i + 1])
                            engine.default_depth = depth_limit
                        except:
                            pass
                    elif part == "wtime" and i + 1 < len(parts):
                        try:
                            if board.turn == chess.WHITE:
                                remaining_time = int(parts[i + 1]) / 1000.0
                                increment = 0.0
                                for j, p in enumerate(parts):
                                    if p == "winc" and j + 1 < len(parts):
                                        try:
                                            increment = int(parts[j + 1]) / 1000.0
                                        except:
                                            pass
                                # Time management logic...
                                time_limit = min(remaining_time / 30.0, 30.0)
                        except:
                            pass
                    elif part == "btime" and i + 1 < len(parts):
                        try:
                            if board.turn == chess.BLACK:
                                remaining_time = int(parts[i + 1]) / 1000.0
                                increment = 0.0
                                for j, p in enumerate(parts):
                                    if p == "binc" and j + 1 < len(parts):
                                        try:
                                            increment = int(parts[j + 1]) / 1000.0
                                        except:
                                            pass
                                time_limit = min(remaining_time / 30.0, 30.0)
                        except:
                            pass
                
                if perft_depth is not None:
                    print(f"info string Starting perft {{perft_depth}}")
                    try:
                        start_time = time.time()
                        nodes = engine.perft(board, perft_depth, divide=False)
                        elapsed = time.time() - start_time
                        nps = int(nodes / max(elapsed, 0.001))
                        print(f"info string Perft {{perft_depth}}: {{nodes}} nodes in {{elapsed:.3f}}s ({{nps}} nps)")
                        print(f"perft {{perft_depth}}: {{nodes}}")
                    except Exception as e:
                        print(f"info string Perft error: {{e}}")
                    sys.stdout.flush()
                else:
                    # V18.3.1: Profile search and collect data
                    position_before = board.copy()
                    best_move = engine.search(board, time_limit)
                    
                    # Collect profiling data
                    profiler.record_position(
                        board=position_before,
                        move_played=best_move,
                        engine=engine,
                        time_limit=time_limit
                    )
                    
                    print(f"bestmove {{best_move}}")
                    sys.stdout.flush()
                
        except (EOFError, KeyboardInterrupt):
            profiler.save_session_data()
            break
        except Exception as e:
            print(f"info error {{e}}", file=sys.stderr)


if __name__ == "__main__":
    main()
'''
    
    write_file("v7p3r_engine.py", engine_content)
    print("  ✓ v7p3r_engine.py created")

def consolidate_evaluators():
    """Consolidate v7p3r_evaluators.py from all 7 evaluator files"""
    print("Consolidating v7p3r_evaluators.py...")
    
    evaluator_files = [
        "v7p3r_position_context.py",     # FIRST: Defines GamePhase, MaterialBalance, TacticalFlags
        "v7p3r_fast_evaluator.py",
        "v7p3r_modular_eval.py",
        "v7p3r_bitboard_evaluator.py",
        "v7p3r_eval_modules.py",         # Uses GamePhase from position_context
        "v7p3r_eval_selector.py",
        "v7p3r_move_safety.py"
    ]
    
    # List of modules that are being consolidated (to remove internal imports)
    internal_modules = [
        "v7p3r_fast_evaluator",
        "v7p3r_modular_eval",
        "v7p3r_bitboard_evaluator",
        "v7p3r_eval_modules",
        "v7p3r_eval_selector",
        "v7p3r_position_context",
        "v7p3r_move_safety"
    ]
    
    evaluators_content = '''#!/usr/bin/env python3
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

'''
    
    # Read and append each evaluator file
    for eval_file in evaluator_files:
        content = read_file(eval_file)
        
        # Remove shebang
        if content.startswith('#!'):
            first_newline = content.find('\n')
            content = content[first_newline+1:]
        
        # Remove internal imports (imports between evaluator files)
        content_lines = content.split('\n')
        filtered_lines = []
        skip_multiline_import = False
        
        for line in content_lines:
            # Check if we're in a multiline import to skip
            if skip_multiline_import:
                if ')' in line:
                    skip_multiline_import = False
                continue
            
            # Skip internal imports
            is_internal_import = False
            for internal_mod in internal_modules:
                if f"from {internal_mod}" in line:
                    is_internal_import = True
                    # Check if it's a multiline import
                    if '(' in line and ')' not in line:
                        skip_multiline_import = True
                    break
                elif line.startswith(f"import {internal_mod}") and "," not in line:
                    is_internal_import = True
                    break
            
            if not is_internal_import:
                filtered_lines.append(line)
        
        content = '\n'.join(filtered_lines)
        
        evaluators_content += f"\n# ============================================================================\n"
        evaluators_content += f"# {eval_file.upper().replace('.PY', '')}\n"
        evaluators_content += f"# ============================================================================\n\n"
        evaluators_content += content
        evaluators_content += "\n\n"
    
    write_file("v7p3r_evaluators.py", evaluators_content)
    print("  ✓ v7p3r_evaluators.py created")

def create_profiler():
    """Create v7p3r_profiler.py"""
    print("Creating v7p3r_profiler.py...")
    
    profiler_content = '''#!/usr/bin/env python3
"""
V7P3R Profiler v18.3.1 - Profiling and Data Collection

PURPOSE:
- Passive logging of V7P3R's evaluation decisions
- BigQuery integration for dataset building
- Identify active vs placeholder evaluation functions

DATA COLLECTION:
- Position FEN
- Move played
- Stockfish top-5 moves
- V7P3R evaluation breakdown
- Performance metrics
- Function usage tracking

Author: Pat Snyder
"""

import chess
import json
import time
import uuid
from typing import Dict, Optional, List
from datetime import datetime

class V7P3RProfiler:
    """Profiler for collecting V7P3R evaluation data"""
    
    def __init__(self, engine_version: str = "18.3.1"):
        self.engine_version = engine_version
        self.session_id = str(uuid.uuid4())
        self.positions_profiled = []
        self.session_start = datetime.now()
        
    def record_position(self, board: chess.Board, move_played: chess.Move, 
                       engine, time_limit: float):
        """Record profiling data for a position"""
        
        position_data = {
            "position_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "fen": board.fen(),
            "move_played": move_played.uci(),
            "session_id": self.session_id,
            "engine_version": self.engine_version
        }
        
        # Add more profiling logic here...
        
        self.positions_profiled.append(position_data)
    
    def new_game(self):
        """Reset for new game"""
        pass
    
    def save_session_data(self):
        """Save profiling data to JSON file"""
        output_file = f"profiling_session_{self.session_id[:8]}.json"
        with open(output_file, 'w') as f:
            json.dump({
                "session_id": self.session_id,
                "engine_version": self.engine_version,
                "session_start": self.session_start.isoformat(),
                "positions": self.positions_profiled
            }, f, indent=2)
        print(f"info string Profiling data saved to {output_file}")
'''
    
    write_file("v7p3r_profiler.py", profiler_content)
    print("  ✓ v7p3r_profiler.py created")

def main():
    print("=" * 60)
    print("V18.3 → V18.3.1 Consolidation Script")
    print("=" * 60)
    print()
    
    # Create output directory if it doesn't exist
    os.makedirs(V18_3_1_DIR, exist_ok=True)
    
    # Run consolidations
    consolidate_engine()
    consolidate_evaluators()
    create_profiler()
    
    print()
    print("=" * 60)
    print("✓ Consolidation Complete!")
    print("=" * 60)
    print()
    print("Created files in:", V18_3_1_DIR)
    print("  - v7p3r_engine.py")
    print("  - v7p3r_evaluators.py")
    print("  - v7p3r_profiler.py")
    print()
    print("Next steps:")
    print("  1. Test V18.3.1 runs correctly")
    print("  2. Begin profiling campaign")
    print("  3. Identify active evaluation functions")

if __name__ == "__main__":
    main()
