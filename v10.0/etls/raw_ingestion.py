# Raw Data Ingestion

import os
from pathlib import Path
from collections import defaultdict
import re
import pandas as pd
import chess.pgn
import json
import pyarrow as pa
import pyarrow.parquet as pq
from dataset_analyzer import DatasetAnalyzer

# Analysis Mode & Configurations
ANALYSIS_MODE = True
FILE_LIMIT = 10
CSV_MAX_ROWS = 10  # Max rows to read from each CSV file for analysis (set to None for no limit)
JSONL_MAX_ROWS = 10  # Max lines to read from each JSONL file for analysis (set to None for no limit)
PGN_MAX_GAMES = 10  # Max games to read from each PGN file for analysis (set to None for no limit)

# Set null, min, and max values
NULL_FEN_HASH = 0
NULL_EVAL = 32767
MIN_EVAL = -32000
MAX_EVAL = 32000
NULL_DEPTH = 255
MIN_DEPTH = 0
MAX_DEPTH = 128
NULL_TIME = 4294967295
MIN_TIME = 0
MAX_TIME = 4294967294
NULL_CLOCK = 65535
MIN_CLOCK = 0
MAX_CLOCK = 7200
NULL_WDL = 127
MIN_WDL = -1
MAX_WDL = 1
NULL_MATERIAL = 65535
MIN_MATERIAL = 0
MAX_MATERIAL = 206
NULL_PHASE = 255
MIN_PHASE = 0
MAX_PHASE = 24
NULL_PIECE_COUNT = 255
MIN_PIECE_COUNT = 0
MAX_PIECE_COUNT = 32
NULL_FEN = "NULL"

# Piece Values
PAWN_VALUE = 100
KNIGHT_VALUE = 320
BISHOP_VALUE = 330
ROOK_VALUE = 500
QUEEN_VALUE = 900
KING_VALUE = 0

# Define the directories to scan
data_directories = [
    r"E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\game_records",
    r"E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\training_data"
]

# Initialize datasets for each file type
pgn_files = []
csv_files = []
json_files = []
jsonl_files = []
db_files = []

# Initialize datasets for structured data
pgn_df = pd.DataFrame()
csv_df = pd.DataFrame()
json_df = pd.DataFrame()
jsonl_df = pd.DataFrame()
db_df = pd.DataFrame()

# File type mapping
file_type_map = {
    '.pgn': pgn_files,
    '.csv': csv_files,
    '.json': json_files,
    '.jsonl': jsonl_files,
    '.db': db_files
}

# Initialize the analyzer
#data_analyzer = DatasetAnalyzer(parquet_dir="path/to/parquet/files")

# Scan directories
for directory in data_directories:
    dir_path = Path(directory)
    if dir_path.exists():
        for file_path in dir_path.rglob('*'):
            if file_path.is_file():
                file_extension = file_path.suffix.lower()
                if file_extension in file_type_map:
                    file_record = {
                        'filename': file_path.name,
                        'filepath': str(file_path),
                        'filetype': file_extension,
                        'filesize': file_path.stat().st_size,
                    }
                    file_type_map[file_extension].append(file_record)

# Calculate total sizes by file type
total_sizes = {}
for file_type, files in file_type_map.items():
    total_gb = sum(f['filesize'] for f in files) / (1024**3)
    total_sizes[file_type] = total_gb

if ANALYSIS_MODE:
    # Display results
    print(f"Found {len(pgn_files)} PGN files ({total_sizes['.pgn']:.2f} GB)")
    print(f"Found {len(csv_files)} CSV files ({total_sizes['.csv']:.2f} GB)")
    print(f"Found {len(json_files)} JSON files ({total_sizes['.json']:.2f} GB)")
    print(f"Found {len(jsonl_files)} JSONL files ({total_sizes['.jsonl']:.2f} GB)")
    print(f"Found {len(db_files)} DB files ({total_sizes['.db']:.2f} GB)")

    # Display sample records (max 2 per file type)
    print("SAMPLE PGN FILES (max 2):")
    for pgn in pgn_files[:2]:
        print(f"  {pgn['filename']} ({round(pgn['filesize']/1000000,2):,} MB)")

    print("SAMPLE CSV FILES (max 2):")
    for csv in csv_files[:2]:
        print(f"  {csv['filename']} ({round(csv['filesize']/1000000,2):,} MB)")

    print("SAMPLE JSON FILES (max 2):")
    for json_file in json_files[:2]:
        print(f"  {json_file['filename']} ({round(json_file['filesize']/1000000,2):,} MB)")

    print("SAMPLE JSONL FILES (max 2):")
    for jsonl in jsonl_files[:2]:
        print(f"  {jsonl['filename']} ({round(jsonl['filesize']/1000000,2):,} MB)")

    print("SAMPLE DB FILES (max 2):")
    for db in db_files[:2]:
        print(f"  {db['filename']} ({round(db['filesize']/1000000,2):,} MB)")


    # Calculate total pgn games
    pgn_game_counts = {}
    total_pgn_games = 0
    for pgn in pgn_files:
        try:
            with open(pgn['filepath'], 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                games = content.split('[Event ')[1:]  # Split on the start of each game
                pgn_game_counts[pgn['filename']] = len(games)
                total_pgn_games += len(games)
                if total_pgn_games >= PGN_MAX_GAMES:
                    print(f"Reached game limit of {PGN_MAX_GAMES} for PGN ingestion. Stopping further counting.")
                    break
        except Exception as e:
            print(f"Error reading {pgn['filename']}: {e}")
            pgn_game_counts[pgn['filename']] = 0

    print(f"Total PGN Games: {total_pgn_games}")

    # Calculate total csv positions
    csv_position_counts = {}
    total_csv_positions = 0
    for csv in csv_files:
        try:
            with open(csv['filepath'], 'r', encoding='utf-8', errors='ignore') as f:
                position_count = sum(1 for line in f) - 1  # Subtract header
                csv_position_counts[csv['filename']] = position_count
                total_csv_positions += position_count
                if total_csv_positions >= CSV_MAX_ROWS:
                    print(f"Reached position limit of {CSV_MAX_ROWS} for CSV ingestion. Stopping further counting.")
                    break
        except Exception as e:
            print(f"Error reading {csv['filename']}: {e}")
            csv_position_counts[csv['filename']] = 0

    print(f"Total CSV Positions: {total_csv_positions}")


    # Calculate total jsonl evaluations
    jsonl_evaluation_counts = {}
    total_jsonl_evaluations = 0
    if False: # too large to run in notebook
        for jsonl in jsonl_files:
            try:
                with open(jsonl['filepath'], 'r', encoding='utf-8', errors='ignore') as f:
                    evaluation_count = sum(1 for line in f)
                    jsonl_evaluation_counts[jsonl['filename']] = evaluation_count
                    total_jsonl_evaluations += evaluation_count
                    if total_jsonl_evaluations >= JSONL_MAX_ROWS:
                        print(f"Reached evaluation limit of {JSONL_MAX_ROWS} for JSONL ingestion. Stopping further counting.")
                        break
            except Exception as e:
                print(f"Error reading {jsonl['filename']}: {e}")
                jsonl_evaluation_counts[jsonl['filename']] = 0
        print(f"Total JSONL Evaluations: {total_jsonl_evaluations}")
    else:
        total_jsonl_evaluations = 388000000 # from previous runs, since counting is too slow in notebook
        print(f"Approximate JSONL Evaluations: {total_jsonl_evaluations}")

# ----------------
# Helper Functions
# ----------------
def extract_eval_and_depth(comment_text: str):
    """Extract eval (cp) and depth from comment text."""
    if not comment_text:
        return NULL_EVAL, NULL_DEPTH

    # [%eval 0.80,4]
    m = re.search(r"\[%eval\s+([^\],\s]+)(?:[,/](\d+))?\]", comment_text)
    if m:
        raw_eval = m.group(1).strip()
        depth = int(m.group(2)) if m.group(2) else 0

        if raw_eval.startswith("#"):
            mate_n = raw_eval[1:]
            if mate_n.startswith("-"):
                return MIN_EVAL, depth
            return MAX_EVAL, depth

        try:
            cp = int(round(float(raw_eval) * 100))
            cp = max(MIN_EVAL, min(MAX_EVAL, cp))
            return cp, depth
        except ValueError:
            pass
    
    # Eval: 0.14
    m = re.search(r"Eval:\s*([+-]?\d+(?:\.\d+)?)", comment_text)
    if m:
        try:
            cp = int(round(float(m.group(1)) * 100))
            cp = max(MIN_EVAL, min(MAX_EVAL, cp))
            return cp, NULL_DEPTH
        except ValueError:
            pass
    
    # (d2-d4 d7-d5 Ng1-f3) +0.70/3 1
    m = re.search(r"\)\s*([+-]?\d+(?:\.\d+)?)(?:[/,](\d+))?", comment_text)
    if m:
        try:
            cp = int(round(float(m.group(1)) * 100))
            cp = max(MIN_EVAL, min(MAX_EVAL, cp))
            depth = int(m.group(2)) if m.group(2) else 0
            return cp, depth
        except ValueError:
            pass
    return NULL_EVAL, NULL_DEPTH


def extract_clk_seconds(comment_text: str):
    """Extract [%clk ...] and return remaining clock time in seconds."""
    if not comment_text:
        return None

    m = re.search(r"\[%clk\s+([0-9:.]+)\]", comment_text)
    if not m:
        return None

    clk_str = m.group(1).strip()
    parts = clk_str.split(":")
    try:
        if len(parts) == 3:
            h = int(parts[0])
            mm = int(parts[1])
            ss = float(parts[2])
            total = h * 3600 + mm * 60 + ss
        elif len(parts) == 2:
            mm = int(parts[0])
            ss = float(parts[1])
            total = mm * 60 + ss
        else:
            total = float(parts[0])

        sec = int(round(total))
        return max(MIN_CLOCK, min(MAX_CLOCK, sec))
    except ValueError:
        return None


def parse_time_control(tc_str: str):
    """
    Parse PGN TimeControl header into (base_seconds, increment_seconds).
    Handles formats: '300+5', '300', '180+2', '-' (unknown), '1/40' (moves/time).
    Returns (base, increment) in seconds, or (None, None) if unparseable.
    """
    if not tc_str or tc_str in ("-", "?", ""):
        return None, None

    # Moves-based format: 40/9000 (ignore increment, use per-move estimate)
    moves_match = re.match(r"(\d+)/(\d+)", tc_str)
    if moves_match:
        moves = int(moves_match.group(1))
        total = int(moves_match.group(2))
        base_per_move = total / moves
        return base_per_move, 0

    # Standard format: base+increment or base
    tc_match = re.match(r"(\d+(?:\.\d+)?)(?:\+(\d+(?:\.\d+)?))?", tc_str)
    if tc_match:
        base = float(tc_match.group(1))
        increment = float(tc_match.group(2)) if tc_match.group(2) else 0.0
        return base, increment

    return None, None


def calculate_move_time(
    prev_clk_side: float | None,
    clk_remaining: float | None,
    increment: float,
    base_seconds: float | None,
    move_number: int,
) -> int:
    """
    Calculate time spent on a move in seconds.

    Uses clock difference when available, accounting for increment added
    after the move. Falls back to a time-control-based estimate when
    clock data is missing.

    Args:
        prev_clk_side: Clock reading before this move (seconds), or None.
        clk_remaining:  Clock reading after this move (seconds), or None.
        increment:      Per-move increment in seconds (0 if none).
        base_seconds:   Total base time for the game (seconds), or None.
        move_number:    Full-move number (used for fallback estimate).

    Returns:
        Estimated move time in miliseconds (clamped 0 to MAX_TIME).
    """
    # Primary: clock difference accounting for increment
    if prev_clk_side is not None and clk_remaining is not None:
        # time_used = prev_clock - curr_clock + increment (increment is added after move)
        time_used = prev_clk_side - clk_remaining + increment
        return max(MIN_TIME, min(MAX_TIME, int(round(time_used))))

    return NULL_TIME

def calculate_game_phase(fen: str) -> int:
    """
    Calculates the game phase from a FEN string based on non-pawn material.
    Returns an integer from 0 (Pure Endgame) to 24 (Pure Middlegame).
    """
    # 1. Define standard engine phase values for each piece type
    # (Kings and Pawns are inherently 0, so they are excluded)
    PHASE_VALUES = {
        'n': 1, 'b': 1, 'r': 2, 'q': 4,  # Black pieces
        'N': 1, 'B': 1, 'R': 2, 'Q': 4   # White pieces
    }
    
    # 2. Isolate the piece placement section of the FEN (the first component)
    piece_placement = fen.split()[0]
    
    # 3. Sum up the phase scores for all surviving pieces
    current_phase_score = 0
    for character in piece_placement:
        if character in PHASE_VALUES:
            current_phase_score += PHASE_VALUES[character]
            
    # 4. Safety catch: clip to maximum starting value (24) just in case a puzzle 
    # FEN has custom promotional pieces that exceed normal starting material.
    if current_phase_score > 24:
        current_phase_score = 24
        
    return current_phase_score


def ingest_pgn_file(pgn_filepath: str) -> pd.DataFrame:
    """
    Parses a PGN file and extracts structured data into a DataFrame.
    Processes ALL games in the file.
    """
    all_records = []
    
    with open(pgn_filepath, "r", encoding="utf-8", errors="ignore") as pgn_file:
        game_count = 0
        while True:
            game = chess.pgn.read_game(pgn_file)  # Read next game
            if game is None or game_count >= PGN_MAX_GAMES:  # No more games or hit limit
                break
            
            game_count += 1

            tc_str = game.headers.get("TimeControl", "")
            base_seconds, increment = parse_time_control(tc_str)

            result = game.headers.get("Result", "*")
            board = game.board()

            fallback_base = base_seconds if (base_seconds and base_seconds > 0) else 60.0
            prev_clk = {chess.WHITE: fallback_base, chess.BLACK: fallback_base}
            inc_val = increment if increment is not None else 0.0

            for node in game.mainline():
                move = node.move
                mover = board.turn
                
                board.push(move)
                fen = board.fen()
                full_move_number = board.fullmove_number
                combined_comment = node.comment if node.comment else ""
                
                # [Rest of your position processing stays the same]
                clk_remaining = extract_clk_seconds(combined_comment)
                node_clk = node.clock()
                if node_clk is not None:
                    clk_remaining = max(MIN_CLOCK, min(MAX_CLOCK, int(round(node_clk))))
                    
                move_time = calculate_move_time(
                    prev_clk_side=prev_clk[mover],
                    clk_remaining=clk_remaining,
                    increment=inc_val,
                    base_seconds=base_seconds,
                    move_number=full_move_number,
                )
                
                if clk_remaining is not None:
                    prev_clk[mover] = clk_remaining

                piece_map = board.piece_map()
                material = sum(
                    {1: PAWN_VALUE, 2: KNIGHT_VALUE, 3: BISHOP_VALUE, 4: ROOK_VALUE, 5: QUEEN_VALUE, 6: KING_VALUE}.get(p.piece_type, 0)
                    for p in piece_map.values()
                )
                piece_count = len(piece_map)

                current_turn = board.turn 

                if result == "1-0":
                    wdl = 1 if current_turn == chess.WHITE else -1
                elif result == "0-1":
                    wdl = 1 if current_turn == chess.BLACK else -1
                elif result == "1/2-1/2":
                    wdl = 0
                else:
                    wdl = NULL_WDL

                node_eval = node.eval()
                if node_eval is not None:
                    # .relative is a property (not a method)
                    parsed_cp = node_eval.relative.score(mate_score=MAX_EVAL)
                    
                    if parsed_cp is not None:
                        eval_cp = max(MIN_EVAL, min(MAX_EVAL, int(parsed_cp)))
                    else:
                        eval_cp = NULL_EVAL
                    
                    node_depth = node.eval_depth()
                    depth = int(node_depth) if node_depth is not None else NULL_DEPTH
                else:
                    # Fallback when node has no evaluation
                    eval_cp, depth = extract_eval_and_depth(combined_comment)

                all_records.append({
                    "fen_hash": hash(fen) & 0xFFFFFFFFFFFFFFFF,
                    "evaluation": eval_cp,
                    "depth": depth,
                    "time": move_time,
                    "clock": clk_remaining if clk_remaining is not None else NULL_CLOCK,
                    "wdl": wdl,
                    "material": material,
                    "phase": calculate_game_phase(fen),
                    "piece_count": piece_count,
                    "fen": fen
                })

    return pd.DataFrame(all_records)

def ingest_jsonl_file(jsonl_filepath: str) -> pd.DataFrame:
    """
    Parses a JSONL file containing chess position evaluations into a structured DataFrame.
    Expects each line to be a JSON object with at least 'fen' and 'evals' fields.
    """
    # Open the jsonl and read the first 10 records
    json_data = []
    with open(jsonl_filepath, "r", encoding="utf-8", errors="ignore") as jsonl_file:
        for _ in range(JSONL_MAX_ROWS):
            line = jsonl_file.readline()
            if not line:
                break
            try:
                record = json.loads(line)
                json_data.append(record)
            except json.JSONDecodeError:
                continue

    # Convert JSONL evaluation example to structured format
    jsonl_records = []

    # Loop through each individual record dictionary inside your list
    for record in json_data:
        
        # Safely look up fields to prevent KeyErrors
        fen_str = record.get("fen", "")
        evals = record.get("evals", [])
        
        # Fallback default values if the 'evals' list is empty
        cp_val = NULL_EVAL
        depth_val = NULL_DEPTH
        
        # FIX: Check if 'evals' exists and is a list before trying to access it
        if evals and isinstance(evals, list):
            # 1. Grab the highest search entry (the first item in the evals list)
            best_eval_entry = evals[0]
            depth_val = best_eval_entry.get("depth", NULL_DEPTH)
            
            # 2. Navigate into the 'pvs' list to grab the top move's centipawn score
            pvs_list = best_eval_entry.get("pvs", [])
            if pvs_list and isinstance(pvs_list, list):
                cp_val = pvs_list[0].get("cp", NULL_EVAL)

        # Calculate piece count from fen string (count pieces by counting uppercase and lowercase letters)
        piece_count = sum(c.isalpha() for c in fen_str)

        # Calculate material count from fen string
        material = sum(
            {"p": PAWN_VALUE, "n": KNIGHT_VALUE, "b": BISHOP_VALUE, "r": ROOK_VALUE, "q": QUEEN_VALUE, "k": KING_VALUE}.get(c.lower(), 0)
            for c in fen_str
        )

        # Calculate WDL from the perspective of the current side to move in the FEN
        board = chess.Board(fen_str)
        if (cp_val > 200 and board.turn == chess.WHITE) or (cp_val < -200 and board.turn == chess.BLACK):
            wdl = 1
        elif (cp_val < -200 and board.turn == chess.WHITE) or (cp_val > 200 and board.turn == chess.BLACK):
            wdl = -1
        else:
            wdl = 0


        # Append the structured record row
        jsonl_records.append(
            {
                "fen_hash": hash(fen_str) & 0xFFFFFFFFFFFFFFFF,
                "evaluation": cp_val,
                "depth": depth_val,
                "time": NULL_TIME,                       # No move time data in JSONL example
                "clock": NULL_CLOCK,                               # No clock data in JSONL example
                "wdl": wdl,                                 # WDL calculated from result and perspective
                "material": material,                     # Calculate material from fen string
                "phase": calculate_game_phase(fen_str),  # Calculated phase based on piece count
                "piece_count": piece_count,               # Fast calculation from fen string
                "fen": fen_str
            }
        )

    # Convert to DataFrame
    return pd.DataFrame(jsonl_records)

def ingest_csv_file(csv_filepath: str) -> pd.DataFrame:
    """Parses a CSV file containing chess puzzles into a structured DataFrame."""
    # --- Load the file straight into raw_df, sampling only the first 10 rows ---
    raw_df = pd.read_csv(csv_filepath, nrows=CSV_MAX_ROWS)

    # Convert csv puzzles to structured format
    csv_records = []

    # Process rows row-by-row
    for _, row in raw_df.iterrows():
        # 1. Initialize the board with this puzzle's starting FEN
        start_fen = str(row["FEN"]).strip()
        board = chess.Board(start_fen)
        
        # 2. Extract and split ALL available moves into an accessible list
        moves_list = str(row["Moves"]).strip().split()

        for p in range(4):  # Loop through plies 0 to 3 (0=starting position, 1=blunder, 2=solution, 3=post-solution)
            wdl = NULL_WDL  # Default WDL since we don't know the starting eval

            # 3. Step forward sequentially through plies
            if p > 0:
                # Ensure the puzzle sequence actually contains enough moves
                if len(moves_list) >= p:
                    # p=1 reads moves_list[0] (Blunder)
                    # p=2 reads moves_list[1] (Solution)
                    next_move = moves_list[p - 1] 
                    
                    try:                       
                        if p % 2 == 1:
                            wdl = 1
                        elif p % 2 == 0:
                            wdl = -1
                        move = board.parse_san(next_move)
                        board.push(move)  # State mutates normally to the next side's turn
                    except ValueError:
                        # If parse_san fails, it means we grabbed an illegal/wrong move
                        pass

            # 4. Gather the newly generated board parameters
            fen_str = board.fen()
            piece_map = board.piece_map()

            # 5. Calculate material score from piece map
            material_score = sum(
                {1: 100, 2: 320, 3: 330, 4: 500, 5: 900, 6: 0}.get(p.piece_type, 0) 
                for p in piece_map.values()
            )

            # Append mapped structured row entries
            csv_records.append(
                {
                    "fen_hash": hash(fen_str) & 0xFFFFFFFFFFFFFFFF,
                    "evaluation": NULL_EVAL,                    # Puzzles don't have evals
                    "depth": NULL_DEPTH,                           # Static puzzle starting points carry no search depth
                    "time": NULL_TIME,                     # No move timer fields present
                    "clock": NULL_CLOCK,                         # No running clocks active
                    "wdl": wdl,                             # No W/D/L data in CSV puzzle example
                    "material": material_score,
                    "phase": calculate_game_phase(fen_str),
                    "piece_count": len(piece_map),
                    "fen": fen_str
                }
            )

    # Wrap into the output DataFrame
    return pd.DataFrame(csv_records)

# PGN Ingestion
pgn_file_counter = 0
for pgn_file in pgn_files:
    pgn_file_counter += 1
    print(f"Parsing PGN file: {pgn_file['filename']} ({round(pgn_file['filesize']/1000000,2)} MB)")
    current_pgn_filepath = pgn_file['filepath']
    pgn_df = pd.concat([pgn_df, ingest_pgn_file(current_pgn_filepath)], ignore_index=True)
    if pgn_file_counter >= FILE_LIMIT:
        print(f"Reached file limit of {FILE_LIMIT} for PGN ingestion. Stopping further parsing.")
        break

# JSONL Ingestion
jsonl_file_counter = 0
for jsonl_file in jsonl_files:
    jsonl_file_counter += 1
    print(f"Parsing JSON file: {jsonl_file['filename']} ({round(jsonl_file['filesize']/1000000,2)} MB)")
    current_jsonl_filepath = jsonl_file['filepath']
    jsonl_df = pd.concat([jsonl_df, ingest_jsonl_file(current_jsonl_filepath)], ignore_index=True)
    if jsonl_file_counter >= FILE_LIMIT:
        print(f"Reached file limit of {FILE_LIMIT} for JSON ingestion. Stopping further parsing.")
        break

# CSV Ingestion
csv_file_counter = 0
for csv_file in csv_files:
    csv_file_counter += 1
    print(f"Parsing CSV file: {csv_file['filename']} ({round(csv_file['filesize']/1000000,2)} MB)")
    current_csv_filepath = csv_file['filepath']
    csv_df = pd.concat([csv_df, ingest_csv_file(current_csv_filepath)], ignore_index=True)
    if csv_file_counter >= FILE_LIMIT:
        print(f"Reached file limit of {FILE_LIMIT} for CSV ingestion. Stopping further parsing.")
        break

# Combine all parsed data into a single master dataset for unified processing
combined_df = pd.concat([pgn_df, jsonl_df, csv_df], ignore_index=True)
combined_df['reserved_bytes'] = [[0] * 66 for _ in range(len(combined_df))]

# Export 
# Define schema matching your 88-byte binary record
schema = pa.schema([
    ("fen_hash", pa.uint64()),         # 8 bytes - uint64
    ("evaluation", pa.int16()),       # 2 bytes - int16 (-32000 to 32000)
    ("depth", pa.uint8()),            # 1 byte  - uint8 (0 to 128)
    ("time", pa.uint32()),            # 4 bytes - uint32 (allows fallback to null=4294967295)
    ("clock", pa.uint16()),           # 2 bytes - uint16 (0 to 7200)
    ("wdl", pa.int8()),               # 1 byte  - int8 (1 / 0 / -1)
    ("material", pa.uint16()),        # 2 bytes - uint16 (0 to 206)
    ("phase", pa.uint8()),            # 1 byte  - uint8 (0 to 24)
    ("piece_count", pa.uint8()),       # 1 byte  - uint8 (0 to 32)
    
    # Strictly reserves fixed 66-byte space block per position row
    ("reserved_bytes", pa.list_(pa.uint8(), 66)), 
    
    # Kept temporarily for advanced feature extraction (will be dropped in step 2)
    ("fen", pa.string())
])

# Export with schema
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
combined_count = len(combined_df)
table = pa.Table.from_pandas(combined_df, schema=schema)
pa.parquet.write_table(
    table,
    f"v10.0/data/raw/ingested_data_{timestamp}_{combined_count}.parquet",
    compression='snappy'
)

if ANALYSIS_MODE:
    # Display results
    print(f"Parsed {len(pgn_df)} positions from PGN files")
    print(f"\nSchema:\n{pgn_df.dtypes}")
    print("\nSample records:")
    print(pgn_df.head(100))

    print(f"Parsed {len(jsonl_df)} positions from JSONL files")
    print(f"\nSchema:\n{jsonl_df.dtypes}")
    print("\nSample records:")
    print(jsonl_df.head(100))

    print(f"Parsed {len(csv_df)} puzzle rows from CSV source.")
    print(f"\nSchema:\n{csv_df.dtypes}")
    print("\nSample records:")
    print(csv_df.head(100))

    print(f"Combined dataset contains {len(combined_df)} total positions.")
