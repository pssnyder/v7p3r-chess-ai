# Chess Puzzle Dataset Preparation
import pandas as pd
import numpy as np
import datetime
import os
import chess
import random

datestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
ONLY_POSITIVE_SENTIMENT = True  # Set to True to filter for only positive overall sentiment positions in the encoded dataset
DIFFICULTY_LEVEL = "beginner"  # Options: "beginner", "intermediate", "expert", or "all"
RATING_DEVIATION_THRESHOLD = 999  # Maximum rating deviation to include (lower means more consistent puzzles)
MIN_PLAY_THRESHOLD = 1  # Minimum number of plays for a puzzle to be included (higher means more popular puzzles)
POSITION_COUNT = 9999999  # Number of positions to include in the final training dataset, based on the top puzzles from the generated curriculum

def extract_pieces(fen):
    # Create a column for each piece type and its positions
    board = fen.split()[0]
    pieces = {p: [] for p in 'PNBRQKpnbrqk'}
    rows = board.split('/')
    for r, row in enumerate(rows):
        c = 0
        for char in row:
            if char.isdigit():
                c += int(char)
            else:
                pieces[char].append((r, c))
                c += 1
    return pieces

def extract_side_to_move(fen):
    # Create a column for the side to move
    return fen.split()[1]

def extract_castling_rights(fen):
    # Create a column for castling rights
    return fen.split()[2]

def extract_en_passant(fen):
    # Create a column for en passant target square
    return fen.split()[3]

def extract_halfmove_clock(fen):
    # Create a column for halfmove clock
    return fen.split()[4]

def extract_fullmove_number(fen):
    # Create a column for fullmove number
    return fen.split()[5]

def flatten_puzzle_dataset(df_puzzles):
    flattened_records = []
    
    # Convert to raw dictionaries instantly to avoid row-by-row serialization
    records = df_puzzles.to_dict('records')
    
    for row in records:
        puzzle_id = row['puzzle_id']
        start_fen = row['setup_fen']
        moves = row['moves_list']
        
        if not moves:
            continue
            
        board = chess.Board(start_fen)
        blunderer_color = "white" if board.turn == chess.WHITE else "black"
        winner_color = "black" if blunderer_color == "white" else "white"
        
        current_fen = start_fen
        sentiment_magnitude = 1
        
        for step_idx, move_str in enumerate(moves, start=1):
            try:
                move = chess.Move.from_uci(move_str)
            except ValueError:
                break
                
            if move in board.legal_moves:
                current_turn = "white" if board.turn == chess.WHITE else "black"
                board.push(move)
                next_fen = board.fen()
                
                if winner_color == "white":
                    w_sent, b_sent = sentiment_magnitude, -sentiment_magnitude
                else:
                    w_sent, b_sent = -sentiment_magnitude, sentiment_magnitude
                
                flattened_records.append({
                    'puzzle_id': puzzle_id,
                    'last_fen': current_fen,
                    'current_fen': next_fen,
                    'move_played': move_str,
                    'last_move_by': current_turn,
                    'to_play': "white" if board.turn == chess.WHITE else "black",
                    'to_win': winner_color,
                    'white_sentiment': w_sent,
                    'black_sentiment': b_sent,
                    'sequence_step': step_idx,
                    'overall_sentiment': w_sent if winner_color == "white" else b_sent
                })
                current_fen = next_fen
                sentiment_magnitude += 1
            else:
                break
                
    return pd.DataFrame(flattened_records)

def calculate_overall_sentiment(row):
    # Calculate overall sentiment for the position based on who is to play and who is winning
    if row['to_win'] == row['last_move_by']:
        return row['white_sentiment'] if row['to_win'] == 'white' else row['black_sentiment']
    else:
        return row['white_sentiment'] if row['to_win'] == 'black' else row['black_sentiment']

def sanity_check_sentiment_labeling(df):
    for idx, row in df.sample(5).iterrows():
        print(f"As part of puzzle {row['puzzle_id']}, after move {row['move_played']} by {row['last_move_by']}, the position transitioned from {row['last_fen']} to {row['current_fen']}")
        print(f"white to {"win" if row['white_sentiment'] > 0 else "lose (or draw)"} and black to {"win" if row['black_sentiment'] > 0 else "lose (or draw)"} and an overall sentiment score of {row['overall_sentiment']} for {row['last_move_by']}.")

def fen_to_tensor(fen):
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    }
    tensor = np.zeros((12, 8, 8), dtype=np.int8)
    parts = fen.split()
    rows = parts[0].split('/')
    for r, row in enumerate(rows):
        c = 0
        for char in row:
            if char.isdigit():
                c += int(char)
            else:
                idx = piece_map[char.upper()]
                if char.islower():
                    idx += 6  # Black pieces
                tensor[idx, r, c] = 1.0
                c += 1
    return tensor

def tensor_to_board(tensor):
    piece_map = {
        0: 'P', 1: 'N', 2: 'B', 3: 'R', 4: 'Q', 5: 'K',
    }
    board = [['.' for _ in range(8)] for _ in range(8)]
    for idx in range(12):
        for r in range(8):
            for c in range(8):
                if tensor[idx, r, c] == 1:
                    piece = piece_map[idx % 6]
                    if idx >= 6:
                        piece = piece.lower()  # Black pieces
                    board[r][c] = piece
    return board

def apply_priority_score(df, play_threshold=50):
    C = df['popularity'].mean()
    m = play_threshold
    df['training_priority'] = (
        (df['num_of_plays'] * df['popularity']) + (m * C)
    ) / (df['num_of_plays'] + m)
    return df.sort_values(by='training_priority', ascending=False)

def generate_training_curriculum(df_puzzles, min_elo=1000, max_elo=2000, max_rd=80, play_threshold=100):
    cohort = df_puzzles[
        (df_puzzles['elo_rating'] >= min_elo) & 
        (df_puzzles['elo_rating'] <= max_elo) & 
        (df_puzzles['rating_deviation'] <= max_rd) &
        (df_puzzles['num_of_plays'] >= play_threshold)
    ].copy()
    
    if cohort.empty:
        return cohort

    if cohort['popularity'].min() < 0:
        norm_raw_pop = (cohort['popularity'] + 100) / 200
    else:
        norm_raw_pop = cohort['popularity'] / 100
        
    global_pop_mean = norm_raw_pop.mean()
    m = play_threshold
    
    cohort['bayesian_popularity'] = (
        (cohort['num_of_plays'] * norm_raw_pop) + (m * global_pop_mean)
    ) / (cohort['num_of_plays'] + m)
    
    log_plays = np.log10(cohort['num_of_plays'] + 1)
    
    def min_max_scale(series):
        if series.max() == series.min():
            return 1.0
        return (series - series.min()) / (series.max() - series.min())
    
    final_pop_score = min_max_scale(cohort['bayesian_popularity'])
    final_play_score = min_max_scale(log_plays)
    
    cohort['training_priority'] = (0.70 * final_pop_score) + (0.30 * final_play_score)
    return cohort.sort_values(by='training_priority', ascending=False)

def flatten_solution_tensor(row):
    p_tensor = np.array(row['puzzle_tensor'], dtype=np.int8)
    s_tensor = np.array(row['solution_tensor'], dtype=np.int8)
    
    departure_grid = np.zeros((8, 8), dtype=np.int8)
    arrival_grid = np.zeros((8, 8), dtype=np.int8)
    
    before_flat = p_tensor.sum(axis=0)
    after_flat = s_tensor.sum(axis=0)
    
    dep_mask = (before_flat == 1) & (after_flat == 0)
    if dep_mask.any():
        departure_grid[dep_mask] = 1
        
    arr_mask = (before_flat == 0) & (after_flat == 1)
    if arr_mask.any():
        arrival_grid[arr_mask] = 1
    else:
        for channel in range(12):
            diff = s_tensor[channel] - p_tensor[channel]
            if (diff == 1).any():
                arrival_grid[diff == 1] = 1
                break

    return np.stack([departure_grid, arrival_grid], axis=0).tolist()

# =============================================================================

def main():
    global datestamp
    print("Step 1: Starting dataset loading and pre-processing...")
    load_path = r'data/lichess_db_puzzle.csv'
    
    print(f"--Loading and filtering puzzles straight from {load_path}...")
    
    if DIFFICULTY_LEVEL == "beginner":
        selected_min_elo, selected_max_elo = 0, 1300
    elif DIFFICULTY_LEVEL == "intermediate":
        selected_min_elo, selected_max_elo = 1301, 1800
    elif DIFFICULTY_LEVEL == "expert":
        selected_min_elo, selected_max_elo = 1801, 9999
    else:
        selected_min_elo, selected_max_elo = 0, 99999

    max_rating_deviation = RATING_DEVIATION_THRESHOLD
    min_play_threshold = MIN_PLAY_THRESHOLD

    filtered_chunks = []
    for chunk in pd.read_csv(load_path, chunksize=100000):
        chunk = chunk.rename(columns={
            'PuzzleId': 'puzzle_id', 'FEN': 'setup_fen', 'Moves': 'moves',
            'Rating': 'elo_rating', 'RatingDeviation': 'rating_deviation',
            'Popularity': 'popularity', 'NbPlays': 'num_of_plays',
            'Themes': 'themes', 'GameUrl': 'game_url', 'OpeningTags': 'opening_tags'
        })
        
        valid_rows = chunk[
            (chunk['elo_rating'] >= selected_min_elo) &
            (chunk['elo_rating'] <= selected_max_elo) &
            (chunk['rating_deviation'] <= max_rating_deviation) &
            (chunk['num_of_plays'] >= min_play_threshold)
        ]
        if not valid_rows.empty:
            filtered_chunks.append(valid_rows)

    if not filtered_chunks:
        print("❌ No puzzles matched your filter criteria.")
        return

    puzzles_df = pd.concat(filtered_chunks, ignore_index=True)
    print(f"----Info: Kept {len(puzzles_df):,} puzzles matching ELO and quality targets.")

    puzzles_df['moves_list'] = puzzles_df['moves'].fillna('').apply(lambda x: x.split())
    
    print("Step 2: Processing training curriculum priority...")
    if puzzles_df['popularity'].min() < 0:
        norm_raw_pop = (puzzles_df['popularity'] + 100) / 200
    else:
        norm_raw_pop = puzzles_df['popularity'] / 100
        
    global_pop_mean = norm_raw_pop.mean()
    m = 100
    puzzles_df['bayesian_popularity'] = ((puzzles_df['num_of_plays'] * norm_raw_pop) + (m * global_pop_mean)) / (puzzles_df['num_of_plays'] + m)
    
    def min_max_scale(series):
        if series.max() == series.min(): return 1.0
        return (series - series.min()) / (series.max() - series.min())
        
    final_pop_score = min_max_scale(puzzles_df['bayesian_popularity'])
    final_play_score = min_max_scale(np.log10(puzzles_df['num_of_plays'] + 1))
    puzzles_df['training_priority'] = (0.70 * final_pop_score) + (0.30 * final_play_score)
    
    puzzles_df = puzzles_df.sort_values(by='training_priority', ascending=False)
    puzzles_df = puzzles_df.head(POSITION_COUNT).reset_index(drop=True)
    
    print("Step 3: Flattening puzzle sequence steps into state transitions...")
    position_sentiment_df = flatten_puzzle_dataset(puzzles_df)
    
    split_dir = os.path.join("data", "encoded", "split")
    os.makedirs(split_dir, exist_ok=True)
    
    train_file = os.path.join(split_dir, f"chess_puzzle_training_dataset_{datestamp}_train.json")
    val_file   = os.path.join(split_dir, f"chess_puzzle_training_dataset_{datestamp}_val.json")
    test_file  = os.path.join(split_dir, f"chess_puzzle_training_dataset_{datestamp}_test.json")

    print("Step 4: Executing tensor serialization stream directly to storage...")
    
    random.seed(42)
    
    with open(train_file, 'w', encoding='utf-8') as f_train, \
         open(val_file, 'w', encoding='utf-8') as f_val, \
         open(test_file, 'w', encoding='utf-8') as f_test:
        
        chunk_size = 5000
        total_sub_rows = len(position_sentiment_df)
        
        for start_idx in range(0, total_sub_rows, chunk_size):
            sub_batch = position_sentiment_df.iloc[start_idx:start_idx + chunk_size].copy()
            
            # FIXED KEY ENCODINGS HERE: targeting 'last_fen' and 'current_fen' from flatten_puzzle_dataset
            sub_batch['puzzle_tensor'] = sub_batch['last_fen'].apply(fen_to_tensor)
            sub_batch['solution_tensor'] = sub_batch['current_fen'].apply(fen_to_tensor)
            
            sub_batch['solution_tensor'] = sub_batch.apply(flatten_solution_tensor, axis=1)
            
            export_batch = sub_batch[['puzzle_id', 'puzzle_tensor', 'solution_tensor', 'overall_sentiment']]
            
            for _, row in export_batch.iterrows():
                line_string = row.to_json() + "\n"
                
                roll = random.random()
                if roll < 0.80:
                    f_train.write(line_string)
                elif roll < 0.90:
                    f_val.write(line_string)
                else:
                    f_test.write(line_string)
                    
            print(f"----Streamed transitions {min(start_idx + chunk_size, total_sub_rows):,} / {total_sub_rows:,}...")

    print(f"\n🏆 Master Streaming Process Complete!")
    print(f"🥇 Train vector: {train_file}")
    print(f"🥈 Val vector:   {val_file}")
    print(f"🥉 Test vector:  {test_file}")

if __name__ == "__main__":
    main()