# Chess Puzzle Dataset Preparation
import pandas as pd
import numpy as np
import datetime
import os
import chess
import random
import datetime

datestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
ONLY_POSITIVE_SENTIMENT = True  # Set to True to filter for only positive overall sentiment positions in the encoded dataset
DIFFICULTY_LEVEL = "beginner"  # Options: "beginner", "intermediate", "expert", or "all"
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

# Display a position as a visual 8x8 board based on the tensors
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
    """
    Computes a Bayesian-smoothed priority score to rank puzzles for ML training.
    
    df requires:
      - 'popularity' (The raw popularity score/rating)
      - 'num_of_plays' (The number of times played)
    """
    # C = Global average popularity across all 4M puzzles
    C = df['popularity'].mean()
    
    # m = Tune this! Higher values pull low-play puzzles harder toward the mean.
    m = play_threshold
    
    # Apply the Bayesian formula
    df['training_priority'] = (
        (df['num_of_plays'] * df['popularity']) + (m * C)
    ) / (df['num_of_plays'] + m)
    
    # Sort descending so the absolute best/most reliable samples are at the top
    return df.sort_values(by='training_priority', ascending=False)

def generate_training_curriculum(df_puzzles, min_elo=1000, max_elo=2000, max_rd=80, play_threshold=100):
    """
    Creates a balanced curriculum priority index directly mapped to your column names:
    'elo_rating', 'rating_deviation', 'popularity', 'num_of_plays'
    """
    # Step 1: Filter down to your current ELO cohort target
    cohort = df_puzzles[
        (df_puzzles['elo_rating'] >= min_elo) & 
        (df_puzzles['elo_rating'] <= max_elo) & 
        (df_puzzles['rating_deviation'] <= max_rd)
    ].copy()
    
    if cohort.empty:
        return cohort

    # Step 2: Safe Bayesian Normalization for Lichess integer values
    # Converts popularity (e.g., 95) into a baseline 0.0 - 1.0 probability float
    # If your dataset contains negative popularities, this maps [-100, 100] -> [0, 1]
    # If it's already [0, 100], it safely maps it to [0, 1] as well.
    if cohort['popularity'].min() < 0:
        norm_raw_pop = (cohort['popularity'] + 100) / 200
    else:
        norm_raw_pop = cohort['popularity'] / 100
        
    global_pop_mean = norm_raw_pop.mean()
    m = play_threshold
    
    # Calculate Bayesian Popularity Index
    cohort['bayesian_popularity'] = (
        (cohort['num_of_plays'] * norm_raw_pop) + (m * global_pop_mean)
    ) / (cohort['num_of_plays'] + m)
    
    # Step 3: Normalize the Play Volumes using log scaling
    # This keeps a 100-play puzzle and an 8070-play puzzle comparable without breaking the math
    log_plays = np.log10(cohort['num_of_plays'] + 1)
    
    def min_max_scale(series):
        if series.max() == series.min():
            return 1.0
        return (series - series.min()) / (series.max() - series.min())
    
    final_pop_score = min_max_scale(cohort['bayesian_popularity'])
    final_play_score = min_max_scale(log_plays)
    
    # Step 4: Combine into a single unified Training Priority scalar
    # 70% emphasis on highly stable upvoted popularity, 30% on high play interaction volume
    cohort['training_priority'] = (0.70 * final_pop_score) + (0.30 * final_play_score)
    
    # Sort descending so your premium verified data rows rise straight to the top
    return cohort.sort_values(by='training_priority', ascending=False)

def flatten_solution_tensor(row):
    """
    Vectorized difference tracking utilizing fast matrix math instead of deep loops.
    """
    p_tensor = np.array(row['puzzle_tensor'], dtype=np.int8)
    s_tensor = np.array(row['solution_tensor'], dtype=np.int8)
    
    departure_grid = np.zeros((8, 8), dtype=np.int8)
    arrival_grid = np.zeros((8, 8), dtype=np.int8)
    
    # Compress 12 channels into flat 2D views of before vs after
    before_flat = p_tensor.sum(axis=0)
    after_flat = s_tensor.sum(axis=0)
    
    # Departure is where a piece disappeared completely
    dep_mask = (before_flat == 1) & (after_flat == 0)
    if dep_mask.any():
        departure_grid[dep_mask] = 1
        
    # Arrival is where a piece appeared on a previously empty square
    arr_mask = (before_flat == 0) & (after_flat == 1)
    if arr_mask.any():
        arrival_grid[arr_mask] = 1
    else:
        # Capture scenario fallback: item moved to an already filled coordinate
        for channel in range(12):
            diff = s_tensor[channel] - p_tensor[channel]
            if (diff == 1).any():
                arrival_grid[diff == 1] = 1
                break

    return np.stack([departure_grid, arrival_grid], axis=0).tolist()

# =============================================================================

def main():
    global datestamp
    
    # Step 1: Load and preprocess the dataset
    print("Step 1: Starting dataset loading and pre-processing...")

    # Load the dataset
    load_path = r'data/lichess_db_puzzle.csv'
    print(f"--Loading dataset from {load_path}...")

    load_count = POSITION_COUNT * 10  # Load more puzzles than needed to ensure we have enough after filtering for the target ELO range and sentiment
    load_count = min(load_count, 5000000)  # Cap at 5 million to avoid memory issues during development, adjust as needed for final runs
    print(f"--Loading the top {load_count} puzzles from the dataset for processing...")

    puzzles_df = pd.read_csv(load_path, nrows=load_count)

    actual_loaded_count = len(puzzles_df)
    print(f"----Info: Loaded {actual_loaded_count} puzzles from the dataset.")

    # Rename columns for clarity
    print("--Renaming columns for clarity...")
    puzzles_df.rename(columns={'PuzzleId': 'puzzle_id', 'FEN': 'setup_fen', 'Moves': 'moves', 'Rating': 'elo_rating', 'RatingDeviation': 'rating_deviation', 'Popularity': 'popularity', 'NbPlays': 'num_of_plays', 'Themes': 'themes', 'GameUrl': 'game_url', 'OpeningTags': 'opening_tags'}, inplace=True)

    # Extracting Metadata
    print("Step 2: Extracting metadata and features from the dataset...")

    print("--Extracting move sequences and categorical features...")
    # Split the Move sequence into individual moves
    puzzles_df['moves_list'] = puzzles_df['moves'].str.split()

    # Split the Themes sequence into individual themes
    puzzles_df['themes_list'] = puzzles_df['themes'].str.split()

    # Split the Opening Tags into individual tags
    puzzles_df['openings_list'] = puzzles_df['opening_tags'].str.split()

    # Replace NaN values with empty lists to avoid issues when splitting
    print("----Subtask: Handling missing values in list columns...")
    puzzles_df['moves_list'] = puzzles_df['moves_list'].apply(lambda x: x if isinstance(x, list) else [])
    puzzles_df['themes_list'] = puzzles_df['themes_list'].apply(lambda x: x if isinstance(x, list) else [])
    puzzles_df['openings_list'] = puzzles_df['openings_list'].apply(lambda x: x if isinstance(x, list) else [])


    # FEN Extractions
    print("--Extracting positional features from FEN strings...")
    # Load the extracted features into the new dataframe
    puzzles_df['pieces_list'] = puzzles_df['setup_fen'].apply(extract_pieces)
    puzzles_df['side_to_move'] = puzzles_df['setup_fen'].apply(extract_side_to_move)
    puzzles_df['castling_rights'] = puzzles_df['setup_fen'].apply(extract_castling_rights)
    puzzles_df['en_passant'] = puzzles_df['setup_fen'].apply(extract_en_passant)
    puzzles_df['halfmove_clock'] = puzzles_df['setup_fen'].apply(extract_halfmove_clock)
    puzzles_df['fullmove_number'] = puzzles_df['setup_fen'].apply(extract_fullmove_number)

    # Position Labeling
    print("--Extracting position transitions into puzzle states...")
    position_sentiment_df = flatten_puzzle_dataset(puzzles_df)
    print("--Extracting sentiment labeling features from puzzles...")
    position_sentiment_df['overall_sentiment'] = position_sentiment_df.apply(calculate_overall_sentiment, axis=1)

    # Sanity check for a few random puzzles to verify the sentiment labeling logic is consistent with the expected outcomes based on the move sequences and locked identities.
    #print("--Performing sanity checks on extracted data...")
    #sanity_check_sentiment_labeling(position_sentiment_df)

    # Encoding
    print("Step 3: Encoding dataset for model interpretation...")
    # Encode position fens into a 12x8x8 binary tensor representation for model input
    encoded_positions_df = position_sentiment_df[['puzzle_id','overall_sentiment']].copy() # Create a new dataframe to hold the encoded FEN tensors
    print("--Encoding puzzle positions into tensor representations...")
    encoded_positions_df['puzzle_tensor'] = position_sentiment_df['last_fen'].apply(fen_to_tensor)
    print("--Encoding solution positions into tensor representations...")
    encoded_positions_df['solution_tensor'] = position_sentiment_df['current_fen'].apply(lambda fen: fen_to_tensor(fen) if fen else None)

    
    if ONLY_POSITIVE_SENTIMENT:
        # Drop all negative overall sentiment samples from the encoded dataset to focus on puzzles seeking winning positions for the player to move, which should provide clearer signals for the model to learn from.
        print("----Optional Subtask: Filtering encoded dataset to focus on positive overall sentiment positions...")
        encoded_positions_df = encoded_positions_df[encoded_positions_df['overall_sentiment'] > 0].reset_index(drop=True)

    # Training Selection
    # Bayesian-Smoothed Priority Scoring for Training Sample Selection
    print("Step 4: Final training dataset refinement and selection...")
    
    # Use the generated priority scores to create a training curriculum that focuses on the most valuable puzzles for the target Elo range, while also ensuring a good mix of puzzle difficulties and themes.
    selected_min_elo = 0
    selected_max_elo = 4000
    if DIFFICULTY_LEVEL == "beginner":
        selected_max_elo = 1200
    elif DIFFICULTY_LEVEL == "intermediate":
        selected_min_elo = 1201
        selected_max_elo = 2400
    elif DIFFICULTY_LEVEL == "expert":
        selected_min_elo = 2401
    print(f"--Generating training curriculum for ELO range {selected_min_elo} to {selected_max_elo} with play threshold of 100...")
    selected_dataset = generate_training_curriculum(puzzles_df, min_elo=selected_min_elo, max_elo=selected_max_elo, play_threshold=100)
    
    # Create a new training dataset of positions based on the top puzzle id's in the selected curriculum, the new dataset should only contain the puzzle id, encoded puzzle tensor, encoded solution tensor, and the overall sentiment score.
    print(f"--Selecting the top puzzles for the final training dataset...")
    top_selected_puzzles = selected_dataset.head(POSITION_COUNT)['puzzle_id']
    training_dataset = position_sentiment_df[position_sentiment_df['puzzle_id'].isin(top_selected_puzzles)][['puzzle_id', 'last_fen', 'current_fen', 'move_played', 'last_move_by', 'to_play', 'to_win', 'white_sentiment', 'black_sentiment', 'sequence_step', 'overall_sentiment']].reset_index(drop=True)
    training_dataset_encoded = encoded_positions_df[encoded_positions_df['puzzle_id'].isin(top_selected_puzzles)][['puzzle_id', 'puzzle_tensor', 'solution_tensor', 'overall_sentiment']].reset_index(drop=True)

    # Final Dataset Preparation
    print("Step 5: Final dataset preparation and export...")
    # Flatten the solution tensor into a 2-channel matrix layer 2 x 8 x 8, which completely isolates the move's action from the rest of the board:
    print("--Flattening solution tensors to isolate move actions...")
    training_dataset_encoded['solution_tensor'] = training_dataset_encoded.apply(flatten_solution_tensor, axis=1)

    # Dataframe cleanup
    print("--Cleaning up intermediate dataframes to focus on the original data and extracted features...")
    puzzles_df.drop(columns=['moves', 'themes', 'opening_tags'], inplace=True) # Drop the intermediate list columns to focus on the original data and the extracted features


    # Export the final dataset to a new json file for use in model training
    final_position_count = len(training_dataset_encoded)
    filename = os.path.join(f"data/",f"chess_puzzle_training_dataset_{final_position_count}_{datestamp}.json")
    encoded_positions_df.to_json(filename, orient='records', lines=True)
    print(f"----Info: Original dataset saved to {filename}")
    # Export the raw dataset
    print(f"--Exporting the final raw dataset...")
    raw_filename = os.path.join(f"data/raw/",f"chess_puzzle_training_dataset_{final_position_count}_{datestamp}_raw.json")
    training_dataset.to_json(raw_filename, orient='records', lines=True)
    print(f"----Info: Final raw dataset saved to {raw_filename}")
    # Set the second type of data we will save
    print(f"--Exporting the final encoded dataset...")
    encoded_filename = os.path.join(f"data/encoded/",f"chess_puzzle_training_dataset_{final_position_count}_{datestamp}_encoded.json")
    training_dataset_encoded.to_json(encoded_filename, orient='records', lines=True)
    print(f"----Info: Final encoded dataset saved to {encoded_filename}")

    # Data Splitting
    print("Step 6: Splitting the dataset into training, validation, and testing sets...")
    
    # 1. Read all lines out of the JSON Lines file cleanly as raw strings
    with open(filename, 'r', encoding='utf-8') as f:
        original_lines = f.readlines()
    original_count = len(original_lines)
    with open(raw_filename, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()
    encoded_count = len(raw_lines)
    with open(encoded_filename, 'r', encoding='utf-8') as f:
        encoded_lines = f.readlines()
    encoded_count = len(encoded_lines)

    # 2. Build your output file paths dynamically
    base_name = os.path.basename(filename)
    split_dir = os.path.join("data", "encoded", "split")
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
        
    train_output_path = os.path.join(split_dir, base_name.replace(".json", "_train.json"))
    val_output_path = os.path.join(split_dir, base_name.replace(".json", "_val.json"))
    test_output_path = os.path.join(split_dir, base_name.replace(".json", "_test.json"))

    # 3. Create shuffled index mapping for reproducibility
    print("--Selecting random seed of 42 and shuffling indicies...")
    indices = list(range(encoded_count))
    random.seed(42)
    random.shuffle(indices)

    # 4. Calculate split boundary sizes
    print("--Calculating splits for dataset...")
    train_count = int(encoded_count * 0.8)
    remaining = encoded_count - train_count
    val_count = remaining // 2
    print(f"--Creating splits with {train_count} training samples, {val_count} validation samples, and {encoded_count - train_count - val_count} testing samples...")
    train_idx = set(indices[:train_count])
    val_idx = set(indices[train_count:train_count + val_count])
    test_idx = set(indices[train_count + val_count:])

    # 5. Distribute lines sequentially based on index mapping
    print("--Saving positions into training, validation, and testing files...")
    with open(train_output_path, 'w', encoding='utf-8') as f_train, \
        open(val_output_path, 'w', encoding='utf-8') as f_val, \
        open(test_output_path, 'w', encoding='utf-8') as f_test:
        
        for i, line in enumerate(encoded_lines):
            # We preserve the JSON Lines structure (no arrays, no commas) 
            # so your upcoming model data loader can stream them efficiently.
            if i in train_idx:
                f_train.write(line)
            elif i in val_idx:
                f_val.write(line)
            elif i in test_idx:
                f_test.write(line)

    print("\n--- Final Training Preparation Summary ---")
    print(f"🎯 Original Dataset: {filename} ({original_count} rows)")
    print(f"📊 Raw Dataset: {raw_filename} ({encoded_count} rows)")
    print(f"📊 Encoded Dataset: {encoded_filename} ({encoded_count} rows)")
    print(f"🔢📚 Training Curriculum: ELO {selected_min_elo} to {selected_max_elo}, top {final_position_count} {DIFFICULTY_LEVEL} puzzles selected")
    print(f"📊 Training/Validation/Testing Split: 80% / 10% / 10%")
    print(f"🥇 Training Set Vector:   {train_output_path} ({len(train_idx)} rows)")
    print(f"🥈 Validation Set Vector: {val_output_path} ({len(val_idx)} rows)")
    print(f"🥉 Testing Set Vector:    {test_output_path} ({len(test_idx)} rows)")


if __name__ == "__main__":
    main()