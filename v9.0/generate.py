# Generation System for Chess Puzzle Dataset
import os
import random
import json
import datetime

GENERATE_DATASET = True  # Set to True to generate a dataset of puzzles with solutions
GEN_DATASET_SIZE = 1000000  # Number of puzzles to generate for the dataset
GEN_BLANKS = range(10, 41)  # Number of 
GEN_MAX_SOLUTIONS = 1  # Maximum number of solutions allowed for each generated puzzle (1 for unique solution)
GEN_DATESTAMP = datetime.datetime.now().strftime("%Y%m%d%H%M") # Timestamp for dataset versioning





def main(generate_dataset=GENERATE_DATASET, dataset_size=GEN_DATASET_SIZE, num_blanks=GEN_BLANKS, max_solutions=GEN_MAX_SOLUTIONS, gen_datestamp=GEN_DATESTAMP):
    if generate_dataset:
        # Create a puzzle dataset with X puzzles each having Y blanks and at most Z solution
        blanks_label = f"{num_blanks.start}-{num_blanks.stop-1}" if isinstance(num_blanks, range) else str(num_blanks)
        print(f"\nGenerating {dataset_size} puzzle dataset with blanks range {blanks_label} (this may take a while)...")
        
        puzzle_dataset = generate_puzzles_with_solution_count(dataset_size, num_blanks, max_solutions)
        print(f"Generated {len(puzzle_dataset)} puzzles.")

        # Export the dataset to a json file
        # Use absolute path relative to this script to avoid folder location issues
        script_dir = os.path.dirname(os.path.abspath(__file__))
        export_dir = os.path.join(script_dir, "data")
        
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
            
        dataset_path = os.path.join(export_dir, f"chess_puzzle_dataset_{dataset_size}_{blanks_label}_{max_solutions}_{gen_datestamp}.json")
        with open(dataset_path, 'w') as f:
            json.dump(puzzle_dataset, f, indent=2)
        print(f"Dataset saved to {dataset_path}")
    else:
        # Test functionality
        print("Testing chess Solver with a sample board:")
        generated_board = generate_board()
        print("Generated Board:")
        for row in generated_board:
            print(row)

        puzzle = generate_puzzle(generated_board, 40)
        print("\nGenerated Puzzle:")
        for row in puzzle:
            print(row)

        solution = brute_force_solve(puzzle)
        print("\nSolved Puzzle:")
        for row in solution:
            print(row)

        solution_check = "Yes" if validate_solution(puzzle, solution) else "No"
        print(f"\nIs the solution valid? {solution_check}")

        solutions_tracker = []
        count_solutions_copy = [row[:] for row in puzzle]
        count_solutions(count_solutions_copy, solutions_tracker)
        print(f"Number of solutions found: {len(solutions_tracker)}")

if __name__ == "__main__":    
    main()