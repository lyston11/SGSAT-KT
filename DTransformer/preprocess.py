import json
import os
import pandas as pd
from collections import defaultdict


def calculate_difficulties(data_path):
    """
    Calculate difficulties for problems and concepts based on student responses.
    Difficulty is defined as the ratio of incorrect answers to total answers.
    """
    problem_difficulty = defaultdict(lambda: {'correct': 0, 'total': 0})
    concept_difficulty = defaultdict(lambda: {'correct': 0, 'total': 0})

    # Read data
    with open(data_path, 'r') as file:
        while True:
            student_id = file.readline().strip()  # Read student ID line
            if not student_id:
                break  # Break if this line is empty (end of file)
            problems_line = file.readline().strip()  # Read problems line
            concepts_line = file.readline().strip()  # Read concepts line
            answers_line = file.readline().strip()  # Read answers line

            if not problems_line or not concepts_line or not answers_line:
                continue  # Skip to the next group if any lines are missing

            problem_ids = problems_line.split(',')
            concept_ids = concepts_line.split(',')
            answers = answers_line.split(',')

            # Process each problem, concept, and answer
            for pid, qid, ans in zip(problem_ids, concept_ids, answers):
                ans = int(ans)  # Convert answer from string to integer

                # Update problem difficulty
                problem_difficulty[pid]['total'] += 1
                if ans == 0:
                    problem_difficulty[pid]['correct'] += 1

                # Update concept difficulty
                concept_difficulty[qid]['total'] += 1
                if ans == 0:
                    concept_difficulty[qid]['correct'] += 1

    # Calculate difficulty as the ratio of incorrect answers
    problem_difficulty = {k: 1 - (v['correct'] / v['total']) for k, v in problem_difficulty.items()}
    concept_difficulty = {k: 1 - (v['correct'] / v['total']) for k, v in concept_difficulty.items()}

    return problem_difficulty, concept_difficulty


def save_difficulties(data_dir, output_dir):
    """
    Process all data files in the directory, calculate difficulties, and save them.
    """
    filenames = ['train.txt', 'test.txt']
    for filename in filenames:
        data_path = os.path.join(data_dir, filename)
        problem_diff, concept_diff = calculate_difficulties(data_path)

        # Save difficulties to JSON files
        with open(os.path.join(output_dir, f'{filename}_problem_difficulties.json'), 'w') as f:
            json.dump(problem_diff, f, indent=4)
        with open(os.path.join(output_dir, f'{filename}_concept_difficulties.json'), 'w') as f:
            json.dump(concept_diff, f, indent=4)


def parse_data_file(data_path):
    """
    Parse the data file with expected four lines per student entry.
    """
    data = []
    with open(data_path, 'r') as file:
        while True:
            student_id = file.readline().strip()  # Read student ID line
            if not student_id:
                break  # Break if this line is empty (end of file)
            problems_line = file.readline().strip()  # Read problems line
            concepts_line = file.readline().strip()  # Read concepts line
            answers_line = file.readline().strip()  # Read answers line

            if not problems_line or not concepts_line or not answers_line:
                continue  # Skip if any lines are missing

            problem_ids = problems_line.split(',')
            concept_ids = concepts_line.split(',')
            answers = answers_line.split(',')

            for pid, qid, ans in zip(problem_ids, concept_ids, answers):
                data.append({
                    'student_id': student_id,
                    'problem_id': pid,
                    'concept_id': qid,
                    'correct': int(ans)
                })

    return pd.DataFrame(data)


def load_difficulties(filename):
    """
    Load difficulties from a JSON file.
    """
    with open(filename, 'r') as file:
        difficulties = json.load(file)
    return difficulties


def merge_difficulties_with_data(data, problem_difficulties, concept_difficulties):
    """
    Merge difficulties with the original dataset.
    """
    problem_diff_df = pd.DataFrame(list(problem_difficulties.items()), columns=['problem_id', 'problem_difficulty'])
    concept_diff_df = pd.DataFrame(list(concept_difficulties.items()), columns=['concept_id', 'concept_difficulty'])

    data = data.merge(problem_diff_df, on='problem_id', how='left')
    data = data.merge(concept_diff_df, on='concept_id', how='left')

    return data


def main(data_dir, output_dir):
    """
    Main processing function to handle the entire workflow.
    """
    save_difficulties(data_dir, output_dir)  # Calculate and save difficulties first

    # Process each data file and merge difficulties
    filenames = ['train.txt', 'test.txt']
    for filename in filenames:
        data_path = os.path.join(data_dir, filename)
        raw_data = parse_data_file(data_path)

        problem_difficulties = load_difficulties(os.path.join(output_dir, f'{filename}_problem_difficulties.json'))
        concept_difficulties = load_difficulties(os.path.join(output_dir, f'{filename}_concept_difficulties.json'))

        enriched_data = merge_difficulties_with_data(raw_data, problem_difficulties, concept_difficulties)

        # Save the enriched data
        enriched_data.to_csv(os.path.join(output_dir, f'enriched_{filename[:-4]}.csv'), index=False)


if __name__ == "__main__":
    DATA_DIR = "data/algebra05"  # Specify your data directory path here
    OUTPUT_DIR = "data/algebra05"  # Specify your output directory path here
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main(DATA_DIR, OUTPUT_DIR)