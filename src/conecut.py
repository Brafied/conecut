import logging
import argparse
import os
from generate_conecut_data import generate_conecut_data
import pickle
import numpy as np
from determine_redundancy import determine_redundancy

logger = logging.getLogger(__name__)

def parse_arguments():
    argument_parser = argparse.ArgumentParser()

    argument_parser.add_argument("--model_id", required=True, type=str, help="The Hugging Face model ID of the model whose representations will be used for redundancy detection.")
    argument_parser.add_argument("--subset_filter", required=True, choices=["chat", "chat_hard", "safety", "reasoning", "full"], help="The subset of RewardBench on which redundancy detection will be run.")
    argument_parser.add_argument("--determine_redundancy", choices=["positive", "negative"], help="Run redundancy detection after generating activations and scores, evaluating the model on the selected subset, and caching the generated data.")
    argument_parser.add_argument("--reconstruction_algorithm", choices=["nnls", "nnomp"], help="The reconstruction algorithm redundancy detection will use.")
    argument_parser.add_argument("--epsilon", type=float, help="The reconstruction coefficient of determination (r^2) value below which an example will be considered non-redundant.")
    argument_parser.add_argument("--nnomp_maximum_nonzero_coefficients", type=int, help="The maximum number of nonzero coefficients nnomp will use in its reconstruction.")

    arguments = argument_parser.parse_args()

    arguments.parsed_model_id = arguments.model_id.replace("/", "_")

    if arguments.determine_redundancy:
        if arguments.reconstruction_algorithm is None or arguments.epsilon is None:
            raise ValueError("When --determine_redundancy is set, both --reconstruction_algorithm and --epsilon must be provided.")
        if not (0.0 <= arguments.epsilon <= 1.0):
            raise ValueError("--epsilon must be between 0 and 1 inclusive.")
        if arguments.reconstruction_algorithm == "nnomp":
            if arguments.nnomp_maximum_nonzero_coefficients is None:
                raise ValueError("When --reconstruction_algorithm is set to 'nnomp', --nnomp_maximum_nonzero_coefficients must be provided.")
            if arguments.nnomp_maximum_nonzero_coefficients <= 0:
                raise ValueError("--nnomp_maximum_nonzero_coefficients must be greater than 0.")
        else:
            if arguments.nnomp_maximum_nonzero_coefficients is not None:
                raise ValueError("When --reconstruction_algorithm is not set to 'nnomp', --nnomp_maximum_nonzero_coefficients should not be provided.")
    else:
        if arguments.reconstruction_algorithm is not None or arguments.epsilon is not None or arguments.nnomp_maximum_nonzero_coefficients is not None:
            raise ValueError("When --determine_redundancy is not set, neither --reconstruction_algorithm, --epsilon, nor --nnomp_maximum_nonzero_coefficients should be provided.")

    return arguments

def configure_logging(arguments):
    os.makedirs("logging", exist_ok=True)

    logging_file_name = f"logging/{arguments.parsed_model_id}_{arguments.subset_filter}"
    if arguments.determine_redundancy:
        logging_file_name += f"_{arguments.determine_redundancy}_{arguments.reconstruction_algorithm}_{arguments.epsilon}"
        if arguments.reconstruction_algorithm == "nnomp":
            logging_file_name += f"_{arguments.nnomp_maximum_nonzero_coefficients}"
    else:
        logging_file_name += "_evaluation"
    logging_file_name += ".txt"

    logging.basicConfig(filename=logging_file_name, level=logging.INFO)

def get_conecut_data(arguments):
    logger.info(f"Attempting to load cached activation differences, scores, and subsets for {arguments.model_id} on subset {arguments.subset_filter}...")
    activation_differences, chosen_scores, rejected_scores, subsets = try_load_cache(arguments)

    if activation_differences is not None:
        logger.info("Cached activation differences, scores, and subsets found.")
        return activation_differences, chosen_scores, rejected_scores, subsets
    
    logger.info("Cached activation differences, scores, and subsets not found.")
    
    logger.info("Generating activation differences, scores, and subsets...")
    activation_differences, chosen_scores, rejected_scores, subsets = generate_conecut_data(arguments)
    logger.info("Completed generating activation differences, scores, and subsets.")

    logger.info("Caching activation differences, scores, and subsets...")
    save_cache(arguments, activation_differences, chosen_scores, rejected_scores, subsets)
    logger.info(f"Completed caching activation differences, scores, and subsets.")
        
    return activation_differences, chosen_scores, rejected_scores, subsets

def try_load_cache(arguments):
    data_directory = f"data/{arguments.parsed_model_id}_{arguments.subset_filter}"
    
    if not os.path.isdir(data_directory):
        return None, None, None, None

    with open(data_directory + "/activation_differences.pkl", "rb") as file:
        activation_differences = pickle.load(file)
    with open(data_directory + "/chosen_scores.pkl", "rb") as file:
        chosen_scores = pickle.load(file)
    with open(data_directory + "/rejected_scores.pkl", "rb") as file:
        rejected_scores = pickle.load(file)
    with open(data_directory + "/subsets.pkl", "rb") as file:
        subsets = pickle.load(file)

    return activation_differences, chosen_scores, rejected_scores, subsets

def save_cache(arguments, activation_differences, chosen_scores, rejected_scores, subsets):
    data_directory = f"data/{arguments.parsed_model_id}_{arguments.subset_filter}/"
    
    os.makedirs(data_directory, exist_ok=True)

    with open(data_directory + "/activation_differences.pkl", "wb") as file:
        pickle.dump(activation_differences, file)
    with open(data_directory + "/chosen_scores.pkl", "wb") as file:
        pickle.dump(chosen_scores, file)
    with open(data_directory + "/rejected_scores.pkl", "wb") as file:
        pickle.dump(rejected_scores, file)
    with open(data_directory + "/subsets.pkl", "wb") as file:
        pickle.dump(subsets, file)

def evaluate_model(chosen_scores, rejected_scores, subsets):
    logger.info(f"Evaluating model on subset...")

    correct_example_mask = (chosen_scores > rejected_scores).numpy()
    correct_example_indices = np.where(correct_example_mask)[0]

    logger.info(f"Accuracy: {len(correct_example_indices) / len(correct_example_mask):.4f} ({len(correct_example_indices)}/{len(correct_example_mask)}).")
    
    for subset in set(subsets):
        subset_example_indices = [i for i, current_subset in enumerate(subsets) if current_subset == subset]   
        subset_correct_example_indices = [i for i in subset_example_indices if i in correct_example_indices]
        logger.info(f"Subset {subset}: Accuracy: {len(subset_correct_example_indices) / len(subset_example_indices):.4f} ({len(subset_correct_example_indices)}/{len(subset_example_indices)}).")

def analyze_redundancy(arguments, activation_differences):
    if arguments.reconstruction_algorithm == "nnomp":
        logger.info(f"Running redundancy detection with reconstruction algorithm {arguments.reconstruction_algorithm}, epsilon {arguments.epsilon}, and maximum nonzero coefficients {arguments.nnomp_maximum_nonzero_coefficients}...")
    else:
        logger.info(f"Running redundancy detection with reconstruction algorithm {arguments.reconstruction_algorithm} and epsilon {arguments.epsilon}...")
    determine_redundancy(arguments, activation_differences)
    logger.info(f"Completed running redundancy detection.")

def main():
    arguments = parse_arguments()
    
    configure_logging(arguments)

    activation_differences, chosen_scores, rejected_scores, subsets = get_conecut_data(arguments)

    evaluate_model(chosen_scores, rejected_scores, subsets)

    if not arguments.determine_redundancy:
        return
    
    analyze_redundancy(arguments, activation_differences)

if __name__ == "__main__":
    main()
