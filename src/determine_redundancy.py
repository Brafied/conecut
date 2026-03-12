import logging
import torch
import numpy as np
import os
import pickle
from scipy.optimize import nnls
from sklearn.metrics import r2_score

logger = logging.getLogger(__name__)

def try_load_checkpoint(checkpoint_file_name):
    if not os.path.exists(checkpoint_file_name):
        return [], [], 0

    with open(checkpoint_file_name, "rb") as file:
        checkpoint_data = pickle.load(file)
        redundant_examples = checkpoint_data["redundant_examples"]
        non_redundant_examples = checkpoint_data["non_redundant_examples"]
        start_index = checkpoint_data["next_index"]

    return redundant_examples, non_redundant_examples, start_index

def non_negative_least_squares(y, X):
    atom_coefficients, _ = nnls(X, y)
    reconstruction = X @ atom_coefficients
    r2 = r2_score(y, reconstruction)

    return [atom_coefficients], [r2]

def non_negative_orthogonal_matching_pursuit(y, X, maximum_nonzero_coefficients):
    coefficients_list = []
    r2_list = []
    
    chosen_atom_indices = []
    residual = y.copy()
    for _ in range(maximum_nonzero_coefficients):
        atom_residual_correlations = X.T @ residual
        atom_residual_correlations[chosen_atom_indices] = -np.inf
        chosen_atom_index = np.argmax(atom_residual_correlations)
        if atom_residual_correlations[chosen_atom_index] <= 0:
            break
        chosen_atom_indices.append(chosen_atom_index)

        chosen_atoms = X[:, chosen_atom_indices]
        chosen_atom_coefficients, _ = nnls(chosen_atoms, y)
        reconstruction = chosen_atoms @ chosen_atom_coefficients
        residual = y - reconstruction

        atom_coefficients = np.zeros(X.shape[1])
        atom_coefficients[chosen_atom_indices] = chosen_atom_coefficients
        coefficients_list.append(atom_coefficients)
        r2 = r2_score(y, reconstruction)
        r2_list.append(r2)

    return coefficients_list, r2_list
    
def determine_redundancy(arguments, activation_differences):
    logger.info("Normalizing activation differences...")
    activation_differences = activation_differences.to(torch.float32).numpy()
    activation_differences /= np.linalg.norm(activation_differences, axis=1, keepdims=True)
    logger.info("Completed normalizing activation differences.")

    data_directory = f"data/{arguments.parsed_model_id}_{arguments.subset_filter}/{arguments.reconstruction_algorithm}_{arguments.epsilon}"
    if arguments.reconstruction_algorithm == "nnomp":
        data_directory += f"_{arguments.nnomp_maximum_nonzero_coefficients}"

    checkpoint_file_name = f"{data_directory}/checkpoint.pkl"

    logger.info("Attempting to load checkpoint...")
    redundant_examples, non_redundant_examples, start_index = try_load_checkpoint(checkpoint_file_name)

    if start_index > 0:        
        logger.info("Checkpoint found.")
    else:
        logger.info("Checkpoint not found.")

    logger.info(f"Beginning redundancy detection at example {start_index + 1}/{len(activation_differences)}.")
    logger.info(f"Current redundant count: {len(redundant_examples)}.")
    for i in range(start_index, len(activation_differences)):
        if i % 5 == 0 and i > start_index:
            logger.info(f"Processing example {i + 1}/{len(activation_differences)}...")
            logger.info(f"Current redundant count: {len(redundant_examples)}.")
        
        if i % 50 == 0 and i > start_index:
            logger.info("Saving checkpoint...")
            os.makedirs(data_directory, exist_ok=True)
            temporary_checkpoint_file_name = checkpoint_file_name + ".tmp"
            with open(temporary_checkpoint_file_name, "wb") as file:
                pickle.dump({
                    "redundant_examples": redundant_examples,
                    "non_redundant_examples": non_redundant_examples,
                    "next_index": i
                }, file)
            os.replace(temporary_checkpoint_file_name, checkpoint_file_name)
            logger.info("Completed saving checkpoint.")

        non_redundant_examples_mask = np.ones(len(activation_differences), dtype=bool)
        for redundant_example in redundant_examples:
            non_redundant_examples_mask[redundant_example["index"]] = False
        non_redundant_examples_mask[i] = False

        X = activation_differences[non_redundant_examples_mask].T
        y = activation_differences[i]

        if arguments.reconstruction_algorithm == "nnls":
            coefficients_list, r2_list = non_negative_least_squares(y, X)
        elif arguments.reconstruction_algorithm == "nnomp":
            coefficients_list, r2_list = non_negative_orthogonal_matching_pursuit(y, X, arguments.nnomp_maximum_nonzero_coefficients)

        index_to_coefficients_list = []
        for coefficients in coefficients_list:
            index_to_coefficients = {}
            for index, coefficient in zip(np.where(non_redundant_examples_mask)[0], coefficients):
                index_to_coefficients[index] = coefficient
            index_to_coefficients_list.append(index_to_coefficients)

        if r2_list[-1] >= arguments.epsilon:
            redundant_examples.append({
                "index": i,
                "r2_list": r2_list,
                "index_to_coefficients_list": index_to_coefficients_list
            })
        else:
            non_redundant_examples.append({
                "index": i,
                "r2_list": r2_list,
                "index_to_coefficients_list": index_to_coefficients_list
            })
    logger.info("Completed redundancy detection.")
    
    logging.info("Saving redundancy results...")
    os.makedirs(data_directory, exist_ok=True)
    with open(data_directory + "/redundant_examples.pkl", "wb") as file:
        pickle.dump(redundant_examples, file)
    with open(data_directory + "/non_redundant_examples.pkl", "wb") as file:
        pickle.dump(non_redundant_examples, file)
    logger.info("Completed saving redundancy results.")

    logger.info("Compiling redundancy statistics...")
    logger.info(f"Total redundant: {len(redundant_examples)}.")
    logger.info(f"Total non-redundant: {len(non_redundant_examples)}.")
    logger.info(f"Percent redundant: {(len(redundant_examples) / len(activation_differences)) * 100:.2f}%.")
    
    if os.path.exists(checkpoint_file_name):
        logger.info("Removing temporary checkpoint file...")
        os.remove(checkpoint_file_name)
        logger.info("Completed removing temporary checkpoint file.")
