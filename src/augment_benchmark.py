import logging
import argparse
import os
import pickle
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from generate_conecut_data import apply_chat_template, run_inference
from determine_redundancy import non_negative_least_squares, non_negative_orthogonal_matching_pursuit, try_load_checkpoint

logger = logging.getLogger(__name__)

def parse_arguments():
    argument_parser = argparse.ArgumentParser()
    
    argument_parser.add_argument("--model_id", required=True, type=str, help="The Hugging Face model ID of the model whose representations will be used for redundancy detection.")
    argument_parser.add_argument("--reconstruction_algorithm", required=True, choices=["nnls", "nnomp"], help="The reconstruction algorithm redundancy detection will use.")
    argument_parser.add_argument("--epsilon", required=True, type=float, help="The reconstruction coefficient of determination (r^2) value below which an example will be considered non-redundant.")
    argument_parser.add_argument("--nnomp_maximum_nonzero_coefficients", type=int, help="The maximum number of nonzero coefficients nnomp will use in its reconstruction.")

    arguments = argument_parser.parse_args()

    arguments.parsed_model_id = arguments.model_id.replace("/", "_")

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

    return arguments

def configure_logging(arguments):
    os.makedirs("logging", exist_ok=True)

    logging_file_name = f"logging/augmentation_{arguments.parsed_model_id}_{arguments.reconstruction_algorithm}_{arguments.epsilon}"
    if arguments.reconstruction_algorithm == "nnomp":
        logging_file_name += f"_{arguments.nnomp_maximum_nonzero_coefficients}"
    logging_file_name += ".txt"

    logging.basicConfig(filename=logging_file_name, level=logging.INFO)

def get_conecut_data(arguments):
    logger.info(f"Loading cached activation differences, redundant examples, and non-redundant examples for {arguments.model_id} on RewardBench...")

    data_directory = f"data/{arguments.parsed_model_id}_full"

    with open(data_directory + "/activation_differences.pkl", "rb") as file:
        activation_differences = pickle.load(file)
    
    data_directory = f"{data_directory}/{arguments.reconstruction_algorithm}_{arguments.epsilon}"
    if arguments.reconstruction_algorithm == "nnomp":
        data_directory += f"_{arguments.nnomp_maximum_nonzero_coefficients}"
        
    with open(data_directory + "/redundant_examples.pkl", "rb") as file:
        redundant_examples = pickle.load(file)
    with open(data_directory + "/non_redundant_examples.pkl", "rb") as file:
        non_redundant_examples = pickle.load(file)

    logger.info("Completed loading cached activation differences, redundant examples, and non-redundant examples.")

    return activation_differences, redundant_examples, non_redundant_examples

def get_augmentation_data(arguments):
    logger.info(f"Attempting to load cached activation differences for {arguments.model_id} on the augmentation dataset.")
    activation_differences = try_load_cache(arguments)

    if activation_differences is not None:
        logger.info("Cached activation differences found.")
        return activation_differences
    
    logger.info("Cached activation differences not found.")
    
    logger.info("Generating activation differences...")
    activation_differences = generate_augmentation_data()
    logger.info("Completed generating activation differences.")

    logger.info("Caching activation differences...")
    save_cache(arguments, activation_differences)
    logger.info(f"Completed caching activation differences.")
        
    return activation_differences

def try_load_cache(arguments):
    data_directory = f"data/{arguments.parsed_model_id}_augmentation_dataset"
    
    if not os.path.isdir(data_directory):
        return None

    with open(data_directory + "/activation_differences.pkl", "rb") as file:
        activation_differences = pickle.load(file)

    return activation_differences

def save_cache(arguments, activation_differences):
    data_directory = f"data/{arguments.parsed_model_id}_augmentation_dataset/"
    
    os.makedirs(data_directory, exist_ok=True)

    with open(data_directory + "/activation_differences.pkl", "wb") as file:
        pickle.dump(activation_differences, file)

def generate_augmentation_data():
    logger.info("Loading training dataset, model, and tokenizer...")
    dataset = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned", split="train")
    dataset = dataset.select(range(1000))
    model = AutoModelForSequenceClassification.from_pretrained(
        "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2",
        num_labels=1,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("Skywork/Skywork-Reward-V2-Llama-3.1-8B")
    logger.info("Completed loading dataset, model, and tokenizer.")

    logger.info("Applying chat template to dataset...")
    templated_chosen_examples = []
    templated_rejected_examples = []
    for example in dataset:
        templated_chosen_examples.append(apply_chat_template(example["prompt"], example["chosen"][1]["content"], tokenizer))
        templated_rejected_examples.append(apply_chat_template(example["prompt"], example["rejected"][1]["content"], tokenizer))
    logger.info("Completed applying chat template to dataset.")

    logger.info("Running inference on the templated examples...")
    _, chosen_activations = run_inference(templated_chosen_examples, model, tokenizer)
    _, rejected_activations = run_inference(templated_rejected_examples, model, tokenizer)
    logger.info("Completed running inference on the templated examples.")

    return chosen_activations - rejected_activations

def augment_redundancy(arguments, activation_differences, redundant_examples, non_redundant_examples, augmentation_activation_differences):
    logger.info("Normalizing base and augmentation activation differences...")
    activation_differences = activation_differences.to(torch.float32).numpy()
    activation_differences /= np.linalg.norm(activation_differences, axis=1, keepdims=True)
    augmentation_activation_differences = augmentation_activation_differences.to(torch.float32).numpy()
    augmentation_activation_differences /= np.linalg.norm(augmentation_activation_differences, axis=1, keepdims=True)
    logger.info("Completed normalizing base and augmentation activation differences.")

    logger.info("Concatenating base and augmentation activation differences...")
    combined_activation_differences = np.concatenate((activation_differences, augmentation_activation_differences), axis=0)
    logger.info("Completed concatenating base and augmentation activation differences.")

    data_directory = f"data/{arguments.parsed_model_id}_augmentation_dataset/{arguments.reconstruction_algorithm}_{arguments.epsilon}"
    if arguments.reconstruction_algorithm == "nnomp":
        data_directory += f"_{arguments.nnomp_maximum_nonzero_coefficients}"

    checkpoint_file_name = f"{data_directory}/checkpoint.pkl"

    logger.info("Attempting to load checkpoint...")
    checkpoint_redundant_examples, checkpoint_non_redundant_examples, start_index = try_load_checkpoint(checkpoint_file_name)

    if start_index > 0:        
        logger.info("Checkpoint found.")
        redundant_examples = checkpoint_redundant_examples
        non_redundant_examples = checkpoint_non_redundant_examples
    else:
        logger.info("Checkpoint not found.")
        redundant_examples = redundant_examples.copy()
        non_redundant_examples = non_redundant_examples.copy()
        start_index = len(activation_differences)

    logger.info(f"Beginning redundancy augmentation at example {start_index + 1}/{len(combined_activation_differences)}.")
    logger.info(f"Current redundant count: {len(redundant_examples)}.")
    for i in range(start_index, len(combined_activation_differences)):
        if i % 5 == 0 and i > start_index:
            logger.info(f"Processing example {i + 1}/{len(combined_activation_differences)}...")
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

        non_redundant_examples_mask = np.zeros(len(combined_activation_differences), dtype=bool)
        for non_redundant_example in non_redundant_examples:
            non_redundant_examples_mask[non_redundant_example["index"]] = True

        X = combined_activation_differences[non_redundant_examples_mask].T
        y = combined_activation_differences[i]

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
    
    logger.info("Saving redundancy results...")
    os.makedirs(data_directory, exist_ok=True)
    with open(data_directory + "/redundant_examples.pkl", "wb") as file:
        pickle.dump(redundant_examples, file)
    with open(data_directory + "/non_redundant_examples.pkl", "wb") as file:
        pickle.dump(non_redundant_examples, file)
    logger.info("Completed saving redundancy results.")

    logger.info("Compiling redundancy statistics...")
    augmentation_redundant = [redundant_example for redundant_example in redundant_examples if redundant_example["index"] >= len(activation_differences)]
    augmentation_non_redundant = [non_redundant_example for non_redundant_example in non_redundant_examples if non_redundant_example["index"] >= len(activation_differences)]

    logger.info(f"Total augmentation redundant: {len(augmentation_redundant)}.")
    logger.info(f"Total augmentation non-redundant: {len(augmentation_non_redundant)}.")
    logger.info(f"Percent augmentation redundant: {(len(augmentation_redundant) / len(augmentation_activation_differences)) * 100:.2f}%.")
    
    if os.path.exists(checkpoint_file_name):
        logger.info("Removing temporary checkpoint file...")
        os.remove(checkpoint_file_name)
        logger.info("Completed removing temporary checkpoint file.")

def main():
    arguments = parse_arguments()
    
    configure_logging(arguments)

    activation_differences, redundant_examples, non_redundant_examples = get_conecut_data(arguments)

    augmentation_activation_differences = get_augmentation_data(arguments)

    augment_redundancy(arguments, activation_differences, redundant_examples, non_redundant_examples, augmentation_activation_differences)

if __name__ == "__main__":
    main()