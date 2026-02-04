import os, argparse, json, pickle as pkl, logging
import torch
import warnings

from reward_bench_adapter import run_rm_subset    
from reward_model_inference_utils import (
    run_redundancy_tests, calculate_accuracy
)
from solvers.nnls.remove_redundancy_nnls import find_redundancy

# --- Configuration & Constants ---
warnings.filterwarnings('ignore')

os.environ["HF_HOME"] = "/scratch/general/vast/u1307785/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "/scratch/general/vast/u1307785/huggingface_cache/transformers"
os.environ["HF_DATASETS_CACHE"]  = "/scratch/general/vast/u1307785/huggingface_cache/datasets"
CACHE_DIR = "/scratch/general/vast/u1307785/huggingface_cache/"

SECTION_MAP = {     
    "chat":       "Chat",
    "chat_hard":  "Chat Hard",
    "safety":     "Safety",
    "reasoning":  "Reasoning",
}

GOLD_STANDARD_MODEL = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"

LOG_DIRECTORY = os.path.expanduser("~/alignment_benchmark_LLM/logging/")

def load_cached_activations(arguments):
    data_directory = f"data/{arguments.model_name}_features"
    activations_differences_path = os.path.join(data_directory, f"features_diff_{arguments.subset_filter}.pkl")
    chosen_scores_path = os.path.join(data_directory, f"scores_chosen_{arguments.subset_filter}.pkl")
    rejected_scores_path = os.path.join(data_directory, f"scores_rejected_{arguments.subset_filter}.pkl")
    subsets_path = os.path.join(data_directory, f"subsets_{arguments.subset_filter}.pkl")

    if not os.path.exists(activations_differences_path):
        return None, None, None, None

    with open(activations_differences_path, "rb") as file:
        activations_difference = pkl.load(file)
    with open(chosen_scores_path, "rb") as file:
        chosen_scores = pkl.load(file)
    with open(rejected_scores_path, "rb") as file:
        rejected_scores = pkl.load(file)
    with open(subsets_path, "rb") as file:
        subsets = pkl.load(file)

    return activations_difference, chosen_scores, rejected_scores, subsets

def configure_logging(arguments):
    if not arguments.evaluate_only:
        os.makedirs(f"data/{arguments.model_name}_features", exist_ok=True)

    os.makedirs(LOG_DIRECTORY, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(LOG_DIRECTORY, f"{arguments.model_name}_{arguments.subset_filter}_{arguments.epsilon if arguments.determine_redundancy else 'evaluation'}.txt"),
        level=logging.INFO,
    )

def determine_activations_and_scores(arguments):
    activation_differences = None

    if arguments.regenerate_activations:
        logging.info(f"Regeneration of activations requested for {arguments.model_name} on subset {arguments.subset_filter}")
    else:
        activation_differences, chosen_scores, rejected_scores, subsets = load_cached_activations(arguments)

    if activation_differences is not None:
        logging.info(f"Found cached activations for {arguments.model_name} on subset {arguments.subset_filter}")
        return activation_differences, chosen_scores, rejected_scores, subsets
    
    logging.info(f"Generating activations for {arguments.model_id} on subset {arguments.subset_filter}")
    # TODO: REFACTOR run_rm_subset
    chosen_activation, rejected_activation, chosen_scores, rejected_scores, subsets = run_rm_subset(
        arguments.model_id,
        section=SECTION_MAP.get(arguments.subset_filter, None),
        batch_size=8,
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_DIR
    )
    logging.info(f"Completed activation generation for {arguments.model_id} on subset {arguments.subset_filter}")
    activation_differences = chosen_activation - rejected_activation
        
    return activation_differences, chosen_scores, rejected_scores, subsets

def evaluate(chosen_scores, rejected_scores, subsets):
    accuracy, correct_count, correct_indices = calculate_accuracy(chosen_scores, rejected_scores)
    logging.info(f"Unweighted Accuracy: {accuracy:.4f} ({correct_count}/{len(chosen_scores)})")
    
    for subset in set(subsets):
        subset_mask = [i for i, current_subset in enumerate(subsets) if current_subset == subset]   
        subset_correct_count = len([i for i in subset_mask if i in correct_indices])
        subset_accuracy = subset_correct_count / len(subset_mask)
        logging.info(f"Subset {subset}: {subset_correct_count}/{len(subset_mask)} = {subset_accuracy:.4f}")

def save_artifacts(arguments, activation_differences, chosen_scores, rejected_scores, subsets):
    data_directory = f"data/{arguments.model_name}_features"

    with open(os.path.join(data_directory, f"scores_chosen_{arguments.subset_filter}.pkl"), "wb") as file:
        pkl.dump(chosen_scores, file)
    with open(os.path.join(data_directory, f"scores_rejected_{arguments.subset_filter}.pkl"), "wb") as file:
        pkl.dump(rejected_scores, file)
    with open(os.path.join(data_directory, f"subsets_{arguments.subset_filter}.pkl"), "wb") as file:
        pkl.dump(subsets, file)
    with open(os.path.join(data_directory, f"features_diff_{arguments.subset_filter}.pkl"), "wb") as file:
        pkl.dump(activation_differences, file)

def determine_redundancy(arguments, chosen_scores, rejected_scores):    
    logging.info(f"Running redundancy detection for model {arguments.model_id} on subset {arguments.subset_filter}")
    
    find_redundancy(
        solve_by="r2", 
        data_name=f"features_diff_{arguments.subset_filter}",
        model_name=arguments.model_name, 
        solver=arguments.solver,
        threshold=arguments.epsilon
    )

    with open(f"data/{arguments.model_name}_features/redundant_features_diff_{arguments.subset_filter}_{arguments.solver}_{arguments.epsilon}.pkl", "rb") as f: 
        redundant_indices = pkl.load(f)

    if redundant_indices is not None:
        print(f"Running ablation tests with {len(redundant_indices)} indices...")
        ablation_results = run_redundancy_tests(chosen_scores, rejected_scores, redundant_indices, len(chosen_scores))
        
        ablation_results_directory = os.path.join(LOG_DIRECTORY, "redundancy_results")
        os.makedirs(ablation_results_directory, exist_ok=True)
        with open(os.path.join(ablation_results_directory, f"{arguments.model_name}_{arguments.subset_filter}_{arguments.solver}_{arguments.epsilon}_redundancy_results_ablation.json"), 'w') as f:
            json.dump(ablation_results, f, indent=4)

        logging.info(f"Redundancy results saved: {json.dumps(ablation_results, indent=2)}")

def parse_arguments():
    argument_parser = argparse.ArgumentParser()

    argument_parser.add_argument("--model_id", required=True, type=str)
    argument_parser.add_argument("--subset_filter", required=True, choices=[*SECTION_MAP.keys(), "full"])
    argument_parser.add_argument("--regenerate_activations", action="store_true")
    argument_parser.add_argument("--evaluation_only", action="store_true")
    argument_parser.add_argument("--determine_redundancy", action="store_true")
    argument_parser.add_argument("--solver", choices=["nnls", "lstsq"])
    argument_parser.add_argument("--epsilon", type=float)

    return argument_parser.parse_args()

def main():
    arguments = parse_arguments()
    arguments.model_name = arguments.model_id.split("/")[-1]
    
    configure_logging(arguments)

    activation_differences, chosen_scores, rejected_scores, subsets = determine_activations_and_scores(arguments)

    evaluate(chosen_scores, rejected_scores, subsets)

    if arguments.evaluation_only:
        return

    save_artifacts(arguments, activation_differences, chosen_scores, rejected_scores, subsets)

    if not arguments.determine_redundancy:
        return
    
    determine_redundancy(arguments, chosen_scores, rejected_scores)

if __name__ == "__main__":
    main()