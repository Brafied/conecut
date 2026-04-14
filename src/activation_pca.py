import logging
import argparse
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from generate_conecut_data import apply_chat_template, run_inference
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

def parse_arguments():
    argument_parser = argparse.ArgumentParser()
    
    argument_parser.add_argument("--model_id", required=True, type=str, help="The Hugging Face model ID of the model whose representations will be used for calculating activation pca.")

    arguments = argument_parser.parse_args()

    arguments.parsed_model_id = arguments.model_id.replace("/", "_")

    return arguments

def configure_logging(arguments):
    os.makedirs("logging", exist_ok=True)

    logging_file_name = f"logging/activation_pca_{arguments.parsed_model_id}.txt"

    logging.basicConfig(filename=logging_file_name, level=logging.INFO)

def generate_activation_pca_data():
    logger.info("Loading dataset, model, and tokenizer...")
    dataset = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned", split="train")
    dataset = dataset.select(range(10_000))
    model = AutoModelForSequenceClassification.from_pretrained(
        "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="eager",
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

    return chosen_activations, rejected_activations


def run_pca(activations, variance_thresholds):
    pca = PCA()
    pca.fit(activations.float().cpu().numpy())

    cumulative = pca.explained_variance_ratio_.cumsum()

    results = {}
    for variance_threshold in variance_thresholds:
        k = (cumulative >= variance_threshold).argmax() + 1
        results[variance_threshold] = k

    return results

def main():
    arguments = parse_arguments()
    
    configure_logging(arguments)

    chosen_activations, rejected_activations = generate_activation_pca_data()
    activation_differences = chosen_activations - rejected_activations
    
    variance_thresholds = [0.9, 0.95, 0.99]
    chosen_activations_pca = run_pca(chosen_activations, variance_thresholds)
    logger.info(f"Chosen activations PCA results: {chosen_activations_pca}")
    rejected_activations_pca = run_pca(rejected_activations, variance_thresholds)
    logger.info(f"Rejected activations PCA results: {rejected_activations_pca}")
    activation_differences_pca = run_pca(activation_differences, variance_thresholds)
    logger.info(f"Activation differences PCA results: {activation_differences_pca}")    

if __name__ == "__main__":
    main()