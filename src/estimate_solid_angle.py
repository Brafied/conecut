import logging
import argparse
import os
import pickle
import torch
import math
import numpy as np

logger = logging.getLogger(__name__)

def parse_arguments():
    argument_parser = argparse.ArgumentParser()
    
    argument_parser.add_argument("--model_id", required=True, type=str, help="The Hugging Face model ID of the model whose representations will be used for calculating activation difference solid angle.")
    argument_parser.add_argument("--data_directory", required=True, type=str, help="The directory whose cached representations will be used for calculating activation difference solid angle.")

    arguments = argument_parser.parse_args()

    arguments.parsed_model_id = arguments.model_id.replace("/", "_")

    return arguments

def configure_logging(arguments):
    os.makedirs("logging", exist_ok=True)

    logging_file_name = f"logging/activation_difference_solid_angle_{arguments.parsed_model_id}.txt"

    logging.basicConfig(filename=logging_file_name, level=logging.INFO)

def get_data(arguments):
    logger.info(f"Loading cached activation differences for {arguments.model_id}...")

    with open(arguments.data_directory + "/activation_differences.pkl", "rb") as file:
        activation_differences = pickle.load(file)

    logger.info("Completed loading cached activation differences.")

    return activation_differences

@torch.no_grad()
def compute_feasible_vector(constraints, maximum_iterations):
    constraints = constraints.to(device="cuda", dtype=torch.float64)
    normalized_constraints = constraints / torch.linalg.vector_norm(constraints, dim=1, keepdim=True)
    current_estimate = normalized_constraints.mean(dim=0)
    for i in range(maximum_iterations):
        scores = normalized_constraints @ current_estimate
        target_vertex_index = torch.argmin(scores)
        target_vertex = normalized_constraints[target_vertex_index]
        current_estimate_squared_norm = torch.dot(current_estimate, current_estimate)
        duality_gap = (current_estimate_squared_norm - scores[target_vertex_index]).item()
        if duality_gap <= 1e-10:
            break
        target_vertex_direction = target_vertex - current_estimate
        step_size = torch.clamp((current_estimate_squared_norm - scores[target_vertex_index]) / torch.dot(target_vertex_direction, target_vertex_direction), 0.0, 1.0)
        current_estimate = current_estimate + step_size * target_vertex_direction
        if (i + 1) % (maximum_iterations // 4) == 0:
            logger.info(
                f"Current iteration: {i + 1}/{maximum_iterations}, "
                f"Norm of current estimate: {torch.linalg.vector_norm(current_estimate).item():.6e}, "
                f"duality gap: {duality_gap:.6e}"
            )
    current_estimate_norm = torch.linalg.vector_norm(current_estimate)
    if current_estimate_norm.item() <= 1e-10:
        return None, None, normalized_constraints
    feasible_vector = current_estimate / current_estimate_norm
    margin = (normalized_constraints @ feasible_vector).min().item()
    return feasible_vector, margin, normalized_constraints

def perform_beta_pilot_search(normalized_constraints, feasible_vector, beta_grid, margin, sample_count, batch_size, generator):
    best_beta = None
    best_relative_standard_error = None
    for beta in beta_grid:
        scaling_factor = float(beta / margin)
        result = estimate_solid_angle(normalized_constraints, feasible_vector, scaling_factor, sample_count, batch_size, generator)
        if best_relative_standard_error is None or result["relative_standard_error"] < best_relative_standard_error:
            best_relative_standard_error = result["relative_standard_error"]
            best_beta = beta
    return best_beta

@torch.no_grad()
def estimate_solid_angle(normalized_constraints, mu, max_samples, max_batch_size, generator):
    batch_count = (max_samples + max_batch_size - 1) // max_batch_size
    
    log_sum_w = torch.tensor(float("-inf"), device="cuda", dtype=torch.float64)
    log_sum_w_squared = torch.tensor(float("-inf"), device="cuda", dtype=torch.float64)
    sample_count = 0
    for i, start in enumerate(range(0, max_samples, max_batch_size)): 
        batch_size = min(max_batch_size, max_samples - start)
        sample_count += batch_size
        
        if (i + 1) % (batch_count // 10) == 0:
            logger.info(f"Processing batch {i + 1} / {batch_count}...")
        
        standard_normal_samples = torch.randn((batch_size, normalized_constraints.shape[1]), generator=generator, device="cuda", dtype=torch.float64)
        proposal_samples = standard_normal_samples + mu.unsqueeze(0)
        
        proposal_samples_hit_mask = torch.all((normalized_constraints @ proposal_samples.T) >= 0, dim=0)
        if proposal_samples_hit_mask.sum().item() == 0:
            continue
        
        log_w = (-(proposal_samples[proposal_samples_hit_mask] @ mu) + 0.5 * torch.dot(mu, mu)).to(torch.float64)
        
        log_sum_w = torch.logaddexp(log_sum_w, torch.logsumexp(log_w, dim=0))
        log_sum_w_squared = torch.logaddexp(log_sum_w_squared, torch.logsumexp(2.0 * log_w, dim=0))
        
        if (i + 1) % (batch_count // 100) == 0:
            log_sample_count = math.log(sample_count)
            log_mean_w = (log_sum_w - log_sample_count).item()
            log_mean_w_squared = (log_sum_w_squared - log_sample_count).item()
            log_standard_error = 0.5 * (log_mean_w_squared + math.log1p(-math.exp(2.0 * log_mean_w - log_mean_w_squared)) - log_sample_count)
            relative_standard_error = math.exp(log_standard_error - log_mean_w)
            if relative_standard_error <= 0.01:
                break
            
    log_sample_count = math.log(sample_count)
    log_mean_w = (log_sum_w - log_sample_count).item()
    log_mean_w_squared = (log_sum_w_squared - log_sample_count).item()
    log_standard_error = 0.5 * (log_mean_w_squared + math.log1p(-math.exp(2.0 * log_mean_w - log_mean_w_squared)) - log_sample_count)
    relative_standard_error = math.exp(log_standard_error - log_mean_w)
    effective_sample_size = math.exp((2.0 * log_sum_w - log_sum_w_squared).item())
    
    return {
        "log_estimate": log_mean_w,
        "log_standard_error": log_standard_error,
        "relative_standard_error": relative_standard_error,
        "effective_sample_size": effective_sample_size
    }

def main():
    arguments = parse_arguments()
    
    configure_logging(arguments)

    activation_differences = get_data(arguments)
    
    generator = torch.Generator(device="cuda")
    
    results = []
    for constraint_count in range(10, activation_differences.shape[0] + 1, 10):
        logger.info(f"Estimating the solid angle of {constraint_count} constraints...")
        
        constraints = activation_differences[:constraint_count]
        feasible_vector, margin, normalized_constraints = compute_feasible_vector(constraints, 1e5)
        if feasible_vector is None:
            logger.info("A feasible vector does not exist.")
            break
        
        beta_grid = np.arange(0.4, 0.8, 0.02)
        beta = perform_beta_pilot_search(normalized_constraints, feasible_vector, beta_grid, margin, 1e9, 2**18, generator)
        mu = float(beta / margin) * feasible_vector
        
        result = estimate_solid_angle(normalized_constraints, mu, 1e10, 2**18, generator)
        results.append(result)
        
        logger.info(f"log_estimates: {[result['log_estimate'] for result in results]}")
        logger.info(f"log_standard_errors: {[result['log_standard_error'] for result in results]}")
        logger.info(f"relative_standard_errors: {[result['relative_standard_error'] for result in results]}")
        logger.info(f"effective_sample_sizes: {[result['effective_sample_size'] for result in results]}")

if __name__ == "__main__":
    main()