import logging
import argparse
import os
import pickle
import torch
import math

logger = logging.getLogger(__name__)

def parse_arguments():
    argument_parser = argparse.ArgumentParser()
    
    argument_parser.add_argument("--model_id", required=True, type=str, help="The Hugging Face model ID of the model whose representations will be used for calculating activation difference solid angle.")
    argument_parser.add_argument("--data_directory", required=True, type=str, help="The directory whose cached representations will be used for calculating activation difference solid angle.")

    arguments = argument_parser.parse_args()

    arguments.parsed_model_id = arguments.model_id.replace("/", "_")

    return arguments

def configure_logging(arguments):
    os.makedirs("logging/rb2", exist_ok=True)

    logging_file_name = f"logging/rb2/activation_difference_solid_angle_{arguments.parsed_model_id}.txt"

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
        
        if (i + 1) % (maximum_iterations // 100) == 0:
            logger.info(
                f"Current iteration: {i + 1}/{maximum_iterations}, "
                f"Norm of current estimate: {torch.linalg.vector_norm(current_estimate).item():.6e}, "
                f"duality gap: {duality_gap:.6e}"
            )
            
    if torch.linalg.vector_norm(current_estimate).item() <= 1e-10:
        return False, normalized_constraints
    
    return True, normalized_constraints

@torch.no_grad()
def estimate_level_probability(constraints, particles, samples_count, batch_size, gpu_generator):
    batch_count = (samples_count + batch_size - 1) // batch_size
    
    survivors = []
    for i, start in enumerate(range(0, samples_count, batch_size)):
        if (i + 1) % (batch_count // 10) == 0:
            logger.info(f"Processing batch {i + 1} / {batch_count}...")
            
        actual_batch_size = min(batch_size, samples_count - start)
        if particles is None:
            batch_samples = torch.randn(
                (actual_batch_size, constraints.shape[1]),
                generator=gpu_generator,
                device="cuda",
                dtype=constraints.dtype,
            )
        else:
            batch_samples = particles[start: start + actual_batch_size]

        batch_hit_mask = torch.all((constraints @ batch_samples.T) >= 0, dim=0)
        if batch_hit_mask.any():
            survivors.append(batch_samples[batch_hit_mask].to("cpu"))

    survivors = torch.cat(survivors, dim=0)
    
    estimate = len(survivors) / samples_count
    log_estimate = math.log(estimate)
    relative_standard_error = math.sqrt((1.0 - estimate) / (samples_count * estimate))
    log_standard_error = (
        -math.inf if relative_standard_error == 0.0
        else math.log(relative_standard_error) + log_estimate
    )
    effective_sample_size = samples_count * estimate

    return {
        "survivors": survivors,
        "estimate": estimate,
        "log_estimate": log_estimate,
        "relative_standard_error": relative_standard_error,
        "log_standard_error": log_standard_error,
        "effective_sample_size": effective_sample_size,
    }
    
@torch.no_grad()
def rejuvenate_particles(survivors, constraints, target_particle_count, cpu_generator, gpu_generator, step_size=0.01, steps=500):
    resample_indices = torch.randint(
        low=0,
        high=survivors.shape[0],
        size=(target_particle_count,),
        generator=cpu_generator,
    )
    particles = survivors[resample_indices].to("cuda").clone()

    for _ in range(steps):
        noise = torch.randn(
            (target_particle_count, particles.shape[1]),
            generator=gpu_generator,
            device="cuda",
            dtype=particles.dtype,
        )
        proposals = particles + step_size * noise

        proposal_hit_mask = torch.all((constraints @ proposals.T) >= 0, dim=0)
        if not proposal_hit_mask.any():
            continue

        log_accept_ratio = 0.5 * (torch.sum(particles[proposal_hit_mask] ** 2, dim=1) - torch.sum(proposals[proposal_hit_mask] ** 2, dim=1))
        uniform_log = torch.log(
            torch.rand(
                log_accept_ratio.shape,
                generator=gpu_generator,
                device="cuda",
                dtype=particles.dtype,
            )
        )
        accepted_indices = torch.nonzero(proposal_hit_mask, as_tuple=False).squeeze(1)[uniform_log < log_accept_ratio]
        particles[accepted_indices] = proposals[accepted_indices]

    return particles

def main():
    arguments = parse_arguments()
    
    configure_logging(arguments)

    activation_differences = get_data(arguments)
    
    logging.info("Computing the max margin feasible vector satisfying all constraint vectors...")
    has_feasible_vector, normalized_constraints = compute_feasible_vector(activation_differences, int(1e5))
    if not has_feasible_vector:
        logger.info("Max margin feasible vector not found.")
        return
    logger.info("Max margin feasible vector found.")
    
    gpu_generator = torch.Generator(device="cuda")
    cpu_generator = torch.Generator(device="cpu")
    particles = None
    results = []
    for constraint_count in range(1, activation_differences.shape[0] + 1, 1):
        logger.info(f"Estimating the solid angle of {constraint_count} constraints...")

        normalized_constraints_slice = normalized_constraints[:constraint_count].to(dtype=torch.float32)
        result = estimate_level_probability(normalized_constraints_slice, particles, 2**19, 2**15, gpu_generator)
        results.append({
            "log_estimate": result["log_estimate"],
            "log_standard_error": result["log_standard_error"],
            "relative_standard_error": result["relative_standard_error"],
            "effective_sample_size": result["effective_sample_size"],
        })
        
        particles = rejuvenate_particles(result["survivors"], normalized_constraints_slice, 2**19, cpu_generator, gpu_generator)
        
        del result
        torch.cuda.empty_cache()

        logger.info(f"cumulative_log_estimate: {sum(result['log_estimate'] for result in results)}")
        logger.info(f"cumulative_relative_standard_error: {math.sqrt(sum(result['relative_standard_error'] ** 2 for result in results))}")
        logger.info(f"log_estimates: {[result['log_estimate'] for result in results]}")
        logger.info(f"log_standard_errors: {[result['log_standard_error'] for result in results]}")
        logger.info(f"relative_standard_errors: {[result['relative_standard_error'] for result in results]}")
        logger.info(f"effective_sample_sizes: {[result['effective_sample_size'] for result in results]}")
        
if __name__ == "__main__":
    main()