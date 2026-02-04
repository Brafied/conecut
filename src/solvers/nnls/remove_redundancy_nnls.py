import argparse
import os
import pickle
import logging
import numpy as np
import torch
import warnings

# Constants
residual_threshold = 0.001
r2_threshold = 0.95
EPS_REL = 1e-6
EPS_TINY = 1e-12


warnings.filterwarnings('ignore')

from utils import load_all_data, verify_linear_dependence

def find_redundancy(solve_by="res", data_name = "grammar", model_name = "skyworks_llama", solver = 'nnls', threshold = 0.95):
    # Use existing logger instead of creating a new one
    logger = logging.getLogger(__name__)
    logger.info("************ redundancy detection started **********************")
    logger.info(f"solving by {solve_by} for data {data_name}")
    # Load data
    try:
        features = load_all_data(features_file_name=data_name, model_name=model_name)
    except FileNotFoundError as e:
        logger.error(f"Could not find features file. Make sure features are saved first. Error: {e}")
        raise
    num_examples = len(features)

    # Define checkpoint path
    save_dir = f'data/{model_name}_features'
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = f'{save_dir}/checkpoint_{data_name}_{solver}_{threshold}.pkl'

    # Initialize variables
    start_index = 0
    redundant_inds = []
    redundancy_details = [] 
    results = []
    
    # Check for existing checkpoint
    if os.path.exists(checkpoint_path):
        try:
            logger.info(f"Found checkpoint at {checkpoint_path}. Loading...")
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
                redundant_inds = checkpoint_data.get('redundant_inds', [])
                redundancy_details = checkpoint_data.get('redundancy_details', [])
                results = checkpoint_data.get('results', [])
                start_index = checkpoint_data.get('next_index', 0)
            
            logger.info(f"Resuming from index {start_index}. Found {len(redundant_inds)} redundant features so far.")
            print(f"Resuming from index {start_index}. Found {len(redundant_inds)} redundant features so far.")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}. Starting from scratch.")
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
            # Reset if loading fails
            start_index = 0
            redundant_inds = []
            redundancy_details = []
            results = []

    # Process each feature starting from start_index
    for i in range(start_index, num_examples):
        if i % 5 == 0:
            logger.info(f"Processing feature {i + 1}/{num_examples}...")
            logger.info(f"Current redundant count: {len(redundant_inds)}")
        
        # Save checkpoint periodically (every 50 iterations)
        if i % 50 == 0 and i > start_index:
            try:
                temp_checkpoint = checkpoint_path + ".tmp"
                with open(temp_checkpoint, 'wb') as f:
                    pickle.dump({
                        'redundant_inds': redundant_inds,
                        'redundancy_details': redundancy_details,
                        'results': results,
                        'next_index': i  # Save current index as next to process if we crash now
                    }, f)
                os.replace(temp_checkpoint, checkpoint_path)
                logger.info(f"Checkpoint saved at index {i}")
            except Exception as e:
                logger.warning(f"Failed to save checkpoint: {e}")

        mask = np.ones(len(features), dtype=bool)
        #do not consider every redundant element plus the current one
        for index_to_mask in redundant_inds:
            mask[index_to_mask] = False
        mask[i] = False
        
        # NEW: Capture the specific indices used as the basis for this iteration
        basis_indices = np.where(mask)[0]
        
        X = features[mask].T
        y = features[i]

        # Normalize
        X_norm = np.linalg.norm(X, axis=0, keepdims=True) + EPS_TINY
        y_norm = np.linalg.norm(y, axis=0, keepdims=True) + EPS_TINY
        X = X / X_norm
        y = y / y_norm

        # Placeholder for the coefficients that determined redundancy
        active_coeffs = None 

        # Choose method based on solve_by
        if solve_by == "res":
            # Check linear dependence
            coeffs_nnls, _, _, _, residual_nnls = verify_linear_dependence(y, X, method='nnls')
            coeffs_lstsq, _, _, _, residual_lstsq = verify_linear_dependence(y, X, method='lstsq')
            residual = min(residual_nnls, residual_lstsq)
            
            if residual_nnls < residual_lstsq:
                method_used = 'nnls'
                active_coeffs = coeffs_nnls
            else:
                method_used = 'lstsq'
                active_coeffs = coeffs_lstsq
                
        else:
            if solver == "nnls":
                # MODIFIED: Unpack the tuple to get coefficients (index 0)
                coeffs_out, _, _, r2_out, _ = verify_linear_dependence(y, X, method='nnls')
                r2 = r2_out
                active_coeffs = coeffs_out
                method_used = 'nnls'
                residual = 'none'
            else:
                # MODIFIED: Unpack the tuple to get coefficients (index 0)
                coeffs_out, _, _, r2_out, _ = verify_linear_dependence(y, X, method='lstsq')
                r2 = r2_out
                active_coeffs = coeffs_out
                method_used = 'lstsq'
                residual = 'none'

        # Determine redundancy
        adaptive_threshold = 1e-8 * np.linalg.norm(X)
        is_zero = torch.count_nonzero(torch.from_numpy(y)).item() == 0
        if solve_by == "res":
            is_redundant = residual <= adaptive_threshold
        else:
            is_redundant = r2 >= threshold

        if is_redundant:
            redundant_inds.append(i)
            
            # NEW: Map coefficients to their original example indices
            # We filter out very small coefficients to save space (sparse storage)
            contributors = {}
            if active_coeffs is not None:
                for idx, coeff in zip(basis_indices, active_coeffs):
                    # Save coefficients that contribute significantly (> 1e-5)
                    if abs(coeff) > 1e-5:
                        contributors[int(idx)] = float(coeff)
            
            redundancy_details.append({
                'redundant_index': int(i),
                'reconstruction_r2': float(r2) if solve_by != "res" else None,
                'reconstruction_residual': float(residual) if isinstance(residual, (int, float)) else None,
                'method': method_used,
                'coefficients': contributors
            })

        # Store result
        results.append({
            'example_num': i,
            'is_redundant': is_redundant,
            'residual': residual,
            'r2_score': r2 if solve_by != "res" else None,
            'method': method_used,
            'is_zero': is_zero
        })

    # Summarize results
    total_redundant = len(redundant_inds)
    total_non_redundant = num_examples - total_redundant
    print(total_non_redundant)
    
    # Recalculate non-redundant indices based on final list
    non_redundant_inds = [i for i in range(num_examples) if i not in redundant_inds]

    logger.info("\nResults Summary:")
    logger.info(f"Total non-redundant features: {total_non_redundant}")
    logger.info(f"Total redundant features: {total_redundant}")
    logger.info(f"Percentage redundant: {(total_redundant / num_examples) * 100:.2f}%")

    print("\nResults Summary:")
    print(f"Total non-redundant features: {total_non_redundant}")
    print(f"Total redundant features: {total_redundant}")
    print(f"Percentage redundant: {(total_redundant / num_examples) * 100:.2f}%")

    # Save indices - using absolute path from current working directory with threshold in path
    
    with open(f'{save_dir}/non_redundant_{data_name}_{solver}_{threshold}.pkl', 'wb') as f:
        pickle.dump(non_redundant_inds, f)

    with open(f'{save_dir}/redundant_{data_name}_{solver}_{threshold}.pkl', 'wb') as f:
        pickle.dump(redundant_inds, f)
        
    # NEW: Save coefficient details
    with open(f'{save_dir}/redundancy_coefficients_{data_name}_{solver}_{threshold}.pkl', 'wb') as f:
        pickle.dump(redundancy_details, f)
    
    logger.info(f"Saved redundancy coefficients details to {save_dir}/redundancy_coefficients_{data_name}_{solver}_{threshold}.pkl")
    logger.info("Non-redundant and redundant indices saved to pickle files.")
    
    # Clean up checkpoint on successful completion
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
            logger.info("Removed temporary checkpoint file.")
        except OSError:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find redundant features")
    parser.add_argument('--data_name', type=str, default="features_trained_from_scratch__grammaticality", help="Name of the dataset")
    parser.add_argument('--model_name', type=str, default="skyworks_llama", help="Name of the model")
    parser.add_argument('--solver', type=str, default="nnls", help="Name of the solver")
    parser.add_argument('--threshold', type=float, default=0.95, help="R2 threshold for redundancy detection")

    args = parser.parse_args()
    os.makedirs(f"data/{args.model_name}_features", exist_ok=True)

    find_redundancy(solve_by = "r2", data_name=args.data_name, model_name=args.model_name, solver=args.solver, threshold=args.threshold)