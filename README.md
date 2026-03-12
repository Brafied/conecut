# Conecut

This repository evaluates reward models on RewardBench and performs redundancy detection on the dataset.

An example is redundant if the constraint it imposes on the reward model's linear reward head weight space lies within the cone of the contstraints imposed by other examples.
This is operationalized using a non-negative reconstruction algorithm: an example is labeled as redundant if its reconstruction R² is greater than some predefined epsilon.

------------------------------------------------------------------------

## Usage

### Run Evaluation

    python conecut.py \
      --model_id Skywork/Skywork-Reward-V2-Llama-3.1-8B \
      --subset_filter safety

### Run Evaluation and Redundancy Detection (NNLS)

    python conecut.py \
      --model_id Skywork/Skywork-Reward-V2-Llama-3.1-8B \
      --subset_filter safety \
      --determine_redundancy \
      --reconstruction_algorithm nnls \
      --epsilon 0.98

### Run Evaluation and Redundancy Detection (NNOMP)

    python conecut.py \
      --model_id Skywork/Skywork-Reward-V2-Llama-3.1-8B \
      --subset_filter safety \
      --determine_redundancy \
      --reconstruction_algorithm nnomp \
      --epsilon 0.98 \
      --nnomp_maximum_nonzero_coefficients 8

------------------------------------------------------------------------

## Output

Activation differences (activation_differences.pkl), scores (chosen_scores.pkl, rejected_scores.pkl), and subset labels (subsets.pkl) are saved to:

    data/<model_id>_<subset_filter>/


Redundancy results (non_redundant_examples.pkl, redundant_examples.pkl) are saved to:

    data/<model_id>_<subset_filter>/<reconstruction_algorithm>_<epsilon>_<nnomp_maximum_nonzero_coefficients>/


Logging outputs (\<model_id\>\_\<subset_filter\>\_\<reconstruction_algorithm\>\_\<epsilon\>\_\<nnomp_maximum_nonzero_coefficients\>.txt) are saved to:

    logging/
