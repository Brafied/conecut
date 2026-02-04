#!/bin/bash

#SBATCH --account=marasovic-gpu-np
#SBATCH --partition=marasovic-gpu-np

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --job-name=determine_redundancy

#SBATCH -o /tmp/slurm-%j.log
#SBATCH -e /tmp/slurm-%j.log

YEAR=$(date +%Y)
MONTH=$(date +%m)
OUTDIR=../../slurm/${YEAR}/${MONTH}
mkdir -p "$OUTDIR"
LOGFILE="$OUTDIR/slurmjob-${SLURM_JOB_ID}-${SLURMD_NODENAME}.log"
exec >"$LOGFILE" 2>&1


module load cuda/12.8.1 miniforge3/latest
conda activate conecut

export HF_HOME=/scratch/general/vast/u1307785/huggingface_cache

python -u src/model_test_llm_features_works_with_rm.py \
    --model_id "Skywork/Skywork-Reward-V2-Llama-3.1-8B" \
    --subset_filter "full" \
    --determine_redundancy \
    --solver "nnls" \
    --epsilon 0.98 \