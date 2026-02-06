#!/usr/bin/env bash
set -e  # stop if any command fails

# -----------------------
# GPU selection
# -----------------------
export CUDA_VISIBLE_DEVICES=1

# -----------------------
# Conda setup (required for non-interactive shells)
# -----------------------
source "$(conda info --base)/etc/profile.d/conda.sh"

# -----------------------
# Project directory
# -----------------------
cd /data1/chethan/interpretAttacks || exit 1

# -----------------------
# Activate environment
# -----------------------
conda activate gemma3

# -----------------------
# Sequential attacks
# -----------------------

for LAYER in $(seq 0 34); do
    python gemma_attack/gemma3AttackImgenet.py --attck_type grill_wass --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 19 --AttackStartLayer $LAYER --numLayerstAtAtime 1
done

for LAYER in $(seq 0 34); do
    python gemma_attack/gemma3AttackImgenet.py --attck_type grill_wass --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 20 --AttackStartLayer $LAYER --numLayerstAtAtime 1
done

for LAYER in $(seq 0 34); do
    python gemma_attack/gemma3AttackImgenet.py --attck_type grill_wass --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 21 --AttackStartLayer $LAYER --numLayerstAtAtime 1
done

