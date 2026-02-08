#!/usr/bin/env bash
set -e  # stop if any command fails

# -----------------------
# GPU selection
# -----------------------
export CUDA_VISIBLE_DEVICES=0

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

for LAYER in $(seq 24 34); do
    python gemma_attack/gemma3AttackImgenet.py --attck_type grill_wass --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 33 --AttackStartLayer $LAYER --numLayerstAtAtime 1
done

for ATTACK_SAMPLE in $(seq 34 250); do
    for LAYER in $(seq 0 34); do
        python gemma_attack/gemma3AttackImgenet.py --attck_type grill_wass --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer $LAYER --numLayerstAtAtime 1
    done
done

