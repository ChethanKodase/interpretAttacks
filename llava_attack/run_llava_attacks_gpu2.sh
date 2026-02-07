#!/usr/bin/env bash
set -e  # stop if any command fails

# -----------------------
# GPU selection
# -----------------------
export CUDA_VISIBLE_DEVICES=2

# -----------------------
# Conda setup (required for non-interactive shells)
# -----------------------
source ~/miniforge3/etc/profile.d/conda.sh

# -----------------------
# Project directory
# -----------------------
cd ~/interpretAttacks || exit 1

# -----------------------
# Activate environment
# -----------------------
conda activate llava15

# -----------------------
# Sequential attacks
# -----------------------


for ATTACK_SAMPLE in $(seq 41 500); do
    for LAYER in $(seq 0 24); do
        python llava_attack/llava_vision_attack_imagenet.py --attck_type grill_wass --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer $LAYER --numLayerstAtAtime 1
    done
done