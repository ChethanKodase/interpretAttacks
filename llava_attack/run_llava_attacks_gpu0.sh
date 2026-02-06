#!/usr/bin/env bash
set -e  # stop if any command fails

# -----------------------
# GPU selection
# -----------------------
export CUDA_VISIBLE_DEVICES=0

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

for LAYER in $(seq 0 32); do
    python llava_attack/llava_attack_imagenet.py --attck_type grill_wass --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 11 --AttackStartLayer $LAYER --numLayerstAtAtime 1
done

for LAYER in $(seq 0 32); do
    python llava_attack/llava_attack_imagenet.py --attck_type grill_wass --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 12 --AttackStartLayer $LAYER --numLayerstAtAtime 1
done

for LAYER in $(seq 0 32); do
    python llava_attack/llava_attack_imagenet.py --attck_type grill_wass --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 13 --AttackStartLayer $LAYER --numLayerstAtAtime 1
done
