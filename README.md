# interpretAttacks

### On Adversarial Robustness of Vision-Language Models

This repository contains the code used to craft, evaluate, and interpret adversarial attacks against Vision-Language Models (VLMs). It includes the proposed **SSGRA** attack alongside a set of baseline attacks (**BSA**, **DRA**, **EGA**, **FDA**, **CE**, **SSPA**), FLOPs-based cost comparisons, and tooling to study how these attacks align with the spectral (singular-value) subspaces of the model's intermediate representations.

All commands below assume you are running from the repository root with the appropriate conda environment activated and a GPU selected via `CUDA_VISIBLE_DEVICES`.

---

## Table of Contents

- [Gemma 3](#gemma-3)
  - [Proposed Method — SSGRA](#proposed-method--ssgra)
  - [Baseline Attacks](#baseline-attacks)
  - [FLOPs Estimation](#flops-estimation)
  - [Singular Value Spectrum & Conditioning](#singular-value-spectrum--conditioning)
  - [Spectral Subspace Alignment — Baseline Representations](#spectral-subspace-alignment--baseline-representations)
  - [Spectral Subspace Alignment — Tracking During Optimization](#spectral-subspace-alignment--tracking-during-optimization)
- [Qwen 2.5](#qwen-25)
- [LLaVA](#llava)

---

## Gemma 3

### Proposed Method — SSGRA

The core method proposed in this work.

```bash
export CUDA_VISIBLE_DEVICES=2
conda activate gemma3
cd interpretAttacks

for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_SSGRA.py \
        --attck_type saa_BSAexp \
        --desired_norm_l_inf 0.0025 \
        --learningRate 0.001 \
        --num_steps 1000 \
        --attackSample $ATTACK_SAMPLE \
        --towardsNull 0.15 \
        --whichMLP up_proj \
        --whichMLPvis fc2 \
        --balancingAlpha 0.5 \
        --chosenLanLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 \
        --chosenVisLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
done
```

### Baseline Attacks

<details open>
<summary><b>BSA</b></summary>

```bash
export CUDA_VISIBLE_DEVICES=0
conda activate gemma3
cd interpretAttacks

for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_BSA.py \
        --attck_type bsa \
        --desired_norm_l_inf 0.009 \
        --learningRate 0.001 \
        --num_steps 1000 \
        --attackSample $ATTACK_SAMPLE
done
```
</details>

<details open>
<summary><b>DRA</b></summary>

```bash
export CUDA_VISIBLE_DEVICES=0
conda activate gemma3
cd interpretAttacks

for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_DRA.py \
        --attck_type dra \
        --desired_norm_l_inf 0.0025 \
        --learningRate 0.001 \
        --num_steps 1000 \
        --attackSample $ATTACK_SAMPLE
done
```
</details>

<details open>
<summary><b>EGA</b></summary>

```bash
export CUDA_VISIBLE_DEVICES=7
conda activate gemma3
cd interpretAttacks

for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_EGA1.py \
        --attck_type ega \
        --desired_norm_l_inf 0.0025 \
        --learningRate 0.001 \
        --num_steps 1000 \
        --attackSample $ATTACK_SAMPLE
done
```
</details>

<details open>
<summary><b>FDA</b></summary>

```bash
export CUDA_VISIBLE_DEVICES=6
conda activate gemma3
cd interpretAttacks

for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_FDAm.py \
        --attck_type fdam \
        --desired_norm_l_inf 0.0025 \
        --learningRate 0.001 \
        --num_steps 1000 \
        --attackSample $ATTACK_SAMPLE
done
```
</details>

<details open>
<summary><b>CE</b></summary>

```bash
export CUDA_VISIBLE_DEVICES=4
conda activate gemma3
cd interpretAttacks

for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_NLLm.py \
        --attck_type nllm \
        --desired_norm_l_inf 0.0025 \
        --learningRate 0.001 \
        --num_steps 1000 \
        --attackSample $ATTACK_SAMPLE
done
```
</details>

<details open>
<summary><b>SSPA</b></summary>

```bash
export CUDA_VISIBLE_DEVICES=5
conda activate gemma3
cd interpretAttacks

for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_SSP.py \
        --attck_type ssp \
        --desired_norm_l_inf 0.0045 \
        --learningRate 0.001 \
        --num_steps 1000 \
        --attackSample $ATTACK_SAMPLE
done
```
</details>

### FLOPs Estimation

To obtain FLOPs estimates for each attack method:

<details open>
<summary><b>BSA</b></summary>

```bash
for ATTACK_SAMPLE in $(seq 50 250); do
    python gemma_attack/FLOPSestimation/gemma3AttackImgenet_BSA_flops.py \
        --attck_type bsa_flops \
        --desired_norm_l_inf 0.05 \
        --learningRate 0.001 \
        --num_steps 1000 \
        --attackSample $ATTACK_SAMPLE
done
```
</details>

<details open>
<summary><b>DRA</b></summary>

```bash
for ATTACK_SAMPLE in $(seq 1 250); do
    python gemma_attack/FLOPSestimation/gemma3AttackImgenet_DRA_flops.py \
        --attck_type dra_flops \
        --desired_norm_l_inf 0.01 \
        --learningRate 0.001 \
        --num_steps 1000 \
        --attackSample $ATTACK_SAMPLE
done
```
</details>

<details open>
<summary><b>EGA</b></summary>

```bash
for ATTACK_SAMPLE in $(seq 50 250); do
    python gemma_attack/FLOPSestimation/gemma3AttackImgenet_EGA1_flops.py \
        --attck_type ega_flops \
        --desired_norm_l_inf 0.05 \
        --learningRate 0.001 \
        --num_steps 1000 \
        --attackSample $ATTACK_SAMPLE
done
```
</details>

<details open>
<summary><b>FDA</b></summary>

```bash
for ATTACK_SAMPLE in $(seq 50 250); do
    python gemma_attack/FLOPSestimation/gemma3AttackImgenet_FDAm_flops.py \
        --attck_type fdam_flops \
        --desired_norm_l_inf 0.05 \
        --learningRate 0.001 \
        --num_steps 1000 \
        --attackSample $ATTACK_SAMPLE
done
```
</details>

<details open>
<summary><b>CE</b></summary>

```bash
for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/FLOPSestimation/gemma3AttackImgenet_NLLm_flops.py \
        --attck_type nllm_flops \
        --desired_norm_l_inf 0.05 \
        --learningRate 0.001 \
        --num_steps 1000 \
        --attackSample $ATTACK_SAMPLE
done
```
</details>

<details open>
<summary><b>SSGRA</b></summary>

```bash
for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/FLOPSestimation/gemma3AttackImgenet_SSGRA_flops.py \
        --attck_type saa_BSAexp_flops \
        --desired_norm_l_inf 0.0045 \
        --learningRate 0.001 \
        --num_steps 1000 \
        --attackSample $ATTACK_SAMPLE \
        --towardsNull 0.15 \
        --whichMLP up_proj \
        --whichMLPvis fc2 \
        --balancingAlpha 0.5 \
        --chosenLanLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 \
        --chosenVisLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
done
```
</details>

<details open>
<summary><b>SSPA</b></summary>

```bash
for ATTACK_SAMPLE in $(seq 50 250); do
    python gemma_attack/FLOPSestimation/gemma3AttackImgenet_SSP_flops.py \
        --attck_type ssp_flops \
        --desired_norm_l_inf 0.05 \
        --learningRate 0.001 \
        --num_steps 1000 \
        --attackSample $ATTACK_SAMPLE
done
```
</details>

### Singular Value Spectrum & Conditioning

To analyze the singular value spectrum and generate conditioning plots:

```bash
python gemma_attack/gemma3ConditioningNoAttention.py
```

### Spectral Subspace Alignment — Baseline Representations

To determine the alignment of baseline attack representations with intermediate spectral subspaces:

**Language-layer sweep**

```bash
export CUDA_VISIBLE_DEVICES=3
conda activate gemma3
cd interpretAttacks

for StudyLayer in $(seq 0 33); do
    python gemma_attack/RealSpectralSubSpaceAlignmentExaminer.py \
        --attck_type bsa --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 \
        --AttackStartLayer 0 --numLayerstAtAtime 1 \
        --VisionLayerTrack 0 --LanLayerTrack $StudyLayer \
        --kthSingVec 20 --attackMode lan

    python gemma_attack/RealSpectralSubSpaceAlignmentExaminer.py \
        --attck_type bsa --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 \
        --AttackStartLayer 0 --numLayerstAtAtime 1 \
        --VisionLayerTrack 0 --LanLayerTrack $StudyLayer \
        --kthSingVec -20 --attackMode lan
done
```

**Vision-layer sweep**

```bash
export CUDA_VISIBLE_DEVICES=2
conda activate gemma3
cd interpretAttacks

for StudyLayer in $(seq 0 26); do
    python gemma_attack/RealSpectralSubSpaceAlignmentExaminer.py \
        --attck_type bsa --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 \
        --AttackStartLayer 0 --numLayerstAtAtime 1 \
        --VisionLayerTrack $StudyLayer --LanLayerTrack 0 \
        --kthSingVec 20 --attackMode vis

    python gemma_attack/RealSpectralSubSpaceAlignmentExaminer.py \
        --attck_type bsa --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 \
        --AttackStartLayer 0 --numLayerstAtAtime 1 \
        --VisionLayerTrack $StudyLayer --LanLayerTrack 0 \
        --kthSingVec -20 --attackMode vis
done
```

### Spectral Subspace Alignment — Tracking During Optimization

To track subspace alignment over the course of attack optimization:

```bash
export CUDA_VISIBLE_DEVICES=3
conda activate gemma3
cd interpretAttacks

python gemma_attack/RealSpectralSubspaceAlignmentTracker.py \
    --attck_type bsaTrack --desired_norm_l_inf 0.05 --learningRate 0.0001 --num_steps 1000 \
    --attackSample 551 --AttackStartLayer 15 --numLayerstAtAtime 1 \
    --VisionLayerTrack 0 --LanLayerTrack 0 --kthSingVec 0
python gemma_attack/RealSpectralSubspaceAlignmentTracker.py \
    --attck_type bsaTrack --desired_norm_l_inf 0.05 --learningRate 0.0001 --num_steps 1000 \
    --attackSample 551 --AttackStartLayer 15 --numLayerstAtAtime 1 \
    --VisionLayerTrack 0 --LanLayerTrack 0 --kthSingVec -1

python gemma_attack/RealSpectralSubspaceAlignmentTracker.py \
    --attck_type bsaTrack --desired_norm_l_inf 0.05 --learningRate 0.0001 --num_steps 1000 \
    --attackSample 551 --AttackStartLayer 15 --numLayerstAtAtime 1 \
    --VisionLayerTrack 1 --LanLayerTrack 1 --kthSingVec 0
python gemma_attack/RealSpectralSubspaceAlignmentTracker.py \
    --attck_type bsaTrack --desired_norm_l_inf 0.05 --learningRate 0.0001 --num_steps 1000 \
    --attackSample 551 --AttackStartLayer 15 --numLayerstAtAtime 1 \
    --VisionLayerTrack 1 --LanLayerTrack 1 --kthSingVec -1

python gemma_attack/RealSpectralSubspaceAlignmentTracker.py \
    --attck_type bsaTrack --desired_norm_l_inf 0.05 --learningRate 0.0001 --num_steps 1000 \
    --attackSample 551 --AttackStartLayer 15 --numLayerstAtAtime 1 \
    --VisionLayerTrack -1 --LanLayerTrack -1 --kthSingVec 0
python gemma_attack/RealSpectralSubspaceAlignmentTracker.py \
    --attck_type bsaTrack --desired_norm_l_inf 0.05 --learningRate 0.0001 --num_steps 1000 \
    --attackSample 551 --AttackStartLayer 15 --numLayerstAtAtime 1 \
    --VisionLayerTrack -1 --LanLayerTrack -1 --kthSingVec -1

python gemma_attack/RealSpectralSubspaceAlignmentTracker.py \
    --attck_type bsaTrack --desired_norm_l_inf 0.05 --learningRate 0.0001 --num_steps 1000 \
    --attackSample 551 --AttackStartLayer 15 --numLayerstAtAtime 1 \
    --VisionLayerTrack -2 --LanLayerTrack -2 --kthSingVec 0
python gemma_attack/RealSpectralSubspaceAlignmentTracker.py \
    --attck_type bsaTrack --desired_norm_l_inf 0.05 --learningRate 0.0001 --num_steps 1000 \
    --attackSample 551 --AttackStartLayer 15 --numLayerstAtAtime 1 \
    --VisionLayerTrack -2 --LanLayerTrack -2 --kthSingVec -1
```

---

## Qwen 2.5

_Similar experiments for Qwen 2.5 will be added here._

---

## LLaVA

_Similar experiments for LLaVA will be added here._
