# interpretAttacks

### On Adversarial Robustness of Vision-Language Models

This repository contains the code used to craft, evaluate, and interpret adversarial attacks against Vision-Language Models (VLMs). It includes the proposed **SSGRA** attack alongside a set of baseline attacks (**BSA**, **DRA**, **EGA**, **FDA**, **CE**, **SSPA**), FLOPs-based cost comparisons, and tooling to study how these attacks align with the spectral (singular-value) subspaces of the model's intermediate representations, across three model families: **Gemma 3**, **Qwen 2.5**, and **LLaVA**.

All commands below assume you are running from the repository root with the appropriate conda environment activated and a GPU selected via `CUDA_VISIBLE_DEVICES`.

## Environment Setup

Each model family has its own conda environment, provided as a `.yml` file:

| Model     | Environment file        | Conda env name |
|-----------|--------------------------|----------------|
| Gemma 3   | `Gemma3environment.yml`  | `gemma3`       |
| Qwen 2.5  | `QwenEnvironment.yml`    | `vlmAttack`    |
| LLaVA     | *(not yet provided)*    | `llava15`      |

```bash
conda env create -f Gemma3environment.yml
conda env create -f QwenEnvironment.yml
```

> **Note:** the environment file for `llava15` hasn't been added yet — the original list named `Gemma3environment.yml` twice, so the duplicate was dropped here. Please add the correct filename for LLaVA once available.

---

## Table of Contents

- [Gemma 3](#gemma-3)
  - [Proposed Method — SSGRA](#proposed-method--ssgra)
  - [Baseline Attacks](#baseline-attacks)
  - [FLOPs Estimation](#flops-estimation)
  - [Singular Value Spectrum & Conditioning](#singular-value-spectrum--conditioning)
  - [Spectral Subspace Alignment — Baseline Representations](#spectral-subspace-alignment--baseline-representations)
  - [Spectral Subspace Alignment — Tracking During Optimization](#spectral-subspace-alignment--tracking-during-optimization)
  - [Quantitative Results](#quantitative-results)
- [Qwen 2.5](#qwen-25)
  - [Singular Value Spectrum & Conditioning](#singular-value-spectrum--conditioning-1)
  - [Proposed Method — SSGRA](#proposed-method--ssgra-1)
  - [Baseline Attacks](#baseline-attacks-1)
  - [FLOPs Estimation](#flops-estimation-1)
  - [Quantitative Results](#quantitative-results-1)
  - [Ablations](#ablations)
- [LLaVA](#llava)
  - [Proposed Method — SSGRA](#proposed-method--ssgra-2)
  - [Baseline Attacks](#baseline-attacks-2)
  - [FLOPs Estimation](#flops-estimation-2)
  - [Quantitative Results](#quantitative-results-2)

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

FLOPs are a property of the forward/backward pass, not of the sample content, so a single representative sample is enough — no need to loop over the full sample range.

```bash
# BSA
python gemma_attack/FLOPSestimation/gemma3AttackImgenet_BSA_flops.py \
    --attck_type bsa_flops --desired_norm_l_inf 0.05 --learningRate 0.001 --num_steps 1000 --attackSample 50

# DRA
python gemma_attack/FLOPSestimation/gemma3AttackImgenet_DRA_flops.py \
    --attck_type dra_flops --desired_norm_l_inf 0.01 --learningRate 0.001 --num_steps 1000 --attackSample 1

# EGA
python gemma_attack/FLOPSestimation/gemma3AttackImgenet_EGA1_flops.py \
    --attck_type ega_flops --desired_norm_l_inf 0.05 --learningRate 0.001 --num_steps 1000 --attackSample 50

# FDA
python gemma_attack/FLOPSestimation/gemma3AttackImgenet_FDAm_flops.py \
    --attck_type fdam_flops --desired_norm_l_inf 0.05 --learningRate 0.001 --num_steps 1000 --attackSample 50

# CE
python gemma_attack/FLOPSestimation/gemma3AttackImgenet_NLLm_flops.py \
    --attck_type nllm_flops --desired_norm_l_inf 0.05 --learningRate 0.001 --num_steps 1000 --attackSample 1

# SSGRA (proposed)
python gemma_attack/FLOPSestimation/gemma3AttackImgenet_SSGRA_flops.py \
    --attck_type saa_BSAexp_flops \
    --desired_norm_l_inf 0.0045 \
    --learningRate 0.001 \
    --num_steps 1000 \
    --attackSample 1 \
    --towardsNull 0.15 \
    --whichMLP up_proj \
    --whichMLPvis fc2 \
    --balancingAlpha 0.5 \
    --chosenLanLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 \
    --chosenVisLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26

# SSPA
python gemma_attack/FLOPSestimation/gemma3AttackImgenet_SSP_flops.py \
    --attck_type ssp_flops --desired_norm_l_inf 0.05 --learningRate 0.001 --num_steps 1000 --attackSample 50
```

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

### Quantitative Results

<details open>
<summary><b>BERTScore</b></summary>

```bash
export CUDA_VISIBLE_DEVICES=6
conda activate gemma3
cd interpretAttacks

python gemma_attack/gemma3BaselinesAndOursComparisionWithEpsilon.py \
    --learningRate 0.001 \
    --num_steps 1000 \
    --AttackStartLayer 0 \
    --AttackStartLayer_vis 11 \
    --numLayerstAtAtime 1 \
    --whichMLP gate_proj \
    --whichMLP_vis fc2 \
    --numSamplesConsidered 50
```
</details>

<details open>
<summary><b>ROUGE-L</b></summary>

```bash
export CUDA_VISIBLE_DEVICES=7
conda activate gemma3
cd interpretAttacks

python gemma_attack/gemma3BaselinesAndOursComparisionWithEpsilonRouge.py \
    --learningRate 0.001 \
    --num_steps 1000 \
    --AttackStartLayer 0 \
    --AttackStartLayer_vis 11 \
    --numLayerstAtAtime 1 \
    --whichMLP gate_proj \
    --whichMLP_vis fc2 \
    --numSamplesConsidered 50
```
</details>

---

## Qwen 2.5

### Singular Value Spectrum & Conditioning

To generate condition-number plots:

```bash
export CUDA_VISIBLE_DEVICES=1
conda deactivate
cd interpretAttacks/
conda activate vlmAttack
export PYTHONNOUSERSITE=1

python qwen/qwen2p5Conditioning.py
```

### Proposed Method — SSGRA

```bash
export CUDA_VISIBLE_DEVICES=3
conda deactivate
cd interpretAttacks/
conda activate vlmAttack
export PYTHONNOUSERSITE=1

for ATTACK_SAMPLE in $(seq 1 50); do
    python qwen/QwenUntargted_SSGRA.py \
        --attck_type saa_loop \
        --desired_norm_l_inf 0.0025 \
        --learningRate 0.001 \
        --num_steps 1000 \
        --attackSample $ATTACK_SAMPLE \
        --AttackStartLayer 0 \
        --numLayerstAtAtime 1 \
        --towardsNull 0.5 \
        --whichMLP gate_proj \
        --whichMLPVis gate_proj \
        --chosenLanLayers 2 \
        --chosenVisLayers 0 1 2 4 5 6 7 8 9 14 24
done
```

### Baseline Attacks

All baseline attacks below share the same environment:

```bash
export CUDA_VISIBLE_DEVICES=2
conda deactivate
cd interpretAttacks/
conda activate vlmAttack
export PYTHONNOUSERSITE=1
```

<details open>
<summary><b>BSA</b></summary>

```bash
for ATTACK_SAMPLE in $(seq 1 50); do
    python qwen/QwenUntargeted_BSA.py \
        --attck_type bsa --desired_norm_l_inf 0.0025 --learningRate 0.001 --num_steps 1000 \
        --attackSample $ATTACK_SAMPLE
done
```
</details>

<details open>
<summary><b>DRA</b></summary>

```bash
for ATTACK_SAMPLE in $(seq 1 50); do
    python qwen/QwenUntargeted_DRA.py \
        --attck_type dra --desired_norm_l_inf 0.0025 --learningRate 0.001 --num_steps 1000 \
        --attackSample $ATTACK_SAMPLE
done
```
</details>

<details open>
<summary><b>EGA</b></summary>

```bash
for ATTACK_SAMPLE in $(seq 1 50); do
    python qwen/QwenUntargeted_EGA.py \
        --attck_type ega --desired_norm_l_inf 0.0025 --learningRate 0.001 --num_steps 1000 \
        --ega_ratio 0.2 --mask_refresh_every 50 --attackSample $ATTACK_SAMPLE
done
```
</details>

<details open>
<summary><b>FDA</b></summary>

```bash
for ATTACK_SAMPLE in $(seq 1 50); do
    python qwen/QwenUntargeted_FDA.py \
        --attck_type fdam --desired_norm_l_inf 0.0025 --learningRate 0.001 --num_steps 1000 \
        --layer_start 1 --attackSample $ATTACK_SAMPLE
done
```
</details>

<details open>
<summary><b>CE</b></summary>

```bash
for ATTACK_SAMPLE in $(seq 1 50); do
    python qwen/QwenUntargeted_NLL.py \
        --attck_type nllm --desired_norm_l_inf 0.0025 --learningRate 0.001 --num_steps 1000 \
        --attackSample $ATTACK_SAMPLE
done
```
</details>

<details open>
<summary><b>SSPA</b></summary>

```bash
for ATTACK_SAMPLE in $(seq 1 50); do
    python qwen/QwenUntargeted_SSP.py \
        --attck_type ssp --desired_norm_l_inf 0.0025 --learningRate 0.001 --num_steps 1000 \
        --attackSample $ATTACK_SAMPLE
done
```
</details>

### FLOPs Estimation

Same reasoning as before — one sample is enough to measure FLOPs, so the sample loop is dropped.

```bash
export CUDA_VISIBLE_DEVICES=3
conda deactivate
cd interpretAttacks/
conda activate vlmAttack
export PYTHONNOUSERSITE=1

# BSA
python qwen/QwenUntargeted_BSA_flops.py \
    --attck_type bsa --desired_norm_l_inf 0.0008 --learningRate 0.001 --num_steps 1000 --attackSample 1

# DRA
python qwen/QwenUntargeted_DRA_flops.py \
    --attck_type dra --desired_norm_l_inf 0.001 --learningRate 0.001 --num_steps 1000 --attackSample 1

# EGA
python qwen/QwenUntargeted_EGA_flops.py \
    --attck_type ega --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 \
    --ega_ratio 0.2 --mask_refresh_every 50 --attackSample 1

# FDA
python qwen/QwenUntargeted_FDA_flops.py \
    --attck_type fdam --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 \
    --layer_start 1 --attackSample 1

# CE
python qwen/QwenUntargeted_NLL_flops.py \
    --attck_type nllm --desired_norm_l_inf 0.001 --learningRate 0.001 --num_steps 1000 --attackSample 1

# SSPA
python qwen/QwenUntargeted_SSP_flops.py \
    --attck_type ssp --desired_norm_l_inf 0.001 --learningRate 0.001 --num_steps 1000 --attackSample 1

# SSGRA (proposed)
python qwen/QwenUntargted_SSGRA_flops.py \
    --attck_type saa_loop \
    --desired_norm_l_inf 0.005 \
    --learningRate 0.001 \
    --num_steps 1000 \
    --attackSample 1 \
    --AttackStartLayer 0 \
    --numLayerstAtAtime 1 \
    --towardsNull 0.5 \
    --whichMLP gate_proj \
    --whichMLPVis gate_proj \
    --chosenLanLayers 2 \
    --chosenVisLayers 0 1 2 4 5 6 7 8 9 14 24
```

### Quantitative Results

```bash
export CUDA_VISIBLE_DEVICES=0
conda deactivate
cd interpretAttacks/
conda activate gemma3

python qwen/QwenBaselinesAndOursComparisionAllEpsilon.py \
    --learningRate 0.001 \
    --num_steps 1000 \
    --AttackStartLayer 0 \
    --numLayerstAtAtime 1 \
    --whichMLP gate_proj \
    --whichMLPVis gate_proj \
    --chosenLanLayers 2 \
    --chosenVisLayers 0 1 2 4 5 6 7 8 9 14 24 \
    --numSamplesConsidered 50

python qwen/QwenBaselinesAndOursComparisionAllEpsilonRouge.py \
    --learningRate 0.001 \
    --num_steps 1000 \
    --AttackStartLayer 0 \
    --numLayerstAtAtime 1 \
    --whichMLP gate_proj \
    --whichMLPVis gate_proj \
    --chosenLanLayers 2 \
    --chosenVisLayers 0 1 2 4 5 6 7 8 9 14 24 \
    --numSamplesConsidered 50
```

### Ablations

**Generate SSGRA-Top samples**

```bash
for ATTACK_SAMPLE in $(seq 1 50); do
    python qwen/QwenUntargted_SSGRATop.py \
        --attck_type saa_loopTop \
        --desired_norm_l_inf 0.002 \
        --learningRate 0.001 \
        --num_steps 1000 \
        --attackSample $ATTACK_SAMPLE \
        --AttackStartLayer 0 \
        --numLayerstAtAtime 1 \
        --towardsNull 0.5 \
        --whichMLP gate_proj \
        --whichMLPVis gate_proj \
        --chosenLanLayers 2 \
        --chosenVisLayers 0 1 2 4 5 6 7 8 9 14 24
done
```

**Ablation plots**

```bash
export CUDA_VISIBLE_DEVICES=3
conda deactivate
cd interpretAttacks/
conda activate gemma3

python qwen/A_ROUGEablationTopBottom.py \
    --learningRate 0.001 \
    --num_steps 1000 \
    --AttackStartLayer 0 \
    --numLayerstAtAtime 1 \
    --whichMLP gate_proj \
    --whichMLPVis gate_proj \
    --chosenLanLayers 2 \
    --chosenVisLayers 0 1 2 4 5 6 7 8 9 14 24 \
    --numSamplesConsidered 50

python qwen/BERTablationTopBottom.py \
    --learningRate 0.001 \
    --num_steps 1000 \
    --AttackStartLayer 0 \
    --numLayerstAtAtime 1 \
    --whichMLP gate_proj \
    --whichMLPVis gate_proj \
    --chosenLanLayers 2 \
    --chosenVisLayers 0 1 2 4 5 6 7 8 9 14 24 \
    --numSamplesConsidered 50

python qwen/QwenBaselinesAndOursComparisionAllEpsilonAblation.py \
    --learningRate 0.001 \
    --num_steps 1000 \
    --AttackStartLayer 0 \
    --numLayerstAtAtime 1 \
    --whichMLP gate_proj \
    --whichMLPVis gate_proj \
    --chosenLanLayers 2 \
    --chosenVisLayers 0 1 2 4 5 6 7 8 9 14 24 \
    --numSamplesConsidered 50

python qwen/QwenBaselinesAndOursComparisionAllEpsilonRougeAblation.py \
    --learningRate 0.001 \
    --num_steps 1000 \
    --AttackStartLayer 0 \
    --numLayerstAtAtime 1 \
    --whichMLP gate_proj \
    --whichMLPVis gate_proj \
    --chosenLanLayers 2 \
    --chosenVisLayers 0 1 2 4 5 6 7 8 9 14 24 \
    --numSamplesConsidered 50
```

---

## LLaVA

### Proposed Method — SSGRA

```bash
export CUDA_VISIBLE_DEVICES=3
cd interpretAttacks/
conda activate llava15

for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopComb.py \
        --attck_type saa_loopC \
        --desired_norm_l_inf 0.002 \
        --learningRate 0.001 \
        --num_steps 1000 \
        --attackSample $ATTACK_SAMPLE \
        --AttackStartLayer 0 \
        --numLayerstAtAtime 1 \
        --towardsNull 0.1 \
        --BalAlpha 0.06 \
        --whichMLP gate_proj \
        --whichMLPVis fc1 \
        --chosenLanLayers 1 3 \
        --chosenVisLayers 7 8 17 13
done
```

### Baseline Attacks

<details open>
<summary><b>BSA</b></summary>

```bash
export CUDA_VISIBLE_DEVICES=0
cd interpretAttacks/
conda activate llava15

for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llavaAttackBSA.py \
        --attck_type bsa --desired_norm_l_inf 0.002 --learningRate 0.001 --num_steps 1000 \
        --attackSample $ATTACK_SAMPLE
done
```
</details>

<details open>
<summary><b>DRA</b></summary>

```bash
export CUDA_VISIBLE_DEVICES=1
cd interpretAttacks/
conda activate llava15

for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llavaAttackDRA.py \
        --attck_type dra --desired_norm_l_inf 0.003 --learningRate 0.001 --num_steps 1000 \
        --attackSample $ATTACK_SAMPLE --WhichLayerDRA 16
done
```
</details>

<details open>
<summary><b>EGA</b></summary>

```bash
export CUDA_VISIBLE_DEVICES=2
cd interpretAttacks/
conda activate llava15

for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llavaAttackEGA.py \
        --attck_type ega --desired_norm_l_inf 0.003 --learningRate 0.001 --num_steps 1000 \
        --attackSample $ATTACK_SAMPLE --ega_ratio 0.2 --mask_refresh_every 50
done
```
</details>

<details open>
<summary><b>FDA</b></summary>

```bash
export CUDA_VISIBLE_DEVICES=3
cd interpretAttacks/
conda activate llava15

for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llavaAttackFDA.py \
        --attck_type fdam --desired_norm_l_inf 0.003 --learningRate 0.001 --num_steps 1000 \
        --attackSample $ATTACK_SAMPLE --layer_start 1
done
```
</details>

<details open>
<summary><b>CE</b></summary>

```bash
export CUDA_VISIBLE_DEVICES=0
cd interpretAttacks/
conda activate llava15

for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llavaAttackNLL.py \
        --attck_type nllm --desired_norm_l_inf 0.003 --learningRate 0.001 --num_steps 1000 \
        --attackSample $ATTACK_SAMPLE
done
```
</details>

<details open>
<summary><b>SSPA</b></summary>

```bash
export CUDA_VISIBLE_DEVICES=1
cd interpretAttacks/
conda activate llava15

for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llavaAttackSSP.py \
        --attck_type ssp --desired_norm_l_inf 0.003 --learningRate 0.001 --num_steps 1000 \
        --attackSample $ATTACK_SAMPLE
done
```
</details>

### FLOPs Estimation

As with the other models, FLOPs measurement doesn't need to loop over samples — a single fixed sample is used instead. The one exception is **DRA**, which sweeps over the `WhichLayerDRA` parameter rather than the sample index, so that loop is kept.

```bash
export CUDA_VISIBLE_DEVICES=3
cd interpretAttacks/
conda activate llava15

# SSGRA (proposed)
python llava_attack/llava_attack_imagenet_KSA_loopCombFlops.py \
    --attck_type saa_loopC_flops \
    --desired_norm_l_inf 0.005 \
    --learningRate 0.001 \
    --num_steps 1000 \
    --attackSample 1 \
    --AttackStartLayer 0 \
    --numLayerstAtAtime 1 \
    --towardsNull 0.1 \
    --BalAlpha 0.06 \
    --whichMLP gate_proj \
    --whichMLPVis fc1 \
    --chosenLanLayers 1 3 \
    --chosenVisLayers 7 8 17 13

# BSA
python llava_attack/llavaAttackBSAFlops.py \
    --attck_type llavaAttackBSAFlops --desired_norm_l_inf 0.0045 --learningRate 0.001 --num_steps 1000 --attackSample 1

# DRA — sweeps over WhichLayerDRA instead of sample index
for LayerDRA in $(seq 0 23); do
    python llava_attack/llavaAttackDRAFlops.py \
        --attck_type dra_flops --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 \
        --attackSample 3 --WhichLayerDRA $LayerDRA
done

# EGA
python llava_attack/llavaAttackEGAFlops.py \
    --attck_type ega_flops --desired_norm_l_inf 0.0045 --learningRate 0.001 --num_steps 1000 \
    --attackSample 1 --ega_ratio 0.2 --mask_refresh_every 50

# FDA
python llava_attack/llavaAttackFDAFlops.py \
    --attck_type fdam_flops --desired_norm_l_inf 0.0045 --learningRate 0.001 --num_steps 1000 \
    --attackSample 1 --layer_start 1

# CE
python llava_attack/llavaAttackNLLFLops.py \
    --attck_type nllm_flops --desired_norm_l_inf 0.0045 --learningRate 0.001 --num_steps 1000 --attackSample 1

# SSPA
python llava_attack/llavaAttackSSP_flops.py \
    --attck_type ssp_flops --desired_norm_l_inf 0.0045 --learningRate 0.001 --num_steps 1000 --attackSample 1
```

### Quantitative Results

<details open>
<summary><b>BERTScore</b></summary>

```bash
export CUDA_VISIBLE_DEVICES=3
conda deactivate
cd interpretAttacks/
conda activate gemma3

python llava_attack/llavaBaselinesAndOursComparisionWithEpsilonActual.py \
    --learningRate 0.001 \
    --num_steps 1000 \
    --AttackStartLayer 0 \
    --numLayerstAtAtime 1 \
    --whichMLP gate_proj \
    --whichMLP_vis fc1 \
    --numSamplesConsidered 50
```
</details>

<details open>
<summary><b>ROUGE-L</b></summary>

```bash
python llava_attack/llavaBaselinesAndOursComparisionWithEpsilonActualRouge.py \
    --learningRate 0.001 \
    --num_steps 1000 \
    --AttackStartLayer 0 \
    --numLayerstAtAtime 1 \
    --whichMLP gate_proj \
    --whichMLP_vis fc1 \
    --numSamplesConsidered 50
```

*(run in the same environment as BERTScore above)*
</details>
