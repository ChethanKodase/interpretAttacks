



'''


##########################################################################################################################################################################################################################################################################################

export CUDA_VISIBLE_DEVICES=4
conda activate gemma3
cd interpretAttacks

for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_KSAm_loop1.py --attck_type saa_loop --desired_norm_l_inf 0.0009 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --towardsNull 0.1 --whichMLP gate_proj --whichMLPvis fc1 --chosenLanLayers 0 --chosenVisLayers 0
done




export CUDA_VISIBLE_DEVICES=1
conda activate gemma3
cd interpretAttacks
for ATTACK_SAMPLE in $(seq 1 5); do
    for CHOSEN_LAN_LAYER in $(seq 0 6); do
        for CHOSEN_VIS_LAYER in $(seq 0 26); do
            python gemma_attack/gemma3AttackImgenet_KSAm_loop1.py --attck_type saa_loop --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 100 --attackSample $ATTACK_SAMPLE --towardsNull 0.1 --whichMLP gate_proj --whichMLPvis fc1 --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER
            python gemma_attack/gemma3AttackImgenet_KSAm_loop1.py --attck_type saa_loop --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 100 --attackSample $ATTACK_SAMPLE --towardsNull 0.5 --whichMLP gate_proj --whichMLPvis fc1 --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER
        done
    done
done

export CUDA_VISIBLE_DEVICES=2
conda activate gemma3
cd interpretAttacks
for ATTACK_SAMPLE in $(seq 1 5); do
    for CHOSEN_LAN_LAYER in $(seq 0 6); do
        for CHOSEN_VIS_LAYER in $(seq 0 26); do
            python gemma_attack/gemma3AttackImgenet_KSAm_loop1.py --attck_type saa_loop --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 100 --attackSample $ATTACK_SAMPLE --towardsNull 0.1 --whichMLP gate_proj --whichMLPvis fc2 --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER
            python gemma_attack/gemma3AttackImgenet_KSAm_loop1.py --attck_type saa_loop --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 100 --attackSample $ATTACK_SAMPLE --towardsNull 0.5 --whichMLP gate_proj --whichMLPvis fc2 --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER
        done
    done
done


for ATTACK_SAMPLE in $(seq 1 5); do
    for CHOSEN_LAN_LAYER in $(seq 0 6); do
        for CHOSEN_VIS_LAYER in $(seq 0 26); do
            python gemma_attack/gemma3AttackImgenet_KSAm_loop1.py --attck_type saa_loop --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 100 --attackSample $ATTACK_SAMPLE --towardsNull 0.1 --whichMLP gate_proj --whichMLPvis out_proj --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER
            python gemma_attack/gemma3AttackImgenet_KSAm_loop1.py --attck_type saa_loop --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 100 --attackSample $ATTACK_SAMPLE --towardsNull 0.5 --whichMLP gate_proj --whichMLPvis out_proj --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER
        done
    done
done

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


export CUDA_VISIBLE_DEVICES=3
conda activate gemma3
cd interpretAttacks
for ATTACK_SAMPLE in $(seq 1 5); do
    for CHOSEN_LAN_LAYER in $(seq 0 6); do
        for CHOSEN_VIS_LAYER in $(seq 0 26); do
            python gemma_attack/gemma3AttackImgenet_KSAm_loop1.py --attck_type saa_loop --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 100 --attackSample $ATTACK_SAMPLE --towardsNull 0.1 --whichMLP up_proj --whichMLPvis fc1 --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER
            python gemma_attack/gemma3AttackImgenet_KSAm_loop1.py --attck_type saa_loop --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 100 --attackSample $ATTACK_SAMPLE --towardsNull 0.5 --whichMLP up_proj --whichMLPvis fc1 --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER
        done
    done
done

export CUDA_VISIBLE_DEVICES=4
conda activate gemma3
cd interpretAttacks
for ATTACK_SAMPLE in $(seq 1 5); do
    for CHOSEN_LAN_LAYER in $(seq 0 6); do
        for CHOSEN_VIS_LAYER in $(seq 0 26); do
            python gemma_attack/gemma3AttackImgenet_KSAm_loop1.py --attck_type saa_loop --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 100 --attackSample $ATTACK_SAMPLE --towardsNull 0.1 --whichMLP up_proj --whichMLPvis fc2 --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER
            python gemma_attack/gemma3AttackImgenet_KSAm_loop1.py --attck_type saa_loop --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 100 --attackSample $ATTACK_SAMPLE --towardsNull 0.5 --whichMLP up_proj --whichMLPvis fc2 --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER
        done
    done
done


export CUDA_VISIBLE_DEVICES=5
conda activate gemma3
cd interpretAttacks
for ATTACK_SAMPLE in $(seq 1 5); do
    for CHOSEN_LAN_LAYER in $(seq 0 6); do
        for CHOSEN_VIS_LAYER in $(seq 0 26); do
            python gemma_attack/gemma3AttackImgenet_KSAm_loop1.py --attck_type saa_loop --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 100 --attackSample $ATTACK_SAMPLE --towardsNull 0.1 --whichMLP up_proj --whichMLPvis out_proj --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER
            python gemma_attack/gemma3AttackImgenet_KSAm_loop1.py --attck_type saa_loop --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 100 --attackSample $ATTACK_SAMPLE --towardsNull 0.5 --whichMLP up_proj --whichMLPvis out_proj --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER
        done
    done
done


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


export CUDA_VISIBLE_DEVICES=6
conda activate gemma3
cd interpretAttacks
for ATTACK_SAMPLE in $(seq 1 5); do
    for CHOSEN_LAN_LAYER in $(seq 0 6); do
        for CHOSEN_VIS_LAYER in $(seq 0 26); do
            python gemma_attack/gemma3AttackImgenet_KSAm_loop1.py --attck_type saa_loop --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 100 --attackSample $ATTACK_SAMPLE --towardsNull 0.1 --whichMLP down_proj --whichMLPvis fc1 --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER
            python gemma_attack/gemma3AttackImgenet_KSAm_loop1.py --attck_type saa_loop --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 100 --attackSample $ATTACK_SAMPLE --towardsNull 0.5 --whichMLP down_proj --whichMLPvis fc1 --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER
        done
    done
done


for ATTACK_SAMPLE in $(seq 1 5); do
    for CHOSEN_LAN_LAYER in $(seq 0 6); do
        for CHOSEN_VIS_LAYER in $(seq 0 26); do
            python gemma_attack/gemma3AttackImgenet_KSAm_loop1.py --attck_type saa_loop --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 100 --attackSample $ATTACK_SAMPLE --towardsNull 0.1 --whichMLP down_proj --whichMLPvis fc2 --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER
            python gemma_attack/gemma3AttackImgenet_KSAm_loop1.py --attck_type saa_loop --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 100 --attackSample $ATTACK_SAMPLE --towardsNull 0.5 --whichMLP down_proj --whichMLPvis fc2 --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER
        done
    done
done


export CUDA_VISIBLE_DEVICES=7
conda activate gemma3
cd interpretAttacks
for ATTACK_SAMPLE in $(seq 1 5); do
    for CHOSEN_LAN_LAYER in $(seq 0 6); do
        for CHOSEN_VIS_LAYER in $(seq 0 26); do
            python gemma_attack/gemma3AttackImgenet_KSAm_loop1.py --attck_type saa_loop --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 100 --attackSample $ATTACK_SAMPLE --towardsNull 0.1 --whichMLP down_proj --whichMLPvis out_proj --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER
            python gemma_attack/gemma3AttackImgenet_KSAm_loop1.py --attck_type saa_loop --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 100 --attackSample $ATTACK_SAMPLE --towardsNull 0.5 --whichMLP down_proj --whichMLPvis out_proj --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER
        done
    done
done



--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

allowed = {"gate_proj", "up_proj", "down_proj"} # for lan
allowed = {"fc1", "fc2", "out_proj"} # for vis


export CUDA_VISIBLE_DEVICES=7
conda activate gemma3
cd interpretAttacks
for ATTACK_SAMPLE in $(seq 4 10); do
    for CHOSEN_LAN_LAYER in $(seq 0 6); do
        for CHOSEN_VIS_LAYER in $(seq 0 26); do
            python gemma_attack/gemma3AttackImgenet_KSAm_loop1.py --attck_type saa_loop --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 100 --attackSample $ATTACK_SAMPLE --towardsNull 0.1 --whichMLP down_proj --whichMLPvis out_proj --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER
            python gemma_attack/gemma3AttackImgenet_KSAm_loop1.py --attck_type saa_loop --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 100 --attackSample $ATTACK_SAMPLE --towardsNull 0.5 --whichMLP down_proj --whichMLPvis out_proj --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER
        done
    done
done


export CUDA_VISIBLE_DEVICES=7
conda activate gemma3
cd interpretAttacks
for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_KSAm_loop1.py --attck_type saa_loop --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --towardsNull 0.5 --whichMLP down_proj --whichMLPvis out_proj --chosenLanLayers 2 --chosenVisLayers 23
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_KSAm_loop1.py --attck_type saa_loop --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --towardsNull 0.5 --whichMLP down_proj --whichMLPvis out_proj --chosenLanLayers 2 --chosenVisLayers 23
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_KSAm_loop1.py --attck_type saa_loop --desired_norm_l_inf 0.003 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --towardsNull 0.5 --whichMLP down_proj --whichMLPvis out_proj --chosenLanLayers 2 --chosenVisLayers 23
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_KSAm_loop1.py --attck_type saa_loop --desired_norm_l_inf 0.002 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --towardsNull 0.5 --whichMLP down_proj --whichMLPvis out_proj --chosenLanLayers 2 --chosenVisLayers 23
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_KSAm_loop1.py --attck_type saa_loop --desired_norm_l_inf 0.001 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --towardsNull 0.5 --whichMLP down_proj --whichMLPvis out_proj --chosenLanLayers 2 --chosenVisLayers 23
done




export CUDA_VISIBLE_DEVICES=0
conda activate gemma3
cd interpretAttacks
for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_KSAm_loop1.py --attck_type saa_loop --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --towardsNull 0.1 --whichMLP gate_proj --whichMLPvis out_proj --chosenLanLayers 0 --chosenVisLayers 26
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_KSAm_loop1.py --attck_type saa_loop --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --towardsNull 0.1 --whichMLP gate_proj --whichMLPvis out_proj --chosenLanLayers 0 --chosenVisLayers 26
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_KSAm_loop1.py --attck_type saa_loop --desired_norm_l_inf 0.003 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --towardsNull 0.1 --whichMLP gate_proj --whichMLPvis out_proj --chosenLanLayers 0 --chosenVisLayers 26
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_KSAm_loop1.py --attck_type saa_loop --desired_norm_l_inf 0.002 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --towardsNull 0.1 --whichMLP gate_proj --whichMLPvis out_proj --chosenLanLayers 0 --chosenVisLayers 26
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_KSAm_loop1.py --attck_type saa_loop --desired_norm_l_inf 0.001 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --towardsNull 0.1 --whichMLP gate_proj --whichMLPvis out_proj --chosenLanLayers 0 --chosenVisLayers 26
done




export CUDA_VISIBLE_DEVICES=5
conda activate gemma3
cd interpretAttacks
for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_KSAm_loop1.py --attck_type saa_loop --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --towardsNull 0.5 --whichMLP gate_proj --whichMLPvis out_proj \
    --chosenLanLayers 0 1 2 3 4 \
    --chosenVisLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26  
done


export CUDA_VISIBLE_DEVICES=4
conda activate gemma3
cd interpretAttacks
for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_KSAm_loop1.py --attck_type saa_loop --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --towardsNull 0.5 --whichMLP gate_proj --whichMLPvis out_proj \
    --chosenLanLayers 0 1 2 3 4 \
    --chosenVisLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26  
done


for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_KSAm_loop1.py --attck_type saa_loop --desired_norm_l_inf 0.003 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --towardsNull 0.5 --whichMLP gate_proj --whichMLPvis out_proj \
    --chosenLanLayers 0 1 2 3 4 \
    --chosenVisLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26  
done



export CUDA_VISIBLE_DEVICES=3
conda activate gemma3
cd interpretAttacks
for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_KSAm_loop1.py --attck_type saa_loop --desired_norm_l_inf 0.002 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --towardsNull 0.1 --whichMLP gate_proj --whichMLPvis out_proj \
    --chosenLanLayers 0 1 2 3 4 \
    --chosenVisLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26  
done


for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_KSAm_loop1.py --attck_type saa_loop --desired_norm_l_inf 0.002 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --towardsNull 0.1 --whichMLP up_proj --whichMLPvis fc1 \
    --chosenLanLayers 4 \
    --chosenVisLayers 9
done




CONFIG: whichMLP=up_proj, whichMLPvis=fc1, towardsNull=0.1, chosenLanLayers=[4], chosenVisLayers=[9]


export CUDA_VISIBLE_DEVICES=0
conda activate gemma3
cd interpretAttacks
for ATTACK_SAMPLE in $(seq 1 50); do
python gemma_attack/gemma3AttackImgenet_Rayleigh.py \
    --desired_norm_l_inf 0.002 \
    --learningRate 0.001 \
    --num_steps 1000 \
    --attackSample $ATTACK_SAMPLE \
    --whichMLP gate_proj \
    --whichMLPvis out_proj \
    --chosenLanLayers 0 \
    --chosenVisLayers 0 2 23 26
done


'''

# ============================================================
# RAYLEIGH QUOTIENT ADVERSARIAL ATTACK
# SSPMA WITHOUT EXPLICIT SVD
# ============================================================

import os
import re
import argparse
import random
import numpy as np

import torch
import torch.nn.functional as F

from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(42)


# ============================================================
# IMAGE UTILS
# ============================================================

def pil_to_tensor01(pil_img: Image.Image) -> torch.Tensor:
    arr = np.array(
        pil_img.convert("RGB"),
        dtype=np.float32
    ) / 255.0

    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    return t


def tensor01_to_pil(t01: torch.Tensor) -> Image.Image:

    if t01.dim() == 4:
        t01 = t01[0]

    t01 = t01.detach().cpu().clamp(0, 1)

    arr = (
        t01.permute(1, 2, 0).numpy() * 255.0
    ).round().clip(0, 255).astype(np.uint8)

    return Image.fromarray(arr)


# ============================================================
# DIFFERENTIABLE PREPROCESSING
# ============================================================

def _get_target_hw(image_processor):

    ip = image_processor

    target_h = target_w = None

    crop = getattr(ip, "crop_size", None)

    if isinstance(crop, dict):

        target_h = crop.get("height", None)
        target_w = crop.get("width", None)

    elif isinstance(crop, int):

        target_h = target_w = crop

    if target_h is None or target_w is None:

        size = getattr(ip, "size", None)

        if isinstance(size, dict):

            if "height" in size and "width" in size:

                target_h = size["height"]
                target_w = size["width"]

            elif "shortest_edge" in size:

                target_h = target_w = size["shortest_edge"]

        elif isinstance(size, int):

            target_h = target_w = size

    if target_h is None or target_w is None:
        target_h = target_w = 896

    return int(target_h), int(target_w)


def resize_keep_aspect_center_crop(
    x: torch.Tensor,
    target_h: int,
    target_w: int,
):

    _, _, H, W = x.shape

    scale = max(target_h / H, target_w / W)

    newH = int(round(H * scale))
    newW = int(round(W * scale))

    x_resized = F.interpolate(
        x,
        size=(newH, newW),
        mode="bilinear",
        align_corners=False,
    )

    top = max((newH - target_h) // 2, 0)
    left = max((newW - target_w) // 2, 0)

    x_crop = x_resized[
        :,
        :,
        top:top + target_h,
        left:left + target_w,
    ]

    pad_h = target_h - x_crop.shape[2]
    pad_w = target_w - x_crop.shape[3]

    if pad_h > 0 or pad_w > 0:

        x_crop = F.pad(
            x_crop,
            (0, max(pad_w, 0), 0, max(pad_h, 0))
        )

    return x_crop


def normalize_like_processor(x01, image_processor):

    mean = torch.tensor(
        image_processor.image_mean,
        dtype=x01.dtype,
        device=x01.device
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        image_processor.image_std,
        dtype=x01.dtype,
        device=x01.device
    ).view(1, 3, 1, 1)

    return (x01 - mean) / std


def gemma_preprocess_differentiable(x01, processor):

    ip = processor.image_processor

    th, tw = _get_target_hw(ip)

    x = resize_keep_aspect_center_crop(
        x01,
        th,
        tw
    )

    x = normalize_like_processor(x, ip)

    return x


# ============================================================
# TEMPLATE INPUTS
# ============================================================

def build_template_inputs(
    processor,
    question,
    pil_image,
    device,
):

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )

    template = processor(
        text=[prompt],
        images=[pil_image],
        return_tensors="pt",
    )

    template = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in template.items()
    }

    return template


# ============================================================
# GENERATION
# ============================================================

def run_generation_with_pixel_values(
    model,
    processor,
    template_inputs,
    pixel_values,
    max_new_tokens=128,
):

    model.eval()

    inputs = {
        k: (v.clone() if torch.is_tensor(v) else v)
        for k, v in template_inputs.items()
    }

    inputs["pixel_values"] = pixel_values

    with torch.no_grad():

        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    input_ids = inputs["input_ids"]

    gen_only = out_ids[:, input_ids.shape[1]:]

    return processor.batch_decode(
        gen_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]


# ============================================================
# HOOK STORAGE
# ============================================================

layer_inputs = {}


def make_pre_hook(name):

    def hook(module, inputs):

        layer_inputs[name] = inputs[0]

    return hook


# ============================================================
# LAYER PARSING
# ============================================================

def extract_language_layer_idx(name):

    m = re.search(
        r"language_model\.model\.layers\.(\d+)\.",
        name
    )

    if m is None:
        return None

    return int(m.group(1))


def extract_vision_layer_idx(name):

    patterns = [
        r"vision_tower\.vision_model\.encoder\.layers\.(\d+)\.",
        r"vision_model\.encoder\.layers\.(\d+)\.",
        r"multi_modal_projector.*layers\.(\d+)\.",
    ]

    for p in patterns:

        m = re.search(p, name)

        if m is not None:
            return int(m.group(1))

    return None


# ============================================================
# TARGET FILTERS
# ============================================================

def is_language_target(
    name,
    chosen_lan_layers_set=None,
    whichMLP="gate_proj",
):

    if "language_model.model.layers." not in name:
        return False

    layer_idx = extract_language_layer_idx(name)

    if layer_idx is None:
        return False

    if (
        chosen_lan_layers_set is not None
        and layer_idx not in chosen_lan_layers_set
    ):
        return False

    return name.endswith(f".mlp.{whichMLP}")


def is_vision_target(
    name,
    chosen_vis_layers_set=None,
    whichMLPvis="fc2",
):

    if "vision" not in name:
        return False

    layer_idx = extract_vision_layer_idx(name)

    if layer_idx is None:
        return False

    if (
        chosen_vis_layers_set is not None
        and layer_idx not in chosen_vis_layers_set
    ):
        return False

    if whichMLPvis == "out_proj":
        return name.endswith(".self_attn.out_proj")

    return name.endswith(f".mlp.{whichMLPvis}")


# ============================================================
# COLLECT MODULES
# ============================================================

def collect_target_modules(
    model,
    chosen_lan_layers=None,
    chosen_vis_layers=None,
    whichMLP="gate_proj",
    whichMLPvis="fc2",
):

    chosen_lan_layers_set = (
        None
        if chosen_lan_layers is None
        else set(chosen_lan_layers)
    )

    chosen_vis_layers_set = (
        None
        if chosen_vis_layers is None
        else set(chosen_vis_layers)
    )

    targets = []

    for name, module in model.named_modules():

        if not hasattr(module, "weight"):
            continue

        if not torch.is_tensor(module.weight):
            continue

        if module.weight.ndim != 2:
            continue

        if is_language_target(
            name,
            chosen_lan_layers_set,
            whichMLP,
        ):

            targets.append({
                "name": name,
                "module": module,
                "kind": "language",
                "layer_idx": extract_language_layer_idx(name),
            })

        elif is_vision_target(
            name,
            chosen_vis_layers_set,
            whichMLPvis,
        ):

            targets.append({
                "name": name,
                "module": module,
                "kind": "vision",
                "layer_idx": extract_vision_layer_idx(name),
            })

    return targets


# ============================================================
# RAYLEIGH QUOTIENT LOSS
# ============================================================

def get_rayleigh_quotient_loss(
    InputToLayer,
    module,
    eps=1e-8,
):

    H = InputToLayer

    if isinstance(H, (tuple, list)):
        H = H[0]

    if H.dim() == 3:

        H = H.reshape(-1, H.shape[-1])

    elif H.dim() == 2:
        pass

    else:

        H = H.view(-1, H.shape[-1])

    H32 = H.to(torch.float32)

    W32 = module.weight.to(torch.float32)

    # transformed features
    WH = H32 @ W32.T

    numerator = (WH ** 2).sum(dim=1)

    denominator = (H32 ** 2).sum(dim=1) + eps

    rayleigh = numerator / denominator

    return rayleigh.mean()


# ============================================================
# BUILD TARGET SPECS
# ============================================================

def build_target_specs_with_subspaces(
    model,
    chosen_lan_layers=None,
    chosen_vis_layers=None,
    whichMLP="gate_proj",
    whichMLPvis="fc2",
):

    targets = collect_target_modules(
        model,
        chosen_lan_layers,
        chosen_vis_layers,
        whichMLP,
        whichMLPvis,
    )

    specs = []

    print("\n========== TARGET MODULES ==========")

    for t in targets:

        spec = {
            "name": t["name"],
            "module": t["module"],
            "kind": t["kind"],
            "layer_idx": t["layer_idx"],
        }

        specs.append(spec)

        print(
            f"{t['kind']:8s} | "
            f"layer={t['layer_idx']:3d} | "
            f"{t['name']}"
        )

    print("====================================\n")

    return specs


# ============================================================
# REGISTER HOOKS
# ============================================================

def register_all_hooks(targets):

    handles = []

    for spec in targets:

        h = spec["module"].register_forward_pre_hook(
            make_pre_hook(spec["name"])
        )

        handles.append(h)

    return handles


# ============================================================
# AGGREGATED LOSS
# ============================================================

def aggregated_rayleigh_loss(
    target_specs,
    device,
):

    lang_losses = []
    vis_losses = []

    for spec in target_specs:

        name = spec["name"]

        if name not in layer_inputs:
            continue

        H = layer_inputs[name]

        loss_i = get_rayleigh_quotient_loss(
            H,
            spec["module"],
        )

        if spec["kind"] == "language":
            lang_losses.append(loss_i)

        elif spec["kind"] == "vision":
            vis_losses.append(loss_i)

    language_loss = torch.tensor(0.0, device=device)
    vision_loss = torch.tensor(0.0, device=device)

    if len(lang_losses) > 0:
        language_loss = torch.stack(lang_losses).mean()

    if len(vis_losses) > 0:
        vision_loss = torch.stack(vis_losses).mean()

    total_used = len(lang_losses) + len(vis_losses)

    return language_loss, vision_loss, total_used


# ============================================================
# ATTACK
# ============================================================

def adam_attack_original_space(
    model,
    processor,
    template_inputs,
    x_orig01,
    num_steps,
    lr,
    epsilon,
    device,
    chosenLanLayers=None,
    chosenVisLayers=None,
    whichMLP="gate_proj",
    whichMLPvis="fc2",
):

    x_orig01 = x_orig01.detach().to(device)

    delta = 0.001 * torch.randn_like(x_orig01)

    delta.requires_grad_(True)

    opt = torch.optim.Adam([delta], lr=lr)

    best_loss = 1e20
    best_delta = delta.detach().clone()

    model.eval()

    target_specs = build_target_specs_with_subspaces(
        model,
        chosenLanLayers,
        chosenVisLayers,
        whichMLP,
        whichMLPvis,
    )

    hook_handles = register_all_hooks(target_specs)

    adv_inputs = {
        k: (v.clone() if torch.is_tensor(v) else v)
        for k, v in template_inputs.items()
    }

    adv_inputs["labels"] = template_inputs["input_ids"]

    adv_inputs["use_cache"] = False

    nllAlpha = 0.0

    try:

        for step in range(num_steps):

            layer_inputs.clear()

            x_adv01 = (
                x_orig01 + delta
            ).clamp(0.0, 1.0)

            x_adv01 = torch.max(
                torch.min(
                    x_adv01,
                    x_orig01 + epsilon,
                ),
                x_orig01 - epsilon,
            ).clamp(0.0, 1.0)

            pv_adv = gemma_preprocess_differentiable(
                x_adv01,
                processor,
            )

            adv_inputs["pixel_values"] = pv_adv

            outputs = model(
                **adv_inputs,
                output_hidden_states=False,
                return_dict=True,
            )

            language_loss, vision_loss, total_used = (
                aggregated_rayleigh_loss(
                    target_specs,
                    device,
                )
            )

            if total_used == 0:
                raise RuntimeError(
                    "No hooked modules used."
                )

            rayleigh_loss = (
                language_loss + vision_loss
            )

            nll_loss = -outputs.loss.float()

            attack_loss = (
                rayleigh_loss
                + nllAlpha * nll_loss
            )

            opt.zero_grad(set_to_none=True)

            attack_loss.backward()

            opt.step()

            with torch.no_grad():

                delta.data.clamp_(
                    -epsilon,
                    epsilon
                )

            lv = float(rayleigh_loss.item())

            if (
                step == 0
                or (step + 1) % 10 == 0
            ):

                print(
                    f"[step {step+1}/{num_steps}] "
                    f"rayleigh={lv:.6f} "
                    f"nll={float(nll_loss.item()):.6f}"
                )

            if lv < best_loss:

                best_loss = lv

                best_delta = delta.detach().clone()

            del outputs

    finally:

        for h in hook_handles:
            h.remove()

    with torch.no_grad():

        x_adv01_final = (
            x_orig01 + best_delta
        ).clamp(0.0, 1.0)

    return x_adv01_final


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--desired_norm_l_inf",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--learningRate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--num_steps",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--attackSample",
        type=str,
        default="nature",
    )

    parser.add_argument(
        "--whichMLP",
        type=str,
        default="gate_proj",
    )

    parser.add_argument(
        "--whichMLPvis",
        type=str,
        default="fc2",
    )

    parser.add_argument(
        "--chosenLanLayers",
        type=int,
        nargs="+",
        default=None,
    )

    parser.add_argument(
        "--chosenVisLayers",
        type=int,
        nargs="+",
        default=None,
    )

    args = parser.parse_args()

    chosenLanLayers = args.chosenLanLayers or []
    chosenVisLayers = args.chosenVisLayers or []

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    dtype = (
        torch.bfloat16
        if device.type == "cuda"
        else torch.float32
    )

    MODEL_PATH = "../illcond/gemma_attack/Gemma3-4b"

    IMAGE_PATH = (
        f"gemma_attack/dataSamplesForQuant/"
        f"{args.attackSample}.JPEG"
    )

    QUESTION = "What is shown in this image?"

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        padding_side="left",
    )

    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
    ).to(device)

    model.eval()

    pil = Image.open(IMAGE_PATH).convert("RGB")

    x_orig01 = pil_to_tensor01(pil).to(device)

    template_inputs = build_template_inputs(
        processor,
        QUESTION,
        pil,
        device,
    )

    pv_clean = gemma_preprocess_differentiable(
        x_orig01,
        processor,
    )

    print("\n=== CLEAN OUTPUT ===")

    clean_text = run_generation_with_pixel_values(
        model,
        processor,
        template_inputs,
        pv_clean,
    )

    print(clean_text)

    x_adv01 = adam_attack_original_space(
        model=model,
        processor=processor,
        template_inputs=template_inputs,
        x_orig01=x_orig01,
        num_steps=args.num_steps,
        lr=args.learningRate,
        epsilon=args.desired_norm_l_inf,
        device=device,
        chosenLanLayers=chosenLanLayers,
        chosenVisLayers=chosenVisLayers,
        whichMLP=args.whichMLP,
        whichMLPvis=args.whichMLPvis,
    )

    pv_adv = gemma_preprocess_differentiable(
        x_adv01,
        processor,
    )

    print("\n=== ADVERSARIAL OUTPUT ===")

    adv_text = run_generation_with_pixel_values(
        model,
        processor,
        template_inputs,
        pv_adv,
    )

    print(adv_text)


if __name__ == "__main__":
    main()