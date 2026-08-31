


'''


Try
export CUDA_VISIBLE_DEVICES=3
conda activate gemma3
cd interpretAttacks
for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_SSGRA2.py --attck_type SSGRA2 --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --towardsNull 0.15 --whichMLP up_proj --whichMLPvis fc2 \
    --balancingAlpha 0.5\
    --chosenLanLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36\
    --chosenVisLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
done



for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_SSGRA2.py --attck_type SSGRA2 --desired_norm_l_inf 0.002 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --towardsNull 0.15 --whichMLP up_proj --whichMLPvis fc2 \
    --balancingAlpha 0.5\
    --chosenLanLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36\
    --chosenVisLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
done


for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_SSGRA2.py --attck_type SSGRA2 --desired_norm_l_inf 0.0025 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --towardsNull 0.15 --whichMLP up_proj --whichMLPvis fc2 \
    --balancingAlpha 0.5\
    --chosenLanLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36\
    --chosenVisLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
done


for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_SSGRA2.py --attck_type SSGRA2 --desired_norm_l_inf 0.003 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --towardsNull 0.15 --whichMLP up_proj --whichMLPvis fc2 \
    --balancingAlpha 0.5\
    --chosenLanLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36\
    --chosenVisLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
done


for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_SSGRA2.py --attck_type SSGRA2 --desired_norm_l_inf 0.0035 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --towardsNull 0.15 --whichMLP up_proj --whichMLPvis fc2 \
    --balancingAlpha 0.5\
    --chosenLanLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36\
    --chosenVisLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
done

for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_SSGRA2.py --attck_type SSGRA2 --desired_norm_l_inf 0.0045 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --towardsNull 0.15 --whichMLP up_proj --whichMLPvis fc2 \
    --balancingAlpha 0.5\
    --chosenLanLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36\
    --chosenVisLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
done

for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_SSGRA2.py --attck_type SSGRA2 --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --towardsNull 0.15 --whichMLP up_proj --whichMLPvis fc2 \
    --balancingAlpha 0.5\
    --chosenLanLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36\
    --chosenVisLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
done

(rest of the launch commands are unchanged from the original script and still apply
 -- just point them at gemma3AttackImgenet_SSGRA_FIXED.py)

'''



import os
import re
import sys
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


# ----------------------------
# Reproducibility
# ----------------------------
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

criterion = nn.MSELoss()


# ----------------------------
# FIX #1 (same root cause as the other two Gemma/Qwen scripts):
# the model runs in bfloat16, which only resolves values near cos()=1 to
# about 2^-8 (~0.0039) spacing. At your epsilon range, clean vs. adv
# hidden states start out extremely close, so cosine-similarity terms
# computed directly in bf16 can round away real (if small) gradient
# signal. Fixed by upcasting to float32 before the normalize/dot-product
# arithmetic. Note `getMeanAlignmentLossWithBottomSubspace` further down
# already did this correctly (`H = InputToLayer.float()`) -- this fix
# just extends the same treatment to `cos()`, `get_bsa_loss`, and
# `get_bsaVis_loss`, which didn't have it.
# ----------------------------
def cos(a, b):
    a = a.view(-1).float()
    b = b.view(-1).float()
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return (a * b).sum()


def wasserstein_distance(tensor_a, tensor_b):
    tensor_a_flat = torch.flatten(tensor_a).float()
    tensor_b_flat = torch.flatten(tensor_b).float()
    tensor_a_sorted, _ = torch.sort(tensor_a_flat)
    tensor_b_sorted, _ = torch.sort(tensor_b_flat)
    wasserstein_dist = torch.mean(torch.abs(tensor_a_sorted - tensor_b_sorted))
    return wasserstein_dist


# ----------------------------
# Losses (legacy / kept for compatibility with attck_type switch)
# ----------------------------
def get_grill_l2(outputs, outputsN):
    loss = 0.0
    for h, hn in zip(outputs.hidden_states, outputsN.hidden_states):
        loss = loss + criterion(h.float(), hn.float())
    return loss * criterion(h.float(), hn.float())


def get_grill_wass(outputs, outputsN, startPos, endPos):
    loss = 0.0
    for h, hn in zip(outputs.hidden_states[startPos:endPos], outputsN.hidden_states[startPos:endPos]):
        loss = loss + wasserstein_distance(h, hn)
    return loss


def get_grill_cos(outputs, outputsN):
    loss = 0.0
    for h, hn in zip(outputs.hidden_states, outputsN.hidden_states):
        loss = loss + (1.0 - cos(h, hn)) ** 2
    return loss * (1.0 - cos(outputs.logits, outputsN.logits)) ** 2


def get_oa_l2(outputs, outputsN):
    return criterion(outputs.logits.float(), outputsN.logits.float())


def get_oa_wass(outputs, outputsN):
    return wasserstein_distance(outputs.logits, outputsN.logits)


def get_oa_cos(outputs, outputsN):
    return (1.0 - cos(outputs.logits, outputsN.logits)) ** 2


# ----------------------------
# Utilities: image <-> tensor
# ----------------------------
def pil_to_tensor01(pil_img: Image.Image) -> torch.Tensor:
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


def tensor01_to_pil(t01: torch.Tensor) -> Image.Image:
    if t01.dim() == 4:
        t01 = t01[0]
    t01 = t01.detach().cpu().clamp(0, 1)
    arr = (t01.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ----------------------------
# Differentiable preprocessing
# ----------------------------
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


# ----------------------------
# FIX #2 (same bug as the plain grill_adv Gemma3 script, carried over into
# this SSGRA one): transformers/models/gemma3/image_processing_gemma3.py
# (Gemma3ImageProcessor) has default_to_square=True and just calls
# self.resize(image, size, resample) with no cropping step -- it's a
# direct stretch to (height, width), not an aspect-preserving
# resize+crop. The old `resize_keep_aspect_center_crop` cropped out part
# of the image and attacked a different region than what the real SigLIP
# tower ever sees at inference. Replaced with a plain stretch-resize to
# match the real pipeline.
# ----------------------------
def resize_square_stretch(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    return F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)


def normalize_like_processor(x01: torch.Tensor, image_processor) -> torch.Tensor:
    mean = torch.tensor(image_processor.image_mean, dtype=x01.dtype, device=x01.device).view(1, 3, 1, 1)
    std = torch.tensor(image_processor.image_std, dtype=x01.dtype, device=x01.device).view(1, 3, 1, 1)
    return (x01 - mean) / std


def gemma_preprocess_differentiable(x01: torch.Tensor, processor) -> torch.Tensor:
    ip = processor.image_processor
    th, tw = _get_target_hw(ip)
    x = resize_square_stretch(x01, th, tw)
    x = normalize_like_processor(x, ip)
    return x


# ----------------------------
# Build template inputs ONCE
# ----------------------------
def build_template_inputs(processor, question: str, pil_image: Image.Image, device):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    template = processor(text=[prompt], images=[pil_image], return_tensors="pt")
    template = {k: v.to(device) if torch.is_tensor(v) else v for k, v in template.items()}
    return template


# ----------------------------
# Generation helper
# ----------------------------
def run_generation_with_pixel_values(model, processor, template_inputs, pixel_values, max_new_tokens=128):
    model.eval()
    inputs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in template_inputs.items()}
    inputs["pixel_values"] = pixel_values

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    input_ids = inputs["input_ids"]
    gen_only = out_ids[:, input_ids.shape[1]:]
    return processor.batch_decode(gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]


# ============================================================
# Attack selected language + vision modules
# ============================================================
layer_inputs = {}


def make_pre_hook(name):
    def hook(module, inputs):
        layer_inputs[name] = inputs[0]
    return hook


def extract_language_layer_idx(name: str):
    m = re.search(r"language_model\.model\.layers\.(\d+)\.", name)
    if m is None:
        return None
    return int(m.group(1))


def extract_vision_layer_idx(name: str):
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


def is_language_target(name: str, chosen_lan_layers_set=None, whichMLP: str = "gate_proj") -> bool:
    if "language_model.model.layers." not in name:
        return False

    layer_idx = extract_language_layer_idx(name)
    if layer_idx is None:
        return False

    if chosen_lan_layers_set is not None and layer_idx not in chosen_lan_layers_set:
        return False

    allowed = {"gate_proj", "up_proj", "down_proj"}
    if whichMLP not in allowed:
        raise ValueError(f"Invalid whichMLP={whichMLP}. Choose from {sorted(allowed)}")

    return name.endswith(f".mlp.{whichMLP}")



def is_vision_target(name: str, chosen_vis_layers_set=None, whichMLPvis: str = "fc2") -> bool:
    if "vision" not in name:
        return False

    layer_idx = extract_vision_layer_idx(name)
    if layer_idx is None:
        return False

    if chosen_vis_layers_set is not None and layer_idx not in chosen_vis_layers_set:
        return False

    allowed = {"fc1", "fc2", "out_proj"}
    if whichMLPvis not in allowed:
        raise ValueError(f"Invalid whichMLPvis={whichMLPvis}. Choose from {sorted(allowed)}")

    if whichMLPvis == "out_proj":
        return name.endswith(".self_attn.out_proj")

    return name.endswith(f".mlp.{whichMLPvis}")


def collect_target_modules(
    model,
    chosen_lan_layers=None,
    chosen_vis_layers=None,
    whichMLP: str = "gate_proj",
    whichMLPvis: str = "fc2",
):
    chosen_lan_layers_set = None if chosen_lan_layers is None else set(chosen_lan_layers)
    chosen_vis_layers_set = None if chosen_vis_layers is None else set(chosen_vis_layers)

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
            chosen_lan_layers_set=chosen_lan_layers_set,
            whichMLP=whichMLP,
        ):
            targets.append({
                "name": name,
                "module": module,
                "kind": "language",
                "layer_idx": extract_language_layer_idx(name),
            })

        elif is_vision_target(
            name,
            chosen_vis_layers_set=chosen_vis_layers_set,
            whichMLPvis=whichMLPvis,
        ):
            targets.append({
                "name": name,
                "module": module,
                "kind": "vision",
                "layer_idx": extract_vision_layer_idx(name),
            })

    return targets


def compute_bottom_singular_subspace(weight: torch.Tensor, towardsNull: float):
    with torch.no_grad():
        W = weight.detach().to(torch.float32)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

        absorbSize = int((S < 1).sum().item())

        if towardsNull == 0:
            bottomInd = 1
        else:
            bottomInd = int(absorbSize * towardsNull)

        bottomInd = max(1, bottomInd)
        bottomInd = min(bottomInd, Vh.shape[0])

        V_bottom = Vh[-bottomInd:].contiguous()
        return V_bottom, S, absorbSize, bottomInd


def compute_top_singular_subspace(weight: torch.Tensor, towardsNull: float):
    with torch.no_grad():
        W = weight.detach().to(torch.float32)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

        absorbSize = int((S < 1).sum().item())

        if towardsNull == 0:
            bottomInd = 1
        else:
            bottomInd = int(absorbSize * towardsNull)

        bottomInd = max(1, bottomInd)
        bottomInd = min(bottomInd, Vh.shape[0])

        V_bottom = Vh[:bottomInd].contiguous()
        return V_bottom, S, absorbSize, bottomInd


def getMeanAlignmentLossWithBottomSubspace(InputToLayer, bottomRightSingularVectors):
    # Already correctly upcast to float32 in the original script -- left as-is.
    H = InputToLayer.float()

    if isinstance(H, (tuple, list)):
        H = H[0]

    if H.dim() == 3:
        H = H.reshape(-1, H.shape[-1])
    elif H.dim() == 2:
        pass
    else:
        H = H.view(-1, H.shape[-1])

    V = bottomRightSingularVectors.to(device=H.device, dtype=torch.float32)

    H_hat = F.normalize(H, dim=1)
    V_hat = F.normalize(V, dim=1)

    coeffs = H_hat @ V_hat.T
    per_token_energy = (coeffs ** 2).sum(dim=1)

    loss = -per_token_energy.mean()
    return loss


def register_all_hooks(targets):
    handles = []

    for spec in targets:
        h = spec["module"].register_forward_pre_hook(make_pre_hook(spec["name"]))
        handles.append(h)

    return handles


def build_target_specs_with_subspaces(
    model,
    towardsNull: float,
    chosen_lan_layers=None,
    chosen_vis_layers=None,
    whichMLP: str = "gate_proj",
    whichMLPvis: str = "fc2",
):
    targets = collect_target_modules(
        model,
        chosen_lan_layers=chosen_lan_layers,
        chosen_vis_layers=chosen_vis_layers,
        whichMLP=whichMLP,
        whichMLPvis=whichMLPvis,
    )

    specs = []

    print("\n========== TARGET MODULES ==========")
    print(f"Language target MLP: {whichMLP}")
    print(f"Vision target module: {whichMLPvis}")

    for t in targets:
        name = t["name"]
        module = t["module"]
        kind = t["kind"]
        layer_idx = t["layer_idx"]

        V_bottom, S, absorbSize, bottomInd = compute_bottom_singular_subspace(
            module.weight,
            towardsNull,
        )

        spec = {
            "name": name,
            "module": module,
            "kind": kind,
            "layer_idx": layer_idx,
            "bottom_vectors": V_bottom,
            "absorbSize": absorbSize,
            "bottomInd": bottomInd,
            "weight_shape": tuple(module.weight.shape),
        }

        specs.append(spec)

        print(
            f"{kind:8s} | layer={layer_idx:3d} | {name} | "
            f"weight_shape={tuple(module.weight.shape)} | "
            f"absorbSize={absorbSize} | bottomInd={bottomInd}"
        )

    num_lang = sum(1 for x in specs if x["kind"] == "language")
    num_vis = sum(1 for x in specs if x["kind"] == "vision")

    print("------------------------------------")
    print(f"Total language targets: {num_lang}")
    print(f"Total vision targets:   {num_vis}")
    print(f"Total targets overall:  {len(specs)}")
    print("====================================\n")

    return specs


def aggregated_bottom_subspace_loss(target_specs, device):
    lang_losses = []
    vis_losses = []

    for spec in target_specs:
        name = spec["name"]

        if name not in layer_inputs:
            continue

        H = layer_inputs[name]
        loss_i = getMeanAlignmentLossWithBottomSubspace(H, spec["bottom_vectors"])

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


# ----------------------------
# FIX #1 continued + FIX #3:
#
# These two functions are the actual BSA term used in the loop
# (`BSALoss = get_bsa_loss(...) + get_bsaVis_loss(...)`). Two problems:
#
#  (a) precision -- F.cosine_similarity was being called directly on bf16
#      hidden states with no upcast, same issue as cos()/cosVis() in the
#      other scripts.
#
#  (b) scale mismatch -- the original summed cos_per_token.sum() across
#      every token AND every layer with no normalization. For Gemma3-4B
#      that's on the order of (hundreds-to-thousands of tokens) x
#      (dozens of layers), so BSALoss lands somewhere in the thousands+
#      early in optimization. AlignmentLoss, by contrast, is a bounded
#      mean (`per_token_energy.mean()`, negated), so it sits at roughly
#      O(1). Combining them as
#      `loss = AlignmentLoss*(1-balancingAlpha) + BSALoss*balancingAlpha`
#      means BSALoss numerically swamps AlignmentLoss almost regardless
#      of balancingAlpha -- the spectral-subspace term (the actual novel
#      part of this attack, controlled by --towardsNull) ends up
#      contributing a negligible fraction of the gradient. Changed both
#      functions to average per-token-per-layer (mean instead of sum) so
#      the result is O(1) like AlignmentLoss, and balancingAlpha actually
#      behaves like an interpolation weight between the two objectives.
#      If you specifically want the old unnormalized-sum behavior back,
#      swap `.mean()` back to `.sum()` in the two loops below.
# ----------------------------
def get_bsa_loss(outputs, outputsN):
    per_layer = []
    for h, hn in zip(outputs.hidden_states, outputsN.hidden_states):
        h = h.float()
        hn = hn.float()
        cos_per_token = F.cosine_similarity(h.squeeze(0), hn.squeeze(0), dim=1)
        per_layer.append(cos_per_token.mean())
    return torch.stack(per_layer).mean()


def get_bsaVis_loss(outputs, outputsN):
    per_layer = []
    for h, hn in zip(outputs.hidden_states, outputsN.hidden_states):
        h = h.float()
        hn = hn.float()
        cos_per_token = F.cosine_similarity(h.squeeze(0), hn.squeeze(0), dim=1)
        per_layer.append(cos_per_token.mean())
    return torch.stack(per_layer).mean()


# ----------------------------
# ORIGINAL-SPACE Adam attack
# ----------------------------
def adam_attack_original_space(
    model,
    processor,
    template_inputs,
    x_orig01,
    attck_type: str,
    num_steps: int,
    lr: float,
    epsilon: float,
    device,
    save_conv_path: str,
    AttackStartLayer: int,
    numLayerstAtAtime: int,
    towardsNull: float,
    balancingAlpha: float,
    whichMLP: str,
    whichMLPvis: str,
    chosenLanLayers=None,
    chosenVisLayers=None,
):
    x_orig01 = x_orig01.detach().to(device)

    delta = 0.001 * torch.randn_like(x_orig01, device=device)
    delta.requires_grad_(True)

    opt = torch.optim.Adam([delta], lr=lr)

    losses_list = []
    best_loss = 1e20
    best_delta = delta.detach().clone()

    model.eval()

    with torch.no_grad():
        pv_clean_fixed = gemma_preprocess_differentiable(x_orig01, processor)

        clean_inputs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in template_inputs.items()}
        clean_inputs["pixel_values"] = pv_clean_fixed
        clean_inputs["labels"] = template_inputs["input_ids"]
        clean_inputs["use_cache"] = False
        outputsN = model(**clean_inputs, output_hidden_states=True, return_dict=True)
        hiddStateLen = len(outputsN.hidden_states)
        print(" Number of hidden states is: ", hiddStateLen)

        vision_model = model.vision_tower.vision_model
        vision_outN = vision_model(pixel_values=clean_inputs["pixel_values"], output_hidden_states=True, return_dict=True,)

        hiddStateLenVis = len(vision_outN.hidden_states)
        print(" Number of vision hidden states is: ", hiddStateLenVis)

    target_specs = build_target_specs_with_subspaces(
        model,
        towardsNull=towardsNull,
        chosen_lan_layers=chosenLanLayers,
        chosen_vis_layers=chosenVisLayers,
        whichMLP=whichMLP,
        whichMLPvis=whichMLPvis,
    )

    if len(target_specs) == 0:
        raise RuntimeError(
            "No target modules found. Check --chosenLanLayers, --chosenVisLayers, "
            "--whichMLP, and --whichMLPvis."
        )

    hook_handles = register_all_hooks(target_specs)

    adv_inputs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in template_inputs.items()}
    adv_inputs["labels"] = template_inputs["input_ids"]
    adv_inputs["use_cache"] = False

    try:
        for step in range(num_steps):
            layer_inputs.clear()

            x_adv01 = (x_orig01 + delta).clamp(0.0, 1.0)
            x_adv01 = torch.max(
                torch.min(x_adv01, x_orig01 + epsilon),
                x_orig01 - epsilon,
            ).clamp(0.0, 1.0)

            pv_adv = gemma_preprocess_differentiable(x_adv01, processor)
            adv_inputs["pixel_values"] = pv_adv

            outputs = model(**adv_inputs, output_hidden_states=True, return_dict=True)
            vision_out = vision_model(pixel_values=adv_inputs["pixel_values"], output_hidden_states=True, return_dict=True)

            BSALoss = get_bsa_loss(outputs, outputsN) + get_bsaVis_loss(vision_out, vision_outN)

            language_loss, vision_loss, total_used = aggregated_bottom_subspace_loss(
                target_specs,
                device=device,
            )

            if total_used == 0:
                raise RuntimeError("No hooked target modules were used in the forward pass.")

            AlignmentLoss = language_loss + vision_loss
            loss = AlignmentLoss * (1 - balancingAlpha) + BSALoss * balancingAlpha

            attack_loss = loss
            opt.zero_grad(set_to_none=True)
            attack_loss.backward()
            opt.step()

            with torch.no_grad():
                delta.data.clamp_(-epsilon, epsilon)

            lv = float(loss.item())
            losses_list.append(lv)

            if (step + 1) % 10 == 0 or step == 0:
                print(
                    f"[step {step + 1}/{num_steps}] "
                    f"total_loss={lv:.6f} "
                    f"language_loss={float(language_loss.item()):.6f} "
                    f"vision_loss={float(vision_loss.item()):.6f} "
                    f"used_modules={total_used}"
                )

            if lv < best_loss:
                best_loss = lv
                best_delta = delta.detach().clone()
                np.save(save_conv_path, np.array(losses_list, dtype=np.float32))

            del outputs, loss, attack_loss, pv_adv

    finally:
        for h in hook_handles:
            h.remove()

    with torch.no_grad():
        x_adv01_final = (x_orig01 + best_delta).clamp(0.0, 1.0)
        x_adv01_final = torch.max(
            torch.min(x_adv01_final, x_orig01 + epsilon),
            x_orig01 - epsilon,
        ).clamp(0.0, 1.0)

    return x_adv01_final, best_delta


# ----------------------------
# MAIN
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Gemma-3 ORIGINAL-image-space adversarial attack with flexible language and vision module selection"
    )

    parser.add_argument(
        "--attck_type",
        type=str,
        default="grill_l2",
        help="kept for compatibility",
    )

    parser.add_argument(
        "--desired_norm_l_inf",
        type=float,
        default=0.03,
        help="epsilon L_inf in ORIGINAL pixel space [0..1]",
    )

    parser.add_argument(
        "--learningRate",
        type=float,
        default=1e-3,
        help="Adam learning rate",
    )

    parser.add_argument(
        "--balancingAlpha",
        type=float,
        default=1.0,
        help="Adam learning rate",
    )

    parser.add_argument(
        "--num_steps",
        type=int,
        default=2000,
        help="Number of Adam steps",
    )

    parser.add_argument(
        "--attackSample",
        type=str,
        default="nature",
        help="which sample",
    )

    parser.add_argument(
        "--AttackStartLayer",
        type=int,
        default=0,
        help="kept for compatibility; not used in selected-layer attack",
    )

    parser.add_argument(
        "--numLayerstAtAtime",
        type=int,
        default=2,
        help="kept for compatibility; not used in selected-layer attack",
    )

    parser.add_argument(
        "--towardsNull",
        type=float,
        default=0.1,
        help="same bottom-k selection logic as before",
    )

    parser.add_argument(
        "--whichMLP",
        type=str,
        default="gate_proj",
        choices=["gate_proj", "up_proj", "down_proj"],
        help="Which language MLP projection to attack",
    )

    parser.add_argument(
        "--whichMLPvis",
        type=str,
        default="fc2",
        choices=["fc1", "fc2", "out_proj"],
        help="Which vision module to attack: fc1, fc2, or out_proj",
    )

    parser.add_argument(
        "--chosenLanLayers",
        type=int,
        nargs="+",
        default=None,
        help="Space-separated language layer indices to attack, e.g. --chosenLanLayers 0 1 2 3 4",
    )

    parser.add_argument(
        "--chosenVisLayers",
        type=int,
        nargs="+",
        default=None,
        help="Space-separated vision layer indices to attack, e.g. --chosenVisLayers 15 16 17 18 19",
    )

    parser.add_argument(
        "--modelDtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help=(
            "Forward-pass dtype for Gemma3. bfloat16 matches the original "
            "script; float16 gives finer precision near cos()=1 but may "
            "risk overflow on some layers; float32 is safest/most "
            "precise but ~2x memory and slower. The loss-side float32 "
            "upcast fix applies regardless of this setting."
        ),
    )

    args = parser.parse_args()

    print("torch.cuda.is_available()", torch.cuda.is_available())

    attck_type = args.attck_type
    epsilon = float(args.desired_norm_l_inf)
    lr = float(args.learningRate)
    num_steps = int(args.num_steps)
    attackSample = str(args.attackSample)
    AttackStartLayer = int(args.AttackStartLayer)
    numLayerstAtAtime = int(args.numLayerstAtAtime)
    towardsNull = float(args.towardsNull)
    balancingAlpha = float(args.balancingAlpha)
    whichMLP = str(args.whichMLP)
    whichMLPvis = str(args.whichMLPvis)

    chosenLanLayers = args.chosenLanLayers
    chosenVisLayers = args.chosenVisLayers

    MODEL_PATH = "../illcond/gemma_attack/Gemma3-4b"
    IMAGE_PATH = f"gemma_attack/dataSamplesForQuant/{attackSample}.JPEG"
    QUESTION = "What is shown in this image?"
    MAX_NEW_TOKENS = 128

    os.makedirs("gemma_attack/outputsStorageImagenet", exist_ok=True)
    os.makedirs(f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}", exist_ok=True)
    os.makedirs(f"gemma_attack/outputsStorageImagenet/convergence/{attackSample}", exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.modelDtype] if device.type == "cuda" else torch.float32

    print(f"device={device}, dtype={dtype}")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, padding_side="left")

    print("Loading model...")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
    ).to(device)

    model.eval()
    model.config.use_cache = False

    print("\n[INFO] This version attacks selected language and vision layers jointly.")
    print("[INFO] CLI args --AttackStartLayer and --numLayerstAtAtime are kept only for compatibility.")
    print(f"[INFO] chosenLanLayers={chosenLanLayers}")
    print(f"[INFO] chosenVisLayers={chosenVisLayers}")
    print(f"[INFO] whichMLP={whichMLP}")
    print(f"[INFO] whichMLPvis={whichMLPvis}")
    print(f"[INFO] towardsNull={towardsNull}\n")

    pil = Image.open(IMAGE_PATH).convert("RGB")
    x_orig01 = pil_to_tensor01(pil).to(device)

    template_inputs = build_template_inputs(processor, QUESTION, pil, device)

    pv_clean = gemma_preprocess_differentiable(x_orig01, processor)

    print("\n=== CLEAN OUTPUT ===")
    clean_text = run_generation_with_pixel_values(
        model,
        processor,
        template_inputs,
        pv_clean,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    print(clean_text)

    if device.type == "cuda":
        torch.cuda.empty_cache()


    if len(chosenVisLayers) > 20:
        conv_path = (
            f"gemma_attack/outputsStorageImagenet/convergence/{attackSample}/"
            f"gemma_ORIG_attack_{attck_type}_lr_{lr}_eps_{epsilon}_"
            f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
            f"num_steps_{num_steps}_towardsNull_{towardsNull}_"
            f"lanMLP_{whichMLP}_visMLP_{whichMLPvis}_"
            f"lanLayers_upto4_visLayers_all_balancingAlpha_{balancingAlpha}.npy"
        )
    else:
        conv_path = (
            f"gemma_attack/outputsStorageImagenet/convergence/{attackSample}/"
            f"gemma_ORIG_attack_{attck_type}_lr_{lr}_eps_{epsilon}_"
            f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
            f"num_steps_{num_steps}_towardsNull_{towardsNull}_"
            f"lanMLP_{whichMLP}_visMLP_{whichMLPvis}_"
            f"lanLayers_{chosenLanLayers}_visLayers_{chosenVisLayers}_balancingAlpha_{balancingAlpha}.npy"
        )



    x_adv01, best_pert = adam_attack_original_space(
        model=model,
        processor=processor,
        template_inputs=template_inputs,
        x_orig01=x_orig01,
        attck_type=attck_type,
        num_steps=num_steps,
        lr=lr,
        epsilon=epsilon,
        device=device,
        save_conv_path=conv_path,
        AttackStartLayer=AttackStartLayer,
        numLayerstAtAtime=numLayerstAtAtime,
        towardsNull=towardsNull,
        balancingAlpha = balancingAlpha,
        whichMLP=whichMLP,
        whichMLPvis=whichMLPvis,
        chosenLanLayers=chosenLanLayers,
        chosenVisLayers=chosenVisLayers,
    )

    if len(chosenVisLayers) > 20:

        adv_img_path = (
            f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/"
            f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
            f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
            f"num_steps_{num_steps}_towardsNull_{towardsNull}_"
            f"lanMLP_{whichMLP}_visMLP_{whichMLPvis}_"
            f"lanLayers_upto4_visLayers_all_balancingAlpha_{balancingAlpha}.png"
        )

        adv_noise_path = (
            f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/"
            f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
            f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
            f"num_steps_{num_steps}_towardsNull_{towardsNull}_"
            f"lanMLP_{whichMLP}_visMLP_{whichMLPvis}_"
            f"lanLayers_upto4_visLayers_all_balancingAlpha_{balancingAlpha}.pt"
        )

    else:
        adv_img_path = (
            f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/"
            f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
            f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
            f"num_steps_{num_steps}_towardsNull_{towardsNull}_"
            f"lanMLP_{whichMLP}_visMLP_{whichMLPvis}_"
            f"lanLayers_{chosenLanLayers}_visLayers_{chosenVisLayers}_balancingAlpha_{balancingAlpha}.png"
        )

        adv_noise_path = (
            f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/"
            f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
            f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
            f"num_steps_{num_steps}_towardsNull_{towardsNull}_"
            f"lanMLP_{whichMLP}_visMLP_{whichMLPvis}_"
            f"lanLayers_{chosenLanLayers}_visLayers_{chosenVisLayers}_balancingAlpha_{balancingAlpha}.pt"
        )


    tensor01_to_pil(x_adv01).save(adv_img_path) # this should be uncommented later
    print(f"\nSaved ORIGINAL-resolution adversarial image to: {adv_img_path}")

    torch.save(best_pert.detach().cpu(), adv_noise_path) # this should be uncommented later
    print(f"Saved perturbation tensor to: {adv_noise_path}")

    pv_adv = gemma_preprocess_differentiable(x_adv01, processor)

    print("\n=== ADVERSARIAL OUTPUT ===")
    adv_text = run_generation_with_pixel_values(
        model,
        processor,
        template_inputs,
        pv_adv,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    print(adv_text)

    cleanOutTxt = f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/cleanOutput.txt"
    with open(cleanOutTxt, "w") as f:
        f.write(clean_text + "\n\n")

    if len(chosenVisLayers) > 20:
        advOutTxt = (
            f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/"
            f"advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
            f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
            f"num_steps_{num_steps}_towardsNull_{towardsNull}_"
            f"lanMLP_{whichMLP}_visMLP_{whichMLPvis}_"
            f"lanLayers_upto4_visLayers_all_balancingAlpha_{balancingAlpha}.txt"
        )
    else:
        advOutTxt = (
            f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/"
            f"advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
            f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
            f"num_steps_{num_steps}_towardsNull_{towardsNull}_"
            f"lanMLP_{whichMLP}_visMLP_{whichMLPvis}_"
            f"lanLayers_{chosenLanLayers}_visLayers_{chosenVisLayers}_balancingAlpha_{balancingAlpha}.txt"
        )



    with open(advOutTxt, "w") as f:
        f.write(adv_text + "\n")

    print(f"\nSaved clean output to: {cleanOutTxt}")
    print(f"Saved adversarial output to: {advOutTxt}")
    print(f"Saved convergence to: {conv_path}")


if __name__ == "__main__":
    main()