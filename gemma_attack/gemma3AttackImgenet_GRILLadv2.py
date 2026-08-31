



'''


##########################################################################################################################################################################################################################################################################################

---------------------------------------------------------------------------------------------------------------------------------------

export CUDA_VISIBLE_DEVICES=7
conda activate gemma3
cd interpretAttacks
for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_GRILLadv2.py --attck_type grill_adv2 --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_GRILLadv2.py --attck_type grill_adv2 --desired_norm_l_inf 0.0035 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_GRILLadv2.py --attck_type grill_adv2 --desired_norm_l_inf 0.0045 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done


export CUDA_VISIBLE_DEVICES=1
conda activate gemma3
cd interpretAttacks
for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_GRILLadv2.py --attck_type grill_adv2 --desired_norm_l_inf 0.003 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done

export CUDA_VISIBLE_DEVICES=2
conda activate gemma3
cd interpretAttacks
for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_GRILLadv2.py --attck_type grill_adv2 --desired_norm_l_inf 0.008 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done

export CUDA_VISIBLE_DEVICES=3
conda activate gemma3
cd interpretAttacks
for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_GRILLadv2.py --attck_type grill_adv2 --desired_norm_l_inf 0.009 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done



export CUDA_VISIBLE_DEVICES=1
conda activate gemma3
cd interpretAttacks
for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_GRILLadv2.py --attck_type grill_adv2 --desired_norm_l_inf 0.007 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done


export CUDA_VISIBLE_DEVICES=0
conda activate gemma3
cd interpretAttacks
for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_GRILLadv2.py --attck_type grill_adv2 --desired_norm_l_inf 0.008 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done


export CUDA_VISIBLE_DEVICES=0
conda activate gemma3
cd interpretAttacks
for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_GRILLadv2.py --attck_type grill_adv2 --desired_norm_l_inf 0.009 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done

for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_GRILLadv2.py --attck_type grill_adv2 --desired_norm_l_inf 0.01 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done



'''

'''
FIXED VERSION — see notes below for what changed and why.

Same launch pattern as before, e.g.:

export CUDA_VISIBLE_DEVICES=7
conda activate gemma3
cd interpretAttacks
for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_GRILLadv2.py --attck_type grill_adv2 --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done


export CUDA_VISIBLE_DEVICES=6
conda activate gemma3
cd interpretAttacks
for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_GRILLadv2.py --attck_type grill_adv2 --desired_norm_l_inf 0.002 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done

for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_GRILLadv2.py --attck_type grill_adv2 --desired_norm_l_inf 0.0025 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done

for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_GRILLadv2.py --attck_type grill_adv2 --desired_norm_l_inf 0.003 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done

for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_GRILLadv2.py --attck_type grill_adv2 --desired_norm_l_inf 0.0035 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done

for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_GRILLadv2.py --attck_type grill_adv2 --desired_norm_l_inf 0.0045 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done

'''

import os
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
# FIX #1 (the big one, same root cause as the Qwen script):
#
# Gemma3 is loaded in bfloat16. bf16 only has ~7-8 mantissa bits, so it
# resolves values near 1.0 to a spacing of about 2^-8 (~0.0039). At the
# epsilon values you're using (0.003-0.0045), clean vs. adversarial hidden
# states start out extremely close, so cos(h_adv, h_clean) sits well
# inside that resolution floor and rounds to *exactly* 1.0 in the forward
# pass. That makes (1 - cos)^2 exactly 0, and the gradient through that
# term is exactly 0 too -- not "small", dead. LLaVA doesn't hit this
# because it runs in float16, which has ~8x finer resolution near 1.0.
#
# The fix is to upcast to float32 before the normalize/dot-product/
# subtract-from-1 arithmetic in cos()/cosVis(). This doesn't recover
# precision already baked into the stored bf16 activations, but it stops
# the reduction itself from rounding a real (if small) difference down to
# a hard zero.
# ----------------------------
def cos(a, b):
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return (a * b).sum()


def cosVis(a, b):
    a = torch.flatten(a).float()
    b = torch.flatten(b).float()
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
# Losses: GRILL + OA (legacy variants kept for parity with the original
# script / attck_type switch; all now route through the fixed cos/cosVis)
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


def get_grill_cos_lossNew(outputs_adv, outputs_clean, acts_adv, acts_clean):
    loss = 0.0
    losses = []
    for h_adv, h_clean in zip(
        outputs_adv.hidden_states,
        outputs_clean.hidden_states
    ):
        loss = loss + (1.0 - cos(h_adv, h_clean)) ** 2
        losses.append(loss)

    # Vision hidden-state distortions
    for v_adv, v_clean in zip(acts_adv.hidden_states, acts_clean.hidden_states):
        loss = loss + (1.0 - cosVis(v_adv, v_clean)) ** 2
        losses.append(loss)

    losses_tensor = torch.stack(losses)
    agg = (losses_tensor.sum() ** 2 - (losses_tensor ** 2).sum()) / 2
    return agg


# ----------------------------
# Utilities: image <-> tensor
# ----------------------------
def pil_to_tensor01(pil_img: Image.Image) -> torch.Tensor:
    """PIL RGB -> torch float tensor in [0,1], shape (1,3,H,W)"""
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    return t


def tensor01_to_pil(t01: torch.Tensor) -> Image.Image:
    """torch tensor [0,1], shape (1,3,H,W) or (3,H,W) -> PIL RGB"""
    if t01.dim() == 4:
        t01 = t01[0]
    t01 = t01.detach().cpu().clamp(0, 1)
    arr = (t01.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ----------------------------
# Differentiable preprocessing (matches Gemma3ImageProcessor)
# ----------------------------
def _get_target_hw(image_processor):
    """
    Infer model target H,W from HF image_processor.
    """
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
# FIX #2: Gemma3ImageProcessor (transformers/models/gemma3/
# image_processing_gemma3.py) has default_to_square=True and calls
# self.resize(image=..., size=size, resample=...) with no cropping step
# anywhere in _preprocess (pan-and-scan aside, which is off by default and
# not used by this script's generate() calls). That resize is a direct
# stretch to (height, width) -- it does NOT preserve aspect ratio and does
# NOT center-crop. The original script's
# `resize_keep_aspect_center_crop` does the opposite (aspect-preserving
# resize + crop), which silently discards part of the image and attacks
# a different crop region than what the real SigLIP tower ever sees at
# inference time. Fixed to a plain stretch-resize to match the real
# pipeline.
# ----------------------------
def resize_square_stretch(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    return F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)


def normalize_like_processor(x01: torch.Tensor, image_processor) -> torch.Tensor:
    mean = torch.tensor(image_processor.image_mean, dtype=x01.dtype, device=x01.device).view(1, 3, 1, 1)
    std = torch.tensor(image_processor.image_std, dtype=x01.dtype, device=x01.device).view(1, 3, 1, 1)
    return (x01 - mean) / std


def gemma_preprocess_differentiable(x01: torch.Tensor, processor) -> torch.Tensor:
    """
    Differentiable equivalent of the processor's image pipeline.
    Produces pixel_values like the processor would (shape 1x3xH'xW').
    """
    ip = processor.image_processor
    th, tw = _get_target_hw(ip)
    x = resize_square_stretch(x01, th, tw)
    x = normalize_like_processor(x, ip)
    return x


# ----------------------------
# Build template inputs ONCE (IMPORTANT)
# Ensures image placeholder tokens exist in input_ids
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
# Generation helper (uses template, swaps pixel_values)
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
    save_conv_path: str
):
    x_orig01 = x_orig01.detach().to(device)

    delta = 0.001 * torch.randn_like(x_orig01, device=device)
    delta.requires_grad_(True)

    opt = torch.optim.Adam([delta], lr=lr)

    losses_list = [0.0]
    best_loss = 1e18
    best_delta = delta.detach().clone()

    model.eval()

    # Precompute clean pixel_values once (no grads needed)
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
        vision_outN = vision_model(pixel_values=clean_inputs["pixel_values"], output_hidden_states=True, return_dict=True)

        hiddStateLenVis = len(vision_outN.hidden_states)
        print(" Number of vision hidden states is: ", hiddStateLenVis)

    adv_inputs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in template_inputs.items()}
    adv_inputs["labels"] = template_inputs["input_ids"]
    adv_inputs["use_cache"] = False

    for step in range(num_steps):
        x_adv01 = (x_orig01 + delta).clamp(0.0, 1.0)
        x_adv01 = torch.max(torch.min(x_adv01, x_orig01 + epsilon), x_orig01 - epsilon).clamp(0.0, 1.0)

        pv_adv = gemma_preprocess_differentiable(x_adv01, processor)
        adv_inputs["pixel_values"] = pv_adv

        outputs = model(**adv_inputs, output_hidden_states=True, return_dict=True)
        vision_out = vision_model(pixel_values=adv_inputs["pixel_values"], output_hidden_states=True, return_dict=True)

        loss = -1 * get_grill_cos_lossNew(outputs, outputsN, vision_out, vision_outN)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        with torch.no_grad():
            delta.data.clamp_(-epsilon, epsilon)

        lv = float(loss.item())
        if (step + 1) % 10 == 0 or step == 0:
            print(f"[step {step+1}/{num_steps}] loss={lv:.6f}")

        if lv < best_loss:
            best_loss = lv
            best_delta = delta.detach().clone()
            losses_list.append(lv)
            np.save(save_conv_path, np.array(losses_list, dtype=np.float32))

        del outputs, vision_out, loss, pv_adv

    with torch.no_grad():
        x_adv01_final = (x_orig01 + best_delta).clamp(0.0, 1.0)
        x_adv01_final = torch.max(torch.min(x_adv01_final, x_orig01 + epsilon), x_orig01 - epsilon).clamp(0.0, 1.0)

    return x_adv01_final, best_delta


# ----------------------------
# MAIN
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Gemma-3 ORIGINAL-image-space adversarial attack (precision-fixed)")
    parser.add_argument("--attck_type", type=str, default="grill_l2",
                        help="grill_l2 | grill_cos | OA_l2 | OA_cos")
    parser.add_argument("--desired_norm_l_inf", type=float, default=0.03)
    parser.add_argument("--learningRate", type=float, default=1e-3)
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--attackSample", type=str, default="nature")
    parser.add_argument(
        "--modelDtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help=(
            "Forward-pass dtype for Gemma3. bfloat16 matches the original "
            "script; float16 gives finer precision near cos()=1 (closer "
            "to LLaVA's fp16 setup) but may risk overflow on some layers; "
            "float32 is safest/most precise but ~2x memory and slower. "
            "The loss-side float32 upcast fix applies regardless."
        ),
    )

    args = parser.parse_args()

    attck_type = args.attck_type
    epsilon = float(args.desired_norm_l_inf)
    lr = float(args.learningRate)
    num_steps = int(args.num_steps)
    attackSample = str(args.attackSample)

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

    pil = Image.open(IMAGE_PATH).convert("RGB")
    x_orig01 = pil_to_tensor01(pil).to(device)

    template_inputs = build_template_inputs(processor, QUESTION, pil, device)

    pv_clean = gemma_preprocess_differentiable(x_orig01, processor)

    print("\n=== CLEAN OUTPUT ===")
    clean_text = run_generation_with_pixel_values(model, processor, template_inputs, pv_clean, max_new_tokens=MAX_NEW_TOKENS)
    print(clean_text)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    conv_path = f"gemma_attack/outputsStorageImagenet/convergence/{attackSample}/gemma_ORIG_attack_{attck_type}_lr_{lr}_eps_{epsilon}_num_steps_{num_steps}_.npy"
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
        save_conv_path=conv_path
    )

    adv_img_path = f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_num_steps_{num_steps}_.png"
    adv_noise_path = f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_num_steps_{num_steps}_.pt"

    tensor01_to_pil(x_adv01).save(adv_img_path)
    print(f"\nSaved ORIGINAL-resolution adversarial image to: {adv_img_path}")

    torch.save(best_pert.detach().cpu(), adv_noise_path)

    pv_adv = gemma_preprocess_differentiable(x_adv01, processor)
    print("\n=== ADVERSARIAL OUTPUT ===")
    adv_text = run_generation_with_pixel_values(model, processor, template_inputs, pv_adv, max_new_tokens=MAX_NEW_TOKENS)
    print(adv_text)

    cleanOutTxt = f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/cleanOutput.txt"
    with open(cleanOutTxt, "w") as f:
        f.write(clean_text + "\n\n")

    advOutTxt = f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_num_steps_{num_steps}_.txt"
    with open(advOutTxt, "w") as f:
        f.write(adv_text + "\n")

    print(f"\nSaved outputs to: {advOutTxt}")
    print(f"Saved convergence to: {conv_path}")


if __name__ == "__main__":
    main()