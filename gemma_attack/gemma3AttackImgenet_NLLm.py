



'''



export CUDA_VISIBLE_DEVICES=4
conda activate gemma3
cd interpretAttacks
for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_NLLm.py --attck_type nllm --desired_norm_l_inf 0.0025 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done

for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_NLLm.py --attck_type nllm --desired_norm_l_inf 0.0035 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done

for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_NLLm.py --attck_type nllm --desired_norm_l_inf 0.0045 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done

for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_NLLm.py --attck_type nllm --desired_norm_l_inf 0.0006 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done

for ATTACK_SAMPLE in $(seq 1 50); do
    python gemma_attack/gemma3AttackImgenet_NLLm.py --attck_type nllm --desired_norm_l_inf 0.0005 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done



export CUDA_VISIBLE_DEVICES=5
conda activate gemma3
cd interpretAttacks
python gemma_attack/gemma3AttackImgenet_NLLm.py --attck_type nllm --desired_norm_l_inf 0.0045 --learningRate 0.001 --num_steps 1000 --attackSample 1
python gemma_attack/gemma3AttackImgenet_NLLm.py --attck_type nllm --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --attackSample 1
python gemma_attack/gemma3AttackImgenet_NLLm.py --attck_type nllm --desired_norm_l_inf 0.0035 --learningRate 0.001 --num_steps 1000 --attackSample 1
python gemma_attack/gemma3AttackImgenet_NLLm.py --attck_type nllm --desired_norm_l_inf 0.003 --learningRate 0.001 --num_steps 1000 --attackSample 1
python gemma_attack/gemma3AttackImgenet_NLLm.py --attck_type nllm --desired_norm_l_inf 0.0025 --learningRate 0.001 --num_steps 1000 --attackSample 1
python gemma_attack/gemma3AttackImgenet_NLLm.py --attck_type nllm --desired_norm_l_inf 0.002 --learningRate 0.001 --num_steps 1000 --attackSample 1


'''




import os
import sys
import argparse
import random
import numpy as np

import torch
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


# ----------------------------
# Utilities: image <-> tensor
# ----------------------------
def pil_to_tensor01(pil_img: Image.Image) -> torch.Tensor:
    """PIL RGB -> torch float tensor in [0,1], shape (1,3,H,W)"""
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


def tensor01_to_pil(t01: torch.Tensor) -> Image.Image:
    """torch tensor [0,1], shape (1,3,H,W) or (3,H,W) -> PIL RGB"""
    if t01.dim() == 4:
        t01 = t01[0]
    t01 = t01.detach().cpu().clamp(0, 1)
    arr = (t01.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ----------------------------
# Differentiable preprocessing (approx Gemma)
# ----------------------------
def _get_target_hw(image_processor):
    """
    Try to infer model target H,W from HF image_processor.
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


def resize_keep_aspect_center_crop(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """
    Differentiable resize + center crop.
    """
    _, _, H, W = x.shape
    scale = max(target_h / H, target_w / W)
    newH = int(round(H * scale))
    newW = int(round(W * scale))

    x_resized = F.interpolate(x, size=(newH, newW), mode="bilinear", align_corners=False)

    top = max((newH - target_h) // 2, 0)
    left = max((newW - target_w) // 2, 0)
    x_crop = x_resized[:, :, top:top + target_h, left:left + target_w]

    pad_h = target_h - x_crop.shape[2]
    pad_w = target_w - x_crop.shape[3]
    if pad_h > 0 or pad_w > 0:
        x_crop = F.pad(x_crop, (0, max(pad_w, 0), 0, max(pad_h, 0)))

    return x_crop


def normalize_like_processor(x01: torch.Tensor, image_processor) -> torch.Tensor:
    mean = torch.tensor(
        image_processor.image_mean, dtype=x01.dtype, device=x01.device
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        image_processor.image_std, dtype=x01.dtype, device=x01.device
    ).view(1, 3, 1, 1)
    return (x01 - mean) / std


def gemma_preprocess_differentiable(x01: torch.Tensor, processor) -> torch.Tensor:
    """
    Differentiable approximation of the processor's image pipeline.
    Produces pixel_values like the processor would.
    """
    ip = processor.image_processor
    th, tw = _get_target_hw(ip)
    x = resize_keep_aspect_center_crop(x01, th, tw)
    x = normalize_like_processor(x, ip)
    return x


# ----------------------------
# Prompt/template helpers
# ----------------------------
def build_user_prompt(processor, question: str):
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
    return prompt


def build_template_inputs(processor, question: str, pil_image: Image.Image, device):
    """
    Prompt-only inputs used for normal generation.
    """
    prompt = build_user_prompt(processor, question)
    template = processor(text=[prompt], images=[pil_image], return_tensors="pt")
    template = {k: v.to(device) if torch.is_tensor(v) else v for k, v in template.items()}
    return template


def build_attack_teacher_forced_inputs(
    processor,
    question: str,
    answer_text: str,
    pil_image: Image.Image,
    device,
):
    """
    Build full teacher-forced inputs:
      [user(image + question)] [assistant(answer_text)]

    Labels are masked so loss is computed ONLY on assistant answer tokens.
    This fixes the main issue in the original script, which computed loss on prompt tokens.
    """
    user_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    full_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer_text}],
        },
    ]

    prompt_only = processor.apply_chat_template(user_messages, add_generation_prompt=True)
    prompt_plus_answer = processor.apply_chat_template(full_messages, add_generation_prompt=False)

    prompt_inputs = processor(text=[prompt_only], images=[pil_image], return_tensors="pt")
    full_inputs = processor(text=[prompt_plus_answer], images=[pil_image], return_tensors="pt")

    prompt_len = prompt_inputs["input_ids"].shape[1]

    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100

    out = {}
    for k, v in full_inputs.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    out["labels"] = labels.to(device)
    return out, prompt_len


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
    return processor.batch_decode(
        gen_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0]


# ----------------------------
# Attack
# ----------------------------
def adam_attack_original_space(
    model,
    processor,
    attack_inputs_base,
    x_orig01,
    attck_type: str,
    num_steps: int,
    lr: float,
    epsilon: float,
    device,
    save_conv_path: str
):
    """
    Optimize delta in ORIGINAL image space:
        x_adv01 = clamp(x_orig01 + delta, 0, 1)
        ||delta||_inf <= epsilon

    Paper-consistent no-label surrogate in this script:
    - generate clean text once externally
    - use that clean text as pseudo-target
    - maximize NLL ONLY on answer tokens (not prompt tokens)

    Note:
    This is still not the exact dual-encoder image-text CE from the paper,
    because this Gemma script does not expose a paired contrastive text encoder.
    """
    if attck_type.lower() != "nllm":
        raise ValueError(f"Only --attck_type nllm is supported in this corrected script, got: {attck_type}")

    x_orig01 = x_orig01.detach().to(device=device, dtype=torch.float32)

    delta = 0.001 * torch.randn_like(x_orig01, device=device)
    delta.requires_grad_(True)

    opt = torch.optim.Adam([delta], lr=lr)

    losses_list = [0.0]
    best_nll = -1e18
    best_delta = delta.detach().clone()

    model.eval()

    adv_inputs = {
        k: (v.clone() if torch.is_tensor(v) else v)
        for k, v in attack_inputs_base.items()
    }
    adv_inputs["use_cache"] = False

    for step in range(num_steps):
        x_adv01 = (x_orig01 + delta).clamp(0.0, 1.0)
        x_adv01 = torch.max(torch.min(x_adv01, x_orig01 + epsilon), x_orig01 - epsilon).clamp(0.0, 1.0)

        pv_adv = gemma_preprocess_differentiable(x_adv01, processor)
        pv_adv = pv_adv.to(dtype=next(model.parameters()).dtype)

        adv_inputs["pixel_values"] = pv_adv

        outputs = model(**adv_inputs, return_dict=True)

        # outputs.loss is CE/NLL over NON-masked labels only, i.e. answer tokens only
        nll = outputs.loss.float()

        # maximize NLL with gradient descent optimizer
        objective = -nll

        opt.zero_grad(set_to_none=True)
        objective.backward()
        opt.step()

        with torch.no_grad():
            delta.data.clamp_(-epsilon, epsilon)

        nll_value = float(nll.item())
        if (step + 1) % 10 == 0 or step == 0:
            print(f"[step {step+1}/{num_steps}] answer_token_nll={nll_value:.6f}")

        # keep the perturbation that maximizes NLL
        if nll_value > best_nll:
            best_nll = nll_value
            best_delta = delta.detach().clone()
            losses_list.append(nll_value)
            np.save(save_conv_path, np.array(losses_list, dtype=np.float32))

        del outputs, nll, objective, pv_adv

    with torch.no_grad():
        x_adv01_final = (x_orig01 + best_delta).clamp(0.0, 1.0)
        x_adv01_final = torch.max(torch.min(x_adv01_final, x_orig01 + epsilon), x_orig01 - epsilon).clamp(0.0, 1.0)

    return x_adv01_final, best_delta


# ----------------------------
# MAIN
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Gemma-3 ORIGINAL-image-space adversarial attack (no squeeze)")
    parser.add_argument(
        "--attck_type",
        type=str,
        default="nll",
        help="nll"
    )
    parser.add_argument(
        "--desired_norm_l_inf",
        type=float,
        default=0.03,
        help="epsilon L_inf in ORIGINAL pixel space [0..1]. Try 0.01~0.08"
    )
    parser.add_argument(
        "--learningRate",
        type=float,
        default=1e-3,
        help="Adam learning rate"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=2000,
        help="Number of Adam steps"
    )
    parser.add_argument(
        "--attackSample",
        type=str,
        default="nature",
        help="which sample"
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
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
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
    x_orig01 = pil_to_tensor01(pil).to(device=device, dtype=torch.float32)

    # Prompt-only inputs for normal clean/adv generation
    template_inputs = build_template_inputs(processor, QUESTION, pil, device)

    pv_clean = gemma_preprocess_differentiable(x_orig01, processor).to(dtype=dtype)

    print("\n=== CLEAN OUTPUT ===")
    clean_text = run_generation_with_pixel_values(
        model,
        processor,
        template_inputs,
        pv_clean,
        max_new_tokens=MAX_NEW_TOKENS
    )
    clean_text = clean_text.strip()
    print(clean_text)

    if clean_text == "":
        clean_text = "unknown object"

    print("\n=== PSEUDO-TARGET USED FOR NLLm ATTACK ===")
    print(clean_text)

    # Build teacher-forced inputs whose labels are ONLY the clean answer tokens
    attack_inputs_base, prompt_len = build_attack_teacher_forced_inputs(
        processor=processor,
        question=QUESTION,
        answer_text=clean_text,
        pil_image=pil,
        device=device,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    conv_path = f"gemma_attack/outputsStorageImagenet/convergence/{attackSample}/gemma_ORIG_attack_{attck_type}_lr_{lr}_eps_{epsilon}_num_steps_{num_steps}_.npy"

    x_adv01, best_pert = adam_attack_original_space(
        model=model,
        processor=processor,
        attack_inputs_base=attack_inputs_base,
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

    pv_adv = gemma_preprocess_differentiable(x_adv01, processor).to(dtype=dtype)

    print("\n=== ADVERSARIAL OUTPUT ===")
    adv_text = run_generation_with_pixel_values(
        model,
        processor,
        template_inputs,
        pv_adv,
        max_new_tokens=MAX_NEW_TOKENS
    )
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