'''




export CUDA_VISIBLE_DEVICES=2
conda deactivate
cd interpretAttacks/
conda activate vlmAttack
export PYTHONNOUSERSITE=1
for ATTACK_SAMPLE in $(seq 51 100); do
    python qwen/QwenUntargeted_NLL.py --attck_type nllm --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done

for ATTACK_SAMPLE in $(seq 51 100); do
    python qwen/QwenUntargeted_NLL.py --attck_type nllm --desired_norm_l_inf 0.0045 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done

for ATTACK_SAMPLE in $(seq 51 100); do
    python qwen/QwenUntargeted_NLL.py --attck_type nllm --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done

for ATTACK_SAMPLE in $(seq 51 100); do
    python qwen/QwenUntargeted_NLL.py --attck_type nllm --desired_norm_l_inf 0.0035 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done

for ATTACK_SAMPLE in $(seq 51 100); do
    python qwen/QwenUntargeted_NLL.py --attck_type nllm --desired_norm_l_inf 0.003 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done

for ATTACK_SAMPLE in $(seq 51 100); do
    python qwen/QwenUntargeted_NLL.py --attck_type nllm --desired_norm_l_inf 0.0025 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done

for ATTACK_SAMPLE in $(seq 51 100); do
    python qwen/QwenUntargeted_NLL.py --attck_type nllm --desired_norm_l_inf 0.002 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done

'''



#!/usr/bin/env python
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import argparse
import random
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

torch.use_deterministic_algorithms(True)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def pil_to_tensor01(pil_img):
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def tensor01_to_pil(t01):
    if t01.dim() == 4:
        t01 = t01[0]
    t01 = t01.detach().cpu().clamp(0, 1)
    arr = (t01.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _get_qwen_resize_hw(image_processor, H, W):
    patch_size = int(getattr(image_processor, "patch_size", 14))
    merge_size = int(getattr(image_processor, "merge_size", 2))
    factor = patch_size * merge_size

    min_pixels = int(getattr(image_processor, "min_pixels", 56 * 56))
    max_pixels = int(getattr(image_processor, "max_pixels", 28 * 28 * 1280))

    def round_by_factor(x, f):
        return int(round(x / f) * f)

    def floor_by_factor(x, f):
        return int(np.floor(x / f) * f)

    def ceil_by_factor(x, f):
        return int(np.ceil(x / f) * f)

    h_bar = max(factor, round_by_factor(H, factor))
    w_bar = max(factor, round_by_factor(W, factor))

    if h_bar * w_bar > max_pixels:
        beta = np.sqrt((H * W) / max_pixels)
        h_bar = max(factor, floor_by_factor(H / beta, factor))
        w_bar = max(factor, floor_by_factor(W / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = np.sqrt(min_pixels / (H * W))
        h_bar = max(factor, ceil_by_factor(H * beta, factor))
        w_bar = max(factor, ceil_by_factor(W * beta, factor))

    return int(h_bar), int(w_bar)


def qwen_preprocess_differentiable(x01, processor):
    ip = processor.image_processor
    _, C, H, W = x01.shape
    assert C == 3

    patch_size = int(ip.patch_size)
    temporal_patch_size = int(ip.temporal_patch_size)
    merge_size = int(ip.merge_size)

    target_h, target_w = _get_qwen_resize_hw(ip, H, W)

    x = F.interpolate(
        x01,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )

    mean = torch.tensor(ip.image_mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(ip.image_std, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std

    x = x.repeat(temporal_patch_size, 1, 1, 1)

    grid_t = x.shape[0] // temporal_patch_size
    grid_h = target_h // patch_size
    grid_w = target_w // patch_size

    patches = x.view(
        grid_t,
        temporal_patch_size,
        3,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )

    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).contiguous()

    pixel_values = patches.view(
        grid_t * grid_h * grid_w,
        3 * temporal_patch_size * patch_size * patch_size,
    )

    image_grid_thw = torch.tensor(
        [[grid_t, grid_h, grid_w]],
        dtype=torch.long,
        device=x01.device,
    )

    return pixel_values, image_grid_thw


def build_user_prompt(processor, question):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]

    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )


def build_template_inputs(processor, question, pil_image, device):
    prompt = build_user_prompt(processor, question)

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


def build_attack_teacher_forced_inputs(
    processor,
    question,
    answer_text,
    pil_image,
    device,
):
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
            "content": [
                {"type": "text", "text": answer_text},
            ],
        },
    ]

    prompt_only = processor.apply_chat_template(
        user_messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    prompt_plus_answer = processor.apply_chat_template(
        full_messages,
        add_generation_prompt=False,
        tokenize=False,
    )

    prompt_inputs = processor(
        text=[prompt_only],
        images=[pil_image],
        return_tensors="pt",
    )

    full_inputs = processor(
        text=[prompt_plus_answer],
        images=[pil_image],
        return_tensors="pt",
    )

    prompt_len = prompt_inputs["input_ids"].shape[1]

    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100

    out = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in full_inputs.items()
    }

    out["labels"] = labels.to(device)
    out["use_cache"] = False

    return out, prompt_len


def run_generation_with_pixel_values(
    model,
    processor,
    template_inputs,
    pixel_values,
    image_grid_thw,
    max_new_tokens=128,
):
    model.eval()

    inputs = {
        k: v.clone() if torch.is_tensor(v) else v
        for k, v in template_inputs.items()
    }

    inputs["pixel_values"] = pixel_values
    inputs["image_grid_thw"] = image_grid_thw

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


def adam_attack_original_space_nllm(
    model,
    processor,
    attack_inputs_base,
    x_orig01,
    attck_type,
    num_steps,
    lr,
    epsilon,
    device,
    save_conv_path,
):
    if attck_type.lower() != "nllm":
        raise ValueError(f"Only --attck_type nllm is supported, got: {attck_type}")

    x_orig01 = x_orig01.detach().to(device=device, dtype=torch.float32)

    delta = 0.001 * torch.randn_like(x_orig01, device=device, dtype=torch.float32)
    delta.requires_grad_(True)

    optimizer = torch.optim.Adam([delta], lr=lr)

    losses_list = [0.0]
    best_nll = -1e18
    best_delta = delta.detach().clone()

    model.eval()
    model.config.use_cache = False

    adv_inputs = {
        k: v.clone() if torch.is_tensor(v) else v
        for k, v in attack_inputs_base.items()
    }
    adv_inputs["use_cache"] = False

    for step in range(num_steps):
        x_adv01 = (x_orig01 + delta).clamp(0.0, 1.0)
        x_adv01 = torch.max(
            torch.min(x_adv01, x_orig01 + epsilon),
            x_orig01 - epsilon,
        ).clamp(0.0, 1.0)

        pv_adv, grid_adv = qwen_preprocess_differentiable(x_adv01, processor)
        pv_adv = pv_adv.to(device=device, dtype=next(model.parameters()).dtype)

        adv_inputs["pixel_values"] = pv_adv
        adv_inputs["image_grid_thw"] = grid_adv

        outputs = model(
            **adv_inputs,
            return_dict=True,
        )

        nll = outputs.loss.float()

        # Maximize answer-token NLL using gradient descent.
        objective = -nll

        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()

        with torch.no_grad():
            delta.data.clamp_(-epsilon, epsilon)

        nll_value = float(nll.item())

        if step == 0 or (step + 1) % 10 == 0:
            print(f"[step {step + 1}/{num_steps}] answer_token_nll={nll_value:.6f}")

        if nll_value > best_nll:
            best_nll = nll_value
            best_delta = delta.detach().clone()
            losses_list.append(nll_value)
            np.save(save_conv_path, np.array(losses_list, dtype=np.float32))

        del outputs, nll, objective, pv_adv, grid_adv

    with torch.no_grad():
        x_adv01_final = (x_orig01 + best_delta).clamp(0.0, 1.0)
        x_adv01_final = torch.max(
            torch.min(x_adv01_final, x_orig01 + epsilon),
            x_orig01 - epsilon,
        ).clamp(0.0, 1.0)

    return x_adv01_final, best_delta


def main():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-VL original-image-space NLLm attack"
    )

    parser.add_argument("--attck_type", type=str, default="nllm")
    parser.add_argument("--desired_norm_l_inf", type=float, default=0.0005)
    parser.add_argument("--learningRate", type=float, default=0.001)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--numSteps", type=int, default=None)
    parser.add_argument("--attackSample", type=str, default="1")

    args = parser.parse_args()

    attck_type = args.attck_type
    epsilon = float(args.desired_norm_l_inf)
    lr = float(args.learningRate)
    num_steps = int(args.numSteps) if args.numSteps is not None else int(args.num_steps)
    attackSample = str(args.attackSample)

    MODEL_PATH = "../illcond/QwenAttack/Qwen2.5-VL-7B-Instruct"
    IMAGE_PATH = f"llava_attack/dataSamplesForQuant/{attackSample}.JPEG"

    QUESTION = "What is shown in this image?"
    MAX_NEW_TOKENS = 128

    os.makedirs("qwen/outputsStorageImagenet", exist_ok=True)
    os.makedirs(f"qwen/outputsStorageImagenet/advOutputs/{attackSample}", exist_ok=True)
    os.makedirs(f"qwen/outputsStorageImagenet/convergence/{attackSample}", exist_ok=True)

    conv_path = (
        f"qwen/outputsStorageImagenet/convergence/{attackSample}/"
        f"qwen_ORIG_attack_{attck_type}_lr_{lr}_eps_{epsilon}_num_steps_{num_steps}_.npy"
    )

    adv_img_path = (
        f"qwen/outputsStorageImagenet/advOutputs/{attackSample}/"
        f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_num_steps_{num_steps}_.png"
    )

    adv_noise_path = (
        f"qwen/outputsStorageImagenet/advOutputs/{attackSample}/"
        f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_num_steps_{num_steps}_.pt"
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"device={device}, dtype={dtype}")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False)

    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        device_map=None,
    ).to(device)

    model.eval()
    model.config.use_cache = False

    pil = Image.open(IMAGE_PATH).convert("RGB")
    x_orig01 = pil_to_tensor01(pil).to(device=device, dtype=torch.float32)

    template_inputs = build_template_inputs(
        processor,
        QUESTION,
        pil,
        device,
    )

    pv_clean, grid_clean = qwen_preprocess_differentiable(x_orig01, processor)
    pv_clean = pv_clean.to(device=device, dtype=dtype)

    print("\n=== CLEAN OUTPUT ===")
    clean_text = run_generation_with_pixel_values(
        model,
        processor,
        template_inputs,
        pv_clean,
        grid_clean,
        max_new_tokens=MAX_NEW_TOKENS,
    ).strip()

    print(clean_text)

    if clean_text == "":
        clean_text = "unknown object"

    print("\n=== PSEUDO-TARGET USED FOR NLLm ATTACK ===")
    print(clean_text)

    attack_inputs_base, prompt_len = build_attack_teacher_forced_inputs(
        processor=processor,
        question=QUESTION,
        answer_text=clean_text,
        pil_image=pil,
        device=device,
    )

    print("prompt_len:", prompt_len)
    print("attack_inputs_base[input_ids].shape:", attack_inputs_base["input_ids"].shape)
    print("attack_inputs_base[labels].shape:", attack_inputs_base["labels"].shape)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\nRunning NLLm attack...")
    x_adv01, best_pert = adam_attack_original_space_nllm(
        model=model,
        processor=processor,
        attack_inputs_base=attack_inputs_base,
        x_orig01=x_orig01,
        attck_type=attck_type,
        num_steps=num_steps,
        lr=lr,
        epsilon=epsilon,
        device=device,
        save_conv_path=conv_path,
    )

    tensor01_to_pil(x_adv01).save(adv_img_path)
    torch.save(best_pert.detach().cpu(), adv_noise_path)

    print(f"\nSaved ORIGINAL-resolution adversarial image to: {adv_img_path}")
    print(f"Saved perturbation to: {adv_noise_path}")

    pv_adv, grid_adv = qwen_preprocess_differentiable(x_adv01, processor)
    pv_adv = pv_adv.to(device=device, dtype=dtype)

    print("\n=== ADVERSARIAL OUTPUT ===")
    adv_text = run_generation_with_pixel_values(
        model,
        processor,
        template_inputs,
        pv_adv,
        grid_adv,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    print(adv_text)

    cleanOutTxt = f"qwen/outputsStorageImagenet/advOutputs/{attackSample}/cleanOutput.txt"
    with open(cleanOutTxt, "w") as f:
        f.write(clean_text + "\n\n")

    advOutTxt = (
        f"qwen/outputsStorageImagenet/advOutputs/{attackSample}/"
        f"advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_num_steps_{num_steps}_.txt"
    )

    with open(advOutTxt, "w") as f:
        f.write(adv_text + "\n")

    print(f"\nSaved outputs to: {advOutTxt}")
    print(f"Saved convergence to: {conv_path}")


if __name__ == "__main__":
    main()