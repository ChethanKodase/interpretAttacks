
'''


export CUDA_VISIBLE_DEVICES=2
conda deactivate
cd interpretAttacks/
conda activate vlmAttack
export PYTHONNOUSERSITE=1
for ATTACK_SAMPLE in $(seq 1 50); do
    python qwen/QwenUntargeted_GRILL_l2C.py --attck_type grill_l2C --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done


export CUDA_VISIBLE_DEVICES=3
conda deactivate
cd interpretAttacks/
conda activate vlmAttack
export PYTHONNOUSERSITE=1
for ATTACK_SAMPLE in $(seq 1 50); do
    python qwen/QwenUntargeted_GRILL_l2C.py --attck_type grill_l2C --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --vision_weight 2 --last_k 4
done

first try
--vision_weight 2 --last_k 4
then try
--vision_weight 1 --last_k 8



export CUDA_VISIBLE_DEVICES=2
conda deactivate
cd interpretAttacks/
conda activate vlmAttack
export PYTHONNOUSERSITE=1
for ATTACK_SAMPLE in $(seq 1 50); do
    python qwen/QwenUntargeted_GRILL_l2C.py --attck_type grill_l2C --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --vision_weight 2 --last_k 4 --vision_weight 1 --last_k 8
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
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


# ----------------------------
# Loss: GRILL-L2 best version
# ----------------------------
def mse(a, b):
    return F.mse_loss(a.float(), b.float(), reduction="mean")


def getGrillL2_best_try(
    acts,
    actsN,
    outputs,
    outputsN,
    vision_start=8,
    vision_end=28,
    lang_start=10,
    last_k=4,
    vision_weight=5.0,
):
    """
    GRILL-style VLM objective:

        objective = log(1 + final_proxy * layer_sum)

    where:
        final_proxy = mean L2 distance over last-K language hidden states
        layer_sum   = weighted vision intermediate distortion + language distortion

    This avoids logits because last hidden states worked better in your experiments.
    """

    vision_layers_adv = acts[vision_start:vision_end]
    vision_layers_clean = actsN[vision_start:vision_end]

    lang_layers_adv = outputs.hidden_states[lang_start:]
    lang_layers_clean = outputsN.hidden_states[lang_start:]

    vis_sum = None
    for h, hn in zip(vision_layers_adv, vision_layers_clean):
        val = mse(h, hn)
        vis_sum = val if vis_sum is None else vis_sum + val

    lan_sum = None
    for h, hn in zip(lang_layers_adv, lang_layers_clean):
        val = mse(h, hn)
        lan_sum = val if lan_sum is None else lan_sum + val

    final_proxy = None
    for h, hn in zip(outputs.hidden_states[-last_k:], outputsN.hidden_states[-last_k:]):
        val = mse(h, hn)
        final_proxy = val if final_proxy is None else final_proxy + val

    if vis_sum is None:
        vis_sum = torch.tensor(0.0, device=outputs.hidden_states[0].device)

    if lan_sum is None:
        lan_sum = torch.tensor(0.0, device=outputs.hidden_states[0].device)

    if final_proxy is None:
        final_proxy = mse(outputs.hidden_states[-1], outputsN.hidden_states[-1])
    else:
        final_proxy = final_proxy / float(last_k)

    layer_sum = vision_weight * vis_sum + lan_sum

    objective = torch.log1p(final_proxy * layer_sum)

    return objective, final_proxy.detach(), vis_sum.detach(), lan_sum.detach()


# ----------------------------
# PIL / tensor helpers
# ----------------------------
def pil_to_tensor01(pil_img):
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def tensor01_to_pil(t01):
    if t01.dim() == 4:
        t01 = t01[0]
    t01 = t01.detach().cpu().clamp(0, 1)
    arr = (
        t01.permute(1, 2, 0).numpy() * 255.0
    ).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ----------------------------
# Differentiable Qwen preprocessing
# ----------------------------
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


# ----------------------------
# Qwen inputs
# ----------------------------
def build_template_inputs(processor, question, pil_image, device):
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
        tokenize=False,
    )

    template = processor(
        text=[prompt],
        images=[pil_image],
        return_tensors="pt",
    )

    return {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in template.items()
    }


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


# ----------------------------
# Vision hooks
# ----------------------------
def run_get_image_features_with_vision_hooks(model, pixel_values, image_grid_thw):
    acts = []
    handles = []

    if hasattr(model, "model") and hasattr(model.model, "visual") and hasattr(model.model.visual, "blocks"):
        blocks = model.model.visual.blocks
    elif hasattr(model, "visual") and hasattr(model.visual, "blocks"):
        blocks = model.visual.blocks
    else:
        raise RuntimeError("Could not find Qwen vision blocks.")

    def hook_fn(module, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        if torch.is_tensor(out):
            acts.append(out)

    for block in blocks:
        handles.append(block.register_forward_hook(hook_fn))

    feat = model.get_image_features(
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )

    for h in handles:
        h.remove()

    if isinstance(feat, tuple):
        feat = feat[0]

    return feat, acts


# ----------------------------
# Attack
# ----------------------------
def adam_attack_original_space(
    model,
    processor,
    template_inputs,
    x_orig01,
    attck_type,
    num_steps,
    lr,
    epsilon,
    device,
    save_conv_path,
    vision_weight=5.0,
    last_k=4,
    vision_start=8,
    vision_end=28,
    lang_start=10,
):
    x_orig01 = x_orig01.detach().to(device)

    delta = 0.001 * torch.randn_like(x_orig01, device=device)
    delta.requires_grad_(True)

    optimizer = torch.optim.Adam([delta], lr=lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_steps,
        eta_min=1e-5,
    )

    scores_list = [0.0]
    best_score = -1e18
    best_delta = delta.detach().clone()

    model.eval()
    model.config.use_cache = False
    model.config.output_hidden_states = True
    model.config.return_dict = True

    with torch.no_grad():
        pv_clean, grid_clean = qwen_preprocess_differentiable(x_orig01, processor)

        clean_inputs = {
            k: v.clone() if torch.is_tensor(v) else v
            for k, v in template_inputs.items()
        }

        clean_inputs["pixel_values"] = pv_clean
        clean_inputs["image_grid_thw"] = grid_clean
        clean_inputs["labels"] = template_inputs["input_ids"]
        clean_inputs["use_cache"] = False

        outputsN = model(
            **clean_inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        _, actsN = run_get_image_features_with_vision_hooks(
            model,
            pv_clean,
            grid_clean,
        )

        print("Number of language hidden states:", len(outputsN.hidden_states))
        print("Number of vision hidden states:", len(actsN))

    adv_inputs = {
        k: v.clone() if torch.is_tensor(v) else v
        for k, v in template_inputs.items()
    }

    adv_inputs["labels"] = template_inputs["input_ids"]
    adv_inputs["use_cache"] = False

    for step in range(num_steps):
        x_adv01 = (x_orig01 + delta).clamp(0.0, 1.0)
        x_adv01 = torch.max(
            torch.min(x_adv01, x_orig01 + epsilon),
            x_orig01 - epsilon,
        ).clamp(0.0, 1.0)

        pv_adv, grid_adv = qwen_preprocess_differentiable(x_adv01, processor)

        adv_inputs["pixel_values"] = pv_adv
        adv_inputs["image_grid_thw"] = grid_adv

        outputs = model(
            **adv_inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        _, acts = run_get_image_features_with_vision_hooks(
            model,
            pv_adv,
            grid_adv,
        )

        objective, final_proxy, vis_sum, lan_sum = getGrillL2_best_try(
            acts=acts,
            actsN=actsN,
            outputs=outputs,
            outputsN=outputsN,
            vision_start=vision_start,
            vision_end=vision_end,
            lang_start=lang_start,
            last_k=last_k,
            vision_weight=vision_weight,
        )

        loss = -objective

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            delta.data.clamp_(-epsilon, epsilon)

        score = float(objective.item())

        if step == 0 or (step + 1) % 10 == 0:
            print(
                f"[Adam step {step + 1}/{num_steps}] "
                f"loss={float(loss.item()):.6f} "
                f"score={score:.6f} "
                f"final_proxy={float(final_proxy.item()):.6f} "
                f"vis_sum={float(vis_sum.item()):.6f} "
                f"lan_sum={float(lan_sum.item()):.6f} "
                f"lr={scheduler.get_last_lr()[0]:.8f}"
            )

        if score > best_score:
            best_score = score
            best_delta = delta.detach().clone()
            scores_list.append(score)
            np.save(save_conv_path, np.array(scores_list, dtype=np.float32))

        del outputs, acts, loss, objective, pv_adv, grid_adv

    with torch.no_grad():
        x_adv01_final = (x_orig01 + best_delta).clamp(0.0, 1.0)
        x_adv01_final = torch.max(
            torch.min(x_adv01_final, x_orig01 + epsilon),
            x_orig01 - epsilon,
        ).clamp(0.0, 1.0)

    print(f"Best GRILL score: {best_score:.6f}")

    return x_adv01_final, best_delta


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-VL original-image-space GRILL-L2 attack"
    )

    parser.add_argument("--attck_type", type=str, default="grill_l2_best")
    parser.add_argument("--desired_norm_l_inf", type=float, default=0.005)
    parser.add_argument("--learningRate", type=float, default=0.001)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--numSteps", type=int, default=None)
    parser.add_argument("--attackSample", type=str, default="nature")

    parser.add_argument("--vision_weight", type=float, default=5.0)
    parser.add_argument("--last_k", type=int, default=4)
    parser.add_argument("--vision_start", type=int, default=8)
    parser.add_argument("--vision_end", type=int, default=28)
    parser.add_argument("--lang_start", type=int, default=10)

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
        f"qwen_ORIG_attack_{attck_type}_lr_{lr}_eps_{epsilon}_num_steps_{num_steps}"
        f"_vw_{args.vision_weight}_lastk_{args.last_k}"
        f"_vs_{args.vision_start}_ve_{args.vision_end}_ls_{args.lang_start}.npy"
    )

    adv_img_path = (
        f"qwen/outputsStorageImagenet/advOutputs/{attackSample}/"
        f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_num_steps_{num_steps}"
        f"_vw_{args.vision_weight}_lastk_{args.last_k}"
        f"_vs_{args.vision_start}_ve_{args.vision_end}_ls_{args.lang_start}.png"
    )

    adv_noise_path = (
        f"qwen/outputsStorageImagenet/advOutputs/{attackSample}/"
        f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_num_steps_{num_steps}"
        f"_vw_{args.vision_weight}_lastk_{args.last_k}"
        f"_vs_{args.vision_start}_ve_{args.vision_end}_ls_{args.lang_start}.pt"
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"device={device}, dtype={dtype}")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False)

    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=dtype,
        device_map=None,
    ).to(device)

    model.eval()
    model.config.use_cache = False

    pil = Image.open(IMAGE_PATH).convert("RGB")
    x_orig01 = pil_to_tensor01(pil).to(device)

    template_inputs = build_template_inputs(
        processor,
        QUESTION,
        pil,
        device,
    )

    pv_clean, grid_clean = qwen_preprocess_differentiable(x_orig01, processor)

    print("\n=== CLEAN OUTPUT ===")
    clean_text = run_generation_with_pixel_values(
        model,
        processor,
        template_inputs,
        pv_clean,
        grid_clean,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    print(clean_text)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\nRunning GRILL-L2 best attack...")
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
        vision_weight=args.vision_weight,
        last_k=args.last_k,
        vision_start=args.vision_start,
        vision_end=args.vision_end,
        lang_start=args.lang_start,
    )

    tensor01_to_pil(x_adv01).save(adv_img_path)
    torch.save(best_pert.detach().cpu(), adv_noise_path)

    print(f"\nSaved ORIGINAL-resolution adversarial image to: {adv_img_path}")
    print(f"Saved perturbation to: {adv_noise_path}")

    pv_adv, grid_adv = qwen_preprocess_differentiable(x_adv01, processor)

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

    cleanOutTxt = (
        f"qwen/outputsStorageImagenet/advOutputs/{attackSample}/cleanOutput.txt"
    )

    with open(cleanOutTxt, "w") as f:
        f.write(clean_text + "\n\n")

    advOutTxt = (
        f"qwen/outputsStorageImagenet/advOutputs/{attackSample}/"
        f"advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_num_steps_{num_steps}"
        f"_vw_{args.vision_weight}_lastk_{args.last_k}"
        f"_vs_{args.vision_start}_ve_{args.vision_end}_ls_{args.lang_start}.txt"
    )

    with open(advOutTxt, "w") as f:
        f.write(adv_text + "\n")

    print(f"\nSaved outputs to: {advOutTxt}")
    print(f"Saved convergence to: {conv_path}")


if __name__ == "__main__":
    main()