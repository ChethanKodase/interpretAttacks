'''



export CUDA_VISIBLE_DEVICES=2
cd interpretAttacks/
conda activate share4v
for ATTACK_SAMPLE in $(seq 1 50); do
    python ShareGPT4V/sharegptAttackEGA.py --attck_type ega --desired_norm_l_inf 0.003 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --ega_ratio 0.2 --mask_refresh_every 50
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python ShareGPT4V/sharegptAttackEGA.py --attck_type ega --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --ega_ratio 0.2 --mask_refresh_every 50
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python ShareGPT4V/sharegptAttackEGA.py --attck_type ega --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --ega_ratio 0.2 --mask_refresh_every 50
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python ShareGPT4V/sharegptAttackEGA.py --attck_type ega --desired_norm_l_inf 0.002 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --ega_ratio 0.2 --mask_refresh_every 50
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python ShareGPT4V/sharegptAttackEGA.py --attck_type ega --desired_norm_l_inf 0.0025 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --ega_ratio 0.2 --mask_refresh_every 50
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python ShareGPT4V/sharegptAttackEGA.py --attck_type ega --desired_norm_l_inf 0.0035 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --ega_ratio 0.2 --mask_refresh_every 50
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python ShareGPT4V/sharegptAttackEGA.py --attck_type ega --desired_norm_l_inf 0.0045 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --ega_ratio 0.2 --mask_refresh_every 50
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python ShareGPT4V/sharegptAttackEGA.py --attck_type ega --desired_norm_l_inf 0.001 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --ega_ratio 0.2 --mask_refresh_every 50
done

'''


#!/usr/bin/env python
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import argparse
import random
import numpy as np

import torch
import torch.nn.functional as F

from PIL import Image

from share4v.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                                DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from share4v.conversation import conv_templates
from share4v.mm_utils import get_model_name_from_path, tokenizer_image_token
from share4v.model.builder import load_pretrained_model
from share4v.utils import disable_torch_init


# ----------------------------
# Reproducibility
# ----------------------------
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


# ----------------------------
# PIL / tensor helpers
# ----------------------------
def pil_to_tensor01(pil_img: Image.Image) -> torch.Tensor:
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def tensor01_to_pil(t01: torch.Tensor) -> Image.Image:
    if t01.dim() == 4:
        t01 = t01[0]
    t01 = t01.detach().cpu().clamp(0, 1)
    arr = (
        t01.permute(1, 2, 0).numpy() * 255.0
    ).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ----------------------------
# Differentiable preprocessing, CLIP-like
# ----------------------------
def _get_target_hw(image_processor):
    ip = image_processor
    crop = getattr(ip, "crop_size", None)
    size = getattr(ip, "size", None)

    target_h = target_w = None
    if isinstance(crop, dict):
        target_h = crop.get("height", None)
        target_w = crop.get("width", None)
    elif isinstance(crop, int):
        target_h = target_w = crop

    resize_short = None
    if isinstance(size, dict) and "shortest_edge" in size:
        resize_short = size["shortest_edge"]
    elif isinstance(size, dict) and "height" in size and "width" in size:
        resize_short = min(size["height"], size["width"])
    elif isinstance(size, int):
        resize_short = size

    if target_h is None or target_w is None:
        target_h = target_w = 224
    if resize_short is None:
        resize_short = min(target_h, target_w)

    return int(resize_short), int(target_h), int(target_w)


def resize_shortest_edge_keep_aspect(x: torch.Tensor, shortest_edge: int) -> torch.Tensor:
    _, _, H, W = x.shape
    scale = shortest_edge / min(H, W)
    newH = int(round(H * scale))
    newW = int(round(W * scale))
    return F.interpolate(x, size=(newH, newW), mode="bilinear", align_corners=False)


def center_crop(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    _, _, H, W = x.shape
    top = max((H - target_h) // 2, 0)
    left = max((W - target_w) // 2, 0)
    x_crop = x[:, :, top:top + target_h, left:left + target_w]

    pad_h = target_h - x_crop.shape[2]
    pad_w = target_w - x_crop.shape[3]
    if pad_h > 0 or pad_w > 0:
        x_crop = F.pad(x_crop, (0, max(pad_w, 0), 0, max(pad_h, 0)))
    return x_crop


def normalize_like_processor(x01: torch.Tensor, image_processor) -> torch.Tensor:
    mean = torch.tensor(image_processor.image_mean, dtype=x01.dtype, device=x01.device).view(1, 3, 1, 1)
    std = torch.tensor(image_processor.image_std, dtype=x01.dtype, device=x01.device).view(1, 3, 1, 1)
    return (x01 - mean) / std


def expand2square_diff(x01: torch.Tensor, background_color) -> torch.Tensor:
    """
    Differentiable equivalent of share4v.mm_utils.expand2square: pads the
    shorter spatial dimension so the image becomes square, filling with
    background_color (the CLIP image_mean, matching ShareGPT4V's
    image_aspect_ratio == "pad" preprocessing).
    """
    _, C, H, W = x01.shape
    if H == W:
        return x01

    bg = torch.tensor(background_color, dtype=x01.dtype, device=x01.device).view(1, C, 1, 1)
    side = max(H, W)
    out = bg.expand(1, C, side, side).clone()

    if W > H:
        top = (W - H) // 2
        out[:, :, top:top + H, :] = x01
    else:
        left = (H - W) // 2
        out[:, :, :, left:left + W] = x01
    return out


def sharegpt_preprocess_differentiable(
    x01: torch.Tensor,
    image_processor,
    image_aspect_ratio: str = "pad",
    target_dtype: torch.dtype = None,
) -> torch.Tensor:
    x = x01
    if image_aspect_ratio == "pad":
        x = expand2square_diff(x, image_processor.image_mean)

    shortest_edge, th, tw = _get_target_hw(image_processor)
    x = resize_shortest_edge_keep_aspect(x, shortest_edge)
    x = center_crop(x, th, tw)
    x = normalize_like_processor(x, image_processor)

    # ShareGPT4V's CLIPVisionTower casts its *input* to the vision tower's
    # dtype internally, but casts the *output* image_features back to
    # whatever dtype `images` had going in. Handing it fp32 makes
    # encode_images() feed fp32 activations into the fp16 mm_projector
    # Linear and crash on a dtype mismatch, so the model always needs fp16
    # input directly here, same as share4v.eval.run_share4v's `.half()`.
    if target_dtype is not None:
        x = x.to(dtype=target_dtype)
    return x


def expand_valid_mask_for_sharegpt(valid_mask_text, input_ids, logits):
    """
    Expands text-token valid mask to match ShareGPT4V's logits length.

    ShareGPT4V's prepare_inputs_labels_for_multimodal replaces the single
    IMAGE_TOKEN_INDEX (-200) placeholder in input_ids with many image patch
    tokens (576 for the 336px/patch14 CLIP tower) before running the LLM.
    Those image patch positions should be ignored for EGA, same as LLaVA.
    """
    B, text_len = input_ids.shape
    logits_len = logits.shape[1]

    expanded_masks = []

    for b in range(B):
        image_pos = torch.nonzero(
            input_ids[b] == IMAGE_TOKEN_INDEX,
            as_tuple=False,
        ).squeeze(-1)

        if image_pos.numel() == 0:
            raise RuntimeError("Could not find IMAGE_TOKEN_INDEX in input_ids.")

        image_pos = int(image_pos[0].item())

        num_image_tokens = logits_len - text_len + 1

        before = valid_mask_text[b, :image_pos]

        image_ignore = torch.zeros(
            num_image_tokens,
            dtype=torch.bool,
            device=valid_mask_text.device,
        )

        after = valid_mask_text[b, image_pos + 1:]

        expanded = torch.cat([before, image_ignore, after], dim=0)

        if expanded.shape[0] != logits_len:
            raise RuntimeError(
                f"Expanded mask length {expanded.shape[0]} "
                f"does not match logits length {logits_len}"
            )

        expanded_masks.append(expanded)

    return torch.stack(expanded_masks, dim=0)


# ----------------------------
# ShareGPT4V prompt / inputs
# ----------------------------
def get_conv_mode(model_name: str) -> str:
    if 'llama-2' in model_name.lower():
        return "share4v_llama_2"
    elif "v1" in model_name.lower():
        return "share4v_v1"
    elif "mpt" in model_name.lower():
        return "mpt"
    else:
        return "share4v_v0"


def build_template_inputs(tokenizer, question: str, model, model_path: str, device):
    """
    Builds the ShareGPT4V conversation prompt (message=None, i.e. ready for
    generation) and tokenizes it with the <image> placeholder replaced by
    IMAGE_TOKEN_INDEX, exactly like share4v.eval.run_share4v.eval_model.
    """
    model_name = get_model_name_from_path(model_path)

    qs = question
    if getattr(model.config, "mm_use_im_start_end", False):
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv_mode = get_conv_mode(model_name)

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)

    return qs, conv_mode, {"input_ids": input_ids, "attention_mask": attention_mask}


def run_generation_with_images(
    model,
    tokenizer,
    template_inputs,
    images,
    max_new_tokens=128,
):
    model.eval()
    input_ids = template_inputs["input_ids"].clone()

    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            images=images,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    gen_only = out_ids[:, input_ids.shape[1]:]
    return tokenizer.decode(gen_only[0], skip_special_tokens=True).strip()


# ----------------------------
# Teacher-forced EGA helpers
# ----------------------------
def build_teacher_forced_inputs(
    tokenizer,
    qs,
    conv_mode,
    answer_text,
    device,
):
    """
    Builds prompt-only and prompt+answer tokenizations via the same
    conversation template / tokenizer_image_token path build_template_inputs
    uses, since plain str concatenation + tokenizer(...) can't handle the
    <image> placeholder (it must become IMAGE_TOKEN_INDEX, not a literal
    vocab token). Everything before prompt_len is masked out of `labels`
    so only the (teacher-forced) answer tokens contribute.
    """
    conv_prompt_only = conv_templates[conv_mode].copy()
    conv_prompt_only.append_message(conv_prompt_only.roles[0], qs)
    conv_prompt_only.append_message(conv_prompt_only.roles[1], None)
    prompt_only_text = conv_prompt_only.get_prompt()

    prompt_ids = tokenizer_image_token(
        prompt_only_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    )
    prompt_len = int(prompt_ids.shape[0])

    conv_full = conv_templates[conv_mode].copy()
    conv_full.append_message(conv_full.roles[0], qs)
    conv_full.append_message(conv_full.roles[1], answer_text)
    full_text = conv_full.get_prompt()

    full_ids = tokenizer_image_token(
        full_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(device)

    labels = full_ids.clone()
    labels[:, :prompt_len] = -100

    attention_mask = torch.ones_like(full_ids)

    full_inputs = {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "use_cache": False,
    }

    return full_inputs, prompt_len


def token_entropy_from_logits(logits):
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def select_top_entropy_positions(entropy, valid_mask, ratio=0.2):
    selected = torch.zeros_like(valid_mask, dtype=torch.bool)
    B, _ = entropy.shape

    for b in range(B):
        idx = torch.nonzero(valid_mask[b], as_tuple=False).squeeze(-1)

        if idx.numel() == 0:
            continue

        k = max(1, int(idx.numel() * ratio))

        vals = entropy[b, idx]
        topk = torch.topk(vals, k=k, largest=True).indices

        selected_idx = idx[topk]
        selected[b, selected_idx] = True

    return selected


# ----------------------------
# ORIGINAL-image-space EGA attack
# ----------------------------
def adam_attack_original_space_ega(
    model,
    image_processor,
    teacher_inputs,
    x_orig01,
    num_steps,
    lr,
    epsilon,
    device,
    save_conv_path,
    image_aspect_ratio: str = "pad",
    ega_ratio=0.2,
    mask_refresh_every=50,
):
    x_orig01 = x_orig01.detach().to(device)
    model_dtype = next(model.parameters()).dtype

    delta = 0.001 * torch.randn_like(x_orig01, device=device)
    delta.requires_grad_(True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    losses_list = []
    best_metric = -1e18
    best_delta = delta.detach().clone()

    model.train()
    model.config.use_cache = False

    with torch.no_grad():
        pv_clean = sharegpt_preprocess_differentiable(
            x_orig01, image_processor, image_aspect_ratio, target_dtype=model_dtype
        )

        clean_inputs = {
            k: v.clone() if torch.is_tensor(v) else v
            for k, v in teacher_inputs.items()
        }
        clean_inputs["images"] = pv_clean
        clean_inputs["use_cache"] = False

        outputs_clean = model(
            **clean_inputs,
            output_hidden_states=False,
            return_dict=True,
        )

        clean_entropy = token_entropy_from_logits(outputs_clean.logits)

        valid_mask_text = clean_inputs["labels"] != -100

        valid_mask = expand_valid_mask_for_sharegpt(
            valid_mask_text=valid_mask_text,
            input_ids=clean_inputs["input_ids"],
            logits=outputs_clean.logits,
        )

        ega_mask = select_top_entropy_positions(
            clean_entropy,
            valid_mask,
            ratio=ega_ratio,
        )

        print("outputs_clean.logits.shape:", outputs_clean.logits.shape)
        print("num selected EGA positions:", int(ega_mask.sum().item()))

    adv_inputs = {
        k: v.clone() if torch.is_tensor(v) else v
        for k, v in teacher_inputs.items()
    }
    adv_inputs["use_cache"] = False

    for step in range(num_steps):
        x_adv01 = (x_orig01 + delta).clamp(0.0, 1.0)
        x_adv01 = torch.max(
            torch.min(x_adv01, x_orig01 + epsilon),
            x_orig01 - epsilon,
        ).clamp(0.0, 1.0)

        pv_adv = sharegpt_preprocess_differentiable(
            x_adv01, image_processor, image_aspect_ratio, target_dtype=model_dtype
        )

        adv_inputs["images"] = pv_adv

        outputs_adv = model(
            **adv_inputs,
            output_hidden_states=False,
            return_dict=True,
        )

        if (
            mask_refresh_every > 0
            and step > 0
            and step % mask_refresh_every == 0
        ):
            with torch.no_grad():
                current_entropy = token_entropy_from_logits(outputs_adv.logits.detach())
                ega_mask = select_top_entropy_positions(
                    current_entropy,
                    valid_mask,
                    ratio=ega_ratio,
                )

        adv_entropy = token_entropy_from_logits(outputs_adv.logits)

        loss_ega = adv_entropy[ega_mask].mean()

        # Maximize entropy on selected answer-token positions.
        loss_total = -loss_ega

        optimizer.zero_grad(set_to_none=True)
        loss_total.backward()
        optimizer.step()

        with torch.no_grad():
            delta.data.clamp_(-epsilon, epsilon)

        current_metric = float(loss_ega.item())
        losses_list.append(current_metric)

        if step == 0 or (step + 1) % 10 == 0:
            print(
                f"[step {step + 1}/{num_steps}] "
                f"ega_entropy={current_metric:.6f} "
                f"n_selected={int(ega_mask.sum().item())}"
            )

        if current_metric > best_metric:
            best_metric = current_metric
            best_delta = delta.detach().clone()
            np.save(save_conv_path, np.array(losses_list, dtype=np.float32))

        del outputs_adv, adv_entropy, loss_ega, loss_total, pv_adv

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
        description="ShareGPT4V-7B original-image-space EGA attack"
    )

    parser.add_argument("--attck_type", type=str, default="ega")
    parser.add_argument("--desired_norm_l_inf", type=float, default=0.001)
    parser.add_argument("--learningRate", type=float, default=0.001)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--numSteps", type=int, default=None)
    parser.add_argument("--attackSample", type=str, default="astronauts68")
    parser.add_argument("--ega_ratio", type=float, default=0.2)
    parser.add_argument("--mask_refresh_every", type=int, default=50)

    args = parser.parse_args()

    if args.attck_type != "ega":
        raise ValueError("This script is only for --attck_type ega")

    attck_type = args.attck_type
    epsilon = float(args.desired_norm_l_inf)
    lr = float(args.learningRate)
    num_steps = int(args.numSteps) if args.numSteps is not None else int(args.num_steps)
    attackSample = str(args.attackSample)
    ega_ratio = float(args.ega_ratio)
    mask_refresh_every = int(args.mask_refresh_every)

    MODEL_PATH = "Lin-Chen/ShareGPT4V-7B"  # or "./checkpoints/ShareGPT4V-7B" if downloaded locally

    # Images are ONLY ever read from this directory.
    IMAGE_DIR = "/home/luser/interpretAttacks/llava_attack/dataSamplesForQuant"
    IMAGE_PATH = f"{IMAGE_DIR}/{attackSample}.JPEG"
    QUESTION = "What is shown in this image?"
    MAX_NEW_TOKENS = 128

    # Everything is written under this directory.
    OUTPUT_ROOT = "/home/luser/interpretAttacks/ShareGPT4V"

    os.makedirs(f"{OUTPUT_ROOT}/outputsStorage", exist_ok=True)
    os.makedirs(f"{OUTPUT_ROOT}/outputsStorage/advOutputs/{attackSample}", exist_ok=True)
    os.makedirs(f"{OUTPUT_ROOT}/outputsStorage/convergence/{attackSample}", exist_ok=True)

    conv_path = (
        f"{OUTPUT_ROOT}/outputsStorage/convergence/{attackSample}/"
        f"sharegpt4v_ORIG_attack_{attck_type}"
        f"_lr_{lr}_eps_{epsilon}"
        f"_num_steps_{num_steps}"
        f"_ratio_{ega_ratio}.npy"
    )

    adv_img_path = (
        f"{OUTPUT_ROOT}/outputsStorage/advOutputs/{attackSample}/"
        f"adv_ORIG_attackType_{attck_type}"
        f"_lr_{lr}_eps_{epsilon}"
        f"_num_steps_{num_steps}"
        f"_ratio_{ega_ratio}.png"
    )

    adv_noise_path = (
        f"{OUTPUT_ROOT}/outputsStorage/advOutputs/{attackSample}/"
        f"adv_ORIG_attackType_{attck_type}"
        f"_lr_{lr}_eps_{epsilon}"
        f"_num_steps_{num_steps}"
        f"_ratio_{ega_ratio}.pt"
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    disable_torch_init()

    print("Loading tokenizer + model + image_processor...")
    model_name = get_model_name_from_path(MODEL_PATH)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=MODEL_PATH,
        model_base=None,
        model_name=model_name,
        device=device.type,
    )

    model.eval()
    model.config.use_cache = False

    image_aspect_ratio = getattr(model.config, "image_aspect_ratio", "pad")
    model_dtype = next(model.parameters()).dtype

    pil = Image.open(IMAGE_PATH).convert("RGB")
    x_orig01 = pil_to_tensor01(pil).to(device)

    qs, conv_mode, template_inputs = build_template_inputs(
        tokenizer, QUESTION, model, MODEL_PATH, device
    )

    pv_clean = sharegpt_preprocess_differentiable(
        x_orig01, image_processor, image_aspect_ratio, target_dtype=model_dtype
    )

    print("\n=== CLEAN OUTPUT ===")
    clean_text = run_generation_with_images(
        model,
        tokenizer,
        template_inputs,
        pv_clean,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    print(clean_text)

    teacher_inputs, prompt_len = build_teacher_forced_inputs(
        tokenizer=tokenizer,
        qs=qs,
        conv_mode=conv_mode,
        answer_text=clean_text,
        device=device,
    )

    print("prompt_len:", prompt_len)
    print("teacher_inputs[input_ids].shape:", teacher_inputs["input_ids"].shape)
    print("teacher_inputs[labels].shape:", teacher_inputs["labels"].shape)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\nRunning EGA attack...")
    x_adv01, best_pert = adam_attack_original_space_ega(
        model=model,
        image_processor=image_processor,
        teacher_inputs=teacher_inputs,
        x_orig01=x_orig01,
        num_steps=num_steps,
        lr=lr,
        epsilon=epsilon,
        device=device,
        save_conv_path=conv_path,
        image_aspect_ratio=image_aspect_ratio,
        ega_ratio=ega_ratio,
        mask_refresh_every=mask_refresh_every,
    )

    tensor01_to_pil(x_adv01).save(adv_img_path)
    torch.save(best_pert.detach().cpu(), adv_noise_path)

    print(f"\nSaved ORIGINAL-resolution adversarial image to: {adv_img_path}")
    print(f"Saved perturbation to: {adv_noise_path}")

    pv_adv = sharegpt_preprocess_differentiable(
        x_adv01, image_processor, image_aspect_ratio, target_dtype=model_dtype
    )

    print("\n=== ADVERSARIAL OUTPUT ===")
    adv_text = run_generation_with_images(
        model,
        tokenizer,
        template_inputs,
        pv_adv,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    print(adv_text)

    cleanOutTxt = f"{OUTPUT_ROOT}/outputsStorage/advOutputs/{attackSample}/cleanOutput.txt"
    with open(cleanOutTxt, "w") as f:
        f.write(clean_text + "\n\n")

    advOutTxt = (
        f"{OUTPUT_ROOT}/outputsStorage/advOutputs/{attackSample}/"
        f"advOutput_attackType_{attck_type}"
        f"_lr_{lr}_eps_{epsilon}"
        f"_num_steps_{num_steps}"
        f"_ratio_{ega_ratio}.txt"
    )
    with open(advOutTxt, "w") as f:
        f.write(adv_text + "\n")

    print(f"\nSaved outputs to: {advOutTxt}")
    print(f"Saved convergence to: {conv_path}")


if __name__ == "__main__":
    main()