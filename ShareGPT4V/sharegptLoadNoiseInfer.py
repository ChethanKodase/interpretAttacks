'''

Reload a perturbation saved by sharegptAttackGRILLadv.py, add it back onto
the corresponding clean image, and reproduce the adversarial output on
ShareGPT4V-7B with the same prompt used during the attack.

IMPORTANT: the attack script optimizes the perturbation against a
*differentiable approximation* of CLIP preprocessing (bilinear resize +
manual center-crop/pad/normalize -- see sharegpt_preprocess_differentiable
in sharegptAttackGRILLadv.py), and its own printed "ADVERSARIAL OUTPUT" is
generated from pixel values produced by that same approximation, not from
share4v's official image_processor.preprocess()/process_images() path.
Those two preprocessing pipelines are close but not bit-identical (different
resize kernel, no PIL round trip), and adversarial perturbations at small
L_inf budgets are exactly the kind of thing that's sensitive to that. So to
reproduce the same text, this script uses the identical differentiable
preprocessing function, not the "real" CLIP processor. A separate,
clearly-labeled block below also runs the real processor so you can see how
much (if any) the attack survives realistic redeployment.

This script also mirrors the attack script's set_seed(42) and cudnn
determinism flags. They don't matter for the perturbation content (that's
loaded verbatim from disk), but cudnn's non-deterministic conv algorithms
can otherwise introduce tiny numeric drift in the vision tower that -- for
a greedy decode over 100+ autoregressive steps -- can compound into a
different token sequence even with identical inputs and weights.

Usage (hyperparameters below must match the attack run whose noise you want
to reload, so the .pt path can be reconstructed the same way the attack
script wrote it):

export CUDA_VISIBLE_DEVICES=1
cd interpretAttacks/
conda activate share4v
python ShareGPT4V/sharegptLoadNoiseInfer.py --attck_type grillAdv --desired_norm_l_inf 0.003 --learningRate 0.001 --num_steps 1000 --attackSample 1

Or skip path reconstruction entirely and point straight at a .pt file:

python ShareGPT4V/sharegptLoadNoiseInfer.py --attackSample 1 --noise_path /home/luser/interpretAttacks/ShareGPT4V/outputsStorage/advOutputs/1/adv_ORIG_attackType_grillAdv_lr_0.001_eps_0.003_AttackStartLayer_0_numLayerstAtAtime_1_num_steps_1000_.pt

'''


import os
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from share4v.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                                DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from share4v.conversation import conv_templates
from share4v.mm_utils import (get_model_name_from_path, process_images,
                               tokenizer_image_token)
from share4v.model.builder import load_pretrained_model
from share4v.utils import disable_torch_init


# ----------------------------
# Reproducibility (must match sharegptAttackGRILLadv.py exactly)
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
# Utilities: image <-> tensor
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
# (copied verbatim from sharegptAttackGRILLadv.py so the pixel values fed
# to the model exactly match what the attack script itself measured)
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

    if target_dtype is not None:
        x = x.to(dtype=target_dtype)
    return x


# ----------------------------
# ShareGPT4V prompt construction (mirrors sharegptAttackGRILLadv.py /
# share4v.eval.run_share4v.eval_model exactly, so the same prompt/token
# layout is reproduced here)
# ----------------------------
def build_template_inputs(tokenizer, question: str, model, model_path: str, device):
    model_name = get_model_name_from_path(model_path)

    qs = question
    if getattr(model.config, "mm_use_im_start_end", False):
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    if 'llama-2' in model_name.lower():
        conv_mode = "share4v_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "share4v_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "share4v_v0"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(device)

    return input_ids


def run_generation(model, tokenizer, input_ids, images, max_new_tokens=128):
    model.eval()
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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Reload a saved GRILL/BSA perturbation, add it to the clean "
            "image, and reproduce the ShareGPT4V-7B adversarial output."
        )
    )

    parser.add_argument("--attck_type", type=str, default="bsa")
    parser.add_argument("--desired_norm_l_inf", type=float, default=0.01)
    parser.add_argument("--learningRate", type=float, default=1e-3)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--numSteps", type=int, default=None)
    parser.add_argument("--attackSample", type=str, default="astronauts68")
    parser.add_argument("--AttackStartLayer", type=int, default=0)
    parser.add_argument("--numLayerstAtAtime", type=int, default=1)
    parser.add_argument(
        "--noise_path",
        type=str,
        default=None,
        help=(
            "Load this .pt perturbation file directly instead of "
            "reconstructing the path from the hyperparameters above."
        ),
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)

    args = parser.parse_args()

    attck_type = args.attck_type
    epsilon = float(args.desired_norm_l_inf)
    lr = float(args.learningRate)
    num_steps = int(args.numSteps) if args.numSteps is not None else int(args.num_steps)
    attackSample = str(args.attackSample)
    AttackStartLayer = int(args.AttackStartLayer)
    numLayerstAtAtime = int(args.numLayerstAtAtime)

    MODEL_PATH = "Lin-Chen/ShareGPT4V-7B"

    # Clean images are ONLY ever read from this directory.
    IMAGE_DIR = "/home/luser/interpretAttacks/llava_attack/dataSamplesForQuant"
    IMAGE_PATH = f"{IMAGE_DIR}/{attackSample}.JPEG"
    QUESTION = "What is shown in this image?"

    # Perturbations / outputs live under this directory.
    OUTPUT_ROOT = "/home/luser/interpretAttacks/ShareGPT4V"

    if args.noise_path is not None:
        adv_noise_path = args.noise_path
    else:
        adv_noise_path = (
            f"{OUTPUT_ROOT}/outputsStorage/advOutputs/{attackSample}/"
            f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
            f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
            f"num_steps_{num_steps}_.pt"
        )

    saved_png_path = (
        f"{OUTPUT_ROOT}/outputsStorage/advOutputs/{attackSample}/"
        f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
        f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
        f"num_steps_{num_steps}_.png"
    )

    reproduced_out_path = (
        f"{OUTPUT_ROOT}/outputsStorage/advOutputs/{attackSample}/"
        f"reproducedFromNoise_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
        f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
        f"num_steps_{num_steps}_.txt"
    )

    if not os.path.exists(adv_noise_path):
        raise FileNotFoundError(f"Could not find saved perturbation at: {adv_noise_path}")

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
    model.config.use_cache = True
    model_dtype = next(model.parameters()).dtype
    image_aspect_ratio = getattr(model.config, "image_aspect_ratio", "pad")

    # ---- Load the clean image (same source dir the attack read it from) ----
    pil_clean = Image.open(IMAGE_PATH).convert("RGB")
    x_orig01 = pil_to_tensor01(pil_clean).to(device)

    # ---- Load the saved perturbation and reconstruct the adversarial image ----
    print(f"Loading perturbation from: {adv_noise_path}")
    delta = torch.load(adv_noise_path, map_location=device)
    delta = delta.to(device=device, dtype=x_orig01.dtype)

    if delta.shape != x_orig01.shape:
        raise ValueError(
            f"Loaded perturbation shape {tuple(delta.shape)} does not match "
            f"clean image shape {tuple(x_orig01.shape)}. Wrong --attackSample "
            f"or mismatched hyperparameters?"
        )

    # Same reconstruction formula sharegptAttackGRILLadv.py used to produce
    # the final adversarial image from best_delta.
    x_adv01 = (x_orig01 + delta).clamp(0.0, 1.0)
    x_adv01 = torch.max(
        torch.min(x_adv01, x_orig01 + epsilon),
        x_orig01 - epsilon,
    ).clamp(0.0, 1.0)

    # ---- Optional sanity check against the PNG the attack script saved ----
    if os.path.exists(saved_png_path):
        saved_pil = Image.open(saved_png_path).convert("RGB")
        saved_arr = np.array(saved_pil, dtype=np.float32)
        recon_arr = np.array(tensor01_to_pil(x_adv01), dtype=np.float32)
        if saved_arr.shape == recon_arr.shape:
            max_diff = float(np.abs(saved_arr - recon_arr).max())
            print(f"[Sanity check] max abs pixel diff vs saved adv PNG (0-255 scale): {max_diff}")
        else:
            print("[Sanity check] saved PNG shape does not match reconstructed image shape, skipping diff.")
    else:
        print(f"[Sanity check] no saved adv PNG found at {saved_png_path}, skipping diff.")

    # ---- Build the same prompt used during the attack ----
    input_ids = build_template_inputs(tokenizer, QUESTION, model, MODEL_PATH, device)

    # ================================================================
    # PRIMARY reproduction: identical differentiable preprocessing the
    # attack script itself used to print "ADVERSARIAL OUTPUT". This is
    # what will actually match sharegptAttackGRILLadv.py's own console
    # output for this sample/hyperparameter combination.
    # ================================================================
    clean_pixel_values_diff = sharegpt_preprocess_differentiable(
        x_orig01, image_processor, image_aspect_ratio, target_dtype=model_dtype
    )
    adv_pixel_values_diff = sharegpt_preprocess_differentiable(
        x_adv01, image_processor, image_aspect_ratio, target_dtype=model_dtype
    )

    print("\n=== CLEAN OUTPUT (differentiable preprocessing, matches attack script) ===")
    clean_text = run_generation(
        model, tokenizer, input_ids, clean_pixel_values_diff,
        max_new_tokens=args.max_new_tokens,
    )
    print(clean_text)

    print("\n=== REPRODUCED ADVERSARIAL OUTPUT (differentiable preprocessing, matches attack script) ===")
    adv_text = run_generation(
        model, tokenizer, input_ids, adv_pixel_values_diff,
        max_new_tokens=args.max_new_tokens,
    )
    print(adv_text)

    # ================================================================
    # SECONDARY check: real, non-differentiable share4v preprocessing
    # (share4v.mm_utils.process_images -- what an actual deployed
    # pipeline would run). Useful to see whether the attack survives
    # realistic redeployment; not expected to match the attack script's
    # printed output exactly.
    # ================================================================
    pil_adv = tensor01_to_pil(x_adv01)
    clean_pixel_values_real = process_images(
        [pil_clean], image_processor, model.config
    ).to(device=device, dtype=model_dtype)
    adv_pixel_values_real = process_images(
        [pil_adv], image_processor, model.config
    ).to(device=device, dtype=model_dtype)

    print("\n=== CLEAN OUTPUT (real share4v preprocessing) ===")
    clean_text_real = run_generation(
        model, tokenizer, input_ids, clean_pixel_values_real,
        max_new_tokens=args.max_new_tokens,
    )
    print(clean_text_real)

    print("\n=== ADVERSARIAL OUTPUT (real share4v preprocessing) ===")
    adv_text_real = run_generation(
        model, tokenizer, input_ids, adv_pixel_values_real,
        max_new_tokens=args.max_new_tokens,
    )
    print(adv_text_real)

    with open(reproduced_out_path, "w") as f:
        f.write("[differentiable preprocessing -- matches attack script]\n")
        f.write("CLEAN OUTPUT:\n" + clean_text + "\n\n")
        f.write("REPRODUCED ADVERSARIAL OUTPUT:\n" + adv_text + "\n\n")
        f.write("[real share4v preprocessing -- realistic redeployment check]\n")
        f.write("CLEAN OUTPUT:\n" + clean_text_real + "\n\n")
        f.write("ADVERSARIAL OUTPUT:\n" + adv_text_real + "\n")

    print(f"\nSaved reproduction log to: {reproduced_out_path}")


if __name__ == "__main__":
    main()