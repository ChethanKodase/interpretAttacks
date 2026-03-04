

'''

export CUDA_VISIBLE_DEVICES=3
cd interpretAttacks/
conda activate llava15
python llava_attack/llavaAttackInferenceImagenet.py --attck_type grill_wass --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 1 --AttackStartLayer 0 --numLayerstAtAtime 1
python llava_attack/llavaAttackInferenceImagenet.py --attck_type grill_wass --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 4 --AttackStartLayer 1 --numLayerstAtAtime 1
python llava_attack/llavaAttackInferenceImagenet.py --attck_type grill_wass --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 4 --AttackStartLayer 2 --numLayerstAtAtime 1
python llava_attack/llavaAttackInferenceImagenet.py --attck_type grill_wass --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 4 --AttackStartLayer 3 --numLayerstAtAtime 1
python llava_attack/llavaAttackInferenceImagenet.py --attck_type grill_wass --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 4 --AttackStartLayer 4 --numLayerstAtAtime 1



'''


import os
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from transformers import (
    LlavaForConditionalGeneration,
    LlavaProcessor,
    CLIPImageProcessor,
    LlamaTokenizer,
)


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



def pil_to_tensor01(pil_img: Image.Image) -> torch.Tensor:
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    return t


def _get_target_hw(image_processor):
    """
    CLIPImageProcessor typically has:
      - crop_size {"height":H,"width":W}
      - size {"shortest_edge":S} or {"height":H,"width":W}
    We'll mimic "resize shortest edge -> center crop to crop_size".
    """
    ip = image_processor
    crop = getattr(ip, "crop_size", None)
    size = getattr(ip, "size", None)

    # crop target
    target_h = target_w = None
    if isinstance(crop, dict):
        target_h = crop.get("height", None)
        target_w = crop.get("width", None)
    elif isinstance(crop, int):
        target_h = target_w = crop

    # resize shortest edge
    resize_short = None
    if isinstance(size, dict) and "shortest_edge" in size:
        resize_short = size["shortest_edge"]
    elif isinstance(size, dict) and "height" in size and "width" in size:
        # some configs specify direct size
        resize_short = min(size["height"], size["width"])
    elif isinstance(size, int):
        resize_short = size

    # fallbacks
    if target_h is None or target_w is None:
        # Many CLIP configs crop to 224
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

def llava_preprocess_differentiable(x01: torch.Tensor, image_processor) -> torch.Tensor:
    """
    Differentiable approximation of CLIPImageProcessor:
      resize shortest edge -> center crop -> normalize
    """
    shortest_edge, th, tw = _get_target_hw(image_processor)
    x = resize_shortest_edge_keep_aspect(x01, shortest_edge)
    x = center_crop(x, th, tw)
    x = normalize_like_processor(x, image_processor)
    return x

# ----------------------------
# Build template inputs ONCE (IMPORTANT)
# Mirrors your Gemma logic: fixed input_ids include image token.
# ----------------------------
def build_template_inputs(tokenizer, question: str, device):
    # LLaVA 1.5 prompt format
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    enc = tokenizer(prompt, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}

# ----------------------------
# Generation helper (swap pixel_values)
# ----------------------------
def run_generation_with_pixel_values(model, tokenizer, template_inputs, pixel_values, max_new_tokens=128):
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
    return tokenizer.decode(gen_only[0], skip_special_tokens=True)


def tensor01_to_pil(t01: torch.Tensor) -> Image.Image:
    if t01.dim() == 4:
        t01 = t01[0]
    t01 = t01.detach().cpu().clamp(0, 1)
    arr = (t01.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def main():
    parser = argparse.ArgumentParser(description="LLaVA-1.5 ORIGINAL-image-space adversarial attack (no squeeze)")
    parser.add_argument("--attck_type", type=str, default="grill_wass",
                        help="grill_wass | grill_l2 | grill_cos | oa_l2 | oa_wass | oa_cos")
    parser.add_argument("--desired_norm_l_inf", type=float, default=0.02,
                        help="epsilon L_inf in ORIGINAL pixel space [0..1]")
    parser.add_argument("--learningRate", type=float, default=1e-3,
                        help="Adam learning rate")
    parser.add_argument("--num_steps", type=int, default=1000,
                        help="Number of Adam steps")
    parser.add_argument("--attackSample", type=str, default="astronauts68",
                        help="image filename stem (without extension)")
    parser.add_argument("--AttackStartLayer", type=int, default=0,
                        help="From which layer do you start attack")
    parser.add_argument("--numLayerstAtAtime", type=int, default=1,
                        help="Number of layers taken at a time to attack")

    args = parser.parse_args()

    attck_type = args.attck_type
    epsilon = float(args.desired_norm_l_inf)
    lr = float(args.learningRate)
    num_steps = int(args.num_steps)
    attackSample = str(args.attackSample)
    AttackStartLayer = int(args.AttackStartLayer)
    numLayerstAtAtime = int(args.numLayerstAtAtime)

    # ---- CONFIG
    MODEL_PATH = "/home/luser/LLaVA/llava-1.5-7b-hf"
    IMAGE_PATH = f"llava_attack/dataSamplesForQuant/{attackSample}.JPEG"  # adjust ext if needed
    QUESTION = "What is shown in this image?"
    MAX_NEW_TOKENS = 128


    adv_img_path = (
        f"llava_attack/outputsStorage/advOutputs/{attackSample}/"
        f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
        f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
        f"num_steps_{num_steps}_.png"
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"device={device}, dtype={dtype}")

    print("Loading tokenizer + image_processor (explicit, avoids processor_config)...")
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    image_processor = CLIPImageProcessor.from_pretrained(MODEL_PATH)

    print("Loading model...")
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if device.type == "cuda" else None,
    )
    model.eval()
    model.config.use_cache = False

    # Load original image (keep original resolution)
    pil = Image.open(IMAGE_PATH).convert("RGB")
    x_orig01 = pil_to_tensor01(pil).to(device)
    template_inputs = build_template_inputs(tokenizer, QUESTION, device)
    pv_clean = llava_preprocess_differentiable(x_orig01, image_processor)
    print("\n=== CLEAN OUTPUT ===")
    clean_text = run_generation_with_pixel_values(model, tokenizer, template_inputs, pv_clean, max_new_tokens=MAX_NEW_TOKENS)
    print(clean_text)


    # Load original image (keep original resolution)
    pil = Image.open(adv_img_path).convert("RGB")
    x_advImage = pil_to_tensor01(pil).to(device)
    template_inputs = build_template_inputs(tokenizer, QUESTION, device)
    pv_clean = llava_preprocess_differentiable(x_advImage, image_processor)
    print("\n=== SAVED ADV IMAGE OUTPUT ===")
    clean_text = run_generation_with_pixel_values(model, tokenizer, template_inputs, pv_clean, max_new_tokens=MAX_NEW_TOKENS)
    print(clean_text)


    if device.type == "cuda":
        torch.cuda.empty_cache()


    adv_noise_path = (
        f"llava_attack/outputsStorage/advOutputs/{attackSample}/"
        f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
        f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
        f"num_steps_{num_steps}_.pt"
    )

    best_delta = torch.load(adv_noise_path).to(device)

    x_adv01_final = (x_orig01 + best_delta).clamp(0.0, 1.0)
    x_adv01_final = torch.max(torch.min(x_adv01_final, x_orig01 + epsilon), x_orig01 - epsilon).clamp(0.0, 1.0)


    template_inputs = build_template_inputs(tokenizer, QUESTION, device)
    pv_adv = llava_preprocess_differentiable(x_adv01_final, image_processor)
    print("\n=== ADVERSARIAL OUTPUT RELOADED FROM NOISE and ADDED to THE IMAGE===")
    adv_text = run_generation_with_pixel_values(model, tokenizer, template_inputs, pv_adv, max_new_tokens=MAX_NEW_TOKENS)
    print(adv_text)

    test1 = np.load("/home/luser/interpretAttacks/llava_attack/outputsStorage/convergence/1/llava_ORIG_attack_grill_wass_lr_0.001_eps_0.02_AttackStartLayer_0_numLayerstAtAtime_1_num_steps_1000_.npy")
    test2 = np.load("/home/luser/interpretAttacks/llava_attack/outputsStorage/convergence/1/llava_ORIG_attack_grill_wass_lr_0.001_eps_0.02_AttackStartLayer_5_numLayerstAtAtime_1_num_steps_1000_.npy")

    print("test1", test1.shape)
    print("test2", test2.shape)

if __name__ == "__main__":
    main()