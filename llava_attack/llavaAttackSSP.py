
'''




export CUDA_VISIBLE_DEVICES=1
cd interpretAttacks/
conda activate llava15
for ATTACK_SAMPLE in $(seq 1 50); do
  python llava_attack/llavaAttackSSP.py --attck_type ssp --desired_norm_l_inf 0.003 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done
for ATTACK_SAMPLE in $(seq 1 50); do
  python llava_attack/llavaAttackSSP.py --attck_type ssp --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done
for ATTACK_SAMPLE in $(seq 1 50); do
  python llava_attack/llavaAttackSSP.py --attck_type ssp --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done
for ATTACK_SAMPLE in $(seq 1 50); do
  python llava_attack/llavaAttackSSP.py --attck_type ssp --desired_norm_l_inf 0.002 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done
for ATTACK_SAMPLE in $(seq 1 50); do
  python llava_attack/llavaAttackSSP.py --attck_type ssp --desired_norm_l_inf 0.0025 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done
for ATTACK_SAMPLE in $(seq 1 50); do
  python llava_attack/llavaAttackSSP.py --attck_type ssp --desired_norm_l_inf 0.0035 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
done
for ATTACK_SAMPLE in $(seq 1 50); do
  python llava_attack/llavaAttackSSP.py --attck_type ssp --desired_norm_l_inf 0.0045 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE
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
from transformers import (
    LlavaForConditionalGeneration,
    CLIPImageProcessor,
    LlamaTokenizer,
)


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


def _get_target_hw(image_processor):
    crop = getattr(image_processor, "crop_size", None)
    size = getattr(image_processor, "size", None)

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


def resize_shortest_edge_keep_aspect(x: torch.Tensor, shortest_edge: int):
    _, _, H, W = x.shape
    scale = shortest_edge / min(H, W)
    newH = int(round(H * scale))
    newW = int(round(W * scale))

    return F.interpolate(
        x,
        size=(newH, newW),
        mode="bilinear",
        align_corners=False,
    )


def center_crop(x: torch.Tensor, target_h: int, target_w: int):
    _, _, H, W = x.shape

    top = max((H - target_h) // 2, 0)
    left = max((W - target_w) // 2, 0)

    x_crop = x[:, :, top:top + target_h, left:left + target_w]

    pad_h = target_h - x_crop.shape[2]
    pad_w = target_w - x_crop.shape[3]

    if pad_h > 0 or pad_w > 0:
        x_crop = F.pad(
            x_crop,
            (0, max(pad_w, 0), 0, max(pad_h, 0)),
        )

    return x_crop


def normalize_like_processor(x01: torch.Tensor, image_processor):
    mean = torch.tensor(
        image_processor.image_mean,
        dtype=x01.dtype,
        device=x01.device,
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        image_processor.image_std,
        dtype=x01.dtype,
        device=x01.device,
    ).view(1, 3, 1, 1)

    return (x01 - mean) / std


def llava_preprocess_differentiable(x01: torch.Tensor, image_processor):
    shortest_edge, th, tw = _get_target_hw(image_processor)
    x = resize_shortest_edge_keep_aspect(x01, shortest_edge)
    x = center_crop(x, th, tw)
    x = normalize_like_processor(x, image_processor)
    return x


def build_template_inputs(tokenizer, question: str, device):
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    enc = tokenizer(prompt, return_tensors="pt")

    return {
        k: v.to(device)
        for k, v in enc.items()
    }


def run_generation_with_pixel_values(
    model,
    tokenizer,
    template_inputs,
    pixel_values,
    max_new_tokens=128,
):
    model.eval()

    inputs = {
        k: v.clone() if torch.is_tensor(v) else v
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

    return tokenizer.decode(
        gen_only[0],
        skip_special_tokens=True,
    )


def get_llava_vision_layers(model):
    if hasattr(model, "vision_tower"):
        vt = model.vision_tower

        if hasattr(vt, "vision_model") and hasattr(vt.vision_model, "encoder"):
            enc = vt.vision_model.encoder
            if hasattr(enc, "layers"):
                return enc.layers

        if hasattr(vt, "encoder") and hasattr(vt.encoder, "layers"):
            return vt.encoder.layers

    if hasattr(model, "model") and hasattr(model.model, "vision_tower"):
        vt = model.model.vision_tower

        if hasattr(vt, "vision_model") and hasattr(vt.vision_model, "encoder"):
            enc = vt.vision_model.encoder
            if hasattr(enc, "layers"):
                return enc.layers

        if hasattr(vt, "encoder") and hasattr(vt.encoder, "layers"):
            return vt.encoder.layers

    raise RuntimeError("Could not find LLaVA vision tower encoder layers.")


def run_vision_tower_with_hooks(model, pixel_values):
    acts = []
    handles = []

    layers = get_llava_vision_layers(model)

    def hook_fn(module, inp, out):
        if isinstance(out, tuple):
            out = out[0]

        if torch.is_tensor(out):
            acts.append(out)

    for layer in layers:
        handles.append(layer.register_forward_hook(hook_fn))

    vision_outputs = model.vision_tower(
        pixel_values,
        output_hidden_states=True,
        return_dict=True,
    )

    for h in handles:
        h.remove()

    return vision_outputs, acts


def get_ssp_vision_loss(acts_adv, acts_clean, use_last_only=True):
    """
    Self-supervised perturbation objective.

    Minimizes negative MSE, i.e. maximizes feature deviation between
    clean and adversarial vision representations.
    """
    if use_last_only:
        return -F.mse_loss(
            acts_adv[-1].float(),
            acts_clean[-1].float(),
        )

    loss = 0.0

    for h_adv, h_clean in zip(acts_adv, acts_clean):
        loss = loss + F.mse_loss(
            h_adv.float(),
            h_clean.float(),
        )

    return -loss


def adam_attack_original_space_ssp(
    model,
    image_processor,
    x_orig01,
    attck_type,
    num_steps,
    lr,
    epsilon,
    device,
    save_conv_path,
    use_last_only=True,
):
    if attck_type.lower() != "ssp":
        raise ValueError(
            f"Only --attck_type ssp is supported, got: {attck_type}"
        )

    model_dtype = next(model.parameters()).dtype

    x_orig01 = x_orig01.detach().to(
        device=device,
        dtype=torch.float32,
    )

    delta = 0.001 * torch.randn_like(
        x_orig01,
        device=device,
        dtype=torch.float32,
    )

    delta.requires_grad_(True)

    optimizer = torch.optim.Adam(
        [delta],
        lr=lr,
    )

    losses_list = [0.0]
    best_loss = 1e18
    best_delta = delta.detach().clone()

    model.eval()
    model.config.use_cache = False

    with torch.no_grad():
        pv_clean = llava_preprocess_differentiable(
            x_orig01,
            image_processor,
        )

        pv_clean = pv_clean.to(
            device=device,
            dtype=model_dtype,
        )

        _, acts_clean = run_vision_tower_with_hooks(
            model,
            pv_clean,
        )

        acts_clean = [
            a.detach()
            for a in acts_clean
        ]

        print("Number of LLaVA vision layer activations:", len(acts_clean))
        print(
            "SSP objective:",
            "last vision layer only" if use_last_only else "all vision layers",
        )

        if len(acts_clean) == 0:
            raise RuntimeError("No LLaVA vision activations found.")

    for step in range(num_steps):
        x_adv01 = (x_orig01 + delta).clamp(0.0, 1.0)

        x_adv01 = torch.max(
            torch.min(
                x_adv01,
                x_orig01 + epsilon,
            ),
            x_orig01 - epsilon,
        ).clamp(0.0, 1.0)

        pv_adv = llava_preprocess_differentiable(
            x_adv01,
            image_processor,
        )

        pv_adv = pv_adv.to(
            device=device,
            dtype=model_dtype,
        )

        _, acts_adv = run_vision_tower_with_hooks(
            model,
            pv_adv,
        )

        loss = get_ssp_vision_loss(
            acts_adv=acts_adv,
            acts_clean=acts_clean,
            use_last_only=use_last_only,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            delta.data.clamp_(-epsilon, epsilon)

        lv = float(loss.item())

        if step == 0 or (step + 1) % 10 == 0:
            print(
                f"[step {step + 1}/{num_steps}] "
                f"loss={lv:.6f}"
            )

        if lv < best_loss:
            best_loss = lv
            best_delta = delta.detach().clone()

            losses_list.append(lv)

            np.save(
                save_conv_path,
                np.array(losses_list, dtype=np.float32),
            )

        del acts_adv, loss, pv_adv

    with torch.no_grad():
        x_adv01_final = (
            x_orig01 + best_delta
        ).clamp(0.0, 1.0)

        x_adv01_final = torch.max(
            torch.min(
                x_adv01_final,
                x_orig01 + epsilon,
            ),
            x_orig01 - epsilon,
        ).clamp(0.0, 1.0)

    return x_adv01_final, best_delta


def main():
    parser = argparse.ArgumentParser(
        description="LLaVA-1.5 original-image-space SSP / SSPA attack"
    )

    parser.add_argument(
        "--attck_type",
        type=str,
        default="ssp",
    )

    parser.add_argument(
        "--desired_norm_l_inf",
        type=float,
        default=0.0005,
    )

    parser.add_argument(
        "--learningRate",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--num_steps",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--numSteps",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--attackSample",
        type=str,
        default="1",
    )

    parser.add_argument(
        "--use_all_vision_layers",
        action="store_true",
        help="Use all LLaVA vision layers instead of only the last layer.",
    )

    args = parser.parse_args()

    attck_type = args.attck_type
    epsilon = float(args.desired_norm_l_inf)
    lr = float(args.learningRate)

    num_steps = (
        int(args.numSteps)
        if args.numSteps is not None
        else int(args.num_steps)
    )

    attackSample = str(args.attackSample)
    use_last_only = not bool(args.use_all_vision_layers)

    MODEL_PATH = "/home/luser/LLaVA/llava-1.5-7b-hf"

    IMAGE_PATH = (
        f"llava_attack/dataSamplesForQuant/"
        f"{attackSample}.JPEG"
    )

    QUESTION = "What is shown in this image?"
    MAX_NEW_TOKENS = 128

    os.makedirs(
        "llava_attack/outputsStorage",
        exist_ok=True,
    )

    os.makedirs(
        f"llava_attack/outputsStorage/"
        f"advOutputs/{attackSample}",
        exist_ok=True,
    )

    os.makedirs(
        f"llava_attack/outputsStorage/"
        f"convergence/{attackSample}",
        exist_ok=True,
    )

    layer_tag = (
        "allVisionLayers"
        if not use_last_only
        else "lastVisionLayer"
    )

    conv_path = (
        f"llava_attack/outputsStorage/"
        f"convergence/{attackSample}/"
        f"llava_ORIG_attack_{attck_type}"
        f"_lr_{lr}_eps_{epsilon}"
        f"_num_steps_{num_steps}"
        f"_{layer_tag}_.npy"
    )

    adv_img_path = (
        f"llava_attack/outputsStorage/"
        f"advOutputs/{attackSample}/"
        f"adv_ORIG_attackType_{attck_type}"
        f"_lr_{lr}_eps_{epsilon}"
        f"_num_steps_{num_steps}"
        f"_{layer_tag}_.png"
    )

    adv_noise_path = (
        f"llava_attack/outputsStorage/"
        f"advOutputs/{attackSample}/"
        f"adv_ORIG_attackType_{attck_type}"
        f"_lr_{lr}_eps_{epsilon}"
        f"_num_steps_{num_steps}"
        f"_{layer_tag}_.pt"
    )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    dtype = (
        torch.float16
        if device.type == "cuda"
        else torch.float32
    )

    print(f"device={device}, dtype={dtype}")

    print("Loading tokenizer + image_processor...")

    tokenizer = LlamaTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=False,
        legacy=True,
    )

    image_processor = CLIPImageProcessor.from_pretrained(
        MODEL_PATH
    )

    print("Loading model...")

    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=None,
    ).to(device)

    model.eval()
    model.config.use_cache = False

    pil = Image.open(IMAGE_PATH).convert("RGB")

    x_orig01 = pil_to_tensor01(pil).to(
        device=device,
        dtype=torch.float32,
    )

    template_inputs = build_template_inputs(
        tokenizer,
        QUESTION,
        device,
    )

    pv_clean = llava_preprocess_differentiable(
        x_orig01,
        image_processor,
    )

    pv_clean = pv_clean.to(
        device=device,
        dtype=dtype,
    )

    print("\n=== CLEAN OUTPUT ===")

    clean_text = run_generation_with_pixel_values(
        model,
        tokenizer,
        template_inputs,
        pv_clean,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    print(clean_text)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\nRunning SSP / SSPA attack...")

    x_adv01, best_pert = adam_attack_original_space_ssp(
        model=model,
        image_processor=image_processor,
        x_orig01=x_orig01,
        attck_type=attck_type,
        num_steps=num_steps,
        lr=lr,
        epsilon=epsilon,
        device=device,
        save_conv_path=conv_path,
        use_last_only=use_last_only,
    )

    tensor01_to_pil(x_adv01).save(
        adv_img_path
    )

    torch.save(
        best_pert.detach().cpu(),
        adv_noise_path,
    )

    print(
        f"\nSaved ORIGINAL-resolution adversarial image to: "
        f"{adv_img_path}"
    )

    print(
        f"Saved perturbation to: "
        f"{adv_noise_path}"
    )

    pv_adv = llava_preprocess_differentiable(
        x_adv01,
        image_processor,
    )

    pv_adv = pv_adv.to(
        device=device,
        dtype=dtype,
    )

    print("\n=== ADVERSARIAL OUTPUT ===")

    adv_text = run_generation_with_pixel_values(
        model,
        tokenizer,
        template_inputs,
        pv_adv,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    print(adv_text)

    cleanOutTxt = (
        f"llava_attack/outputsStorage/"
        f"advOutputs/{attackSample}/"
        f"cleanOutput.txt"
    )

    with open(cleanOutTxt, "w") as f:
        f.write(clean_text + "\n\n")

    advOutTxt = (
        f"llava_attack/outputsStorage/"
        f"advOutputs/{attackSample}/"
        f"advOutput_attackType_{attck_type}"
        f"_lr_{lr}_eps_{epsilon}"
        f"_num_steps_{num_steps}"
        f"_{layer_tag}_.txt"
    )

    with open(advOutTxt, "w") as f:
        f.write(adv_text + "\n")

    print(f"\nSaved outputs to: {advOutTxt}")
    print(f"Saved convergence to: {conv_path}")


if __name__ == "__main__":
    main()