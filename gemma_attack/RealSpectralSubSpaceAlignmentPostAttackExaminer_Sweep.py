"""
RealSpectralSubSpaceAlignmentPostAttackExaminer_Sweep.py

Consolidated sweep version of RealSpectralSubSpaceAlignmentPostAttackExaminerFullHistogramQuantitative.py.

What changed vs. the original per-layer script (and why):

1. Bug fix in the non-attention-head alignment function: the per-rank spectrum
   must be obtained by averaging over TOKENS (dim=0), not over singular-vector
   ranks (dim=1). The original `coeffs.mean(dim=1)` averaged over ranks and
   returned a per-token vector, which contradicts the "top singular vector on
   the left, bottom on the right" framing entirely. Fixed to `coeffs.mean(dim=0)`.

2. For attention-head taps (q/k/v, lan q/k/v), the original code flattened
   (heads, ranks) together. That is NOT one monotonic top->bottom axis -- it's
   `num_heads` repeated top->bottom blocks concatenated. Averaged over heads
   here instead, so every tap point produces one clean rank-ordered spectrum.

3. Normalization bug fix: the previous script normalized a SIGNED quantity by
   dividing by its sum (or by min-max scaling independently per distribution),
   which does not produce a valid probability mass function -- this is exactly
   what caused the earlier `nan`/`inf` outputs. Distributions are now built
   from ENERGY (squared alignment coefficient), which is non-negative by
   construction, requires no clipping/smoothing hacks, and has a direct
   physical meaning (how strongly the representation engages that singular
   direction).

4. Distribution-shift measurement: for an ordered rank axis (top -> bottom),
   the combination that avoids both failure modes discussed is:
     - CentroidShift: signed, can cancel if mass moves both ways
     - WassersteinRankMagnitude: unsigned total redistribution, never cancels
     - TowardBottom / TowardTop: the two directional components, decomposed
       from the signed CDF difference so nothing is lost to cancellation
     - JS / Hellinger / TV distance: computed on the same valid energy pmf
     - Wasserstein / KS test on the raw (signed) per-rank spectrum: sample-
       based, doesn't require any binning/normalization at all
     - MeanShift_Raw: simple signed sanity check

5. Orchestration: loads the model ONCE, preloads all images/templates/deltas
   ONCE (none of that depends on which layer is being probed), and loops over
   LanLayerTrack (mode="lan", 0..lan_layer_max) and VisionLayerTrack
   (mode="vis", 0..vis_layer_max) internally instead of relaunching the
   process per layer via bash. Hooks are registered and removed for every
   layer setting so no stale hooks from a previous layer ever fire again
   (the original script never removed hooks, which is fine for one-shot runs
   but breaks silently the moment you sweep layers in one process).

6. Results are accumulated into a single pandas DataFrame with columns
   [attackMode, VisionLayerTrack, LanLayerTrack, Projection, <all metrics>],
   written to CSV after every layer (safe against interruption on a long
   sweep) and printed as one table at the end.

Run:
export CUDA_VISIBLE_DEVICES=7
conda activate gemma3
cd interpretAttacks
python gemma_attack/RealSpectralSubSpaceAlignmentPostAttackExaminer_Sweep.py \
    --attck_type saa_BSAexp --desired_norm_l_inf 0.005 --learningRate 0.001 \
    --num_steps 1000 --num_samples 49 \
    --lan_layer_max 33 --vis_layer_max 26 \
    --output_csv gemma_attack/outputsStorageImagenet/alignment_shift_sweep.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from scipy.stats import wasserstein_distance as scipy_wasserstein_distance, ks_2samp
from scipy.spatial.distance import jensenshannon

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


# ---------------------------------------------------------------------------
# Image preprocessing (differentiable approximation of the HF processor)
# ---------------------------------------------------------------------------
def pil_to_tensor01(pil_img: Image.Image) -> torch.Tensor:
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _get_target_hw(image_processor):
    ip = image_processor
    target_h = target_w = None
    crop = getattr(ip, "crop_size", None)
    if isinstance(crop, dict):
        target_h, target_w = crop.get("height"), crop.get("width")
    elif isinstance(crop, int):
        target_h = target_w = crop
    if target_h is None or target_w is None:
        size = getattr(ip, "size", None)
        if isinstance(size, dict):
            if "height" in size and "width" in size:
                target_h, target_w = size["height"], size["width"]
            elif "shortest_edge" in size:
                target_h = target_w = size["shortest_edge"]
        elif isinstance(size, int):
            target_h = target_w = size
    if target_h is None or target_w is None:
        target_h = target_w = 896
    return int(target_h), int(target_w)


def resize_keep_aspect_center_crop(x, target_h, target_w):
    _, _, H, W = x.shape
    scale = max(target_h / H, target_w / W)
    newH, newW = int(round(H * scale)), int(round(W * scale))
    x_resized = F.interpolate(x, size=(newH, newW), mode="bilinear", align_corners=False)
    top = max((newH - target_h) // 2, 0)
    left = max((newW - target_w) // 2, 0)
    x_crop = x_resized[:, :, top:top + target_h, left:left + target_w]
    pad_h = target_h - x_crop.shape[2]
    pad_w = target_w - x_crop.shape[3]
    if pad_h > 0 or pad_w > 0:
        x_crop = F.pad(x_crop, (0, max(pad_w, 0), 0, max(pad_h, 0)))
    return x_crop


def normalize_like_processor(x01, image_processor):
    mean = torch.tensor(image_processor.image_mean, dtype=x01.dtype, device=x01.device).view(1, 3, 1, 1)
    std = torch.tensor(image_processor.image_std, dtype=x01.dtype, device=x01.device).view(1, 3, 1, 1)
    return (x01 - mean) / std


def gemma_preprocess_differentiable(x01, processor):
    ip = processor.image_processor
    th, tw = _get_target_hw(ip)
    x = resize_keep_aspect_center_crop(x01, th, tw)
    return normalize_like_processor(x, ip)


def build_template_inputs(processor, question, pil_image, device):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    template = processor(text=[prompt], images=[pil_image], return_tensors="pt")
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in template.items()}


# ---------------------------------------------------------------------------
# Alignment spectrum extraction
# ---------------------------------------------------------------------------
def spectrum_from_plain_input(InputToLayer, V):
    """Non-head projections (vis-out-proj, FC1/FC2, mm-proj, gate/up/down).
    Returns a (K,) rank-ordered spectrum: mean alignment over TOKENS."""
    H = InputToLayer[0]
    V = V.to(H)
    H_hat = F.normalize(H, dim=1)
    V_hat = F.normalize(V, dim=1)
    coeffs = H_hat @ V_hat.T          # (N tokens, K ranks)
    return coeffs.mean(dim=0)         # FIXED: average over tokens, not ranks


def spectrum_from_head_input(InputToLayer, V):
    """Attention-head projections (vis q/k/v, lan q/k/v).
    Returns a (K,) rank-ordered spectrum: mean alignment over tokens AND heads,
    so the axis is one clean top->bottom spectrum rather than `num_heads`
    concatenated repeats."""
    H = InputToLayer[0]
    V = V.to(H)
    H_hat = F.normalize(H, dim=1)
    V_hat = F.normalize(V, dim=2)
    coeffs = torch.einsum('nd,hkd->hnk', H_hat, V_hat)   # (heads, tokens, ranks)
    return coeffs.mean(dim=1).mean(dim=0)                # -> (ranks,)


# ---------------------------------------------------------------------------
# Hook machinery
# ---------------------------------------------------------------------------
TAP_ORDER = ["qry0", "key0", "val0", "visOutProj", "FC1", "FC2", "MulModProj",
             "qryLan", "keyLan", "valLan", "gate", "up", "down"]
HEAD_TAPS = {"qry0", "key0", "val0", "qryLan", "keyLan", "valLan"}

_captured = {t: None for t in TAP_ORDER}


def _make_hook(tap):
    def _hook(module, inputs):
        _captured[tap] = inputs[0]
    return _hook


def register_all_hooks(model, VisionLayerTrack, LanLayerTrack):
    vis_layer = model.vision_tower.vision_model.encoder.layers[VisionLayerTrack]
    lan_layer = model.language_model.model.layers[LanLayerTrack]
    tap_modules = {
        "qry0": vis_layer.self_attn.q_proj,
        "key0": vis_layer.self_attn.k_proj,
        "val0": vis_layer.self_attn.v_proj,
        "visOutProj": vis_layer.self_attn.out_proj,
        "FC1": vis_layer.mlp.fc1,
        "FC2": vis_layer.mlp.fc2,
        "MulModProj": model.multi_modal_projector,
        "qryLan": lan_layer.self_attn.q_proj,
        "keyLan": lan_layer.self_attn.k_proj,
        "valLan": lan_layer.self_attn.v_proj,
        "gate": lan_layer.mlp.gate_proj,
        "up": lan_layer.mlp.up_proj,
        "down": lan_layer.mlp.down_proj,
    }
    handles = [module.register_forward_pre_hook(_make_hook(tap)) for tap, module in tap_modules.items()]
    return handles, tap_modules


def remove_hooks(handles):
    for h in handles:
        h.remove()


# ---------------------------------------------------------------------------
# SVD extraction (right singular vectors, full spectrum, rank 0 = dominant)
# ---------------------------------------------------------------------------
def svd_right(linear_layer):
    return torch.linalg.svd(linear_layer.weight.to(torch.float32))[2]


def svd_right_mmproj(mm_proj):
    return torch.linalg.svd(mm_proj.mm_input_projection_weight.T.to(torch.float32))[2]


def svd_right_heads(param_module, num_heads):
    param = param_module.weight.to(torch.float32)      # (out_features, in_features)
    d_model = param.shape[1]
    d_head = param.shape[0] // num_heads
    param_heads = param.view(num_heads, d_head, d_model)
    vh = [torch.linalg.svd(param_heads[h])[2] for h in range(num_heads)]
    return torch.stack(vh, 0)                            # (num_heads, d_head, d_model)


def build_singular_vectors(tap_modules, num_heads_vis, num_heads_lan):
    sv = {}
    sv["qry0"] = svd_right_heads(tap_modules["qry0"], num_heads_vis)
    sv["key0"] = svd_right_heads(tap_modules["key0"], num_heads_vis)
    sv["val0"] = svd_right_heads(tap_modules["val0"], num_heads_vis)
    sv["visOutProj"] = svd_right(tap_modules["visOutProj"])
    sv["FC1"] = svd_right(tap_modules["FC1"])
    sv["FC2"] = svd_right(tap_modules["FC2"])
    sv["MulModProj"] = svd_right_mmproj(tap_modules["MulModProj"])
    sv["qryLan"] = svd_right_heads(tap_modules["qryLan"], num_heads_lan)
    sv["keyLan"] = svd_right_heads(tap_modules["keyLan"], num_heads_lan)
    sv["valLan"] = svd_right_heads(tap_modules["valLan"], num_heads_lan)
    sv["gate"] = svd_right(tap_modules["gate"])
    sv["up"] = svd_right(tap_modules["up"])
    sv["down"] = svd_right(tap_modules["down"])
    return sv


# ---------------------------------------------------------------------------
# Distribution-shift metrics (magnitude + direction, top vs. bottom)
# ---------------------------------------------------------------------------
def compute_shift_metrics(orig_raw, adv_raw):
    orig_raw = np.asarray(orig_raw, dtype=np.float64)
    adv_raw = np.asarray(adv_raw, dtype=np.float64)
    eps = 1e-12

    # Non-negative energy pmf -- valid probability distribution, no smoothing needed.
    orig_energy = orig_raw ** 2
    adv_energy = adv_raw ** 2
    p = orig_energy / (orig_energy.sum() + eps)
    q = adv_energy / (adv_energy.sum() + eps)

    ranks = np.arange(len(p))
    centroid_orig = (ranks * p).sum()
    centroid_adv = (ranks * q).sum()
    centroid_shift = centroid_adv - centroid_orig    # >0: toward bottom, <0: toward top (can cancel)

    cdf_p, cdf_q = np.cumsum(p), np.cumsum(q)
    D = cdf_p - cdf_q
    toward_bottom = D[D > 0].sum()
    toward_top = (-D[D < 0]).sum()
    wasserstein_rank = toward_bottom + toward_top    # total magnitude, never cancels
    net_direction = toward_bottom - toward_top        # cross-check vs centroid_shift

    js_distance = jensenshannon(p, q)
    hellinger = np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2)
    tv_distance = 0.5 * np.abs(p - q).sum()

    w1_raw = scipy_wasserstein_distance(orig_raw, adv_raw)
    ks_stat, ks_p = ks_2samp(orig_raw, adv_raw)
    mean_shift_raw = float(adv_raw.mean() - orig_raw.mean())

    return {
        "CentroidShift": float(centroid_shift),
        "NetDirection_CDF": float(net_direction),
        "WassersteinRankMagnitude": float(wasserstein_rank),
        "TowardBottom": float(toward_bottom),
        "TowardTop": float(toward_top),
        "JS_Distance": float(js_distance),
        "Hellinger_Distance": float(hellinger),
        "TV_Distance": float(tv_distance),
        "Wasserstein_RawValues": float(w1_raw),
        "KS_Statistic": float(ks_stat),
        "KS_pvalue": float(ks_p),
        "MeanShift_Raw": mean_shift_raw,
    }


# ---------------------------------------------------------------------------
# Forward pass -> per-tap spectra for one image at one layer setting
# ---------------------------------------------------------------------------
def collect_spectra(model, processor, template_inputs, x_orig01, delta, epsilon, singular_vectors):
    x_adv01 = (x_orig01 + delta).clamp(0.0, 1.0)
    x_adv01 = torch.max(torch.min(x_adv01, x_orig01 + epsilon), x_orig01 - epsilon).clamp(0.0, 1.0)
    pv = gemma_preprocess_differentiable(x_adv01, processor)

    inputs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in template_inputs.items()}
    inputs["pixel_values"] = pv
    inputs["labels"] = template_inputs["input_ids"]
    inputs["use_cache"] = False

    with torch.no_grad():
        model(**inputs, output_hidden_states=False, return_dict=True)
        spectra = []
        for tap in TAP_ORDER:
            fn = spectrum_from_head_input if tap in HEAD_TAPS else spectrum_from_plain_input
            spectra.append(fn(_captured[tap], singular_vectors[tap]).detach().to(torch.float32).cpu())
    return spectra


# ---------------------------------------------------------------------------
# Preload samples once (image / template / adversarial delta do not depend
# on which layer is being probed)
# ---------------------------------------------------------------------------
def preload_samples(processor, device, image_dir, question, num_samples,
                     attck_type, epsilon, lr, num_steps,
                     towardsNullR, AttackStartLayerR, numLayerstAtAtimeR,
                     whichMLPR, whichMLPvisR, balancingAlphaR):
    samples = []
    for idx in range(1, num_samples + 1):
        image_path = os.path.join(image_dir, f"{idx}.JPEG")
        adv_noise_path = (
            f"gemma_attack/outputsStorageImagenet/advOutputs/{idx}/"
            f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
            f"AttackStartLayer_{AttackStartLayerR}_numLayerstAtAtime_{numLayerstAtAtimeR}_"
            f"num_steps_{num_steps}_towardsNull_{towardsNullR}_"
            f"lanMLP_{whichMLPR}_visMLP_{whichMLPvisR}_"
            f"lanLayers_upto4_visLayers_all_balancingAlpha_{balancingAlphaR}.pt"
        )
        if not os.path.exists(image_path) or not os.path.exists(adv_noise_path):
            print(f"[skip] missing file for sample {idx}: "
                  f"{image_path if not os.path.exists(image_path) else adv_noise_path}")
            continue
        pil = Image.open(image_path).convert("RGB")
        x_orig01 = pil_to_tensor01(pil).to(device)
        template_inputs = build_template_inputs(processor, question, pil, device)
        best_delta = torch.load(adv_noise_path, map_location="cpu", weights_only=False).to(device)
        samples.append({"x_orig01": x_orig01, "template_inputs": template_inputs, "best_delta": best_delta})
    print(f"Preloaded {len(samples)} samples.")
    return samples


# ---------------------------------------------------------------------------
# Per-layer-setting analysis
# ---------------------------------------------------------------------------
POINT_LABELS_VIS = ["query proj", "key proj", "value proj", "att output proj", "MLP exp", "MLP out proj"]
POINT_LABELS_LAN = ["query proj", "key proj", "value proj", "MLP gate proj", "MLP up proj", "MLP down proj"]


def analyze_setup(model, processor, samples, attackMode, VisionLayerTrack, LanLayerTrack,
                   epsilon, num_heads_vis, num_heads_lan):
    handles, tap_modules = register_all_hooks(model, VisionLayerTrack, LanLayerTrack)
    try:
        with torch.no_grad():
            singular_vectors = build_singular_vectors(tap_modules, num_heads_vis, num_heads_lan)

        orig_sum, adv_sum, n = None, None, 0
        for s in samples:
            adv_spectra = collect_spectra(model, processor, s["template_inputs"], s["x_orig01"],
                                           s["best_delta"], epsilon, singular_vectors)
            orig_spectra = collect_spectra(model, processor, s["template_inputs"], s["x_orig01"],
                                            s["best_delta"] * 0, epsilon, singular_vectors)
            if orig_sum is None:
                orig_sum = [t.clone() for t in orig_spectra]
                adv_sum = [t.clone() for t in adv_spectra]
            else:
                for k in range(len(orig_sum)):
                    orig_sum[k] += orig_spectra[k]
                    adv_sum[k] += adv_spectra[k]
            n += 1
        orig_mean = [t / n for t in orig_sum]
        adv_mean = [t / n for t in adv_sum]
    finally:
        remove_hooks(handles)

    # tap order is: [qry0,key0,val0,visOutProj,FC1,FC2, MulModProj, qryLan,keyLan,valLan,gate,up,down]
    # vis-relevant taps = first 6, lan-relevant taps = last 6 (skip index 6, the vis->lan projector)
    if attackMode == "vis":
        orig_sel, adv_sel, labels = orig_mean[:6], adv_mean[:6], POINT_LABELS_VIS
    else:
        orig_sel, adv_sel, labels = orig_mean[7:], adv_mean[7:], POINT_LABELS_LAN

    rows = []
    for label, o, a in zip(labels, orig_sel, adv_sel):
        metrics = compute_shift_metrics(o.numpy(), a.numpy())
        row = {"attackMode": attackMode, "VisionLayerTrack": VisionLayerTrack,
               "LanLayerTrack": LanLayerTrack, "Projection": label}
        row.update(metrics)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Sweep alignment-shift metrics across all vision/language layers")
    parser.add_argument("--model_path", type=str, default="../illcond/gemma_attack/Gemma3-4b")
    parser.add_argument("--image_dir", type=str, default="gemma_attack/dataSamplesForQuant")
    parser.add_argument("--question", type=str, default="What is shown in this image?")
    parser.add_argument("--attck_type", type=str, default="saa_BSAexp")
    parser.add_argument("--desired_norm_l_inf", type=float, default=0.005)
    parser.add_argument("--learningRate", type=float, default=0.001)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--num_samples", type=int, default=49)
    parser.add_argument("--vis_layer_max", type=int, default=26, help="sweep VisionLayerTrack 0..this (inclusive)")
    parser.add_argument("--lan_layer_max", type=int, default=33, help="sweep LanLayerTrack 0..this (inclusive)")
    # fixed hyperparams that identify which precomputed adversarial delta file to load
    parser.add_argument("--towardsNull", type=float, default=0.15)
    parser.add_argument("--attack_start_layer", type=int, default=0)
    parser.add_argument("--num_layers_at_a_time", type=int, default=2)
    parser.add_argument("--which_mlp_lan", type=str, default="up_proj")
    parser.add_argument("--which_mlp_vis", type=str, default="fc2")
    parser.add_argument("--balancing_alpha", type=float, default=0.5)
    parser.add_argument("--output_csv", type=str, default="/data1/chethan/interpretAttacks/gemma_attack/HistogramTabulate/alignment_shift_sweep.csv")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"device={device}, dtype={dtype}")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_path, padding_side="left")

    print("Loading model...")
    model = Gemma3ForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=dtype).to(device)
    model.eval()
    model.config.use_cache = False

    num_heads_vis = model.vision_tower.vision_model.encoder.layers[0].self_attn.num_heads
    num_heads_lan = model.language_model.config.num_attention_heads

    samples = preload_samples(
        processor, device, args.image_dir, args.question, args.num_samples,
        args.attck_type, args.desired_norm_l_inf, args.learningRate, args.num_steps,
        args.towardsNull, args.attack_start_layer, args.num_layers_at_a_time,
        args.which_mlp_lan, args.which_mlp_vis, args.balancing_alpha,
    )
    if len(samples) == 0:
        raise RuntimeError("No samples could be preloaded -- check --image_dir and the adv delta file naming.")

    all_rows = []
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    for LanLayerTrack in tqdm(range(0, args.lan_layer_max + 1), desc="lan sweep"):
        try:
            rows = analyze_setup(model, processor, samples, "lan", 0, LanLayerTrack,
                                  args.desired_norm_l_inf, num_heads_vis, num_heads_lan)
            all_rows.extend(rows)
        except Exception as e:
            print(f"[warn] lan layer {LanLayerTrack} failed: {e}")
        pd.DataFrame(all_rows).to_csv(args.output_csv, index=False)

    for VisionLayerTrack in tqdm(range(0, args.vis_layer_max + 1), desc="vis sweep"):
        try:
            rows = analyze_setup(model, processor, samples, "vis", VisionLayerTrack, 0,
                                  args.desired_norm_l_inf, num_heads_vis, num_heads_lan)
            all_rows.extend(rows)
        except Exception as e:
            print(f"[warn] vis layer {VisionLayerTrack} failed: {e}")
        pd.DataFrame(all_rows).to_csv(args.output_csv, index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output_csv, index=False)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print(f"\nSaved full table to {args.output_csv}")


if __name__ == "__main__":
    main()