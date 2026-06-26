'''

export CUDA_VISIBLE_DEVICES=2
conda deactivate
cd interpretAttacks/
conda activate vlmAttack
export PYTHONNOUSERSITE=1
for LayerTrackNum in $(seq 0 31); do
    python qwen/qwen2_5SingularValAttackImgenetAlignmentTracker_Cos.py --attck_type track_cos --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 1 --AttackStartLayer $LayerTrackNum --numLayerstAtAtime 1 --VisionLayerTrack $LayerTrackNum --LanLayerTrack $LayerTrackNum --kthSingVec -1
    python qwen/qwen2_5SingularValAttackImgenetAlignmentTracker_Cos.py --attck_type track_cos --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 1 --AttackStartLayer $LayerTrackNum --numLayerstAtAtime 1 --VisionLayerTrack $LayerTrackNum --LanLayerTrack $LayerTrackNum --kthSingVec 0
done

for LayerTrackNum in $(seq 32 35); do
    python qwen/qwen2_5SingularValAttackImgenetAlignmentTracker_Cos.py --attck_type track_cos --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 1 --AttackStartLayer $LayerTrackNum --numLayerstAtAtime 1 --VisionLayerTrack 31 --LanLayerTrack $LayerTrackNum --kthSingVec -1
    python qwen/qwen2_5SingularValAttackImgenetAlignmentTracker_Cos.py --attck_type track_cos --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 1 --AttackStartLayer $LayerTrackNum --numLayerstAtAtime 1 --VisionLayerTrack 31 --LanLayerTrack $LayerTrackNum --kthSingVec 0
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
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# ── reproducibility ──────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

set_seed()
torch.use_deterministic_algorithms(True)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32       = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


# ── differentiable Qwen image preprocessing ──────────────────────────────────
def _get_qwen_resize_hw(ip, H, W):
    patch_size = int(getattr(ip, "patch_size", 14))
    merge_size = int(getattr(ip, "merge_size",  2))
    factor     = patch_size * merge_size
    min_pixels = int(getattr(ip, "min_pixels", 56*56))
    max_pixels = int(getattr(ip, "max_pixels", 28*28*1280))
    def rb(x, f): return int(round(x/f)*f)
    def fb(x, f): return int(np.floor(x/f)*f)
    def cb(x, f): return int(np.ceil(x/f)*f)
    h, w = max(factor, rb(H, factor)), max(factor, rb(W, factor))
    if h*w > max_pixels:
        b = np.sqrt((H*W)/max_pixels); h,w = max(factor,fb(H/b,factor)), max(factor,fb(W/b,factor))
    elif h*w < min_pixels:
        b = np.sqrt(min_pixels/(H*W)); h,w = max(factor,cb(H*b,factor)), max(factor,cb(W*b,factor))
    return h, w

def qwen_preprocess_differentiable(x01, processor):
    ip  = processor.image_processor
    _,C,H,W = x01.shape
    ps  = int(ip.patch_size); tps = int(ip.temporal_patch_size); ms = int(ip.merge_size)
    th, tw = _get_qwen_resize_hw(ip, H, W)
    x = F.interpolate(x01, size=(th,tw), mode="bilinear", align_corners=False)
    x = (x - torch.tensor(ip.image_mean,dtype=x.dtype,device=x.device).view(1,3,1,1)) \
      /      torch.tensor(ip.image_std, dtype=x.dtype,device=x.device).view(1,3,1,1)
    x = x.repeat(tps,1,1,1)
    gt,gh,gw = x.shape[0]//tps, th//ps, tw//ps
    x = x.view(gt,tps,3,gh//ms,ms,ps,gw//ms,ms,ps).permute(0,3,6,4,7,2,1,5,8).contiguous()
    pv  = x.view(gt*gh*gw, 3*tps*ps*ps)
    thw = torch.tensor([[gt,gh,gw]], dtype=torch.long, device=x01.device)
    return pv, thw


# ── template / generation ─────────────────────────────────────────────────────
def build_inputs(processor, question, pil, device):
    msgs   = [{"role":"user","content":[{"type":"image"},{"type":"text","text":question}]}]
    prompt = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    tmpl   = processor(text=[prompt], images=[pil], return_tensors="pt")
    return {k: v.to(device) if torch.is_tensor(v) else v for k,v in tmpl.items()}

def generate(model, processor, tmpl, pv, thw, max_new_tokens=128):
    inp = {k: v.clone() if torch.is_tensor(v) else v for k,v in tmpl.items()}
    inp["pixel_values"] = pv; inp["image_grid_thw"] = thw
    with torch.no_grad():
        ids = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(ids[:,inp["input_ids"].shape[1]:],
                                  skip_special_tokens=True,
                                  clean_up_tokenization_spaces=True)[0]


# ── loss ──────────────────────────────────────────────────────────────────────
def cos_flat(a, b):
    a = F.normalize(a.reshape(-1), dim=0)
    b = F.normalize(b.reshape(-1), dim=0)
    return (a*b).sum()

def attack_loss(outputs, outputsN, s, e):
    loss = 0.0
    for h,hn in zip(outputs.hidden_states[s:e], outputsN.hidden_states[s:e]):
        loss = loss + cos_flat(h, hn)
    return loss


# ── SVD helpers ───────────────────────────────────────────────────────────────
def right_sv(W, k):
    return torch.linalg.svd(W.float(), full_matrices=False)[2][k]       # (d_in,)

def left_sv(W, k):
    return torch.linalg.svd(W.float(), full_matrices=False)[0][:,k]     # (d_out,)

def right_sv_per_head(W, nh, dh, k):
    # W: (nh*dh, d_in)  →  (nh, d_in)
    d_in = W.shape[1]
    Wh   = W.float().view(nh, dh, d_in)
    return torch.stack([torch.linalg.svd(Wh[h], full_matrices=False)[2][k]
                        for h in range(nh)])

def left_sv_per_head(W, nh, dh, k):
    # W: (nh*dh, d_in)  →  (nh, dh)
    d_in = W.shape[1]
    Wh   = W.float().view(nh, dh, d_in)
    return torch.stack([torch.linalg.svd(Wh[h], full_matrices=False)[0][:,k]
                        for h in range(nh)])


# ── alignment ─────────────────────────────────────────────────────────────────
def _2d(t):
    """any (..., d) → (T, d)"""
    return t.reshape(-1, t.shape[-1])

def align_right(act, sv):
    """act: (...,d)  sv: (d,)  →  scalar"""
    H = F.normalize(_2d(act).to(sv), dim=1)      # (T, d)
    v = F.normalize(sv, dim=0)                   # (d,)
    return (H @ v).abs().mean().item()

def align_right_heads(act, sv_heads):
    """act: (...,d)  sv_heads: (nh, d)  →  scalar"""
    H = F.normalize(_2d(act).to(sv_heads), dim=1)        # (T, d)
    V = F.normalize(sv_heads, dim=1)                      # (nh, d)
    return (H @ V.T).abs().mean().item()


# ── hooks ─────────────────────────────────────────────────────────────────────
_store = {}

def _pre(key):
    def h(mod, inp): _store[key] = inp[0].detach()
    return h

def _fwd(key):
    def h(mod, inp, out):
        _store[key] = (out[0] if isinstance(out,tuple) else out).detach()
    return h

def register_hooks(model, vis_layer, lan_layer):
    vis = model.model.visual.blocks[vis_layer]
    lan = model.model.language_model.layers[lan_layer]
    # vision: qkv (fused), proj, gate, down
    vis.attn.qkv.register_forward_pre_hook(    _pre("vis_qkv_in"))
    vis.attn.proj.register_forward_pre_hook(   _pre("vis_proj_in"))
    vis.mlp.gate_proj.register_forward_pre_hook(_pre("vis_gate_in"))
    vis.mlp.down_proj.register_forward_pre_hook(_pre("vis_down_in"))
    # merger: hook mlp[0] so input dim == 5120 == weight input dim
    model.model.visual.merger.mlp[0].register_forward_pre_hook(_pre("merger_in"))
    # language: q, k, v, gate, up, down
    lan.self_attn.q_proj.register_forward_pre_hook(_pre("lan_q_in"))
    lan.self_attn.k_proj.register_forward_pre_hook(_pre("lan_k_in"))
    lan.self_attn.v_proj.register_forward_pre_hook(_pre("lan_v_in"))
    lan.mlp.gate_proj.register_forward_pre_hook(   _pre("lan_gate_in"))
    lan.mlp.up_proj.register_forward_pre_hook(     _pre("lan_up_in"))
    lan.mlp.down_proj.register_forward_pre_hook(   _pre("lan_down_in"))


# ── singular vectors ──────────────────────────────────────────────────────────
def compute_svs(model, vis_layer, lan_layer,
                nh_vis, dh_vis, nh_q, nh_kv, k):
    vis = model.model.visual.blocks[vis_layer]
    lan = model.model.language_model.layers[lan_layer]

    W   = vis.attn.qkv.weight   # (3*nh*dh, d_model_vis)  = (3840, 1280)
    d   = nh_vis * dh_vis        # 1280

    sv = {}
    # vision qkv: split into q/k/v slices, per-head SVD
    for name, sl in [("vis_q", W[:d]), ("vis_k", W[d:2*d]), ("vis_v", W[2*d:])]:
        sv[name] = right_sv_per_head(sl, nh_vis, dh_vis, k)   # (nh_vis, d_model_vis)

    # vision proj, gate, down: standard SVD (input-space right SV)
    sv["vis_proj"] = right_sv(vis.attn.proj.weight,    k)   # (d_model_vis,)
    sv["vis_gate"] = right_sv(vis.mlp.gate_proj.weight, k)  # (d_model_vis,)
    sv["vis_down"] = right_sv(vis.mlp.down_proj.weight, k)  # (d_mlp_vis,)

    # merger mlp[0]: (5120, 5120) → right SV is (5120,)
    sv["merger"]   = right_sv(model.model.visual.merger.mlp[0].weight, k)

    # language q (nh_q heads), k/v (nh_kv heads)
    for name, mod, nh in [("lan_q", lan.self_attn.q_proj, nh_q),
                           ("lan_k", lan.self_attn.k_proj, nh_kv),
                           ("lan_v", lan.self_attn.v_proj, nh_kv)]:
        dh = mod.weight.shape[0] // nh
        sv[name] = right_sv_per_head(mod.weight, nh, dh, k)  # (nh, d_model_lan)

    # language mlp
    sv["lan_gate"] = right_sv(lan.mlp.gate_proj.weight, k)
    sv["lan_up"]   = right_sv(lan.mlp.up_proj.weight,   k)
    sv["lan_down"] = right_sv(lan.mlp.down_proj.weight,  k)

    return sv


# ── measure alignments ────────────────────────────────────────────────────────
def measure(sv):
    # vision qkv input is shared for q/k/v (fused projection)
    row = [
        align_right_heads(_store["vis_qkv_in"],  sv["vis_q"]),    # 0
        align_right_heads(_store["vis_qkv_in"],  sv["vis_k"]),    # 1
        align_right_heads(_store["vis_qkv_in"],  sv["vis_v"]),    # 2
        align_right(      _store["vis_proj_in"], sv["vis_proj"]), # 3
        align_right(      _store["vis_gate_in"], sv["vis_gate"]), # 4
        align_right(      _store["vis_down_in"], sv["vis_down"]), # 5
        align_right(      _store["merger_in"],   sv["merger"]),   # 6
        align_right_heads(_store["lan_q_in"],    sv["lan_q"]),    # 7
        align_right_heads(_store["lan_k_in"],    sv["lan_k"]),    # 8
        align_right_heads(_store["lan_v_in"],    sv["lan_v"]),    # 9
        align_right(      _store["lan_gate_in"], sv["lan_gate"]), # 10
        align_right(      _store["lan_up_in"],   sv["lan_up"]),   # 11
        align_right(      _store["lan_down_in"], sv["lan_down"]), # 12
    ]
    return row


# ── attack loop ───────────────────────────────────────────────────────────────
def run_attack(model, processor, tmpl, x_orig, sv,
               num_steps, lr, eps, device,
               start_layer, num_layers, conv_path):

    x_orig = x_orig.detach().to(device)
    delta  = 0.001 * torch.randn_like(x_orig)
    delta.requires_grad_(True)
    opt    = torch.optim.Adam([delta], lr=lr)

    best_loss, best_delta = -1e18, delta.detach().clone()
    losses, align_log     = [0.0], []

    # clean reference forward
    with torch.no_grad():
        pv0, thw0 = qwen_preprocess_differentiable(x_orig, processor)
        ci = {k: v.clone() if torch.is_tensor(v) else v for k,v in tmpl.items()}
        ci.update({"pixel_values": pv0, "image_grid_thw": thw0,
                   "labels": tmpl["input_ids"], "use_cache": False})
        refN = model(**ci, output_hidden_states=True, return_dict=True)
        hs   = len(refN.hidden_states)
        s, e = start_layer, start_layer + num_layers
        print(f"hidden_states={hs}  attack=[{s},{e})")
        assert e <= hs, f"endPos {e} > {hs}"

    adv = {k: v.clone() if torch.is_tensor(v) else v for k,v in tmpl.items()}
    adv.update({"labels": tmpl["input_ids"], "use_cache": False})

    for step in range(num_steps):
        x_adv = torch.max(torch.min((x_orig+delta).clamp(0,1), x_orig+eps),
                          x_orig-eps).clamp(0,1)
        pv, thw = qwen_preprocess_differentiable(x_adv, processor)
        adv["pixel_values"] = pv;  adv["image_grid_thw"] = thw

        out  = model(**adv, output_hidden_states=True, return_dict=True)

        if step % 20 == 0:
            with torch.no_grad():
                align_log.append(measure(sv))

        loss = attack_loss(out, refN, s, e)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            delta.data.clamp_(-eps, eps)

        lv = loss.item()
        if step == 0 or (step+1) % 10 == 0:
            print(f"[{step+1}/{num_steps}] loss={lv:.6f}")
        if lv > best_loss:
            best_loss  = lv
            best_delta = delta.detach().clone()
            losses.append(lv)
            np.save(conv_path, np.array(losses, dtype=np.float32))

        del out, loss, pv, thw

    with torch.no_grad():
        x_final = torch.max(torch.min((x_orig+best_delta).clamp(0,1), x_orig+eps),
                            x_orig-eps).clamp(0,1)
    return x_final, best_delta, np.array(align_log)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--attck_type",         type=str,   default="track_cos")
    pa.add_argument("--desired_norm_l_inf",  type=float, default=0.02)
    pa.add_argument("--learningRate",        type=float, default=0.001)
    pa.add_argument("--num_steps",           type=int,   default=1000)
    pa.add_argument("--attackSample",        type=str,   default="nature")
    pa.add_argument("--AttackStartLayer",    type=int,   default=0)
    pa.add_argument("--numLayerstAtAtime",   type=int,   default=1)
    pa.add_argument("--VisionLayerTrack",    type=int,   default=0)
    pa.add_argument("--LanLayerTrack",       type=int,   default=0)
    pa.add_argument("--kthSingVec",          type=int,   default=0)
    args = pa.parse_args()

    MODEL_PATH = "../illcond/QwenAttack/Qwen2.5-VL-7B-Instruct"
    IMAGE_PATH = f"llava_attack/dataSamplesForQuant/{args.attackSample}.JPEG"
    QUESTION   = "What is shown in this image?"

    for d in ["qwen/outputsStorageImagenet",
              f"qwen/outputsStorageImagenet/advOutputs/{args.attackSample}",
              f"qwen/outputsStorageImagenet/convergence/{args.attackSample}",
              "qwen/outputsStorageImagenet/Alignments"]:
        os.makedirs(d, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"device={device}  dtype={dtype}")

    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False)
    model     = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    MODEL_PATH, dtype=dtype, device_map=None).to(device)
    model.eval()
    model.config.use_cache = False

    # ── architecture dims (read from weight shapes, no guessing) ──
    nh_vis  = model.model.visual.config.num_heads                                      # 16
    dh_vis  = model.model.visual.blocks[0].attn.qkv.weight.shape[1] // nh_vis         # 80
    nh_q    = model.model.language_model.config.num_attention_heads                    # 28
    d_lan   = model.model.language_model.layers[0].self_attn.q_proj.weight.shape[1]   # 3584
    dh_lan  = d_lan // nh_q                                                            # 128
    nh_kv   = model.model.language_model.layers[0].self_attn.k_proj.weight.shape[0] // dh_lan  # 4
    print(f"vis nh={nh_vis} dh={dh_vis} | lan nh_q={nh_q} nh_kv={nh_kv} dh={dh_lan}")

    # ── hooks + singular vectors ──
    register_hooks(model, args.VisionLayerTrack, args.LanLayerTrack)
    with torch.no_grad():
        sv = compute_svs(model, args.VisionLayerTrack, args.LanLayerTrack,
                         nh_vis, dh_vis, nh_q, nh_kv, args.kthSingVec)

    # ── image ──
    pil      = Image.open(IMAGE_PATH).convert("RGB")
    x_orig   = pil_to_tensor01(pil).to(device)
    tmpl     = build_inputs(processor, QUESTION, pil, device)
    pv0, thw0 = qwen_preprocess_differentiable(x_orig, processor)

    print("\n=== CLEAN OUTPUT ===")
    clean_text = generate(model, processor, tmpl, pv0, thw0)
    print(clean_text)
    if device.type == "cuda": torch.cuda.empty_cache()

    # ── file paths ──
    tag = (f"{args.attck_type}_lr_{args.learningRate}_eps_{args.desired_norm_l_inf}"
           f"_start_{args.AttackStartLayer}_n_{args.numLayerstAtAtime}"
           f"_steps_{args.num_steps}")
    base     = f"qwen/outputsStorageImagenet"
    adv_path   = f"{base}/advOutputs/{args.attackSample}/adv_{tag}.png"
    noise_path = f"{base}/advOutputs/{args.attackSample}/adv_{tag}.pt"
    conv_path  = f"{base}/convergence/{args.attackSample}/conv_{tag}.npy"
    align_path = (f"{base}/Alignments/{args.attackSample}_align_{tag}"
                  f"_visL_{args.VisionLayerTrack}_lanL_{args.LanLayerTrack}"
                  f"_k_{args.kthSingVec}.npy")

    # ── attack ──
    x_adv, best_pert, align_log = run_attack(
        model=model, processor=processor, tmpl=tmpl, x_orig=x_orig, sv=sv,
        num_steps=args.num_steps, lr=args.learningRate,
        eps=args.desired_norm_l_inf, device=device,
        start_layer=args.AttackStartLayer, num_layers=args.numLayerstAtAtime,
        conv_path=conv_path)

    # ── save ──
    arr = np.array(x_adv[0].detach().cpu().clamp(0,1).permute(1,2,0).numpy()*255,
                   dtype=np.uint8)
    Image.fromarray(arr).save(adv_path)
    torch.save(best_pert.cpu(), noise_path)
    np.save(align_path, align_log)

    pv_adv, thw_adv = qwen_preprocess_differentiable(x_adv, processor)
    print("\n=== ADVERSARIAL OUTPUT ===")
    adv_text = generate(model, processor, tmpl, pv_adv, thw_adv)
    print(adv_text)

    with open(f"{base}/advOutputs/{args.attackSample}/cleanOutput.txt",  "w") as f: f.write(clean_text+"\n")
    with open(f"{base}/advOutputs/{args.attackSample}/adv_{tag}.txt",    "w") as f: f.write(adv_text+"\n")
    print(f"\nSaved:\n  {adv_path}\n  {noise_path}\n  {align_path}")


def pil_to_tensor01(pil):
    arr = np.array(pil.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)


if __name__ == "__main__":
    main()