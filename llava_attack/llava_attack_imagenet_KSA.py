

'''


-----------Vis outproj




export CUDA_VISIBLE_DEVICES=0
cd interpretAttacks/
conda activate llava15
for ATTACK_SAMPLE in $(seq 1 500); do
    for ALIGNLAYER in $(seq 0 23); do
        python llava_attack/llava_attack_imagenet_KSA.py --attck_type saa --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AlignLayer $ALIGNLAYER --towardsNull 0.1 --whichLayMod vis --whichMLP out_proj
        python llava_attack/llava_attack_imagenet_KSA.py --attck_type saa --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AlignLayer $ALIGNLAYER --towardsNull 0 --whichLayMod vis --whichMLP out_proj

        python llava_attack/llava_attack_imagenet_KSA.py --attck_type saa --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AlignLayer $ALIGNLAYER --towardsNull 0.1 --whichLayMod vis --whichMLP fc1
        python llava_attack/llava_attack_imagenet_KSA.py --attck_type saa --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AlignLayer $ALIGNLAYER --towardsNull 0 --whichLayMod vis --whichMLP fc1
    done
done




export CUDA_VISIBLE_DEVICES=1
cd interpretAttacks/
conda activate llava15
for ATTACK_SAMPLE in $(seq 1 500); do
    for ALIGNLAYER in $(seq 0 23); do
        python llava_attack/llava_attack_imagenet_KSA.py --attck_type saa --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AlignLayer $ALIGNLAYER --towardsNull 0.1 --whichLayMod vis --whichMLP fc2
        python llava_attack/llava_attack_imagenet_KSA.py --attck_type saa --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AlignLayer $ALIGNLAYER --towardsNull 0 --whichLayMod vis --whichMLP fc2
    done
done




export CUDA_VISIBLE_DEVICES=2
cd interpretAttacks/
conda activate llava15
for ATTACK_SAMPLE in $(seq 1 500); do
    for ALIGNLAYER in $(seq 0 31); do
        python llava_attack/llava_attack_imagenet_KSA.py --attck_type saa --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AlignLayer $ALIGNLAYER --towardsNull 0.1 --whichLayMod lan --whichMLP up_proj
        python llava_attack/llava_attack_imagenet_KSA.py --attck_type saa --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AlignLayer $ALIGNLAYER --towardsNull 0 --whichLayMod lan --whichMLP up_proj
    done
done


export CUDA_VISIBLE_DEVICES=3
cd interpretAttacks/
conda activate llava15
for ATTACK_SAMPLE in $(seq 1 500); do
    for ALIGNLAYER in $(seq 0 31); do
        python llava_attack/llava_attack_imagenet_KSA.py --attck_type saa --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AlignLayer $ALIGNLAYER --towardsNull 0.1 --whichLayMod lan --whichMLP down_proj
        python llava_attack/llava_attack_imagenet_KSA.py --attck_type saa --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AlignLayer $ALIGNLAYER --towardsNull 0 --whichLayMod lan --whichMLP down_proj
    done
done


-----------Vis fc1

-----------Vis fc2

-----------lan up_proj

-----------lan down_proj



parser.add_argument("--AlignLayer", type=int, default=1,
                    help="values taken are 0 to 23 for vis and 0 to 31 for lan ")
parser.add_argument("--whichLayMod", type=str, default="vis",
                    help="values taken are vis and lan")
parser.add_argument("--whichMLP", type=str, default="fc1",
                    help="values taken : down_proj, up_proj, fc1, fc2, out_proj")

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



def cos(a, b):
    a = a.view(-1)
    b = b.view(-1)
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return (a * b).sum()

def wasserstein_distance(tensor_a, tensor_b):
    tensor_a_flat = torch.flatten(tensor_a)
    tensor_b_flat = torch.flatten(tensor_b)
    tensor_a_sorted, _ = torch.sort(tensor_a_flat)
    tensor_b_sorted, _ = torch.sort(tensor_b_flat)
    wasserstein_dist = torch.mean(torch.abs(tensor_a_sorted - tensor_b_sorted))
    return wasserstein_dist

# ----------------------------
# Losses: GRILL + OA (same style)
# ----------------------------
def get_grill_l2(outputs, outputsN):
    loss = 0.0
    for h, hn in zip(outputs.hidden_states, outputsN.hidden_states):
        loss = loss + criterion(h, hn)
    return loss * criterion(h, hn)

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
    return criterion(outputs.logits, outputsN.logits)

def get_oa_wass(outputs, outputsN):
    return wasserstein_distance(outputs.logits, outputsN.logits)

def get_oa_cos(outputs, outputsN):
    return (1.0 - cos(outputs.logits, outputsN.logits)) ** 2

# ----------------------------
# Utilities: image <-> tensor
# ----------------------------
def pil_to_tensor01(pil_img: Image.Image) -> torch.Tensor:
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    return t

def tensor01_to_pil(t01: torch.Tensor) -> Image.Image:
    if t01.dim() == 4:
        t01 = t01[0]
    t01 = t01.detach().cpu().clamp(0, 1)
    arr = (t01.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)

# ----------------------------
# Differentiable preprocessing (CLIP-like)
# ----------------------------
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


def getMeanAlignmentLossWithTopRightSingularVector(InputToLayer, topRightSingularVector):

    H = InputToLayer[0]        
    V = topRightSingularVector   
    V = V.to(H)

    H_hat = F.normalize(H, dim=1)      
    V_hat = F.normalize(V, dim=1)     
    #print("H_hat.shape", H_hat.shape)
    #print("V_hat.shape", V_hat.shape)
    coeffs = H_hat @ V_hat.T        # (N, k)
    #print("coeffs.shape", coeffs.shape)
    # Energy per token (sum of squared coefficients)
    per_token_energy = (coeffs ** 2).sum(dim=1)  # (N,)
    #print("per_token_energy.shape", per_token_energy.shape)
    # Mean across tokens
    loss = ((1 - per_token_energy)**2).mean()
    #mean_energy = per_token_energy.mean().item()

    return loss

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

# ----------------------------
# ORIGINAL-SPACE Adam attack (same style as yours)
# ----------------------------

layer_inputs = {}
def down_proj_pre_hook(module, inputs):
    # inputs is a tuple; for Linear it's (hidden_tensor,)
    # hidden_tensor shape: (batch, seq_len, d_ff)
    layer_inputs["down_in"] = inputs[0]

def adam_attack_original_space(
    model,
    tokenizer,
    image_processor,
    template_inputs,
    x_orig01,
    attck_type: str,
    num_steps: int,
    lr: float,
    epsilon: float,
    device,
    save_conv_path: str,
    AlignLayer: int,
    towardsNull: float,
    whichLayMod: str,
    whichMLP: str
):
    x_orig01 = x_orig01.detach().to(device)

    delta = 0.001 * torch.randn_like(x_orig01, device=device)
    delta.requires_grad_(True)
    opt = torch.optim.Adam([delta], lr=lr)

    losses_list = [0.0]
    best_loss = 1e5
    best_delta = delta.detach().clone()

    model.train()
    model.config.use_cache = False

    '''for name, param in model.named_parameters():
        print(f"{name:60s} {tuple(param.shape)}")'''


    #paramWt = model.language_model.model.layers[AlignLayer].mlp.down_proj.weight # down proj
    #paramWt = model.language_model.model.layers[AlignLayer].mlp.up_proj.weight # up proj

    # vision trial 
    if whichMLP == "down_proj" and whichLayMod == "lan":
        paramWt = model.language_model.model.layers[AlignLayer].mlp.down_proj.weight

    if whichMLP == "up_proj" and whichLayMod == "lan":
        paramWt = model.language_model.model.layers[AlignLayer].mlp.up_proj.weight

    if whichMLP == "fc1" and whichLayMod == "vis":
        paramWt = model.vision_tower.vision_model.encoder.layers[AlignLayer].mlp.fc1.weight

    if whichMLP == "fc2" and whichLayMod == "vis":
        paramWt = model.vision_tower.vision_model.encoder.layers[AlignLayer].mlp.fc2.weight

    if whichMLP == "out_proj" and whichLayMod == "vis":
        paramWt = model.vision_tower.vision_model.encoder.layers[AlignLayer].self_attn.out_proj.weight
    
    print("paramWt.shape", paramWt.shape)
    U, S, Vh = torch.linalg.svd(paramWt.to(torch.float32))
    print("S.shape", S.shape)
    print("Vh.shape", Vh.shape)


    with torch.no_grad():
        # detach so SVD is NOT part of autograd graph
        U, S, Vh = torch.linalg.svd(paramWt.detach().to(torch.float32), full_matrices=False)
        print("Vh.shape", Vh.shape)
        print("S.shape", S.shape)
        print("S.max(), S.min()", S.max(), S.min())
        print("S", S)
        print("how many less than 1 ?len(S[S<1])", len(S[S<1]))
        #towardsNull = 0
        absorbSize = len(S[S<1])
        #topRightSingularVector = Vh[-1].contiguous()  # constant tensor now
        if towardsNull == 0:
            bottomInd = 1
        else:
            bottomInd = int(absorbSize*towardsNull)
        print("bottomInd", bottomInd)
        topkRightSingularVector = Vh[-bottomInd:].contiguous()
        print("topkRightSingularVector.shape", topkRightSingularVector.shape)


    # Precompute clean pixel_values + clean outputsN once
    with torch.no_grad():
        pv_clean_fixed = llava_preprocess_differentiable(x_orig01, image_processor)

        clean_inputs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in template_inputs.items()}
        clean_inputs["pixel_values"] = pv_clean_fixed
        clean_inputs["labels"] = template_inputs["input_ids"]
        clean_inputs["use_cache"] = False

        outputsN = model(**clean_inputs, output_hidden_states=True, return_dict=True)

        hiddStateLen = len(outputsN.hidden_states)
        print(" Number of hidden states is: ", hiddStateLen)



    adv_inputs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in template_inputs.items()}
    adv_inputs["labels"] = template_inputs["input_ids"]
    adv_inputs["use_cache"] = False
    for step in range(num_steps):
        layer_inputs.clear() 

        # original-space constraint
        x_adv01 = (x_orig01 + delta).clamp(0.0, 1.0)
        x_adv01 = torch.max(torch.min(x_adv01, x_orig01 + epsilon), x_orig01 - epsilon).clamp(0.0, 1.0)

        pv_adv = llava_preprocess_differentiable(x_adv01, image_processor)

        adv_inputs["pixel_values"] = pv_adv

        outputs = model(**adv_inputs, output_hidden_states=True, return_dict=True)

        '''with torch.no_grad():
            clean_inputs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in template_inputs.items()}
            clean_inputs["pixel_values"] = pv_clean_fixed
            clean_inputs["labels"] = template_inputs["input_ids"]
            clean_inputs["use_cache"] = False
            outputsN = model(**clean_inputs, output_hidden_states=True, return_dict=True)'''


        #loss = get_grill_wass(outputs, outputsN, startPos, endPos)

        gated_hidden = layer_inputs["down_in"]  # must exist now

        loss = getMeanAlignmentLossWithTopRightSingularVector(gated_hidden, topkRightSingularVector)

        attack_loss = loss  # maximize loss

        opt.zero_grad(set_to_none=True)
        attack_loss.backward()
        opt.step()

        with torch.no_grad():
            delta.data.clamp_(-epsilon, epsilon)

        lv = float(loss.item())
        if (step + 1) % 10 == 0 or step == 0:
            print(f"[step {step+1}/{num_steps}] loss={lv:.6f}")

        '''if lv < best_loss:
            best_loss = lv'''
        best_delta = delta.detach().clone()
        losses_list.append(lv)
        np.save(save_conv_path, np.array(losses_list, dtype=np.float32))

        del outputs, loss, attack_loss, pv_adv
        '''if device.type == "cuda":
            torch.cuda.empty_cache()'''

    with torch.no_grad():
        x_adv01_final = (x_orig01 + best_delta).clamp(0.0, 1.0)
        x_adv01_final = torch.max(torch.min(x_adv01_final, x_orig01 + epsilon), x_orig01 - epsilon).clamp(0.0, 1.0)

    return x_adv01_final, best_delta

# ----------------------------
# MAIN
# ----------------------------
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

    parser.add_argument("--AlignLayer", type=int, default=1,
                        help="values taken are 0 to 23 for vis and 0 to 33 for lan ")
    parser.add_argument("--whichLayMod", type=str, default="vis",
                        help="values taken are vis and lan")
    parser.add_argument("--whichMLP", type=str, default="fc1",
                        help="values taken : down_proj, up_proj, fc1, fc2, out_proj")
    parser.add_argument("--towardsNull", type=float, default=1,
                        help="values 0 to 1")



    args = parser.parse_args()

    attck_type = args.attck_type
    epsilon = float(args.desired_norm_l_inf)
    lr = float(args.learningRate)
    num_steps = int(args.num_steps)
    attackSample = str(args.attackSample)
    
    AlignLayer = int(args.AlignLayer)
    towardsNull = float(args.towardsNull)
    whichLayMod = str(args.whichLayMod)
    whichMLP = str(args.whichMLP)
    #AlignLayer = 22
    #whichLayMod = "vis"
    #whichMLP = "fc1"
    #whichMLP = "out_proj"


    # ---- CONFIG
    MODEL_PATH = "/home/luser/LLaVA/llava-1.5-7b-hf"
    IMAGE_PATH = f"llava_attack/dataSamplesForQuant/{attackSample}.JPEG"  # adjust ext if needed
    QUESTION = "What is shown in this image?"
    MAX_NEW_TOKENS = 128

    os.makedirs("llava_attack/outputsStorage", exist_ok=True)
    os.makedirs(f"llava_attack/outputsStorage/advOutputs/{attackSample}", exist_ok=True)
    os.makedirs(f"llava_attack/outputsStorage/convergence/{attackSample}", exist_ok=True)

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


    #down_proj = model.language_model.model.layers[AlignLayer].mlp.down_proj
    #hook_handle = down_proj.register_forward_pre_hook(down_proj_pre_hook)


    #up_proj = model.language_model.model.layers[AlignLayer].mlp.up_proj
    #hook_handle = up_proj.register_forward_pre_hook(down_proj_pre_hook)

    # vision trial

    #fc1 = model.vision_tower.vision_model.encoder.layers[AlignLayer].mlp.fc2
    #hook_handle = fc1.register_forward_pre_hook(down_proj_pre_hook)



    if whichMLP == "down_proj" and whichLayMod == "lan":
        down_proj = model.language_model.model.layers[AlignLayer].mlp.down_proj
        hook_handle = down_proj.register_forward_pre_hook(down_proj_pre_hook)

    if whichMLP == "up_proj" and whichLayMod == "lan":
        up_proj = model.language_model.model.layers[AlignLayer].mlp.up_proj
        hook_handle = up_proj.register_forward_pre_hook(down_proj_pre_hook)

    if whichMLP == "fc1" and whichLayMod == "vis":
        fc1 = model.vision_tower.vision_model.encoder.layers[AlignLayer].mlp.fc1
        hook_handle = fc1.register_forward_pre_hook(down_proj_pre_hook)


    if whichMLP == "fc2" and whichLayMod == "vis":
        fc2 = model.vision_tower.vision_model.encoder.layers[AlignLayer].mlp.fc2
        hook_handle = fc2.register_forward_pre_hook(down_proj_pre_hook)


    if whichMLP == "out_proj" and whichLayMod == "vis":
        out_proj = model.vision_tower.vision_model.encoder.layers[AlignLayer].self_attn.out_proj
        hook_handle = out_proj.register_forward_pre_hook(down_proj_pre_hook)


    # Load original image (keep original resolution)
    pil = Image.open(IMAGE_PATH).convert("RGB")
    x_orig01 = pil_to_tensor01(pil).to(device)

    # Build template inputs ONCE (fixed image placeholder tokens)
    template_inputs = build_template_inputs(tokenizer, QUESTION, device)

    # Clean output: preprocess original (differentiable) then generate
    pv_clean = llava_preprocess_differentiable(x_orig01, image_processor)

    print("\n=== CLEAN OUTPUT ===")
    clean_text = run_generation_with_pixel_values(model, tokenizer, template_inputs, pv_clean, max_new_tokens=MAX_NEW_TOKENS)
    print(clean_text)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Attack
    conv_path = (
        f"llava_attack/outputsStorage/convergence/{attackSample}/"
        f"llava_ORIG_attack_{attck_type}_lr_{lr}_eps_{epsilon}_"
        f"num_steps_{num_steps}_AlignLayer_{AlignLayer}_towardsNull_{towardsNull}_whichLayMod_{whichLayMod}_whichMLP_{whichMLP}_.npy"
    )

    x_adv01, best_pert = adam_attack_original_space(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        template_inputs=template_inputs,
        x_orig01=x_orig01,
        attck_type=attck_type,
        num_steps=num_steps,
        lr=lr,
        epsilon=epsilon,
        device=device,
        save_conv_path=conv_path,
        AlignLayer = AlignLayer,
        towardsNull = towardsNull,
        whichLayMod = whichLayMod,
        whichMLP = whichMLP
    )

    # Save ORIGINAL-resolution adversarial image
    adv_img_path = (
        f"llava_attack/outputsStorage/advOutputs/{attackSample}/"
        f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
        f"num_steps_{num_steps}_AlignLayer_{AlignLayer}_towardsNull_{towardsNull}_whichLayMod_{whichLayMod}_whichMLP_{whichMLP}_.png"
    )

    adv_noise_path = (
        f"llava_attack/outputsStorage/advOutputs/{attackSample}/"
        f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
        f"num_steps_{num_steps}_AlignLayer_{AlignLayer}_towardsNull_{towardsNull}_whichLayMod_{whichLayMod}_whichMLP_{whichMLP}_.pt"
    )

    tensor01_to_pil(x_adv01).save(adv_img_path)
    print(f"\nSaved ORIGINAL-resolution adversarial image to: {adv_img_path}")
    #print("best_pert.shape", best_pert.shape)
    torch.save(best_pert.detach().cpu(), adv_noise_path)
    # Adv output
    pv_adv = llava_preprocess_differentiable(x_adv01, image_processor)
    print("\n=== ADVERSARIAL OUTPUT ===")
    adv_text = run_generation_with_pixel_values(model, tokenizer, template_inputs, pv_adv, max_new_tokens=MAX_NEW_TOKENS)
    print(adv_text)

    # Save text outputs
    cleanOutTxt = f"llava_attack/outputsStorage/advOutputs/{attackSample}/cleanOutput.txt"
    with open(cleanOutTxt, "w") as f:
        f.write(clean_text + "\n\n")

    advOutTxt = (
        f"llava_attack/outputsStorage/advOutputs/{attackSample}/"
        f"advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
        f"num_steps_{num_steps}_AlignLayer_{AlignLayer}_towardsNull_{towardsNull}_whichLayMod_{whichLayMod}_whichMLP_{whichMLP}_.txt"
    )
    with open(advOutTxt, "w") as f:
        f.write(adv_text + "\n")

    print(f"\nSaved outputs to: {advOutTxt}")
    print(f"Saved convergence to: {conv_path}")

if __name__ == "__main__":
    main()