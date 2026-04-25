


'''



export CUDA_VISIBLE_DEVICES=7
conda deactivate
cd interpretAttacks/
conda activate vlmAttack
export PYTHONNOUSERSITE=1
for ATTACK_SAMPLE in $(seq 100 200); do
    python qwen/QwenUntargted_KSA.py --attck_type saa --desired_norm_l_inf 0.05 --learningRate 0.001 --numSteps 1000 --attackSample $ATTACK_SAMPLE --towardsNull 0 --AlignLayer 1
done

export CUDA_VISIBLE_DEVICES=6
conda deactivate
cd interpretAttacks/
conda activate vlmAttack
export PYTHONNOUSERSITE=1
for ATTACK_SAMPLE in $(seq 100 200); do
    python qwen/QwenUntargted_KSA.py --attck_type saa --desired_norm_l_inf 0.05 --learningRate 0.001 --numSteps 1000 --attackSample $ATTACK_SAMPLE --towardsNull 0.5 --AlignLayer 1
done



48 0.02

'''

#!/usr/bin/env python
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import torch.nn as nn
import torch.nn.functional as F


import random
import os

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"   # or ":4096:8"

import torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Disable flash / mem-efficient SDP attention (common nondeterminism source)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


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


import argparse

parser = argparse.ArgumentParser(description='Adversarial attack on VLM')
parser.add_argument('--attck_type', type=str, default="lip", help='Segment index')
parser.add_argument('--desired_norm_l_inf', type=float, default="lip", help='Segment index')
parser.add_argument('--learningRate', type=float, default="lip", help='Segment index')
parser.add_argument('--numSteps', type=int, default="num steps", help='num steps')
parser.add_argument("--attackSample", type=str, default="nature",
                help="which sample")
parser.add_argument('--towardsNull', type=float, default="0.5", help='how much towards null')
parser.add_argument('--AlignLayer', type=int, default="1", help='AlignLayer')

args = parser.parse_args()


attck_type = args.attck_type
desired_norm_l_inf = args.desired_norm_l_inf
learningRate = float(args.learningRate)
numSteps = args.numSteps
attackSample = str(args.attackSample)
towardsNull = float(args.towardsNull)
AlignLayer = int(args.AlignLayer)
# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------


MODEL_PATH = "../illcond/QwenAttack/Qwen2.5-VL-7B-Instruct"

#IMAGE_PATH = "/home/luser/vlm_learn/dataSamples/dogs68.jpg"

IMAGE_PATH = f"qwen/dataSamplesForQuant/{attackSample}.JPEG"

#AlignLayer = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16  # you can also try torch.float16

criterion = nn.MSELoss()


os.makedirs("qwen/outputsStorageImagenet", exist_ok=True)
os.makedirs(f"qwen/outputsStorageImagenet/advOutputs/{attackSample}", exist_ok=True)
os.makedirs(f"qwen/outputsStorageImagenet/convergence/{attackSample}", exist_ok=True)

conv_path = f"qwen/outputsStorageImagenet/convergence/{attackSample}/gemma_ORIG_attack_{attck_type}_lr_{learningRate}_eps_{desired_norm_l_inf}_num_steps_{numSteps}_towardsNull_{towardsNull}_AlignLayer_{AlignLayer}_.npy"
adv_img_path = f"qwen/outputsStorageImagenet/advOutputs/{attackSample}/adv_ORIG_attackType_{attck_type}_lr_{learningRate}_eps_{desired_norm_l_inf}_num_steps_{numSteps}_towardsNull_{towardsNull}_AlignLayer_{AlignLayer}_.png"
adv_noise_path = f"qwen/outputsStorageImagenet/advOutputs/{attackSample}/adv_ORIG_attackType_{attck_type}_lr_{learningRate}_eps_{desired_norm_l_inf}_num_steps_{numSteps}_towardsNull_{towardsNull}_AlignLayer_{AlignLayer}_.pt"


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


layer_inputs = {}

def down_proj_pre_hook(module, inputs):
    # inputs is a tuple; for Linear it's (hidden_tensor,)
    # hidden_tensor shape: (batch, seq_len, d_ff)
    layer_inputs["down_in"] = inputs[0]


# -------------------------------------------------------------------
# BUILD MULTIMODAL INPUTS (uses PIL internally, but outside grad path)
# -------------------------------------------------------------------
def build_inputs(processor, image_path: str, question: str):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": image_path},
                {"type": "text", "text": question},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    return inputs.to(DEVICE)


# -------------------------------------------------------------------
# CLEAN GENERATION (no gradients)
# -------------------------------------------------------------------
def run_clean_generation(model, processor, inputs, max_new_tokens: int = 128):
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    input_ids = inputs["input_ids"]
    gen_only = output_ids[:, input_ids.shape[1]:]
    texts = processor.batch_decode(
        gen_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return texts[0]


'''def getGrillLoss(outputs,outputsN):
    loss = 0
    for hiddenState, hiddenStateN in zip(outputs.hidden_states,outputsN.hidden_states):
        loss = loss + criterion(hiddenState, hiddenStateN)
    return loss * criterion(outputs.logits, outputsN.logits)'''

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

def getGrillLoss(outputs,outputsN):
    loss = 0
    for hiddenState, hiddenStateN in zip(outputs.hidden_states,outputsN.hidden_states):
        loss = loss + criterion(hiddenState, hiddenStateN)
    return loss * criterion(hiddenState, hiddenStateN)

'''def getGrillWassLoss(outputs,outputsN):
    loss = 0
    for hiddenState, hiddenStateN in zip(outputs.hidden_states,outputsN.hidden_states):
        loss = loss + wasserstein_distance(hiddenState, hiddenStateN)
    return loss * wasserstein_distance(hiddenState, hiddenStateN)'''

def getGrillWassLoss(outputs, outputsN, startPos, endPos):
    loss = 0.0
    for h, hn in zip(outputs.hidden_states[startPos:endPos], outputsN.hidden_states[startPos:endPos]):
        loss = loss + wasserstein_distance(h, hn)
    return loss #* wasserstein_distance(h, hn)

def getGrillCosLoss(outputs,outputsN):
    loss = 0
    for hiddenState, hiddenStateN in zip(outputs.hidden_states,outputsN.hidden_states):
        loss = loss + (1.0-cos(hiddenState, hiddenStateN))**2
    return loss * (1.0-cos(outputs.logits, outputsN.logits))**2

def getOALoss(outputs,outputsN):
    return criterion(outputs.logits, outputsN.logits)

def getOAWassLoss(outputs,outputsN):
    return wasserstein_distance(outputs.logits, outputsN.logits)

def getOACosLoss(outputs,outputsN):
    return (1.0-cos(outputs.logits, outputsN.logits))**2

# -------------------------------------------------------------------
# ADAM ATTACK IN PATCH (pixel_values) SPACE
# -------------------------------------------------------------------
def adam_attack_on_patches(
    model,
    original_inputs,
    num_steps: int = 3,   # keep small; bump up if it runs fine
    lr: float = 0.05,
    epsilon: float = 0.2,
):
    """
    Iterative adversarial attack on `pixel_values` using Adam.

    We optimize a noise tensor `delta` such that:
        adv_pixel = orig_pixel_values + delta
    and clamp delta to [-epsilon, epsilon] in this patch space.

    Objective: maximize language modeling loss (untargeted attack).
    """

    # Clone inputs to avoid modifying original dict
    inputs = {
        k: (v.clone().to(DEVICE) if torch.is_tensor(v) else v)
        for k, v in original_inputs.items()
    }

    if "pixel_values" not in inputs:
        raise ValueError("pixel_values not found in inputs. Check build_inputs().")

    orig_pixel_values = inputs["pixel_values"].detach()

    # Initialize noise (delta) as zeros
    #delta = torch.zeros_like(orig_pixel_values, device=DEVICE, dtype=orig_pixel_values.dtype)
    
    delta = 0.01 * torch.randn_like(orig_pixel_values, device=DEVICE, dtype=orig_pixel_values.dtype)

    delta.requires_grad_(True)

    optimizer = torch.optim.Adam([delta], lr=lr)

    # All other inputs remain fixed
    static_keys = [k for k in inputs.keys() if k != "pixel_values"]
    static_inputs = {k: inputs[k] for k in static_keys}


    model.eval()
    lossesList = [0.0]


    #paramWt = model.language_model.layers[AlignLayer].mlp.down_proj.weight 
    paramWt = model.language_model.layers[AlignLayer].mlp.up_proj.weight 

    with torch.no_grad():
        # detach so SVD is NOT part of autograd graph
        U, S, Vh = torch.linalg.svd(paramWt.detach().to(torch.float32), full_matrices=False)
        print("Vh.shape", Vh.shape)
        print("S.shape", S.shape)
        print("S.max(), S.min()", S.max(), S.min())
        print("S", S)
        print("how many less than 1 ?len(S[S<1])", len(S[S<1]))
        #towardsNull = 0.5
        absorbSize = len(S[S<1])
        #topRightSingularVector = Vh[-1].contiguous()  # constant tensor now
        if towardsNull == 0:
            bottomInd = 1
        else:
            bottomInd = int(absorbSize*towardsNull)
        print("bottomInd", bottomInd)
        topkRightSingularVector = Vh[-bottomInd:].contiguous()
        print("topkRightSingularVector.shape", topkRightSingularVector.shape)

    with torch.no_grad():
        step_inputsN = dict(static_inputs)
        step_inputsN["pixel_values"] = orig_pixel_values
        step_inputsN["labels"] = static_inputs["input_ids"]
        step_inputsN["use_cache"] = False  # critical to save memory
        outputsN = model(**step_inputsN, output_hidden_states=True)


        hiddStateLen = len(outputsN.hidden_states)
        print(" Number of hidden states is: ", hiddStateLen)

    # Build model inputs for this step
    step_inputs = dict(static_inputs)
    #step_inputs["pixel_values"] = adv_pixel
    step_inputs["labels"] = static_inputs["input_ids"]
    step_inputs["use_cache"] = False  # critical to save memory

    for step in range(num_steps):
        layer_inputs.clear() 
        # Build adversarial pixel_values within epsilon-ball
        adv_pixel = orig_pixel_values + delta
        adv_pixel = torch.max(
            torch.min(adv_pixel, orig_pixel_values + epsilon),
            orig_pixel_values - epsilon,
        )

        step_inputs["pixel_values"] = adv_pixel

        # Forward
        outputs = model(**step_inputs, output_hidden_states=True)
        gated_hidden = layer_inputs["down_in"]


        #loss = getGrillWassLoss(outputs, outputsN, startPos, endPos)
        loss = getMeanAlignmentLossWithTopRightSingularVector(gated_hidden, topkRightSingularVector)


        attack_loss = loss

        optimizer.zero_grad(set_to_none=True)
        attack_loss.backward()
        optimizer.step()

        # Project delta back into [-epsilon, epsilon] after Adam update
        with torch.no_grad():
            delta.data.clamp_(-epsilon, epsilon)

        if (step + 1) % 10 == 0 or step == 0:
            print(f"[Adam step {step+1}/{num_steps}] loss = {loss.item():.8f}")
            #print("lossesList", lossesList)
        if max(lossesList) < loss.item():
            consideredDelta = delta
            lossesList.append(float(loss.item()))
            np.save(conv_path, np.array(lossesList, dtype=np.float32))

        del outputs, loss, attack_loss, adv_pixel
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


    with torch.no_grad():
        adv_pixel_final = orig_pixel_values + consideredDelta
        adv_pixel_final = torch.max(
            torch.min(adv_pixel_final, orig_pixel_values + epsilon),
            orig_pixel_values - epsilon,
        )

    return adv_pixel_final, consideredDelta


# -------------------------------------------------------------------
# UNPATCHIFY & SAVE ADVERSARIAL IMAGE
# -------------------------------------------------------------------
def save_adv_image_from_pixel_values(
    adv_pixel_values: torch.Tensor,
    inputs,
    processor,
    save_path: str = "adv_image.png",
):

    ip = processor.image_processor

    pv = adv_pixel_values.detach().cpu().float()

    grid_thw = inputs["image_grid_thw"][0].cpu().tolist()
    grid_t, grid_h, grid_w = grid_thw  # T_grid, H_grid, W_grid

    # Hyperparameters from image processor
    patch_size = ip.patch_size           # typically 14
    temporal_patch_size = ip.temporal_patch_size  # typically 2
    merge_size = ip.merge_size           # typically 2
    channel = 3

    num_patches, D = pv.shape
    assert num_patches == grid_t * grid_h * grid_w, \
        f"num_patches mismatch: {num_patches} vs {grid_t*grid_h*grid_w}"
    assert D == channel * temporal_patch_size * patch_size * patch_size, \
        f"patch dim mismatch: {D} vs {channel*temporal_patch_size*patch_size*patch_size}"

    # Step 1: undo final flatten -> 9D tensor in permuted space
    patches = pv.view(
        grid_t,
        grid_h // merge_size,
        grid_w // merge_size,
        merge_size,
        merge_size,
        channel,
        temporal_patch_size,
        patch_size,
        patch_size,
    )

    # Step 2: invert the permute from the official preprocessing
    # Forward permute was: (0, 3, 6, 4, 7, 2, 1, 5, 8)
    # So we map back to original dims: (0, 1, 2, 3, 4, 5, 6, 7, 8)
    patches_orig = patches.permute(0, 6, 5, 1, 3, 7, 2, 4, 8).contiguous()
    # patches_orig shape:
    # (grid_t, temporal_patch_size, channel,
    #  grid_h // merge_size, merge_size, patch_size,
    #  grid_w // merge_size, merge_size, patch_size)

    # Step 3: collapse spatial/merge dims back into H', W'
    H_resized = (grid_h // merge_size) * merge_size * patch_size
    W_resized = (grid_w // merge_size) * merge_size * patch_size

    vid = patches_orig.view(
        grid_t,
        temporal_patch_size,
        channel,
        H_resized,
        W_resized,
    )
    # vid shape: (grid_t, temporal_patch_size, C, H, W)

    # Step 4: collapse temporal grid into full frames dimension
    vid = vid.view(grid_t * temporal_patch_size, channel, H_resized, W_resized)
    # For images, Qwen2-VL uses 2 identical frames; take the first one
    frame0 = vid[0]  # (C, H, W), still normalized

    # Step 5: denormalize (inverse of CLIP normalization)
    mean = torch.tensor(ip.image_mean, dtype=torch.float32).view(channel, 1, 1)
    std = torch.tensor(ip.image_std, dtype=torch.float32).view(channel, 1, 1)

    img = frame0 * std + mean  # back to [0,1] approx
    img = img.clamp(0.0, 1.0)

    # Step 6: to uint8 and save
    img_np = (img.numpy().transpose(1, 2, 0) * 255.0).round().clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)
    pil_img.save(save_path)
    print(f"Saved adversarial image (model-resolution) to: {save_path}")


# -------------------------------------------------------------------
# RUN GENERATION WITH ADVERSARIAL PATCHES
# -------------------------------------------------------------------
def run_adversarial_generation(
    model,
    processor,
    original_inputs,
    adv_pixel_values,
    max_new_tokens: int = 128,
):
    model.eval()
    adv_inputs = {
        k: (v.clone().to(DEVICE) if torch.is_tensor(v) else v)
        for k, v in original_inputs.items()
    }
    adv_inputs["pixel_values"] = adv_pixel_values

    with torch.no_grad():
        output_ids = model.generate(
            **adv_inputs,
            max_new_tokens=max_new_tokens,
        )

    input_ids = adv_inputs["input_ids"]
    gen_only = output_ids[:, input_ids.shape[1]:]
    texts = processor.batch_decode(
        gen_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return texts[0]


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    print("Loading model and processor...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=DTYPE,
        device_map=None,
    ).to(DEVICE)

    model.config.use_cache = False
    #model.gradient_checkpointing_enable()

    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    question = "What is shown in this image?"

    print("Building inputs (one-time preprocessing with PIL)...")
    inputs = build_inputs(processor, IMAGE_PATH, question)

    print("\n=== CLEAN (NO ATTACK) OUTPUT ===")
    clean_text = run_clean_generation(model, processor, inputs)
    print(clean_text)

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    #down_proj = model.language_model.layers[AlignLayer].mlp.down_proj
    #hook_handle = down_proj.register_forward_pre_hook(down_proj_pre_hook)

    up_proj = model.language_model.layers[AlignLayer].mlp.up_proj
    hook_handle = up_proj.register_forward_pre_hook(down_proj_pre_hook)
    

    print("\nRunning Adam-based iterative attack on patch tensor (pixel_values)...")
    adv_pixel_values, best_pert = adam_attack_on_patches(
        model,
        inputs,
        num_steps=numSteps,   # start small; if OK, try 4–5   10000
        lr=learningRate,
        epsilon=desired_norm_l_inf # it was 0.05 before
    )

    # --- NEW: save adversarial image reconstructed from adv_pixel_values ---
    save_adv_image_from_pixel_values(
        adv_pixel_values,
        inputs,
        processor,
        save_path=adv_img_path
    )

    torch.save(best_pert.detach().cpu(), adv_noise_path)

    print("\n=== ADVERSARIAL OUTPUT (PATCH-SPACE, ADAM ATTACK) ===")
    adv_text = run_adversarial_generation(model, processor, inputs, adv_pixel_values)
    print(adv_text)



    # Save text outputs
    cleanOutTxt = f"qwen/outputsStorageImagenet/advOutputs/{attackSample}/cleanOutput.txt"
    with open(cleanOutTxt, "w") as f:
        f.write(clean_text + "\n\n")


    advOutTxt = f"qwen/outputsStorageImagenet/advOutputs/{attackSample}/advOutput_attackType_{attck_type}_lr_{learningRate}_eps_{desired_norm_l_inf}_num_steps_{numSteps}_towardsNull_{towardsNull}_AlignLayer_{AlignLayer}_.txt"
    with open(advOutTxt, "w") as f:
        f.write(adv_text + "\n")

    print(f"\nSaved outputs to: {advOutTxt}")
    print(f"Saved convergence to: {conv_path}")




if __name__ == "__main__":
    main()