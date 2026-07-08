



'''

export CUDA_VISIBLE_DEVICES=3
conda activate gemma3
cd interpretAttacks
python gemma_attack/gemma3ConditioningNew.py 


'''

import os
import sys
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

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

plotFinishedThings = False

def conv_singular_values_via_fft(weight, grid=64, device=None):
    # weight: (out_ch, in_ch, kh, kw)
    if device is None:
        device = weight.device

    # FFT doesn't support bfloat16 -> use float32 for analysis
    W = weight.detach().to(device=device, dtype=torch.float32)

    out_ch, in_ch, kh, kw = W.shape

    # zero-pad kernel to grid size
    if grid < kh or grid < kw:
        raise ValueError(f"grid={grid} must be >= kernel size ({kh},{kw})")

    Wpad = F.pad(W, (0, grid - kw, 0, grid - kh))  # (out_ch, in_ch, grid, grid)

    # FFT over spatial dims -> complex64/complex128 depending on input dtype (here complex64)
    Wfft = torch.fft.fft2(Wpad, dim=(-2, -1))  # (out_ch, in_ch, grid, grid), complex

    s_vals = []
    Wfft_np = Wfft.detach().cpu().numpy()  # complex numpy array
    for u in range(grid):
        for v in range(grid):
            H = Wfft_np[:, :, u, v]  # (out_ch, in_ch) complex
            sv = np.linalg.svd(H, compute_uv=False)
            s_vals.append(sv)

    return np.concatenate(s_vals)

def main():
    MODEL_PATH = "../illcond/gemma_attack/Gemma3-4b"

    os.makedirs("outputsStorage", exist_ok=True)
    os.makedirs("outputsStorage/convergence", exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"device={device}, dtype={dtype}")

    print("Loading processor...")

    print("Loading model...")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    model.config.use_cache = False

    print("\n=== MODEL PARAMETERS (name → shape) ===")

    if plotFinishedThings:
        # Identifying important information
        pos = model.vision_tower.vision_model.embeddings.position_embedding.weight
        print("positional embedding pos.shape", pos.shape)
        numTockens = pos.shape[0]
        NumPatchesEachSide = int(numTockens**0.5)
        eachPatchSideLen = model.vision_tower.vision_model.embeddings.patch_embedding.weight.shape[-1]
        ImageInputSideLen = eachPatchSideLen * NumPatchesEachSide

        print("ImageInputSideLen", ImageInputSideLen)
        print("eachPatchSideLen", eachPatchSideLen)
        print("NumPatchesEachSide", NumPatchesEachSide)
        print("numTockens", numTockens)



        
        print(model.language_model.config)
        print(model.language_model.config.num_attention_heads)
        print(model.language_model.config.num_key_value_heads)
        print(model.language_model.config.hidden_size)

        d_modelT = model.language_model.config.hidden_size
        num_headsT = model.language_model.config.num_attention_heads




        #--------------- Singular value of all Queries parameters -------------------------
        query_s_per_head_all_layers = []
        for lay in range(len(model.language_model.model.layers)):
            for name, param in model.language_model.model.layers[lay].named_parameters():
                print(f"{name:60s} {tuple(param.shape)}")
                if len(param.shape)==2 and 'q_proj' in name:
                    d_headT = param.shape[0] // num_headsT
                    param_heads = param.view(num_headsT, d_headT, d_modelT)
                    query_s_per_head = []
                    for h in range(num_headsT):
                        Wh = param_heads[h]            # shape (d_head, d_model)
                        U, S, Vh = torch.linalg.svd(Wh.to(torch.float32) )
                        query_s_per_head.append(S)
                    query_s_per_head = torch.stack(query_s_per_head, 0)
            query_s_per_head_all_layers.append(query_s_per_head)

        
        query_s_per_head_all_layers = torch.stack(query_s_per_head_all_layers, 0)
        print("query_s_per_head.shape", query_s_per_head_all_layers.shape)
        #--------------- Singular value of all Queries parameters -------------------------


        #--------------- Singular value of all Key parameters -------------------------
        key_s_per_head_all_layers = []
        for lay in range(len(model.language_model.model.layers)):
            for name, param in model.language_model.model.layers[lay].named_parameters():
                #print(f"{name:60s} {tuple(param.shape)}")
                if len(param.shape)==2 and 'k_proj' in name:
                    d_headT = param.shape[0] // num_headsT
                    param_heads = param.view(num_headsT, d_headT, d_modelT)
                    key_s_per_head = []
                    for h in range(num_headsT):
                        Wh = param_heads[h]            # shape (d_head, d_model)
                        U, S, Vh = torch.linalg.svd(Wh.to(torch.float32) )
                        key_s_per_head.append(S)
                    key_s_per_head = torch.stack(key_s_per_head, 0)
            key_s_per_head_all_layers.append(key_s_per_head)
            #print("layer  is  over")
            #print()
        
        key_s_per_head_all_layers = torch.stack(key_s_per_head_all_layers, 0)
        print("key_s_per_head_all_layers.shape", key_s_per_head_all_layers.shape)
        #--------------- Singular value of all Key parameters -------------------------



        #--------------- Singular value of all Key parameters -------------------------
        value_s_per_head_all_layers = []
        for lay in range(len(model.language_model.model.layers)):
            for name, param in model.language_model.model.layers[lay].named_parameters():
                #print(f"{name:60s} {tuple(param.shape)}")
                if len(param.shape)==2 and 'v_proj' in name:
                    d_headT = param.shape[0] // num_headsT
                    param_heads = param.view(num_headsT, d_headT, d_modelT)
                    value_s_per_head = []
                    for h in range(num_headsT):
                        Wh = param_heads[h]            # shape (d_head, d_model)
                        U, S, Vh = torch.linalg.svd(Wh.to(torch.float32) )
                        value_s_per_head.append(S)
                    value_s_per_head = torch.stack(value_s_per_head, 0)
            value_s_per_head_all_layers.append(value_s_per_head)
            #print("layer  is  over")
            #print()
        
        value_s_per_head_all_layers = torch.stack(value_s_per_head_all_layers, 0)
        print("value_s_per_head_all_layers.shape", value_s_per_head_all_layers.shape)
        #--------------- Singular value of all Key parameters -------------------------

        #--------------- Singular value of all out projections -------------------------
        out_proj_s_all_layers = []
        for name, param in model.language_model.model.layers.named_parameters():
            if len(param.shape)==2 and 'o_proj' in name:
                U, S, Vh = torch.linalg.svd(param.to(torch.float32))
                out_proj_s_all_layers.append(S)
        out_proj_s_all_layers = torch.stack(out_proj_s_all_layers, 0)
        print("out_proj_s_all_layers.shape", out_proj_s_all_layers.shape)

        #--------------- Singular value of all out projections -------------------------

        #--------------- Singular value of all out projections -------------------------
        mlpFc1_proj_s_all_layers = []
        for name, param in model.language_model.model.layers.named_parameters():
            if len(param.shape)==2 and 'mlp.gate_proj' in name:
                U, S, Vh = torch.linalg.svd(param.to(torch.float32))

                mlpFc1_proj_s_all_layers.append(S)
        mlpFc1_proj_s_all_layers = torch.stack(mlpFc1_proj_s_all_layers, 0)
        print("mlpFc1_proj_s_all_layers.shape", mlpFc1_proj_s_all_layers.shape)
        #--------------- Singular value of all out projections -------------------------

        #--------------- Singular value of all out projections -------------------------
        mlpFc2_proj_s_all_layers = []
        for name, param in model.language_model.model.layers.named_parameters():
            if len(param.shape)==2 and 'mlp.up_proj' in name:
                #print("param.shape", param.shape)
                U, S, Vh = torch.linalg.svd(param.to(torch.float32))
                mlpFc2_proj_s_all_layers.append(S)
        mlpFc2_proj_s_all_layers = torch.stack(mlpFc2_proj_s_all_layers, 0)
        print("mlpFc2_proj_s_all_layers.shape", mlpFc2_proj_s_all_layers.shape)
        #--------------- Singular value of all out projections -------------------------


        #--------------- Singular value of all out projections -------------------------
        mlpFc3_proj_s_all_layers = []
        for name, param in model.language_model.model.layers.named_parameters():
            if len(param.shape)==2 and 'mlp.down_proj' in name:
                #print("param.shape", param.shape)
                U, S, Vh = torch.linalg.svd(param.to(torch.float32))
                mlpFc3_proj_s_all_layers.append(S)
        mlpFc3_proj_s_all_layers = torch.stack(mlpFc3_proj_s_all_layers, 0)
        print("mlpFc3_proj_s_all_layers.shape", mlpFc3_proj_s_all_layers.shape)
        #--------------- Singular value of all out projections -------------------------

        #--------------plotting maximum singular values-----------------


        #-------------------------------------------------QuerySingularValuesPlots--------------------------------------------------
        print("query executing ?")
        max_vals = (query_s_per_head_all_layers.max(dim=-1, keepdim=True).values).squeeze(-1)
        Z = max_vals.detach().cpu().numpy()
        layers = np.arange(1, Z.shape[0] + 1) 
        heads  = np.arange(1, Z.shape[1] + 1) 
        X, Y = np.meshgrid(heads, layers)       
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True, shade=True)
        #fig.colorbar(surf, ax=ax, shrink=0.5, aspect=12, pad=0.1, label="Largest singular value")
        ax.set_xlabel("Head (1-16)")
        ax.set_ylabel("Layer (1-27)")
        ax.set_zlabel("Max singular value")
        #ax.set_title("Largest singular values per head across layers")
        ax.view_init(elev=30, azim=60)
        plt.tight_layout()
        plt.savefig("/data1/chethan/interpretAttacks/gemma_attack/AllPlots/singularValuesPlot/queryTextMaxSingularPlot.png", dpi=300)
        plt.show()
        plt.close()

        min_vals = (query_s_per_head_all_layers.min(dim=-1, keepdim=True).values.squeeze(-1))
        Z = min_vals.detach().cpu().numpy()
        eps = 1e-6 
        Z_log = np.log10(Z + eps)
        layers = np.arange(1, Z.shape[0] + 1)   # 1..27
        heads  = np.arange(1, Z.shape[1] + 1)   # 1..16
        X, Y = np.meshgrid(heads, layers)       # X=head, Y=layer
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, Z_log, cmap="viridis",linewidth=0, antialiased=True, shade=True)
        z_ticks = np.arange(np.floor(Z_log.min()),np.ceil(Z_log.max()) + 1)
        ax.set_zticks(z_ticks)
        ax.set_zticklabels([rf"$10^{{{int(t)}}}$" for t in z_ticks])
        ax.set_xlabel("Head (1-16)")
        ax.set_ylabel("Layer (1-27)")
        ax.set_zlabel("Min singular value")
        ax.view_init(elev=30, azim=-60)
        plt.tight_layout()
        plt.savefig("/data1/chethan/interpretAttacks/gemma_attack/AllPlots/singularValuesPlot/queryTextMinSingularPlot_log.png", dpi=300)
        plt.show()
        plt.close()

        max_vals = (query_s_per_head_all_layers.max(dim=-1, keepdim=True).values).squeeze(-1)
        min_vals = (query_s_per_head_all_layers.min(dim=-1, keepdim=True).values.squeeze(-1))
        conditionNum = max_vals/min_vals
        Z = conditionNum.detach().cpu().numpy()
        eps = 1e-6 
        Z_log = np.log10(Z + eps)
        layers = np.arange(1, Z.shape[0] + 1)   # 1..27
        heads  = np.arange(1, Z.shape[1] + 1)   # 1..16
        X, Y = np.meshgrid(heads, layers)       # X=head, Y=layer
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, Z_log, cmap="viridis",linewidth=0, antialiased=True, shade=True)
        z_ticks = np.arange(np.floor(Z_log.min()),np.ceil(Z_log.max()) + 1)
        ax.set_zticks(z_ticks)
        ax.set_zticklabels([rf"$10^{{{int(t)}}}$" for t in z_ticks])
        ax.set_xlabel("Head (1-16)")
        ax.set_ylabel("Layer (1-27)")
        ax.set_zlabel("Condition number")
        ax.view_init(elev=30, azim=60)
        plt.tight_layout()
        plt.savefig("/data1/chethan/interpretAttacks/gemma_attack/AllPlots/singularValuesPlot/queryTextConditionNumber.png", dpi=300)
        plt.show()
        plt.close()
        #-------------------------------------------------QuerySingularValuesPlots--------------------------------------------------


        #-------------------------------------------------KeySingularValuesPlots--------------------------------------------------
        max_vals = (key_s_per_head_all_layers.max(dim=-1, keepdim=True).values).squeeze(-1)
        Z = max_vals.detach().cpu().numpy()
        layers = np.arange(1, Z.shape[0] + 1) 
        heads  = np.arange(1, Z.shape[1] + 1) 
        X, Y = np.meshgrid(heads, layers)       
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True, shade=True)
        #fig.colorbar(surf, ax=ax, shrink=0.5, aspect=12, pad=0.1, label="Largest singular value")
        ax.set_xlabel("Head (1-16)")
        ax.set_ylabel("Layer (1-27)")
        ax.set_zlabel("Max singular value")
        #ax.set_title("Largest singular values per head across layers")
        ax.view_init(elev=30, azim=60)
        plt.tight_layout()
        plt.savefig("/data1/chethan/interpretAttacks/gemma_attack/AllPlots/singularValuesPlot/keyTextMaxSingularPlot.png", dpi=300)
        plt.show()
        plt.close()

        min_vals = (key_s_per_head_all_layers.min(dim=-1, keepdim=True).values.squeeze(-1))
        Z = min_vals.detach().cpu().numpy()
        eps = 1e-6 
        Z_log = np.log10(Z + eps)
        layers = np.arange(1, Z.shape[0] + 1)   # 1..27
        heads  = np.arange(1, Z.shape[1] + 1)   # 1..16
        X, Y = np.meshgrid(heads, layers)       # X=head, Y=layer
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, Z_log, cmap="viridis",linewidth=0, antialiased=True, shade=True)
        z_ticks = np.arange(np.floor(Z_log.min()),np.ceil(Z_log.max()) + 1)
        ax.set_zticks(z_ticks)
        ax.set_zticklabels([rf"$10^{{{int(t)}}}$" for t in z_ticks])
        ax.set_xlabel("Head (1-16)")
        ax.set_ylabel("Layer (1-27)")
        ax.set_zlabel("Min singular value")
        ax.view_init(elev=30, azim=-60)
        plt.tight_layout()
        plt.savefig("/data1/chethan/interpretAttacks/gemma_attack/AllPlots/singularValuesPlot/keyTextMinSingularPlot_log.png", dpi=300)
        plt.show()
        plt.close()

        max_vals = (key_s_per_head_all_layers.max(dim=-1, keepdim=True).values).squeeze(-1)
        min_vals = (key_s_per_head_all_layers.min(dim=-1, keepdim=True).values.squeeze(-1))
        conditionNum = max_vals/min_vals
        Z = conditionNum.detach().cpu().numpy()
        eps = 1e-6 
        Z_log = np.log10(Z + eps)
        layers = np.arange(1, Z.shape[0] + 1)   # 1..27
        heads  = np.arange(1, Z.shape[1] + 1)   # 1..16
        X, Y = np.meshgrid(heads, layers)       # X=head, Y=layer
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, Z_log, cmap="viridis",linewidth=0, antialiased=True, shade=True)
        z_ticks = np.arange(np.floor(Z_log.min()),np.ceil(Z_log.max()) + 1)
        ax.set_zticks(z_ticks)
        ax.set_zticklabels([rf"$10^{{{int(t)}}}$" for t in z_ticks])
        ax.set_xlabel("Head (1-16)")
        ax.set_ylabel("Layer (1-27)")
        ax.set_zlabel("Condition number")
        ax.view_init(elev=30, azim=60)
        plt.tight_layout()
        plt.savefig("/data1/chethan/interpretAttacks/gemma_attack/AllPlots/singularValuesPlot/keyTextConditionNumber.png", dpi=300)
        plt.show()
        plt.close()
        #-------------------------------------------------KeySingularValuesPlots--------------------------------------------------


        #-------------------------------------------------ValueSingularValuesPlots--------------------------------------------------
        max_vals = (value_s_per_head_all_layers.max(dim=-1, keepdim=True).values).squeeze(-1)
        Z = max_vals.detach().cpu().numpy()
        layers = np.arange(1, Z.shape[0] + 1) 
        heads  = np.arange(1, Z.shape[1] + 1) 
        X, Y = np.meshgrid(heads, layers)       
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True, shade=True)
        #fig.colorbar(surf, ax=ax, shrink=0.5, aspect=12, pad=0.1, label="Largest singular value")
        ax.set_xlabel("Head (1-16)")
        ax.set_ylabel("Layer (1-27)")
        ax.set_zlabel("Max singular value")
        #ax.set_title("Largest singular values per head across layers")
        ax.view_init(elev=30, azim=60)
        plt.tight_layout()
        plt.savefig("/data1/chethan/interpretAttacks/gemma_attack/AllPlots/singularValuesPlot/ValueTextMaxSingularPlot.png", dpi=300)
        plt.show()
        plt.close()

        min_vals = (value_s_per_head_all_layers.min(dim=-1, keepdim=True).values.squeeze(-1))
        Z = min_vals.detach().cpu().numpy()
        eps = 1e-6 
        Z_log = np.log10(Z + eps)
        layers = np.arange(1, Z.shape[0] + 1)   # 1..27
        heads  = np.arange(1, Z.shape[1] + 1)   # 1..16
        X, Y = np.meshgrid(heads, layers)       # X=head, Y=layer
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, Z_log, cmap="viridis",linewidth=0, antialiased=True, shade=True)
        z_ticks = np.arange(np.floor(Z_log.min()),np.ceil(Z_log.max()) + 1)
        ax.set_zticks(z_ticks)
        ax.set_zticklabels([rf"$10^{{{int(t)}}}$" for t in z_ticks])
        ax.set_xlabel("Head (1-16)")
        ax.set_ylabel("Layer (1-27)")
        ax.set_zlabel("Min singular value")
        ax.view_init(elev=30, azim=-60)
        plt.tight_layout()
        plt.savefig("/data1/chethan/interpretAttacks/gemma_attack/AllPlots/singularValuesPlot/ValueTextMinSingularPlot_log.png", dpi=300)
        plt.show()
        plt.close()

        max_vals = (value_s_per_head_all_layers.max(dim=-1, keepdim=True).values).squeeze(-1)
        min_vals = (value_s_per_head_all_layers.min(dim=-1, keepdim=True).values.squeeze(-1))
        conditionNum = max_vals/min_vals
        Z = conditionNum.detach().cpu().numpy()
        eps = 1e-6 
        Z_log = np.log10(Z + eps)
        layers = np.arange(1, Z.shape[0] + 1)   # 1..27
        heads  = np.arange(1, Z.shape[1] + 1)   # 1..16
        X, Y = np.meshgrid(heads, layers)       # X=head, Y=layer
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, Z_log, cmap="viridis",linewidth=0, antialiased=True, shade=True)
        z_ticks = np.arange(np.floor(Z_log.min()),np.ceil(Z_log.max()) + 1)
        ax.set_zticks(z_ticks)
        ax.set_zticklabels([rf"$10^{{{int(t)}}}$" for t in z_ticks])
        ax.set_xlabel("Head (1-16)")
        ax.set_ylabel("Layer (1-27)")
        ax.set_zlabel("Condition number")
        ax.view_init(elev=30, azim=60)
        plt.tight_layout()
        plt.savefig("/data1/chethan/interpretAttacks/gemma_attack/AllPlots/singularValuesPlot/ValueTextConditionNumber.png", dpi=300)
        plt.show()
        plt.close()
        #-------------------------------------------------ValueSingularValuesPlots--------------------------------------------------


        #-------------------------------------------------outProjSingularValuesPlots--------------------------------------------------


        min_valsOutProj = (out_proj_s_all_layers.min(dim=-1, keepdim=True).values).squeeze(-1)
        valuesOutProj = min_valsOutProj.detach().cpu().numpy()

        min_valsFc1 = (mlpFc1_proj_s_all_layers.min(dim=-1, keepdim=True).values).squeeze(-1)
        valuesFc1 = min_valsFc1.detach().cpu().numpy()

        min_valsFc2 = (mlpFc2_proj_s_all_layers.min(dim=-1, keepdim=True).values).squeeze(-1)
        valuesFc2 = min_valsFc2.detach().cpu().numpy()

        min_valsFc3 = (mlpFc3_proj_s_all_layers.min(dim=-1, keepdim=True).values).squeeze(-1)
        valuesFc3 = min_valsFc3.detach().cpu().numpy()

        n_layers = len(valuesOutProj)
        y = np.arange(1, n_layers+1)   # layer positions
        height = 0.18             # thickness of each bar
        plt.figure(figsize=(4,6))
        plt.barh(y - 1.5*height, valuesOutProj, height, label='AOP', color='tab:blue')
        plt.barh(y - 0.5*height, valuesFc1,    height, label='FC1', color='tab:orange')
        plt.barh(y + 0.5*height, valuesFc2,    height, label='FC2', color='tab:green')
        plt.barh(y + 1.5*height, valuesFc3,    height, label='FC3', color='tab:red')


        plt.xscale('log')   # log scale for singular values (since now horizontal)

        plt.ylabel("Layer Index")
        plt.xlabel("Min. Singular Value (log scale)")
        plt.yticks(y)
        #plt.legend()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig("/data1/chethan/interpretAttacks/gemma_attack/AllPlots/singularValuesPlot/outProjTextFc1Fc2MinSingularPlot_horizontal.png", dpi=300)

        plt.show()
        plt.close()


        max_valsOutProj = (out_proj_s_all_layers.max(dim=-1, keepdim=True).values).squeeze(-1)
        valuesOutProj = max_valsOutProj.detach().cpu().numpy()

        max_valsFc1 = (mlpFc1_proj_s_all_layers.max(dim=-1, keepdim=True).values).squeeze(-1)
        valuesFc1 = max_valsFc1.detach().cpu().numpy()

        max_valsFc2 = (mlpFc2_proj_s_all_layers.max(dim=-1, keepdim=True).values).squeeze(-1)
        valuesFc2 = max_valsFc2.detach().cpu().numpy()

        max_valsFc3 = (mlpFc3_proj_s_all_layers.max(dim=-1, keepdim=True).values).squeeze(-1)
        valuesFc3 = max_valsFc3.detach().cpu().numpy()

        n_layers = len(valuesOutProj)
        y = np.arange(1, n_layers+1)   # layer positions
        height = 0.25             # thickness of each bar
        plt.figure(figsize=(4,6))
        plt.barh(y - 1.5*height, valuesOutProj, height, label='AOP', color='tab:blue')
        plt.barh(y - 0.5*height, valuesFc1,    height, label='FC1', color='tab:orange')
        plt.barh(y + 0.5*height, valuesFc2,    height, label='FC2', color='tab:green')
        plt.barh(y + 1.5*height, valuesFc3,    height, label='FC3', color='tab:red')


        #plt.xscale('log')   # log scale for singular values (since now horizontal)

        plt.ylabel("Layer Index")
        plt.xlabel("Max. Singular Value ")
        plt.yticks(y)
        #plt.legend()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig("/data1/chethan/interpretAttacks/gemma_attack/AllPlots/singularValuesPlot/outProjTextFc1Fc2MaxSingularPlot_horizontal.png", dpi=300)

        plt.show()
        plt.close()



        valuesOutProj = (max_valsOutProj/min_valsOutProj).detach().cpu().numpy()
        valuesFc1 = (max_valsFc1/min_valsFc1).detach().cpu().numpy()
        valuesFc2 = (max_valsFc2/min_valsFc2).detach().cpu().numpy()
        valuesFc3 = (max_valsFc3/min_valsFc3).detach().cpu().numpy()


        n_layers = len(valuesOutProj)
        y = np.arange(1, n_layers+1)   # layer positions
        height = 0.18             # thickness of each bar
        plt.figure(figsize=(4,6))
        plt.barh(y - 1.5*height, valuesOutProj, height, label='AOP', color='tab:blue')
        plt.barh(y - 0.5*height, valuesFc1,    height, label='FC1', color='tab:orange')
        plt.barh(y + 0.5*height, valuesFc2,    height, label='FC2', color='tab:green')
        plt.barh(y + 1.5*height, valuesFc3,    height, label='FC3', color='tab:red')


        plt.xscale('log')   # log scale for singular values (since now horizontal)

        plt.ylabel("Layer Index")
        plt.xlabel("Condition number (log scale)")
        plt.yticks(y)
        #plt.legend()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig("/data1/chethan/interpretAttacks/gemma_attack/AllPlots/singularValuesPlot/outProjTextFc1Fc2ConditionNumberPlot_horizontal.png", dpi=300)

        plt.show()
        plt.close()


    '''for name, param in model.vision_tower.vision_model.named_parameters():
        print(f"{name:60s} {tuple(param.shape)}")'''


    multiModalProj = model.multi_modal_projector.mm_input_projection_weight
    U, S, Vh = torch.linalg.svd(multiModalProj.to(torch.float32))
    maxSingularMultiModalProj = S.max()
    minSingularMultiModalProj = S.min()
    ConditionNumMultiModalProj = S.max()/S.min()

    print("maxSingularMultiModalProj", maxSingularMultiModalProj)
    print("minSingularMultiModalProj", minSingularMultiModalProj)
    print("ConditionNumMultiModalProj", ConditionNumMultiModalProj)

    textEmbedder = model.language_model.model.embed_tokens.weight
    print("textEmbedder.shape", textEmbedder.shape)
    # U, S, Vh = torch.linalg.svd(textEmbedder.to(torch.float32)) # this is way too large of a matrix to do SVD. And since it acts more like a look table, it doesn't make sense to do SVD


    print("multiModalProj.shape", multiModalProj.shape)
    print()
    print("full")
    for name, param in model.named_parameters():
        print(f"{name:60s} {tuple(param.shape)}")
        #U, S, Vh = torch.linalg.svd(param.to(torch.float32))

if __name__ == "__main__":
    main()

