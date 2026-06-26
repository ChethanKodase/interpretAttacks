



'''

export CUDA_VISIBLE_DEVICES=5
conda activate gemma3
cd interpretAttacks
python gemma_attack/gemma3ConditioningNoAttention.py 


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


    allCondNums = []
    allSmallSingvals = []
    alllargestSingVal = []
    for name, param in model.named_parameters():
        print(f"{name:60s} {tuple(param.shape)}")

        if 'weight' in name and len(param.shape)>1 and "qkv" not in name and "q_proj" not in name and "k_proj" not in name and "v_proj" not in name :
            print("param.shape", param.shape)
            print("len(param.shape)", len(param.shape))
            W_matrix = param.view(param.shape[0], -1)  # Flatten kernels into a 2D matrix
            U, S, Vt = torch.linalg.svd(W_matrix.float(), full_matrices=False)
            condition_number = S.max() / S.min()
            print("condition_number", condition_number)
            allCondNums.append(condition_number.item())
            allSmallSingvals.append(S.min().item())
            alllargestSingVal.append(S.max().item())

    print("allCondNums", allCondNums)
    print('allSmallSingvals', allSmallSingvals)
    print('alllargestSingVal', alllargestSingVal)


    largestAmongMaxes = max(alllargestSingVal)
    print("largestAmongMaxes", largestAmongMaxes)
    ####################################### actual condition numbers

    ####################################### minimum condition numbers
    #####################----------------------------------



    vals = np.array(allSmallSingvals, dtype=float)

    fig, ax = plt.subplots(figsize=(4, 6))  # wider helps a lot
    ax.barh(np.arange(len(vals)), vals, color="blue", alpha=0.7)

    linthresh = 1e-4
    ax.set_xscale("symlog", linthresh=linthresh)

    # set x-limits to your data range (DON'T force left=0 unless you need it)
    pos = vals[vals > 0]
    xmin = pos.min()
    xmax = pos.max()
    ax.set_xlim(0, xmax * 1.05)

    # --- choose a small set of decade ticks within data range ---
    dmin = int(np.floor(np.log10(max(xmin, linthresh))))
    dmax = int(np.ceil(np.log10(xmax)))
    decade_ticks = [10.0**k for k in range(dmin, dmax + 1)]

    # include 0 tick + a few decades only
    ticks = [0.0] + decade_ticks
    ax.xaxis.set_major_locator(FixedLocator(ticks))

    def exp_formatter(x, _):
        if x == 0:
            return "0"
        return f"{x:.0e}".replace("+", "").replace("e0", "e")  # 1e-3 style

    ax.xaxis.set_major_formatter(FuncFormatter(exp_formatter))

    ax.set_ylabel("Layer index", fontsize=28)
    ax.set_xlabel(r"$\sigma_{min}$", fontsize=28)

    ax.tick_params(axis="x", labelsize=22, rotation=45)  # rotation=0 avoids the pile-up

    step = 100
    yticks = list(range(1, len(vals), step))
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=28)

    fig.tight_layout()
    fig.savefig("gemma_attack/NonAttnCondAnalysis/gemma_min_sing_valsCC.png", dpi=300)
    plt.show()
    plt.close()



    #-----------------#

    vals = np.array(alllargestSingVal, dtype=float)

    fig, ax = plt.subplots(figsize=(4, 6))  # wider helps a lot
    ax.barh(np.arange(len(vals)), vals, color="blue", alpha=0.7)

    linthresh = 1e1
    ax.set_xscale("symlog", linthresh=linthresh)

    # set x-limits to your data range (DON'T force left=0 unless you need it)
    pos = vals[vals > 0]
    xmin = pos.min()
    xmax = pos.max()
    ax.set_xlim(0, xmax * 1.05)

    # --- choose a small set of decade ticks within data range ---
    dmin = int(np.floor(np.log10(max(xmin, linthresh))))
    dmax = int(np.ceil(np.log10(xmax)))
    decade_ticks = [10.0**k for k in range(dmin, dmax + 1)]

    # include 0 tick + a few decades only
    ticks = [0.0] + decade_ticks
    ax.xaxis.set_major_locator(FixedLocator(ticks))

    def exp_formatter(x, _):
        if x == 0:
            return "0"
        return f"{x:.0e}".replace("+", "").replace("e0", "e")  # 1e-3 style

    ax.xaxis.set_major_formatter(FuncFormatter(exp_formatter))

    ax.set_ylabel("Layer index", fontsize=28)
    ax.set_xlabel(r"$\sigma_{max}$", fontsize=28)

    ax.tick_params(axis="x", labelsize=22, rotation=45)  # rotation=0 avoids the pile-up

    step = 100
    yticks = list(range(1, len(vals), step))
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=28)

    fig.tight_layout()
    fig.savefig("gemma_attack/NonAttnCondAnalysis/gemma_max_sing_valsCC.png", dpi=300)
    plt.show()




    #-----------------#


    vals = np.array(allCondNums, dtype=float)

    '''# Keep only finite values for axis scaling / ticks
    finite = np.isfinite(vals)
    vals_finite = vals[finite]

    # If you want to *plot* inf/nan as a capped value instead of dropping them:
    cap = np.nanmax(vals_finite) * 1.05
    vals_plot = vals.copy()
    vals_plot[~finite] = cap  # put inf/nan at the right edge'''

    fig, ax = plt.subplots(figsize=(4, 6))
    #ax.barh(np.arange(len(allCondNums)), allCondNums, color="blue", alpha=0.7)

    ax.barh(range(len(allCondNums)),
        allCondNums,
        color='blue',
        edgecolor='blue',
        linewidth=2)


    linthresh = 1e3
    ax.set_xscale("symlog", linthresh=linthresh)

    # Safe limits (finite only)
    xmax = np.max(vals)
    ax.set_xlim(0, xmax * 1.05)

    # decade ticks
    xmin_pos = np.min(vals[vals > 0])
    dmin = int(np.floor(np.log10(max(xmin_pos, linthresh))))
    dmax = int(np.ceil(np.log10(xmax)))
    decade_ticks = [10.0**k for k in range(dmin, dmax + 1)]

    ticks = [0.0] + decade_ticks
    ax.xaxis.set_major_locator(FixedLocator(ticks))

    def exp_formatter(x, _):
        if x == 0:
            return "0"
        return f"{x:.0e}".replace("+", "").replace("e0", "e")

    ax.xaxis.set_major_formatter(FuncFormatter(exp_formatter))

    ax.set_ylabel("Layer index", fontsize=28)
    ax.set_xlabel(r"$\kappa$", fontsize=28)
    ax.tick_params(axis="x", labelsize=22, rotation=45)

    step = 100
    yticks = list(range(1, len(vals), step))
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=28)

    fig.tight_layout()
    fig.savefig("gemma_attack/NonAttnCondAnalysis/CondNumCC.png", dpi=300)
    plt.show()



    allSmallSingvalsCheck = np.array(allSmallSingvals, dtype=float)
    alllargestSingValCheck = np.array(alllargestSingVal, dtype=float)
    allCondNumsCheck = np.array(allCondNums, dtype=float)

    total_num_nonAttn_layers = len(allSmallSingvalsCheck)

    # ----- helper -----
    def count_and_pct(values, threshold, mode):
        if mode == "gt":
            count = np.sum(values > threshold)
        elif mode == "lt":
            count = np.sum(values < threshold)
        else:
            raise ValueError("mode must be 'gt' or 'lt'")

        pct = 100.0 * count / total_num_nonAttn_layers
        return int(count), pct


    # ----- sigma_max thresholds -----
    sigma_max_thresholds = [1.5, 10, 100, 1000, 10000]
    sigma_max_stats = {
        thr: count_and_pct(alllargestSingValCheck, thr, "gt")
        for thr in sigma_max_thresholds
    }

    # ----- sigma_min thresholds -----
    sigma_min_thresholds = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    sigma_min_stats = {
        thr: count_and_pct(allSmallSingvalsCheck, thr, "lt")
        for thr in sigma_min_thresholds
    }


    # ----- console output -----
    print("=== Spectral Threshold Statistics ===")
    print(f"Total non-attention layers: {total_num_nonAttn_layers}\n")

    for thr, (count, pct) in sigma_max_stats.items():
        print(
            f"sigma_max > {thr:g} : "
            f"{count}/{total_num_nonAttn_layers} ({pct:.2f}%)"
        )

    print()

    for thr, (count, pct) in sigma_min_stats.items():
        print(
            f"sigma_min < {thr:g} : "
            f"{count}/{total_num_nonAttn_layers} ({pct:.2f}%)"
        )


    # ----- save report -----
    report_path = "gemma_attack/NonAttnCondAnalysis/spectral_stats_report.txt"

    with open(report_path, "w") as f:
        f.write("=== Spectral Statistics Report ===\n\n")

        f.write(f"Total non-attention layers: {total_num_nonAttn_layers}\n\n")

        f.write("----- Sigma Max Threshold Counts -----\n")
        for thr, (count, pct) in sigma_max_stats.items():
            f.write(
                f"Layers with sigma_max > {thr:g} : "
                f"{count}/{total_num_nonAttn_layers} "
                f"({pct:.2f}%)\n"
            )

        f.write("\n----- Sigma Min Threshold Counts -----\n")
        for thr, (count, pct) in sigma_min_stats.items():
            f.write(
                f"Layers with sigma_min < {thr:g} : "
                f"{count}/{total_num_nonAttn_layers} "
                f"({pct:.2f}%)\n"
            )

        f.write("\n----- Sigma Max Statistics -----\n")
        f.write(f"Median sigma_max : {np.median(alllargestSingValCheck):.6f}\n")
        f.write(f"Mean sigma_max   : {np.mean(alllargestSingValCheck):.6f}\n")
        f.write(f"Min sigma_max    : {np.min(alllargestSingValCheck):.6f}\n")
        f.write(f"Max sigma_max    : {np.max(alllargestSingValCheck):.6f}\n\n")

        f.write("----- Sigma Min Statistics -----\n")
        f.write(f"Median sigma_min : {np.median(allSmallSingvalsCheck):.6e}\n")
        f.write(f"Mean sigma_min   : {np.mean(allSmallSingvalsCheck):.6e}\n")
        f.write(f"Min sigma_min    : {np.min(allSmallSingvalsCheck):.6e}\n")
        f.write(f"Max sigma_min    : {np.max(allSmallSingvalsCheck):.6e}\n\n")

        f.write("----- Condition Number Statistics -----\n")
        f.write(f"Median kappa     : {np.median(allCondNumsCheck):.6f}\n")
        f.write(f"Mean kappa       : {np.mean(allCondNumsCheck):.6f}\n")
        f.write(f"Min kappa        : {np.min(allCondNumsCheck):.6f}\n")
        f.write(f"Max kappa        : {np.max(allCondNumsCheck):.6f}\n")

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()

