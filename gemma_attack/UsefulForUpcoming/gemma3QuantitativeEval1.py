#!/usr/bin/env python3
"""
Gemma-3 quantitative evaluation with BERTScore.

Usage (example):
export CUDA_VISIBLE_DEVICES=5
conda activate gemma3
cd interpretAttacks
python gemma_attack/gemma3QuantitativeEval1.py --attck_type grill_wass --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1

"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from bert_score import score


# ----------------------------
# Plot styling (ECCV-friendly)
# ----------------------------
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.5,
})


# ----------------------------
# Helpers
# ----------------------------
def read_text(path: str) -> str:
    with open(path, "r") as f:
        return f.read().strip()


'''def sanitize_text(s: str, first_paragraph_only: bool = True) -> str:
    s = s.strip()
    if first_paragraph_only:
        # Helps remove accidental appended logs / extra sections
        s = s.split("\n\n")[0].strip()
    return s'''


def sanitize_text(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_mean_std(x, mean_arr, std_arr, ylabel, out_path, ylim=None, clip_band=True):
    mean_arr = np.asarray(mean_arr, dtype=np.float32)
    std_arr  = np.asarray(std_arr, dtype=np.float32)

    lower = mean_arr - std_arr
    upper = mean_arr + std_arr

    # BERTScore is bounded by [-1, 1] (rescaled often within that too).
    # Mean±std can exceed bounds; clip for visualization clarity.
    if clip_band:
        lower = np.maximum(lower, -1.0)
        upper = np.minimum(upper,  1.0)

    plt.figure(figsize=(3.4, 2.6))
    plt.plot(x, mean_arr, label=f"Mean {ylabel}")
    plt.fill_between(x, lower, upper, alpha=0.3, label="±1 Std")
    plt.xlabel("Hidden state index")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    # Uncomment if you want legend
    # plt.legend(frameon=False)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Gemma-3 BERTScore evaluation over hidden-state attack layers")
    parser.add_argument("--attck_type", type=str, default="grill_l2",
                        help="grill_l2 | grill_cos | grill_wass | OA_l2 | OA_cos | OA_wass")
    parser.add_argument("--desired_norm_l_inf", type=float, default=0.03,
                        help="epsilon L_inf in ORIGINAL pixel space [0..1].")
    parser.add_argument("--learningRate", type=float, default=1e-3,
                        help="Adam learning rate used during attack (for filename matching).")
    parser.add_argument("--num_steps", type=int, default=2000,
                        help="Number of Adam steps used during attack (for filename matching).")
    parser.add_argument("--AttackStartLayer", type=int, default=0,
                        help="(Ignored for sweep) kept for filename matching if needed.")
    parser.add_argument("--numLayerstAtAtime", type=int, default=2,
                        help="Number of layers attacked at a time (for filename matching).")

    # Evaluation controls
    parser.add_argument("--numHiddenStates", type=int, default=35,
                        help="Number of hidden states to sweep (default 35).")
    parser.add_argument("--sample_start", type=int, default=1,
                        help="First sample id (inclusive).")
    parser.add_argument("--sample_end", type=int, default=14,
                        help="Last sample id (inclusive).")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for BERTScore model, e.g., cuda:0 or cpu.")
    parser.add_argument("--model_type", type=str, default="roberta-large",
                        help="BERTScore backbone, e.g., roberta-large.")
    parser.add_argument("--rescale_with_baseline", action="store_true", default=True,
                        help="Use BERTScore baseline rescaling (default True).")
    parser.add_argument("--no_rescale_with_baseline", dest="rescale_with_baseline",
                        action="store_false", help="Disable baseline rescaling.")
    parser.add_argument("--first_paragraph_only", action="store_true", default=True,
                        help="Only evaluate the first paragraph from each output text (default True).")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Print per-sample P/R/F1.")

    args = parser.parse_args()

    attck_type = args.attck_type
    epsilon = float(args.desired_norm_l_inf)
    lr = float(args.learningRate)
    num_steps = int(args.num_steps)
    numHiddenStates = int(args.numHiddenStates)
    numLayerstAtAtime = int(args.numLayerstAtAtime)

    sample_ids = list(range(int(args.sample_start), int(args.sample_end) + 1))

    out_dir = "gemma_attack/AllPlots/bertScores/sampleSpecific"
    ensure_dir(out_dir)

    # Aggregate arrays across layers
    PmeanList, RmeanList, F1meanList = [], [], []
    PstdList,  RstdList,  F1stdList  = [], [], []

    for layer in range(numHiddenStates):
        cands = []
        refs  = []
        used_samples = []

        # Collect texts for this layer across all samples
        for sample_id in sample_ids:
            advOutputPath = (
                f"gemma_attack/outputsStorageImagenet/advOutputs/{sample_id}/"
                f"advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
                f"AttackStartLayer_{layer}_numLayerstAtAtime_{numLayerstAtAtime}_"
                f"num_steps_{num_steps}_.txt"
            )
            cleanOutputPath = (
                f"gemma_attack/outputsStorageImagenet/advOutputs/{sample_id}/cleanOutput.txt"
            )

            '''if not (os.path.exists(advOutputPath) and os.path.exists(cleanOutputPath)):
                if args.verbose:
                    print(f"[layer {layer}] Missing file(s) for sample {sample_id}")
                continue'''

            adv_txt = sanitize_text(read_text(advOutputPath))
            ref_txt = sanitize_text(read_text(cleanOutputPath))

            '''if len(adv_txt) == 0 or len(ref_txt) == 0:
                if args.verbose:
                    print(f"[layer {layer}] Empty text for sample {sample_id}")
                continue'''

            cands.append(adv_txt)
            refs.append(ref_txt)
            used_samples.append(sample_id)

        if len(cands) == 0:
            # No data for this layer
            PmeanList.append(0.0); PstdList.append(0.0)
            RmeanList.append(0.0); RstdList.append(0.0)
            F1meanList.append(0.0); F1stdList.append(0.0)
            continue

        # One batched call per layer (fast + avoids repeated warnings)
        P, R, F1 = score(
            cands,
            refs,
            lang="en",
            model_type=args.model_type,
            rescale_with_baseline=args.rescale_with_baseline,
            device=args.device
        )

        P_np = P.detach().cpu().numpy()
        R_np = R.detach().cpu().numpy()
        F1_np = F1.detach().cpu().numpy()

        if args.verbose:
            for sid, p, r, f1 in zip(used_samples, P_np, R_np, F1_np):
                print(f"[layer {layer:02d}] sample {sid:02d} | P={p:.4f} R={r:.4f} F1={f1:.4f}")

        PmeanList.append(float(P_np.mean()))
        PstdList.append(float(P_np.std()))
        RmeanList.append(float(R_np.mean()))
        RstdList.append(float(R_np.std()))
        F1meanList.append(float(F1_np.mean()))
        F1stdList.append(float(F1_np.std()))

        print(f"[layer {layer:02d}] N={len(P_np)} | "
              f"P={P_np.mean():.4f}±{P_np.std():.4f} "
              f"R={R_np.mean():.4f}±{R_np.std():.4f} "
              f"F1={F1_np.mean():.4f}±{F1_np.std():.4f}")

    # X axis: 1..numHiddenStates
    x = np.arange(1, numHiddenStates + 1, dtype=np.int32)

    # Save plots
    plot_mean_std(
        x, PmeanList, PstdList, "Precision",
        os.path.join(out_dir, "Precision_vs_hiddenstate.png"),
        ylim=None, clip_band=True
    )
    plot_mean_std(
        x, RmeanList, RstdList, "Recall",
        os.path.join(out_dir, "Recall_vs_hiddenstate.png"),
        ylim=None, clip_band=True
    )
    plot_mean_std(
        x, F1meanList, F1stdList, "F1 score",
        os.path.join(out_dir, "F1score_vs_hiddenstate.png"),
        ylim=None, clip_band=True
    )

    # Also save arrays for later reuse
    np.save(os.path.join(out_dir, f"bertScore_means_{attck_type}_eps{epsilon}_lr{lr}_steps{num_steps}.npy"),
            np.stack([PmeanList, RmeanList, F1meanList], axis=0).astype(np.float32))
    np.save(os.path.join(out_dir, f"bertScore_stds_{attck_type}_eps{epsilon}_lr{lr}_steps{num_steps}.npy"),
            np.stack([PstdList, RstdList, F1stdList], axis=0).astype(np.float32))

    print(f"\nSaved plots to: {out_dir}")


if __name__ == "__main__":
    main()