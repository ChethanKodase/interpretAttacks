

'''

export CUDA_VISIBLE_DEVICES=2
conda deactivate
cd interpretAttacks/
conda activate gemma3
python qwen/BERTablationTopBottom.py \
    --learningRate 0.001 \
    --num_steps 1000 \
    --AttackStartLayer 0 \
    --numLayerstAtAtime 1 \
    --whichMLP gate_proj \
    --whichMLPVis gate_proj \
    --chosenLanLayers 2 \
    --chosenVisLayers 0 1 2 4 5 6 7 8 9 14 24 \
    --numSamplesConsidered 22

'''

from bert_score import score
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# IJCAI-style plotting settings
# ============================================================

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "lines.linewidth": 2.8,
    "axes.linewidth": 1.0,
})

parser = argparse.ArgumentParser(description="Qwen BERTScore boxplot comparison: saa_loop vs saa_loopTop")

parser.add_argument("--learningRate", type=float, default=1e-3)
parser.add_argument("--num_steps", type=int, default=2000)
parser.add_argument("--AttackStartLayer", type=int, default=0)
parser.add_argument("--numLayerstAtAtime", type=int, default=2)
parser.add_argument("--whichMLP", type=str, default="fc1")
parser.add_argument("--whichMLPVis", type=str, default="fc1")
parser.add_argument("--numSamplesConsidered", type=int, default=50)
parser.add_argument("--chosenLanLayers", type=int, nargs="+", default=None)
parser.add_argument("--chosenVisLayers", type=int, nargs="+", default=None)

args = parser.parse_args()

lr = float(args.learningRate)
num_steps = int(args.num_steps)
AttackStartLayer = int(args.AttackStartLayer)
numLayerstAtAtime = int(args.numLayerstAtAtime)
whichMLP = str(args.whichMLP)
whichMLPVis = str(args.whichMLPVis)
numSamplesConsidered = int(args.numSamplesConsidered)
chosenLanLayers = args.chosenLanLayers
chosenVisLayers = args.chosenVisLayers

towardsNull = 0.5
epsilon = 0.002

# Both methods share the same path template
def get_adv_path(attck_type, sample):
    return (
        f"qwen/outputsStorageImagenet/advOutputs/{sample}/"
        f"advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
        f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
        f"num_steps_{num_steps}_towardsNull_{towardsNull}_"
        f"{whichMLP}_{whichMLPVis}_{chosenLanLayers}_{chosenVisLayers}.txt"
    )

def get_clean_path(sample):
    return (
        f"qwen/outputsStorageImagenet/advOutputs/{sample}/"
        f"cleanOutput.txt"
    )

# ============================================================
# Collect per-sample BERTScores for each method
# ============================================================

methods = ["saa_loop", "saa_loopTop"]
method_labels = ["SSPMA-botom", "SSPMA-Top"]  # display names

results = {m: {"P": [], "R": [], "F1": []} for m in methods}

for attck_type in methods:
    print(f"\nEvaluating: {attck_type}")
    for attackSample in range(1, numSamplesConsidered):
        adv_path = get_adv_path(attck_type, attackSample)
        clean_path = get_clean_path(attackSample)

        if not os.path.exists(adv_path):
            print(f"  Missing adv output: {adv_path}")
            continue
        if not os.path.exists(clean_path):
            print(f"  Missing clean output: {clean_path}")
            continue

        with open(adv_path, "r") as f:
            adv_output = [f.read().strip()]
        with open(clean_path, "r") as f:
            clean_output = [f.read().strip()]

        P, R, F1 = score(
            adv_output,
            clean_output,
            lang="en",
            model_type="roberta-large",
            rescale_with_baseline=False
        )

        results[attck_type]["P"].append(P.item())
        results[attck_type]["R"].append(R.item())
        results[attck_type]["F1"].append(F1.item())

    print(f"  Collected {len(results[attck_type]['F1'])} samples")

# ============================================================
# Box plot
# ============================================================

COLORS = ["#4C72B0", "#DD8452"]  # blue, orange

def plot_boxplot(metric_key, ylabel, filename_suffix):
    save_dir = "qwen/AllPlots/boxplotComparision"
    os.makedirs(save_dir, exist_ok=True)

    data = [results[m][metric_key] for m in methods]

    fig, ax = plt.subplots(figsize=(4.5, 5.5))

    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color="black", linewidth=2.2),
        whiskerprops=dict(linewidth=1.6),
        capprops=dict(linewidth=1.6),
        flierprops=dict(
            marker="o",
            markersize=4,
            linestyle="none",
            markeredgewidth=0.8
        ),
        boxprops=dict(linewidth=1.4),
    )

    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Overlay individual data points (strip plot style)
    for i, (m, color) in enumerate(zip(methods, COLORS), start=1):
        y = results[m][metric_key]
        x = np.random.normal(i, 0.07, size=len(y))  # jitter
        ax.scatter(x, y, color=color, alpha=0.55, s=18, zorder=3, edgecolors="none")

    ax.set_xticks([1, 2])
    ax.set_xticklabels(method_labels, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(rf"$\epsilon = {epsilon}$", fontsize=13, pad=8)

    ax.tick_params(axis="y", labelsize=12, width=1.2, length=5)
    ax.tick_params(axis="x", width=1.2, length=4)
    ax.grid(True, axis="y", linewidth=0.7, alpha=0.35)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    fig.subplots_adjust(left=0.20, right=0.97, bottom=0.12, top=0.90)

    save_path = os.path.join(
        save_dir,
        f"boxplot_{filename_suffix}_eps_{epsilon}_"
        f"num_steps_{num_steps}_"
        f"AttackStartLayer_{AttackStartLayer}_"
        f"towardsNull_{towardsNull}_"
        f"numSamples_{numSamplesConsidered}_"
        f"{whichMLP}_{whichMLPVis}_"
        f"{chosenLanLayers}_{chosenVisLayers}.png"
    )

    plt.savefig(save_path, dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Saved: {save_path}")


plot_boxplot("P",  "BERT Precision", "Precision")
plot_boxplot("R",  "BERT Recall",    "Recall")
plot_boxplot("F1", "BERT F1 Score",  "F1")

# ============================================================
# Print summary statistics
# ============================================================

print("\n========== Summary ==========")
for m, label in zip(methods, method_labels):
    for metric in ["P", "R", "F1"]:
        arr = np.array(results[m][metric])
        if len(arr) > 0:
            print(f"{label:>12} | {metric} | mean={arr.mean():.4f}  std={arr.std():.4f}  "
                  f"median={np.median(arr):.4f}  n={len(arr)}")