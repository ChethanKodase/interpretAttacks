

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
    --numSamplesConsidered 38

'''

from bert_score import score
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import os

# ============================================================
# Paper-ready plotting settings
# ============================================================
# - Larger fonts so the figure is legible at column width (~3.3in) or
#   double-column width (~7in) in a printed PDF, not just on a screen.
# - fonttype 42 keeps text as real, editable/searchable text when embedded
#   in LaTeX (no font substitution issues at submission time).
# - Save as PDF (vector) for the camera-ready version; PNG is just for
#   quick viewing.

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 13,
    "axes.labelsize": 16,
    "axes.titlesize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "lines.linewidth": 2.8,
    "axes.linewidth": 1.1,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
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
method_labels = ["SSGRA-bottom", "SSGRA-top"]  # display names

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
# Combined, space-efficient box plot (P / R / F1 as one figure)
# ============================================================

COLORS = ["#4C72B0", "#DD8452"]  # blue, orange
METRICS = [("P", "Precision"), ("R", "Recall"), ("F1", "F1")]


def _draw_boxplot(ax, metric_key):
    data = [results[m][metric_key] for m in methods]

    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="black", linewidth=2.2),
        whiskerprops=dict(linewidth=1.6),
        capprops=dict(linewidth=1.6),
        flierprops=dict(marker="o", markersize=4, linestyle="none", markeredgewidth=0.8),
        boxprops=dict(linewidth=1.4),
    )
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    for i, (m, color) in enumerate(zip(methods, COLORS), start=1):
        y = results[m][metric_key]
        x = np.random.normal(i, 0.06, size=len(y))
        ax.scatter(x, y, color=color, alpha=0.55, s=16, zorder=3, edgecolors="none")

    ax.set_xticks([1, 2])
    ax.set_xticklabels(method_labels, fontsize=13)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis="both", width=1.1, length=4)
    ax.grid(True, axis="y", linewidth=0.6, alpha=0.3)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
    return bp


def plot_boxplot(metric_key, ylabel, filename_suffix):
    save_dir = "qwen/AllPlots/boxplotComparision"
    os.makedirs(save_dir, exist_ok=True)

    # Single-panel figure sized for a paper column (~3.3-3.5in wide).
    # constrained_layout removes the need for manual subplots_adjust
    # tuning and keeps whitespace tight around the larger fonts.
    fig, ax = plt.subplots(figsize=(4.2, 4.6), constrained_layout=True)

    bp = _draw_boxplot(ax, metric_key)

    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(rf"$\epsilon = {epsilon}$", fontsize=15, pad=6)

    base_name = (
        f"boxplot_{filename_suffix}_eps_{epsilon}_"
        f"num_steps_{num_steps}_"
        f"AttackStartLayer_{AttackStartLayer}_"
        f"numSamples_{numSamplesConsidered}"
    )
    pdf_path = os.path.join(save_dir, base_name + ".pdf")
    png_path = os.path.join(save_dir, base_name + ".png")

    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)          # for the paper
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.05)  # for quick viewing
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


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