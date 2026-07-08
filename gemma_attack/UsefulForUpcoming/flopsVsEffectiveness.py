

'''

export CUDA_VISIBLE_DEVICES=3
conda deactivate
cd interpretAttacks/
conda activate gemma3
python gemma_attack/flopsVsEffectiveness.py

'''

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

#whichPlot = "Precision"
#whichPlot = "Recall"
whichPlot = "F1"

methods = ["BSA", "DRA", "EGA", "FDA", "SSP", "NLL", "SSPMA", "SSGRA"]

flops = np.array([
    25896860154024,
    10102865928192,
    19515596799726,
    10103237549901,
    15614300402710,
    19561743997328,
    12270110084825,
    26010423057294
], dtype=float)



bertPrecisionScores = np.array([
    0.879147,  # BSA
    0.914639,  # DRA
    0.907173,  # EGA
    0.923928,  # FDA
    0.918544,  # SSP  (SSPA)
    0.878676,  # NLL  (CE)
    0.917156,  # SSPMA
    0.873870,  # SSGRA
])

bertRecallScores = np.array([
    0.880452,  # BSA
    0.914070,  # DRA
    0.909753,  # EGA
    0.922930,  # FDA
    0.914500,  # SSP  (SSPA)
    0.879471,  # NLL  (CE)
    0.916893,  # SSPMA
    0.872882,  # SSGRA
])

bertF1Scores = np.array([
    0.879675,  # BSA
    0.914233,  # DRA
    0.908390,  # EGA
    0.923329,  # FDA
    0.916401,  # SSP  (SSPA)
    0.878972,  # NLL  (CE)
    0.916922,  # SSPMA
    0.873236,  # SSGRA
])


if whichPlot == "Precision":
    bert_scores = bertPrecisionScores
if whichPlot == "Recall":
    bert_scores = bertRecallScores
if whichPlot == "F1":
    bert_scores = bertF1Scores

save_dir = "gemma_attack/AllPlots/MethodEffectivenessEfficiency"
os.makedirs(save_dir, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 7,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(3.35, 2.25))  # single-column paper width

colors = plt.cm.tab10(np.arange(len(methods)))

ax.scatter(
    flops / 1e12,
    bert_scores,
    s=100,
    c=colors,
    edgecolor="black",
    linewidth=0.35,
    alpha=0.9,
    zorder=3
)

# Compact manual label offsets
if whichPlot == "Precision":
    offsets = {
        "BSA": (3, 5),
        "DRA": (3, 5),
        "EGA": (-13, -11),
        "FDA": (3, 5),
        "SSP": (3, -10),
        "NLL": (-13, 6),
        "SSPMA": (3, 4),
        "SSGRA": (3, 4)
    }
if whichPlot == "Recall":
    offsets = {
        "BSA": (3, 5),
        "DRA": (3, 5),
        "EGA": (-13, 7),
        "FDA": (3, 5),
        "SSP": (3, -10),
        "NLL": (-13, -12),
        "SSPMA": (3, 4),
        "SSGRA": (3, 4)
    }
if whichPlot == "F1":
    offsets = {
        "BSA": (3, 5),
        "DRA": (-20, -10),
        "EGA": (-13, 7),
        "FDA": (-20, 6),
        "SSP": (-10, -12),
        "NLL": (-13, -12),
        "SSPMA": (-10, 6),
        "SSGRA": (3, -12)
    }


for m, x, y in zip(methods, flops / 1e12, bert_scores):
    dx, dy = offsets[m]
    ax.annotate(
        m,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=7,
        fontweight="bold"
    )

ax.axhline(0, color="gray", linewidth=0.6, linestyle=":", zorder=1)

ax.set_xlabel(r"FLOPs ($\times 10^{12}$)")


if whichPlot == "Precision":
    ax.set_ylabel("BERT Precision")

    ax.set_xlim(5.8, 30.8)
    ax.set_ylim(0.65, 1.0)

    ax.set_xticks([5, 10, 15, 20, 25, 30])
    ax.set_yticks([0.7, 0.7 + 0.3/4, 0.7 + 2 * 0.3/4, 0.7 + 3* 0.3/4, 0.7 + 4* 0.3/4])

if whichPlot == "Recall":
    ax.set_ylabel("BERT Recall")

    ax.set_xlim(5.8, 30.8)
    ax.set_ylim(0.7, 1.0)

    ax.set_xticks([5, 10, 15, 20, 25, 30])
    ax.set_yticks([0.7, 0.7 + 0.3/4, 0.7 + 2 * 0.3/4, 0.7 + 3* 0.3/4, 0.7 + 4* 0.3/4])

if whichPlot == "F1":
    ax.set_ylabel("BERT F1 Score")

    ax.set_xlim(5.8, 30.8)
    ax.set_ylim(0.84, 1.0)

    ax.set_xticks([5, 10, 15, 20, 25, 30])
    ax.set_yticks([0.84, 0.84 + 0.16/4, 0.84 + 2 * 0.16/4, 0.84 + 3* 0.16/4, 0.84 + 4* 0.16/4])




# Tight axis limits


ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.35, zorder=0)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout(pad=0.15)

png_path = os.path.join(save_dir, "flops_vs_bert_score_"+whichPlot+".png")
pdf_path = os.path.join(save_dir, "flops_vs_bert_score_"+whichPlot+".pdf")

fig.savefig(png_path, bbox_inches="tight", pad_inches=0.015, dpi=600)
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.015)

print(f"Saved PNG to: {png_path}")
print(f"Saved PDF to: {pdf_path}")

plt.show()





'''

================ EPSILON = 0.004 ================

Precision Mean
BSA                 : 0.205270
DRA                 : 0.611677
EGA                 : 0.193518
FDA                 : 0.558698
SSPA                : 0.524780
CE                  : 0.216357
SSPMA               : -0.627923

'''



'''
Recall Mean
BSA                 : 0.143267
DRA                 : 0.616609
EGA                 : 0.297238
FDA                 : 0.511168
SSPA                : 0.488341

CE                  : 0.249979
SSPMA               : -0.249979


'''

'''


F1 Mean
BSA                 : 0.174908
DRA                 : 0.614432
EGA                 : 0.244249
FDA                 : 0.535178
SSPA                : 0.506837
CE                  : 0.233364
SSPMA               : -0.457617

'''