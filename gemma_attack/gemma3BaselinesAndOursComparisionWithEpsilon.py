
'''




export CUDA_VISIBLE_DEVICES=7
conda activate gemma3
cd interpretAttacks
python gemma_attack/gemma3BaselinesAndOursComparision.py  --attck_type saa --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --AttackStartLayer_vis 11 --numLayerstAtAtime 1 --whichMLP gate_proj --whichMLP_vis fc2 --numSamplesConsidered 50


export CUDA_VISIBLE_DEVICES=6
conda activate gemma3
cd interpretAttacks
python gemma_attack/gemma3BaselinesAndOursComparisionWithEpsilon.py \
  --learningRate 0.001 \
  --num_steps 1000 \
  --AttackStartLayer 0 \
  --AttackStartLayer_vis 11 \
  --numLayerstAtAtime 1 \
  --whichMLP gate_proj \
  --whichMLP_vis fc2 \
  --numSamplesConsidered 50

'''

from bert_score import score
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import FuncFormatter

# ============================================================
# Paper-style plotting settings
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

# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser(
    description="Gemma-3 BERT-score comparison across epsilon values"
)

parser.add_argument("--learningRate", type=float, default=1e-3)
parser.add_argument("--num_steps", type=int, default=1000)
parser.add_argument("--AttackStartLayer", type=int, default=0)
parser.add_argument("--AttackStartLayer_vis", type=int, default=11)
parser.add_argument("--numLayerstAtAtime", type=int, default=1)
parser.add_argument("--whichMLP", type=str, default="gate_proj")
parser.add_argument("--whichMLP_vis", type=str, default="fc2")
parser.add_argument("--numSamplesConsidered", type=int, default=50)

args = parser.parse_args()

lr = float(args.learningRate)
num_steps = int(args.num_steps)
AttackStartLayer = int(args.AttackStartLayer)
AttackStartLayer_vis = int(args.AttackStartLayer_vis)
numLayerstAtAtime = int(args.numLayerstAtAtime)
whichMLP = str(args.whichMLP)
whichMLP_vis = str(args.whichMLP_vis)
numSamplesConsidered = int(args.numSamplesConsidered)

# ============================================================
# Fixed settings from your comparison code
# ============================================================

allEpsilons = [0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005]
#allEpsilons = [0.0005, 0.0006, 0.0007, 0.0008, 0.0009]
#allEpsilons = [0.002, 0.003, 0.004]
towardsNull = 0.1
ega_ratio = 0.2

chosenLanLayers = [0]
chosenVisLayers = [11]

#     Spectral-Subspace-Guided Representation Similarity Attack (SSGRA)
# Spectral Subspace Projection Attack (SSPA)
all_attck_types = [
    "bsa",
    "dra",
    "fdam",
    "ssp",
    "ega",
    "nllm",
    #"saa_loopR",
    "saa_BSAexpTN_P15_BAp05"
]

AllAttckTypes = [
    "BSA",
    "DRA",
    "FDA",
    "SSPA",
    "EGA",
    "CE",
    #"SSPMA",
    "SSGRA"
]

# ============================================================
# Path function
# ============================================================

def get_adv_output_path(attck_type, attackSample, epsilon):

    if attck_type == "saa":
        return (
            f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/"
            f"advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
            f"AttackStartLayer_{AttackStartLayer}_"
            f"numLayerstAtAtime_{numLayerstAtAtime}_"
            f"num_steps_{num_steps}_towardsNull_{towardsNull}_"
            f"{whichMLP}.txt"
        )

    if attck_type == "saa_loopR":
        #advOutput_attackType_saa_loopR_lr_0.001_eps_0.002_AttackStartLayer_0_numLayerstAtAtime_2_num_steps_1000_towardsNull_0.1_lanMLP_down_proj_visMLP_out_proj_lanLayers_[3, 4, 5]_visLayers_[1, 3, 5, 10, 11, 12, 15, 17, 21, 23]

        whichMLPR = "down_proj"
        whichMLPvisR = "out_proj"
        chosenLanLayersR = [3, 4, 5]
        chosenVisLayersR = [1, 3, 5, 10, 11, 12, 15, 17, 21, 23]
        towardsNullR = 0.1
        numLayerstAtAtimeR = 2
        return (
            f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/"
            f"advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
            f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtimeR}_"
            f"num_steps_{num_steps}_towardsNull_{towardsNullR}_"
            f"lanMLP_{whichMLPR}_visMLP_{whichMLPvisR}_"
            f"lanLayers_{chosenLanLayersR}_visLayers_{chosenVisLayersR}.txt"
        )
    


    elif attck_type == "saav":
        return (
            f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/"
            f"advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
            f"AttackStartLayer_{AttackStartLayer_vis}_"
            f"numLayerstAtAtime_{numLayerstAtAtime}_"
            f"num_steps_{num_steps}_towardsNull_{towardsNull}_"
            f"{whichMLP_vis}.txt"
        )

    elif attck_type == "ega":
        return (
            f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/"
            f"advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
            f"num_steps_{num_steps}_ratio_{ega_ratio}.txt"
        )
    

    elif attck_type == "saa_BSApo2":
        attck_typeR = "saa_BSA"
        towardsNullR = 0.02
        AttackStartLayerR = 0
        numLayerstAtAtimeR = 2
        whichMLPR = "up_proj"
        whichMLPvisR = "fc2"
        chosenLanLayersR = [0]
        chosenVisLayersR = [0]
        balancingAlphaR = 0.06

        return (
            f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/"
            f"advOutput_attackType_{attck_typeR}_lr_{lr}_eps_{epsilon}_"
            f"AttackStartLayer_{AttackStartLayerR}_"
            f"numLayerstAtAtime_{numLayerstAtAtimeR}_"
            f"num_steps_{num_steps}_towardsNull_{towardsNullR}_"
            f"lanMLP_{whichMLPR}_visMLP_{whichMLPvisR}_"
            f"lanLayers_{chosenLanLayersR}_visLayers_{chosenVisLayersR}_"
            f"balancingAlpha_{balancingAlphaR}.txt"
        )

    elif attck_type == "saa_BSAexpTN_P15_BAp05":
        attck_typeR = "saa_BSAexp"
        towardsNullR = 0.15
        AttackStartLayerR = 0
        numLayerstAtAtimeR = 2
        whichMLPR = "up_proj"
        whichMLPvisR = "fc2"
        balancingAlphaR = 0.5

        return (
            f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/"
            f"advOutput_attackType_{attck_typeR}_lr_{lr}_eps_{epsilon}_"
            f"AttackStartLayer_{AttackStartLayerR}_"
            f"numLayerstAtAtime_{numLayerstAtAtimeR}_"
            f"num_steps_{num_steps}_towardsNull_{towardsNullR}_"
            f"lanMLP_{whichMLPR}_visMLP_{whichMLPvisR}_"
            f"lanLayers_upto4_visLayers_all_"
            f"balancingAlpha_{balancingAlphaR}.txt"
        )

    else:
        return (
            f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/"
            f"advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
            f"num_steps_{num_steps}_.txt"
        )


# ============================================================
# Storage
# Shape after computation:
# rows = epsilons, columns = attack methods
# ============================================================

precisionMeanSeries = []
precisionStdSeries = []

recallMeanSeries = []
recallStdSeries = []

f1MeanSeries = []
f1StdSeries = []

# ============================================================
# Compute BERT scores
# ============================================================

precisionMinimumSeries = []
recallMinimumSeries = []
f1MinimumSeries = []

for epsilon in allEpsilons:

    precisionMeans = []
    precisionStds = []
    MinimumPrecision = []


    recallMeans = []
    recallStds = []
    MinimumRecall = []


    f1Means = []
    f1Stds = []
    Minimumf1 = []

    print(f"\nProcessing epsilon = {epsilon}")

    for attck_type in all_attck_types:

        sampleAggP = []
        sampleAggR = []
        sampleAggF1 = []

        print("  Attack type:", attck_type)

        for attackSample in range(1, numSamplesConsidered):

            advOutputPath = get_adv_output_path(
                attck_type=attck_type,
                attackSample=attackSample,
                epsilon=epsilon
            )

            cleanOutputPath = (
                f"gemma_attack/outputsStorageImagenet/advOutputs/"
                f"{attackSample}/cleanOutput.txt"
            )

            if not os.path.exists(advOutputPath):
                print("Missing adv file:", advOutputPath)
                continue

            if not os.path.exists(cleanOutputPath):
                print("Missing clean file:", cleanOutputPath)
                continue

            with open(advOutputPath, "r") as f:
                advOutput = [f.read().strip()]

            with open(cleanOutputPath, "r") as f:
                cleanOutput = [f.read().strip()]

            P, R, F1 = score(
                advOutput,
                cleanOutput,
                lang="en",
                model_type="roberta-large",
                rescale_with_baseline=False
            )

            sampleAggP.append(P.item())
            sampleAggR.append(R.item())
            sampleAggF1.append(F1.item())

        sampleAggP = np.array(sampleAggP)
        sampleAggR = np.array(sampleAggR)
        sampleAggF1 = np.array(sampleAggF1)

        precisionMeans.append(sampleAggP.mean())
        precisionStds.append(sampleAggP.std())
        MinimumPrecision.append(sampleAggP.min())

        recallMeans.append(sampleAggR.mean())
        recallStds.append(sampleAggR.std())
        MinimumRecall.append(sampleAggR.min())

        f1Means.append(sampleAggF1.mean())
        f1Stds.append(sampleAggF1.std())
        Minimumf1.append(sampleAggF1.min())

    precisionMeanSeries.append(np.array(precisionMeans))
    precisionStdSeries.append(np.array(precisionStds))
    precisionMinimumSeries.append(np.array(MinimumPrecision))


    recallMeanSeries.append(np.array(recallMeans))
    recallStdSeries.append(np.array(recallStds))
    recallMinimumSeries.append(np.array(MinimumRecall))

    f1MeanSeries.append(np.array(f1Means))
    f1StdSeries.append(np.array(f1Stds))
    f1MinimumSeries.append(np.array(Minimumf1))


precisionMeanSeries = np.array(precisionMeanSeries)
precisionStdSeries = np.array(precisionStdSeries)
precisionMinimumSeries = np.array(precisionMinimumSeries)

recallMeanSeries = np.array(recallMeanSeries)
recallStdSeries = np.array(recallStdSeries)
recallMinimumSeries = np.array(recallMinimumSeries)

f1MeanSeries = np.array(f1MeanSeries)
f1StdSeries = np.array(f1StdSeries)
f1MinimumSeries = np.array(f1MinimumSeries)

# ============================================================
# Formatter
# ============================================================

def format_func(x, pos):
    if x == 0:
        return "0"

    exponent = int(np.floor(np.log10(abs(x))))
    coeff = x / (10 ** exponent)

    return r"${:.1f}\times10^{{{}}}$".format(coeff, exponent)

# ============================================================
# Plot function
# ============================================================

def plot_metric(means, stds, ylabel, save_name):

    fig, ax = plt.subplots(figsize=(4.8, 7.8))

    for i in range(means.shape[1]):

        mean = means[:, i]
        std = stds[:, i]

        ax.plot(
            allEpsilons,
            mean,
            label=AllAttckTypes[i],
            linewidth=3.2,
            marker="o",
            markersize=5.5,
            markeredgewidth=1.1,
        )

        ax.fill_between(
            allEpsilons,
            mean - std,
            mean + std,
            alpha=0.16,
            linewidth=0,
        )

    ax.set_xlabel(r"$c$", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)

    ax.set_xticks(allEpsilons)
    ax.xaxis.set_major_formatter(FuncFormatter(format_func))

    ax.tick_params(axis="x", rotation=35, labelsize=13, width=1.2, length=5)
    ax.tick_params(axis="y", labelsize=13, width=1.2, length=5)

    ax.grid(True, linewidth=0.7, alpha=0.35)

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=3,
        frameon=False,
        fontsize=12,
        handlelength=2.7,
        columnspacing=1.2,
        handletextpad=0.5,
    )

    for legline in legend.get_lines():
        legline.set_linewidth(4.0)

    for spine in ax.spines.values():
        spine.set_linewidth(1.3)

    fig.subplots_adjust(
        left=0.20,
        right=0.98,
        bottom=0.26,
        top=0.78,
    )

    save_path = os.path.join(save_dir, save_name + ".png")

    plt.savefig(
        save_path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.close()

# ============================================================
# Save directory
# ============================================================

save_dir = "gemma_attack/AllPlots/comparisionFinTestEpsilonSeries"
os.makedirs(save_dir, exist_ok=True)

base_name = (
    f"eps_0002_0003_0004_"
    f"num_steps_{num_steps}_"
    f"AttackStartLayer_{AttackStartLayer}_"
    f"AttackStartLayerVis_{AttackStartLayer_vis}_"
    f"towardsNull_{towardsNull}_"
    f"numSamplesConsidered_{numSamplesConsidered}_"
    f"{whichMLP}_{whichMLP_vis}_"
    f"lanLayers_{chosenLanLayers}_visLayers_{chosenVisLayers}"
)

# ============================================================
# Save plots
# ============================================================

plot_metric(
    precisionMeanSeries,
    precisionStdSeries,
    "BERT Precision",
    "PrecisionComparisonSeries_" + base_name,
)

plot_metric(
    recallMeanSeries,
    recallStdSeries,
    "BERT Recall",
    "RecallComparisonSeries_" + base_name,
)

plot_metric(
    f1MeanSeries,
    f1StdSeries,
    "BERT F1 Score",
    "F1ComparisonSeries_" + base_name,
)

# ============================================================
# Print results
# ============================================================

print("\nPrecision Means")
print(precisionMeanSeries)

print("\nPrecision STDs")
print(precisionStdSeries)

print("\nPrecision Minimum")
print(precisionMinimumSeries)


print("\nRecall Means")
print(recallMeanSeries)

print("\nRecall STDs")
print(recallStdSeries)

print("\nRecall Minimum")
print(recallMinimumSeries)

print("\nF1 Means")
print(f1MeanSeries)

print("\nF1 STDs")
print(f1StdSeries)

print("\nF1 Minimum")
print(f1MinimumSeries)

results_file = os.path.join(
    save_dir,
    f"BERTScoreStatistics_{base_name}.txt"
)


with open(results_file, "w") as f:

    f.write("Attack Methods:\n")
    f.write(str(AllAttckTypes))
    f.write("\n\n")

    for eps_idx, eps in enumerate(allEpsilons):

        f.write(f"\n================ EPSILON = {eps} ================\n")

        f.write("\nPrecision Mean\n")
        for m, val in zip(AllAttckTypes, precisionMeanSeries[eps_idx]):
            f.write(f"{m:20s}: {val:.6f}\n")

        f.write("\nPrecision Std\n")
        for m, val in zip(AllAttckTypes, precisionStdSeries[eps_idx]):
            f.write(f"{m:20s}: {val:.6f}\n")

        f.write("\nPrecision Minimum\n")
        for m, val in zip(AllAttckTypes, precisionMinimumSeries[eps_idx]):
            f.write(f"{m:20s}: {val:.6f}\n")

        #-------

        f.write("\nRecall Mean\n")
        for m, val in zip(AllAttckTypes, recallMeanSeries[eps_idx]):
            f.write(f"{m:20s}: {val:.6f}\n")

        f.write("\nRecall Std\n")
        for m, val in zip(AllAttckTypes, recallStdSeries[eps_idx]):
            f.write(f"{m:20s}: {val:.6f}\n")

        f.write("\nRecall Minimum\n")
        for m, val in zip(AllAttckTypes, recallMinimumSeries[eps_idx]):
            f.write(f"{m:20s}: {val:.6f}\n")

        #-------

        f.write("\nF1 Mean\n")
        for m, val in zip(AllAttckTypes, f1MeanSeries[eps_idx]):
            f.write(f"{m:20s}: {val:.6f}\n")

        f.write("\nF1 Std\n")
        for m, val in zip(AllAttckTypes, f1StdSeries[eps_idx]):
            f.write(f"{m:20s}: {val:.6f}\n")

        f.write("\nF1 Minimum\n")
        for m, val in zip(AllAttckTypes, f1MinimumSeries[eps_idx]):
            f.write(f"{m:20s}: {val:.6f}\n")
