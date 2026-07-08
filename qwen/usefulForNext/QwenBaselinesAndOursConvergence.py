
'''


export CUDA_VISIBLE_DEVICES=0
conda deactivate
cd interpretAttacks/
conda activate gemma3
python qwen/QwenBaselinesAndOursConvergence.py \
    --learningRate 0.001 \
    --num_steps 1000 \
    --AttackStartLayer 0 \
    --numLayerstAtAtime 1 \
    --whichMLP gate_proj \
    --whichMLPVis gate_proj \
    --chosenLanLayers 2 \
    --chosenVisLayers 0 1 2 4 5 6 7 8 9 14 24 \
    --numSamplesConsidered 1

'''

from bert_score import score
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import FuncFormatter


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


parser = argparse.ArgumentParser(
    description="Qwen BERTScore comparison across perturbation budgets"
)

parser.add_argument("--learningRate", type=float, default=1e-3)
parser.add_argument("--num_steps", type=int, default=2000)
parser.add_argument("--AttackStartLayer", type=int, default=0)
parser.add_argument("--AttackStartLayer_vis", type=int, default=0)
parser.add_argument("--numLayerstAtAtime", type=int, default=2)

parser.add_argument("--whichMLP", type=str, default="fc1")
parser.add_argument("--whichMLPVis", type=str, default="fc1")

parser.add_argument("--numSamplesConsidered", type=int, default=50)

parser.add_argument(
    "--chosenLanLayers",
    type=int,
    nargs="+",
    default=None,
    help="Example: --chosenLanLayers 0 1 2"
)

parser.add_argument(
    "--chosenVisLayers",
    type=int,
    nargs="+",
    default=None,
    help="Example: --chosenVisLayers 11 12 13"
)

args = parser.parse_args()

lr = float(args.learningRate)
num_steps = int(args.num_steps)
AttackStartLayer = int(args.AttackStartLayer)
AttackStartLayer_vis = int(args.AttackStartLayer_vis)
numLayerstAtAtime = int(args.numLayerstAtAtime)
whichMLP = str(args.whichMLP)
whichMLPVis = str(args.whichMLPVis)
numSamplesConsidered = int(args.numSamplesConsidered)

chosenLanLayers = args.chosenLanLayers
chosenVisLayers = args.chosenVisLayers

towardsNull = 0.5
ega_ratio = 0.2

#allEpsilons = [0.001, 0.002, 0.003, 0.004, 0.005]

#allEpsilons = [0.002, 0.003, 0.004, 0.005]
#allEpsilons = [0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005]
allEpsilons = [0.005]
#allEpsilons = [0.002, 0.003, 0.004]

#all_attck_types = ["bsa", "dra", "fdam", "ssp", "ega", "nllm", "saa_loop", "saa_loopC"]
#all_attck_types = ["bsa", "dra", "fdam", "ssp", "ega", "nllm", "saa_loop", "saa_loop"]
#AllAttckTypes = ["BSA", "DRA", "FDA", "SSPA", "EGA", "CE", "SSPMA", "SSGRA"]


all_attck_types = ["bsa", "dra", "fdam", "ssp", "ega", "nllm", "saa_loop"]
#AllAttckTypes = ["BSA", "DRA", "FDA", "SSPA", "EGA", "CE", "SSPMA\n\SSGRA"]
AllAttckTypes = ["BSA", "DRA", "FDA", "SSPA", "EGA", "CE", "SSGRA"]


precisionMeanForAttacksSeries = []
precisionStdForAttacksSeries = []

recallMeanForAttacksSeries = []
recallStdForAttacksSeries = []

f1MeanForAttacksSeries = []
f1StdForAttacksSeries = []



# ============================================================
# Convergence plotting
# ============================================================

save_dir = "/home/luser/interpretAttacks/qwen/AllPlots/convergence"
os.makedirs(save_dir, exist_ok=True)

steps = np.arange(1, num_steps + 1)
attack_curves = {}

def pad_to_num_steps(arr, num_steps):
    arr = np.asarray(arr).flatten()

    if len(arr) == 0:
        raise ValueError("Empty convergence array found.")

    if len(arr) < num_steps:
        last_value = arr[-1]
        padding = np.full(num_steps - len(arr), last_value)
        arr = np.concatenate([arr, padding])

    elif len(arr) > num_steps:
        arr = arr[:num_steps]

    return arr


for epsilon in allEpsilons:

    print(f"\nEvaluating epsilon = {epsilon}")

    attack_curves = {}

    for attck_type in all_attck_types:

        sample_curves = []

        print("attack type:", attck_type)

        for attackSample in range(5, 5+numSamplesConsidered):

            if attck_type == "saa_loop":
                advOutputPath = (
                    f"qwen/outputsStorageImagenet/convergence/{attackSample}/"
                    f"qwen_ORIG_attack_{attck_type}_lr_{lr}_eps_{epsilon}_"
                    f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
                    f"num_steps_{num_steps}_towardsNull_{towardsNull}_"
                    f"{whichMLP}_{whichMLPVis}_{chosenLanLayers}_{chosenVisLayers}.npy"
                )

            elif attck_type == "ega":
                advOutputPath = (
                    f"qwen/outputsStorageImagenet/convergence/{attackSample}/"
                    f"qwen_ORIG_attack_ega_lr_{lr}_eps_{epsilon}_"
                    f"num_steps_{num_steps}_ratio_{ega_ratio}.npy"
                )

            else:
                advOutputPath = (
                    f"qwen/outputsStorageImagenet/convergence/{attackSample}/"
                    f"qwen_ORIG_attack_{attck_type}_lr_{lr}_eps_{epsilon}_"
                    f"num_steps_{num_steps}_.npy"
                )

            testIt = np.load(advOutputPath)
            print("testIt.shape", testIt.shape)

            testIt = pad_to_num_steps(abs(testIt), num_steps)
            sample_curves.append(testIt)

        sample_curves = np.stack(sample_curves, axis=0)
        mean_curve = np.mean(sample_curves, axis=0)

        attack_curves[attck_type] = mean_curve

# ========================================================
    # Plot each attack separately for this epsilon
    # ========================================================

    for i, attck_type in enumerate(all_attck_types):

        plt.figure(figsize=(10, 6))

        plt.plot(
            steps,
            attack_curves[attck_type],
            label=AllAttckTypes[i],
            color="tab:blue"
        )

        plt.xlabel("Optimization Step")
        plt.ylabel("Convergence Value")
        plt.title(f"Convergence Plot - {AllAttckTypes[i]}, epsilon = {epsilon}")
        plt.xlim(1, num_steps)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(loc="best", frameon=True)
        plt.tight_layout()

        save_path = os.path.join(
            save_dir,
            f"qwen_convergence_{attck_type}_eps_{epsilon}_lr_{lr}_steps_{num_steps}.png"
        )

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved plot to: {save_path}")