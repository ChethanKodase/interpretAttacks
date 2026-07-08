
'''

export CUDA_VISIBLE_DEVICES=0
conda activate gemma3
cd interpretAttacks
python gemma_attack/gemma3BaselinesAndOursComparisionAllEpsilon_toTestGrillSubReadyTabulation.py --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --AttackStartLayer_vis 11 --whichMLP na --whichMLP_vis na --numSamplesConsidered 45



export CUDA_VISIBLE_DEVICES=0
conda activate gemma3
cd interpretAttacks
python gemma_attack/gemma3BaselinesAndOursComparisionAllEpsilon_toTestGrillSubReadyTabulation.py --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --AttackStartLayer_vis 11 --whichMLP na --whichMLP_vis na --numSamplesConsidered 50


'''




from bert_score import score
import argparse
import numpy as np
import os

# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser(
    description="Create per-sample BERT-score tables for Gemma-3 attacks"
)

parser.add_argument("--learningRate", type=float, default=1e-3)
parser.add_argument("--num_steps", type=int, default=1000)

parser.add_argument("--AttackStartLayer", type=int, default=0)
parser.add_argument("--AttackStartLayer_vis", type=int, default=11)

parser.add_argument("--numLayerstAtAtime", type=int, default=1)

parser.add_argument("--whichMLP", type=str, default="na")
parser.add_argument("--whichMLP_vis", type=str, default="na")

parser.add_argument("--numSamplesConsidered", type=int, default=50)

parser.add_argument("--perturbationScale", type=str, default="tinyTiny")

args = parser.parse_args()

# ============================================================
# Parameters
# ============================================================

lr = float(args.learningRate)
num_steps = int(args.num_steps)

AttackStartLayer = int(args.AttackStartLayer)
AttackStartLayer_vis = int(args.AttackStartLayer_vis)

numLayerstAtAtime = int(args.numLayerstAtAtime)

whichMLP = str(args.whichMLP)
whichMLP_vis = str(args.whichMLP_vis)

numSamplesConsidered = int(args.numSamplesConsidered)
perturbationScale = str(args.perturbationScale)

towardsNull = 0.1
ega_ratio = 0.2

# ============================================================
# Epsilon settings
# ============================================================

epsilonTypes = perturbationScale

if epsilonTypes == "tiny":
    allEpsilons = [0.001, 0.002, 0.003, 0.004, 0.005]
    all_attck_types = ["bsa", "dra", "fdam", "ssp", "ega", "grill_cos"]

elif epsilonTypes == "tinyTiny":
    allEpsilons = [0.0005, 0.0006, 0.0007, 0.0008, 0.0009]
    all_attck_types = ["bsa", "dra", "fdam", "ssp", "ega", "grill_cos"]

else:
    allEpsilons = [0.01, 0.02, 0.03, 0.04, 0.05]
    all_attck_types = ["bsa", "dra", "fdam", "ssp", "ega", "grill_cos"]

# ============================================================
# Save directory
# ============================================================

save_dir = "gemma_attack/AllPlots/grillComparisionSeries/perSampleTables"
os.makedirs(save_dir, exist_ok=True)

# ============================================================
# Helper: get adversarial output path
# ============================================================

def get_adv_output_path(attck_type, epsilon, attackSample):

    if attck_type == "saa":

        advOutputPath = (
            f"gemma_attack/outputsStorageImagenet/advOutputs/"
            f"{attackSample}/"
            f"advOutput_attackType_{attck_type}_lr_{lr}_"
            f"eps_{epsilon}_AttackStartLayer_{AttackStartLayer}_"
            f"numLayerstAtAtime_{numLayerstAtAtime}_"
            f"num_steps_{num_steps}_towardsNull_{towardsNull}_"
            f"{whichMLP}.txt"
        )

    elif attck_type == "saav":

        advOutputPath = (
            f"gemma_attack/outputsStorageImagenet/advOutputs/"
            f"{attackSample}/"
            f"advOutput_attackType_{attck_type}_lr_{lr}_"
            f"eps_{epsilon}_AttackStartLayer_{AttackStartLayer_vis}_"
            f"numLayerstAtAtime_{numLayerstAtAtime}_"
            f"num_steps_{num_steps}_towardsNull_{towardsNull}_"
            f"{whichMLP_vis}.txt"
        )

    elif attck_type == "ega":

        advOutputPath = (
            f"gemma_attack/outputsStorageImagenet/advOutputs/"
            f"{attackSample}/"
            f"advOutput_attackType_{attck_type}_lr_{lr}_"
            f"eps_{epsilon}_num_steps_{num_steps}_"
            f"ratio_{ega_ratio}.txt"
        )

    else:

        advOutputPath = (
            f"gemma_attack/outputsStorageImagenet/advOutputs/"
            f"{attackSample}/"
            f"advOutput_attackType_{attck_type}_lr_{lr}_"
            f"eps_{epsilon}_num_steps_{num_steps}_.txt"
        )

    return advOutputPath


# ============================================================
# Main: compute and save per-sample tables
# ============================================================

for epsilon in allEpsilons:

    for attck_type in all_attck_types:

        rows = []

        print(f"\nProcessing attack={attck_type}, epsilon={epsilon}")

        for attackSample in range(1, numSamplesConsidered + 1):

            advOutputPath = get_adv_output_path(
                attck_type,
                epsilon,
                attackSample
            )

            cleanOutputPath = (
                f"gemma_attack/outputsStorageImagenet/advOutputs/"
                f"{attackSample}/cleanOutput.txt"
            )

            if not os.path.exists(advOutputPath):
                print(f"Missing adversarial file: {advOutputPath}")
                continue

            if not os.path.exists(cleanOutputPath):
                print(f"Missing clean file: {cleanOutputPath}")
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
                rescale_with_baseline=True
            )

            rows.append([
                attackSample,
                P.item(),
                R.item(),
                F1.item()
            ])

        # ====================================================
        # Save one table for this attack type and epsilon
        # ====================================================

        table_path = os.path.join(
            save_dir,
            f"bert_scores_attackType_{attck_type}_eps_{epsilon}_"
            f"numSamples_{numSamplesConsidered}.txt"
        )

        with open(table_path, "w") as f:

            f.write(f"Attack Type: {attck_type}\n")
            f.write(f"Epsilon: {epsilon}\n")
            f.write(f"Number of Samples Requested: {numSamplesConsidered}\n")
            f.write(f"Number of Samples Found: {len(rows)}\n")
            f.write("\n")

            f.write("Sample\tPrecision\tRecall\tF1\n")

            for sample_id, p, r, f1 in rows:
                f.write(
                    f"{sample_id}\t"
                    f"{p:.8f}\t"
                    f"{r:.8f}\t"
                    f"{f1:.8f}\n"
                )

            if len(rows) > 0:

                rows_np = np.array(rows)

                mean_p = rows_np[:, 1].mean()
                mean_r = rows_np[:, 2].mean()
                mean_f1 = rows_np[:, 3].mean()

                std_p = rows_np[:, 1].std()
                std_r = rows_np[:, 2].std()
                std_f1 = rows_np[:, 3].std()

                f.write("\n")
                f.write("Mean\t")
                f.write(f"{mean_p:.8f}\t{mean_r:.8f}\t{mean_f1:.8f}\n")

                f.write("STD\t")
                f.write(f"{std_p:.8f}\t{std_r:.8f}\t{std_f1:.8f}\n")

        print(f"Saved table: {table_path}")

print("\nDone.")