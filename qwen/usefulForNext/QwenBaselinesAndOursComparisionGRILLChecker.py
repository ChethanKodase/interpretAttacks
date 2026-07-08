
'''


export CUDA_VISIBLE_DEVICES=3
conda deactivate
cd interpretAttacks/
conda activate gemma3
python qwen/QwenBaselinesAndOursComparisionGRILLChecker.py --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --whichMLP gate_proj --whichMLPVis gate_proj --chosenLanLayers 2 --chosenVisLayers  0 1 2 4 5 6 7 8 9 14 24 --numSamplesConsidered 38
python qwen/QwenBaselinesAndOursComparisionGRILLChecker.py --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --whichMLP gate_proj --whichMLPVis gate_proj --chosenLanLayers 2 --chosenVisLayers  0 1 2 4 5 6 7 8 9 14 24 --numSamplesConsidered 38



'''


from bert_score import score
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.5,
})


parser = argparse.ArgumentParser(
    description="Qwen BERTScore comparison: BSA vs GRILL attacks"
)

parser.add_argument("--desired_norm_l_inf", type=float, default=0.03)
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
    default=None
)

parser.add_argument(
    "--chosenVisLayers",
    type=int,
    nargs="+",
    default=None
)

args = parser.parse_args()

epsilon = float(args.desired_norm_l_inf)
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


ega_ratio = 0.2
towardsNull = 0.5
vision_weight = 1.0


all_attck_types = [
    "bsa",
    "dra",
    "fdam",
    "ssp",
    "ega",
    "nllm",
    "grill_cos",
    "grill_cos2",
    "grill_l2C"
]

AllAttckTypes = [
    "BSA",
    "DRA",
    "FDA",
    "SSPA",
    "EGA",
    "CE",
    "GRILL-cos",
    "GRILL-cos2",
    "GRILL-L2C"
]

grill_attack_types = [
    "grill_cos",
    "grill_cos2",
    "grill_l2C"
]


def get_adv_output_path(attck_type, attackSample):
    if attck_type == "saa_loop":
        return (
            f"qwen/outputsStorageImagenet/advOutputs/{attackSample}/"
            f"advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
            f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
            f"num_steps_{num_steps}_towardsNull_{towardsNull}_"
            f"{whichMLP}_{whichMLPVis}_{chosenLanLayers}_{chosenVisLayers}.txt"
        )

    elif attck_type == "ega":
        return (
            f"qwen/outputsStorageImagenet/advOutputs/{attackSample}/"
            f"advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
            f"num_steps_{num_steps}_ratio_{ega_ratio}.txt"
        )

    elif attck_type == "grill_l2C" and vision_weight == 2.0:
        last_k = 4
        vision_start = 8
        vision_end = 28
        lang_start = 10

        return (
            f"qwen/outputsStorageImagenet/advOutputs/{attackSample}/"
            f"advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
            f"num_steps_{num_steps}_vw_{vision_weight}_lastk_{last_k}"
            f"_vs_{vision_start}_ve_{vision_end}_ls_{lang_start}.txt"
        )

    elif attck_type == "grill_l2C" and vision_weight == 1.0:
        last_k = 8
        vision_start = 8
        vision_end = 28
        lang_start = 10

        return (
            f"qwen/outputsStorageImagenet/advOutputs/{attackSample}/"
            f"advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
            f"num_steps_{num_steps}_vw_{vision_weight}_lastk_{last_k}"
            f"_vs_{vision_start}_ve_{vision_end}_ls_{lang_start}.txt"
        )

    else:
        return (
            f"qwen/outputsStorageImagenet/advOutputs/{attackSample}/"
            f"advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
            f"num_steps_{num_steps}_.txt"
        )


scores_by_attack = {attack: {} for attack in all_attck_types}
missing_files = []

type_sampleAggP = []
type_sampleAggR = []
type_samleAggF1 = []


for attck_type in all_attck_types:
    sampleAggP = []
    sampleAggR = []
    samleAggF1 = []

    print("\nattck_type:", attck_type)

    for attackSample in range(1, numSamplesConsidered):
        advOutputPath = get_adv_output_path(attck_type, attackSample)

        cleanOutputPath = (
            f"qwen/outputsStorageImagenet/advOutputs/{attackSample}/"
            f"cleanOutput.txt"
        )

        if not os.path.exists(advOutputPath):
            missing_files.append((attck_type, attackSample, advOutputPath))
            print(f"Skipping missing adv file: sample={attackSample}, attack={attck_type}")
            continue

        if not os.path.exists(cleanOutputPath):
            missing_files.append(("clean", attackSample, cleanOutputPath))
            print(f"Skipping missing clean file: sample={attackSample}")
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

        p_val = P.item()
        r_val = R.item()
        f1_val = F1.item()

        sampleAggP.append(p_val)
        sampleAggR.append(r_val)
        samleAggF1.append(f1_val)

        scores_by_attack[attck_type][attackSample] = {
            "P": p_val,
            "R": r_val,
            "F1": f1_val
        }

    sampleAggP = np.array(sampleAggP)
    sampleAggR = np.array(sampleAggR)
    samleAggF1 = np.array(samleAggF1)

    print("sampleAggP.shape:", sampleAggP.shape)
    print("sampleAggR.shape:", sampleAggR.shape)
    print("samleAggF1.shape:", samleAggF1.shape)

    type_sampleAggP.append(sampleAggP)
    type_sampleAggR.append(sampleAggR)
    type_samleAggF1.append(samleAggF1)


print("\nFinished BERTScore calculation.")
print("len(type_sampleAggP):", len(type_sampleAggP))
print("len(type_sampleAggR):", len(type_sampleAggR))
print("len(type_samleAggF1):", len(type_samleAggF1))


# ---------------------------------------------------------
# Compare GRILL attacks against BSA
# ---------------------------------------------------------

print("\n====================================================")
print("Samples where GRILL-based attacks outperform BSA")
print("Criterion: GRILL F1 < BSA F1")
print("Lower BERT F1 means stronger attack.")
print("====================================================\n")

grill_outperforms_bsa = {
    attack: [] for attack in grill_attack_types
}

grill_comparison_details = {
    attack: [] for attack in grill_attack_types
}

for grill_attack in grill_attack_types:
    print(f"\nChecking {grill_attack} vs BSA")

    for sample_idx, grill_scores in scores_by_attack[grill_attack].items():
        if sample_idx not in scores_by_attack["bsa"]:
            continue

        bsa_f1 = scores_by_attack["bsa"][sample_idx]["F1"]
        grill_f1 = grill_scores["F1"]

        if grill_f1 < bsa_f1:
            diff = bsa_f1 - grill_f1

            grill_outperforms_bsa[grill_attack].append(sample_idx)
            grill_comparison_details[grill_attack].append({
                "sample_idx": sample_idx,
                "bsa_f1": bsa_f1,
                "grill_f1": grill_f1,
                "diff": diff
            })

            print(
                f"Sample {sample_idx}: {grill_attack} outperforms BSA | "
                f"BSA F1={bsa_f1:.4f}, "
                f"{grill_attack} F1={grill_f1:.4f}, "
                f"diff={diff:.4f}"
            )


print("\n====================================================")
print("Summary: sample indices per GRILL attack")
print("====================================================")

for grill_attack, sample_indices in grill_outperforms_bsa.items():
    print(f"{grill_attack}: {sample_indices}")


all_grill_better_samples = sorted(
    set(
        idx
        for indices in grill_outperforms_bsa.values()
        for idx in indices
    )
)

print("\nUnique sample indices where any GRILL attack outperforms BSA:")
print(all_grill_better_samples)


# ---------------------------------------------------------
# Optional: also print samples where all GRILL attacks beat BSA
# ---------------------------------------------------------

sets_for_available_grill = [
    set(indices)
    for indices in grill_outperforms_bsa.values()
    if len(indices) > 0
]

if len(sets_for_available_grill) > 0:
    samples_where_all_available_grill_beat_bsa = sorted(
        set.intersection(*sets_for_available_grill)
    )
else:
    samples_where_all_available_grill_beat_bsa = []

print("\nSample indices where all available GRILL attacks outperform BSA:")
print(samples_where_all_available_grill_beat_bsa)


# ---------------------------------------------------------
# Save comparison details as text
# ---------------------------------------------------------

save_dir = "qwen/AllPlots/comparisionGRILLCheck"
os.makedirs(save_dir, exist_ok=True)

comparison_txt_path = (
    f"{save_dir}/grill_vs_bsa_indices_"
    f"lr_{lr}_eps_{epsilon}_num_steps_{num_steps}_"
    f"numSamples_{numSamplesConsidered}.txt"
)

with open(comparison_txt_path, "w") as f:
    f.write("Samples where GRILL-based attacks outperform BSA\n")
    f.write("Criterion: GRILL F1 < BSA F1\n\n")

    for grill_attack, details in grill_comparison_details.items():
        f.write(f"{grill_attack}\n")
        f.write("-" * 50 + "\n")

        if len(details) == 0:
            f.write("No samples found.\n\n")
            continue

        for item in details:
            f.write(
                f"Sample {item['sample_idx']}: "
                f"BSA F1={item['bsa_f1']:.6f}, "
                f"{grill_attack} F1={item['grill_f1']:.6f}, "
                f"diff={item['diff']:.6f}\n"
            )

        f.write("\n")

    f.write("Summary indices:\n")
    for grill_attack, sample_indices in grill_outperforms_bsa.items():
        f.write(f"{grill_attack}: {sample_indices}\n")

    f.write("\nUnique sample indices where any GRILL attack outperforms BSA:\n")
    f.write(str(all_grill_better_samples) + "\n")

    f.write("\nSample indices where all available GRILL attacks outperform BSA:\n")
    f.write(str(samples_where_all_available_grill_beat_bsa) + "\n")


print(f"\nSaved GRILL vs BSA comparison to:\n{comparison_txt_path}")


# ---------------------------------------------------------
# Print missing files
# ---------------------------------------------------------

print("\n====================================================")
print("Missing files skipped")
print("====================================================")
print(f"Total missing files: {len(missing_files)}")

for item in missing_files:
    print(item)


# ---------------------------------------------------------
# Boxplots
# ---------------------------------------------------------

valid_plot_data_P = []
valid_plot_data_R = []
valid_plot_data_F1 = []
valid_plot_labels = []

for label, p_data, r_data, f1_data in zip(
    AllAttckTypes,
    type_sampleAggP,
    type_sampleAggR,
    type_samleAggF1
):
    if len(f1_data) == 0:
        print(f"Skipping plot for {label} because no valid samples were found.")
        continue

    valid_plot_labels.append(label)
    valid_plot_data_P.append(p_data)
    valid_plot_data_R.append(r_data)
    valid_plot_data_F1.append(f1_data)


if len(valid_plot_data_F1) > 0:
    plt.figure(figsize=(5, 3))
    plt.boxplot(valid_plot_data_P, labels=valid_plot_labels)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("BERT Precision score")
    plt.xlabel("Attack Type")
    plt.title("Distribution of Precision across Attack Types")
    plt.tight_layout()
    plt.savefig(
        f"{save_dir}/precision_boxplot_"
        f"lr_{lr}_eps_{epsilon}_num_steps_{num_steps}_"
        f"numSamples_{numSamplesConsidered}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()

    plt.figure(figsize=(5, 3))
    plt.boxplot(valid_plot_data_R, labels=valid_plot_labels)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("BERT Recall score")
    plt.xlabel("Attack Type")
    plt.title("Distribution of Recall across Attack Types")
    plt.tight_layout()
    plt.savefig(
        f"{save_dir}/recall_boxplot_"
        f"lr_{lr}_eps_{epsilon}_num_steps_{num_steps}_"
        f"numSamples_{numSamplesConsidered}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()

    plt.figure(figsize=(5, 3))
    plt.boxplot(valid_plot_data_F1, labels=valid_plot_labels)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("BERT F1 score")
    plt.xlabel("Attack Type")
    plt.title("Distribution of F1 score across Attack Types")
    plt.tight_layout()
    plt.savefig(
        f"{save_dir}/f1score_boxplot_"
        f"lr_{lr}_eps_{epsilon}_num_steps_{num_steps}_"
        f"numSamples_{numSamplesConsidered}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()

else:
    print("No valid data available for plotting.")