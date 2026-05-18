
'''


export CUDA_VISIBLE_DEVICES=3
conda deactivate
cd interpretAttacks/
conda activate vlmAttack
export PYTHONNOUSERSITE=1
python qwen/layerAnalysis.py
'''


import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from bert_score import score
from tqdm import tqdm

# =========================
# CONFIGURATION
# =========================
BASE_DIR = "/mdadm0/chethan_krishnamurth/interpretAttacks/qwen/outputsStorageImagenet/advOutputs"
NUM_SAMPLES = 50
LAYERS = list(range(28))  # AttackStartLayer_0 to AttackStartLayer_27
LR = "0.001"
EPS = "0.04"

# =========================
# FILE PATTERNS
# =========================
# Adversarial file pattern
ADV_PATTERN = (
    "advOutput_attackType_grill_wass_"
    "lr_{lr}_eps_{eps}_"
    "AttackStartLayer_{layer}_"
    "numLayerstAtAtime_1_num_steps_1000_.txt"
)

# Clean output filename
CLEAN_FILENAME = "cleanOutput.txt"

# =========================
# HELPERS
# =========================
def read_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


def find_clean_file(sample_dir):
    """
    Clean output location:
    <sample_dir>/cleanOutput.txt
    """

    clean_path = os.path.join(sample_dir, CLEAN_FILENAME)

    if os.path.exists(clean_path):
        return clean_path

    return None


# =========================
# STORAGE
# =========================
all_precision = defaultdict(list)
all_recall = defaultdict(list)
all_f1 = defaultdict(list)

# =========================
# MAIN LOOP
# =========================
for sample_idx in tqdm(range(1, NUM_SAMPLES + 1), desc="Samples"):

    sample_dir = os.path.join(BASE_DIR, str(sample_idx))

    if not os.path.exists(sample_dir):
        print(f"[WARNING] Missing sample directory: {sample_dir}")
        continue

    clean_file = find_clean_file(sample_dir)

    if clean_file is None:
        print(f"[WARNING] No clean file found in: {sample_dir}")
        continue

    clean_output = read_text(clean_file)

    for layer in LAYERS:

        adv_filename = ADV_PATTERN.format(
            lr=LR,
            eps=EPS,
            layer=layer
        )

        adv_path = os.path.join(sample_dir, adv_filename)

        if not os.path.exists(adv_path):
            print(f"[WARNING] Missing file: {adv_path}")
            continue

        adv_output = read_text(adv_path)

        try:
            P, R, F1 = score(
                [adv_output],
                [clean_output],
                lang="en",
                model_type="roberta-large",
                rescale_with_baseline=True
            )

            p_val = P.item()
            r_val = R.item()
            f1_val = F1.item()

            all_precision[layer].append(p_val)
            all_recall[layer].append(r_val)
            all_f1[layer].append(f1_val)

        except Exception as e:
            print(f"[ERROR] Failed on layer {layer}, sample {sample_idx}: {e}")

# =========================
# COMPUTE STATISTICS
# =========================
layers_present = sorted(all_f1.keys())

precision_mean = []
precision_std = []

recall_mean = []
recall_std = []

f1_mean = []
f1_std = []

for layer in layers_present:
    p_vals = np.array(all_precision[layer])
    r_vals = np.array(all_recall[layer])
    f1_vals = np.array(all_f1[layer])

    precision_mean.append(np.mean(p_vals))
    precision_std.append(np.std(p_vals))

    recall_mean.append(np.mean(r_vals))
    recall_std.append(np.std(r_vals))

    f1_mean.append(np.mean(f1_vals))
    f1_std.append(np.std(f1_vals))

# =========================
# FIND LOWEST BERT SCORE LAYER
# =========================
lowest_f1_idx = int(np.argmin(f1_mean))
lowest_layer = layers_present[lowest_f1_idx]
lowest_value = f1_mean[lowest_f1_idx]

print("\n====================================")
print(f"Lowest Mean BERTScore F1 Layer: {lowest_layer}")
print(f"Mean F1 Score: {lowest_value:.6f}")
print("====================================")

# =========================
# SAVE STATS
# =========================
output_stats = "bertscore_layer_statistics.txt"

with open(output_stats, "w") as f:
    f.write("Layer\tP_mean\tP_std\tR_mean\tR_std\tF1_mean\tF1_std\n")

    for i, layer in enumerate(layers_present):
        f.write(
            f"{layer}\t"
            f"{precision_mean[i]:.6f}\t{precision_std[i]:.6f}\t"
            f"{recall_mean[i]:.6f}\t{recall_std[i]:.6f}\t"
            f"{f1_mean[i]:.6f}\t{f1_std[i]:.6f}\n"
        )

print(f"Saved statistics to: {output_stats}")

# =========================
# PLOTTING
# =========================
plt.figure(figsize=(14, 8))

# Precision
plt.plot(layers_present, precision_mean, label="Precision (Mean)")
plt.fill_between(
    layers_present,
    np.array(precision_mean) - np.array(precision_std),
    np.array(precision_mean) + np.array(precision_std),
    alpha=0.2
)

# Recall
plt.plot(layers_present, recall_mean, label="Recall (Mean)")
plt.fill_between(
    layers_present,
    np.array(recall_mean) - np.array(recall_std),
    np.array(recall_mean) + np.array(recall_std),
    alpha=0.2
)

# F1
plt.plot(layers_present, f1_mean, label="F1 (Mean)", linewidth=3)
plt.fill_between(
    layers_present,
    np.array(f1_mean) - np.array(f1_std),
    np.array(f1_mean) + np.array(f1_std),
    alpha=0.2
)

# Highlight lowest layer
plt.axvline(
    x=lowest_layer,
    linestyle="--",
    linewidth=2,
    label=f"Lowest F1 Layer = {lowest_layer}"
)

plt.xlabel("AttackStartLayer")
plt.ylabel("BERTScore")
plt.title("BERTScore vs AttackStartLayer\nMean ± Standard Deviation Across Samples")
plt.xticks(layers_present)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plot_path = "bertscore_layer_analysis.png"
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"Saved plot to: {plot_path}")

# =========================
# OPTIONAL: INDIVIDUAL DISTRIBUTION PLOTS
# =========================
fig, axes = plt.subplots(3, 1, figsize=(14, 16), sharex=True)

# Precision
axes[0].plot(layers_present, precision_mean)
axes[0].fill_between(
    layers_present,
    np.array(precision_mean) - np.array(precision_std),
    np.array(precision_mean) + np.array(precision_std),
    alpha=0.25
)
axes[0].set_title("Precision Mean ± Std")
axes[0].set_ylabel("Precision")
axes[0].grid(True, alpha=0.3)

# Recall
axes[1].plot(layers_present, recall_mean)
axes[1].fill_between(
    layers_present,
    np.array(recall_mean) - np.array(recall_std),
    np.array(recall_mean) + np.array(recall_std),
    alpha=0.25
)
axes[1].set_title("Recall Mean ± Std")
axes[1].set_ylabel("Recall")
axes[1].grid(True, alpha=0.3)

# F1
axes[2].plot(layers_present, f1_mean, linewidth=3)
axes[2].fill_between(
    layers_present,
    np.array(f1_mean) - np.array(f1_std),
    np.array(f1_mean) + np.array(f1_std),
    alpha=0.25
)
axes[2].axvline(x=lowest_layer, linestyle="--")
axes[2].set_title("F1 Mean ± Std")
axes[2].set_xlabel("AttackStartLayer")
axes[2].set_ylabel("F1")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()

plot_path_2 = "bertscore_distributions.png"
plt.savefig(plot_path_2, dpi=300)
plt.close()

print(f"Saved detailed plots to: {plot_path_2}")

print("\nDone.")