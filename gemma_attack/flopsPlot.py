

'''


export CUDA_VISIBLE_DEVICES=1
conda activate gemma3
cd interpretAttacks
python gemma_attack/flopsPlot.py

'''


import matplotlib.pyplot as plt
import numpy as np





# Data
flops = {
    "BSA": 25896860154024,
    "DRA": 10102865928192,
    "FDA": 10103237549901,
    "SSPA": 15614300402710,
    "EGA": 19515596799726,
    "SSPMA-E": 6199382708224,
    "SSPMA-L": 12266970663496,
    "SSPMA-EL": 12270110084825
}



#     "NLLM": 19561743997328

labels = list(flops.keys())
values = np.array(list(flops.values()))

# Scale
scale = 1e13
values_scaled = values / scale

save_dir = "gemma_attack/AllPlots/flopsPlots"

# ---- NeurIPS-style plotting ----
plt.rcParams.update({
    "font.size": 8,          # base font
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8
})

# Figure size: single-column NeurIPS (~3.25 inches wide)
plt.figure(figsize=(3.25, 2.2))

bars = plt.bar(labels, values_scaled)
max_val = max(values_scaled)
plt.ylim(0, max_val * 1.15)
# Labels
plt.ylabel("FLOPs (×10¹³)")
plt.title("FLOPs Comparison")

# Grid (light)
plt.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.5)

# Rotate x labels slightly for readability
plt.xticks(rotation=30, ha='right')

# Optional: add value labels on top (compact)
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.2f}",
        ha='center',
        va='bottom',
        fontsize=6
    )

plt.tight_layout(pad=0.5)

plt.savefig(f"{save_dir}/flopsPlotPaper.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()




gpuAlloc = {
    "BSA": 20661.36,
    "DRA": 12866.08,
    "FDA": 13252.55,
    "SSPA": 19298.89,
    "EGA": 20881.19,
    "SSPMA-E": 12948.40,
    "SSPMA-L": 15651.01,
    "SSPMA-EL": 15220.13

}

#     "NLLM": 19561743997328

labels = list(gpuAlloc.keys())
values = np.array(list(gpuAlloc.values()))

# Scale
scale = 1e3
values_scaled = values / scale

save_dir = "gemma_attack/AllPlots/flopsPlots"

# ---- NeurIPS-style plotting ----
plt.rcParams.update({
    "font.size": 8,          # base font
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8
})

# Figure size: single-column NeurIPS (~3.25 inches wide)
plt.figure(figsize=(3.25, 2.2))

bars = plt.bar(labels, values_scaled)
max_val = max(values_scaled)
plt.ylim(0, max_val * 1.15)
# Labels
plt.ylabel("Memory (GB)")
plt.title("Memory Comparison")

# Grid (light)
plt.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.5)

# Rotate x labels slightly for readability
plt.xticks(rotation=30, ha='right')

# Optional: add value labels on top (compact)
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.2f}",
        ha='center',
        va='bottom',
        fontsize=6
    )

plt.tight_layout(pad=0.5)

plt.savefig(f"{save_dir}/memoryPlotPaper.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()