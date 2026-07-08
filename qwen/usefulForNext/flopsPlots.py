

'''
python qwen/flopsPlots.py



'''

import os
import matplotlib.pyplot as plt

# Data
methods = ["BSA", "GRILL-cos", "DRA", "EGA", "FDA", "SSP", "NLL", "SSPMA"]
values = [
    29423739941824,
    14004157656213,
    14204866649119,
    26764193043082,
    6139307805634,
    6139183765772,
    26851063319698,
    13090508399113
]

# Save directory
save_dir = "/home/luser/interpretAttacks/qwen/AllPlots/flops"
os.makedirs(save_dir, exist_ok=True)

# File path
save_path = os.path.join(save_dir, "flops_bar_chart.png")

# Create figure
plt.figure(figsize=(12, 6))

# Bar chart
bars = plt.bar(methods, values)

# Labels and title
plt.xlabel("Methods")
plt.ylabel("FLOPs")
plt.title("FLOPs Comparison Across Methods")

# Rotate x-axis labels
plt.xticks(rotation=20)

# Add values above bars
for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2e}",
        ha='center',
        va='bottom',
        fontsize=9
    )

# Scientific notation for y-axis
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

# Tight layout
plt.tight_layout()

# Save figure
plt.savefig(save_path, dpi=300, bbox_inches='tight')

# Optional: display plot
plt.show()

print(f"Plot saved to: {save_path}")