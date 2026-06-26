
'''
export CUDA_VISIBLE_DEVICES=2
conda deactivate
cd interpretAttacks/
conda activate vlmAttack
export PYTHONNOUSERSITE=1

python qwen/AlignmentTrackerPlots.py

'''


'''
# Vision + language layers together (VisionLayerTrack == LanLayerTrack)
for LayerTrackNum in $(seq 0 31); do
    python qwen/AlignmentTrackerPlots.py --attck_type track_cos --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 550 --AttackStartLayer $LayerTrackNum --numLayerstAtAtime 1 --VisionLayerTrack $LayerTrackNum --LanLayerTrack $LayerTrackNum --kthSingVec -1
    python qwen/AlignmentTrackerPlots.py --attck_type track_cos --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 550 --AttackStartLayer $LayerTrackNum --numLayerstAtAtime 1 --VisionLayerTrack $LayerTrackNum --LanLayerTrack $LayerTrackNum --kthSingVec 0
done

# Language-only layers (fix VisionLayerTrack=31)
for LayerTrackNum in $(seq 32 35); do
    python qwen/AlignmentTrackerPlots.py --attck_type track_cos --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 550 --AttackStartLayer $LayerTrackNum --numLayerstAtAtime 1 --VisionLayerTrack 31 --LanLayerTrack $LayerTrackNum --kthSingVec -1
    python qwen/AlignmentTrackerPlots.py --attck_type track_cos --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 550 --AttackStartLayer $LayerTrackNum --numLayerstAtAtime 1 --VisionLayerTrack 31 --LanLayerTrack $LayerTrackNum --kthSingVec 0
done
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--attck_type",         type=str,   default="track_cos")
parser.add_argument("--desired_norm_l_inf",  type=float, default=0.02)
parser.add_argument("--learningRate",        type=float, default=1e-3)
parser.add_argument("--num_steps",           type=int,   default=1000)
parser.add_argument("--attackSample",        type=str,   default="nature")
parser.add_argument("--AttackStartLayer",    type=int,   default=0)
parser.add_argument("--numLayerstAtAtime",   type=int,   default=1)
parser.add_argument("--VisionLayerTrack",    type=int,   default=0)
parser.add_argument("--LanLayerTrack",       type=int,   default=0)
parser.add_argument("--kthSingVec",          type=int,   default=0)
args = parser.parse_args()

attck_type        = args.attck_type
epsilon           = float(args.desired_norm_l_inf)
lr                = float(args.learningRate)
num_steps         = int(args.num_steps)
attackSample      = str(args.attackSample)
AttackStartLayer  = int(args.AttackStartLayer)
numLayerstAtAtime = int(args.numLayerstAtAtime)
VisionLayerTrack  = int(args.VisionLayerTrack)
LanLayerTrack     = int(args.LanLayerTrack)
kthSingVec        = int(args.kthSingVec)

# ── load alignment file (shape: num_snapshots x 13) ──────────────────────────
align_path = (
    f"qwen/outputsStorageImagenet/Alignments/"
    f"{attackSample}_align_{attck_type}_lr_{lr}_eps_{epsilon}"
    f"_start_{AttackStartLayer}_n_{numLayerstAtAtime}_steps_{num_steps}"
    f"_visL_{VisionLayerTrack}_lanL_{LanLayerTrack}_k_{kthSingVec}.npy"
)

test = np.load(align_path)           # (num_snapshots, 13)
steps, n_points = test.shape
print("test.shape", test.shape)

# Column layout (13 total):
#  0  vis q        3  vis proj     6  merger
#  1  vis k        4  vis gate     7  lan q
#  2  vis v        5  vis down     8  lan k
#                                  9  lan v
#                                 10  lan gate
#                                 11  lan up
#                                 12  lan down

point_labels_all = [
    "query (vis)",        # 0
    "key (vis)",          # 1
    "value (vis)",        # 2
    "att output (vis)",   # 3
    "MLP gate (vis)",     # 4
    "MLP down (vis)",     # 5
    "Vis-to-lan",         # 6
    "query (lan)",        # 7
    "key (lan)",          # 8
    "value (lan)",        # 9
    "MLP gate (lan)",     # 10
    "MLP up (lan)",       # 11
    "MLP down (lan)",     # 12
]

# ── plot settings ─────────────────────────────────────────────────────────────
BACK_MARGIN       = 1.5
show_shadow_on_XZ = True
show_shadow_on_XY = True
drop_start        = True
drop_end          = True

# ── helper: one 3-D waterfall plot ───────────────────────────────────────────
def make_waterfall(data, labels, title_suffix, save_path):
    s, np_ = data.shape
    z_plane   = float(data.min())
    y_plane   = (np_ - 1) + BACK_MARGIN

    plt.style.use("default")
    fig = plt.figure(figsize=(10, 6))
    ax  = fig.add_subplot(111, projection="3d")
    ax.tick_params(labelsize=16)

    x      = np.linspace(0, num_steps, s)
    colors = cm.tab20(np.linspace(0, 1, np_))

    for j in range(np_):
        y = np.full(s, j)
        z = data[:, j]
        c = colors[j]

        ax.plot(x, y, z, linewidth=4.2, color=c, alpha=0.95)

        if show_shadow_on_XZ:
            ax.plot(x, np.full(s, y_plane), z, linewidth=1.2, color=c, alpha=0.22)
        if show_shadow_on_XY:
            ax.plot(x, y, np.full(s, z_plane), linewidth=1.2, color=c, alpha=0.22)

        def drop_both(xp, yp, zp):
            ax.plot([xp, xp], [yp, y_plane], [zp, zp], color=c, alpha=0.7, linewidth=1.2)
            ax.plot([xp, xp], [yp, yp], [zp, z_plane], color=c, alpha=0.7, linewidth=1.2)

        if drop_start: drop_both(x[0],  j, z[0])
        if drop_end:   drop_both(x[-1], j, z[-1])

        ax.scatter([x[0], x[-1]], [j, j], [z[0], z[-1]], color=c, s=22, alpha=0.9)

    ax.set_xlabel("Step",      fontsize=16, labelpad=10)
    ax.set_zlabel("Alignment", fontsize=16, labelpad=10)
    ax.set_zlim(0, 0.7)
    ax.set_ylim(-0.5, y_plane + 0.5)

    ax.set_yticks(np.arange(np_))
    ax.set_yticklabels([])
    ax.tick_params(axis='y', length=0)

    x_label = x[-1] + 100
    for j, lab in enumerate(labels):
        ax.text(x_label, j, z_plane, lab, ha="left", va="center", fontsize=16)

    ax.view_init(elev=25, azim=-65)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", save_path)


# ── save dirs ─────────────────────────────────────────────────────────────────
base_tag = (f"S_{attackSample}_{attck_type}_lr_{lr}_eps_{epsilon}"
            f"_ASL{AttackStartLayer}_NLAT{numLayerstAtAtime}"
            f"_NS{num_steps}_visL{VisionLayerTrack}_lanL{LanLayerTrack}_k{kthSingVec}")

# ── Plot 1: all 13 columns ────────────────────────────────────────────────────
save_dir_all = (f"qwen/outputsStorageImagenet/AlignmentPlots/"
                f"allLayers_AttackStartLayer_{AttackStartLayer}")
os.makedirs(save_dir_all, exist_ok=True)
make_waterfall(
    data      = test,
    labels    = point_labels_all,
    title_suffix = "all layers",
    save_path = os.path.join(save_dir_all, f"{base_tag}_all.png")
)

# ── Plot 2: vision only (cols 0-5) ───────────────────────────────────────────
save_dir_vis = (f"qwen/outputsStorageImagenet/AlignmentPlots/"
                f"visLayers_AttackStartLayer_{AttackStartLayer}")
os.makedirs(save_dir_vis, exist_ok=True)
make_waterfall(
    data      = test[:, :6],
    labels    = point_labels_all[:6],
    title_suffix = "vision layers",
    save_path = os.path.join(save_dir_vis, f"{base_tag}_vis.png")
)

# ── Plot 3: language only (cols 6-12, mirrors Gemma text-layer plot) ──────────
save_dir_lan = (f"qwen/outputsStorageImagenet/AlignmentPlots/"
                f"textLayers_AttackStartLayer_{AttackStartLayer}")
os.makedirs(save_dir_lan, exist_ok=True)
make_waterfall(
    data      = test[:, 6:],
    labels    = point_labels_all[6:],
    title_suffix = "language layers",
    save_path = os.path.join(save_dir_lan, f"{base_tag}_lan.png")
)