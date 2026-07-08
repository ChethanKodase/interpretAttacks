
'''
Used to plot tracking trajectory spectral subspace alignment in different text layers


------------------------------------------------------------------------------------------------------------------------------------------------------------------------
550, 1, 2, 3, 4, 5
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=2
conda activate gemma3
cd interpretAttacks
for LayerTrackNum in $(seq 0 26); do
    python gemma_attack/AlignmentTrackerPlots1TextLayers.py --attck_type track_cos --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 2 --AttackStartLayer $LayerTrackNum --numLayerstAtAtime 1 --VisionLayerTrack $LayerTrackNum --LanLayerTrack $LayerTrackNum --kthSingVec -1
    python gemma_attack/AlignmentTrackerPlots1TextLayers.py --attck_type track_cos --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 2 --AttackStartLayer $LayerTrackNum --numLayerstAtAtime 1 --VisionLayerTrack $LayerTrackNum --LanLayerTrack $LayerTrackNum --kthSingVec 0
done

for LayerTrackNum in $(seq 26 33); do
    python gemma_attack/AlignmentTrackerPlots1TextLayers.py --attck_type track_cos --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 2 --AttackStartLayer $LayerTrackNum --numLayerstAtAtime 1 --VisionLayerTrack 26 --LanLayerTrack $LayerTrackNum --kthSingVec -1
    python gemma_attack/AlignmentTrackerPlots1TextLayers.py --attck_type track_cos --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 2 --AttackStartLayer $LayerTrackNum --numLayerstAtAtime 1 --VisionLayerTrack 26 --LanLayerTrack $LayerTrackNum --kthSingVec 0
done

for LayerTrackNum in $(seq 0 26); do
    python gemma_attack/AlignmentTrackerPlots1TextLayers.py --attck_type track_cos --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 3 --AttackStartLayer $LayerTrackNum --numLayerstAtAtime 1 --VisionLayerTrack $LayerTrackNum --LanLayerTrack $LayerTrackNum --kthSingVec -1
    python gemma_attack/AlignmentTrackerPlots1TextLayers.py --attck_type track_cos --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 3 --AttackStartLayer $LayerTrackNum --numLayerstAtAtime 1 --VisionLayerTrack $LayerTrackNum --LanLayerTrack $LayerTrackNum --kthSingVec 0
done

for LayerTrackNum in $(seq 26 33); do
    python gemma_attack/AlignmentTrackerPlots1TextLayers.py --attck_type track_cos --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 3 --AttackStartLayer $LayerTrackNum --numLayerstAtAtime 1 --VisionLayerTrack 26 --LanLayerTrack $LayerTrackNum --kthSingVec -1
    python gemma_attack/AlignmentTrackerPlots1TextLayers.py --attck_type track_cos --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 3 --AttackStartLayer $LayerTrackNum --numLayerstAtAtime 1 --VisionLayerTrack 26 --LanLayerTrack $LayerTrackNum --kthSingVec 0
done

for LayerTrackNum in $(seq 0 26); do
    python gemma_attack/AlignmentTrackerPlots1TextLayers.py --attck_type track_cos --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 4 --AttackStartLayer $LayerTrackNum --numLayerstAtAtime 1 --VisionLayerTrack $LayerTrackNum --LanLayerTrack $LayerTrackNum --kthSingVec -1
    python gemma_attack/AlignmentTrackerPlots1TextLayers.py --attck_type track_cos --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 4 --AttackStartLayer $LayerTrackNum --numLayerstAtAtime 1 --VisionLayerTrack $LayerTrackNum --LanLayerTrack $LayerTrackNum --kthSingVec 0
done

for LayerTrackNum in $(seq 26 33); do
    python gemma_attack/AlignmentTrackerPlots1TextLayers.py --attck_type track_cos --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 4 --AttackStartLayer $LayerTrackNum --numLayerstAtAtime 1 --VisionLayerTrack 26 --LanLayerTrack $LayerTrackNum --kthSingVec -1
    python gemma_attack/AlignmentTrackerPlots1TextLayers.py --attck_type track_cos --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 4 --AttackStartLayer $LayerTrackNum --numLayerstAtAtime 1 --VisionLayerTrack 26 --LanLayerTrack $LayerTrackNum --kthSingVec 0
done

for LayerTrackNum in $(seq 0 26); do
    python gemma_attack/AlignmentTrackerPlots1TextLayers.py --attck_type track_cos --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 5 --AttackStartLayer $LayerTrackNum --numLayerstAtAtime 1 --VisionLayerTrack $LayerTrackNum --LanLayerTrack $LayerTrackNum --kthSingVec -1
    python gemma_attack/AlignmentTrackerPlots1TextLayers.py --attck_type track_cos --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 5 --AttackStartLayer $LayerTrackNum --numLayerstAtAtime 1 --VisionLayerTrack $LayerTrackNum --LanLayerTrack $LayerTrackNum --kthSingVec 0
done

for LayerTrackNum in $(seq 26 33); do
    python gemma_attack/AlignmentTrackerPlots1TextLayers.py --attck_type track_cos --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 5 --AttackStartLayer $LayerTrackNum --numLayerstAtAtime 1 --VisionLayerTrack 26 --LanLayerTrack $LayerTrackNum --kthSingVec -1
    python gemma_attack/AlignmentTrackerPlots1TextLayers.py --attck_type track_cos --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 5 --AttackStartLayer $LayerTrackNum --numLayerstAtAtime 1 --VisionLayerTrack 26 --LanLayerTrack $LayerTrackNum --kthSingVec 0
done

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------------------------------------------------------------


------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# can be good results
for LayerTrackNum in $(seq 0 20); do
    python gemma_attack/AlignmentTrackerPlots1TextLayers.py --attck_type grill_wass --desired_norm_l_inf 0.02 --learningRate 0.0001 --num_steps 10000 --attackSample 550 --AttackStartLayer 15 --numLayerstAtAtime 1 --VisionLayerTrack $LayerTrackNum --LanLayerTrack $LayerTrackNum --kthSingVec 0
    python gemma_attack/AlignmentTrackerPlots1TextLayers.py --attck_type grill_wass --desired_norm_l_inf 0.02 --learningRate 0.0001 --num_steps 10000 --attackSample 550 --AttackStartLayer 15 --numLayerstAtAtime 1 --VisionLayerTrack $LayerTrackNum --LanLayerTrack $LayerTrackNum --kthSingVec -1
done

#checkj
export CUDA_VISIBLE_DEVICES=5
conda activate gemma3
cd interpretAttacks
for LayerTrackNum in $(seq 0 20); do
    python gemma_attack/AlignmentTrackerPlots1TextLayers.py --attck_type grill_wass --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 550 --AttackStartLayer 33 --numLayerstAtAtime 1 --VisionLayerTrack $LayerTrackNum --LanLayerTrack $LayerTrackNum --kthSingVec 0
    python gemma_attack/AlignmentTrackerPlots1TextLayers.py --attck_type grill_wass --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 550 --AttackStartLayer 33 --numLayerstAtAtime 1 --VisionLayerTrack $LayerTrackNum --LanLayerTrack $LayerTrackNum --kthSingVec -1
done


export CUDA_VISIBLE_DEVICES=5
conda activate gemma3
cd interpretAttacks
for LayerTrackNum in $(seq 0 33); do
    python gemma_attack/AlignmentTrackerPlots1TextLayers.py --attck_type grill_wass --desired_norm_l_inf 0.02 --learningRate 0.0001 --num_steps 10000 --attackSample 550 --AttackStartLayer 15 --numLayerstAtAtime 1 --VisionLayerTrack $LayerTrackNum --LanLayerTrack $LayerTrackNum --kthSingVec 0
    python gemma_attack/AlignmentTrackerPlots1TextLayers.py --attck_type grill_wass --desired_norm_l_inf 0.02 --learningRate 0.0001 --num_steps 10000 --attackSample 550 --AttackStartLayer 15 --numLayerstAtAtime 1 --VisionLayerTrack $LayerTrackNum --LanLayerTrack $LayerTrackNum --kthSingVec -1
done


export CUDA_VISIBLE_DEVICES=6
conda activate gemma3
cd interpretAttacks
for LayerTrackNum in $(seq 20 26); do
    python gemma_attack/AlignmentTrackerPlots1TextLayers.py --attck_type bsaTrack --desired_norm_l_inf 0.02 --learningRate 0.0001 --num_steps 1000 --attackSample 550 --AttackStartLayer 15 --numLayerstAtAtime 1 --VisionLayerTrack $LayerTrackNum --LanLayerTrack $LayerTrackNum --kthSingVec 0
    python gemma_attack/AlignmentTrackerPlots1TextLayers.py --attck_type bsaTrack --desired_norm_l_inf 0.02 --learningRate 0.0001 --num_steps 1000 --attackSample 550 --AttackStartLayer 15 --numLayerstAtAtime 1 --VisionLayerTrack $LayerTrackNum --LanLayerTrack $LayerTrackNum --kthSingVec -1
done


for LayerTrackNum in $(seq 26 33); do
    python gemma_attack/AlignmentTrackerPlots1TextLayers.py --attck_type bsaTrack --desired_norm_l_inf 0.02 --learningRate 0.0001 --num_steps 1000 --attackSample 550 --AttackStartLayer 15 --numLayerstAtAtime 1 --VisionLayerTrack 26 --LanLayerTrack $LayerTrackNum --kthSingVec 0
    python gemma_attack/AlignmentTrackerPlots1TextLayers.py --attck_type bsaTrack --desired_norm_l_inf 0.02 --learningRate 0.0001 --num_steps 1000 --attackSample 550 --AttackStartLayer 15 --numLayerstAtAtime 1 --VisionLayerTrack 26 --LanLayerTrack $LayerTrackNum --kthSingVec -1
done

python gemma_attack/RealSpectralSubspaceAlignmentTracker.py --attck_type bsaTrack --desired_norm_l_inf 0.05 --learningRate 0.0001 --num_steps 1000 --attackSample 550 --AttackStartLayer 15 --numLayerstAtAtime 1 --VisionLayerTrack 0 --LanLayerTrack 0 --kthSingVec -1


'''


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse

parser = argparse.ArgumentParser(description="Gemma-3 ORIGINAL-image-space adversarial attack (no squeeze)")
parser.add_argument("--attck_type", type=str, default="grill_l2",
                    help="grill_l2 | grill_cos | OA_l2 | OA_cos")
parser.add_argument("--desired_norm_l_inf", type=float, default=0.03,
                    help="epsilon L_inf in ORIGINAL pixel space [0..1]. Try 0.01~0.08")
parser.add_argument("--learningRate", type=float, default=1e-3,
                    help="Adam learning rate")
parser.add_argument("--num_steps", type=int, default=2000,
                    help="Number of Adam steps")
parser.add_argument("--attackSample", type=str, default="nature",
                    help="which sample")
parser.add_argument("--AttackStartLayer", type=int, default=0,
                    help="From which layer do you start attack")
parser.add_argument("--numLayerstAtAtime", type=int, default=2,
                    help="Number of layers taken at a time to attack")
parser.add_argument("--VisionLayerTrack", type=int, default=2,
                    help="which vision layer you want to track")
parser.add_argument("--LanLayerTrack", type=int, default=2,
                    help="which language layer you want to track")
parser.add_argument("--kthSingVec", type=int, default=0,
                    help="Among the k singular vectors which one do you want")

args = parser.parse_args()

attck_type = args.attck_type
epsilon = float(args.desired_norm_l_inf)
lr = float(args.learningRate)
num_steps = int(args.num_steps)
attackSample = str(args.attackSample)
AttackStartLayer = int(args.AttackStartLayer)
numLayerstAtAtime = int(args.numLayerstAtAtime)
VisionLayerTrack = int(args.VisionLayerTrack)
LanLayerTrack = int(args.LanLayerTrack)
kthSingVec = int(args.kthSingVec)

RightAlignMentTrackerPath = (
    f"gemma_attack/outputsStorageImagenet/Alignments/"
    f"{attackSample}_RightAlignment_adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
    f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_num_steps_{num_steps}_"
    f"VisionLayerTrack_{VisionLayerTrack}_LanLayerTrack_{LanLayerTrack}_kthSingVec_{kthSingVec}_.npy"
)

test = np.load(RightAlignMentTrackerPath)

test = test[:,7:]
steps, n_points = test.shape

print(" test.shape",  test.shape)

save_dir = f"/data1/chethan/interpretAttacks/gemma_attack/AllPlots/AlignmentPlots/textLayers_AttackStartLayer_{AttackStartLayer}"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(
    save_dir,
    f"S_{attackSample}RtAl{attck_type}_lr_{lr}_"
    f"eps_{epsilon}ASL{AttackStartLayer}NLAT{numLayerstAtAtime}_"
    f"NS{num_steps}_VisionLayerTrack_{VisionLayerTrack}_LanLayerTrack_{LanLayerTrack}_"
    f"kthSingVec_{kthSingVec}_.png"
)

# -----------------------------
# Planes you want to drop onto
# -----------------------------
z_plane = float(test.min())  # XY plane (floor). Or set to 0.0 if preferred.

BACK_MARGIN = 1.5
y_plane = (n_points - 1) + BACK_MARGIN  # back XZ plane location

show_shadow_on_XZ = True
show_shadow_on_XY = True
drop_start = True
drop_end = True

point_labels = [
    "query (vis)", "key (vis)", "value (vis)", "att output (vis)",
    "MLP exp (vis)", "MLP out (vis)", "Vis-to-lan", "query",
    "key", "value", "MLP gate", "MLP up",
    "MLP down"
]

point_labels = point_labels[7:]
#print("point_labels", point_labels)


plt.style.use("default")
#fig = plt.figure(figsize=(18, 13))
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")

#------------------------------------------------------------------------------------
ax.tick_params(labelsize=16)

#----------------------------------------------------------------------------------------------------------------

#x = np.arange(steps)*20
x = np.linspace(0, 1000, steps)
colors = cm.tab20(np.linspace(0, 1, n_points))

for j in range(n_points):
    y = np.full(steps, j)
    z = test[:, j]
    c = colors[j]

    # Main ribbon
    ax.plot(x, y, z, linewidth=4.2, color=c, alpha=0.95)

    # Shadows (optional)
    if show_shadow_on_XZ:
        ax.plot(x, np.full(steps, y_plane), z, linewidth=1.2, color=c, alpha=0.22)
    if show_shadow_on_XY:
        ax.plot(x, y, np.full(steps, z_plane), linewidth=1.2, color=c, alpha=0.22)

    def drop_both(xp, yp, zp):
        # Perpendicular to XZ plane at y=y_plane (along y)
        ax.plot([xp, xp], [yp, y_plane], [zp, zp], color=c, alpha=0.7, linewidth=1.2)
        # Perpendicular to XY plane at z=z_plane (along z)
        ax.plot([xp, xp], [yp, yp], [zp, z_plane], color=c, alpha=0.7, linewidth=1.2)

    if drop_start:
        drop_both(x[0], j, z[0])
    if drop_end:
        drop_both(x[-1], j, z[-1])

    ax.scatter([x[0], x[-1]], [j, j], [z[0], z[-1]], color=c, s=22, alpha=0.9)

#ax.set_title("3D Waterfall with drops to BACK XZ plane and to XY floor", pad=18)
'''ax.set_xlabel("Step")
ax.set_ylabel("Layer type", labelpad=20)
ax.set_zlabel("Alignment")'''

ax.set_xlabel("Step", fontsize=16, labelpad=10)
#ax.set_ylabel("Layer Type", fontsize=14, labelpad=40)
ax.set_zlabel("Alignment", fontsize=16, labelpad=10)

ax.set_zlim(0, 0.7)
# Extend axis so the back plane is in view
ax.set_ylim(-0.5, y_plane + 0.5)

# -----------------------------
# OPTION A: Custom Y labels (fixes mplot3d tick-label projection shift)
# -----------------------------
# Keep ticks at the correct y locations but hide the built-in y tick labels
ax.set_yticks(np.arange(n_points))
ax.set_yticklabels([])              # hide default labels (these "look shifted")
ax.tick_params(axis='y', length=0)  # optional: hide tick marks too (keep grid)

# Put your own labels at EXACT y=j coordinates.
# Choose an anchor position slightly "outside" the plotted box:
#x_label = x[-1] + max(10, int(0.03 * steps))  # a bit to the right of last step
x_label = x[-1] + 100 
z_label = z_plane                              # on the floor plane

for j, lab in enumerate(point_labels):
    ax.text(
        x_label, j, z_label, lab,
        ha="left", va="center", fontsize=16
    )

# View
ax.view_init(elev=25, azim=-65)

#plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.close(fig)

print("Saved:", save_path)
print(f"XZ projection plane at y={y_plane:.2f} (back), XY floor at z={z_plane:.6g}")