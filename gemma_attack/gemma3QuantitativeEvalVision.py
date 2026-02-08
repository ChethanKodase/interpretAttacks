
'''

export CUDA_VISIBLE_DEVICES=3
conda activate gemma3
cd interpretAttacks
python gemma_attack/gemma3QuantitativeEvalVision.py  --attck_type grill_wass --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1


'''


from bert_score import score
import argparse

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.5,
})

parser = argparse.ArgumentParser(description="Gemma-3 ORIGINAL-image-space adversarial attack (no squeeze)")
parser.add_argument("--attck_type", type=str, default="grill_l2",
                    help="grill_l2 | grill_cos | OA_l2 | OA_cos")
parser.add_argument("--desired_norm_l_inf", type=float, default=0.03,
                    help="epsilon L_inf in ORIGINAL pixel space [0..1]. Try 0.01~0.08")
parser.add_argument("--learningRate", type=float, default=1e-3,
                    help="Adam learning rate")
parser.add_argument("--num_steps", type=int, default=2000,
                    help="Number of Adam steps")
parser.add_argument("--AttackStartLayer", type=int, default=0,
                    help="From which layer do you start attack")
parser.add_argument("--numLayerstAtAtime", type=int, default=2,
                    help="Number of layers taken at a time to attack")



args = parser.parse_args()

attck_type = args.attck_type
epsilon = float(args.desired_norm_l_inf)
lr = float(args.learningRate)
num_steps = int(args.num_steps)
AttackStartLayer = int(args.AttackStartLayer)
numLayerstAtAtime = int(args.numLayerstAtAtime)

numHiddenStates = 28

PmeanList = []
RmeanList = []
F1meanList = []

PstdList = []
RstdList = []
F1stdList = []

for AttackStartLayer in range(numHiddenStates):

    sampleAggP = []
    sampleAggR = []
    samleAggF1 = []
    for attackSample in range(1,13):
        advOutputPath = f"gemma_attack/outputsStorageImagenet/advOutputsVis/{attackSample}/advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_num_steps_{num_steps}_.txt"


        with open(advOutputPath, "r") as f:
            advOutput = [f.read().strip()]
            print("advOutput", advOutput)

        #cleanOutputPath = "/data1/chethan/interpretAttacks/gemma_attack/outputsStorageImagenet/advOutputs/1/cleanOutput.txt"
        cleanOutputPath = f"gemma_attack/outputsStorageImagenet/advOutputsVis/{attackSample}/cleanOutput.txt"

        with open(cleanOutputPath, "r") as f:
            cleanOutput = [f.read().strip()]
            print("cleanOutput", cleanOutput)


        P, R, F1 = score(
            advOutput,
            cleanOutput,
            lang="en",              # language
            model_type="roberta-large",  # standard choice
            rescale_with_baseline=True   # recommended
        )

        print("Precision:", P.item())
        print("Recall:", R.item())
        print("F1:", F1.item())



        sampleAggP.append(P.item())
        sampleAggR.append(R.item())
        samleAggF1.append(F1.item())

    sampleAggP = np.array(sampleAggP)
    sampleAggR = np.array(sampleAggR)
    samleAggF1 = np.array(samleAggF1)

    Pmean = sampleAggP.mean()
    Pstd = sampleAggP.std()

    Rmean = sampleAggR.mean()
    Rstd = sampleAggR.std()

    F1mean = samleAggF1.mean()
    F1std = samleAggF1.std()


    PmeanList.append(Pmean.item())
    RmeanList.append(Rmean.item())
    F1meanList.append(F1mean.item())

    PstdList.append(Pstd.item())
    RstdList.append(Rstd.item())
    F1stdList.append(F1std.item())

ConsideredEndPts = [k+1 for k in range(numHiddenStates)]


plt.figure(figsize=(3.4, 2.6))

PmeanArr = np.array(PmeanList)
PstdArr  = np.array(PstdList)

plt.plot(ConsideredEndPts, PmeanArr, label="Mean Precision")
plt.fill_between(
    ConsideredEndPts,
    PmeanArr - PstdArr,
    PmeanArr + PstdArr,
    alpha=0.3,
    label="±1 Std"
)

plt.xlabel("Hidden state index")
plt.ylabel("Precision")
#plt.yscale("symlog", linthresh=0.9, linscale=50.0)
#yticks = [-0.1, 0, 0.1, 0.2, 0.5, 0.8]
#plt.yticks(yticks, [f"{y:g}" for y in yticks])

#plt.title("Precision vs Hidden State")
#plt.legend()
plt.grid(True)

plt.savefig(
    "gemma_attack/AllPlots/bertScores/sampleSpecific/Vis_Precision_vs_hiddenstate.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


plt.figure(figsize=(3.4, 2.6))

RmeanArr = np.array(RmeanList)
RstdArr  = np.array(RstdList)

plt.plot(ConsideredEndPts, RmeanArr, label="Mean Recall")
plt.fill_between(
    ConsideredEndPts,
    RmeanArr - RstdArr,
    RmeanArr + RstdArr,
    alpha=0.3,
    label="±1 Std"
)

plt.xlabel("Hidden state index")
plt.ylabel("Recall")
#plt.yscale("symlog", linthresh=0.1, linscale=1.0)

#plt.title("Recall vs Hidden State")
#plt.legend()
plt.grid(True)

plt.savefig(
    "gemma_attack/AllPlots/bertScores/sampleSpecific/Vis_recall_vs_hiddenstate.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


plt.figure(figsize=(3.4, 2.6))

F1meanArr = np.array(F1meanList)
F1stdArr  = np.array(F1stdList)

plt.plot(ConsideredEndPts, F1meanArr, label="Mean F1")
plt.fill_between(
    ConsideredEndPts,
    F1meanArr - F1stdArr,
    F1meanArr + F1stdArr,
    alpha=0.3,
    label="±1 Std"
)

plt.xlabel("Hidden state index")
plt.ylabel("F1 score")
#plt.yscale("symlog", linthresh=0.1, linscale=1.0)

#plt.title("F1 vs Hidden State")
#plt.legend()
plt.grid(True)

plt.savefig(
    "gemma_attack/AllPlots/bertScores/sampleSpecific/Vis_f1score_vs_hiddenstate.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

