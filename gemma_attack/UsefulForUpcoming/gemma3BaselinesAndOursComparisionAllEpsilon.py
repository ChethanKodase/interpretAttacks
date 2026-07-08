
'''

export CUDA_VISIBLE_DEVICES=0
conda activate gemma3
cd interpretAttacks
python gemma_attack/gemma3BaselinesAndOursComparisionAllEpsilon.py --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --AttackStartLayer_vis 11 --whichMLP gate_proj --whichMLP_vis fc2 --numSamplesConsidered 50



export CUDA_VISIBLE_DEVICES=0
conda activate gemma3
cd interpretAttacks
python gemma_attack/gemma3BaselinesAndOursComparisionAllEpsilon.py --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --AttackStartLayer_vis 11 --whichMLP gate_proj --whichMLP_vis fc2 --numSamplesConsidered 50 --perturbationScale tinyTiny

export CUDA_VISIBLE_DEVICES=1
conda activate gemma3
cd interpretAttacks
python gemma_attack/gemma3BaselinesAndOursComparisionAllEpsilon.py --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --AttackStartLayer_vis 11 --whichMLP gate_proj --whichMLP_vis fc2 --numSamplesConsidered 50 --perturbationScale tiny


export CUDA_VISIBLE_DEVICES=2
conda activate gemma3
cd interpretAttacks
python gemma_attack/gemma3BaselinesAndOursComparisionAllEpsilon.py --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --numLayerstAtAtime 1 --AttackStartLayer_vis 11 --whichMLP gate_proj --whichMLP_vis fc2 --numSamplesConsidered 50 --perturbationScale moderate


export CUDA_VISIBLE_DEVICES=3
conda activate gemma3
cd interpretAttacks
python gemma_attack/gemma3BaselinesAndOursComparisionAllEpsilon.py --learningRate 0.001 --num_steps 100 --AttackStartLayer 0 --numLayerstAtAtime 1 --AttackStartLayer_vis 11 --whichMLP gate_proj --whichMLP_vis fc2 --numSamplesConsidered 50 --perturbationScale tiny
'''


from bert_score import score
import argparse

import matplotlib.pyplot as plt
import numpy as np

import os
from matplotlib.ticker import FuncFormatter


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
#parser.add_argument("--attck_type", type=str, default="grill_l2",
                    #help="grill_l2 | grill_cos | OA_l2 | OA_cos")
#parser.add_argument("--desired_norm_l_inf", type=float, default=0.03,
                    #help="epsilon L_inf in ORIGINAL pixel space [0..1]. Try 0.01~0.08")
parser.add_argument("--learningRate", type=float, default=1e-3,
                    help="Adam learning rate")
parser.add_argument("--num_steps", type=int, default=2000,
                    help="Number of Adam steps")
parser.add_argument("--AttackStartLayer", type=int, default=0,
                    help="From which layer do you start attack")
parser.add_argument("--AttackStartLayer_vis", type=int, default=0,
                    help="From which layer do you start attack")

parser.add_argument("--numLayerstAtAtime", type=int, default=2,
                    help="Number of layers taken at a time to attack")
parser.add_argument("--whichMLP", type=str, default="fc1",
                    help="values taken : down_proj, up_proj, fc1, fc2, out_proj")

parser.add_argument("--whichMLP_vis", type=str, default="fc1",
                    help="values taken : down_proj, up_proj, fc1, fc2, out_proj")

parser.add_argument("--numSamplesConsidered", type=int, default=50,
                    help="Number of samples considered")

parser.add_argument("--perturbationScale", type=str, default="tinyTiny",
                    help="types to be used are tiny, tinyTiny and moderate ")


args = parser.parse_args()

#attck_type = args.attck_type
#epsilon = float(args.desired_norm_l_inf)
lr = float(args.learningRate)
num_steps = int(args.num_steps)
AttackStartLayer = int(args.AttackStartLayer)
AttackStartLayer_vis = int(args.AttackStartLayer_vis)

numLayerstAtAtime = int(args.numLayerstAtAtime)
whichMLP = str(args.whichMLP)
whichMLP_vis = str(args.whichMLP_vis)
perturbationScale = str(args.perturbationScale)


numSamplesConsidered = int(args.numSamplesConsidered)
chosenLanLayers = [0]
chosenVisLayers = [11]


numHiddenStates = 35
#numSamplesConsidered = 100
#numSamplesConsidered = 250
#numSamplesConsidered = 250

#numSamplesConsidered = 50

PmeanList = []
RmeanList = []
F1meanList = []

PstdList = []
RstdList = []
F1stdList = []

#AttackStartLayer = 0

#for AttackStartLayer in range(numHiddenStates):
#towardsNull = 0
#towardsNull = 0.5
#towardsNull = 1.0
towardsNull = 0.1
ega_ratio = 0.2

#all_attck_types = ["bsa", "dra", "fda", "ssp","saa"]
#all_attck_types = ["bsa", "bsa_flat", "bsa_flat_lan", "bsa_flat_vis", "dra", "fda", "ssp", "nll", "ega", "saa"]
#all_attck_types = ["bsa", "dra", "fda", "ssp", "nll", "ega", "saa"]
#all_attck_types = ["bsa", "dra", "fdam", "ssp", "ega", "saa"]
#all_attck_types = ["bsa", "dra", "fdam", "ssp", "ega", "saav", "saa"]
#all_attck_types = ["bsa", "dra", "fdam", "ssp", "ega", "nllm", "saav", "saa", "saa_loop"]

#epsilon = 0.02

#allEpsilons = [0.01, 0.02, 0.03, 0.04, 0.05]

#epsilonTypes = "tiny"
#epsilonTypes = "tinyTiny"
#epsilonTypes = "moderate"
epsilonTypes = perturbationScale

if epsilonTypes == "tiny":
    allEpsilons = [0.001, 0.002, 0.003, 0.004, 0.005]
    all_attck_types = ["bsa", "dra", "fdam", "ssp", "ega", "nllm", "saa_loop"]
elif epsilonTypes == "tinyTiny":
    allEpsilons = [0.0005, 0.0006, 0.0007, 0.0008, 0.0009]
    all_attck_types = ["bsa", "dra", "fdam", "ssp", "ega", "nllm", "saa_loop"]
else:
    allEpsilons = [0.01, 0.02, 0.03, 0.04, 0.05]
    all_attck_types = ["bsa", "dra", "fdam", "ssp", "ega", "nllm", "saa_loop"]

precisionMeanForAttacksSeries = []
precisionStdForAttacksSeries = []

recallMeanForAttacksSeries = []
recallStdForAttacksSeries = []

f1MeanForAttacksSeries = []
f1StdForAttacksSeries = []

for epsilon in allEpsilons:

    precisionMeanForAttacks = []
    precisionStdForAttacks = []

    recallMeanForAttacks = []
    recallStdForAttacks = []

    f1MeanForAttacks = []
    f1StdForAttacks = []

    for attck_type in all_attck_types:
        sampleAggP = []
        sampleAggR = []
        samleAggF1 = []
        for attackSample in range(1,numSamplesConsidered):
            #advOutputPath = f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_num_steps_{num_steps}_.txt"

            if attck_type == "saa":
                #advOutputPath = f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_num_steps_{num_steps}_towardsNull_{towardsNull}.txt"
                advOutputPath = f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_num_steps_{num_steps}_towardsNull_{towardsNull}_{whichMLP}.txt"
            elif attck_type == "saav":
                #advOutputPath = f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_num_steps_{num_steps}_towardsNull_{towardsNull}.txt"
                advOutputPath = f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_AttackStartLayer_{AttackStartLayer_vis}_numLayerstAtAtime_{numLayerstAtAtime}_num_steps_{num_steps}_towardsNull_{towardsNull}_{whichMLP_vis}.txt"
            elif attck_type == "ega":
                advOutputPath = f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_num_steps_{num_steps}_ratio_{ega_ratio}.txt"
            elif attck_type == "saa_loop":
                advOutputPath = f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_num_steps_{num_steps}_towardsNull_{towardsNull}_{whichMLP}_{chosenLanLayers}_{chosenVisLayers}.txt"
            else:
                advOutputPath = f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_num_steps_{num_steps}_.txt"



            with open(advOutputPath, "r") as f:
                advOutput = [f.read().strip()]
                #print("advOutput", advOutput)

            #cleanOutputPath = "/data1/chethan/interpretAttacks/gemma_attack/outputsStorageImagenet/advOutputs/1/cleanOutput.txt"
            cleanOutputPath = f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/cleanOutput.txt"

            with open(cleanOutputPath, "r") as f:
                cleanOutput = [f.read().strip()]
                #print("cleanOutput", cleanOutput)

            P, R, F1 = score(
                advOutput,
                cleanOutput,
                lang="en",              # language
                model_type="roberta-large",  # standard choice
                rescale_with_baseline=True   # recommended
            )


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

        precisionMeanForAttacks.append(Pmean)
        precisionStdForAttacks.append(Pstd)

        recallMeanForAttacks.append(Rmean)
        recallStdForAttacks.append(Rstd)

        f1MeanForAttacks.append(F1mean)
        f1StdForAttacks.append(F1std)

    precisionMeanForAttacks = np.array(precisionMeanForAttacks)
    precisionStdForAttacks = np.array(precisionStdForAttacks)


    recallMeanForAttacks = np.array(recallMeanForAttacks)
    recallStdForAttacks = np.array(recallStdForAttacks)

    f1MeanForAttacks = np.array(f1MeanForAttacks)
    f1StdForAttacks = np.array(f1StdForAttacks)


    precisionMeanForAttacksSeries.append(precisionMeanForAttacks)
    precisionStdForAttacksSeries.append(precisionStdForAttacks)

    recallMeanForAttacksSeries.append(recallMeanForAttacks)
    recallStdForAttacksSeries.append(recallStdForAttacks)

    f1MeanForAttacksSeries.append(f1MeanForAttacks)
    f1StdForAttacksSeries.append(f1StdForAttacks)


precisionMeanForAttacksSeries = np.array(precisionMeanForAttacksSeries)
precisionStdForAttacksSeries = np.array(precisionStdForAttacksSeries)

recallMeanForAttacksSeries = np.array(recallMeanForAttacksSeries)
recallStdForAttacksSeries = np.array(recallStdForAttacksSeries)

f1MeanForAttacksSeries = np.array(f1MeanForAttacksSeries)
f1StdForAttacksSeries = np.array(f1StdForAttacksSeries)



#AllAttckTypes = ["BSA", "DRA", "FDA", "SSPA","SSPMA"]
#AllAttckTypes = ["BSA", "BSA\nFLAT", "BSA\nLAN", "BSA VIS", "DRA", "FDA", "SSPA", "CE", "EGA", "SSPMA"]
#AllAttckTypes = ["BSA", "DRA", "FDA", "SSPA", "CE", "EGA", "SSPMA"]
#AllAttckTypes = ["BSA", "DRA", "FDA", "SSPA", "EGA", "SSPMA"]
#AllAttckTypes = ["BSA", "DRA", "FDA", "SSPA", "EGA", "SSPMA-E", "SSPMA-L"]
if epsilonTypes == "tiny":
    AllAttckTypes = ["BSA", "DRA", "FDA", "SSPA", "EGA", "CE", "SSPMA"]
else:
    AllAttckTypes = ["BSA", "DRA", "FDA", "SSPA", "EGA", "CE", "SSPMA"]


#---------------------------------------------------------------------------------------------------------------------------------------------------
means = precisionMeanForAttacksSeries
stds = precisionStdForAttacksSeries
time_points = allEpsilons #np.arange(means.shape[0])  # [0, 1, 2]
save_dir = "gemma_attack/AllPlots/comparisionSeries"
os.makedirs(save_dir, exist_ok=True)
plt.figure()
for i in range(means.shape[1]):  # 5 objects
    mean = means[:, i]
    std = stds[:, i]
    
    plt.plot(time_points, mean, label=AllAttckTypes[i], linewidth=2.5)
    plt.fill_between(time_points, mean - std, mean + std, alpha=0.2)
plt.xlabel("c", fontsize=14)
plt.ylabel("BERT Precision", fontsize=14)

# Increase tick font sizes
# Set tick positions
plt.xticks(time_points)

# Format ticks as scientific notation (e.g., 2 × 10^-2)
def format_func(x, pos):
    if x == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    coeff = x / (10**exponent)
    return r"${:.0f} \times 10^{{{}}}$".format(coeff, exponent)

plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))

# Rotate ticks
plt.xticks(rotation=45, ha='right', fontsize=12)

plt.yticks(fontsize=12)

plt.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, 1.02),  # (x, y) — centered above plot
    ncol=3,  # adjust based on number of legend items
    fontsize=12,
    frameon=False
)
plt.grid(True)
save_path = os.path.join(save_dir, f"PrecisionComparisionSeries_num_steps_{num_steps}_AttackStartLayer_{AttackStartLayer}_towardsNull_{towardsNull}_numSamplesConsidered_{numSamplesConsidered}_{whichMLP}_{whichMLP_vis}_{chosenLanLayers}_{chosenVisLayers}_{epsilonTypes}.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print()
print("precision")
print("means", means)
print("stds", stds)

#---------------------------------------------------------------------------------------------------------------------------------------------------
means = recallMeanForAttacksSeries
stds = recallStdForAttacksSeries
time_points = allEpsilons #np.arange(means.shape[0])  # [0, 1, 2]
save_dir = "gemma_attack/AllPlots/comparisionSeries"
os.makedirs(save_dir, exist_ok=True)
plt.figure()
for i in range(means.shape[1]):  # 5 objects
    mean = means[:, i]
    std = stds[:, i]
    
    plt.plot(time_points, mean, label=AllAttckTypes[i], linewidth=2.5)
    plt.fill_between(time_points, mean - std, mean + std, alpha=0.2)
plt.xlabel("c", fontsize=14)
plt.ylabel("BERT Recall", fontsize=14)

# Increase tick font sizes
# Set tick positions
plt.xticks(time_points)

# Format ticks as scientific notation (e.g., 2 × 10^-2)
def format_func(x, pos):
    if x == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    coeff = x / (10**exponent)
    return r"${:.0f} \times 10^{{{}}}$".format(coeff, exponent)

plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))

# Rotate ticks
plt.xticks(rotation=45, ha='right', fontsize=12)

plt.yticks(fontsize=12)

plt.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, 1.02),  # (x, y) — centered above plot
    ncol=3,  # adjust based on number of legend items
    fontsize=12,
    frameon=False
)
plt.grid(True)
save_path = os.path.join(save_dir, f"RecallComparisionSeries_num_steps_{num_steps}_AttackStartLayer_{AttackStartLayer}_towardsNull_{towardsNull}_numSamplesConsidered_{numSamplesConsidered}_{whichMLP}_{whichMLP_vis}_{chosenLanLayers}_{chosenVisLayers}_{epsilonTypes}.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
plt.close()
print()
print("recall")
print("means", means)
print("stds", stds)


#---------------------------------------------------------------------------------------------------------------------------------------------------
means = f1MeanForAttacksSeries
stds = f1StdForAttacksSeries
time_points = allEpsilons #np.arange(means.shape[0])  # [0, 1, 2]
save_dir = "gemma_attack/AllPlots/comparisionSeries"
os.makedirs(save_dir, exist_ok=True)
plt.figure()
for i in range(means.shape[1]):  # 5 objects
    mean = means[:, i]
    std = stds[:, i]
    
    plt.plot(time_points, mean, label=AllAttckTypes[i], linewidth=2.5)
    plt.fill_between(time_points, mean - std, mean + std, alpha=0.2)
plt.xlabel("c", fontsize=14)
plt.ylabel("BERT F1 score", fontsize=14)

# Increase tick font sizes
# Set tick positions
plt.xticks(time_points)

# Format ticks as scientific notation (e.g., 2 × 10^-2)
def format_func(x, pos):
    if x == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    coeff = x / (10**exponent)
    return r"${:.0f} \times 10^{{{}}}$".format(coeff, exponent)

plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))

# Rotate ticks
plt.xticks(rotation=45, ha='right', fontsize=12)

plt.yticks(fontsize=12)

plt.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, 1.02),  # (x, y) — centered above plot
    ncol=3,  # adjust based on number of legend items
    fontsize=12,
    frameon=False
)

plt.grid(True)
save_path = os.path.join(save_dir, f"F1ComparisionSeries_num_steps_{num_steps}_AttackStartLayer_{AttackStartLayer}_towardsNull_{towardsNull}_numSamplesConsidered_{numSamplesConsidered}_{whichMLP}_{whichMLP_vis}_{chosenLanLayers}_{chosenVisLayers}_{epsilonTypes}.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print()
print("f1 score")
print("means", means)
print("stds", stds)



'''
for tiny epsilons
precision
means [[0.59478385 0.59185296 0.60350214 0.58517771 0.58922769 0.38496362
  0.64838977 0.57273559 0.21096493]
 [0.46470169 0.54508884 0.56334753 0.52809967 0.56516678 0.35055463
  0.58706534 0.52564388 0.24143836]
 [0.36483745 0.54195785 0.55589309 0.49029173 0.46990986 0.31558682
  0.5952499  0.48137848 0.08448873]
 [0.25698312 0.47245277 0.54423776 0.50600796 0.51228989 0.33327272
  0.576406   0.45268898 0.19378585]
 [0.23540371 0.48829753 0.5279206  0.47199363 0.5155321  0.3296533
  0.58948071 0.47420553 0.09933627]]
stds [[0.16306618 0.16692738 0.164446   0.17908852 0.15852156 0.18549218
  0.16522526 0.18606786 0.10595221]
 [0.18534654 0.17787589 0.21989384 0.15775507 0.15969921 0.17058742
  0.15975448 0.2004845  0.11898452]
 [0.19993509 0.160861   0.20133209 0.16922624 0.17159157 0.18078699
  0.17098149 0.18183199 0.06307958]
 [0.21739638 0.1955733  0.1862699  0.16570377 0.15647517 0.19008256
  0.15844333 0.19251216 0.09215992]
 [0.22400912 0.16545549 0.18402241 0.17228164 0.16086883 0.16376098
  0.15288625 0.17741285 0.09085071]]

recall
means [[ 0.59656205  0.5898196   0.59436509  0.57471819  0.59254314  0.38025833
   0.62812679  0.58515075  0.26085275]
 [ 0.46840879  0.54535906  0.54993975  0.52762156  0.57444671  0.34705309
   0.58101955  0.52093205  0.26575367]
 [ 0.36128057  0.54383884  0.54450598  0.47929949  0.48272183  0.30732111
   0.58630721  0.4875462  -0.03227971]
 [ 0.27093578  0.48533067  0.54943208  0.48005821  0.53243367  0.33006997
   0.578026    0.47147693  0.22733215]
 [ 0.25362658  0.49991567  0.52987065  0.48155101  0.51478195  0.33898534
   0.60367331  0.50786154  0.16919973]]
stds [[0.15475767 0.17911387 0.18666614 0.17008961 0.17367759 0.19495587
  0.19346042 0.16590421 0.09461046]
 [0.19333064 0.1647255  0.23189228 0.16123144 0.14234135 0.18151143
  0.19496178 0.21160892 0.08919432]
 [0.20312489 0.15894926 0.2154008  0.20801731 0.17838109 0.18332482
  0.19828031 0.19858859 0.09275559]
 [0.20995597 0.19136564 0.19666781 0.18503648 0.15997407 0.18363722
  0.14862878 0.18287629 0.07292205]
 [0.18727101 0.16211584 0.18233247 0.18662362 0.13905396 0.16434114
  0.14455747 0.14831945 0.05320716]]

f1 score
means [[0.59584383 0.59107639 0.59897637 0.57996559 0.5910459  0.38289967
  0.63828456 0.57900166 0.23682028]
 [0.46695808 0.54527893 0.55673486 0.52793878 0.57023672 0.34917921
  0.58397621 0.52345937 0.25447261]
 [0.36361845 0.54309312 0.55037835 0.48499558 0.47654075 0.31186213
  0.59103071 0.48465944 0.02676955]
 [0.26437943 0.479106   0.54711112 0.49336304 0.52252911 0.33209156
  0.57755061 0.46236181 0.21152239]
 [0.24456225 0.49422046 0.52911689 0.47718653 0.51548353 0.33487706
  0.59691932 0.49127487 0.13509764]]
stds [[0.15071576 0.16652986 0.16653839 0.16451851 0.15770396 0.1804944
  0.17179336 0.16723678 0.09500389]
 [0.1828826  0.16082164 0.21890577 0.14814076 0.14635488 0.16601963
  0.1669811  0.19848681 0.09768793]
 [0.19543723 0.15074688 0.20150742 0.18104484 0.16550812 0.17206258
  0.17916959 0.18136103 0.07184832]                                                                                                                                                                                                                                                                                                                       4-Ap2 24-Apr-26
 [0.20469263 0.18505617 0.18530438 0.16871694 0.14819602 0.17828224                                                                                                                                                                                                                                                                                    24-
  0.14739687 0.17960258 0.07471502]                                                                                                                                                                                                                                                                                                                  24
 [0.19395709 0.15238461 0.17525837 0.173263   0.14209239 0.15591524
  0.14327481 0.15496429 0.06276228]]                                      

'''