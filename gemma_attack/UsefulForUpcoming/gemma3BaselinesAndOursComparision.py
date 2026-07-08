
'''




export CUDA_VISIBLE_DEVICES=1
conda activate gemma3
cd interpretAttacks
python gemma_attack/gemma3BaselinesAndOursComparision.py  --attck_type saa --desired_norm_l_inf 0.05 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --AttackStartLayer_vis 11 --numLayerstAtAtime 1 --whichMLP gate_proj --whichMLP_vis fc2 --numSamplesConsidered 50
python gemma_attack/gemma3BaselinesAndOursComparision.py  --attck_type saa --desired_norm_l_inf 0.04 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --AttackStartLayer_vis 11 --numLayerstAtAtime 1 --whichMLP gate_proj --whichMLP_vis fc2 --numSamplesConsidered 50
python gemma_attack/gemma3BaselinesAndOursComparision.py  --attck_type saa --desired_norm_l_inf 0.03 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --AttackStartLayer_vis 11 --numLayerstAtAtime 1 --whichMLP gate_proj --whichMLP_vis fc2 --numSamplesConsidered 50
python gemma_attack/gemma3BaselinesAndOursComparision.py  --attck_type saa --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --AttackStartLayer_vis 11 --numLayerstAtAtime 1 --whichMLP gate_proj --whichMLP_vis fc2 --numSamplesConsidered 50
python gemma_attack/gemma3BaselinesAndOursComparision.py  --attck_type saa --desired_norm_l_inf 0.01 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --AttackStartLayer_vis 11 --numLayerstAtAtime 1 --whichMLP gate_proj --whichMLP_vis fc2 --numSamplesConsidered 50


export CUDA_VISIBLE_DEVICES=7
conda activate gemma3
cd interpretAttacks
python gemma_attack/gemma3BaselinesAndOursComparision.py  --attck_type saa --desired_norm_l_inf 0.05 --learningRate 0.001 --num_steps 10 --AttackStartLayer 0 --AttackStartLayer_vis 11 --numLayerstAtAtime 1 --whichMLP gate_proj --whichMLP_vis fc2 --numSamplesConsidered 50
python gemma_attack/gemma3BaselinesAndOursComparision.py  --attck_type saa --desired_norm_l_inf 0.04 --learningRate 0.001 --num_steps 10 --AttackStartLayer 0 --AttackStartLayer_vis 11 --numLayerstAtAtime 1 --whichMLP gate_proj --whichMLP_vis fc2 --numSamplesConsidered 50
python gemma_attack/gemma3BaselinesAndOursComparision.py  --attck_type saa --desired_norm_l_inf 0.03 --learningRate 0.001 --num_steps 10 --AttackStartLayer 0 --AttackStartLayer_vis 11 --numLayerstAtAtime 1 --whichMLP gate_proj --whichMLP_vis fc2 --numSamplesConsidered 50
python gemma_attack/gemma3BaselinesAndOursComparision.py  --attck_type saa --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 10 --AttackStartLayer 0 --AttackStartLayer_vis 11 --numLayerstAtAtime 1 --whichMLP gate_proj --whichMLP_vis fc2 --numSamplesConsidered 50
python gemma_attack/gemma3BaselinesAndOursComparision.py  --attck_type saa --desired_norm_l_inf 0.01 --learningRate 0.001 --num_steps 10 --AttackStartLayer 0 --AttackStartLayer_vis 11 --numLayerstAtAtime 1 --whichMLP gate_proj --whichMLP_vis fc2 --numSamplesConsidered 50


export CUDA_VISIBLE_DEVICES=2
conda activate gemma3
cd interpretAttacks
python gemma_attack/gemma3BaselinesAndOursComparision.py  --attck_type saa --desired_norm_l_inf 0.05 --learningRate 0.001 --num_steps 100 --AttackStartLayer 0 --AttackStartLayer_vis 11 --numLayerstAtAtime 1 --whichMLP gate_proj --whichMLP_vis fc2 --numSamplesConsidered 50
python gemma_attack/gemma3BaselinesAndOursComparision.py  --attck_type saa --desired_norm_l_inf 0.04 --learningRate 0.001 --num_steps 100 --AttackStartLayer 0 --AttackStartLayer_vis 11 --numLayerstAtAtime 1 --whichMLP gate_proj --whichMLP_vis fc2 --numSamplesConsidered 50
python gemma_attack/gemma3BaselinesAndOursComparision.py  --attck_type saa --desired_norm_l_inf 0.03 --learningRate 0.001 --num_steps 100 --AttackStartLayer 0 --AttackStartLayer_vis 11 --numLayerstAtAtime 1 --whichMLP gate_proj --whichMLP_vis fc2 --numSamplesConsidered 50
python gemma_attack/gemma3BaselinesAndOursComparision.py  --attck_type saa --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 100 --AttackStartLayer 0 --AttackStartLayer_vis 11 --numLayerstAtAtime 1 --whichMLP gate_proj --whichMLP_vis fc2 --numSamplesConsidered 50
python gemma_attack/gemma3BaselinesAndOursComparision.py  --attck_type saa --desired_norm_l_inf 0.01 --learningRate 0.001 --num_steps 100 --AttackStartLayer 0 --AttackStartLayer_vis 11 --numLayerstAtAtime 1 --whichMLP gate_proj --whichMLP_vis fc2 --numSamplesConsidered 50



export CUDA_VISIBLE_DEVICES=1
conda activate gemma3
cd interpretAttacks
python gemma_attack/gemma3BaselinesAndOursComparision.py  --attck_type saa --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --AttackStartLayer_vis 11 --numLayerstAtAtime 1 --whichMLP gate_proj --whichMLP_vis fc2 --numSamplesConsidered 50
python gemma_attack/gemma3BaselinesAndOursComparision.py  --attck_type saa --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --AttackStartLayer_vis 11 --numLayerstAtAtime 1 --whichMLP gate_proj --whichMLP_vis fc2 --numSamplesConsidered 50
python gemma_attack/gemma3BaselinesAndOursComparision.py  --attck_type saa --desired_norm_l_inf 0.003 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --AttackStartLayer_vis 11 --numLayerstAtAtime 1 --whichMLP gate_proj --whichMLP_vis fc2 --numSamplesConsidered 50
python gemma_attack/gemma3BaselinesAndOursComparision.py  --attck_type saa --desired_norm_l_inf 0.002 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --AttackStartLayer_vis 11 --numLayerstAtAtime 1 --whichMLP gate_proj --whichMLP_vis fc2 --numSamplesConsidered 50
python gemma_attack/gemma3BaselinesAndOursComparision.py  --attck_type saa --desired_norm_l_inf 0.001 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --AttackStartLayer_vis 11 --numLayerstAtAtime 1 --whichMLP gate_proj --whichMLP_vis fc2 --numSamplesConsidered 50


export CUDA_VISIBLE_DEVICES=0
conda activate gemma3
cd interpretAttacks
python gemma_attack/gemma3BaselinesAndOursComparision.py  --attck_type saa --desired_norm_l_inf 0.0009 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --AttackStartLayer_vis 11 --numLayerstAtAtime 1 --whichMLP gate_proj --whichMLP_vis fc2 --numSamplesConsidered 50
python gemma_attack/gemma3BaselinesAndOursComparision.py  --attck_type saa --desired_norm_l_inf 0.0008 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --AttackStartLayer_vis 11 --numLayerstAtAtime 1 --whichMLP gate_proj --whichMLP_vis fc2 --numSamplesConsidered 50
python gemma_attack/gemma3BaselinesAndOursComparision.py  --attck_type saa --desired_norm_l_inf 0.0007 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --AttackStartLayer_vis 11 --numLayerstAtAtime 1 --whichMLP gate_proj --whichMLP_vis fc2 --numSamplesConsidered 50
python gemma_attack/gemma3BaselinesAndOursComparision.py  --attck_type saa --desired_norm_l_inf 0.0006 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --AttackStartLayer_vis 11 --numLayerstAtAtime 1 --whichMLP gate_proj --whichMLP_vis fc2 --numSamplesConsidered 50
python gemma_attack/gemma3BaselinesAndOursComparision.py  --attck_type saa --desired_norm_l_inf 0.0005 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --AttackStartLayer_vis 11 --numLayerstAtAtime 1 --whichMLP gate_proj --whichMLP_vis fc2 --numSamplesConsidered 50



#command to run the below code 
export CUDA_VISIBLE_DEVICES=7
conda activate gemma3
cd interpretAttacks
python gemma_attack/gemma3BaselinesAndOursComparision.py  --attck_type saa --desired_norm_l_inf 0.0045 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --AttackStartLayer_vis 11 --numLayerstAtAtime 1 --whichMLP gate_proj --whichMLP_vis fc2 --numSamplesConsidered 50



export CUDA_VISIBLE_DEVICES=7
conda activate gemma3
cd interpretAttacks
python gemma_attack/gemma3BaselinesAndOursComparision.py  --attck_type saa --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --AttackStartLayer_vis 11 --numLayerstAtAtime 1 --whichMLP gate_proj --whichMLP_vis fc2 --numSamplesConsidered 50




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


parser.add_argument(
    "--chosenLanLayers",
    type=int,
    nargs="+",
    default=None,
    help="Space-separated language layer indices to attack, e.g. --chosenLanLayers 0 1 2 3 4"
)
parser.add_argument(
    "--chosenVisLayers",
    type=int,
    nargs="+",
    default=None,
    help="Space-separated vision layer indices to attack, e.g. --chosenVisLayers 15 16 17 18 19"
)

args = parser.parse_args()

attck_type = args.attck_type
epsilon = float(args.desired_norm_l_inf)
lr = float(args.learningRate)
num_steps = int(args.num_steps)

AttackStartLayer = int(args.AttackStartLayer)
AttackStartLayer_vis = int(args.AttackStartLayer_vis)

numLayerstAtAtime = int(args.numLayerstAtAtime)

whichMLP = str(args.whichMLP)
whichMLP_vis = str(args.whichMLP_vis)

numSamplesConsidered = int(args.numSamplesConsidered)

chosenLanLayers = [0]
chosenVisLayers = [11]

numHiddenStates = 35
#numSamplesConsidered = 15

PmeanList = []
RmeanList = []
F1meanList = []

PstdList = []
RstdList = []
F1stdList = []

#AttackStartLayer = 0

ega_ratio = 0.2
#for AttackStartLayer in range(numHiddenStates):
#towardsNull = 0
#towardsNull = 0.5
#towardsNull = 1.0
#towardsNull = 0.5
towardsNull = 0.1

#all_attck_types = ["bsa", "bsa_flat", "bsa_flat_lan", "bsa_flat_vis", "dra", "fda", "ssp", "nll", "ega", "saa"]
#all_attck_types = ["bsa", "dra", "fda", "ssp", "nllm", "ega", "saav"]

#all_attck_types = ["saa_loopRR", "saa_loopR", "bsa", "dra", "fdam", "ssp", "ega", "nllm", "saav", "saa", "saa_loop"]
#all_attck_types = ["saa_BSAexpTN_P15_BAp05", "saa_BSAexpTN_P05_BAp006", "saa_BSAexpTN_P05_BAp05", "saa_BSApo6", "saa_loopRR", "bsa", "dra", "fdam", "ssp", "ega", "nllm", "saav", "saa", "saa_loop"]

all_attck_types = ["saa_BSApo0", "saa_BSAexpTN_P15_BAp05", "saa_BSAexpTN_P05_BAp006", "saa_BSAexpTN_P05_BAp05", "saa_BSApo6", "bsa", "dra", "fdam", "ssp", "ega", "nllm", "saav", "saa", "saa_loop"]


#all_attck_types = ["saa_BSAexpTN_P15_BAp05", "saa_BSApo2",  "bsa", "dra", "fdam", "ssp", "ega", "nllm", "saav", "saa", "saa_loop"]
all_attck_types = ["saa_BSAexpTN_P15_BAp05", "saa_BSApo2",  "bsa", "dra", "fdam", "ssp", "ega", "nllm"]


type_sampleAggP = []
type_sampleAggR = []
type_samleAggF1 = []
for attck_type in all_attck_types:
    sampleAggP = []
    sampleAggR = []
    samleAggF1 = []
    print("attck_type", attck_type)
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
        elif attck_type == "saa_loopR":
            attck_typeR = "saa_loopR"
            towardsNullR = 0.1
            AttackStartLayerR = 0
            numLayerstAtAtimeR = 2
            whichMLPR = "down_proj"
            whichMLPvisR = "out_proj"
            chosenLanLayersR = [3, 4, 5 ]
            chosenVisLayersR = [1, 3, 5, 12, 17, 21]
            advOutputPath = (
                f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/"
                f"advOutput_attackType_{attck_typeR}_lr_{lr}_eps_{epsilon}_"
                f"AttackStartLayer_{AttackStartLayerR}_numLayerstAtAtime_{numLayerstAtAtimeR}_"
                f"num_steps_{num_steps}_towardsNull_{towardsNullR}_"
                f"lanMLP_{whichMLPR}_visMLP_{whichMLPvisR}_"
                f"lanLayers_{chosenLanLayersR}_visLayers_{chosenVisLayersR}.txt"
            )

        elif attck_type == "saa_loopRR":
            attck_typeR = "saa_loopR"
            towardsNullR = 0.5
            AttackStartLayerR = 0
            numLayerstAtAtimeR = 2
            whichMLPR = "up_proj"
            whichMLPvisR = "fc2"
            chosenLanLayersR = [0]
            chosenVisLayersR = [0]  
            advOutputPath = (
                f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/"
                f"advOutput_attackType_{attck_typeR}_lr_{lr}_eps_{epsilon}_"
                f"AttackStartLayer_{AttackStartLayerR}_numLayerstAtAtime_{numLayerstAtAtimeR}_"
                f"num_steps_{num_steps}_towardsNull_{towardsNullR}_"
                f"lanMLP_{whichMLPR}_visMLP_{whichMLPvisR}_"
                f"lanLayers_{chosenLanLayersR}_visLayers_{chosenVisLayersR}.txt"
            )
        elif attck_type == "saa_BSApo2":
            attck_typeR = "saa_BSA"
            towardsNullR = 0.02
            AttackStartLayerR = 0
            numLayerstAtAtimeR = 2
            whichMLPR = "up_proj"
            whichMLPvisR = "fc2"
            chosenLanLayersR = [0]
            chosenVisLayersR = [0]  
            balancingAlphaR = 0.06
            advOutputPath = (
                f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/"
                f"advOutput_attackType_{attck_typeR}_lr_{lr}_eps_{epsilon}_"
                f"AttackStartLayer_{AttackStartLayerR}_numLayerstAtAtime_{numLayerstAtAtimeR}_"
                f"num_steps_{num_steps}_towardsNull_{towardsNullR}_"
                f"lanMLP_{whichMLPR}_visMLP_{whichMLPvisR}_"
                f"lanLayers_{chosenLanLayersR}_visLayers_{chosenVisLayersR}_balancingAlpha_{balancingAlphaR}.txt"
            )

        elif attck_type == "saa_BSAp15":
            attck_typeR = "saa_BSA"
            towardsNullR = 0.15
            AttackStartLayerR = 0
            numLayerstAtAtimeR = 2
            whichMLPR = "up_proj"
            whichMLPvisR = "fc2"
            chosenLanLayersR = [0]
            chosenVisLayersR = [0]  
            balancingAlphaR = 0.06
            advOutputPath = (
                f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/"
                f"advOutput_attackType_{attck_typeR}_lr_{lr}_eps_{epsilon}_"
                f"AttackStartLayer_{AttackStartLayerR}_numLayerstAtAtime_{numLayerstAtAtimeR}_"
                f"num_steps_{num_steps}_towardsNull_{towardsNullR}_"
                f"lanMLP_{whichMLPR}_visMLP_{whichMLPvisR}_"
                f"lanLayers_{chosenLanLayersR}_visLayers_{chosenVisLayersR}_balancingAlpha_{balancingAlphaR}.txt"
            )


        elif attck_type == "saa_BSApo0":
            attck_typeR = "saa_BSA"
            towardsNullR = 0.0
            AttackStartLayerR = 0
            numLayerstAtAtimeR = 2
            whichMLPR = "up_proj"
            whichMLPvisR = "fc2"
            chosenLanLayersR = [0]
            chosenVisLayersR = [0]  
            balancingAlphaR = 0.06
            advOutputPath = (
                f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/"
                f"advOutput_attackType_{attck_typeR}_lr_{lr}_eps_{epsilon}_"
                f"AttackStartLayer_{AttackStartLayerR}_numLayerstAtAtime_{numLayerstAtAtimeR}_"
                f"num_steps_{num_steps}_towardsNull_{towardsNullR}_"
                f"lanMLP_{whichMLPR}_visMLP_{whichMLPvisR}_"
                f"lanLayers_{chosenLanLayersR}_visLayers_{chosenVisLayersR}_balancingAlpha_{balancingAlphaR}.txt"
            )


        elif attck_type == "saa_BSAexpTN_P05_BAp05":
            #advOutput_attackType_saa_BSAexp_lr_0.001_eps_0.002_AttackStartLayer_0_numLayerstAtAtime_2_num_steps_1000_towardsNull_0.5_lanMLP_up_proj_visMLP_fc2_lanLayers_upto4_visLayers_all_balancingAlpha_0.5.txt
            attck_typeR = "saa_BSAexp"
            towardsNullR = 0.5
            AttackStartLayerR = 0
            numLayerstAtAtimeR = 2
            whichMLPR = "up_proj"
            whichMLPvisR = "fc2"
            chosenLanLayersR = [0]
            chosenVisLayersR = [0]  
            balancingAlphaR = 0.5
            advOutputPath = (
                f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/"
                f"advOutput_attackType_{attck_typeR}_lr_{lr}_eps_{epsilon}_"
                f"AttackStartLayer_{AttackStartLayerR}_numLayerstAtAtime_{numLayerstAtAtimeR}_"
                f"num_steps_{num_steps}_towardsNull_{towardsNullR}_"
                f"lanMLP_{whichMLPR}_visMLP_{whichMLPvisR}_"
                f"lanLayers_upto4_visLayers_all_balancingAlpha_{balancingAlphaR}.txt"
            )

        elif attck_type == "saa_BSAexpTN_P05_BAp006":
            #advOutput_attackType_saa_BSAexp_lr_0.001_eps_0.002_AttackStartLayer_0_numLayerstAtAtime_2_num_steps_1000_towardsNull_0.5_lanMLP_up_proj_visMLP_fc2_lanLayers_upto4_visLayers_all_balancingAlpha_0.5.txt
            attck_typeR = "saa_BSAexp"
            towardsNullR = 0.5
            AttackStartLayerR = 0
            numLayerstAtAtimeR = 2
            whichMLPR = "up_proj"
            whichMLPvisR = "fc2"
            chosenLanLayersR = [0]
            chosenVisLayersR = [0]  
            balancingAlphaR = 0.06
            advOutputPath = (
                f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/"
                f"advOutput_attackType_{attck_typeR}_lr_{lr}_eps_{epsilon}_"
                f"AttackStartLayer_{AttackStartLayerR}_numLayerstAtAtime_{numLayerstAtAtimeR}_"
                f"num_steps_{num_steps}_towardsNull_{towardsNullR}_"
                f"lanMLP_{whichMLPR}_visMLP_{whichMLPvisR}_"
                f"lanLayers_upto4_visLayers_all_balancingAlpha_{balancingAlphaR}.txt"
            )



        elif attck_type == "saa_BSAexpTN_P15_BAp05":
            #advOutput_attackType_saa_BSAexp_lr_0.001_eps_0.002_AttackStartLayer_0_numLayerstAtAtime_2_num_steps_1000_towardsNull_0.5_lanMLP_up_proj_visMLP_fc2_lanLayers_upto4_visLayers_all_balancingAlpha_0.5.txt
            attck_typeR = "saa_BSAexp"
            towardsNullR = 0.15
            AttackStartLayerR = 0
            numLayerstAtAtimeR = 2
            whichMLPR = "up_proj"
            whichMLPvisR = "fc2"
            chosenLanLayersR = [0]
            chosenVisLayersR = [0]  
            balancingAlphaR = 0.5
            advOutputPath = (
                f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/"
                f"advOutput_attackType_{attck_typeR}_lr_{lr}_eps_{epsilon}_"
                f"AttackStartLayer_{AttackStartLayerR}_numLayerstAtAtime_{numLayerstAtAtimeR}_"
                f"num_steps_{num_steps}_towardsNull_{towardsNullR}_"
                f"lanMLP_{whichMLPR}_visMLP_{whichMLPvisR}_"
                f"lanLayers_upto4_visLayers_all_balancingAlpha_{balancingAlphaR}.txt"
            )

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

        #print("Precision:", P.item())
        #print("Recall:", R.item())
        #print("F1:", F1.item())

        if F1.item() < -3 :
            checkInd = attackSample

        sampleAggP.append(P.item())
        sampleAggR.append(R.item())
        samleAggF1.append(F1.item())

    sampleAggP = np.array(sampleAggP)
    sampleAggR = np.array(sampleAggR)
    samleAggF1 = np.array(samleAggF1)

    print("sampleAggP.shape", sampleAggP.shape)
    print("sampleAggR.shape", sampleAggR.shape)
    print("samleAggF1.shape", samleAggF1.shape)

    type_sampleAggP.append(sampleAggP)
    type_sampleAggR.append(sampleAggR)
    type_samleAggF1.append(samleAggF1)


print("len(type_sampleAggP)", len(type_sampleAggP))
print("len(type_sampleAggR)", len(type_sampleAggR))
print("len(type_samleAggF1)", len(type_samleAggF1))


print(type_sampleAggP[0].shape)
print(type_sampleAggP[-1].shape)

print(type_sampleAggR[0].shape)
print(type_sampleAggR[-1].shape)

print(type_samleAggF1[0].shape)
print(type_samleAggF1[-1].shape)


import os
import matplotlib.pyplot as plt

#AllAttckTypes = ["BSA", "BSA\nFLAT", "BSA\nLAN", "BSA VIS", "DRA", "FDA", "SSPA", "CE", "EGA", "SSPMA"]
#AllAttckTypes = ["BSA", "DRA", "FDA", "SSPA", "CE", "EGA", "SSPMA"]
#AllAttckTypes = ["BSA", "DRA", "FDA", "SSPA", "EGA", "SSPMA"]
#AllAttckTypes = ["SSPMA-RR","SSPMA-R", "BSA", "DRA", "FDA", "SSPA", "EGA", "CE", "SSPMA-E", "SSPMA-L", "SSPMA-EL"]
#AllAttckTypes = ["saa_BSAexpTN_P15_BAp05", "saa_BSAexpTN_P05_BAp006", "saa_BSAexpTN_P05_BAp05", "saa_BSApo6", "SSPMA-RR", "BSA", "DRA", "FDA", "SSPA", "EGA", "CE", "SSPMA-E", "SSPMA-L", "SSPMA-EL"]

#AllAttckTypes = ["saa_BSApo0", "saa_BSAexpTN_P15_BAp05", "saa_BSAexpTN_P05_BAp006", "saa_BSAexpTN_P05_BAp05", "saa_BSApo6", "BSA", "DRA", "FDA", "SSPA", "EGA", "CE", "SSPMA-E", "SSPMA-L", "SSPMA-EL"]

#AllAttckTypes = ["saa_BSAexpTN_P15_BAp05", "saa_BSApo2" ,"BSA", "DRA", "FDA", "SSPA", "EGA", "CE", "SSPMA-E", "SSPMA-L", "SSPMA-EL"]
AllAttckTypes = ["saa_BSAexpTN_P15_BAp05", "saa_BSApo2" ,"BSA", "DRA", "FDA", "SSPA", "EGA", "CE"]

save_dir = "gemma_attack/AllPlots/comparisionFinTest"
os.makedirs(save_dir, exist_ok=True)  # create folder if it doesn't exist

plt.figure(figsize=(5, 7))
plt.boxplot(type_sampleAggP, labels=AllAttckTypes)
plt.xticks(rotation=90, ha="right")

plt.ylabel("BERT Precision score")
plt.xlabel("Attack Type")
plt.title("Distribution of Precision across Attack Types")
plt.tight_layout()
plt.savefig(f"{save_dir}/precision_boxplot_eps_{epsilon}_num_steps_{num_steps}_AttackStartLayer_{AttackStartLayer}_towardsNull_{towardsNull}_numSamplesConsidered_{numSamplesConsidered}_{whichMLP}_nll_fdam_cor_coherent_{chosenLanLayers}_{chosenVisLayers}.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


plt.figure(figsize=(5, 7))
plt.boxplot(type_sampleAggR, labels=AllAttckTypes)
plt.xticks(rotation=90, ha="right")

plt.ylabel("BERT Recall score")
plt.xlabel("Attack Type")
plt.title("Distribution of Recall across Attack Types")
plt.tight_layout()
plt.savefig(f"{save_dir}/recall_boxplot_eps_{epsilon}_num_steps_{num_steps}_AttackStartLayer_{AttackStartLayer}_towardsNull_{towardsNull}_numSamplesConsidered_{numSamplesConsidered}_{whichMLP}_nll_fdam_cor_coherent_{chosenLanLayers}_{chosenVisLayers}.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


plt.figure(figsize=(5, 7))
plt.boxplot(type_samleAggF1, labels=AllAttckTypes)
plt.xticks(rotation=90, ha="right")

plt.ylabel("BERT F1 score")
plt.xlabel("Attack Type")
plt.title("Distribution of F1 score across Attack Types")
plt.tight_layout()
plt.savefig(f"{save_dir}/f1score_boxplot_eps_{epsilon}_num_steps_{num_steps}_AttackStartLayer_{AttackStartLayer}_towardsNull_{towardsNull}_numSamplesConsidered_{numSamplesConsidered}_{whichMLP}_nll_fdam_cor_coherent_{chosenLanLayers}_{chosenVisLayers}.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


print("checkInd", checkInd)