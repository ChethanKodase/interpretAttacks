
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
export CUDA_VISIBLE_DEVICES=1
conda activate gemma3
cd interpretAttacks
python gemma_attack/gemma3BaselinesAndOursComparisionLocalDangerScore.py  --attck_type saa --desired_norm_l_inf 0.002 --learningRate 0.001 --num_steps 1000 --AttackStartLayer 0 --AttackStartLayer_vis 11 --numLayerstAtAtime 1 --whichMLP gate_proj --whichMLP_vis fc2 --numSamplesConsidered 50



export CUDA_VISIBLE_DEVICES=1

conda activate gemma3

cd /data1/chethan/interpretAttacks

python gemma_attack/gemma3BaselinesAndOursComparisionLocalDangerScore.py \
    --attck_type saa \
    --desired_norm_l_inf 0.002 \
    --learningRate 0.001 \
    --num_steps 1000 \
    --AttackStartLayer 0 \
    --AttackStartLayer_vis 11 \
    --numLayerstAtAtime 1 \
    --whichMLP gate_proj \
    --whichMLP_vis fc2 \
    --numSamplesConsidered 50


'''


import argparse
import os
import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from bert_score import score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM


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

ega_ratio = 0.2
towardsNull = 0.1

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

# ---------------- Local danger-score models ----------------

print("Loading SentenceTransformer model...")
sbert_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)

print("Loading fluency model...")
fluency_model_name = "distilgpt2"
fluency_tokenizer = AutoTokenizer.from_pretrained(fluency_model_name)
fluency_model = AutoModelForCausalLM.from_pretrained(fluency_model_name).to(device)
fluency_model.eval()

if fluency_tokenizer.pad_token is None:
    fluency_tokenizer.pad_token = fluency_tokenizer.eos_token


def compute_sbert_similarity(clean_text, adv_text):
    clean_emb = sbert_model.encode(clean_text, convert_to_numpy=True)
    adv_emb = sbert_model.encode(adv_text, convert_to_numpy=True)

    sim = cosine_similarity([clean_emb], [adv_emb])[0][0]
    sim = float(sim)

    # Clamp to [0, 1]
    sim = max(0.0, min(1.0, sim))
    return sim


def compute_perplexity(text, max_length=512):
    if text.strip() == "":
        return 1e6

    encodings = fluency_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    input_ids = encodings.input_ids.to(device)
    attention_mask = encodings.attention_mask.to(device)

    with torch.no_grad():
        outputs = fluency_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )

    loss = outputs.loss.item()

    if math.isnan(loss) or math.isinf(loss):
        return 1e6

    ppl = math.exp(min(loss, 20))
    return float(ppl)


def perplexity_to_plausibility(ppl):
    # Maps perplexity to [0, 1].
    # Lower perplexity means more fluent/plausible.
    #
    # ppl around 20-50: good/normal text
    # ppl around 100+: weaker
    # ppl very high: likely garbage
    plausibility = 1.0 / (1.0 + math.log(max(ppl, 1.0)))

    plausibility = max(0.0, min(1.0, plausibility))
    return plausibility


def compute_danger_score(clean_text, adv_text):
    # BERTScore similarity
    _, _, bert_f1 = score(
        [adv_text],
        [clean_text],
        lang="en",
        model_type="roberta-large",
        rescale_with_baseline=True,
        verbose=False,
    )

    bert_f1_value = float(bert_f1.item())

    # BERTScore with rescale_with_baseline=True can sometimes be negative.
    # Normalize/clamp to [0, 1].
    bert_similarity = max(0.0, min(1.0, bert_f1_value))

    # Attack failure is high when attacked output differs from clean output.
    task_failure = 1.0 - bert_similarity

    # Semantic preservation is high when attacked output remains on-topic.
    semantic_preservation = compute_sbert_similarity(clean_text, adv_text)

    # Plausibility is high when attacked output is fluent/natural.
    ppl = compute_perplexity(adv_text)
    plausibility = perplexity_to_plausibility(ppl)

    danger_score = task_failure * semantic_preservation * plausibility

    return task_failure, plausibility, semantic_preservation, danger_score, bert_similarity, ppl


all_attck_types = [
    "saa_loopR",
    "bsa",
    "dra",
    "fdam",
    "ssp",
    "ega",
    "nllm",
    "saav",
    "saa",
    "saa_loop",
]

type_sampleAggP = []
type_sampleAggR = []
type_samleAggF1 = []

checkInd = -1

for attck_type in all_attck_types:
    sampleAggP = []
    sampleAggR = []
    samleAggF1 = []

    print("attck_type", attck_type)

    for attackSample in range(1, numSamplesConsidered):
        if attck_type == "saa":
            advOutputPath = f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_num_steps_{num_steps}_towardsNull_{towardsNull}_{whichMLP}.txt"

        elif attck_type == "saav":
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
            chosenLanLayersR = [3, 4, 5]
            chosenVisLayersR = [1, 3, 5, 12, 17, 21]

            advOutputPath = (
                f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/"
                f"advOutput_attackType_{attck_typeR}_lr_{lr}_eps_{epsilon}_"
                f"AttackStartLayer_{AttackStartLayerR}_numLayerstAtAtime_{numLayerstAtAtimeR}_"
                f"num_steps_{num_steps}_towardsNull_{towardsNullR}_"
                f"lanMLP_{whichMLPR}_visMLP_{whichMLPvisR}_"
                f"lanLayers_{chosenLanLayersR}_visLayers_{chosenVisLayersR}.txt"
            )

        else:
            advOutputPath = f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_num_steps_{num_steps}_.txt"

        cleanOutputPath = f"gemma_attack/outputsStorageImagenet/advOutputs/{attackSample}/cleanOutput.txt"

        with open(advOutputPath, "r") as f:
            advOutput = f.read().strip()

        with open(cleanOutputPath, "r") as f:
            cleanOutput = f.read().strip()

        (
            task_failure_score,
            plausibility_score,
            semantic_preservation_score,
            danger_score,
            bert_similarity,
            ppl,
        ) = compute_danger_score(cleanOutput, advOutput)

        print(
            "attackSample",
            attackSample,
            "task_failure",
            task_failure_score,
            "plausibility",
            plausibility_score,
            "semantic_preservation",
            semantic_preservation_score,
            "danger_score",
            danger_score,
            "bert_similarity",
            bert_similarity,
            "ppl",
            ppl,
        )

        # Preserving your old variable structure:
        # sampleAggP  -> Task Failure
        # sampleAggR  -> Plausibility
        # samleAggF1  -> Danger Score
        sampleAggP.append(task_failure_score)
        sampleAggR.append(plausibility_score)
        samleAggF1.append(danger_score)

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


AllAttckTypes = [
    "SSPMA-R",
    "BSA",
    "DRA",
    "FDA",
    "SSPA",
    "EGA",
    "CE",
    "SSPMA-E",
    "SSPMA-L",
    "SSPMA-EL",
]

save_dir = "gemma_attack/AllPlots/comparisionDanger"
os.makedirs(save_dir, exist_ok=True)


plt.figure(figsize=(5, 3))
plt.boxplot(type_sampleAggP, labels=AllAttckTypes)
plt.xticks(rotation=45, ha="right")

plt.ylabel("Task Failure score")
plt.xlabel("Attack Type")
plt.title("Distribution of Task Failure across Attack Types")
plt.tight_layout()
plt.savefig(
    f"{save_dir}/precision_boxplot_eps_{epsilon}_num_steps_{num_steps}_AttackStartLayer_{AttackStartLayer}_towardsNull_{towardsNull}_numSamplesConsidered_{numSamplesConsidered}_{whichMLP}_nll_fdam_cor_coherent_{chosenLanLayers}_{chosenVisLayers}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
plt.close()


plt.figure(figsize=(5, 3))
plt.boxplot(type_sampleAggR, labels=AllAttckTypes)
plt.xticks(rotation=45, ha="right")

plt.ylabel("Plausibility score")
plt.xlabel("Attack Type")
plt.title("Distribution of Plausibility across Attack Types")
plt.tight_layout()
plt.savefig(
    f"{save_dir}/recall_boxplot_eps_{epsilon}_num_steps_{num_steps}_AttackStartLayer_{AttackStartLayer}_towardsNull_{towardsNull}_numSamplesConsidered_{numSamplesConsidered}_{whichMLP}_nll_fdam_cor_coherent_{chosenLanLayers}_{chosenVisLayers}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
plt.close()


plt.figure(figsize=(5, 3))
plt.boxplot(type_samleAggF1, labels=AllAttckTypes)
plt.xticks(rotation=45, ha="right")

plt.ylabel("Danger score")
plt.xlabel("Attack Type")
plt.title("Distribution of Danger Score across Attack Types")
plt.tight_layout()
plt.savefig(
    f"{save_dir}/f1score_boxplot_eps_{epsilon}_num_steps_{num_steps}_AttackStartLayer_{AttackStartLayer}_towardsNull_{towardsNull}_numSamplesConsidered_{numSamplesConsidered}_{whichMLP}_nll_fdam_cor_coherent_{chosenLanLayers}_{chosenVisLayers}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
plt.close()


print("checkInd", checkInd)