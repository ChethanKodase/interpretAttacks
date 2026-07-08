



'''



export CUDA_VISIBLE_DEVICES=2
conda deactivate
cd interpretAttacks/
conda activate vlmAttack
python qwen/Qwen2p5ConditioningNew.py 

'''

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import random
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(42)

plotFinishedThings = False

def main():
    MODEL_PATH = "../illcond/QwenAttack/Qwen2.5-VL-7B-Instruct"

    #os.makedirs("outputsStorage", exist_ok=True)
    #os.makedirs("outputsStorage/convergence", exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16  # you can also try torch.float16


    print("Loading processor...")

    print("Loading model and processor...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=DTYPE,
        device_map=None,
    ).to(DEVICE)

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    print("\n=== MODEL PARAMETERS (name → shape) ===")

 
    for name, param in model.named_parameters():
        print(f"{name:60s} {tuple(param.shape)}")
        #U, S, Vh = torch.linalg.svd(param.to(torch.float32))

if __name__ == "__main__":
    main()

