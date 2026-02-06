



'''

export CUDA_VISIBLE_DEVICES=5
conda activate gemma3
cd interpretAttacks
python gemma_attack/gemma3ConditioningNew.py 


'''

import os
import sys
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


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


def main():
    MODEL_PATH = "../illcond/gemma_attack/Gemma3-4b"

    os.makedirs("outputsStorage", exist_ok=True)
    os.makedirs("outputsStorage/convergence", exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"device={device}, dtype={dtype}")

    print("Loading processor...")

    print("Loading model...")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    model.config.use_cache = False

    print("\n=== MODEL PARAMETERS (name → shape) ===")

    for name, param in model.named_parameters():
        print(f"{name:60s} {tuple(param.shape)}")


if __name__ == "__main__":
    main()

