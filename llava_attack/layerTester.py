'''

export CUDA_VISIBLE_DEVICES=0
cd interpretAttacks/
conda activate llava15
python llava_attack/layerTester.py



Imp : 
1

FILE NAME: advOutput_attackType_saa_lr_0.001_eps_0.02_num_steps_1000_AlignLayer_11_towardsNull_0.0_whichLayMod_vis_whichMLP_fc2_.txt
FILE NAME: advOutput_attackType_saa_lr_0.001_eps_0.02_num_steps_1000_AlignLayer_14_towardsNull_0.1_whichLayMod_vis_whichMLP_fc1_.txt

FILE NAME: advOutput_attackType_saa_lr_0.001_eps_0.02_num_steps_1000_AlignLayer_8_towardsNull_0.1_whichLayMod_vis_whichMLP_fc2_.txt
FILE NAME: advOutput_attackType_saa_lr_0.001_eps_0.02_num_steps_1000_AlignLayer_9_towardsNull_0.1_whichLayMod_vis_whichMLP_fc1_.txt

FILE NAME: advOutput_attackType_saa_lr_0.001_eps_0.02_num_steps_1000_AlignLayer_11_towardsNull_0.0_whichLayMod_vis_whichMLP_fc2_.txt
FILE NAME: advOutput_attackType_saa_lr_0.001_eps_0.02_num_steps_1000_AlignLayer_12_towardsNull_0.1_whichLayMod_vis_whichMLP_fc1_.txt

advOutput_attackType_saa_lr_0.001_eps_0.02_num_steps_1000_AlignLayer_23_towardsNull_0.0_whichLayMod_vis_whichMLP_fc2_.txt

FILE NAME: advOutput_attackType_saa_lr_0.001_eps_0.02_num_steps_1000_AlignLayer_8_towardsNull_0.1_whichLayMod_vis_whichMLP_fc1_.txt

FILE NAME: advOutput_attackType_saa_lr_0.001_eps_0.02_num_steps_1000_AlignLayer_8_towardsNull_0.1_whichLayMod_vis_whichMLP_fc2_.txt

FILE NAME: advOutput_attackType_saa_lr_0.001_eps_0.02_num_steps_1000_AlignLayer_9_towardsNull_0.0_whichLayMod_vis_whichMLP_fc1_.txt


2:
FILE NAME: advOutput_attackType_saa_lr_0.001_eps_0.02_num_steps_1000_AlignLayer_20_towardsNull_0.0_whichLayMod_vis_whichMLP_fc1_.txt


'''

import os
from pathlib import Path

which_sample = 4
# Input directory containing the .txt files
input_dir = Path("/home/luser/interpretAttacks/llava_attack/outputsStorage/advOutputsN/"+str(which_sample)+"")

# Output file
output_file = Path("/home/luser/interpretAttacks/llava_attack/outputsStorage/which_sample_"+str(which_sample)+".txt")

# Collect all txt files and sort them for reproducibility
txt_files = sorted(input_dir.glob("*.txt"))

with open(output_file, "w", encoding="utf-8") as outfile:
    for txt_file in txt_files:
        outfile.write("=" * 100 + "\n")
        outfile.write(f"FILE NAME: {txt_file.name}\n")
        outfile.write("=" * 100 + "\n\n")

        try:
            with open(txt_file, "r", encoding="utf-8") as infile:
                content = infile.read()
                outfile.write(content)
        except UnicodeDecodeError:
            # Fallback for files with non-UTF8 encoding
            with open(txt_file, "r", encoding="latin-1") as infile:
                content = infile.read()
                outfile.write(content)
        except Exception as e:
            outfile.write(f"[ERROR READING FILE: {e}]")

        outfile.write("\n\n\n")

print(f"Combined file saved to: {output_file}")
print(f"Processed {len(txt_files)} files.")