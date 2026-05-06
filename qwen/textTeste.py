'''
conda activate gemma3



'''


import os

# Directory containing the txt files
dir_path = "/home/luser/interpretAttacks/qwen/outputsStorageImagenet/advOutputs/1"

output_file = os.path.join(dir_path, "aaaa.txt")

with open(output_file, "w") as outfile:
    for filename in sorted(os.listdir(dir_path)):
        if filename.endswith(".txt") and filename != "aaaa.txt":
            file_path = os.path.join(dir_path, filename)
            
            with open(file_path, "r") as infile:
                content = infile.read()
            
            # Write in required format
            outfile.write(f"{filename} : : {content}\n")
            outfile.write("-------------------\n")

print(f"Combined file saved at: {output_file}")