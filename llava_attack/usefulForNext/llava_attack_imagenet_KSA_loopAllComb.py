

'''


-----------Vis outproj




export CUDA_VISIBLE_DEVICES=0
cd interpretAttacks/
conda activate llava15
for ATTACK_SAMPLE in $(seq 1 500); do
    for ALIGNLAYER in $(seq 0 23); do
        python llava_attack/llava_attack_imagenet_KSA.py --attck_type saa --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AlignLayer $ALIGNLAYER --towardsNull 0.1 --whichLayMod vis --whichMLP out_proj
        python llava_attack/llava_attack_imagenet_KSA.py --attck_type saa --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AlignLayer $ALIGNLAYER --towardsNull 0 --whichLayMod vis --whichMLP out_proj

        python llava_attack/llava_attack_imagenet_KSA.py --attck_type saa --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AlignLayer $ALIGNLAYER --towardsNull 0.1 --whichLayMod vis --whichMLP fc1
        python llava_attack/llava_attack_imagenet_KSA.py --attck_type saa --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AlignLayer $ALIGNLAYER --towardsNull 0 --whichLayMod vis --whichMLP fc1
    done
done




export CUDA_VISIBLE_DEVICES=1
cd interpretAttacks/
conda activate llava15
for ATTACK_SAMPLE in $(seq 1 500); do
    for ALIGNLAYER in $(seq 0 23); do
        python llava_attack/llava_attack_imagenet_KSA.py --attck_type saa --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AlignLayer $ALIGNLAYER --towardsNull 0.1 --whichLayMod vis --whichMLP fc2
        python llava_attack/llava_attack_imagenet_KSA.py --attck_type saa --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AlignLayer $ALIGNLAYER --towardsNull 0 --whichLayMod vis --whichMLP fc2
    done
done




export CUDA_VISIBLE_DEVICES=2
cd interpretAttacks/
conda activate llava15
for ATTACK_SAMPLE in $(seq 1 500); do
    for ALIGNLAYER in $(seq 0 31); do
        python llava_attack/llava_attack_imagenet_KSA.py --attck_type saa --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AlignLayer $ALIGNLAYER --towardsNull 0.1 --whichLayMod lan --whichMLP up_proj
        python llava_attack/llava_attack_imagenet_KSA.py --attck_type saa --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AlignLayer $ALIGNLAYER --towardsNull 0 --whichLayMod lan --whichMLP up_proj
    done
done


export CUDA_VISIBLE_DEVICES=3
cd interpretAttacks/
conda activate llava15
for ATTACK_SAMPLE in $(seq 1 500); do
    for ALIGNLAYER in $(seq 0 31); do
        python llava_attack/llava_attack_imagenet_KSA.py --attck_type saa --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AlignLayer $ALIGNLAYER --towardsNull 0.1 --whichLayMod lan --whichMLP down_proj
        python llava_attack/llava_attack_imagenet_KSA.py --attck_type saa --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AlignLayer $ALIGNLAYER --towardsNull 0 --whichLayMod lan --whichMLP down_proj
    done
done


-----------Vis fc1

-----------Vis fc2

-----------lan up_proj

-----------lan down_proj



parser.add_argument("--AlignLayer", type=int, default=1,
                    help="values taken are 0 to 23 for vis and 0 to 31 for lan ")
parser.add_argument("--whichLayMod", type=str, default="vis",
                    help="values taken are vis and lan")
parser.add_argument("--whichMLP", type=str, default="fc1",
                    help="values taken : down_proj, up_proj, fc1, fc2, out_proj")



------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------                    

export CUDA_VISIBLE_DEVICES=0
cd interpretAttacks/
conda activate llava15
for ATTACK_SAMPLE in $(seq 1 5); do
    for CHOSEN_LAN_LAYER in $(seq 0 5); do
        for CHOSEN_VIS_LAYER in $(seq 0 23); do
            python llava_attack/llava_attack_imagenet_KSA_loop.py --attck_type saa_loop --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.5 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER 
            python llava_attack/llava_attack_imagenet_KSA_loop.py --attck_type saa_loop --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER 
        done
    done
done



export CUDA_VISIBLE_DEVICES=1
cd interpretAttacks/
conda activate llava15
for ATTACK_SAMPLE in $(seq 1 5); do
    for CHOSEN_LAN_LAYER in $(seq 0 5); do
        for CHOSEN_VIS_LAYER in $(seq 0 23); do
            python llava_attack/llava_attack_imagenet_KSA_loop.py --attck_type saa_loop --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.5 --whichMLP gate_proj --whichMLPVis fc2 --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER 
            python llava_attack/llava_attack_imagenet_KSA_loop.py --attck_type saa_loop --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --whichMLP gate_proj --whichMLPVis fc2 --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER 
        done
    done
done


export CUDA_VISIBLE_DEVICES=2
cd interpretAttacks/
conda activate llava15
for ATTACK_SAMPLE in $(seq 1 5); do
    for CHOSEN_LAN_LAYER in $(seq 0 5); do
        for CHOSEN_VIS_LAYER in $(seq 0 23); do
            python llava_attack/llava_attack_imagenet_KSA_loop.py --attck_type saa_loop --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.5 --whichMLP gate_proj --whichMLPVis out_proj --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER 
            python llava_attack/llava_attack_imagenet_KSA_loop.py --attck_type saa_loop --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --whichMLP gate_proj --whichMLPVis out_proj --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER 
        done
    done
done

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------                    

export CUDA_VISIBLE_DEVICES=3
cd interpretAttacks/
conda activate llava15
for ATTACK_SAMPLE in $(seq 1 5); do
    for CHOSEN_LAN_LAYER in $(seq 0 5); do
        for CHOSEN_VIS_LAYER in $(seq 0 23); do
            python llava_attack/llava_attack_imagenet_KSA_loop.py --attck_type saa_loop --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.5 --whichMLP up_proj --whichMLPVis fc1 --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER 
            python llava_attack/llava_attack_imagenet_KSA_loop.py --attck_type saa_loop --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --whichMLP up_proj --whichMLPVis fc1 --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER 
        done
    done
done



export CUDA_VISIBLE_DEVICES=1
cd interpretAttacks/
conda activate llava15
for ATTACK_SAMPLE in $(seq 1 5); do
    for CHOSEN_LAN_LAYER in $(seq 0 5); do
        for CHOSEN_VIS_LAYER in $(seq 0 23); do
            python llava_attack/llava_attack_imagenet_KSA_loop.py --attck_type saa_loop --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.5 --whichMLP up_proj --whichMLPVis fc2 --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER 
            python llava_attack/llava_attack_imagenet_KSA_loop.py --attck_type saa_loop --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --whichMLP up_proj --whichMLPVis fc2 --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER 
        done
    done
done


export CUDA_VISIBLE_DEVICES=2
cd interpretAttacks/
conda activate llava15
for ATTACK_SAMPLE in $(seq 1 5); do
    for CHOSEN_LAN_LAYER in $(seq 0 5); do
        for CHOSEN_VIS_LAYER in $(seq 0 23); do
            python llava_attack/llava_attack_imagenet_KSA_loop.py --attck_type saa_loop --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.5 --whichMLP up_proj --whichMLPVis out_proj --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER 
            python llava_attack/llava_attack_imagenet_KSA_loop.py --attck_type saa_loop --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --whichMLP up_proj --whichMLPVis out_proj --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER 
        done
    done
done

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------                    

export CUDA_VISIBLE_DEVICES=0
cd interpretAttacks/
conda activate llava15
for ATTACK_SAMPLE in $(seq 1 5); do
    for CHOSEN_LAN_LAYER in $(seq 0 5); do
        for CHOSEN_VIS_LAYER in $(seq 0 23); do
            python llava_attack/llava_attack_imagenet_KSA_loop.py --attck_type saa_loop --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.5 --whichMLP down_proj --whichMLPVis fc1 --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER 
            python llava_attack/llava_attack_imagenet_KSA_loop.py --attck_type saa_loop --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --whichMLP down_proj --whichMLPVis fc1 --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER 
        done
    done
done



export CUDA_VISIBLE_DEVICES=1
cd interpretAttacks/
conda activate llava15
for ATTACK_SAMPLE in $(seq 1 5); do
    for CHOSEN_LAN_LAYER in $(seq 0 5); do
        for CHOSEN_VIS_LAYER in $(seq 0 23); do
            python llava_attack/llava_attack_imagenet_KSA_loop.py --attck_type saa_loop --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.5 --whichMLP down_proj --whichMLPVis fc2 --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER 
            python llava_attack/llava_attack_imagenet_KSA_loop.py --attck_type saa_loop --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --whichMLP down_proj --whichMLPVis fc2 --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER 
        done
    done
done


export CUDA_VISIBLE_DEVICES=2
cd interpretAttacks/
conda activate llava15
for ATTACK_SAMPLE in $(seq 1 5); do
    for CHOSEN_LAN_LAYER in $(seq 0 5); do
        for CHOSEN_VIS_LAYER in $(seq 0 23); do
            python llava_attack/llava_attack_imagenet_KSA_loop.py --attck_type saa_loop --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.5 --whichMLP down_proj --whichMLPVis out_proj --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER 
            python llava_attack/llava_attack_imagenet_KSA_loop.py --attck_type saa_loop --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --whichMLP down_proj --whichMLPVis out_proj --chosenLanLayers $CHOSEN_LAN_LAYER --chosenVisLayers $CHOSEN_VIS_LAYER 
        done
    done
done

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


export CUDA_VISIBLE_DEVICES=0
cd interpretAttacks/
conda activate llava15
python llava_attack/llava_attack_imagenet_KSA_loop.py --attck_type saa_loop --desired_norm_l_inf 0.02 --learningRate 0.0001 --num_steps 1000 --attackSample 1 --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --whichMLP gate_proj --whichMLPVis fc2 --chosenLanLayers 0 --chosenVisLayers 0
#  The image shows a white flower with a yellow center, sitting on top of a black and white checkered surface. The flower is positioned in the center of the image, and the checkered surface is visible around it.

export CUDA_VISIBLE_DEVICES=0
cd interpretAttacks/
conda activate llava15
python llava_attack/llava_attack_imagenet_KSA_loop.py --attck_type saa_loop --desired_norm_l_inf 0.04 --learningRate 0.001 --num_steps 1000 --attackSample 1 --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.5 --whichMLP gate_proj --whichMLPVis fc2 --chosenLanLayers 0 --chosenVisLayers 0
# The image shows a large green field with a grassy area where a dragonfly is resting. The dragonfly is positioned towards the right side of the image, and it appears to be a close-up shot of the insect. The field provides a natural and serene environment for the dragonfly to rest in.                       


export CUDA_VISIBLE_DEVICES=1
cd interpretAttacks/
conda activate llava15
python llava_attack/llava_attack_imagenet_KSA_loop.py --attck_type saa_loop --desired_norm_l_inf 0.04 --learningRate 0.001 --num_steps 100 --attackSample 1 --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.5 --whichMLP gate_proj --whichMLPVis fc2 --chosenLanLayers 0 --chosenVisLayers 0
#The image shows a close-up of a white flower with a yellow center, surrounded by green grass. The flower is located in the foreground, while a snake is seen in the background, partially hidden by the grass. The scene appears to be a mix of nature, with the flower and the snake coexisting in the same environment.

export CUDA_VISIBLE_DEVICES=1
cd interpretAttacks/
conda activate llava15
python llava_attack/llava_attack_imagenet_KSA_loop.py --attck_type saa_loop --desired_norm_l_inf 0.02 --learningRate 0.001 --num_steps 1000 --attackSample 1 --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.6 --whichMLP gate_proj --whichMLPVis fc2 --chosenLanLayers 0 --chosenVisLayers 0


export CUDA_VISIBLE_DEVICES=0
cd interpretAttacks/
conda activate llava15
python llava_attack/llava_attack_imagenet_KSA_loop.py --attck_type saa_loop --desired_norm_l_inf 0.04 --learningRate 0.001 --num_steps 1000 --attackSample 1 --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.5 --whichMLP gate_proj --whichMLPVis fc2 --chosenLanLayers 0 --chosenVisLayers 0


export CUDA_VISIBLE_DEVICES=0
cd interpretAttacks/
conda activate llava15
python llava_attack/llava_attack_imagenet_KSA_loop.py --attck_type saa_loop --desired_norm_l_inf 0.002 --learningRate 0.001 --num_steps 1000 --attackSample 1 --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.5 --whichMLP gate_proj --whichMLPVis fc2 --chosenLanLayers 0 --chosenVisLayers 0


lan 31
vis 23




gate_proj fc1
gate_proj fc2
gate_proj out_proj

up_proj fc1
up_proj fc2
up_proj out_proj

down_proj fc1
down_proj fc2
down_proj out_proj

3 16 may be the 2 number configuration





export CUDA_VISIBLE_DEVICES=1
cd interpretAttacks/
conda activate llava15
python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --attackSample 1 --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 --chosenVisLayers 8



export CUDA_VISIBLE_DEVICES=1
cd interpretAttacks/
conda activate llava15
python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --attackSample 1 --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.5 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13


export CUDA_VISIBLE_DEVICES=2
cd interpretAttacks/
conda activate llava15
python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.002 --learningRate 0.001 --num_steps 1000 --attackSample 1 --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.05 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13




export CUDA_VISIBLE_DEVICES=3
cd interpretAttacks/
conda activate llava15
python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --attackSample 1 --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 1.0 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13


export CUDA_VISIBLE_DEVICES=0
cd interpretAttacks/
conda activate llava15
python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.002 --learningRate 0.001 --num_steps 1000 --attackSample 1 --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.01 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

export CUDA_VISIBLE_DEVICES=0
cd interpretAttacks/
conda activate llava15
for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.002 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.01 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13
done

for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.01 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13
done

for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.0025 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.01 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13
done

for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.0045 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.01 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13
done

export CUDA_VISIBLE_DEVICES=1
cd interpretAttacks/
conda activate llava15
for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.002 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.05 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13
done

for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.05 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13
done

for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.0025 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.05 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13
done

for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.0045 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.05 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13
done
--------------------------------------------------------------------

export CUDA_VISIBLE_DEVICES=2
cd interpretAttacks/
conda activate llava15
for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.003 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.01 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13
done

for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.01 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13
done

for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.0035 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.01 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13
done

export CUDA_VISIBLE_DEVICES=3
cd interpretAttacks/
conda activate llava15
for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.003 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.05 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.05 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.0035 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.05 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13
done


-----------------------------
----------------------------
export CUDA_VISIBLE_DEVICES=0
cd interpretAttacks/
conda activate llava15
for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.002 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.02 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.0025 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.02 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.003 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.02 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.0035 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.02 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.02 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.0045 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.02 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.02 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13
done


export CUDA_VISIBLE_DEVICES=1
cd interpretAttacks/
conda activate llava15
for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.002 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.03 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.0025 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.03 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.003 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.03 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.0035 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.03 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.03 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.0045 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.03 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.03 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13
done


export CUDA_VISIBLE_DEVICES=2
cd interpretAttacks/
conda activate llava15
for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.002 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.04 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.0025 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.04 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.003 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.04 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.0035 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.04 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.04 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.0045 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.04 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.04 --whichMLP gate_proj --whichMLPVis fc1 --chosenLanLayers 1 3 --chosenVisLayers 7 8 17 13
done



export CUDA_VISIBLE_DEVICES=0
cd interpretAttacks/
conda activate llava15
for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.003 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.06 --whichMLP gate_proj --whichMLPVis fc1 \
        --chosenLanLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31\
        --chosenVisLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.004 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.06 --whichMLP gate_proj --whichMLPVis fc1 \
        --chosenLanLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31\
        --chosenVisLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
done



export CUDA_VISIBLE_DEVICES=1
cd interpretAttacks/
conda activate llava15
for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.002 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.06 --whichMLP gate_proj --whichMLPVis fc1 \
        --chosenLanLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31\
        --chosenVisLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
done
for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.0025 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.06 --whichMLP gate_proj --whichMLPVis fc1 \
        --chosenLanLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31\
        --chosenVisLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
done


export CUDA_VISIBLE_DEVICES=2
cd interpretAttacks/
conda activate llava15
for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.0035 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.06 --whichMLP gate_proj --whichMLPVis fc1 \
        --chosenLanLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31\
        --chosenVisLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
done

for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.0045 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.06 --whichMLP gate_proj --whichMLPVis fc1 \
        --chosenLanLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31\
        --chosenVisLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
done



export CUDA_VISIBLE_DEVICES=3
cd interpretAttacks/
conda activate llava15
for ATTACK_SAMPLE in $(seq 1 50); do
    python llava_attack/llava_attack_imagenet_KSA_loopAllComb.py --attck_type saa_loopAllC --desired_norm_l_inf 0.005 --learningRate 0.001 --num_steps 1000 --attackSample $ATTACK_SAMPLE --AttackStartLayer 0 --numLayerstAtAtime 1 --towardsNull 0.1 --BalAlpha 0.06 --whichMLP gate_proj --whichMLPVis fc1 \
        --chosenLanLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31\
        --chosenVisLayers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
done
'''



#!/usr/bin/env python
import os
import re
import argparse
import random
import numpy as np

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import (
    LlavaForConditionalGeneration,
    CLIPImageProcessor,
    LlamaTokenizer,
)

# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

try:
    torch.use_deterministic_algorithms(True)
except Exception as e:
    print(f"[WARN] deterministic algorithms failed: {e}")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


criterion = nn.MSELoss()
layer_inputs = {}


# ----------------------------
# Image helpers
# ----------------------------
def pil_to_tensor01(pil_img: Image.Image) -> torch.Tensor:
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def tensor01_to_pil(t01: torch.Tensor) -> Image.Image:
    if t01.dim() == 4:
        t01 = t01[0]
    t01 = t01.detach().cpu().clamp(0, 1)
    arr = (t01.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ----------------------------
# Differentiable CLIP preprocessing
# ----------------------------
def _get_target_hw(image_processor):
    crop = getattr(image_processor, "crop_size", None)
    size = getattr(image_processor, "size", None)

    target_h = target_w = None

    if isinstance(crop, dict):
        target_h = crop.get("height", None)
        target_w = crop.get("width", None)
    elif isinstance(crop, int):
        target_h = target_w = crop

    resize_short = None

    if isinstance(size, dict) and "shortest_edge" in size:
        resize_short = size["shortest_edge"]
    elif isinstance(size, dict) and "height" in size and "width" in size:
        resize_short = min(size["height"], size["width"])
    elif isinstance(size, int):
        resize_short = size

    if target_h is None or target_w is None:
        target_h = target_w = 224

    if resize_short is None:
        resize_short = min(target_h, target_w)

    return int(resize_short), int(target_h), int(target_w)


def resize_shortest_edge_keep_aspect(x: torch.Tensor, shortest_edge: int) -> torch.Tensor:
    _, _, H, W = x.shape
    scale = shortest_edge / min(H, W)
    newH = int(round(H * scale))
    newW = int(round(W * scale))
    return F.interpolate(x, size=(newH, newW), mode="bilinear", align_corners=False)


def center_crop(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    _, _, H, W = x.shape

    top = max((H - target_h) // 2, 0)
    left = max((W - target_w) // 2, 0)

    x_crop = x[:, :, top:top + target_h, left:left + target_w]

    pad_h = target_h - x_crop.shape[2]
    pad_w = target_w - x_crop.shape[3]

    if pad_h > 0 or pad_w > 0:
        x_crop = F.pad(x_crop, (0, max(pad_w, 0), 0, max(pad_h, 0)))

    return x_crop


def normalize_like_processor(x01: torch.Tensor, image_processor) -> torch.Tensor:
    mean = torch.tensor(image_processor.image_mean, dtype=x01.dtype, device=x01.device).view(1, 3, 1, 1)
    std = torch.tensor(image_processor.image_std, dtype=x01.dtype, device=x01.device).view(1, 3, 1, 1)
    return (x01 - mean) / std


def llava_preprocess_differentiable(x01: torch.Tensor, image_processor) -> torch.Tensor:
    shortest_edge, th, tw = _get_target_hw(image_processor)
    x = resize_shortest_edge_keep_aspect(x01, shortest_edge)
    x = center_crop(x, th, tw)
    x = normalize_like_processor(x, image_processor)
    return x


def get_bsa_loss(outputs, outputsN):
    """
    Token-wise cosine similarity over LLaVA language hidden states.
    The attack minimizes this value, matching your Qwen BSA implementation.
    """
    hidden_states = outputs.hidden_states
    clean_hidden_states = outputsN.hidden_states

    loss = 0.0
    for h, hn in zip(hidden_states, clean_hidden_states):
        cos_per_token = F.cosine_similarity(h.squeeze(0), hn.squeeze(0), dim=1)
        loss = loss + cos_per_token.sum()
    return loss


def get_bsa_vision_loss(acts, actsN):
    """
    Token-wise cosine similarity over CLIP vision tower hidden states.
    """
    loss = 0.0
    for h, hn in zip(acts, actsN):
        cos_per_token = F.cosine_similarity(h, hn, dim=-1)
        loss = loss + cos_per_token.sum()
    return loss


# ----------------------------
# Prompt helpers
# ----------------------------
def build_template_inputs(tokenizer, question: str, device):
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    enc = tokenizer(prompt, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


def run_generation_with_pixel_values(model, tokenizer, template_inputs, pixel_values, max_new_tokens=128):
    model.eval()

    inputs = {
        k: v.clone() if torch.is_tensor(v) else v
        for k, v in template_inputs.items()
    }

    inputs["pixel_values"] = pixel_values

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    input_ids = inputs["input_ids"]
    gen_only = out_ids[:, input_ids.shape[1]:]

    return tokenizer.decode(gen_only[0], skip_special_tokens=True)


# ============================================================
# Flexible Qwen-style LLaVA bottom-subspace targeting
# ============================================================
def make_pre_hook(name):
    def hook(module, inputs):
        layer_inputs[name] = inputs[0]
    return hook


def extract_language_layer_idx(name: str):
    patterns = [
        r"language_model\.model\.layers\.(\d+)\.",
        r"language_model\.layers\.(\d+)\.",
        r"model\.layers\.(\d+)\.",
    ]

    for p in patterns:
        m = re.search(p, name)
        if m:
            return int(m.group(1))

    return None


def extract_vision_layer_idx(name: str):
    patterns = [
        r"vision_tower\.vision_model\.encoder\.layers\.(\d+)\.",
        r"vision_model\.encoder\.layers\.(\d+)\.",
    ]

    for p in patterns:
        m = re.search(p, name)
        if m:
            return int(m.group(1))

    return None


def is_language_target(name: str, chosen_lan_layers_set=None, whichMLP="gate_proj") -> bool:
    layer_idx = extract_language_layer_idx(name)

    if layer_idx is None:
        return False

    if chosen_lan_layers_set is not None and layer_idx not in chosen_lan_layers_set:
        return False

    valid_suffixes = {
        "gate_proj": ".mlp.gate_proj",
        "up_proj": ".mlp.up_proj",
        "down_proj": ".mlp.down_proj",
    }

    suffix = valid_suffixes[whichMLP]
    return name.endswith(suffix)


def is_vision_target(name: str, chosen_vis_layers_set=None, whichMLPVis="fc2") -> bool:
    layer_idx = extract_vision_layer_idx(name)

    if layer_idx is None:
        return False

    if chosen_vis_layers_set is not None and layer_idx not in chosen_vis_layers_set:
        return False

    valid_suffixes = {
        "fc1": ".mlp.fc1",
        "fc2": ".mlp.fc2",
        "q_proj": ".self_attn.q_proj",
        "k_proj": ".self_attn.k_proj",
        "v_proj": ".self_attn.v_proj",
        "out_proj": ".self_attn.out_proj",
    }

    suffix = valid_suffixes[whichMLPVis]
    return name.endswith(suffix)


def collect_target_modules(
    model,
    chosen_lan_layers=None,
    chosen_vis_layers=None,
    whichMLP="gate_proj",
    whichMLPVis="fc2",
):
    chosen_lan_layers_set = None if chosen_lan_layers is None else set(chosen_lan_layers)
    chosen_vis_layers_set = None if chosen_vis_layers is None else set(chosen_vis_layers)

    targets = []

    for name, module in model.named_modules():
        if not hasattr(module, "weight"):
            continue

        if not torch.is_tensor(module.weight):
            continue

        if module.weight.ndim != 2:
            continue

        if is_language_target(name, chosen_lan_layers_set, whichMLP):
            targets.append(
                {
                    "name": name,
                    "module": module,
                    "kind": "language",
                    "layer_idx": extract_language_layer_idx(name),
                }
            )

        elif is_vision_target(name, chosen_vis_layers_set, whichMLPVis):
            targets.append(
                {
                    "name": name,
                    "module": module,
                    "kind": "vision",
                    "layer_idx": extract_vision_layer_idx(name),
                }
            )

    return targets


def compute_bottom_singular_subspace(weight: torch.Tensor, towardsNull: float):
    with torch.no_grad():
        W = weight.detach().to(torch.float32)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

        absorbSize = int((S < 1).sum().item())

        if towardsNull == 0:
            bottomInd = 1
        else:
            bottomInd = int(absorbSize * towardsNull)

        bottomInd = max(1, bottomInd)
        bottomInd = min(bottomInd, Vh.shape[0])

        V_bottom = Vh[-bottomInd:].contiguous()

    return V_bottom, S, absorbSize, bottomInd


def getMeanAlignmentLossWithBottomSubspace(InputToLayer, bottomRightSingularVectors):
    H = InputToLayer

    if isinstance(H, (tuple, list)):
        H = H[0]

    if H.dim() == 3:
        H = H.reshape(-1, H.shape[-1])
    elif H.dim() == 2:
        pass
    else:
        H = H.view(-1, H.shape[-1])

    V = bottomRightSingularVectors.to(device=H.device, dtype=H.dtype)

    H_hat = F.normalize(H, dim=1)
    V_hat = F.normalize(V, dim=1)

    coeffs = H_hat @ V_hat.T
    per_token_energy = (coeffs ** 2).sum(dim=1)

    loss = ((1.0 - per_token_energy) ** 2).mean()
    return loss

'''def getMeanAlignmentLossWithBottomSubspace(InputToLayer, bottomRightSingularVectors):
    #H = InputToLayer
    H = InputToLayer.float()

    if isinstance(H, (tuple, list)):
        H = H[0]

    if H.dim() == 3:
        H = H.reshape(-1, H.shape[-1])
    elif H.dim() == 2:
        pass
    else:
        H = H.view(-1, H.shape[-1])

    #V = bottomRightSingularVectors.to(device=H.device, dtype=H.dtype)
    V = bottomRightSingularVectors.to(device=H.device, dtype=torch.float32)

    H_hat = F.normalize(H, dim=1)
    V_hat = F.normalize(V, dim=1)

    coeffs = H_hat @ V_hat.T
    per_token_energy = (coeffs ** 2).sum(dim=1)

    #loss = ((1.0 - per_token_energy) ** 2).mean()
    loss = -per_token_energy.mean()
    return loss'''

# ----------------------------
# Vision hooks for LLaVA CLIP tower
# ----------------------------
def get_llava_vision_layers(model):
    """
    Supports the common HuggingFace LLaVA structure:
      model.vision_tower.vision_model.encoder.layers
    plus a few common wrappers.
    """
    if hasattr(model, "vision_tower"):
        vt = model.vision_tower
        if hasattr(vt, "vision_model") and hasattr(vt.vision_model, "encoder"):
            enc = vt.vision_model.encoder
            if hasattr(enc, "layers"):
                return enc.layers
        if hasattr(vt, "encoder") and hasattr(vt.encoder, "layers"):
            return vt.encoder.layers

    if hasattr(model, "model") and hasattr(model.model, "vision_tower"):
        vt = model.model.vision_tower
        if hasattr(vt, "vision_model") and hasattr(vt.vision_model, "encoder"):
            enc = vt.vision_model.encoder
            if hasattr(enc, "layers"):
                return enc.layers

    raise RuntimeError("Could not find LLaVA vision tower encoder layers.")


def run_vision_tower_with_hooks(model, pixel_values):
    """
    Runs only the vision tower and records activations from each CLIP encoder layer.
    These activations are differentiable w.r.t. pixel_values.
    """
    acts = []
    handles = []
    layers = get_llava_vision_layers(model)

    def hook_fn(module, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        if torch.is_tensor(out):
            acts.append(out)

    for layer in layers:
        handles.append(layer.register_forward_hook(hook_fn))

    vision_outputs = model.vision_tower(
        pixel_values,
        output_hidden_states=True,
        return_dict=True,
    )

    for h in handles:
        h.remove()

    return vision_outputs, acts


def build_target_specs_with_subspaces(
    model,
    towardsNull: float,
    chosen_lan_layers=None,
    chosen_vis_layers=None,
    whichMLP="gate_proj",
    whichMLPVis="fc2",
):
    targets = collect_target_modules(
        model,
        chosen_lan_layers=chosen_lan_layers,
        chosen_vis_layers=chosen_vis_layers,
        whichMLP=whichMLP,
        whichMLPVis=whichMLPVis,
    )

    specs = []

    print("\n========== TARGET MODULES ==========")

    for t in targets:
        name = t["name"]
        module = t["module"]

        V_bottom, S, absorbSize, bottomInd = compute_bottom_singular_subspace(
            module.weight,
            towardsNull,
        )

        spec = {
            "name": name,
            "module": module,
            "kind": t["kind"],
            "layer_idx": t["layer_idx"],
            "bottom_vectors": V_bottom,
            "absorbSize": absorbSize,
            "bottomInd": bottomInd,
            "weight_shape": tuple(module.weight.shape),
        }

        specs.append(spec)

        print(
            f"{t['kind']:8s} | layer={t['layer_idx']:3d} | {name} "
            f"| weight_shape={tuple(module.weight.shape)} "
            f"| absorbSize={absorbSize} | bottomInd={bottomInd}"
        )

    num_lang = sum(1 for s in specs if s["kind"] == "language")
    num_vis = sum(1 for s in specs if s["kind"] == "vision")

    print("====================================")
    print(f"Total language targets: {num_lang}")
    print(f"Total vision targets:   {num_vis}")
    print(f"Total targets overall:  {len(specs)}")
    print("====================================\n")

    return specs


def register_all_hooks(target_specs):
    handles = []

    for spec in target_specs:
        handle = spec["module"].register_forward_pre_hook(make_pre_hook(spec["name"]))
        handles.append(handle)

    return handles


def aggregated_bottom_subspace_loss(target_specs, device):
    lang_losses = []
    vis_losses = []

    for spec in target_specs:
        name = spec["name"]

        if name not in layer_inputs:
            continue

        loss_i = getMeanAlignmentLossWithBottomSubspace(
            layer_inputs[name],
            spec["bottom_vectors"],
        )

        if spec["kind"] == "language":
            lang_losses.append(loss_i)
        elif spec["kind"] == "vision":
            vis_losses.append(loss_i)

    language_loss = torch.tensor(0.0, device=device)
    vision_loss = torch.tensor(0.0, device=device)

    if len(lang_losses) > 0:
        language_loss = torch.stack(lang_losses).mean()

    if len(vis_losses) > 0:
        vision_loss = torch.stack(vis_losses).mean()

    total_used = len(lang_losses) + len(vis_losses)

    return language_loss, vision_loss, total_used


# ----------------------------
# ORIGINAL-space Adam attack

#BalAlpha = 1
# ----------------------------
def adam_attack_original_space(
    model,
    tokenizer,
    image_processor,
    template_inputs,
    x_orig01,
    attck_type: str,
    num_steps: int,
    lr: float,
    epsilon: float,
    device,
    save_conv_path: str,
    AttackStartLayer: int,
    numLayerstAtAtime: int,
    towardsNull: float,
    BalAlpha: float,
    whichMLP: str,
    whichMLPVis: str,
    chosenLanLayers=None,
    chosenVisLayers=None
):
    x_orig01 = x_orig01.detach().to(device)

    delta = 0.001 * torch.randn_like(x_orig01, device=device)
    delta.requires_grad_(True)

    opt = torch.optim.Adam([delta], lr=lr)

    losses_list = [0.0]
    best_loss = 1e20
    best_delta = delta.detach().clone()

    model.eval()
    model.config.use_cache = False

    target_specs = build_target_specs_with_subspaces(
        model=model,
        towardsNull=towardsNull,
        chosen_lan_layers=chosenLanLayers,
        chosen_vis_layers=chosenVisLayers,
        whichMLP=whichMLP,
        whichMLPVis=whichMLPVis,
    )

    if len(target_specs) == 0:
        raise RuntimeError(
            "No target modules found. Check --chosenLanLayers, --chosenVisLayers, "
            "--whichMLP, and --whichMLPVis."
        )

    hook_handles = register_all_hooks(target_specs)

    with torch.no_grad():
        pv_clean_fixed = llava_preprocess_differentiable(x_orig01, image_processor)

        clean_inputs = {
            k: v.clone() if torch.is_tensor(v) else v
            for k, v in template_inputs.items()
        }

        clean_inputs["pixel_values"] = pv_clean_fixed
        clean_inputs["labels"] = template_inputs["input_ids"]
        clean_inputs["use_cache"] = False

        outputsN = model(
            **clean_inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        _, actsN = run_vision_tower_with_hooks(model, pv_clean_fixed)

        #print("Number of language hidden states:", len(outputsN.hidden_states))

        hidden_len = len(outputsN.hidden_states)
        vision_len = len(actsN)
        print("Number of language hidden states:", hidden_len)
        print("Number of vision hidden states:", vision_len)

    adv_inputs = {
        k: v.clone() if torch.is_tensor(v) else v
        for k, v in template_inputs.items()
    }

    adv_inputs["labels"] = template_inputs["input_ids"]
    adv_inputs["use_cache"] = False

    for step in range(num_steps):
        layer_inputs.clear()

        x_adv01 = (x_orig01 + delta).clamp(0.0, 1.0)

        x_adv01 = torch.max(
            torch.min(x_adv01, x_orig01 + epsilon),
            x_orig01 - epsilon,
        ).clamp(0.0, 1.0)

        pv_adv = llava_preprocess_differentiable(x_adv01, image_processor)

        adv_inputs["pixel_values"] = pv_adv

        outputs = model(
            **adv_inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        language_loss, vision_loss, total_used = aggregated_bottom_subspace_loss(
            target_specs,
            device=device,
        )
        _, acts = run_vision_tower_with_hooks(model, pv_adv)

        BSALoss = get_bsa_loss(outputs, outputsN) + get_bsa_vision_loss(acts, actsN)

        if total_used == 0:
            raise RuntimeError("No hooked target modules were used in the forward pass.")

        SSloss = language_loss + vision_loss
        attack_loss = BSALoss * BalAlpha + SSloss * (1-BalAlpha)

        opt.zero_grad(set_to_none=True)
        attack_loss.backward()
        opt.step()

        with torch.no_grad():
            delta.data.clamp_(-epsilon, epsilon)

        lv = float(attack_loss.item())
        losses_list.append(lv)

        if step == 0 or (step + 1) % 10 == 0:
            print(
                f"[step {step + 1}/{num_steps}] "
                f"total_loss={lv:.6f} "
                f"language_loss={float(language_loss.item()):.6f} "
                f"vision_loss={float(vision_loss.item()):.6f} "
                f"used_modules={total_used}"
            )

        if lv < best_loss:
            best_loss = lv
            best_delta = delta.detach().clone()
            np.save(save_conv_path, np.array(losses_list, dtype=np.float32))

        del outputs, attack_loss, pv_adv

    for h in hook_handles:
        h.remove()

    with torch.no_grad():
        x_adv01_final = (x_orig01 + best_delta).clamp(0.0, 1.0)
        x_adv01_final = torch.max(
            torch.min(x_adv01_final, x_orig01 + epsilon),
            x_orig01 - epsilon,
        ).clamp(0.0, 1.0)

    return x_adv01_final, best_delta


# ----------------------------
# MAIN
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Flexible LLaVA bottom-subspace attack, Qwen-style selected language + vision layers"
    )

    parser.add_argument("--attck_type", type=str, default="saa_loop")
    parser.add_argument("--desired_norm_l_inf", type=float, default=0.02)
    parser.add_argument("--learningRate", type=float, default=1e-3)

    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--numSteps", type=int, default=None)

    parser.add_argument("--attackSample", type=str, default="astronauts68")

    parser.add_argument("--AttackStartLayer", type=int, default=0)
    parser.add_argument("--numLayerstAtAtime", type=int, default=1)

    parser.add_argument("--towardsNull", type=float, default=0.5)

    parser.add_argument("--BalAlpha", type=float, default=0.5)

    parser.add_argument(
        "--whichMLP",
        type=str,
        default="gate_proj",
        choices=["gate_proj", "up_proj", "down_proj"],
        help="Language MLP projection to target.",
    )

    parser.add_argument(
        "--whichMLPVis",
        type=str,
        default="fc2",
        choices=["fc1", "fc2", "q_proj", "k_proj", "v_proj", "out_proj"],
        help="Vision module projection to target.",
    )

    parser.add_argument(
        "--chosenLanLayers",
        type=int,
        nargs="+",
        default=None,
        help="Language layers to attack, e.g. --chosenLanLayers 2 3 4",
    )

    parser.add_argument(
        "--chosenVisLayers",
        type=int,
        nargs="+",
        default=None,
        help="Vision layers to attack, e.g. --chosenVisLayers 0 1 2 4 5",
    )

    args = parser.parse_args()

    attck_type = str(args.attck_type)
    epsilon = float(args.desired_norm_l_inf)
    lr = float(args.learningRate)

    num_steps = args.num_steps if args.num_steps is not None else args.numSteps
    if num_steps is None:
        num_steps = 1000
    num_steps = int(num_steps)

    attackSample = str(args.attackSample)

    AttackStartLayer = int(args.AttackStartLayer)
    numLayerstAtAtime = int(args.numLayerstAtAtime)

    towardsNull = float(args.towardsNull)
    BalAlpha = float(args.BalAlpha)
    whichMLP = str(args.whichMLP)
    whichMLPVis = str(args.whichMLPVis)

    chosenLanLayers = args.chosenLanLayers
    chosenVisLayers = args.chosenVisLayers

    MODEL_PATH = "/home/luser/LLaVA/llava-1.5-7b-hf"
    IMAGE_PATH = f"llava_attack/dataSamplesForQuant/{attackSample}.JPEG"

    QUESTION = "What is shown in this image?"
    MAX_NEW_TOKENS = 128

    os.makedirs("llava_attack/outputsStorage", exist_ok=True)
    os.makedirs(f"llava_attack/outputsStorage/advOutputsN/{attackSample}", exist_ok=True)
    os.makedirs(f"llava_attack/outputsStorage/convergence/{attackSample}", exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"device={device}, dtype={dtype}")

    print("Loading tokenizer + image_processor...")
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    image_processor = CLIPImageProcessor.from_pretrained(MODEL_PATH)

    print("Loading model...")
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if device.type == "cuda" else None,
    )

    model.eval()
    model.config.use_cache = False

    print("\n[INFO] Flexible LLaVA bottom-subspace attack")
    print(f"[INFO] chosenLanLayers={chosenLanLayers}")
    print(f"[INFO] chosenVisLayers={chosenVisLayers}")
    print(f"[INFO] whichMLP={whichMLP}")
    print(f"[INFO] whichMLPVis={whichMLPVis}")
    print(f"[INFO] towardsNull={towardsNull}\n")

    pil = Image.open(IMAGE_PATH).convert("RGB")
    x_orig01 = pil_to_tensor01(pil).to(device)

    template_inputs = build_template_inputs(tokenizer, QUESTION, device)

    pv_clean = llava_preprocess_differentiable(x_orig01, image_processor)

    print("\n=== CLEAN OUTPUT ===")
    clean_text = run_generation_with_pixel_values(
        model,
        tokenizer,
        template_inputs,
        pv_clean,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    print(clean_text)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    conv_path = (
        f"llava_attack/outputsStorage/convergence/{attackSample}/"
        f"llava_ORIG_attack_{attck_type}_lr_{lr}_eps_{epsilon}_"
        f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
        f"num_steps_{num_steps}_towardsNull_{towardsNull}_"
        f"{whichMLP}_{whichMLPVis}_allLayers_{BalAlpha}.npy"
    )

    x_adv01, best_pert = adam_attack_original_space(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        template_inputs=template_inputs,
        x_orig01=x_orig01,
        attck_type=attck_type,
        num_steps=num_steps,
        lr=lr,
        epsilon=epsilon,
        device=device,
        save_conv_path=conv_path,
        AttackStartLayer=AttackStartLayer,
        numLayerstAtAtime=numLayerstAtAtime,
        towardsNull=towardsNull,
        BalAlpha = BalAlpha,
        whichMLP=whichMLP,
        whichMLPVis=whichMLPVis,
        chosenLanLayers=chosenLanLayers,
        chosenVisLayers=chosenVisLayers,
    )

    adv_img_path = (
        f"llava_attack/outputsStorage/advOutputsN/{attackSample}/"
        f"adv_ORIG_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
        f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
        f"num_steps_{num_steps}_towardsNull_{towardsNull}_"
        f"{whichMLP}_{whichMLPVis}_allLayers_{BalAlpha}.png"
    )

    adv_noise_path = adv_img_path.replace(".png", ".pt")

    tensor01_to_pil(x_adv01).save(adv_img_path)
    torch.save(best_pert.detach().cpu(), adv_noise_path)

    print(f"\nSaved ORIGINAL-resolution adversarial image to: {adv_img_path}")
    print(f"Saved perturbation to: {adv_noise_path}")

    pv_adv = llava_preprocess_differentiable(x_adv01, image_processor)

    print("\n=== ADVERSARIAL OUTPUT ===")
    adv_text = run_generation_with_pixel_values(
        model,
        tokenizer,
        template_inputs,
        pv_adv,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    print(adv_text)

    cleanOutTxt = f"llava_attack/outputsStorage/advOutputsN/{attackSample}/cleanOutput.txt"
    with open(cleanOutTxt, "w") as f:
        f.write(clean_text + "\n\n")

    advOutTxt = (
        f"llava_attack/outputsStorage/advOutputsN/{attackSample}/"
        f"advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
        f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
        f"num_steps_{num_steps}_towardsNull_{towardsNull}_"
        f"{whichMLP}_{whichMLPVis}_allLayers_{BalAlpha}.txt"
    )

    with open(advOutTxt, "w") as f:
        f.write(adv_text + "\n")

    print(f"\nSaved outputs to: {advOutTxt}")
    print(f"Saved convergence to: {conv_path}")


if __name__ == "__main__":
    main()