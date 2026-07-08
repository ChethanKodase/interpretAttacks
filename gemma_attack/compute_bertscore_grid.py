'''

Hyperparameter estimation : layer selection:
Used to determine which layers to use for for the spectral loss formulation


conda activate gemma3
cd interpretAttacks
python gemma_attack/compute_bertscore_grid.py


'''


import os
import itertools
import numpy as np
from bert_score import score

BASE_DIR = "gemma_attack/outputsStorageImagenet/advOutputs"

attack_samples = range(1, 6)
lan_layers = range(0, 7)
vis_layers = range(0, 27)

which_mlps = ["gate_proj", "up_proj", "down_proj"]
which_vis_mlps = ["fc1", "fc2", "out_proj"]
towards_nulls = [0.1, 0.5]

attck_type = "saa_loopR"
lr = 0.001
epsilon = 0.005
num_steps = 100
AttackStartLayer = 0
numLayerstAtAtime = 2

output_file = "gemma_attack/outputsStorageImagenet/bert_score_all_configurationsR.txt"


def adv_path(sample, towards_null, which_mlp, which_mlpvis, lan_layer, vis_layer):
    return (
        f"{BASE_DIR}/{sample}/"
        f"advOutput_attackType_{attck_type}_lr_{lr}_eps_{epsilon}_"
        f"AttackStartLayer_{AttackStartLayer}_numLayerstAtAtime_{numLayerstAtAtime}_"
        f"num_steps_{num_steps}_towardsNull_{towards_null}_"
        f"lanMLP_{which_mlp}_visMLP_{which_mlpvis}_"
        f"lanLayers_[{lan_layer}]_visLayers_[{vis_layer}].txt"
    )


def clean_path(sample):
    return f"{BASE_DIR}/{sample}/cleanOutput.txt"


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def main():
    all_results = []

    with open(output_file, "w", encoding="utf-8") as out:
        out.write("BERTScore tabulation for Gemma-3 adversarial outputs\n")
        out.write("=" * 90 + "\n\n")

        for which_mlp, which_mlpvis, towards_null, lan_layer, vis_layer in itertools.product(
            which_mlps,
            which_vis_mlps,
            towards_nulls,
            lan_layers,
            vis_layers,
        ):
            sample_scores = []
            missing = []

            for sample in attack_samples:
                adv_txt_path = adv_path(
                    sample,
                    towards_null,
                    which_mlp,
                    which_mlpvis,
                    lan_layer,
                    vis_layer,
                )
                clean_txt_path = clean_path(sample)

                if not os.path.exists(adv_txt_path):
                    missing.append(sample)
                    continue

                if not os.path.exists(clean_txt_path):
                    missing.append(sample)
                    continue

                adv_output = [read_text(adv_txt_path)]
                clean_output = [read_text(clean_txt_path)]

                P, R, F1 = score(
                    adv_output,
                    clean_output,
                    lang="en",
                    model_type="roberta-large",
                    rescale_with_baseline=True,
                    verbose=False,
                )

                sample_scores.append(
                    {
                        "sample": sample,
                        "P": float(P[0]),
                        "R": float(R[0]),
                        "F1": float(F1[0]),
                    }
                )

            if len(sample_scores) == 0:
                continue

            mean_f1 = float(np.mean([x["F1"] for x in sample_scores]))
            mean_p = float(np.mean([x["P"] for x in sample_scores]))
            mean_r = float(np.mean([x["R"] for x in sample_scores]))

            config_result = {
                "whichMLP": which_mlp,
                "whichMLPvis": which_mlpvis,
                "towardsNull": towards_null,
                "chosenLanLayer": lan_layer,
                "chosenVisLayer": vis_layer,
                "mean_F1": mean_f1,
                "mean_P": mean_p,
                "mean_R": mean_r,
                "scores": sample_scores,
                "missing": missing,
            }
            all_results.append(config_result)

            out.write(
                f"CONFIG: whichMLP={which_mlp}, "
                f"whichMLPvis={which_mlpvis}, "
                f"towardsNull={towards_null}, "
                f"chosenLanLayers=[{lan_layer}], "
                f"chosenVisLayers=[{vis_layer}]\n"
            )
            out.write("-" * 90 + "\n")
            out.write(f"{'Sample':<10}{'Precision':<15}{'Recall':<15}{'F1':<15}\n")

            for s in sample_scores:
                out.write(
                    f"{s['sample']:<10}"
                    f"{s['P']:<15.6f}"
                    f"{s['R']:<15.6f}"
                    f"{s['F1']:<15.6f}\n"
                )

            out.write("-" * 90 + "\n")
            out.write(f"Mean Precision: {mean_p:.6f}\n")
            out.write(f"Mean Recall:    {mean_r:.6f}\n")
            out.write(f"Mean F1:        {mean_f1:.6f}\n")

            if missing:
                out.write(f"Missing samples: {missing}\n")

            out.write("\n\n")

        out.write("\n")
        out.write("=" * 90 + "\n")
        out.write("RANKING BY LOWEST MEAN F1\n")
        out.write("=" * 90 + "\n\n")

        all_results_sorted = sorted(all_results, key=lambda x: x["mean_F1"])

        out.write(
            f"{'Rank':<6}{'Mean F1':<12}{'Mean P':<12}{'Mean R':<12}"
            f"{'whichMLP':<12}{'whichMLPvis':<12}"
            f"{'towardsNull':<14}{'LanLayer':<10}{'VisLayer':<10}\n"
        )

        for rank, r in enumerate(all_results_sorted, start=1):
            out.write(
                f"{rank:<6}"
                f"{r['mean_F1']:<12.6f}"
                f"{r['mean_P']:<12.6f}"
                f"{r['mean_R']:<12.6f}"
                f"{r['whichMLP']:<12}"
                f"{r['whichMLPvis']:<12}"
                f"{r['towardsNull']:<14}"
                f"{r['chosenLanLayer']:<10}"
                f"{r['chosenVisLayer']:<10}\n"
            )

    print(f"Saved BERTScore tabulation to: {output_file}")


if __name__ == "__main__":
    main()