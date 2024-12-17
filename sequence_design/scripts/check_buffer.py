import argparse
import os
import pickle

from utils import import_protein_env

FULL_LIST_RESULTS = {
    "m1f1": {
        "SR-s42": "ckpts_iclr2025/ready-SR-1seq-128rs-s42",
        "SR-s45": "ckpts_iclr2025/ready-SR-1seq-128rs-s45",
        "SR-s49": "ckpts_iclr2025/ready-SR-1seq-128rs-s49",
        "EI-s42": "ckpts_iclr2025/ready-EI-1seq-128rs-s42",
        "EI-s45": "ckpts_iclr2025/ready-EI-1seq-128rs-s45",
        "EI-s49": "ckpts_iclr2025/ready-EI-1seq-128rs-s49",
        "PI-s42": "ckpts_iclr2025/ready-PI-1seq-128rs-s42",
        "PI-s45": "ckpts_iclr2025/ready-PI-1seq-128rs-s45",
        "PI-s49": "ckpts_iclr2025/ready-PI-1seq-128rs-s49",
        "UCB-s42": "ckpts_iclr2025/ready-UCB-1seq-128rs-s42",
        "UCB-s45": "ckpts_iclr2025/ready-UCB-1seq-128rs-s45",
        "UCB-s49": "ckpts_iclr2025/ready-UCB-1seq-128rs-s49",
        "KG-s42": "ckpts_iclr2025/ready-KG-1seq-128rs-s42",
        "KG-s45": "ckpts_iclr2025/ready-KG-1seq-128rs-s45",
        "KG-s49": "ckpts_iclr2025/ready-KG-1seq-128rs-s49",
        "Ours-s42": "ckpts_iclr2025/ready-HES-11-1seq-128rs-s42",
        "Ours-s45": "ckpts_iclr2025/ready-HES-11-1seq-128rs-s45",
        "Ours-s49": "ckpts_iclr2025/ready-HES-11-1seq-128rs-s49",
    },
    "m1f2": {
        "SR-s42": "ckpts/m1f2-SR-1seq-64rs-s42",
        "SR-s45": "ckpts/m1f2-SR-1seq-64rs-s45",
        "SR-s49": "ckpts/m1f2-SR-1seq-64rs-s49",
        "EI-s42": "ckpts/m1f2-EI-1seq-64rs-s42",
        "EI-s45": "ckpts/m1f2-EI-1seq-64rs-s45",
        "EI-s49": "ckpts/m1f2-EI-1seq-64rs-s49",
        "PI-s42": "ckpts/m1f2-PI-1seq-64rs-s42",
        "PI-s45": "ckpts/m1f2-PI-1seq-64rs-s45",
        "PI-s49": "ckpts/m1f2-PI-1seq-64rs-s49",
        "UCB-s42": "ckpts/m1f2-UCB-1seq-64rs-s42",
        "UCB-s45": "ckpts/m1f2-UCB-1seq-64rs-s45",
        "UCB-s49": "ckpts/m1f2-UCB-1seq-64rs-s49",
        "KG-s42": "ckpts/m1f2-KG-1seq-64rs-s42",
        "KG-s45": "ckpts/m1f2-KG-1seq-64rs-s45",
        "KG-s49": "ckpts/m1f2-KG-1seq-64rs-s49",
        "Ours-s42": "ckpts/m1f2-HES-11-1seq-64rs-s42",
        "Ours-s45": "ckpts/m1f2-HES-11-1seq-64rs-s45",
        "Ours-s49": "ckpts/m1f2-HES-11-1seq-64rs-s49",
    },
    "m2f1": {
        "SR-s42": "ckpts/m2f1-SR-1seq-64rs-s42",
        "SR-s45": "ckpts/m2f1-SR-1seq-64rs-s45",
        "SR-s49": "ckpts/m2f1-SR-1seq-64rs-s49",
        "EI-s42": "ckpts/m2f1-EI-1seq-64rs-s42",
        "EI-s45": "ckpts/m2f1-EI-1seq-64rs-s45",
        "EI-s49": "ckpts/m2f1-EI-1seq-64rs-s49",
        "PI-s42": "ckpts/m2f1-PI-1seq-64rs-s42",
        "PI-s45": "ckpts/m2f1-PI-1seq-64rs-s45",
        "PI-s49": "ckpts/m2f1-PI-1seq-64rs-s49",
        "UCB-s42": "ckpts/m2f1-UCB-1seq-64rs-s42",
        "UCB-s45": "ckpts/m2f1-UCB-1seq-64rs-s45",
        "UCB-s49": "ckpts/m2f1-UCB-1seq-64rs-s49",
        "KG-s42": "ckpts/m2f1-KG-1seq-64rs-s42",
        "KG-s45": "ckpts/m2f1-KG-1seq-64rs-s45",
        "KG-s49": "ckpts/m2f1-KG-1seq-64rs-s49",
        "Ours-s42": "ckpts/m2f1-HES-11-1seq-64rs-s42",
        "Ours-s45": "ckpts/m2f1-HES-11-1seq-64rs-s45",
        "Ours-s49": "ckpts/m2f1-HES-11-1seq-64rs-s49",
    },
    "m2f2": {
        "SR-s42": "ckpts/m2f2-SR-1seq-64rs-s42",
        "SR-s45": "ckpts/m2f2-SR-1seq-64rs-s45",
        "SR-s49": "ckpts/m2f2-SR-1seq-64rs-s49",
        "EI-s42": "ckpts/m2f2-EI-1seq-64rs-s42",
        "EI-s45": "ckpts/m2f2-EI-1seq-64rs-s45",
        "EI-s49": "ckpts/m2f2-EI-1seq-64rs-s49",
        "PI-s42": "ckpts/m2f2-PI-1seq-64rs-s42",
        "PI-s45": "ckpts/m2f2-PI-1seq-64rs-s45",
        "PI-s49": "ckpts/m2f2-PI-1seq-64rs-s49",
        "UCB-s42": "ckpts/m2f2-UCB-1seq-64rs-s42",
        "UCB-s45": "ckpts/m2f2-UCB-1seq-64rs-s45",
        "UCB-s49": "ckpts/m2f2-UCB-1seq-64rs-s49",
        "KG-s42": "ckpts/m2f2-KG-1seq-64rs-s42",
        "KG-s45": "ckpts/m2f2-KG-1seq-64rs-s45",
        "KG-s49": "ckpts/m2f2-KG-1seq-64rs-s49",
        "Ours-s42": "ckpts/m2f2-HES-11-1seq-64rs-s42",
        "Ours-s45": "ckpts/m2f2-HES-11-1seq-64rs-s45",
        "Ours-s49": "ckpts/m2f2-HES-11-1seq-64rs-s49",
    },
    "m3f1": {
        "SR-s42": "ckpts/m3f1-SR-1seq-64rs-s42",
        "SR-s45": "ckpts/m3f1-SR-1seq-64rs-s45",
        "SR-s49": "ckpts/m3f1-SR-1seq-64rs-s49",
        "EI-s42": "ckpts/m3f1-EI-1seq-64rs-s42",
        "EI-s45": "ckpts/m3f1-EI-1seq-64rs-s45",
        "EI-s49": "ckpts/m3f1-EI-1seq-64rs-s49",
        "PI-s42": "ckpts/m3f1-PI-1seq-64rs-s42",
        "PI-s45": "ckpts/m3f1-PI-1seq-64rs-s45",
        "PI-s49": "ckpts/m3f1-PI-1seq-64rs-s49",
        "UCB-s42": "ckpts/m3f1-UCB-1seq-64rs-s42",
        "UCB-s45": "ckpts/m3f1-UCB-1seq-64rs-s45",
        "UCB-s49": "ckpts/m3f1-UCB-1seq-64rs-s49",
        "KG-s42": "ckpts/m3f1-KG-1seq-64rs-s42",
        "KG-s45": "ckpts/m3f1-KG-1seq-64rs-s45",
        "KG-s49": "ckpts/m3f1-KG-1seq-64rs-s49",
        "Ours-s42": "ckpts/m3f1-HES-11-1seq-64rs-s42",
        "Ours-s45": "ckpts/m3f1-HES-11-1seq-64rs-s45",
        "Ours-s49": "ckpts/m3f1-HES-11-1seq-64rs-s49",
    },
    "m3f2": {
        "SR-s42": "ckpts/m3f2-SR-1seq-64rs-s42",
        "SR-s45": "ckpts/m3f2-SR-1seq-64rs-s45",
        "SR-s49": "ckpts/m3f2-SR-1seq-64rs-s49",
        "EI-s42": "ckpts/m3f2-EI-1seq-64rs-s42",
        "EI-s45": "ckpts/m3f2-EI-1seq-64rs-s45",
        "EI-s49": "ckpts/m3f2-EI-1seq-64rs-s49",
        "PI-s42": "ckpts/m3f2-PI-1seq-64rs-s42",
        "PI-s45": "ckpts/m3f2-PI-1seq-64rs-s45",
        "PI-s49": "ckpts/m3f2-PI-1seq-64rs-s49",
        "UCB-s42": "ckpts/m3f2-UCB-1seq-64rs-s42",
        "UCB-s45": "ckpts/m3f2-UCB-1seq-64rs-s45",
        "UCB-s49": "ckpts/m3f2-UCB-1seq-64rs-s49",
        "KG-s42": "ckpts/m3f2-KG-1seq-64rs-s42",
        "KG-s45": "ckpts/m3f2-KG-1seq-64rs-s45",
        "KG-s49": "ckpts/m3f2-KG-1seq-64rs-s49",
        "Ours-s42": "ckpts/m3f2-HES-11-1seq-64rs-s42",
        "Ours-s45": "ckpts/m3f2-HES-11-1seq-64rs-s45",
        "Ours-s49": "ckpts/m3f2-HES-11-1seq-64rs-s49",
    },
}

LIST_ALGOS = [
    "SR",
    "EI",
    "PI",
    "UCB",
    "KG",
    "Ours",
]

SEEDS = [42, 45, 49]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutant_ver", type=str, default="v1")
    parser.add_argument("--fn_ver", type=str, default="v1")
    args = parser.parse_args()

    _, INIT_SEQ, _, _, _ = import_protein_env(args.mutant_ver)
    result_key = f"m{args.mutant_ver[1:]}f{args.fn_ver[1:]}"
    LIST_RESULTS = FULL_LIST_RESULTS[result_key]

    for algo in LIST_ALGOS:
        for seed in SEEDS:
            folder = LIST_RESULTS[algo + "-s" + str(seed)]
            try:
                buffer = pickle.load(open(os.path.join(folder, "buffer.pkl"), "rb"))
                print(f"{algo}-s{seed}")
                new_buffer = {
                    "x": buffer["x"],
                    "y": buffer["y"],
                }

                # Save the new buffer
                with open(os.path.join(folder, "lite_buffer.pkl"), "wb") as f:
                    pickle.dump(new_buffer, f)

            except:
                pass
