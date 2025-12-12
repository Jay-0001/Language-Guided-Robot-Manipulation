import os
import json
import numpy as np
from PIL import Image
import csv

from grounding_stage import run_grounding_on_image,load_model    # your module
from metrics import box_iou, center_error

###############################################
# COLOR → PHRASE MAP
###############################################

SPHERE_PHRASES = [
    "red sphere",
    "green sphere",
    "blue sphere",
    "yellow sphere",
    "black sphere",
    "pink sphere"
]


###############################################
# EVALUATION CORE
###############################################

def evaluate_folder(model, folder_path, sphere_count):
    results_path = os.path.join(folder_path, f"metrics_{sphere_count}.csv")

    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample", "color", "gt_box", "pred_box", "IoU", "center_err"
        ])

        samples = sorted(os.listdir(folder_path))
        samples = [s for s in samples if s.startswith("sample")]

        for sample in samples:
            sample_dir = os.path.join(folder_path, sample)

            rgb = Image.open(os.path.join(sample_dir, "rgb.png"))
            with open(os.path.join(sample_dir, "object_bounding_boxes.json")) as fp:
                gt_boxes = json.load(fp)

            # Evaluate each sphere separately
            for i in range(sphere_count):
                phrase = SPHERE_PHRASES[i]
                gt_box = gt_boxes[i]

                pred_box = run_grounding_on_image(model, rgb, phrase)

                iou = box_iou(gt_box, pred_box)
                ce  = center_error(gt_box, pred_box)

                writer.writerow([
                    sample,
                    phrase,
                    gt_box,
                    pred_box,
                    round(iou, 4),
                    round(ce, 2)
                ])

            print(f"[✓] Evaluated {sample} with {sphere_count} spheres")

    print(f"\nSaved evaluation file → {results_path}")


###############################################
# MASTER EVALUATION LOOP
###############################################

def evaluate_dataset(model, root="dataset"):
    for N in [3,4,5,6]:
        folder = os.path.join(root, f"spheres_{N}")
        if not os.path.exists(folder):
            print(f"Skipping {folder}, not found.")
            continue

        print(f"\n=== Evaluating N = {N} spheres ===")
        evaluate_folder(model, folder, N)


if __name__ == "__main__":
    # ---- UPDATE THESE ----
    CONFIG = "/home/jay/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    CHECKPOINT = "/home/jay/GroundingDINO/weights/groundingdino_swint_ogc.pth"

    model = load_model(CONFIG, CHECKPOINT)
    root="/home/jay/Language Aware Manipulation/dataset"
    evaluate_dataset(model)