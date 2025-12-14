import pybullet as p
import pybullet_data
import numpy as np
import time
import torch
from collections import Counter, defaultdict

# Real pipeline components
from intgrn_two_links import (
    capture_image,
    run_grounding,
    project_point,
)
from grounding_stage import load_model


# ============================================================
# Local eval sphere spawner — deterministic color order
# ============================================================
def spawn_eval_spheres():
    """
    Spawns FOUR spheres and returns:
        sphere_ids: list of PyBullet IDs
        sphere_color_map: {id: "color"}
    """
    colors = [
        ("red",    (1, 0, 0, 1)),
        ("green",  (0, 1, 0, 1)),
        ("blue",   (0, 0, 1, 1)),
        ("yellow", (1, 1, 0, 1)),
    ]

    sphere_ids = []
    sphere_color_map = {}

    radius = 0.03
    x_range = (0.45, 0.65)
    y_range = (-0.15, 0.15)
    z_range = (0.12, 0.20)

    for (name, rgba) in colors:
        x = np.random.uniform(*x_range)
        y = np.random.uniform(*y_range)
        z = np.random.uniform(*z_range)

        visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba)
        collision = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)

        uid = p.createMultiBody(
            baseMass=0,
            basePosition=[x, y, z],
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual
        )

        sphere_ids.append(uid)
        sphere_color_map[uid] = name

    return sphere_ids, sphere_color_map


# ============================================================
# Single Trial Evaluation
# ============================================================
def eval_trial(model, prompt):
    """
    Evaluates ONE GUI-based trial for a given color prompt.
    Returns:
        success (bool),
        reason (FAIL TYPE),
        true_color,
        predicted_color
    """

    # GUI mode required for valid image rendering
    p.connect(p.GUI)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.loadURDF("plane.urdf")
    p.setGravity(0, 0, 0)

    panda = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

    sphere_ids, sphere_color_map = spawn_eval_spheres()

    for _ in range(80):
        p.stepSimulation()
        time.sleep(1/240)

    cam_pos = [1.0, 0, 0.7]
    cam_target = [0, 0, 0.05]
    rgb_pil, depth, view, proj = capture_image(cam_pos, cam_target)
    W, H = rgb_pil.size

    # -------------
    # Grounding
    # -------------
    boxes, phrases, scores = run_grounding(model, rgb_pil, prompt)
    if len(boxes) == 0:
        p.disconnect()
        return False, "GROUNDING_FAIL", prompt.split()[0], None

    best_idx = torch.argmax(scores).item()
    box = boxes[best_idx]

    cx, cy = float(box[0]) * W, float(box[1]) * H
    bw, bh = float(box[2]) * W, float(box[3]) * H

    xmin, xmax = int(cx - bw/2), int(cx + bw/2)
    ymin, ymax = int(cy - bh/2), int(cy + bh/2)

    # -------------------
    # True ID
    # -------------------
    prompt_color = prompt.split()[0].lower()
    true_id = next(sid for sid, col in sphere_color_map.items() if col == prompt_color)

    # -------------------
    # Projection + Matching
    # -------------------
    candidate = None
    best_depth = 1e9

    for sid in sphere_ids:
        pos, _ = p.getBasePositionAndOrientation(sid)
        px, py, depth_val = project_point(pos, view, proj, W, H)
        if px is None:
            continue
        if xmin <= px <= xmax and ymin <= py <= ymax:
            if depth_val < best_depth:
                best_depth = depth_val
                candidate = sid

    if candidate is None:
        p.disconnect()
        return False, "NO_PROJECTION_MATCH", prompt_color, None

    predicted_color = sphere_color_map[candidate]
    success = (candidate == true_id)

    p.disconnect()
    return success, ("SUCCESS" if success else "WRONG_MATCH"), prompt_color, predicted_color


# ============================================================
# MULTI-PROMPT EVALUATION
# ============================================================
def evaluate_all(model, prompts=None, trials_per_color=25):

    if prompts is None:
        prompts = ["red sphere", "green sphere", "blue sphere", "yellow sphere"]

    confusion = defaultdict(lambda: Counter())
    success_by_color = Counter()
    total_by_color = Counter()
    reason_counts = Counter()

    for prompt in prompts:
        p_color = prompt.split()[0]
        print(f"\n===== Evaluating {p_color.upper()} for {trials_per_color} trials =====")

        for t in range(trials_per_color):
            ok, reason, true_color, predicted_color = eval_trial(model, prompt)
            print(f"[{t}] {ok}, Reason={reason}, True={true_color}, Pred={predicted_color}")

            reason_counts[reason] += 1
            total_by_color[true_color] += 1

            if ok:
                success_by_color[true_color] += 1

            if predicted_color is not None:
                confusion[true_color][predicted_color] += 1

    # Final aggregated stats
    print("\n==================== FINAL SUMMARY ====================")

    # Overall success
    total_successes = sum(success_by_color.values())
    total_trials = sum(total_by_color.values())
    overall_acc = total_successes / total_trials if total_trials > 0 else 0.0

    print(f"Overall Accuracy: {overall_acc:.3f}")
    print(f"Failures Breakdown: {dict(reason_counts)}")

    print("\nPer-Color Accuracy:")
    for color in total_by_color:
        acc = success_by_color[color] / total_by_color[color]
        print(f"  {color}: {acc:.3f}")

    print("\nConfusion Matrix (True → Pred Count):")
    for true_c, row in confusion.items():
        print(f"  {true_c}: {dict(row)}")

    print("========================================================")

    return {
        "overall_accuracy": overall_acc,
        "success_by_color": dict(success_by_color),
        "total_by_color": dict(total_by_color),
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "failure_reasons": dict(reason_counts),
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    CONFIG = "/home/jay/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    CHECKPOINT = "/home/jay/GroundingDINO/weights/groundingdino_swint_ogc.pth"

    print("Loading GroundingDINO…")
    model = load_model(CONFIG, CHECKPOINT)

    # Run about 100 trials (25 × 4 colors)
    evaluate_all(model, trials_per_color=25)
