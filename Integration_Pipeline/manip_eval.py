import pybullet as p
import pybullet_data
import numpy as np
import random, csv, torch, time

from intgrn_two_links import capture_image, run_grounding, project_point
from manip import spawn_scattered_static_spheres, pick_and_place

# ================================================================
# Only 4 spheres → these MUST match the real spawn order in manip.py
# ================================================================
COLOR_LIST = ["red", "green", "blue", "yellow"]


def make_color_map(sphere_ids):
    """Map colors → sphere IDs based on deterministic spawn order."""
    mapping = {}
    for i, sid in enumerate(sphere_ids):
        if i < len(COLOR_LIST):
            mapping[COLOR_LIST[i]] = sid
    return mapping


def bbox_to_pixel(box, W, H):
    cx = float(box[0]) * W
    cy = float(box[1]) * H
    bw = float(box[2]) * W
    bh = float(box[3]) * H
    return int(cx - bw/2), int(cx + bw/2), int(cy - bh/2), int(cy + bh/2)


def bbox_valid(xmin, ymin, xmax, ymax, W, H):
    if xmax <= xmin or ymax <= ymin:
        return False
    if xmax < 0 or xmin > W or ymax < 0 or ymin > H:
        return False
    if (xmax - xmin) * (ymax - ymin) < 10:
        return False
    return True


def center_dist(px, py, xmin, ymin, xmax, ymax):
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    return np.sqrt((px - cx)**2 + (py - cy)**2)


# ================================================================
# ONE TRIAL — Ground → Project → Match → Manipulate
# ================================================================
def run_trial(model, trial_id, gt_color):

    print(f"\n=== Trial {trial_id} — Target: {gt_color} sphere ===")
    record = {"trial_id": trial_id, "gt_color": gt_color}

    # ------------------------------
    # RESET SIMULATION
    # ------------------------------
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    p.setGravity(0, 0, 0)

    panda = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
    p.changeDynamics(panda, 9, lateralFriction=2.0)
    p.changeDynamics(panda, 10, lateralFriction=2.0)

    # Spawn exactly 4 spheres
    spheres = spawn_scattered_static_spheres(4)
    cmap = make_color_map(spheres)

    # Safety: color not present
    if gt_color not in cmap:
        record["status"] = "COLOR_NOT_IN_SCENE"
        return record

    true_id = cmap[gt_color]
    record["true_id"] = true_id

    for _ in range(5):
        p.stepSimulation()

    # ------------------------------
    # CAMERA
    # ------------------------------
    cam_pos = [1.0, 0, 0.7]
    cam_target = [0, 0, 0.05]

    rgb_pil, depth_img, view, proj = capture_image(cam_pos, cam_target)
    W, H = rgb_pil.size

    # ------------------------------
    # GROUNDING
    # ------------------------------
    prompt = f"{gt_color} sphere"
    boxes, phrases, scores = run_grounding(model, rgb_pil, prompt)

    if len(boxes) == 0:
        record["status"] = "GROUNDING_FAIL"
        return record

    best_idx = torch.argmax(scores).item()
    best_box = boxes[best_idx]
    xmin, ymin, xmax, ymax = bbox_to_pixel(best_box, W, H)
    record["bbox"] = [xmin, ymin, xmax, ymax]

    if not bbox_valid(xmin, ymin, xmax, ymax, W, H):
        record["status"] = "BBOX_INVALID"
        return record

    # ------------------------------
    # PROJECTION + MATCHING
    # ------------------------------
    candidate = None
    candidate_color = None
    best_depth = 9999

    projected_log = []

    for col, sid in cmap.items():
        pos, _ = p.getBasePositionAndOrientation(sid)
        px, py, depth = project_point(pos, view, proj, W, H)

        projected_log.append({"color": col, "id": sid,
                              "px": px, "py": py, "depth": depth})

        if px is None:
            continue

        inside = (xmin <= px <= xmax and ymin <= py <= ymax)

        if inside and depth < best_depth:
            candidate = sid
            candidate_color = col
            best_depth = depth

    record["projected"] = projected_log
    record["matched_color"] = candidate_color
    record["matched_id"] = candidate

    if candidate is None:
        record["status"] = "NO_PROJECTION_MATCH"
        return record

    # ------------------------------
    # SUCCESS LOGIC: Correct match
    # ------------------------------
    if candidate == true_id:
        record["status"] = "SUCCESS"
    else:
        record["status"] = "WRONG_MATCH"

    # Run pick anyway (not evaluated)
    pick_and_place(panda, candidate, [0.6, 0.0, 0.042])

    return record


# ================================================================
# BULK EVALUATION
# ================================================================
def evaluate(model, N=100, save_csv="diagnostic_results.csv"):
    logs = []
    success_count = 0

    for t in range(N):
        gt_color = random.choice(COLOR_LIST)
        rec = run_trial(model, t, gt_color)
        logs.append(rec)

        if rec["status"] == "SUCCESS":
            success_count += 1

    # Save CSV
    with open(save_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trial_id", "gt_color", "status",
            "matched_color", "bbox",
            "projected", "true_id", "matched_id"
        ])
        for r in logs:
            writer.writerow([
                r["trial_id"],
                r["gt_color"],
                r["status"],
                r.get("matched_color", None),
                r.get("bbox", None),
                r.get("projected", None),
                r.get("true_id", None),
                r.get("matched_id", None)
            ])

    rate = success_count / N * 100
    print(f"\n=== MATCH SUCCESS RATE: {success_count}/{N} = {rate:.2f}% ===")
    print(f"Logs saved to {save_csv}")

    return logs


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":

    print("Loading GroundingDINO…")
    from grounding_stage import load_model

    CONFIG = "/home/jay/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    CHECKPOINT = "/home/jay/GroundingDINO/weights/groundingdino_swint_ogc.pth"

    model = load_model(CONFIG, CHECKPOINT)

    p.connect(p.DIRECT)
    evaluate(model, N=100)
    p.disconnect()
