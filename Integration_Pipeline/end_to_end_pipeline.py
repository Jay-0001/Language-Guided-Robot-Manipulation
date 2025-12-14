import pybullet as p
import pybullet_data
import numpy as np
import time
import random

from PIL import Image

import torch
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
import groundingdino.datasets.transforms as T

from language_parser import language_module
from grounding_stage import load_model,get_boxes,visualize
from camera_utils import get_view_matrix, get_projection_matrix
from manip import pick_and_place, setup_environment, spawn_scattered_static_spheres 

# ================================================================
# Scene generation
# ================================================================
WORKSPACE = {
    "x": (-0.25, 0.25),
    "y": (-0.25, 0.25),
    "z": (0.02, 0.05)
}

SPHERE_COLORS = {
    "red": (1, 0, 0, 1),
    "green": (0, 1, 0, 1),
    "blue": (0, 0, 1, 1),
    "yellow": (1, 1, 0, 1),
    "black": (0.1, 0.1, 0.1, 1),
    "pink": (1, 0.4, 0.7, 1)
}

def random_pos():
    return [
        random.uniform(*WORKSPACE["x"]),
        random.uniform(*WORKSPACE["y"]),
        random.uniform(*WORKSPACE["z"]),
    ]

def spawn_6_spheres():
    for rgba in SPHERE_COLORS.values():
        pos = random_pos()
        col_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=rgba)
        p.createMultiBody(
            baseMass=0.05,
            baseVisualShapeIndex=col_id,
            basePosition=pos
        )

# ================================================================
# Camera capture
# ================================================================

def capture_image(camera_pos, target, fov=70, width=640, height=480):
    view = p.computeViewMatrix(camera_pos, target, [0, 0, 1])
    proj = p.computeProjectionMatrixFOV(
        fov=fov,
        aspect=width / height,
        nearVal=0.01,
        farVal=2.0
    )

    img = p.getCameraImage(
        width,
        height,
        view,
        proj,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )

    rgb = np.reshape(img[2], (height, width, 4))[:, :, :3]
    depth = np.reshape(img[3], (height, width))
    rgb_pil = Image.fromarray(rgb)

    return rgb_pil, depth, view, proj


# ================================================================
# GroundingDINO
# ================================================================

def load_image_from_pil(image_pil):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])
    tensor, _ = transform(image_pil, None)
    return tensor

def run_grounding(model, rgb_pil, prompt):
    device = next(model.parameters()).device
    image_tensor = load_image_from_pil(rgb_pil).to(device)

    boxes, phrases, scores = get_boxes(
        model,
        image_tensor,
        prompt,
        box_threshold=0.5,
        text_threshold=0.4
    )

    return boxes, phrases, scores


def pick_bbox(boxes, scores):
    if len(boxes) == 0:
        return None
    best_idx = torch.argmax(scores).item()
    return boxes[best_idx]


# ================================================================
# Pixel → 3D (uses inference camera)
# ================================================================

def pixel_to_world(px, py, depth, view, proj, width=640, height=480):
    # PyBullet depth is non-linear → convert to real-world Z
    near, far = 0.01, 2.0
    z_buffer = depth[py, px]
    real_z = near * far / (far - (far - near) * z_buffer)

    # Normalized Device Coordinates
    ndc_x = (px / width - 0.5) * 2.0
    ndc_y = (0.5 - py / height) * 2.0
    ndc_z = (real_z - near) / (far - near) * 2.0 - 1.0  # Convert to OpenGL clip depth

    clip = np.array([ndc_x, ndc_y, ndc_z, 1.0], dtype=np.float32)

    # Matrices from computeViewMatrix and computeProjectionMatrix (row-major)
    view_m = np.array(view).reshape(4, 4).T
    proj_m = np.array(proj).reshape(4, 4).T

    inv_view = np.linalg.inv(view_m)
    inv_proj = np.linalg.inv(proj_m)

    world = inv_view @ (inv_proj @ clip)
    world /= world[3]

    return world[:3].tolist()


# ================================================================
# Different integration idea -- grounding and manipulation chaining
# ================================================================
def project_point(pos, view_matrix, proj_matrix, img_w, img_h):
    # PyBullet gives row-major matrices → convert to column-major for math
    vm = np.array(view_matrix).reshape(4,4).T
    pm = np.array(proj_matrix).reshape(4,4).T

    x, y, z = pos
    vec = np.array([x, y, z, 1.0])

    clip = pm @ (vm @ vec)

    # If clip[3] is 0 or negative, projection is invalid
    if clip[3] == 0:
        return None, None, 99999

    ndc_x = clip[0] / clip[3]
    ndc_y = clip[1] / clip[3]
    depth = clip[2] / clip[3]

    # Reject invalid projections
    if not (-1 <= ndc_x <= 1 and -1 <= ndc_y <= 1):
        return None, None, 99999

    px = int((ndc_x + 1) * 0.5 * img_w)
    py = int((1 - ndc_y) * 0.5 * img_h)

    return px, py, depth




# ================================================================
# Visual Debugging
# ================================================================

def draw_bounding_cube(center, size=0.05, lifetime=0):
    x, y, z = center
    s = size
    pts = [
        [x - s, y - s, z - s],
        [x - s, y - s, z + s],
        [x - s, y + s, z - s],
        [x - s, y + s, z + s],
        [x + s, y - s, z - s],
        [x + s, y - s, z + s],
        [x + s, y + s, z - s],
        [x + s, y + s, z + s],
    ]
    edges = [
        (0, 1), (0, 2), (0, 4),
        (3, 1), (3, 2), (3, 7),
        (5, 1), (5, 4), (5, 7),
        (6, 2), (6, 4), (6, 7)
    ]
    for (i, j) in edges:
        p.addUserDebugLine(pts[i], pts[j], [1, 0, 0], 2.0, lifetime)

def draw_label(center, text, lifetime=0):
    x, y, z = center
    p.addUserDebugText(text, [x, y, z + 0.07], [1, 1, 1], 1.4, lifetime)

def draw_marker(xyz, lifetime=0):
    p.addUserDebugText("●", xyz, [1, 0, 0], 2, lifetime)
    p.addUserDebugLine(xyz, [xyz[0], xyz[1], xyz[2] + 0.1],
                       [1, 0, 0], 1, lifetime)


# ================================================================
# MAIN PIPELINE (FIX C APPLIED)
# ================================================================

def run_pipeline(model):

    # ==========================================================
    # 1. Start & set up simulation
    # ==========================================================
    if p.getConnectionInfo()['isConnected'] == 0:
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")

        # Disable gravity (spheres float)
        p.setGravity(0, 0, 0)

        # Load Panda arm
        panda = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

        # Finger friction (important!)
        p.changeDynamics(panda, 9, lateralFriction=2.0)
        p.changeDynamics(panda, 10, lateralFriction=2.0)

        # Spawn sphere scene from manip.py
        spheres = spawn_scattered_static_spheres(5)

        # Let scene settle
        for _ in range(150):
            p.stepSimulation()
            time.sleep(1/240)

    #--------------------------------------------------------------
    #LAnguage module -- without messing with Camera positions
    #--------------------------------------------------------------
    prompt = language_module()

    # ==========================================================
    # 2. Camera configuration
    # ==========================================================
    cam_pos    = [1.0, 0, 0.7]
    cam_target = [0, 0, 0.05]

    rgb_pil, depth_img, view, proj = capture_image(cam_pos, cam_target)
    rgb_pil.save("debug_rgb.png")

    # Sync GUI camera to inference camera
    dx = cam_pos[0] - cam_target[0]
    dy = cam_pos[1] - cam_target[1]
    dz = cam_pos[2] - cam_target[2]

    dist = np.sqrt(dx*dx + dy*dy + dz*dz)
    yaw = np.degrees(np.arctan2(dy, dx))
    pitch = -np.degrees(np.arctan2(dz, np.sqrt(dx*dx + dy*dy)))

    p.resetDebugVisualizerCamera(
        cameraDistance=dist,
        cameraYaw=yaw,
        cameraPitch=pitch,
        cameraTargetPosition=cam_target
    )

    # Image dimensions (must match capture_image)
    W, H = rgb_pil.size

    # ==========================================================
    # 3. Grounding (2D Bounding Box)
    # ==========================================================
    boxes, phrases, scores = run_grounding(model, rgb_pil, prompt)

    if len(boxes) == 0:
        print("No objects matched the prompt.")
        return

    best_idx = torch.argmax(scores).item()
    best_box = boxes[best_idx]
    best_phrase = phrases[best_idx]

    #COnverting to pixel co-ordinates
    cx = float(best_box[0]) * W
    cy = float(best_box[1]) * H
    bw = float(best_box[2]) * W
    bh = float(best_box[3]) * H

    xmin = int(cx - bw/2)
    xmax = int(cx + bw/2)
    ymin = int(cy - bh/2)
    ymax = int(cy + bh/2)

    visualize(rgb_pil, [best_box], [best_phrase], "ground_rgb.png")

    print(f"Grounded phrase: {best_phrase}")
    print(f"Grounded bbox: {best_box}")
    print("Pixel bbox:", xmin, ymin, xmax, ymax)
    # ==========================================================
    # ⭐ 4. BOUNDING BOX → OBJECT-ID MATCHING (3D→2D PROJECTION)
    # ==========================================================
    candidate_id = None
    best_depth = 1e9

    for sphere_id in spheres:
        pos, _ = p.getBasePositionAndOrientation(sphere_id)
        px, py, depth_val = project_point(pos, view, proj, W, H)

        print(f"Sphere {sphere_id} projected → px={px}, py={py}, depth={depth_val}")
        # Only consider sphere if its projection lies inside the grounded box
        if xmin <= px <= xmax and ymin <= py <= ymax:
            if depth_val < best_depth:
                best_depth = depth_val
                candidate_id = sphere_id

    if candidate_id is None:
        print("No object found inside grounding region.")
        return

    print(f"Matched simulation object ID: {candidate_id}")

    # ==========================================================
    # 5. Manipulation: pick-and-place using object ID
    # ==========================================================
    place_position = [0.6, 0.0, 0.042]  # ground placement point

    pick_and_place(panda, candidate_id, place_position)

    # ==========================================================
    # 6. Keep simulation alive
    # ==========================================================
    while True:
        p.stepSimulation()
        time.sleep(1/240)


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":
    CONFIG = "/home/jay/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    CHECKPOINT = "/home/jay/GroundingDINO/weights/groundingdino_swint_ogc.pth"

    model = load_model(CONFIG, CHECKPOINT)
    xyz = run_pipeline(model)
