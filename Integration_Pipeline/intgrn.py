import pybullet as p
import pybullet_data
import numpy as np
import time
from PIL import Image

import torch
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
import groundingdino.datasets.transforms as T



from grounding_stage import load_model,get_boxes,visualize   # <-- your existing grounding function
from camera_utils import get_view_matrix, get_projection_matrix
import random





#Scene generation
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

# -------------------------------------------------------------------
# 1. PyBullet Camera Capture (RGB + Depth + Matrices)
# -------------------------------------------------------------------

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


# -------------------------------------------------------------------
# 2. Run GroundingDINO on an RGB PIL Image
# -------------------------------------------------------------------
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
    # Convert PIL to tensor (your existing code handles this)
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


# -------------------------------------------------------------------
# 3. Pick the grounding result (simplest: pick first box)
# -------------------------------------------------------------------

def pick_bbox(boxes, scores):
    """Pick the bbox with highest confidence score."""
    if len(boxes) == 0:
        return None
    best_idx = torch.argmax(scores).item()
    return boxes[best_idx]



# -------------------------------------------------------------------
# 4. Pixel → 3D unprojection
# -------------------------------------------------------------------


#inference camera pos vs GUI camera pos mismatch issues
def pixel_to_world(px, py, depth, view, proj, width=640, height=480):
    """
    Convert pixel to world coordinate using depth buffer + camera matrices.
    """

    # Convert depth buffer from [0,1] to depth in world space
    z = depth[py, px]

    # Compute normalized device coordinates
    ndc_x = (px / width - 0.5) * 2.0
    ndc_y = (0.5 - py / height) * 2.0
    ndc_z = z * 2.0 - 1.0

    clip = np.array([ndc_x, ndc_y, ndc_z, 1.0], dtype=np.float32)

    # Convert flat list → 4x4 matrix
    view_m = np.array(view).reshape(4, 4).T
    proj_m = np.array(proj).reshape(4, 4).T

    inv_view = np.linalg.inv(view_m)
    inv_proj = np.linalg.inv(proj_m)

    # Unproject
    world = inv_view @ (inv_proj @ clip)
    world = world / world[3]

    return world[:3].tolist()

# ----------------------------------------------------------
# RAYCAST-BASED PIXEL → 3D
# ----------------------------------------------------------
'''
def pixel_to_ray(px, py, width, height, view, proj):
    """
    Convert pixel (px, py) into a 3D ray using camera matrices.
    Returns (ray_from, ray_to) suitable for p.rayTest.
    """

    # Convert lists → matrices
    view_m = np.array(view).reshape(4, 4).T
    proj_m = np.array(proj).reshape(4, 4).T

    inv_view = np.linalg.inv(view_m)
    inv_proj = np.linalg.inv(proj_m)

    # Convert pixel to Normalized Device Coordinates (NDC)
    ndc_x = (px / width - 0.5) * 2.0
    ndc_y = (0.5 - py / height) * 2.0

    # Create clip-space coordinates for near & far planes
    near_clip = np.array([ndc_x, ndc_y, -1.0, 1.0])
    far_clip  = np.array([ndc_x, ndc_y,  1.0, 1.0])

    # Unproject near/far
    near_world = inv_view @ (inv_proj @ near_clip)
    far_world  = inv_view @ (inv_proj @ far_clip)

    near_world /= near_world[3]
    far_world  /= far_world[3]

    ray_from = near_world[:3]
    ray_to   = far_world[:3]

    return ray_from, ray_to


def raycast_from_pixel(px, py, view, proj, width=640, height=480):
    """
    Raycast into the scene using pixel → (ray_from, ray_to).
    Returns: (hit_position, hit_object_id)
    """

    ray_from, ray_to = pixel_to_ray(px, py, width, height, view, proj)
    hit = p.rayTest(ray_from, ray_to)[0]

    hit_uid = hit[0]
    hit_pos = hit[3]

    if hit_uid < 0:
        return None, None

    return hit_pos, hit_uid
'''
# -------------------------------------------------------------------
# 5. Visualization in PyBullet (Debug Marker)
# -------------------------------------------------------------------

def draw_bounding_cube(center, size=0.05, lifetime=0):
    """
    Draw a wireframe cube centered at the grounded object's 3D coordinate.
    Size is half-extent (0.05 means cube ~10cm wide).
    """

    x, y, z = center
    s = size

    # 8 vertices of the cube
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

    # 12 edges of cube
    edges = [
        (0, 1), (0, 2), (0, 4),
        (3, 1), (3, 2), (3, 7),
        (5, 1), (5, 4), (5, 7),
        (6, 2), (6, 4), (6, 7)
    ]

    for (i, j) in edges:
        p.addUserDebugLine(
            pts[i], pts[j],
            lineColorRGB=[1, 0, 0],
            lineWidth=2.0,
            lifeTime=lifetime
        )

def draw_label(center, text, lifetime=0):
    x, y, z = center
    p.addUserDebugText(
        text,
        [x, y, z + 0.07],     # float above the object
        textColorRGB=[1, 1, 1],
        textSize=1.4,
        lifeTime=lifetime
    )

def draw_marker(xyz):
    p.addUserDebugText(
        text="●",
        textPosition=xyz,
        textColorRGB=[1, 0, 0],
        textSize=2,
        lifeTime=3
    )
    p.addUserDebugLine(
        xyz, [xyz[0], xyz[1], xyz[2] + 0.1],
        [1, 0, 0], 1, 3
    )


# -------------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------------

def run_pipeline(model, prompt="red sphere"):

     # ---- Start physics server ----
    if p.getConnectionInfo()['isConnected'] == 0:
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load plane + robot
        plane = p.loadURDF("plane.urdf")
        panda = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

        # Load your 6 sphere scene
        spawn_6_spheres()

        # Let objects settle
        for _ in range(120):
            p.stepSimulation()
            time.sleep(1.0/240)

    # Camera placement (matches your working settings)
    cam_pos = [1.0, 0, 0.7]
    cam_target = [0, 0, 0.05]

    # Step 1: Capture from PyBullet
    rgb_pil, depth, view, proj = capture_image(cam_pos, cam_target)
    rgb_pil.save("debug_rgb.png")     # optional

    # Step 2: Grounding inference
    boxes, phrases, scores = run_grounding(model, rgb_pil, prompt)
    best_box = pick_bbox(boxes,scores)

    if best_box is None:
        print("No objects matched the prompt.")
        return

    #visualizing the grounding results just in case
    best_idx = torch.argmax(scores).item()
    best_box = boxes[best_idx]
    best_phrase = phrases[best_idx]
    visualize(rgb_pil, [best_box], [best_phrase], "ground_rgb.png")
    
    # Step 3: Convert box → pixel center
    x1, y1, x2, y2 = best_box
    px = int((x1 + x2) / 2 * 640)
    py = int((y1 + y2) / 2 * 480)

    # Step 4: Pixel → 3D world coordinate
    xyz, obj_hit = raycast_from_pixel(px, py, view, proj)
    print("Grounded 3D coordinate:", xyz)
    if xyz is None:
        print("Raycast failed — no hit detected.")
        return

    print("Grounded 3D coordinate:", xyz)

    # Step 5: Visualize in PyBullet
    draw_marker(xyz)
    draw_bounding_cube(xyz, size=0.05)
    draw_label(xyz, prompt)

    #SImulation maneuvering
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1)

    #Keep the simulation active
    while True:
        p.stepSimulation()
        time.sleep(1.0/240)

    return xyz


# -------------------------------------------------------------------
# Usage
# -------------------------------------------------------------------

if __name__ == "__main__":
    # Load your trained GroundingDINO model
    CONFIG = "/home/jay/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    CHECKPOINT = "/home/jay/GroundingDINO/weights/groundingdino_swint_ogc.pth"

    model = load_model(CONFIG, CHECKPOINT)

    # Example prompt
    target = "red sphere"

    # Run integration
    xyz = run_pipeline(model, target)
    print("Final grounded point:", xyz)
