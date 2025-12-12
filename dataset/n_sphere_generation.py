import pybullet as p
import pybullet_data
import numpy as np
import json
import os
import time
from PIL import Image

###############################################
#  CAMERA + PROJECTION UTILS (unchanged logic)
###############################################

def capture_image(camera_pos, camera_target, fov=70, width=640, height=480):
    view = p.computeViewMatrix(camera_pos, camera_target, [0, 0, 1])
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


def project_point(pos, view_matrix, proj_matrix, img_w, img_h):
    vm = np.array(view_matrix).reshape(4,4).T
    pm = np.array(proj_matrix).reshape(4,4).T

    x, y, z = pos
    vec = np.array([x, y, z, 1.0])

    clip = pm @ (vm @ vec)

    if clip[3] == 0:
        return None, None, 99999

    ndc_x = clip[0] / clip[3]
    ndc_y = clip[1] / clip[3]
    ndc_z = clip[2] / clip[3]

    if not (-1 <= ndc_x <= 1 and -1 <= ndc_y <= 1):
        return None, None, 99999

    px = int((ndc_x + 1) * 0.5 * img_w)
    py = int((1 - ndc_y) * 0.5 * img_h)

    return px, py, ndc_z


###############################################
#  MODULAR: GROUND-TRUTH BOUNDING BOX BUILDER
###############################################

def compute_bounding_boxes(sphere_positions, radius, view, proj, W, H):
    boxes = []

    for pos in sphere_positions:
        cx, cy, depth = project_point(pos, view, proj, W, H)

        if cx is None:
            boxes.append(None)
            continue

        # sphere image radius approx = (focal * r) / depth
        # focal length derivation from projection matrix
        pm = np.array(proj).reshape(4,4).T
        f = pm[0,0]  # projection scale

        if depth == 0:
            boxes.append(None)
            continue

        r_img = abs(f * radius / depth)

        w = int(2 * r_img)
        h = int(2 * r_img)

        boxes.append([int(cx), int(cy), int(w), int(h)])

    return boxes


###############################################
#  SCENE + COLLISION UTILITIES
###############################################

def load_scene():
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    plane = p.loadURDF("plane.urdf")
    panda = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
    return panda


def is_valid_position(candidate, existing_positions, radius):
    """Validate sphere-sphere collision."""
    for p0 in existing_positions:
        if np.linalg.norm(np.array(candidate)-np.array(p0)) < (2*radius + 0.01):
            return False
    return True


def collides_with_robot(panda_id, sphere_id):
    pts = p.getClosestPoints(panda_id, sphere_id, distance=0.005)
    return len(pts) > 0


def spawn_sphere(color, pos, radius=0.025):
    visual = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=color
    )
    collision = p.createCollisionShape(
        p.GEOM_SPHERE,
        radius=radius
    )
    sphere_id = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=pos
    )
    return sphere_id


def sample_valid_sphere_position(existing, radius):
    """Repeated random sampling until a valid non-colliding position is found."""
    for _ in range(200):
        x = np.random.uniform(0.2, 0.6)
        y = np.random.uniform(-0.3, 0.3)
        z = 0.05  # on plane

        pos = [x, y, z]
        if is_valid_position(pos, existing, radius):
            return pos

    return None  # rare but handle gracefully


###############################################
#  DATASET GENERATOR
###############################################

def generate_dataset(N_spheres, samples=50, W=640, H=480):
    folder = f"dataset/spheres_{N_spheres}"
    os.makedirs(folder, exist_ok=True)

    RADIUS = 0.025
    COLORS = [
        [1, 0, 0, 1],   # red
        [0, 1, 0, 1],   # green
        [0, 0, 1, 1],   # blue
        [1, 1, 0, 1],   # yellow
        [0, 0, 0, 1],   # black
        [1, 0, 1, 1],   # pink
    ]

    for idx in range(samples):
        panda_id = load_scene()

        sphere_positions = []
        sphere_ids = []

        for k in range(N_spheres):
            pos = sample_valid_sphere_position(sphere_positions, RADIUS)
            if pos is None:
                print("Warning: Failed to place sphere safely.")
                continue

            sphere_id = spawn_sphere(COLORS[k % len(COLORS)], pos, RADIUS)
            sphere_positions.append(pos)
            sphere_ids.append(sphere_id)

            p.stepSimulation()

            if collides_with_robot(panda_id, sphere_id):
                print("Robot collision detected, resampling.")
                p.removeBody(sphere_id)
                sphere_positions.pop()
                continue

        # CAMERA PARAMS
        cam_pos = [1.0, 0, 0.7]
        cam_tgt = [0, 0, 0.05]

        rgb_pil, depth, view, proj = capture_image(cam_pos, cam_tgt, width=W, height=H)

        # COMPUTE GT BOXES
        gt_boxes = compute_bounding_boxes(sphere_positions, RADIUS, view, proj, W, H)

        # SAVE
        out = os.path.join(folder, f"sample_{idx:04d}")
        os.makedirs(out, exist_ok=True)

        rgb_pil.save(os.path.join(out, "rgb.png"))
        np.save(os.path.join(out, "depth.npy"), depth)

        json.dump(sphere_positions, open(os.path.join(out, "object_positions.json"), "w"))
        json.dump(gt_boxes, open(os.path.join(out, " object_bounding_boxes.json"),"w"))
        json.dump({"view":view, "proj":proj}, open(os.path.join(out, "camera_matrices.json"),"w"))

        print(f"[✓] Sample {idx} complete ({N_spheres} spheres).")

    print(f"\nFinished dataset for N={N_spheres} spheres → {folder}")


###############################################
#  MAIN EXECUTION
###############################################
if __name__ == "__main__":
    p.connect(p.GUI)

    for N in [3, 4, 5, 6]:
        generate_dataset(N_spheres=N, samples=30)

    p.disconnect()
