import pybullet as p
import pybullet_data
import random
import time
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from camera_utils import (
    get_view_matrix,
    get_projection_matrix,
    setup_camera,
    save_image
)

from object_factory import (
    make_sphere,
    make_cube,
    make_cylinder,
    sample_color
)

WORKSPACE = {
    "x": (-0.25, 0.25),
    "y": (-0.25, 0.25),
    "z": (0.02, 0.05)
}

SHAPES = ["sphere", "cube", "cylinder"]
OBJ_RANGE = (5, 10)  # <-- your constraint
OUTPUT_ROOT = "dataset/"

def random_pos():
    return [
        random.uniform(*WORKSPACE["x"]),
        random.uniform(*WORKSPACE["y"]),
        random.uniform(*WORKSPACE["z"]),
    ]

def generate_sample():
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    plane = p.loadURDF("plane.urdf")
    panda = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

    num_objects = random.randint(*OBJ_RANGE)
    annotations = []

    for obj_id in range(num_objects):
        shape = random.choice(SHAPES)
        color_name, color_rgba = sample_color()
        pos = random_pos()

        if shape == "sphere":
            make_sphere(pos, color_rgba)
        elif shape == "cube":
            make_cube(pos, color_rgba)
        elif shape == "cylinder":
            make_cylinder(pos, color_rgba)

        annotations.append({
            "id": obj_id,
            "shape": shape,
            "color_name": color_name,
            "color_rgba": color_rgba,
            "position": pos
        })

    # settle physics
    for _ in range(120):
        p.stepSimulation()

    # cameras
    proj = get_projection_matrix(fov=90, aspect=4/3)

    cams = {
        "front": ([1.0, 0, 0.7], [0, 0, 0.05]),
        "side": ([0, 1.0, 0.7], [0, 0, 0.05]),
        "overhead": ([0, -0.001, 1.0], [0, 0, 0.1]),
        "iso": ([0.9, 0.9, 0.9], [0, 0, 0]),
    }

    sample_id = f"sample_{random.randint(10000, 99999)}"
    sample_dir = os.path.join(OUTPUT_ROOT, sample_id)
    os.makedirs(sample_dir, exist_ok=True)

    for cam_name, (pos, tgt) in cams.items():
        view = get_view_matrix(pos, tgt)
        rgb = setup_camera(view, proj)
        save_image(rgb, f"{sample_dir}/{cam_name}.png")

    with open(f"{sample_dir}/annotations.json", "w") as f:
        json.dump({"objects": annotations}, f, indent=4)

    p.disconnect()
    return sample_dir


if __name__ == "__main__":
    out = generate_sample()
    print("Generated:", out)
