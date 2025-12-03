import pybullet as p
import pybullet_data
import time
import random
import numpy as np

from camera_utils import (
    get_view_matrix,
    get_projection_matrix,
    setup_camera,
    save_image
)

# ---------------------------------------------
# Scene Setup
# ---------------------------------------------
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

plane = p.loadURDF("plane.urdf")
panda = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

workspace = {
    "x": (-0.3, 0.3),
    "y": (-0.3, 0.3),
    "z": (0.02, 0.05),
}

colors = {
    "red": (1, 0, 0, 1),
    "green": (0, 1, 0, 1),
    "blue": (0, 0, 1, 1),
    "yellow": (1, 1, 0, 1),
    "black": (0.1, 0.1, 0.1, 1),
    "pink": (1, 0.4, 0.7, 1),
}

def spawn_sphere(position, rgba):
    col_id = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.03,
        rgbaColor=rgba
    )
    return p.createMultiBody(
        baseMass=0.05,
        baseVisualShapeIndex=col_id,
        basePosition=position
    )

# Spawn spheres
sphere_positions = []
for rgba in colors.values():
    pos = [
        random.uniform(*workspace["x"]),
        random.uniform(*workspace["y"]),
        random.uniform(*workspace["z"]),
    ]
    sphere_positions.append(pos)
    spawn_sphere(pos, rgba)

# ---------------------------------------------
# Camera Configurations
# ---------------------------------------------
target = [0, 0, 0]   # Center of workspace
proj = get_projection_matrix(fov=90, aspect=640/480)

cameras = {
    "overhead": {
        "pos": [0.183, 0.13, 1.1],
        "target": [0, 0, 0.05],
    },
    "front": {
        "pos": [1.0, 0, 0.7],
        "target": [0, 0, 0.05],
    },
    "side": {
        "pos": [0, 1.0, 0.7],
        "target": [0, 0, 0.05],
    },
    "isometric": {  # optional — very useful for grounding
        "pos": [0.9, 0.9, 0.9],
        "target": [0, 0, 0],
    }
}


# ---------------------------------------------
# Main Loop
# ---------------------------------------------
for _ in range(120):
    p.stepSimulation()
    time.sleep(1.0/240)

# Capture images from all cameras
for cam_name, cfg in cameras.items():
    view = get_view_matrix(cfg["pos"], cfg["target"])
    rgb = setup_camera(view, proj)
    save_image(rgb, f"captured/{cam_name}.png")

print("Saved images to ./captured/")
