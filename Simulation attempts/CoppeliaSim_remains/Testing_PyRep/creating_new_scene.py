from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
import numpy as np
import random

# --- CONFIG ---
SCENE_FILE = join(dirname(abspath(__file__)), 'scene_panda_reach_target.ttt')
SAVE_AS = join(dirname(abspath(__file__)), 'scene_panda_6_spheres.ttt')

# Workspace bounds (adjust if needed)
# Keeping it slightly inside your original reach target volume
pos_min = [0.65, -0.25, 0.80]
pos_max = [0.95,  0.25, 1.20]

# Colors (R,G,B)
COLORS = {
    "red":    [1.0, 0.0, 0.0],
    "green":  [0.0, 1.0, 0.0],
    "blue":   [0.0, 0.0, 1.0],
    "yellow": [1.0, 1.0, 0.0],
    "black":  [0.0, 0.0, 0.0],
    "pink":   [1.0, 0.4, 0.7]
}

# Sphere radius
RADIUS = 0.04   # 4 cm

# --- MAIN SCRIPT ---
pr = PyRep()
pr.launch(SCENE_FILE, headless=False)
pr.start()

print("Creating spheres...")

created_spheres = []

for name, color in COLORS.items():
    # Random position inside the workspace box
    pos = [
        np.random.uniform(pos_min[0], pos_max[0]),
        np.random.uniform(pos_min[1], pos_max[1]),
        np.random.uniform(pos_min[2], pos_max[2]),
    ]

    sphere = Shape.create(
        type=PrimitiveShape.SPHERE,
        size=[RADIUS, RADIUS, RADIUS],
        color=color,
        static=True,
        respondable=False
    )
    sphere.set_name(f"{name}_sphere")
    sphere.set_position(pos)

    created_spheres.append(sphere)
    print(f"→ Created {name} sphere at {pos}")

# Give simulation a few steps to settle
for _ in range(10):
    pr.step()

print("Saving modified scene as:", SAVE_AS)
pr.export_scene(SAVE_AS)

pr.stop()
print("Scene saved and PyRep stopped.")
