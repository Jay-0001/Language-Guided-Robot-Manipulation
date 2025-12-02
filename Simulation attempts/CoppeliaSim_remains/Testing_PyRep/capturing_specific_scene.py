from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.objects.vision_sensor import VisionSensor
import numpy as np
from PIL import Image
import os

SCENE_FILE = join(dirname(abspath(__file__)), 'scene_panda_6_spheres.ttt')
OUTPUT_DIR = join(dirname(abspath(__file__)), 'output')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define safe, standard camera resolution
RES = [640, 480]

# ----------------------------------------------------------
# LAUNCH SIM
# ----------------------------------------------------------
pr = PyRep()
pr.launch(SCENE_FILE, headless=False)
pr.start()

# ----------------------------------------------------------
# CREATE CAMERAS
# ----------------------------------------------------------

# Top-down camera
cam_top = VisionSensor.create(RES)
cam_top.set_name("cam_top")
cam_top.set_position([0.85, 0.0, 1.55])
cam_top.set_orientation([0.0, 0.0, 0.0])

# Angled camera
cam_side = VisionSensor.create(RES)
cam_side.set_name("cam_side")
cam_side.set_position([0.40, 0.0, 1.20])
cam_side.set_orientation([0.0, 0.0, 0.0])
cam_side.rotate([np.radians(-45), 0.0, 0.0])

# ----------------------------------------------------------
# HELPERS
# ----------------------------------------------------------

def save_rgb(camera: VisionSensor, filepath: str):
    img = camera.capture_rgb()
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(filepath)

def save_depth(camera: VisionSensor, filepath: str):
    depth = camera.capture_depth()
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    Image.fromarray(depth_uint8).save(filepath)

# ----------------------------------------------------------
# CAPTURE
# ----------------------------------------------------------

pr.step()  # one render step

print("Capturing images...")

save_rgb(cam_top,  join(OUTPUT_DIR, "cam_top.png"))
save_rgb(cam_side, join(OUTPUT_DIR, "cam_side.png"))

save_depth(cam_top,  join(OUTPUT_DIR, "cam_top_depth.png"))
save_depth(cam_side, join(OUTPUT_DIR, "cam_side_depth.png"))

print("Saved to:", OUTPUT_DIR)

pr.stop()
