import pybullet as p
import random

COLOR_PALETTE = {
    "red": (1, 0, 0, 1),
    "green": (0, 1, 0, 1),
    "blue": (0, 0, 1, 1),
    "yellow": (1, 1, 0, 1),
    "black": (0.1, 0.1, 0.1, 1),
    "pink": (1, 0.4, 0.7, 1),
    "cyan": (0.2, 1, 1, 1),
    "purple": (0.6, 0.2, 0.8, 1),
}

def sample_color():
    name = random.choice(list(COLOR_PALETTE.keys()))
    rgba = COLOR_PALETTE[name]
    return name, rgba

def make_sphere(position, color):
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=color)
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.03)
    p.createMultiBody(baseMass=0.05, baseVisualShapeIndex=vis,
                      baseCollisionShapeIndex=col, basePosition=position)

def make_cube(position, color):
    half = 0.04
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[half]*3, rgbaColor=color)
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half]*3)
    p.createMultiBody(baseMass=0.05, baseVisualShapeIndex=vis,
                      baseCollisionShapeIndex=col, basePosition=position)

def make_cylinder(position, color):
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.03, length=0.1, rgbaColor=color)
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.03, height=0.1)
    p.createMultiBody(baseMass=0.05, baseVisualShapeIndex=vis,
                      baseCollisionShapeIndex=col, basePosition=position)
