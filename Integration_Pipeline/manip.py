import pybullet as p
import pybullet_data
import time
import math
import random

# ============================
# 1. LOW-LEVEL MOTION HELPERS
# ============================

def move_to(panda, eef_link, pos, orn, steps=120):
    joint_angles = p.calculateInverseKinematics(
        panda,
        eef_link,
        pos,
        orn,
        maxNumIterations=60,
        residualThreshold=1e-4
    )

    for _ in range(steps):
        for j in range(7):   # 7 DoF arm
            p.setJointMotorControl2(
                panda, j, p.POSITION_CONTROL,
                joint_angles[j], force=200
            )
        p.stepSimulation()
        time.sleep(1/240)


def close_gripper(panda, tightness=-0.01, force=150):
    p.setJointMotorControl2(panda, 9, p.POSITION_CONTROL, tightness, force=force)
    p.setJointMotorControl2(panda, 10, p.POSITION_CONTROL, tightness, force=force)
    for _ in range(50):
        p.stepSimulation()
        time.sleep(1/240)


def open_gripper(panda):
    p.setJointMotorControl2(panda, 9, p.POSITION_CONTROL, 0.04, force=150)
    p.setJointMotorControl2(panda, 10, p.POSITION_CONTROL, 0.04, force=150)
    for _ in range(50):
        p.stepSimulation()
        time.sleep(1/240)


# ============================
# 2. OBJECT SPAWN HELPERS
# ============================

def spawn_scattered_static_spheres(n=6):
    colors = [
        (1,0,0,1), (0,1,0,1), (0,0,1,1),
        (1,1,0,1), (0,0,0,1), (1,0,1,1)
    ]

    sphere_ids = []
    radius = 0.04

    x_range = (0.45, 0.65)
    y_range = (-0.15, 0.15)
    z_range = (0.12, 0.20)

    for i in range(n):
        x = random.uniform(*x_range)
        y = random.uniform(*y_range)
        z = random.uniform(*z_range)

        visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=colors[i]
        )
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)

        body = p.createMultiBody(
            baseMass=0,         
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=visual,
            basePosition=[x,y,z]
        )
        sphere_ids.append(body)

    return sphere_ids


# ============================
# 3. HIGH-LEVEL MANIPULATION
# ============================

def pick_and_place(panda, sphere_id, place_position):
    """
    Core high-level API:
    - Robot moves to sphere
    - Grasp
    - Lift
    - Move to place position
    - Release
    - Lift away
    """

    eef_link = 11
    orn = p.getQuaternionFromEuler([math.pi, 0, 0])

    # Get the sphere pose
    pos, _ = p.getBasePositionAndOrientation(sphere_id)
    print("\n=== PICK TARGET ===")
    print("Sphere ID:", sphere_id)
    print("World Position:", pos)
    print("===================\n")

    # Pre-grasp
    pre = [pos[0], pos[1], pos[2] + 0.15]
    move_to(panda, eef_link, pre, orn)

    # Grasp
    grasp = [pos[0], pos[1], pos[2] + 0.03]
    move_to(panda, eef_link, grasp, orn)

    close_gripper(panda)

    # Lift while still static
    lift = [pos[0], pos[1], pos[2] + 0.25]
    move_to(panda, eef_link, lift, orn)

    # Now enable physics only after grasp lift
    p.changeDynamics(sphere_id, -1, mass=0.05)
    p.changeDynamics(sphere_id, -1, restitution=0.0)
    p.resetBaseVelocity(sphere_id, [0,0,0], [0,0,0])

    # Place
    px, py, pz = place_position
    place_pre = [px, py, pz + 0.20]
    move_to(panda, eef_link, place_pre, orn)

    place_final = [px, py, pz]
    move_to(panda, eef_link, place_final, orn)

    open_gripper(panda)

    # Enable gravity after release
    p.setGravity(0, 0, -9.8)

    # Retreat upward
    retreat = [px, py, pz + 0.20]
    move_to(panda, eef_link, retreat, orn)


# ============================
# 4. ENVIRONMENT SETUP
# ============================

def setup_environment():
    """
    Creates PyBullet GUI, plane, Panda arm,
    sets finger friction and disables gravity initially.
    """
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    p.setGravity(0, 0, 0)

    panda = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

    # Improve grip friction
    p.changeDynamics(panda, 9, lateralFriction=2.0)
    p.changeDynamics(panda, 10, lateralFriction=2.0)

    return panda


# ============================
# 5. MAIN (only for testing)
# ============================

if __name__ == "__main__":

    panda = setup_environment()
    spheres = spawn_scattered_static_spheres()

    sphere_id = spheres[0]          # pick the red sphere
    place_pos = [0.6, 0.0, 0.041]   # placement on ground

    pick_and_place(panda, sphere_id, place_pos)

    while True:
        p.stepSimulation()
        time.sleep(1/240)