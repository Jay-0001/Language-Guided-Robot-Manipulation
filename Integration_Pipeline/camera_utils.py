import pybullet as p
import pybullet_data
import numpy as np
import cv2
import os

def setup_camera(view_matrix, projection_matrix):
    """Captures an RGB frame using the given camera matrices."""
    width, height = 640, 480

    _, _, rgb, _, _ = p.getCameraImage(
        width,
        height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )

    rgb = np.reshape(rgb, (height, width, 4))  # RGBA
    rgb = rgb[:, :, :3]  # Drop alpha

    return rgb


def save_image(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)


def get_view_matrix(camera_pos, target_pos, up_vec=[0, 0, 1]):
    return p.computeViewMatrix(
        cameraEyePosition=camera_pos,
        cameraTargetPosition=target_pos,
        cameraUpVector=up_vec
    )


def get_projection_matrix(fov=60, aspect=1.0, near=0.01, far=2.0):
    return p.computeProjectionMatrixFOV(
        fov=fov,
        aspect=aspect,
        nearVal=near,
        farVal=far
    )
