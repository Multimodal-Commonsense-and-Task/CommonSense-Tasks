import os
from datetime import datetime

import numpy as np
import torch
from PIL import Image

from logging_utils import get_logger
from utils import compute_tilt

logger = get_logger(__name__)


def ensure_log_directory(base_log_dir: str, action_name: str, drm_call_count: int) -> str:
    log_dir = f"{base_log_dir}/{drm_call_count}_{action_name}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def save_manipulation_image(
    image: np.ndarray,
    log_dir: str,
    prefix: str,
    target_name: str,
    rerun_visualizer=None,
) -> str:
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{log_dir}/{prefix}_{target_name}_{current_time}.png"
    Image.fromarray(image).save(save_path)

    if rerun_visualizer is not None:
        rerun_visualizer.queue.put({
            "type": "log_custom_2d_image",
            "identity_name": "/observation_similar_to_text",
            "img": image
        })

    return save_path


def compute_head_tilt_for_target(
    camera_xyz: np.ndarray,
    target_point: np.ndarray,
    default_tilt: float = -0.65,
    max_height: float = 2.0,
    clamp_height: float = 1.0,
) -> float:
    if isinstance(target_point, torch.Tensor):
        target = target_point.numpy()
    else:
        target = target_point.copy()
    if target[2] >= max_height:
        target[2] = clamp_height
    return compute_tilt(camera_xyz, target)


def determine_gripper_width(width: float, robot_edition: int = 3) -> float:
    if robot_edition == 3:
        if width < 0.05:
            return 0.45
        elif width < 0.075:
            return 0.6
    return 1.0


def retract_arm_after_place(manip_wrapper, max_lift: float = 1.1):
    current_lift = manip_wrapper.robot.get_six_joints()[1]

    manip_wrapper.move_to_position(lift_pos=min(current_lift + 0.4, max_lift))

    manip_wrapper.move_to_position(gripper_pos=1, lift_pos=1.05, arm_pos=0)
    manip_wrapper.move_to_position(wrist_pitch=-1.57)
