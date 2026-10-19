"""Head control utilities for robot actions."""
import numpy as np
import torch
from logging_utils import get_logger
from typing import TYPE_CHECKING
from utils import T_global_to_robot
from utils import RESET, YELLOW

if TYPE_CHECKING:
    from stretch.agent.robot_agent_dynamem import RobotAgent

logger = get_logger(__name__)

def look_at_target(agent: "RobotAgent", target_xyz: np.ndarray, rerun_str="world/irm_localized_point", log_dir=None):
    agent.rerun_visualizer.clear_identity(rerun_str)
    agent.rerun_visualizer.log_custom_pointcloud(rerun_str, torch.tensor([target_xyz[0], target_xyz[1], target_xyz[2]]), torch.Tensor([0, 1, 0]), 0.1)

    robot_x, robot_y, robot_theta = agent.robot.get_base_pose()
    target_x, target_y, target_z = target_xyz
    T_robot_global = T_global_to_robot(robot_x, robot_y, robot_theta)
    target_robot = T_robot_global @ np.array([target_x, target_y, 1])
    target_robot_theta = np.arctan2(target_robot[1], target_robot[0])

    camera_xyz = agent.robot.get_head_pose()[:3, 3]
    dx = target_x - robot_x; dy = target_y - robot_y; dz = target_z - camera_xyz[2]
    dist_xy = np.sqrt(dx ** 2 + dy ** 2)
    tilt = np.arctan2(dz, dist_xy)

    if -np.pi <= target_robot_theta <= np.pi / 4:
        agent.robot.head_to(target_robot_theta, tilt, blocking=True, reliable=True)
    elif np.pi / 4 < target_robot_theta <= (5.0 / 8.0) * np.pi:
        logger.notice(f"{YELLOW}4/pi <= target_robot_theta <= 5/8*pi -> rotate small to the left{RESET}")
        base_rotation = target_robot_theta - np.pi / 4
        xyt = agent.robot.get_base_pose()
        xyt[2] += base_rotation
        agent.robot.move_base_to(xyt, blocking=True, reliable=True)
        agent.robot.head_to(np.pi / 4, tilt, blocking=True, reliable=True)
    elif (5.0 / 8.0) * np.pi < target_robot_theta <= np.pi:
        logger.notice(f"{YELLOW}5/8*pi < target_robot_theta <= pi -> rotate small to the right{RESET}")
        base_rotation = np.pi - target_robot_theta
        xyt = agent.robot.get_base_pose()
        xyt[2] -= base_rotation
        agent.robot.move_base_to(xyt, blocking=True, reliable=True)
        agent.robot.head_to(-np.pi, tilt, blocking=True, reliable=True)
    else:
        raise ValueError(f"target_robot_theta: {target_robot_theta} is out of range")
