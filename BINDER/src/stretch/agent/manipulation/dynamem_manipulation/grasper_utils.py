# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import numpy as np
import pinocchio as pin
import os
import pickle
import copy
import cv2

from stretch.agent.manipulation.dynamem_manipulation.image_publisher import ImagePublisher
from stretch.agent.manipulation.dynamem_manipulation.place import Placing

from typing import TYPE_CHECKING
from logging_utils import get_logger
from PIL import Image
from datetime import datetime

logger = get_logger(__name__)

if TYPE_CHECKING:
    from stretch.agent.manipulation.dynamem_manipulation.dynamem_manipulation import DynamemManipulationWrapper as ManipulationWrapper

def recv_rgb_img(socket):
    img = socket.recv()
    img = np.frombuffer(img, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return img

def process_image_for_placing(obj, hello_robot, detection_model, save_dir=None):
    if save_dir is not None:
        save_dir = save_dir + "/" + obj
    placing = Placing(hello_robot.robot, detection_model, save_dir=save_dir)
    retries = [
        [0, 0],
        [0, -0.1],
        [0, 0.1],
        [-0.1, 0],
        [-0.1, -0.1],
        [-0.1, 0.1],
        [0.1, 0],
        [0.1, -0.1],
        [0.1, 0.1],
    ]
    success = False
    head_tilt = hello_robot.tilt
    head_pan = hello_robot.pan
    base_trans = 0

    for i in range(9):
        delta_base, delta_tilt = retries[i]
        hello_robot.move_to_position(
            base_trans=base_trans + delta_base, head_tilt=head_tilt + delta_tilt, head_pan=head_pan
        )
        actions = placing.process(obj, 1, head_tilt=head_tilt + delta_tilt)
        if actions is not None:
            base_trans, head_tilt = actions
            hello_robot.move_to_position(
                base_trans=base_trans, head_tilt=head_tilt, head_pan=head_pan
            )
            success = True
            break

    if not success:
        return None, None

    base_trans = 0
    head_tilt = hello_robot.tilt
    head_pan = hello_robot.pan

    for i in range(9):
        delta_base, delta_tilt = retries[i]
        hello_robot.move_to_position(
            base_trans=base_trans + delta_base, head_tilt=head_tilt + delta_tilt, head_pan=head_pan
        )
        translation = placing.process(obj, 2, head_tilt=head_tilt + delta_tilt)
        if translation is not None:
            return [0], np.array([-translation[1], -translation[0], -translation[2]])

    return None, None


def apply_se3_transform(se3_obj, point):
    homogeneous_point = np.append(point.flatten(), 1)
    transformed_homogeneous_point = se3_obj.homogeneous.dot(homogeneous_point)
    transformed_point = transformed_homogeneous_point[:3]

    return transformed_point

def capture_and_process_image(mode, obj, socket, hello_robot, init_head_tilt=None, ok_robot=False):

    image_publisher = ImagePublisher(hello_robot.robot, socket)

    # Centering the object
    head_tilt_angles = [0, -0.1, 0.1]
    tilt_retries, side_retries = 1, 0
    retry_flag = True
    head_tilt = -0.65
    head_pan = -1.57
    head_tilt = init_head_tilt if init_head_tilt is not None else -0.65

    while retry_flag:
        translation, rotation, depth, width, retry_flag = image_publisher.publish_image(
            obj, mode, head_tilt=head_tilt
        )

        if retry_flag == 1:
            base_trans = translation[0]
            if not ok_robot:
                head_tilt += rotation[0]

            hello_robot.move_to_position(
                base_trans=base_trans, head_pan=head_pan, head_tilt=head_tilt
            )

        elif retry_flag != 0 and side_retries == 3:
            if mode == "place":
                return None, None, None
            else:
                return None, None, None, None, None

        elif side_retries == 2 and tilt_retries == 3:
            hello_robot.move_to_position(base_trans=0.1, head_tilt=head_tilt)
            side_retries = 3

        elif retry_flag == 2:
            if tilt_retries == 3:
                if side_retries == 0:
                    hello_robot.move_to_position(base_trans=0.1, head_tilt=head_tilt)
                    side_retries = 1
                else:
                    hello_robot.move_to_position(base_trans=-0.2, head_tilt=head_tilt)
                    side_retries = 2
                tilt_retries = 1
            else:
                hello_robot.move_to_position(
                    head_pan=head_pan, head_tilt=head_tilt + head_tilt_angles[tilt_retries]
                )
                tilt_retries += 1

    if mode == "place":
        translation = np.array([-translation[1], -translation[0], -translation[2]])
        # Get the place best pose image
        socket.send_string("Waiting for place best pose image")
        place_best_pose_image = recv_rgb_img(socket)

    if mode == "pick":
        socket.send_string("Waiting for head best pose image")
        head_best_pose_image = recv_rgb_img(socket)
        return rotation, translation, depth, width, head_best_pose_image
    else:
        return rotation, translation, place_best_pose_image


def move_to_point(robot, point, base_node, gripper_node, move_mode=1, pitch_rotation=0):
    rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    dest_frame = pin.SE3(rotation, point)
    transform = robot.get_joint_transform(base_node, gripper_node)

    transformed_frame = transform * dest_frame

    transformed_frame.translation[2] -= 0.2
    

    robot.move_to_pose(
        [
            transformed_frame.translation[0],
            transformed_frame.translation[1],
            transformed_frame.translation[2],
        ],
        [pitch_rotation, 0, 0],
        [1],
        move_mode=move_mode,
    )

def finalize_open_loop(robot: "ManipulationWrapper", save_dict: dict, t_base_C: np.ndarray, ref_diff: float, gripper_width: float, log=None):
    is_logging_enabled = log is not None
    def _log(key, value):
        if is_logging_enabled:
            save_dict[key] = copy.deepcopy(value)
    
    R_gripper_base = robot.get_joint_transform("base_link", "link_gripper_s3_body")
    t_gripper_C_new = apply_se3_transform(R_gripper_base, t_base_C)
    cur_diff = t_gripper_C_new[2]
    _log('t_gripper_C_new', t_gripper_C_new)

    diff = abs(cur_diff - ref_diff)
    if diff > 0.08:
        dist = diff - 0.08
        state = robot.robot.get_six_joints()
        state[1] += 0.02
        state[2] += 0.02
        robot.robot.arm_to(state, blocking=True)
        robot.move_to_pose([0, 0, dist], [0, 0, 0], [1])
        diff = diff - dist
    
    while diff > 0.01:
        dist = min(0.03, diff)
        robot.move_to_pose([0, 0, dist], [0, 0, 0], [1])
        diff = diff - dist
    
    if is_logging_enabled:
        with open(os.path.join(log, "pickup.pkl"), "wb") as f:
            pickle.dump(save_dict, f)
    
    robot.robot.started_gripper_closing = True
    robot.pickup(gripper_width)

def finalize_closed_loop(robot: "ManipulationWrapper", save_dict: dict, translation: np.ndarray, rotation: np.ndarray,
                         depth: float, width: float, log=None):
    is_logging_enabled = log is not None
    def _log(key, value):
        if is_logging_enabled:
            save_dict[key] = copy.deepcopy(value)
    
    t_gripper_camera_C = np.array([translation[0], -translation[1], translation[2]])
    R_gripper_camera_anygrasp = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
    )
    R_anygrasp_C = np.array(
        [
            [rotation[0][0], rotation[0][1], rotation[0][2]],
            [rotation[1][0], rotation[1][1], rotation[1][2]],
            [rotation[2][0], rotation[2][1], rotation[2][2]],
        ]
    )
    R_gripper_camera_C = np.dot(R_gripper_camera_anygrasp, R_anygrasp_C)
    se3_gripper_camera_C = pin.SE3(R_gripper_camera_C, t_gripper_camera_C)
    R_gripper_body_gripper_camera = robot.get_joint_transform("gripper_camera_depth_optical_frame", "link_gripper_s3_body")
    se3_gripper_body_C = R_gripper_body_gripper_camera * se3_gripper_camera_C

    R_C_C_prime = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]])
    final_rotation = np.dot(se3_gripper_body_C.rotation, R_C_C_prime)
    rpy_angles = pin.rpy.matrixToRpy(final_rotation)

    R_base_gripper_camera = robot.get_joint_transform("gripper_camera_depth_optical_frame", "base_link")
    t_base_C = apply_se3_transform(R_base_gripper_camera, t_gripper_camera_C)

    robot.move_to_pose([0, 0, 0], [rpy_angles[0], rpy_angles[1], rpy_angles[2]], [1])

    R_gripper_body_base = robot.get_joint_transform("base_link", "link_gripper_s3_body")
    t_gripper_body = apply_se3_transform(R_gripper_body_base, t_base_C)

    diff_value = 0.225 - depth - 0.03
    t_gripper_body[2] -= diff_value
    robot.move_to_pose(
        translation_tensor=[t_gripper_body[0], t_gripper_body[1], t_gripper_body[2] - 0.03],
        rotational_tensor=[0, 0, 0],
        gripper=[1],
        move_mode=1,
    )

    robot.robot.started_gripper_closing = True
    robot.pickup(width)    
    

def capped_step_in_gripper_frame(robot, step_g, ratio=0.7):
    T_base_gripper_body = robot.get_joint_transform("link_gripper_s3_body", "base_link")
    R_base_gripper_body = T_base_gripper_body.rotation; t_base_gripper_body = T_base_gripper_body.translation

    gripper_step_base = R_base_gripper_body.dot(step_g)
    gripper_step_base[2] *= ratio

    T_gripper_body_base = robot.get_joint_transform("base_link", "link_gripper_s3_body")
    R_gripper_body_base = T_gripper_body_base.rotation; t_gripper_body_base = T_gripper_body_base.translation
    return R_gripper_body_base.dot(gripper_step_base)

def pickup(
    robot: "ManipulationWrapper",
    rotation,
    translation,
    base_node,
    gripper_node,
    gripper_height=0.03,
    gripper_depth=0.03,
    gripper_width=1,
    log=None,
    ablation=False,
):
    is_logging_enabled = log is not None
    save_dict = {}

    if is_logging_enabled:
        os.makedirs(log, exist_ok=True)

    def _log(key, value):
        if is_logging_enabled:
            save_dict[key] = copy.deepcopy(value)

    _log('rotation', rotation)
    _log('translation', translation)

    if is_logging_enabled:
        image_publisher = ImagePublisher(robot.robot, socket=None, which="head")
        head_dict = image_publisher.save_dict(return_dict=True)
        _log('head_dict', head_dict)

    pin_point = np.array([-translation[1], -translation[0], translation[2]])
    rotation1_bottom_mat = np.array( # R_head<-anygrasp
        [
            [0.0000000, -1.0000000, 0.0000000],
            [-1.0000000, 0.0000000, 0.0000000],
            [0.0000000, 0.0000000, 1.0000000],
        ]
    )

    rotation1_mat = np.array(
        [
            [rotation[0][0], rotation[0][1], rotation[0][2]],
            [rotation[1][0], rotation[1][1], rotation[1][2]],
            [rotation[2][0], rotation[2][1], rotation[2][2]],
        ]
    )

    pin_rotation = np.dot(rotation1_bottom_mat, rotation1_mat)
    _log('pin_point', pin_point)
    _log('rotation1_bottom_mat', rotation1_bottom_mat)
    _log('rotation1_mat', rotation1_mat)
    _log('pin_rotation', pin_rotation)

    pin_dest_frame = pin.SE3(np.array(pin_rotation), np.array(pin_point))
    _log('pin_dest_frame', pin_dest_frame)

    pin_cam2gripper_transform = robot.get_joint_transform(base_node, gripper_node)
    pin_transformed_frame = pin_cam2gripper_transform * pin_dest_frame
    _log('pin_cam2gripper_transform_start', pin_cam2gripper_transform)
    _log('pin_transformed_frame_start', pin_transformed_frame)
    _log('joints_pin_start', robot.joints_pin)

    robot.move_to_position(gripper_pos=gripper_width)
    robot.move_to_position(lift_pos=1.05, head_pan=None, head_tilt=None)

    temp = robot.get_joint_transform(base_node, gripper_node)
    _log('pin_cam2gripper_transform_after_lift', temp)
    _log('joints_pin_after_lift', robot.joints_pin)

    rotation2_top_mat = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]])
    pin_final_rotation = np.dot(pin_transformed_frame.rotation, rotation2_top_mat)

    rpy_angles = pin.rpy.matrixToRpy(pin_final_rotation);
    _log('rotation2_top_mat', rotation2_top_mat)
    _log('pin_final_rotation', pin_final_rotation)
    _log('rpy_angles', rpy_angles)

    robot.move_to_pose(
        [0, 0, 0],
        [rpy_angles[0], rpy_angles[1], rpy_angles[2]],
        [1],
    )

    pin_cam2gripper_transform = robot.get_joint_transform(base_node, gripper_node)
    pin_transformed_point1 = apply_se3_transform(pin_cam2gripper_transform, pin_point)
    _log('pin_cam2gripper_transform_after_gripper_rotation', pin_cam2gripper_transform)
    _log('pin_transformed_point1', pin_transformed_point1)
    _log('joints_pin_after_gripper_rotation', robot.joints_pin)

    pin_cam2base_transform = robot.get_joint_transform(base_node, "base_link")
    pin_base_point = apply_se3_transform(pin_cam2base_transform, pin_point)
    _log('pin_cam2base_transform', pin_cam2base_transform)
    _log('pin_base_point', pin_base_point)
    _log('joints_pin_after_gripper_rotation_2', robot.joints_pin)

    diff_value = (
        0.225 - gripper_depth - gripper_height
    )
    pin_transformed_point1[2] -= diff_value
    ref_diff = diff_value

    robot.move_to_pose(
        translation_tensor=[pin_transformed_point1[0], pin_transformed_point1[1], pin_transformed_point1[2] - 0.2],
        rotational_tensor=[0, 0, 0],
        gripper=[1],
        move_mode=1,
    )

    temp = robot.get_joint_transform(base_node, gripper_node)
    _log('joints_pin_after_lowering', robot.joints_pin)
    
    if is_logging_enabled:
        image_publisher = ImagePublisher(robot.robot, socket=None, which="ee")
        ee_dict = image_publisher.save_dict(return_dict=True)
        _log('ee_dict', ee_dict)

    pin_base2gripper_transform = robot.get_joint_transform("base_link", gripper_node)
    pin_transformed_point2 = apply_se3_transform(pin_base2gripper_transform, pin_base_point)
    curr_diff = pin_transformed_point2[2]

    _log('pin_base2gripper_transform', pin_base2gripper_transform)
    _log('pin_transformed_point2', pin_transformed_point2)
    _log('joints_pin_after_lowering_2', robot.joints_pin)

    diff = abs(curr_diff - ref_diff)
    if diff > 0.08:
        dist = diff - 0.08
        state = robot.robot.get_six_joints()
        state[1] += 0.02
        state[2] += 0.02
        robot.robot.arm_to(state, blocking=True)

        step_g = capped_step_in_gripper_frame(robot, np.array([0.0, 0.0, dist]), ratio=0.7)
        robot.move_to_pose([step_g[0], step_g[1], step_g[2]], [0, 0, 0], [1])
        diff = diff - dist
    
    temp = robot.get_joint_transform(base_node, gripper_node)
    _log('pin_base2gripper_transform_2', temp)
    _log('joints_pin_after_coarse_moving', robot.joints_pin)

    while diff > 0.01:
        dist = min(0.03, diff)
        robot.move_to_pose([0, 0, dist], [0, 0, 0], [1])
        diff = diff - dist

    if is_logging_enabled:
        with open(os.path.join(log, "pickup.pkl"), "wb") as f:
            pickle.dump(save_dict, f)

    robot.robot.started_gripper_closing = True        
    robot.pickup(gripper_width)

    robot.move_to_position(lift_pos=min(robot.robot.get_six_joints()[1] + 0.2, 1.1))

    robot.move_to_position(arm_pos=0.01)
    robot.move_to_position(wrist_pitch=0.0)
    robot.move_to_position(lift_pos=min(robot.robot.get_six_joints()[1], 0.9), wrist_yaw=2.5)
    robot.move_to_position(lift_pos=min(robot.robot.get_six_joints()[1], 0.55))

def pickup_with_ee_camera(robot: "ManipulationWrapper", rotation, translation, base_node, gripper_node, 
                          gripper_height=0.03, gripper_depth=0.03, gripper_width=1, log=None, text=None,
                          socket=None):
    if log is not None:
        os.makedirs(log, exist_ok=True)
        save_dict = {'rotation': rotation, 'translation': translation}

        image_publisher = ImagePublisher(robot.robot, socket=None, which="head")
        head_dict = image_publisher.save_dict(return_dict=True)
        save_dict['head_dict'] = copy.deepcopy(head_dict)
    
    pin_point = np.array([-translation[1], -translation[0], translation[2]])
    rotation1_bottom_mat = np.array(
        [
            [0.0000000, -1.0000000, 0.0000000],
            [-1.0000000, 0.0000000, 0.0000000],
            [0.0000000, 0.0000000, 1.0000000],
        ]
    )
    rotation1_mat = np.array(
        [
            [rotation[0][0], rotation[0][1], rotation[0][2]],
            [rotation[1][0], rotation[1][1], rotation[1][2]],
            [rotation[2][0], rotation[2][1], rotation[2][2]],
        ]
    )
    pin_rotation = np.dot(rotation1_bottom_mat, rotation1_mat)
    if log is not None:
        save_dict['pin_point'] = pin_point; save_dict['rotation1_bottom_mat'] = rotation1_bottom_mat; save_dict['rotation1_mat'] = rotation1_mat
        save_dict['pin_rotation'] = pin_rotation

    pin_dest_frame = pin.SE3(np.array(pin_rotation), np.array(pin_point))
    if log is not None:
        save_dict['pin_dest_frame'] = copy.deepcopy(pin_dest_frame)

    pin_cam2gripper_transform = robot.get_joint_transform(base_node, gripper_node)
    pin_transformed_frame = pin_cam2gripper_transform * pin_dest_frame
    if log is not None:
        save_dict['pin_cam2gripper_transform_start'] = copy.deepcopy(pin_cam2gripper_transform)
        save_dict['pin_transformed_frame_start'] = copy.deepcopy(pin_transformed_frame)
        save_dict['joints_pin_start'] = copy.deepcopy(robot.joints_pin)

    robot.move_to_position(gripper_pos=gripper_width)
    robot.move_to_position(lift_pos=1.05, head_pan=None, head_tilt=None)

    temp = robot.get_joint_transform(base_node, gripper_node)
    if log is not None:
        save_dict['pin_cam2gripper_transform_after_lift'] = copy.deepcopy(temp)
        save_dict['joints_pin_after_lift'] = copy.deepcopy(robot.joints_pin)

    rotation2_top_mat = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]])
    pin_final_rotation = np.dot(pin_transformed_frame.rotation, rotation2_top_mat)
    rpy_angles = pin.rpy.matrixToRpy(pin_final_rotation)
    if log is not None:
        save_dict['rotation2_top_mat'] = rotation2_top_mat
        save_dict['pin_final_rotation'] = pin_final_rotation
        save_dict['rpy_angles'] = rpy_angles

    robot.move_to_pose(
        [0, 0, 0],
        [rpy_angles[0], rpy_angles[1], rpy_angles[2]],
        [1],
    )

    pin_cam2gripper_transform = robot.get_joint_transform(base_node, gripper_node)
    pin_transformed_point1 = apply_se3_transform(pin_cam2gripper_transform, pin_point)
    if log is not None:
        save_dict['pin_cam2gripper_transform_after_gripper_rotation'] = copy.deepcopy(pin_cam2gripper_transform)
        save_dict['pin_transformed_point1'] = pin_transformed_point1
        save_dict['joints_pin_after_gripper_rotation'] = copy.deepcopy(robot.joints_pin)
    
    pin_cam2base_transform = robot.get_joint_transform(base_node, "base_link")
    pin_base_point = apply_se3_transform(pin_cam2base_transform, pin_point)
    if log is not None:
        save_dict['pin_cam2base_transform'] = copy.deepcopy(pin_cam2base_transform)
        save_dict['pin_base_point'] = pin_base_point
        save_dict['joints_pin_after_gripper_rotation_2'] = copy.deepcopy(robot.joints_pin)

    diff_value = (
        0.225 - gripper_depth - gripper_height
    )
    pin_transformed_point1[2] -= diff_value
    ref_diff = diff_value

    robot.move_to_pose(
        translation_tensor=[pin_transformed_point1[0], pin_transformed_point1[1], pin_transformed_point1[2] - 0.2],
        rotational_tensor=[0, 0, 0],
        gripper=[1],
        move_mode=1,
    )
    
    image_publisher = ImagePublisher(robot.robot, socket=socket, which="ee")
    translation, rotation, depth, width, retry_flag = image_publisher.publish_image(
        text=text,
        mode="pick",
        head_tilt=None,
    )

    pin_point = np.array([translation[0], -translation[1], translation[2]])
    R_gripper_camera_anygrasp = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
    )
    rotation1_mat = np.array(
        [
            [rotation[0][0], rotation[0][1], rotation[0][2]],
            [rotation[1][0], rotation[1][1], rotation[1][2]],
            [rotation[2][0], rotation[2][1], rotation[2][2]],
        ]
    )
    pin_rotation = np.dot(R_gripper_camera_anygrasp, rotation1_mat)
    pin_dest_frame = pin.SE3(np.array(pin_rotation), np.array(pin_point))

    pin_gripper_camera_to_gripper_transform = robot.get_joint_transform("gripper_camera_depth_optical_frame", "link_gripper_s3_body")
    pin_transformed_frame = pin_gripper_camera_to_gripper_transform * pin_dest_frame

    rotation2_top_mat = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]])
    pin_final_rotation = np.dot(pin_transformed_frame.rotation, rotation2_top_mat)
    rpy_angles = pin.rpy.matrixToRpy(pin_final_rotation)

    pin_gripper_camera_to_base_transform = robot.get_joint_transform("gripper_camera_depth_optical_frame", "base_link")
    pin_transformed_point1 = apply_se3_transform(pin_gripper_camera_to_base_transform, pin_point)
    
    robot.move_to_pose(
        [0, 0, 0],
        [rpy_angles[0], rpy_angles[1], rpy_angles[2]],
        [1],
    )

    pin_base_to_gripper_transform = robot.get_joint_transform("base_link", "link_gripper_s3_body")
    pin_transformed_point1_gripper = apply_se3_transform(pin_base_to_gripper_transform, pin_transformed_point1)

    diff_value = 0.225 - depth - gripper_height
    pin_transformed_point1_gripper[2] -= diff_value
    robot.move_to_pose(
        translation_tensor=[pin_transformed_point1_gripper[0], pin_transformed_point1_gripper[1], pin_transformed_point1_gripper[2] - 0.03],
        rotational_tensor=[0, 0, 0],
        gripper=[1],
        move_mode=1,
    )


def pickup_k(robot: "ManipulationWrapper",
               rotation,
               translation,
               base_node,
               gripper_node,
               gripper_height=0.03,
               gripper_depth=0.03,
               gripper_width=1,
               log=None,
               text=None,
               socket=None,
               k=2,
               irm_event_queue=None,
               rerun_visualizer=None,
               enable_irm=False,
):
    is_logging_enabled = log is not None
    save_dict = {}
    if is_logging_enabled:
        os.makedirs(log, exist_ok=True)
    def _log(key, value):
        if is_logging_enabled:
            save_dict[key] = copy.deepcopy(value)

    _log('rotation', rotation)
    _log('translation', translation)
    
    t_head_C = np.array([-translation[1], -translation[0], translation[2]])
    R_head_anygrasp = np.array(
        [
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    R_head_C = np.dot(R_head_anygrasp, rotation)
    se3_head_C = pin.SE3(np.array(R_head_C), np.array(t_head_C))
    _log('R_head_C', R_head_C)
    _log('se3_head_C', se3_head_C)

    R_gripper_body_head = robot.get_joint_transform("camera_depth_optical_frame", "link_gripper_s3_body")
    se3_gripper_body_C = R_gripper_body_head * se3_head_C
    _log('se3_gripper_body_C', se3_gripper_body_C)
    _log('joints_pin_start', robot.joints_pin)

    robot.move_to_position(gripper_pos=gripper_width)
    robot.move_to_position(lift_pos=1.05, head_pan=None, head_tilt=None)

    R_C_C_prime = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]])
    R_gripper_C_prime = np.dot(se3_gripper_body_C.rotation, R_C_C_prime)
    rpy_angles = pin.rpy.matrixToRpy(R_gripper_C_prime)
    _log('R_gripper_C_prime', R_gripper_C_prime)
    _log('rpy_angles', rpy_angles)

    robot.move_to_pose([0, 0, 0], [rpy_angles[0], rpy_angles[1], rpy_angles[2]], [1])

    
    se3_gripper_head = robot.get_joint_transform("camera_depth_optical_frame", "link_gripper_s3_body")
    t_gripper_C = apply_se3_transform(se3_gripper_head, t_head_C)
    _log('t_gripper_C', t_gripper_C)

    se3_base_head = robot.get_joint_transform("camera_depth_optical_frame", "base_link")
    t_base_C = apply_se3_transform(se3_base_head, t_head_C)
    _log('t_base_C', t_base_C)

    diff_value = 0.225 - gripper_depth - gripper_height
    t_gripper_C[2] -= diff_value
    ref_diff = diff_value
    robot.robot.started_gripper_lowering = True
    robot.move_to_pose([t_gripper_C[0], t_gripper_C[1], t_gripper_C[2] - 0.2], [0, 0, 0], [1], move_mode=1)

    if not enable_irm:
        is_grasp_feasible = True
    else:
        is_grasp_feasible = check_grasp_feasibility(irm_event_queue)
    if not is_grasp_feasible:
        image_publisher = ImagePublisher(robot.robot, socket=socket, which="ee")
        translation, rotation, depth, width, retry_flag, ee_best_pose_image = image_publisher.publish_image(
            text=text,
            mode="pick",
            head_tilt=None,
        )
        if retry_flag:
            finalize_open_loop(robot, save_dict, t_base_C, ref_diff, gripper_width, log)
        else:
            if ee_best_pose_image is not None:
                rerun_visualizer.queue.put({"type": "log_custom_2d_image", "identity_name": "/observation_similar_to_text", "img": ee_best_pose_image})
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                Image.fromarray(ee_best_pose_image).save(f"{log}/anygrasp_{text}_{current_time}.png")
            t_base_C, ref_diff, gripper_width, save_dict = finalize_closed_loop_k(robot, save_dict, translation, rotation, depth, width, log, is_last_stage=True)

    else:
        finalize_open_loop(robot, save_dict, t_base_C, ref_diff, gripper_width, log)
        return
    
def check_grasp_feasibility(irm_event_queue):
    # The IRM reports a grasp via the guidance-conditioned mode m_t: CONTINUE means the
    # pose is feasible (proceed open-loop); ADJUST means re-grasp closed-loop. Compare on
    # the mode name so this upstream helper stays decoupled from the agent's ExecutionMode.
    response = irm_event_queue.get()
    if response.get("type") != "grasping_refinement_result":
        return False
    mode = response.get("mode")
    return getattr(mode, "name", None) == "CONTINUE"

    

def finalize_closed_loop_k(robot: "ManipulationWrapper", save_dict: dict, translation: np.ndarray, rotation: np.ndarray,
                         depth: float, width: float, log=None, is_last_stage=False):
    is_logging_enabled = log is not None
    def _log(key, value):
        if is_logging_enabled:
            save_dict[key] = copy.deepcopy(value)
    
    t_gripper_camera_C = np.array([translation[0], -translation[1], translation[2]])
    R_gripper_camera_anygrasp = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
    )
    R_anygrasp_C = np.array(
        [
            [rotation[0][0], rotation[0][1], rotation[0][2]],
            [rotation[1][0], rotation[1][1], rotation[1][2]],
            [rotation[2][0], rotation[2][1], rotation[2][2]],
        ]
    )
    R_gripper_camera_C = np.dot(R_gripper_camera_anygrasp, R_anygrasp_C)
    se3_gripper_camera_C = pin.SE3(R_gripper_camera_C, t_gripper_camera_C) # [R_gripper_camera<-C, t_gripper_camera_C]
    R_gripper_body_gripper_camera = robot.get_joint_transform("gripper_camera_depth_optical_frame", "link_gripper_s3_body")
    se3_gripper_body_C = R_gripper_body_gripper_camera * se3_gripper_camera_C

    R_C_C_prime = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]])
    final_rotation = np.dot(se3_gripper_body_C.rotation, R_C_C_prime)
    rpy_angles = pin.rpy.matrixToRpy(final_rotation)

    R_base_gripper_camera = robot.get_joint_transform("gripper_camera_depth_optical_frame", "base_link")
    t_base_C = apply_se3_transform(R_base_gripper_camera, t_gripper_camera_C)

    robot.move_to_pose([0, 0, 0], [rpy_angles[0], rpy_angles[1], rpy_angles[2]], [1])

    R_gripper_body_base = robot.get_joint_transform("base_link", "link_gripper_s3_body")
    t_gripper_body = apply_se3_transform(R_gripper_body_base, t_base_C)

    if not is_last_stage:
        diff_value = 0.225 - depth - 0.03
        t_gripper_body[2] -= diff_value
        t_gripper_body[0] /= 2.0; t_gripper_body[1] /= 2.0; t_gripper_body[2] /= 2.0
        robot.move_to_pose(
            translation_tensor=[t_gripper_body[0], t_gripper_body[1], t_gripper_body[2] - 0.015],
            rotational_tensor=[0, 0, 0],
            gripper=[1],
            move_mode=1,
        )
    else:
        diff_value = 0.225 - depth - 0.02
        t_gripper_body[2] -= diff_value
        robot.move_to_pose(
            translation_tensor=[t_gripper_body[0], t_gripper_body[1], t_gripper_body[2] - 0.01],
            rotational_tensor=[0, 0, 0],
            gripper=[1],
            move_mode=1,
        )
        robot.robot.started_gripper_closing = True
        robot.pickup(width)
    
    ref_diff = diff_value

    return t_base_C, ref_diff, width, save_dict
    