# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import cv2
import numpy as np
import pickle
from datetime import datetime

from logging_utils import get_logger
logger = get_logger(__name__)

def recv_rgb_img(socket):
    img = socket.recv()
    img = np.frombuffer(img, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return img

def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    return socket.send_pyobj(A)


def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    return socket.recv_pyobj()

def send_rgb_img(socket, img):
    img = img.astype(np.uint8)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    _, img_encoded = cv2.imencode(".jpg", img, encode_param)
    socket.send(img_encoded.tobytes())


def send_depth_img(socket, depth_img):
    depth_img = (depth_img * 1000).astype(np.uint16)
    encode_param = [
        int(cv2.IMWRITE_PNG_COMPRESSION),
        3,
    ]
    _, depth_img_encoded = cv2.imencode(".png", depth_img, encode_param)
    socket.send(depth_img_encoded.tobytes())


class DynamemCamera:
    def __init__(self, robot, which="head"):
        self.robot = robot
        self.which = which

        # Camera intrinsics
        if self.which == "head":
            intrinsics = self.robot.get_camera_K()
            self.fy = intrinsics[0, 0]
            self.fx = intrinsics[1, 1]
            self.cy = intrinsics[0, 2]
            self.cx = intrinsics[1, 2]

            self.ix, self.iy = None, None
        else:
            servo_obs = self.robot.get_servo_observation()
            self.ee_camera_K = servo_obs.ee_camera_K
            self.fx = self.ee_camera_K[0, 0]
            self.fy = self.ee_camera_K[1, 1]
            self.cx = self.ee_camera_K[0, 2]
            self.cy = self.ee_camera_K[1, 2]
            self.ee_camera_pose = servo_obs.ee_camera_pose
            self.ee_depth_scaling = servo_obs.ee_depth_scaling
            self.joint_state = self.robot.get_joint_positions()

    def capture_image(self):
        if self.which == "head":
            self.rgb_image, self.depth_image, self.points = self.robot.get_images(compute_xyz=True)
            self.rgb_image = np.rot90(self.rgb_image, k=1)[:, :, [2, 1, 0]]
            self.depth_image = np.rot90(self.depth_image, k=1)
            self.points = np.rot90(self.points, k=1)
            self.rgb_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB)
            return self.rgb_image, self.depth_image, self.points
        else:
            rgb, depth = self.robot._servo.ee_rgb, self.robot._servo.ee_depth
            self.ee_rgb_image = rgb; self.ee_depth_image = depth
            self.ee_rgb_image = np.rot90(self.ee_rgb_image, k=1)[:, :, [2, 1, 0]]
            self.ee_depth_image = np.rot90(self.ee_depth_image, k=1)
            self.ee_rgb_image = cv2.cvtColor(self.ee_rgb_image, cv2.COLOR_BGR2RGB)
            return self.ee_rgb_image, self.ee_depth_image, None


class ImagePublisher:
    def __init__(self, robot, socket, which="head"):
        self.camera = DynamemCamera(robot, which)
        self.socket = socket
    
    def save_dict(self, return_dict=True):
        image, depth, points = self.camera.capture_image()
        rotated_image = np.rot90(image, k=-1)
        rotated_depth = np.rot90(depth, k=-1)
        
        if self.camera.which == "head":
            rotated_point = np.rot90(points, k=-1)
            result_dict = {
                "image": rotated_image,
                "colors": rotated_image,
                "depth": rotated_depth,
                "fx": self.camera.fx,
                "fy": self.camera.fy,
                "cx": self.camera.cx,
                "cy": self.camera.cy,
            }
        else:
            result_dict = {
                "image": rotated_image,
                "colors": rotated_image,
                "depth": rotated_depth,
                "fx": self.camera.fx,
                "fy": self.camera.fy,
                "cx": self.camera.cx,
                "cy": self.camera.cy,
                "pose": self.camera.ee_camera_pose,
                "joint_state": self.camera.joint_state,
            }
        if return_dict:
            return result_dict
        else:
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"ee_image_{current_time}.pkl"
            with open(file_name, "wb") as f:
                pickle.dump(result_dict, f)
            logger.info(f"EE image saved as {file_name}")
            return file_name

    def publish_image(self, text, mode, head_tilt=-1):
        image, depth, points = self.camera.capture_image()

        rotated_image = np.rot90(image, k=-1)
        rotated_depth = np.rot90(depth, k=-1)

        if self.camera.which == "head":
            rotated_point = np.rot90(points, k=-1)

            self.socket.send_string("head")
            print(f"Server: {self.socket.recv_string()}") # Camera type received

            send_rgb_img(self.socket, rotated_image)
            print(f"Server: {self.socket.recv_string()}") # RGB received
            send_depth_img(self.socket, rotated_depth)
            print(f"Server: {self.socket.recv_string()}") # Depth received
            send_array(
                self.socket,
                np.array(
                    [
                        self.camera.fy,
                        self.camera.fx,
                        self.camera.cy,
                        self.camera.cx,
                        int(head_tilt * 100),
                    ]
                ),
            )
            print(f"Server: {self.socket.recv_string()}")
            self.socket.send_string(text)
            print(f"Server: {self.socket.recv_string()}")
            self.socket.send_string(mode)
            print(f"Server: {self.socket.recv_string()}")

            self.socket.send_string("Waiting for gripper pose/ base and head trans from workstation")
            translation = recv_array(self.socket)
            self.socket.send_string("translation received by robot")
            rotation = recv_array(self.socket)
            self.socket.send_string("rotation received by robot")
            add_data = recv_array(self.socket)
            self.socket.send_string(f"Additional data received robot")

            depth = add_data[0]
            width = add_data[1]
            retry = add_data[2]
            print(f"Server: {self.socket.recv_string()}")
            return translation, rotation, depth, width, retry
        else:
            self.socket.send_string("ee")
            print(f"Server: {self.socket.recv_string()}")

            send_rgb_img(self.socket, rotated_image)
            print(f"Server: {self.socket.recv_string()}")
            send_depth_img(self.socket, rotated_depth)
            print(f"Server: {self.socket.recv_string()}")
            send_array(
                self.socket,
                np.array(
                    [
                        self.camera.fx,
                        self.camera.fy,
                        self.camera.cx,
                        self.camera.cy,
                    ]
                ),
            )
            print(f"Server: {self.socket.recv_string()}")

            self.socket.send_string(text)
            print(f"Server: {self.socket.recv_string()}")
            self.socket.send_string(mode)
            print(f"Server: {self.socket.recv_string()}")

            self.socket.send_string("Waiting for rotation / translation with respect to Anygrasp frame")
            translation = recv_array(self.socket)
            self.socket.send_string("translation received by robot")
            rotation = recv_array(self.socket)
            self.socket.send_string("rotation received by robot")
            add_data = recv_array(self.socket)
            self.socket.send_string(f"Additional data received robot")
            depth = add_data[0]
            width = add_data[1]
            retry = int(add_data[2])
            print(f"Server: {self.socket.recv_string()}")

            self.socket.send_string("Waiting for ee best pose image")
            ee_best_pose_image = recv_rgb_img(self.socket)

            return translation, rotation, depth, width, retry, ee_best_pose_image