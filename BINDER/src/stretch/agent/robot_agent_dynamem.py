# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.


# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from threading import Lock
from typing import Any, Dict, List, Optional, Union
from threading import Lock
from typing import TYPE_CHECKING

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import zmq

from logging_utils import get_logger
logger = get_logger(__name__)

if TYPE_CHECKING:
    from stretch.visualization.rerun import RerunVisualizer

from stretch.agent.manipulation.dynamem_manipulation.dynamem_manipulation import (
    DynamemManipulationWrapper as ManipulationWrapper,
)
from stretch.agent.manipulation.dynamem_manipulation.grasper_utils import (
    capture_and_process_image,
    pickup,
)
from stretch.agent.robot_agent import RobotAgent as RobotAgentBase
from stretch.audio.text_to_speech import get_text_to_speech
from stretch.core.interfaces import Observations
from stretch.core.parameters import Parameters
from stretch.core.robot import AbstractGraspClient, AbstractRobotClient
from stretch.mapping.instance import Instance
from stretch.mapping.voxel import SparseVoxelMapDynamem as SparseVoxelMap
from stretch.mapping.voxel import (
    SparseVoxelMapNavigationSpaceDynamem as SparseVoxelMapNavigationSpace,
)
from stretch.mapping.voxel import SparseVoxelMapProxy
from stretch.motion.algo.a_star import AStar
from stretch.perception.detection.owl import OwlPerception
from stretch.perception.encoders import MaskSiglipEncoder
from stretch.perception.wrapper import OvmmPerception
from instance_reasoning import IRM

INIT_LIFT_POS = 0.45
INIT_WRIST_PITCH = -1.57
INIT_ARM_POS = 0
INIT_WRIST_ROLL = 0
INIT_WRIST_YAW = 0
INIT_HEAD_PAN = -1.57
INIT_HEAD_TILT = -0.65

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

class RobotAgent(RobotAgentBase):
    def __init__(
        self,
        robot: AbstractRobotClient,
        parameters: Union[Parameters, Dict[str, Any]],
        semantic_sensor: Optional[OvmmPerception] = None,
        grasp_client: Optional[AbstractGraspClient] = None,
        voxel_map: Optional[SparseVoxelMap] = None,
        debug_instances: bool = True,
        show_instances_detected: bool = False,
        use_instance_memory: bool = False,
        realtime_updates: bool = False,
        obs_sub_port: int = 4450,
        re: int = 3,
        manip_port: int = 5559,
        log: Optional[str] = None,
        server_ip: Optional[str] = "127.0.0.1",
        vlm1_ip: Optional[str] = "localhost",
        vlm1_port: int = 1207,
        mllm: bool = False,
        manipulation_only: bool = False,
        enable_irm: bool = False,
        binder_config: Dict[str, Any] = None,
    ):
        self.binder_config = binder_config
        self.reset_object_plans()
        if isinstance(parameters, Dict):
            self.parameters = Parameters(**parameters)
        elif isinstance(parameters, Parameters):
            self.parameters = parameters
        else:
            raise RuntimeError(f"parameters of unsupported type: {type(parameters)}")
        self.robot = robot
        self.grasp_client = grasp_client
        self.debug_instances = debug_instances
        self.show_instances_detected = show_instances_detected

        self.vlm1_ip = vlm1_ip
        self.vlm1_port = vlm1_port

        self.semantic_sensor = semantic_sensor
        self.pos_err_threshold = parameters["trajectory_pos_err_threshold"]
        self.rot_err_threshold = parameters["trajectory_rot_err_threshold"]
        self.current_state = "WAITING"

        self.rerun_visualizer: RerunVisualizer = self.robot._rerun
        self.setup_custom_blueprint()

        self.mllm = mllm
        self.manipulation_only = manipulation_only
        self.owl_sam_detector = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if not os.path.exists("dynamem_log"):
            os.makedirs("dynamem_log")

        if log is None:
            raise ValueError("log is None")
        else:
            self.log = log
            logger.notice(f'-' * 100)
            logger.notice(f'-' * 100)
            logger.notice(f"Logging to {self.log}/run.log")
            logger.notice(f'-' * 100)
            logger.notice(f'-' * 100)

        self.create_obstacle_map(parameters)

        self._voxel_map_lock = Lock()
        self.voxel_map_proxy = SparseVoxelMapProxy(self.voxel_map, self._voxel_map_lock)

        self.obs_history: List[Observations] = []

        self.guarantee_instance_is_reachable = self.parameters.guarantee_instance_is_reachable
        self.use_scene_graph = self.parameters["use_scene_graph"]
        self.tts = get_text_to_speech(self.parameters["tts_engine"])
        self._use_instance_memory = use_instance_memory
        self._realtime_updates = realtime_updates

        self._sweep_head_on_update = parameters["agent"]["sweep_head_on_update"]

        self.current_receptacle: Instance = None
        self.current_object: Instance = None
        self.target_object = None
        self.target_receptacle = None
        self._is_match_threshold = parameters.get("encoder_args/feature_match_threshold", 0.05)
        self._grasp_match_threshold = parameters.get(
            "encoder_args/grasp_feature_match_threshold", 0.05
        )

        self._default_expand_frontier_size = parameters["motion_planner"]["frontier"][
            "default_expand_frontier_size"
        ]
        self._frontier_min_dist = parameters["motion_planner"]["frontier"]["min_dist"]
        self._frontier_step_dist = parameters["motion_planner"]["frontier"]["step_dist"]
        self._manipulation_radius = parameters["motion_planner"]["goals"]["manipulation_radius"]
        self._voxel_size = parameters["voxel_size"]

        context = zmq.Context()
        self.manip_socket = context.socket(zmq.REQ)
        self.manip_socket.connect("tcp://" + server_ip + ":" + str(manip_port))

        if re == 1 or re == 2:
            stretch_gripper_max = 0.3
            end_link = "link_straight_gripper"
        else:
            stretch_gripper_max = 0.64
            end_link = "link_gripper_s3_body"
        self.transform_node = end_link
        self.manip_wrapper = ManipulationWrapper(
            self.robot, stretch_gripper_max=stretch_gripper_max, end_link=end_link
        )
        self.robot.move_to_nav_posture()

        self.reset_object_plans()

        self.re = re

        self.scene_graph = None
        self._previous_goal = None

        self._running = True

        self._start_threads()

        self.enable_irm = enable_irm

        self.irm = IRM(
            robot=self.robot,
            vlm1_ip=self.vlm1_ip,
            vlm1_port=self.vlm1_port,
            fps=8,
            num_frames_for_description=32,
            num_frames_for_irm=32,
            enable_irm=self.enable_irm,
            binder_config=self.binder_config,
        )

        if self.enable_irm:
            self.irm.set_image_buffer_update_loop(True)
            self.irm.set_ee_image_buffer_update_loop(True)
    
    def force_stop(self):
        logger.info("Force stop!!!")
        self.robot.send_message({"force_stop": "true"})


    def create_obstacle_map(self, parameters):
        if self.manipulation_only:
            self.encoder = None
        else:
            self.encoder = MaskSiglipEncoder(device=self.device, version="so400m")
        if self.manipulation_only:
            self.detection_model = None
            semantic_memory_resolution = 0.1
            image_shape = (360, 270)
        elif self.mllm:
            self.detection_model = OwlPerception(
                version="owlv2-B-p16", device=self.device, confidence_threshold=0.01
            )
            semantic_memory_resolution = 0.1
            image_shape = (360, 270)
        else:
            self.detection_model = OwlPerception(
                version="owlv2-L-p14-ensemble", device=self.device, confidence_threshold=0.2
            )
            semantic_memory_resolution = 0.05
            image_shape = (480, 360)
        self.voxel_map = SparseVoxelMap(
            resolution=parameters["voxel_size"],
            semantic_memory_resolution=semantic_memory_resolution,
            local_radius=parameters["local_radius"],
            obs_min_height=parameters["obs_min_height"],
            obs_max_height=parameters["obs_max_height"],
            obs_min_density=parameters["obs_min_density"],
            grid_resolution=0.1,
            min_depth=parameters["min_depth"],
            max_depth=parameters["max_depth"],
            pad_obstacles=parameters["pad_obstacles"],
            add_local_radius_points=parameters.get("add_local_radius_points", default=True),
            remove_visited_from_obstacles=parameters.get(
                "remove_visited_from_obstacles", default=False
            ),
            smooth_kernel_size=parameters.get("filters/smooth_kernel_size", -1),
            use_median_filter=parameters.get("filters/use_median_filter", False),
            median_filter_size=parameters.get("filters/median_filter_size", 5),
            median_filter_max_error=parameters.get("filters/median_filter_max_error", 0.01),
            use_derivative_filter=parameters.get("filters/use_derivative_filter", False),
            derivative_filter_threshold=parameters.get("filters/derivative_filter_threshold", 0.5),
            detection=self.detection_model,
            encoder=self.encoder,
            image_shape=image_shape,
            log=self.log,
            mllm=self.mllm,
        )
        self.space = SparseVoxelMapNavigationSpace(
            self.voxel_map,
            rotation_step_size=parameters.get("motion_planner/rotation_step_size", 0.2),
            dilate_frontier_size=parameters.get("motion_planner/frontier/dilate_frontier_size", 2),
            dilate_obstacle_size=parameters.get("motion_planner/frontier/dilate_obstacle_size", 0),
        )
        self.planner = AStar(self.space)

    def setup_custom_blueprint(self):
        main = rrb.Horizontal(
            rrb.Spatial3DView(name="3D View", origin="world"),
            rrb.Vertical(
                rrb.TextDocumentView(name="text", origin="robot_monologue"),
                rrb.TextDocumentView(name="IRM log", origin="instance_reasoning_log"),
                rrb.Spatial2DView(name="image", origin="/observation_similar_to_text"),
            ),
            rrb.Vertical(
                rrb.Spatial2DView(name="head_rgb", origin="/world/head_camera"),
                rrb.Spatial2DView(name="ee_rgb", origin="/world/ee_camera"),
            ),
            column_shares=[2, 1, 1],
        )
        my_blueprint = rrb.Blueprint(
            rrb.Vertical(main, rrb.TimePanel(state=True)),
            collapse_panels=True,
        )
        rr.send_blueprint(my_blueprint)

    def compute_blur_metric(self, image):
        """
        Computes a blurriness metric for an image tensor using gradient magnitudes.

        Parameters:
        - image (torch.Tensor): The input image tensor. Shape is [H, W, C].

        Returns:
        - blur_metric (float): The computed blurriness metric.
        """

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Compute gradients using the Sobel operator
        Gx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        Gy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

        # Compute gradient magnitude
        G = cv2.magnitude(Gx, Gy)

        # Compute the mean of gradient magnitudes
        blur_metric = G.mean()

        return blur_metric

    def update(self):
        obs = self.robot.get_observation()
        rgb, depth, K, camera_pose = obs.rgb, obs.depth, obs.camera_K, obs.camera_pose
        self.voxel_map.process_rgbd_images(rgb, depth, K, camera_pose, rerun_visualizer=self.rerun_visualizer)
        if self.voxel_map.voxel_pcd._points is not None:
            self.rerun_visualizer.update_voxel_map(space=self.space) # Update voxel_pcd
        if self.voxel_map.semantic_memory._points is not None:
            self.rerun_visualizer.log_custom_pointcloud( # Update semantic_memory
                "world/semantic_memory/pointcloud",
                self.voxel_map.semantic_memory._points.detach().cpu(),
                self.voxel_map.semantic_memory._rgb.detach().cpu() / 255.0,
                0.03,
            )

    def look_around(self):
        logger.info(f"{'*' * 10} Look around to check {'*' * 10}")
        for pan in [0.6, -0.2, -1.0, -1.8]:
            for tilt in [-0.7]:
                self.robot.head_to(pan, tilt, blocking=True)
                self.update()

    def rotate_in_place(self):
        logger.info(f"{'*' * 10} Rotate in place {'*' * 10}")
        xyt = self.robot.get_base_pose()
        self.robot.head_to(head_pan=0, head_tilt=-0.6, blocking=True)
        for i in range(8):
            xyt[2] += 2 * np.pi / 8
            self.robot.move_base_to(xyt, blocking=True)
            if not self._realtime_updates:
                self.update()

    def visualize_points(self, points, color, radius, identity):
        self.rerun_visualizer.clear_identity(identity)
        for i, point in enumerate(points):
            point_str = f"{identity}/{i}"
            self.rerun_visualizer.log_custom_pointcloud(point_str, [point[0], point[1], 1.5], color, radius)

    def get_voxel_map(self):
        return self.voxel_map

    def manipulate(
        self,
        text,
        init_tilt=INIT_HEAD_TILT,
        base_node="camera_depth_optical_frame",
        skip_confirmation: bool = False,
        ablation: bool = False,
    ):
        self.robot.switch_to_manipulation_mode()
        self.robot.look_at_ee()

        gripper_pos = 1

        self.manip_wrapper.move_to_position(
            arm_pos=INIT_ARM_POS,
            head_pan=INIT_HEAD_PAN,
            head_tilt=init_tilt,
            gripper_pos=gripper_pos,
            lift_pos=INIT_LIFT_POS,
            wrist_pitch=INIT_WRIST_PITCH,
            wrist_roll=INIT_WRIST_ROLL,
            wrist_yaw=INIT_WRIST_YAW,
        )

        rotation, translation, depth, width, _ = capture_and_process_image(
            mode="pick",
            obj=text,
            socket=self.manip_socket,
            hello_robot=self.manip_wrapper,
            ok_robot=True
        )

        if rotation is None:
            return False

        if width < 0.05 and self.re == 3:
            gripper_width = 0.45
        elif width < 0.075 and self.re == 3:
            gripper_width = 0.6
        else:
            gripper_width = 1

        if skip_confirmation or input("Do you want to do this manipulation? Y or N ") != "N":
            pickup(
                self.manip_wrapper,
                rotation,
                translation,
                base_node,
                self.transform_node,
                gripper_depth=depth,
                gripper_width=gripper_width,
                ablation=ablation,
            )

        self.manip_wrapper.move_to_position(
            base_trans=-self.manip_wrapper.robot.get_six_joints()[0]
        )

        return True
