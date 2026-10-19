import numpy as np
import torch
import os
import queue
import threading
from typing import Dict, List, Callable
from typing import TYPE_CHECKING
from utils import show_maps
from logging_utils import get_logger
from scipy import ndimage as ndi
from scipy.ndimage import binary_dilation, binary_erosion
from datetime import datetime

from action.utils.localization import (
    localize_single_object_last_observations,
    localize_single_object_with_mllm,
    find_alignment_over_model_last_observations,
)
from action.utils.head_control import look_at_target

logger = get_logger(__name__)

if TYPE_CHECKING:
    from deliberate_reasoning import DRM
    from instance_reasoning import IRM
    from stretch.agent.robot_agent_dynamem import RobotAgent
    from stretch.mapping.voxel import SparseVoxelMapDynamem as SparseVoxelMap
    from stretch.mapping.voxel.voxel_map_dynamem import SparseVoxelMapNavigationSpace

class BaseAction:
    def __init__(self, drm: "DRM", irm: "IRM", args: Dict, binder_config: Dict):
        self.drm = drm
        self.irm = irm
        self.args = args
        if args is not None:
            self.enable_irm = args["enable_irm"]
        else:
            self.enable_irm = False
        self.binder_config = binder_config
        self.robot = drm.robot
        self.agent: RobotAgent = drm.agent
        self.voxel_map: "SparseVoxelMap" = self.agent.voxel_map
        self.space: "SparseVoxelMapNavigationSpace" = self.agent.space
        self._stop_event_processing_flag = False
        self._event_processing_thread = None


    def run(self):
        self.before_action_task()
        if self.enable_irm:
            self._start_event_processing()
        try:
            result = self.execute()
        finally:
            if self.enable_irm:
                self._stop_event_processing()
                self.irm.request_stop_all_tasks_and_wait()

        self.update_drm_history(result)
        return result

    def _start_event_processing(self):
        self._stop_event_processing_flag = False
        self._event_processing_thread = threading.Thread(
            target=self._event_processing_loop,
            daemon=True,
        )
        self._event_processing_thread.start()

    def _event_processing_loop(self):
        while not self._stop_event_processing_flag:
            try:
                event = self.irm.event_queue.get(timeout=0.1)
                self.handle_irm_event(event)
            except queue.Empty:
                continue

    def _stop_event_processing(self):
        self._stop_event_processing_flag = True
        if self._event_processing_thread is not None:
            self._event_processing_thread.join(timeout=1.0)
    
    def handle_irm_event(self, event):
        pass
    
    def predict_irm_tasks(self):
        return None

    def before_action_task(self):
        if not self.enable_irm:
            return
        irm_tasks = self.predict_irm_tasks()
        if irm_tasks is not None:
            self.irm.set_tasks(irm_tasks)

    def execute(self):
        raise NotImplementedError

    def update_drm_history(self, result=None):
        pass

    def after_action_task(self):
        pass

    def update(self):
        obs = self.robot.get_observation()
        rgb, depth, K, camera_pose = obs.rgb, obs.depth, obs.camera_K, obs.camera_pose
        self.voxel_map.process_rgbd_images(rgb, depth, K, camera_pose, rerun_visualizer=self.agent.rerun_visualizer)
        if self.voxel_map.voxel_pcd._points is not None:
            self.agent.rerun_visualizer.update_voxel_map(space=self.agent.space) # Update voxel_pcd
        if self.voxel_map.semantic_memory._points is not None:
            self.agent.rerun_visualizer.log_custom_pointcloud( # Update semantic_memory
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

    def look_around_and_update(self, update_memory: bool = True, target_objects: List[str] = None, save_dir: str = None):
        self.robot.look_front()
        self.look_around()
        self.robot.look_front()
        self.robot.switch_to_navigation_mode()

        if target_objects is None:
            target_objects = self.drm.memory['target_objects']
        if update_memory:
            found_new_objects, found_objects, object_moved_or_removed, moved_or_removed_objects = self.refresh_object_memory(last_observations_count=4, save_dir=save_dir)
        else:
            found_new_objects, found_objects, object_moved_or_removed, moved_or_removed_objects = False, [], False, []

        return found_new_objects, found_objects, object_moved_or_removed, moved_or_removed_objects
    
    def rotate_90_degrees(self, reverse: bool = False):
        self.robot.switch_to_navigation_mode()
        xyt = self.robot.get_base_pose()
        xyt[2] = xyt[2] + np.pi / 2 if not reverse else xyt[2] - np.pi / 2
        self.robot.move_base_to(xyt, blocking=True)
        
    def move_to_reachable_nearest_point(self, target_point: np.ndarray, visualize: bool = True,
                                        interrupt_condition: Callable[[], bool] = None, log_dir: str = None, target_object: str = None):
        finished = False
        iter = 0
        object_moved_or_removed = False
        current_obstacle_margin = self.binder_config['navigation']['obstacle_margin']
        current_close_threshold = self.binder_config['navigation']['close_threshold']
        while not finished:
            iter += 1
            if interrupt_condition is not None and interrupt_condition():
                break
            elif object_moved_or_removed:
                break
            start_pose = self.robot.get_base_pose()
            self.agent.rerun_visualizer.log_custom_pointcloud("world/target_point", [target_point[0], target_point[1], 1.5], torch.Tensor([1, 0, 0]), 0.1)

            nearest_point, is_close_to_target, distance_to_target = self.agent.space.sample_navigation(
                start=start_pose,
                planner=self.agent.planner,
                point=target_point,
                edge_margin=self.binder_config['navigation']['edge_margin'],
                rerun_visualizer=self.agent.rerun_visualizer,
                is_exploration=False,
                obstacle_margin=current_obstacle_margin,
                close_threshold=current_close_threshold
            )
            if is_close_to_target:
                self.agent.rerun_visualizer.log_text("robot_monologue", f"### ✅ The target point is reachable in a single step. Distance to the target point: {distance_to_target:.2f}m")
                finished = True
            else:
                distance_robot_to_nearest = np.linalg.norm(nearest_point[:2] - start_pose[:2])
                if distance_robot_to_nearest < self.binder_config['navigation']['distance_robot_to_nearest_threshold']:
                    logger.notice(f"The nearest point ({nearest_point}) is too close to the robot (distance: {distance_robot_to_nearest:.2f}m). Do not move the robot and just update the map only.")
                    logger.notice(f"Current pose: {start_pose}, Target point: {target_point}, Nearest point: {nearest_point}")
                    logger.notice(f"Distance (nearest point ~ target point): {distance_to_target:.2f}m")
                    if distance_to_target < 1.0:
                        current_obstacle_margin = max(0, current_obstacle_margin - 1)
                        current_close_threshold += 0.15
                    self.agent.rerun_visualizer.log_text("robot_monologue", f"### ❌ The nearest point is too close to the robot (distance: {distance_robot_to_nearest:.2f}m). Do not move the robot and just update the map only.")
                    current_xyt = self.robot.get_base_pose()
                    theta = self.agent.space.compute_theta(current_xyt[0], current_xyt[1], target_point[0], target_point[1])
                    self.robot.move_base_to(np.array([current_xyt[0], current_xyt[1], theta]), blocking=True)
                    if not (interrupt_condition is not None and interrupt_condition()):
                        _, _, object_moved_or_removed, _ = self.look_around_and_update()
                    
                    if target_object in self.drm.memory["object_positions"]:
                        target_point = self.drm.memory["object_positions"][target_object]["xyz"]
                    continue
                else:
                    logger.notice(f"The target point ({target_point}) is not reachable in a single step. Should expand the map once more. Distance to the target point: {distance_to_target:.2f}m")
                    logger.notice(f"Current pose: {start_pose}, Target point: {target_point}, Nearest point: {nearest_point}")
                    logger.notice(f"Distance (nearest point ~ target point): {distance_to_target:.2f}m")
                    theta = self.agent.space.compute_theta(nearest_point[0], nearest_point[1], target_point[0], target_point[1])
                    nearest_point[2] = theta
                    self.agent.rerun_visualizer.log_text("robot_monologue", f"### ❌ The target point is not reachable in a single step. Should expand the map once more. Distance to the target point: {distance_to_target:.2f}m")
                    self.agent.rerun_visualizer.log_custom_pointcloud("world/navigation_endpoint", [nearest_point[0], nearest_point[1], 1.5], torch.Tensor([1, 0, 0]), 0.1)

            
            if visualize:
                self.agent.rerun_visualizer.log_custom_pointcloud("world/navigation_endpoint", [nearest_point[0], nearest_point[1], 1.5], torch.Tensor([0, 1, 0]), 0.1)
            
            res = self.agent.planner.plan(start_pose, nearest_point)
            if res is None:
                raise ValueError(f"Could not find any trajectory to the target point: {target_point}")
            waypoints = [pt.state for pt in res.trajectory]
            if visualize:
                self.agent.visualize_points(waypoints, torch.Tensor([0, 0, 1]), 0.05, "world/waypoint")
            
            traj = self.agent.planner.clean_path_for_xy(waypoints)
            if visualize:
                self.agent.visualize_points(traj, torch.Tensor([1, 1, 0]), 0.12, "world/cleaned_path")
            
            self.robot.execute_trajectory(traj, blocking=True, interrupt_condition=interrupt_condition)

            if not finished and not (interrupt_condition is not None and interrupt_condition()):
                pass
        return finished, object_moved_or_removed


    
    def check_objects_in_roi(self, last_observations_count: int = 4, target_objects: List[str] = None) -> bool:
        points, _, _, _ = self.agent.voxel_map.semantic_memory.get_pointcloud()
        obs_ids = self.agent.voxel_map.semantic_memory._obs_counts.cpu()
        observation_ids = sorted(obs_ids.unique().numpy().tolist())
        last_observation_ids = observation_ids[-last_observations_count:]
        mask = torch.isin(obs_ids, torch.tensor(last_observation_ids))
        points = points[mask]

        binary_mask = torch.zeros((1024, 1024), dtype=torch.bool)
        for point in points:
            x, y = self.agent.planner.to_pt(point)
            binary_mask[y, x] = True
        str_el = ndi.generate_binary_structure(2, 1)
        binary_mask = binary_dilation(binary_mask, structure=str_el, iterations=1)
        binary_mask = binary_erosion(binary_mask, structure=str_el, iterations=1)

        lbl, n = ndi.label(binary_mask)
        if n > 1:
            counts = ndi.sum(binary_mask, lbl, index=range(1, n+1))
            keep = (lbl == counts.argmax() + 1)
            binary_mask = keep
        binary_mask = binary_erosion(binary_mask, structure=str_el, iterations=1)

        object_positions = self.drm.memory["object_positions"]
        if target_objects is not None:
            object_positions = {obj: object_positions[obj] for obj in target_objects}

        img = np.zeros((1024, 1024, 3), dtype=np.float32)
        img[binary_mask] = np.array([0, 1, 0])

        results = {}
        for obj in object_positions:
            x, y = self.agent.planner.to_pt(object_positions[obj]["xyz"])
            img[y, x] = np.array([1, 0, 0])
            results[obj] = binary_mask[y, x]
        
        save_path = f"{self.agent.log}/localization_results"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        show_maps([img], titles=["Object in ROI"], zoom=11.0, save_path=os.path.join(save_path, f"{current_time}_object_in_roi.png"))
        return results
        
    
    def refresh_object_memory(self, last_observations_count: int = 4, remove_objects: bool = False, save_dir: str = None):
        found_new_objects = False; found_objects = []; object_moved_or_removed = False; moved_or_removed_object = []
        if save_dir is not None:
            save_path = f"{save_dir}/localization_results"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        target_objects_already_localized = set(self.drm.memory["object_positions"].keys())
        target_objects_not_localized = set(self.drm.memory['target_objects']) - target_objects_already_localized
        logger.info(f"Localizing the already localized objects: {target_objects_already_localized}")
        in_boundary = self.check_objects_in_roi(last_observations_count=last_observations_count)
        for obj in target_objects_already_localized:
            if in_boundary[obj]:
                logger.notice(f"❗ {obj} is in the boundary of the last {last_observations_count} observations.")
                logger.notice(f"Checking whether it still exists...")
                save_path = os.path.join(save_dir, f"{obj}.png") if save_dir is not None else None
                target_point, image_id, score = self.localize_single_object_last_observations(obj, last_observations_count=last_observations_count, save_path=save_path)
                if target_point is None:
                    if remove_objects:
                        logger.notice(f"❌ {obj} is not in the boundary of the last {last_observations_count} observations. Remove it from the memory.")
                        self.drm.memory["object_positions"].pop(obj)
                        self.agent.rerun_visualizer.clear_identity(f"world/localized_objects/{obj.replace(' ', '_')}")
                        object_moved_or_removed = True; moved_or_removed_object.append(obj)
                else:
                    logger.notice(f"✅ {obj} is still in the boundary of the last {last_observations_count} observations. Update the memory.")
                    logger.notice(f"Before: {self.drm.memory['object_positions'][obj]['xyz']}, After: {target_point}")
                    if np.linalg.norm(self.drm.memory['object_positions'][obj]['xyz'] - target_point) > 2:
                        object_moved_or_removed = True; moved_or_removed_object.append(obj)
                    self.drm.memory["object_positions"][obj] = {
                        "xyz": target_point,
                        "image_id": image_id,
                        "score": score,
                    }
                        
            else:
                logger.info(f"{obj} is not in the boundary of the last {last_observations_count} observations. Skip updating the memory.")
                continue


        logger.info(f"Localizing the NOT localized objects: {target_objects_not_localized}")
        for target_object in target_objects_not_localized:
            logger.info(f"Target object: {target_object}")
            save_path = os.path.join(save_dir, f"{target_object}.png") if save_dir is not None else None
            target_point, image_id, score = self.localize_single_object_last_observations(target_object, last_observations_count=last_observations_count, save_path=save_path)
            if target_point is None:
                logger.info(f"Failed to localize {target_object} with last {last_observations_count} observations")
            else:
                logger.info(f"Successfully localized {target_object} with last {last_observations_count} observations")
                found_new_objects = True; found_objects.append(target_object)
                self.drm.memory["object_positions"][target_object] = {
                    "xyz": target_point,
                    "image_id": image_id,
                    "score": score,
                }

        return found_new_objects, found_objects, object_moved_or_removed, moved_or_removed_object
    
    
    def find_alignment_over_model_last_observations(self, queries: str, last_observations_count: int = 1, save_path: str = None):
        return find_alignment_over_model_last_observations(self.agent, queries, last_observations_count, save_path)

    def localize_single_object_last_observations(self, target_object: str, last_observations_count: int = 1, save_path: str = None, gpt_verification: bool = True):
        return localize_single_object_last_observations(
            self.agent, target_object, last_observations_count, save_path, gpt_verification, self.binder_config['models']['drm']['model_name']
        )

    def localize_single_object_with_mllm(self, target_object: str, return_shortest: bool = True, visualize_rerun: bool = True):
        return localize_single_object_with_mllm(self.agent, target_object, return_shortest, visualize_rerun)

    def look_at_target(self, target_xyz, rerun_str="world/irm_localized_point", log_dir=None):
        return look_at_target(self.agent, target_xyz, rerun_str, log_dir)
