import os
import numpy as np
import torch
from typing import TYPE_CHECKING
from overrides import overrides
from PIL import Image

from action.base import BaseAction
from utils import BOLD, RED, GREEN, RESET, get_grid_image
from irm_task.explore_multiple_targets import ExploreMultipleTargetsTask
from irm_task.base import ExecutionMode
from prompts import get_default_guidance
from logging_utils import get_logger

from action.utils.exploration_sampling import (
    sample_k_points_from_heuristics,
    find_optimal_look_order,
    grid_points_to_xyt,
)
from action.utils.exploration_llm import (
    run_llm,
)
from action.utils.exploration_visualization import (
    generate_visual_prompt,
    save_heuristics_visualization,
)

logger = get_logger(__name__)

if TYPE_CHECKING:
    from stretch.motion.algo import AStar
    from stretch.mapping.voxel.voxel_map_dynamem import SparseVoxelMapNavigationSpace


class ExploreEnvironmentAction(BaseAction):
    def __init__(self, drm, irm, args, binder_config, random=False):
        super().__init__(drm, irm, args, binder_config)
        self.num_max_tries = 5
        self.num_tries = 0
        self.planner: AStar = self.agent.planner
        self.space: SparseVoxelMapNavigationSpace = self.agent.space
        self.previous_trajectories = []
        self.should_interrupt = False
        self.interrupt_reason = None
        self.random = random
        # Safely get model name with fallback
        self.model = binder_config.get('models', {}).get('drm', {}).get('model_name', 'gpt-4o')
    def handle_irm_event(self, event):
        # Act on the IRM execution mode m_t (paper Eq. 3), not the event name.
        mode = event.get("mode")
        data = event.get("data", {})
        if mode == ExecutionMode.REPLAN:
            reason = data.get("reason", "IRM REPLAN")
            logger.notice(f"{BOLD}{GREEN}IRM REPLAN: {data.get('target_object')} - {reason}{RESET}")
            self.should_interrupt = True
            self.interrupt_reason = reason
        elif mode == ExecutionMode.CONTINUE and "target_point" in data:
            logger.info(f"{BOLD}{GREEN}Updating object position: {data['target_object']} at {data['target_point']}{RESET}")
            self.drm.memory["object_positions"][data["target_object"]] = {
                "xyz": data["target_point"],
                "image_id": data["image_idx"],
                "score": data["score"],
            }

    @overrides
    def predict_irm_tasks(self):
        seen_objects = list(self.drm.memory["object_positions"].keys())
        unseen_objects = list(set(self.drm.memory["target_objects"]) - set(seen_objects))
        current_target_object = self.args["target_object"]

        if len(seen_objects) == 0 and len(unseen_objects) == 0:
            logger.warning(f"{RED} No target objects to localize{RESET}")
            return None

        guidance = self.args.get("guidance") or get_default_guidance(
            "explore_environment", self.drm.memory["target_objects"]
        )
        irm_config = {
            "action": "explore_environment",
            "seen_objects": seen_objects,
            "unseen_objects": unseen_objects,
            "current_target_object": current_target_object,
            "guidance": guidance,
            "verbose": True,
            "activate_only_on_moving_base": True,
            "log_dir": self.log_dir,
        }
        rerun_output = f"### ❗ IRM will localize {unseen_objects} and {seen_objects}"
        self.agent.rerun_visualizer.log_text("instance_reasoning_log", rerun_output)

        localize_objects_task = ExploreMultipleTargetsTask(
            irm=self.irm,
            robot=self.irm.robot,
            agent=self.agent,
            get_drm_memory_fn=self.drm.get_drm_memory,
            vlm1_socket=self.irm.vlm1_socket,
            image_buffer=self.irm._image_buffer,
            _ee_image_buffer=self.irm._ee_image_buffer,
            fps=self.irm._fps,
            config=irm_config,
        )
        return [localize_objects_task]

    @overrides
    def run(self):
        success = False
        for i in range(self.num_max_tries):
            self.log_dir = f"{self.agent.log}/{self.drm.drm_call_count}_explore_environment/{i}"
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

            self.before_action_task()
            if self.enable_irm:
                self._start_event_processing()

            try:
                self.execute(k=3)
            finally:
                if self.enable_irm:
                    self._stop_event_processing()
                    self.irm.request_stop_all_tasks_and_wait(timeout=10.0)

            if self.should_interrupt:
                logger.notice(f"{BOLD}{GREEN}IRM found the target object {self.target_object} and stopped the trajectory.{RESET}")
                logger.notice(f"{BOLD}{GREEN}Updating the map...{RESET}")
                self.update()
                self.should_interrupt = False
                success = True
                break
            else:
                found_new_objects = self.after_action_task()
                if found_new_objects:
                    if self.target_object in self.drm.memory["object_positions"]:
                        logger.notice(f"{GREEN}Robot found the target object {self.target_object} after IRM interrupted{RESET}")
                        success = True
                        break
                    else:
                        logger.notice(f"{RED}Robot found other target object, but not {self.target_object}{RESET}")
                        continue

        # Update DRM history
        status = f"success - found target object - {self.target_object}" if success else f"failed - not found target object - {self.target_object}"
        self.drm.action_history.append({
            "action": "explore_environment",
            "arguments": {"target_object": self.args["target_object"]},
            "status": status,
            "result": None,
        })

    def should_interrupt_navigation(self):
        return self.should_interrupt

    @overrides
    def before_action_task(self):
        if self.enable_irm:
            irm_tasks = self.predict_irm_tasks()
            if irm_tasks is not None:
                self.irm.set_tasks(irm_tasks)
        self.target_object = self.args["target_object"]

    @overrides
    def after_action_task(self):
        if self.enable_irm:
            self.irm.unset_tasks(is_interrupted=False)

        found_new_objects, found_objects, _, _ = self.look_around_and_update(save_dir=self.log_dir)
        if found_new_objects:
            if self.target_object in found_objects:
                self.agent.rerun_visualizer.log_text("robot_monologue", f"### ✅ Target object - {self.target_object} is localized")
            else:
                self.agent.rerun_visualizer.log_text("robot_monologue", f"### ✅ Localized other objects - {found_objects} - but not target object - {self.target_object}")
        else:
            self.agent.rerun_visualizer.log_text("robot_monologue", f"### ❌ Not found any new objects")

        return found_new_objects

    @overrides
    def execute(self, k=None):
        if k is None:
            k = self.binder_config.get('exploration', {}).get('num_exploration_points', 3)
        self.robot.switch_to_manipulation_mode()
        start_pose = self.robot.get_base_pose()

        exploration_config = self.binder_config.get('exploration', {})
        sampling_result = sample_k_points_from_heuristics(
            space=self.space,
            planner=self.planner,
            xyt=start_pose,
            target_object=self.target_object,
            k=k,
            grid_to_xy_resolution=self.binder_config['exploration']['grid_to_xy_resolution'],
            min_distance_from_other_points=self.binder_config['exploration']['min_distance_from_other_points'],
            min_distance_from_robot=self.binder_config['exploration']['min_distance_from_robot'],
            alignment_smooth=self.binder_config['exploration']['alignment_smooth'],
            alignment_threshold=self.binder_config['exploration']['alignment_threshold'],
        )
        localized_k_points = grid_points_to_xyt(sampling_result["grid_points"], self.space.voxel_map)

        if len(localized_k_points) == 0:
            # No reachable frontier this iteration (e.g. area fully explored). Returning lets
            # run()'s retry loop record a "failed - not found" history entry instead of
            # crashing the whole DRM loop on localized_k_points[0].
            logger.warning(f"No exploration points found based on time heuristics")
            return

        xy_grid = np.array(self.planner.to_pt(start_pose[:2]))[::-1]
        result_img = save_heuristics_visualization(
            history_soft=sampling_result["history_soft"],
            time_heuristics=sampling_result["time_heuristics"],
            alignment_heuristics=sampling_result["alignment_heuristics"],
            total_heuristics=sampling_result["total_heuristics"],
            outside_frontier=sampling_result["outside_frontier"],
            grid_points=sampling_result["grid_points"],
            robot_grid_pos=xy_grid,
            save_path=f"{self.log_dir}/exploration_points.png",
        )
        self.agent.rerun_visualizer.log_custom_2d_image("/observation_similar_to_text",
            result_img
        )

        if len(localized_k_points) >= 2:
            localized_point = self._select_best_point_with_llm(localized_k_points, k)
        else:
            localized_point = localized_k_points[0]

        self.agent.rerun_visualizer.log_custom_pointcloud(
            "world/exploration_target_point",
            [localized_point[0], localized_point[1], 1.5],
            torch.Tensor([1, 0, 0]), 0.1
        )

        self._navigate_to_point(start_pose, localized_point)

    def _select_best_point_with_llm(self, localized_k_points, k):
        visual_prompt = generate_visual_prompt(
            space=self.space,
            planner=self.planner,
            candidate_points=localized_k_points,
            previous_trajectories=self.previous_trajectories,
            base_pose=self.robot.get_base_pose(),
            save_path=f"{self.log_dir}/visual_prompt.png",
        )
        self.agent.rerun_visualizer.log_text("robot_monologue", f"### ❗ Use visual prompt to choose the best exploration point for {self.target_object}")
        self.agent.rerun_visualizer.log_custom_2d_image("/observation_similar_to_text", visual_prompt)

        sequence = find_optimal_look_order(localized_k_points, self.robot.get_base_pose())
        logger.info(f"{GREEN}[ExploreEnvironmentAction-execute] Sequence: {sequence}{RESET}")

        rgb_images = [None] * len(sequence)
        for idx in sequence:
            self.agent.rerun_visualizer.log_text("robot_monologue", f"### ❗ Look at the exploration target point {idx + 1}")
            self.look_at_target(localized_k_points[idx], rerun_str=f"world/exploration_target_point")
            obs = self.robot.get_observation()
            rgb_images[idx] = obs.rgb

        pil_images = [Image.fromarray(img) for img in rgb_images]
        grid = get_grid_image(pil_images, cols=k)
        self.agent.rerun_visualizer.log_custom_2d_image("/observation_similar_to_text", grid)

        self.robot.look_front()
        self.agent.rerun_visualizer.log_text("robot_monologue", "### Waiting GPT to choose the best image")

        objects_of_interest = self._get_objects_of_interest()

        if self.random:
            best_image_index = np.random.randint(1, len(localized_k_points) + 1)
            reason = "Randomly selected the exploration point."
        else:
            best_image_index, reason = run_llm(
                visual_prompt=visual_prompt,
                rgb_images=rgb_images,
                objects_of_interest=objects_of_interest,
                save_dir=self.log_dir,
                model=self.model,
            )

        rerun_str = f"### ❗ Best image index: {best_image_index}\n### {reason}"
        self.agent.rerun_visualizer.log_text("robot_monologue", rerun_str)
        self.agent.rerun_visualizer.log_custom_2d_image("/observation_similar_to_text", Image.fromarray(rgb_images[best_image_index - 1]))

        return localized_k_points[best_image_index - 1]

    def _get_objects_of_interest(self):
        if self.args["target_object"] in self.drm.memory["object_receptacle_mapping"].keys():
            seen_objects = list(self.drm.memory["object_positions"].keys())
            unseen_objects = list(set(self.drm.memory["target_objects"]) - set(seen_objects))
            return [obj for obj in unseen_objects if obj in self.drm.memory["object_receptacle_mapping"].keys()]
        else:
            return [self.args["target_object"]]

    def _navigate_to_point(self, start_pose, localized_point):
        point, _, _ = self.space.sample_navigation(
            start=start_pose,
            planner=self.planner,
            point=localized_point,
            edge_margin=self.binder_config['navigation']['edge_margin'],
            rerun_visualizer=self.agent.rerun_visualizer,
            is_exploration=True,
            obstacle_margin=self.binder_config['navigation']['obstacle_margin'],
            close_threshold=self.binder_config['navigation']['close_threshold']
        )
        self.agent.rerun_visualizer.log_custom_pointcloud(
            "world/exploration_target_point_reachable",
            [point[0], point[1], 1.5],
            torch.Tensor([0, 1, 0]), 0.1
        )

        res = self.planner.plan(start_pose, point)
        waypoints = [pt.state for pt in res.trajectory]
        self.agent.visualize_points(waypoints, torch.Tensor([0, 0, 1]), 0.05, "world/waypoint")

        traj = self.planner.clean_path_for_xy(waypoints)
        self.agent.visualize_points(traj, torch.Tensor([1, 1, 0]), 0.11, "world/cleaned_path")

        self.previous_trajectories.append(traj)
        self.robot.execute_trajectory(traj, blocking=True, interrupt_condition=self.should_interrupt_navigation)
