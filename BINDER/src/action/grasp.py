import os
import queue
import time

import torch
from overrides import overrides

from action.base import BaseAction
from action.utils.manipulation import (
    ensure_log_directory,
    save_manipulation_image,
    compute_head_tilt_for_target,
    determine_gripper_width,
)
from irm_task.grasping_refinement_task import GraspingRefinementTask
from irm_task.watch_grasping_task import WatchGraspingTask
from irm_task.base import ExecutionMode
from prompts import get_default_guidance
from logging_utils import get_logger
from stretch.agent.manipulation.dynamem_manipulation.grasper_utils import (
    capture_and_process_image,
    pickup_k,
)
from stretch.agent.robot_agent_dynamem import (
    INIT_ARM_POS,
    INIT_HEAD_PAN,
    INIT_HEAD_TILT,
    INIT_LIFT_POS,
    INIT_WRIST_PITCH,
    INIT_WRIST_ROLL,
    INIT_WRIST_YAW,
)

logger = get_logger(__name__)


class GraspAction(BaseAction):
    def __init__(self, drm, irm, args, binder_config):
        super().__init__(drm, irm, args, binder_config)
        self.num_max_tries = 3
        self.num_tries = 0
        self.grasp_object = None
        self.retry_grasp_point = None
        self.retry_grasp_image_id = None
        self.object_disappeared = False
        # Set by _check_object_visibility_after_grasp(), but that runs at the end of
        # _manipulate() — which returns early when AnyGrasp finds no pose. Initialize here
        # so after_failed_action_task() never reads an unset attribute.
        self.try_backward = False

        self.manip_wrapper = self.agent.manip_wrapper
        self.manip_socket = self.agent.manip_socket
        self.re = self.agent.re

    @overrides
    def predict_irm_tasks(self):
        log_dir = f"{self.agent.log}/{self.drm.drm_call_count}_grasp"

        guidance = self.args.get("guidance") or get_default_guidance(
            "_grasp", [self.args["target_object"]]
        )

        grasping_refinement_config = {
            "action": "grasp",
            "target_object": self.args["target_object"],
            "guidance": guidance,
            "verbose": True,
            "log_dir": log_dir,
        }
        grasping_refinement_task = GraspingRefinementTask(
            irm=self.irm,
            robot=self.irm.robot,
            agent=self.drm.agent,
            get_drm_memory_fn=self.drm.get_drm_memory,
            vlm1_socket=self.irm.vlm1_socket,
            image_buffer=self.irm._image_buffer,
            _ee_image_buffer=self.irm._ee_image_buffer,
            fps=self.irm._fps,
            config=grasping_refinement_config,
        )

        watch_grasping_config = {
            "action": "grasp",
            "target_object": self.args["target_object"],
            "guidance": guidance,
            "verbose": True,
            "log_dir": log_dir,
        }
        watch_grasping_task = WatchGraspingTask(
            irm=self.irm,
            robot=self.irm.robot,
            agent=self.drm.agent,
            get_drm_memory_fn=self.drm.get_drm_memory,
            vlm1_socket=self.irm.vlm1_socket,
            image_buffer=self.irm._image_buffer,
            _ee_image_buffer=self.irm._ee_image_buffer,
            fps=self.irm._fps,
            config=watch_grasping_config,
        )
        return [watch_grasping_task, grasping_refinement_task]

    @overrides
    def run(self):
        if self.drm.memory["held_object"] is not None:
            self._handle_invalid_grasp_while_holding()
            return

        success = False
        recovery_result = False

        while (
            not success
            and self.num_tries <= self.num_max_tries
            and not self.object_disappeared
        ):
            self.before_action_task()
            success = self.execute()
            self.num_tries += 1

            if not success:
                recovery_result = self.after_failed_action_task()
                if not recovery_result:
                    break
            else:
                self.after_success_action_task()
                break

        self.update_drm_history(success=success, recovery_result=recovery_result)

    def _handle_invalid_grasp_while_holding(self):
        held = self.drm.memory["held_object"]
        self.drm.action_history.append({
            "action": "grasp",
            "arguments": {"target_object": self.args["target_object"]},
            "status": f"failed - cannot grasp while holding {held}. "
                      "Place the held object first.",
            "result": None,
        })

    @overrides
    def before_action_task(self):
        if self.enable_irm:
            self.irm_tasks = self.predict_irm_tasks()
            if self.irm_tasks is not None:
                self.irm.set_tasks(self.irm_tasks)
        else:
            self.irm_tasks = []

        self.target_object = self.args["target_object"]
        if self.num_tries == 0:
            self.target_point = self.drm.memory["object_positions"][self.target_object]["xyz"]
        else:
            self.target_point = self.retry_grasp_point

    @overrides
    def execute(self):
        self.robot.switch_to_manipulation_mode()
        camera_xyz = self.robot.get_head_pose()[:3, 3]

        theta = compute_head_tilt_for_target(camera_xyz, self.target_point)

        if self.grasp_object is not None:
            self._execute_visual_servoing_grasp(theta)
        else:
            self._execute_anygrasp_manipulation(theta)

        if not self.enable_irm:
            # No IRM => no post-grasp verification signal is available; assume the attempt succeeded.
            return True
        else:
            return self._wait_for_irm_verification()

    def _execute_visual_servoing_grasp(self, theta: float):
        state = self.robot.get_six_joints()
        state[1] = 1.0
        self.robot.arm_to(state, blocking=True)

        self.grasp_object(
            target_object=self.target_object,
            object_xyz=self.target_point,
            match_method="feature",
            show_object_to_grasp=False,
            show_servo_gui=True,
            delete_object_after_grasp=False,
        )

    def _execute_anygrasp_manipulation(self, theta: float):
        self._manipulate(self.target_object, theta)

    def _manipulate(
        self,
        text: str,
        init_tilt: float = INIT_HEAD_TILT,
        base_node: str = "camera_depth_optical_frame",
    ):
        self.robot.switch_to_manipulation_mode()
        self.robot.look_at_ee()

        self.manip_wrapper.move_to_position(
            arm_pos=INIT_ARM_POS,
            head_pan=INIT_HEAD_PAN,
            head_tilt=init_tilt,
            gripper_pos=1,
            lift_pos=INIT_LIFT_POS,
            wrist_pitch=INIT_WRIST_PITCH,
            wrist_roll=INIT_WRIST_ROLL,
            wrist_yaw=INIT_WRIST_YAW,
        )

        rotation, translation, depth, width, head_best_pose_image = capture_and_process_image(
            mode="pick",
            obj=text,
            socket=self.manip_socket,
            hello_robot=self.manip_wrapper,
            init_head_tilt=init_tilt,
        )

        if rotation is None:
            return False

        gripper_width = determine_gripper_width(width, self.re)

        log_dir = ensure_log_directory(
            self.agent.log, "grasp", self.drm.drm_call_count
        )

        if head_best_pose_image is not None:
            save_manipulation_image(
                head_best_pose_image,
                log_dir,
                "anygrasp",
                text,
                self.agent.rerun_visualizer,
            )

        pickup_k(
            self.manip_wrapper,
            rotation,
            translation,
            base_node,
            "link_gripper_s3_body",
            gripper_depth=depth,
            gripper_width=gripper_width,
            log=log_dir,
            text=text,
            socket=self.manip_socket,
            k=2,
            irm_event_queue=self.irm.event_queue,
            rerun_visualizer=self.agent.rerun_visualizer,
            enable_irm=self.enable_irm,
        )

        self._retract_arm_after_grasp()

        self._check_object_visibility_after_grasp()

        self.robot.move_to_nav_posture()
        self.robot.finished_gripper_closing = True

        return True

    def _retract_arm_after_grasp(self):
        current_lift = self.manip_wrapper.robot.get_six_joints()[1]
        self.manip_wrapper.move_to_position(lift_pos=min(current_lift + 0.07, 1.1))

        self.manip_wrapper.move_to_position(arm_pos=0.01)

        current_lift = self.manip_wrapper.robot.get_six_joints()[1]
        self.manip_wrapper.move_to_position(
            lift_pos=min(current_lift, 0.9), wrist_yaw=2.5
        )
        self.manip_wrapper.move_to_position(
            lift_pos=min(self.manip_wrapper.robot.get_six_joints()[1], 0.55)
        )

        self.manip_wrapper.move_to_position(
            base_trans=-self.manip_wrapper.robot.get_six_joints()[0]
        )

    def _check_object_visibility_after_grasp(self):
        obs = self.robot.get_observation()
        rgb = torch.Tensor(obs.rgb)
        depth = torch.Tensor(obs.depth)
        camera_K = torch.Tensor(obs.camera_K)
        camera_pose = torch.Tensor(obs.camera_pose)

        target_point, _, score = self.drm.agent.detection_model.compute_obj_coord(
            self.target_object,
            rgb,
            depth,
            camera_K,
            camera_pose,
            confidence_threshold=0.3,
            depth_threshold=3.0,
        )

        if score is None:
            self.try_backward = True
        else:
            self.try_backward = False

    def _wait_for_irm_verification(self) -> bool:
        try:
            response = self.irm.event_queue.get(timeout=30.0)  # 30 second timeout
        except queue.Empty:
            logger.error("Timeout waiting for IRM verification response")
            return False

        # The type selects which IRM task's report this is; the mode m_t is the decision.
        if response.get("type") == "watch_grasping_result":
            return response.get("mode") == ExecutionMode.CONTINUE   # grasp succeeded
        return False

    def refresh_object_memory(self, after_success: bool = True):
        save_path = ensure_log_directory(
            self.agent.log, "grasp", self.drm.drm_call_count
        )

        is_boundary = self.check_objects_in_roi(
            target_objects=[self.target_object], last_observations_count=1
        )

        if not is_boundary[self.target_object]:
            return

        target_point, image_id, score = self.localize_single_object_last_observations(
            self.target_object,
            last_observations_count=1,
            save_path=os.path.join(save_path, f"after_grasp_{self.target_object}.png"),
        )

        # On a failed grasp, if the object can no longer be re-localized, mark it as gone.
        if not after_success and target_point is None:
            self.object_disappeared = True

    def after_success_action_task(self):
        if self.enable_irm:
            self.irm.unset_tasks()

        self.rotate_90_degrees(reverse=True)
        self.robot.move_to_nav_posture()
        self.robot.look_front_down()
        time.sleep(0.5)

        self.agent.update()
        self.refresh_object_memory(after_success=True)
        self.robot.look_front()

        self.robot.grasping_just_finished = False
        self.robot.grasping_initial_result = None
        self.robot.grasping_verification_result = None

        target_object_name = f"{self.target_object.replace(' ', '_')}"
        self.agent.rerun_visualizer.clear_identity(
            f"world/localized_objects/{target_object_name}"
        )

        self.drm.memory["target_objects"].remove(self.target_object)

    def after_failed_action_task(self) -> bool:
        if self.enable_irm:
            self.irm.unset_tasks()
        self.robot.move_to_nav_posture()
        start_pose = self.robot.get_base_pose()

        if not self.try_backward:
            self.retry_grasp_point = self.target_point
            return True

        recovery_point = self.agent.space.sample_point_for_recovery_rectangular(
            start_pose,
            self.target_point,
            self.agent.planner,
            rerun_visualizer=self.agent.rerun_visualizer,
        )

        res = self.agent.planner.plan(start_pose, recovery_point)
        if res is None:
            raise ValueError(f"Could not find trajectory to recovery point: {recovery_point}")

        waypoints = [pt.state for pt in res.trajectory]
        traj = self.agent.planner.clean_path_for_xy(waypoints)

        self.robot.execute_trajectory(traj, blocking=True)

        self.agent.update()
        self.refresh_object_memory(after_success=False)

        if self.object_disappeared:
            return False

        self.retry_grasp_point = self.drm.memory["object_positions"][self.target_object]["xyz"]
        self.move_to_reachable_nearest_point(self.retry_grasp_point, visualize=True)

        return True

    def update_drm_history(self, success: bool, recovery_result: bool):
        if success:
            self.drm.memory["object_positions"].pop(self.target_object)
            self.drm.memory["held_object"] = self.target_object
            self.drm.action_history.append({
                "action": "grasp",
                "arguments": {"target_object": self.target_object},
                "status": "success",
                "result": None,
            })
        else:
            self.drm.memory["held_object"] = None
            if recovery_result:
                self.drm.action_history.append({
                    "action": "grasp",
                    "arguments": {"target_object": self.target_object},
                    "status": "failed",
                    "result": None,
                })
            else:
                self.drm.action_history.append({
                    "action": "grasp",
                    "arguments": {"target_object": self.target_object},
                    "status": "failed - object disappeared, need to explore again.",
                    "result": None,
                })
                self.drm.memory["object_positions"].pop(self.target_object)
