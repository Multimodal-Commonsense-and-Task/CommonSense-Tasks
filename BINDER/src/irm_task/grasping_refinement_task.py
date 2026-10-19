import os
import time

import numpy as np

from irm_task.base import IRMBaseTask, TaskStatus
from irm_task.utils.image_processing import (
    get_batch_images_from_buffer,
    send_images_to_vlm,
    extract_rgb_array,
    log_irm_response_to_rerun,
)
from irm_task.base import ExecutionMode
from guidance_parser import GuidanceParser
from prompts import compose_irm_prompt
from logging_utils import get_logger

logger = get_logger(__name__)


class GraspingRefinementTask(IRMBaseTask):
    NUM_FRAMES_FOR_REASONING = 64
    GRIPPER_LOWERING_WAIT_TIME = 8.0

    def __init__(self, irm, robot, agent, get_drm_memory_fn,
                 vlm1_socket, image_buffer, _ee_image_buffer, fps, config):
        super().__init__(
            irm, robot, agent, get_drm_memory_fn,
            vlm1_socket, image_buffer, _ee_image_buffer, fps, config
        )
        self.target_object = config["target_object"]
        self.verbose = config.get("verbose", True)
        self.guidance = config.get("guidance", "")
        self.guidance_parser = GuidanceParser(getattr(self.irm, "model", None))
        self.log_dir = os.path.join(config["log_dir"], "GraspingRefinementTask")

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self._num_frames_for_irm = self.NUM_FRAMES_FOR_REASONING
        self.robot.started_gripper_lowering = False
        self.started_gripper_lowering_time = None

    def should_run(self) -> bool:
        if self.get_status() != TaskStatus.ACTIVE:
            return False

        if not self.robot.started_gripper_lowering:
            return False

        if self.robot.started_gripper_lowering and self.started_gripper_lowering_time is None:
            self.started_gripper_lowering_time = time.time()
            return False

        if time.time() - self.started_gripper_lowering_time >= self.GRIPPER_LOWERING_WAIT_TIME:
            return True

        return False

    def _get_batch_images(self):
        return get_batch_images_from_buffer(
            image_buffer=self._ee_image_buffer,
            buffer_lock=self.irm._ee_image_buffer_lock,
            task_class_name=self.__class__.__name__,
            num_frames=self._num_frames_for_irm,
        )

    def _evaluate_with_vlm(self, images: np.ndarray):
        # IRM prompt = DRM guidance G + grasping-refinement response format.
        prompt = compose_irm_prompt(self.guidance, "grasping_refinement", [self.target_object])
        result = send_images_to_vlm(
            vlm_socket=self.vlm1_socket,
            images=images,
            prompt=prompt,
            fps=self._fps,
            verbose=self.verbose,
            task_name="Grasping refinement",
        )

        log_irm_response_to_rerun(
            self.agent.rerun_visualizer,
            "Grasping refinement",
            result,
        )

        parsed = self.guidance_parser.parse(result, "grasping_refinement", [self.target_object])
        return parsed.mode == ExecutionMode.CONTINUE, parsed.mode

    def _emit_result(self, grasp_feasible: bool, mode: ExecutionMode):
        self.irm.event_queue.put({
            "type": "grasping_refinement_result",
            "data": grasp_feasible,
            "mode": mode,
        })

    def run_step(self):
        if self.get_status() == TaskStatus.STOPPING:
            with self._lock:
                self._status = TaskStatus.STOPPED
            return

        if not self.should_run():
            return

        batch_images = self._get_batch_images()
        if len(batch_images) == 0:
            return

        images = extract_rgb_array(batch_images)
        grasp_feasible, mode = self._evaluate_with_vlm(images)
        self._emit_result(grasp_feasible, mode)
        self.set_status(TaskStatus.STOPPING)
