from typing import Any
import numpy as np
import torch
import os
from irm_task.base import IRMBaseTask
from prompts import compose_irm_prompt
from guidance_parser import GuidanceParser
from stretch.agent.manipulation.dynamem_manipulation.image_publisher import send_array
from PIL import Image
from datetime import datetime, timezone
from utils import get_grid_image
from logging_utils import get_logger
from irm_task.base import TaskStatus, ExecutionMode
logger = get_logger(__name__)

BOLD = "\033[1m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

class ExploreMultipleTargetsTask(IRMBaseTask):
    # Constants
    DYNAMIC_UPDATE_THRESHOLD = 1.0
    
    def __init__(self, irm, robot, agent, get_drm_memory_fn, vlm1_socket, image_buffer, _ee_image_buffer, fps, config):
        super().__init__(irm, robot, agent, get_drm_memory_fn, vlm1_socket, image_buffer, _ee_image_buffer, fps, config)
        self.seen_objects = config["seen_objects"]
        self.unseen_objects = config["unseen_objects"]
        self.current_target_object = config["current_target_object"]
        self.total_target_objects = self.seen_objects + self.unseen_objects + [self.current_target_object]
        # Guidance prompt G from the DRM (paper Eq. 1) and the guidance-conditioned
        # parser Phi_G (paper Eq. 3) used to turn the VLM report Z_t into (X_t, m_t).
        self.guidance = config.get("guidance", "")
        self.guidance_parser = GuidanceParser(getattr(self.irm, "model", None))
        self._num_frames_for_irm = config.get("_num_frames_for_irm", 8)
        self.verbose = config.get("verbose", True)
        self.activate_only_on_moving_base = config.get("activate_only_on_moving_base", True)
        self.interrupt_object = config.get("interrupt_object", None)
        self.irm_result = None
        self.log_dir = os.path.join(config["log_dir"], "ExploreMultipleTargetsTask")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        self.dynamic_update_threshold = config.get("dynamic_update_threshold", self.DYNAMIC_UPDATE_THRESHOLD)
        
    def should_run(self):
        if self.get_status() != TaskStatus.ACTIVE:
            return False
        if self.activate_only_on_moving_base and not self.robot.moving_base:
            return False
        return True

    def get_batch_images_for_irm(self):
        with self.irm._image_buffer_lock:
            consecutive_unused = []
            for image in reversed(self.image_buffer):
                if self.__class__.__name__ in image["used_by"]:
                    break
                consecutive_unused.append(image)
            if len(consecutive_unused) < self._num_frames_for_irm:
                return []
            batch_images = list(reversed(consecutive_unused[:self._num_frames_for_irm]))
            for image in batch_images:
                image["used_by"].add(self.__class__.__name__)

        return batch_images
    
    def send_images_to_vlm(self, images, prompt, timestamps):
        send_dict = {
            "images": images,
            "fps": self._fps,
            "post_time_instruction_prompt": prompt,
        }
        send_array(self.vlm1_socket, send_dict)
        result = self.vlm1_socket.recv_pyobj()['description']
        return result
    
    def _sort_matched_objects_by_priority(self, matched_objects):
        sorted_objects = []
        if self.action == "goto_target_point":
            sorted_objects.extend([obj for obj in matched_objects if obj in self.unseen_objects])
            if self.current_target_object in matched_objects:
                sorted_objects.append(self.current_target_object)
            sorted_objects.extend([obj for obj in matched_objects if obj in self.seen_objects and obj != self.current_target_object])
        elif self.action == "explore_environment":
            if self.current_target_object in matched_objects:
                sorted_objects.append(self.current_target_object)
            sorted_objects.extend([obj for obj in matched_objects if obj in self.unseen_objects and obj != self.current_target_object])
            sorted_objects.extend([obj for obj in matched_objects if obj in self.seen_objects])
        
        return sorted_objects

    def _log_irm_result(self, matched_objects, raw_outputs, grid_image):
        self.agent.rerun_visualizer.queue.put({
            "type": "log_custom_2d_image", 
            "identity_name": "/observation_similar_to_text", 
            "img": grid_image
        })
        irm_output = f"### ❗ IRM found {matched_objects}\n### Raw output: {raw_outputs}"
        self.agent.rerun_visualizer.queue.put({
            "type": "log_text", 
            "identity_name": "instance_reasoning_log", 
            "text": irm_output
        })
    
    def _log_localization_result(self, target_object, success, max_score):
        if success:
            message = f"### ✅ Successfully localized {target_object}, max score: {max_score}"
        else:
            message = f"### ❌ Failed to localize {target_object}, max score: {max_score}"
        
        self.agent.rerun_visualizer.queue.put({
            "type": "log_text", 
            "identity_name": "robot_monologue", 
            "text": message
        })
    
    def _log_target_point(self, target_point, target_object, color):
        self.agent.rerun_visualizer.queue.put({
            "type": "log_text_label", 
            "target_point": target_point, 
            "name": target_object.replace(' ', '_'), 
            "box_size": 0.05, 
            "color": color
        })
    
    def _detect_object_in_images(self, batch_images, target_object, depth_threshold=3.0, confidence_threshold=0.2):
        target_point = None; image_idx = None; max_score = -1.0
        for i, image_dict in enumerate(batch_images):
            rgb = torch.Tensor(image_dict["rgb"])
            depth = torch.Tensor(image_dict["depth"])
            camera_K = torch.Tensor(image_dict["camera_K"])
            camera_pose = image_dict["camera_pose"]
            xyz, (rgb_with_text, _), score = self.agent.detection_model.compute_obj_coord(target_object, rgb, depth, camera_K, camera_pose, depth_threshold=depth_threshold, confidence_threshold=confidence_threshold)
            if score is not None and score.item() > confidence_threshold:
                target_point = xyz; image_idx = i; score = score.item()
                max_score = max(max_score, score)
                break

        return target_point, image_idx, max_score, rgb_with_text
    
    def find_target_object_in_batch(self, matched_objects, batch_images, raw_outputs, depth_threshold=3.0, seen_confidence_threshold=0.35, unseen_confidence_threshold=0.2):
        pil_images = [Image.fromarray(image["rgb"]) for image in batch_images]
        grid = get_grid_image(pil_images)
        self._log_irm_result(matched_objects, raw_outputs, grid)

        sorted_matched_objects = self._sort_matched_objects_by_priority(matched_objects)
        target_object = sorted_matched_objects[0]
        
        if target_object in self.unseen_objects:
            confidence_threshold = unseen_confidence_threshold
        else:
            confidence_threshold = seen_confidence_threshold

        target_point, image_idx, max_score, rgb_with_text = self._detect_object_in_images(batch_images, target_object, depth_threshold, confidence_threshold)

        if target_point is None:
            self._log_localization_result(target_object, False, max_score)
            return None
        else:
            self._log_localization_result(target_object, True, max_score)
    
        return_dict = {
            target_object: {
                'target_point': target_point,
                'image_idx': image_idx,
                'score': max_score,
                'rgb_with_text': rgb_with_text,
                'timestamp': batch_images[image_idx]["timestamp"],
            }   
        }
        return return_dict
    
    def _should_update_position(self, target_object, new_position):
        object_positions = self.get_drm_memory_fn()["object_positions"]
        if target_object not in object_positions:
            return True
        
        previous_position = object_positions[target_object].get("xyz", None)
        if previous_position is None:
            return True
        
        distance = np.linalg.norm(previous_position - new_position.cpu().numpy())
        update = distance > self.dynamic_update_threshold
        if update:
            logger.warning(f"{BOLD}{RED}Target object {target_object} is too far from the previous position. Update the memory{RESET}")
            logger.warning(f"Before: {previous_position}, After: {new_position}")
            return True
        else:
            logger.notice(f"{BOLD}{GREEN}Target object {target_object} is NOT too far from the previous position. Skip updating the memory{RESET}")
            logger.notice(f"Before: {previous_position}, After: {new_position}")
            return False

    def _emit_update_position_event(self, target_object, target_point, image_idx, score):
        # Non-critical object update -> CONTINUE (just refresh memory R_t, keep moving).
        self.emit_event("update_object_position", {
            "target_object": target_object,
            "target_point": target_point,
            "image_idx": image_idx,
            "score": score,
        }, mode=ExecutionMode.CONTINUE)
        self.agent.rerun_visualizer.queue.put({
            "type": "log_text_label", 
            "target_point": target_point, 
            "name": target_object.replace(' ', '_'), 
            "box_size": 0.05, 
            "color": [255, 0, 0]
        })
    
    def run_step(self):
        if self.get_status() == TaskStatus.STOPPING:
            with self._lock:
                self._status = TaskStatus.STOPPED
            return

        if not self.should_run():
            return

        batch_images = self.get_batch_images_for_irm()
        if len(batch_images) == 0:
            return
        
        timestamps = [datetime.fromisoformat(image["timestamp"]) for image in batch_images]
        images = [image["rgb"] for image in batch_images]
        images = np.array(images)
        
        # Compose the IRM prompt = DRM guidance G (focus) + response-format template.
        prompt = compose_irm_prompt(self.guidance, "explore_multiple_targets", self.total_target_objects)
        result = self.send_images_to_vlm(images, prompt, timestamps)
        self.agent.rerun_visualizer.queue.put({"type": "log_text", "identity_name": "instance_reasoning_log", "text": f"### IRM: {result}"})

        parsed_report = self.guidance_parser.parse(result, "explore_multiple_targets",
                                                   self.total_target_objects, self._replan_targets())
        match_list = parsed_report.objects

        if not match_list:
            return
        target_object_dict = self.find_target_object_in_batch(match_list, batch_images, result)
        if target_object_dict is None:
            irm_output = f"### ❌ IRM responded {match_list}, but failed to localize in batch images"
            self.agent.rerun_visualizer.queue.put({"type": "log_text", "identity_name": "instance_reasoning_log", "text": irm_output})
            logger.warning(f"Failed to localize {match_list} in batch images")
            return
        
        target_object = list[Any](target_object_dict.keys())[0]
        timestamp_matched_image = datetime.fromisoformat(target_object_dict[target_object]["timestamp"])
        total_time = (datetime.now(timezone.utc) - timestamp_matched_image).total_seconds()
        target_point = target_object_dict[target_object]["target_point"]
        image_idx = target_object_dict[target_object]["image_idx"]
        rgb_with_text = target_object_dict[target_object]["rgb_with_text"]
        score = target_object_dict[target_object]["score"]

        self.agent.rerun_visualizer.queue.put({"type": "log_text", "identity_name": "instance_reasoning_log", "text": f"### ❗ IRM found {target_object} in image {image_idx} (total time: {total_time:.2f}s.)"})
        self.agent.rerun_visualizer.queue.put({"type": "log_custom_2d_image", "identity_name": "/observation_similar_to_text", "img": rgb_with_text})

        self.after_localize_task(target_object, target_point, image_idx, score, parsed_report.mode)
    
    def _interrupt_navigation(self, target_object, reason):
        logger.warning(f"{BOLD}{RED}{reason}{RESET}")
        self.agent.rerun_visualizer.queue.put({
            "type": "log_text", 
            "identity_name": "instance_reasoning_log", 
            "text": f"### ❗ {reason}"
        })
        
        # Task-critical change (current target appeared/moved) -> REPLAN.
        self.emit_event("object_moved", {
            "target_object": target_object,
            "reason": reason,
        }, mode=ExecutionMode.REPLAN)
        self.set_status(TaskStatus.STOPPING)

    def _replan_targets(self):
        if self.action == "goto_target_point":
            return [self.current_target_object]
        memory = self.get_drm_memory_fn()
        empty_hand = memory["held_object"] is None
        return [obj for obj in self.unseen_objects
                if obj == self.current_target_object
                or (empty_hand and obj in memory["object_receptacle_mapping"])]

    def after_localize_task(self, target_object, target_point, image_idx, score, mode):
        if target_object in self.seen_objects and not self._should_update_position(target_object, target_point):
            return
        self._emit_update_position_event(target_object, target_point, image_idx, score)
        if mode == ExecutionMode.REPLAN and target_object in self._replan_targets():
            self._interrupt_navigation(target_object, f"Force stop since {target_object} appeared or moved")
