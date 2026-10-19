from typing import List

import numpy as np

from logging_utils import get_logger
from stretch.agent.manipulation.dynamem_manipulation.image_publisher import send_array

logger = get_logger(__name__)


def get_batch_images_from_buffer(
    image_buffer: List[dict],
    buffer_lock,
    task_class_name: str,
    num_frames: int,
) -> List[dict]:
    with buffer_lock:
        consecutive_unused = []
        for image in reversed(image_buffer):
            if task_class_name in image["used_by"]:
                break
            consecutive_unused.append(image)

        if len(consecutive_unused) < num_frames:
            return []

        batch_images = list(reversed(consecutive_unused[:num_frames]))
        for image in batch_images:
            image["used_by"].add(task_class_name)

    return batch_images


def send_images_to_vlm(
    vlm_socket,
    images: np.ndarray,
    prompt: str,
    fps: int,
    verbose: bool = True,
    task_name: str = "IRM Task",
) -> str:
    send_dict = {
        "images": images,
        "fps": fps,
        "post_time_instruction_prompt": prompt,
    }

    try:
        send_array(vlm_socket, send_dict)

        if vlm_socket.poll(30000):
            result = vlm_socket.recv_pyobj()['description']
        else:
            return "ERROR: VLM response timeout"
    except Exception as e:
        return f"ERROR: {str(e)}"


    return result


def extract_rgb_array(batch_images: List[dict]) -> np.ndarray:
    images = [image["rgb"] for image in batch_images]
    return np.array(images)


def log_irm_response_to_rerun(
    rerun_visualizer,
    task_name: str,
    response: str,
):
    if rerun_visualizer is not None:
        rerun_visualizer.queue.put({
            "type": "log_text",
            "identity_name": "instance_reasoning_log",
            "text": f"### ❗ {task_name} response: {response}"
        })
