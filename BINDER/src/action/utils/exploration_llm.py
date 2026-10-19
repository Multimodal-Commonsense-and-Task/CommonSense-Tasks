import base64
import json
import queue
import threading
from io import BytesIO
from typing import List, Optional, Tuple

import openai
from PIL import Image, ImageDraw

from logging_utils import get_logger

logger = get_logger(__name__)

RED = "\033[91m"
RESET = "\033[0m"


def build_exploration_system_prompt(targets_str: str) -> str:
    return f"""
You are an expert vision-language assistant for a mobile robot.

Input you receive:
1. Map image (top-down):
 - Green pixels indicate the robot's reachable area and red pixels indicate obstacles.
   (Note: Candidates located in regions dominated by red pixels (obstacles) are likely near walls or blocked areas and should be avoided.)
 - Blue points (labelled "candidate i") = exploration candidates (linked to the candidate image)
 - Yellow square (labelled "robot") = robot's current position
2. k candidate images: Each has a red circle overlaid to indicate a possible exploration point that the robot can move to.
 - If a candidate clearly shows wires/cables or clutter that would block the robot, treat it as unsafe and exclude it.
3. Gray dashed poly-line = already traversed; avoid revisiting if possible.

Objects of interest: [{targets_str}]

Your task:
- Select the single best exploration candidate to maximize the chance of finding at least ONE object from the list above.
- In your reasoning, briefly mention the pros and cons of each candidate image (e.g., "Image 1 is good because..., Image 2 is blocked by a wall, Image 3 looks cluttered...").
- Also briefly consider pros/cons for each candidate using both the map context and what is visible in its view.
- Then state why you selected the best candidate.
- In most cases, the first image will be the best choice. If there is a clear reason to select another image, then select that image.

Instructions:
Return your answer as a JSON with keys:
- "selected_image": the index (starting from 1) of the best exploration point,
- "reason": your explanation, including a brief comment on each candidate image.

Example:
Input:
    The object of interest is a red ball.
    (Suppose none of the images directly shows the red ball.)
Output:
    {{
        "selected_image": 1,
        "reason": "Image 1 is promising because the red circle is near a hallway that could reveal new areas. Image 2 is directly in front of a wall, offering little new information. Image 3 is cluttered with many cables, which may block movement. Therefore, Image 1 is the best choice for exploration."
    }}
"""


def encode_image_to_base64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def add_center_circle_to_image(image: Image.Image, radius: int = 30, width: int = 5) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    cx, cy = image.size[0] // 2, image.size[1] // 2
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        outline="red",
        width=width,
    )
    return image


def build_user_messages(
    visual_prompt,
    rgb_images: List,
    targets_str: str,
    save_dir: Optional[str] = None,
) -> List[dict]:
    messages = []

    # Add map image
    map_image = Image.fromarray(visual_prompt)
    base64_encoded = encode_image_to_base64(map_image)
    messages.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{base64_encoded}",
            "detail": "high",
        },
    })

    # Add candidate images with red circle overlay
    for i, rgb in enumerate(rgb_images):
        image = Image.fromarray(rgb)
        image = add_center_circle_to_image(image)

        if save_dir:
            image.save(f"{save_dir}/k_candidate_{i + 1}.png")

        base64_encoded = encode_image_to_base64(image)
        messages.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_encoded}",
                "detail": "high",
            },
        })

    # Add text instruction
    messages.append({
        "type": "text",
        "text": (
            f"You are given one map image and {len(rgb_images)} candidate images, total {len(rgb_images) + 1} images."
            f" In each candidate image, the red circle marks a possible exploration point."
            f" Objects of interest: [{targets_str}]."
            f" Pick the single best exploration point to find ANY of them."
            " Please give your answer as a JSON object."
        ),
    })

    return messages


def call_llm_for_exploration(
    visual_prompt,
    rgb_images: List,
    objects_of_interest: List[str],
    save_dir: Optional[str] = None,
    model: str = "gpt-4o",
) -> Tuple[int, str]:
    targets_str = ", ".join(objects_of_interest)
    system_prompt = build_exploration_system_prompt(targets_str)
    user_messages = build_user_messages(visual_prompt, rgb_images, targets_str, save_dir)

    completion = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_messages},
        ],
        response_format={"type": "json_object"},
        service_tier="priority",
    )
    response = completion.choices[0].message.content

    try:
        response_json = json.loads(response)
        logger.info(response_json)
        return response_json["selected_image"], response_json["reason"]
    except json.JSONDecodeError:
        logger.warning(f"{RED}Invalid JSON response - {response}{RESET}")
        return 1, ""


def run_llm(
    visual_prompt,
    rgb_images: List,
    objects_of_interest: List[str],
    save_dir: Optional[str] = None,
    model: str = "gpt-4o",
) -> Tuple[int, str]:
    result_q = queue.Queue()

    def llm_worker():
        try:
            best_idx, reason = call_llm_for_exploration(
                visual_prompt, rgb_images, objects_of_interest, save_dir, model
            )
            result_q.put((best_idx, reason))
        except Exception as e:
            result_q.put((None, str(e)))

    t = threading.Thread(target=llm_worker, daemon=True)
    t.start()

    return result_q.get()
