import base64
import json
import openai
import numpy as np
import torch
import torch.nn.functional as F
from io import BytesIO
from PIL import Image
from logging_utils import get_logger
from typing import TYPE_CHECKING
from utils import show_maps
from utils import RED, RESET

if TYPE_CHECKING:
    from stretch.agent.robot_agent_dynamem import RobotAgent

logger = get_logger(__name__)


def find_alignment_over_model_last_observations(agent, queries: str, last_observations_count: int = 1, save_path: str = None) -> np.ndarray:
    clip_text_tokens = agent.voxel_map.encoder.encode_text(queries).cpu()
    points, features, weights, _ = agent.voxel_map.semantic_memory.get_pointcloud()
    if points is None:
        return None
    features = F.normalize(features, p=2, dim=-1).cpu()

    obs_ids = agent.voxel_map.semantic_memory._obs_counts.cpu()
    observation_ids = sorted(obs_ids.unique().numpy().tolist())
    last_observation_ids = observation_ids[-last_observations_count:]
    mask = torch.isin(obs_ids, torch.tensor(last_observation_ids))

    features = features[mask]; points = points[mask]; obs_ids = obs_ids[mask]
    point_alignments = clip_text_tokens.float() @ features.float().T


    if save_path is not None:
        H, W = 1024, 1024
        scores = point_alignments.squeeze().numpy()
        min_score = scores.min(); max_score = scores.max()
        points_pt = [agent.planner.to_pt(p) for p in points]
        img = np.zeros((H, W, 3), dtype=np.float32)
        for (x, y), score in zip(points_pt, scores):
            norm_score = (score - min_score) / (max_score - min_score + 1e-8)
            color = [norm_score, norm_score, 1 - norm_score]
            img[y, x] = color
        show_maps([img], titles=["Alignment over last observations"], zoom=11.0, save_path=save_path)

    return point_alignments, points, obs_ids


def localize_single_object_last_observations(agent: "RobotAgent", target_object: str, last_observations_count: int = 1, save_path: str = None, gpt_verification: bool = True, model: str = "gpt-4o") -> np.ndarray:
    point_alignments, points, obs_ids = find_alignment_over_model_last_observations(agent, target_object, last_observations_count, save_path=save_path)
    if point_alignments is None:
        return None, None, None
    
    obs_id = obs_ids[point_alignments.argmax(dim=-1)].item()
    rgb = agent.voxel_map.observations[obs_id - 1].rgb
    pose = agent.voxel_map.observations[obs_id - 1].camera_pose
    depth = agent.voxel_map.observations[obs_id - 1].depth
    K = agent.voxel_map.observations[obs_id - 1].camera_K
    target_point, (rgb_with_text, rgb_bbox_only), score = agent.detection_model.compute_obj_coord(target_object, rgb, depth, K, pose)

    if target_point is None:
        return None, None, None
    else:
        if gpt_verification:
            match, reason = verify_with_gpt(rgb_bbox_only, target_object, model)
            if not match:
                logger.warning(f"OWL detected {target_object}({score.item()}) but GPT said it's not. Skip this observation. Reason: {reason}")
                return None, None, None

        agent.rerun_visualizer.log_custom_2d_image("/observation_similar_to_text", rgb_with_text)
        agent.rerun_visualizer.log_text_label(target_point, target_object.replace(' ', '_'), box_size=0.05, color=[255, 0, 0])
        Image.fromarray(rgb_with_text.cpu().numpy().astype(np.uint8)).save(f"{agent.log}/localization_results/{target_object}_bbox.png")

        target_point_np = target_point.detach().cpu().numpy() if torch.is_tensor(target_point) else target_point
        return target_point_np, obs_id, score.item()


def verify_with_gpt(rgb_bbox_only, target_object, model: str = "gpt-4o"):
    system_prompt = (
        "You are an expert vision-language assistant for a mobile robot.\n"
        "The robot gives you a full RGB image where the candidate object is "
        "highlighted with a RED bounding box. Your task is to look ONLY at the "
        "area **inside** that red rectangle and decide whether it is a **plausible match** "
        f"for the expected object: \"{target_object}\".\n\n"
        "The object detector has already made a reasonable selection, so only respond 'false' "
        "if the highlighted item is **clearly and objectively different** from the expected object "
        "(e.g., completely different category, color, or shape).\n"
        "Minor differences are acceptable — be practical, not overly strict.\n\n"
        "Answer **ONLY** with valid JSON in this format:\n"
        '{\n  "match": true | false,\n  "reason": "<concise English explanation>"\n}\n'
        "Do NOT output anything else."
    )

    user_messages = []
    buf = BytesIO()
    visual_prompt = Image.fromarray(rgb_bbox_only.cpu().numpy().astype(np.uint8))
    visual_prompt.save(buf, format="PNG")
    base64_encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    user_messages.append(
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_encoded}",
                "detail": "high",
            },
        }        
    )

    user_messages.append(
        {
            "type": "text",
            "text": (
                    "Here is the image. Judge only what's **inside** the red box "
                    f"and return the JSON described above. The target object is '{target_object}'."
                    "Please return the JSON format only."
                ),
        }
    )

    completion = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_messages}
        ],
        response_format={"type": "json_object"},
    )
    result = completion.choices[0].message.content

    try:
        response_json = json.loads(result)
        match = response_json["match"]
        if not isinstance(match, bool):
            logger.warning(f"verify_with_gpt: non-bool 'match' in response {result!r}; failing closed (treat as no-match)")
            return False, response_json.get("reason")
        return match, response_json.get("reason")
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"verify_with_gpt: unparseable response {result!r} ({e}); failing closed (treat as no-match)")
        return False, None


def localize_single_object_with_mllm(agent: "RobotAgent", target_object: str, return_shortest: bool = True, visualize_rerun: bool = True):
    image_ids, points, alignments = agent.voxel_map.find_all_images(target_object)
    target_ids = agent.voxel_map.llm_locator(image_ids, target_object)

    if target_ids is None or len(target_ids) == 0:
        return None, None, None
    if isinstance(target_ids, int):
        target_ids = [target_ids]

    candidate_points = []; candidate_image_ids = []; candidate_scores = []
    for target_id in target_ids:
        target_id -= 1
        target_point = points[target_id]
        image_id = image_ids[target_id]

        rgb = agent.voxel_map.observations[image_id - 1].rgb
        pose = agent.voxel_map.observations[image_id - 1].camera_pose
        depth = agent.voxel_map.observations[image_id - 1].depth
        K = agent.voxel_map.observations[image_id - 1].camera_K
        res, (rgb_with_text, _), score = agent.detection_model.compute_obj_coord(target_object, rgb, depth, K, pose)

        if res is not None:
            candidate_points.append(res); candidate_image_ids.append(image_id); candidate_scores.append(score)

    if len(candidate_points) == 0:
        logger.warning(f"{RED}[Warning - localize_single_object_with_mllm] {target_object} is not found in all images{RESET}")
        target_point_np = target_point.detach().cpu().numpy() if torch.is_tensor(target_point) else target_point
        return target_point_np, image_ids[target_ids[0]], candidate_scores[0]

    best_idx = max(range(len(candidate_scores)), key=lambda i: candidate_scores[i])
    localized_point = candidate_points[best_idx]
    best_image_id = candidate_image_ids[best_idx]
    best_score = candidate_scores[best_idx]
    
    if visualize_rerun:
        agent.rerun_visualizer.log_text_label(localized_point, target_object.replace(' ', '_'), box_size=0.05, color=[255, 0, 0])
        agent.rerun_visualizer.log_custom_2d_image("/observation_similar_to_text", rgb_with_text)

    localized_point_np = localized_point.detach().cpu().numpy() if torch.is_tensor(localized_point) else localized_point
    return localized_point_np, best_image_id.item(), best_score.item()



