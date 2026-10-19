import openai
import json
import textwrap
import torch
import numpy as np
import threading
import queue
from PIL import Image
from action.goto_target_point import GotoTargetPointAction
from action.grasp import GraspAction
from action.place import PlaceAction
from action.explore_environment import ExploreEnvironmentAction
from stretch.agent.robot_agent_dynamem import RobotAgent
from stretch.agent.zmq_client import HomeRobotZmqClient
from instance_reasoning import IRM
from stretch.utils.morphology import get_edges
from utils import crop_nonzero_with_margin, draw_marker_on_zoom, pil_to_base64_str
from prompts import DRM_GUIDANCE_OUTPUT_INSTRUCTION
from logging_utils import get_logger
from typing import Dict, Any

logger = get_logger(__name__)

BOLD = "\033[1m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

class DRM:
    def __init__(
            self,
            agent: RobotAgent,
            irm: IRM,
            enable_irm: bool = True,
            binder_config: Dict[str, Any] = None,
    ):
        self.drm_call_count = 0
        self.agent = agent
        self.robot: HomeRobotZmqClient = agent.robot
        self.irm = irm
        self.task_prompt = ""
        self.localized_objects = None
        self.action_history = []
        self.memory = {
            'target_objects': set(),  # populated from the task prompt via set_object_receptacle()
            'object_positions': {},
            'object_receptacle_mapping': {},
            'held_object': None,
        }
        self.enable_irm = enable_irm
        self.binder_config = binder_config

    def __str__(self):
        seen_objects = set(self.memory['object_positions'].keys())
        result = ""
        result += f"{BOLD}{GREEN}[DRM] Memory:{RESET}\n"
        result += f"  - target_objects: {self.memory['target_objects']}\n"
        result += f"  - seen_objects: {seen_objects}\n"
        result += f"  - held_object: {self.memory['held_object']}\n"

        result += f"{BOLD}{GREEN}[DRM] Action history:{RESET}\n"
        for i, history_dict in enumerate(self.action_history):
            result += f"{i+1}. action: {history_dict['action']}\n"
            result += f"   - arguments: {history_dict['arguments']}\n"
            result += f"   - status: {history_dict['status']}\n"
            result += f"   - result: {history_dict['result']}\n"
        return result


    def decompose_object_recepatcle_mapping(self):
        prompt = textwrap.dedent(f"""
            Extract the object-to-receptacle mapping from the task description.
            The task description is as follows: {self.task_prompt}

            Return the mapping in the following format:
            {{
                "object_name": "receptacle_name",
                ...
            }}

            Example:
            TASK: Clean this room. Place the banana, Rubik's cube in the yellow basket and orange in the red box.
            MAPPING: {{"banana": "yellow basket", "Rubik's cube": "yellow basket", "orange": "red box"}}

            Caution:
            - Do not change the object names or receptacle names including their capitalizations.
            - Only return the dictionary in JSON format without any additional text.
        """)

        response = openai.chat.completions.create(
            model=self.binder_config['models']['drm']['model_name'],
            messages=[{
                "role": "user",
                "content": prompt,
            }
            ],
            response_format={"type": "json_object"},
            service_tier="priority"
        )
        return json.loads(response.choices[0].message.content.strip())

    def set_object_receptacle(self):
        target_objects = set()
        for object, receptacle in self.memory['object_receptacle_mapping'].items():
            target_objects.add(object)
            target_objects.add(receptacle)
        self.memory['target_objects'] = target_objects

    def set_task_prompt(self, prompt):
        self.task_prompt = prompt
        self.action_history = []
        self.memory["object_positions"].clear()
        self.memory["held_object"] = None
        self.memory["object_receptacle_mapping"] = self.decompose_object_recepatcle_mapping()
        self.set_object_receptacle()

    def generate_visual_prompt(self):
        obstacles, explored, history_soft = self.agent.space.voxel_map.get_2d_map(return_history_id=True)
        base_pose = self.robot.get_base_pose()
        reachable_points = self.agent.planner.get_reachable_points(self.agent.planner.to_pt(base_pose))
        reachable_xs, reachable_ys = zip(*reachable_points)
        reachable_xs = np.array(reachable_xs); reachable_ys = np.array(reachable_ys)
        reachable_map = torch.zeros_like(explored)
        reachable_map[reachable_xs, reachable_ys] = 1
        reachable_map = reachable_map.to(torch.bool)
        edges = get_edges(reachable_map)

        obstacles_np = obstacles.cpu().numpy().astype(bool)
        reachable_map_np = reachable_map.cpu().numpy().astype(bool)
        edges_np = edges.cpu().numpy().astype(bool)

        H, W = obstacles_np.shape
        img = np.zeros((H, W, 3), dtype=np.float32)
        img[reachable_map_np] = [0, 1, 0]
        img[obstacles_np] = [1, 0, 0]

        img_zoom, crop_origin, crop_size = crop_nonzero_with_margin(img, margin=7)
        base_pose_pt = self.agent.planner.to_pt(base_pose)

        # Mark the robot's current position
        marked_image = draw_marker_on_zoom(img_zoom, base_pose_pt, crop_origin=crop_origin, crop_size=crop_size, color=(255,255,0), label='robot', marker_type="square")

        # Mark objects
        object_positions = self.memory["object_positions"]
        for object, object_dict in object_positions.items():
            xyz_pt = self.agent.planner.to_pt(object_dict['xyz'])
            label_name = f"{object}"
            marked_image = draw_marker_on_zoom(marked_image, xyz_pt, crop_origin=crop_origin, crop_size=crop_size,
                                                   color=(0,0, 255), label=label_name, marker_type="circle")
        return Image.fromarray(marked_image)
        


    def decide_next_step(self):
        visual_prompt = self.generate_visual_prompt()
        visual_prompt_base64 = pil_to_base64_str(visual_prompt)

        result_q = queue.Queue()

        def llm_worker():
            try:
                prompt = self._build_llm_prompt()
                logger.info(f"Prompt: {prompt}")
                response = openai.chat.completions.create(
                    model=self.binder_config['models']['drm']['model_name'],
                    messages=[
                        {"role": "system", "content": "You are a high-level robotic decision-making agent."},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{visual_prompt_base64}"
                                    }
                                }
                            ]
                        }
                    ],
                    response_format={"type": "json_object"},
                    service_tier="priority"
                )
                raw = response.choices[0].message.content
                parsed = raw.strip().strip('`').strip('json').strip()
                d = json.loads(parsed)
                # Paper Eq. 1: DRM emits (action a, target o in args, guidance prompt G).
                result_q.put((d['action'], d['args'], d.get('reason', ''), d.get('guidance', '')))
            except Exception as e:
                result_q.put((None, None, str(e), ''))

        t = threading.Thread(target=llm_worker, daemon=True)
        t.start()

        action, args, reason, guidance = result_q.get()

        self.agent.rerun_visualizer.log_custom_2d_image("/observation_similar_to_text", visual_prompt)

        return action, args, reason, guidance

    def execute_next_step(self, action=None, args=None, reason=None, guidance=None):
        if action is None:
            action, args, reason, guidance = self.decide_next_step()

        # Carry the DRM guidance prompt G into the action so it can refocus the IRM.
        if args is None:
            args = {}
        args["guidance"] = guidance or ""

        logger.notice(f"{BOLD}{GREEN}[DRM] Executing action: {action} with args: {args}{RESET}")
        logger.notice(f"{BOLD}{GREEN}[DRM] Reason: {reason}{RESET}")
        logger.notice(f"{BOLD}{GREEN}[DRM] Guidance (G): {guidance}{RESET}")
        log_str = f"### action: {action}\n### args: {args}\n##### {reason}\n##### guidance: {guidance}"
        self.agent.rerun_visualizer.log_text("robot_monologue", log_str)
        action_obj = ActionFactory.create(action, self, self.irm, args, self.binder_config)
        action_obj.run()
        self.drm_call_count += 1
    
    
    def get_drm_memory(self):
        return self.memory


    def _build_llm_prompt(self):
        seen_objects = set(self.memory['object_positions'].keys())
        prompt = textwrap.dedent(f"""\
            You are a high-level robotic decision-making agent.
            Your job is to choose the next action for a robot that is performing the following task:

            TASK: {self.task_prompt}
            OBJECT-TO-RECEPTACLE MAPPING: {self.memory["object_receptacle_mapping"]}

            Task Rules:
            - The robot cannot grasp an object before going to its location.
            - After grasping an object, the robot must move to or find its correct receptacle, and then place the object on the receptacle.
            - Only interact with the correct receptacle based on the currently held object.
            - *IMPORTANT* The object-to-receptacle mapping is defined in the task description.
              - For example, if the task description says "Place the black toy in the blue box",
                then the robot should place the black toy in the **blue box**.
            - If the robot has just performed 'goto_target_point' to move to an object's location, it should perform '_grasp' or '_place' action next.
            - The robot can hold only one object at a time. That is, if the robot is holding an object, it should perform '_place' action with **corresponding receptacle** next.

            The robot has previously taken these actions:
        """)
        for i, history_dict in enumerate(self.action_history):
            prompt += f"{i+1}. action: {history_dict['action']}\n"
            prompt += f"   - arguments: {history_dict['arguments']}\n"
            prompt += f"   - status: {history_dict['status']}\n"
            prompt += f"   - result: {history_dict['result']}\n"
        
        prompt += f"Currently localized (position-known) objects: {str(seen_objects)}\n"
        prompt += textwrap.dedent(f"""\
            These are the objects the robot already knows the positions of.
            You may consider using them as target_object or target_receptacle accordingly.

            You must now decide the next action the robot should take. You can choose from the following actions:

            1. goto_target_point(target_object: str)
            - This function moves the robot to the location of a previously localized object.
            - You can only use this action if the target_object is in the set seen_objects - {str(seen_objects)}
            - DO NOT choose any object that is not in seen_objects
            - target_object is the name of the object, like "black toy".
            - If multiple objects were localized, prioritize moving to one of them (excluding receptacles if possible).

            2. _grasp(target_object: str)
            - This function makes the robot grasp a previously localized object.
            - target_object is the name of the object, like "black toy".

            3. _place(target_receptacle: str)
            - This function places the currently held object onto the specified receptacle.
            - target_receptacle is the name of the receptacle, like "blue box".
            - Can only be used **after the robot has grasped the object and moved near the target receptacle**.

            4. explore_environment(target_object: str)
            - This function makes the robot explore the environment to search for the specified object.
            - This action is only allowed if the target_object is **NOT** in seen_objects.
            - Use this when the robot needs to find a specific object (e.g., a receptacle that has not yet been localized).

            Return the next action you want the robot to perform as a Python dictionary.
            Use the following format:


            {{
                "action": "goto_target_point",
                "args": {{
                    "target_object": "black toy",
                }},
                "reason": "Explanation of the reasoning for the next action based on the visual information and the action history",
                "guidance": "Current action is navigation; alert me when the black toy or any task receptacle becomes visible or appears to have moved."
            }}

            Or:

            {{
                "action": "_grasp",
                "args": {{
                    "target_object": "black toy",
                }},
                "reason": "Explanation of the reasoning for the next action based on the visual information and the action history"
            }}

            Or:

            {{
                "action": "goto_target_point",
                "args": {{
                    "target_object": "blue box",
                }},
                "reason": "Explanation of the reasoning for the next action based on the visual information and the action history"
            }}
                                  
            Or:

            {{
                "action": "_place",
                "args": {{
                    "target_receptacle": "blue box",
                }},
                "reason": "Explanation of the reasoning for the next action based on the visual information and the action history"
            }}

            Or:

            {{
            "action": "explore_environment",
            "args": {{
                "target_object": "blue box"
                }},
                "reason": "Explanation of the reasoning for the next action based on the visual information and the action history"
            }}

            ---
            Below is a visual map of the current environment.
            - Black areas: unexplored regions
            - Green areas: navigable (free) space
            - Red areas: obstacles (not traversable)
            - Blue markers: objects of interest (localized objects, with labels)
            - Yellow marker: robot's current position

            Use both the action history, known object positions, and the **visual information** to decide the next action.

            Every response MUST also include a "guidance" field, as described below.
            {DRM_GUIDANCE_OUTPUT_INSTRUCTION}

            Only return the dictionary in JSON format.
        """)
        return prompt

class ActionFactory:
    @staticmethod
    def create(action_name, drm, irm, args, binder_config):
        args["enable_irm"] = drm.enable_irm
        if action_name == "goto_target_point":
            return GotoTargetPointAction(drm, irm, args, binder_config)
        elif "grasp" in action_name:
            return GraspAction(drm, irm, args, binder_config)
        elif "place" in action_name:
            return PlaceAction(drm, irm, args, binder_config)
        elif action_name == "explore_environment":
            return ExploreEnvironmentAction(drm, irm, args, binder_config)
        else:
            raise ValueError(f"Unknown action {action_name}")
