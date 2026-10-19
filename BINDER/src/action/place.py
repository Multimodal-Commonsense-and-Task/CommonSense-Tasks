from overrides import overrides

from action.base import BaseAction
from action.utils.manipulation import (
    ensure_log_directory,
    save_manipulation_image,
    retract_arm_after_place,
)
from logging_utils import get_logger
from stretch.agent.manipulation.dynamem_manipulation.grasper_utils import (
    capture_and_process_image,
    move_to_point,
)
from utils import compute_tilt

INIT_HEAD_PAN = -1.57
INIT_HEAD_TILT = -0.65

logger = get_logger(__name__)

class PlaceAction(BaseAction):
    def __init__(self, drm, irm, args, binder_config):
        super().__init__(drm, irm, args, binder_config)
        self.manip_wrapper = self.agent.manip_wrapper
        self.manip_socket = self.agent.manip_socket
        self.transform_node = self.agent.transform_node

    @overrides
    def predict_irm_tasks(self):
        return None

    @overrides
    def run(self):
        result = self.before_action_task()
        if not result:
            return
        success = self.execute()
        self.after_action_task()
        self.update_drm_history(success=success)

    def _validate_receptacle(self) -> bool:
        if self.target_receptacle not in self.drm.memory['object_positions']:
            self.drm.action_history.append({
                "action": "place",
                "arguments": {"target_receptacle": self.target_receptacle},
                "status": "failed - Cannot place before localizing receptacle.",
                "result": None,
            })
            return False
        return True

    @overrides
    def before_action_task(self):
        self.irm_tasks = self.predict_irm_tasks()
        if self.irm_tasks is not None:
            self.irm.set_tasks(self.irm_tasks)

        self.target_receptacle = self.args["target_receptacle"]

        if not self._validate_receptacle():
            return False

        self.target_point = self.drm.memory['object_positions'][self.target_receptacle]['xyz']
        return True

    @overrides
    def execute(self):
        base_node = "camera_depth_optical_frame"
        self.robot.switch_to_manipulation_mode()

        camera_xyz = self.robot.get_head_pose()[:3, 3]
        theta = compute_tilt(camera_xyz, self.target_point)

        self.robot.look_at_ee()
        self.manip_wrapper.move_to_position(head_pan=INIT_HEAD_PAN, head_tilt=theta)

        rotation, translation, place_best_pose_image = capture_and_process_image(
            mode="place",
            obj=self.target_receptacle,
            socket=self.manip_socket,
            hello_robot=self.manip_wrapper,
        )

        log_dir = ensure_log_directory(
            self.agent.log, "place", self.drm.drm_call_count
        )

        if place_best_pose_image is not None:
            save_manipulation_image(
                place_best_pose_image,
                log_dir,
                "place_best_pose",
                self.target_receptacle,
                self.agent.rerun_visualizer,
            )

        if rotation is None:
            return False

        self._execute_place_sequence(translation, base_node)

        self.robot.move_to_nav_posture()
        return True

    def _execute_place_sequence(self, translation, base_node: str):
        self.manip_wrapper.move_to_position(lift_pos=1.05)
        self.manip_wrapper.move_to_position(wrist_yaw=0, wrist_pitch=0)

        move_to_point(
            self.manip_wrapper,
            translation,
            base_node,
            self.transform_node,
            move_mode=0,
        )
        self.manip_wrapper.move_to_position(gripper_pos=1, blocking=True)

        retract_arm_after_place(self.manip_wrapper)

    @overrides
    def after_action_task(self):
        pass

    def update_drm_history(self, success: bool):
        # Only release the held object on a successful place. On failure the gripper never
        # opened (execute() returned False before opening), so the robot is still holding
        # it — clearing held_object here would desync memory from reality and strand the
        # object (its position was already popped at grasp time), making it unrecoverable.
        if success:
            self.drm.memory["held_object"] = None
        self.drm.action_history.append({
            "action": "place",
            "arguments": {"target_receptacle": self.target_receptacle},
            "status": "success" if success else "failed",
            "result": None,
        })
        