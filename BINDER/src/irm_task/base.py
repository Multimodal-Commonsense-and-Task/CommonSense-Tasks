"""Base class for IRM (Instance Reasoning Module) tasks.

This module provides the base class for all IRM tasks that run
background monitoring during action execution.
"""

from enum import Enum, auto
from threading import Lock
from typing import TYPE_CHECKING

from logging_utils import get_logger

if TYPE_CHECKING:
    from instance_reasoning import IRM
    from stretch.agent.robot_agent_dynamem import RobotAgent
    from stretch.agent.zmq_client import HomeRobotZmqClient

logger = get_logger(__name__)


class TaskStatus(Enum):
    """Status of an IRM task (internal task lifecycle)."""

    ACTIVE = auto()
    PAUSED = auto()
    STOPPING = auto()
    STOPPED = auto()


class ExecutionMode(Enum):
    """Execution mode m_t that the IRM hands back to the DRM (paper Eq. 3).

    The guidance-conditioned parser (Phi_G, see guidance_parser.py) maps each IRM
    video report Z_t to one of these modes:

    - CONTINUE: nothing actionable; keep executing the current action (e.g. a memory
      update for a non-critical object).
    - ADJUST:   apply a local, in-place correction without full replanning
      (e.g. grasp refinement / gripper re-alignment).
    - REPLAN:   a task-critical change occurred; stop the current action and let the
      DRM replan (e.g. the current target appeared/moved, or a grasp failed).
    """

    CONTINUE = auto()
    ADJUST = auto()
    REPLAN = auto()


class IRMBaseTask:
    """Base class for IRM background tasks.

    IRM tasks run in background threads during action execution,
    monitoring the robot state and emitting events when conditions are met.

    Attributes:
        irm: IRM reference
        robot: Robot client for hardware control
        agent: Robot agent with perception and mapping
        config: Task-specific configuration
        action: Current action name
    """

    def __init__(
        self,
        irm: "IRM",
        robot: "HomeRobotZmqClient",
        agent: "RobotAgent",
        get_drm_memory_fn,
        vlm1_socket,
        image_buffer,
        _ee_image_buffer,
        fps,
        config,
    ):
        """Initialize IRM base task.

        Args:
            irm: IRM
            robot: Robot client
            agent: Robot agent
            get_drm_memory_fn: Function to get DRM memory
            vlm1_socket: VLM socket for inference
            image_buffer: Head camera image buffer
            _ee_image_buffer: End-effector camera image buffer
            fps: Frame rate
            config: Task configuration (action, seen_objects, unseen_objects, etc.)
        """
        self.irm = irm
        self.robot = robot
        self.agent = agent
        self.get_drm_memory_fn = get_drm_memory_fn
        self.vlm1_socket = vlm1_socket
        self.image_buffer = image_buffer
        self._ee_image_buffer = _ee_image_buffer
        self._fps = fps
        self.config = config
        self.action = config["action"]
        self._status = TaskStatus.ACTIVE
        self._lock = Lock()

    def emit_event(self, event_type: str, data: dict, mode: "ExecutionMode" = None):
        """Emit event to the event queue.

        Args:
            event_type: Type of event
            data: Event data payload
            mode: execution mode m_t (CONTINUE/ADJUST/REPLAN) this event corresponds to.
        """
        event = {
            "type": event_type,
            "data": data,
            "mode": mode,
            "source": self.__class__.__name__,
        }
        self.irm.event_queue.put(event)

    def request_stop(self):
        """Request the task to stop gracefully."""
        with self._lock:
            self._status = TaskStatus.STOPPING

    def stop(self):
        """Stop the task immediately."""
        with self._lock:
            self._status = TaskStatus.STOPPED

    def get_status(self) -> TaskStatus:
        """Get current task status."""
        with self._lock:
            return self._status

    def set_status(self, status: TaskStatus):
        """Set task status.

        Args:
            status: New status value
        """
        with self._lock:
            self._status = status

    def should_run(self) -> bool:
        """Check if task should execute this step.

        Override in subclass to implement task-specific conditions.

        Returns:
            bool: True if task should run
        """
        return False

    def run_step(self):
        """Execute one step of the task.

        Override in subclass to implement task logic.
        """
        pass
