
import time
import timeit
import zmq
import queue
from collections import deque
from threading import Lock, Thread
from logging_utils import get_logger
from sentence_transformers import SentenceTransformer
from nltk import download
from irm_task.base import TaskStatus
from typing import Dict, Any
download('wordnet')
download('omw-1.4')
download('punkt')

logger = get_logger(__name__)

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"

class IRM:
    def __init__(
            self,
            robot,
            vlm1_ip=None,
            vlm1_port=None,
            fps=None,
            num_frames_for_description=None,
            num_frames_for_irm=None,
            enable_irm=False,
            binder_config: Dict[str, Any] = None,
    ):
        # Load from config if not provided
        self.binder_config = binder_config
        self.robot = robot
        self.vlm1_ip = vlm1_ip or self.binder_config['network']['vlm_server']['ip']
        self.vlm1_port = vlm1_port or self.binder_config['network']['vlm_server']['port']
        self._fps = fps or self.binder_config['irm']['fps']
        self._capture_interval = 1.0 / self._fps
        self._ee_capture_interval = 1.0 / self._fps
        self._num_frames_for_description = num_frames_for_description or self.binder_config['irm']['num_frames_for_description']
        self._num_frames_for_irm = num_frames_for_irm or self.binder_config['irm']['num_frames_for_irm']
        self.enable_irm = enable_irm
        self.current_irm_tasks = []
        self.event_queue = queue.Queue()

        self.false_positive_force_stop = False

        # Image buffer to save real-time images
        self._image_buffer_lock = Lock()
        self._image_buffer = deque(maxlen=100)
        self._start_time = timeit.default_timer()

        self._ee_image_buffer_lock = Lock()
        self._ee_image_buffer = deque(maxlen=100)
        self._ee_start_time = timeit.default_timer()

        self.start_image_buffer_update_loop = False
        self.start_ee_image_buffer_update_loop = False

        self.context = zmq.Context()

        if self.enable_irm:
            self.prepare_irm()
        
        if self.enable_irm:
            self._image_buffer_update_thread = Thread(target=self.image_buffer_update_loop)
            self._image_buffer_update_thread.start()
            self._ee_image_buffer_update_thread = Thread(target=self.ee_image_buffer_update_loop)
            self._ee_image_buffer_update_thread.start()

        if self.enable_irm:
            logger.info(f"{BOLD}{GREEN}[IRM] Starting IRM thread{RESET}")
            self._irm_thread = Thread(target=self.update_loop)
            self._irm_thread.start()

    def unset_tasks(self, is_interrupted=False):
        for task in self.current_irm_tasks:
            task.active = False
        self.current_irm_tasks = []
        logger.info(f"Unsetted tasks: {self.current_irm_tasks} - is_interrupted: {is_interrupted}")

    def set_tasks(self, task_list: list):
        if task_list is not None:
            logger.info(f"Setting {len(task_list)} IRM tasks: {[task.__class__.__name__ for task in task_list]}")
            for task in task_list:
                task.active = True
                logger.info(f"Activated task: {task.__class__.__name__}")
            self.current_irm_tasks = task_list
        else:
            logger.info("No IRM tasks to set (task_list is None)")
            self.current_irm_tasks = []

    def set_image_buffer_update_loop(self, start: bool):
        self.start_image_buffer_update_loop = start

    def set_ee_image_buffer_update_loop(self, start: bool):
        self.start_ee_image_buffer_update_loop = start

    def prepare_irm(self):
        logger.info(f"IRM: connecting to VLM1 at {self.vlm1_ip}:{self.vlm1_port}")
        self.model = SentenceTransformer(self.binder_config['models']['sentence_transformer']['model_name'])

        self.vlm1_socket = self.context.socket(zmq.REQ)
        self.vlm1_socket.setsockopt(zmq.RCVTIMEO, 30000)
        self.vlm1_socket.setsockopt(zmq.SNDTIMEO, 5000)
        self.vlm1_socket.setsockopt(zmq.LINGER, 0)

        try:
            self.vlm1_socket.connect(f"tcp://{self.vlm1_ip}:{self.vlm1_port}")
            logger.info(f"Successfully connected to VLM socket at {self.vlm1_ip}:{self.vlm1_port}")
        except Exception as e:
            logger.error(f"Failed to connect to VLM socket: {e}")
            raise
    
    def image_buffer_update_loop(self, verbose=False):
        logger.info(f"Image buffer update loop started. self.robot.running: {self.robot.running}, self.start_image_buffer_update_loop: {self.start_image_buffer_update_loop}")
        loop_count = 0
        while self.robot.running:
            time.sleep(self._capture_interval)
            if not self.start_image_buffer_update_loop:
                if loop_count % 50 == 0:
                    logger.debug(f"Image buffer update loop waiting... (start_image_buffer_update_loop={self.start_image_buffer_update_loop})")
                loop_count += 1
                continue
            loop_count = 0
            current_obs = self.robot._obs
            current_obs_rgb = current_obs["rgb"]
            timestamp = current_obs["timestamp"]
            if current_obs_rgb is None:
                logger.warning(f"Image buffer update loop: No image found")
            else:
                with self._image_buffer_lock:
                    self._image_buffer.append({
                        "timestamp": timestamp,
                        "rgb": current_obs_rgb,
                        "depth": current_obs["depth"],
                        "camera_K": current_obs["camera_K"],
                        "camera_pose": current_obs["camera_pose"],
                        "used_by": set()
                    })
                    if len(self._image_buffer) % 10 == 1:
                        logger.info(f"Added image to buffer. Buffer size now: {len(self._image_buffer)}")
                if verbose:
                    logger.info(f"Image buffer update loop: Image buffer size: {len(self._image_buffer)}")

    def ee_image_buffer_update_loop(self, verbose=False):
        logger.info(f"self.robot.running: {self.robot.running}, self.start_ee_image_buffer_update_loop: {self.start_ee_image_buffer_update_loop}")
        while self.robot.running:
            time.sleep(self._ee_capture_interval)
            if not self.start_ee_image_buffer_update_loop:
                continue
            current_obs_servo = self.robot._servo
            current_obs_rgb = current_obs_servo.ee_rgb
            timestamp = current_obs_servo.timestamp
            if current_obs_rgb is None:
                logger.warning("EE image buffer update loop: No image found")
            else:
                with self._ee_image_buffer_lock:
                    self._ee_image_buffer.append({
                        "timestamp": timestamp,
                        "rgb": current_obs_rgb,
                        "used_by": set()
                    })
                if verbose:
                    logger.info(f"EE image buffer update loop: Image buffer size: {len(self._ee_image_buffer)}")
    
    
    def request_stop_all_tasks_and_wait(self, timeout=10.0):
        for task in self.current_irm_tasks:
            task.request_stop()
        logger.notice(f"{BOLD}{RED}Requested to stop all tasks, waiting for them to finish...{RESET}")
        start_time = time.time()
        while time.time() - start_time < timeout:
            time.sleep(0.1)
            if len(self.current_irm_tasks) == 0 or all(task.get_status() == TaskStatus.STOPPED for task in self.current_irm_tasks):
                logger.notice(f"{BOLD}{GREEN}All tasks finished in {time.time() - start_time:.2f} seconds, go back to main thread...{RESET}")
                return True
        logger.warning(f"{BOLD}{RED}Timeout: {timeout} seconds passed, but some tasks are still running{RESET}")
        return False

    def update_loop(self, verbose=True):
        logger.info("IRM update_loop started")
        while self.robot.running:
            time.sleep(0.1)
            current_tasks = self.current_irm_tasks.copy()
            if current_tasks and verbose:
                logger.debug(f"Running {len(current_tasks)} IRM tasks")
            for task in current_tasks:
                try:
                    task.run_step()
                except Exception as e:
                    logger.error(f"Error running IRM task {task.__class__.__name__}: {e}")

                if task.get_status() == TaskStatus.STOPPED:
                    self.current_irm_tasks.remove(task)
                    logger.notice(f"Removed task: {task}")