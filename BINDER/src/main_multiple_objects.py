import warnings; warnings.filterwarnings("ignore", message=".*asarray.*copy.*")
import os
import pytz
from datetime import datetime
from logging_utils import setup, get_logger
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core.parameters import get_parameters
from stretch.agent.robot_agent_dynamem import RobotAgent
from deliberate_reasoning import DRM
from action.grasp import BaseAction
from configs.config import BINDER_CONFIG

# Set OpenAI API key from config
os.environ["OPENAI_API_KEY"] = BINDER_CONFIG['credentials']['openai_api_key']


def init_logging_once():
    log_dir = os.environ.get("RUN_LOG_DIR")
    if not log_dir:
        timezone = pytz.timezone(BINDER_CONFIG['logging']['timezone'])
        current_datetime = datetime.now(timezone)
        log_dir = f"{BINDER_CONFIG['logging']['base_dir']}/{current_datetime.strftime('%m-%d_%H-%M-%S')}"
        os.makedirs(log_dir, exist_ok=True)
        os.environ["RUN_LOG_DIR"] = log_dir

    setup(log_path=f"{log_dir}/run.log")
    logger = get_logger(__name__)
    return log_dir, logger


# Load parameters
parameters = get_parameters("dynav_config.yaml")

def main(log_dir):
    robot = HomeRobotZmqClient(
        robot_ip=BINDER_CONFIG['network']['robot']['ip'],
        doridori=BINDER_CONFIG['robot']['doridori'],
        log_dir=log_dir,
        binder_config=BINDER_CONFIG
    )

    # Create agent - pass essential params directly, others via CONFIG
    agent = RobotAgent(
        robot=robot,
        parameters=parameters,
        semantic_sensor=None,
        log=log_dir,
        server_ip=BINDER_CONFIG['network']['manipulation_server']['ip'],
        vlm1_ip=BINDER_CONFIG['network']['vlm_server']['ip'],
        vlm1_port=BINDER_CONFIG['network']['vlm_server']['port'],
        manip_port=BINDER_CONFIG['network']['manipulation_server']['port'],
        re=BINDER_CONFIG['robot']['edition'],
        mllm=BINDER_CONFIG['task']['use_mllm'],
        manipulation_only=BINDER_CONFIG['task']['manipulation_only'],
        enable_irm=BINDER_CONFIG['task']['enable_irm'],
        binder_config=BINDER_CONFIG
    )
    agent.start()
    robot.agent = agent

    drm = DRM(
        agent=agent,
        irm=agent.irm,
        enable_irm=BINDER_CONFIG['task']['enable_irm'],
        binder_config=BINDER_CONFIG
    )
    drm.set_task_prompt(BINDER_CONFIG['task']['prompt'])
    BaseAction(drm, drm.irm, None, binder_config=BINDER_CONFIG).look_around_and_update()
    while True:
        drm.execute_next_step()

if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method('fork')
    except RuntimeError:
        pass

    log_dir, logger = init_logging_once()
    main(log_dir)