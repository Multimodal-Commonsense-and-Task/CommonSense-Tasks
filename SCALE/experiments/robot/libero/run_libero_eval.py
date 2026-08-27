"""
run_libero_eval.py

Evaluate a SCALE-enabled OpenVLA policy in the LIBERO simulator.

Usage (greedy / SCALE / baseline decoding):
    CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
        --task_suite libero_spatial \
        --decoding_mode scale            # one of: greedy | temp | topk | topp | scale

The checkpoint is auto-selected from `--task_suite` (the official OpenVLA finetuned checkpoint for that
suite); pass `--pretrained_checkpoint` only to override it. SCALE hyperparameters default to the values
in `configs/scale.yaml`; any of them can be overridden on the command line (e.g. `--T0 1.0 --kappa 2.0`).
"""

import os
import sys

# Set EGL rendering for headless mode (before importing any rendering libraries)
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

# Cap CPU threads per process (must be set BEFORE importing numpy/torch/tensorflow). OpenVLA eval is
# GPU-bound, so this barely affects single-run speed but prevents CPU oversubscription when running many
# evals in parallel (e.g. one per GPU). Override by exporting these env vars before launching.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ.setdefault(_v, "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
import yaml
from libero.libero import benchmark

import wandb

# Append repo root so the interpreter can find the `experiments.robot` package
sys.path.append(str(Path(__file__).resolve().parents[3]))
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

# SCALE default-hyperparameter file (paper values for OpenVLA)
SCALE_CONFIG_PATH = str(Path(__file__).resolve().parents[3] / "configs" / "scale.yaml")
# SCALE hyperparameters: any left as None on the CLI are filled from `scale_config`
SCALE_PARAMS = ("T0", "kappa", "alpha", "epsilon", "num_logits", "attn_sensitivity")
VALID_DECODING_MODES = ("greedy", "temp", "topk", "topp", "scale")


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Checkpoint (local dir or HF Hub ID). If empty, auto-selected from `task_suite`
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # Decoding parameters
    #################################################################################################################
    decoding_mode: str = "greedy"                    # One of: greedy | temp | topk | topp | scale

    # -- Baseline sampling hyperparameters --
    temperature: float = 1.0                         # "temp"/"topk": sampling temperature T
    top_k: int = 40                                  # "topk": number of top tokens to keep (k)
    top_p: float = 0.9                               # "topp": nucleus probability mass (p)

    # -- SCALE hyperparameters (None => load default from `scale_config`) --
    scale_config: str = SCALE_CONFIG_PATH            # Path to SCALE default-hyperparameter YAML
    T0: Optional[float] = None                       # base/max sampling temperature; tau = T0*sigmoid(u)  (Eq. 4)
    kappa: Optional[float] = None                    # visual-attention bound; gamma in (1/kappa, kappa)   (Eq. 8)
    alpha: Optional[float] = None                    # EMA temporal smoothing of step-level uncertainty    (Eq. 7)
    epsilon: Optional[float] = None                  # smoothing of the one-hot reference q_low
    num_logits: Optional[int] = None                 # # of top action logits used to compute u
    attn_sensitivity: Optional[float] = None         # sensitivity inside tanh(attn_sensitivity * delta_u)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite: str = "libero_spatial"               # Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    task_ids: Optional[str] = None                   # Comma-separated task IDs (e.g. "0,1,2"). If None, evaluates all tasks
    more_time: bool = False                          # If True, multiply max_steps by 1.5

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    save_video: bool = True                          # Save an MP4 replay of each episode

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


def _apply_scale_defaults(cfg: GenerateConfig) -> None:
    """Fill any unset SCALE hyperparameter (None) from the SCALE config YAML."""
    with open(cfg.scale_config, "r") as f:
        defaults = yaml.safe_load(f)
    for name in SCALE_PARAMS:
        if getattr(cfg, name) is None:
            setattr(cfg, name, defaults[name])


def _build_run_id(cfg: GenerateConfig) -> str:
    """Construct a descriptive run id (mode + relevant hyperparameters)."""
    run_id = f"EVAL-{cfg.task_suite}-{cfg.model_family}-{cfg.decoding_mode}"
    if cfg.decoding_mode == "temp":
        run_id += f"--t_{cfg.temperature}"
    elif cfg.decoding_mode == "topk":
        run_id += f"--k_{cfg.top_k}--t_{cfg.temperature}"
    elif cfg.decoding_mode == "topp":
        run_id += f"--p_{cfg.top_p}"
    elif cfg.decoding_mode == "scale":
        run_id += f"--T0_{cfg.T0}--kappa_{cfg.kappa}--alpha_{cfg.alpha}--s_{cfg.attn_sensitivity}"
    if cfg.task_ids is not None:
        run_id += f"--tasks_{cfg.task_ids.replace(',', '_')}"
    run_id += f"--seed_{cfg.seed}--{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    return run_id


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.decoding_mode in VALID_DECODING_MODES, (
        f"Unknown decoding_mode '{cfg.decoding_mode}'. Choose from {VALID_DECODING_MODES}."
    )

    # Auto-select the official OpenVLA finetuned checkpoint for the task suite (override with --pretrained_checkpoint)
    if not cfg.pretrained_checkpoint:
        cfg.pretrained_checkpoint = f"openvla/openvla-7b-finetuned-{cfg.task_suite.replace('_', '-')}"
        print(f"[*] No --pretrained_checkpoint given; using {cfg.pretrained_checkpoint} for task_suite={cfg.task_suite}")
    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Fill SCALE defaults from the YAML config
    _apply_scale_defaults(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = get_processor(cfg) if cfg.model_family == "openvla" else None

    # Initialize local logging
    run_id = _build_run_id(cfg)
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite]()
    num_tasks_in_suite = task_suite.n_tasks

    # Determine which tasks to evaluate
    if cfg.task_ids is not None:
        task_ids_list = [int(tid.strip()) for tid in cfg.task_ids.split(",")]
        for tid in task_ids_list:
            if tid < 0 or tid >= num_tasks_in_suite:
                raise ValueError(f"Task ID {tid} is out of range [0, {num_tasks_in_suite - 1}]")
    else:
        task_ids_list = list(range(num_tasks_in_suite))
    print(f"Task suite: {cfg.task_suite} (evaluating tasks: {task_ids_list})")
    log_file.write(f"Task suite: {cfg.task_suite} (evaluating tasks: {task_ids_list})\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Maximum number of steps per episode (per task suite)
    max_steps_per_suite = {
        "libero_spatial": 220,  # longest training demo has 193 steps
        "libero_object": 280,   # longest training demo has 254 steps
        "libero_goal": 300,     # longest training demo has 270 steps
        "libero_10": 520,       # longest training demo has 505 steps
        "libero_90": 400,       # longest training demo has 373 steps
    }

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(task_ids_list):
        # Get task and its default initial states
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment and set initial state
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            uncertainty_history = None  # SCALE per-episode state (Eq. 6-8); reset every episode

            max_steps = max_steps_per_suite[cfg.task_suite]
            if cfg.more_time:
                max_steps = int(max_steps * 1.5)

            print(f"Starting episode {task_episodes + 1}...")
            log_file.write(f"Starting episode {task_episodes + 1}...\n")
            done = False
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    # Get preprocessed image (and save for the replay video)
                    img = get_libero_image(obs, resize_size)
                    replay_images.append(img)

                    # Prepare observations dict (OpenVLA does not take proprio state as input)
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    # Query model to get action (threads SCALE uncertainty history across the episode)
                    action, step_info = get_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                        uncertainty_history=uncertainty_history,
                    )
                    uncertainty_history = step_info.get("uncertainty_history", uncertainty_history)

                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            if cfg.save_video:
                save_rollout_video(
                    replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
                )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Log per-task results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

    # Final summary
    overall_sr = float(total_successes) / float(total_episodes) if total_episodes else 0.0
    print("\n" + "=" * 80)
    print(f"EVALUATION SUMMARY  ({cfg.task_suite}, decoding={cfg.decoding_mode})")
    print(f"Total Episodes: {total_episodes} | Total Successes: {total_successes} | "
          f"Overall Success Rate: {overall_sr * 100:.1f}%")
    print("=" * 80)
    log_file.write(f"\nOverall success rate: {overall_sr * 100:.1f}% ({total_successes}/{total_episodes})\n")
    log_file.close()

    if cfg.use_wandb:
        wandb.log({"success_rate/total": overall_sr, "num_episodes/total": total_episodes})
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()
