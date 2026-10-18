from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image
import os
from libero.libero import get_libero_path


benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_object" # can also choose libero_spatial, libero_object, etc.
task_suite = benchmark_dict[task_suite_name]()

# retrieve a specific task
task_id = 0
task = task_suite.get_task(task_id)
task_name = task.name
task_description = task.language
task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
      f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

# step over the environment
env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 128,
    "camera_widths": 128
}
env = OffScreenRenderEnv(**env_args)
env.seed(0)
env.reset()
init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
init_state_id = 0
env.set_init_state(init_states[init_state_id])

dummy_action = [0.] * 7

save_dir = "saved_images"
os.makedirs(save_dir, exist_ok=True)

for step in range(10):
    obs, reward, done, info = env.step(dummy_action)

    if step == 9:
        imgs = {
            "agentview": obs["agentview_image"][::-1, ::-1],
            "agentview_right": obs["agentview_right_image"][::-1, ::-1],
            "agentview_left": obs["agentview_left_image"][::-1, ::-1],
            "agentview_right_back": obs["agentview_right_back_image"][::-1, ::-1],
            "agentview_left_back": obs["agentview_left_back_image"][::-1, ::-1],
        }

        for name, img_arr in imgs.items():
            img = Image.fromarray(img_arr)
            img.save(os.path.join(save_dir, f"{name}_step{step}.png"))

env.close()