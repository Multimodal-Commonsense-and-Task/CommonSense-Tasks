# Image Generation Process

## Prompt Generation
- All related code is located in the `nameonly/prompt_generation` folder.
- Since there are only a few dependencies, they can be installed manually (or use the `generate` environment).

### Ours
1. Use `generate_prompts_gpt4_static_100.py`.
2. Change the OpenAI key.
3. `metaprompt_json_path`: Path to save the result of the first stage (metaprompt JSON).
4. `totalprompt_json_path`: Path to save the result of the second stage (totalprompt JSON), which will be used for image generation.
5. Set the following parameters:
   - `num_metaprompts`
   - `num_prompts_per_metaprompt`
   - `max_prompts` (the total number of prompts to generate)

### synclr, synthclip, fake-f
1. **synclr**: Use `generate_backgrounds_synclr.py` and `generate_prompts_synclr.py`.
   - A background JSON file must be provided for prompt generation.
2. **synthclip**: Use `generate_prompts_synthclip.py`.
3. **fake_f**: Use `generate_prompts_fake_f.py`.

---

## Image Generation
- All related code is located in the `./nameonly/generate_twostage` folder.

### Dependency Installation
1. **For Anaconda (excluding cogview2)**:
   - Run `bash ./new_requirements.sh` to install the dependencies.

2. **For Docker (cogview2)**:
   #### 2.1. If the Docker container (cogview2) does not exist:
   - Run:
     ```bash
     docker run --gpus all -it --ipc=host --net=host -v /home/user:/workspace --name cogview2 pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
     ```
   - Access the container:
     ```bash
     docker exec -it cogview2 bash
     ```
   - Navigate to `./nameonly/generate_twostage` and run:
     ```bash
     bash ./cogview2_setup.sh
     ```
     This will automatically download the required checkpoint and generate a sample image.
   - Exit the container:
     ```bash
     exit
     ```

   #### 2.2. If the Docker container (cogview2) already exists:
   - Start the container if it is not running:
     ```bash
     docker start cogview2
     ```
   - Access the container:
     ```bash
     docker exec -it cogview2 bash
     ```
   - Navigate to `./nameonly/generate_twostage` and run:
     ```bash
     bash ./cogview2_setup.sh
     ```
     This will automatically download the required checkpoint and generate a sample image.

---

### Image Generation
1. Use `run.sh`:
   - **DATASET**: Supported datasets include `PACS`, `cifar10`, `DomainNet`, `ImageNet`, etc.
   - **IMAGE_DIR**: Path to save the generated images.
   - **GENERATIVE_MODEL**: Specify the generative model, such as `sdxl`, `floyd`, `cogview2`, `sd3`, `auraflow`.
   - **START_CLASS, END_CLASS**: Define the range of classes for image generation (important—if a `task.txt` file already exists, the range will be ignored).
   - **PROMPT_DIR**: Path to the directory where the prompt JSON files are stored.
   - **INCREASE_RATIO**: Determines the dataset size relative to `MA` (defined in `classes.py`). A ratio of `1.15x` is recommended.
   - **GPU_ID**: Specify the GPU ID to use. For example, in `{1:-0}`, the default GPU is `0`. Update this or pass it as an argument when running the script.

2. **task.txt File Creation (Important)**:
   - Running `run.sh` will generate a `task.txt` file.
   - This file acts as a queue to manage jobs and their statuses: `pending`, `in_progress`, or `done`.
   - To add more jobs or GPUs without stopping all tasks, modify or delete the `task.txt` file and rerun `run.sh`.
   - **Important**: If you want to change the class range, either edit the `task.txt` file directly or delete it and rerun the script.
