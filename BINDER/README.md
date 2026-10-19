# BINDER: Instantly Adaptive Mobile Manipulation with Open-Vocabulary Commands (ICRA 2026)

Official implementation of **"BINDER: Instantly Adaptive Mobile Manipulation with Open-Vocabulary Commands"** (ICRA 2026).

[`Seongwon Cho`](https://seongwon980.github.io/)\* | [`Daechul Ahn`](https://dcahn12.github.io/)\* | [`Donghyun Shin`](https://github.com/Donghyun1228) | [`Hyeonbeom Choi`](https://hyeonbeomchoi.github.io) | [`San Kim`](https://mounKim.github.io/) | [`Jonghyun Choi`](https://ppolon.github.io/)†

<sub><span style="color:gray">\* Equal contribution &nbsp;&nbsp; † Corresponding author</span></sub>

BINDER is a fork of [Stretch AI](https://github.com/hello-robot/stretch_ai) for the Hello Robot
**Stretch 3**. It turns a natural-language instruction (e.g. *"put the toy in the blue box, the
clothes in the green basket"*) into an explore → navigate → grasp → place loop, using two modules:

- **DRM** (`src/deliberate_reasoning.py`): an OpenAI LLM planner that picks the next action
  from the task, the action history and the map.
- **IRM** (`src/instance_reasoning.py`, `src/irm_task/`): background threads that stream camera
  frames to a Video-LLM (Qwen2.5-VL on vLLM, over ZMQ) during exploration and grasping and send
  events back to the DRM. `src/guidance_parser.py` turns those reports into DRM actions.

The main loop is `src/main_multiple_objects.py`. For a complete DRM↔IRM round trip, read
`src/action/grasp.py`.

## Layout

| Path | Contents |
|------|----------|
| `src/` | The agent: DRM, IRM, `action/`, `irm_task/`, `configs/` |
| `src/stretch/` | Upstream `stretch` package (mapping, navigation, perception, ZMQ client) |
| `src/stretch_ros2_bridge/` | ROS 2 bridge server that runs on the robot |
| `irm/` | Video-LLM ZMQ server (`main.py`, port 1207) |
| `docker/`, `scripts/` | GPU image and launch helpers |
| `third_party/` | Vendored Detectron2 / SAM2 |

## Requirements

- Stretch 3 with ROS 2
- NVIDIA GPU workstation. Tested on an RTX PRO 5000 (Blackwell) and an RTX 6000 Ada with
  driver 580.
- Docker with the NVIDIA Container Toolkit (`scripts/install_nvidia_container_toolkit.sh`)
- `OPENAI_API_KEY` exported in your shell

## Running

```bash
# 1. Robot: ROS 2 bridge server
./scripts/run_stretch_ai_ros2_bridge_server.sh

# 2. Workstation: build the image (deps only; source is bind-mounted at /app)
docker build -f docker/Dockerfile.all -t binder-all .

# 3. Workstation: Video-LLM server (ZMQ 1207)
docker run --rm -it --gpus all --shm-size=8g --network host \
  -v "$PWD":/app -v "$HOME/.cache/huggingface":/models/hf \
  -w /app/irm binder-all python main.py

# 4. Workstation: agent
docker run --rm -it --network host -v "$PWD":/app \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -w /app/src binder-all python main_multiple_objects.py
```

Without Docker, run `cd src && pip install -e .[dev] && python main_multiple_objects.py`, and
start `irm/main.py` separately.

## Configuration

Settings live in `src/configs/default_config.yaml`. To override them, copy it to `config.yaml`.

- `network.robot.ip`: the robot's IP (or run `scripts/set_robot_ip.sh <ip>`)
- `network.vlm_server.ip`: the host running the Video-LLM server
- `task.prompt`: the instruction (target objects are parsed from it)
- `task.enable_irm`: turns the IRM on or off

| Port | Used by |
|------|---------|
| 4401–4404, 4409 | Robot observations, actions, state, servoing, IRM messages |
| 1207 | Video-LLM server |
| 5559 | Manipulation server |

## License

See `LICENSE` and `META_LICENSE`. Upstream Stretch AI terms apply to `src/stretch/` and
`third_party/`.

## Acknowledgements

This research was partially supported by AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data (IITP) funded by the Ministry of Science and ICT (2022-0-00077).
