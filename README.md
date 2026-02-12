# Traffic Simulator

## Project Overview
Traffic Simulator is a lightweight, procedural four-way intersection simulator designed for reinforcement learning experiments. It includes:
- Four incoming lanes with queues, stop-line logic, and collision handling.
- Emergency vehicles with higher speeds and emergency-aware traffic light behavior.
- A Gym-compatible environment (`TrafficEnv`) with dense reward shaping and optional headless execution.
- A Pygame UI for PPO model execution with speed control, spawn-rate toggle, and save/load model controls.

## Run The Demo
1. Install dependencies:
   - `pip install -r requirements.txt`
2. Launch rendered PPO execution mode:
   - `python src/main.py`

UI Controls:
- Play/Stop buttons: Start or pause the live simulation
- Speed slider: 0.5x–2.0x playback speed
- Spawn Rate toggle: LOW / MED / HIGH
- Load Model: load a trained PPO `.zip`
- Train Model button: prompts for episodes + steps per episode + model name, then trains in the background with a branded progress splash
- Keyboard:
  - `Space` toggles pause, `.` single-steps when paused
  - `Ctrl+L` opens PPO model loader

## Train Headless (Fastest)
Training runs outside renderer only, and PPO is required.

Command:
```
python src/rl/train_agent.py --algo PPO
```

Examples:
```
python src/rl/train_agent.py --algo PPO --timesteps 300000 --model-path models/traffic_ppo --fast
python src/rl/train_agent.py --algo PPO --episodes 60 --episode-steps 2000 --model-path models/traffic_ppo_base
```

Recommended base model:
- Name: `models/traffic_ppo_base.zip`
- Loop: `60` episodes with `2000` steps per episode (good default for a reusable startup model)
- Startup behavior: if `models/traffic_ppo_base.zip` exists, renderer loads it automatically on launch.

## Run A Trained Model (No Render)
Use the controller runner with `--headless` to run a saved model without rendering.

Example:
```
python src/run_controller.py --mode rl --algo PPO --headless --model-path models/traffic_ppo.zip
```

Developer training options (terminal):
```
python src/rl/train_agent.py --algo PPO --timesteps 300000 --model-path models/traffic_ppo
python src/rl/train_agent.py --algo PPO --episodes 40 --episode-steps 2000 --model-path models/traffic_ppo_ep
```

Notes:
- `--headless` disables rendering for maximum speed.
- If you launch the executable directly, it defaults to the rendered UI.

## Notes
- Reward shaping prioritizes crash avoidance, then emergency clearance, then queue/wait minimization, with small positive bonuses for safe flow.
- The renderer does not train; it only executes loaded PPO models.
