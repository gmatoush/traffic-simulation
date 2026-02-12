# Traffic Simulator RL

Traffic Simulator RL is a 4-way intersection simulator with a PPO-based RL workflow.  
It supports:

- Interactive rendering for model execution
- GUI-triggered background training
- Headless terminal training for faster iteration
- Side-by-side live model comparison

## What This Project Does

- Simulates traffic flow across four incoming lanes (`NORTH`, `SOUTH`, `EAST`, `WEST`)
- Uses a traffic light controller with green/yellow phases
- Spawns normal and emergency vehicles
- Detects collisions near the intersection
- Exposes a Gym-compatible RL environment (`TrafficEnv`)
- Trains/evaluates PPO policies with Stable-Baselines3

## Core Architecture

- `src/main.py`
  - Primary GUI application (start page + simulation + comparison + GUI training flow)
- `src/env/traffic_env.py`
  - Gym/Gymnasium environment, observations, reward shaping, action handling
- `src/sim/world.py`
  - World state, vehicle spawning, queue updates, collision logic, counters
- `src/render/pygame_renderer.py`
  - Pygame rendering, HUD/control panel, comparison UI
- `src/rl/train_agent.py`
  - Headless PPO trainer for terminal/dev workflow
- `src/run_controller.py`
  - Baseline/RL runner with optional headless mode

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:

- `gymnasium`
- `stable-baselines3`
- `pygame`
- `numpy`

## Quick Start

Launch the main app:

```bash
python src/main.py
```

At startup, choose one mode:

1. `Run Simulation`
2. `Train Model`
3. `Compare Models`

Press `C` in major UI views to return to the start page.

## Mode Details

### 1) Run Simulation

Purpose: load a PPO model and run it in rendered simulation.

Control panel features:

- Speed slider: `0.5x` to `2.0x`
- Spawn Rate toggle: `LOW / MED / HIGH` (reset on change)
- Play / Stop buttons
- Load Model button
- Train Model button
- Compare Models button
- `Model: <name>` display

Keyboard:

- `Space`: pause/resume
- `.`: single-step when paused
- `Ctrl+L`: open model picker
- `+ / -`: adjust simulation speed
- `R`: cycle spawn risk level and reset
- `C`: return to start page

### 2) Train Model (GUI)

Purpose: start PPO training from GUI without leaving app.

Flow:

1. Enter episodes
2. Enter steps per episode
3. Enter model name
4. Training starts in background and shows branded progress view

Behavior:

- Press `C` while on training view to return to start page
- Training continues in background
- You can use other app modes while training is still running

### 3) Compare Models

Purpose: compare two PPO models side-by-side with live metrics.

Flow:

1. Select first model
2. Select second model
3. Comparison screen runs both models headlessly and updates live stats

Comparison UI includes:

- Shared elapsed timer (single global timer)
- Per-model stat boxes
- Comparison speed slider: `0.5x` to `10.0x`
- Centered legend

Metric coloring:

- Green = better
- Red = worse
- Yellow = tie

Current comparison metrics:

- Avg stopped wait (lower is better)
- Vehicles per minute (higher is better)
- Crashes per minute (lower is better)
- Vehicles cleared (higher is better)

Notes:

- Stats remain cumulative across crash resets during comparison
- Press `C` to return to start page

## Headless Terminal Training (Recommended for Speed)

Use:

```bash
python src/rl/train_agent.py --algo PPO
```

Useful examples:

```bash
python src/rl/train_agent.py --algo PPO --timesteps 300000 --model-path models/traffic_ppo --fast
python src/rl/train_agent.py --algo PPO --episodes 60 --episode-steps 2000 --model-path models/traffic_ppo_base
```

Trainer notes:

- PPO-only (`--algo PPO`)
- Supports both `--timesteps` and episode-based training (`--episodes`, `--episode-steps`)
- Periodic evaluation during training
- Checkpoints saved under `models/checkpoint/`
- Interactive keys during terminal training:
  - `y`: stop and save
  - `s`: toggle live stats line
  - `n`: rename output model

## Running Controllers Without GUI

You can run baseline or RL controller from terminal:

```bash
python src/run_controller.py --mode baseline --headless
python src/run_controller.py --mode rl --algo PPO --model-path models/traffic_ppo_base.zip --headless
```

## Training Defaults and Base Model

Recommended starter base model:

- Path: `models/traffic_ppo_base.zip`
- Example loop: `60` episodes, `2000` steps/episode

Main app behavior:

- If base model exists and you choose run mode, it is used as preferred startup model path.

## Environment and Reward Summary

`TrafficEnv` observations include normalized queue/wait state, phase timing/switchability, emergency presence, and current phase.

Reward shaping emphasizes:

- Heavy crash penalties (dominant)
- Emergency wait penalties
- Queue/wait penalties
- Small bonuses for no-crash/flow/clearing behavior

This keeps safety primary while still encouraging throughput and reduced waiting.

## Performance Notes

- Headless training is much faster than rendering
- GUI training runs on CPU and can reduce responsiveness on low-end machines
- Comparison mode at high speed (`10x`) increases background simulation load
- Keep model size moderate for smoother rendering on weaker hardware

## Project Structure (High-Level)

```text
src/
  config/
  env/
  render/
  rl/
  sim/
  main.py
  run_controller.py
assets/
models/
requirements.txt
README.md
```

## Troubleshooting

- If model selection fails:
  - Ensure `.zip` model exists and is a valid PPO checkpoint
- If training appears slow:
  - Use terminal training with `--fast`
  - Lower episode length and total episodes while testing
- If rendering lags:
  - Reduce speed and spawn pressure
  - Use a smaller/lighter model checkpoint

## License / Usage

Use this project for experimentation, prototyping, and RL learning workflows.  
Add your own license terms if this repository is public/distributed.
