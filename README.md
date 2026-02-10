# Traffic RL Simulation

## Project overview
Traffic RL Simulation is a lightweight, procedural traffic intersection simulator designed for RL experiments. It includes:
- A four-way intersection with lane queues, pedestrians, emergency vehicles, and a pedestrian phase.
- A Gym-compatible environment with configurable observation/action spaces and rewards.
- A fixed-time baseline controller for benchmarking.
- Optional rendering via Pygame, with a headless fast-sim mode for training.

## How to run the demo
1. Install dependencies:
   - `pip install -r requirements.txt`
2. Run the interactive demo:
   - `python src/main.py`

Controls:
- `Space`: Pause/Resume
- `.`: Single-step (when paused)
- `+`/`-`: Speed up/down
- `Tab`: Toggle debug overlays

## How to train the RL agent
1. Ensure headless mode for fast training:
   - Set `RENDER_ENABLED = False` in `src/config/sim_config.py`
2. Train with Stable-Baselines3:
   - `python src/rl/train_agent.py`

## Design tradeoffs
- Discrete movement keeps the simulator simple and fast, but sacrifices smooth dynamics. A smoothing layer can be added later.
- Headless training maximizes throughput but removes visual debugging unless overlays are enabled in render mode.
- Fixed-time baseline is intentionally simple for benchmarking, not optimality, to give a clear RL comparison point.
