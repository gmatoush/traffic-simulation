# Traffic Simulator

## Project Overview
Traffic Simulator is a lightweight, procedural four-way intersection simulator designed for reinforcement learning experiments. It includes:
- Four incoming lanes with queues, stop-line logic, and collision handling.
- Emergency vehicles with higher speeds and emergency-aware traffic light behavior.
- A Gym-compatible environment (`TrafficEnv`) with dense reward shaping and optional headless execution.
- A Pygame UI with live training, speed control, spawn-rate toggle, and save/load model controls.

## Run The Demo
1. Install dependencies:
   - `pip install -r requirements.txt`
2. Launch the simulator:
   - `python src/main.py`

UI Controls:
- Play/Stop buttons: Start or pause the live simulation
- Speed slider: 0.5x–2.0x playback speed
- Spawn Rate toggle: LOW / MED / HIGH
- Save/Load Model: Choose a model file via file dialog
- Keyboard: `Space` toggles pause, `.` single-steps when paused

## Train Headless (Fastest)
Training already runs without rendering by default.

Command:
```
python src/rl/train_agent.py
```

## Run A Trained Model (No Render)
Use the controller runner with `--headless` to run a saved model without rendering.

Example:
```
python src/run_controller.py --mode rl --headless --model-path models/traffic_rl_light.zip
```

Notes:
- `--headless` disables rendering for maximum speed.
- If you launch the executable directly, it defaults to the rendered UI.

## Notes
- Reward shaping prioritizes crash avoidance, then emergency clearance, then queue/wait minimization, with small positive bonuses for safe flow.
- The UI automatically speeds up training when the window is minimized/hidden.
