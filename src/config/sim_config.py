"""Simulation configuration and defaults."""

# Simulation seconds per tick. Smaller values increase update frequency.
SIM_DT: float = 0.1

# Real-time multiplier for simulation pacing. For example, 2.0 targets
# running twice as fast as real time, while 0.5 slows it down.
RENDER_SPEED: float = 1.0

# Toggle rendering. When False, run in headless mode with no pygame usage.
RENDER_ENABLED: bool = True

# Number of simulation ticks to run in headless mode.
HEADLESS_STEPS: int = 10000

# Controller selection for run_controller.py: "baseline" or "rl".
CONTROLLER_MODE: str = "baseline"

# Baseline controller configuration.
BASELINE_PHASE_DURATION: float = 5.0
GREEN_PHASE_DURATION: float = 6.0
YELLOW_PHASE_DURATION: float = 2.0

# RL controller configuration.
RL_ALGO: str = "DQN"
RL_MODEL_PATH: str = "models/traffic_rl"
RL_ACTION_REPEAT: int = 5
RL_CURRICULUM_EPISODES: int = 300

# Training configuration.
RL_TRAIN_TIMESTEPS: int = 10000

# Per-direction spawn probabilities (per tick).
SPAWN_RATE_NORTH: float = 0.12
SPAWN_RATE_SOUTH: float = 0.12
SPAWN_RATE_EAST: float = 0.12
SPAWN_RATE_WEST: float = 0.12
SPAWN_RATE_EMERGENCY: float = 0.005
