"""Run a baseline or RL controller with optional rendering."""

from __future__ import annotations

from config.sim_config import (
    BASELINE_PHASE_DURATION,
    CONTROLLER_MODE,
    HEADLESS_STEPS,
    RENDER_ENABLED,
    RL_ALGO,
    RL_MODEL_PATH,
)
from env.fixed_time_controller import FixedTimeController
from env.traffic_env import TrafficEnv
from rl.rl_controller import RLController


def run() -> None:
    env = TrafficEnv(render_enabled=RENDER_ENABLED, max_steps=HEADLESS_STEPS)
    obs, _ = env.reset()

    mode = CONTROLLER_MODE.lower()
    if mode == "baseline":
        controller = FixedTimeController(phase_duration=BASELINE_PHASE_DURATION)
        controller.reset()

        while True:
            action = controller.act(env.dt)
            obs, _, terminated, truncated, _ = env.step(action)
            if RENDER_ENABLED:
                env.render()
            if terminated or truncated:
                break
    elif mode == "rl":
        controller = RLController(algo=RL_ALGO, model_path=RL_MODEL_PATH)
        while True:
            action = controller.act(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            if RENDER_ENABLED:
                env.render()
            if terminated or truncated:
                break
    else:
        raise ValueError(f"Unsupported controller mode: {CONTROLLER_MODE}")

    env.close()


if __name__ == "__main__":
    run()
