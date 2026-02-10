"""Run a baseline or RL controller with optional rendering."""

from __future__ import annotations

import argparse

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
    parser = argparse.ArgumentParser(description="Run a controller with optional rendering.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable rendering for fastest execution.",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "rl"],
        default=CONTROLLER_MODE.lower(),
        help="Controller mode to run.",
    )
    parser.add_argument(
        "--algo",
        default=RL_ALGO,
        help="RL algorithm (e.g., DQN or PPO).",
    )
    parser.add_argument(
        "--model-path",
        default=RL_MODEL_PATH,
        help="Path to a saved RL model.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=HEADLESS_STEPS,
        help="Maximum number of steps to run.",
    )
    args = parser.parse_args()

    render_enabled = RENDER_ENABLED and not args.headless
    env = TrafficEnv(render_enabled=render_enabled, max_steps=args.steps)
    obs, _ = env.reset()

    mode = args.mode
    if mode == "baseline":
        controller = FixedTimeController(phase_duration=BASELINE_PHASE_DURATION)
        controller.reset()

        while True:
            action = controller.act(env.dt)
            obs, _, terminated, truncated, _ = env.step(action)
            if render_enabled:
                env.render()
            if terminated or truncated:
                break
    elif mode == "rl":
        controller = RLController(algo=args.algo, model_path=args.model_path)
        while True:
            action = controller.act(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            if render_enabled:
                env.render()
            if terminated or truncated:
                break
    else:
        raise ValueError(f"Unsupported controller mode: {CONTROLLER_MODE}")

    env.close()


if __name__ == "__main__":
    run()
