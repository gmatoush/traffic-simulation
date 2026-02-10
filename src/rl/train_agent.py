"""Train a PPO/DQN agent on the traffic environment."""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from config.sim_config import RL_ALGO, RL_MODEL_PATH, RL_TRAIN_TIMESTEPS
from env.traffic_env import TrafficEnv


def _read_key() -> str | None:
    """Return a single key press if available."""
    try:
        import msvcrt  # type: ignore

        if msvcrt.kbhit():
            key = msvcrt.getch()
            try:
                return key.decode("utf-8")
            except Exception:
                return None
        return None
    except Exception:
        import select

        if select.select([sys.stdin], [], [], 0.0)[0]:
            return sys.stdin.read(1)
        return None


def _is_stop_key(key: str | None) -> bool:
    return key in ("y", "Y")


def _is_stats_key(key: str | None) -> bool:
    return key in ("s", "S")


def _is_name_key(key: str | None) -> bool:
    return key in ("n", "N")


def train(algo: str = RL_ALGO, timesteps: int = RL_TRAIN_TIMESTEPS, model_path: str = RL_MODEL_PATH) -> None:
    algo = algo.upper()
    try:
        from stable_baselines3 import DQN, PPO
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "stable-baselines3 is required for training. Install it before running."
        ) from exc

    from stable_baselines3.common.vec_env import DummyVecEnv

    # Headless training: no rendering and no artificial pacing.
    env = DummyVecEnv([lambda: TrafficEnv(render_enabled=False, max_steps=10**9)])
    models_dir = os.path.dirname(model_path) or os.getcwd()

    if algo == "PPO":
        model = PPO("MultiInputPolicy", env, verbose=0)
    elif algo == "DQN":
        model = DQN("MultiInputPolicy", env, verbose=0)
    else:
        raise ValueError(f"Unsupported RL algorithm: {algo}")

    print("Training headless. Press 'y' to stop and save. Press 's' for stats. Press 'n' to set filename.")
    steps = 0
    last_avg_reward = 0.0
    while True:
        model.learn(total_timesteps=256, reset_num_timesteps=False, progress_bar=False)
        steps += 256
        key = _read_key()
        if _is_name_key(key):
            new_name = input("Enter model filename (without extension): ").strip()
            if new_name:
                model_path = os.path.join(models_dir, f"{new_name}.zip")
        if _is_stats_key(key):
            avg_reward = _compute_avg_reward(model, last_avg_reward)
            last_avg_reward = avg_reward
            print("\n" * 35)
            print("Press 'y' to stop and save. Press 's' for stats. Press 'n' to set filename.")
            print(f"Steps: {steps}")
            print(f"Avg reward: {avg_reward:.2f}")
            print(f"Model path: {model_path}")
        if _is_stop_key(key):
            break
    model.save(model_path)


def _compute_avg_reward(model, fallback: float) -> float:
    """Compute average reward from SB3 buffers; fall back to last value."""
    ep_buf = getattr(model, "ep_info_buffer", None)
    if ep_buf:
        rewards = [info.get("r", 0.0) for info in ep_buf if isinstance(info, dict)]
        if rewards:
            return sum(rewards) / len(rewards)
    logger = getattr(model, "logger", None)
    if logger is not None:
        value = logger.name_to_value.get("rollout/ep_rew_mean", None)
        if value is not None:
            try:
                return float(value)
            except Exception:
                pass
    return float(fallback)


if __name__ == "__main__":
    train()
