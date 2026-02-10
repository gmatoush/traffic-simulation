"""Train a PPO/DQN agent on the traffic environment."""

from __future__ import annotations

from config.sim_config import RL_ALGO, RL_MODEL_PATH, RL_TRAIN_TIMESTEPS
from env.traffic_env import TrafficEnv


def train(algo: str = RL_ALGO, timesteps: int = RL_TRAIN_TIMESTEPS, model_path: str = RL_MODEL_PATH) -> None:
    algo = algo.upper()
    try:
        from stable_baselines3 import DQN, PPO
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "stable-baselines3 is required for training. Install it before running."
        ) from exc

    env = TrafficEnv(render_enabled=False, max_steps=timesteps)

    if algo == "PPO":
        model = PPO("MultiInputPolicy", env, verbose=1)
    elif algo == "DQN":
        model = DQN("MultiInputPolicy", env, verbose=1)
    else:
        raise ValueError(f"Unsupported RL algorithm: {algo}")

    model.learn(total_timesteps=timesteps)
    model.save(model_path)


if __name__ == "__main__":
    train()
