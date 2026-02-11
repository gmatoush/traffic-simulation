"""Train a PPO/DQN agent on the traffic environment."""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from config.sim_config import (
    RL_ACTION_REPEAT,
    RL_ALGO,
    RL_CURRICULUM_EPISODES,
    RL_MODEL_PATH,
    RL_TRAIN_TIMESTEPS,
)
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


def train(
    algo: str = RL_ALGO,
    timesteps: int = RL_TRAIN_TIMESTEPS,
    model_path: str = RL_MODEL_PATH,
    uniform_speed: bool = False,
    fast: bool = False,
) -> None:
    # Keep headless training fully compatible with the rendered UI model.
    algo = "DQN"
    try:
        from stable_baselines3 import DQN, PPO
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "stable-baselines3 is required for training. Install it before running."
        ) from exc
    try:
        import torch
    except Exception:
        torch = None

    from stable_baselines3.common.vec_env import DummyVecEnv

    # Headless training: no rendering and no artificial pacing.
    # Use finite episodes and loop forever across episodes.
    episode_length = 2000 if fast else 5000
    n_envs = 12 if fast else 1

    def _make_env():
        env = TrafficEnv(
            render_enabled=False,
            max_steps=episode_length,
            action_repeat=RL_ACTION_REPEAT,
            use_curriculum=True,
            curriculum_episodes=RL_CURRICULUM_EPISODES,
        )
        env.world.uniform_speed_enabled = uniform_speed
        return env

    env = DummyVecEnv([_make_env for _ in range(n_envs)])
    models_dir = os.path.dirname(model_path) or os.getcwd()
    checkpoint_dir = os.path.join(models_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_freq = 4 if fast else 1
    gradient_steps = 4 if fast else 1
    batch_size = 64 if fast else 32
    buffer_size = 50000 if fast else 5000
    learning_starts = 1000 if fast else 100

    device = "cpu"
    if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        device = "cuda"

    model = DQN(
        "MultiInputPolicy",
        env,
        verbose=0,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        device=device,
    )

    print("Training headless. Press 'y' to stop and save. Press 's' for stats. Press 'n' to set filename.")
    episodes = 0
    last_avg_reward = 0.0
    best_avg_reward = float("-inf")
    reward_buffer: list[float] = []
    checkpoint_best_path = os.path.join(checkpoint_dir, "best_model.zip")
    checkpoint_latest_path = os.path.join(checkpoint_dir, "latest_model.zip")
    checkpoint_interval_episodes = 5
    live_stats = False

    class RewardTracker:
        def __init__(self, buf):
            self.buf = buf

        def __call__(self, locals_, globals_):
            rewards = locals_.get("rewards", None)
            if rewards is not None:
                try:
                    if hasattr(rewards, "__len__"):
                        self.buf.append(float(sum(rewards) / len(rewards)))
                    else:
                        self.buf.append(float(rewards))
                    if len(self.buf) > 2000:
                        self.buf.pop(0)
                except Exception:
                    pass
            return True

    reward_cb = RewardTracker(reward_buffer)
    while True:
        # Train in full-episode chunks so loops align to episode boundaries.
        model.learn(
            total_timesteps=episode_length * n_envs,
            reset_num_timesteps=False,
            progress_bar=False,
            callback=reward_cb,
        )
        episodes += 1
        avg_reward = _compute_avg_reward(model, last_avg_reward)
        if reward_buffer:
            avg_reward = sum(reward_buffer) / len(reward_buffer)
        last_avg_reward = avg_reward

        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            try:
                model.save(checkpoint_best_path)
            except Exception:
                pass
        if episodes % checkpoint_interval_episodes == 0:
            try:
                model.save(checkpoint_latest_path)
            except Exception:
                pass

        key = _read_key()
        if _is_name_key(key):
            new_name = input("Enter model filename (without extension): ").strip()
            if new_name:
                model_path = os.path.join(models_dir, f"{new_name}.zip")
        if _is_stats_key(key):
            live_stats = not live_stats
            if live_stats:
                print("\n" * 5)
                print("Live stats ON (press 's' to stop updates). Press 'y' to stop and save. Press 'n' to set filename.")
            else:
                print("\n" * 2)
                print("Live stats OFF. Press 's' to show updates again.")
        if _is_stop_key(key):
            if live_stats:
                sys.stdout.write("\n")
            break
        if live_stats:
            line = (
                f"\rEpisodes: {episodes} | "
                f"Avg reward (rolling): {avg_reward:.2f} | "
                f"Best: {best_avg_reward:.2f} | "
                f"Device: {device} | "
                f"Model: {os.path.basename(model_path)}"
            )
            sys.stdout.write(line)
            sys.stdout.flush()
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
    import argparse

    parser = argparse.ArgumentParser(description="Headless training runner.")
    parser.add_argument(
        "--uniform-speed",
        action="store_true",
        help="Use uniform vehicle speed for easier training.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable parallel headless training for maximum throughput.",
    )
    args = parser.parse_args()
    train(uniform_speed=args.uniform_speed, fast=args.fast)
