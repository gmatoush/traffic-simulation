"""Train a PPO agent on the traffic environment (headless only)."""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from config.sim_config import (
    RL_ACTION_REPEAT,
    RL_CURRICULUM_EPISODES,
    RL_ALGO,
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


def _make_env(
    episode_length: int,
    uniform_speed: bool,
    use_curriculum: bool,
    curriculum_episodes: int,
) -> TrafficEnv:
    env = TrafficEnv(
        render_enabled=False,
        max_steps=episode_length,
        action_repeat=RL_ACTION_REPEAT,
        use_curriculum=use_curriculum,
        curriculum_episodes=curriculum_episodes,
        reset_on_crash=False,
        crash_pause_duration=0.0,
    )
    env.world.uniform_speed_enabled = uniform_speed
    return env


def train(
    algo: str = RL_ALGO,
    timesteps: int = RL_TRAIN_TIMESTEPS,
    episodes: int | None = None,
    episode_steps: int | None = None,
    model_path: str = RL_MODEL_PATH,
    uniform_speed: bool = False,
    fast: bool = False,
    eval_interval: int = 5000,
    eval_episodes: int = 5,
) -> None:
    algo = algo.upper()
    if algo != "PPO":
        raise ValueError("Only PPO is supported for training.")
    if eval_interval <= 0:
        raise ValueError("eval_interval must be positive")
    if eval_episodes <= 0:
        raise ValueError("eval_episodes must be positive")
    if episodes is not None and episodes <= 0:
        raise ValueError("episodes must be positive when provided")
    if episode_steps is not None and episode_steps <= 0:
        raise ValueError("episode_steps must be positive when provided")

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.evaluation import evaluate_policy
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "stable-baselines3 is required for training. Install it before running."
        ) from exc

    try:
        import torch
    except Exception:
        torch = None

    episode_length = episode_steps if episode_steps is not None else (2000 if fast else 5000)
    n_envs = 12 if fast else 1
    env = DummyVecEnv(
        [
            (lambda: _make_env(episode_length, uniform_speed, True, RL_CURRICULUM_EPISODES))
            for _ in range(n_envs)
        ]
    )
    eval_env = DummyVecEnv(
        [
            (lambda: _make_env(episode_length, uniform_speed, False, RL_CURRICULUM_EPISODES))
        ]
    )

    models_dir = os.path.dirname(model_path) or os.getcwd()
    checkpoint_dir = os.path.join(models_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = "cpu"
    if (
        torch is not None
        and getattr(torch, "cuda", None) is not None
        and torch.cuda.is_available()
    ):
        device = "cuda"

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=0,
        n_steps=256,
        batch_size=128 if fast else 64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        learning_rate=3e-4,
        device=device,
    )

    print(
        "Headless training started. Press 'y' to stop and save, "
        "'s' for live stats, 'n' to rename output."
    )
    trained_steps = 0
    best_eval_reward = float("-inf")
    checkpoint_best_path = os.path.join(checkpoint_dir, "best_model.zip")
    checkpoint_latest_path = os.path.join(checkpoint_dir, "latest_model.zip")
    live_stats = False
    if episodes is not None:
        target_timesteps = max(0, int(episodes) * int(episode_length))
    else:
        target_timesteps = max(0, int(timesteps))

    while True:
        remaining = None if target_timesteps == 0 else (target_timesteps - trained_steps)
        if remaining is not None and remaining <= 0:
            break
        block_steps = eval_interval if remaining is None else min(eval_interval, remaining)

        model.learn(
            total_timesteps=block_steps,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        trained_steps += block_steps

        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=eval_episodes,
            deterministic=True,
            render=False,
        )
        try:
            model.save(checkpoint_latest_path)
        except Exception:
            pass
        if mean_reward > best_eval_reward:
            best_eval_reward = mean_reward
            try:
                model.save(checkpoint_best_path)
            except Exception:
                pass

        key = _read_key()
        if _is_name_key(key):
            new_name = input("Enter model filename (without extension): ").strip()
            if new_name:
                model_path = os.path.join(models_dir, f"{new_name}.zip")
        if _is_stats_key(key):
            live_stats = not live_stats
        if _is_stop_key(key):
            break

        completed_episodes = trained_steps / float(max(1, episode_length))
        if target_timesteps == 0:
            target_episodes: str | float = "unbounded"
        else:
            target_episodes = target_timesteps / float(max(1, episode_length))

        if live_stats:
            line = (
                f"\rAlgo: {algo} | Episodes: {completed_episodes:.1f}/{target_episodes} | "
                f"Eval mean: {mean_reward:.2f} +/- {std_reward:.2f} | "
                f"Best eval: {best_eval_reward:.2f} | "
                f"Device: {device} | Model: {os.path.basename(model_path)}"
            )
            sys.stdout.write(line)
            sys.stdout.flush()
        else:
            print(
                f"Episodes: {completed_episodes:.1f} | Eval mean: {mean_reward:.2f} +/- {std_reward:.2f} | "
                f"Best eval: {best_eval_reward:.2f}"
            )

    if live_stats:
        sys.stdout.write("\n")
    model.save(model_path)
    print(f"Saved model to {model_path}.zip")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Headless training runner.")
    parser.add_argument(
        "--algo",
        choices=["PPO", "ppo"],
        default=RL_ALGO,
        help="RL algorithm to train (PPO only).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=RL_TRAIN_TIMESTEPS,
        help="Total training timesteps. Use 0 for unbounded training.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Optional episode count. If set, total steps = episodes * episode-steps.",
    )
    parser.add_argument(
        "--episode-steps",
        type=int,
        default=None,
        help="Optional steps per episode (default: 2000 fast / 5000 normal).",
    )
    parser.add_argument(
        "--model-path",
        default=RL_MODEL_PATH,
        help="Output model path (without extension is fine).",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=5000,
        help="Evaluation cadence in timesteps.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Number of deterministic episodes per evaluation pass.",
    )
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
    train(
        algo=args.algo,
        timesteps=args.timesteps,
        episodes=args.episodes,
        episode_steps=args.episode_steps,
        model_path=args.model_path,
        uniform_speed=args.uniform_speed,
        fast=args.fast,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
    )
