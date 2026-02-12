"""Stable-Baselines3 RL controller wrapper."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RLController:
    """Load a trained model and provide actions."""

    algo: str
    model_path: str

    def __post_init__(self) -> None:
        algo = self.algo.upper()
        try:
            from stable_baselines3 import DQN, PPO
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise ImportError(
                "stable-baselines3 is required for RLController."
            ) from exc

        if algo == "PPO":
            self._model = PPO.load(self.model_path, device="cpu")
        elif algo == "DQN":
            self._model = DQN.load(self.model_path, device="cpu")
        else:
            raise ValueError(f"Unsupported RL algorithm: {self.algo}")

    def act(self, obs) -> int:
        action, _ = self._model.predict(obs, deterministic=True)
        return int(action)

    def save(self, path: str) -> None:
        self._model.save(path)
