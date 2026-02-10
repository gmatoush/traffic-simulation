"""Gym-compatible environment wrapper for the traffic simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - fallback for gym
    import gym
    from gym import spaces

from config.sim_config import RENDER_ENABLED, SIM_DT
from env.actions import HOLD_PHASE, SWITCH_PHASE
from sim.traffic_light import TrafficLight, TrafficPhase
from sim.world import Lane, World

@dataclass
class TrafficEnv(gym.Env):
    """Gym-compatible environment for traffic signal control."""

    render_enabled: bool = RENDER_ENABLED
    max_steps: int = 1000
    emergency_wait_penalty: float = 5.0
    crash_penalty: float = 50.0

    def __post_init__(self) -> None:
        self.dt = SIM_DT
        self._steps = 0
        self.world = World(traffic_light=TrafficLight())
        self._renderer = None

        if self.render_enabled:
            from render.pygame_renderer import PygameRenderer

            self._renderer = PygameRenderer()

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Dict(
            {
                "queue_lengths": spaces.Box(
                    low=0, high=1000, shape=(len(Lane),), dtype=int
                ),
                "emergency_waiting": spaces.Discrete(2),
                "light_phase": spaces.Discrete(len(TrafficPhase)),
            }
        )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        self._steps = 0
        self.world = World(traffic_light=TrafficLight())
        return self._get_obs(), {}

    def step(self, action: int) -> tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if action == SWITCH_PHASE and isinstance(self.world.traffic_light, TrafficLight):
            light = self.world.traffic_light
            light.switch_phase(light._next_phase(light.current_phase))

        self.world.update(self.dt)
        self._steps += 1

        reward = -self.total_wait_time()
        if self.world.emergency_waiting():
            reward -= self.emergency_wait_penalty
        reward -= self.crash_penalty * self.world.consume_crash_events()

        terminated = False
        truncated = self._steps >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self) -> None:
        if self._renderer is None:
            return
        self._renderer.render(self.world, sim_speed=1.0)

    def close(self) -> None:
        if self._renderer is None:
            return
        import pygame

        pygame.quit()

    @property
    def steps(self) -> int:
        return self._steps

    def _get_obs(self) -> Dict[str, Any]:
        queue_lengths = [len(self.world.lane_queues[lane]) for lane in Lane]
        emergency_waiting = int(self.world.emergency_waiting())
        phase_index = list(TrafficPhase).index(self.world.traffic_light.current_phase)
        return {
            "queue_lengths": queue_lengths,
            "emergency_waiting": emergency_waiting,
            "light_phase": phase_index,
        }

    def total_wait_time(self) -> float:
        vehicle_wait = sum(getattr(v, "wait_time", 0.0) for v in self.world.vehicles)
        return vehicle_wait
