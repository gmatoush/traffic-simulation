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
from sim.vehicles import EmergencyVehicle
from sim.world import Lane, World

# Reward weights (crash avoidance must dominate all other terms).
CRASH_PENALTY_PER_VEHICLE = -100.0
EMERGENCY_WAIT_PENALTY = -20.0
QUEUE_PENALTY = -1.0
WAIT_TIME_PENALTY = -0.05
SWITCH_PENALTY = -0.1
# Small positive rewards to stabilize learning (must not outweigh crash penalties).
NO_CRASH_BONUS = 1.0
LOW_QUEUE_BONUS = 0.5
LOW_QUEUE_THRESHOLD = 4
FLOW_BONUS = 0.2

@dataclass
class TrafficEnv(gym.Env):
    """Gym-compatible environment for traffic signal control."""

    render_enabled: bool = RENDER_ENABLED
    max_steps: int = 1000
    emergency_wait_penalty: float = abs(EMERGENCY_WAIT_PENALTY)
    crash_penalty: float = abs(CRASH_PENALTY_PER_VEHICLE)
    wait_weight: float = abs(WAIT_TIME_PENALTY)

    def __post_init__(self) -> None:
        self.dt = SIM_DT
        self._steps = 0
        self.world = World(traffic_light=TrafficLight())
        self._renderer = None
        self._prev_vehicle_count = 0

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
        self._prev_vehicle_count = 0
        return self._get_obs(), {}

    def step(self, action: int) -> tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        prev_phase = None
        if isinstance(self.world.traffic_light, TrafficLight):
            prev_phase = self.world.traffic_light.current_phase
        if action == SWITCH_PHASE and isinstance(self.world.traffic_light, TrafficLight):
            light = self.world.traffic_light
            light.switch_phase(light._next_phase(light.current_phase))

        prev_count = len(self.world.vehicles)
        self.world.update(self.dt)
        self._steps += 1

        crash_events = self.world.consume_crash_events()
        crashed_count = sum(1 for v in self.world.vehicles if getattr(v, "crashed", False))

        # Crash avoidance dominates: one crash outweighs all other gains.
        crash_penalty = CRASH_PENALTY_PER_VEHICLE * max(crashed_count, crash_events)

        emergency_wait_time = sum(
            getattr(v, "wait_time", 0.0)
            for v in self.world.vehicles
            if isinstance(v, EmergencyVehicle) and getattr(v, "wait_time", 0.0) > 0.0
        )
        emergency_penalty = EMERGENCY_WAIT_PENALTY * emergency_wait_time

        waiting_count = sum(1 for v in self.world.vehicles if getattr(v, "wait_time", 0.0) > 0.0)
        queue_penalty = QUEUE_PENALTY * waiting_count

        total_wait = self.total_wait_time()
        wait_time_penalty = WAIT_TIME_PENALTY * total_wait

        phase_switched = (
            prev_phase is not None
            and isinstance(self.world.traffic_light, TrafficLight)
            and self.world.traffic_light.current_phase != prev_phase
        )
        switch_penalty = SWITCH_PENALTY if phase_switched else 0.0

        # Positive rewards: small, stabilizing incentives for safe, efficient flow.
        no_crash_bonus = NO_CRASH_BONUS if crashed_count == 0 and crash_events == 0 else 0.0
        low_queue_bonus = LOW_QUEUE_BONUS if waiting_count < LOW_QUEUE_THRESHOLD else 0.0
        prev_count = len(self.world.vehicles) + crash_events
        cleared = max(0, prev_count - len(self.world.vehicles))
        flow_bonus = FLOW_BONUS * cleared

        reward = (
            crash_penalty
            + emergency_penalty
            + queue_penalty
            + wait_time_penalty
            + switch_penalty
            + no_crash_bonus
            + low_queue_bonus
            + flow_bonus
        )

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

    def average_wait_time(self) -> float:
        vehicle_count = len(self.world.vehicles)
        if vehicle_count == 0:
            return 0.0
        return self.total_wait_time() / vehicle_count
