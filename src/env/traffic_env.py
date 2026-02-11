"""Gym-compatible environment wrapper for the traffic simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - fallback for gym
    import gym
    from gym import spaces

from config.sim_config import (
    RL_ACTION_REPEAT,
    RL_CURRICULUM_EPISODES,
    RENDER_ENABLED,
    SIM_DT,
    SPAWN_RATE_EAST,
    SPAWN_RATE_NORTH,
    SPAWN_RATE_SOUTH,
    SPAWN_RATE_WEST,
)
from env.actions import SWITCH_PHASE
from sim.traffic_light import TrafficLight, TrafficPhase
from sim.vehicles import EmergencyVehicle
from sim.world import Lane, World

# Reward weights (crash avoidance must dominate all other terms).
CRASH_PENALTY_PER_VEHICLE = -100.0
EMERGENCY_WAIT_PENALTY = -20.0
QUEUE_PENALTY = -1.0
WAIT_TIME_PENALTY = -0.05
SWITCH_PENALTY = -0.1
INVALID_SWITCH_PENALTY = -0.2
# Small positive rewards to stabilize learning (must not outweigh crash penalties).
NO_CRASH_BONUS = 1.0
LOW_QUEUE_BONUS = 0.5
LOW_QUEUE_THRESHOLD = 4
FLOW_BONUS = 0.2
DELTA_WAIT_REWARD = 0.35
DELTA_QUEUE_REWARD = 0.5
QUEUE_NORMALIZER = 40.0

@dataclass
class TrafficEnv(gym.Env):
    """Gym-compatible environment for traffic signal control."""

    render_enabled: bool = RENDER_ENABLED
    max_steps: int = 1000
    action_repeat: int = RL_ACTION_REPEAT
    use_curriculum: bool = False
    curriculum_episodes: int = RL_CURRICULUM_EPISODES
    emergency_wait_penalty: float = abs(EMERGENCY_WAIT_PENALTY)
    crash_penalty: float = abs(CRASH_PENALTY_PER_VEHICLE)
    wait_weight: float = abs(WAIT_TIME_PENALTY)

    def __post_init__(self) -> None:
        if self.action_repeat <= 0:
            raise ValueError("action_repeat must be positive")
        if self.curriculum_episodes <= 0:
            raise ValueError("curriculum_episodes must be positive")
        self.dt = SIM_DT
        self._steps = 0
        self._episodes = 0
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
                    low=0.0, high=1.0, shape=(len(Lane),), dtype=np.float32
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
        if self.use_curriculum:
            self._apply_curriculum()
        self._prev_vehicle_count = 0
        self._episodes += 1
        return self._get_obs(), {}

    def step(self, action: int) -> tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; expected 0 or 1.")

        prev_total_wait = self.total_wait_time()
        prev_waiting_count = sum(
            1 for v in self.world.vehicles if getattr(v, "wait_time", 0.0) > 0.0
        )
        prev_count = len(self.world.vehicles)
        invalid_switch = False
        phase_switched = False

        for idx in range(self.action_repeat):
            if idx == 0 and action == SWITCH_PHASE and isinstance(self.world.traffic_light, TrafficLight):
                light = self.world.traffic_light
                prev_phase = light.current_phase
                switched = light.switch_phase(light._next_phase(light.current_phase))
                invalid_switch = not switched
                phase_switched = phase_switched or (light.current_phase != prev_phase)
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
        switch_penalty = SWITCH_PENALTY if phase_switched else 0.0
        invalid_switch_penalty = INVALID_SWITCH_PENALTY if invalid_switch else 0.0

        # Positive rewards: small, stabilizing incentives for safe, efficient flow.
        no_crash_bonus = NO_CRASH_BONUS if crashed_count == 0 and crash_events == 0 else 0.0
        low_queue_bonus = LOW_QUEUE_BONUS if waiting_count < LOW_QUEUE_THRESHOLD else 0.0
        cleared = max(0, prev_count + crash_events - len(self.world.vehicles))
        flow_bonus = FLOW_BONUS * cleared
        delta_wait_reward = DELTA_WAIT_REWARD * (prev_total_wait - total_wait)
        delta_queue_reward = DELTA_QUEUE_REWARD * (prev_waiting_count - waiting_count)

        reward = (
            crash_penalty
            + emergency_penalty
            + queue_penalty
            + wait_time_penalty
            + switch_penalty
            + invalid_switch_penalty
            + no_crash_bonus
            + low_queue_bonus
            + flow_bonus
            + delta_wait_reward
            + delta_queue_reward
        )

        terminated = False
        truncated = self._steps >= self.max_steps
        info = {"action_mask": self._action_mask()}
        return self._get_obs(), reward, terminated, truncated, info

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
        queue_lengths = np.array(
            [len(self.world.lane_queues[lane]) for lane in Lane], dtype=np.float32
        )
        queue_lengths = np.clip(queue_lengths / QUEUE_NORMALIZER, 0.0, 1.0)
        emergency_waiting = int(self.world.emergency_waiting())
        phase_index = list(TrafficPhase).index(self.world.traffic_light.current_phase)
        return {
            "queue_lengths": queue_lengths,
            "emergency_waiting": emergency_waiting,
            "light_phase": phase_index,
        }

    def _apply_curriculum(self) -> None:
        progress = min(1.0, self._episodes / float(self.curriculum_episodes))
        spawn_scale = 0.35 + 0.65 * progress
        self.world.car_spawn_probabilities = {
            Lane.NORTH: SPAWN_RATE_NORTH * spawn_scale,
            Lane.SOUTH: SPAWN_RATE_SOUTH * spawn_scale,
            Lane.EAST: SPAWN_RATE_EAST * spawn_scale,
            Lane.WEST: SPAWN_RATE_WEST * spawn_scale,
        }
        self.world.risk_factor = 0.1 + 0.9 * progress
        self.world.max_vehicles = int(14 + 26 * progress)
        # Keep emergency vehicles enabled throughout curriculum.
        self.world.emergency_enabled = True

    def _action_mask(self) -> np.ndarray:
        can_switch = 0
        if isinstance(self.world.traffic_light, TrafficLight) and self.world.traffic_light.can_switch():
            can_switch = 1
        return np.array([1, can_switch], dtype=np.int8)

    def total_wait_time(self) -> float:
        vehicle_wait = sum(getattr(v, "wait_time", 0.0) for v in self.world.vehicles)
        return vehicle_wait

    def average_wait_time(self) -> float:
        vehicle_count = len(self.world.vehicles)
        if vehicle_count == 0:
            return 0.0
        return self.total_wait_time() / vehicle_count
