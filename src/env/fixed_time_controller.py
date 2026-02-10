"""Fixed-time traffic signal controller baseline and benchmark helpers."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Dict

from config.sim_config import SIM_DT
from env.actions import HOLD_PHASE, SWITCH_PHASE
from sim.traffic_light import TrafficLight
from sim.world import World


@dataclass
class FixedTimeController:
    """Cycle phases on a fixed schedule in seconds."""

    phase_duration: float = 5.0
    _timer: float = 0.0

    def reset(self) -> None:
        self._timer = 0.0

    def act(self, dt: float) -> int:
        self._timer += dt
        if self._timer >= self.phase_duration:
            self._timer = 0.0
            return SWITCH_PHASE
        return HOLD_PHASE

    def update(self, light: TrafficLight, dt: float) -> None:
        action = self.act(dt)
        if action == SWITCH_PHASE:
            light.switch_phase(light._next_phase(light.current_phase))


def run_fixed_time_benchmark(
    steps: int = 2000, phase_duration: float = 5.0, seed: int = 42
) -> Dict[str, Any]:
    """Run a fixed-time benchmark and return summary metrics."""
    random.seed(seed)
    world = World(traffic_light=TrafficLight())
    controller = FixedTimeController(phase_duration=phase_duration)

    total_wait = 0.0
    for _ in range(steps):
        action = controller.act(SIM_DT)
        if action == SWITCH_PHASE:
            world.traffic_light.switch_phase(
                world.traffic_light._next_phase(world.traffic_light.current_phase)
            )
        world.update(SIM_DT, update_traffic_light=False)
        total_wait += _total_wait_time(world)

    avg_wait = total_wait / max(1, steps)
    return {
        "steps": steps,
        "avg_wait_time": avg_wait,
        "phase_duration": phase_duration,
        "seed": seed,
    }


def _total_wait_time(world: World) -> float:
    vehicle_wait = sum(getattr(v, "wait_time", 0.0) for v in world.vehicles)
    return vehicle_wait
