"""Traffic light state and control abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from config.sim_config import GREEN_PHASE_DURATION, YELLOW_PHASE_DURATION


class TrafficPhase(str, Enum):
    """Discrete traffic light phases for a single intersection."""

    NS_GREEN = "ns_green"
    NS_YELLOW = "ns_yellow"
    EW_GREEN = "ew_green"
    EW_YELLOW = "ew_yellow"


@dataclass
class TrafficLight:
    """Traffic light controller with minimum phase durations.

    Emergency handling:
        Call `request_emergency_phase()` to temporarily override normal cycling.
        The controller will switch to the requested phase as soon as the minimum
        phase duration allows, and no later than the configured maximum delay
        unless the minimum duration constraint prevents it.
    """

    current_phase: TrafficPhase = TrafficPhase.NS_GREEN
    phase_timer: float = 0.0
    min_phase_duration: float = GREEN_PHASE_DURATION
    yellow_phase_duration: float = YELLOW_PHASE_DURATION
    emergency_max_delay: float = 10.0
    _emergency_target: TrafficPhase | None = None
    _emergency_timer: float = 0.0

    def update(self, dt: float) -> None:
        """Advance the phase timer.

        Args:
            dt: Simulation timestep in seconds.
        """
        if dt <= 0:
            raise ValueError("dt must be positive")

        self.phase_timer += dt

        if self._emergency_target is not None:
            self._emergency_timer += dt
            if (
                self.current_phase != self._emergency_target
                and self.phase_timer >= self._current_phase_min_duration()
                and self._emergency_timer >= self.emergency_max_delay
            ):
                self.switch_phase(self._emergency_target)
                self._clear_emergency_request()
                return

        if self.phase_timer >= self._current_phase_min_duration():
            self.switch_phase(self._next_phase(self.current_phase))

    def switch_phase(self, next_phase: TrafficPhase) -> bool:
        """Switch to a new phase if the minimum duration has elapsed.

        Returns:
            True if the phase changed, otherwise False.
        """
        if self.phase_timer < self._current_phase_min_duration():
            return False

        if next_phase == self.current_phase:
            return False

        self.current_phase = next_phase
        self.phase_timer = 0.0

        return True

    def request_emergency_phase(
        self, target_phase: TrafficPhase, max_delay: float | None = None
    ) -> None:
        """Request an emergency override to a target phase.

        The override respects minimum phase duration and aims to switch within
        the configured maximum delay.
        """
        if max_delay is not None:
            if max_delay <= 0:
                raise ValueError("max_delay must be positive")
            self.emergency_max_delay = max_delay

        if target_phase == self.current_phase:
            self._clear_emergency_request()
            return

        if self._emergency_target != target_phase:
            self._emergency_target = target_phase
            self._emergency_timer = 0.0

    def _clear_emergency_request(self) -> None:
        self._emergency_target = None
        self._emergency_timer = 0.0

    def _next_phase(self, phase: TrafficPhase) -> TrafficPhase:
        """Return the next phase in the fixed cycle."""
        order = (
            TrafficPhase.NS_GREEN,
            TrafficPhase.NS_YELLOW,
            TrafficPhase.EW_GREEN,
            TrafficPhase.EW_YELLOW,
        )
        index = order.index(phase)
        return order[(index + 1) % len(order)]

    def _current_phase_min_duration(self) -> float:
        if self.current_phase in (TrafficPhase.NS_YELLOW, TrafficPhase.EW_YELLOW):
            return self.yellow_phase_duration
        return self.min_phase_duration

    def can_switch(self) -> bool:
        return self.phase_timer >= self._current_phase_min_duration()
