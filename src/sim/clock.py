"""Simulation clock and time-stepping utilities."""

from __future__ import annotations


class SimulationClock:
    """Decouple simulation time from real time.

    The clock advances the simulation by a fixed `dt` each time `tick()` is
    called. The `speed` attribute describes how fast the simulation should
    progress relative to real time (for example, `speed=2.0` means the
    simulation is intended to run twice as fast as real time). This class does
    not sleep or measure wall-clock time; it only provides consistent step
    sizes so callers can decide how to enforce the desired speed externally.
    """

    def __init__(self, dt: float, speed: float = 1.0) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive")
        if speed <= 0:
            raise ValueError("speed must be positive")

        self.dt = float(dt)
        self.speed = float(speed)

    def tick(self) -> float:
        """Advance the simulation by one step and return `dt`."""
        return self.dt