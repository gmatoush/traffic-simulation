"""Vehicle models and behaviors for the traffic simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.world import Lane


@dataclass
class Vehicle:
    """Base vehicle entity.

    Intended to capture shared state across all moving agents. Concrete
    subclasses will refine movement, interaction, and rules of the road.
    """

    speed: float
    position: float
    wait_time: float
    lane: "Lane | None"
    entry_direction: str = "UNKNOWN"
    exit_direction: str = "UNKNOWN"
    length: float = 18.0
    width: float = 10.0
    min_gap: float = 28.0
    sprite_kind: str = "normal"
    sprite_category: str = "medium"
    center_x: float = 0.0
    center_y: float = 0.0
    crashed: bool = False

    def update(self, dt: float, stopped: bool = False, move_step: float = 1.0) -> None:
        """Update per-tick bookkeeping and discrete movement."""
        if dt <= 0:
            raise ValueError("dt must be positive")

        if self.crashed:
            self.wait_time += dt
            return

        if stopped or abs(move_step) < 1e-6:
            # Accumulate waiting time when stopped at signals or in queues.
            self.wait_time += dt
            return

        # Continuous linear movement (no physics yet).
        # Stop-line and spacing logic is enforced by the world update.
        # Vehicles only stop when constrained by the stop line or a lead vehicle.
        self.position += move_step


@dataclass
class Car(Vehicle):
    """Standard passenger vehicle.

    Behavior will include lane-keeping and speed regulation in future logic.
    """




@dataclass
class EmergencyVehicle(Vehicle):
    """Emergency vehicle with priority handling.

    Priority vehicles may override normal traffic rules when enabled.
    """

    priority: bool = True
