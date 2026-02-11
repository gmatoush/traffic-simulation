"""Simulation world definitions for the traffic simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import random
from typing import Dict, List

from sim.vehicles import Car, EmergencyVehicle
from config.sim_config import (
    SPAWN_RATE_EAST,
    SPAWN_RATE_EMERGENCY,
    SPAWN_RATE_NORTH,
    SPAWN_RATE_SOUTH,
    SPAWN_RATE_WEST,
)
from sim.traffic_light import TrafficLight, TrafficPhase


class Lane(str, Enum):
    """Incoming lanes for the four-way intersection."""

    NORTH = "NORTH"
    SOUTH = "SOUTH"
    EAST = "EAST"
    WEST = "WEST"




@dataclass
class World:
    """Container for simulation state and update coordination."""

    vehicles: List[object] = field(default_factory=list)
    lane_queues: Dict[Lane, List[object]] = field(
        default_factory=lambda: {lane: [] for lane in Lane}
    )
    car_spawn_probabilities: Dict[Lane, float] = field(
        default_factory=lambda: {
            Lane.NORTH: SPAWN_RATE_NORTH,
            Lane.SOUTH: SPAWN_RATE_SOUTH,
            Lane.EAST: SPAWN_RATE_EAST,
            Lane.WEST: SPAWN_RATE_WEST,
        }
    )
    emergency_spawn_probabilities: Dict[Lane, float] = field(
        default_factory=lambda: {lane: SPAWN_RATE_EMERGENCY for lane in Lane}
    )
    entry_distance: float = 220.0
    exit_distance: float = 240.0
    # Stop line is placed before the intersection to prevent entering on red.
    stop_line_distance: float = 80.0
    min_follow_gap: float = 55.0
    max_spawns_per_lane_per_tick: int = 4
    lane_offset: float = 30.0
    render_road_width: float = 120.0
    crash_freeze_duration: float = 5.0
    lane_freeze_timers: Dict[Lane, float] = field(
        default_factory=lambda: {lane: 0.0 for lane in Lane}
    )
    crash_events: int = 0
    emergency_enabled: bool = True
    risk_factor: float = 0.0
    uniform_speed_enabled: bool = False
    uniform_speed_value: float = 26.0
    max_vehicles: int = 20
    emergency_spawn_timer: float = 0.0
    emergency_spawn_interval: float = 0.0
    traffic_light: object | None = None
    time: float = 0.0

    def __post_init__(self) -> None:
        self._validate_spawn_probabilities(self.car_spawn_probabilities, Lane)
        self._validate_spawn_probabilities(self.emergency_spawn_probabilities, Lane)
        self._reset_emergency_timer()

    def _reset_emergency_timer(self) -> None:
        self.emergency_spawn_timer = 0.0
        self.emergency_spawn_interval = random.uniform(0.0, 30.0)

    @staticmethod
    def _validate_spawn_probabilities(
        probabilities: Dict[Enum, float], enum_type: type[Enum]
    ) -> None:
        for enum_value in enum_type:
            if enum_value not in probabilities:
                raise ValueError(f"Missing spawn probability for {enum_value}")
            prob = probabilities[enum_value]
            if prob < 0.0 or prob > 1.0:
                raise ValueError(f"Spawn probability out of range for {enum_value}")

    def enqueue_vehicle(self, vehicle: object, lane: Lane) -> None:
        """Add a vehicle to a lane queue and register it with the world."""
        if len(self.vehicles) >= self.max_vehicles:
            return
        if hasattr(vehicle, "lane"):
            setattr(vehicle, "lane", lane)

        self.lane_queues[lane].append(vehicle)
        self.vehicles.append(vehicle)


    def _lane_has_green(self, lane: Lane) -> bool:
        return self._lane_signal(lane) == "green"

    def _lane_signal(self, lane: Lane) -> str:
        """Return the signal state for a lane: green, yellow, or red."""
        if not isinstance(self.traffic_light, TrafficLight):
            return "green"

        phase = self.traffic_light.current_phase
        if lane in (Lane.NORTH, Lane.SOUTH):
            if phase == TrafficPhase.NS_GREEN:
                return "green"
            if phase == TrafficPhase.NS_YELLOW:
                return "yellow"
            return "red"
        if phase == TrafficPhase.EW_GREEN:
            return "green"
        if phase == TrafficPhase.EW_YELLOW:
            return "yellow"
        return "red"

    def _lane_direction(self, lane: Lane) -> int:
        if lane in (Lane.NORTH, Lane.WEST):
            return 1
        return -1

    @staticmethod
    def _lane_directions(lane: Lane) -> tuple[str, str]:
        if lane == Lane.WEST:
            return "WEST", "EAST"
        if lane == Lane.EAST:
            return "EAST", "WEST"
        if lane == Lane.NORTH:
            return "NORTH", "SOUTH"
        return "SOUTH", "NORTH"

    def _lane_stop_line(self, lane: Lane) -> float:
        if lane in (Lane.NORTH, Lane.WEST):
            return -self.stop_line_distance
        return self.stop_line_distance

    def _update_vehicle_center(self, vehicle: object) -> None:
        """Update center-of-mass coordinates based on lane and position."""
        lane = getattr(vehicle, "lane", None)
        lane_value = getattr(lane, "value", lane)
        pos = float(getattr(vehicle, "position", 0.0))
        # Keep collision box aligned with render sizing.
        lane_width = self.render_road_width / 2.0
        car_width = max(18.0, lane_width * 0.8)
        car_length = max(28.0, lane_width * 1.2)
        setattr(vehicle, "width", car_width)
        setattr(vehicle, "length", car_length)
        if lane_value == "NORTH":
            setattr(vehicle, "center_x", -self.lane_offset)
            setattr(vehicle, "center_y", pos)
        elif lane_value == "SOUTH":
            setattr(vehicle, "center_x", self.lane_offset)
            setattr(vehicle, "center_y", pos)
        elif lane_value == "EAST":
            setattr(vehicle, "center_x", pos)
            setattr(vehicle, "center_y", -self.lane_offset)
        elif lane_value == "WEST":
            setattr(vehicle, "center_x", pos)
            setattr(vehicle, "center_y", self.lane_offset)

    def _lane_frozen(self, lane: Lane) -> bool:
        return self.lane_freeze_timers.get(lane, 0.0) > 0.0

    def _register_crash(self, vehicle_a: object, vehicle_b: object) -> None:
        # Crash counts for all lanes whenever any sprites collide.
        for lane in Lane:
            self.lane_freeze_timers[lane] = self.crash_freeze_duration
        self.crash_events += 1

    def _update_freeze_timers(self, dt: float) -> None:
        for lane in list(self.lane_freeze_timers.keys()):
            if self.lane_freeze_timers[lane] > 0.0:
                self.lane_freeze_timers[lane] = max(
                    0.0, self.lane_freeze_timers[lane] - dt
                )

    def _remove_crashed_after_freeze(self) -> None:
        # After the freeze window, delete any vehicles that are still crashed or touching.
        touching: set[int] = set()
        vehicles = list(self.vehicles)
        if len(vehicles) <= 1:
            return
        for i in range(len(vehicles)):
            v1 = vehicles[i]
            lane1 = getattr(v1, "lane", None)
            b1 = self._vehicle_world_bbox(v1)
            for j in range(i + 1, len(vehicles)):
                v2 = vehicles[j]
                lane2 = getattr(v2, "lane", None)
                if lane1 is not None and lane1 == lane2:
                    continue
                b2 = self._vehicle_world_bbox(v2)
                if (
                    b1[0] <= b2[2]
                    and b1[2] >= b2[0]
                    and b1[1] <= b2[3]
                    and b1[3] >= b2[1]
                ):
                    touching.add(id(v1))
                    touching.add(id(v2))

        for lane, queue in self.lane_queues.items():
            if self.lane_freeze_timers.get(lane, 0.0) > 0.0:
                continue
            crashed = [
                v
                for v in queue
                if getattr(v, "crashed", False) or id(v) in touching
            ]
            for vehicle in crashed:
                if vehicle in queue:
                    queue.remove(vehicle)
                if vehicle in self.vehicles:
                    self.vehicles.remove(vehicle)

    def consume_crash_events(self) -> int:
        events = self.crash_events
        self.crash_events = 0
        return events


    def _update_lane_queue(self, lane: Lane, dt: float) -> None:
        queue = self.lane_queues[lane]
        if not queue:
            return

        if self._lane_frozen(lane):
            for vehicle in queue:
                if hasattr(vehicle, "update"):
                    vehicle.update(dt, stopped=True, move_step=0.0)
            return

        lane_signal = self._lane_signal(lane)
        direction = self._lane_direction(lane)
        stop_line = self._lane_stop_line(lane)

        # Ordered list per lane by position along the lane axis.
        # Front-most vehicle is evaluated first for spacing checks.
        queue.sort(key=lambda v: getattr(v, "position", 0.0), reverse=direction > 0)

        lead_pos: float | None = None
        lead_vehicle: object | None = None
        for vehicle in queue:
            speed = float(getattr(vehicle, "speed", 0.0))
            position = float(getattr(vehicle, "position", 0.0))
            target = position + direction * speed * dt
            stopped = False

            # Following model:
            # Each vehicle tracks the one ahead and enforces a minimum gap equal
            # to (lead.length + follower.min_gap). This prevents overlap at all speeds.
            if lead_pos is not None and lead_vehicle is not None:
                lead_length = float(getattr(lead_vehicle, "length", 0.0))
                follower_length = float(getattr(vehicle, "length", 0.0))
                follower_gap = float(getattr(vehicle, "min_gap", self.min_follow_gap))
                lead_back = lead_pos - direction * (lead_length / 2.0)
                # Enforce gap using the follower's back vs. the lead's back.
                if direction > 0:
                    separator_pos = lead_back - (follower_gap + follower_length / 2.0)
                    if target > separator_pos:
                        target = separator_pos
                        stopped = target == position
                else:
                    separator_pos = lead_back + (follower_gap + follower_length / 2.0)
                    if target < separator_pos:
                        target = separator_pos
                        stopped = target == position
            # Stop line logic:
            # On red, vehicles must stop at the stop line and never enter the intersection.
            # On yellow, vehicles already past the stop line may continue; others must stop.
            front_pos = position + direction * (float(getattr(vehicle, "length", 0.0)) / 2.0)
            can_proceed = lane_signal == "green"
            if lane_signal == "yellow":
                if direction > 0:
                    can_proceed = front_pos > stop_line
                else:
                    can_proceed = front_pos < stop_line

            if not can_proceed:
                if direction > 0:
                    if front_pos <= stop_line:
                        target = min(target, stop_line - direction * (float(getattr(vehicle, "length", 0.0)) / 2.0))
                        stopped = target == position
                    # If already past stop line, allow continued movement.
                else:
                    if front_pos >= stop_line:
                        target = max(target, stop_line - direction * (float(getattr(vehicle, "length", 0.0)) / 2.0))
                        stopped = target == position

            move_step = target - position
            if hasattr(vehicle, "update"):
                if abs(move_step) < 1e-6:
                    stopped = True
                vehicle.update(dt, stopped=stopped, move_step=move_step)
                self._update_vehicle_center(vehicle)

            lead_pos = target
            lead_vehicle = vehicle

        # Remove vehicles that have cleared the intersection.
        cleared = []
        for vehicle in queue:
            position = float(getattr(vehicle, "position", 0.0))
            if direction > 0 and position > self.exit_distance:
                cleared.append(vehicle)
            elif direction < 0 and position < -self.exit_distance:
                cleared.append(vehicle)

        for vehicle in cleared:
            if vehicle in queue:
                queue.remove(vehicle)
            if vehicle in self.vehicles:
                self.vehicles.remove(vehicle)

    def spawn_cars(self) -> None:
        """Spawn cars randomly per lane based on configured probabilities."""
        # Reserve a slot for emergencies so the toggle can actually spawn them.
        effective_max = self.max_vehicles - (1 if self.emergency_enabled else 0)
        for lane, probability in self.car_spawn_probabilities.items():
            for _ in range(self.max_spawns_per_lane_per_tick):
                if len(self.vehicles) >= effective_max:
                    return
                if random.random() >= probability:
                    continue
                position = -self.entry_distance if lane in (Lane.NORTH, Lane.WEST) else self.entry_distance
                # If the entry is occupied, stack new spawns behind the last vehicle.
                queue = self.lane_queues[lane]
                entry_direction, exit_direction = self._lane_directions(lane)
                if self.uniform_speed_enabled:
                    speed_category = "medium"
                    speed = self.uniform_speed_value
                    gap = random.uniform(self.min_follow_gap * 0.9, self.min_follow_gap * 1.1)
                else:
                    speed_category = random.choice(["slow", "medium", "fast"])
                    if speed_category == "slow":
                        speed = random.uniform(5.0, 8.0)
                        gap = random.uniform(self.min_follow_gap * 0.6, self.min_follow_gap * 0.8)
                    elif speed_category == "fast":
                        speed = random.uniform(22.0, 30.0)
                        gap = random.uniform(self.min_follow_gap * 1.2, self.min_follow_gap * 1.5)
                    else:
                        speed = random.uniform(12.0, 16.0)
                        gap = random.uniform(self.min_follow_gap * 0.9, self.min_follow_gap * 1.1)
                gap = max(self.min_follow_gap, gap)
                speed *= 1.0 + 0.75 * self.risk_factor
                if queue:
                    # Only spawn at the road entry if there is enough space.
                    direction = self._lane_direction(lane)
                    tail_pos = min(
                        (getattr(v, "position", position) for v in queue),
                        default=position,
                    ) if direction > 0 else max(
                        (getattr(v, "position", position) for v in queue),
                        default=position,
                    )
                    spawn_gap = self.min_follow_gap * 1.5
                    if direction > 0:
                        if tail_pos - spawn_gap > position:
                            continue
                    else:
                        if tail_pos + spawn_gap < position:
                            continue
                car = Car(
                    speed=speed,
                    position=position,
                    wait_time=0.0,
                    lane=lane,
                    entry_direction=entry_direction,
                    exit_direction=exit_direction,
                    min_gap=gap,
                    sprite_kind="normal",
                    sprite_category=speed_category,
                    sprite_seed=random.randint(0, 1_000_000_000),
                )
                self.enqueue_vehicle(car, lane)
                self._update_vehicle_center(car)

    def spawn_emergency_vehicles(self) -> None:
        """Spawn emergency vehicles randomly at low probability."""
        if not self.emergency_enabled:
            return
        if self.emergency_spawn_timer < self.emergency_spawn_interval:
            return
        if len(self.vehicles) >= self.max_vehicles:
            return

        lane = random.choice(list(Lane))
        position = -self.entry_distance if lane in (Lane.NORTH, Lane.WEST) else self.entry_distance
        queue = self.lane_queues[lane]
        entry_direction, exit_direction = self._lane_directions(lane)
        gap = random.uniform(self.min_follow_gap * 1.1, self.min_follow_gap * 1.4)
        gap = max(self.min_follow_gap, gap)
        speed = random.uniform(18.0, 24.0)
        speed *= 1.0 + 0.75 * self.risk_factor
        if queue:
            direction = self._lane_direction(lane)
            tail_pos = min(
                (getattr(v, "position", position) for v in queue),
                default=position,
            ) if direction > 0 else max(
                (getattr(v, "position", position) for v in queue),
                default=position,
            )
            spawn_gap = self.min_follow_gap * 1.5
            if direction > 0:
                if tail_pos - spawn_gap > position:
                    return
            else:
                if tail_pos + spawn_gap < position:
                    return
        vehicle = EmergencyVehicle(
            speed=speed,
            position=position,
            wait_time=0.0,
            lane=lane,
            entry_direction=entry_direction,
            exit_direction=exit_direction,
            min_gap=gap,
            sprite_kind="emergency",
            sprite_category="emergency",
            sprite_seed=random.randint(0, 1_000_000_000),
        )
        self.enqueue_vehicle(vehicle, lane)
        self._update_vehicle_center(vehicle)
        self._reset_emergency_timer()


    def emergency_waiting(self) -> bool:
        """Return True if any emergency vehicle is currently waiting in a queue."""
        for queue in self.lane_queues.values():
            for vehicle in queue:
                if isinstance(vehicle, EmergencyVehicle) and vehicle.wait_time > 0:
                    return True
        return False

    def emergency_waiting_lane(self) -> Lane | None:
        """Return the lane with a waiting emergency vehicle, if any."""
        for lane, queue in self.lane_queues.items():
            for vehicle in queue:
                if isinstance(vehicle, EmergencyVehicle) and vehicle.wait_time > 0:
                    return lane
        return None

    @staticmethod
    def _lane_priority_phase(lane: Lane) -> TrafficPhase:
        if lane in (Lane.NORTH, Lane.SOUTH):
            return TrafficPhase.NS_GREEN
        return TrafficPhase.EW_GREEN

    def update(self, dt: float, update_traffic_light: bool = True) -> None:
        """Advance simulation time and update entities.

        Args:
            dt: Simulation timestep in seconds.
            update_traffic_light: When False, skip internal traffic light updates.
        """
        if dt <= 0:
            raise ValueError("dt must be positive")

        self.time += dt
        if self.emergency_enabled:
            self.emergency_spawn_timer += dt
        self.spawn_cars()
        self.spawn_emergency_vehicles()

        self._update_freeze_timers(dt)

        if isinstance(self.traffic_light, TrafficLight):
            emergency_lane = self.emergency_waiting_lane()
            if emergency_lane is not None:
                target_phase = self._lane_priority_phase(emergency_lane)
                self.traffic_light.request_emergency_phase(target_phase)

        for lane in Lane:
            self._update_lane_queue(lane, dt)

        self._detect_collisions()
        self._remove_crashed_after_freeze()

        if (
            update_traffic_light
            and self.traffic_light is not None
            and hasattr(self.traffic_light, "update")
        ):
            if self.crash_events > 0:
                return
            self.traffic_light.update(dt)

    def _vehicle_world_bbox(self, vehicle: object) -> tuple[float, float, float, float]:
        """Return AABB (min_x, min_y, max_x, max_y) in world coordinates."""
        lane = getattr(vehicle, "lane", None)
        lane_value = getattr(lane, "value", lane)
        pos = float(getattr(vehicle, "position", 0.0))
        length = float(getattr(vehicle, "length", 0.0))
        width = float(getattr(vehicle, "width", 0.0))

        if lane_value in ("NORTH", "SOUTH"):
            x = -self.lane_offset if lane_value == "NORTH" else self.lane_offset
            y = pos
            half_w = width / 2.0
            half_l = length / 2.0
            return (x - half_w, y - half_l, x + half_w, y + half_l)

        x = pos
        y = -self.lane_offset if lane_value == "EAST" else self.lane_offset
        half_w = width / 2.0
        half_l = length / 2.0
        return (x - half_l, y - half_w, x + half_l, y + half_w)

    def _detect_collisions(self) -> None:
        """Crash vehicles when any two overlap in world space."""
        vehicles = list(self.vehicles)
        if len(vehicles) <= 1:
            return
        for i in range(len(vehicles)):
            v1 = vehicles[i]
            if getattr(v1, "crashed", False):
                continue
            lane1 = getattr(v1, "lane", None)
            b1 = self._vehicle_world_bbox(v1)
            for j in range(i + 1, len(vehicles)):
                v2 = vehicles[j]
                lane2 = getattr(v2, "lane", None)
                if lane1 is not None and lane1 == lane2:
                    continue
                if getattr(v2, "crashed", False):
                    continue
                b2 = self._vehicle_world_bbox(v2)
                if (
                    b1[0] <= b2[2]
                    and b1[2] >= b2[0]
                    and b1[1] <= b2[3]
                    and b1[3] >= b2[1]
                ):
                    setattr(v1, "crashed", True)
                    setattr(v2, "crashed", True)
                    setattr(v1, "speed", 0.0)
                    setattr(v2, "speed", 0.0)
                    self._register_crash(v1, v2)
