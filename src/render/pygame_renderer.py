"""Pygame-based renderer for the traffic simulation."""

from __future__ import annotations

from dataclasses import dataclass
import os
import random

import pygame


@dataclass
class PygameRenderer:
    """Render a top-down view of a single intersection using pygame."""

    width: int = 800
    height: int = 600
    road_width: int = 120

    def __post_init__(self) -> None:
        pygame.display.set_caption("Traffic RL Demo")
        self.screen = pygame.display.set_mode(
            (self.width, self.height), pygame.RESIZABLE
        )
        self.font = pygame.font.Font(None, 24)
        self._flash_timer = 0.0
        self._vehicle_sprites: dict[int, pygame.Surface] = {}
        self._sprite_pool: dict[str, list[pygame.Surface]] = {}
        self._load_vehicle_sprites()

    def resize(self, width: int, height: int) -> None:
        self.width = max(480, int(width))
        self.height = max(360, int(height))
        self.screen = pygame.display.set_mode(
            (self.width, self.height), pygame.RESIZABLE
        )

    def _load_vehicle_sprites(self) -> None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        cars_dir = os.path.join(base_dir, "assets", "cars")
        normal_dir = os.path.join(cars_dir, "normal_cars")
        emergency_dir = os.path.join(cars_dir, "emergency")
        self._sprite_pool = {
            "slow": self._load_sprite_dir(os.path.join(normal_dir, "slow")),
            "medium": self._load_sprite_dir(os.path.join(normal_dir, "medium")),
            "fast": self._load_sprite_dir(os.path.join(normal_dir, "fast")),
            "emergency": self._load_sprite_dir(emergency_dir),
        }

    def _load_sprite_dir(self, directory: str) -> list[pygame.Surface]:
        if not os.path.isdir(directory):
            return []
        sprites: list[pygame.Surface] = []
        for name in os.listdir(directory):
            if name.lower().endswith(".png"):
                path = os.path.join(directory, name)
                try:
                    sprites.append(pygame.image.load(path).convert_alpha())
                except pygame.error:
                    continue
        return sprites

    def render(
        self,
        world,
        sim_speed: float | None = None,
        show_overlays: bool = True,
        controls: dict | None = None,
    ) -> None:
        """Render the world state without mutating it.

        Args:
            world: Simulation world containing vehicles and traffic light.
            sim_speed: Optional simulation speed multiplier for HUD display.
            show_overlays: Toggle debug overlay rendering.
            controls: Optional UI control state for on-screen sliders/toggles.
        """
        self._flash_timer += 1.0 / 60.0
        self._draw_background()

        # Draw a simple cross intersection.
        horizontal = pygame.Rect(
            0, self.height // 2 - self.road_width // 2, self.width, self.road_width
        )
        vertical = pygame.Rect(
            self.width // 2 - self.road_width // 2, 0, self.road_width, self.height
        )
        pygame.draw.rect(self.screen, self.palette["road"], horizontal)
        pygame.draw.rect(self.screen, self.palette["road"], vertical)
        self._draw_lane_markings()
        # Render vehicles as sprites (fallback to rectangles if none loaded).
        for vehicle in getattr(world, "vehicles", []):
            rect = self._vehicle_rect(vehicle)
            if any(self._sprite_pool.values()):
                sprite = self._get_vehicle_sprite(vehicle)
                sprite = self._orient_and_scale_sprite(sprite, rect, vehicle)
                self.screen.blit(sprite, rect.topleft)
            else:
                color = self._vehicle_color(vehicle)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (8, 8, 10), rect, width=1)

        # Render central traffic light indicator.
        light = getattr(world, "traffic_light", None)
        if light is not None:
            self._draw_central_traffic_light(light)

        if controls is not None:
            self._draw_controls(controls)

        pygame.display.flip()

    def _vehicle_rect(self, vehicle) -> pygame.Rect:
        """Map vehicle position/lane to a rectangle in screen space."""
        # Scale vehicle size relative to lane width for readability.
        # Orientation aligns with lane travel: vertical lanes are tall, horizontal lanes are wide.
        lane = getattr(vehicle, "lane", "horizontal")
        position = float(getattr(vehicle, "position", 0.0))
        lane_value = getattr(lane, "value", lane)

        lane_width = self.road_width // 2
        car_width = max(18, int(lane_width * 0.8))
        car_length = max(28, int(lane_width * 1.2))

        if lane_value in ("EAST", "WEST"):
            width = car_length
            height = car_width
        else:
            width = car_width
            height = car_length

        center_x = self.width // 2
        center_y = self.height // 2
        # Align vehicles to their half of the two-way road (keep center line clear).
        lane_offset = 30
        entry_offset = self.road_width // 2 + 30

        if lane_value == "NORTH":
            x = center_x - lane_offset - width // 2
            y = int(center_y + position - height // 2)
        elif lane_value == "SOUTH":
            x = center_x + lane_offset - width // 2
            y = int(center_y + position - height // 2)
        elif lane_value == "EAST":
            x = int(center_x + position - width // 2)
            y = center_y - lane_offset - height // 2
        elif lane_value == "WEST":
            x = int(center_x + position - width // 2)
            y = center_y + lane_offset - height // 2
        else:
            x = int(center_x - self.road_width // 2 + position)
            y = center_y - height // 2

        return pygame.Rect(x, y, width, height)

    def _vehicle_color(self, vehicle) -> tuple[int, int, int]:
        """Pick a color based on vehicle type."""
        name = vehicle.__class__.__name__
        if name == "EmergencyVehicle":
            return self._flashing_emergency_color()
        return self.palette["vehicle"]

    def _get_vehicle_sprite(self, vehicle) -> pygame.Surface:
        key = id(vehicle)
        if key not in self._vehicle_sprites:
            kind = getattr(vehicle, "sprite_kind", "normal")
            category = getattr(vehicle, "sprite_category", "medium")
            pool = self._sprite_pool.get("emergency" if kind == "emergency" else category, [])
            if not pool:
                fallback = []
                for sprites in self._sprite_pool.values():
                    fallback.extend(sprites)
                pool = fallback
            self._vehicle_sprites[key] = random.choice(pool)
        return self._vehicle_sprites[key]

    def _orient_and_scale_sprite(
        self, sprite: pygame.Surface, rect: pygame.Rect, vehicle
    ) -> pygame.Surface:
        lane = getattr(vehicle, "lane", None)
        lane_value = getattr(lane, "value", lane)
        oriented = sprite
        # Sprites are vertical; rotate/flip based on travel direction.
        if lane_value == "NORTH":
            oriented = pygame.transform.rotate(oriented, 180)
        elif lane_value == "EAST":
            oriented = pygame.transform.rotate(oriented, 90)
        elif lane_value == "WEST":
            oriented = pygame.transform.rotate(oriented, -90)
        scaled = pygame.transform.smoothscale(oriented, (rect.width, rect.height))
        return scaled

    def _traffic_light_colors(self, light) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Map traffic light phase to north-south and east-west colors."""
        phase = getattr(light, "current_phase", "ns_green")
        phase_value = getattr(phase, "value", phase)

        green = self.palette["green"]
        yellow = self.palette["yellow"]
        red = self.palette["red"]

        if phase_value == "ns_green":
            return green, red
        if phase_value == "ns_yellow":
            return yellow, red
        if phase_value == "ew_green":
            return red, green
        if phase_value == "ew_yellow":
            return red, yellow
        return red, red

    def _draw_central_traffic_light(self, light) -> None:
        center_x = self.width // 2
        center_y = self.height // 2
        ns_color, ew_color = self._traffic_light_colors(light)

        # Four single-bulb lights (top/bottom for NS, left/right for EW).
        radius = 8
        offset = 28
        # NS controls top/bottom, EW controls left/right.
        self._draw_light_box(center_x, center_y - offset, ns_color, radius)
        self._draw_light_box(center_x, center_y + offset, ns_color, radius)
        self._draw_light_box(center_x - offset, center_y, ew_color, radius)
        self._draw_light_box(center_x + offset, center_y, ew_color, radius)


    def _phase_to_signal_color(self, phase_value: str) -> tuple[int, int, int]:
        if phase_value in ("ns_green", "ew_green"):
            return (70, 220, 120)
        if phase_value in ("ns_yellow", "ew_yellow"):
            return (250, 200, 60)
        return (235, 60, 60)

    def _draw_light_box(
        self, x: int, y: int, color: tuple[int, int, int], radius: int
    ) -> None:
        """Draw a single light with its own compact black housing."""
        box_size = radius * 3
        box = pygame.Rect(x - box_size // 2, y - box_size // 2, box_size, box_size)
        pygame.draw.rect(self.screen, (8, 8, 10), box, border_radius=4)
        pygame.draw.rect(self.screen, (50, 50, 58), box, width=1, border_radius=4)
        pygame.draw.circle(self.screen, color, (x, y), radius)

    def _draw_controls(self, controls: dict) -> None:
        """Draw on-screen sliders/toggles for runtime control."""
        panel_w = 280
        panel_h = 160
        x = self.width - panel_w - 12
        y = 12
        panel = pygame.Rect(x, y, panel_w, panel_h)
        pygame.draw.rect(self.screen, (15, 18, 24), panel, border_radius=6)
        pygame.draw.rect(self.screen, (60, 60, 70), panel, width=1, border_radius=6)

        speed = float(controls.get("speed", 1.0))
        risk_label = str(controls.get("risk_label", "LOW")).upper()
        emergency = bool(controls.get("emergency", True))
        phase_label = str(controls.get("phase", ""))

        label_color = self.palette["hud_text"]
        self.screen.blit(self.font.render(f"Speed: {speed:.2f}x", True, label_color), (x + 10, y + 8))
        self.screen.blit(self.font.render("Risk", True, label_color), (x + 10, y + 44))
        self.screen.blit(self.font.render(f"Risk: {risk_label}", True, label_color), (x + 120, y + 44))
        em_text = "Emergency: ON" if emergency else "Emergency: OFF"
        self.screen.blit(self.font.render(em_text, True, label_color), (x + 10, y + 78))
        # Phase display removed per request.

        self._draw_slider(x + 90, y + 18, 160, speed, 0.5, 2.0)
        # Risk is a toggle (LOW/MED/HIGH), so no slider is drawn.

    def _draw_slider(
        self, x: int, y: int, width: int, value: float, vmin: float, vmax: float
    ) -> None:
        track = pygame.Rect(x, y + 6, width, 4)
        pygame.draw.rect(self.screen, (70, 70, 80), track, border_radius=2)
        t = 0.0 if vmax == vmin else (value - vmin) / (vmax - vmin)
        t = max(0.0, min(1.0, t))
        knob_x = x + int(t * width)
        pygame.draw.circle(self.screen, (230, 230, 230), (knob_x, y + 8), 6)


    def _draw_background(self) -> None:
        self.palette = {
            "asphalt": (22, 26, 34),
            "road": (46, 52, 64),
            "lane_mark": (210, 210, 210),
            "lane_mark_dim": (160, 160, 160),
            "vehicle": (88, 170, 240),
            "green": (70, 210, 120),
            "yellow": (240, 200, 90),
            "red": (225, 80, 80),
            "hud_text": (235, 235, 235),
        }
        self.screen.fill(self.palette["asphalt"])

        # Subtle vignette
        vignette = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        vignette.fill((0, 0, 0, 60))
        pygame.draw.rect(
            vignette,
            (0, 0, 0, 0),
            pygame.Rect(60, 60, self.width - 120, self.height - 120),
        )
        self.screen.blit(vignette, (0, 0))

    def _draw_lane_markings(self) -> None:
        center_x = self.width // 2
        center_y = self.height // 2
        # Single solid yellow center lines for two-way roads.
        center_color = (230, 200, 60)
        center_width = 4

        # North-south road center line.
        pygame.draw.line(
            self.screen,
            center_color,
            (center_x, 0),
            (center_x, self.height),
            center_width,
        )

        # East-west road center line.
        pygame.draw.line(
            self.screen,
            center_color,
            (0, center_y),
            (self.width, center_y),
            center_width,
        )

    def _flashing_emergency_color(self) -> tuple[int, int, int]:
        # Simple flash between red and white.
        pulse = int(self._flash_timer * 6) % 2
        return (240, 70, 70) if pulse == 0 else (240, 240, 240)
