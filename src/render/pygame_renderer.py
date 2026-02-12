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
        pygame.display.set_caption("Traffic Simulator")
        self.screen = pygame.display.set_mode(
            (self.width, self.height), pygame.RESIZABLE
        )
        self._set_window_icon()
        self.font = pygame.font.Font(None, 24)
        self._flash_timer = 0.0
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
        self._static_scene_surface: pygame.Surface | None = None
        self._vehicle_sprites: dict[int, pygame.Surface] = {}
        self._vehicle_sprite_by_instance: dict[int, pygame.Surface] = {}
        self._transformed_sprite_cache: dict[tuple[int, str, int, int], pygame.Surface] = {}
        self._dimmed_sprite_cache: dict[int, pygame.Surface] = {}
        self._sprite_pool: dict[str, list[pygame.Surface]] = {}
        self._control_icons: dict[str, pygame.Surface] = {}
        self._load_vehicle_sprites()
        self._load_control_icons()

    def resize(self, width: int, height: int) -> None:
        self.width = max(480, int(width))
        self.height = max(360, int(height))
        self.screen = pygame.display.set_mode(
            (self.width, self.height), pygame.RESIZABLE
        )
        self._static_scene_surface = None

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

    def _load_control_icons(self) -> None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        play_path = os.path.join(base_dir, "assets", "start_button.png")
        stop_path = os.path.join(base_dir, "assets", "stop_button.png")
        icons: dict[str, pygame.Surface] = {}
        try:
            if os.path.isfile(play_path):
                icons["play"] = pygame.image.load(play_path).convert_alpha()
            if os.path.isfile(stop_path):
                icons["stop"] = pygame.image.load(stop_path).convert_alpha()
        except pygame.error:
            icons = {}
        self._control_icons = icons

    def _set_window_icon(self) -> None:
        """Set the pygame window icon if an app icon is available."""
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        icon_path = os.path.join(base_dir, "assets", "app_icon.png")
        if not os.path.isfile(icon_path):
            return
        try:
            icon = pygame.image.load(icon_path).convert_alpha()
        except pygame.error:
            return
        pygame.display.set_icon(icon)

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
        stats: dict | None = None,
    ) -> None:
        """Render the world state without mutating it.

        Args:
            world: Simulation world containing vehicles and traffic light.
            sim_speed: Optional simulation speed multiplier for HUD display.
            show_overlays: Toggle debug overlay rendering.
            controls: Optional UI control state for on-screen sliders/toggles.
            stats: Optional stats for left-side display.
        """
        self._flash_timer += 1.0 / 60.0
        self._draw_static_scene()
        self._prune_vehicle_sprite_cache(world)
        # Render vehicles as sprites (fallback to rectangles if none loaded).
        has_any_sprites = any(self._sprite_pool.values())
        for vehicle in getattr(world, "vehicles", []):
            rect = self._vehicle_rect(vehicle)
            if vehicle.__class__.__name__ == "EmergencyVehicle":
                self._draw_emergency_glow(rect)
            if has_any_sprites:
                sprite = self._get_vehicle_sprite(vehicle)
                sprite = self._orient_and_scale_sprite(sprite, rect, vehicle)
                if getattr(vehicle, "crashed", False):
                    sprite = self._dim_sprite(sprite)
                self.screen.blit(sprite, rect.topleft)
            else:
                color = self._vehicle_color(vehicle)
                if getattr(vehicle, "crashed", False):
                    color = (90, 90, 90)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (8, 8, 10), rect, width=1)

        self._draw_crash_effects(world)
        self._draw_crash_banner(world)

        # Render central traffic light indicator.
        light = getattr(world, "traffic_light", None)
        if light is not None:
            self._draw_central_traffic_light(light)

        if controls is not None:
            self._draw_controls(controls)
        if stats is not None:
            self._draw_left_stats(stats)

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
        instance_key = id(vehicle)
        cached_instance_sprite = self._vehicle_sprite_by_instance.get(instance_key)
        if cached_instance_sprite is not None:
            return cached_instance_sprite

        kind = getattr(vehicle, "sprite_kind", "normal")
        category = getattr(vehicle, "sprite_category", "medium")
        seed = getattr(vehicle, "sprite_seed", 0)
        key = (kind, category, seed)
        if key not in self._vehicle_sprites:
            pool = self._sprite_pool.get("emergency" if kind == "emergency" else category, [])
            if not pool:
                fallback = []
                for sprites in self._sprite_pool.values():
                    fallback.extend(sprites)
                pool = fallback
            chooser = random.Random(seed)
            self._vehicle_sprites[key] = chooser.choice(pool)
        chosen = self._vehicle_sprites[key]
        self._vehicle_sprite_by_instance[instance_key] = chosen
        return chosen

    def _prune_vehicle_sprite_cache(self, world) -> None:
        vehicles = getattr(world, "vehicles", [])
        active_ids = {id(v) for v in vehicles}
        stale = [k for k in self._vehicle_sprite_by_instance.keys() if k not in active_ids]
        for key in stale:
            self._vehicle_sprite_by_instance.pop(key, None)

    def _orient_and_scale_sprite(
        self, sprite: pygame.Surface, rect: pygame.Rect, vehicle
    ) -> pygame.Surface:
        lane = getattr(vehicle, "lane", None)
        lane_value = getattr(lane, "value", lane)
        cache_key = (id(sprite), str(lane_value), rect.width, rect.height)
        cached = self._transformed_sprite_cache.get(cache_key)
        if cached is not None:
            return cached

        oriented = sprite
        # Sprites are vertical; rotate/flip based on travel direction.
        if lane_value == "NORTH":
            oriented = pygame.transform.rotate(oriented, 180)
        elif lane_value == "EAST":
            oriented = pygame.transform.rotate(oriented, 90)
        elif lane_value == "WEST":
            oriented = pygame.transform.rotate(oriented, -90)
        scaled = pygame.transform.smoothscale(oriented, (rect.width, rect.height))
        self._transformed_sprite_cache[cache_key] = scaled
        return scaled

    def _dim_sprite(self, sprite: pygame.Surface) -> pygame.Surface:
        cached = self._dimmed_sprite_cache.get(id(sprite))
        if cached is not None:
            return cached
        dimmed = sprite.copy()
        overlay = pygame.Surface(dimmed.get_size(), pygame.SRCALPHA)
        overlay.fill((60, 60, 60, 160))
        dimmed.blit(overlay, (0, 0))
        self._dimmed_sprite_cache[id(sprite)] = dimmed
        return dimmed

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
        show_mode_algo = "mode" in controls or "algo" in controls
        panel_h = 312 if show_mode_algo else 242
        x = self.width - panel_w - 12
        y = 12
        panel = pygame.Rect(x, y, panel_w, panel_h)
        pygame.draw.rect(self.screen, (15, 18, 24), panel, border_radius=6)
        pygame.draw.rect(self.screen, (60, 60, 70), panel, width=1, border_radius=6)

        speed = float(controls.get("speed", 1.0))
        risk_label = str(controls.get("risk_label", "LOW")).upper()
        phase_label = str(controls.get("phase", ""))
        mode_label = str(controls.get("mode", "")).upper()
        algo_label = str(controls.get("algo", "")).upper()

        label_color = self.palette["hud_text"]
        self.screen.blit(self.font.render(f"Speed: {speed:.2f}x", True, label_color), (x + 10, y + 8))
        self.screen.blit(self.font.render(f"Spawn Rate: {risk_label}", True, label_color), (x + 10, y + 44))
        if phase_label:
            self.screen.blit(self.font.render(phase_label, True, label_color), (x + 10, y + 62))

        self._draw_slider(x + 90, y + 18, 160, speed, 0.5, 2.0)
        # Risk is a toggle (LOW/MED/HIGH), so no slider is drawn.
        self._draw_icon_button(x + 10, y + 82, 120, 44, "play")
        self._draw_icon_button(x + 150, y + 82, 120, 44, "stop")
        self._draw_button(x + 10, y + 136, 260, 28, "Load Model")
        self._draw_button(x + 10, y + 170, 260, 28, "Train Model")
        if show_mode_algo:
            self.screen.blit(self.font.render("Mode", True, label_color), (x + 10, y + 214))
            self._draw_button(x + 90, y + 208, 180, 28, mode_label)
            self.screen.blit(self.font.render("Algo", True, label_color), (x + 10, y + 250))
            self._draw_button(x + 90, y + 244, 180, 28, algo_label)

    def _draw_left_stats(self, stats: dict) -> None:
        """Draw key metrics on the upper-left."""
        panel_x = 12
        panel_y = 12
        x = panel_x + 10
        y = panel_y + 8
        avg_wait = stats.get("avg_wait", None)
        learn = stats.get("learning", None)
        elapsed = stats.get("elapsed", None)
        throughput = stats.get("throughput", None)
        completed = stats.get("completed", None)
        crashes_per_min = stats.get("crashes_per_min", None)

        elapsed_surface = None
        elapsed_h = 0
        elapsed_w = 0
        if elapsed is not None:
            big_font = pygame.font.Font(None, 40)
            elapsed_surface = big_font.render(
                f"Elapsed: {elapsed:.1f}s", True, self.palette["hud_text"]
            )
            elapsed_w = elapsed_surface.get_width()
            elapsed_h = 34

        lines = []
        if avg_wait is not None:
            lines.append(f"Avg stopped wait: {avg_wait:.2f}s")
        if throughput is not None:
            lines.append(f"Vehicles per minute (vehicles/min): {throughput:.1f}")
        if crashes_per_min is not None:
            lines.append(f"Crashes per minute (crashes/min): {crashes_per_min:.2f}")
        if completed is not None:
            lines.append(f"Vehicles cleared: {int(completed)}")
        if learn is not None:
            lines.append(f"Avg reward(100): {learn:.2f}")

        line_width = 0
        for line in lines:
            line_width = max(line_width, self.font.size(line)[0])
        content_w = max(elapsed_w, line_width)
        panel_w = max(260, content_w + 20)
        panel_h = 16 + elapsed_h + (18 * len(lines)) + 8
        panel = pygame.Rect(panel_x, panel_y, panel_w, panel_h)
        pygame.draw.rect(self.screen, (15, 18, 24), panel, border_radius=6)
        pygame.draw.rect(self.screen, (60, 60, 70), panel, width=1, border_radius=6)

        if elapsed_surface is not None:
            self.screen.blit(elapsed_surface, (x, y))
            y += elapsed_h
        for line in lines:
            surface = self.font.render(line, True, self.palette["hud_text"])
            self.screen.blit(surface, (x, y))
            y += 18

    def _draw_crash_effects(self, world) -> None:
        """Draw crash indicators for crashed vehicles."""
        vehicles = getattr(world, "vehicles", [])
        if not vehicles:
            return
        for vehicle in vehicles:
            if not getattr(vehicle, "crashed", False):
                continue
            lane = getattr(vehicle, "lane", None)
            timer = 0.0
            if lane is not None and hasattr(world, "lane_freeze_timers"):
                timer = float(world.lane_freeze_timers.get(lane, 0.0))
            if timer <= 0.0:
                continue
            rect = self._vehicle_rect(vehicle)
            cx, cy = rect.center
            size = max(rect.width, rect.height) // 2 + 6
            color = (220, 60, 60)
            pygame.draw.line(self.screen, color, (cx - size, cy - size), (cx + size, cy + size), 3)
            pygame.draw.line(self.screen, color, (cx - size, cy + size), (cx + size, cy - size), 3)

    def _draw_crash_banner(self, world) -> None:
        """Draw a clear banner when any crashed vehicle is present."""
        crashed = any(getattr(v, "crashed", False) for v in getattr(world, "vehicles", []))
        if not crashed:
            return
        banner_w = 300
        banner_h = 44
        x = (self.width - banner_w) // 2
        y = 14
        rect = pygame.Rect(x, y, banner_w, banner_h)
        pygame.draw.rect(self.screen, (130, 20, 20), rect, border_radius=8)
        pygame.draw.rect(self.screen, (245, 215, 215), rect, width=2, border_radius=8)
        text = self.font.render("CRASH DETECTED - RESETTING...", True, (255, 240, 240))
        text_rect = text.get_rect(center=rect.center)
        self.screen.blit(text, text_rect)

    def _draw_slider(
        self, x: int, y: int, width: int, value: float, vmin: float, vmax: float
    ) -> None:
        track = pygame.Rect(x, y + 6, width, 4)
        pygame.draw.rect(self.screen, (70, 70, 80), track, border_radius=2)
        t = 0.0 if vmax == vmin else (value - vmin) / (vmax - vmin)
        t = max(0.0, min(1.0, t))
        knob_x = x + int(t * width)
        pygame.draw.circle(self.screen, (230, 230, 230), (knob_x, y + 8), 6)

    def _draw_button(self, x: int, y: int, width: int, height: int, label: str) -> None:
        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, (24, 28, 36), rect, border_radius=6)
        pygame.draw.rect(self.screen, (70, 70, 80), rect, width=1, border_radius=6)
        text = self.font.render(label, True, self.palette["hud_text"])
        text_rect = text.get_rect(center=rect.center)
        self.screen.blit(text, text_rect)

    def _draw_checkbox(self, x: int, y: int, size: int, checked: bool) -> None:
        rect = pygame.Rect(x, y, size, size)
        pygame.draw.rect(self.screen, (24, 28, 36), rect, border_radius=4)
        pygame.draw.rect(self.screen, (70, 70, 80), rect, width=1, border_radius=4)
        if checked:
            inner = rect.inflate(-6, -6)
            pygame.draw.rect(self.screen, (200, 220, 200), inner, border_radius=3)

    def _draw_icon_button(self, x: int, y: int, width: int, height: int, icon: str) -> None:
        rect = pygame.Rect(x, y, width, height)
        sprite = self._control_icons.get(icon)
        if sprite is None:
            return
        target = pygame.transform.smoothscale(sprite, (height, height))
        icon_rect = target.get_rect(center=rect.center)
        self.screen.blit(target, icon_rect)


    def _draw_background(self, surface: pygame.Surface) -> None:
        surface.fill(self.palette["asphalt"])

        # Subtle vignette
        vignette = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        vignette.fill((0, 0, 0, 60))
        pygame.draw.rect(
            vignette,
            (0, 0, 0, 0),
            pygame.Rect(60, 60, self.width - 120, self.height - 120),
        )
        surface.blit(vignette, (0, 0))

    def _draw_lane_markings(self, surface: pygame.Surface) -> None:
        center_x = self.width // 2
        center_y = self.height // 2
        # Dashed yellow center lines for two-way roads.
        center_color = (230, 200, 60)
        center_width = 4
        dash = 18
        gap = 14

        # North-south dashed center line.
        y = 0
        while y < self.height:
            pygame.draw.line(
                surface,
                center_color,
                (center_x, y),
                (center_x, min(self.height, y + dash)),
                center_width,
            )
            y += dash + gap

        # East-west dashed center line.
        x = 0
        while x < self.width:
            pygame.draw.line(
                surface,
                center_color,
                (x, center_y),
                (min(self.width, x + dash), center_y),
                center_width,
            )
            x += dash + gap

    def _draw_static_scene(self) -> None:
        if (
            self._static_scene_surface is None
            or self._static_scene_surface.get_width() != self.width
            or self._static_scene_surface.get_height() != self.height
        ):
            scene = pygame.Surface((self.width, self.height))
            self._draw_background(scene)
            horizontal = pygame.Rect(
                0, self.height // 2 - self.road_width // 2, self.width, self.road_width
            )
            vertical = pygame.Rect(
                self.width // 2 - self.road_width // 2, 0, self.road_width, self.height
            )
            pygame.draw.rect(scene, self.palette["road"], horizontal)
            pygame.draw.rect(scene, self.palette["road"], vertical)
            self._draw_lane_markings(scene)
            self._static_scene_surface = scene
        self.screen.blit(self._static_scene_surface, (0, 0))

    def _flashing_emergency_color(self) -> tuple[int, int, int]:
        # Flash between red and blue.
        pulse = int(self._flash_timer * 6) % 2
        return (240, 70, 70) if pulse == 0 else (60, 120, 240)

    def _draw_emergency_glow(self, rect: pygame.Rect) -> None:
        """Draw a faint circular red/blue glow that fades toward the edges."""
        color = self._flashing_emergency_color()
        radius = max(rect.width, rect.height) // 2 + 8
        size = radius * 2
        glow = pygame.Surface((size, size), pygame.SRCALPHA)
        # Radial falloff using concentric circles to keep a smooth fade.
        for r in range(radius, 0, -4):
            alpha = int(70 * (r / radius))
            pygame.draw.circle(glow, (*color, alpha), (radius, radius), r)
        self.screen.blit(glow, (rect.centerx - radius, rect.centery - radius))
