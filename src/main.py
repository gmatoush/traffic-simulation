"""Entry point for the traffic simulation demo."""

from __future__ import annotations

from config.sim_config import HEADLESS_STEPS, RENDER_ENABLED, RENDER_SPEED, SIM_DT
from sim.clock import SimulationClock
from sim.traffic_light import TrafficLight
from sim.world import World


def main() -> None:
    sim_clock = SimulationClock(dt=SIM_DT, speed=RENDER_SPEED)
    world = World(traffic_light=TrafficLight())

    if not RENDER_ENABLED:
        # Headless mode: run as fast as possible without pygame.
        for _ in range(HEADLESS_STEPS):
            world.update(sim_clock.tick())
        return

    import pygame
    from render.pygame_renderer import PygameRenderer

    pygame.init()
    renderer = PygameRenderer()
    frame_clock = pygame.time.Clock()
    sim_accumulator = 0.0
    running = True
    paused = False
    show_overlays = True
    dragging_speed = False
    ui_speed = sim_clock.speed
    risk_levels = ["LOW", "MED", "HIGH"]
    risk_index = 0
    emergency_enabled = True

    while running:
        real_dt = frame_clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                renderer.resize(event.w, event.h)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_PERIOD:
                    if paused:
                        world.update(sim_clock.tick())
                elif event.key in (pygame.K_TAB, pygame.K_d):
                    show_overlays = not show_overlays
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    sim_clock.speed = min(sim_clock.speed * 1.25, 10.0)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    sim_clock.speed = max(sim_clock.speed / 1.25, 0.1)
                elif event.key == pygame.K_e:
                    emergency_enabled = not emergency_enabled
                elif event.key == pygame.K_r:
                    risk_index = (risk_index + 1) % len(risk_levels)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    sx = _speed_slider_rect(renderer.width, renderer.height)
                    ex = _emergency_toggle_rect(renderer.width, renderer.height)
                    rx = _risk_toggle_rect(renderer.width, renderer.height)
                    if sx.collidepoint(event.pos):
                        dragging_speed = True
                    elif rx.collidepoint(event.pos):
                        risk_index = (risk_index + 1) % len(risk_levels)
                    elif ex.collidepoint(event.pos):
                        emergency_enabled = not emergency_enabled
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging_speed = False
            elif event.type == pygame.MOUSEMOTION:
                if dragging_speed:
                    ui_speed = _slider_value(
                        event.pos[0], _speed_slider_rect(renderer.width, renderer.height), 0.5, 2.0
                    )

        sim_clock.speed = ui_speed
        risk_value = [0.0, 0.5, 1.0][risk_index]
        world.risk_factor = risk_value
        world.emergency_enabled = emergency_enabled

        if not paused:
            sim_accumulator += real_dt * sim_clock.speed
            while sim_accumulator >= sim_clock.dt:
                world.update(sim_clock.tick())
                sim_accumulator -= sim_clock.dt

        renderer.render(
            world,
            sim_speed=sim_clock.speed,
            show_overlays=show_overlays,
            controls={
                "speed": ui_speed,
                "risk_label": risk_levels[risk_index],
                "emergency": emergency_enabled,
            },
        )

    pygame.quit()


def _speed_slider_rect(width: int, height: int):
    import pygame
    x = width - 280 - 12
    y = 12
    return pygame.Rect(x + 90, y + 18, 160, 16)


def _slider_value(mouse_x: int, rect, vmin: float, vmax: float) -> float:
    t = (mouse_x - rect.x) / float(rect.width)
    t = max(0.0, min(1.0, t))
    return vmin + t * (vmax - vmin)


def _emergency_toggle_rect(width: int, height: int):
    import pygame
    x = width - 280 - 12
    y = 12
    return pygame.Rect(x + 10, y + 76, 220, 22)


def _risk_toggle_rect(width: int, height: int):
    import pygame
    x = width - 280 - 12
    y = 12
    return pygame.Rect(x + 10, y + 40, 220, 22)


if __name__ == "__main__":
    main()
