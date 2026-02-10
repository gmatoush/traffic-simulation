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

        if not paused:
            sim_accumulator += real_dt * sim_clock.speed
            while sim_accumulator >= sim_clock.dt:
                world.update(sim_clock.tick())
                sim_accumulator -= sim_clock.dt

        renderer.render(world, sim_speed=sim_clock.speed, show_overlays=show_overlays)

    pygame.quit()


if __name__ == "__main__":
    main()
