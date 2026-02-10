"""Entry point for the traffic simulation demo."""

from __future__ import annotations

from collections import deque
import os
import threading

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
    from env.traffic_env import TrafficEnv
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    pygame.init()
    renderer = PygameRenderer()
    frame_clock = pygame.time.Clock()
    running = True
    paused = False
    show_overlays = True
    dragging_speed = False
    ui_speed = sim_clock.speed
    risk_levels = ["LOW", "MED", "HIGH"]
    risk_index = 0
    uniform_speed = False

    class LiveTrainingCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.step_rewards = deque(maxlen=100)
            self.episode_return = 0.0

        def _on_step(self) -> bool:
            rewards = self.locals.get("rewards", [0.0])
            dones = self.locals.get("dones", [False])
            reward = float(rewards[0]) if rewards is not None else 0.0
            self.step_rewards.append(reward)
            self.episode_return += reward
            if dones[0]:
                self.episode_return = 0.0
            return True

        def avg_reward(self) -> float:
            if not self.step_rewards:
                return 0.0
            return sum(self.step_rewards) / len(self.step_rewards)

    models_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(models_dir, exist_ok=True)
    live_model_path = os.path.join(models_dir, "traffic_rl_light")
    heavy_model_path = os.path.join(models_dir, "traffic_rl_heavy")

    live_env = DummyVecEnv([lambda: TrafficEnv(render_enabled=False, max_steps=10**9)])
    model = DQN(
        "MultiInputPolicy",
        live_env,
        verbose=0,
        buffer_size=5000,
        learning_starts=100,
        batch_size=32,
        train_freq=1,
        gradient_steps=1,
    )
    train_cb = LiveTrainingCallback()
    train_accum = 0.0
    stop_event = threading.Event()

    def _background_trainer() -> None:
        heavy_env = DummyVecEnv([lambda: TrafficEnv(render_enabled=False, max_steps=10**9)])
        heavy_model = PPO(
            "MultiInputPolicy",
            heavy_env,
            verbose=0,
            n_steps=256,
            batch_size=128,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            learning_rate=3e-4,
        )
        steps = 0
        while not stop_event.is_set():
            heavy_model.learn(total_timesteps=256, reset_num_timesteps=False)
            steps += 256
            if steps % 2048 == 0:
                heavy_model.save(heavy_model_path)

    bg_thread = threading.Thread(target=_background_trainer, daemon=True)
    bg_thread.start()

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
                elif event.key == pygame.K_r:
                    risk_index = (risk_index + 1) % len(risk_levels)
                elif event.key == pygame.K_u:
                    uniform_speed = not uniform_speed
                elif event.key == pygame.K_s and (event.mod & pygame.KMOD_CTRL):
                    model.save(live_model_path)
                elif event.key == pygame.K_l and (event.mod & pygame.KMOD_CTRL):
                    try:
                        model = DQN.load(live_model_path, env=live_env)
                    except Exception:
                        pass
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    sx = _speed_slider_rect(renderer.width, renderer.height)
                    rx = _risk_toggle_rect(renderer.width, renderer.height)
                    ux = _uniform_speed_rect(renderer.width, renderer.height)
                    play_btn = _play_button_rect(renderer.width, renderer.height)
                    stop_btn = _stop_button_rect(renderer.width, renderer.height)
                    save_btn = _save_button_rect(renderer.width, renderer.height)
                    load_btn = _load_button_rect(renderer.width, renderer.height)
                    if sx.collidepoint(event.pos):
                        dragging_speed = True
                    elif rx.collidepoint(event.pos):
                        risk_index = (risk_index + 1) % len(risk_levels)
                        new_risk_value = [0.0, 0.5, 1.0][risk_index]
                        live_env.envs[0].world = World(traffic_light=TrafficLight())
                        env_world = live_env.envs[0].world
                        env_world.risk_factor = new_risk_value
                        env_world.uniform_speed_enabled = uniform_speed
                        env_world.max_vehicles = 20 + 10 * risk_index
                    elif ux.collidepoint(event.pos):
                        uniform_speed = not uniform_speed
                        live_env.envs[0].world = World(traffic_light=TrafficLight())
                        env_world = live_env.envs[0].world
                        env_world.risk_factor = risk_value
                        env_world.uniform_speed_enabled = uniform_speed
                    elif play_btn.collidepoint(event.pos):
                        paused = False
                    elif stop_btn.collidepoint(event.pos):
                        paused = True
                    elif save_btn.collidepoint(event.pos):
                        path = _choose_model_path(models_dir, save=True)
                        if path:
                            try:
                                model.save(path)
                                _show_dialog("Save Model", f"Saved model to:\n{path}")
                            except Exception as exc:
                                _show_dialog("Save Model Failed", f"Failed to save model:\n{exc}")
                    elif load_btn.collidepoint(event.pos):
                        path = _choose_model_path(models_dir, save=False)
                        if path:
                            try:
                                model = DQN.load(path, env=live_env)
                                _show_dialog("Load Model", f"Loaded model from:\n{path}")
                            except Exception as exc:
                                _show_dialog("Load Model Failed", f"Failed to load model:\n{exc}")
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

        env_world = live_env.envs[0].world
        env_world.risk_factor = risk_value
        env_world.uniform_speed_enabled = uniform_speed
        env_world.max_vehicles = 20 + 10 * risk_index
        world = env_world

        window_active = pygame.display.get_active()

        if not paused:
            if not window_active:
                # When minimized/hidden, train as fast as possible without rendering.
                for _ in range(200):
                    model.learn(total_timesteps=1, reset_num_timesteps=False, callback=train_cb)
            else:
                train_accum += real_dt * sim_clock.speed
                while train_accum >= sim_clock.dt:
                    model.learn(total_timesteps=1, reset_num_timesteps=False, callback=train_cb)
                    train_accum -= sim_clock.dt

        stopped = [v for v in world.vehicles if getattr(v, "wait_time", 0.0) > 0.0]
        avg_wait = sum(v.wait_time for v in stopped) / len(stopped) if stopped else 0.0
        if window_active:
            renderer.render(
                world,
                sim_speed=sim_clock.speed,
                show_overlays=show_overlays,
            controls={
                "speed": ui_speed,
                "risk_label": risk_levels[risk_index],
                "uniform_speed": uniform_speed,
            },
                stats={
                    "avg_wait": avg_wait,
                    "elapsed": world.time,
                    "learning": train_cb.avg_reward(),
                },
            )

    pygame.quit()
    stop_event.set()


def _speed_slider_rect(width: int, height: int):
    import pygame
    x = width - 280 - 12
    y = 12
    return pygame.Rect(x + 90, y + 18, 160, 16)


def _slider_value(mouse_x: int, rect, vmin: float, vmax: float) -> float:
    t = (mouse_x - rect.x) / float(rect.width)
    t = max(0.0, min(1.0, t))
    return vmin + t * (vmax - vmin)


def _risk_toggle_rect(width: int, height: int):
    import pygame
    x = width - 280 - 12
    y = 12
    return pygame.Rect(x + 10, y + 40, 220, 22)


def _uniform_speed_rect(width: int, height: int):
    import pygame
    x = width - 280 - 12
    y = 12
    return pygame.Rect(x + 10, y + 132, 220, 22)


def _save_button_rect(width: int, height: int):
    import pygame
    x = width - 280 - 12
    y = 12
    return pygame.Rect(x + 10, y + 170, 120, 28)


def _load_button_rect(width: int, height: int):
    import pygame
    x = width - 280 - 12
    y = 12
    return pygame.Rect(x + 150, y + 170, 120, 28)


def _play_button_rect(width: int, height: int):
    import pygame
    x = width - 280 - 12
    y = 12
    return pygame.Rect(x + 10, y + 82, 120, 44)


def _stop_button_rect(width: int, height: int):
    import pygame
    x = width - 280 - 12
    y = 12
    return pygame.Rect(x + 150, y + 82, 120, 44)


def _choose_model_path(models_dir: str, save: bool) -> str | None:
    """Open a file dialog rooted in the models directory."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None
    root = tk.Tk()
    root.withdraw()
    filetypes = [("Model files", "*.zip"), ("All files", "*.*")]
    if save:
        path = filedialog.asksaveasfilename(
            initialdir=models_dir,
            defaultextension=".zip",
            filetypes=filetypes,
            title="Save RL Model",
        )
    else:
        path = filedialog.askopenfilename(
            initialdir=models_dir,
            filetypes=filetypes,
            title="Load RL Model",
        )
    root.destroy()
    return path or None


def _show_dialog(title: str, message: str) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox
    except Exception:
        return
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(title, message)
    root.destroy()


if __name__ == "__main__":
    main()
