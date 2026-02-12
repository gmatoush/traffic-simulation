"""Entry point for rendered PPO inference (training is headless/offline)."""

from __future__ import annotations

import os
import threading

from config.sim_config import HEADLESS_STEPS, RENDER_ENABLED, RENDER_SPEED, SIM_DT
from env.traffic_env import TrafficEnv
from rl.rl_controller import RLController
from sim.clock import SimulationClock
from sim.traffic_light import TrafficLight
from sim.world import World

DEFAULT_BASE_MODEL_NAME = "traffic_ppo_base.zip"


def main() -> None:
    sim_clock = SimulationClock(dt=SIM_DT, speed=RENDER_SPEED)
    world = World(traffic_light=TrafficLight())

    if not RENDER_ENABLED:
        for _ in range(HEADLESS_STEPS):
            world.update(sim_clock.tick())
        return

    import pygame
    from render.pygame_renderer import PygameRenderer

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

    env = TrafficEnv(
        render_enabled=False,
        max_steps=10**9,
        action_repeat=1,
        reset_on_crash=True,
        crash_pause_duration=5.0,
    )
    obs, _ = env.reset()
    step_seconds = env.dt * env.action_repeat
    step_accum = 0.0
    max_steps_per_frame = 3
    models_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(models_dir, exist_ok=True)
    default_model_path = os.path.join(models_dir, DEFAULT_BASE_MODEL_NAME)
    elapsed_offset = 0.0
    completed_offset = 0
    crash_offset = 0
    last_world_time = env.world.time
    last_completed = int(getattr(env.world, "completed_vehicles", 0))
    last_total_crashes = int(getattr(env.world, "total_crashes", 0))
    training_state: dict[str, object] = {
        "active": False,
        "progress": 0.0,
        "message": "",
        "result_path": None,
        "error": None,
    }
    state_lock = threading.Lock()

    loaded_startup = _load_default_or_prompt_model(
        models_dir=models_dir,
        default_model_path=default_model_path,
        renderer=renderer,
    )
    if loaded_startup is None:
        _show_dialog("Model Required", "No PPO model selected. Exiting.")
        pygame.quit()
        env.close()
        return
    rl_controller, loaded_model_name = loaded_startup
    compare_mode = False
    compare_session: dict[str, object] | None = None

    def _reset_simulation() -> None:
        nonlocal obs
        nonlocal step_accum
        nonlocal elapsed_offset
        nonlocal completed_offset
        nonlocal crash_offset
        nonlocal last_world_time
        nonlocal last_completed
        nonlocal last_total_crashes
        obs, _ = env.reset()
        step_accum = 0.0
        elapsed_offset = 0.0
        completed_offset = 0
        crash_offset = 0
        last_world_time = env.world.time
        last_completed = int(getattr(env.world, "completed_vehicles", 0))
        last_total_crashes = int(getattr(env.world, "total_crashes", 0))

    def _init_compare_agent_state(model_path: str) -> dict[str, object]:
        agent_env = TrafficEnv(
            render_enabled=False,
            max_steps=10**9,
            action_repeat=1,
            reset_on_crash=False,
            crash_pause_duration=0.0,
        )
        agent_obs, _ = agent_env.reset()
        model_name = os.path.basename(model_path)
        return {
            "name": model_name,
            "controller": RLController(algo="PPO", model_path=model_path),
            "env": agent_env,
            "obs": agent_obs,
            "step_accum": 0.0,
            "elapsed_offset": 0.0,
            "completed_offset": 0,
            "crash_offset": 0,
            "last_world_time": agent_env.world.time,
            "last_completed": int(getattr(agent_env.world, "completed_vehicles", 0)),
            "last_total_crashes": int(getattr(agent_env.world, "total_crashes", 0)),
            "episodes": 0,
            "wall_elapsed": 0.0,
            "avg_wait_sum": 0.0,
            "avg_wait_samples": 0,
        }

    def _close_compare_session() -> None:
        nonlocal compare_session
        if compare_session is None:
            return
        for key in ("left", "right"):
            side = compare_session.get(key)
            if isinstance(side, dict):
                side_env = side.get("env")
                if side_env is not None and hasattr(side_env, "close"):
                    side_env.close()
        compare_session = None

    def _step_compare_agent(agent: dict[str, object], real_seconds: float) -> dict[str, object]:
        agent_env = agent["env"]
        if not isinstance(agent_env, TrafficEnv):
            raise TypeError("Invalid compare agent env")
        controller = agent["controller"]
        if not isinstance(controller, RLController):
            raise TypeError("Invalid compare agent controller")

        scaled_real_seconds = real_seconds * sim_clock.speed
        agent["wall_elapsed"] = float(agent["wall_elapsed"]) + scaled_real_seconds
        step_accum_local = float(agent["step_accum"])
        step_accum_local += scaled_real_seconds
        steps_this_frame = 0
        while step_accum_local >= step_seconds and steps_this_frame < max_steps_per_frame:
            obs_local = agent["obs"]
            obs_local, _, terminated, truncated, _ = agent_env.step(controller.act(obs_local))
            current_time = agent_env.world.time
            current_completed = int(getattr(agent_env.world, "completed_vehicles", 0))
            current_total_crashes = int(getattr(agent_env.world, "total_crashes", 0))

            if current_time < float(agent["last_world_time"]):
                agent["elapsed_offset"] = float(agent["elapsed_offset"]) + float(agent["last_world_time"])
                agent["completed_offset"] = int(agent["completed_offset"]) + int(agent["last_completed"])
                agent["crash_offset"] = int(agent["crash_offset"]) + int(agent["last_total_crashes"])

            if terminated or truncated:
                agent["elapsed_offset"] = float(agent["elapsed_offset"]) + current_time
                agent["completed_offset"] = int(agent["completed_offset"]) + current_completed
                agent["crash_offset"] = int(agent["crash_offset"]) + current_total_crashes
                agent["episodes"] = int(agent["episodes"]) + 1
                obs_local, _ = agent_env.reset()
                current_time = agent_env.world.time
                current_completed = int(getattr(agent_env.world, "completed_vehicles", 0))
                current_total_crashes = int(getattr(agent_env.world, "total_crashes", 0))

            agent["obs"] = obs_local
            agent["last_world_time"] = current_time
            agent["last_completed"] = current_completed
            agent["last_total_crashes"] = current_total_crashes
            step_accum_local -= step_seconds
            steps_this_frame += 1

        step_accum_local = min(step_accum_local, step_seconds * max_steps_per_frame)
        agent["step_accum"] = step_accum_local

        world_local = agent_env.world
        stopped = [v for v in world_local.vehicles if getattr(v, "wait_time", 0.0) > 0.0]
        instant_avg_wait = sum(v.wait_time for v in stopped) / len(stopped) if stopped else 0.0
        agent["avg_wait_sum"] = float(agent["avg_wait_sum"]) + instant_avg_wait
        agent["avg_wait_samples"] = int(agent["avg_wait_samples"]) + 1
        avg_wait = float(agent["avg_wait_sum"]) / float(max(1, int(agent["avg_wait_samples"])))
        elapsed_total = float(agent["wall_elapsed"])
        completed = int(agent["completed_offset"]) + int(getattr(world_local, "completed_vehicles", 0))
        total_crashes = int(agent["crash_offset"]) + int(getattr(world_local, "total_crashes", 0))
        throughput_per_min = 0.0
        crashes_per_min = 0.0
        if elapsed_total > 0.0:
            throughput_per_min = (completed * 60.0) / elapsed_total
            crashes_per_min = (total_crashes * 60.0) / elapsed_total
        return {
            "name": str(agent["name"]),
            "elapsed": elapsed_total,
            "avg_wait": avg_wait,
            "throughput": throughput_per_min,
            "completed": completed,
            "crashes_per_min": crashes_per_min,
        }

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
                elif event.key == pygame.K_PERIOD and paused:
                    obs, _, terminated, truncated, _ = env.step(rl_controller.act(obs))
                    if terminated or truncated:
                        obs, _ = env.reset()
                elif event.key in (pygame.K_TAB, pygame.K_d):
                    show_overlays = not show_overlays
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    ui_speed = min(ui_speed * 1.25, 2.0)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    ui_speed = max(ui_speed / 1.25, 0.5)
                elif event.key == pygame.K_r:
                    risk_index = (risk_index + 1) % len(risk_levels)
                    _reset_simulation()
                elif event.key == pygame.K_l and (event.mod & pygame.KMOD_CTRL):
                    loaded = _prompt_load_ppo_model(models_dir, renderer)
                    if loaded[0] is not None:
                        rl_controller, loaded_model_name = loaded
                elif event.key == pygame.K_c and compare_mode:
                    compare_mode = False
                    _close_compare_session()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                sx = _speed_slider_rect(renderer.width, renderer.height)
                rx = _risk_toggle_rect(renderer.width, renderer.height)
                play_btn = _play_button_rect(renderer.width, renderer.height)
                stop_btn = _stop_button_rect(renderer.width, renderer.height)
                load_btn = _load_button_rect(renderer.width, renderer.height)
                train_btn = _train_button_rect(renderer.width, renderer.height)
                compare_btn = _compare_button_rect(renderer.width, renderer.height)
                if sx.collidepoint(event.pos):
                    dragging_speed = True
                    ui_speed = _slider_value(event.pos[0], sx, 0.5, 2.0)
                elif rx.collidepoint(event.pos):
                    risk_index = (risk_index + 1) % len(risk_levels)
                    _reset_simulation()
                elif play_btn.collidepoint(event.pos):
                    paused = False
                elif stop_btn.collidepoint(event.pos):
                    paused = True
                elif load_btn.collidepoint(event.pos):
                    loaded = _prompt_load_ppo_model(models_dir, renderer)
                    if loaded[0] is not None:
                        rl_controller, loaded_model_name = loaded
                elif train_btn.collidepoint(event.pos):
                    with state_lock:
                        if bool(training_state["active"]):
                            continue
                    params = _prompt_training_params()
                    if params is not None:
                        episodes, episode_steps, model_name = params
                        model_path = os.path.join(models_dir, f"{model_name}.zip")
                        with state_lock:
                            training_state.update(
                                {
                                    "active": True,
                                    "progress": 0.0,
                                    "message": f"Training {model_name}.zip (0/{episodes} episodes)",
                                    "result_path": None,
                                    "error": None,
                                }
                            )
                        thread = threading.Thread(
                            target=_train_model_background,
                            args=(episodes, episode_steps, model_path, training_state, state_lock),
                            daemon=True,
                        )
                        thread.start()
                elif compare_btn.collidepoint(event.pos):
                    if compare_mode:
                        compare_mode = False
                        _close_compare_session()
                        continue
                    model_paths = _prompt_compare_ppo_models(models_dir, renderer)
                    if model_paths is None:
                        continue
                    left_path, right_path = model_paths
                    left_state = None
                    right_state = None
                    try:
                        left_state = _init_compare_agent_state(left_path)
                        right_state = _init_compare_agent_state(right_path)
                    except Exception as exc:
                        if isinstance(left_state, dict):
                            left_env = left_state.get("env")
                            if left_env is not None and hasattr(left_env, "close"):
                                left_env.close()
                        if isinstance(right_state, dict):
                            right_env = right_state.get("env")
                            if right_env is not None and hasattr(right_env, "close"):
                                right_env.close()
                        _show_dialog("Compare Models Failed", f"Could not start comparison:\n{exc}")
                        continue
                    compare_session = {
                        "left": left_state,
                        "right": right_state,
                    }
                    compare_mode = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging_speed = False
            elif event.type == pygame.MOUSEMOTION and dragging_speed:
                ui_speed = _slider_value(
                    event.pos[0], _speed_slider_rect(renderer.width, renderer.height), 0.5, 2.0
                )

        with state_lock:
            training_active = bool(training_state["active"])
            train_progress = float(training_state["progress"])
            train_message = str(training_state["message"])
            train_result = training_state["result_path"]
            train_error = training_state["error"]

        if training_active:
            _draw_branding_splash(
                renderer,
                hint=train_message,
                progress=train_progress,
                show_spinner=True,
            )
            continue

        if train_result is not None:
            try:
                rl_controller = RLController(algo="PPO", model_path=str(train_result))
                loaded_model_name = os.path.basename(str(train_result))
                _show_dialog("Training Complete", f"Model trained and loaded:\n{train_result}")
            except Exception as exc:
                _show_dialog("Training Complete", f"Model trained but failed to load:\n{exc}")
            with state_lock:
                training_state["result_path"] = None
        if train_error is not None:
            _show_dialog("Training Failed", str(train_error))
            with state_lock:
                training_state["error"] = None

        sim_clock.speed = ui_speed
        if compare_mode and compare_session is not None:
            left_state = compare_session.get("left")
            right_state = compare_session.get("right")
            if not isinstance(left_state, dict) or not isinstance(right_state, dict):
                compare_mode = False
                _close_compare_session()
                continue
            left_stats = _step_compare_agent(left_state, real_dt)
            right_stats = _step_compare_agent(right_state, real_dt)
            renderer.render_model_comparison(
                left_stats,
                right_stats,
                title="PPO Model Comparison",
            )
            continue

        risk_value = [0.0, 0.5, 1.0][risk_index]
        env.world.risk_factor = risk_value
        env.world.uniform_speed_enabled = True
        env.world.max_vehicles = 8 + 4 * risk_index
        world = env.world

        if not paused:
            step_accum += real_dt * sim_clock.speed
            steps_this_frame = 0
            while step_accum >= step_seconds and steps_this_frame < max_steps_per_frame:
                obs, _, terminated, truncated, _ = env.step(rl_controller.act(obs))
                current_time = env.world.time
                current_completed = int(getattr(env.world, "completed_vehicles", 0))
                current_total_crashes = int(getattr(env.world, "total_crashes", 0))
                if current_time < last_world_time:
                    elapsed_offset += last_world_time
                    completed_offset += last_completed
                    crash_offset += last_total_crashes
                if terminated or truncated:
                    elapsed_offset += current_time
                    completed_offset += current_completed
                    crash_offset += current_total_crashes
                    obs, _ = env.reset()
                    current_time = env.world.time
                    current_completed = int(getattr(env.world, "completed_vehicles", 0))
                    current_total_crashes = int(getattr(env.world, "total_crashes", 0))
                last_world_time = current_time
                last_completed = current_completed
                last_total_crashes = current_total_crashes
                step_accum -= step_seconds
                steps_this_frame += 1
            # Drop excessive backlog so render responsiveness stays smooth.
            step_accum = min(step_accum, step_seconds * max_steps_per_frame)

        stopped = [v for v in world.vehicles if getattr(v, "wait_time", 0.0) > 0.0]
        avg_wait = sum(v.wait_time for v in stopped) / len(stopped) if stopped else 0.0
        elapsed_total = elapsed_offset + world.time
        completed = completed_offset + int(getattr(world, "completed_vehicles", 0))
        total_crashes = crash_offset + int(getattr(world, "total_crashes", 0))
        throughput_per_min = 0.0
        crashes_per_min = 0.0
        if elapsed_total > 0.0:
            throughput_per_min = (completed * 60.0) / elapsed_total
            crashes_per_min = (total_crashes * 60.0) / elapsed_total
        renderer.render(
            world,
            sim_speed=sim_clock.speed,
            show_overlays=show_overlays,
            controls={
                "speed": ui_speed,
                "risk_label": risk_levels[risk_index],
                "phase": f"PPO | {loaded_model_name}",
            },
            stats={
                "avg_wait": avg_wait,
                "elapsed": elapsed_total,
                "throughput": throughput_per_min,
                "completed": completed,
                "crashes_per_min": crashes_per_min,
            },
        )

    pygame.quit()
    _close_compare_session()
    env.close()


def _prompt_load_ppo_model(models_dir: str, renderer) -> tuple[RLController | None, str]:
    while True:
        _draw_branding_splash(renderer)
        path = _choose_model_path(
            models_dir,
            save=False,
            renderer=renderer,
            hint="Select a PPO model to load.",
        )
        if not path:
            return None, "None"
        try:
            controller = RLController(algo="PPO", model_path=path)
            _show_dialog("Load Model", f"Loaded PPO model:\n{path}")
            return controller, os.path.basename(path)
        except Exception as exc:
            _show_dialog("Load Model Failed", f"Failed to load PPO model:\n{exc}")


def _draw_branding_splash(
    renderer,
    hint: str = "Select a PPO model to start rendering.",
    progress: float | None = None,
    show_spinner: bool = False,
) -> None:
    import pygame

    renderer._draw_static_scene()
    branding_path = os.path.join(os.getcwd(), "assets", "branding", "traffic_sim.png")
    image = None
    if os.path.isfile(branding_path):
        try:
            image = pygame.image.load(branding_path).convert_alpha()
        except pygame.error:
            image = None

    if image is not None:
        max_w = int(renderer.width * 0.6)
        max_h = int(renderer.height * 0.6)
        iw, ih = image.get_size()
        scale = min(max_w / max(1, iw), max_h / max(1, ih), 1.0)
        target = pygame.transform.smoothscale(
            image, (max(1, int(iw * scale)), max(1, int(ih * scale)))
        )
        rect = target.get_rect(center=(renderer.width // 2, renderer.height // 2 - 20))
        renderer.screen.blit(target, rect)

    text_color = (235, 235, 235)
    hint_surface = renderer.font.render(hint, True, text_color)
    hint_rect = hint_surface.get_rect(center=(renderer.width // 2, renderer.height - 64))
    renderer.screen.blit(hint_surface, hint_rect)
    if progress is not None:
        bar_w = min(420, renderer.width - 120)
        bar_h = 14
        bx = (renderer.width - bar_w) // 2
        by = renderer.height - 42
        pygame.draw.rect(renderer.screen, (60, 60, 70), pygame.Rect(bx, by, bar_w, bar_h), border_radius=6)
        fill_w = int(max(0.0, min(1.0, progress)) * bar_w)
        pygame.draw.rect(renderer.screen, (70, 210, 120), pygame.Rect(bx, by, fill_w, bar_h), border_radius=6)
    if show_spinner:
        cx = renderer.width // 2
        cy = renderer.height - 96
        phase = (pygame.time.get_ticks() // 100) % 12
        for i in range(12):
            alpha = 60 + ((i - phase) % 12) * 16
            color = (235, 235, 235, max(0, min(255, alpha)))
            spinner = pygame.Surface((6, 18), pygame.SRCALPHA)
            spinner.fill(color)
            rot = pygame.transform.rotate(spinner, i * 30)
            rect = rot.get_rect(center=(cx, cy))
            renderer.screen.blit(rot, rect)
    pygame.display.flip()


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


def _load_button_rect(width: int, height: int):
    import pygame

    x = width - 280 - 12
    y = 12
    return pygame.Rect(x + 10, y + 136, 260, 28)


def _train_button_rect(width: int, height: int):
    import pygame

    x = width - 280 - 12
    y = 12
    return pygame.Rect(x + 10, y + 170, 260, 28)


def _compare_button_rect(width: int, height: int):
    import pygame

    x = width - 280 - 12
    y = 12
    return pygame.Rect(x + 10, y + 204, 260, 28)


def _load_default_or_prompt_model(
    models_dir: str, default_model_path: str, renderer
) -> tuple[RLController, str] | None:
    if os.path.isfile(default_model_path):
        try:
            return RLController(algo="PPO", model_path=default_model_path), os.path.basename(
                default_model_path
            )
        except Exception:
            _show_dialog(
                "Default Model Failed",
                f"Failed to load default base model:\n{default_model_path}\nChoose another model.",
            )

    loaded = _prompt_load_ppo_model(models_dir, renderer)
    if loaded[0] is not None:
        return loaded[0], loaded[1]
    return None


def _prompt_compare_ppo_models(models_dir: str, renderer) -> tuple[str, str] | None:
    first = _choose_model_path(
        models_dir,
        save=False,
        renderer=renderer,
        hint="Choose first PPO model for comparison.",
    )
    if not first:
        return None
    second = _choose_model_path(
        models_dir,
        save=False,
        renderer=renderer,
        hint="Choose second PPO model for comparison.",
    )
    if not second:
        return None
    return first, second


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


def _choose_model_path(
    models_dir: str,
    save: bool,
    renderer=None,
    hint: str | None = None,
) -> str | None:
    """Open a file dialog rooted in the models directory."""
    if renderer is not None:
        _draw_branding_splash(
            renderer,
            hint=hint or ("Choose a model save path." if save else "Choose a PPO model."),
        )
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
            title="Load PPO Model",
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


def _prompt_training_params() -> tuple[int, int, str] | None:
    try:
        import tkinter as tk
        from tkinter import simpledialog
    except Exception:
        return None
    root = tk.Tk()
    root.withdraw()
    episodes = simpledialog.askinteger(
        "Train Model",
        "How many episodes?",
        minvalue=1,
        initialvalue=20,
    )
    if episodes is None:
        root.destroy()
        return None
    episode_steps = simpledialog.askinteger(
        "Train Model",
        "How many steps per episode?",
        minvalue=100,
        initialvalue=2000,
    )
    if episode_steps is None:
        root.destroy()
        return None
    model_name = simpledialog.askstring(
        "Train Model",
        "Model name (without extension):",
        initialvalue="traffic_ppo_base",
    )
    root.destroy()
    if not model_name:
        return None
    safe_name = "".join(ch for ch in model_name if ch.isalnum() or ch in ("_", "-", ".")).strip(". ")
    if not safe_name:
        return None
    return episodes, episode_steps, safe_name


def _train_model_background(
    episodes: int,
    episode_steps: int,
    model_path: str,
    state: dict[str, object],
    state_lock: threading.Lock,
) -> None:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv

        episode_length = max(100, int(episode_steps))
        env = DummyVecEnv(
            [
                lambda: TrafficEnv(
                    render_enabled=False,
                    max_steps=episode_length,
                    action_repeat=1,
                    use_curriculum=True,
                    curriculum_episodes=300,
                    reset_on_crash=False,
                    crash_pause_duration=0.0,
                )
            ]
        )

        class EpisodeProgressCallback(BaseCallback):
            def __init__(self, total_episodes: int, steps_per_episode: int, done_episodes: int) -> None:
                super().__init__()
                self.total_episodes = max(1, total_episodes)
                self.steps_per_episode = max(1, steps_per_episode)
                self.done_episodes = done_episodes

            def _on_step(self) -> bool:
                within_episode = min(
                    1.0, float(self.num_timesteps) / float(self.steps_per_episode)
                )
                overall = (self.done_episodes + within_episode) / float(self.total_episodes)
                with state_lock:
                    state["progress"] = max(0.0, min(1.0, overall))
                    state["message"] = (
                        f"Training {os.path.basename(model_path)} "
                        f"({self.done_episodes + within_episode:.1f}/{self.total_episodes} episodes)"
                    )
                return True

        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=0,
            n_steps=128,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            learning_rate=3e-4,
            device="cpu",
        )
        for ep in range(episodes):
            model.learn(
                total_timesteps=episode_length,
                reset_num_timesteps=False,
                progress_bar=False,
                callback=EpisodeProgressCallback(episodes, episode_length, ep),
            )
            with state_lock:
                state["progress"] = float(ep + 1) / float(max(1, episodes))
                state["message"] = (
                    f"Training {os.path.basename(model_path)} ({ep + 1}/{episodes} episodes)"
                )
        model.save(model_path)
        with state_lock:
            state["progress"] = 1.0
            state["result_path"] = model_path
    except Exception as exc:
        with state_lock:
            state["error"] = str(exc)
    finally:
        with state_lock:
            state["active"] = False


if __name__ == "__main__":
    main()
