"""Roll out policies and record per-episode metrics."""

from __future__ import annotations

import numpy as np

from src.config import LANE_LEFT, LANE_RIGHT, Settings
from src.envs import front_gap_m, make_env
from src.failure import classify_failure
from src.metrics import mean_abs_jerk


def run_episode(env, seed: int, action_fn, settings: Settings) -> dict:
    """Run one deterministic episode and return raw fields used in the CSV."""
    obs, _info = env.reset(seed=int(seed))
    speeds: list[float] = []
    rewards: list[float] = []
    actions: list[int] = []
    min_gap = 1.0e9
    crashed = False
    offroad = False
    steps = 0
    done = False
    while not done and steps < settings.max_episode_steps:
        gap = front_gap_m(env)
        if gap < min_gap:
            min_gap = gap
        action = int(action_fn(env, obs))
        obs, reward, terminated, truncated, info = env.step(action)
        vehicle = env.unwrapped.vehicle
        speeds.append(float(info.get("speed", vehicle.speed)))
        rewards.append(float(reward))
        actions.append(action)
        crashed = crashed or bool(vehicle.crashed) or bool(info.get("crashed", False))
        offroad = offroad or (not bool(vehicle.on_road))
        done = bool(terminated or truncated)
        steps += 1
    recent_window = max(settings.policy_freq, 1)
    recent = actions[-recent_window:]
    lane_change_recent = any(action in (LANE_LEFT, LANE_RIGHT) for action in recent)
    vehicle = env.unwrapped.vehicle
    if tuple(vehicle.lane_index) != tuple(vehicle.target_lane_index):
        lane_change_recent = True
    failed = bool(crashed or offroad)
    return {
        "seed": int(seed),
        "crashed": int(failed),
        "offroad": int(offroad),
        "sum_speed": float(sum(speeds)),
        "n_steps": int(len(speeds)),
        "mean_abs_jerk": mean_abs_jerk(speeds, settings.dt),
        "return": float(sum(rewards)),
        "progress_m": float(sum(speeds) * settings.dt),
        "min_front_gap_m": float(min_gap if min_gap < 1.0e8 else -1.0),
        "failure_mode": classify_failure(failed, offroad, lane_change_recent, min_gap),
    }


def evaluate_seeds(env, seeds: list[int], action_fn, settings: Settings) -> list[dict]:
    return [run_episode(env, seed, action_fn, settings) for seed in seeds]


def render_collision_gif(scenario: str, settings: Settings, seed: int, action_fn, path) -> int:
    """Replay one seed with rgb frames and write a GIF. Returns the frame count."""
    from src.gifs import save_gif

    env = make_env(scenario, settings, render=True)
    frames: list[np.ndarray] = []
    try:
        obs, _info = env.reset(seed=int(seed))
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame))
        done = False
        steps = 0
        while not done and steps < settings.max_episode_steps:
            action = int(action_fn(env, obs))
            obs, _reward, terminated, truncated, _info = env.step(action)
            steps += 1
            if steps % 2 == 0:
                frame = env.render()
                if frame is not None:
                    frames.append(np.asarray(frame))
            done = bool(terminated or truncated)
        if not frames:
            raise RuntimeError("renderer returned no frames")
        save_gif(frames, path)
        return len(frames)
    finally:
        env.close()


def random_action_fn(env, _obs) -> int:
    available = [int(action) for action in env.unwrapped.action_type.get_available_actions()]
    if not available:
        return 1
    index = int(env.unwrapped.np_random.integers(len(available)))
    return available[index]


def recent_lane_changes(actions: deque[int]) -> bool:
    return any(action in (LANE_LEFT, LANE_RIGHT) for action in actions)
