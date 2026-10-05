"""highway-env and merge-env construction."""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import gymnasium as gym
import highway_env
import numpy as np

from src.config import ENV_IDS, FASTER, IDLE, SLOWER, SPEED_STEP, Settings

warnings.filterwarnings("ignore", message=".*out of date.*")
gym.register_envs(highway_env)


class SyncSetpoint(gym.Wrapper):
    """Make IDLE hold the measured speed.

    highway-env applies FASTER and SLOWER relative to the current speed, but
    IDLE keeps the previous target. Resetting the setpoint before each action
    removes that hidden state, so the same observation always means the same
    command.
    """

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._sync()
        return obs, info

    def step(self, action):
        self._sync()
        action = int(np.asarray(action).reshape(-1)[0])
        # One meta-action has to be able to brake. A single 1 m/s step cannot.
        if action in (FASTER, SLOWER):
            self._shift_speed(action)
            action = IDLE
        return self.env.step(action)

    def _shift_speed(self, action: int) -> None:
        ego = self.env.unwrapped.vehicle
        delta = SPEED_STEP if action == FASTER else -SPEED_STEP
        limit = int(ego.target_speeds.size) - 1
        ego.speed_index = int(np.clip(int(ego.speed_index) + delta, 0, limit))
        ego.target_speed = float(ego.index_to_speed(ego.speed_index))

    def _sync(self) -> None:
        ego = self.env.unwrapped.vehicle
        low = float(ego.target_speeds[0])
        high = float(ego.target_speeds[-1])
        speed = float(np.clip(ego.speed, low, high))
        ego.speed_index = int(ego.speed_to_index(speed))
        ego.target_speed = float(ego.index_to_speed(ego.speed_index))


class RadarObservation(gym.ObservationWrapper):
    """Append the lead-vehicle gaps that IDM and MOBIL read from the road.

    The kinematic matrix orders vehicles by distance, so the same-lane leader
    is not a fixed row. Columns 5-8 on the ego row are the current-lane gap,
    the range rate, and the forward gaps in the left and right lanes.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        vehicles, features = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(vehicles), int(features) + 4),
            dtype=np.float32,
        )

    def observation(self, observation):
        base = np.asarray(observation, dtype=np.float32)
        extra = np.zeros((base.shape[0], 4), dtype=np.float32)
        gap, rate, left, right = self._leads()
        extra[0, 0] = np.float32(gap / 80.0)
        extra[0, 1] = np.float32(rate / 10.0)
        extra[0, 2] = np.float32(left / 80.0)
        extra[0, 3] = np.float32(right / 80.0)
        return np.concatenate([base, extra], axis=1)

    def _leads(self) -> tuple[float, float, float, float]:
        ego = self.env.unwrapped.vehicle
        gap, rate = 200.0, 0.0
        front = _front_on_lane(ego, ego.lane_index)
        if front is not None:
            gap = _lane_gap(ego, front)
            rate = float(np.dot(ego.velocity - front.velocity, ego.direction))
        left = _side_gap(ego, -1)
        right = _side_gap(ego, 1)
        return gap, rate, left, right


def _front_on_lane(ego, lane_index):
    if lane_index is None or ego.road is None:
        return None
    try:
        front, _rear = ego.road.neighbour_vehicles(ego, lane_index)
    except Exception:
        return None
    return front


def _lane_gap(ego, other) -> float:
    try:
        return float(max(ego.lane_distance_to(other), 0.0))
    except Exception:
        return 200.0


def _side_gap(ego, direction: int) -> float:
    if ego.lane_index is None:
        return 200.0
    side = (ego.lane_index[0], ego.lane_index[1], int(ego.lane_index[2]) + direction)
    try:
        ego.road.network.get_lane(side)
    except Exception:
        return 200.0
    front = _front_on_lane(ego, side)
    if front is None:
        return 200.0
    return _lane_gap(ego, front)


def make_env(scenario: str, settings: Settings, *, render: bool = False) -> gym.Env:
    """Create one scenario env. Caller owns ``reset`` and ``close``."""
    if scenario not in ENV_IDS:
        raise KeyError(scenario)
    env = gym.make(
        ENV_IDS[scenario],
        config=settings.env_config(),
        render_mode="rgb_array" if render else None,
    )
    return RadarObservation(SyncSetpoint(env))


def front_gap_m(env: gym.Env) -> float:
    """Distance to the lead vehicle on the current lane, or a large number if none."""
    ego = env.unwrapped.vehicle
    if ego.lane is None:
        return 1.0e6
    try:
        front, _rear = ego.road.neighbour_vehicles(ego)
    except Exception:
        return 1.0e6
    if front is None:
        return 1.0e6
    try:
        return float(ego.lane_distance_to(front))
    except Exception:
        return 1.0e6
