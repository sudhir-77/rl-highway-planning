"""Discretized IDM + MOBIL expert and the lane-keeping IDM baseline.

highway-env's IDMVehicle already implements both models. The ego vehicle in
this project is an MDPVehicle, so the expert builds a temporary IDM proxy at
the same state, reads its lane-change decision and acceleration, and maps
those onto the five meta-actions. The proxy is not left on the road, and the
real ego is removed during the query so it is not treated as its own neighbor.
"""

from __future__ import annotations

import numpy as np

from src.config import ACCEL_ACCEL, BRAKE_ACCEL, FASTER, IDLE, LANE_LEFT, LANE_RIGHT, SLOWER


def expert_action(env, *, enable_lane_change: bool, brake_accel: float | None = None, accel_accel: float | None = None) -> int:
    """Return a legal discrete meta-action for the current ego state."""
    from highway_env.vehicle.behavior import IDMVehicle

    ego = env.unwrapped.vehicle
    road = ego.road
    dt = 1.0 / float(env.unwrapped.config["policy_frequency"])
    original = list(road.vehicles)
    road.vehicles[:] = [vehicle for vehicle in original if vehicle is not ego]
    lane_action: int | None = None
    acc = 0.0
    try:
        timer = float(getattr(ego, "_expert_timer", 1.01))
        speed_limit = 30.0
        if ego.lane is not None and ego.lane.speed_limit:
            speed_limit = float(ego.lane.speed_limit)
        desired_cruise = float(min(30.0, speed_limit))
        proxy = IDMVehicle(
            road,
            np.array(ego.position, dtype=np.float64, copy=True),
            heading=float(ego.heading),
            speed=float(ego.speed),
            target_lane_index=ego.target_lane_index,
            target_speed=desired_cruise,
            route=getattr(ego, "route", None),
            enable_lane_change=enable_lane_change,
            timer=timer,
        )
        # A fresh IDMVehicle otherwise draws DELTA at random, so the same
        # traffic state would not have one expert action.
        proxy.DELTA = 4.0
        before = int(proxy.target_lane_index[2])
        if enable_lane_change:
            proxy.change_lane_policy()
        after = int(proxy.target_lane_index[2])
        ego._expert_timer = float(proxy.timer) + dt
        front, _rear = road.neighbour_vehicles(proxy, proxy.lane_index)
        acc = float(proxy.acceleration(ego_vehicle=proxy, front_vehicle=front))
        if proxy.lane_index != proxy.target_lane_index:
            front_target, _rear_target = road.neighbour_vehicles(proxy, proxy.target_lane_index)
            acc = min(acc, float(proxy.acceleration(ego_vehicle=proxy, front_vehicle=front_target)))
        if after < before:
            lane_action = LANE_LEFT
        elif after > before:
            lane_action = LANE_RIGHT
    finally:
        road.vehicles[:] = original

    # A deadband on IDM acceleration is a stable function of the lead vehicle.
    # Quantizing onto a speed index flickers between IDLE and SLOWER.
    brake_level = float(BRAKE_ACCEL if brake_accel is None else brake_accel)
    accel_level = float(ACCEL_ACCEL if accel_accel is None else accel_accel)
    if acc <= -brake_level:
        long_action = SLOWER
    elif acc >= accel_level:
        long_action = FASTER
    else:
        long_action = IDLE

    available = {int(action) for action in env.unwrapped.action_type.get_available_actions()}
    # MOBIL lane changes have to win over a mild brake. On merge-v0 the ramp
    # ends if the ego never moves into the main lanes.
    if lane_action is not None and lane_action in available:
        return lane_action
    if long_action in available:
        return long_action
    if IDLE in available:
        return IDLE
    return int(min(available))
