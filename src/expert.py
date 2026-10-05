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

# highway-env's do_every uses a strict greater-than against this delay.
LANE_CHANGE_DELAY = 1.0


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
        # MOBIL is consulted once the delay has elapsed. Passing the raw
        # elapsed time fails that check when the policy step equals the delay,
        # which skipped every lane change on a 1 Hz policy.
        elapsed = float(getattr(ego, "_expert_elapsed", LANE_CHANGE_DELAY))
        ongoing = tuple(ego.lane_index) != tuple(ego.target_lane_index)
        due = (not ongoing) and elapsed + 1e-9 >= LANE_CHANGE_DELAY
        proxy_timer = (LANE_CHANGE_DELAY + 0.01) if due else 0.0
        speed_limit = 30.0
        if ego.lane is not None and ego.lane.speed_limit:
            speed_limit = float(ego.lane.speed_limit)
        desired_cruise = float(min(30.0, speed_limit))
        # route=None so MOBIL maximizes acceleration instead of treating the
        # current lane id as a forced route and rejecting every lane change.
        proxy = IDMVehicle(
            road,
            np.array(ego.position, dtype=np.float64, copy=True),
            heading=float(ego.heading),
            speed=float(ego.speed),
            target_lane_index=ego.target_lane_index,
            target_speed=desired_cruise,
            route=None,
            enable_lane_change=enable_lane_change,
            timer=proxy_timer,
        )
        # A fresh IDMVehicle otherwise draws DELTA at random, so the same
        # traffic state would not have one expert action.
        proxy.DELTA = 4.0
        before = int(proxy.target_lane_index[2])
        if enable_lane_change:
            proxy.change_lane_policy()
        after = int(proxy.target_lane_index[2])
        if ongoing:
            ego._expert_elapsed = 0.0
        elif due:
            ego._expert_elapsed = dt
        else:
            ego._expert_elapsed = elapsed + dt
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
    # MOBIL wins over the longitudinal command. That is the lane change the
    # expert is supposed to make on highway-v0 and on merge-v0. The 5 m/s
    # brake on other steps is what keeps a closing gap from becoming a crash.
    if lane_action is not None and lane_action in available:
        return lane_action
    if long_action in available:
        return long_action
    if IDLE in available:
        return IDLE
    return int(min(available))
