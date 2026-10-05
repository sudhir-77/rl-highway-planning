import numpy as np

from src.metrics import (
    aggregate_episodes,
    collision_pct,
    mean_abs_jerk,
    select_learned_policy,
    speed_pct_of_expert,
    steps_fraction_to_match,
    time_weighted_mean_speed,
)


def test_collision_pct_hand_computed():
    assert collision_pct([1, 0, 0, 1]) == 50.0
    assert collision_pct([]) == 0.0


def test_time_weighted_speed_hand_computed():
    # Two episodes: 2 steps at speed 10 (sum 20) and 2 steps at speed 30 (sum 60).
    assert time_weighted_mean_speed([20.0, 60.0], [2.0, 2.0]) == 20.0
    assert time_weighted_mean_speed([0.0], [0.0]) == 0.0


def test_mean_abs_jerk_hand_computed():
    # speeds 0, 1, 3, 6 at dt=1 -> acc 1, 2, 3 -> jerk 1, 1 -> mean abs 1
    assert mean_abs_jerk([0, 1, 3, 6], dt=1.0) == 1.0
    assert mean_abs_jerk([5.0, 5.0], dt=0.2) == 0.0
    # dt=0.5: speeds 0, 1, 3 -> acc 2, 4 -> jerk 4
    assert mean_abs_jerk(np.array([0.0, 1.0, 3.0]), dt=0.5) == 4.0


def test_speed_pct_of_expert():
    assert speed_pct_of_expert(19.0, 20.0) == 95.0
    assert speed_pct_of_expert(10.0, 0.0) == 0.0


def test_steps_fraction_hand_computed():
    steps = [0, 10, 20, 40]
    returns = [1.0, 2.0, 5.0, 5.0]
    assert steps_fraction_to_match(steps, returns, scratch_final=5.0, total_steps=40) == 0.5
    assert steps_fraction_to_match(steps, returns, scratch_final=9.0, total_steps=40) == 1.0
    assert steps_fraction_to_match(steps, returns, scratch_final=1.0, total_steps=40) == 0.0
    assert steps_fraction_to_match(steps, returns, scratch_final=0.0, total_steps=0) == 1.0


def test_aggregate_episodes():
    rows = [
        {"crashed": 1, "sum_speed": 10.0, "n_steps": 2, "mean_abs_jerk": 1.0, "return": 0.0, "progress_m": 2.0},
        {"crashed": 0, "sum_speed": 30.0, "n_steps": 2, "mean_abs_jerk": 3.0, "return": 4.0, "progress_m": 6.0},
    ]
    stats = aggregate_episodes(rows)
    assert stats["n"] == 2
    assert stats["collision_pct"] == 50.0
    assert stats["mean_speed"] == 10.0
    assert stats["mean_abs_jerk"] == 2.0
    assert stats["mean_return"] == 2.0
    assert stats["mean_progress_m"] == 4.0


def test_select_learned_policy_prefers_feasible():
    rows = [
        {"policy": "slow_safe", "collision_pct": 0.0, "mean_speed": 10.0},
        {"policy": "matched", "collision_pct": 1.0, "mean_speed": 20.0},
        {"policy": "fast_crash", "collision_pct": 30.0, "mean_speed": 29.0},
    ]
    # Expert collision 0, speed 20. +2 points allows collision <= 2. 95% of speed is 19.
    assert select_learned_policy(rows, expert_collision_pct=0.0, expert_speed=20.0) == "matched"


def test_select_learned_policy_breaks_ties_by_collision_then_speed():
    rows = [
        {"policy": "a", "collision_pct": 1.0, "mean_speed": 20.0},
        {"policy": "b", "collision_pct": 0.5, "mean_speed": 20.0},
    ]
    assert select_learned_policy(rows, 0.0, 20.0) == "b"
