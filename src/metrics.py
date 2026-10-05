"""Metric definitions used by training curves and the held-out evaluation.

Every success-criterion number is computed with these functions so the same
math can be checked on hand-written examples and on the CSV artifacts.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def collision_pct(crashed: Sequence[float] | np.ndarray) -> float:
    """Percent of episodes that crashed. Empty input returns 0."""
    values = np.asarray(list(crashed), dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.mean(values) * 100.0)


def time_weighted_mean_speed(sum_speed: Sequence[float], n_steps: Sequence[float]) -> float:
    """Step-weighted mean speed in m/s. Equivalent to total distance over total time."""
    speed = np.asarray(list(sum_speed), dtype=np.float64)
    steps = np.asarray(list(n_steps), dtype=np.float64)
    denom = float(steps.sum())
    if denom <= 0:
        return 0.0
    return float(speed.sum() / denom)


def mean_abs_jerk(speeds: Sequence[float] | np.ndarray, dt: float) -> float:
    """Mean absolute longitudinal jerk (m/s^3) from a speed trace.

    Acceleration and jerk use first differences with the policy timestep ``dt``.
    Traces shorter than 3 samples have undefined jerk and return 0.
    """
    series = np.asarray(list(speeds), dtype=np.float64)
    if series.size < 3 or dt <= 0:
        return 0.0
    acceleration = np.diff(series) / dt
    jerk = np.diff(acceleration) / dt
    return float(np.mean(np.abs(jerk)))


def speed_pct_of_expert(policy_speed: float, expert_speed: float) -> float:
    if expert_speed <= 0:
        return 0.0
    return float(100.0 * policy_speed / expert_speed)


def steps_fraction_to_match(
    steps: Sequence[float],
    returns: Sequence[float],
    scratch_final: float,
    total_steps: float,
) -> float:
    """Fraction of ``total_steps`` at which ``returns`` first reaches ``scratch_final``.

    Returns 1 when the curve never reaches the target. A point at step 0 counts,
    so a behavior-cloned initialization that is already there scores 0.
    """
    if total_steps <= 0:
        return 1.0
    for step, value in zip(steps, returns):
        if float(value) >= float(scratch_final) - 1e-6:
            return float(step) / float(total_steps)
    return 1.0


def aggregate_episodes(episodes: Iterable[dict]) -> dict[str, float]:
    rows = list(episodes)
    if not rows:
        return {
            "n": 0.0,
            "collision_pct": 0.0,
            "mean_speed": 0.0,
            "mean_abs_jerk": 0.0,
            "mean_return": 0.0,
            "mean_progress_m": 0.0,
        }
    return {
        "n": float(len(rows)),
        "collision_pct": collision_pct([row["crashed"] for row in rows]),
        "mean_speed": time_weighted_mean_speed(
            [row["sum_speed"] for row in rows],
            [row["n_steps"] for row in rows],
        ),
        "mean_abs_jerk": float(np.mean([row["mean_abs_jerk"] for row in rows])),
        "mean_return": float(np.mean([row["return"] for row in rows])),
        "mean_progress_m": float(np.mean([row["progress_m"] for row in rows])),
    }


def select_learned_policy(
    rows: Sequence[dict],
    expert_collision_pct: float,
    expert_speed: float,
) -> str:
    """Pick a learned policy from validation rows only.

    Feasible policies are those within +2 collision points of the validation
    expert and at least 95% of its speed. Ties break toward fewer collisions,
    then higher speed. If none are feasible, the same key still ranks the
    closest policies without looking at test seeds.
    """
    if not rows:
        raise ValueError("no validation rows to select from")

    def sort_key(row: dict) -> tuple:
        collision = float(row["collision_pct"])
        speed = float(row["mean_speed"])
        excess = max(0.0, collision - (expert_collision_pct + 2.0))
        ratio = speed_pct_of_expert(speed, expert_speed) / 100.0
        deficit = max(0.0, 0.95 - ratio)
        feasible = excess == 0.0 and deficit == 0.0
        return (0 if feasible else 1, excess, deficit, collision, -speed, row["policy"])

    return str(min(rows, key=sort_key)["policy"])
