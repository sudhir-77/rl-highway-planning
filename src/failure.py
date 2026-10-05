"""Classify how an episode failed, using only signals recorded during the rollout."""

from __future__ import annotations


def classify_failure(
    crashed: bool,
    offroad: bool,
    lane_change_recent: bool,
    min_front_gap_m: float,
) -> str:
    """Return a short failure tag. Successful episodes are ``none``."""
    if not crashed and not offroad:
        return "none"
    if lane_change_recent:
        return "lane_change"
    if min_front_gap_m < 25.0:
        return "rear_end"
    if offroad:
        return "offroad"
    return "other"


def pick_diverse_cases(cases: list[dict], limit: int) -> list[dict]:
    """Prefer different failure modes, then earlier seeds, up to ``limit`` cases."""
    chosen: list[dict] = []
    used_modes: set[str] = set()
    ordered = sorted(cases, key=lambda row: (str(row["failure_mode"]), int(row["seed"]), str(row["policy"])))
    for row in ordered:
        mode = str(row["failure_mode"])
        if mode in used_modes:
            continue
        chosen.append(row)
        used_modes.add(mode)
        if len(chosen) >= limit:
            return chosen
    for row in ordered:
        if row in chosen:
            continue
        chosen.append(row)
        if len(chosen) >= limit:
            break
    return chosen
