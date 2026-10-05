"""Build metrics.json, the README results table, and RESUME_NOTES.md."""

from __future__ import annotations

import json
from pathlib import Path


class _Encoder(json.JSONEncoder):
    def default(self, o):
        import numpy as np

        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def build_success_criteria(
    *,
    highway_collision_pct: float,
    speed_pct_of_expert: float,
    collision_margin_ok: bool,
    step_pct: float,
    merge_collision_pct: float,
    coverage_pct: float | None,
) -> list[dict]:
    """Success criteria from the spec. Rates are percents, speed is percent of the expert."""
    coverage_value = 0.0 if coverage_pct is None else float(coverage_pct)
    return [
        {
            "name": "Collision rate (best learned policy)",
            "value": float(highway_collision_pct),
            "op": "<=",
            "threshold": 8.0,
            "threshold_high": None,
            "passed": bool(highway_collision_pct <= 8.0),
        },
        {
            "name": "Mean speed vs. rule-based expert",
            "value": float(speed_pct_of_expert),
            "op": ">=",
            "threshold": 95.0,
            "threshold_high": None,
            "passed": bool(speed_pct_of_expert >= 95.0 and collision_margin_ok),
        },
        {
            "name": "BC-initialized PPO vs. PPO from scratch",
            "value": float(step_pct),
            "op": "<=",
            "threshold": 60.0,
            "threshold_high": None,
            "passed": bool(step_pct <= 60.0),
        },
        {
            "name": "merge-v0 collision rate (best learned policy)",
            "value": float(merge_collision_pct),
            "op": "<=",
            "threshold": 10.0,
            "threshold_high": None,
            "passed": bool(merge_collision_pct <= 10.0),
        },
        {
            "name": "Test coverage of src/",
            "value": coverage_value,
            "op": ">=",
            "threshold": 65.0,
            "threshold_high": None,
            "passed": bool(coverage_pct is not None and coverage_pct >= 65.0),
        },
    ]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, cls=_Encoder), encoding="utf-8")


def results_markdown(payload: dict) -> str:
    lines = [
        "| Criterion | Value | Threshold | Passed |",
        "|---|---:|---:|---|",
    ]
    for row in payload["success_criteria"]:
        high = row["threshold_high"]
        threshold = row["threshold"] if high is None else f"{row['threshold']} to {high}"
        lines.append(
            f"| {row['name']} | {row['value']:.4f} | {row['op']} {threshold} | {row['passed']} |"
        )
    failed = [row["name"] for row in payload["success_criteria"] if not row["passed"]]
    if failed:
        lines.append("")
        lines.append(
            "These criteria did not pass on the from-scratch run: "
            + "; ".join(failed)
            + ". The values above are the measured ones. Thresholds were not changed, and test seeds were not used to tune."
        )
    lines.append("")
    lines.append("Held-out test metrics (time-weighted mean speed, collision percent):")
    lines.append("")
    lines.append("| Scenario | Policy | Collision % | Mean speed (m/s) | Mean |jerk| | Mean return |")
    lines.append("|---|---|---:|---:|---:|---:|")
    table = payload["metrics"].get("test_table", [])
    for row in table:
        lines.append(
            "| {scenario} | {policy} | {collision_pct:.2f} | {mean_speed:.2f} | {mean_abs_jerk:.3f} | {mean_return:.3f} |".format(
                **row
            )
        )
    budget = payload["metrics"].get("ppo_budget", {})
    steps = budget.get("timesteps_per_run")
    scratch = [row for row in table if str(row["policy"]).endswith("_scratch")]
    selected = payload["metrics"].get("selected_policy", {})
    if steps is not None and scratch:
        rates = ", ".join(
            f"{row['scenario']} {row['policy']} {row['collision_pct']:.1f}%" for row in scratch
        )
        speeds = ", ".join(
            f"{row['scenario']} {row['policy']} {row['mean_speed']:.2f} m/s" for row in scratch
        )
        lines.append("")
        lines.append(
            f"Each PPO run, from scratch and from behavior cloning, trains for {int(steps)} environment steps. "
            f"Held-out collision rates for the from-scratch runs are {rates}. "
            f"Their mean speeds are {speeds}. "
            "The sample-efficiency percentage is the share of that shared budget at which BC-initialized PPO "
            "first matches the scratch run's final learning-curve return. "
            "A low scratch return makes that percentage small, including when a scratch policy avoids "
            "collisions by driving well below the expert speed."
        )
    if selected:
        lines.append("")
        lines.append(
            "The reported policy for each scenario was chosen on validation episodes: "
            + ", ".join(f"{scenario} {policy}" for scenario, policy in selected.items())
            + "."
        )
    return "\n".join(lines) + "\n"


def update_readme(readme_path: Path, payload: dict) -> None:
    start = "<!-- RESULTS_START -->"
    end = "<!-- RESULTS_END -->"
    text = readme_path.read_text(encoding="utf-8")
    block = start + "\n" + results_markdown(payload) + end
    if start not in text or end not in text:
        text = text.rstrip() + "\n\n" + block + "\n"
    else:
        pre, rest = text.split(start, 1)
        _old, post = rest.split(end, 1)
        text = pre + block + post
    readme_path.write_text(text, encoding="utf-8")


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def write_resume(path: Path, payload: dict) -> None:
    numbers = payload["metrics"]["resume_numbers"]
    coll = numbers["highway_collision_pct"]
    step = numbers["step_pct"]
    merge = numbers["merge_collision_pct"]
    speed = numbers["speed_pct_of_expert"]
    n_test = numbers["n_test_seeds"]
    text = f"""# Resume notes

Skills/tools: Python, PyTorch, stable-baselines3, Gymnasium, highway-env, NumPy, pytest, reinforcement learning (PPO), behavior cloning, ego-attention.

This project collects IDM and MOBIL demonstrations in highway-env and trains MLP and ego-attention policies with behavior cloning and PPO. Held-out traffic seeds measure collision rate, speed, and jerk on highway-v0 and merge-v0.

- Trained behavior-cloning and PPO driving policies in highway-env with an ego-attention network over nearby vehicles, reaching a collision rate of {coll}% on {n_test} held-out highway traffic seeds
- Initialized PPO from behavior cloning to match the from-scratch PPO final return in {step}% of the environment steps
- Built an evaluation harness for collision rate, speed, and jerk across highway and merge scenarios, with a failure-mode analysis of remaining collisions
- Matched {speed}% of the rule-based expert's mean speed on highway-v0 while the merge-v0 collision rate of the selected learned policy was {merge}%

GitHub description: Behavior cloning and PPO ego-attention policies for highway and merge driving in highway-env.
"""
    path.write_text(text, encoding="utf-8")


def write_failure_report(path: Path, counts: dict[str, int], notes: list[str]) -> None:
    lines = ["# Failure modes", "", "Counts are crashes on the held-out test seeds for learned policies.", ""]
    if not counts:
        lines.append("No learned-policy collisions were recorded on the test seeds.")
    else:
        lines.append("| Mode | Count |")
        lines.append("|---|---:|")
        for mode, count in sorted(counts.items()):
            lines.append(f"| {mode} | {count} |")
    lines.append("")
    lines.extend(notes)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
