"""Plots written from the evaluation tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curves(curves: list[dict], path: Path) -> None:
    """``curves`` rows have scenario, architecture, init, timestep, mean_return."""
    path.parent.mkdir(parents=True, exist_ok=True)
    scenarios = list(dict.fromkeys(row["scenario"] for row in curves))
    if not scenarios:
        return
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5.2 * len(scenarios), 3.6), squeeze=False)
    for axis, scenario in zip(axes[0], scenarios):
        for arch, init, style in (
            ("attention", "scratch", "-"),
            ("attention", "bc", "--"),
            ("mlp", "scratch", ":"),
            ("mlp", "bc", "-."),
        ):
            rows = [
                row
                for row in curves
                if row["scenario"] == scenario and row["architecture"] == arch and row["init"] == init
            ]
            rows = sorted(rows, key=lambda row: row["timestep"])
            if not rows:
                continue
            axis.plot(
                [row["timestep"] for row in rows],
                [row["mean_return"] for row in rows],
                linestyle=style,
                label=f"{arch} {init}",
            )
        axis.set_title(scenario)
        axis.set_xlabel("environment steps")
        axis.set_ylabel("mean return")
        axis.legend(fontsize=8)
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_grouped_bars(rows: list[dict], value_key: str, ylabel: str, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scenarios = list(dict.fromkeys(row["scenario"] for row in rows))
    policies = list(dict.fromkeys(row["policy"] for row in rows))
    if not scenarios or not policies:
        return
    x = np.arange(len(policies))
    width = 0.35 if len(scenarios) > 1 else 0.6
    fig, axis = plt.subplots(figsize=(10, 4))
    for index, scenario in enumerate(scenarios):
        values = []
        for policy in policies:
            match = [row[value_key] for row in rows if row["scenario"] == scenario and row["policy"] == policy]
            values.append(match[0] if match else 0.0)
        offset = (index - (len(scenarios) - 1) / 2) * width
        axis.bar(x + offset, values, width=width, label=scenario)
    axis.set_xticks(x, policies, rotation=30, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend()
    axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
