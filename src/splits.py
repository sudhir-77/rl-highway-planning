"""Episode-level dataset splits.

Transitions from one episode stay in one split so adjacent frames cannot leak
from train into val or test. The assignment is a pure function of the episode
ids and a seed, and callers persist the result the first time it is created.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def assign_splits(
    episode_ids: np.ndarray,
    seed: int,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[int, str]:
    """Map each episode id to train, val, or test.

    Episodes are shuffled with ``seed``. When there are at least three
    episodes, val and test are non-empty. Smaller collections keep the
    partitions disjoint and put the remainder in train.
    """
    episodes = np.array(sorted({int(x) for x in np.asarray(episode_ids).tolist()}), dtype=np.int64)
    if episodes.size == 0:
        return {}
    order = episodes.copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    n = int(order.size)
    if n >= 3:
        n_test = max(1, int(round(n * ratios[2])))
        n_val = max(1, int(round(n * ratios[1])))
        if n_test + n_val >= n:
            n_test = 1
            n_val = 1
        n_train = n - n_val - n_test
        labels = (["train"] * n_train) + (["val"] * n_val) + (["test"] * n_test)
    elif n == 2:
        labels = ["train", "val"]
    else:
        labels = ["train"]
    return {int(episode): label for episode, label in zip(order.tolist(), labels)}


def transition_split_ids(
    episode_ids: np.ndarray,
    mapping: dict[int, str],
    split: str,
) -> np.ndarray:
    """Row indexes whose episode belongs to ``split``."""
    mask = np.fromiter((mapping[int(ep)] == split for ep in episode_ids), dtype=bool, count=len(episode_ids))
    return np.flatnonzero(mask).astype(np.int64)


def assert_split_integrity(episode_ids: np.ndarray, mapping: dict[int, str]) -> None:
    """Raise if an episode is missing, duplicated across splits, or mixed inside itself."""
    seen: dict[int, str] = {}
    for episode in np.asarray(episode_ids).tolist():
        episode = int(episode)
        label = mapping[episode]
        if episode in seen and seen[episode] != label:
            raise AssertionError(f"episode {episode} is in two splits")
        seen[episode] = label
    labels = set(mapping.values())
    if not labels <= {"train", "val", "test"}:
        raise AssertionError(f"unexpected split labels: {labels}")
    # Every transition of an episode must share the mapping's label. The mapping
    # itself is one label per episode, so the check is that no episode id is absent.
    missing = {int(ep) for ep in episode_ids.tolist()} - set(mapping)
    if missing:
        raise AssertionError(f"episodes missing from the split map: {sorted(missing)[:5]}")


def split_counts(episode_ids: np.ndarray, mapping: dict[int, str]) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    for episode in np.asarray(episode_ids).tolist():
        counts[mapping[int(episode)]] += 1
    return counts


def save_splits(path: Path, per_scenario: dict[str, dict[int, str]], fingerprint: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fingerprint": fingerprint,
        "scenarios": {
            scenario: {str(episode): label for episode, label in mapping.items()}
            for scenario, mapping in per_scenario.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_splits(path: Path, fingerprint: str) -> dict[str, dict[int, str]] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("fingerprint") != fingerprint:
        return None
    return {
        scenario: {int(episode): label for episode, label in mapping.items()}
        for scenario, mapping in payload["scenarios"].items()
    }
