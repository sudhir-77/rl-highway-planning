"""Collect, cache, and verify expert transitions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.config import Settings
from src.envs import make_env
from src.expert import expert_action


class DatasetError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_dataset(
    path: Path,
    manifest_path: Path,
    *,
    observations: np.ndarray,
    actions: np.ndarray,
    episode_ids: np.ndarray,
    seeds: np.ndarray,
    fingerprint: str,
    stats: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int8),
        episode_ids=np.asarray(episode_ids, dtype=np.int32),
        seeds=np.asarray(seeds, dtype=np.int32),
    )
    manifest = {
        "rows": int(observations.shape[0]),
        "sha256": file_sha256(path),
        "fingerprint": fingerprint,
        "stats": stats,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_dataset(path: Path, manifest_path: Path, fingerprint: str) -> dict:
    """Load a cached dataset and reject a bad checksum, row count, or fingerprint."""
    if not path.exists() or not manifest_path.exists():
        raise DatasetError(f"missing dataset at {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("fingerprint") != fingerprint:
        raise DatasetError("dataset fingerprint does not match this configuration")
    actual = file_sha256(path)
    if actual != manifest["sha256"]:
        raise DatasetError("dataset checksum mismatch")
    with np.load(path) as blob:
        observations = np.asarray(blob["observations"], dtype=np.float32)
        actions = np.asarray(blob["actions"], dtype=np.int8)
        episode_ids = np.asarray(blob["episode_ids"], dtype=np.int32)
        seeds = np.asarray(blob["seeds"], dtype=np.int32)
    if observations.shape[0] != int(manifest["rows"]):
        raise DatasetError("dataset row count does not match the manifest")
    if observations.shape[0] != actions.shape[0]:
        raise DatasetError("observations and actions have different lengths")
    return {
        "observations": observations,
        "actions": actions,
        "episode_ids": episode_ids,
        "seeds": seeds,
        "stats": manifest.get("stats", {}),
        "rows": int(manifest["rows"]),
    }


def _try_load(path: Path, manifest_path: Path, fingerprint: str, rows: int) -> dict | None:
    if not path.exists() or not manifest_path.exists():
        return None
    try:
        data = load_dataset(path, manifest_path, fingerprint)
    except DatasetError:
        return None
    if data["rows"] != rows:
        return None
    return data


def collect_expert(
    scenario: str,
    settings: Settings,
    path: Path,
    manifest_path: Path,
) -> dict:
    """Generate exactly ``settings.expert_transitions`` expert rows, or reload them."""
    fingerprint = settings.data_fingerprint()
    cached = _try_load(path, manifest_path, fingerprint, settings.expert_transitions)
    if cached is not None:
        print(f"[{scenario}] loaded {cached['rows']} cached expert transitions", flush=True)
        return cached

    reserved = settings.reserved_seeds(scenario)
    target = settings.expert_transitions
    n_veh = settings.n_obs_vehicles
    n_feat = settings.n_features
    observations = np.zeros((target, n_veh, n_feat), dtype=np.float32)
    actions = np.zeros((target,), dtype=np.int8)
    episode_ids = np.zeros((target,), dtype=np.int32)
    seeds = np.zeros((target,), dtype=np.int32)
    filled = 0
    episode_index = 0
    crashes = 0
    episodes = 0
    seed = settings.expert_seed_start
    env = make_env(scenario, settings, render=False)
    try:
        while filled < target:
            while seed in reserved:
                seed += 1
            obs, _info = env.reset(seed=int(seed))
            if tuple(obs.shape) != (n_veh, n_feat):
                raise DatasetError(f"unexpected observation shape {obs.shape}")
            done = False
            crashed = False
            steps = 0
            while not done and filled < target:
                action = int(expert_action(env, enable_lane_change=True))
                observations[filled] = obs
                actions[filled] = action
                episode_ids[filled] = episode_index
                seeds[filled] = seed
                filled += 1
                obs, _reward, terminated, truncated, info = env.step(action)
                crashed = crashed or bool(info.get("crashed", False)) or bool(env.unwrapped.vehicle.crashed)
                done = bool(terminated or truncated)
                steps += 1
                if steps > settings.max_episode_steps:
                    break
            crashes += int(crashed)
            episodes += 1
            episode_index += 1
            seed += 1
            if episodes % 25 == 0 or filled >= target:
                print(
                    f"[{scenario}] expert transitions {filled}/{target} episodes {episodes}",
                    flush=True,
                )
    finally:
        env.close()

    if filled != target:
        raise DatasetError(f"collected {filled} rows, expected {target}")
    stats = {
        "episodes": episodes,
        "crashes": crashes,
        "train_seed_collision_pct": float(100.0 * crashes / max(episodes, 1)),
        "action_counts": {str(i): int(np.sum(actions == i)) for i in range(5)},
        "min_seed": int(seeds.min()),
        "max_seed": int(seeds.max()),
    }
    if set(seeds.tolist()) & reserved:
        raise DatasetError("expert data used a reserved evaluation seed")
    save_dataset(
        path,
        manifest_path,
        observations=observations,
        actions=actions,
        episode_ids=episode_ids,
        seeds=seeds,
        fingerprint=fingerprint,
        stats=stats,
    )
    print(f"[{scenario}] wrote {target} expert transitions", flush=True)
    return load_dataset(path, manifest_path, fingerprint)
