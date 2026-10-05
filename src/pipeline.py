"""End-to-end collection, training, evaluation, and report writing."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.bc import predict_actions, train_bc
from src.config import (
    ALL_POLICIES,
    BASELINE_POLICIES,
    LEARNED_POLICIES,
    PROJECT_SLUG,
    SCENARIOS,
    Settings,
    make_settings,
)
from src.dataset import collect_expert
from src.envs import make_env
from src.evaluate import evaluate_seeds, random_action_fn, render_collision_gif, run_episode
from src.expert import expert_action
from src.failure import pick_diverse_cases
from src.metrics import (
    aggregate_episodes,
    select_learned_policy,
    speed_pct_of_expert,
    steps_fraction_to_match,
)
from src.networks import build_policy
from src.plots import plot_grouped_bars, plot_learning_curves
from src.ppo import load_ppo, train_ppo
from src.report import (
    build_success_criteria,
    update_readme,
    write_failure_report,
    write_json,
    write_resume,
)
from src.runtime import PeakRSS, hardware_info, limit_threads
from src.splits import (
    assert_split_integrity,
    assign_splits,
    load_splits,
    save_splits,
    split_counts,
    transition_split_ids,
)

ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available. Using CPU.", flush=True)
        return "cpu"
    return requested


def _bc_seed(settings: Settings, scenario: str, arch: str) -> int:
    return settings.master_seed + (0 if scenario == "highway" else 17) + (0 if arch == "mlp" else 3)


def _train_bc_model(scenario: str, arch: str, data: dict, mapping: dict[int, str], settings: Settings, path: Path):
    fingerprint = settings.train_fingerprint() + f":bc:{scenario}:{arch}"
    if path.exists():
        blob = torch.load(path, map_location=settings.device, weights_only=False)
        if blob.get("fingerprint") == fingerprint:
            model = build_policy(arch, settings)
            model.load_state_dict(blob["state_dict"])
            model.to(settings.device)
            print(f"[{scenario}] loaded BC {arch}", flush=True)
            return model, blob["info"]
    train_idx = transition_split_ids(data["episode_ids"], mapping, "train")
    val_idx = transition_split_ids(data["episode_ids"], mapping, "val")
    test_idx = transition_split_ids(data["episode_ids"], mapping, "test")
    if set(train_idx.tolist()) & set(test_idx.tolist()):
        raise RuntimeError("training rows overlap the test split")
    torch.manual_seed(_bc_seed(settings, scenario, arch))
    model = build_policy(arch, settings)
    info = train_bc(
        model,
        data["observations"],
        data["actions"],
        train_idx,
        val_idx,
        epochs=settings.bc_epochs,
        batch_size=min(settings.bc_batch, max(len(train_idx), 1)),
        lr=settings.bc_lr,
        patience=settings.bc_patience,
        device=settings.device,
        seed=_bc_seed(settings, scenario, arch),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"fingerprint": fingerprint, "state_dict": model.state_dict(), "info": info, "arch": arch},
        path,
    )
    print(
        f"[{scenario}] BC {arch} val accuracy {info['val_accuracy']:.3f} epochs {info['epochs_ran']}",
        flush=True,
    )
    return model, info


def _action_functions(settings: Settings, bc_models: dict, ppo_models: dict, scenario: str) -> dict:
    def expert(enable_lane_change: bool):
        def fn(env, _obs, flag=enable_lane_change):
            return expert_action(env, enable_lane_change=flag)

        return fn

    fns = {
        "idm_mobil": expert(True),
        "idm_lane_keep": expert(False),
    }
    for arch in ("mlp", "attention"):
        fns[f"bc_{arch}"] = _bc_action(bc_models[(scenario, arch)], settings.device)
        for init in ("scratch", "bc"):
            fns[f"ppo_{arch}_{init}"] = _ppo_action(ppo_models[(scenario, arch, init)])
    return fns


def _bc_action(model, device: str):
    def fn(_env, obs):
        return model.act(obs, device)

    return fn


def _ppo_action(model):
    model.policy.set_training_mode(False)

    def fn(_env, obs):
        action, _state = model.predict(obs, deterministic=True)
        return int(np.asarray(action).reshape(-1)[0])

    return fn


def _evaluate(scenario: str, split: str, seeds: list[int], fns: dict, settings: Settings) -> list[dict]:
    env = make_env(scenario, settings, render=False)
    rows: list[dict] = []
    try:
        for policy in ALL_POLICIES:
            print(f"[{scenario}] eval {split} {policy} on {len(seeds)} seeds", flush=True)
            episodes = evaluate_seeds(env, seeds, fns[policy], settings)
            for episode in episodes:
                episode = dict(episode)
                episode["scenario"] = scenario
                episode["env"] = scenario
                episode["policy"] = policy
                episode["split"] = split
                episode["episode_id"] = f"{scenario}:{split}:{policy}:{episode['seed']}"
                rows.append(episode)
    finally:
        env.close()
    return rows


def _summary_rows(episodes: list[dict], scenario: str, split: str) -> list[dict]:
    rows = []
    for policy in ALL_POLICIES:
        subset = [ep for ep in episodes if ep["scenario"] == scenario and ep["policy"] == policy and ep["split"] == split]
        stats = aggregate_episodes(subset)
        rows.append({"scenario": scenario, "policy": policy, "split": split, **stats})
    return rows


def _step_pct(curve_rows: list[dict], scenario: str, arch: str, total_steps: int) -> float:
    scratch = [
        row
        for row in curve_rows
        if row["scenario"] == scenario and row["architecture"] == arch and row["init"] == "scratch"
    ]
    cloned = [
        row
        for row in curve_rows
        if row["scenario"] == scenario and row["architecture"] == arch and row["init"] == "bc"
    ]
    scratch = sorted(scratch, key=lambda row: row["timestep"])
    cloned = sorted(cloned, key=lambda row: row["timestep"])
    if not scratch or not cloned:
        return 100.0
    fraction = steps_fraction_to_match(
        [row["timestep"] for row in cloned],
        [row["mean_return"] for row in cloned],
        float(scratch[-1]["mean_return"]),
        total_steps,
    )
    return float(100.0 * fraction)


def _measure_coverage(root: Path, output_json: Path) -> float | None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "--cov=src",
        f"--cov-report=json:{output_json}",
        "--cov-report=term-missing",
        "-q",
        "--override-ini",
        "addopts=",
    ]
    print("measuring test coverage", flush=True)
    completed = subprocess.run(command, cwd=root, check=False)
    if not output_json.exists():
        print(f"coverage json missing (pytest exit {completed.returncode})", flush=True)
        return None
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    return float(payload["totals"]["percent_covered"])


def run_pipeline(
    *,
    quick: bool,
    device: str,
    output_dir: Path,
    data_dir: Path,
    artifacts_dir: Path,
    measure_coverage: bool = False,
    readme_path: Path | None = None,
    resume_path: Path | None = None,
) -> dict:
    """Run the full project pipeline and write ``output_dir``."""
    limit_threads(4)
    device = _resolve_device(device)
    settings = make_settings(quick, device)
    torch.manual_seed(settings.master_seed)
    np.random.seed(settings.master_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    peak = PeakRSS().start()
    started = time.perf_counter()
    try:
        datasets = {}
        for scenario in SCENARIOS:
            datasets[scenario] = collect_expert(
                scenario,
                settings,
                data_dir / f"{scenario}.npz",
                data_dir / f"{scenario}.manifest.json",
            )
        split_path = data_dir / "splits.json"
        splits = load_splits(split_path, settings.data_fingerprint())
        if splits is None:
            splits = {}
            for scenario in SCENARIOS:
                splits[scenario] = assign_splits(
                    datasets[scenario]["episode_ids"],
                    seed=settings.master_seed,
                )
            save_splits(split_path, splits, settings.data_fingerprint())
        for scenario in SCENARIOS:
            assert_split_integrity(datasets[scenario]["episode_ids"], splits[scenario])
            overlap = set(datasets[scenario]["seeds"].tolist()) & settings.reserved_seeds(scenario)
            if overlap:
                raise RuntimeError(f"{scenario} expert seeds overlap reserved seeds: {sorted(overlap)[:5]}")

        bc_models = {}
        bc_info = {}
        predictions: list[dict] = []
        for scenario in SCENARIOS:
            for arch in ("mlp", "attention"):
                model, info = _train_bc_model(
                    scenario,
                    arch,
                    datasets[scenario],
                    splits[scenario],
                    settings,
                    artifacts_dir / f"bc_{scenario}_{arch}.pt",
                )
                bc_models[(scenario, arch)] = model
                bc_info[f"{scenario}_{arch}"] = {
                    "epochs_ran": info["epochs_ran"],
                    "train_accuracy": info["train_accuracy"],
                    "val_accuracy": info["val_accuracy"],
                    "best_val_loss": info["best_val_loss"],
                }
                test_idx = transition_split_ids(datasets[scenario]["episode_ids"], splits[scenario], "test")
                pred = predict_actions(
                    model,
                    datasets[scenario]["observations"][test_idx],
                    settings.device,
                )
                truth = datasets[scenario]["actions"][test_idx]
                for row_index, label, guess in zip(test_idx.tolist(), truth.tolist(), pred.tolist()):
                    predictions.append(
                        {
                            "transition_id": int(row_index),
                            "scenario": scenario,
                            "architecture": arch,
                            "action": int(label),
                            "prediction": int(guess),
                        }
                    )

        curve_rows: list[dict] = []
        ppo_models = {}
        for scenario in SCENARIOS:
            for arch in ("mlp", "attention"):
                for init in ("scratch", "bc"):
                    checkpoint = artifacts_dir / f"ppo_{scenario}_{arch}_{init}.zip"
                    curve = train_ppo(
                        scenario,
                        arch,
                        init,
                        settings,
                        bc_models[(scenario, arch)] if init == "bc" else None,
                        settings.seed_block(scenario, "curve"),
                        checkpoint,
                        checkpoint.with_suffix(".curve.json"),
                    )
                    for point in curve:
                        curve_rows.append(
                            {
                                "scenario": scenario,
                                "architecture": arch,
                                "init": init,
                                "timestep": int(point["timestep"]),
                                "steps": int(point["timestep"]),
                                "mean_return": float(point["mean_return"]),
                                "return": float(point["mean_return"]),
                                "n_seeds": int(point["n_seeds"]),
                            }
                        )
                    seed = settings.master_seed + (0 if scenario == "highway" else 100) + (0 if arch == "mlp" else 10)
                    ppo_models[(scenario, arch, init)] = load_ppo(scenario, arch, settings, checkpoint, seed)

        episodes: list[dict] = []
        val_summaries: list[dict] = []
        selected: dict[str, str] = {}
        for scenario in SCENARIOS:
            fns = _action_functions(settings, bc_models, ppo_models, scenario)
            val_eps = _evaluate(scenario, "val", settings.seed_block(scenario, "val"), fns, settings)
            episodes.extend(val_eps)
            summaries = _summary_rows(val_eps, scenario, "val")
            val_summaries.extend(summaries)
            expert_val = next(row for row in summaries if row["policy"] == "idm_mobil")
            learned = [row for row in summaries if row["policy"] in LEARNED_POLICIES]
            selected[scenario] = select_learned_policy(
                learned,
                expert_val["collision_pct"],
                expert_val["mean_speed"],
            )
            print(f"[{scenario}] selected {selected[scenario]} on validation", flush=True)

        test_summaries: list[dict] = []
        for scenario in SCENARIOS:
            fns = _action_functions(settings, bc_models, ppo_models, scenario)
            test_eps = _evaluate(scenario, "test", settings.seed_block(scenario, "test"), fns, settings)
            episodes.extend(test_eps)
            test_summaries.extend(_summary_rows(test_eps, scenario, "test"))

        gif_notes: list[str] = []
        gif_rows: list[dict] = []
        collisions = [
            {
                "scenario": ep["scenario"],
                "policy": ep["policy"],
                "seed": ep["seed"],
                "failure_mode": ep["failure_mode"],
                "episode_id": ep["episode_id"],
            }
            for ep in episodes
            if ep["split"] == "test" and ep["policy"] in LEARNED_POLICIES and ep["crashed"]
        ]
        chosen = pick_diverse_cases(collisions, settings.gif_count)
        if len(chosen) < settings.gif_count:
            gif_notes.append(
                "Learned policies did not produce enough test collisions, so extra GIFs use a random policy on unused seeds."
            )
            probe_env_seeds = list(range(300_000, 300_000 + 30))
            for scenario in SCENARIOS:
                if len(chosen) >= settings.gif_count:
                    break
                env = make_env(scenario, settings, render=False)
                try:
                    for seed in probe_env_seeds:
                        episode = run_episode(env, seed, random_action_fn, settings)
                        if episode["crashed"]:
                            chosen.append(
                                {
                                    "scenario": scenario,
                                    "policy": "random_probe",
                                    "seed": seed,
                                    "failure_mode": episode["failure_mode"],
                                    "episode_id": f"{scenario}:probe:random_probe:{seed}",
                                }
                            )
                        if len(chosen) >= settings.gif_count:
                            break
                finally:
                    env.close()
        for index, case in enumerate(chosen, start=1):
            fns = _action_functions(settings, bc_models, ppo_models, case["scenario"])
            action_fn = fns.get(case["policy"], random_action_fn)
            gif_path = output_dir / "collisions" / f"collision_{index:02d}.gif"
            frames = render_collision_gif(case["scenario"], settings, int(case["seed"]), action_fn, gif_path)
            gif_rows.append({**case, "gif": str(gif_path.name), "frames": frames})
            print(f"wrote {gif_path.name} ({case['policy']} seed {case['seed']})", flush=True)

        for model in ppo_models.values():
            env = model.get_env()
            if env is not None:
                env.close()

        failure_counts: dict[str, int] = {}
        for ep in episodes:
            if ep["split"] == "test" and ep["policy"] in LEARNED_POLICIES and ep["failure_mode"] != "none":
                failure_counts[ep["failure_mode"]] = failure_counts.get(ep["failure_mode"], 0) + 1
        write_failure_report(output_dir / "failure_modes.md", failure_counts, gif_notes)

        plot_learning_curves(curve_rows, output_dir / "learning_curves.png")
        plot_grouped_bars(
            test_summaries,
            "collision_pct",
            "collision rate (%)",
            output_dir / "collision_rates.png",
            "Held-out collision rate",
        )
        plot_grouped_bars(
            test_summaries,
            "mean_speed",
            "mean speed (m/s)",
            output_dir / "speed_comparison.png",
            "Held-out mean speed",
        )

        def _find(summaries: list[dict], scenario: str, policy: str) -> dict:
            return next(row for row in summaries if row["scenario"] == scenario and row["policy"] == policy)

        highway_best = _find(test_summaries, "highway", selected["highway"])
        highway_expert = _find(test_summaries, "highway", "idm_mobil")
        merge_best = _find(test_summaries, "merge", selected["merge"])
        merge_expert = _find(test_summaries, "merge", "idm_mobil")
        speed_pct = speed_pct_of_expert(highway_best["mean_speed"], highway_expert["mean_speed"])
        margin_ok = highway_best["collision_pct"] <= highway_expert["collision_pct"] + 2.0
        step_pct = _step_pct(curve_rows, "highway", "attention", settings.ppo_timesteps)
        step_detail = {
            f"{scenario}_{arch}": _step_pct(curve_rows, scenario, arch, settings.ppo_timesteps)
            for scenario in SCENARIOS
            for arch in ("mlp", "attention")
        }

        coverage_pct = None
        if measure_coverage and not os.environ.get("PYTEST_CURRENT_TEST"):
            coverage_pct = _measure_coverage(ROOT, output_dir / "coverage.json")

        criteria = build_success_criteria(
            highway_collision_pct=highway_best["collision_pct"],
            speed_pct_of_expert=speed_pct,
            collision_margin_ok=margin_ok,
            step_pct=step_pct,
            merge_collision_pct=merge_best["collision_pct"],
            coverage_pct=coverage_pct,
        )
        test_table = [
            {
                "scenario": row["scenario"],
                "policy": row["policy"],
                "collision_pct": row["collision_pct"],
                "mean_speed": row["mean_speed"],
                "mean_abs_jerk": row["mean_abs_jerk"],
                "mean_return": row["mean_return"],
            }
            for row in test_summaries
        ]
        resume_numbers = {
            "highway_collision_pct": f"{highway_best['collision_pct']:.2f}",
            "step_pct": f"{step_pct:.2f}",
            "merge_collision_pct": f"{merge_best['collision_pct']:.2f}",
            "speed_pct_of_expert": f"{speed_pct:.2f}",
            "n_test_seeds": int(settings.test_count),
        }
        payload = {
            "project": PROJECT_SLUG,
            "success_criteria": criteria,
            "metrics": {
                "selected_policy": selected,
                "test_table": test_table,
                "validation_table": [
                    {
                        "scenario": row["scenario"],
                        "policy": row["policy"],
                        "collision_pct": row["collision_pct"],
                        "mean_speed": row["mean_speed"],
                        "mean_return": row["mean_return"],
                    }
                    for row in val_summaries
                ],
                "sample_efficiency_step_pct": step_detail,
                "highway_attention_step_pct": step_pct,
                "ppo_budget": {
                    "timesteps_per_run": int(settings.ppo_timesteps),
                    "n_envs": int(settings.ppo_n_envs),
                    "n_steps": int(settings.ppo_n_steps),
                    "policy_hz": int(settings.policy_freq),
                    "simulation_hz": int(settings.sim_freq),
                    "final_timesteps": {
                        f"{scenario}/{arch}/{init}": max(
                            (
                                int(row["timestep"])
                                for row in curve_rows
                                if row["scenario"] == scenario
                                and row["architecture"] == arch
                                and row["init"] == init
                            ),
                            default=0,
                        )
                        for scenario in SCENARIOS
                        for arch in ("mlp", "attention")
                        for init in ("scratch", "bc")
                    },
                },
                "speed_criterion": {
                    "speed_pct_of_expert": speed_pct,
                    "learned_collision_pct": highway_best["collision_pct"],
                    "expert_collision_pct": highway_expert["collision_pct"],
                    "collision_margin_points": 2.0,
                    "collision_margin_ok": margin_ok,
                },
                "bc": bc_info,
                "dataset": {scenario: datasets[scenario]["stats"] for scenario in SCENARIOS},
                "resume_numbers": resume_numbers,
                "eval_seeds": {
                    scenario: {
                        "val": settings.seed_block(scenario, "val"),
                        "curve": settings.seed_block(scenario, "curve"),
                        "test": settings.seed_block(scenario, "test"),
                    }
                    for scenario in SCENARIOS
                },
            },
            "baseline": {
                "highway": {
                    "idm_mobil": _public_stats(highway_expert),
                    "idm_lane_keep": _public_stats(_find(test_summaries, "highway", "idm_lane_keep")),
                },
                "merge": {
                    "idm_mobil": _public_stats(merge_expert),
                    "idm_lane_keep": _public_stats(_find(test_summaries, "merge", "idm_lane_keep")),
                },
            },
            "splits": {
                scenario: split_counts(datasets[scenario]["episode_ids"], splits[scenario])
                for scenario in SCENARIOS
            },
            "seeds": [settings.master_seed],
            "hardware": {},
            "runtime_seconds": 0.0,
        }
        # Fill hardware and runtime after coverage so the peak includes that work.
        payload["hardware"] = hardware_info(peak.stop(), settings.device)
        payload["runtime_seconds"] = round(time.perf_counter() - started, 3)
        write_json(output_dir / "metrics.json", payload)
        _write_episode_artifacts(output_dir, episodes, curve_rows, predictions, datasets, splits, gif_rows)
        if readme_path is not None and readme_path.exists():
            update_readme(readme_path, payload)
        if resume_path is not None:
            write_resume(resume_path, payload)
        print("success criteria:", flush=True)
        for row in criteria:
            print(f"  {row['passed']!s:5} {row['name']}: {row['value']:.4f} {row['op']} {row['threshold']}", flush=True)
        return payload
    except Exception:
        peak.stop()
        raise


def _public_stats(row: dict) -> dict:
    return {
        "collision_pct": row["collision_pct"],
        "mean_speed": row["mean_speed"],
        "mean_abs_jerk": row["mean_abs_jerk"],
        "mean_return": row["mean_return"],
        "mean_progress_m": row["mean_progress_m"],
        "n": row["n"],
    }


def _write_episode_artifacts(output_dir, episodes, curve_rows, predictions, datasets, splits, gif_rows) -> None:
    _write_csv(
        output_dir / "episodes.csv",
        episodes,
        [
            "episode_id",
            "scenario",
            "env",
            "policy",
            "seed",
            "split",
            "crashed",
            "collided",
            "offroad",
            "sum_speed",
            "n_steps",
            "mean_speed",
            "mean_abs_jerk",
            "return",
            "progress_m",
            "min_front_gap_m",
            "failure_mode",
        ],
    )
    _write_csv(
        output_dir / "learning_curves.csv",
        curve_rows,
        ["scenario", "architecture", "init", "timestep", "steps", "mean_return", "return", "n_seeds"],
    )
    _write_csv(
        output_dir / "bc_test_predictions.csv",
        predictions,
        ["transition_id", "scenario", "architecture", "action", "prediction"],
    )
    split_rows = []
    for scenario, mapping in splits.items():
        ids = datasets[scenario]["episode_ids"]
        for episode, label in sorted(mapping.items()):
            split_rows.append(
                {
                    "episode_id": int(episode),
                    "scenario": scenario,
                    "split": label,
                    "n_transitions": int(np.sum(ids == episode)),
                }
            )
    _write_csv(
        output_dir / "split_assignments.csv",
        split_rows,
        ["episode_id", "scenario", "split", "n_transitions"],
    )
    _write_csv(
        output_dir / "gif_index.csv",
        gif_rows,
        ["gif", "scenario", "policy", "seed", "failure_mode", "episode_id", "frames"],
    )


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train and evaluate highway driving policies.")
    parser.add_argument("--quick", action="store_true", help="Small subset that finishes in a few minutes.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output", default="results")
    args = parser.parse_args(argv)
    root = ROOT
    os.chdir(root)
    mode = "quick" if args.quick else "full"
    run_pipeline(
        quick=args.quick,
        device=args.device,
        output_dir=root / args.output,
        data_dir=root / "data" / mode,
        artifacts_dir=root / "artifacts" / mode,
        measure_coverage=not args.quick,
        readme_path=root / "README.md",
        resume_path=root / "RESUME_NOTES.md",
    )
