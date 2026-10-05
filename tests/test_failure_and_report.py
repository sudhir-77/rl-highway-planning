import numpy as np
from PIL import Image

from src.failure import classify_failure, pick_diverse_cases
from src.gifs import save_gif
from src.report import build_success_criteria, results_markdown, write_resume


def test_classify_failure_hand_cases():
    assert classify_failure(False, False, False, 5.0) == "none"
    assert classify_failure(True, False, True, 4.0) == "lane_change"
    assert classify_failure(True, False, False, 10.0) == "rear_end"
    assert classify_failure(False, True, False, 100.0) == "offroad"
    assert classify_failure(True, False, False, 80.0) == "other"


def test_pick_diverse_cases_prefers_new_modes():
    cases = [
        {"failure_mode": "rear_end", "seed": 2, "policy": "a"},
        {"failure_mode": "rear_end", "seed": 1, "policy": "b"},
        {"failure_mode": "lane_change", "seed": 3, "policy": "a"},
        {"failure_mode": "offroad", "seed": 4, "policy": "a"},
    ]
    chosen = pick_diverse_cases(cases, limit=2)
    assert [row["failure_mode"] for row in chosen] == ["lane_change", "offroad"]


def test_save_gif_roundtrip(tmp_path):
    frames = [np.zeros((8, 8, 3), dtype=np.uint8), np.full((8, 8, 3), 255, dtype=np.uint8)]
    path = tmp_path / "clip.gif"
    save_gif(frames, path, duration_ms=40)
    with Image.open(path) as image:
        assert image.n_frames == 2


def test_success_criteria_pass_and_fail():
    passed = build_success_criteria(
        highway_collision_pct=7.0,
        speed_pct_of_expert=96.0,
        collision_margin_ok=True,
        step_pct=40.0,
        merge_collision_pct=9.0,
        coverage_pct=70.0,
    )
    assert all(row["passed"] for row in passed)
    failed = build_success_criteria(
        highway_collision_pct=9.0,
        speed_pct_of_expert=96.0,
        collision_margin_ok=False,
        step_pct=80.0,
        merge_collision_pct=12.0,
        coverage_pct=60.0,
    )
    assert not any(row["passed"] for row in failed)
    # Speed value can clear 95 while the collision margin still fails the criterion.
    mixed = build_success_criteria(
        highway_collision_pct=4.0,
        speed_pct_of_expert=97.0,
        collision_margin_ok=False,
        step_pct=10.0,
        merge_collision_pct=1.0,
        coverage_pct=None,
    )
    by_name = {row["name"]: row for row in mixed}
    assert by_name["Mean speed vs. rule-based expert"]["passed"] is False
    assert by_name["Test coverage of src/"]["passed"] is False
    text = results_markdown({"success_criteria": passed, "metrics": {"test_table": []}})
    assert "7.0000" in text
    detailed = results_markdown(
        {
            "success_criteria": passed,
            "metrics": {
                "test_table": [
                    {
                        "scenario": "highway",
                        "policy": "ppo_mlp_scratch",
                        "collision_pct": 0.0,
                        "mean_speed": 5.5,
                        "mean_abs_jerk": 0.2,
                        "mean_return": 1.0,
                    }
                ],
                "ppo_budget": {"timesteps_per_run": 50176},
                "selected_policy": {"highway": "bc_attention"},
            },
        }
    )
    assert "50176" in detailed
    assert "5.50 m/s" in detailed
    assert "bc_attention" in detailed


def test_resume_uses_only_supplied_numbers(tmp_path):
    payload = {
        "metrics": {
            "resume_numbers": {
                "highway_collision_pct": "1.25",
                "step_pct": "0.00",
                "merge_collision_pct": "3.50",
                "speed_pct_of_expert": "97.10",
                "n_test_seeds": 200,
            }
        }
    }
    path = tmp_path / "RESUME_NOTES.md"
    write_resume(path, payload)
    text = path.read_text(encoding="utf-8")
    for token in ("1.25", "0.00", "3.50", "97.10", "200"):
        assert token in text
    assert "—" not in text
    assert "leveraged" not in text.lower()
