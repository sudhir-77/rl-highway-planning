from src.config import PROJECT_SLUG, make_settings
from src.pipeline import run_pipeline


def test_quick_pipeline(tmp_path):
    payload = run_pipeline(
        quick=True,
        device="cpu",
        output_dir=tmp_path / "results",
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        measure_coverage=False,
        readme_path=None,
        resume_path=tmp_path / "RESUME_NOTES.md",
    )
    assert payload["project"] == PROJECT_SLUG == "rl-highway-planning"
    assert len(payload["success_criteria"]) == 5
    assert (tmp_path / "results" / "episodes.csv").exists()
    assert (tmp_path / "results" / "learning_curves.csv").exists()
    assert (tmp_path / "results" / "bc_test_predictions.csv").exists()
    assert (tmp_path / "results" / "metrics.json").exists()
    assert (tmp_path / "RESUME_NOTES.md").exists()
    # The quick subset is not the 200-seed evaluation.
    assert payload["metrics"]["resume_numbers"]["n_test_seeds"] == 2
    full = make_settings(False, "cpu")
    assert full.ppo_timesteps >= 50_000
    assert payload["metrics"]["ppo_budget"]["timesteps_per_run"] == 128
