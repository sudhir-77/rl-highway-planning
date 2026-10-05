import numpy as np

from src.config import IDLE, FASTER, LANE_LEFT, LANE_RIGHT, SLOWER, SPEED_STEP, make_settings
from src.dataset import DatasetError, file_sha256, load_dataset, save_dataset
from src.envs import make_env
from src.evaluate import recent_lane_changes, run_episode
from src.expert import expert_action
from src.plots import plot_grouped_bars, plot_learning_curves


def test_dataset_checksum_detects_corruption(tmp_path):
    path = tmp_path / "tiny.npz"
    manifest = tmp_path / "tiny.manifest.json"
    obs = np.zeros((4, 2, 3), dtype=np.float32)
    actions = np.array([0, 1, 1, 4], dtype=np.int8)
    episodes = np.array([0, 0, 1, 1], dtype=np.int32)
    seeds = np.array([3, 3, 4, 4], dtype=np.int32)
    save_dataset(
        path,
        manifest,
        observations=obs,
        actions=actions,
        episode_ids=episodes,
        seeds=seeds,
        fingerprint="fp",
        stats={"episodes": 2},
    )
    loaded = load_dataset(path, manifest, "fp")
    assert loaded["rows"] == 4
    assert file_sha256(path) != "0"
    blob = path.read_bytes()
    path.write_bytes(blob[:-1] + bytes([blob[-1] ^ 0xFF]))
    try:
        load_dataset(path, manifest, "fp")
    except DatasetError as exc:
        assert "checksum" in str(exc)
    else:
        raise AssertionError("corrupt dataset was accepted")


def test_expert_actions_are_legal_and_lane_keep_is_longitudinal():
    settings = make_settings(True, "cpu")
    env = make_env("highway", settings)
    try:
        env.reset(seed=0)
        assert env.observation_space.shape == (settings.n_obs_vehicles, settings.n_features)
        for _ in range(6):
            action = expert_action(env, enable_lane_change=True)
            available = {int(item) for item in env.unwrapped.action_type.get_available_actions()}
            assert action in available
            kept = expert_action(env, enable_lane_change=False)
            assert kept in {IDLE, FASTER, SLOWER}
            assert kept not in {LANE_LEFT, LANE_RIGHT}
            _obs, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                env.reset(seed=1)
        episode = run_episode(env, 11, lambda e, _o: expert_action(e, enable_lane_change=True), settings)
        assert episode["n_steps"] > 0
        assert episode["failure_mode"] in {"none", "lane_change", "rear_end", "offroad", "other"}
    finally:
        env.close()


def test_idle_holds_measured_speed_not_a_stale_setpoint():
    settings = make_settings(True, "cpu")
    env = make_env("highway", settings)
    try:
        env.reset(seed=0)
        ego = env.unwrapped.vehicle
        speed = float(np.clip(ego.speed, ego.target_speeds[0], ego.target_speeds[-1]))
        expected = float(ego.index_to_speed(ego.speed_to_index(speed)))
        ego.speed_index = 0
        ego.target_speed = float(ego.target_speeds[0])
        env.step(IDLE)
        assert abs(float(ego.target_speed) - expected) < 1e-5
    finally:
        env.close()


def test_slower_drops_the_setpoint_by_one_speed_step():
    settings = make_settings(True, "cpu")
    env = make_env("highway", settings)
    try:
        env.reset(seed=0)
        ego = env.unwrapped.vehicle
        ego.speed = 20.0
        env.step(IDLE)
        held = float(ego.target_speed)
        env.step(SLOWER)
        assert abs((held - float(ego.target_speed)) - SPEED_STEP) < 1e-5
    finally:
        env.close()


def test_expert_changes_lanes_on_highway_and_merge():
    settings = make_settings(True, "cpu")
    for scenario in ("highway", "merge"):
        env = make_env(scenario, settings)
        lane_actions = 0
        try:
            for seed in range(12):
                env.reset(seed=seed)
                done = False
                steps = 0
                while not done and steps < settings.max_episode_steps:
                    action = expert_action(env, enable_lane_change=True)
                    available = {int(item) for item in env.unwrapped.action_type.get_available_actions()}
                    assert action in available
                    if action in (LANE_LEFT, LANE_RIGHT):
                        lane_actions += 1
                    _obs, _reward, terminated, truncated, _info = env.step(action)
                    done = bool(terminated or truncated)
                    steps += 1
        finally:
            env.close()
        assert lane_actions >= 1, scenario


def test_merge_observation_shape():
    settings = make_settings(True, "cpu")
    env = make_env("merge", settings)
    try:
        obs, _info = env.reset(seed=2)
        assert obs.shape == (settings.n_obs_vehicles, settings.n_features)
        assert float(obs[0, 0]) == 1.0
    finally:
        env.close()


def test_recent_lane_changes_and_plots(tmp_path):
    assert recent_lane_changes([1, 3, 0]) is True
    assert recent_lane_changes([1, 4, 1]) is False
    plot_learning_curves(
        [
            {"scenario": "highway", "architecture": "attention", "init": "scratch", "timestep": 0, "mean_return": 1.0},
            {"scenario": "highway", "architecture": "attention", "init": "bc", "timestep": 10, "mean_return": 2.0},
        ],
        tmp_path / "curves.png",
    )
    plot_grouped_bars(
        [{"scenario": "highway", "policy": "bc_mlp", "collision_pct": 1.0}],
        "collision_pct",
        "collision %",
        tmp_path / "bars.png",
        "title",
    )
    assert (tmp_path / "curves.png").exists()
    assert (tmp_path / "bars.png").exists()
