"""Fixed experiment settings.

Hyperparameters and seed blocks are chosen before any test-seed evaluation.
The test block is never used for collection, training, or model selection.
"""

from __future__ import annotations

from dataclasses import dataclass


FEATURES = ["presence", "x", "y", "vx", "vy"]
# 1 m/s steps. The expert emits SLOWER/FASTER from an IDM acceleration
# deadband, and the controller moves the setpoint by one step.
TARGET_SPEEDS = list(range(0, 31))
BRAKE_ACCEL = 1.0
ACCEL_ACCEL = 0.5
# Kinematic columns, then same-lane gap, same-lane range rate, left gap, right gap.
FEATURE_SCALE = [1.0, 200.0, 20.0, 40.0, 40.0, 1.0, 1.0, 1.0, 1.0]

# DiscreteMetaAction indexes from highway-env.
LANE_LEFT = 0
IDLE = 1
LANE_RIGHT = 2
FASTER = 3
SLOWER = 4

LEARNED_POLICIES = (
    "bc_mlp",
    "bc_attention",
    "ppo_mlp_scratch",
    "ppo_mlp_bc",
    "ppo_attention_scratch",
    "ppo_attention_bc",
)
BASELINE_POLICIES = ("idm_mobil", "idm_lane_keep")
ALL_POLICIES = BASELINE_POLICIES + LEARNED_POLICIES
SCENARIOS = ("highway", "merge")
ENV_IDS = {"highway": "highway-v0", "merge": "merge-v0"}


@dataclass(frozen=True)
class Settings:
    quick: bool
    device: str
    config_version: int = 5
    master_seed: int = 0
    n_obs_vehicles: int = 10
    n_features: int = 9
    sim_freq: int = 5
    policy_freq: int = 5
    duration_s: int = 20
    lanes: int = 4
    spawn_vehicles: int = 8
    expert_transitions: int = 100_000
    expert_seed_start: int = 0
    val_count: int = 40
    test_count: int = 200
    curve_count: int = 8
    bc_epochs: int = 12
    bc_batch: int = 512
    bc_lr: float = 1e-3
    bc_patience: int = 4
    ppo_timesteps: int = 4096
    ppo_n_steps: int = 512
    ppo_batch: int = 128
    ppo_epochs: int = 4
    ppo_lr: float = 3e-4
    ppo_lr_bc: float = 1e-4
    ppo_ent_scratch: float = 0.01
    ppo_ent_bc: float = 0.0
    ppo_clip: float = 0.2
    ppo_clip_bc: float = 0.1
    curve_every: int = 2048
    gif_count: int = 3
    features_dim: int = 64
    attn_heads: int = 4

    def __post_init__(self) -> None:
        if self.ppo_timesteps % self.ppo_n_steps != 0:
            raise ValueError("ppo_timesteps must be a multiple of ppo_n_steps")
        if self.curve_every % self.ppo_n_steps != 0:
            raise ValueError("curve_every must be a multiple of ppo_n_steps")
        if self.ppo_n_steps % self.ppo_batch != 0:
            raise ValueError("ppo_n_steps must be a multiple of ppo_batch")
        # One-step episodes would consume one seed per transition. That block
        # must end before the lowest reserved seed for the scenario.
        for scenario in SCENARIOS:
            last_possible = self.expert_seed_start + self.expert_transitions - 1
            first_reserved = min(self.seed_block(scenario, "val"))
            if last_possible >= first_reserved:
                raise ValueError(f"{scenario} expert seeds can overlap reserved seeds")

    @property
    def dt(self) -> float:
        return 1.0 / float(self.policy_freq)

    @property
    def max_episode_steps(self) -> int:
        return int(self.duration_s * self.policy_freq + 5)

    def seed_block(self, scenario: str, split: str) -> list[int]:
        """Disjoint seed blocks. Quick and full use different ranges."""
        if scenario not in SCENARIOS:
            raise KeyError(scenario)
        if self.quick:
            starts = {
                ("highway", "val"): 1_000,
                ("merge", "val"): 1_100,
                ("highway", "curve"): 1_200,
                ("merge", "curve"): 1_300,
                ("highway", "test"): 2_000,
                ("merge", "test"): 2_100,
            }
            counts = {"val": self.val_count, "curve": self.curve_count, "test": self.test_count}
        else:
            starts = {
                ("highway", "val"): 100_000,
                ("merge", "val"): 110_000,
                ("highway", "curve"): 120_000,
                ("merge", "curve"): 130_000,
                ("highway", "test"): 200_000,
                ("merge", "test"): 210_000,
            }
            counts = {"val": self.val_count, "curve": self.curve_count, "test": self.test_count}
        start = starts[(scenario, split)]
        return list(range(start, start + counts[split]))

    def reserved_seeds(self, scenario: str) -> set[int]:
        reserved: set[int] = set()
        for split in ("val", "curve", "test"):
            reserved.update(self.seed_block(scenario, split))
        return reserved

    def env_config(self) -> dict:
        return {
            "simulation_frequency": self.sim_freq,
            "policy_frequency": self.policy_freq,
            "duration": self.duration_s,
            "lanes_count": self.lanes,
            "vehicles_count": self.spawn_vehicles,
            "vehicles_density": 1.0,
            "normalize_reward": False,
            "right_lane_reward": 0.0,
            "high_speed_reward": 0.4,
            "lane_change_reward": 0.0,
            "collision_reward": -1.0,
            "reward_speed_range": [20, 30],
            "offroad_terminal": True,
            "show_trajectories": False,
            "real_time_rendering": False,
            "neighbour_vehicles_connected_lanes": True,
            "observation": {
                "type": "Kinematics",
                "vehicles_count": self.n_obs_vehicles,
                "features": list(FEATURES),
                "absolute": False,
                "order": "sorted",
                "normalize": False,
                "clip": False,
                "see_behind": True,
                "observe_intentions": False,
            },
            "action": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": True,
                "target_speeds": list(TARGET_SPEEDS),
            },
        }

    def data_fingerprint(self) -> str:
        import hashlib
        import json

        payload = {
            "config_version": self.config_version,
            "env": self.env_config(),
            "expert_transitions": self.expert_transitions,
            "expert_seed_start": self.expert_seed_start,
            "brake_accel": BRAKE_ACCEL,
            "accel_accel": ACCEL_ACCEL,
            "idm_delta": 4.0,
            "quick": self.quick,
        }
        raw = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(raw).hexdigest()[:16]

    def train_fingerprint(self) -> str:
        import hashlib
        import json

        payload = {
            "data": self.data_fingerprint(),
            "bc_epochs": self.bc_epochs,
            "bc_batch": self.bc_batch,
            "bc_lr": self.bc_lr,
            "bc_patience": self.bc_patience,
            "bc_loss": "sqrt_moderate3_rare8",
            "ppo_timesteps": self.ppo_timesteps,
            "ppo_n_steps": self.ppo_n_steps,
            "ppo_batch": self.ppo_batch,
            "ppo_epochs": self.ppo_epochs,
            "ppo_lr": self.ppo_lr,
            "ppo_lr_bc": self.ppo_lr_bc,
            "features_dim": self.features_dim,
            "attn_heads": self.attn_heads,
            "master_seed": self.master_seed,
        }
        raw = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()[:16]


def make_settings(quick: bool, device: str) -> Settings:
    if quick:
        return Settings(
            quick=True,
            device=device,
            duration_s=8,
            expert_transitions=200,
            val_count=2,
            test_count=2,
            curve_count=2,
            bc_epochs=2,
            bc_batch=64,
            bc_patience=1,
            ppo_timesteps=256,
            ppo_n_steps=128,
            ppo_batch=64,
            ppo_epochs=2,
            curve_every=256,
            gif_count=1,
        )
    return Settings(quick=False, device=device)
