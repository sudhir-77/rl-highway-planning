"""PPO from scratch and PPO initialized from a behavior-cloning checkpoint."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from torch import nn

from src.config import Settings
from src.envs import make_env
from src.networks import BCPolicy, EgoAttentionEncoder, MLPEncoder


class MLPFeaturesExtractor(BaseFeaturesExtractor):
    """SB3 feature extractor around :class:`MLPEncoder`."""

    def __init__(self, observation_space, features_dim: int = 64) -> None:
        super().__init__(observation_space, features_dim)
        n_veh, n_feat = observation_space.shape
        self.encoder = MLPEncoder(int(n_veh), int(n_feat), features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.encoder(observations)


class AttentionFeaturesExtractor(BaseFeaturesExtractor):
    """SB3 feature extractor around :class:`EgoAttentionEncoder`."""

    def __init__(self, observation_space, features_dim: int = 64, n_heads: int = 4) -> None:
        super().__init__(observation_space, features_dim)
        _n_veh, n_feat = observation_space.shape
        self.encoder = EgoAttentionEncoder(int(n_feat), features_dim, n_heads)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.encoder(observations)


def extractor_class(arch: str):
    if arch == "mlp":
        return MLPFeaturesExtractor
    if arch == "attention":
        return AttentionFeaturesExtractor
    raise KeyError(arch)


def copy_bc_to_ppo(model: PPO, bc_policy: BCPolicy) -> None:
    """Copy the cloned encoder and action head. The value head stays randomly initialized."""
    model.policy.features_extractor.encoder.load_state_dict(bc_policy.encoder.state_dict())
    with torch.no_grad():
        model.policy.action_net.weight.copy_(bc_policy.head.weight)
        model.policy.action_net.bias.copy_(bc_policy.head.bias)
    model.policy.set_training_mode(False)


def _dummy_env(scenario: str, settings: Settings, seed: int) -> DummyVecEnv:
    def _init():
        env = make_env(scenario, settings, render=False)
        env.reset(seed=seed)
        return env

    return DummyVecEnv([_init])


def _subprocess_env(scenario: str, settings: Settings, seed: int, n_envs: int) -> SubprocVecEnv:
    """Parallel env steps. Workers stay at one thread so four of them fit in memory."""

    def _thunk(rank: int):
        def _init():
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
            os.environ["NUMEXPR_NUM_THREADS"] = "1"
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            env = make_env(scenario, settings, render=False)
            env.reset(seed=seed + rank)
            return env

        return _init

    return SubprocVecEnv([_thunk(rank) for rank in range(n_envs)], start_method="spawn")


def _vec_env(scenario: str, settings: Settings, seed: int, n_envs: int | None = None) -> VecEnv:
    count = settings.ppo_n_envs if n_envs is None else n_envs
    if count <= 1:
        return _dummy_env(scenario, settings, seed)
    return _subprocess_env(scenario, settings, seed, count)


def build_ppo(
    scenario: str,
    arch: str,
    settings: Settings,
    *,
    seed: int,
    learning_rate: float,
    ent_coef: float,
    clip_range: float,
) -> tuple[PPO, VecEnv]:
    env = _vec_env(scenario, settings, seed)
    kwargs = {
        "features_extractor_class": extractor_class(arch),
        "features_extractor_kwargs": {"features_dim": settings.features_dim},
        "net_arch": dict(pi=[], vf=[64]),
        "activation_fn": nn.ReLU,
    }
    if arch == "attention":
        kwargs["features_extractor_kwargs"] = {
            "features_dim": settings.features_dim,
            "n_heads": settings.attn_heads,
        }
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=settings.ppo_n_steps,
        batch_size=settings.ppo_batch,
        n_epochs=settings.ppo_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=seed,
        device=settings.device,
        verbose=0,
        policy_kwargs=kwargs,
    )
    return model, env


def deterministic_return(model: PPO, scenario: str, settings: Settings, seeds: list[int]) -> float:
    """Mean undiscounted return of the deterministic policy on ``seeds``."""
    from src.evaluate import run_episode

    model.policy.set_training_mode(False)
    env = make_env(scenario, settings, render=False)
    returns: list[float] = []
    try:
        def action_fn(_env, obs):
            action, _state = model.predict(obs, deterministic=True)
            return int(np.asarray(action).reshape(-1)[0])

        for seed in seeds:
            episode = run_episode(env, seed, action_fn, settings)
            returns.append(float(episode["return"]))
    finally:
        env.close()
    if not returns:
        return 0.0
    return float(np.mean(returns))


def train_ppo(
    scenario: str,
    arch: str,
    init: str,
    settings: Settings,
    bc_policy: BCPolicy | None,
    curve_seeds: list[int],
    checkpoint: Path,
    curve_path: Path,
) -> list[dict]:
    """Train PPO and return the deterministic-return curve, including step 0.

    ``init`` is ``scratch`` or ``bc``. Checkpoints are reused when the training
    fingerprint matches.
    """
    fingerprint = settings.train_fingerprint() + f":{scenario}:{arch}:{init}"
    if checkpoint.exists() and curve_path.exists():
        meta_path = checkpoint.with_suffix(".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("fingerprint") == fingerprint:
                print(f"[{scenario}] loaded PPO {arch}/{init}", flush=True)
                return list(meta["curve"])

    seed = settings.master_seed + (0 if scenario == "highway" else 100) + (0 if arch == "mlp" else 10)
    lr = settings.ppo_lr if init == "scratch" else settings.ppo_lr_bc
    ent = settings.ppo_ent_scratch if init == "scratch" else settings.ppo_ent_bc
    clip = settings.ppo_clip if init == "scratch" else settings.ppo_clip_bc
    model, env = build_ppo(
        scenario,
        arch,
        settings,
        seed=seed,
        learning_rate=lr,
        ent_coef=ent,
        clip_range=clip,
    )
    try:
        if init == "bc":
            if bc_policy is None:
                raise ValueError("BC initialization requires a behavior-cloning policy")
            copy_bc_to_ppo(model, bc_policy)
        curve: list[dict] = []
        step0 = deterministic_return(model, scenario, settings, curve_seeds)
        curve.append({"timestep": 0, "mean_return": step0, "n_seeds": len(curve_seeds)})
        print(f"[{scenario}] PPO {arch}/{init} step 0 return {step0:.3f}", flush=True)
        trained = 0
        first = True
        while trained < settings.ppo_timesteps:
            chunk = min(settings.curve_every, settings.ppo_timesteps - trained)
            model.learn(total_timesteps=chunk, reset_num_timesteps=first, progress_bar=False)
            first = False
            trained = int(model.num_timesteps)
            value = deterministic_return(model, scenario, settings, curve_seeds)
            curve.append({"timestep": trained, "mean_return": value, "n_seeds": len(curve_seeds)})
            print(f"[{scenario}] PPO {arch}/{init} step {trained} return {value:.3f}", flush=True)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if int(model.num_timesteps) < settings.ppo_timesteps:
            raise RuntimeError(
                f"PPO {scenario}/{arch}/{init} stopped at {model.num_timesteps} steps, "
                f"below the budget of {settings.ppo_timesteps}"
            )
        model.save(str(checkpoint))
        meta = {"fingerprint": fingerprint, "curve": curve, "num_timesteps": int(model.num_timesteps)}
        checkpoint.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
        curve_path.write_text(json.dumps(curve), encoding="utf-8")
        return curve
    finally:
        env.close()


def load_ppo(scenario: str, arch: str, settings: Settings, checkpoint: Path, seed: int) -> PPO:
    """Load a checkpoint for deterministic evaluation. Schedules are not resumed."""
    env = _vec_env(scenario, settings, seed, n_envs=1)
    model = PPO.load(
        str(checkpoint),
        env=env,
        device=settings.device,
        custom_objects={"learning_rate": 0.0, "clip_range": 0.2},
    )
    return model
