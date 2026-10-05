import numpy as np
import torch

from src.config import make_settings
from src.networks import build_policy
from src.ppo import build_ppo, copy_bc_to_ppo


def test_bc_weights_match_ppo_actions_before_training():
    settings = make_settings(True, "cpu")
    torch.manual_seed(0)
    bc_policy = build_policy("attention", settings)
    model, env = build_ppo(
        "highway",
        "attention",
        settings,
        seed=0,
        learning_rate=1e-4,
        ent_coef=0.0,
        clip_range=0.1,
    )
    try:
        copy_bc_to_ppo(model, bc_policy)
        obs, _info = env.envs[0].reset(seed=5)
        bc_action = bc_policy.act(obs, "cpu")
        ppo_action, _state = model.predict(obs, deterministic=True)
        assert int(np.asarray(ppo_action).reshape(-1)[0]) == bc_action
    finally:
        env.close()
