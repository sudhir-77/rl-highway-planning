import numpy as np
import torch

from src.bc import action_class_weights, predict_actions, train_bc
from src.config import make_settings
from src.networks import EgoAttentionEncoder, MLPEncoder, build_policy


def test_attention_ignores_padding_and_permutes_other_vehicles():
    torch.manual_seed(0)
    encoder = EgoAttentionEncoder(n_features=5, features_dim=64, n_heads=4)
    encoder.eval()
    obs = torch.zeros(2, 10, 5)
    obs[:, :, 0] = 1.0
    obs[:, 0, 3] = 25.0
    obs[:, 1, 1] = 20.0
    obs[:, 1, 3] = -2.0
    obs[:, 2, 1] = -15.0
    obs[:, 2, 3] = 1.0
    base = encoder(obs)
    swapped = obs.clone()
    swapped[:, [1, 2]] = swapped[:, [2, 1]]
    assert torch.allclose(base, encoder(swapped), atol=1e-5)
    padded = obs.clone()
    padded[:, 5, 0] = 0.0
    padded[:, 5, 1] = 999.0
    padded[:, 5, 3] = -50.0
    masked = obs.clone()
    masked[:, 5, 0] = 0.0
    assert torch.allclose(encoder(padded), encoder(masked), atol=1e-5)
    flat = MLPEncoder(10, 5, 64)
    flat.eval()
    # The MLP sees vehicle order, so this check only locks the output shape.
    assert flat(obs).shape == (2, 64)


def test_class_weights_cap_rare_actions_and_ignore_absent_ones():
    # Braking is the majority here, so it is not upweighted. The rare lane change is.
    actions = np.array([4, 4, 4, 4, 1, 1, 0], dtype=np.int64)
    weights = action_class_weights(actions)
    assert weights[4] == weights[1]
    assert weights[0] > weights[4]
    assert weights[2] == 0.0
    assert weights[3] == 0.0
    rare_brake = np.array([1] * 90 + [4] * 10, dtype=np.int64)
    rare_weights = action_class_weights(rare_brake)
    assert rare_weights[4] > rare_weights[1]
    assert rare_weights[4] < 2.0


def test_bc_overfits_a_linear_rule_and_early_stops_when_frozen():
    settings = make_settings(True, "cpu")
    torch.manual_seed(0)
    model = build_policy("mlp", settings)
    n = 24
    obs = np.zeros((n, settings.n_obs_vehicles, settings.n_features), dtype=np.float32)
    obs[:, :, 0] = 1.0
    obs[:, 0, 3] = np.linspace(-30, 30, n).astype(np.float32)
    actions = (obs[:, 0, 3] > 0).astype(np.int64)
    idx = np.arange(n)
    info = train_bc(
        model,
        obs,
        actions,
        idx,
        idx[:8],
        epochs=40,
        batch_size=8,
        lr=1e-2,
        patience=40,
        device="cpu",
        seed=0,
    )
    assert info["train_accuracy"] >= 0.8
    pred = predict_actions(model, obs, "cpu", batch_size=8)
    assert pred.shape == (n,)
    frozen = build_policy("attention", settings)
    stopped = train_bc(
        frozen,
        obs,
        actions,
        idx,
        idx[:4],
        epochs=6,
        batch_size=8,
        lr=0.0,
        patience=1,
        device="cpu",
        seed=1,
    )
    assert stopped["epochs_ran"] == 2


def test_build_policy_attention_forward():
    settings = make_settings(True, "cpu")
    policy = build_policy("attention", settings)
    obs = torch.zeros(3, settings.n_obs_vehicles, settings.n_features)
    obs[:, 0, 0] = 1.0
    logits = policy(obs)
    assert logits.shape == (3, 5)
    action = policy.act(obs[0].numpy(), "cpu")
    assert action in range(5)
