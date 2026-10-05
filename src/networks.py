"""MLP and ego-attention policies shared by behavior cloning and PPO."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.config import FEATURE_SCALE


def _nearest_ahead(mask: torch.Tensor, longitudinal: torch.Tensor, vx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gap_source = torch.where(mask, longitudinal, torch.full_like(longitudinal, 1.0e6))
    gap, index = gap_source.min(dim=1)
    rel = vx.gather(1, index.unsqueeze(1)).squeeze(1)
    missing = gap > 1.0e5
    gap = torch.where(missing, torch.full_like(gap, 200.0), gap)
    rel = torch.where(missing, torch.zeros_like(rel), rel)
    return gap, rel


def leader_features(obs: torch.Tensor) -> torch.Tensor:
    """Same-lane gap and relative speed, plus the adjacent-lane gaps.

    Row 0 is the ego vehicle. IDM brakes for the vehicle in the current lane,
    and MOBIL looks at the neighbouring lanes. Missing leaders use a far gap.
    """
    # Wrapped envs store the simulator's own lead gaps on the ego row.
    # Those match the vehicles IDM and MOBIL actually use.
    if obs.shape[-1] >= 9:
        return torch.cat(
            [
                obs[:, 0, 5:9],
                (obs[:, 0, 3] / 30.0).unsqueeze(1),
                (obs[:, 0, 2] / 8.0).unsqueeze(1),
            ],
            dim=-1,
        )
    others = obs[:, 1:, :]
    present = others[..., 0] >= 0.5
    longitudinal = others[..., 1]
    lateral = others[..., 2]
    vx = others[..., 3]
    ahead = present & (longitudinal > 0)
    gap, rel = _nearest_ahead(ahead & (lateral.abs() <= 2.2), longitudinal, vx)
    left_gap, _left_rel = _nearest_ahead(ahead & (lateral < -2.2) & (lateral > -6.5), longitudinal, vx)
    right_gap, _right_rel = _nearest_ahead(ahead & (lateral > 2.2) & (lateral < 6.5), longitudinal, vx)
    return torch.stack(
        [
            gap / 80.0,
            rel / 10.0,
            left_gap / 80.0,
            right_gap / 80.0,
            obs[:, 0, 3] / 30.0,
            obs[:, 0, 2] / 8.0,
        ],
        dim=-1,
    )


def _scale_buffer(n_features: int) -> torch.Tensor:
    scale = torch.tensor(FEATURE_SCALE, dtype=torch.float32)
    if int(scale.numel()) < n_features:
        raise ValueError(f"expected at least {n_features} feature scales, got {scale.numel()}")
    return scale[:n_features]


class MLPEncoder(nn.Module):
    """Flatten the vehicle matrix and map it with a two-layer MLP."""

    def __init__(self, n_vehicles: int, n_features: int, features_dim: int = 64) -> None:
        super().__init__()
        self.n_vehicles = n_vehicles
        self.n_features = n_features
        self.register_buffer("scale", _scale_buffer(n_features))
        self.net = nn.Sequential(
            nn.Linear(n_vehicles * n_features + 6, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        scaled = obs / self.scale
        flat = scaled.reshape(obs.shape[0], -1)
        return self.net(torch.cat([flat, leader_features(obs)], dim=-1))


class EgoAttentionEncoder(nn.Module):
    """Ego query attends over nearby vehicles.

    Row 0 is the ego vehicle, matching highway-env's kinematic observation.
    Vehicles with presence 0 are masked out. The block is permutation-invariant
    over the non-ego rows.
    """

    def __init__(
        self,
        n_features: int,
        features_dim: int = 64,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        if features_dim % n_heads != 0:
            raise ValueError("features_dim must be divisible by n_heads")
        self.n_features = n_features
        self.register_buffer("scale", _scale_buffer(n_features))
        self.embed = nn.Sequential(
            nn.Linear(n_features, features_dim),
            nn.ReLU(),
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=features_dim,
            num_heads=n_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(features_dim)
        self.proj = nn.Sequential(
            nn.Linear(features_dim + 6, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        scaled = obs / self.scale
        presence = obs[..., 0]
        hidden = self.embed(scaled)
        query = hidden[:, :1, :]
        mask = presence < 0.5
        mask = mask.clone()
        mask[:, 0] = False
        attended, _weights = self.attn(
            query,
            hidden,
            hidden,
            key_padding_mask=mask,
            need_weights=False,
        )
        mixed = self.norm(hidden[:, 0, :] + attended.squeeze(1))
        return self.proj(torch.cat([mixed, leader_features(obs)], dim=-1))


class BCPolicy(nn.Module):
    """Encoder plus a linear action head. The same encoder is copied into PPO."""

    def __init__(self, encoder: nn.Module, features_dim: int, n_actions: int = 5) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(features_dim, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(obs))

    def act(self, obs: np.ndarray, device: str) -> int:
        self.eval()
        tensor = torch.from_numpy(np.ascontiguousarray(obs, dtype=np.float32)).unsqueeze(0)
        tensor = tensor.to(device)
        with torch.no_grad():
            logits = self.forward(tensor)
        return int(torch.argmax(logits, dim=-1).item())


def build_policy(arch: str, settings) -> BCPolicy:
    if arch == "mlp":
        encoder: nn.Module = MLPEncoder(settings.n_obs_vehicles, settings.n_features, settings.features_dim)
    elif arch == "attention":
        encoder = EgoAttentionEncoder(settings.n_features, settings.features_dim, settings.attn_heads)
    else:
        raise KeyError(arch)
    return BCPolicy(encoder, settings.features_dim, n_actions=5)
