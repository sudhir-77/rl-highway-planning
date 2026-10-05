"""Behavior cloning with early stopping on the validation split."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from src.networks import BCPolicy


def _accuracy(model: BCPolicy, obs: torch.Tensor, actions: torch.Tensor, idx: np.ndarray, device: str, batch: int) -> float:
    if len(idx) == 0:
        return 0.0
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, len(idx), batch):
            batch_idx = idx[start : start + batch]
            logits = model(obs[batch_idx].to(device))
            pred = torch.argmax(logits, dim=-1).cpu()
            target = actions[batch_idx]
            correct += int((pred == target).sum().item())
            total += int(target.numel())
    return float(correct / max(total, 1))


def _loss(
    model: BCPolicy,
    obs: torch.Tensor,
    actions: torch.Tensor,
    idx: np.ndarray,
    device: str,
    batch: int,
    criterion: nn.Module,
) -> float:
    if len(idx) == 0:
        return 0.0
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(idx), batch):
            batch_idx = idx[start : start + batch]
            logits = model(obs[batch_idx].to(device))
            target = actions[batch_idx].to(device)
            total += float(criterion(logits, target).item()) * len(batch_idx)
            count += len(batch_idx)
    return float(total / max(count, 1))


def action_class_weights(actions: np.ndarray, n_actions: int = 5) -> np.ndarray:
    """Weight rare lane-change labels only.

    Braking is not upweighted. A large SLOWER weight made the cloned policy
    brake on ordinary gaps and fall behind the expert's speed.
    """
    counts = np.bincount(np.asarray(actions, dtype=np.int64), minlength=n_actions).astype(np.float64)
    weights = np.zeros(n_actions, dtype=np.float64)
    present = counts > 0
    weights[present] = 1.0
    total = float(counts.sum())
    for lane in (0, 2):
        if counts[lane] <= 0 or total <= 0:
            continue
        fraction = float(counts[lane] / total)
        if fraction < 0.15:
            weights[lane] = min(2.5, float(np.sqrt(1.0 / max(fraction, 1e-6))))
    if present.any():
        weights[present] = weights[present] / weights[present].mean()
    return weights


def train_bc(
    model: BCPolicy,
    observations: np.ndarray,
    actions: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    device: str,
    seed: int,
) -> dict:
    """Minimize cross-entropy on ``train_idx`` and restore the best ``val_idx`` weights.

    ``train_idx`` and ``val_idx`` are row numbers. The test split is not read.
    """
    if len(train_idx) == 0:
        raise ValueError("behavior cloning received an empty training split")
    obs = torch.from_numpy(np.ascontiguousarray(observations, dtype=np.float32))
    act = torch.from_numpy(np.ascontiguousarray(actions, dtype=np.int64))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    class_weights = torch.tensor(
        action_class_weights(np.asarray(actions)[train_idx]),
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    best = float("inf")
    wait = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_ran = 0
    history: list[dict] = []
    for epoch in range(epochs):
        model.train()
        generator = torch.Generator()
        generator.manual_seed(seed + epoch)
        order = torch.randperm(len(train_idx), generator=generator).numpy()
        shuffled = train_idx[order]
        for start in range(0, len(shuffled), batch_size):
            batch_idx = shuffled[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(obs[batch_idx].to(device))
            loss = criterion(logits, act[batch_idx].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_loss = _loss(model, obs, act, val_idx, device, batch_size, criterion)
        train_loss = _loss(model, obs, act, train_idx, device, batch_size, criterion)
        epochs_ran = epoch + 1
        history.append({"epoch": epochs_ran, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best - 1e-4:
            best = val_loss
            wait = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    train_accuracy = _accuracy(model, obs, act, train_idx, device, batch_size)
    val_accuracy = _accuracy(model, obs, act, val_idx, device, batch_size)
    return {
        "best_val_loss": float(best if best < float("inf") else 0.0),
        "epochs_ran": epochs_ran,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "history": history,
    }


def predict_actions(model: BCPolicy, observations: np.ndarray, device: str, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    if len(observations) == 0:
        return np.zeros((0,), dtype=np.int64)
    obs = torch.from_numpy(np.ascontiguousarray(observations, dtype=np.float32))
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(obs), batch_size):
            logits = model(obs[start : start + batch_size].to(device))
            preds.append(torch.argmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(preds).astype(np.int64)
