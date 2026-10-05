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
    # Braking is about 10% of highway labels and lane changes are under 2%.
    # Square-root weights give those classes a larger share of the loss.
    # The cap is lower for the common rare class so the policy does not
    # brake on every close gap. Unseen actions get weight 0.
    train_actions = np.asarray(actions)[train_idx]
    counts = np.bincount(train_actions, minlength=5).astype(np.float64)
    total = float(counts.sum())
    fraction = counts / max(total, 1.0)
    raw_weights = np.ones(5, dtype=np.float64)
    present = counts > 0
    moderate = present & (fraction < 0.15)
    raw_weights[moderate] = np.sqrt(1.0 / np.maximum(fraction[moderate], 1e-6))
    raw_weights[moderate] = np.clip(raw_weights[moderate], 1.0, 3.0)
    very_rare = present & (fraction < 0.02)
    raw_weights[very_rare] = np.clip(np.sqrt(1.0 / np.maximum(fraction[very_rare], 1e-6)), 1.0, 8.0)
    raw_weights[~present] = 0.0
    if present.any():
        raw_weights[present] = raw_weights[present] / raw_weights[present].mean()
    class_weights = torch.tensor(raw_weights, dtype=torch.float32, device=device)
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
