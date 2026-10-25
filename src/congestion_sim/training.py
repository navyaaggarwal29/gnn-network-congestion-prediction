"""Deterministic mini-batch training and checkpoint utilities."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .models import LinkGRU, TemporalGNN, parameter_count, regression_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return device
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_split(path: Path) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Load normalized features and labels as [samples,time,nodes,features]."""
    with np.load(path, allow_pickle=False) as data:
        x = data["x"].transpose(0, 1, 3, 2).copy()
        y = data["y"].copy()
        edge_index = data["edge_index"].copy()
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"Non-finite values found in {path}")
    return torch.from_numpy(x), torch.from_numpy(y), edge_index


def _predict(model: torch.nn.Module, loader: DataLoader, edge_index: torch.Tensor | None,
             device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    predictions, targets = [], []
    loss_sum, count = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x, edge_index) if isinstance(model, TemporalGNN) else model(x)
            loss_sum += torch.nn.functional.mse_loss(pred, y, reduction="sum").item()
            count += y.numel()
            predictions.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
    return loss_sum / max(count, 1), np.concatenate(predictions), np.concatenate(targets)


def train_one(model_name: str, x_train: torch.Tensor, y_train: torch.Tensor,
              x_validation: torch.Tensor, y_validation: torch.Tensor,
              edge_index_array: np.ndarray, output_dir: Path, *, seed: int = 42,
              device_name: str = "auto", batch_size: int = 32, epochs: int = 60,
              learning_rate: float = 1e-3, hidden_dim: int = 64, dropout: float = 0.2,
              patience: int = 8, min_delta: float = 1e-5,
              positive_weight: float | None = None, prediction_horizon: int = 5,
              feature_normalization: dict[str, Any] | None = None) -> dict[str, Any]:
    """Train one model, select by validation MAE, and save its best checkpoint."""
    seed_everything(seed)
    device = resolve_device(device_name)
    input_dim = x_train.shape[-1]
    if model_name.startswith("gnn"):
        model: torch.nn.Module = TemporalGNN(input_dim, hidden_dim, dropout=dropout)
    elif model_name == "gru":
        model = LinkGRU(input_dim, hidden_dim, dropout=dropout)
        positive_weight = None
    else:
        raise ValueError(f"Unknown model: {model_name}")
    model.to(device)
    edge_index = torch.as_tensor(edge_index_array, dtype=torch.long, device=device)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size,
                              shuffle=True, generator=generator, drop_last=False)
    validation_loader = DataLoader(TensorDataset(x_validation, y_validation), batch_size=batch_size,
                                   shuffle=False, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{model_name}.pt"
    best_mae, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history: list[dict[str, float | int | str]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        objective_total, standard_mse_total, target_count = 0.0, 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, edge_index) if isinstance(model, TemporalGNN) else model(x)
            loss = regression_loss(pred, y, positive_weight=positive_weight)
            standard_mse_total += torch.nn.functional.mse_loss(pred.detach(), y, reduction="sum").item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            objective_total += loss.item() * y.numel()
            target_count += y.numel()
        val_mse, val_pred, val_true = _predict(model, validation_loader, edge_index, device)
        val_mae = float(np.mean(np.abs(val_pred - val_true)))
        scheduler.step(val_mae)
        history.append({"model": model_name, "epoch": epoch,
                        "train_loss": objective_total / max(target_count, 1),
                        "train_mse": standard_mse_total / max(target_count, 1),
                        "validation_mse": val_mse, "validation_mae": val_mae,
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        if val_mae < best_mae - min_delta:
            best_mae, best_epoch, stale = val_mae, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError(f"Training did not produce a checkpoint for {model_name}")
    model.load_state_dict(best_state)
    checkpoint = {
        "model_name": model_name,
        "model_config": {"input_dim": input_dim, "hidden_dim": hidden_dim, "dropout": dropout,
                         "window": int(x_train.shape[1]), "prediction_horizon": prediction_horizon},
        "state_dict": best_state,
        "feature_normalization": feature_normalization,
        "best_epoch": best_epoch,
        "best_validation_mae": best_mae,
        "parameter_count": parameter_count(model),
        "positive_weight": positive_weight,
        "seed": seed,
    }
    torch.save(checkpoint, checkpoint_path)
    return {"model": model_name, "device": str(device), "epochs_trained": len(history),
            "best_epoch": best_epoch, "best_validation_mae": best_mae,
            "parameter_count": parameter_count(model), "positive_weight": positive_weight,
            "checkpoint": str(checkpoint_path), "history": history}


def load_model_checkpoint(path: Path, device: torch.device | str = "cpu") -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    if checkpoint["model_name"].startswith("gnn"):
        model = TemporalGNN(config["input_dim"], config["hidden_dim"], dropout=config["dropout"])
    elif checkpoint["model_name"] == "gru":
        model = LinkGRU(config["input_dim"], config["hidden_dim"], dropout=config["dropout"])
    else:
        raise ValueError(f"Unknown checkpoint model {checkpoint['model_name']}")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
