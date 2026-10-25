"""Temporal link dataset creation, graph construction, and run-level splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .simulator import NetworkSimulator, TrafficProfile
from .topology import NetworkTopology


FEATURE_COLUMNS = ["utilization", "queue_occupancy", "delay_ms", "packet_loss_pct",
                   "offered_demand_mbps", "bandwidth_mbps"]
TRAFFIC_PROFILES = ("normal", "moderate", "burst", "sustained_high")


def directed_link_nodes(topology: NetworkTopology) -> list[tuple[str, str]]:
    """Return stable GNN node order, one node for every directed physical link."""
    return [(u, v) for a, b in topology.links for u, v in ((a, b), (b, a))]


def directed_link_edge_index(topology: NetworkTopology) -> np.ndarray:
    """Build directed line-graph edges for legal consecutive link traversal.

    Nodes i -> j when link i ends at the router where link j starts. An
    immediate reversal over the same physical cable is excluded.
    """
    links = directed_link_nodes(topology)
    edge_pairs = []
    for i, (source, router) in enumerate(links):
        for j, (next_source, destination) in enumerate(links):
            if router == next_source and not (source == destination):
                edge_pairs.append((i, j))
    if not edge_pairs:
        return np.empty((2, 0), dtype=np.int64)
    return np.asarray(edge_pairs, dtype=np.int64).T


@dataclass
class TemporalDataset:
    runs: dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]]
    link_nodes: list[tuple[str, str]]
    edge_index: np.ndarray
    window: int
    horizon: int


def generate_runs(topology: NetworkTopology, seeds: list[int], steps: int = 60,
                  window: int = 10, horizon: int = 5) -> tuple[TemporalDataset, pd.DataFrame]:
    """Simulate independent seed/profile runs and create aligned windows.

    A target is the utilization observed ``horizon`` intervals after the final
    observation in its input window. Split assignment is performed by run ID.
    """
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("Provide at least three unique seeds")
    if steps < window + horizon:
        raise ValueError("steps must be at least window + horizon")
    links = directed_link_nodes(topology)
    link_ids = [f"{u}->{v}" for u, v in links]
    all_measurements, runs = [], {}
    base_time = pd.Timestamp("2025-01-01T00:00:00Z")
    profile_index = {name: i for i, name in enumerate(TRAFFIC_PROFILES)}
    for seed in seeds:
        for profile_name in TRAFFIC_PROFILES:
            run_id = f"{profile_name}-seed-{seed}"
            sim_seed = seed * 1009 + profile_index[profile_name]
            simulator = NetworkSimulator(topology, seed=sim_seed)
            frame = simulator.run(steps, TrafficProfile(kind=profile_name))
            frame["run_id"] = run_id
            frame["traffic_profile"] = profile_name
            frame["seed"] = seed
            frame["timestamp"] = base_time + pd.to_timedelta(frame["time_seconds"], unit="s")
            frame["link_id"] = pd.Categorical(frame["link_id"], categories=link_ids, ordered=True)
            frame = frame.sort_values(["step", "link_id"]).reset_index(drop=True)
            pivot = frame.pivot(index="step", columns="link_id", values=FEATURE_COLUMNS)
            x = pivot[FEATURE_COLUMNS].to_numpy(dtype=np.float32).reshape(
                steps, len(FEATURE_COLUMNS), len(links))
            utilization = pivot["utilization"].to_numpy(dtype=np.float32)
            sample_x, sample_y = [], []
            for start in range(steps - window - horizon + 1):
                target_step = start + window - 1 + horizon
                sample_x.append(x[start:start + window])
                sample_y.append(utilization[target_step])
            runs[run_id] = (np.stack(sample_x), np.stack(sample_y), frame)
            frame["future_target_step"] = frame["step"] + horizon
            future = frame[["run_id", "step", "link_id", "utilization"]].rename(
                columns={"step": "future_target_step", "utilization": "future_utilization"})
            frame = frame.merge(future, on=["run_id", "future_target_step", "link_id"], how="left")
            all_measurements.append(frame)
    return TemporalDataset(runs, links, directed_link_edge_index(topology), window, horizon), pd.concat(all_measurements, ignore_index=True)


def split_run_ids(run_ids: list[str], train_fraction: float = 0.70,
                  validation_fraction: float = 0.15) -> dict[str, list[str]]:
    """Split by seed, keeping all traffic profiles for a seed in one split."""
    seeds = sorted({int(run_id.rsplit("-", 1)[1]) for run_id in run_ids})
    if len(seeds) < 3:
        raise ValueError("At least three independent seed groups are required")
    n_train = max(1, int(round(len(seeds) * train_fraction)))
    n_validation = max(1, int(round(len(seeds) * validation_fraction)))
    if n_train + n_validation >= len(seeds):
        n_train = len(seeds) - n_validation - 1
    train_seeds = set(seeds[:n_train])
    val_seeds = set(seeds[n_train:n_train + n_validation])
    test_seeds = set(seeds[n_train + n_validation:])
    split = {
        "train": [r for r in run_ids if int(r.rsplit("-", 1)[1]) in train_seeds],
        "validation": [r for r in run_ids if int(r.rsplit("-", 1)[1]) in val_seeds],
        "test": [r for r in run_ids if int(r.rsplit("-", 1)[1]) in test_seeds],
    }
    return split


def prepare_splits(dataset: TemporalDataset, split: dict[str, list[str]]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Concatenate split samples and normalize using training windows only."""
    raw: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, ids in split.items():
        raw[name] = (np.concatenate([dataset.runs[r][0] for r in ids]),
                     np.concatenate([dataset.runs[r][1] for r in ids]))
    train_values = raw["train"][0].transpose(0, 1, 3, 2).reshape(-1, len(FEATURE_COLUMNS))
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std[std == 0] = 1.0
    normalized = {name: {"x": ((x - mean[None, None, :, None]) / std[None, None, :, None]).astype(np.float32),
                         "y": y.astype(np.float32)}
                  for name, (x, y) in raw.items()}
    return normalized, {"feature_columns": FEATURE_COLUMNS, "mean": mean.tolist(), "std": std.tolist(),
                        "normalization_fit": "training_windows_only"}


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = np.asarray(y_pred) - np.asarray(y_true)
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2)))}


def evaluate_baselines(dataset: TemporalDataset, test_run_ids: list[str]) -> dict[str, Any]:
    y_true, persistence, moving_average = [], [], []
    for run_id in test_run_ids:
        x, y, _ = dataset.runs[run_id]
        # feature column 0 is current utilization; window ends at the latest
        # available point and the label is five steps beyond that point.
        y_true.append(y)
        persistence.append(x[:, -1, 0, :])
        moving_average.append(x[:, :, 0, :].mean(axis=1))
    truth = np.concatenate(y_true)
    p = np.concatenate(persistence)
    m = np.concatenate(moving_average)
    threshold = 0.85
    return {
        "persistence": regression_metrics(truth, p),
        "moving_average": regression_metrics(truth, m),
        "congestion_threshold": threshold,
        "test_positive_count": int((truth >= threshold).sum()),
        "test_label_count": int(truth.size),
    }
