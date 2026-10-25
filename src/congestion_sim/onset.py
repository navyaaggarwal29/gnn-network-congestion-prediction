"""Scenario datasets, causal features, and episode-level onset metrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .simulator import NetworkSimulator, ScenarioTrafficProfile
from .topology import NetworkTopology
from .dataset import directed_link_edge_index, directed_link_nodes

ONSET_THRESHOLD = 0.85
SCENARIOS = ("gradual_ramp", "sustained_bottleneck", "periodic", "correlated", "sudden_burst")
ONSET_FEATURES = ["utilization", "queue_occupancy", "delay_ms", "packet_loss_pct",
                  "offered_demand_mbps", "bandwidth_mbps", "utilization_trend",
                  "queue_growth_rate", "offered_load_trend", "utilization_rolling_std",
                  "recent_drop_rate"]


def episode_lengths(values: np.ndarray, threshold: float = ONSET_THRESHOLD) -> list[int]:
    """Return lengths of contiguous above-threshold intervals."""
    hot = np.asarray(values) >= threshold
    starts = np.flatnonzero(hot & ~np.r_[False, hot[:-1]])
    ends = np.flatnonzero(hot & ~np.r_[hot[1:], False])
    return [int(end - start + 1) for start, end in zip(starts, ends)]


def add_causal_temporal_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add current/past-only link statistics; never reads future rows."""
    result = frame.sort_values(["link_id", "step"]).copy()
    groups = result.groupby("link_id", sort=False)
    result["utilization_trend"] = groups.utilization.diff().fillna(0.0)
    result["queue_growth_rate"] = groups.queue_occupancy.diff().fillna(0.0)
    result["offered_load_trend"] = groups.offered_demand_mbps.diff().fillna(0.0)
    result["utilization_rolling_std"] = groups.utilization.transform(
        lambda values: values.rolling(4, min_periods=1).std(ddof=0))
    result["recent_drop_rate"] = groups.packet_loss_pct.transform(
        lambda values: values.rolling(3, min_periods=1).mean())
    return result.sort_values(["step", "link_id"]).reset_index(drop=True)


@dataclass
class OnsetData:
    x: np.ndarray
    onset: np.ndarray
    onset_mask: np.ndarray
    continued: np.ndarray
    already_congested: np.ndarray
    run_id: np.ndarray
    scenario: np.ndarray
    seed: np.ndarray
    decision_step: np.ndarray
    utilization_by_run: dict[str, np.ndarray]
    measurements: pd.DataFrame
    link_nodes: list[tuple[str, str]]
    edge_index: np.ndarray
    window: int
    horizon: int


def generate_onset_data(topology: NetworkTopology, seeds: list[int], steps: int = 80,
                        window: int = 10, horizon: int = 5) -> OnsetData:
    """Simulate all scenarios; split assignment is left to the seed-level caller."""
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("At least three unique simulation seeds are required")
    if steps <= window + horizon:
        raise ValueError("steps must exceed window + horizon")
    links = directed_link_nodes(topology)
    link_ids = [f"{u}->{v}" for u, v in links]
    samples, onset_labels, masks, continued_labels, current_states = [], [], [], [], []
    sample_runs, sample_scenarios, sample_seeds, sample_steps = [], [], [], []
    utilizations: dict[str, np.ndarray] = {}
    frames = []
    for seed in seeds:
        for scenario_index, scenario in enumerate(SCENARIOS):
            run_id = f"{scenario}-seed-{seed}"
            simulator = NetworkSimulator(topology, seed=seed * 1009 + scenario_index)
            frame = simulator.run(steps, ScenarioTrafficProfile(kind=scenario))
            frame["run_id"], frame["scenario"], frame["seed"] = run_id, scenario, seed
            frame["link_id"] = pd.Categorical(frame["link_id"], categories=link_ids, ordered=True)
            frame = add_causal_temporal_features(frame)
            pivot = frame.pivot(index="step", columns="link_id", values=ONSET_FEATURES)
            features = pivot[ONSET_FEATURES].to_numpy(dtype=np.float32).reshape(steps, len(ONSET_FEATURES), len(links))
            features = features.transpose(0, 2, 1)
            utilization = pivot["utilization"].to_numpy(dtype=np.float32)
            utilizations[run_id] = utilization
            for decision in range(window - 1, steps - horizon):
                start = decision - window + 1
                future = utilization[decision + 1:decision + horizon + 1]
                current = utilization[decision]
                is_onset_eligible = current < ONSET_THRESHOLD
                onsets = (future >= ONSET_THRESHOLD).any(axis=0) & is_onset_eligible
                continued = (future >= ONSET_THRESHOLD).any(axis=0) & (current >= ONSET_THRESHOLD)
                samples.append(features[start:decision + 1])
                onset_labels.append(onsets.astype(np.float32))
                masks.append(is_onset_eligible.astype(np.float32))
                continued_labels.append(continued.astype(np.float32))
                current_states.append((current >= ONSET_THRESHOLD).astype(np.float32))
                sample_runs.append(run_id); sample_scenarios.append(scenario)
                sample_seeds.append(seed); sample_steps.append(decision)
            frames.append(frame)
    return OnsetData(
        x=np.stack(samples), onset=np.stack(onset_labels), onset_mask=np.stack(masks),
        continued=np.stack(continued_labels), already_congested=np.stack(current_states),
        run_id=np.asarray(sample_runs, dtype=str), scenario=np.asarray(sample_scenarios, dtype=str),
        seed=np.asarray(sample_seeds, dtype=np.int64), decision_step=np.asarray(sample_steps, dtype=np.int64),
        utilization_by_run=utilizations, measurements=pd.concat(frames, ignore_index=True),
        link_nodes=links, edge_index=directed_link_edge_index(topology), window=window, horizon=horizon)


def split_onset_indices(data: OnsetData, train_fraction: float = .70,
                        validation_fraction: float = .15) -> dict[str, np.ndarray]:
    """Assign complete independent seed groups to chronological splits."""
    seeds = sorted(set(data.seed.tolist()))
    n_train = max(1, int(round(len(seeds) * train_fraction)))
    n_validation = max(1, int(round(len(seeds) * validation_fraction)))
    if n_train + n_validation >= len(seeds):
        n_train = len(seeds) - n_validation - 1
    train, validation = set(seeds[:n_train]), set(seeds[n_train:n_train + n_validation])
    test = set(seeds[n_train + n_validation:])
    return {name: np.flatnonzero(np.isin(data.seed, list(group))) for name, group in
            (("train", train), ("validation", validation), ("test", test))}


def normalize_onset_features(x: np.ndarray, splits: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    """Normalize by train windows only, retaining static/near-constant feature stability."""
    train = x[splits["train"]].reshape(-1, x.shape[-1])
    mean, std = train.mean(0), train.std(0)
    std[std < 1e-8] = 1.0
    normalized = ((x - mean) / std).astype(np.float32)
    return normalized, {"feature_columns": ONSET_FEATURES, "mean": mean.tolist(), "std": std.tolist(),
                       "normalization_fit": "training_windows_only"}


def episode_event_metrics(scores: np.ndarray, data: OnsetData, indices: np.ndarray,
                          threshold: float, decision_delay: int = 2) -> dict[str, float | int | None]:
    """Match alert episodes to actual onset episodes once, avoiding overlapping-window double counts."""
    alerts_per_run: dict[str, int] = {}
    tp = fp = fn = eligible_events = useful = 0
    leads: list[int] = []
    for run_id in dict.fromkeys(data.run_id[indices].tolist()):
        sample_ids = [int(i) for i in indices if data.run_id[i] == run_id]
        sample_ids.sort(key=lambda i: int(data.decision_step[i]))
        util = data.utilization_by_run[run_id]
        link_count = util.shape[1]
        run_alert_episodes = 0
        for link in range(link_count):
            series = util[:, link]
            congested = series >= ONSET_THRESHOLD
            onset_steps = np.flatnonzero(congested & ~np.r_[False, congested[:-1]])
            for onset_step in onset_steps:
                possible_decisions = [int(data.decision_step[i]) for i in sample_ids
                                      if int(data.decision_step[i]) < onset_step
                                      and onset_step - int(data.decision_step[i]) <= data.horizon
                                      and series[int(data.decision_step[i])] < ONSET_THRESHOLD]
                if possible_decisions:
                    eligible_events += 1
            alert_steps = []
            for i in sample_ids:
                decision = int(data.decision_step[i])
                if series[decision] < ONSET_THRESHOLD and scores[i, link] >= threshold:
                    alert_steps.append(decision)
            # Merge adjacent alert windows into one alert episode for this link.
            alert_groups: list[list[int]] = []
            for decision in alert_steps:
                if not alert_groups or decision > alert_groups[-1][-1] + 1:
                    alert_groups.append([decision])
                else:
                    alert_groups[-1].append(decision)
            used_events: set[int] = set()
            for group in alert_groups:
                start = group[0]
                matches = [int(onset) for onset in onset_steps if int(onset) not in used_events
                           and start < onset <= start + data.horizon]
                if matches:
                    onset = min(matches)
                    used_events.add(onset); tp += 1
                    lead_steps = onset - start
                    leads.append(lead_steps)
                    useful += int(lead_steps >= decision_delay)
                else:
                    fp += 1; run_alert_episodes += 1
            fn += sum(1 for onset in onset_steps if int(onset) not in used_events
                      and any(int(data.decision_step[i]) < onset
                              and onset - int(data.decision_step[i]) <= data.horizon
                              and series[int(data.decision_step[i])] < ONSET_THRESHOLD for i in sample_ids))
        alerts_per_run[run_id] = run_alert_episodes
    precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(1e-12, precision + recall),
            "true_positive_episodes": tp, "false_alert_episodes": fp,
            "missed_episodes": fn, "eligible_onset_episodes": eligible_events,
            "episode_detection_rate": tp / max(1, eligible_events),
            "false_alert_episodes_per_run": fp / max(1, len(alerts_per_run)),
            "mean_lead_steps": float(np.mean(leads)) if leads else None,
            "useful_lead_rate": useful / max(1, tp), "decision_delay_steps": decision_delay}
