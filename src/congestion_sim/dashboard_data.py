"""Validated, run-scoped data loaders and calculations used by the dashboard."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

STRATEGY_LABELS = {
    "static": "Static Dijkstra",
    "reactive": "Reactive routing",
    "gru": "GRU proactive",
    "gnn": "GNN proactive",
}
STRATEGY_ORDER = list(STRATEGY_LABELS)
RUN_COLUMNS = [
    "step", "time_seconds", "link_id", "source_router", "destination_router",
    "bandwidth_mbps", "propagation_delay_ms", "queue_capacity_bytes",
    "offered_demand_mbps", "utilization", "queue_occupancy", "delay_ms",
    "packet_loss_pct", "packets_offered", "packets_dropped", "measurement_source",
    "scenario", "seed", "strategy", "current_route", "predicted_congestion_probability",
    "routing_decision", "alternative_route", "delivered_throughput_mbps", "flow_routes",
    "flow_id", "target_link", "applied_step",
]
METRIC_COLUMNS = [
    "scenario", "seed", "strategy", "mean_delay_ms", "p95_delay_ms",
    "packet_delivery_ratio_hop_proxy", "packet_loss_rate_pct_hop_proxy",
    "delivered_throughput_mbps_hop_sum", "congestion_duration_steps",
    "mean_queue_occupancy", "reroutes", "rejected_reroutes", "routing_oscillations",
    "prediction_triggered_actions", "successful_congestion_avoidance_events",
]


class DashboardDataError(RuntimeError):
    """Raised when required saved artifacts are missing or malformed."""


def load_dashboard_data(project_root: str | Path) -> dict[str, Any]:
    """Read Phase 4 and Phase 3.6 artifacts, retaining only required log fields."""
    root = Path(project_root)
    paths = {
        "log": root / "results/phase4/simulation_log.csv",
        "metrics": root / "results/phase4/per_run_metrics.csv",
        "scenario_metrics": root / "results/phase4/scenario_summary.csv",
        "model_metrics": root / "results/phase3_6/test_metrics.csv",
        "scenario_model_metrics": root / "results/phase3_6/scenario_test_metrics.csv",
        "selection": root / "results/phase3_6/frozen_selection.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise DashboardDataError("Required results are missing: " + ", ".join(missing))
    try:
        log = pd.read_csv(paths["log"], usecols=lambda column: column in RUN_COLUMNS,
                          low_memory=False)
        # Event rows are emitted by the same run and carry scenario + seed. The
        # separate routing_decisions.csv predates those identifiers, so it is
        # deliberately not joined to replay data.
        log["step"] = pd.to_numeric(log["step"], errors="coerce")
        log["seed"] = pd.to_numeric(log["seed"], errors="coerce")
        log["link_id"] = log["link_id"].astype("string")
        metrics = pd.read_csv(paths["metrics"])
        scenario_metrics = pd.read_csv(paths["scenario_metrics"])
        model_metrics = pd.read_csv(paths["model_metrics"])
        scenario_model_metrics = pd.read_csv(paths["scenario_model_metrics"])
        selection = json.loads(paths["selection"].read_text())
    except Exception as exc:
        raise DashboardDataError(f"Could not parse saved experiment artifacts: {exc}") from exc
    required = {"scenario", "seed", "strategy", "step", "link_id", "utilization"}
    if not required.issubset(log.columns):
        raise DashboardDataError(f"Simulation log lacks required fields: {sorted(required - set(log.columns))}")
    return {"log": log, "metrics": metrics, "scenario_metrics": scenario_metrics,
            "model_metrics": model_metrics, "scenario_model_metrics": scenario_model_metrics,
            "selection": selection}


def filter_run(log: pd.DataFrame, scenario: str, strategy: str, seed: int) -> pd.DataFrame:
    """Return only one exact scenario/strategy/seed, preserving event rows."""
    subset = log[(log.scenario == scenario) & (log.strategy == strategy) &
                 (pd.to_numeric(log.seed, errors="coerce") == int(seed))].copy()
    if subset.empty:
        raise DashboardDataError(f"No saved replay for {scenario}, {strategy}, seed {seed}.")
    return subset.sort_values(["step", "link_id"], na_position="last", kind="stable").reset_index(drop=True)


def measurement_rows(run: pd.DataFrame) -> pd.DataFrame:
    """Extract directional-link measurement rows, excluding controller event rows."""
    frame = run[run.link_id.notna()].copy()
    if frame.empty:
        raise DashboardDataError("This run contains no directional-link measurements.")
    frame["step"] = pd.to_numeric(frame["step"], errors="coerce").astype(int)
    return frame


def event_rows(run: pd.DataFrame) -> pd.DataFrame:
    """Return contextual controller-event rows belonging to this exact run."""
    if "link_id" not in run:
        return run.iloc[0:0].copy()
    return run[run.link_id.isna() & run.routing_decision.notna()].copy()


def decision_counts(decisions: pd.DataFrame, step: int) -> dict[str, int]:
    """Count scheduled, applied, and rejected controller outcomes through a step."""
    if decisions.empty:
        return {"pending": 0, "applied": 0, "rejected": 0, "issued": 0}
    applied_step = pd.to_numeric(decisions.applied_step, errors="coerce")
    decision_step = pd.to_numeric(decisions.step, errors="coerce")
    has_route = decisions.alternative_route.map(lambda value: bool(parse_json_value(value, expected_type=list)))
    occurred = decision_step.le(int(step))
    return {
        "pending": int((occurred & has_route & applied_step.gt(int(step))).sum()),
        "applied": int((has_route & applied_step.le(int(step))).sum()),
        "rejected": int((occurred & ~has_route).sum()),
        "issued": int(occurred.sum()),
    }


def parse_json_value(value: Any, *, expected_type: type | tuple[type, ...] | None = None) -> Any:
    """Parse a serialized JSON value without turning missing data into defaults."""
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    if isinstance(value, (list, dict)):
        result = value
    else:
        try:
            result = json.loads(str(value))
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
    if expected_type is not None and not isinstance(result, expected_type):
        return None
    return result


def comparison_data(metrics: pd.DataFrame, scenario: str, metric: str) -> pd.DataFrame:
    """Aggregate actual per-run values by strategy; preserve missing metrics."""
    if metric not in metrics.columns:
        raise DashboardDataError(f"Metric is unavailable: {metric}")
    selected = metrics[metrics.scenario == scenario].copy()
    if selected.empty:
        raise DashboardDataError(f"No comparison data for scenario {scenario}.")
    selected[metric] = pd.to_numeric(selected[metric], errors="coerce")
    result = selected.groupby("strategy", as_index=False).agg(
        value_mean=(metric, "mean"), value_std=(metric, "std"), seed_count=(metric, "count"))
    result["strategy_label"] = result.strategy.map(STRATEGY_LABELS)
    return result


def metric_improvements(metrics: pd.DataFrame, scenario: str, strategy: str) -> dict[str, float | None]:
    """Compute seed-paired percentage deltas versus static Dijkstra."""
    subset = metrics[metrics.scenario == scenario]
    static = subset[subset.strategy == "static"].set_index("seed")
    candidate = subset[subset.strategy == strategy].set_index("seed")
    joined = pd.merge(static.reset_index(), candidate.reset_index(), on="seed",
                      suffixes=("_static", "_candidate"), how="inner")
    if joined.empty:
        return {"mean_delay_improvement_pct": None, "delivery_proxy_delta_pp": None,
                "congestion_duration_improvement_pct": None}
    delay_base = pd.to_numeric(joined.mean_delay_ms_static, errors="coerce")
    delay_new = pd.to_numeric(joined.mean_delay_ms_candidate, errors="coerce")
    congestion_base = pd.to_numeric(joined.congestion_duration_steps_static, errors="coerce")
    congestion_new = pd.to_numeric(joined.congestion_duration_steps_candidate, errors="coerce")
    pdr_base = pd.to_numeric(joined.packet_delivery_ratio_hop_proxy_static, errors="coerce")
    pdr_new = pd.to_numeric(joined.packet_delivery_ratio_hop_proxy_candidate, errors="coerce")
    def mean_or_none(values: pd.Series) -> float | None:
        clean = values.replace([np.inf, -np.inf], np.nan).dropna()
        return float(clean.mean()) if len(clean) else None
    return {
        "mean_delay_improvement_pct": mean_or_none((delay_base - delay_new) / delay_base.abs().replace(0, np.nan) * 100),
        "delivery_proxy_delta_pp": mean_or_none((pdr_new - pdr_base) * 100),
        "congestion_duration_improvement_pct": mean_or_none(
            (congestion_base - congestion_new) / congestion_base.abs().replace(0, np.nan) * 100),
    }
