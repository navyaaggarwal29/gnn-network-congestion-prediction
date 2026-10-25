"""Run Phase 4 pilot or paired closed-loop routing experiments.

All measurements and traffic are simulator-generated. Final seeds are fresh
and controller parameters are fixed before those runs.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from congestion_sim.onset import SCENARIOS
from congestion_sim.routing import OnsetPredictor, RoutingController
from congestion_sim.simulator import NetworkSimulator, ScenarioTrafficProfile
from congestion_sim.topology import build_default_topology

STRATEGIES = ("static", "reactive", "gru", "gnn")

def run_one(scenario: str, seed: int, strategy: str, artifact_dir: Path,
            steps: int = 80) -> tuple[pd.DataFrame, list[dict], dict]:
    topology = build_default_topology()
    simulator = NetworkSimulator(topology, seed=seed * 1009 + SCENARIOS.index(scenario))
    simulator.set_scenario_horizon(steps)
    profile = ScenarioTrafficProfile(kind=scenario)
    predictor = OnsetPredictor(f"{strategy}_onset", artifact_dir, topology) if strategy in {"gru", "gnn"} else None
    controller = RoutingController(topology, strategy, predictor, decision_delay=2, cooldown=6)
    measurements, logs = [], []
    for step in range(steps):
        controller.apply_due(simulator, step)
        frame = simulator.step(profile)
        measurements.append(frame)
        history = pd.concat(measurements, ignore_index=True)
        scores = predictor.predict(history) if predictor else None
        issued = controller.observe(simulator, frame, history, step, scores)
        score_map = ({link: float(scores[i]) for i, link in enumerate(predictor.link_ids)}
                     if scores is not None else {})
        route_summary = json.dumps([{"flow_id": f["flow_id"], "route": f["route"]}
                                    for f in simulator.flow_state], separators=(",", ":"))
        decision_summary = json.dumps([d.__dict__ for d in issued], separators=(",", ":"))
        flows = simulator.flow_state
        routes = {f"{f['source_router']}->{f['destination_router']}": f["route"] for f in flows}
        for row in frame.to_dict("records"):
            logs.append({**row, "scenario": scenario, "seed": seed, "strategy": strategy,
                         "source_destination": None, "current_route": route_summary,
                         "observed_link_utilization": row["utilization"],
                         "predicted_congestion_probability": score_map.get(row["link_id"]),
                         "routing_decision": decision_summary,
                         "alternative_route": json.dumps([d.alternative_route for d in issued]),
                         "queue_occupancy": row["queue_occupancy"], "packet_loss": row["packet_loss_pct"],
                         "delivered_throughput_mbps": row["utilization"] * row["bandwidth_mbps"],
                         "flow_routes": json.dumps(routes, separators=(",", ":"))})
    data = pd.concat(measurements, ignore_index=True)
    total_offered = float(data.packets_offered.sum())
    dropped = float(data.packets_dropped.sum())
    metrics = {
        "scenario": scenario, "seed": seed, "strategy": strategy,
        "mean_delay_ms": float(data.delay_ms.mean()), "p95_delay_ms": float(data.delay_ms.quantile(.95)),
        # This is a hop-level proxy because the simulator does not model packet IDs
        # across successive routers as a discrete-event end-to-end packet system.
        "packet_delivery_ratio_hop_proxy": max(0.0, 1.0 - dropped / max(1.0, total_offered)),
        "packet_loss_rate_pct_hop_proxy": 100.0 * dropped / max(1.0, total_offered),
        "delivered_throughput_mbps_hop_sum": float((data.utilization * data.bandwidth_mbps).sum() / steps),
        "congestion_duration_steps": int(data.groupby("step").utilization.max().ge(.85).sum()),
        "mean_queue_occupancy": float(data.queue_occupancy.mean()),
        "reroutes": sum(d.applied_step is not None for d in controller.decisions),
        "scheduled_reroutes": sum(d.alternative_route is not None for d in controller.decisions),
        "rejected_reroutes": sum(d.alternative_route is None for d in controller.decisions),
        "routing_oscillations": sum(1 for a,b in zip(controller.decisions, controller.decisions[1:])
                                    if a.flow_id == b.flow_id and a.alternative_route == b.old_route),
        "prediction_triggered_actions": sum(d.strategy in {"gru", "gnn"} for d in controller.decisions),
        "successful_congestion_avoidance_events": 0,
    }
    # Count an action successful when its target link is below threshold after
    # application and remains so for the next two observed steps.
    for decision in controller.decisions:
        if decision.applied_step is None or decision.target_link is None:
            continue
        target = data[data.link_id == decision.target_link].set_index("step").utilization
        if all(target.get(s, 1.0) < .85 for s in range(decision.applied_step, min(steps, decision.applied_step + 2))):
            metrics["successful_congestion_avoidance_events"] += 1
    events = [d.__dict__ for d in controller.decisions]
    # Include delayed application records, including rejected / retained decisions.
    for event in events:
        logs.append({"step": event["step"], "time_seconds": event["step"], "scenario": scenario,
                     "seed": seed, "strategy": strategy, "source_destination": event["source"] and f"{event['source']}->{event['destination']}",
                     "current_route": json.dumps(event["old_route"]), "observed_link_utilization": None,
                     "predicted_congestion_probability": event["probability"], "routing_decision": event["reason"],
                     "alternative_route": json.dumps(event["alternative_route"]),
                     "queue_occupancy": None, "packet_loss": None, "delivered_throughput_mbps": None,
                     "flow_id": event["flow_id"], "target_link": event["target_link"], "applied_step": event["applied_step"]})
    return pd.DataFrame(logs), events, metrics

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "final"], required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    out = args.out or ROOT / "results" / ("phase4_pilot" if args.mode == "pilot" else "phase4")
    out.mkdir(parents=True, exist_ok=True)
    seeds = [2900] if args.mode == "pilot" else list(range(3000, 3000 + args.seeds))
    logs, events, metrics = [], [], []
    for scenario in SCENARIOS:
        for seed in seeds:
            for strategy in STRATEGIES:
                print(f"{args.mode}: {scenario}, seed={seed}, strategy={strategy}", flush=True)
                log, event, metric = run_one(scenario, seed, strategy, ROOT / "results/phase3_6", args.steps)
                logs.append(log); events.extend(event); metrics.append(metric)
    log_frame = pd.concat(logs, ignore_index=True)
    metric_frame = pd.DataFrame(metrics)
    log_frame.to_csv(out / "simulation_log.csv", index=False)
    pd.DataFrame(events).to_csv(out / "routing_decisions.csv", index=False)
    metric_frame.to_csv(out / "per_run_metrics.csv", index=False)
    summary = metric_frame.groupby(["scenario", "strategy"]).agg(
        **{f"{c}_mean": (c, "mean") for c in metric_frame.select_dtypes("number").columns if c not in {"seed"}},
        **{f"{c}_std": (c, "std") for c in metric_frame.select_dtypes("number").columns if c not in {"seed"}}).reset_index()
    summary.to_csv(out / "scenario_summary.csv", index=False)
    static = metric_frame[metric_frame.strategy == "static"].set_index(["scenario", "seed"])
    paired = []
    higher_better = {"packet_delivery_ratio_hop_proxy", "delivered_throughput_mbps_hop_sum"}
    for _, row in metric_frame[metric_frame.strategy != "static"].iterrows():
        base = static.loc[(row.scenario, row.seed)]
        item = {"scenario": row.scenario, "seed": row.seed, "strategy": row.strategy}
        for col in [c for c in metric_frame.select_dtypes("number").columns if c not in {"seed"}]:
            item[f"{col}_change_pct"] = ((row[col] - base[col]) / abs(base[col]) * 100
                                         if base[col] != 0 else np.nan)
            if col in higher_better:
                item[f"{col}_improvement_pct"] = item[f"{col}_change_pct"]
            else:
                item[f"{col}_improvement_pct"] = -item[f"{col}_change_pct"]
        paired.append(item)
    pd.DataFrame(paired).to_csv(out / "paired_vs_static.csv", index=False)
    paired_frame = pd.DataFrame(paired)
    paired_summary = paired_frame.groupby(["scenario", "strategy"]).agg(
        **{f"{c}_mean": (c, "mean") for c in paired_frame.select_dtypes("number").columns
           if c.endswith("_improvement_pct")},
        **{f"{c}_std": (c, "std") for c in paired_frame.select_dtypes("number").columns
           if c.endswith("_improvement_pct")}).reset_index()
    for col in [c for c in paired_summary if c.endswith("_mean")]:
        raw = col[:-5]
        std = f"{raw}_std"
        if std in paired_summary:
            # Normal-approximation 95% paired CI; clearly labeled and interpreted
            # cautiously because final evaluation has five independent seeds.
            paired_summary[f"{raw}_ci95_low"] = paired_summary[col] - 1.96 * paired_summary[std] / np.sqrt(len(seeds))
            paired_summary[f"{raw}_ci95_high"] = paired_summary[col] + 1.96 * paired_summary[std] / np.sqrt(len(seeds))
    paired_summary.to_csv(out / "paired_summary_ci.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for metric, ax in zip(["mean_delay_ms", "packet_delivery_ratio_hop_proxy"], axes):
        metric_frame.groupby("strategy")[metric].mean().reindex(STRATEGIES).plot.bar(ax=ax)
        ax.set_title(metric.replace("_", " ")); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(out / "routing_comparison.png", dpi=160); plt.close(fig)
    metadata = {"source": "simulated", "mode": args.mode, "seeds": seeds, "scenarios": SCENARIOS,
                "strategies": STRATEGIES, "steps": args.steps, "decision_delay": 2,
                "cooldown": 6, "trigger": .85, "fresh_seed_range": [min(seeds), max(seeds)],
                "checkpoint_dir": "results/phase3_6/checkpoints", "pdr_limitation": "hop-level proxy; no packet identity across the end-to-end route"}
    (out / "experiment_config.json").write_text(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False))

if __name__ == "__main__": main()
