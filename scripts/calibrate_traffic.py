"""Run multi-seed traffic calibration experiments and export actual summaries."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from congestion_sim import NetworkSimulator, TrafficProfile, build_default_topology
from congestion_sim.dataset import TRAFFIC_PROFILES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase1_5"))
    args = parser.parse_args()
    topology = build_default_topology()
    frames = []
    for profile_index, profile in enumerate(TRAFFIC_PROFILES):
        for seed in range(args.seeds):
            run_seed = seed * 1009 + profile_index
            frame = NetworkSimulator(topology, seed=run_seed).run(args.steps, TrafficProfile(kind=profile))
            frame["traffic_profile"] = profile
            frame["seed"] = seed
            frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    active = data[data.offered_demand_mbps > 0]
    summary = {"simulation_runs": args.seeds * len(TRAFFIC_PROFILES), "seeds_per_profile": args.seeds,
               "steps_per_run": args.steps, "measurement_source": "simulated",
               "profiles": {}}
    for profile, group in data.groupby("traffic_profile", sort=False):
        active_profile = group[group.offered_demand_mbps > 0]
        summary["profiles"][profile] = {
            "all_link_utilization": {"min": float(group.utilization.min()), "mean": float(group.utilization.mean()),
                                     "p95": float(group.utilization.quantile(.95)), "max": float(group.utilization.max())},
            "active_link_utilization": {"min": float(active_profile.utilization.min()), "mean": float(active_profile.utilization.mean()),
                                        "p95": float(active_profile.utilization.quantile(.95)), "max": float(active_profile.utilization.max())},
            "queue_occupancy": {"mean": float(group.queue_occupancy.mean()), "p95": float(group.queue_occupancy.quantile(.95)),
                                "max": float(group.queue_occupancy.max())},
            "active_queue_occupancy": {"mean": float(active_profile.queue_occupancy.mean()),
                                       "p95": float(active_profile.queue_occupancy.quantile(.95)),
                                       "max": float(active_profile.queue_occupancy.max())},
            "packet_loss_pct": {"mean": float(group.packet_loss_pct.mean()), "p95": float(group.packet_loss_pct.quantile(.95)),
                                "max": float(group.packet_loss_pct.max())},
            "active_measurements": int(len(active_profile)),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output_dir / "calibration_measurements.csv", index=False)
    (args.output_dir / "calibration_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
