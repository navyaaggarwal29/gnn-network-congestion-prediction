"""Run a reproducible Phase 1 simulation and export simulated measurements."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from congestion_sim import NetworkSimulator, TrafficProfile, build_default_topology


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--profile", choices=("normal", "moderate", "burst", "sustained_high"), default="burst")
    parser.add_argument("--output", type=Path, default=Path("data/phase1_measurements.csv"))
    args = parser.parse_args()
    topology = build_default_topology()
    simulator = NetworkSimulator(topology, seed=args.seed)
    measurements = simulator.run(args.steps, TrafficProfile(kind=args.profile))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    measurements.to_csv(args.output, index=False)
    print(f"Routers: {topology.graph.number_of_nodes()}; links: {topology.graph.number_of_edges()}")
    print(f"Simulated intervals: {args.steps}; rows: {len(measurements)}; seed: {args.seed}")
    print(f"Mean utilization: {measurements.utilization.mean():.3f}; max: {measurements.utilization.max():.3f}")
    print(f"Measurements source: {measurements.measurement_source.unique().tolist()}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
