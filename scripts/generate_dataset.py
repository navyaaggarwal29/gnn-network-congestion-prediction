"""Generate leakage-safe temporal link samples and graph artifacts."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from congestion_sim import build_default_topology
from congestion_sim.dataset import evaluate_baselines, generate_runs, prepare_splits, split_run_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0, help="First independent seed ID")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated/phase2"))
    args = parser.parse_args()
    topology = build_default_topology()
    dataset, measurements = generate_runs(topology, list(range(args.seed_start, args.seed_start + args.seeds)), args.steps, args.window, args.horizon)
    split = split_run_ids(list(dataset.runs))
    arrays, normalization = prepare_splits(dataset, split)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    measurements.to_csv(args.output_dir / "measurements.csv", index=False)
    link_nodes = np.asarray([f"{u}->{v}" for u, v in dataset.link_nodes])
    for name, values in arrays.items():
        np.savez_compressed(args.output_dir / f"{name}.npz", x=values["x"], y=values["y"],
                            edge_index=dataset.edge_index, link_nodes=link_nodes)
    test_predictions = evaluate_baselines(dataset, split["test"])
    baseline_rows = []
    for method in ("persistence", "moving_average"):
        baseline_rows.append({"split": "test", "method": method,
                              **test_predictions[method], "threshold": .85,
                              "congestion_positive_count": test_predictions["test_positive_count"],
                              "label_count": test_predictions["test_label_count"]})
    import pandas as pd
    pd.DataFrame(baseline_rows).to_csv(args.output_dir / "baseline_metrics.csv", index=False)
    split_summary = {name: {"runs": ids, "run_count": len(ids), "sample_count": len(arrays[name]["y"]),
                            "positive_labels": int((arrays[name]["y"] >= .85).sum()),
                            "label_count": int(arrays[name]["y"].size)} for name, ids in split.items()}
    result = {"measurement_source": "simulated", "profile_names": ["normal", "moderate", "burst", "sustained_high"],
              "independent_seed_count": args.seeds, "independent_run_count": len(dataset.runs),
              "steps_per_run": args.steps, "window": args.window, "prediction_horizon": args.horizon,
              "link_node_count": len(dataset.link_nodes), "graph_edge_count": int(dataset.edge_index.shape[1]),
              "splits": split_summary, "normalization": normalization, "test_baselines": test_predictions,
              "test_congestion_positive_count": split_summary["test"]["positive_labels"]}
    (args.output_dir / "dataset_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
