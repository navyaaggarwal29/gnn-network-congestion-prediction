"""Train GRU and temporal GNN models using train and validation splits only."""

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import torch
import torch_geometric

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from congestion_sim.training import load_split, seed_everything, train_one, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/generated/phase2"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase3"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1 or args.epochs < 1 or args.hidden_dim < 1 or args.threads < 1:
        parser.error("batch-size, epochs, hidden-dim, and threads must be positive")
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    x_train, y_train, edge_index = load_split(args.data_dir / "train.npz")
    x_val, y_val, val_edge = load_split(args.data_dir / "validation.npz")
    if not np.array_equal(edge_index, val_edge):
        raise ValueError("Training and validation graph connectivity differs")
    summary = json.loads((args.data_dir / "dataset_summary.json").read_text())
    normalization = summary["normalization"]
    positives = int((y_train.numpy() >= .85).sum())
    negatives = y_train.numel() - positives
    class_weight = min(float(np.sqrt(negatives / positives)), 20.0) if positives else 1.0
    config = {"seed": args.seed, "batch_size": args.batch_size, "max_epochs": args.epochs,
              "learning_rate": args.learning_rate, "hidden_dim": args.hidden_dim,
              "dropout": args.dropout, "patience": args.patience, "device_requested": args.device,
              "threads": args.threads, "window": summary["window"],
              "torch_version": torch.__version__, "torch_geometric_version": torch_geometric.__version__,
              "mps_available": bool(torch.backends.mps.is_available()),
              "cuda_available": bool(torch.cuda.is_available()),
              "prediction_horizon": summary["prediction_horizon"],
              "feature_count": x_train.shape[-1], "node_count": x_train.shape[-2],
              "graph_edge_count": int(edge_index.shape[1]), "training_positive_targets": positives,
              "training_negative_targets": negatives,
              "weighted_loss_positive_weight": class_weight,
              "weighted_loss_formula": "min(sqrt(negative_count / positive_count), 20), fit on training labels only",
              "model_selection": "lowest validation MAE; test split not loaded during training"}
    write_json(args.output_dir / "config.json", config)
    checkpoint_dir = args.output_dir / "checkpoints"
    histories = []
    results = []
    for name, weight in (("gnn_standard", None), ("gnn_weighted", class_weight), ("gru", None)):
        print(f"Training {name} on {args.device} (batch={args.batch_size}, max_epochs={args.epochs})", flush=True)
        result = train_one(name, x_train, y_train, x_val, y_val, edge_index, checkpoint_dir,
                           seed=args.seed, device_name=args.device, batch_size=args.batch_size,
                           epochs=args.epochs, learning_rate=args.learning_rate,
                           hidden_dim=args.hidden_dim, dropout=args.dropout,
                           patience=args.patience, positive_weight=weight,
                           prediction_horizon=summary["prediction_horizon"],
                           feature_normalization=normalization)
        print(f"{name}: best epoch {result['best_epoch']}, validation MAE {result['best_validation_mae']:.6f}, "
              f"parameters {result['parameter_count']}, device {result['device']}", flush=True)
        histories.extend(result.pop("history"))
        results.append(result)
    gnn_results = [r for r in results if r["model"].startswith("gnn")]
    selected = min(gnn_results, key=lambda r: r["best_validation_mae"])
    shutil.copyfile(selected["checkpoint"], checkpoint_dir / "best_gnn.pt")
    write_json(args.output_dir / "selected_gnn.json", {"selected_model": selected["model"],
              "selection_metric": "validation_mae", "best_validation_mae": selected["best_validation_mae"],
              "checkpoint": str(checkpoint_dir / "best_gnn.pt")})
    pd.DataFrame(histories).to_csv(args.output_dir / "training_history.csv", index=False)
    training_summary = {"device": results[0]["device"], "model_results": results,
                        "selected_gnn": selected["model"],
                        "best_validation_performance": {"mae": selected["best_validation_mae"],
                                                        "epoch": selected["best_epoch"]}}
    write_json(args.output_dir / "training_summary.json", training_summary)
    print(json.dumps(training_summary, indent=2))


if __name__ == "__main__":
    main()
