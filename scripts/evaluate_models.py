"""Evaluate trained models on the untouched test split and make figures."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "gnn-congestion-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, precision_score,
                             precision_recall_curve, r2_score, recall_score)
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from congestion_sim.training import load_model_checkpoint, load_split, resolve_device, write_json


THRESHOLD = 0.85


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual_flat, pred_flat = actual.reshape(-1), predicted.reshape(-1)
    high = actual_flat >= THRESHOLD
    return {
        "mae": float(mean_absolute_error(actual_flat, pred_flat)),
        "rmse": float(np.sqrt(mean_squared_error(actual_flat, pred_flat))),
        "r2": float(r2_score(actual_flat, pred_flat)) if actual_flat.size > 1 and np.var(actual_flat) > 0 else None,
        "high_utilization_mae": float(mean_absolute_error(actual_flat[high], pred_flat[high])) if high.any() else None,
        "high_utilization_target_count": int(high.sum()),
    }


def congestion_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    truth, guess = actual.reshape(-1) >= THRESHOLD, predicted.reshape(-1) >= THRESHOLD
    positives = int(truth.sum())
    return {
        "threshold": THRESHOLD,
        "precision": float(precision_score(truth, guess, zero_division=0)),
        "recall": float(recall_score(truth, guess, zero_division=0)),
        "f1": float(f1_score(truth, guess, zero_division=0)),
        "pr_auc_average_precision": float(average_precision_score(truth, predicted.reshape(-1))) if positives else None,
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(truth, guess, labels=[False, True]).ravel().tolist(),
        "positive_targets": positives,
        "target_count": int(truth.size),
    }


def choose_validation_threshold(actual: np.ndarray, predicted_scores: np.ndarray) -> float:
    labels = (actual.reshape(-1) >= THRESHOLD).astype(np.int8)
    scores = predicted_scores.reshape(-1)
    if labels.sum() == 0:
        return 1.0
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if not len(thresholds):
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def thresholded_metrics(actual: np.ndarray, predicted_scores: np.ndarray, threshold: float) -> dict:
    truth = actual.reshape(-1) >= THRESHOLD
    guess = predicted_scores.reshape(-1) >= threshold
    tn, fp, fn, tp = confusion_matrix(truth, guess, labels=[False, True]).ravel()
    return {"decision_threshold": threshold,
            "precision": float(precision_score(truth, guess, zero_division=0)),
            "recall": float(recall_score(truth, guess, zero_division=0)),
            "f1": float(f1_score(truth, guess, zero_division=0)),
            "confusion_matrix_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)],
            "positive_targets": int(truth.sum()), "target_count": int(truth.size)}


def predict(model, x: torch.Tensor, edge: torch.Tensor, device: torch.device,
            batch_size: int) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)
    output = []
    with torch.no_grad():
        for (features,) in loader:
            features = features.to(device)
            if model.__class__.__name__ == "TemporalGNN":
                prediction = model(features, edge)
            else:
                prediction = model(features)
            output.append(prediction.cpu().numpy())
    return np.concatenate(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/generated/phase2"))
    parser.add_argument("--experiment-dir", type=Path, default=Path("results/phase3"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    x_test, y_test, edge_np = load_split(args.data_dir / "test.npz")
    x_val, y_val, val_edge_np = load_split(args.data_dir / "validation.npz")
    if not np.array_equal(edge_np, val_edge_np):
        raise ValueError("Validation and test graph connectivity differs")
    edge = torch.as_tensor(edge_np, dtype=torch.long, device=device)
    normalization = json.loads((args.data_dir / "dataset_summary.json").read_text())["normalization"]
    mean = normalization["mean"][0]
    std = normalization["std"][0]
    # Feature index 0 is normalized utilization, restored here for the
    # persistence and historical-average baselines.
    raw_utilization = x_test[:, :, :, 0].numpy() * std + mean
    val_utilization = x_val[:, :, :, 0].numpy() * std + mean
    predictions: dict[str, np.ndarray] = {
        "persistence": raw_utilization[:, -1, :],
        "moving_average": raw_utilization.mean(axis=1),
    }
    validation_predictions: dict[str, np.ndarray] = {
        "persistence": val_utilization[:, -1, :],
        "moving_average": val_utilization.mean(axis=1),
    }
    checkpoint_dir = args.experiment_dir / "checkpoints"
    checkpoints = [checkpoint_dir / "gnn_standard.pt", checkpoint_dir / "gnn_weighted.pt", checkpoint_dir / "gru.pt"]
    checkpoint_info = {}
    for path in checkpoints:
        model, info = load_model_checkpoint(path, device=device)
        predictions[info["model_name"]] = predict(model, x_test, edge, device, args.batch_size)
        validation_predictions[info["model_name"]] = predict(model, x_val, edge, device, args.batch_size)
        checkpoint_info[info["model_name"]] = {"parameter_count": info["parameter_count"],
                                               "best_epoch": info["best_epoch"],
                                               "best_validation_mae": info["best_validation_mae"]}
    actual = y_test.numpy()
    result: dict = {"measurement_source": "simulated", "test_threshold": THRESHOLD,
                    "device": str(device), "target_count": int(actual.size),
                    "positive_targets": int((actual >= THRESHOLD).sum()),
                    "models": {}, "checkpoint_selection": checkpoint_info}
    for name, prediction in predictions.items():
        result["models"][name] = {"regression": regression_metrics(actual, prediction),
                                  "congestion": congestion_metrics(actual, prediction)}
    validation_actual = y_val.numpy()
    result["validation_tuned_congestion"] = {}
    for name, prediction in predictions.items():
        decision_threshold = choose_validation_threshold(validation_actual, validation_predictions[name])
        result["validation_tuned_congestion"][name] = thresholded_metrics(actual, prediction, decision_threshold)

    data_summary = json.loads((args.data_dir / "dataset_summary.json").read_text())
    test_runs = data_summary["splits"]["test"]["runs"]
    samples_per_run = len(actual) // len(test_runs)
    profiles = np.repeat([run.rsplit("-seed-", 1)[0] for run in test_runs], samples_per_run)
    result["samples_per_test_run"] = samples_per_run
    result["by_traffic_profile"] = {}
    for profile in sorted(set(profiles)):
        mask = profiles == profile
        result["by_traffic_profile"][profile] = {
            name: regression_metrics(actual[mask], prediction[mask]) for name, prediction in predictions.items()
        }
    selected = json.loads((args.experiment_dir / "selected_gnn.json").read_text())["selected_model"]
    result["selected_gnn"] = selected
    result["test_split_not_used_for_selection"] = True
    result["baseline_comparison"] = {
        "moving_average_mae": result["models"]["moving_average"]["regression"]["mae"],
        "selected_gnn_mae": result["models"][selected]["regression"]["mae"],
        "selected_gnn_minus_moving_average_mae": result["models"][selected]["regression"]["mae"] -
                                                  result["models"]["moving_average"]["regression"]["mae"],
        "lower_is_better": True,
    }
    output_dir = args.experiment_dir
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "evaluation_metrics.json", result)
    metric_rows = []
    for name, metrics in result["models"].items():
        metric_rows.append({"model": name, **metrics["regression"], **{
            f"congestion_{key}": value for key, value in metrics["congestion"].items()
            if key not in {"confusion_matrix_tn_fp_fn_tp"}}, **{
            f"validation_tuned_{key}": value for key, value in result["validation_tuned_congestion"][name].items()
            if key not in {"confusion_matrix_tn_fp_fn_tp"}}})
    pd.DataFrame(metric_rows).to_csv(output_dir / "model_comparison.csv", index=False)

    with np.load(args.data_dir / "test.npz", allow_pickle=False) as data:
        link_nodes = data["link_nodes"].astype(str)
    sample_profiles = np.repeat(profiles, actual.shape[1])
    sample_ids = np.repeat(np.arange(len(actual)), actual.shape[1])
    link_ids = np.tile(link_nodes, len(actual))
    base = {"sample_index": sample_ids, "traffic_profile": sample_profiles, "link_id": link_ids,
            "actual_utilization": actual.reshape(-1)}
    pred_frame = pd.DataFrame(base)
    for name, prediction in predictions.items():
        pred_frame[f"{name}_prediction"] = prediction.reshape(-1)
    pred_frame.to_csv(output_dir / "test_predictions.csv", index=False)

    # Plot actual experimental training histories with the same unweighted MSE
    # scale, including for the congestion-weighted training run.
    history = pd.read_csv(output_dir / "training_history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, group in history.groupby("model"):
        axes[0].plot(group.epoch, group.train_mse, label=name)
        axes[1].plot(group.epoch, group.validation_mse, label=name)
    axes[0].set(title="Training MSE", xlabel="Epoch", ylabel="MSE")
    axes[1].set(title="Validation MSE", xlabel="Epoch", ylabel="MSE")
    for axis in axes:
        axis.grid(alpha=.25); axis.legend()
    fig.tight_layout(); fig.savefig(plot_dir / "training_validation_loss.png", dpi=160); plt.close(fig)

    chosen_pred = predictions[selected].reshape(-1)
    indices = np.linspace(0, actual.size - 1, min(5000, actual.size), dtype=int)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual.reshape(-1)[indices], chosen_pred[indices], s=7, alpha=.25)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="Actual utilization", ylabel="Predicted utilization", title=f"Actual vs predicted ({selected})", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(plot_dir / "actual_vs_predicted.png", dpi=160); plt.close(fig)

    names = list(predictions)
    fig, ax = plt.subplots(figsize=(10, 5))
    xpos = np.arange(len(names)); width = .38
    maes = [result["models"][name]["regression"]["mae"] for name in names]
    rmses = [result["models"][name]["regression"]["rmse"] for name in names]
    ax.bar(xpos - width/2, maes, width, label="MAE"); ax.bar(xpos + width/2, rmses, width, label="RMSE")
    ax.set_xticks(xpos, names, rotation=20, ha="right"); ax.set_ylabel("Error")
    ax.set_title("Test regression error comparison"); ax.legend(); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(plot_dir / "regression_comparison.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for metric in ("precision", "recall", "f1"):
        ax.plot(names, [result["validation_tuned_congestion"][name][metric] for name in names], marker="o", label=metric)
    ax.set_ylim(0, 1); ax.set_ylabel("Score"); ax.set_title("Test congestion metrics; decision thresholds tuned on validation")
    ax.tick_params(axis="x", rotation=20); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(plot_dir / "congestion_metrics.png", dpi=160); plt.close(fig)

    chosen_gnn_pred = predictions[selected]
    run_to_plot = "sustained_high-seed-19"
    run_index = test_runs.index(run_to_plot) if run_to_plot in test_runs else len(test_runs) - 1
    start, stop = run_index * samples_per_run, (run_index + 1) * samples_per_run
    run_actual = actual[start:stop]
    chosen_links = np.argsort(run_actual.mean(axis=0))[-3:][::-1]
    fig, axes = plt.subplots(len(chosen_links), 1, figsize=(10, 8), sharex=True)
    for ax, link in zip(np.atleast_1d(axes), chosen_links):
        ax.plot(run_actual[:, link], label="actual", lw=1.7)
        ax.plot(chosen_gnn_pred[start:stop, link], label=selected, lw=1.4)
        ax.axhline(THRESHOLD, color="red", ls="--", lw=1, label="85% threshold")
        ax.set_ylabel(str(link_nodes[link])); ax.set_ylim(0, 1); ax.grid(alpha=.2)
    axes[-1].set_xlabel("Consecutive window sample (target steps advance by one)")
    axes[0].legend(ncol=3, fontsize=8); fig.suptitle(f"Five-step predictions: {run_to_plot}")
    fig.tight_layout(); fig.savefig(plot_dir / "example_time_series.png", dpi=160); plt.close(fig)

    tn, fp, fn, tp = result["models"][selected]["congestion"]["confusion_matrix_tn_fp_fn_tp"]
    matrix = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(matrix, cmap="Blues")
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, f"{value:,}", ha="center", va="center", color="black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["not congested", "congested"],
           yticklabels=["not congested", "congested"], xlabel="Predicted", ylabel="Actual",
           title=f"Confusion matrix ({selected}; threshold 0.85)")
    fig.colorbar(image, ax=ax); fig.tight_layout(); fig.savefig(plot_dir / "confusion_matrix.png", dpi=160); plt.close(fig)

    tn, fp, fn, tp = result["validation_tuned_congestion"][selected]["confusion_matrix_tn_fp_fn_tp"]
    matrix = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(matrix, cmap="Blues")
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, f"{value:,}", ha="center", va="center", color="black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["not congested", "congested"],
           yticklabels=["not congested", "congested"], xlabel="Predicted", ylabel="Actual",
           title=f"Confusion matrix ({selected}; validation-tuned cutoff)")
    fig.colorbar(image, ax=ax); fig.tight_layout(); fig.savefig(plot_dir / "confusion_matrix_validation_tuned.png", dpi=160); plt.close(fig)

    print(json.dumps({"device": str(device), "selected_gnn": selected,
                      "test_models": {n: result["models"][n] for n in predictions},
                      "plots": [str(p) for p in sorted(plot_dir.glob("*.png"))]}, indent=2))


if __name__ == "__main__":
    main()
