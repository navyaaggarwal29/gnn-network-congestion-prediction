# Graph Neural Network-Based Network Congestion Prediction

Academic project for studying **simulated** network congestion and proactive traffic management. The completed phases provide a deterministic NetworkX topology, full-duplex directional queues, traffic calibration, temporal prediction datasets, and graph structures for a later GNN. All measurements are simulated; they do not represent a real network.

## Setup

Python 3.11 or newer is required. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Phase 1: simulator

Run a 60-step burst simulation (seed 7):

```bash
python scripts/simulate.py --steps 60 --seed 7 --profile burst --output data/phase1_measurements.csv
```

The topology has 10 routers and 15 full-duplex links. Dijkstra routing minimizes propagation delay. Each direction has a finite 252,000-byte queue, sized to an integer number of 1,200-byte packets. Traffic is quantized to whole packets and service occurs during each one-second interval. Actual transmitted utilization cannot exceed link service capacity; excess packets wait in the queue or are dropped when it is full.

Available profiles are `normal`, `moderate`, `burst`, and `sustained_high`. Their default offered rates are 40 Mbps, 80 Mbps, 40 Mbps with a 4x burst from step 20 for 8 steps, and 160 Mbps respectively. `--steps`, `--seed`, and `--output` are configurable.

## Phase 1.5: traffic calibration

Run the multi-seed calibration experiment and write raw measurements and measured distributions:

```bash
python scripts/calibrate_traffic.py --seeds 10 --steps 80 --output-dir results/phase1_5
```

Outputs: `calibration_measurements.csv` and `calibration_summary.json`. Summaries distinguish active-link utilization from all-link utilization and include queue occupancy and packet loss per profile.

## Phase 2: dataset generation

Generate independent runs using seeds 0–19, 60 steps per run, a 10-step window, and a 5-step horizon:

```bash
python scripts/generate_dataset.py --seeds 20 --steps 60 --window 10 --horizon 5 --output-dir data/generated/phase2
```

Outputs include raw timestamped `measurements.csv`; compressed `train.npz`, `validation.npz`, and `test.npz`; `baseline_metrics.csv`; and `dataset_summary.json`. Each archive contains `x` (`samples × window × features × directed-link nodes`), future utilization labels `y`, `edge_index`, and stable `link_nodes`. `edge_index` uses PyTorch Geometric's `[2, graph_edges]` convention and can be converted directly to a `torch.long` tensor. Graph edges connect legal consecutive directed links at routers and exclude immediate reversals over the same physical link.

All profiles belonging to a seed remain in one split. The default split uses 14 training, 3 validation, and 3 test seeds (70/15/15%). Feature means and standard deviations are fit from training windows only. Baselines are persistence and a historical moving average; summaries report test MAE/RMSE and 85% congestion-label counts.

## Tests

Run the automated checks:

```bash
python -m unittest discover -s tests -v
```

The tests cover topology and routing, packet/queue bounds, deterministic generation, direction-aware graph connectivity, temporal alignment, seed isolation, and training-only normalization.

## Phase 3: temporal GNN training and evaluation

Install the expanded requirements, then train the non-graph GRU, standard temporal GNN, and congestion-weighted temporal GNN. Training and early stopping read only the training and validation splits; the test split is loaded by the separate evaluation command after model selection.

```bash
python -m pip install -r requirements.txt
python scripts/train_models.py --data-dir data/generated/phase2 --output-dir results/phase3 --batch-size 32 --epochs 60 --learning-rate 0.001 --hidden-dim 64 --dropout 0.2 --patience 8 --seed 42 --device auto
python scripts/evaluate_models.py --data-dir data/generated/phase2 --experiment-dir results/phase3 --device auto
```

`--device auto` selects Apple Silicon MPS when PyTorch reports it available, then CUDA, then CPU. Use `--device cpu` to force CPU. The model uses two directed PyG `GCNConv` layers shared across the ten input times, a per-link GRU, and a sigmoid utilization output. GCN messages follow the stored source-to-target edges; the model does not symmetrize the topology. The regression output is bounded to [0, 1].

Training compares standard MSE with a congestion-weighted MSE. The positive class weight is derived from training labels only as `min(sqrt(negative_count / positive_count), 20)`. Validation MAE selects the best GNN variant. Regression congestion metrics use the fixed 0.85 prediction and target threshold. A separate classification cutoff is selected by validation F1 and reported on test; test labels are never used to select a cutoff. Average precision is reported as PR-AUC.

Artifacts are written under `results/phase3/`: `checkpoints/` (ignored by Git), configuration and training histories, `evaluation_metrics.json`, `model_comparison.csv`, `test_predictions.csv`, and seven PNG evaluation plots. These are actual results from the saved simulated dataset; training may be rerun deterministically with the same command and seed.

Run all existing and Phase 3 checks with:

```bash
python -m unittest discover -s tests -v
```

Phase 3 ends with model training and evaluation. Proactive routing and the dashboard are not part of this phase.
