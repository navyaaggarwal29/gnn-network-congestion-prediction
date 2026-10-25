# Graph Neural Network-Based Network Congestion Prediction

## Phase 4: Experimental closed-loop routing

Phase 4 adds a reproducible, simulator-only comparison of static Dijkstra,
reactive congestion-aware routing, Phase 3.6 GRU onset routing, and Phase 3.6
GNN onset routing. It is an experiment, not a production routing system. The
controllers run against live measurements from their own controlled runs; no
predictions are replayed from a static-routing trace. The saved Phase 3.6
feature order, training-only normalization, checkpoints, and validation-picked
thresholds are loaded from `results/phase3_6`.

Run a separate one-seed-per-scenario controller pilot first:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/gnn_mpl venv/bin/python scripts/phase4_experiment.py --mode pilot --steps 80
```

Then run the frozen comparison with five fresh, paired seeds per traffic
scenario (seeds 3000–3004):

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/gnn_mpl venv/bin/python scripts/phase4_experiment.py --mode final --steps 80 --seeds 5
```

Outputs are written to ignored generated-artifact directories:

- Pilot: `results/phase4_pilot/`
- Final: `results/phase4/`
- `per_run_metrics.csv`, `scenario_summary.csv`, `paired_vs_static.csv`, and
  `paired_summary_ci.csv` contain measurements and paired comparisons.
- `simulation_log.csv` and `routing_decisions.csv` contain per-step/link logs
  and every controller decision, including rejected and delayed actions.
- `routing_comparison.png` plots results measured by the simulator.
- `experiment_config.json` records seeds and controller settings.

The final experiment uses a two-step action delay, six-step target-link
cooldown, a 0.85 reactive trigger, and a 95% projected-capacity reserve. These
settings are fixed before final evaluation and are not tuned against its seeds.
The GRU and GNN use their saved Phase 3.6 validation thresholds.

### Phase 4 interpretation and limitations

All network measurements are simulated. The simulator has finite per-direction
queues and capacity-bounded packet service, but computes interval-level queue
delay and does not preserve packet identities across every hop. Therefore the
exported packet delivery ratio and packet loss fields are explicitly
hop-level proxies, and delivered throughput is a sum of delivered link-hop
Mbps; none is an end-to-end packet-level measurement. Congestion duration is
the number of intervals in which any directed link reaches 85% utilization.
Successful avoidance counts an applied reroute whose target link stays below
85% utilization during the next two measured intervals; it is an operational
indicator, not a causal counterfactual proof.

Phase 3.6 classifiers were trained under static routing. Applying their
decisions changes the input distribution and can create feedback effects, so
closed-loop generalization is uncertain. Congestion is scenario-dependent,
especially for the sudden unannounced burst, and five evaluation seeds per
scenario provide only modest statistical power. Confidence intervals use a
normal approximation over paired seeds. No real traffic was used.

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

## Phase 3.5: fresh-seed prediction and event diagnostics

The Phase 3.5 study reserves seed groups 1000–1029 independently of the earlier dataset. Run it with the project virtual environment after installing requirements:

```bash
python scripts/phase35_experiment.py --seeds 30 --seed-start 1000 --steps 60 --epochs 12 --out results/phase3_5 --device auto
```

It regenerates deterministic matching scenarios at 1, 3, and 5 step horizons, fits each horizon with training-only normalization, and holds out whole seed groups. Model and threshold selection use only validation. The final test arrays are opened once after the selection is frozen. Outputs include per-horizon measurement data and checkpoints, `validation_comparison.csv`, `final_test_comparison.csv`, event diagnostics, plots, a frozen selection record, and `DIAGNOSTIC_REPORT.md`. A thresholded regression score is not a dedicated onset classifier; review the report's event-onset metrics and GO/NO-GO recommendation before any routing integration. Results are simulator measurements only.

To reproduce the earlier tests and the Phase 3.5 experiment environment:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Phase 3.6: traffic scenarios and onset classification

Phase 3.6 preserves the original traffic profiles and adds five seeded scenarios with fixed flow routes per run. The dedicated onset models use causal utilization, queue, offered-load, variability, and drop-history features. An onset is a first 85% utilization crossing within the prediction horizon, conditioned on being below threshold at decision time. GNN and non-graph GRU classifiers use positive weights derived from training labels only; no artificial balancing is used. Validation selects model and threshold before final test evaluation. Overlapping alerts are merged into episodes so one congestion onset is counted once.

Reproduce on CPU:

```bash
python scripts/train_onset.py --seeds 25 --seed-start 2000 --steps 80 --window 10 --horizon 5 --epochs 16 --hidden 24 --decision-delay 2 --out results/phase3_6 --device cpu
```

Scenarios include a gradual ramp, sustained bottleneck, periodic load, AR(1)-correlated demand, and a seeded unannounced four-step burst. Congestion, drops, delays, and queue occupancy arise from packet arrivals and per-link service. Outputs include split data, checkpoints, scenario and event metrics, PR curves, event timelines, and `DIAGNOSTIC_REPORT.md`. All results are simulated. Verify with `python -m unittest discover -s tests -v`.

## Phase 5: interactive network monitoring dashboard

Install project dependencies from the repository root, then launch Streamlit:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run dashboard/app.py
```

The dashboard reads the saved Phase 4 log and paired strategy metrics, plus
Phase 3.6 test metrics, preprocessing, thresholds, and GRU/GNN checkpoints.
Expected files are under `results/phase4/` and `results/phase3_6/`; these are
generated artifacts and intentionally ignored by Git. If they are absent,
reproduce them with the Phase 3.6 and Phase 4 commands above. The dashboard
reports missing files instead of substituting metrics.

### Dashboard sections

- **Live monitor** replays one exact scenario, strategy, and seed from Phase 4,
  or executes a new 80-step simulation when **Run a new simulation** is chosen.
  Playback, link inspection, and route highlighting use recorded or newly
  generated simulator outputs.
- **Routing comparison** plots saved Phase 4 per-run metrics across four
  strategies and shows scenario-level means and seed-paired changes versus
  static Dijkstra.
- **Model performance** shows held-out Phase 3.6 event precision, recall, F1,
  PR-AUC, false alerts per run, lead time, and scenario breakdown.
- **Demo guide** provides the periodic replay walkthrough.

The topology shows physical links as edges and reports the two directional
measurements separately when a link is inspected. Solid edge colors show
observed utilization; dashed violet marks model risk above its saved threshold
before observed congestion. The event timeline uses events from the same
scenario/strategy/seed. It takes controller outcomes from the contextual event
rows embedded in `simulation_log.csv`; the separate `routing_decisions.csv`
lacks run identifiers and is not joined to replay data.

New simulations use the existing Phase 4 simulator and controller, the
existing causal Phase 3.6 feature pipeline, and cached checkpoints and
validation thresholds. Playback advances through measured intervals; it does
not animate packet movement.

All measurements remain simulated. Delivery and loss are hop-level proxies,
and throughput is a sum of link-hop Mbps because the simulator does not track
packet identity end-to-end. The onset models were trained under static
routing and can experience distribution shift in closed loop. The saved
Phase 3.6 results show the GRU outperforming the GNN on event precision and F1;
neither model has demonstrated broad generalization.

### Troubleshooting

- **Missing results/checkpoints:** run the documented Phase 3.6 training and
  Phase 4 final experiment commands, then restart Streamlit.
- **Port already in use:** use `python -m streamlit run dashboard/app.py --server.port 8502`.
- **Slow first load:** the Phase 4 log is read once and cached for the session.
- **Live model run fails:** verify both onset checkpoints, `preprocessing.json`,
  and `frozen_selection.json` under `results/phase3_6/`.
- **Dependency installation fails:** use Python 3.11 or newer and install the
  full `requirements.txt` in the active project environment.
