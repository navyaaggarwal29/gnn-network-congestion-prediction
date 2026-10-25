"""Causal routing strategies and safe rerouting controller for Phase 4."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import json

import networkx as nx
import numpy as np
import pandas as pd
import torch

from .dataset import directed_link_edge_index, directed_link_nodes
from .models import TemporalOnsetGNN, TemporalOnsetGRU
from .onset import ONSET_FEATURES, add_causal_temporal_features
from .simulator import NetworkSimulator
from .topology import NetworkTopology


@dataclass
class Decision:
    step: int
    strategy: str
    flow_id: int | None
    source: str | None
    destination: str | None
    old_route: list[str] | None
    alternative_route: list[str] | None
    reason: str
    target_link: str | None = None
    probability: float | None = None
    applied_step: int | None = None


class OnsetPredictor:
    """Load Phase 3.6 weights and reproduce its saved feature normalization."""

    def __init__(self, model_name: str, artifact_dir: str | Path, topology: NetworkTopology,
                 device: str = "cpu"):
        if model_name not in {"gru_onset", "gnn_onset"}:
            raise ValueError("model_name must be gru_onset or gnn_onset")
        self.model_name, self.device = model_name, torch.device(device)
        artifact_dir = Path(artifact_dir)
        self.prep = json.loads((artifact_dir / "preprocessing.json").read_text())
        self.selection = json.loads((artifact_dir / "frozen_selection.json").read_text())
        if self.prep["feature_columns"] != ONSET_FEATURES:
            raise ValueError("Saved feature order does not match Phase 3.6")
        if self.prep["normalization_fit"] != "training_windows_only":
            raise ValueError("Checkpoint lacks training-only normalization metadata")
        self.threshold = float(self.selection["threshold_by_model"][model_name])
        self.links = directed_link_nodes(topology)
        self.link_ids = [f"{u}->{v}" for u, v in self.links]
        self.edge = torch.as_tensor(directed_link_edge_index(topology), dtype=torch.long, device=self.device)
        checkpoint = artifact_dir / "checkpoints" / f"{model_name}.pt"
        self.model = (TemporalOnsetGNN(len(ONSET_FEATURES), 32) if model_name == "gnn_onset"
                      else TemporalOnsetGRU(len(ONSET_FEATURES), 32)).to(self.device)
        payload = torch.load(checkpoint, map_location=self.device, weights_only=True)
        state = payload["state_dict"] if "state_dict" in payload else payload
        config = payload.get("model_config", {}) if isinstance(payload, dict) else {}
        hidden_dim = int(config.get("hidden_dim", 32))
        self.model = (TemporalOnsetGNN(len(ONSET_FEATURES), hidden_dim) if model_name == "gnn_onset"
                      else TemporalOnsetGRU(len(ONSET_FEATURES), hidden_dim)).to(self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        self.mean = np.asarray(self.prep["mean"], dtype=np.float32)
        self.std = np.asarray(self.prep["std"], dtype=np.float32)
        self.window = int(self.prep["window"])

    @torch.inference_mode()
    def predict(self, history: pd.DataFrame) -> np.ndarray | None:
        """Predict per-link onset probabilities from measurements through now."""
        if history["step"].nunique() < self.window:
            return None
        frame = add_causal_temporal_features(history)
        frame["link_id"] = pd.Categorical(frame["link_id"], categories=self.link_ids, ordered=True)
        pivot = frame.pivot(index="step", columns="link_id", values=ONSET_FEATURES).sort_index()
        if pivot.shape[0] < self.window or pivot.isna().any().any():
            return None
        raw = pivot[ONSET_FEATURES].to_numpy(dtype=np.float32).reshape(-1, len(ONSET_FEATURES), len(self.links)).transpose(0, 2, 1)
        x = ((raw[-self.window:] - self.mean) / self.std).astype(np.float32)[None, ...]
        tensor = torch.from_numpy(x).to(self.device)
        logits = self.model(tensor, self.edge) if self.model_name == "gnn_onset" else self.model(tensor)
        return torch.sigmoid(logits)[0].cpu().numpy()


class RoutingController:
    """Decision-delay, capacity-aware rerouting with cooldown and hysteresis."""
    def __init__(self, topology: NetworkTopology, strategy: str, predictor: OnsetPredictor | None = None,
                 decision_delay: int = 2, cooldown: int = 6, trigger: float = .85,
                 reserve_fraction: float = .95):
        self.topology, self.strategy, self.predictor = topology, strategy, predictor
        self.decision_delay, self.cooldown, self.trigger = decision_delay, cooldown, trigger
        self.reserve_fraction = reserve_fraction
        self.pending: list[tuple[int, int, list[str], Decision]] = []
        self.last_action: dict[str, int] = {}
        self.decisions: list[Decision] = []
        self._alerts: dict[str, int] = {}

    def apply_due(self, simulator: NetworkSimulator, step: int) -> None:
        due, keep = [], []
        for item in self.pending:
            (due if item[0] <= step else keep).append(item)
        self.pending = keep
        for _, flow_id, route, decision in due:
            try:
                simulator.reroute_scenario_flow(flow_id, route)
                decision.applied_step = step
                self.last_action[decision.target_link or ""] = step
            except (ValueError, IndexError) as exc:
                decision.reason += f"; application rejected: {exc}"

    def observe(self, simulator: NetworkSimulator, frame: pd.DataFrame, history: pd.DataFrame,
                step: int, predicted: np.ndarray | None = None) -> list[Decision]:
        if self.strategy == "static":
            return []
        candidates: list[tuple[str, float, str]] = []
        if self.strategy == "reactive":
            hot = frame[(frame.utilization >= self.trigger) | (frame.queue_occupancy >= self.trigger)]
            candidates = [(str(row.link_id), float(max(row.utilization, row.queue_occupancy)), "observed congestion")
                          for row in hot.itertuples()]
        elif self.strategy in {"gru", "gnn"} and predicted is not None:
            model = self.predictor
            assert model is not None
            candidates = [(model.link_ids[i], float(score), "predicted onset risk")
                          for i, score in enumerate(predicted) if score >= model.threshold]
        if not candidates:
            return []
        # Hysteresis: only one alert/action per target link until cooldown expires.
        candidates.sort(key=lambda item: item[1], reverse=True)
        emitted: list[Decision] = []
        for link_id, score, reason in candidates:
            if step - self._alerts.get(link_id, -10**9) < self.cooldown:
                continue
            self._alerts[link_id] = step
            a, b = link_id.split("->")
            affected = [f for f in simulator.flow_state if any(
                {x, y} == {a, b} for x, y in zip(f["route"], f["route"][1:]))]
            pending_flows = {item[1] for item in self.pending}
            affected = [f for f in affected if int(f["flow_id"]) not in pending_flows]
            chosen = None
            reject_reason = "no affected persistent flow"
            load = {(row.source_router, row.destination_router): float(row.offered_demand_mbps)
                    for row in frame.itertuples()}
            # Use current per-direction load; subtract a moved flow's demand from
            # old hops, add it to candidate hops, and require every hop to retain reserve.
            for flow in affected:
                graph = self.topology.graph.copy()
                graph.remove_edge(a, b)
                src, dst = str(flow["source_router"]), str(flow["destination_router"])
                try:
                    alternatives = sorted(nx.all_simple_paths(graph, src, dst, cutoff=len(self.topology.routers)-1),
                                          key=lambda p: sum(graph[u][v]["propagation_delay_ms"] for u,v in zip(p,p[1:])))
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    alternatives = []
                rate = float(flow["offered_demand_mbps"])
                for route in alternatives:
                    projected = dict(load)
                    for x,y in zip(flow["route"], flow["route"][1:]):
                        projected[(x,y)] = max(0., projected.get((x,y), 0.) - rate)
                    feasible = True
                    for x,y in zip(route, route[1:]):
                        projected[(x,y)] = projected.get((x,y), 0.) + rate
                        cap = float(self.topology.graph[x][y]["bandwidth_mbps"])
                        if projected[(x,y)] > cap * self.reserve_fraction:
                            feasible = False; reject_reason = f"alternate {route} exceeds reserve on {x}->{y}"; break
                    if feasible:
                        chosen = (flow, route); break
                if chosen: break
            if chosen:
                flow, route = chosen
                decision = Decision(step, self.strategy, int(flow["flow_id"]), str(flow["source_router"]),
                                    str(flow["destination_router"]), list(flow["route"]), list(route),
                                    reason + f"; scheduled after {self.decision_delay} steps", link_id, score)
                self.pending.append((step + self.decision_delay, int(flow["flow_id"]), list(route), decision))
            else:
                decision = Decision(step, self.strategy, None, None, None, None, None,
                                    reason + "; retained route: " + reject_reason, link_id, score)
            self.decisions.append(decision); emitted.append(decision)
            # At most one reroute decision per interval.
            break
        return emitted
