"""Discrete-time packet traffic simulator with finite per-direction queues."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Literal

import numpy as np
import pandas as pd

from .topology import NetworkTopology


@dataclass(frozen=True)
class TrafficProfile:
    """Traffic generation settings. Rates are aggregate offered Mbps per tick."""

    kind: Literal["normal", "moderate", "burst", "sustained_high"] = "normal"
    mean_demand_mbps: float = 40.0
    burst_start: int = 20
    burst_duration: int = 8
    burst_multiplier: float = 4.0

    def __post_init__(self) -> None:
        if self.kind not in {"normal", "moderate", "burst", "sustained_high"}:
            raise ValueError(f"Unsupported traffic profile: {self.kind}")
        if self.mean_demand_mbps <= 0 or self.burst_duration < 1 or self.burst_multiplier < 1:
            raise ValueError("Traffic rates and burst parameters must be positive")

    def demand(self, step: int) -> float:
        if self.kind == "moderate":
            return self.mean_demand_mbps * 2.0
        if self.kind == "sustained_high":
            return self.mean_demand_mbps * 4
        if self.kind == "burst" and self.burst_start <= step < self.burst_start + self.burst_duration:
            return self.mean_demand_mbps * self.burst_multiplier
        return self.mean_demand_mbps


@dataclass(frozen=True)
class ScenarioTrafficProfile:
    """Temporally structured aggregate demand used by Phase 3.6 experiments.

    Rates are offered Mbps over one simulator interval. Scenario flow paths are
    fixed for a run; queues, utilization, delay, and loss still emerge from the
    ordinary packet service and finite-buffer mechanics.
    """

    kind: Literal["gradual_ramp", "sustained_bottleneck", "periodic",
                  "correlated", "sudden_burst"]
    base_demand_mbps: float = 40.0
    peak_demand_mbps: float = 180.0
    period_steps: int = 24

    def __post_init__(self) -> None:
        supported = {"gradual_ramp", "sustained_bottleneck", "periodic", "correlated", "sudden_burst"}
        if self.kind not in supported:
            raise ValueError(f"Unsupported scenario profile: {self.kind}")
        if self.base_demand_mbps <= 0 or self.peak_demand_mbps < self.base_demand_mbps or self.period_steps < 2:
            raise ValueError("Scenario demand parameters are invalid")

    def demand(self, step: int, steps: int, rng: np.random.Generator,
               state: dict[str, float | int]) -> float:
        """Return current demand and retain only past state for correlated input."""
        if self.kind == "gradual_ramp":
            ramp_end = max(1, int(steps * 0.65))
            fraction = min(1.0, step / ramp_end)
            return self.base_demand_mbps + fraction * (self.peak_demand_mbps - self.base_demand_mbps)
        if self.kind == "sustained_bottleneck":
            # A measurable low-load baseline lets the run contain an actual
            # onset before the queue remains loaded for the rest of the run.
            warmup = max(3, int(steps * 0.15))
            return self.base_demand_mbps if step < warmup else self.peak_demand_mbps
        if self.kind == "periodic":
            amplitude = (self.peak_demand_mbps - self.base_demand_mbps) / 2
            # Start at the trough, then rise; the phase itself is predictable.
            return self.base_demand_mbps + amplitude * (1 - np.cos(2 * np.pi * step / self.period_steps))
        if self.kind == "correlated":
            # Common AR(1) demand factor creates correlated load across fixed flows.
            base = max(self.base_demand_mbps, 80.0)
            previous = float(state.get("factor", 1.0))
            factor = max(0.25, 1.0 + 0.88 * (previous - 1.0) + float(rng.normal(0, 0.12)))
            state["factor"] = factor
            return min(self.peak_demand_mbps, max(base, base * factor))
        if "burst_start" not in state:
            low = max(1, int(steps * 0.30))
            high = max(low + 1, int(steps * 0.75))
            state["burst_start"] = int(rng.integers(low, high))
        burst_start = int(state["burst_start"])
        # Four-step unannounced pulse; demand returns to baseline immediately.
        if burst_start <= step < burst_start + 4:
            return self.peak_demand_mbps
        return self.base_demand_mbps


class NetworkSimulator:
    """Simulate traffic and link-level measurements at fixed time intervals.

    Each undirected edge represents two independent full-duplex queues. Offered
    traffic is divided into source/destination flows and routed by Dijkstra.
    Queue capacity is enforced in bytes; excess arriving packets are dropped.
    """

    def __init__(self, topology: NetworkTopology, seed: int = 7, interval_seconds: float = 1.0,
                 packet_size_bytes: int = 1200):
        if interval_seconds <= 0 or packet_size_bytes <= 0:
            raise ValueError("Interval and packet size must be positive")
        self.topology = topology
        self.seed = seed
        self.interval_seconds = interval_seconds
        self.packet_size_bytes = packet_size_bytes
        self._rng = np.random.default_rng(seed)
        self._py_rng = random.Random(seed)
        self._queues: dict[tuple[str, str], float] = {}
        for u, v in topology.graph.edges:
            self._queues[(u, v)] = 0.0
            self._queues[(v, u)] = 0.0
        self._step = 0
        self._scenario_state: dict[str, float | int] = {}
        self._scenario_steps: int | None = None
        self._scenario_paths: list[list[str]] | None = None
        self._scenario_weights: np.ndarray | None = None

    def _init_scenario(self, profile: ScenarioTrafficProfile) -> None:
        routers = self.topology.routers
        if profile.kind == "sustained_bottleneck":
            # Multiple flows share a fixed shortest path with a 50 Mbps hop.
            pairs = [("R1", "R5")] * 8 if {"R1", "R5"}.issubset(routers) else [(routers[0], routers[-1])] * 8
        else:
            # Half the independent conversations traverse the same low-capacity
            # hop; the remaining flows use varied paths to preserve background load.
            pairs = [("R1", "R5")] * 4 if {"R1", "R5"}.issubset(routers) else []
            pairs.extend((routers[i % len(routers)], routers[(i * 3 + 4) % len(routers)])
                         for i in range(len(pairs), 8))
            pairs = [(a, b if b != a else routers[(routers.index(a) + 1) % len(routers)]) for a, b in pairs]
        self._scenario_paths = [self.topology.shortest_path(source, destination) for source, destination in pairs]
        self._scenario_weights = self._rng.dirichlet(np.ones(len(pairs)))

    def step(self, profile: TrafficProfile | ScenarioTrafficProfile | None = None) -> pd.DataFrame:
        """Advance one interval, returning one simulated measurement per link."""
        profile = profile or TrafficProfile()
        if isinstance(profile, ScenarioTrafficProfile):
            if self._scenario_paths is None:
                self._init_scenario(profile)
            offered_mbps = profile.demand(self._step, self._scenario_steps or 0, self._rng, self._scenario_state)
            weights = self._scenario_weights
            paths = self._scenario_paths
        else:
            profile = profile or TrafficProfile()
            offered_mbps = profile.demand(self._step)
            routers = self.topology.routers
            # Legacy profile paths and shares remain independently sampled per tick.
            flow_count = max(4, len(routers) // 2)
            weights = self._rng.dirichlet(np.ones(flow_count))
            paths = [self.topology.shortest_path(*self._py_rng.sample(routers, 2)) for _ in weights]
        flows = []
        for path, weight in zip(paths, weights):
            expected_packets = offered_mbps * float(weight) * 1_000_000 * self.interval_seconds / (self.packet_size_bytes * 8)
            packet_count = int(np.floor(expected_packets)) + int(self._rng.random() < expected_packets % 1)
            flows.append((path, packet_count))

        arrivals: dict[tuple[str, str], float] = {key: 0.0 for key in self._queues}
        demand_by_link: dict[tuple[str, str], float] = {key: 0.0 for key in self._queues}
        packet_bits = self.packet_size_bytes * 8
        for path, packet_count in flows:
            bits = packet_count * packet_bits
            rate = bits / (1_000_000 * self.interval_seconds)
            for a, b in zip(path, path[1:]):
                key = (a, b)
                arrivals[key] += bits
                demand_by_link[key] += rate

        rows = []
        for u, v in self.topology.links:
            for a, b in ((u, v), (v, u)):
                key = (a, b)
                edge = self.topology.graph[a][b]
                capacity_bits = edge["bandwidth_mbps"] * 1_000_000 * self.interval_seconds
                service_bits = int(capacity_bits // packet_bits) * packet_bits
                queue_limit_bits = edge["queue_capacity_bytes"] * 8
                previous = self._queues[key]
                # Traffic arrives throughout the interval while the serializer
                # is transmitting. Reserve service capacity before deciding
                # whether new arrivals overflow the finite queue.
                accepted_limit = max(0.0, queue_limit_bits + service_bits - previous)
                accepted_limit = int(accepted_limit // packet_bits) * packet_bits
                accepted = min(arrivals[key], accepted_limit)
                dropped = arrivals[key] - accepted
                queued = previous + accepted
                transmitted = min(queued, service_bits)
                self._queues[key] = queued - transmitted
                occupancy = self._queues[key] / queue_limit_bits
                utilization = min(1.0, transmitted / capacity_bits)
                queue_delay_ms = ((previous + max(0.0, accepted - transmitted) / 2)
                                  / (edge["bandwidth_mbps"] * 1_000_000)) * 1000
                demand = demand_by_link[key]
                offered_packets = arrivals[key] / packet_bits
                rows.append({
                    "step": self._step, "time_seconds": self._step * self.interval_seconds,
                    "link_id": f"{a}->{b}", "source_router": a, "destination_router": b,
                    "bandwidth_mbps": float(edge["bandwidth_mbps"]),
                    "propagation_delay_ms": float(edge["propagation_delay_ms"]),
                    "queue_capacity_bytes": int(edge["queue_capacity_bytes"]),
                    "offered_demand_mbps": demand, "utilization": utilization,
                    "queue_occupancy": occupancy,
                    "delay_ms": float(edge["propagation_delay_ms"] + queue_delay_ms),
                    "packet_loss_pct": (dropped / arrivals[key] * 100.0) if arrivals[key] else 0.0,
                    "packets_offered": offered_packets,
                    "packets_dropped": dropped / packet_bits,
                    "measurement_source": "simulated",
                })
            # Preserve a representative attribute for simple topology displays.
            edge = self.topology.graph[u][v]
            edge["utilization"] = max(
                rows[-1]["utilization"], rows[-2]["utilization"]
            )
        result = pd.DataFrame(rows)
        self._step += 1
        return result

    def run(self, steps: int, profile: TrafficProfile | ScenarioTrafficProfile | None = None) -> pd.DataFrame:
        """Run a profile for ``steps`` intervals and concatenate link metrics."""
        if steps < 1:
            raise ValueError("steps must be at least 1")
        if isinstance(profile, ScenarioTrafficProfile):
            self._scenario_steps = steps
        return pd.concat([self.step(profile) for _ in range(steps)], ignore_index=True)
