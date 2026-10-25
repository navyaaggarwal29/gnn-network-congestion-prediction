"""Network topology construction and shortest-path routing."""

from __future__ import annotations

from dataclasses import dataclass
import networkx as nx


@dataclass(frozen=True)
class Link:
    """Undirected full-duplex link properties; capacity is per direction."""

    bandwidth_mbps: float
    propagation_delay_ms: float
    queue_capacity_bytes: int


class NetworkTopology:
    """Router graph with validated physical link attributes."""

    def __init__(self, graph: nx.Graph):
        if graph.is_directed() or graph.number_of_nodes() < 2:
            raise ValueError("Topology must be an undirected graph with at least two routers")
        if not nx.is_connected(graph):
            raise ValueError("Topology must be connected")
        for u, v, data in graph.edges(data=True):
            for key in ("bandwidth_mbps", "propagation_delay_ms", "queue_capacity_bytes"):
                if key not in data or float(data[key]) <= 0:
                    raise ValueError(f"Link {u}-{v} requires positive {key}")
        self.graph = graph.copy()
        self.graph.graph.setdefault("units", "bandwidth Mbps, delay ms, queue bytes")

    @property
    def routers(self) -> list[str]:
        return sorted(self.graph.nodes, key=str)

    @property
    def links(self) -> list[tuple[str, str]]:
        return sorted((tuple(sorted((u, v), key=str)) for u, v in self.graph.edges), key=lambda e: (str(e[0]), str(e[1])))

    def shortest_path(self, source: str, destination: str) -> list[str]:
        """Return Dijkstra path minimizing propagation delay in milliseconds."""
        return nx.shortest_path(self.graph, source, destination, weight="propagation_delay_ms", method="dijkstra")

    def path_links(self, path: list[str]) -> list[tuple[str, str]]:
        return [tuple(sorted((a, b), key=str)) for a, b in zip(path, path[1:])]


def build_default_topology() -> NetworkTopology:
    """Build a connected 10-router mesh with heterogeneous link properties."""
    graph = nx.Graph()
    graph.add_nodes_from(f"R{i}" for i in range(1, 11))
    # A ring plus chords gives multiple paths while keeping the demonstration legible.
    edges = [
        ("R1", "R2", 100, 4), ("R2", "R3", 80, 5), ("R3", "R4", 100, 4),
        ("R4", "R5", 60, 7), ("R5", "R6", 100, 4), ("R6", "R7", 80, 5),
        ("R7", "R8", 100, 4), ("R8", "R9", 60, 7), ("R9", "R10", 100, 4),
        ("R10", "R1", 80, 5), ("R1", "R6", 50, 10), ("R2", "R7", 70, 8),
        ("R3", "R8", 50, 9), ("R4", "R9", 70, 8), ("R5", "R10", 50, 10),
    ]
    for u, v, bandwidth, delay in edges:
        graph.add_edge(u, v, bandwidth_mbps=bandwidth, propagation_delay_ms=delay,
                       queue_capacity_bytes=252_000, utilization=0.0)
    return NetworkTopology(graph)
