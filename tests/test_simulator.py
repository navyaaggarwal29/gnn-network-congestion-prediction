import sys
from pathlib import Path

import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from congestion_sim import NetworkSimulator, TrafficProfile, build_default_topology
from congestion_sim.dataset import (directed_link_edge_index, directed_link_nodes,
                                   generate_runs, prepare_splits, split_run_ids)


class SimulatorTests(unittest.TestCase):
  def test_default_topology_and_dijkstra(self):
    topology = build_default_topology()
    self.assertEqual(topology.graph.number_of_nodes(), 10)
    self.assertEqual(topology.graph.number_of_edges(), 15)
    path = topology.shortest_path("R1", "R6")
    self.assertEqual((path[0], path[-1]), ("R1", "R6"))


  def test_simulation_is_reproducible_and_well_formed(self):
    profile = TrafficProfile(kind="burst", burst_start=2, burst_duration=3)
    first = NetworkSimulator(build_default_topology(), seed=19).run(8, profile)
    second = NetworkSimulator(build_default_topology(), seed=19).run(8, profile)
    self.assertTrue(first.equals(second))
    self.assertEqual(len(first), 8 * 15 * 2)
    self.assertEqual(set(first.measurement_source), {"simulated"})
    self.assertTrue(first.utilization.between(0, 1).all())
    self.assertTrue(first.queue_occupancy.between(0, 1).all())
    self.assertTrue(first.packet_loss_pct.between(0, 100).all())
    self.assertTrue((first.utilization <= 1).all())
    self.assertTrue((first.packets_offered % 1 == 0).all())
    self.assertTrue((first.packets_dropped % 1 == 0).all())
    self.assertTrue((first.packets_dropped <= first.packets_offered).all())


  def test_queue_drops_under_extreme_sustained_load(self):
    profile = TrafficProfile(kind="sustained_high", mean_demand_mbps=1000)
    metrics = NetworkSimulator(build_default_topology(), seed=3).run(4, profile)
    self.assertGreater(metrics.packets_dropped.sum(), 0)
    self.assertGreater(metrics.packet_loss_pct.max(), 0)


  def test_invalid_topology_rejected(self):
    import networkx as nx
    graph = nx.Graph()
    graph.add_edge("a", "b", bandwidth_mbps=0, propagation_delay_ms=1, queue_capacity_bytes=100)
    from congestion_sim.topology import NetworkTopology
    with self.assertRaises(ValueError):
      NetworkTopology(graph)

  def test_directional_line_graph_only_connects_consecutive_links(self):
    topology = build_default_topology()
    nodes = directed_link_nodes(topology)
    edges = directed_link_edge_index(topology)
    edge_pairs = set(map(tuple, edges.T.tolist()))
    forward = nodes.index(("R1", "R2"))
    continuation = nodes.index(("R2", "R3"))
    reverse = nodes.index(("R2", "R1"))
    self.assertIn((forward, continuation), edge_pairs)
    self.assertNotIn((forward, reverse), edge_pairs)

  def test_window_target_alignment_and_reproducibility(self):
    topology = build_default_topology()
    first, measurements = generate_runs(topology, [1, 2, 3], steps=18, window=4, horizon=2)
    second, _ = generate_runs(topology, [1, 2, 3], steps=18, window=4, horizon=2)
    run_id = "normal-seed-1"
    x, y, _ = first.runs[run_id]
    self.assertTrue((x == second.runs[run_id][0]).all())
    # Sample 0 observes steps 0..3 and targets step 5.
    expected = measurements[(measurements.run_id == run_id) & (measurements.step == 5)]
    expected = expected.set_index("link_id").loc[[f"{u}->{v}" for u, v in first.link_nodes]]
    import numpy as np
    self.assertTrue(np.allclose(y[0], expected.utilization.to_numpy()))
    current = measurements[(measurements.run_id == run_id) & (measurements.step == 0)]
    self.assertTrue(np.allclose(current.sort_values("link_id").future_utilization.to_numpy(),
                                measurements[(measurements.run_id == run_id) & (measurements.step == 2)]
                                .sort_values("link_id").utilization.to_numpy()))
    self.assertEqual(x.shape[1], 4)
    timestamp_row = measurements[(measurements.run_id == run_id) & (measurements.step == 0)].iloc[0]
    self.assertEqual(timestamp_row.timestamp.isoformat(), "2025-01-01T00:00:00+00:00")

  def test_seed_group_splits_are_isolated_and_normalization_uses_train(self):
    dataset, _ = generate_runs(build_default_topology(), list(range(10)), steps=17, window=3, horizon=2)
    split = split_run_ids(list(dataset.runs))
    seed_groups = [{int(run.rsplit("-", 1)[1]) for run in ids} for ids in split.values()]
    self.assertTrue(seed_groups[0].isdisjoint(seed_groups[1]))
    self.assertTrue(seed_groups[0].isdisjoint(seed_groups[2]))
    self.assertTrue(seed_groups[1].isdisjoint(seed_groups[2]))
    arrays, stats = prepare_splits(dataset, split)
    self.assertEqual(stats["normalization_fit"], "training_windows_only")
    self.assertTrue(np_allclose_mean_zero(arrays["train"]["x"]))
    self.assertTrue(np_allclose_std_one(arrays["train"]["x"]))


def np_allclose_mean_zero(values):
  import numpy as np
  return np.allclose(values.mean(axis=(0, 1, 3)), 0, atol=1e-5)


def np_allclose_std_one(values):
  import numpy as np
  return np.allclose(values.std(axis=(0, 1, 3)), 1, atol=5e-4)


if __name__ == "__main__":
  unittest.main()
