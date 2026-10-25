from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from congestion_sim.onset import add_causal_temporal_features
from congestion_sim.routing import OnsetPredictor, RoutingController
from congestion_sim.simulator import NetworkSimulator, ScenarioTrafficProfile
from congestion_sim.topology import build_default_topology


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.topology = build_default_topology()
        self.profile = ScenarioTrafficProfile("sustained_bottleneck")

    def test_route_validation_and_loop_prevention(self):
        sim = NetworkSimulator(self.topology, seed=41)
        sim.step(self.profile)
        flow = sim.flow_state[0]
        good = self.topology.shortest_path(flow["source_router"], flow["destination_router"])
        sim.reroute_scenario_flow(0, good)
        with self.assertRaises(ValueError):
            sim.reroute_scenario_flow(0, [flow["source_router"], flow["source_router"], flow["destination_router"]])

    def test_delayed_route_is_applied_after_two_steps_and_changes_flow(self):
        sim = NetworkSimulator(self.topology, seed=42)
        initial = sim.step(self.profile)
        route = list(sim.flow_state[0]["route"])
        a, b = route[0], route[1]
        frame = initial.copy()
        frame.loc[(frame.source_router == a) & (frame.destination_router == b), "utilization"] = .95
        frame.loc[(frame.source_router == a) & (frame.destination_router == b), "queue_occupancy"] = .95
        ctl = RoutingController(self.topology, "reactive", decision_delay=2, cooldown=6)
        ctl.observe(sim, frame, initial, 0)
        self.assertTrue(ctl.pending)
        ctl.apply_due(sim, 1)
        self.assertEqual(sim.flow_state[0]["route"], route)
        sim.step(self.profile)
        ctl.apply_due(sim, 2)
        self.assertNotEqual(sim.flow_state[0]["route"], route)
        self.assertEqual(sim.flow_state[0]["route"][0], sim.flow_state[0]["source_router"])
        self.assertEqual(len(sim.flow_state[0]["route"]), len(set(sim.flow_state[0]["route"])))
        changed = sim.step(self.profile)
        static = NetworkSimulator(self.topology, seed=42)
        baseline = static.run(3, self.profile)
        expected = baseline[baseline.step == 2].reset_index(drop=True)
        actual = changed.reset_index(drop=True)
        self.assertFalse(np.allclose(actual.offered_demand_mbps, expected.offered_demand_mbps))

    def test_routing_strategies_preserve_exogenous_flow_demands(self):
        first = NetworkSimulator(self.topology, seed=47)
        second = NetworkSimulator(self.topology, seed=47)
        first.step(self.profile); second.step(self.profile)
        original = first.flow_state
        alt = self.topology.shortest_path(original[0]["source_router"], original[0]["destination_router"])
        # Set another valid route by excluding the first hop when available.
        graph = self.topology.graph.copy(); a, b = original[0]["route"][:2]; graph.remove_edge(a, b)
        import networkx as nx
        alternate = nx.shortest_path(graph, original[0]["source_router"], original[0]["destination_router"], weight="propagation_delay_ms")
        first.reroute_scenario_flow(0, alternate)
        self.assertEqual([f["offered_demand_mbps"] for f in first.flow_state],
                         [f["offered_demand_mbps"] for f in second.flow_state])
        self.assertEqual(sum(f["offered_demand_mbps"] for f in first.flow_state),
                         sum(f["offered_demand_mbps"] for f in second.flow_state))

    def test_cooldown_suppresses_duplicate_link_decisions(self):
        sim = NetworkSimulator(self.topology, seed=43)
        frame = sim.step(self.profile)
        a, b = sim.flow_state[0]["route"][:2]
        hot = frame.copy()
        mask = (hot.source_router == a) & (hot.destination_router == b)
        hot.loc[mask, ["utilization", "queue_occupancy"]] = .95
        ctl = RoutingController(self.topology, "reactive", decision_delay=2, cooldown=6)
        first = ctl.observe(sim, hot, frame, 0)
        second = ctl.observe(sim, hot, frame, 1)
        self.assertEqual(len(first), 1)
        self.assertEqual(second, [])

    def test_causal_features_do_not_depend_on_future_rows(self):
        sim = NetworkSimulator(self.topology, seed=44)
        frame = sim.run(12, self.profile)
        prefix = frame[frame.step <= 8].copy()
        from_prefix = add_causal_temporal_features(prefix)
        from_full = add_causal_temporal_features(frame)
        columns = ["utilization_trend", "queue_growth_rate", "offered_load_trend",
                   "utilization_rolling_std", "recent_drop_rate"]
        left = from_prefix.set_index(["step", "link_id"])[columns]
        right = from_full[from_full.step <= 8].set_index(["step", "link_id"])[columns]
        pd.testing.assert_frame_equal(left, right)

    def test_checkpoint_loads_real_phase36_weights_and_threshold(self):
        artifacts = Path(__file__).resolve().parents[1] / "results" / "phase3_6"
        if not (artifacts / "checkpoints/gru_onset.pt").exists():
            self.skipTest("Phase 3.6 checkpoint artifacts are not installed")
        model = OnsetPredictor("gru_onset", artifacts, self.topology)
        self.assertAlmostEqual(model.threshold, .9972271353006363)
        self.assertEqual(model.window, 10)
        frame = NetworkSimulator(self.topology, seed=45).run(10, self.profile)
        scores = model.predict(frame)
        self.assertEqual(scores.shape, (len(model.links),))
        self.assertTrue(np.isfinite(scores).all())

    def test_scenario_replay_is_deterministic(self):
        left = NetworkSimulator(self.topology, seed=46).run(20, self.profile)
        right = NetworkSimulator(self.topology, seed=46).run(20, self.profile)
        pd.testing.assert_frame_equal(left, right)

    def test_stepwise_scenario_horizon_matches_run_for_ramp_and_burst(self):
        for scenario in ("gradual_ramp", "sudden_burst"):
            profile = ScenarioTrafficProfile(scenario)
            via_run = NetworkSimulator(self.topology, seed=48).run(20, profile)
            stepwise = NetworkSimulator(self.topology, seed=48)
            stepwise.set_scenario_horizon(20)
            via_steps = pd.concat([stepwise.step(profile) for _ in range(20)], ignore_index=True)
            pd.testing.assert_frame_equal(via_run, via_steps)


if __name__ == "__main__":
    unittest.main()
