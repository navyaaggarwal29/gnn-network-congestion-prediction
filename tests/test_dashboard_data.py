from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from congestion_sim.dashboard_data import (
    DashboardDataError, comparison_data, decision_counts, event_rows, filter_run, load_dashboard_data,
    measurement_rows, metric_improvements, parse_json_value,
)


def sample_log() -> pd.DataFrame:
    common = {"scenario": "periodic", "seed": 3000, "strategy": "gnn", "step": 10,
              "time_seconds": 10.0, "measurement_source": "simulated", "routing_decision": None,
              "alternative_route": None, "applied_step": np.nan, "predicted_congestion_probability": .98,
              "utilization": .7, "queue_occupancy": .2, "delay_ms": 8.0, "packet_loss_pct": 0.0,
              "flow_routes": '{"R1->R5":["R1","R10","R5"]}', "source_router": "R1",
              "destination_router": "R6", "bandwidth_mbps": 100.0, "offered_demand_mbps": 20.0,
              "packets_offered": 200, "packets_dropped": 0, "queue_capacity_bytes": 252000,
              "propagation_delay_ms": 4.0, "delivered_throughput_mbps": 20.0, "flow_id": np.nan,
              "target_link": None, "current_route": None, "link_id": "R1->R6"}
    measure = dict(common)
    event = dict(common, link_id=None, step=11, routing_decision="predicted onset risk; scheduled after 2 steps",
                 alternative_route='["R1","R10","R5"]', applied_step=13, flow_id=0,
                 target_link="R1->R6", predicted_congestion_probability=.99,
                 current_route='["R1","R6","R5"]')
    unrelated = dict(measure, seed=3001, utilization=.1)
    return pd.DataFrame([measure, event, unrelated])


class DashboardDataTests(unittest.TestCase):
    def test_run_filter_and_event_rows_keep_scenario_seed_identity(self):
        filtered = filter_run(sample_log(), "periodic", "gnn", 3000)
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered.seed.unique().tolist(), [3000])
        self.assertEqual(len(measurement_rows(filtered)), 1)
        events = event_rows(filtered)
        self.assertEqual(len(events), 1)
        self.assertEqual(parse_json_value(events.iloc[0].alternative_route), ["R1", "R10", "R5"])
        self.assertAlmostEqual(float(events.iloc[0].predicted_congestion_probability), .99)
        self.assertEqual(events.iloc[0].target_link, "R1->R6")
        self.assertEqual(decision_counts(events, 11), {"pending": 1, "applied": 0, "rejected": 0, "issued": 1})
        self.assertEqual(decision_counts(events, 13), {"pending": 0, "applied": 1, "rejected": 0, "issued": 1})

    def test_replay_is_deterministic_and_missing_json_is_not_zero(self):
        log = sample_log()
        first = filter_run(log, "periodic", "gnn", 3000)
        second = filter_run(log.copy(), "periodic", "gnn", 3000)
        pd.testing.assert_frame_equal(first, second)
        self.assertIsNone(parse_json_value(np.nan))
        self.assertIsNone(parse_json_value(""))

    def test_comparison_uses_actual_values_and_strategy_mapping(self):
        metrics = pd.DataFrame({"scenario": ["periodic"] * 8,
            "strategy": ["static", "static", "reactive", "reactive", "gru", "gru", "gnn", "gnn"],
            "seed": [3000, 3001] * 4, "mean_delay_ms": [10, 12, 8, 10, 9, 11, 10, 13],
            "packet_delivery_ratio_hop_proxy": [.8, .9, .9, .95, .85, .92, .8, .91],
            "congestion_duration_steps": [10, 12, 8, 9, 9, 10, 10, 12]})
        table = comparison_data(metrics, "periodic", "mean_delay_ms")
        self.assertEqual(set(table.strategy), {"static", "reactive", "gru", "gnn"})
        reactive = metric_improvements(metrics, "periodic", "reactive")
        self.assertAlmostEqual(reactive["mean_delay_improvement_pct"], (20 + 100 * 2 / 12) / 2)
        self.assertAlmostEqual(reactive["delivery_proxy_delta_pp"], 7.5)

    def test_missing_pairs_remain_missing_not_fabricated_defaults(self):
        metrics = pd.DataFrame({"scenario": ["periodic"], "strategy": ["static"], "seed": [3000],
            "mean_delay_ms": [10.0], "packet_delivery_ratio_hop_proxy": [.8],
            "congestion_duration_steps": [5]})
        values = metric_improvements(metrics, "periodic", "gnn")
        self.assertIsNone(values["mean_delay_improvement_pct"])
        self.assertTrue(np.isnan(parse_json_value(np.nan) or np.nan))
        with self.assertRaises(DashboardDataError):
            comparison_data(metrics, "periodic", "not_a_metric")

    def test_loader_handles_required_files_and_reports_missing_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(DashboardDataError):
                load_dashboard_data(root)
            (root / "results/phase4").mkdir(parents=True)
            (root / "results/phase3_6").mkdir(parents=True)
            sample_log().to_csv(root / "results/phase4/simulation_log.csv", index=False)
            pd.DataFrame({"scenario": [], "seed": [], "strategy": []}).to_csv(root / "results/phase4/per_run_metrics.csv", index=False)
            pd.DataFrame({"scenario": [], "seed": [], "strategy": []}).to_csv(root / "results/phase4/scenario_summary.csv", index=False)
            pd.DataFrame({"model": []}).to_csv(root / "results/phase3_6/test_metrics.csv", index=False)
            pd.DataFrame({"model": []}).to_csv(root / "results/phase3_6/scenario_test_metrics.csv", index=False)
            (root / "results/phase3_6/frozen_selection.json").write_text(json.dumps({"threshold_by_model": {}}))
            loaded = load_dashboard_data(root)
            self.assertEqual(len(measurement_rows(loaded["log"])), 2)


if __name__ == "__main__":
    unittest.main()
