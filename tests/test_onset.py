import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from congestion_sim import (NetworkSimulator, ScenarioTrafficProfile,
                            build_default_topology)
from congestion_sim.models import TemporalOnsetGNN, TemporalOnsetGRU
from congestion_sim.onset import (ONSET_THRESHOLD, OnsetData, add_causal_temporal_features,
    episode_event_metrics, episode_lengths, generate_onset_data, normalize_onset_features,
    split_onset_indices)


class OnsetScenarioTests(unittest.TestCase):
    def test_all_new_scenarios_are_reproducible_and_physically_bounded(self):
        for name in ("gradual_ramp", "sustained_bottleneck", "periodic", "correlated", "sudden_burst"):
            profile = ScenarioTrafficProfile(name)
            a = NetworkSimulator(build_default_topology(), seed=9).run(32, profile)
            b = NetworkSimulator(build_default_topology(), seed=9).run(32, profile)
            self.assertTrue(a.equals(b), name)
            self.assertTrue(a.utilization.between(0, 1).all())
            self.assertTrue(a.queue_occupancy.between(0, 1).all())
            self.assertTrue(a.packet_loss_pct.between(0, 100).all())
            self.assertTrue((a.packets_dropped <= a.packets_offered).all())
            self.assertEqual(a.measurement_source.unique().tolist(), ["simulated"])

    def test_scenario_demand_shapes(self):
        rng = np.random.default_rng(4)
        state = {}
        ramp = [ScenarioTrafficProfile("gradual_ramp").demand(i, 20, rng, {}) for i in range(20)]
        self.assertLess(ramp[0], ramp[-1])
        periodic = [ScenarioTrafficProfile("periodic", period_steps=8).demand(i, 24, rng, {}) for i in range(24)]
        self.assertGreater(max(periodic), min(periodic))
        burst = ScenarioTrafficProfile("sudden_burst")
        rates = [burst.demand(i, 40, rng, state) for i in range(40)]
        self.assertEqual(sum(rate > 40 for rate in rates), 4)
        self.assertGreater(max(rates), min(rates))

    def test_sustained_bottleneck_creates_queue_congestion_and_drops(self):
        metrics = NetworkSimulator(build_default_topology(), seed=3).run(
            30, ScenarioTrafficProfile("sustained_bottleneck"))
        choke = metrics[(metrics.link_id == "R1->R6")]
        self.assertLess(choke[choke.step == 0].utilization.iloc[0], ONSET_THRESHOLD)
        self.assertGreater((choke.utilization >= ONSET_THRESHOLD).sum(), 20)
        self.assertGreater(choke.packets_dropped.sum(), 0)
        self.assertTrue((choke.queue_occupancy <= 1).all())

    def test_temporal_features_are_causal(self):
        frame = pd.DataFrame({"link_id": ["a"] * 6, "step": range(6),
            "utilization": [.1,.2,.3,.4,.5,.6], "queue_occupancy": [.1,.1,.2,.3,.4,.5],
            "offered_demand_mbps": [10,11,12,13,14,15], "packet_loss_pct": [0,0,0,5,0,0]})
        first = add_causal_temporal_features(frame)
        altered = frame.copy(); altered.loc[4:, ["utilization","queue_occupancy","offered_demand_mbps","packet_loss_pct"]] = 99
        second = add_causal_temporal_features(altered)
        cols = ["utilization_trend","queue_growth_rate","offered_load_trend","utilization_rolling_std","recent_drop_rate"]
        pd.testing.assert_frame_equal(first.loc[:3,cols], second.loc[:3,cols])

    def test_onset_labels_and_seed_isolation(self):
        data = generate_onset_data(build_default_topology(), [2,3,4,5,6], steps=24, window=4, horizon=3)
        splits = split_onset_indices(data)
        seed_sets = [{int(v) for v in data.seed[ix]} for ix in splits.values()]
        self.assertFalse(seed_sets[0]&seed_sets[1] or seed_sets[0]&seed_sets[2] or seed_sets[1]&seed_sets[2])
        i = next(i for i, run in enumerate(data.run_id) if run == "sustained_bottleneck-seed-2")
        decision = int(data.decision_step[i])
        util = data.utilization_by_run[data.run_id[i]]
        expected = ((util[decision] < ONSET_THRESHOLD) &
                    (util[decision+1:decision+4] >= ONSET_THRESHOLD).any(axis=0))
        np.testing.assert_array_equal(data.onset[i].astype(bool), expected)
        np.testing.assert_array_equal(data.already_congested[i].astype(bool), util[decision] >= ONSET_THRESHOLD)
        self.assertTrue(np.isfinite(data.x).all())

    def test_feature_normalization_uses_training_samples_only(self):
        x=np.zeros((6,2,1,2),dtype=np.float32)
        x[0:2,:,:,0]=2; x[2:4,:,:,0]=7; x[4:6,:,:,0]=1000
        splits={"train":np.array([0,1]),"validation":np.array([2,3]),"test":np.array([4,5])}
        normalized,meta=normalize_onset_features(x,splits)
        self.assertEqual(meta["normalization_fit"],"training_windows_only")
        self.assertAlmostEqual(meta["mean"][0],2.0)
        self.assertGreater(float(normalized[4].mean()),100)

    def test_episode_metrics_deduplicate_overlapping_alert_windows(self):
        # One actual event at step 4, with alerts on adjacent decision steps 1 and 2.
        util = np.zeros((8, 2), dtype=np.float32); util[4:6,0] = .9
        x = np.zeros((5,1,2,1),dtype=np.float32)
        data = OnsetData(x=x,onset=np.zeros((5,2)),onset_mask=np.ones((5,2)),continued=np.zeros((5,2)),
            already_congested=np.zeros((5,2)),run_id=np.array(["r"]*5),scenario=np.array(["s"]*5),
            seed=np.ones(5,dtype=int),decision_step=np.arange(5),utilization_by_run={"r":util},
            measurements=pd.DataFrame(),link_nodes=[("a","b"),("b","c")],edge_index=np.empty((2,0),int),window=1,horizon=3)
        scores=np.zeros((5,2)); scores[1:3,0]=.9; scores[2,1]=.9
        m=episode_event_metrics(scores,data,np.arange(5),.5,decision_delay=2)
        self.assertEqual(m["true_positive_episodes"],1)
        self.assertEqual(m["false_alert_episodes"],1)
        self.assertEqual(m["missed_episodes"],0)
        self.assertEqual(m["mean_lead_steps"],3)
        self.assertAlmostEqual(m["false_alert_episodes_per_run"],1)
        self.assertEqual(episode_lengths(np.array([0,.9,.9,0,.9])),[2,1])

    def test_onset_gnn_and_gru_logits_have_gradient_flow(self):
        edge=torch.tensor([[0,1],[1,2]],dtype=torch.long)
        for model in (TemporalOnsetGRU(5,hidden_dim=8),TemporalOnsetGNN(5,hidden_dim=8)):
            x=torch.randn(2,4,3,5,requires_grad=True)
            logits=model(x,edge)
            self.assertEqual(tuple(logits.shape),(2,3))
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.rand_like(logits))
            loss.backward()
            self.assertIsNotNone(x.grad)
            self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == "__main__":
    unittest.main()
