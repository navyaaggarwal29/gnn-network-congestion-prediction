import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from congestion_sim.models import LinkGRU, TemporalGNN, batch_edge_index, regression_loss
from congestion_sim.training import load_model_checkpoint, load_split


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/generated/phase2"


class TemporalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with np.load(DATA / "train.npz", allow_pickle=False) as data:
            cls.edge = torch.as_tensor(data["edge_index"], dtype=torch.long)
            cls.nodes = len(data["link_nodes"])

    def test_temporal_gnn_dimensions_bounds_and_cpu_backward(self):
        model = TemporalGNN(input_dim=6, hidden_dim=16, dropout=0.0).cpu()
        features = torch.randn(3, 10, self.nodes, 6, requires_grad=True)
        prediction = model(features, self.edge)
        self.assertEqual(tuple(prediction.shape), (3, self.nodes))
        self.assertTrue(torch.all((prediction >= 0) & (prediction <= 1)).item())
        regression_loss(prediction, torch.rand_like(prediction)).backward()
        self.assertIsNotNone(features.grad)

    def test_graph_batching_matches_independent_graphs(self):
        torch.manual_seed(8)
        model = TemporalGNN(input_dim=6, hidden_dim=8, dropout=0.0).eval()
        batch = torch.randn(2, 4, self.nodes, 6)
        with torch.no_grad():
            together = model(batch, self.edge)
            separate = torch.cat([model(batch[i:i + 1], self.edge) for i in range(2)], dim=0)
        self.assertTrue(torch.allclose(together, separate, atol=1e-6))
        duplicated = batch_edge_index(self.edge, 2, self.nodes)
        self.assertEqual(tuple(duplicated.shape), (2, 2 * self.edge.shape[1]))
        self.assertLess(int(duplicated.min()), self.nodes)
        self.assertGreaterEqual(int(duplicated.max()), self.nodes)

    def test_gcn_respects_source_to_target_direction(self):
        torch.manual_seed(21)
        model = TemporalGNN(input_dim=6, hidden_dim=8, dropout=0.0).eval()
        features = torch.arange(18, dtype=torch.float32).reshape(1, 1, 3, 6) / 10
        forward_edge = torch.tensor([[0], [1]], dtype=torch.long)
        reverse_edge = torch.tensor([[1], [0]], dtype=torch.long)
        with torch.no_grad():
            forward = model(features, forward_edge)
            reverse = model(features, reverse_edge)
        self.assertFalse(torch.allclose(forward, reverse))

    def test_non_graph_gru_dimensions_backward_and_repeatability(self):
        torch.manual_seed(11)
        model = LinkGRU(input_dim=6, hidden_dim=12, dropout=0.0).cpu().eval()
        features = torch.randn(2, 10, self.nodes, 6, requires_grad=True)
        first = model(features)
        second = model(features)
        self.assertEqual(tuple(first.shape), (2, self.nodes))
        self.assertTrue(torch.equal(first, second))
        first.mean().backward()
        self.assertIsNotNone(features.grad)

    def test_checkpoint_round_trip_preserves_predictions(self):
        torch.manual_seed(13)
        model = TemporalGNN(input_dim=6, hidden_dim=8, dropout=0.0).eval()
        x = torch.randn(1, 3, self.nodes, 6)
        with torch.no_grad():
            expected = model(x, self.edge)
        checkpoint = {"model_name": "gnn_standard", "model_config":
                      {"input_dim": 6, "hidden_dim": 8, "dropout": 0.0},
                      "state_dict": model.state_dict(), "best_epoch": 1,
                      "best_validation_mae": 0.1, "parameter_count": sum(p.numel() for p in model.parameters()),
                      "positive_weight": None, "seed": 13}
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.pt"
            torch.save(checkpoint, path)
            loaded, metadata = load_model_checkpoint(path, device="cpu")
            with torch.no_grad():
                actual = loaded(x, self.edge)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(metadata["best_epoch"], 1)

    def test_existing_splits_are_finite_and_seed_disjoint(self):
        x, y, edge = load_split(DATA / "test.npz")
        self.assertEqual(tuple(x.shape[1:]), (10, 30, 6))
        self.assertEqual(tuple(y.shape[1:]), (30,))
        self.assertEqual(edge.shape[0], 2)
        summary = json.loads((DATA / "dataset_summary.json").read_text())
        seeds = [{int(run.rsplit("-", 1)[1]) for run in summary["splits"][name]["runs"]}
                 for name in ("train", "validation", "test")]
        self.assertTrue(seeds[0].isdisjoint(seeds[1]))
        self.assertTrue(seeds[0].isdisjoint(seeds[2]))
        self.assertTrue(seeds[1].isdisjoint(seeds[2]))
        self.assertTrue(torch.isfinite(x).all().item())
        self.assertTrue(torch.isfinite(y).all().item())
        self.assertEqual(summary["normalization"]["normalization_fit"], "training_windows_only")


if __name__ == "__main__":
    unittest.main()
