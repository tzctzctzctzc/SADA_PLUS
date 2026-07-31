import importlib.util
from pathlib import Path
import unittest

import torch


MODULE_PATH = Path(__file__).parents[1] / "tools/temporal_boundary_search.py"
SPEC = importlib.util.spec_from_file_location(
    "temporal_boundary_search", MODULE_PATH
)
search = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(search)


class TemporalBoundarySearchTest(unittest.TestCase):
    def setUp(self):
        self.points = torch.arange(20, dtype=torch.float32)
        actionness = torch.zeros(20)
        actionness[5:15] = 1.0
        self.actionness = actionness
        self.features = torch.stack((1.0 - actionness, actionness), dim=1)
        self.inside = torch.tensor([0.0, 1.0])
        self.outside = torch.tensor([1.0, 0.0])
        self.parameters = {
            "search_radius_ratio": 0.4,
            "window_ratio": 0.1,
            "feature_weight": 0.5,
            "semantic_weight": 0.5,
            "distance_weight": 0.1,
            "margin_threshold": 0.0,
            "max_shift_ratio": 0.4,
        }

    def test_directional_evidence_recovers_start_and_end(self):
        refined, details = search.refine_segment(
            self.points, self.features, self.actionness,
            torch.tensor([7.0, 13.0]), self.inside, self.outside,
            self.parameters,
        )
        self.assertEqual(refined.tolist(), [5.0, 15.0])
        self.assertTrue(details["left"]["applied"])
        self.assertTrue(details["right"]["applied"])

    def test_margin_gate_preserves_identity(self):
        parameters = {**self.parameters, "margin_threshold": 100.0}
        segment = torch.tensor([7.0, 13.0])
        refined, details = search.refine_segment(
            self.points, self.features, self.actionness, segment,
            self.inside, self.outside, parameters,
        )
        self.assertTrue(torch.equal(refined, segment))
        self.assertFalse(details["left"]["applied"])
        self.assertFalse(details["right"]["applied"])

    def test_max_shift_gate_preserves_large_correction(self):
        parameters = {**self.parameters, "max_shift_ratio": 0.1}
        segment = torch.tensor([7.0, 13.0])
        refined, _ = search.refine_segment(
            self.points, self.features, self.actionness, segment,
            self.inside, self.outside, parameters,
        )
        self.assertTrue(torch.equal(refined, segment))

    def test_aligned_iou(self):
        value = search.aligned_segment_iou(
            torch.tensor([[0.0, 10.0], [2.0, 8.0]]),
            torch.tensor([[0.0, 10.0], [0.0, 10.0]]),
        )
        self.assertTrue(torch.allclose(value, torch.tensor([1.0, 0.6])))


if __name__ == "__main__":
    unittest.main()
