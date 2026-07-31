import importlib.util
from pathlib import Path
import unittest

import torch



MODULE_PATH = (
    Path(__file__).parents[1]
    / "libs/models/SADA/boundary_target_utils.py"
)
SPEC = importlib.util.spec_from_file_location("boundary_target_utils", MODULE_PATH)
boundary_target_utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(boundary_target_utils)
boundary_focal_loss = boundary_target_utils.boundary_focal_loss
build_boundary_heatmaps = boundary_target_utils.build_boundary_heatmaps
masked_side_mse = boundary_target_utils.masked_side_mse
refine_segment_from_boundary_maps = (
    boundary_target_utils.refine_segment_from_boundary_maps
)


class BoundaryTargetGenerationTest(unittest.TestCase):
    def test_heatmap_peak_and_scale_assignment(self):
        points = [
            torch.tensor([[float(i), 0.0, 4.0, 1.0] for i in range(12)]),
            torch.tensor([[float(i * 2), 2.0, 8.0, 2.0] for i in range(6)]),
        ]
        masks = [
            torch.ones(1, 1, 12, dtype=torch.bool),
            torch.ones(1, 1, 6, dtype=torch.bool),
        ]
        starts, ends, start_masks, end_masks = build_boundary_heatmaps(
            points, masks, [torch.tensor([[2.0, 8.0]])],
            [[0, 4], [4, 8]], sigma=1.0,
        )
        self.assertEqual(int(starts[0][0].argmax()), 2)
        self.assertEqual(int(ends[0][0].argmax()), 8)
        self.assertEqual(float(starts[1].sum()), 0.0)
        self.assertTrue(start_masks[0].all())
        self.assertTrue(end_masks[0].all())

    def test_truncated_side_is_not_a_positive_and_crop_edge_is_ignored(self):
        points = [
            torch.tensor([[float(i), 0.0, 8.0, 1.0] for i in range(10)])
        ]
        masks = [torch.ones(1, 1, 10, dtype=torch.bool)]
        starts, ends, start_masks, end_masks = build_boundary_heatmaps(
            points,
            masks,
            [torch.tensor([[0.0, 5.0]])],
            [[0, 8]],
            gt_boundary_valid=[torch.tensor([[False, True]])],
            crop_edges=[torch.tensor([True, False])],
            edge_ignore_cells=3,
        )
        self.assertEqual(float(starts[0].sum()), 0.0)
        self.assertEqual(int(ends[0][0].argmax()), 5)
        self.assertFalse(start_masks[0][0, :3].any())
        self.assertFalse(end_masks[0][0, :3].any())
        self.assertTrue(start_masks[0][0, 3:].all())

    def test_boundary_loss_has_gradient(self):
        start = [torch.zeros(1, 5, requires_grad=True)]
        end = [torch.zeros(1, 5, requires_grad=True)]
        target_start = [torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]])]
        target_end = [torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]])]
        masks = [torch.ones(1, 5, dtype=torch.bool)]
        loss = boundary_focal_loss(
            start, end, target_start, target_end, masks, masks
        )
        loss.backward()
        self.assertGreater(float(loss), 0.0)
        self.assertIsNotNone(start[0].grad)
        self.assertIsNotNone(end[0].grad)

    def test_selective_gate_accepts_only_supported_side(self):
        points = torch.arange(0, 11).float()
        start = torch.tensor([0.0, 0.0, 0.1, 0.2, 0.95, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        end = torch.full((11,), 0.2)
        params = {
            "search_radius_ratio": 0.3,
            "absolute_threshold": 0.5,
            "gain_threshold": 0.2,
            "margin_threshold": 0.2,
            "max_shift_ratio": 0.3,
        }
        refined, detail = refine_segment_from_boundary_maps(
            points, start, end, torch.tensor([3.0, 8.0]), params
        )
        self.assertEqual(refined.tolist(), [4.0, 8.0])
        self.assertTrue(detail["left"]["accepted"])
        self.assertFalse(detail["right"]["accepted"])

    def test_rejected_side_is_not_used_by_loss(self):
        predictions = torch.tensor([[[3.0, 9.0]]], requires_grad=True)
        targets = torch.tensor([[[4.0, 7.0]]])
        loss = masked_side_mse(
            predictions, targets,
            torch.tensor([[True]]), torch.tensor([[False]]),
        )
        loss.backward()
        self.assertAlmostEqual(float(loss), 1.0)
        self.assertAlmostEqual(float(predictions.grad[0, 0, 0]), -2.0)
        self.assertAlmostEqual(float(predictions.grad[0, 0, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
