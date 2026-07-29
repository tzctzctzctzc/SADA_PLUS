import importlib.util
from pathlib import Path
import unittest

import torch


MODULE_PATH = Path(__file__).parents[1] / "libs/models/SADA/stage2_utils.py"
SPEC = importlib.util.spec_from_file_location("stage2_utils", MODULE_PATH)
stage2_utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage2_utils)


class Stage2UtilsTest(unittest.TestCase):
    def test_alqa_same_budget_can_rescue_below_alpha_candidate(self):
        confidence = torch.tensor([0.9, 0.8, 0.5, 0.4])
        tiou = torch.tensor([0.2, 0.3, 0.95, 0.1])
        tnd = torch.zeros(4)
        score = stage2_utils.alqa_joint_score(
            confidence, tiou, tnd
        )
        indices, budget = stage2_utils.select_same_budget_alqa(
            confidence,
            score,
            torch.ones(4, dtype=torch.bool),
            alpha=0.6,
        )
        self.assertEqual(budget, 2)
        self.assertEqual(indices.tolist(), [2, 1])
        self.assertEqual(
            int((confidence[indices] < 0.6).sum()), 1
        )

    def test_alqa_same_budget_respects_usable_mask(self):
        indices, budget = stage2_utils.select_same_budget_alqa(
            torch.tensor([0.9, 0.8, 0.7]),
            torch.tensor([0.1, 0.9, 0.8]),
            torch.tensor([True, False, True]),
            alpha=0.6,
        )
        self.assertEqual(budget, 2)
        self.assertEqual(indices.tolist(), [2, 0])

    def test_npl_ambiguity_detects_boundary_outlier(self):
        segments = torch.tensor([
            [10.0, 30.0],
            [10.5, 30.5],
            [10.0, 30.0],
            [25.0, 45.0],
            [10.0, 30.0],
        ])
        ambiguity, counts = stage2_utils.npl_boundary_ambiguity(
            segments, torch.ones(5, dtype=torch.bool), (1,)
        )
        self.assertEqual(counts[:, 0].tolist(), [2, 3, 3, 3, 2])
        self.assertLess(float(ambiguity[0, 0]), float(ambiguity[2, 0]))

    def test_npl_ambiguity_excludes_invalid_neighbors(self):
        segments = torch.tensor([
            [10.0, 30.0],
            [11.0, 31.0],
            [12.0, 32.0],
        ])
        ambiguity, counts = stage2_utils.npl_boundary_ambiguity(
            segments, torch.tensor([True, False, True]), (1, 2)
        )
        self.assertTrue(torch.isinf(ambiguity[0, 0]))
        self.assertEqual(int(counts[0, 0]), 1)
        self.assertEqual(int(counts[0, 1]), 2)

    def test_match_post_nms_to_pre_nms(self):
        pre_segments = torch.tensor([[1.0, 3.0], [4.0, 7.0], [8.0, 9.0]])
        pre_scores = torch.tensor([0.7, 0.9, 0.8])
        pre_labels = torch.tensor([1, 2, 3])
        indices = stage2_utils.match_post_nms_to_pre_nms(
            pre_segments[[1, 0]], pre_scores[[1, 0]], pre_labels[[1, 0]],
            pre_segments, pre_scores, pre_labels,
        )
        self.assertEqual(indices.tolist(), [1, 0])

    def test_matched_budget_has_exact_size(self):
        indices = stage2_utils.matched_budget_indices(
            torch.tensor([0.5, 0.5, 0.4, 0.3]), 0.5
        )
        self.assertEqual(indices.tolist(), [0, 1])

    def test_aligned_tiou_tnd(self):
        tiou, tnd = stage2_utils.aligned_tiou_tnd(
            torch.tensor([[0.0, 10.0], [2.0, 8.0]]),
            torch.tensor([[0.0, 10.0], [0.0, 10.0]]),
        )
        self.assertTrue(torch.allclose(tiou, torch.tensor([1.0, 0.6])))
        self.assertTrue(torch.allclose(tnd, torch.tensor([0.0, 0.0])))

    def test_aligned_tiou_tnd_penalizes_center_shift(self):
        tiou, tnd = stage2_utils.aligned_tiou_tnd(
            torch.tensor([[2.0, 12.0]]),
            torch.tensor([[0.0, 10.0]]),
        )
        self.assertAlmostEqual(float(tiou), 8.0 / 12.0, places=6)
        self.assertAlmostEqual(float(tnd), 4.0 / 144.0, places=6)


if __name__ == "__main__":
    unittest.main()
