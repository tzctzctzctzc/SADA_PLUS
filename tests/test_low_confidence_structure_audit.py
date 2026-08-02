import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch


for package_name in ("libs", "libs.models", "libs.models.SADA"):
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = []
        sys.modules[package_name] = package

UTILS_PATH = Path(__file__).parents[1] / "libs/models/SADA/stage1_utils.py"
UTILS_SPEC = importlib.util.spec_from_file_location(
    "libs.models.SADA.stage1_utils", UTILS_PATH
)
stage1_utils = importlib.util.module_from_spec(UTILS_SPEC)
UTILS_SPEC.loader.exec_module(stage1_utils)
sys.modules["libs.models.SADA.stage1_utils"] = stage1_utils

TOPK_PATH = Path(__file__).parents[1] / "tools/audit_low_confidence_topk_consensus.py"
TOPK_SPEC = importlib.util.spec_from_file_location(
    "tools.audit_low_confidence_topk_consensus", TOPK_PATH
)
topk = importlib.util.module_from_spec(TOPK_SPEC)
TOPK_SPEC.loader.exec_module(topk)
sys.modules["tools.audit_low_confidence_topk_consensus"] = topk

MODULE_PATH = Path(__file__).parents[1] / "tools/audit_low_confidence_structure.py"
SPEC = importlib.util.spec_from_file_location("low_confidence_structure", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class LowConfidenceStructureAuditTest(unittest.TestCase):
    def test_shifted_has_no_wraparound(self):
        values = torch.tensor([[1, 2, 3, 4]])
        self.assertEqual(audit.shifted(values, 1).tolist(), [[2, 3, 4, 0]])
        self.assertEqual(audit.shifted(values, -1).tolist(), [[0, 1, 2, 3]])

    def test_same_level_scores_reward_coherent_segments(self):
        probabilities = torch.tensor([[[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]]])
        offsets = torch.ones(1, 3, 2) * 2
        points = torch.tensor([
            [0.0, 0.0, 4.0, 1.0],
            [1.0, 0.0, 4.0, 1.0],
            [2.0, 0.0, 4.0, 1.0],
        ])
        valid = torch.ones(1, 3, dtype=torch.bool)
        scores = audit.same_level_scores(
            probabilities, offsets, points, valid, radius=1
        )
        self.assertAlmostEqual(scores["local_continuity_r1"][0, 1].item(), 0.8)
        self.assertAlmostEqual(scores["segment_vote_r1"][0, 1].item(), 0.8)
        self.assertGreater(scores["segment_iou_r1"][0, 1].item(), 0.4)

    def test_grouped_selection_preserves_each_confidence_bin(self):
        universe = torch.ones(8, dtype=torch.bool)
        ranking = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        levels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        bins = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
        selected = audit.grouped_top_fraction(
            universe, ranking, levels, bins, fraction=0.5
        )
        self.assertEqual(selected.tolist(), [True, False] * 4)

    def test_cross_level_support_uses_same_absolute_time(self):
        probabilities = [
            torch.tensor([[[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]]]),
            torch.tensor([[[0.6, 0.4], [0.4, 0.6]]]),
        ]
        masks = [
            torch.ones(1, 3, dtype=torch.bool),
            torch.ones(1, 2, dtype=torch.bool),
        ]
        points = [
            torch.tensor([[0.0, 0, 4, 1], [1.0, 0, 4, 1], [2.0, 0, 4, 1]]),
            torch.tensor([[0.0, 0, 4, 2], [2.0, 0, 4, 2]]),
        ]
        scores = audit.cross_level_scores(probabilities, masks, points, [0, 1])
        self.assertTrue(torch.allclose(
            scores[0]["cross_level_support"],
            torch.tensor([[0.6, 0.5, 0.4]]),
        ))
        self.assertEqual(
            scores[0]["cross_level_has_support"].tolist(), [[1.0, 1.0, 1.0]]
        )


if __name__ == "__main__":
    unittest.main()
