import importlib.util
from pathlib import Path
import unittest

import torch


MODULE_PATH = Path(__file__).parents[1] / "libs/models/SADA/stage2_utils.py"
SPEC = importlib.util.spec_from_file_location("stage2_utils", MODULE_PATH)
stage2_utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage2_utils)


class PLRAuditTests(unittest.TestCase):
    def test_temporal_roi_interpolation_is_linear(self):
        features = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        centers = torch.tensor([0.5, 1.5, 2.5, 3.5])
        roi = stage2_utils.temporal_roi_interpolate(
            features, torch.tensor([1.0, 3.0]), centers, 4, 5
        )
        torch.testing.assert_close(
            roi, torch.tensor([[0.5, 1.0, 1.5, 2.0, 2.5]])
        )

    def test_boundary_offsets_preserve_positive_duration(self):
        segments = torch.tensor([[2.0, 6.0]])
        offsets = torch.tensor([[1.0, -1.0]])
        refined = stage2_utils.apply_boundary_offsets(
            segments, offsets, 0.45
        )
        torch.testing.assert_close(refined, torch.tensor([[3.8, 4.2]]))
        self.assertGreater(float(refined[0, 1] - refined[0, 0]), 0.0)

    def test_clip_segments_uses_fallback_for_collapsed_interval(self):
        refined = torch.tensor([[5.0, 8.0]])
        fallback = torch.tensor([[3.0, 4.0]])
        clipped = stage2_utils.clip_segments(
            refined, torch.tensor([5.0]), fallback=fallback
        )
        torch.testing.assert_close(clipped, fallback)


if __name__ == "__main__":
    unittest.main()
