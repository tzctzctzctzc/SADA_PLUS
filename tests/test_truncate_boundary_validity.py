import importlib.util
from pathlib import Path
import unittest

import torch


MODULE_PATH = Path(__file__).parents[1] / "libs/datasets/data_utils.py"
SPEC = importlib.util.spec_from_file_location("data_utils", MODULE_PATH)
data_utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(data_utils)
truncate_feats = data_utils.truncate_feats


class FixedCropRng:
    def __init__(self, start):
        self.start = start

    def randint(self, low, high):
        return self.start


class TruncateBoundaryValidityTest(unittest.TestCase):
    def test_marks_only_the_cropped_side_invalid(self):
        sample = {
            "feats": torch.zeros(2, 12),
            "segments": torch.tensor([[2.0, 8.0], [5.0, 9.0]]),
            "labels": torch.tensor([0, 1]),
            "metadata": {},
        }
        cropped = truncate_feats(
            sample, max_seq_len=6, trunc_thresh=0.3,
            rng=FixedCropRng(4),
        )
        self.assertEqual(cropped["segments"].tolist(), [[0.0, 4.0], [1.0, 5.0]])
        self.assertEqual(
            cropped["boundary_valid"].tolist(),
            [[False, True], [True, True]],
        )
        self.assertEqual(cropped["metadata"]["crop_start"], 4)

    def test_uncropped_sample_keeps_both_sides_valid(self):
        sample = {
            "feats": torch.zeros(2, 6),
            "segments": torch.tensor([[1.0, 4.0]]),
            "labels": torch.tensor([0]),
            "metadata": {},
        }
        result = truncate_feats(
            sample, max_seq_len=6, trunc_thresh=0.3
        )
        self.assertEqual(result["boundary_valid"].tolist(), [[True, True]])


if __name__ == "__main__":
    unittest.main()
