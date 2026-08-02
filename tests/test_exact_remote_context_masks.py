import importlib.util
from pathlib import Path
import unittest

import torch


MODULE_PATH = (
    Path(__file__).parents[1] / "tools/audit_exact_remote_context_masks.py"
)
SPEC = importlib.util.spec_from_file_location("remote_context_masks", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class ExactRemoteContextMasksTest(unittest.TestCase):
    def test_three_masks_keep_fixed_grid_and_local_core(self):
        inputs = torch.ones((1, 2, 12))
        masks = torch.ones((1, 1, 12), dtype=torch.bool)
        masked_inputs, masked_masks = audit.build_remote_context_batch(
            inputs,
            masks,
            centers=[5, 5, 5],
            radii=[2, 2, 2],
            modes=["past", "future", "bilateral"],
        )
        expected = torch.tensor([
            [False, False, False, True, True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True, False, False, False, False],
            [False, False, False, True, True, True, True, True, False, False, False, False],
        ])
        self.assertTrue(torch.equal(masked_masks[:, 0], expected))
        self.assertEqual(tuple(masked_inputs.shape), (3, 2, 12))
        self.assertTrue(torch.equal(masked_inputs != 0, masked_masks.expand(-1, 2, -1)))

    def test_hash_sampling_is_stable(self):
        first = audit.deterministic_keep(7, "video", 11, 2, 19, 0.5)
        second = audit.deterministic_keep(7, "video", 11, 2, 19, 0.5)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
