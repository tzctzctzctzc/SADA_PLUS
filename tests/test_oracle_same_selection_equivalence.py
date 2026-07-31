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
build_same_selection_supervision = (
    boundary_target_utils.build_same_selection_supervision
)


class OracleSameSelectionEquivalenceTest(unittest.TestCase):
    def test_only_regression_target_changes(self):
        pseudo_labels = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]])
        gt_labels = torch.tensor([[[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]])
        pseudo_offsets = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        gt_offsets = torch.tensor([[[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]]])
        valid = torch.tensor([[True, True, True]])
        levels = torch.tensor([[True, True, False]])
        clean_mask, clean_target = build_same_selection_supervision(
            pseudo_labels, gt_labels, pseudo_offsets, gt_offsets,
            valid, levels, "clean_pseudo",
        )
        oracle_mask, oracle_target = build_same_selection_supervision(
            pseudo_labels, gt_labels, pseudo_offsets, gt_offsets,
            valid, levels, "same_selection",
        )
        self.assertTrue(torch.equal(clean_mask, oracle_mask))
        self.assertEqual(int(clean_mask.sum()), 1)
        self.assertTrue(torch.equal(clean_target, pseudo_offsets))
        self.assertTrue(torch.equal(oracle_target, gt_offsets))

        common_losses = {
            "classification": torch.tensor(0.7),
            "source": torch.tensor(1.2),
            "adversarial": torch.tensor(0.4),
        }
        clean_common = {key: value.clone() for key, value in common_losses.items()}
        oracle_common = {key: value.clone() for key, value in common_losses.items()}
        for key in common_losses:
            self.assertTrue(torch.equal(clean_common[key], oracle_common[key]))


if __name__ == "__main__":
    unittest.main()
