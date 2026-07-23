import unittest
import importlib.util
from pathlib import Path
import sys
import types

import torch

MODULE_PATH = Path(__file__).parents[1] / "libs/models/SADA/stage1_utils.py"
SPEC = importlib.util.spec_from_file_location("stage1_utils", MODULE_PATH)
stage1_utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage1_utils)
for package_name in ("libs", "libs.models", "libs.models.SADA"):
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = []
        sys.modules[package_name] = package
sys.modules["libs.models.SADA.stage1_utils"] = stage1_utils
CALIBRATION_PATH = Path(__file__).parents[1] / "libs/models/SADA/stage1_calibration.py"
CALIBRATION_SPEC = importlib.util.spec_from_file_location(
    "libs.models.SADA.stage1_calibration", CALIBRATION_PATH
)
stage1_calibration = importlib.util.module_from_spec(CALIBRATION_SPEC)
CALIBRATION_SPEC.loader.exec_module(stage1_calibration)
build_routing_masks = stage1_utils.build_routing_masks
interpolate_probabilities = stage1_utils.interpolate_probabilities
shift_features = stage1_utils.shift_features
weighted_quantile = stage1_utils.weighted_quantile


class Stage1UtilsTest(unittest.TestCase):
    def test_shift_has_no_wraparound(self):
        features = torch.arange(5).float().view(1, 5)
        mask = torch.ones(1, 5, dtype=torch.bool)
        shifted, shifted_mask = shift_features(features, mask, 2)
        self.assertEqual(shifted.tolist(), [[0, 0, 0, 1, 2]])
        self.assertEqual(
            shifted_mask.tolist(), [[False, False, True, True, True]]
        )

    def test_interpolation_uses_two_valid_endpoints(self):
        points = torch.tensor([[0.0, 0, 0, 1], [4.0, 0, 0, 4], [8.0, 0, 0, 4]])
        probabilities = torch.tensor([[[0.0], [0.4], [0.8]]])
        mask = torch.tensor([[True, True, True]])
        output, valid = interpolate_probabilities(
            probabilities, mask, points, torch.tensor([2.0, 6.0, 9.0])
        )
        self.assertTrue(torch.allclose(output[0, :2, 0], torch.tensor([0.2, 0.6])))
        self.assertEqual(valid.tolist(), [[True, True, False]])

    def test_routing_is_mutually_exclusive(self):
        original = torch.tensor([[[0.80, 0.10], [0.02, 0.03], [0.40, 0.30]]])
        perturbed = torch.tensor([[[0.76, 0.12], [0.03, 0.02], [0.35, 0.34]]])
        route = build_routing_masks(
            original, perturbed, torch.ones(1, 3, dtype=torch.bool),
            alpha=0.6, beta=0.1, tau_act=0.05, tau_bg=0.02,
        )
        self.assertEqual(route["reliable_action"].tolist(), [[True, False, False]])
        self.assertEqual(route["stable_background"].tolist(), [[False, True, False]])
        self.assertTrue(torch.all(
            route["reliable_action"].int()
            + route["stable_background"].int()
            + route["ignore"].int()
            == 1
        ))

    def test_weighted_quantile(self):
        value = weighted_quantile(
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([1.0, 1.0, 8.0]),
            0.5,
        )
        self.assertEqual(value.item(), 3.0)

    def test_layer_balanced_calibration(self):
        records = [{
            "level": torch.tensor([0, 0, 1, 1]),
            "is_action": torch.tensor([False, True, False, True]),
            "r": torch.tensor([0.01, 0.8, 0.02, 0.9]),
            "action_eligible": torch.tensor([False, True, False, True]),
            "action_std": torch.tensor([0.0, 0.03, 0.0, 0.04]),
            "background_valid": torch.ones(4, dtype=torch.bool),
            "background_std": torch.tensor([0.01, 0.2, 0.02, 0.3]),
        }]
        output = stage1_calibration.calibrate_thresholds(
            records, quantile=0.9, epsilon_bg=0.01
        )
        self.assertAlmostEqual(output["beta"], 0.02, places=6)
        self.assertAlmostEqual(output["tau_act"], 0.04, places=6)
        self.assertAlmostEqual(output["tau_bg"], 0.02, places=6)

    def test_negative_shift_has_no_wraparound(self):
        features = torch.arange(5).float().view(1, 5)
        mask = torch.ones(1, 5, dtype=torch.bool)
        shifted, shifted_mask = shift_features(features, mask, -2)
        self.assertEqual(shifted.tolist(), [[2, 3, 4, 0, 0]])
        self.assertEqual(
            shifted_mask.tolist(), [[True, True, True, False, False]]
        )


if __name__ == "__main__":
    unittest.main()
