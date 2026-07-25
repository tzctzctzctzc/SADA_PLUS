import unittest
import importlib.util
from pathlib import Path
import sys
import tempfile
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
build_oracle_routing = stage1_utils.build_oracle_routing
context_view_on_cpu = stage1_utils.context_view_on_cpu
interpolate_probabilities = stage1_utils.interpolate_probabilities
shift_features = stage1_utils.shift_features
weighted_quantile = stage1_utils.weighted_quantile
source_action_eligible = stage1_utils.source_action_eligible
validate_stage1_resume = stage1_utils.validate_stage1_resume

DATA_UTILS_PATH = Path(__file__).parents[1] / "libs/datasets/data_utils.py"
DATA_UTILS_SPEC = importlib.util.spec_from_file_location(
    "stage1_data_utils", DATA_UTILS_PATH
)
stage1_data_utils = importlib.util.module_from_spec(DATA_UTILS_SPEC)
DATA_UTILS_SPEC.loader.exec_module(stage1_data_utils)


class Stage1UtilsTest(unittest.TestCase):
    @staticmethod
    def _stage1_checkpoint(
        perturbation="shift", alpha=0.6, epoch=5, include_new_fields=True
    ):
        stage1 = {"calibrated": True}
        if include_new_fields:
            stage1.update({
                "enabled": True,
                "perturbation": perturbation,
                "alpha": alpha,
            })
        return {
            "epoch": epoch,
            "stage1": stage1,
            "config": {
                "stage1": {
                    "enabled": True,
                    "perturbation": perturbation,
                },
                "model": {"pl_threshold": alpha},
            },
        }

    def test_stage1_resume_rejects_forgotten_cli_mode(self):
        with self.assertRaisesRegex(ValueError, "--stage1 off"):
            validate_stage1_resume(
                self._stage1_checkpoint(),
                current_enabled=False,
                current_perturbation="shift",
                current_alpha=0.6,
                calibration_epoch=5,
            )

    def test_stage1_resume_rejects_perturbation_mismatch(self):
        with self.assertRaisesRegex(ValueError, "perturbation mismatch"):
            validate_stage1_resume(
                self._stage1_checkpoint(perturbation="shift"),
                current_enabled=True,
                current_perturbation="context",
                current_alpha=0.6,
                calibration_epoch=5,
            )

    def test_stage1_resume_rejects_alpha_mismatch(self):
        with self.assertRaisesRegex(ValueError, "alpha mismatch"):
            validate_stage1_resume(
                self._stage1_checkpoint(alpha=0.6),
                current_enabled=True,
                current_perturbation="shift",
                current_alpha=0.5,
                calibration_epoch=5,
            )

    def test_stage1_resume_accepts_legacy_config_metadata(self):
        checkpoint = self._stage1_checkpoint(include_new_fields=False)
        checkpoint["config"] = types.SimpleNamespace(
            config=checkpoint["config"]
        )
        state = validate_stage1_resume(
            checkpoint,
            current_enabled=True,
            current_perturbation="shift",
            current_alpha=0.6,
            calibration_epoch=5,
        )
        self.assertIs(state, checkpoint["stage1"])

    def test_stage1_resume_rejects_missing_post_calibration_state(self):
        checkpoint = {"epoch": 4, "config": {}}
        self.assertIsNone(validate_stage1_resume(
            checkpoint,
            current_enabled=True,
            current_perturbation="shift",
            current_alpha=0.6,
            calibration_epoch=5,
        ))
        checkpoint["epoch"] = 5
        with self.assertRaisesRegex(ValueError, "no Stage 1 state"):
            validate_stage1_resume(
                checkpoint,
                current_enabled=True,
                current_perturbation="shift",
                current_alpha=0.6,
                calibration_epoch=5,
            )

    def test_shift_has_no_wraparound(self):
        features = torch.arange(5).float().view(1, 5)
        mask = torch.ones(1, 5, dtype=torch.bool)
        shifted, shifted_mask = shift_features(features, mask, 2)
        self.assertEqual(shifted.tolist(), [[0, 0, 0, 1, 2]])
        self.assertEqual(
            shifted_mask.tolist(), [[False, False, True, True, True]]
        )

    def test_context_valid_and_invalid_views_share_cpu_device(self):
        full = torch.arange(12).float().view(1, 12)
        original = full[:, 2:8].clone()
        valid_view = context_view_on_cpu(
            original, full, start=3, length=6
        )
        invalid_view = context_view_on_cpu(
            original, full, start=2, length=6
        )
        fallback_view = context_view_on_cpu(
            original, None, start=0, length=6
        )

        self.assertEqual(valid_view.device.type, "cpu")
        self.assertEqual(invalid_view.device.type, "cpu")
        self.assertEqual(fallback_view.device.type, "cpu")
        self.assertEqual(valid_view.tolist(), [[3, 4, 5, 6, 7, 8]])
        self.assertEqual(invalid_view.tolist(), [[2, 3, 4, 5, 6, 7]])
        self.assertTrue(torch.equal(fallback_view, original))

    def test_interpolation_uses_two_valid_endpoints(self):
        points = torch.tensor([[0.0, 0, 0, 1], [4.0, 0, 0, 4], [8.0, 0, 0, 4]])
        probabilities = torch.tensor([[[0.0], [0.4], [0.8]]])
        mask = torch.tensor([[True, True, True]])
        output, valid = interpolate_probabilities(
            probabilities, mask, points, torch.tensor([2.0, 6.0, 9.0])
        )
        self.assertTrue(torch.allclose(output[0, :2, 0], torch.tensor([0.2, 0.6])))
        self.assertEqual(valid.tolist(), [[True, True, False]])

    def test_exact_interpolation_uses_only_exact_anchor_validity(self):
        points = torch.tensor([[0.0, 0, 0, 1], [4.0, 0, 0, 4], [8.0, 0, 0, 4]])
        probabilities = torch.tensor([[[0.0], [0.4], [0.8]]])
        mask = torch.tensor([[False, True, True]])
        output, valid = interpolate_probabilities(
            probabilities, mask, points, torch.tensor([4.0])
        )
        self.assertTrue(torch.allclose(output[0, 0], torch.tensor([0.4])))
        self.assertEqual(valid.tolist(), [[True]])

    def test_multilabel_source_action_accepts_any_positive_class(self):
        labels = torch.tensor([[[0.0, 1.0, 0.0, 1.0]]])
        eligible = source_action_eligible(
            labels,
            pseudo_class=torch.tensor([[3]]),
            perturbed_class=torch.tensor([[3]]),
            score_original=torch.tensor([[0.8]]),
            valid=torch.tensor([[True]]),
            alpha=0.6,
        )
        self.assertEqual(eligible.tolist(), [[True]])

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

    def test_oracle_routes_isolate_the_three_diagnostics(self):
        probabilities = torch.tensor([
            [0.80, 0.10, 0.05],
            [0.20, 0.30, 0.10],
            [0.05, 0.10, 0.90],
            [0.20, 0.10, 0.30],
        ])
        gt_labels = torch.tensor([
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])

        o1 = build_oracle_routing(probabilities, gt_labels, 0.6, "o1")
        self.assertEqual(o1["action"].tolist(), [True, False, False, False])
        self.assertEqual(o1["background"].tolist(), [False, False, True, True])
        self.assertEqual(o1["ignore"].tolist(), [False, True, False, False])
        self.assertEqual(o1["action_labels"][0].tolist(), [1.0, 0.0, 0.0])

        o2 = build_oracle_routing(probabilities, gt_labels, 0.6, "o2")
        self.assertEqual(o2["action"].tolist(), [True, False, True, False])
        self.assertEqual(o2["background"].tolist(), [False, True, False, True])
        self.assertFalse(o2["ignore"].any())
        self.assertEqual(o2["action_labels"][0].tolist(), [0.0, 1.0, 1.0])
        self.assertEqual(o2["action_labels"][2].tolist(), [0.0, 0.0, 1.0])

        o3 = build_oracle_routing(probabilities, gt_labels, 0.6, "o3")
        self.assertEqual(o3["action"].tolist(), [True, True, False, False])
        self.assertEqual(o3["background"].tolist(), [False, False, True, True])
        self.assertFalse(o3["ignore"].any())
        self.assertTrue(torch.equal(o3["action_labels"], gt_labels))

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

    def test_calibration_crop_reuses_action_preserving_truncation(self):
        class FixedRng:
            def __init__(self):
                self.values = iter((0, 4))

            def randint(self, _lower, _upper):
                return next(self.values)

        item = {
            "feats": torch.arange(10).float().view(1, 10),
            "segments": torch.tensor([[6.0, 9.0]]),
            "labels": torch.tensor([2]),
            "metadata": {},
        }
        cropped = stage1_data_utils.truncate_feats(
            item,
            max_seq_len=5,
            trunc_thresh=0.3,
            crop_ratio=None,
            rng=FixedRng(),
        )
        self.assertEqual(cropped["metadata"]["crop_start"], 4)
        self.assertEqual(cropped["segments"].tolist(), [[2.0, 5.0]])
        self.assertEqual(cropped["labels"].tolist(), [2])

    def test_calibration_artifact_validates_manifest_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            metadata = {"perturbation": "shift", "alpha": 0.6}
            stage1_calibration.save_calibration_artifact(
                path,
                {"beta": 0.1, "tau_act": 0.2, "tau_bg": 0.03},
                [{"video_id": "v1"}],
                metadata,
            )
            loaded = stage1_calibration.load_calibration_artifact(
                path, metadata
            )
            self.assertEqual(loaded["beta"], 0.1)
            with self.assertRaises(ValueError):
                stage1_calibration.load_calibration_artifact(
                    path, {"perturbation": "context", "alpha": 0.6}
                )

    def test_training_rng_resume_matches_continuous_sequence(self):
        continuous = torch.Generator(device="cpu")
        continuous.manual_seed(3407)
        torch.randint(0, 2, (1,), generator=continuous)
        checkpoint_state = continuous.get_state()
        expected = torch.randint(0, 2, (8,), generator=continuous)

        resumed = torch.Generator(device="cpu")
        resumed.set_state(checkpoint_state)
        actual = torch.randint(0, 2, (8,), generator=resumed)
        self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
