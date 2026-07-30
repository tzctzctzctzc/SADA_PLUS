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

    def test_proposal_loss_terms_remove_anchor_amplification(self):
        numerator, denominator, weight_square_sum, proposal_count = (
            stage2_utils.proposal_loss_terms(
                torch.tensor([1.0, 1.0, 1.0, 3.0]),
                torch.tensor([0, 0, 0, 1]),
                torch.tensor([1.0, 1.0]),
            )
        )
        self.assertAlmostEqual(float(numerator / denominator), 2.0)
        self.assertEqual(float(denominator), 2.0)
        self.assertEqual(float(weight_square_sum), 2.0)
        self.assertEqual(float(proposal_count), 2.0)

    def test_proposal_loss_terms_apply_detached_quality_weights(self):
        weights = torch.tensor([1.0, 3.0], requires_grad=True)
        numerator, denominator, _, _ = stage2_utils.proposal_loss_terms(
            torch.tensor([1.0, 3.0], requires_grad=True),
            torch.tensor([0, 1]),
            weights,
        )
        loss = numerator / denominator
        loss.backward()
        self.assertIsNone(weights.grad)
        self.assertAlmostEqual(float(loss), 2.5)

    def test_aligned_boundary_iou_from_offsets(self):
        first = torch.tensor([[2.0, 3.0], [1.0, 1.0]])
        second = torch.tensor([[2.0, 3.0], [3.0, 3.0]])
        stability = stage2_utils.aligned_boundary_iou_from_offsets(
            first, second
        )
        self.assertTrue(torch.allclose(
            stability, torch.tensor([1.0, 1.0 / 3.0])
        ))

    def test_weighted_anchor_mse_matches_equal_mse_for_unit_weights(self):
        predictions = torch.tensor(
            [[[1.0, 3.0], [2.0, 4.0]]], requires_grad=True
        )
        targets = torch.tensor([[[0.0, 1.0], [1.0, 1.0]]])
        mask = torch.tensor([[True, True]])
        loss = stage2_utils.weighted_anchor_mse(
            predictions, targets, torch.ones_like(mask, dtype=torch.float), mask
        )
        expected = torch.nn.functional.mse_loss(predictions, targets)
        self.assertTrue(torch.allclose(loss, expected))

    def test_weighted_anchor_mse_detaches_stability(self):
        predictions = torch.tensor(
            [[[1.0, 1.0], [3.0, 3.0]]], requires_grad=True
        )
        targets = torch.zeros_like(predictions)
        weights = torch.tensor([[1.0, 3.0]], requires_grad=True)
        loss = stage2_utils.weighted_anchor_mse(
            predictions,
            targets,
            weights,
            torch.tensor([[True, True]]),
        )
        loss.backward()
        self.assertIsNone(weights.grad)
        self.assertAlmostEqual(float(loss), 7.0)

    def test_top_fraction_mask_is_independent_per_sample(self):
        scores = torch.tensor([
            [0.9, 0.8, 0.7, 0.6],
            [0.1, 0.4, 0.3, 0.2],
        ])
        eligible = torch.tensor([
            [True, True, True, True],
            [False, True, True, True],
        ])
        kept = stage2_utils.top_fraction_per_sample_mask(
            scores, eligible, 0.5
        )
        self.assertEqual(kept.tolist(), [
            [True, True, False, False],
            [False, True, True, False],
        ])

    def test_top_fraction_mask_keeps_one_sparse_anchor(self):
        kept = stage2_utils.top_fraction_per_sample_mask(
            torch.tensor([[0.2, 0.9]]),
            torch.tensor([[False, True]]),
            0.25,
        )
        self.assertEqual(kept.tolist(), [[False, True]])

    def test_gaussian_perturbation_is_sample_independent_and_reproducible(self):
        inputs = torch.zeros(2, 3, 4)
        masks = torch.ones(2, 1, 4, dtype=torch.bool)
        first = stage2_utils.deterministic_gaussian_perturbation(
            inputs, masks, ['video-a:0', 'video-b:0'], 0.01, 123, 5
        )
        repeated = stage2_utils.deterministic_gaussian_perturbation(
            inputs, masks, ['video-a:0', 'video-b:0'], 0.01, 123, 5
        )
        swapped = stage2_utils.deterministic_gaussian_perturbation(
            inputs, masks, ['video-b:0', 'video-a:0'], 0.01, 123, 5
        )
        self.assertTrue(torch.equal(first, repeated))
        self.assertFalse(torch.equal(first[0], first[1]))
        self.assertTrue(torch.equal(first[0], swapped[1]))
        self.assertTrue(torch.equal(first[1], swapped[0]))

    def test_gaussian_perturbation_preserves_padding(self):
        inputs = torch.zeros(1, 2, 4)
        masks = torch.tensor([[[True, True, False, False]]])
        perturbed = stage2_utils.deterministic_gaussian_perturbation(
            inputs, masks, ['video-a:10'], 0.01, 123, 5
        )
        self.assertFalse(torch.equal(perturbed[..., :2], inputs[..., :2]))
        self.assertTrue(torch.equal(perturbed[..., 2:], inputs[..., 2:]))

    def test_channel_dropout_is_sample_independent_and_reproducible(self):
        inputs = torch.ones(2, 64, 4)
        masks = torch.ones(2, 1, 4, dtype=torch.bool)
        first = stage2_utils.deterministic_channel_dropout_perturbation(
            inputs, masks, ['video-a:0', 'video-b:0'], 0.25, 123, 5
        )
        repeated = stage2_utils.deterministic_channel_dropout_perturbation(
            inputs, masks, ['video-a:0', 'video-b:0'], 0.25, 123, 5
        )
        swapped = stage2_utils.deterministic_channel_dropout_perturbation(
            inputs, masks, ['video-b:0', 'video-a:0'], 0.25, 123, 5
        )
        self.assertTrue(torch.equal(first, repeated))
        self.assertFalse(torch.equal(first[0], first[1]))
        self.assertTrue(torch.equal(first[0], swapped[1]))
        self.assertTrue(torch.equal(first[1], swapped[0]))
        self.assertTrue(torch.equal(first[:, :, 0], first[:, :, -1]))

    def test_channel_dropout_preserves_padding(self):
        inputs = torch.ones(1, 64, 4)
        masks = torch.tensor([[[True, True, False, False]]])
        perturbed = stage2_utils.deterministic_channel_dropout_perturbation(
            inputs, masks, ['video-a:10'], 0.25, 123, 5
        )
        self.assertTrue(torch.equal(perturbed[..., 2:], inputs[..., 2:]))
        self.assertTrue(torch.any(perturbed[..., :2] == 0.0))


if __name__ == "__main__":
    unittest.main()
