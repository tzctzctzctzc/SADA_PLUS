import unittest

import torch

from tools.audit_conditional_gradient_influence import (
    conditional_target_coefficients,
    first_order_feature_influence,
    summarize_records,
)


class ConditionalGradientInfluenceTests(unittest.TestCase):
    def test_class_balanced_coefficients_preserve_original_weights(self):
        pseudo_class = torch.tensor([0, 0, 1, 2, 2, 2])
        coefficients, present = conditional_target_coefficients(
            pseudo_class,
            num_classes=3,
            background_weight=0.3,
            global_scale=2.0,
        )

        self.assertEqual(present, 3)
        self.assertAlmostEqual(coefficients[pseudo_class == 0].sum().item(), 2 / 3)
        self.assertAlmostEqual(coefficients[pseudo_class == 1].sum().item(), 2 / 3)
        self.assertAlmostEqual(coefficients[pseudo_class == 2].sum().item(), 0.2)

    def test_positive_influence_means_gt_loss_increases(self):
        oracle = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        domain = torch.tensor([[2.0, 0.0], [-3.0, 0.0]])
        coefficients = torch.tensor([0.5, 0.25])

        update, influence, cosine = first_order_feature_influence(
            oracle, domain, coefficients, beta=0.5
        )

        self.assertTrue(torch.allclose(update[:, 0], torch.tensor([0.5, -0.375])))
        self.assertTrue(torch.allclose(influence, torch.tensor([0.5, -0.375])))
        self.assertTrue(torch.allclose(cosine, torch.tensor([1.0, -1.0])))

    def test_summary_separates_pseudo_error_types_and_harm(self):
        records = {
            "is_action": torch.tensor([True, True, False, False]),
            "gt_action": torch.tensor([True, False, True, False]),
            "correct_class": torch.tensor([True, False, False, False]),
            "alignment_correct": torch.tensor([True, False, False, True]),
            "domain_loss": torch.tensor([0.2, 0.8, 0.6, 0.1]),
            "confidence": torch.tensor([0.9, 0.7, 0.2, 0.1]),
            "oracle_cls_loss": torch.ones(4),
            "oracle_gradient_norm": torch.ones(4),
            "domain_update_norm": torch.ones(4),
            "influence": torch.tensor([2.0, -1.0, 1.0, -3.0]),
            "cosine": torch.tensor([1.0, -1.0, 1.0, -1.0]),
        }

        summary = summarize_records(records)
        all_group = summary["groups"]["all"]
        self.assertEqual(summary["groups"]["false_action"]["n"], 1)
        self.assertEqual(summary["groups"]["missed_action_background"]["n"], 1)
        self.assertAlmostEqual(all_group["harmful_rate"], 0.5)
        self.assertAlmostEqual(all_group["net_predicted_delta"], -1.0)
        self.assertAlmostEqual(all_group["positive_harm"], 3.0)
        self.assertAlmostEqual(all_group["negative_benefit"], 4.0)
        self.assertAlmostEqual(all_group["positive_magnitude_fraction"], 3 / 7)


if __name__ == "__main__":
    unittest.main()
