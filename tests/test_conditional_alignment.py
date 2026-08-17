import importlib.util
import unittest
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "libs"
    / "models"
    / "SADA"
    / "conditional_alignment.py"
)
SPEC = importlib.util.spec_from_file_location("conditional_alignment", MODULE_PATH)
conditional_alignment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(conditional_alignment)
aggregate_conditional_class_losses = (
    conditional_alignment.aggregate_conditional_class_losses
)


class ConditionalAlignmentAggregationTests(unittest.TestCase):
    def _rows(self):
        return [
            {
                "source_loss": torch.tensor(1.0),
                "target_loss": torch.tensor(2.0),
                "source_samples": 4,
                "target_samples": 3,
                "class_weight": 1.0,
            },
            {
                "source_loss": torch.tensor(0.0),
                "target_loss": torch.tensor(6.0),
                "source_samples": 0,
                "target_samples": 2,
                "class_weight": 1.0,
            },
            {
                "source_loss": torch.tensor(3.0),
                "target_loss": torch.tensor(0.0),
                "source_samples": 5,
                "target_samples": 0,
                "class_weight": 1.0,
            },
        ]

    def test_default_keeps_independent_class_denominators(self):
        result = aggregate_conditional_class_losses(
            self._rows(), torch.tensor(0.0), paired_support=False
        )

        self.assertAlmostEqual(result["source_loss"].item(), 2.0)
        self.assertAlmostEqual(result["target_loss"].item(), 4.0)
        self.assertEqual(result["source_classes"], 2)
        self.assertEqual(result["target_classes"], 2)
        self.assertAlmostEqual(
            result["forgotten_target_objective"].item(), 3.0
        )

    def test_paired_support_forgets_unmatched_classes_symmetrically(self):
        result = aggregate_conditional_class_losses(
            self._rows(), torch.tensor(0.0), paired_support=True
        )

        self.assertAlmostEqual(result["source_loss"].item(), 1.0)
        self.assertAlmostEqual(result["target_loss"].item(), 2.0)
        self.assertEqual(result["source_classes"], 1)
        self.assertEqual(result["target_classes"], 1)
        self.assertEqual(result["paired_classes"], 1)
        self.assertEqual(result["forgotten_target_classes"], 1)
        self.assertEqual(result["forgotten_target_samples"], 2)

    def test_no_pair_returns_differentiable_zero(self):
        source_loss = torch.tensor(1.0, requires_grad=True)
        target_loss = torch.tensor(2.0, requires_grad=True)
        rows = [{
            "source_loss": source_loss,
            "target_loss": target_loss,
            "source_samples": 0,
            "target_samples": 3,
            "class_weight": 1.0,
        }]

        result = aggregate_conditional_class_losses(
            rows,
            source_loss + target_loss,
            paired_support=True,
        )
        total = result["source_loss"] + result["target_loss"]
        total.backward()

        self.assertEqual(total.item(), 0.0)
        self.assertEqual(source_loss.grad.item(), 0.0)
        self.assertEqual(target_loss.grad.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
