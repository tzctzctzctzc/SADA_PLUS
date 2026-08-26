import importlib.util
from pathlib import Path
import unittest

import torch

_MODULE_PATH = Path(__file__).parents[1] / 'libs' / 'models' / 'SADA' / 'transport_alignment.py'
_SPEC = importlib.util.spec_from_file_location('transport_alignment', _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

(
    ResidualAdapter,
    action_label_distribution,
    joint_unbalanced_ot_loss,
    unbalanced_sinkhorn,
) = (
    _MODULE.ResidualAdapter,
    _MODULE.action_label_distribution,
    _MODULE.joint_unbalanced_ot_loss,
    _MODULE.unbalanced_sinkhorn,
)


class TransportAlignmentTests(unittest.TestCase):
    def test_action_distribution_adds_background(self):
        labels = torch.tensor([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
        distribution = action_label_distribution(action_labels=labels)
        self.assertEqual(tuple(distribution.shape), (3, 3))
        self.assertTrue(torch.equal(distribution[1], torch.tensor([0.0, 0.0, 1.0])))
        self.assertTrue(torch.allclose(distribution.sum(dim=1), torch.ones(3)))

    def test_sinkhorn_returns_finite_unbalanced_plan(self):
        cost = torch.tensor([[0.1, 1.0], [1.0, 0.1]])
        plan = unbalanced_sinkhorn(cost, epsilon=0.2, marginal_relaxation=0.3)
        self.assertEqual(tuple(plan.shape), (2, 2))
        self.assertTrue(torch.isfinite(plan).all())
        self.assertGreater(float(plan.diag().sum()), float(plan[0, 1] + plan[1, 0]))

    def test_uot_loss_keeps_target_gradient(self):
        source = torch.randn(8, 4, requires_grad=True)
        target = torch.randn(7, 4, requires_grad=True)
        labels = torch.zeros(8, 2)
        labels[:4, 0] = 1.0
        labels[4:, 1] = 1.0
        target_probs = torch.sigmoid(torch.randn(7, 2))
        loss, stats = joint_unbalanced_ot_loss(
            source,
            target,
            labels,
            target_probs,
            max_points=16,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(target.grad)
        self.assertGreater(float(target.grad.abs().sum()), 0.0)
        self.assertGreaterEqual(float(stats['rejected_mass']), 0.0)

    def test_adapter_is_zero_initialized(self):
        adapter = ResidualAdapter(4, bottleneck=2)
        features = torch.randn(2, 4, 5)
        self.assertTrue(torch.allclose(adapter(features), features))


if __name__ == '__main__':
    unittest.main()
