import unittest

import torch

from libs.trainer.gradient_surgery import project_conflicting_gradients


class GradientSurgeryTests(unittest.TestCase):
    def test_projects_conflicting_component(self):
        projected, cosine, conflict, coefficient = project_conflicting_gradients(
            [torch.tensor([3.0, 4.0])], [torch.tensor([-5.0, 3.0])]
        )
        self.assertLess(cosine.item(), 0.0)
        self.assertEqual(conflict.item(), 1.0)
        self.assertGreater(coefficient.item(), 0.0)
        self.assertAlmostEqual(
            torch.dot(torch.tensor([3.0, 4.0]), projected[0]).item(),
            0.0,
            places=6,
        )

    def test_preserves_non_conflicting_gradient(self):
        auxiliary = torch.tensor([4.0, 3.0])
        projected, _, conflict, coefficient = project_conflicting_gradients(
            [torch.tensor([3.0, 4.0])], [auxiliary]
        )
        self.assertEqual(conflict.item(), 0.0)
        self.assertEqual(coefficient.item(), 0.0)
        self.assertTrue(torch.equal(projected[0], auxiliary))


if __name__ == '__main__':
    unittest.main()
