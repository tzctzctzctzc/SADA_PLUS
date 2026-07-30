import importlib.util
import unittest
from pathlib import Path

import torch

SPEC = importlib.util.spec_from_file_location(
    "lr_schedulers",
    Path(__file__).parents[1] / "libs" / "utils" / "lr_schedulers.py",
)
LR_SCHEDULERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LR_SCHEDULERS)
LinearWarmupCosineAnnealingLR = (
    LR_SCHEDULERS.LinearWarmupCosineAnnealingLR
)


class OptimizerTest(unittest.TestCase):
    def test_scheduler_accepts_scientific_notation_string_eta_min(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.SGD([parameter], lr=1e-3)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=1,
            max_epochs=10,
            eta_min="5e-05",
        )
        optimizer.step()
        scheduler.step()
        optimizer.step()
        scheduler.step()
        self.assertAlmostEqual(scheduler.eta_min, 5e-5)


if __name__ == "__main__":
    unittest.main()
